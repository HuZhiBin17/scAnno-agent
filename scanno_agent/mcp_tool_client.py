from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from jsonschema import ValidationError, validate
except ImportError:  # pragma: no cover
    ValidationError = Exception  # type: ignore[assignment]
    validate = None  # type: ignore[assignment]

try:
    from agent_memory import AgentMemoryStore, stable_hash
    from agent_tools_core import (
        query_go_terms_core,
        query_kegg_pathways_core,
        rag_annotate_core,
        search_literature_core,
        validate_annotation_core,
    )
except ImportError:  # pragma: no cover
    from .agent_memory import AgentMemoryStore, stable_hash
    from .agent_tools_core import (
        query_go_terms_core,
        query_kegg_pathways_core,
        rag_annotate_core,
        search_literature_core,
        validate_annotation_core,
    )


TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "search_literature": {
        "name": "search_literature",
        "description": "Search the local literature RAG index for relevant chunks.",
        "inputSchema": {
            "type": "object",
            "properties": {"query": {"type": "string", "minLength": 1}},
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    "query_go_terms": {
        "name": "query_go_terms",
        "description": "Query GO terms for a list of marker genes.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "gene_list": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 20}
            },
            "required": ["gene_list"],
            "additionalProperties": False,
        },
    },
    "query_kegg_pathways": {
        "name": "query_kegg_pathways",
        "description": "Query KEGG pathways for a list of marker genes.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "gene_list": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 20}
            },
            "required": ["gene_list"],
            "additionalProperties": False,
        },
    },
    "rag_annotate": {
        "name": "rag_annotate",
        "description": "Run the RAG annotation pipeline for one cluster.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "markers": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                "tissue": {"type": "string"},
                "technology": {"type": "string"},
                "cluster_id": {"type": "string"},
            },
            "required": ["markers"],
            "additionalProperties": False,
        },
    },
    "validate_annotation": {
        "name": "validate_annotation",
        "description": "Validate a structured annotation result.",
        "inputSchema": {
            "type": "object",
            "properties": {"annotation": {"type": "object"}},
            "required": ["annotation"],
            "additionalProperties": False,
        },
    },
}


LOCAL_TOOLS = {
    "search_literature": lambda args: search_literature_core(args["query"]),
    "query_go_terms": lambda args: query_go_terms_core(args["gene_list"]),
    "query_kegg_pathways": lambda args: query_kegg_pathways_core(args["gene_list"]),
    "rag_annotate": lambda args: rag_annotate_core(
        args["markers"],
        tissue=args.get("tissue", "unknown"),
        technology=args.get("technology", "scRNA-seq"),
        cluster_id=args.get("cluster_id", "0"),
    ),
    "validate_annotation": lambda args: validate_annotation_core(args["annotation"]),
}


def _clean_subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    return env


@dataclass
class ToolCallResult:
    tool_name: str
    args: dict[str, Any]
    result: dict[str, Any]
    ok: bool
    error: str | None
    duration_ms: int
    cache_hit: bool

    def trace_payload(self, step_index: int) -> dict[str, Any]:
        return {
            "step_index": step_index,
            "tool_name": self.tool_name,
            "args_hash": stable_hash(self.args),
            "status": "ok" if self.ok else "error",
            "error": self.error,
            "duration_ms": self.duration_ms,
            "cache_hit": self.cache_hit,
            "result_preview": self.result.get("display_text", "")[:500] if isinstance(self.result, dict) else "",
        }


class MCPToolClient:
    def __init__(
        self,
        backend: str = "local",
        timeout_seconds: int = 60,
        memory_store: AgentMemoryStore | None = None,
        cache_enabled: bool = True,
    ):
        self.backend = (backend or "local").lower()
        self.timeout_seconds = int(timeout_seconds)
        self.memory_store = memory_store
        self.cache_enabled = cache_enabled

    def list_tools(self) -> list[dict[str, Any]]:
        if self.backend == "mcp":
            try:
                return asyncio.run(self._list_tools_mcp())
            except Exception:
                return list(TOOL_SCHEMAS.values())
        return list(TOOL_SCHEMAS.values())

    def validate_args(self, tool_name: str, args: dict[str, Any]) -> str | None:
        schema = TOOL_SCHEMAS.get(tool_name)
        if not schema:
            return f"Unknown tool: {tool_name}"
        if validate is None:
            return None
        try:
            validate(instance=args, schema=schema["inputSchema"])
            return None
        except ValidationError as exc:
            return str(exc.message)

    def call_tool(
        self,
        tool_name: str,
        args: dict[str, Any],
        run_id: str = "",
        step_index: int = 0,
    ) -> ToolCallResult:
        started = time.time()
        args = args or {}

        validation_error = self.validate_args(tool_name, args)
        if validation_error:
            return self._finish_call(
                run_id,
                step_index,
                tool_name,
                args,
                {"ok": False, "error": validation_error, "display_text": f"Tool error: {validation_error}"},
                started,
                cache_hit=False,
            )

        if self.cache_enabled and self.memory_store:
            cached = self.memory_store.get_tool_cache(tool_name, args)
            if cached is not None:
                return self._finish_call(run_id, step_index, tool_name, args, cached, started, cache_hit=True)

        try:
            if self.backend == "mcp":
                result = asyncio.run(asyncio.wait_for(self._call_tool_mcp(tool_name, args), timeout=self.timeout_seconds))
            else:
                result = LOCAL_TOOLS[tool_name](args)
        except Exception as exc:
            result = {"ok": False, "error": str(exc), "display_text": f"Tool error: {exc}"}

        return self._finish_call(run_id, step_index, tool_name, args, result, started, cache_hit=False)

    def _finish_call(
        self,
        run_id: str,
        step_index: int,
        tool_name: str,
        args: dict[str, Any],
        result: dict[str, Any],
        started: float,
        cache_hit: bool,
    ) -> ToolCallResult:
        duration_ms = int((time.time() - started) * 1000)
        ok = bool(result.get("ok", False))
        error = result.get("error") if isinstance(result, dict) else None
        payload = ToolCallResult(
            tool_name=tool_name,
            args=args,
            result=result,
            ok=ok,
            error=error,
            duration_ms=duration_ms,
            cache_hit=cache_hit,
        )
        if self.memory_store:
            self.memory_store.record_tool_call(
                run_id=run_id,
                step_index=step_index,
                tool_name=tool_name,
                args=args,
                result=result,
                status="ok" if ok else "error",
                error=error,
                duration_ms=duration_ms,
                cache_hit=cache_hit,
            )
        return payload

    async def _list_tools_mcp(self) -> list[dict[str, Any]]:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        server_path = Path(__file__).with_name("mcp_server.py")
        params = StdioServerParameters(
            command=sys.executable,
            args=[str(server_path)],
            env=_clean_subprocess_env(),
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools_response = await session.list_tools()
                tools = []
                for tool in tools_response.tools:
                    tools.append(
                        {
                            "name": tool.name,
                            "description": tool.description or "",
                            "inputSchema": getattr(tool, "inputSchema", None)
                            or getattr(tool, "input_schema", None)
                            or {},
                        }
                    )
                return tools

    async def _call_tool_mcp(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        server_path = Path(__file__).with_name("mcp_server.py")
        params = StdioServerParameters(
            command=sys.executable,
            args=[str(server_path)],
            env=_clean_subprocess_env(),
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments=args)
                return self._parse_mcp_tool_result(result)

    @staticmethod
    def _parse_mcp_tool_result(result: Any) -> dict[str, Any]:
        structured = getattr(result, "structuredContent", None) or getattr(result, "structured_content", None)
        if isinstance(structured, dict):
            return structured

        content = getattr(result, "content", None) or []
        texts: list[str] = []
        for item in content:
            text = getattr(item, "text", None)
            if text:
                texts.append(text)
        joined = "\n".join(texts).strip()
        if joined:
            try:
                parsed = json.loads(joined)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
        is_error = bool(getattr(result, "isError", False) or getattr(result, "is_error", False))
        return {"ok": not is_error, "error": joined if is_error else None, "display_text": joined}
