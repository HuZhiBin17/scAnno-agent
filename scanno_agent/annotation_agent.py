"""
Step 7 — ReAct Agent：自动化注释 + 归因推理
──────────────────────────────────────────────
Agent 拥有以下工具：
  1. search_literature(query)    — 检索知识库
  2. query_go_term(gene_list)    — 查询 Gene Ontology（via QuickGO REST API）
  3. query_kegg_pathway(genes)   — 查询 KEGG 通路（via KEGG REST API）
  4. call_rag_annotate(markers)  — 调用 RAG pipeline 生成注释
  5. validate_annotation(result) — 检验注释是否有足够置信度

ReAct 循环：Thought → Action → Observation → … → Final Answer
最大循环次数：MAX_AGENT_STEPS（config.py 中配置）
"""
from __future__ import annotations
import re
import json
import time
import logging
import requests
from typing import Any, Optional

from config import (  # pyright: ignore[reportImplicitRelativeImport]
    LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS, OPENAI_API_KEY, OPENAI_BASE_URL,
    MAX_AGENT_STEPS, RERANK_TOP_K, AGENT_MEMORY_ENABLED, AGENT_MEMORY_DB,
    AGENT_CACHE_ENABLED, AGENT_REUSE_EXACT_MATCH, AGENT_SIMILARITY_THRESHOLD,
    AGENT_TOOL_BACKEND, MCP_TOOL_TIMEOUT_SECONDS, get_llm_runtime_config,
)
from agent_memory import AgentMemoryStore  # pyright: ignore[reportImplicitRelativeImport]
from agent_trace import AgentTrace  # pyright: ignore[reportImplicitRelativeImport]
from agent_tools_core import (  # pyright: ignore[reportImplicitRelativeImport]
    query_go_terms_core,
    query_kegg_pathways_core,
    rag_annotate_core,
    search_literature_core,
    validate_annotation_core,
)
from mcp_tool_client import MCPToolClient, TOOL_SCHEMAS, ToolCallResult  # pyright: ignore[reportImplicitRelativeImport]

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ─── Tool 实现 ────────────────────────────────────────────────────────────────

retriever = None
rag_pipeline = None


def tool_search_literature(query: str) -> str:
    """检索知识库，返回 top chunks 摘要"""
    chunks = retriever.retrieve(query)
    if not chunks:
        return "No relevant literature found."
    parts = []
    for c in chunks:
        parts.append(f"- [{c.pmid}] {c.title} ({c.year})\n  {c.text[:300]}...")
    return "\n".join(parts)


def tool_query_go_terms(gene_list: list[str]) -> str:
    """
    查询基因集相关 GO terms（优先 QuickGO，失败时回退 Enrichr）
    """
    if not gene_list:
        return "No genes provided."

    genes = [g.strip().upper() for g in gene_list if g and g.strip()]
    genes = genes[:20]
    if not genes:
        return "No valid genes provided."

    # 方案 A：QuickGO（对标识符要求严格，gene symbol 经常 400）
    quickgo_url = "https://www.ebi.ac.uk/QuickGO/services/annotation/search"
    try:
        resp = requests.get(
            quickgo_url,
            timeout=15,
            headers={"Accept": "application/json"},
            params={
                "geneProductSymbol": ",".join(genes[:10]),
                "taxonId": "9606",
                "limit": 20,
                "page": 1,
            },
        )
        if resp.status_code == 200:
            data = resp.json()
            results = data.get("results", [])
            lines = []
            seen_go: set[str] = set()
            for r in results[:20]:
                go_id = str(r.get("goId", "")).strip()
                raw_go_name = r.get("goName", "")
                go_name = "" if raw_go_name is None else str(raw_go_name).strip()
                aspect = str(r.get("goAspect", "")).strip()
                # QuickGO 某些记录缺失 goName，避免返回低质量结果，改走 fallback。
                if go_id and go_name and go_name.lower() != "none" and go_id not in seen_go:
                    lines.append(f"{go_id} ({aspect}): {go_name}")
                    seen_go.add(go_id)
            if lines:
                return "\n".join(lines[:10])
        else:
            log.warning(f"QuickGO returned {resp.status_code}, fallback to Enrichr.")
    except Exception as e:
        log.warning(f"QuickGO query failed, fallback to Enrichr: {e}")

    # 方案 B：Enrichr（对 gene symbol 友好，当前网络可达性更好）
    add_list_url = "https://maayanlab.cloud/Enrichr/addList"
    enrich_url = "https://maayanlab.cloud/Enrichr/enrich"
    try:
        add_resp = requests.post(
            add_list_url,
            timeout=20,
            files={
                "list": (None, "\n".join(genes)),
                "description": (None, "scanno_agent_go_fallback"),
            },
        )
        if add_resp.status_code != 200:
            return f"GO query failed (QuickGO + Enrichr): {add_resp.status_code}"
        add_data = add_resp.json()
        user_list_id = add_data.get("userListId")
        if not user_list_id:
            return "GO query failed: Enrichr did not return userListId."

        dbs = [
            ("GO_Biological_Process_2023", "BP"),
            ("GO_Molecular_Function_2023", "MF"),
            ("GO_Cellular_Component_2023", "CC"),
        ]
        terms: list[tuple[float, str]] = []
        for db_name, tag in dbs:
            enr_resp = requests.get(
                enrich_url,
                timeout=20,
                params={"userListId": user_list_id, "backgroundType": db_name},
            )
            if enr_resp.status_code != 200:
                continue
            enr_data = enr_resp.json()
            rows = enr_data.get(db_name, [])
            for row in rows[:8]:
                # row: [rank, term_name, p_value, z_score, combined_score, overlapping_genes, ...]
                if not isinstance(row, list) or len(row) < 3:
                    continue
                term_name = str(row[1]).strip()
                p_value = row[2]
                if not term_name:
                    continue
                go_match = re.search(r"(GO:\d{7})", term_name)
                go_id = go_match.group(1) if go_match else "GO:NA"
                if isinstance(p_value, (int, float)):
                    terms.append((float(p_value), f"{go_id} ({tag}, p={p_value:.2e}): {term_name}"))
                else:
                    terms.append((1.0, f"{go_id} ({tag}): {term_name}"))

        if not terms:
            return "No GO annotations found."
        terms.sort(key=lambda x: x[0])
        lines = [t[1] for t in terms[:10]]
        return "\n".join(lines)
    except Exception as e:
        return f"GO query failed: {e}"


def tool_query_kegg_pathways(gene_list: list[str]) -> str:
    """
    通过 KEGG REST API 查询人类 KEGG 通路
    """
    if not gene_list:
        return "No genes provided."
    pathway_counts: dict[str, int] = {}
    pathway_names: dict[str, str] = {}

    for gene in gene_list[:8]:   # 限制 8 个基因
        url = f"https://rest.kegg.jp/find/hsa/{gene}"
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                continue
            # 获取 KEGG gene ID
            lines = resp.text.strip().split("\n")
            if not lines or not lines[0]:
                continue
            kegg_gene_id = lines[0].split("\t")[0]

            # 查询该基因参与的通路
            path_resp = requests.get(
                f"https://rest.kegg.jp/link/pathway/{kegg_gene_id}",
                timeout=10
            )
            for line in path_resp.text.strip().split("\n"):
                parts = line.split("\t")
                if len(parts) == 2:
                    pathway_id = parts[1].strip()
                    pathway_counts[pathway_id] = pathway_counts.get(pathway_id, 0) + 1

            time.sleep(0.1)   # KEGG 速率限制
        except Exception:
            continue

    if not pathway_counts:
        return "No KEGG pathways found."

    # 获取通路名称（取 top 5）
    top_pathways = sorted(pathway_counts.items(), key=lambda x: -x[1])[:5]
    result_lines = []
    for pid, cnt in top_pathways:
        try:
            name_resp = requests.get(
                f"https://rest.kegg.jp/list/{pid}", timeout=8
            )
            name = name_resp.text.split("\t")[-1].strip() if name_resp.ok else pid
        except Exception:
            name = pid
        result_lines.append(f"{pid}: {name} (hit by {cnt} genes)")

    return "\n".join(result_lines)


def tool_rag_annotate(
    markers: list[str],
    tissue: str = "unknown",
    technology: str = "scRNA-seq",
    cluster_id: str = "0",
) -> str:
    """调用 RAG pipeline 生成细胞类型注释"""
    result = rag_pipeline.annotate_cluster(
        markers=markers,
        tissue=tissue,
        technology=technology,
        cluster_id=cluster_id,
    )
    return json.dumps(result, ensure_ascii=False, indent=2)


def tool_validate_annotation(annotation_json: str = "", annotation: str = "") -> str:
    """检验注释质量，如置信度低则给出改进建议。兼容 annotation/annotation_json 两种入参。"""
    payload = annotation_json or annotation
    if not payload:
        return "Invalid JSON annotation."
    try:
        ann = json.loads(payload)
    except Exception:
        return "Invalid JSON annotation."

    confidence = ann.get("confidence", 0)
    markers    = ann.get("key_markers", [])
    refs       = ann.get("supporting_references", [])

    issues = []
    if confidence < 0.5:
        issues.append("Low confidence (<0.5). Need more specific marker evidence.")
    if len(markers) < 2:
        issues.append("Too few key markers identified.")
    if not refs:
        issues.append("No supporting references found.")
    if not ann.get("go_terms"):
        issues.append("GO terms missing. Consider querying GO database.")
    if not ann.get("kegg_pathways"):
        issues.append("KEGG pathways missing. Consider querying KEGG.")

    if not issues:
        return f"Annotation validated. Confidence: {confidence:.2f}. Cell type: {ann['cell_type']}."
    return "Issues found:\n" + "\n".join(f"- {i}" for i in issues)


# ─── Tool 注册表 ──────────────────────────────────────────────────────────────

TOOLS = {
    "search_literature":  tool_search_literature,
    "query_go_terms":     tool_query_go_terms,
    "query_kegg_pathways":tool_query_kegg_pathways,
    "rag_annotate":       tool_rag_annotate,
    "validate_annotation":tool_validate_annotation,
}

TOOL_DESCRIPTIONS = """
Available Tools:
1. search_literature(query: str) -> str
   Search the single-cell literature knowledge base for relevant papers.

2. query_go_terms(gene_list: list[str]) -> str
   Query Gene Ontology (GO) annotations for a list of genes via EBI QuickGO.

3. query_kegg_pathways(gene_list: list[str]) -> str
   Query KEGG human pathways enriched in a gene list.

4. rag_annotate(markers: list[str], tissue: str, technology: str, cluster_id: str) -> str
   Run the full RAG pipeline to annotate a cell cluster and return JSON result.

5. validate_annotation(annotation_ref: "last_rag_result") -> str
   Validate the in-memory last_rag_result annotation and report issues.
"""

REACT_SYSTEM = f"""You are a single-cell biology expert agent.
You MUST follow a strict JSON protocol on every step.

{TOOL_DESCRIPTIONS}

Output MUST be exactly one JSON object with this envelope schema:
{{
  "type": "ACTION" | "FINAL",
  "thought": "<optional short reasoning>",
  "action": {{"name": "<tool_name>", "args": {{...}}}},
  "final": {{...final annotation json...}}
}}

Rules:
- If type is ACTION, include action and DO NOT include final.
- If type is FINAL, include final and DO NOT include action.
- Never output Observation blocks or markdown.
- rag_annotate is executed by program BEFORE your loop; do not call it.
- If you call validate_annotation, use {{"annotation_ref": "last_rag_result"}}.
- Validate before returning FINAL when possible.
- Do not invent data not present in observations.
"""


# ─── ReAct Agent ─────────────────────────────────────────────────────────────

class SingleCellAgent:

    def __init__(
        self,
        max_steps: int = MAX_AGENT_STEPS,
        tool_backend: str | None = None,
        memory_enabled: bool | None = None,
        cache_enabled: bool | None = None,
        reuse_exact_match: bool | None = None,
        trace_enabled: bool = True,
    ):
        self.max_steps = max_steps
        self.history: list[dict[str, Any]] = []
        self.parse_fail_budget = 2
        self.empty_resp_budget = 2
        self.tool_error_budget = 2
        self.tool_backend = (tool_backend or AGENT_TOOL_BACKEND or "local").lower()
        self.memory_enabled = AGENT_MEMORY_ENABLED if memory_enabled is None else bool(memory_enabled)
        self.cache_enabled = AGENT_CACHE_ENABLED if cache_enabled is None else bool(cache_enabled)
        self.reuse_exact_match = AGENT_REUSE_EXACT_MATCH if reuse_exact_match is None else bool(reuse_exact_match)
        self.trace_enabled = trace_enabled
        self._last_llm_cache_hit = False
        self._last_llm_model: str | None = None
        self._last_llm_prompt_payload: dict[str, Any] | None = None
        self._last_llm_response: str | None = None
        self.memory_store = AgentMemoryStore(AGENT_MEMORY_DB, enabled=self.memory_enabled)
        self.tool_client = MCPToolClient(
            backend=self.tool_backend,
            timeout_seconds=MCP_TOOL_TIMEOUT_SECONDS,
            memory_store=self.memory_store if self.memory_enabled else None,
            cache_enabled=self.cache_enabled,
        )

    def _extract_first_json_object(self, text: str) -> Optional[str]:
        """从文本中提取第一个完整 JSON 对象，避免前后杂质影响解析。"""
        start = text.find("{")
        if start < 0:
            return None
        depth = 0
        in_str = False
        escaped = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if escaped:
                    escaped = False
                    continue
                if ch == "\\":
                    escaped = True
                    continue
                if ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        return None

    def _parse_envelope(self, text: str) -> Optional[dict[str, Any]]:
        """严格解析 AgentEnvelope。"""
        cleaned = text.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned).strip()
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()

        candidates = [cleaned]
        first_obj = self._extract_first_json_object(cleaned)
        if first_obj and first_obj not in candidates:
            candidates.append(first_obj)

        for c in candidates:
            try:
                payload = json.loads(c)
                if isinstance(payload, dict):
                    return payload
            except Exception:
                continue
        return None

    def _validate_action_args(self, tool_name: str, args: Any) -> Optional[str]:
        if not isinstance(args, dict):
            return "action.args must be a JSON object"

        def _is_str_list(v: Any) -> bool:
            return isinstance(v, list) and all(isinstance(x, str) for x in v)

        if tool_name == "search_literature":
            if not isinstance(args.get("query"), str):
                return "search_literature requires {'query': str}"
        elif tool_name in {"query_go_terms", "query_kegg_pathways"}:
            if not _is_str_list(args.get("gene_list")):
                return f"{tool_name} requires {{'gene_list': list[str]}}"
        elif tool_name == "rag_annotate":
            return "rag_annotate is program-controlled and cannot be called by LLM"
        elif tool_name == "validate_annotation":
            if args.get("annotation_ref") != "last_rag_result":
                return "validate_annotation requires {'annotation_ref': 'last_rag_result'}"
        return None

    def _validate_action_envelope(
        self,
        payload: dict[str, Any],
    ) -> tuple[Optional[tuple[str, dict[str, Any]]], Optional[str]]:
        if payload.get("type") != "ACTION":
            return None, "envelope.type must be ACTION"
        if "final" in payload and payload["final"] not in (None, {}):
            return None, "ACTION envelope must not include final"

        action = payload.get("action")
        if not isinstance(action, dict):
            return None, "ACTION envelope requires action object"
        tool_name = action.get("name")
        args = action.get("args", {})
        if not isinstance(tool_name, str) or tool_name not in TOOL_SCHEMAS:
            return None, f"Unknown tool: {tool_name}"
        if tool_name == "rag_annotate":
            return None, "rag_annotate is program-controlled and cannot be called by LLM"
        if tool_name != "validate_annotation":
            schema_error = self.tool_client.validate_args(tool_name, args)
            if schema_error:
                return None, schema_error
        arg_error = self._validate_action_args(tool_name, args)
        if arg_error:
            return None, arg_error
        return (tool_name, args), None

    def _validate_final_payload(self, payload: Any) -> tuple[Optional[dict[str, Any]], Optional[str]]:
        if not isinstance(payload, dict):
            return None, "final must be a JSON object"
        payload = dict(payload)
        if "key_markers" not in payload:
            for alias in ("markers", "marker_genes", "supporting_markers"):
                if alias in payload:
                    payload["key_markers"] = payload.get(alias)
                    break
        payload.setdefault("subtype", None)
        payload.setdefault("go_terms", [])
        payload.setdefault("kegg_pathways", [])
        payload.setdefault("supporting_references", [])
        payload.setdefault("attribution", payload.get("thought", "Validated model final fallback."))

        required = [
            "cell_type",
            "subtype",
            "confidence",
            "key_markers",
            "go_terms",
            "kegg_pathways",
            "attribution",
            "supporting_references",
        ]
        for k in required:
            if k not in payload:
                return None, f"final missing required key: {k}"

        cell_type = payload.get("cell_type")
        if not isinstance(cell_type, str) or not cell_type.strip():
            return None, "final.cell_type must be non-empty string"
        subtype = payload.get("subtype")
        if subtype is not None and not isinstance(subtype, str):
            return None, "final.subtype must be string or null"
        if not isinstance(payload.get("attribution"), str):
            return None, "final.attribution must be string"

        for list_key in ("key_markers", "go_terms", "kegg_pathways", "supporting_references"):
            value = payload.get(list_key)
            if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
                return None, f"final.{list_key} must be list[str]"

        confidence_raw = payload.get("confidence")
        if confidence_raw is None or not isinstance(confidence_raw, (int, float, str)):
            return None, "final.confidence must be numeric"
        try:
            confidence = float(confidence_raw)
        except Exception:
            return None, "final.confidence must be numeric"
        payload["confidence"] = max(0.0, min(1.0, confidence))
        return payload, None

    def _build_step_prompt(
        self,
        state: str,
        cluster_info: dict[str, Any],
        observations: list[dict[str, str]],
        expected_type: str,
        memory_context: list[dict[str, Any]] | None = None,
    ) -> str:
        payload = {
            "protocol": "Return exactly one JSON object. No markdown, no prose, no Observation field.",
            "state": state,
            "expected_type": expected_type,
            "cluster": {
                "cluster_id": cluster_info.get("cluster_id", "0"),
                "markers": cluster_info.get("markers", []),
                "tissue": cluster_info.get("tissue", "unknown"),
                "technology": cluster_info.get("technology", "scRNA-seq"),
            },
            "recent_observations": observations[-4:],
            "similar_memories": memory_context or [],
            "envelope_schema": {
                "type": "ACTION|FINAL",
                "thought": "optional short string",
                "action": {"name": "tool_name", "args": {}},
                "final": {"cell_type": "str", "confidence": "0..1", "...": "..."},
            },
            "constraints": [
                "ACTION and FINAL are mutually exclusive",
                "when ACTION: include action only",
                "when FINAL: include final only",
                "rag_annotate is program-executed and forbidden in ACTION",
                "validate_annotation must use annotation_ref=last_rag_result",
            ],
        }
        return json.dumps(payload, ensure_ascii=False)

    def _call_tool_result(
        self,
        tool_name: str,
        args: Any,
        run_id: str = "",
        step_index: int = 0,
    ) -> ToolCallResult:
        if not isinstance(args, dict):
            args = {}
        return self.tool_client.call_tool(tool_name, args, run_id=run_id, step_index=step_index)

    def _call_llm_cached(self, system: str, user: str, run_id: str) -> str:
        runtime = get_llm_runtime_config()
        model = runtime["model"] or LLM_MODEL
        prompt_payload = {"system": system, "user": user, "model": model}
        self._last_llm_model = model
        self._last_llm_prompt_payload = prompt_payload
        self._last_llm_response = None
        if self.cache_enabled and self.memory_enabled:
            cached = self.memory_store.get_llm_cache(model, prompt_payload)
            if cached is not None:
                self._last_llm_cache_hit = True
                self._last_llm_response = cached
                self.memory_store.record_llm_call(
                    run_id=run_id,
                    model=model,
                    prompt_payload=prompt_payload,
                    response_text=cached,
                    status="ok",
                    error=None,
                    duration_ms=0,
                    cache_hit=True,
                )
                return cached

        started = time.time()
        try:
            from rag_pipeline import call_llm  # pyright: ignore[reportImplicitRelativeImport]
            response = call_llm(system, user, max_retries=2)
            self._last_llm_response = response
            self._last_llm_cache_hit = False
            duration_ms = int((time.time() - started) * 1000)
            llm_status = "ok" if response and response.strip() else "empty"
            if self.memory_enabled:
                self.memory_store.record_llm_call(
                    run_id=run_id,
                    model=model,
                    prompt_payload=prompt_payload,
                    response_text=response,
                    status=llm_status,
                    error=None if llm_status == "ok" else "empty response",
                    duration_ms=duration_ms,
                    cache_hit=False,
                )
            return response
        except Exception as exc:
            duration_ms = int((time.time() - started) * 1000)
            if self.memory_enabled:
                self.memory_store.record_llm_call(
                    run_id=run_id,
                    model=model,
                    prompt_payload=prompt_payload,
                    response_text=None,
                    status="error",
                    error=str(exc),
                    duration_ms=duration_ms,
                    cache_hit=False,
            )
            raise

    def _mark_last_llm_response_unusable(self, run_id: str, status: str, error: str) -> None:
        if not self.memory_enabled or not self._last_llm_model or not self._last_llm_prompt_payload:
            return
        self.memory_store.record_llm_call(
            run_id=run_id,
            model=self._last_llm_model,
            prompt_payload=self._last_llm_prompt_payload,
            response_text=self._last_llm_response,
            status=status,
            error=error,
            duration_ms=0,
            cache_hit=self._last_llm_cache_hit,
        )

    def _call_tool(self, tool_name: str, args: Any) -> str:
        """执行工具调用"""
        if tool_name not in TOOLS:
            return f"Unknown tool: {tool_name}"
        func: Any = TOOLS[tool_name]
        try:
            if isinstance(args, dict):
                return func(**args)
            elif isinstance(args, list):
                return func(args)
            else:
                return func(args)
        except Exception as e:
            return f"Tool error: {e}"

    def _build_failsafe_final(
        self,
        cid: str,
        last_rag_result: Optional[dict[str, Any]],
        reason: str,
    ) -> dict[str, Any]:
        if isinstance(last_rag_result, dict):
            fallback = dict(last_rag_result)
            fallback.setdefault("cell_type", "Unknown")
            fallback.setdefault("subtype", None)
            raw_conf = fallback.get("confidence", 0.0)
            try:
                conf = float(raw_conf) if raw_conf is not None else 0.0
            except Exception:
                conf = 0.0
            fallback["confidence"] = max(0.0, min(1.0, conf))
            fallback.setdefault("key_markers", [])
            fallback.setdefault("go_terms", [])
            fallback.setdefault("kegg_pathways", [])
            fallback.setdefault("attribution", "Fallback to last rag_annotate result due to agent control failure.")
            fallback.setdefault("supporting_references", [])
            fallback["agent_warning"] = reason
            fallback["cluster_id"] = cid
            return fallback
        return {
            "cluster_id": cid,
            "cell_type": "Unknown",
            "subtype": None,
            "confidence": 0.0,
            "key_markers": [],
            "go_terms": [],
            "kegg_pathways": [],
            "attribution": "Agent failed to produce a valid final answer.",
            "supporting_references": [],
            "error": reason,
        }

    def _apply_enrichment_gate(
        self,
        payload: dict[str, Any],
        go_called: bool,
        kegg_called: bool,
    ) -> dict[str, Any]:
        """仅在调用过对应富集工具后保留字段，否则强制置空。"""
        if not go_called:
            payload["go_terms"] = []
        if not kegg_called:
            payload["kegg_pathways"] = []
        return payload

    def _parse_go_terms_observation(self, observation: str) -> list[str]:
        lines = [ln.strip().lstrip("-").strip() for ln in (observation or "").splitlines() if ln.strip()]
        if not lines:
            return []
        head = lines[0].lower()
        if (
            head.startswith("no ")
            or head.startswith("go query failed")
            or head.startswith("tool error:")
            or head.startswith("unknown tool:")
        ):
            return []
        return lines[:10]

    def _parse_kegg_observation(self, observation: str) -> list[str]:
        lines = [ln.strip().lstrip("-").strip() for ln in (observation or "").splitlines() if ln.strip()]
        if not lines:
            return []
        head = lines[0].lower()
        if (
            head.startswith("no ")
            or head.startswith("tool error:")
            or head.startswith("unknown tool:")
        ):
            return []
        parsed: list[str] = []
        for line in lines:
            text = line
            if ":" in line:
                text = line.split(":", 1)[1].strip()
            text = re.sub(r"\s*\(hit by \d+ genes\)\s*$", "", text).strip()
            if text:
                parsed.append(text)
        return parsed[:10]

    def run(self, cluster_info: dict[str, Any]) -> dict[str, Any]:
        """
        运行 Agent 对单个簇进行注释归因

        cluster_info: {
            "cluster_id": str,
            "markers": list[str],
            "tissue": str,
            "technology": str,
        }
        """
        markers   = cluster_info.get("markers", [])
        tissue    = cluster_info.get("tissue", "unknown")
        technology= cluster_info.get("technology", "scRNA-seq")
        cid       = cluster_info.get("cluster_id", "0")
        cluster_payload = {
            "cluster_id": cid,
            "markers": markers,
            "tissue": tissue,
            "technology": technology,
        }
        run_id = self.memory_store.new_run_id()
        run_started = time.time()
        trace = AgentTrace(run_id, enabled=self.trace_enabled)
        trace.write_input(cluster_payload)
        trace.add_event(
            "run_start",
            backend=self.tool_backend,
            memory_enabled=self.memory_enabled,
            cache_enabled=self.cache_enabled,
        )

        if self.memory_enabled and self.reuse_exact_match:
            exact = self.memory_store.lookup_exact(cluster_payload)
            if exact is not None:
                exact = dict(exact)
                exact["cluster_id"] = cid
                exact["memory_hit"] = "exact"
                trace.add_memory_hit("exact", {"summary": "Reused exact historical final result."})
                duration_ms = int((time.time() - run_started) * 1000)
                self.memory_store.record_run(
                    run_id=run_id,
                    cluster=cluster_payload,
                    backend=self.tool_backend,
                    final=exact,
                    status="memory_hit",
                    fallback_reason=None,
                    duration_ms=duration_ms,
                )
                trace.finalize(exact, status="memory_hit")
                return exact

        similar_memories = []
        if self.memory_enabled:
            similar_memories = self.memory_store.search_similar(
                cluster_payload,
                threshold=AGENT_SIMILARITY_THRESHOLD,
                limit=3,
            )
            for memory_hit in similar_memories:
                trace.add_memory_hit("similar", memory_hit)

        state = "INIT"
        observations: list[dict[str, str]] = []
        final_answer: Optional[dict[str, Any]] = None
        last_rag_result: Optional[dict[str, Any]] = None
        validation_passed = False
        parse_fail_count = 0
        empty_resp_count = 0
        tool_error_count = 0
        fail_reason = "Agent max steps exceeded"
        agent_step = 0
        go_enrichment_called = False
        kegg_enrichment_called = False

        # 必经步骤：由程序直接执行 rag_annotate，不由 LLM 决定。
        rag_args = {
            "markers": markers if isinstance(markers, list) else [],
            "tissue": tissue,
            "technology": technology,
            "cluster_id": cid,
        }
        rag_call = self._call_tool_result("rag_annotate", rag_args, run_id=run_id, step_index=0)
        trace.add_tool_call(rag_call.trace_payload(step_index=0))
        rag_observation = rag_call.result
        rag_obs_text = str(rag_observation.get("display_text", rag_observation))[:2000]
        observations.append({"tool": "rag_annotate", "observation": rag_obs_text})
        self.history.append({"step": 0, "state": "INIT", "tool": "rag_annotate", "observation": rag_obs_text})
        if not rag_call.ok:
            fail_reason = (rag_call.error or "rag_annotate failed")[:200]
        else:
            parsed_rag = rag_observation.get("annotation")
            if isinstance(parsed_rag, dict):
                last_rag_result = self._apply_enrichment_gate(
                    parsed_rag,
                    go_called=go_enrichment_called,
                    kegg_called=kegg_enrichment_called,
                )
            else:
                fail_reason = "Program pre-run rag_annotate returned non-object JSON."

        while agent_step < self.max_steps:
            if state == "INIT":
                state = "ACT"

            if state == "EVAL":
                # 程序侧自动验证当前状态，避免依赖 LLM 再生成 FINAL JSON。
                if isinstance(last_rag_result, dict):
                    auto_validation = tool_validate_annotation(
                        annotation_json=json.dumps(last_rag_result, ensure_ascii=False)
                    )
                    if auto_validation.startswith("Annotation validated."):
                        validation_passed = True
                # 终止条件：validate 通过后优先返回可校验的 RAG 结果；否则继续推进。
                if validation_passed and isinstance(last_rag_result, dict):
                    validated_fallback, fallback_error = self._validate_final_payload(last_rag_result)
                    if validated_fallback is not None and fallback_error is None:
                        final_answer = validated_fallback
                        break
                if validation_passed or agent_step >= self.max_steps - 1:
                    state = "FINALIZE"
                else:
                    state = "ACT"
                continue

            expected_type = "FINAL" if state == "FINALIZE" else "ACTION|FINAL"
            llm_user = self._build_step_prompt(
                state=state,
                cluster_info=cluster_payload,
                observations=observations,
                expected_type=expected_type,
                memory_context=similar_memories,
            )
            try:
                response = self._call_llm_cached(REACT_SYSTEM, llm_user, run_id=run_id)
                trace.add_event(
                    "llm_call",
                    state=state,
                    status="ok",
                    cache_hit=self._last_llm_cache_hit,
                    model=get_llm_runtime_config()["model"],
                )
            except Exception as exc:
                fail_reason = f"LLM call failed: {exc}"
                trace.add_event("llm_call", state=state, status="error", error=str(exc))
                break
            log.info(f"Agent step {agent_step + 1}/{self.max_steps} state={state} response={response[:300]}")

            if not response or not response.strip():
                empty_resp_count += 1
                fail_reason = "Empty LLM response."
                if empty_resp_count > self.empty_resp_budget:
                    break
                # 空响应只消耗 empty 预算，不消耗主步骤预算
                continue

            envelope = self._parse_envelope(response)
            if envelope is None:
                parse_fail_count += 1
                fail_reason = "Unparseable LLM envelope."
                self._mark_last_llm_response_unusable(run_id, "parse_error", fail_reason)
                # FINALIZE 阶段若出现截断，优先回退到已存在且可校验的 RAG 结果
                if state == "FINALIZE" and isinstance(last_rag_result, dict):
                    validated_fallback, fallback_error = self._validate_final_payload(last_rag_result)
                    if validated_fallback is not None and fallback_error is None:
                        final_answer = validated_fallback
                        break
                if parse_fail_count > self.parse_fail_budget:
                    break
                # 不可解析只消耗 parse 预算，不消耗主步骤预算
                continue

            envelope_type = str(envelope.get("type", "")).upper().strip()

            if state == "FINALIZE" and envelope_type != "FINAL":
                parse_fail_count += 1
                fail_reason = "FINALIZE state requires type=FINAL."
                self._mark_last_llm_response_unusable(run_id, "parse_error", fail_reason)
                if parse_fail_count > self.parse_fail_budget:
                    break
                continue

            if envelope_type == "ACTION":
                parsed_action, action_error = self._validate_action_envelope(envelope)
                if action_error or parsed_action is None:
                    parse_fail_count += 1
                    fail_reason = action_error or "Invalid ACTION envelope."
                    self._mark_last_llm_response_unusable(run_id, "parse_error", fail_reason)
                    if parse_fail_count > self.parse_fail_budget:
                        break
                    continue

                tool_name, args = parsed_action
                if tool_name == "validate_annotation":
                    if not isinstance(last_rag_result, dict):
                        tool_call = ToolCallResult(
                            tool_name=tool_name,
                            args=args,
                            result={
                                "ok": False,
                                "error": "validate_annotation requires last_rag_result in state.",
                                "display_text": "Tool error: validate_annotation requires last_rag_result in state.",
                            },
                            ok=False,
                            error="validate_annotation requires last_rag_result in state.",
                            duration_ms=0,
                            cache_hit=False,
                        )
                    else:
                        tool_call = self._call_tool_result(
                            "validate_annotation",
                            {"annotation": last_rag_result},
                            run_id=run_id,
                            step_index=agent_step + 1,
                        )
                else:
                    tool_call = self._call_tool_result(tool_name, args, run_id=run_id, step_index=agent_step + 1)
                trace.add_tool_call(tool_call.trace_payload(step_index=agent_step + 1))
                observation_payload = tool_call.result
                observation = str(observation_payload.get("display_text", observation_payload))
                obs_text = observation[:2000]
                observations.append({"tool": tool_name, "observation": obs_text})
                self.history.append({"step": agent_step + 1, "state": state, "tool": tool_name, "observation": obs_text})

                if tool_name == "query_go_terms" and tool_call.ok:
                    go_enrichment_called = True
                    if isinstance(last_rag_result, dict):
                        terms = observation_payload.get("terms") if isinstance(observation_payload, dict) else None
                        if isinstance(terms, list):
                            last_rag_result["go_terms"] = [
                                f"{t.get('go_id', '')} {t.get('name', '')}".strip()
                                for t in terms
                                if isinstance(t, dict)
                            ]
                        else:
                            last_rag_result["go_terms"] = self._parse_go_terms_observation(observation)
                        self._apply_enrichment_gate(
                            last_rag_result,
                            go_called=go_enrichment_called,
                            kegg_called=kegg_enrichment_called,
                        )
                if tool_name == "query_kegg_pathways" and tool_call.ok:
                    kegg_enrichment_called = True
                    if isinstance(last_rag_result, dict):
                        pathways = observation_payload.get("pathways") if isinstance(observation_payload, dict) else None
                        if isinstance(pathways, list):
                            last_rag_result["kegg_pathways"] = [
                                str(p.get("name", "")).strip()
                                for p in pathways
                                if isinstance(p, dict) and p.get("name")
                            ]
                        else:
                            last_rag_result["kegg_pathways"] = self._parse_kegg_observation(observation)
                        self._apply_enrichment_gate(
                            last_rag_result,
                            go_called=go_enrichment_called,
                            kegg_called=kegg_enrichment_called,
                        )

                if tool_name == "validate_annotation" and bool(observation_payload.get("valid", False)):
                    validation_passed = True

                if not tool_call.ok:
                    tool_error_count += 1
                    fail_reason = (tool_call.error or observation)[:200]
                    if tool_error_count > self.tool_error_budget:
                        break

                agent_step += 1
                state = "EVAL"
                continue

            if envelope_type == "FINAL":
                # FINAL 由程序基于 last_rag_result 生成，不消费 LLM final.payload。
                if isinstance(last_rag_result, dict):
                    validated_fallback, fallback_error = self._validate_final_payload(last_rag_result)
                    if validated_fallback is not None and fallback_error is None:
                        final_answer = self._apply_enrichment_gate(
                            validated_fallback,
                            go_called=go_enrichment_called,
                            kegg_called=kegg_enrichment_called,
                        )
                        agent_step += 1
                        break
                llm_final = envelope.get("final") if isinstance(envelope, dict) else None
                validated_llm_final, llm_final_error = self._validate_final_payload(llm_final)
                if validated_llm_final is not None and llm_final_error is None:
                    final_answer = self._apply_enrichment_gate(
                        validated_llm_final,
                        go_called=go_enrichment_called,
                        kegg_called=kegg_enrichment_called,
                    )
                    final_answer["agent_warning"] = (
                        "Used schema-validated LLM final because rag_annotate did not produce a valid result."
                    )
                    agent_step += 1
                    break
                parse_fail_count += 1
                fail_reason = "Program-controlled finalize requires valid last_rag_result."
                self._mark_last_llm_response_unusable(run_id, "parse_error", fail_reason)
                if parse_fail_count > self.parse_fail_budget:
                    break
                state = "ACT"
                continue

            parse_fail_count += 1
            fail_reason = f"Invalid envelope.type: {envelope.get('type')}"
            self._mark_last_llm_response_unusable(run_id, "parse_error", fail_reason)
            if parse_fail_count > self.parse_fail_budget:
                break

        if final_answer is None:
            log.warning(f"Agent failsafe triggered: {fail_reason}")
            final_answer = self._build_failsafe_final(
                cid=cid,
                last_rag_result=last_rag_result,
                reason=fail_reason,
            )

        assert final_answer is not None
        final_answer = self._apply_enrichment_gate(
            final_answer,
            go_called=go_enrichment_called,
            kegg_called=kegg_enrichment_called,
        )

        final_answer["cluster_id"] = cid
        status = "ok" if "error" not in final_answer else "error"
        fallback_reason = final_answer.get("agent_warning") or final_answer.get("error")
        duration_ms = int((time.time() - run_started) * 1000)
        if self.memory_enabled:
            self.memory_store.record_run(
                run_id=run_id,
                cluster=cluster_payload,
                backend=self.tool_backend,
                final=final_answer,
                status=status,
                fallback_reason=fallback_reason,
                duration_ms=duration_ms,
            )
            if status == "ok":
                self.memory_store.upsert_task_memory(run_id, cluster_payload, final_answer)
        trace.finalize(final_answer, status=status, fallback_reason=fallback_reason)
        return final_answer


# ─── CLI 入口 ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    agent = SingleCellAgent()

    test_clusters = [
        {
            "cluster_id": "cluster_0",
            "markers": ["CD8A", "CD8B", "GZMB", "GZMK", "PRF1", "PDCD1", "LAG3", "HAVCR2"],
            "tissue": "lung adenocarcinoma",
            "technology": "scRNA-seq",
        },
        {
            "cluster_id": "cluster_1",
            "markers": ["FOXP3", "IL2RA", "CTLA4", "IKZF2", "TNFRSF9", "TIGIT"],
            "tissue": "colorectal cancer",
            "technology": "scRNA-seq",
        },
    ]

    for cluster in test_clusters:
        print(f"\n{'='*60}")
        print(f"Annotating {cluster['cluster_id']}")
        print(f"{'='*60}")
        result = agent.run(cluster)
        print(json.dumps(result, indent=2, ensure_ascii=False))
