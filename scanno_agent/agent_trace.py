from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

try:
    from config import RESULTS_DIR
except ImportError:  # pragma: no cover
    from .config import RESULTS_DIR


def _json_dump(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class AgentTrace:
    def __init__(self, run_id: str, enabled: bool = True):
        self.run_id = run_id
        self.enabled = enabled
        self.started_at = time.time()
        self.events: list[dict[str, Any]] = []
        self.tool_calls: list[dict[str, Any]] = []
        self.memory_hits: list[dict[str, Any]] = []
        self.run_dir = RESULTS_DIR / "runs" / run_id
        if self.enabled:
            self.run_dir.mkdir(parents=True, exist_ok=True)

    def write_input(self, cluster: dict[str, Any]) -> None:
        if self.enabled:
            _json_dump(self.run_dir / "input.json", cluster)

    def add_event(self, name: str, **fields: Any) -> None:
        self.events.append(
            {
                "ts": time.time(),
                "name": name,
                **fields,
            }
        )

    def add_memory_hit(self, kind: str, hit: dict[str, Any]) -> None:
        payload = {"kind": kind, **hit}
        self.memory_hits.append(payload)
        self.add_event("memory_hit", kind=kind, summary=hit.get("summary"), cache_hit=True)

    def add_tool_call(self, call: dict[str, Any]) -> None:
        self.tool_calls.append(call)
        self.add_event(
            "tool_call",
            tool_name=call.get("tool_name"),
            status=call.get("status"),
            cache_hit=call.get("cache_hit", False),
            duration_ms=call.get("duration_ms", 0),
            error=call.get("error"),
        )

    def finalize(self, final: dict[str, Any], status: str, fallback_reason: str | None = None) -> None:
        if not self.enabled:
            return
        duration_ms = int((time.time() - self.started_at) * 1000)
        _json_dump(self.run_dir / "final.json", final)
        _json_dump(self.run_dir / "memory_hits.json", self.memory_hits)
        _json_dump(self.run_dir / "tool_calls.json", self.tool_calls)
        _json_dump(
            self.run_dir / "trace.json",
            {
                "run_id": self.run_id,
                "status": status,
                "fallback_reason": fallback_reason,
                "duration_ms": duration_ms,
                "events": self.events,
            },
        )

