from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
PKG_DIR = ROOT / "scanno_agent"
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))

from config import RESULTS_DIR  # noqa: E402


def load_clusters(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return [payload]
    return payload


def variant_config(name: str) -> dict[str, Any]:
    normalized = name.strip().lower()
    if normalized in {"baseline", "local"}:
        return {"tool_backend": "local", "memory_enabled": False, "cache_enabled": False, "reuse_exact_match": False}
    if normalized in {"agent_memory", "mcp_memory"}:
        return {"tool_backend": "mcp", "memory_enabled": True, "cache_enabled": True, "reuse_exact_match": True}
    if normalized == "local_memory":
        return {"tool_backend": "local", "memory_enabled": True, "cache_enabled": True, "reuse_exact_match": True}
    raise ValueError(f"Unknown variant: {name}")


def latest_run_dir(before: set[Path]) -> Path | None:
    runs_dir = RESULTS_DIR / "runs"
    if not runs_dir.exists():
        return None
    candidates = [p for p in runs_dir.iterdir() if p.is_dir() and p not in before]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def summarize_trace(run_dir: Path | None) -> dict[str, Any]:
    if not run_dir:
        return {
            "tool_success_count": 0,
            "tool_call_count": 0,
            "cache_hit_count": 0,
            "memory_hit_count": 0,
            "llm_call_count": 0,
            "fallback": False,
        }
    tool_calls_path = run_dir / "tool_calls.json"
    memory_hits_path = run_dir / "memory_hits.json"
    trace_path = run_dir / "trace.json"
    tool_calls = json.loads(tool_calls_path.read_text(encoding="utf-8")) if tool_calls_path.exists() else []
    memory_hits = json.loads(memory_hits_path.read_text(encoding="utf-8")) if memory_hits_path.exists() else []
    trace = json.loads(trace_path.read_text(encoding="utf-8")) if trace_path.exists() else {}
    return {
        "tool_success_count": sum(1 for call in tool_calls if call.get("status") == "ok"),
        "tool_call_count": len(tool_calls),
        "cache_hit_count": sum(1 for call in tool_calls if call.get("cache_hit")),
        "memory_hit_count": len(memory_hits),
        "llm_call_count": sum(1 for event in trace.get("events", []) if event.get("name") == "llm_call"),
        "fallback": bool(trace.get("fallback_reason")),
    }


def run_variant(name: str, clusters: list[dict[str, Any]]) -> dict[str, Any]:
    from annotation_agent import SingleCellAgent

    cfg = variant_config(name)
    outputs = []
    durations = []
    trace_summaries = []
    runs_dir = RESULTS_DIR / "runs"
    before = set(runs_dir.iterdir()) if runs_dir.exists() else set()

    for cluster in clusters:
        agent = SingleCellAgent(**cfg)
        started = time.time()
        try:
            final = agent.run(cluster)
            error = None
        except Exception as exc:
            final = {"error": str(exc)}
            error = str(exc)
        durations.append(int((time.time() - started) * 1000))
        run_dir = latest_run_dir(before)
        if run_dir:
            before.add(run_dir)
        trace_summaries.append(summarize_trace(run_dir))
        outputs.append({"final": final, "error": error})

    total = max(len(outputs), 1)
    json_valid = sum(1 for out in outputs if isinstance(out["final"], dict) and "cell_type" in out["final"])
    tool_calls = sum(x["tool_call_count"] for x in trace_summaries)
    tool_success = sum(x["tool_success_count"] for x in trace_summaries)
    cache_hits = sum(x["cache_hit_count"] for x in trace_summaries)
    memory_hits = sum(x["memory_hit_count"] for x in trace_summaries)
    llm_calls = sum(x["llm_call_count"] for x in trace_summaries)
    fallbacks = sum(1 for x in trace_summaries if x["fallback"])

    return {
        "variant": name,
        "json_valid_rate": json_valid / total,
        "tool_success_rate": (tool_success / tool_calls) if tool_calls else 0.0,
        "cache_hit_rate": (cache_hits / tool_calls) if tool_calls else 0.0,
        "memory_hit_rate": memory_hits / total,
        "fallback_rate": fallbacks / total,
        "average_latency_ms": sum(durations) / total,
        "llm_call_count": llm_calls,
        "tool_call_count": tool_calls,
        "outputs": outputs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Agent variants.")
    parser.add_argument("--input", required=True, help="Cluster JSON path.")
    parser.add_argument("--variants", default="baseline,agent_memory", help="Comma-separated variants.")
    args = parser.parse_args()

    clusters = load_clusters(args.input)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    results = [run_variant(variant, clusters) for variant in variants]

    out_dir = RESULTS_DIR / "eval" / time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Agent Evaluation Summary", ""]
    for row in results:
        lines.append(
            f"- {row['variant']}: json_valid={row['json_valid_rate']:.2f}, "
            f"tool_success={row['tool_success_rate']:.2f}, cache_hit={row['cache_hit_rate']:.2f}, "
            f"memory_hit={row['memory_hit_rate']:.2f}, fallback={row['fallback_rate']:.2f}, "
            f"latency_ms={row['average_latency_ms']:.0f}"
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote eval results to {out_dir}")


if __name__ == "__main__":
    main()
