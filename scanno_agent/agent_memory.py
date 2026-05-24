from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

try:
    from config import RESULTS_DIR
except ImportError:  # pragma: no cover - package execution fallback
    from .config import RESULTS_DIR


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_hash(value: Any) -> str:
    import hashlib

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def normalize_markers(markers: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    normalized: list[str] = []
    for marker in markers or []:
        if not isinstance(marker, str):
            continue
        gene = marker.strip().upper()
        if gene and gene not in seen:
            seen.add(gene)
            normalized.append(gene)
    return normalized


def normalize_cluster(cluster: dict[str, Any]) -> dict[str, Any]:
    return {
        "cluster_id": str(cluster.get("cluster_id", "")),
        "markers": normalize_markers(cluster.get("markers", [])),
        "tissue": str(cluster.get("tissue", "unknown")).strip().lower(),
        "technology": str(cluster.get("technology", "scRNA-seq")).strip().lower(),
    }


def task_fingerprint(cluster: dict[str, Any]) -> str:
    return stable_hash(normalize_cluster(cluster))


def marker_jaccard(a: Iterable[Any], b: Iterable[Any]) -> float:
    left = set(normalize_markers(a))
    right = set(normalize_markers(b))
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


class AgentMemoryStore:
    def __init__(self, db_path: str | Path | None = None, enabled: bool = True):
        self.enabled = enabled
        self.db_path = Path(db_path) if db_path else RESULTS_DIR / "agent_memory.sqlite"
        if self.enabled:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        # Some shared Windows workspaces disallow journal deletion; MEMORY keeps
        # this demo-friendly store usable without extra filesystem privileges.
        conn.execute("PRAGMA journal_mode=MEMORY")
        conn.execute("PRAGMA synchronous=OFF")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS agent_runs (
                    run_id TEXT PRIMARY KEY,
                    task_hash TEXT NOT NULL,
                    backend TEXT NOT NULL,
                    input_json TEXT NOT NULL,
                    final_json TEXT,
                    status TEXT NOT NULL,
                    fallback_reason TEXT,
                    duration_ms INTEGER DEFAULT 0,
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_agent_runs_task_hash
                    ON agent_runs(task_hash);

                CREATE TABLE IF NOT EXISTS tool_calls (
                    call_id TEXT PRIMARY KEY,
                    run_id TEXT,
                    step_index INTEGER DEFAULT 0,
                    tool_name TEXT NOT NULL,
                    args_hash TEXT NOT NULL,
                    args_json TEXT NOT NULL,
                    result_json TEXT,
                    status TEXT NOT NULL,
                    error TEXT,
                    duration_ms INTEGER DEFAULT 0,
                    cache_hit INTEGER DEFAULT 0,
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_tool_calls_cache
                    ON tool_calls(tool_name, args_hash, status, created_at);

                CREATE TABLE IF NOT EXISTS llm_calls (
                    call_id TEXT PRIMARY KEY,
                    run_id TEXT,
                    model TEXT NOT NULL,
                    prompt_hash TEXT NOT NULL,
                    prompt_json TEXT NOT NULL,
                    response_text TEXT,
                    status TEXT NOT NULL,
                    error TEXT,
                    duration_ms INTEGER DEFAULT 0,
                    cache_hit INTEGER DEFAULT 0,
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_llm_calls_cache
                    ON llm_calls(model, prompt_hash, status, created_at);

                CREATE TABLE IF NOT EXISTS task_memory (
                    task_hash TEXT PRIMARY KEY,
                    markers_json TEXT NOT NULL,
                    tissue TEXT NOT NULL,
                    technology TEXT NOT NULL,
                    final_json TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );
                """
            )

    def new_run_id(self) -> str:
        return time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]

    def lookup_exact(self, cluster: dict[str, Any]) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        task_hash = task_fingerprint(cluster)
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT final_json FROM task_memory
                WHERE task_hash = ?
                """,
                (task_hash,),
            ).fetchone()
        if not row:
            return None
        return json.loads(row["final_json"])

    def search_similar(
        self,
        cluster: dict[str, Any],
        threshold: float = 0.75,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        normalized = normalize_cluster(cluster)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT task_hash, markers_json, tissue, technology, final_json,
                       summary, run_id, updated_at
                FROM task_memory
                ORDER BY updated_at DESC
                LIMIT 200
                """
            ).fetchall()

        matches: list[dict[str, Any]] = []
        for row in rows:
            markers = json.loads(row["markers_json"])
            score = marker_jaccard(normalized["markers"], markers)
            if score < threshold:
                continue
            matches.append(
                {
                    "task_hash": row["task_hash"],
                    "similarity": score,
                    "markers": markers,
                    "tissue": row["tissue"],
                    "technology": row["technology"],
                    "summary": row["summary"],
                    "final": json.loads(row["final_json"]),
                    "run_id": row["run_id"],
                }
            )
        matches.sort(key=lambda x: x["similarity"], reverse=True)
        return matches[:limit]

    def get_tool_cache(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        args_hash = stable_hash(args)
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT result_json FROM tool_calls
                WHERE tool_name = ? AND args_hash = ? AND status = 'ok'
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (tool_name, args_hash),
            ).fetchone()
        if not row or not row["result_json"]:
            return None
        return json.loads(row["result_json"])

    def record_tool_call(
        self,
        run_id: str,
        step_index: int,
        tool_name: str,
        args: dict[str, Any],
        result: dict[str, Any] | None,
        status: str,
        error: str | None,
        duration_ms: int,
        cache_hit: bool,
    ) -> None:
        if not self.enabled:
            return
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO tool_calls (
                    call_id, run_id, step_index, tool_name, args_hash, args_json,
                    result_json, status, error, duration_ms, cache_hit, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    uuid.uuid4().hex,
                    run_id,
                    step_index,
                    tool_name,
                    stable_hash(args),
                    canonical_json(args),
                    canonical_json(result) if result is not None else None,
                    status,
                    error,
                    int(duration_ms),
                    1 if cache_hit else 0,
                    time.time(),
                ),
            )

    def get_llm_cache(self, model: str, prompt_payload: dict[str, Any]) -> str | None:
        if not self.enabled:
            return None
        prompt_hash = stable_hash(prompt_payload)
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT response_text, status FROM llm_calls
                WHERE model = ? AND prompt_hash = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (model, prompt_hash),
            ).fetchone()
        if row is None or row["status"] != "ok":
            return None
        if not row["response_text"] or not str(row["response_text"]).strip():
            return None
        return row["response_text"]

    def record_llm_call(
        self,
        run_id: str,
        model: str,
        prompt_payload: dict[str, Any],
        response_text: str | None,
        status: str,
        error: str | None,
        duration_ms: int,
        cache_hit: bool,
    ) -> None:
        if not self.enabled:
            return
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO llm_calls (
                    call_id, run_id, model, prompt_hash, prompt_json,
                    response_text, status, error, duration_ms, cache_hit, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    uuid.uuid4().hex,
                    run_id,
                    model,
                    stable_hash(prompt_payload),
                    canonical_json(prompt_payload),
                    response_text,
                    status,
                    error,
                    int(duration_ms),
                    1 if cache_hit else 0,
                    time.time(),
                ),
            )

    def record_run(
        self,
        run_id: str,
        cluster: dict[str, Any],
        backend: str,
        final: dict[str, Any] | None,
        status: str,
        fallback_reason: str | None,
        duration_ms: int,
    ) -> None:
        if not self.enabled:
            return
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO agent_runs (
                    run_id, task_hash, backend, input_json, final_json, status,
                    fallback_reason, duration_ms, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    task_fingerprint(cluster),
                    backend,
                    canonical_json(cluster),
                    canonical_json(final) if final is not None else None,
                    status,
                    fallback_reason,
                    int(duration_ms),
                    time.time(),
                ),
            )

    def upsert_task_memory(self, run_id: str, cluster: dict[str, Any], final: dict[str, Any]) -> None:
        if not self.enabled:
            return
        normalized = normalize_cluster(cluster)
        summary = self._build_summary(normalized, final)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO task_memory (
                    task_hash, markers_json, tissue, technology, final_json,
                    summary, run_id, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task_fingerprint(cluster),
                    canonical_json(normalized["markers"]),
                    normalized["tissue"],
                    normalized["technology"],
                    canonical_json(final),
                    summary,
                    run_id,
                    time.time(),
                ),
            )

    @staticmethod
    def _build_summary(cluster: dict[str, Any], final: dict[str, Any]) -> str:
        cell_type = final.get("cell_type", "Unknown")
        subtype = final.get("subtype")
        confidence = final.get("confidence", 0)
        markers = ", ".join(cluster.get("markers", [])[:8])
        label = f"{cell_type}" + (f" ({subtype})" if subtype else "")
        return (
            f"Past task markers=[{markers}], tissue={cluster.get('tissue')}, "
            f"technology={cluster.get('technology')} -> {label}, confidence={confidence}."
        )
