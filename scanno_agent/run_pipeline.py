"""
run_pipeline.py — 一键运行完整 Pipeline
──────────────────────────────────────────
Usage:
  # 完整流程（爬取 → 解析 → 切片 → 索引 → 注释）
  python run_pipeline.py --all

  # 仅运行注释（已有索引）
  python run_pipeline.py --annotate --input clusters.json

  # 单步运行
  python run_pipeline.py --step crawl
  python run_pipeline.py --step parse
  python run_pipeline.py --step chunk
  python run_pipeline.py --step index
"""
from __future__ import annotations
import json
import argparse
import logging
from typing import Any
from pathlib import Path
from datetime import datetime

from config import RESULTS_DIR  # pyright: ignore[reportImplicitRelativeImport]

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(RESULTS_DIR / "pipeline.log"),
    ]
)

EXAMPLE_CLUSTERS = [
    {
        "cluster_id": "cluster_0",
        "markers": ["CD8A", "CD8B", "GZMB", "GZMK", "PRF1", "PDCD1", "LAG3", "HAVCR2"],
        "tissue": "lung adenocarcinoma tumor microenvironment",
        "technology": "scRNA-seq",
    },
    {
        "cluster_id": "cluster_1",
        "markers": ["FOXP3", "IL2RA", "CTLA4", "IKZF2", "TNFRSF9", "TIGIT", "IL10"],
        "tissue": "colorectal cancer",
        "technology": "scRNA-seq",
    },
    {
        "cluster_id": "cluster_2",
        "markers": ["CD14", "CD68", "CSF1R", "MRC1", "APOE", "C1QA", "C1QB", "TREM2"],
        "tissue": "lung adenocarcinoma",
        "technology": "scRNA-seq",
    },
    {
        "cluster_id": "cluster_3",
        "markers": ["EPCAM", "KRT8", "KRT18", "KRT19", "CDH1", "MUC1", "ERBB2"],
        "tissue": "breast cancer",
        "technology": "scRNA-seq",
    },
]


def normalize_confidence(value: object) -> float:
    """将模型返回的 confidence 统一转换为 0~1 浮点数。"""
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))

    if isinstance(value, str):
        s = value.strip().lower()
        level_map = {
            "high": 0.85,
            "medium": 0.60,
            "low": 0.30,
        }
        if s in level_map:
            return level_map[s]
        try:
            return max(0.0, min(1.0, float(s)))
        except ValueError:
            pass

    return 0.0


def step_crawl():
    log.info("=" * 50 + " STEP 1: CRAWL " + "=" * 50)
    import importlib
    paper_crawler = importlib.import_module("paper_crawler")
    paper_crawler.crawl_and_download()


def step_parse():
    log.info("=" * 50 + " STEP 2: PARSE " + "=" * 50)
    import importlib
    pdf_parser = importlib.import_module("pdf_parser")
    pdf_parser.parse_all()


def step_chunk():
    log.info("=" * 50 + " STEP 3: CHUNK " + "=" * 50)
    import importlib
    chunker = importlib.import_module("chunker")
    chunker.chunk_all()


def step_index():
    log.info("=" * 50 + " STEP 4: INDEX " + "=" * 50)
    import importlib
    embedding_indexer = importlib.import_module("embedding_indexer")
    embedding_indexer.build_index()


def step_annotate(
    clusters: list[dict[str, Any]],
    use_agent: bool = True,
    agent_backend: str | None = None,
    memory: str = "env",
):
    log.info("=" * 50 + " STEP 5: ANNOTATE " + "=" * 50)
    results = []

    if use_agent:
        import importlib
        annotation_agent = importlib.import_module("annotation_agent")
        memory_enabled = None if memory == "env" else memory == "on"
        agent = annotation_agent.SingleCellAgent(
            tool_backend=agent_backend,
            memory_enabled=memory_enabled,
        )
        for cluster in clusters:
            log.info(f"Agent annotating {cluster['cluster_id']}...")
            result = agent.run(cluster)
            results.append(result)
    else:
        import importlib
        rag_pipeline = importlib.import_module("rag_pipeline")
        pipeline = rag_pipeline.SingleCellRAGPipeline()
        results = pipeline.annotate_batch(clusters)

    # 保存结果
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"annotations_{ts}.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    log.info(f"注释结果已保存: {out_path}")

    # 打印摘要
    print("\n" + "=" * 60)
    print("ANNOTATION SUMMARY")
    print("=" * 60)
    for r in results:
        cid        = r.get("cluster_id", "?")
        cell_type  = r.get("cell_type", "Unknown")
        subtype    = r.get("subtype") or ""
        confidence = normalize_confidence(r.get("confidence", 0))
        r["confidence"] = confidence
        label = f"{cell_type}" + (f" ({subtype})" if subtype else "")
        bar = "█" * int(confidence * 20) + "░" * (20 - int(confidence * 20))
        print(f"  {cid:<15} {label:<35} [{bar}] {confidence:.2f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Single-cell RAG Annotation Pipeline")
    parser.add_argument("--all",       action="store_true", help="运行完整 pipeline")
    parser.add_argument("--step",      choices=["crawl", "parse", "chunk", "index"])
    parser.add_argument("--annotate",  action="store_true", help="仅运行注释步骤")
    parser.add_argument("--input",     type=str, default=None, help="簇信息 JSON 文件路径")
    parser.add_argument("--no-agent",  action="store_true", help="使用简单 RAG 而非 Agent")
    parser.add_argument("--agent-backend", choices=["local", "mcp"], default=None,
                        help="Agent tool backend: local or mcp")
    parser.add_argument("--memory", choices=["env", "on", "off"], default="env",
                        help="Agent memory mode")
    args = parser.parse_args()

    # 加载输入簇信息
    if args.input:
        clusters = json.loads(Path(args.input).read_text())
    else:
        clusters = EXAMPLE_CLUSTERS

    if args.all:
        step_crawl()
        step_parse()
        step_chunk()
        step_index()
        step_annotate(
            clusters,
            use_agent=not args.no_agent,
            agent_backend=args.agent_backend,
            memory=args.memory,
        )

    elif args.step == "crawl":
        step_crawl()
    elif args.step == "parse":
        step_parse()
    elif args.step == "chunk":
        step_chunk()
    elif args.step == "index":
        step_index()
    elif args.annotate:
        step_annotate(
            clusters,
            use_agent=not args.no_agent,
            agent_backend=args.agent_backend,
            memory=args.memory,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
