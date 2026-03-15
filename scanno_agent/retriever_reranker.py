"""
Step 5 — 检索 + Rerank
───────────────────────
两阶段检索：
  Stage 1: ChromaDB 向量近邻搜索（top-N = 20）
  Stage 2: BGE Reranker 精排（取 top-K = 5）

支持混合查询扩展：
  • 原始 query
  • 生物学术语归一化（gene symbol 大写等）
  • 查询扩展（同义词补充）
"""
from __future__ import annotations
import re
import logging
from dataclasses import dataclass, field
from typing import Optional

from FlagEmbedding import FlagReranker

from config import (
    RERANK_MODEL, RERANK_TOP_K, RETRIEVE_TOP_N,
    EMBED_DEVICE,
)
from embedding_indexer import get_embed_model, get_chroma_collection, embed_texts

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ─── 数据结构 ─────────────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    chunk_id:    str
    pmid:        str
    title:       str
    year:        str
    section:     str
    text:        str
    score:       float          # rerank 分数（越高越相关）
    vector_dist: float = 0.0    # 初步向量距离

    def to_context_str(self) -> str:
        return (
            f"[Source: {self.title[:60]} ({self.year}), Section: {self.section}]\n"
            f"{self.text}"
        )


# ─── Query 扩展 ───────────────────────────────────────────────────────────────

# 单细胞领域常用同义词/缩写映射
SC_SYNONYMS = {
    "t cell": ["T lymphocyte", "CD3+ cell", "T-cell"],
    "b cell": ["B lymphocyte", "CD19+ cell", "B-cell"],
    "nk cell": ["natural killer cell", "CD56+ cell"],
    "macrophage": ["Mφ", "MΦ", "monocyte-derived macrophage"],
    "marker gene": ["marker genes", "cell type marker", "signature gene"],
    "scrna-seq": ["scRNA-seq", "single-cell RNA sequencing", "single cell transcriptomics"],
    "scatac-seq": ["scATAC-seq", "single-cell ATAC-seq", "chromatin accessibility"],
    "annotation": ["cell type annotation", "cell labeling", "cluster annotation"],
}

def expand_query(query: str) -> list[str]:
    """返回原始 query + 同义词扩展变体（最多 3 个）"""
    queries = [query]
    q_lower = query.lower()
    added = 0
    for key, synonyms in SC_SYNONYMS.items():
        if key in q_lower and added < 2:
            for syn in synonyms[:1]:
                expanded = re.sub(re.escape(key), syn, query, flags=re.I)
                if expanded != query:
                    queries.append(expanded)
                    added += 1
    return queries


# ─── Reranker 单例 ────────────────────────────────────────────────────────────

_reranker: Optional[FlagReranker] = None

def get_reranker() -> FlagReranker:
    global _reranker
    if _reranker is None:
        log.info(f"加载 Reranker {RERANK_MODEL}")
        device = "cuda" if EMBED_DEVICE == "cuda" else "cpu"
        _reranker = FlagReranker(RERANK_MODEL, use_fp16=True, device=device)
    return _reranker


# ─── 检索器 ───────────────────────────────────────────────────────────────────

class SingleCellRetriever:
    """
    两阶段检索器：向量召回 → Cross-Encoder Rerank
    """

    def __init__(self, top_n: int = RETRIEVE_TOP_N, top_k: int = RERANK_TOP_K):
        self.top_n = top_n
        self.top_k = top_k

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        """主入口：给定 query，返回 top-k RetrievedChunk"""
        # 1. Query 扩展
        queries = expand_query(query)
        log.info(f"Query 扩展: {queries}")

        # 2. 多路向量检索（去重）
        collection = get_chroma_collection()
        seen_ids: set[str] = set()
        candidates: list[dict] = []

        for q in queries:
            q_vec = embed_texts([q])[0]
            results = collection.query(
                query_embeddings=[q_vec],
                n_results=min(self.top_n, collection.count()),
                include=["documents", "metadatas", "distances"],
            )

            ids       = results["ids"][0]
            docs      = results["documents"][0]
            metas     = results["metadatas"][0]
            distances = results["distances"][0]

            for cid, doc, meta, dist in zip(ids, docs, metas, distances):
                if cid not in seen_ids:
                    seen_ids.add(cid)
                    candidates.append({
                        "chunk_id":    cid,
                        "pmid":        meta["pmid"],
                        "title":       meta["title"],
                        "year":        meta["year"],
                        "section":     meta["section"],
                        "text":        meta.get("raw_text", doc[:500]),
                        "vector_dist": dist,
                    })

        if not candidates:
            log.warning("向量检索无结果")
            return []

        log.info(f"向量召回候选数: {len(candidates)}")

        # 3. Rerank
        reranker = get_reranker()
        pairs = [(query, c["text"]) for c in candidates]
        scores = reranker.compute_score(pairs, normalize=True)

        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)

        # 4. 按 rerank 分数排序，取 top-k
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        top = candidates[:self.top_k]

        log.info(
            f"Rerank top-{self.top_k} scores: "
            + ", ".join(f"{c['rerank_score']:.3f}" for c in top)
        )

        return [
            RetrievedChunk(
                chunk_id    = c["chunk_id"],
                pmid        = c["pmid"],
                title       = c["title"],
                year        = c["year"],
                section     = c["section"],
                text        = c["text"],
                score       = c["rerank_score"],
                vector_dist = c["vector_dist"],
            )
            for c in top
        ]


# ─── 快速测试 ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    retriever = SingleCellRetriever()
    test_query = "marker genes for CD8 T cells in tumor microenvironment scRNA-seq"
    results = retriever.retrieve(test_query)
    print(f"\n=== Top {len(results)} Results ===")
    for i, r in enumerate(results):
        print(f"\n[{i+1}] score={r.score:.3f} | {r.title[:60]} ({r.year})")
        print(f"     Section: {r.section}")
        print(f"     {r.text[:200]}...")
