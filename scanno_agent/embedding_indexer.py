"""
Step 4 — 向量 Embedding 与 ChromaDB 索引构建
─────────────────────────────────────────────
使用 BAAI/bge-m3 生成 dense embedding（1024 维）
批量写入 ChromaDB 本地持久化向量库
ChromaDB 默认使用余弦相似度（cosine）

索引结构：
  collection: singlecell_papers
  documents : chunk["embed_text"]   (带 section 前缀的文本)
  metadatas : {pmid, title, year, section, chunk_id, is_high_value}
  ids        : chunk["chunk_id"]
"""
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import torch

from config import (
    CHUNKS_DIR, INDEX_DIR,
    EMBED_MODEL, EMBED_BATCH, EMBED_DEVICE,
    CHROMA_COLLECTION,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ─── 全局单例（避免重复加载模型）─────────────────────────────────────────────

_embed_model: Optional[SentenceTransformer] = None
_chroma_client: Optional[chromadb.PersistentClient] = None
_collection = None


def get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        device = EMBED_DEVICE if torch.cuda.is_available() else "cpu"
        log.info(f"加载 Embedding 模型 {EMBED_MODEL} on {device}")
        _embed_model = SentenceTransformer(EMBED_MODEL, device=device)
    return _embed_model


def get_chroma_collection():
    global _chroma_client, _collection
    if _collection is None:
        _chroma_client = chromadb.PersistentClient(
            path=str(INDEX_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        # cosine 相似度：先 normalize embedding 再用内积
        _collection = _chroma_client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


# ─── Embedding 工具 ───────────────────────────────────────────────────────────

def embed_texts(texts: list[str]) -> list[list[float]]:
    """批量 embed，返回 list of float vectors"""
    model = get_embed_model()
    # BGE 系列：encode 时 normalize_embeddings=True 效果更好
    vecs = model.encode(
        texts,
        batch_size=EMBED_BATCH,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 50,
        convert_to_numpy=True,
    )
    return vecs.tolist()


# ─── 索引构建 ─────────────────────────────────────────────────────────────────

def build_index():
    """将 all_chunks.jsonl 全量写入 ChromaDB"""
    chunks_path = CHUNKS_DIR / "all_chunks.jsonl"
    if not chunks_path.exists():
        log.error("all_chunks.jsonl 不存在，请先运行 03_chunker.py")
        return

    collection = get_chroma_collection()

    # 已有 ID 集合（增量更新）
    existing_ids: set[str] = set()
    try:
        existing = collection.get(include=[])
        existing_ids = set(existing["ids"])
        log.info(f"ChromaDB 已有 {len(existing_ids)} 条向量")
    except Exception:
        pass

    # 读取所有 chunk
    chunks = []
    with open(chunks_path) as f:
        for line in f:
            c = json.loads(line)
            if c["chunk_id"] not in existing_ids:
                chunks.append(c)

    log.info(f"待索引新 chunk: {len(chunks)}")
    if not chunks:
        log.info("无新增，索引已是最新 ✓")
        return

    # 分批写入
    for i in range(0, len(chunks), EMBED_BATCH):
        batch = chunks[i:i + EMBED_BATCH]
        texts     = [c["embed_text"] for c in batch]
        ids       = [c["chunk_id"]   for c in batch]
        metadatas = [
            {
                "pmid":         c["pmid"],
                "title":        c["title"][:200],
                "year":         c["year"],
                "section":      c["section"],
                "chunk_id":     c["chunk_id"],
                "char_count":   c["char_count"],
                "is_high_value": str(c.get("is_high_value", False)),
                "raw_text":     c["text"][:500],   # 存前 500 字供快速显示
            }
            for c in batch
        ]

        embeddings = embed_texts(texts)

        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        log.info(f"  indexed {i+len(batch)}/{len(chunks)}")

    log.info(f"索引构建完成，总计 {collection.count()} 条向量 ✓")


if __name__ == "__main__":
    build_index()
