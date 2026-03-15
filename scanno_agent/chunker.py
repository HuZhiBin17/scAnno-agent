"""
Step 3 — 语义切片（Chunking）
──────────────────────────────
策略：
  1. 按 section 边界优先切分（保留生物学语义单元）
  2. 在 section 内使用递归字符切分（RecursiveCharacterTextSplitter）
  3. 每个 chunk 保留完整元数据（pmid、标题、section heading、chunk_id）
  4. 输出写入 data/chunks/all_chunks.jsonl

每条 chunk 结构：
  {
    "chunk_id":   "pmid_secIdx_chkIdx",
    "pmid":       "12345678",
    "title":      "Single-cell RNA sequencing...",
    "year":       "2023",
    "section":    "Results",
    "text":       "...",
    "char_count": 480
  }
"""
from __future__ import annotations
import re
import json
import logging
from pathlib import Path
from typing import Iterator

from config import PARSED_DIR, CHUNKS_DIR, CHUNK_SIZE, CHUNK_OVERLAP, SEPARATORS

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ─── 递归字符切分（纯 Python 实现，无需 langchain）────────────────────────────

class RecursiveCharSplitter:
    """
    递归字符文本切分器（模仿 LangChain RecursiveCharacterTextSplitter）。
    chunk_size 以字符数计；chunk_overlap 为重叠字符数。
    """

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE * 3,   # 512 tokens ≈ 1536 chars（中英混合取 3x）
        chunk_overlap: int = CHUNK_OVERLAP * 3,
        separators: list[str] = None,
    ):
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators    = separators or SEPARATORS

    def split_text(self, text: str) -> list[str]:
        return list(self._split(text, self.separators))

    def _split(self, text: str, seps: list[str]) -> Iterator[str]:
        if not seps:
            # 最细粒度：直接按 chunk_size 切
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
                yield text[i:i + self.chunk_size]
            return

        sep = seps[0]
        remaining_seps = seps[1:]

        # 以当前 sep 分割
        if sep:
            parts = text.split(sep)
        else:
            parts = list(text)   # 字符级

        current = []
        current_len = 0

        for part in parts:
            part_len = len(part) + len(sep)
            if current_len + part_len > self.chunk_size and current:
                # 输出当前块
                chunk_text = sep.join(current)
                if len(chunk_text) <= self.chunk_size:
                    yield chunk_text
                else:
                    # 块太大，递归切
                    yield from self._split(chunk_text, remaining_seps)
                # 保留 overlap
                while current and current_len > self.chunk_overlap:
                    removed = current.pop(0)
                    current_len -= len(removed) + len(sep)
            current.append(part)
            current_len += part_len

        if current:
            chunk_text = sep.join(current)
            if len(chunk_text) <= self.chunk_size:
                yield chunk_text
            else:
                yield from self._split(chunk_text, remaining_seps)


# ─── Section 级切片 ───────────────────────────────────────────────────────────

SPLITTER = RecursiveCharSplitter()

# 摘要和 Methods 通常信息密度高，单独保留完整 chunk
HIGH_VALUE_SECTIONS = re.compile(
    r"(abstract|introduction|result|discussion|conclusion|finding|marker|"
    r"cell type|annotation|cluster|pathway|go term|kegg)",
    re.I
)

LOW_VALUE_SECTIONS = re.compile(
    r"(method|material|statistic|protocol|supplementar|figure|table legend)",
    re.I
)


def chunk_document(doc: dict) -> list[dict]:
    """将单篇解析后的文档切成 chunk 列表"""
    pmid    = doc["pmid"]
    title   = doc.get("title", "")
    year    = doc.get("year", "")
    authors = doc.get("authors", [])
    chunks  = []

    for sec_idx, section in enumerate(doc.get("sections", [])):
        heading = section.get("heading", "")
        text    = section.get("text", "").strip()
        if not text:
            continue

        # 低价值段落降低切片粒度（不切太细，减少噪声）
        if LOW_VALUE_SECTIONS.search(heading):
            # 只保留整段，超大则按 2x chunk_size 切
            if len(text) <= SPLITTER.chunk_size * 2:
                sub_chunks = [text]
            else:
                sub_chunks = SPLITTER.split_text(text)
        else:
            sub_chunks = SPLITTER.split_text(text)

        for chk_idx, chunk_text in enumerate(sub_chunks):
            chunk_text = chunk_text.strip()
            if len(chunk_text) < 60:     # 过短的噪声块跳过
                continue

            # 在每个 chunk 前注入上下文前缀（提升检索精度）
            prefix = f"[Paper: {title[:80]}] [Section: {heading}]\n"
            enriched = prefix + chunk_text

            chunks.append({
                "chunk_id":   f"{pmid}_{sec_idx}_{chk_idx}",
                "pmid":       pmid,
                "title":      title,
                "year":       year,
                "authors":    authors[:3],   # 只保留前 3 作者
                "section":    heading,
                "text":       chunk_text,          # 原始文本（用于展示）
                "embed_text": enriched,            # 带前缀（用于 embedding）
                "char_count": len(chunk_text),
                "is_high_value": bool(HIGH_VALUE_SECTIONS.search(heading)),
            })

    return chunks


def chunk_all():
    """批量切片所有解析文档"""
    out_path = CHUNKS_DIR / "all_chunks.jsonl"
    existing_pmids = set()

    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                existing_pmids.add(json.loads(line)["pmid"])
        log.info(f"已有切片来自 {len(existing_pmids)} 篇论文")

    json_files = sorted(PARSED_DIR.glob("*.json"))
    total_new = 0

    with open(out_path, "a", encoding="utf-8") as out_f:
        for json_file in json_files:
            pmid = json_file.stem
            if pmid in existing_pmids:
                continue

            doc = json.loads(json_file.read_text(encoding="utf-8"))
            chunks = chunk_document(doc)
            for chunk in chunks:
                out_f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            total_new += len(chunks)
            log.info(f"  {pmid}: {len(chunks)} chunks")

    log.info(f"切片完成：新增 {total_new} 个 chunks ✓")


if __name__ == "__main__":
    chunk_all()
