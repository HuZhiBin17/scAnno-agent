"""
Step 2 — PDF 解析与结构化抽取
──────────────────────────────
使用 PyMuPDF (fitz) 逐页解析 PDF：
  • 剔除页眉页脚、参考文献、图表说明
  • 基于字体大小启发式识别段落层级（标题/正文/摘要）
  • 输出结构化 JSON：{pmid, title, sections:[{heading, text}]}
  • 写入 data/parsed/<pmid>.json
"""
from __future__ import annotations
import re
import json
import logging
from pathlib import Path
from typing import Optional
from statistics import mean, stdev

import fitz  # PyMuPDF

from config import PARSED_DIR, PDF_DIR, MIN_CHARS_PER_PAGE, TITLE_FONT_RATIO

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# 需要过滤的章节关键词（参考文献、利益声明等）
SKIP_HEADINGS = re.compile(
    r"^(references?|bibliography|acknowledge?ments?|conflict|declaration|"
    r"supplementar|funding|author contributions?|data availability)\b",
    re.I
)
# 页眉页脚特征：行极短 & 含数字或期刊名缩写
HEADER_FOOTER_RE = re.compile(r"^\s*(\d+|\w{1,3}\s+\d{4})\s*$")


# ─── 工具函数 ─────────────────────────────────────────────────────────────────

def _median_fontsize(page) -> float:
    """计算页面正文中位字体大小"""
    sizes = []
    for block in page.get_text("dict")["blocks"]:
        if block["type"] != 0:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                if span["text"].strip():
                    sizes.append(span["size"])
    return mean(sizes) if sizes else 10.0


def _clean_text(text: str) -> str:
    """清洗乱码、多余空白、连字符断行"""
    text = re.sub(r"-\n(\w)", r"\1", text)        # 连字符断行合并
    text = re.sub(r"\n{3,}", "\n\n", text)         # 多空行压缩
    text = re.sub(r"[ \t]{2,}", " ", text)         # 多空格
    text = re.sub(r"\x0c", "", text)               # 换页符
    return text.strip()


def _is_heading(span_size: float, median: float, text: str) -> bool:
    """启发式判断是否为标题行"""
    return (
        span_size > median * TITLE_FONT_RATIO
        and len(text.split()) <= 12
        and not text.endswith(".")
    )


# ─── 核心解析器 ───────────────────────────────────────────────────────────────

def parse_pdf(pdf_path: Path, pmid: str, metadata: dict) -> Optional[dict]:
    """
    解析单个 PDF，返回结构化文档 dict。
    结构：
      {
        pmid, title, year, authors, abstract,
        sections: [{"heading": str, "text": str}],
        full_text: str   # 所有正文拼接（用于 fallback 切片）
      }
    """
    try:
        doc = fitz.open(str(pdf_path))
    except Exception as e:
        log.warning(f"fitz.open failed for {pdf_path}: {e}")
        return None

    sections = []
    current_heading = "Introduction"
    current_paragraphs = []
    skip_mode = False

    for page_num, page in enumerate(doc):
        raw_text = page.get_text("text")
        if len(raw_text.strip()) < MIN_CHARS_PER_PAGE:
            continue   # 跳过图片页/扫描页

        median_sz = _median_fontsize(page)

        # 逐 block 处理
        blocks_data = page.get_text("dict")["blocks"]
        for block in blocks_data:
            if block["type"] != 0:   # 只处理文字块
                continue

            block_lines = []
            block_is_heading = False
            first_span_size = 0.0

            for line in block["lines"]:
                line_text = " ".join(s["text"] for s in line["spans"]).strip()
                if not line_text or HEADER_FOOTER_RE.match(line_text):
                    continue
                if line["spans"]:
                    sz = line["spans"][0]["size"]
                    if page_num == 0 and _is_heading(sz, median_sz, line_text):
                        block_is_heading = True
                        first_span_size = sz
                    elif page_num > 0 and _is_heading(sz, median_sz, line_text):
                        block_is_heading = True
                        first_span_size = sz
                block_lines.append(line_text)

            if not block_lines:
                continue

            block_text = " ".join(block_lines)

            if block_is_heading:
                heading_candidate = block_text.strip()
                # 遇到 References 等则进入跳过模式
                if SKIP_HEADINGS.match(heading_candidate):
                    skip_mode = True
                    continue
                if skip_mode:
                    # 如果跳过模式中出现新的合法标题则退出（部分期刊把致谢放中间）
                    skip_mode = False

                # 保存上一节
                if current_paragraphs:
                    sections.append({
                        "heading": current_heading,
                        "text": _clean_text("\n".join(current_paragraphs))
                    })
                current_heading = heading_candidate
                current_paragraphs = []
            else:
                if not skip_mode:
                    current_paragraphs.append(block_text)

    # 保存最后一节
    if current_paragraphs and not skip_mode:
        sections.append({
            "heading": current_heading,
            "text": _clean_text("\n".join(current_paragraphs))
        })

    doc.close()

    if not sections:
        log.warning(f"No sections parsed from {pdf_path}")
        return None

    full_text = "\n\n".join(
        f"## {s['heading']}\n{s['text']}" for s in sections
    )

    # 如果 metadata 里有摘要，把摘要也注入到第一节前
    abstract = metadata.get("abstract", "")
    if abstract and (not sections or "abstract" not in sections[0]["heading"].lower()):
        sections.insert(0, {"heading": "Abstract", "text": abstract})

    result = {
        "pmid": pmid,
        "title": metadata.get("title", ""),
        "year": metadata.get("year", ""),
        "authors": metadata.get("authors", []),
        "abstract": abstract,
        "sections": sections,
        "full_text": full_text,
        "pdf_path": str(pdf_path),
    }
    return result


# ─── 批量解析 ─────────────────────────────────────────────────────────────────

def parse_all():
    """读取 metadata.jsonl，对每篇有 PDF 的论文执行解析"""
    meta_path = PARSED_DIR / "metadata.jsonl"
    if not meta_path.exists():
        log.error("metadata.jsonl 不存在，请先运行 01_paper_crawler.py")
        return

    papers = []
    with open(meta_path, encoding="utf-8") as f:
        for line in f:
            papers.append(json.loads(line))

    ok_count = 0
    for paper in papers:
        pmid = paper["pmid"]
        out_path = PARSED_DIR / f"{pmid}.json"

        if out_path.exists():
            log.info(f"[skip] {pmid} 已解析")
            continue

        pdf_path_str = paper.get("pdf_path")
        if not pdf_path_str:
            log.info(f"[skip] {pmid} 无 PDF")
            continue

        pdf_path = Path(pdf_path_str)
        if not pdf_path.exists():
            log.warning(f"[missing] PDF 文件不存在: {pdf_path}")
            continue

        log.info(f"解析 {pmid}: {paper.get('title','')[:60]}")
        parsed = parse_pdf(pdf_path, pmid, paper)
        if parsed:
            out_path.write_text(json.dumps(parsed, ensure_ascii=False, indent=2))
            ok_count += 1

    log.info(f"解析完成：{ok_count} 篇 ✓")


if __name__ == "__main__":
    parse_all()
