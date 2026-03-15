"""
单细胞多组学 RAG 系统 — 全局配置
"""
from __future__ import annotations
import os
from pathlib import Path

# ─── 目录结构 ─────────────────────────────────────────────────────────────────
ROOT_DIR        = Path(__file__).parent
DATA_DIR        = ROOT_DIR / "data"
PDF_DIR         = DATA_DIR / "pdfs"
PARSED_DIR      = DATA_DIR / "parsed"
CHUNKS_DIR      = DATA_DIR / "chunks"
INDEX_DIR       = DATA_DIR / "index"
RESULTS_DIR     = ROOT_DIR / "results"

for d in [PDF_DIR, PARSED_DIR, CHUNKS_DIR, INDEX_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── PubMed 爬虫 ──────────────────────────────────────────────────────────────
PUBMED_EMAIL    = os.getenv("PUBMED_EMAIL", "2038182056emiya@gmail.com")
PUBMED_API_KEY  = os.getenv("PUBMED_API_KEY", "")          # 可选，有 key 限速更高
PUBMED_QUERIES  = [
    "single cell RNA sequencing cell type annotation marker genes",
    "scRNA-seq clustering cell type identification",
    "single cell ATAC-seq chromatin accessibility annotation",
    "single cell multiomics integration cell annotation",
    "scRNA-seq tumor microenvironment cell type",
]
MAX_PAPERS_PER_QUERY = 20                                   # 每条查询最多下载篇数
PUBMED_MAX_RETRIES   = 3

# Unpaywall / Open Access PDF 下载
UNPAYWALL_EMAIL = os.getenv("UNPAYWALL_EMAIL", PUBMED_EMAIL)

# ─── PDF 解析 ─────────────────────────────────────────────────────────────────
MIN_CHARS_PER_PAGE  = 100     # 少于此字符的页面视为图/扫描页，跳过
TITLE_FONT_RATIO    = 1.2     # 字体大于正文均值 * 比例 → 视为标题

# ─── 文本切片 ─────────────────────────────────────────────────────────────────
CHUNK_SIZE      = 512         # token 数（近似字符 * 0.75）
CHUNK_OVERLAP   = 64
SEPARATORS      = ["\n\n", "\n", "。", ". ", " ", ""]

# ─── Embedding ────────────────────────────────────────────────────────────────
# 推荐使用 BAAI/bge-m3（中英双语，2048 token 上下文）
EMBED_MODEL     = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
EMBED_BATCH     = 32
EMBED_DEVICE    = "cuda"      # 无 GPU 则改 "cpu"

# ─── 向量库（ChromaDB 本地持久化）────────────────────────────────────────────
CHROMA_COLLECTION = "singlecell_papers"

# ─── Reranker ────────────────────────────────────────────────────────────────
RERANK_MODEL    = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
RERANK_TOP_K    = 5           # rerank 后保留 Top-K 送入 LLM

# ─── 检索参数 ────────────────────────────────────────────────────────────────
RETRIEVE_TOP_N  = 20          # 初步向量检索条数，再 rerank 缩减到 RERANK_TOP_K

# ─── LLM ─────────────────────────────────────────────────────────────────────
LLM_PROVIDER    = os.getenv("LLM_PROVIDER", "openai")       # openai | deepseek | zhipu
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL       = os.getenv("LLM_MODEL", "gpt-4.1-mini")
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS  = 1024

# ─── Agent ───────────────────────────────────────────────────────────────────
MAX_AGENT_STEPS = 5           # ReAct 最大循环轮次
