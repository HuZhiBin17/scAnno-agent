"""
单细胞多组学 RAG 系统 — 全局配置
"""
from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env 环境变量
load_dotenv()

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
EMBED_DEVICE    = os.getenv("EMBED_DEVICE", "cuda")      # 无 GPU 则改 "cpu"

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
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
LLM_MODEL       = os.getenv("LLM_MODEL", "")
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS  = int(os.getenv("LLM_MAX_TOKENS", "2048"))


def _is_truthy(value: str) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


HF_HUB_DIR = Path(os.getenv("HF_HOME", str(Path.home() / ".cache" / "huggingface"))) / "hub"
HF_AUTO_LOCAL_ONLY = _is_truthy(os.getenv("HF_AUTO_LOCAL_ONLY", "1"))


def resolve_local_model_path(model_id: str) -> str:
    """
    从 HuggingFace 本地缓存中解析模型快照目录。
    若不存在可用缓存，返回空字符串。
    """
    if not model_id or "/" not in model_id:
        return ""

    cache_dir = HF_HUB_DIR / f"models--{model_id.replace('/', '--')}"
    snapshots_dir = cache_dir / "snapshots"
    if not snapshots_dir.exists():
        return ""

    # 优先 refs/main 指向的快照
    ref_main = cache_dir / "refs" / "main"
    if ref_main.exists():
        revision = ref_main.read_text(encoding="utf-8").strip()
        target = snapshots_dir / revision
        if target.exists():
            return str(target)

    # 回退：取最近修改的快照目录
    snapshots = [p for p in snapshots_dir.iterdir() if p.is_dir()]
    if not snapshots:
        return ""
    snapshots.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(snapshots[0])


def _normalize_openai_base_url(url: str) -> str:
    """
    将 OpenAI 兼容地址规范化为包含 /v1 的形式。
    例如：
    - https://api.deepseek.com -> https://api.deepseek.com/v1
    - https://api.deepseek.com/v1 -> 保持不变
    """
    base = (url or "").strip().rstrip("/")
    if not base:
        return base
    if base.endswith("/v1"):
        return base
    if base.endswith("/anthropic"):
        return base
    return f"{base}/v1"


def get_llm_runtime_config() -> dict[str, str]:
    """
    解析当前运行时的 LLM 连接配置，统一返回 provider/api_key/base_url/model。
    支持：
    - openai（默认）
    - deepseek（可选）
    """
    provider = (LLM_PROVIDER or "openai").strip().lower()

    if provider == "deepseek":
        return {
            "provider": "deepseek",
            "api_key": DEEPSEEK_API_KEY,
            "base_url": _normalize_openai_base_url(DEEPSEEK_BASE_URL),
            "model": LLM_MODEL or "deepseek-v4-flash",
        }

    # 默认走 OpenAI 兼容接口
    return {
        "provider": "openai",
        "api_key": OPENAI_API_KEY,
        "base_url": _normalize_openai_base_url(OPENAI_BASE_URL),
        "model": LLM_MODEL or "gpt-4.1-mini",
    }

# ─── Agent ───────────────────────────────────────────────────────────────────
MAX_AGENT_STEPS = 5           # ReAct 最大循环轮次
# Agent memory / tool runtime
AGENT_MEMORY_ENABLED = _is_truthy(os.getenv("AGENT_MEMORY_ENABLED", "true"))
AGENT_MEMORY_DB = os.getenv("AGENT_MEMORY_DB", str(RESULTS_DIR / "agent_memory.sqlite"))
AGENT_CACHE_ENABLED = _is_truthy(os.getenv("AGENT_CACHE_ENABLED", "true"))
AGENT_REUSE_EXACT_MATCH = _is_truthy(os.getenv("AGENT_REUSE_EXACT_MATCH", "true"))
AGENT_SIMILARITY_THRESHOLD = float(os.getenv("AGENT_SIMILARITY_THRESHOLD", "0.75"))
AGENT_TOOL_BACKEND = os.getenv("AGENT_TOOL_BACKEND", "mcp").strip().lower()
MCP_SERVER_MODE = os.getenv("MCP_SERVER_MODE", "stdio").strip().lower()
MCP_TOOL_TIMEOUT_SECONDS = int(os.getenv("MCP_TOOL_TIMEOUT_SECONDS", "60"))
