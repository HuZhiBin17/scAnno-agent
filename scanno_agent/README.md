# 基于 RAG 与 Agent 的单细胞多组学自动化注释系统

## 系统架构

```
PubMed API                   PDF 文件
    │                            │
    ▼                            ▼
paper_crawler.py         ────►  data/pdfs/
    │  esearch + efetch           │
    │  Unpaywall OA PDF          │
    ▼                            │
metadata.jsonl           ◄───────┘
    │
    ▼
pdf_parser.py
    │  PyMuPDF 逐页解析
    │  标题/正文/摘要识别
    │  剔除参考文献/页眉
    ▼
data/parsed/<pmid>.json
    │
    ▼
chunker.py
    │  RecursiveCharSplitter
    │  Section 边界优先切分
    │  元数据注入（pmid/section/year）
    ▼
data/chunks/all_chunks.jsonl
    │
    ▼
embedding_indexer.py
    │  BAAI/bge-m3 (1024-dim dense)
    │  ChromaDB cosine 索引
    ▼
data/index/  (ChromaDB 持久化)
    │
    ├──────────────────────────────────────────────┐
    ▼                                              │
retriever_reranker.py                          │
    │  Stage 1: 向量近邻检索 (top-20)              │
    │  Query 扩展（同义词）                         │
    │  Stage 2: BGE Reranker v2-m3 精排 (→ top-5) │
    ▼                                              │
RetrievedChunk[]                                  │
    │                                              │
    ▼                                              │
rag_pipeline.py                                │
    │  RAG Prompt 构建                             │
    │  [Context] + [Markers] + [Tissue]            │
    │  LLM (gpt-4o / deepseek / zhipu)            │
    │  JSON 结构化输出                              │
    ▼                                              │
AnnotationResult (JSON)                           │
    │                                              │
    ▼                                              │
annotation_agent.py  ◄─────────────────────────┘
    │  ReAct: Thought → Action → Observation
    │  Tools:
    │    ├─ search_literature()  → 知识库检索
    │    ├─ query_go_terms()     → EBI QuickGO API
    │    ├─ query_kegg_pathways()→ KEGG REST API
    │    ├─ rag_annotate()       → 调用 RAG pipeline
    │    └─ validate_annotation()→ 质检 + 改进建议
    ▼
Final Annotation JSON
    {
      "cell_type": "CD8+ Exhausted T cell",
      "subtype": "Terminally exhausted",
      "confidence": 0.91,
      "key_markers": ["CD8A", "PDCD1", "HAVCR2", "GZMB"],
      "go_terms": ["GO:0042110 T cell activation", ...],
      "kegg_pathways": ["hsa04660 T cell receptor signaling"],
      "attribution": "...",
      "supporting_references": ["PMID:35436658", ...]
    }
```

## 快速开始

### 1. 环境安装
```bash
# 创建虚拟环境
conda create -n scrag python=3.11 -y
conda activate scrag

# 安装依赖
pip install -r requirements.txt

# GPU 版 PyTorch（可选）
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 2. 配置环境变量
```bash
export PUBMED_EMAIL="your@email.com"
export OPENAI_API_KEY="sk-..."
export OPENAI_BASE_URL="https://api.openai.com/v1"   # DeepSeek: https://api.deepseek.com/v1
export LLM_MODEL="gpt-4o-mini"                       # DeepSeek: deepseek-chat
```

### 3. 运行完整 Pipeline
```bash
# 一键全流程
python run_pipeline.py --all

# 仅注释（已有索引）
python run_pipeline.py --annotate

# 使用自定义簇信息
python run_pipeline.py --annotate --input my_clusters.json

# 快速模式（跳过 Agent，直接 RAG）
python run_pipeline.py --annotate --no-agent
```

### 4. 自定义簇信息格式 (`my_clusters.json`)
```json
[
  {
    "cluster_id": "cluster_0",
    "markers": ["CD8A", "CD8B", "GZMB", "GZMK", "PDCD1", "LAG3"],
    "tissue": "lung adenocarcinoma",
    "technology": "scRNA-seq"
  }
]
```

## 模块说明

| 文件 | 功能 | 关键技术 |
|------|------|----------|
| `config.py` | 全局配置 | - |
| `paper_crawler.py` | PubMed 爬取 + PDF 下载 | NCBI E-utilities, Unpaywall API |
| `pdf_parser.py` | PDF 结构化解析 | PyMuPDF, 字体大小启发式 |
| `chunker.py` | 语义切片 | RecursiveCharSplitter, Section 边界 |
| `embedding_indexer.py` | 向量索引构建 | BAAI/bge-m3, ChromaDB |
| `retriever_reranker.py` | 两阶段检索 | 向量召回 + BGE Reranker v2 |
| `rag_pipeline.py` | RAG 注释推理 | RAG Prompt, OpenAI JSON mode |
| `annotation_agent.py` | ReAct 自动化 Agent | ReAct, GO/KEGG API, 多工具调用 |
| `run_pipeline.py` | 一键运行入口 | argparse, 全流程编排 |

## 技术亮点（简历要点）

1. **两阶段检索**：向量召回（BGE-M3, top-20）→ Cross-Encoder 精排（BGE-Reranker-v2-m3, top-5），召回率↑ 精确率↑
2. **Query 扩展**：单细胞领域同义词映射（T cell ↔ CD3+/T lymphocyte），减少漏召
3. **Context-Enriched Chunking**：每个 chunk 注入 Paper Title + Section 前缀，embedding 语义更准确
4. **Section 感知切片**：Abstract/Results/Discussion 细切，Methods/Supplementary 粗切，降噪
5. **ReAct Agent + 多工具**：LLM 自主调用 GO/KEGG API 进行富集验证，形成闭环
6. **结构化 JSON 输出**：cell_type + confidence + evidence，可直接集成到 Scanpy AnnData 注释流程
