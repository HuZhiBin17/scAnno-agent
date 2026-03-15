"""
Step 7 — ReAct Agent：自动化注释 + 归因推理
──────────────────────────────────────────────
Agent 拥有以下工具：
  1. search_literature(query)    — 检索知识库
  2. query_go_term(gene_list)    — 查询 Gene Ontology（via QuickGO REST API）
  3. query_kegg_pathway(genes)   — 查询 KEGG 通路（via KEGG REST API）
  4. call_rag_annotate(markers)  — 调用 RAG pipeline 生成注释
  5. validate_annotation(result) — 检验注释是否有足够置信度

ReAct 循环：Thought → Action → Observation → … → Final Answer
最大循环次数：MAX_AGENT_STEPS（config.py 中配置）
"""
from __future__ import annotations
import re
import json
import time
import logging
import requests
from typing import Any, Optional

from config import (
    LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS,
    OPENAI_API_KEY, OPENAI_BASE_URL,
    MAX_AGENT_STEPS, RERANK_TOP_K,
)
from retriever_reranker import SingleCellRetriever
from rag_pipeline import SingleCellRAGPipeline, call_llm

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ─── Tool 实现 ────────────────────────────────────────────────────────────────

retriever = SingleCellRetriever()
rag_pipeline = SingleCellRAGPipeline()


def tool_search_literature(query: str) -> str:
    """检索知识库，返回 top chunks 摘要"""
    chunks = retriever.retrieve(query)
    if not chunks:
        return "No relevant literature found."
    parts = []
    for c in chunks:
        parts.append(f"- [{c.pmid}] {c.title} ({c.year})\n  {c.text[:300]}...")
    return "\n".join(parts)


def tool_query_go_terms(gene_list: list[str]) -> str:
    """
    通过 QuickGO REST API 查询基因集的 GO term 富集
    (公共 API，无需 key)
    """
    if not gene_list:
        return "No genes provided."
    genes_str = ",".join(gene_list[:10])   # 限制查询数量
    url = (
        "https://www.ebi.ac.uk/QuickGO/services/annotation/search"
        f"?geneProductId={genes_str}&limit=5&page=1"
    )
    try:
        resp = requests.get(url, timeout=15,
                            headers={"Accept": "application/json"})
        if resp.status_code != 200:
            return f"QuickGO API error: {resp.status_code}"
        data = resp.json()
        results = data.get("results", [])
        if not results:
            return "No GO annotations found."
        lines = []
        seen_go = set()
        for r in results[:10]:
            go_id   = r.get("goId", "")
            go_name = r.get("goName", "")
            aspect  = r.get("goAspect", "")
            if go_id not in seen_go:
                lines.append(f"{go_id} ({aspect}): {go_name}")
                seen_go.add(go_id)
        return "\n".join(lines) if lines else "No GO terms retrieved."
    except Exception as e:
        return f"QuickGO query failed: {e}"


def tool_query_kegg_pathways(gene_list: list[str]) -> str:
    """
    通过 KEGG REST API 查询人类 KEGG 通路
    """
    if not gene_list:
        return "No genes provided."
    pathway_counts: dict[str, int] = {}
    pathway_names: dict[str, str] = {}

    for gene in gene_list[:8]:   # 限制 8 个基因
        url = f"https://rest.kegg.jp/find/hsa/{gene}"
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                continue
            # 获取 KEGG gene ID
            lines = resp.text.strip().split("\n")
            if not lines or not lines[0]:
                continue
            kegg_gene_id = lines[0].split("\t")[0]

            # 查询该基因参与的通路
            path_resp = requests.get(
                f"https://rest.kegg.jp/link/pathway/{kegg_gene_id}",
                timeout=10
            )
            for line in path_resp.text.strip().split("\n"):
                parts = line.split("\t")
                if len(parts) == 2:
                    pathway_id = parts[1].strip()
                    pathway_counts[pathway_id] = pathway_counts.get(pathway_id, 0) + 1

            time.sleep(0.1)   # KEGG 速率限制
        except Exception:
            continue

    if not pathway_counts:
        return "No KEGG pathways found."

    # 获取通路名称（取 top 5）
    top_pathways = sorted(pathway_counts.items(), key=lambda x: -x[1])[:5]
    result_lines = []
    for pid, cnt in top_pathways:
        try:
            name_resp = requests.get(
                f"https://rest.kegg.jp/list/{pid}", timeout=8
            )
            name = name_resp.text.split("\t")[-1].strip() if name_resp.ok else pid
        except Exception:
            name = pid
        result_lines.append(f"{pid}: {name} (hit by {cnt} genes)")

    return "\n".join(result_lines)


def tool_rag_annotate(
    markers: list[str],
    tissue: str = "unknown",
    technology: str = "scRNA-seq",
    cluster_id: str = "0",
) -> str:
    """调用 RAG pipeline 生成细胞类型注释"""
    result = rag_pipeline.annotate_cluster(
        markers=markers,
        tissue=tissue,
        technology=technology,
        cluster_id=cluster_id,
    )
    return json.dumps(result, ensure_ascii=False, indent=2)


def tool_validate_annotation(annotation_json: str) -> str:
    """检验注释质量，如置信度低则给出改进建议"""
    try:
        ann = json.loads(annotation_json)
    except Exception:
        return "Invalid JSON annotation."

    confidence = ann.get("confidence", 0)
    markers    = ann.get("key_markers", [])
    refs       = ann.get("supporting_references", [])

    issues = []
    if confidence < 0.5:
        issues.append("Low confidence (<0.5). Need more specific marker evidence.")
    if len(markers) < 2:
        issues.append("Too few key markers identified.")
    if not refs:
        issues.append("No supporting references found.")
    if not ann.get("go_terms"):
        issues.append("GO terms missing. Consider querying GO database.")
    if not ann.get("kegg_pathways"):
        issues.append("KEGG pathways missing. Consider querying KEGG.")

    if not issues:
        return f"Annotation validated. Confidence: {confidence:.2f}. Cell type: {ann['cell_type']}."
    return "Issues found:\n" + "\n".join(f"- {i}" for i in issues)


# ─── Tool 注册表 ──────────────────────────────────────────────────────────────

TOOLS = {
    "search_literature":  tool_search_literature,
    "query_go_terms":     tool_query_go_terms,
    "query_kegg_pathways":tool_query_kegg_pathways,
    "rag_annotate":       tool_rag_annotate,
    "validate_annotation":tool_validate_annotation,
}

TOOL_DESCRIPTIONS = """
Available Tools:
1. search_literature(query: str) -> str
   Search the single-cell literature knowledge base for relevant papers.

2. query_go_terms(gene_list: list[str]) -> str
   Query Gene Ontology (GO) annotations for a list of genes via EBI QuickGO.

3. query_kegg_pathways(gene_list: list[str]) -> str
   Query KEGG human pathways enriched in a gene list.

4. rag_annotate(markers: list[str], tissue: str, technology: str, cluster_id: str) -> str
   Run the full RAG pipeline to annotate a cell cluster and return JSON result.

5. validate_annotation(annotation_json: str) -> str
   Validate the quality of an annotation and report issues.
"""

REACT_SYSTEM = f"""You are a single-cell biology expert agent.
Use the ReAct framework: Thought → Action → Observation → repeat.

{TOOL_DESCRIPTIONS}

Output format for each step:
Thought: <your reasoning>
Action: <tool_name>({{"arg1": "val1", ...}})

When you have the final answer, output:
Final Answer: <complete JSON annotation>

Rules:
- Always call rag_annotate first to get initial annotation.
- Enrich with GO terms and KEGG pathways if missing.
- Validate before returning Final Answer.
- Do not invent data not present in observations.
"""


# ─── ReAct Agent ─────────────────────────────────────────────────────────────

class SingleCellAgent:

    def __init__(self, max_steps: int = MAX_AGENT_STEPS):
        self.max_steps = max_steps
        self.history: list[dict] = []

    def _parse_action(self, text: str) -> Optional[tuple[str, Any]]:
        """从 LLM 输出解析 Action 行，支持 Action: tool_name({"key": "val"})"""
        match = re.search(r"Action:\s*(\w+)\((.*)\)\s*$", text, re.M | re.S)
        if not match:
            return None
        tool_name = match.group(1)
        raw_args  = match.group(2).strip()
        
        if not raw_args:
            return tool_name, {}

        # 尝试解析 JSON 参数
        try:
            args = json.loads(raw_args)
            return tool_name, args
        except json.JSONDecodeError:
            # 兼容非标准 JSON（如 key 没有引号，或者用了冒号但没用大括号）
            log.warning(f"JSON 解析失败，尝试启发式提取: {raw_args}")
            # 处理 markers: [...] 这种常见的 LLM 错误格式
            if ":" in raw_args:
                try:
                    # 尝试转换成字典
                    processed = re.sub(r'(\w+):', r'"\1":', raw_args)
                    if not processed.startswith("{"):
                        processed = "{" + processed + "}"
                    return tool_name, json.loads(processed)
                except Exception:
                    pass
            
            # 如果还是不行，退回到原始字符串（可能是单参数工具）
            return tool_name, raw_args

    def _call_tool(self, tool_name: str, args: Any) -> str:
        """执行工具调用"""
        if tool_name not in TOOLS:
            return f"Unknown tool: {tool_name}"
        func = TOOLS[tool_name]
        try:
            if isinstance(args, dict):
                return func(**args)
            elif isinstance(args, list):
                return func(args)
            else:
                return func(args)
        except Exception as e:
            return f"Tool error: {e}"

    def run(self, cluster_info: dict) -> dict:
        """
        运行 Agent 对单个簇进行注释归因

        cluster_info: {
            "cluster_id": str,
            "markers": list[str],
            "tissue": str,
            "technology": str,
        }
        """
        markers   = cluster_info.get("markers", [])
        tissue    = cluster_info.get("tissue", "unknown")
        technology= cluster_info.get("technology", "scRNA-seq")
        cid       = cluster_info.get("cluster_id", "0")

        initial_prompt = (
            f"Annotate cluster {cid}.\n"
            f"Tissue: {tissue}\nTechnology: {technology}\n"
            f"Top Markers: {', '.join(markers)}\n\n"
            f"Start with rag_annotate, then enrich GO/KEGG, then validate."
        )

        messages = [
            {"role": "system", "content": REACT_SYSTEM},
            {"role": "user",   "content": initial_prompt},
        ]

        final_answer = None
        for step in range(self.max_steps):
            log.info(f"Agent step {step+1}/{self.max_steps}")
            response = call_llm(REACT_SYSTEM, "\n\n".join(
                f"{m['role'].upper()}: {m['content']}" for m in messages[1:]
            ))
            log.info(f"LLM response:\n{response}")

            # 检查 Final Answer
            fa_match = re.search(r"Final Answer:\s*(.+)", response, re.S)
            if fa_match:
                raw_fa = fa_match.group(1).strip()
                try:
                    final_answer = json.loads(raw_fa)
                except Exception:
                    final_answer = {"raw_final_answer": raw_fa}
                log.info(f"Agent finished at step {step+1}")
                break

            # 解析并执行 Action
            action = self._parse_action(response)
            if action:
                tool_name, args = action
                log.info(f"  Tool: {tool_name}({str(args)[:80]})")
                observation = self._call_tool(tool_name, args)
                log.info(f"  Observation: {observation[:200]}")

                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user",
                    "content": f"Observation: {observation[:2000]}"  # 截断避免超 context
                })
            else:
                log.warning("No action parsed from response, stopping.")
                break

        if final_answer is None:
            log.warning("Max steps reached without Final Answer")
            final_answer = {"cell_type": "Unknown", "confidence": 0.0,
                            "error": "Agent max steps exceeded"}

        final_answer["cluster_id"] = cid
        return final_answer


# ─── CLI 入口 ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    agent = SingleCellAgent()

    test_clusters = [
        {
            "cluster_id": "cluster_0",
            "markers": ["CD8A", "CD8B", "GZMB", "GZMK", "PRF1", "PDCD1", "LAG3", "HAVCR2"],
            "tissue": "lung adenocarcinoma",
            "technology": "scRNA-seq",
        },
        {
            "cluster_id": "cluster_1",
            "markers": ["FOXP3", "IL2RA", "CTLA4", "IKZF2", "TNFRSF9", "TIGIT"],
            "tissue": "colorectal cancer",
            "technology": "scRNA-seq",
        },
    ]

    for cluster in test_clusters:
        print(f"\n{'='*60}")
        print(f"Annotating {cluster['cluster_id']}")
        print(f"{'='*60}")
        result = agent.run(cluster)
        print(json.dumps(result, indent=2, ensure_ascii=False))
