"""
Step 6 — RAG Pipeline：Prompt 构建 + LLM 调用
──────────────────────────────────────────────
功能：
  • 接收细胞簇的 marker gene 列表
  • 构建结构化 RAG prompt（系统提示 + 检索上下文 + 用户问题）
  • 调用 LLM（OpenAI 兼容 API）
  • 解析结构化 JSON 输出（细胞类型 + 置信度 + 证据）
"""
from __future__ import annotations
import re
import json
import logging
from typing import Any
import time
from openai import OpenAI
import openai

from config import LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS, get_llm_runtime_config  # pyright: ignore[reportImplicitRelativeImport]
from retriever_reranker import SingleCellRetriever, RetrievedChunk  # pyright: ignore[reportImplicitRelativeImport]

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ─── Prompt 模板 ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert computational biologist specializing in single-cell genomics \
and cell type annotation. You have deep knowledge of marker genes, GO terms, \
KEGG pathways, and the tumor microenvironment.

Your task is to annotate a cell cluster from single-cell RNA-seq or scATAC-seq \
data based on its marker genes and relevant literature evidence provided in the context.

Output MUST be valid JSON with this schema:
{
  "cell_type": "<canonical cell type name>",
  "subtype": "<specific subtype if applicable, else null>",
  "confidence": <float 0.0-1.0>,
  "key_markers": ["<gene1>", "<gene2>", ...],
  "go_terms": ["<GO:XXXXXXX description>", ...],
  "kegg_pathways": ["<pathway name>", ...],
  "attribution": "<1-2 sentence mechanistic explanation referencing the context>",
  "supporting_references": ["<PMID or title>", ...]
}

Rules:
- Base your annotation ONLY on the provided context and your expert knowledge.
- If context is insufficient, set confidence < 0.5 and explain in attribution.
- gene symbols should be uppercase (e.g., CD8A, FOXP3, CD19).
- Do NOT add markdown formatting outside the JSON block.
"""

RAG_PROMPT_TEMPLATE = """\
## Retrieved Literature Context
{context}

---

## Cell Cluster to Annotate
- **Tissue/Disease**: {tissue}
- **Technology**: {technology}
- **Top Marker Genes** (ranked by log2FC): {markers}
- **Additional Info**: {extra_info}

Based on the literature context above, annotate this cell cluster.
Return ONLY a valid JSON object as specified.
"""


# ─── LLM 客户端 ───────────────────────────────────────────────────────────────

def get_llm_client() -> OpenAI:
    runtime = get_llm_runtime_config()
    api_key = runtime["api_key"]
    base_url = runtime["base_url"]
    provider = runtime["provider"]

    if not api_key:
        raise ValueError(
            f"{provider} API key 未配置。请设置环境变量："
            f"{'DEEPSEEK_API_KEY' if provider == 'deepseek' else 'OPENAI_API_KEY'}"
        )

    return OpenAI(api_key=api_key, base_url=base_url)


def call_llm(system: str, user: str, max_retries: int = 3) -> str:
    """调用 LLM，带重试机制以应对不稳定的网络环境"""
    client = get_llm_client()
    runtime = get_llm_runtime_config()
    model_name = runtime["model"] or LLM_MODEL
    prompt = f"{system}\n\n{user}"
    
    for attempt in range(max_retries):
        try:
            if runtime["provider"] == "deepseek":
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=LLM_TEMPERATURE,
                    max_tokens=LLM_MAX_TOKENS,
                )
                return (resp.choices[0].message.content or "").strip()

            resp = client.responses.create(
                model=model_name,
                input=prompt,
                store=True,
            )
            return resp.output_text
        except (openai.PermissionDeniedError, openai.APIConnectionError, openai.RateLimitError) as e:
            if isinstance(e, openai.RateLimitError):
                wait_time = 25  # 强制等待至少 20s，给 25s 更稳
                log.warning(f"触发频率限制 (429): {e}. 强制等待 {wait_time}s 后重试...")
            else:
                wait_time = (attempt + 1) * 5
                log.warning(f"LLM 调用失败 (尝试 {attempt+1}/{max_retries}): {e}. 等待 {wait_time}s 后重试...")
            
            if attempt < max_retries - 1:
                time.sleep(wait_time)
            else:
                log.error("已达到最大重试次数，任务失败。")
                raise e
        except Exception as e:
            log.error(f"LLM 发生未知错误: {e}")
            raise e

    raise RuntimeError("LLM 调用失败：超过最大重试次数。")


# ─── RAG Pipeline ─────────────────────────────────────────────────────────────

class SingleCellRAGPipeline:
    """
    给定 marker gene 列表 → 检索文献 → 构建 prompt → 调用 LLM → 返回注释结果
    """

    def __init__(self, top_n: int = 20, top_k: int = 5):
        self.retriever = SingleCellRetriever(top_n=top_n, top_k=top_k)

    def _build_query(self, markers: list[str], tissue: str, technology: str) -> str:
        """从 marker genes 构建检索 query"""
        top_markers = ", ".join(markers[:8])
        return (
            f"cell type annotation marker genes {top_markers} "
            f"{tissue} {technology} scRNA-seq single cell"
        )

    def _build_context(self, chunks: list[RetrievedChunk]) -> str:
        """将检索结果格式化为 prompt 上下文"""
        parts = []
        for i, chunk in enumerate(chunks):
            parts.append(
                f"[{i+1}] {chunk.title} ({chunk.year}) — {chunk.section}\n"
                f"{chunk.text}\n"
                f"(Relevance Score: {chunk.score:.2f})"
            )
        return "\n\n".join(parts)

    def annotate_cluster(
        self,
        markers: list[str],
        tissue: str = "unknown",
        technology: str = "scRNA-seq",
        extra_info: str = "",
        cluster_id: str = "",
    ) -> dict[str, Any]:
        """
        主入口：注释单个细胞簇

        Args:
            markers:    marker gene 列表（已按 log2FC 排序）
            tissue:     组织/疾病类型（如 "lung adenocarcinoma"）
            technology: 测序技术（scRNA-seq / scATAC-seq / multiome）
            extra_info: 其他已知信息（如 "cluster 5, 342 cells"）
            cluster_id: 簇 ID（用于日志/输出）

        Returns:
            dict with keys: cell_type, subtype, confidence, key_markers,
                            go_terms, kegg_pathways, attribution,
                            supporting_references, retrieved_chunks
        """
        log.info(f"Annotating cluster {cluster_id}: markers={markers[:5]}")

        # Stage 1: 检索
        query = self._build_query(markers, tissue, technology)
        chunks = self.retriever.retrieve(query)

        if not chunks:
            log.warning("无检索结果，仅凭 LLM 参数知识注释")
            context_str = "No relevant literature found in the knowledge base."
        else:
            context_str = self._build_context(chunks)

        # Stage 2: 构建 prompt
        user_prompt = RAG_PROMPT_TEMPLATE.format(
            context    = context_str,
            tissue     = tissue,
            technology = technology,
            markers    = ", ".join(markers),
            extra_info = extra_info or "N/A",
        )

        # Stage 3: 调用 LLM
        runtime = get_llm_runtime_config()
        log.info(f"Calling LLM ({runtime['provider']} / {runtime['model']})...")
        raw_output = call_llm(SYSTEM_PROMPT, user_prompt)

        # Stage 4: 解析 JSON 输出
        result = self._parse_output(raw_output)
        if result.get("cell_type") == "Unknown":
            log.warning("首次 JSON 解析失败，尝试一次严格 JSON 重试...")
            repair_prompt = (
                user_prompt
                + "\n\nIMPORTANT: Return ONLY one complete valid JSON object. "
                  "No markdown, no explanation, no trailing text."
            )
            raw_output_retry = call_llm(SYSTEM_PROMPT, repair_prompt, max_retries=2)
            retry_result = self._parse_output(raw_output_retry)
            if retry_result.get("cell_type") != "Unknown":
                result = retry_result

        result["cluster_id"]        = cluster_id
        result["query_used"]        = query
        result["retrieved_chunks"]  = [
            {"pmid": c.pmid, "title": c.title, "score": c.score}
            for c in chunks
        ]
        return result

    def _extract_first_json_object(self, text: str) -> str | None:
        """从文本中提取第一个完整 JSON 对象字符串。"""
        start = text.find("{")
        if start < 0:
            return None

        depth = 0
        in_str = False
        quote_char = ""
        escaped = False

        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if escaped:
                    escaped = False
                    continue
                if ch == "\\":
                    escaped = True
                    continue
                if ch == quote_char:
                    in_str = False
                continue

            if ch in ("'", '"'):
                in_str = True
                quote_char = ch
                continue
            if ch == "{":
                depth += 1
                continue
            if ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
                continue
        return None

    def _parse_output(self, raw: str) -> dict[str, Any]:
        """安全解析 LLM JSON 输出"""
        # 去除可能的 markdown 代码块包裹
        clean = re.sub(r"```(?:json)?\s*", "", raw).strip()
        clean = re.sub(r"\s*```$", "", clean).strip()

        try:
            return json.loads(clean)
        except json.JSONDecodeError as e:
            extracted = self._extract_first_json_object(clean)
            if extracted:
                try:
                    return json.loads(extracted)
                except json.JSONDecodeError:
                    pass
            log.error(f"JSON parse failed: {e}\nRaw: {raw[:200]}")
            return {
                "cell_type":    "Unknown",
                "subtype":      None,
                "confidence":   0.0,
                "key_markers":  [],
                "go_terms":     [],
                "kegg_pathways":[],
                "attribution":  "LLM output parsing failed.",
                "supporting_references": [],
                "raw_output":   raw,
            }

    def annotate_batch(
        self,
        cluster_list: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        批量注释多个簇

        cluster_list 每条格式：
          {"cluster_id": "0", "markers": ["CD8A","GZMB",...], "tissue": "...", ...}
        """
        results = []
        for i, cluster in enumerate(cluster_list):
            log.info(f"Processing cluster {i+1}/{len(cluster_list)}")
            res = self.annotate_cluster(**cluster)
            results.append(res)
        return results


# ─── 快速测试 ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pipeline = SingleCellRAGPipeline()

    # 示例：肺癌肿瘤浸润 T 细胞簇
    result = pipeline.annotate_cluster(
        markers    = ["CD8A", "CD8B", "GZMB", "GZMK", "PRF1", "IFNG", "LAG3", "PDCD1"],
        tissue     = "lung adenocarcinoma tumor microenvironment",
        technology = "scRNA-seq",
        extra_info = "Cluster 3, 512 cells, highly exhausted phenotype",
        cluster_id = "cluster_3",
    )

    print("\n=== Annotation Result ===")
    print(json.dumps(result, indent=2, ensure_ascii=False))
