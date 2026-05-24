from __future__ import annotations

import json
import re
import time
from typing import Any

import requests
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from rag_pipeline import SingleCellRAGPipeline
    from retriever_reranker import SingleCellRetriever


_retriever: "SingleCellRetriever | None" = None
_rag_pipeline: "SingleCellRAGPipeline | None" = None


def get_retriever() -> SingleCellRetriever:
    global _retriever
    if _retriever is None:
        try:
            from retriever_reranker import SingleCellRetriever
        except ImportError:  # pragma: no cover
            from .retriever_reranker import SingleCellRetriever
        _retriever = SingleCellRetriever()
    return _retriever


def get_rag_pipeline() -> SingleCellRAGPipeline:
    global _rag_pipeline
    if _rag_pipeline is None:
        try:
            from rag_pipeline import SingleCellRAGPipeline
        except ImportError:  # pragma: no cover
            from .rag_pipeline import SingleCellRAGPipeline
        _rag_pipeline = SingleCellRAGPipeline()
    return _rag_pipeline


def search_literature_core(query: str) -> dict[str, Any]:
    if not isinstance(query, str) or not query.strip():
        return {"ok": False, "error": "query must be a non-empty string", "chunks": [], "display_text": ""}
    try:
        chunks = get_retriever().retrieve(query)
        results = [
            {
                "chunk_id": c.chunk_id,
                "pmid": c.pmid,
                "title": c.title,
                "year": c.year,
                "section": c.section,
                "score": c.score,
                "text": c.text,
            }
            for c in chunks
        ]
        display = "\n".join(
            f"- [{c['pmid']}] {c['title']} ({c['year']})\n  {c['text'][:300]}..."
            for c in results
        )
        return {"ok": True, "error": None, "chunks": results, "display_text": display or "No relevant literature found."}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "chunks": [], "display_text": f"Tool error: {exc}"}


def query_go_terms_core(gene_list: list[str]) -> dict[str, Any]:
    if not gene_list:
        return {"ok": False, "error": "No genes provided.", "terms": [], "display_text": "No genes provided."}
    genes = [g.strip().upper() for g in gene_list if isinstance(g, str) and g.strip()][:20]
    if not genes:
        return {"ok": False, "error": "No valid genes provided.", "terms": [], "display_text": "No valid genes provided."}

    quickgo_url = "https://www.ebi.ac.uk/QuickGO/services/annotation/search"
    try:
        resp = requests.get(
            quickgo_url,
            timeout=15,
            headers={"Accept": "application/json"},
            params={
                "geneProductSymbol": ",".join(genes[:10]),
                "taxonId": "9606",
                "limit": 20,
                "page": 1,
            },
        )
        if resp.status_code == 200:
            data = resp.json()
            seen_go: set[str] = set()
            terms: list[dict[str, str]] = []
            for row in data.get("results", [])[:20]:
                go_id = str(row.get("goId", "")).strip()
                go_name = str(row.get("goName", "") or "").strip()
                aspect = str(row.get("goAspect", "") or "").strip()
                if go_id and go_name and go_name.lower() != "none" and go_id not in seen_go:
                    terms.append({"go_id": go_id, "name": go_name, "aspect": aspect, "source": "QuickGO"})
                    seen_go.add(go_id)
            if terms:
                return {
                    "ok": True,
                    "error": None,
                    "terms": terms[:10],
                    "display_text": "\n".join(f"{t['go_id']} ({t['aspect']}): {t['name']}" for t in terms[:10]),
                }
    except Exception:
        pass

    add_list_url = "https://maayanlab.cloud/Enrichr/addList"
    enrich_url = "https://maayanlab.cloud/Enrichr/enrich"
    try:
        add_resp = requests.post(
            add_list_url,
            timeout=20,
            files={
                "list": (None, "\n".join(genes)),
                "description": (None, "scanno_agent_go_fallback"),
            },
        )
        if add_resp.status_code != 200:
            return {
                "ok": False,
                "error": f"GO query failed (QuickGO + Enrichr): {add_resp.status_code}",
                "terms": [],
                "display_text": f"GO query failed (QuickGO + Enrichr): {add_resp.status_code}",
            }
        user_list_id = add_resp.json().get("userListId")
        if not user_list_id:
            return {"ok": False, "error": "Enrichr did not return userListId.", "terms": [], "display_text": "GO query failed."}

        rows: list[tuple[float, dict[str, Any]]] = []
        for db_name, tag in [
            ("GO_Biological_Process_2023", "BP"),
            ("GO_Molecular_Function_2023", "MF"),
            ("GO_Cellular_Component_2023", "CC"),
        ]:
            enr_resp = requests.get(enrich_url, timeout=20, params={"userListId": user_list_id, "backgroundType": db_name})
            if enr_resp.status_code != 200:
                continue
            for row in enr_resp.json().get(db_name, [])[:8]:
                if not isinstance(row, list) or len(row) < 3:
                    continue
                term_name = str(row[1]).strip()
                p_value = row[2]
                if not term_name:
                    continue
                go_match = re.search(r"(GO:\d{7})", term_name)
                go_id = go_match.group(1) if go_match else "GO:NA"
                p_sort = float(p_value) if isinstance(p_value, (int, float)) else 1.0
                rows.append(
                    (
                        p_sort,
                        {
                            "go_id": go_id,
                            "name": term_name,
                            "aspect": tag,
                            "p_value": p_value,
                            "source": "Enrichr",
                        },
                    )
                )
        rows.sort(key=lambda item: item[0])
        terms = [row[1] for row in rows[:10]]
        if not terms:
            return {"ok": True, "error": None, "terms": [], "display_text": "No GO annotations found."}
        return {
            "ok": True,
            "error": None,
            "terms": terms,
            "display_text": "\n".join(
                f"{t['go_id']} ({t['aspect']}, p={t.get('p_value', 'NA')}): {t['name']}"
                for t in terms
            ),
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc), "terms": [], "display_text": f"GO query failed: {exc}"}


def query_kegg_pathways_core(gene_list: list[str]) -> dict[str, Any]:
    if not gene_list:
        return {"ok": False, "error": "No genes provided.", "pathways": [], "display_text": "No genes provided."}

    genes = [g.strip().upper() for g in gene_list if isinstance(g, str) and g.strip()][:8]
    pathway_counts: dict[str, int] = {}

    for gene in genes:
        try:
            resp = requests.get(f"https://rest.kegg.jp/find/hsa/{gene}", timeout=10)
            if resp.status_code != 200:
                continue
            lines = resp.text.strip().split("\n")
            if not lines or not lines[0]:
                continue
            kegg_gene_id = lines[0].split("\t")[0]
            path_resp = requests.get(f"https://rest.kegg.jp/link/pathway/{kegg_gene_id}", timeout=10)
            for line in path_resp.text.strip().split("\n"):
                parts = line.split("\t")
                if len(parts) == 2:
                    pathway_counts[parts[1].strip()] = pathway_counts.get(parts[1].strip(), 0) + 1
            time.sleep(0.1)
        except Exception:
            continue

    if not pathway_counts:
        return {"ok": True, "error": None, "pathways": [], "display_text": "No KEGG pathways found."}

    pathways: list[dict[str, Any]] = []
    for pathway_id, hit_count in sorted(pathway_counts.items(), key=lambda x: -x[1])[:5]:
        try:
            name_resp = requests.get(f"https://rest.kegg.jp/list/{pathway_id}", timeout=8)
            name = name_resp.text.split("\t")[-1].strip() if name_resp.ok else pathway_id
        except Exception:
            name = pathway_id
        pathways.append({"pathway_id": pathway_id, "name": name, "hit_count": hit_count})

    return {
        "ok": True,
        "error": None,
        "pathways": pathways,
        "display_text": "\n".join(f"{p['pathway_id']}: {p['name']} (hit by {p['hit_count']} genes)" for p in pathways),
    }


def rag_annotate_core(
    markers: list[str],
    tissue: str = "unknown",
    technology: str = "scRNA-seq",
    cluster_id: str = "0",
) -> dict[str, Any]:
    try:
        result = get_rag_pipeline().annotate_cluster(
            markers=markers,
            tissue=tissue,
            technology=technology,
            cluster_id=cluster_id,
        )
        return {"ok": True, "error": None, "annotation": result, "display_text": json.dumps(result, ensure_ascii=False, indent=2)}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "annotation": None, "display_text": f"Tool error: {exc}"}


def validate_annotation_core(annotation: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(annotation, dict):
        return {"ok": False, "error": "Invalid JSON annotation.", "issues": ["Invalid JSON annotation."], "display_text": "Invalid JSON annotation."}

    confidence = annotation.get("confidence", 0)
    try:
        confidence_value = float(confidence)
    except Exception:
        confidence_value = 0.0
    markers = annotation.get("key_markers", [])
    refs = annotation.get("supporting_references", [])

    issues: list[str] = []
    if confidence_value < 0.5:
        issues.append("Low confidence (<0.5). Need more specific marker evidence.")
    if not isinstance(markers, list) or len(markers) < 2:
        issues.append("Too few key markers identified.")
    if not isinstance(refs, list) or not refs:
        issues.append("No supporting references found.")
    if not annotation.get("go_terms"):
        issues.append("GO terms missing. Consider querying GO database.")
    if not annotation.get("kegg_pathways"):
        issues.append("KEGG pathways missing. Consider querying KEGG.")

    if issues:
        return {
            "ok": True,
            "valid": False,
            "issues": issues,
            "display_text": "Issues found:\n" + "\n".join(f"- {issue}" for issue in issues),
        }
    return {
        "ok": True,
        "valid": True,
        "issues": [],
        "display_text": f"Annotation validated. Confidence: {confidence_value:.2f}. Cell type: {annotation.get('cell_type')}.",
    }
