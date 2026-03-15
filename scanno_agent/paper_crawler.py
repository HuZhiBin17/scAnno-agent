"""
Step 1 — 论文爬取与 PDF 下载
────────────────────────────
流程：
  PubMed E-utilities (esearch → efetch) 获取 PMID 列表
  → 解析 PubMed XML 获取元数据 + PMC/DOI
  → 优先从 PMC Open Access 下载全文 PDF
  → 其次通过 Unpaywall API 查询 Open Access PDF 链接
  → 保存 PDF 到 data/pdfs/，元数据写入 data/parsed/metadata.jsonl
"""
from __future__ import annotations
import time
import json
import hashlib
import logging
import requests
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional
from urllib.parse import quote_plus
import os
os.environ['NO_PROXY'] = '*'
from config import (
    PDF_DIR, PARSED_DIR,
    PUBMED_EMAIL, PUBMED_API_KEY,
    PUBMED_QUERIES, MAX_PAPERS_PER_QUERY,
    PUBMED_MAX_RETRIES, UNPAYWALL_EMAIL
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PMC_PDF_URL = "https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/"
UNPAYWALL_URL = "https://api.unpaywall.org/v2/{doi}?email={email}"


# ─── 工具函数 ─────────────────────────────────────────────────────────────────

def _get(url: str, params: dict = None, timeout: int = 30) -> Optional[requests.Response]:
    """带重试的 GET 请求"""
    # 强制设置 连接超时 和 读取/下载超时，防止 socket 假死
    strict_timeout = (timeout, timeout)
    for attempt in range(PUBMED_MAX_RETRIES):
        try:
            # stream=True 防止直接下载超大文件或卡死
            resp = requests.get(url, params=params, timeout=strict_timeout, stream=True,
                                headers={"User-Agent": f"scRAG/1.0 ({PUBMED_EMAIL})"})
            resp.raise_for_status()
            return resp
        except Exception as e:
            log.warning(f"Request failed ({attempt+1}/{PUBMED_MAX_RETRIES}): {e}")
            time.sleep(2 ** attempt)
    return None


def _uid(pmid: str) -> str:
    return hashlib.md5(pmid.encode()).hexdigest()[:8]


# ─── PubMed E-utilities ───────────────────────────────────────────────────────

def esearch_pmids(query: str, max_results: int) -> list[str]:
    """esearch：返回 PMID 列表"""
    params = dict(
        db="pubmed", term=query, retmax=max_results,
        retmode="json", email=PUBMED_EMAIL,
        usehistory="y",
    )
    if PUBMED_API_KEY:
        params["api_key"] = PUBMED_API_KEY

    resp = _get(f"{EUTILS_BASE}/esearch.fcgi", params)
    if not resp:
        return []
    data = resp.json()
    pmids = data.get("esearchresult", {}).get("idlist", [])
    log.info(f"esearch '{query[:50]}' → {len(pmids)} PMIDs")
    return pmids


def efetch_metadata(pmids: list[str]) -> list[dict]:
    """efetch：批量获取 PubMed XML 元数据"""
    if not pmids:
        return []
    params = dict(
        db="pubmed", id=",".join(pmids),
        rettype="xml", retmode="xml",
        email=PUBMED_EMAIL,
    )
    if PUBMED_API_KEY:
        params["api_key"] = PUBMED_API_KEY

    resp = _get(f"{EUTILS_BASE}/efetch.fcgi", params)
    if not resp:
        return []

    root = ET.fromstring(resp.text)
    papers = []
    for article in root.findall(".//PubmedArticle"):
        try:
            pmid   = article.findtext(".//PMID", "")
            title  = article.findtext(".//ArticleTitle", "")
            year   = article.findtext(".//PubDate/Year", "")
            # 作者列表
            authors = [
                f"{a.findtext('LastName','')} {a.findtext('ForeName','')}".strip()
                for a in article.findall(".//Author")
            ]
            # 摘要（可能多段）
            abstract_parts = article.findall(".//AbstractText")
            abstract = " ".join(a.text or "" for a in abstract_parts)
            # DOI & PMC
            doi = ""
            pmcid = ""
            for aid in article.findall(".//ArticleId"):
                if aid.attrib.get("IdType") == "doi":
                    doi = aid.text or ""
                elif aid.attrib.get("IdType") == "pmc":
                    pmcid = aid.text or ""
            papers.append(dict(
                pmid=pmid, title=title, year=year,
                authors=authors, abstract=abstract,
                doi=doi, pmcid=pmcid
            ))
        except Exception as e:
            log.warning(f"Parse error: {e}")
    return papers


# ─── PDF 下载 ─────────────────────────────────────────────────────────────────

def _save_pdf(content: bytes, pmid: str, title: str) -> Optional[Path]:
    """将 PDF bytes 写入磁盘，返回路径"""
    safe_title = "".join(c if c.isalnum() or c in "-_ " else "_" for c in title)[:60]
    filename = f"{pmid}_{safe_title}.pdf"
    path = PDF_DIR / filename
    if path.exists():
        log.info(f"  [skip] already downloaded: {filename}")
        return path
    path.write_bytes(content)
    log.info(f"  [saved] {filename} ({len(content)//1024} KB)")
    return path


def download_pmc_pdf(pmcid: str, pmid: str, title: str) -> Optional[Path]:
    """从 PMC Open Access 下载 PDF"""
    if not pmcid:
        return None
    url = PMC_PDF_URL.format(pmcid=pmcid)
    resp = _get(url, timeout=30)
    if resp and resp.headers.get("content-type", "").startswith("application/pdf"):
        return _save_pdf(resp.content, pmid, title)
    return None


def download_unpaywall_pdf(doi: str, pmid: str, title: str) -> Optional[Path]:
    """通过 Unpaywall 查找 Open Access PDF"""
    if not doi:
        return None
    url = UNPAYWALL_URL.format(doi=doi, email=UNPAYWALL_EMAIL)
    resp = _get(url, timeout=15)
    if not resp:
        return None
    data = resp.json()
    # 优先 best_oa_location
    loc = data.get("best_oa_location") or {}
    pdf_url = loc.get("url_for_pdf") or loc.get("url")
    if not pdf_url:
        return None
    pdf_resp = _get(pdf_url, timeout=60)
    if pdf_resp and "pdf" in pdf_resp.headers.get("content-type", ""):
        return _save_pdf(pdf_resp.content, pmid, title)
    return None


# ─── 主流程 ───────────────────────────────────────────────────────────────────

def crawl_and_download():
    """全量爬取入口"""
    meta_path = PARSED_DIR / "metadata.jsonl"
    existing_pmids: set[str] = set()

    # 读取已有元数据，避免重复
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                existing_pmids.add(row["pmid"])
    log.info(f"已有元数据 {len(existing_pmids)} 篇")

    all_pmids: set[str] = set()
    for query in PUBMED_QUERIES:
        pmids = esearch_pmids(query, MAX_PAPERS_PER_QUERY)
        all_pmids.update(pmids)
        time.sleep(0.4)   # 遵守 PubMed 速率限制

    new_pmids = [p for p in all_pmids if p not in existing_pmids]
    log.info(f"新增待处理 PMID: {len(new_pmids)}")
    if not new_pmids:
        return

    # 批量获取元数据（每批 100 条）
    papers_meta = []
    for i in range(0, len(new_pmids), 100):
        batch = new_pmids[i:i+100]
        papers_meta.extend(efetch_metadata(batch))
        time.sleep(0.5)

    # 下载 PDF
    with open(meta_path, "a", encoding="utf-8") as mf:
        for paper in papers_meta:
            pmid, pmcid, doi = paper["pmid"], paper["pmcid"], paper["doi"]
            title = paper["title"]
            log.info(f"处理 PMID={pmid}: {title[:60]}")

            pdf_path = (
                download_pmc_pdf(pmcid, pmid, title)
                or download_unpaywall_pdf(doi, pmid, title)
            )
            paper["pdf_path"] = str(pdf_path) if pdf_path else None
            paper["pdf_status"] = "ok" if pdf_path else "unavailable"

            mf.write(json.dumps(paper, ensure_ascii=False) + "\n")
            time.sleep(0.3)

    log.info("爬取完成 ✓")


if __name__ == "__main__":
    crawl_and_download()
