from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PACKAGE_DIR = Path(__file__).resolve().parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover - depends on optional runtime dependency
    FastMCP = None  # type: ignore[assignment]
    _MCP_IMPORT_ERROR = exc
else:
    _MCP_IMPORT_ERROR = None

from agent_tools_core import (  # noqa: E402
    query_go_terms_core,
    query_kegg_pathways_core,
    rag_annotate_core,
    search_literature_core,
    validate_annotation_core,
)


def create_server() -> Any:
    if FastMCP is None:
        raise RuntimeError(
            "The MCP Python SDK is not installed. Install it with `pip install mcp` "
            "or use AGENT_TOOL_BACKEND=local."
        ) from _MCP_IMPORT_ERROR

    mcp = FastMCP("scAnno Agent Tools", json_response=True)

    @mcp.tool()
    def search_literature(query: str) -> dict[str, Any]:
        """Search the local literature RAG index for relevant chunks."""
        return search_literature_core(query)

    @mcp.tool()
    def query_go_terms(gene_list: list[str]) -> dict[str, Any]:
        """Query GO terms for a list of marker genes."""
        return query_go_terms_core(gene_list)

    @mcp.tool()
    def query_kegg_pathways(gene_list: list[str]) -> dict[str, Any]:
        """Query KEGG pathways for a list of marker genes."""
        return query_kegg_pathways_core(gene_list)

    @mcp.tool()
    def rag_annotate(
        markers: list[str],
        tissue: str = "unknown",
        technology: str = "scRNA-seq",
        cluster_id: str = "0",
    ) -> dict[str, Any]:
        """Run the RAG annotation pipeline for one cluster."""
        return rag_annotate_core(markers, tissue=tissue, technology=technology, cluster_id=cluster_id)

    @mcp.tool()
    def validate_annotation(annotation: dict[str, Any]) -> dict[str, Any]:
        """Validate a structured annotation result."""
        return validate_annotation_core(annotation)

    return mcp


def main() -> None:
    server = create_server()
    server.run(transport="stdio")


if __name__ == "__main__":
    main()

