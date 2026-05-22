"""
search_code.py

Tool: search_code(query, top_k, filters)

Semantic search over the indexed codebase using ChromaDB + embeddings.
Supports optional metadata filters to narrow results by language,
chunk_type, module, or file.

This is the primary retrieval tool — used by the agent for most queries
before deciding whether deeper tools (read_file, find_usages) are needed.
"""

from typing import Any
from core.vectorstore.chroma_store import get_store
from core.config import SEARCH_DEFAULT_TOP_K


def search_code(
    query: str,
    top_k: int = SEARCH_DEFAULT_TOP_K,
    filters: dict[str, Any] | None = None,
) -> dict:
    """
    Perform semantic search over the indexed codebase.

    Args:
        query:   Natural language search query.
                 e.g. "how does Session handle authentication"
        top_k:   Number of results to return (default 5, max 20).
        filters: Optional metadata filters. Supported keys:
                   language    : "python" | "markdown" | "text" | "config"
                   chunk_type  : "function" | "class" | "module" | "plaintext"
                   module      : e.g. "requests.adapters"
                   file_name   : e.g. "sessions.py"
                 Example: {"language": "python", "chunk_type": "function"}

    Returns:
        Dict with keys:
          query      : echoed back
          total      : number of results found
          results    : list of result dicts, each with:
                         rank, symbol, file, lines, chunk_type,
                         module, docstring, snippet (first 300 chars)
          tool       : "search_code"
    """
    if not query or not query.strip():
        return _error("Query cannot be empty.", "search_code")

    top_k = max(1, min(top_k, 20))   # clamp 1–20

    # Normalize module filter: the vector store indexes modules with their
    # full path-based name (e.g. "src.requests.sessions"). If the LLM passes
    # "requests.sessions" (without the "src." prefix), try both variants.
    filters_to_try = [filters]
    if filters and "module" in filters:
        raw_module = filters["module"]
        # If it doesn't already start with "src.", add a fallback with "src." prefix
        if not raw_module.startswith("src."):
            alt_filters = dict(filters)
            alt_filters["module"] = f"src.{raw_module}"
            filters_to_try = [alt_filters, filters]  # try src. version first

    store = get_store()
    raw = None
    for f in filters_to_try:
        raw = store.query(query.strip(), top_k=top_k, filters=f)
        if raw:
            break

    if not raw:
        return {
            "tool": "search_code",
            "query": query,
            "total": 0,
            "results": [],
            "message": "No results found. Try a broader query or different keywords.",
        }

    results = []
    for i, item in enumerate(raw):
        meta = item["metadata"]
        snippet = item["text"][:300].strip()
        if len(item["text"]) > 300:
            snippet += "..."

        results.append({
            "rank":       i + 1,
            "symbol":     meta.get("symbol_name", ""),
            "file":       meta.get("source", ""),
            "lines":      f"{meta.get('start_line', '?')}–{meta.get('end_line', '?')}",
            "chunk_type": meta.get("chunk_type", ""),
            "module":     meta.get("module", ""),
            "docstring":  meta.get("docstring", ""),
            "language":   meta.get("language", ""),
            "score":      round(item["score"], 4),
            "snippet":    snippet,
        })

    return {
        "tool":    "search_code",
        "query":   query,
        "filters": filters or {},
        "total":   len(results),
        "results": results,
    }


def _error(msg: str, tool: str) -> dict:
    return {"tool": tool, "error": msg, "results": []}