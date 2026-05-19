"""
chroma_store.py

Persistent vector store built on ChromaDB with local sentence-transformers
embeddings. No API key required for embeddings — everything runs locally.

Responsibilities:
  - Initialise ChromaDB client (persists to data/vectorstore/ on disk)
  - Embed chunks using sentence-transformers all-MiniLM-L6-v2
  - Add chunks in batches (handles large codebases without OOM)
  - Query with optional metadata filters (language, chunk_type, module, etc.)
  - Check if index already exists (skip-re-index logic for indexer)
  - Delete and recreate collection when force re-index is requested

Design decisions:
  - We use ChromaDB's built-in embedding function interface BUT we manage
    the embedding model ourselves (sentence-transformers) so we can control
    the device (cpu/cuda) and batch size explicitly.
  - Embeddings are computed BEFORE adding to ChromaDB so we pass
    pre-computed embeddings directly — this avoids ChromaDB trying to
    call an external embedding API.
  - Metadata values must be str/int/float/bool for ChromaDB — we enforce
    this with _sanitize_metadata().
"""

import time
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, BarColumn,
    TextColumn, MofNCompleteColumn, TimeElapsedColumn
)

from core.config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
    EMBEDDING_MODEL,
    EMBEDDING_DEVICE,
)

console = Console()

#  Batch size for embedding + inserting 
# Keeps memory usage bounded on standard PCs
EMBED_BATCH_SIZE = 64


# ═══════════════════════════════════════════════════════════════════════════════
#  ChromaStore class
# ═══════════════════════════════════════════════════════════════════════════════

class ChromaStore:
    """
    Wrapper around ChromaDB for the codebase_qa system.

    Usage:
        store = ChromaStore()
        store.add_chunks(chunks)           # add list of chunk dicts
        results = store.query("how does Session work?", top_k=5)
        results = store.query("auth flow", filters={"language": "python"})
    """

    def __init__(self):
        self._client = None
        self._collection = None
        self._embedder = None
        self._initialised = False

    #  Lazy initialisation 

    def initialise(self) -> None:
        """
        Set up ChromaDB client, collection, and embedding model.
        Called once — subsequent calls are no-ops.
        """
        if self._initialised:
            return

        #  ChromaDB persistent client 
        persist_dir = Path(CHROMA_PERSIST_DIR)
        persist_dir.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )

        # Get or create collection
        # Using cosine distance — best for semantic similarity of text embeddings
        self._collection = self._client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        #  Embedding model
        console.print(
            f"[dim]Loading embedding model:[/] {EMBEDDING_MODEL} "
            f"[dim](device: {EMBEDDING_DEVICE})[/]"
        )
        self._embedder = SentenceTransformer(
            EMBEDDING_MODEL,
            device=EMBEDDING_DEVICE,
        )

        self._initialised = True
        console.print(
            f"[green]✓ Vector store ready[/] "
            f"[dim]({self._collection.count()} chunks already indexed)[/]"
        )

    #  Index check 

    def is_populated(self) -> bool:
        """
        Returns True if the collection already has chunks.
        Used by the indexer to skip re-indexing on restart.
        """
        self.initialise()
        return self._collection.count() > 0

    def count(self) -> int:
        """Return number of chunks currently in the store."""
        self.initialise()
        return self._collection.count()

    #  Add chunks 

    def add_chunks(self, chunks: list[dict]) -> None:
        """
        Embed and insert a list of chunk dicts into ChromaDB.

        Args:
            chunks: List of dicts with keys: id, text, metadata
                    (as produced by parser.py)
        """
        self.initialise()

        if not chunks:
            console.print("[yellow]⚠ No chunks to add.[/]")
            return

        total = len(chunks)
        console.print(f"\n[bold]Embedding and indexing[/] {total:,} chunks...")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Indexing...", total=total)

            for batch_start in range(0, total, EMBED_BATCH_SIZE):
                batch = chunks[batch_start : batch_start + EMBED_BATCH_SIZE]

                ids       = [c["id"] for c in batch]
                texts     = [c["text"] for c in batch]
                metadatas = [_sanitize_metadata(c["metadata"]) for c in batch]

                # Compute embeddings locally
                embeddings = self._embedder.encode(
                    texts,
                    batch_size=EMBED_BATCH_SIZE,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                ).tolist()

                # Insert into ChromaDB
                # upsert = insert or update if id already exists
                # This makes re-runs safe without duplicates
                self._collection.upsert(
                    ids=ids,
                    documents=texts,
                    embeddings=embeddings,
                    metadatas=metadatas,
                )

                progress.advance(task, len(batch))

        console.print(
            f"[bold green]✓ Indexed {total:,} chunks[/]  "
            f"[dim](total in store: {self._collection.count():,})[/]"
        )

    #  Query 

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict]:
        """
        Semantic search over the indexed chunks.

        Args:
            query_text: Natural language query string.
            top_k:      Number of results to return.
            filters:    Optional metadata filters. Examples:
                        {"language": "python"}
                        {"chunk_type": "function"}
                        {"module": "requests.adapters"}
                        {"language": "python", "chunk_type": "class"}

        Returns:
            List of result dicts, each containing:
            {
                "id":       chunk id,
                "text":     chunk text content,
                "metadata": chunk metadata dict,
                "score":    cosine distance (lower = more similar),
            }
            Sorted by relevance (most relevant first).
        """
        self.initialise()

        if not query_text or not query_text.strip():
            return []

        # Embed the query
        query_embedding = self._embedder.encode(
            [query_text.strip()],
            show_progress_bar=False,
            convert_to_numpy=True,
        ).tolist()

        # Build ChromaDB where clause from filters
        where_clause = _build_where_clause(filters) if filters else None

        try:
            results = self._collection.query(
                query_embeddings=query_embedding,
                n_results=min(top_k, self._collection.count()),
                where=where_clause,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            console.print(f"[yellow]⚠ Query error: {e}[/]")
            return []

        # Unpack ChromaDB response format
        formatted = []
        if results and results.get("ids") and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                formatted.append({
                    "id":       chunk_id,
                    "text":     results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "score":    results["distances"][0][i],
                })

        return formatted

    #  Fetch by metadata 

    def get_by_source(self, relative_path: str) -> list[dict]:
        """
        Retrieve all chunks from a specific file by its relative path.
        Used by read_file tool to get all chunks of one file.

        Args:
            relative_path: e.g. "requests/adapters.py"

        Returns:
            List of chunk dicts sorted by start_line.
        """
        self.initialise()
        try:
            results = self._collection.get(
                where={"source": relative_path},
                include=["documents", "metadatas"],
            )
            chunks = []
            if results and results.get("ids"):
                for i, chunk_id in enumerate(results["ids"]):
                    chunks.append({
                        "id":       chunk_id,
                        "text":     results["documents"][i],
                        "metadata": results["metadatas"][i],
                        "score":    0.0,
                    })
            # Sort by start_line
            chunks.sort(key=lambda c: c["metadata"].get("start_line", 0))
            return chunks
        except Exception as e:
            console.print(f"[yellow]⚠ get_by_source error: {e}[/]")
            return []

    def get_structure_map(self) -> dict | None:
        """
        Retrieve the pre-built repository structure map chunk.
        Returns None if not found.
        """
        self.initialise()
        try:
            results = self._collection.get(
                where={"chunk_type": "structure_map"},
                include=["documents", "metadatas"],
            )
            if results and results.get("ids") and results["ids"]:
                return {
                    "id":       results["ids"][0],
                    "text":     results["documents"][0],
                    "metadata": results["metadatas"][0],
                    "score":    0.0,
                }
        except Exception as e:
            console.print(f"[yellow]⚠ get_structure_map error: {e}[/]")
        return None

    #  Delete / reset 

    def reset(self) -> None:
        """
        Delete and recreate the collection.
        Used when --reindex flag is passed.
        """
        self.initialise()
        try:
            self._client.delete_collection(CHROMA_COLLECTION_NAME)
            self._collection = self._client.get_or_create_collection(
                name=CHROMA_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            console.print("[green]✓ Vector store reset. Ready for fresh indexing.[/]")
        except Exception as e:
            console.print(f"[bold red]Error resetting store:[/] {e}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _sanitize_metadata(metadata: dict) -> dict:
    """
    ChromaDB only accepts str, int, float, bool as metadata values.
    Convert anything else to string. Never let None slip through.
    """
    sanitized = {}
    for key, value in metadata.items():
        if value is None:
            sanitized[key] = ""
        elif isinstance(value, (str, int, float, bool)):
            sanitized[key] = value
        else:
            sanitized[key] = str(value)
    return sanitized


def _build_where_clause(filters: dict[str, Any]) -> dict | None:
    """
    Build a ChromaDB $and where clause from a flat filters dict.

    Single filter  → {"key": {"$eq": value}}
    Multiple       → {"$and": [{"k1": {"$eq": v1}}, {"k2": {"$eq": v2}}]}

    Returns None if filters is empty.
    """
    if not filters:
        return None

    conditions = [
        {key: {"$eq": value}}
        for key, value in filters.items()
        if value is not None
    ]

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


#  Module-level singleton 
# One shared instance used across the entire application.
# Lazy-initialised on first use.
_store_instance: ChromaStore | None = None


def get_store() -> ChromaStore:
    """
    Return the global ChromaStore singleton.
    Initialises on first call.
    """
    global _store_instance
    if _store_instance is None:
        _store_instance = ChromaStore()
    return _store_instance