"""
indexer.py

Orchestrates the full ingestion pipeline:
  1. Check if vector store already has data → skip if yes (unless force)
  2. Build repository structure map → store as first chunk
  3. Walk all indexable files via cloner.get_indexable_files()
  4. Parse each file via parser.parse_file()
  5. Batch-insert all chunks into ChromaDB via chroma_store.add_chunks()
  6. Print a final summary table of what was indexed

Re-index safety:
  - On clean start: full pipeline runs
  - On restart (store exists): skips everything, prints status
  - With --reindex flag: resets store, runs full pipeline fresh
"""

import time
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from core.config import (
    ALLOWED_EXTENSIONS,
    REPO_LOCAL_PATH,
)
from core.ingestion.cloner import get_indexable_files
from core.ingestion.parser import parse_file, build_structure_map, get_parser_stats
from core.vectorstore.chroma_store import get_store

console = Console()


def build_index(force_reindex: bool = False) -> None:
    """
    Main indexing entry point called from main.py.

    Args:
        force_reindex: If True, wipe existing index and rebuild from scratch.
                       If False, skip indexing if store already has data.
    """
    store = get_store()
    repo_path = Path(REPO_LOCAL_PATH)

    # Guard: repo must exist before indexing
    if not repo_path.exists():
        console.print(
            "[bold red]Error:[/] Repository not found at "
            f"{repo_path}. Run setup first."
        )
        return

    # Skip-if-exists logic
    if not force_reindex and store.is_populated():
        count = store.count()
        console.print(
            f"[green]✓ Vector store already populated[/] "
            f"[dim]({count:,} chunks — skipping re-index)[/]\n"
            f"  [dim]Use --reindex to force a fresh index.[/]"
        )
        return

    # Force reindex: wipe existing data
    if force_reindex and store.is_populated():
        console.print("[yellow]⚠ Force reindex: wiping existing vector store...[/]")
        store.reset()

    # Start pipeline
    start_time = time.time()
    console.print(f"\n[bold]Starting indexing pipeline[/] for [green]{repo_path.name}[/]\n")

    # Step 1: Structure map (stored first — always retrievable)
    console.print("[bold cyan]Step 1/3:[/] Building repository structure map...")
    structure_chunk = build_structure_map(repo_path)
    store.add_chunks([structure_chunk])
    console.print(
        f"  [green]✓[/] Structure map stored "
        f"[dim]({len(structure_chunk['text'].splitlines())} lines)[/]"
    )

    # Step 2: Discover all indexable files
    console.print("\n[bold cyan]Step 2/3:[/] Discovering indexable files...")
    files = get_indexable_files(repo_path, ALLOWED_EXTENSIONS)
    console.print(f"  [green]✓[/] Found [bold]{len(files):,}[/] indexable files")

    # Step 3: Parse all files and collect chunks
    console.print("\n[bold cyan]Step 3/3:[/] Parsing and indexing files...")
    all_chunks = []
    skipped = 0
    file_errors = []

    for file_path in files:
        try:
            chunks = parse_file(file_path, repo_path)
            if chunks:
                all_chunks.extend(chunks)
            else:
                skipped += 1
        except Exception as e:
            file_errors.append((str(file_path.name), str(e)))
            skipped += 1

    # Report any parse errors (non-fatal)
    if file_errors:
        console.print(
            f"\n  [yellow]⚠ {len(file_errors)} file(s) had parse errors (skipped):[/]"
        )
        for fname, err in file_errors[:5]:
            console.print(f"    [dim]{fname}: {err[:80]}[/]")

    # Insert all chunks into ChromaDB
    if all_chunks:
        store.add_chunks(all_chunks)
    else:
        console.print("[yellow]⚠ No chunks were produced. Check your repo path.[/]")
        return

    # Final summary
    elapsed = time.time() - start_time
    stats = get_parser_stats(all_chunks)
    _print_summary(stats, files, skipped, elapsed, store.count())


def _print_summary(
    stats: dict,
    files: list,
    skipped: int,
    elapsed: float,
    total_in_store: int,
) -> None:
    """Print a rich summary table after indexing completes."""

    table = Table(
        title="Indexing Summary",
        box=box.ROUNDED,
        border_style="cyan",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Category", style="bold")
    table.add_column("Count", justify="right", style="green")

    table.add_row("Files discovered", f"{len(files):,}")
    table.add_row("Files skipped", f"{skipped:,}")
    table.add_row("Files parsed", f"{stats['files_parsed']:,}")
    table.add_row("─" * 20, "─" * 8)
    table.add_row("Total chunks", f"{stats['total']:,}")

    for ctype, count in sorted(stats["by_type"].items()):
        table.add_row(f"  └─ {ctype}", f"{count:,}")

    table.add_row("─" * 20, "─" * 8)

    for lang, count in sorted(stats["by_language"].items()):
        table.add_row(f"Language: {lang}", f"{count:,}")

    table.add_row("─" * 20, "─" * 8)
    table.add_row("Total in vector store", f"{total_in_store:,}")
    table.add_row("Time elapsed", f"{elapsed:.1f}s")

    console.print()
    console.print(table)
    console.print(
        Panel(
            "[bold green]✓ Indexing complete![/]  "
            "Vector store persisted to [dim]data/vectorstore/[/]\n"
            "Re-runs will skip indexing automatically.",
            border_style="green",
            padding=(0, 2),
        )
    )