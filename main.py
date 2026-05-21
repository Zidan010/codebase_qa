"""
main.py
───────
Single entry point for codebase_qa.

Usage:
    python main.py             # auto setup + launch CLI
    python main.py --setup     # only run setup (clone + index)
    python main.py --reindex   # force re-index even if store exists
    python main.py --cli       # skip setup check, go straight to CLI
    python main.py --help      # show help
"""

import sys
import argparse
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()


def print_banner() -> None:
    banner = Text(justify="center")
    banner.append("  CODEBASE Q&A\n", style="bold cyan")
    banner.append("  Agentic Q&A on ", style="dim")
    banner.append("psf/requests", style="bold green")
    banner.append("  •  Groq llama-3.3-70b  •  LangGraph  •  ChromaDB", style="dim")
    console.print(Panel(banner, border_style="cyan", padding=(1, 4)))


def check_python_version() -> None:
    if sys.version_info < (3, 10):
        console.print(
            f"[bold red]Error:[/] Python 3.10+ required. "
            f"You have {sys.version_info.major}.{sys.version_info.minor}."
        )
        sys.exit(1)


def run_setup(force_reindex: bool = False) -> None:
    """Clone repo and build vector index."""
    from core.config import validate_config
    from core.ingestion.cloner import clone_repository
    from core.ingestion.indexer import build_index

    errors = validate_config()
    if errors:
        for err in errors:
            console.print(f"[bold red]Config Error:[/] {err}")
        sys.exit(1)

    console.print("\n[bold cyan]Step 1/2:[/] Checking repository...")
    clone_repository()

    console.print("\n[bold cyan]Step 2/2:[/] Checking vector index...")
    build_index(force_reindex=force_reindex)

    console.print(
        "\n[bold green]✓ Setup complete![/] "
        "\nLaunching conversational interface...\n"
    )


def run_cli() -> None:
    """Launch the conversational CLI."""
    from cli import start_cli
    start_cli()

def main() -> None:
    check_python_version()
    print_banner()

    parser = argparse.ArgumentParser(
        description="codebase_qa — Agentic Q&A over psf/requests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py              # full setup + launch CLI\n"
            "  python main.py --setup      # setup only\n"
            "  python main.py --reindex    # force fresh index\n"
            "  python main.py --cli        # skip setup, go to CLI\n"
        )
    )
    parser.add_argument(
        "--setup", action="store_true",
        help="Run setup only (clone repo + build index)"
    )
    parser.add_argument(
        "--reindex", action="store_true",
        help="Force re-index even if vector store already exists"
    )
    parser.add_argument(
        "--cli", action="store_true",
        help="Skip setup check and go straight to CLI"
    )
    args = parser.parse_args()

    if args.cli:
        run_cli()
    elif args.setup:
        run_setup(force_reindex=args.reindex)
        console.print("[dim]Run [bold]python main.py --cli[/] to start the interface.[/]")
    else:
        # Default: setup if needed, then launch CLI
        run_setup(force_reindex=args.reindex)
        run_cli()


if __name__ == "__main__":
    main()