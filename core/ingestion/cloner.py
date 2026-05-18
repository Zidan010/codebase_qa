"""
cloner.py

Clones the target repository (psf/requests) into data/repos/ at runtime.

Behaviour:
- If the repo already exists locally → skip clone, print status
- Shows real-time clone progress via Rich
- Handles network errors, invalid paths, and git failures gracefully
- Never crashes on unexpected input
"""

import sys
import shutil
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
import git
from git import Repo, GitCommandError, InvalidGitRepositoryError

from core.config import REPO_URL, REPO_NAME, REPO_LOCAL_PATH

console = Console()


#  Progress callback for git clone 

class CloneProgress(git.RemoteProgress):
    """
    Hooks into GitPython's RemoteProgress to show live clone progress.
    Called by GitPython during clone operation.
    """

    def __init__(self, progress_obj, task_id):
        super().__init__()
        self._progress = progress_obj
        self._task_id = task_id
        self._last_message = ""

    def update(self, op_code, cur_count, max_count=None, message=""):
        """Called by GitPython on each progress update from git."""
        if message and message != self._last_message:
            self._last_message = message
            self._progress.update(
                self._task_id,
                description=f"[cyan]Cloning psf/requests...[/] {message}"
            )


#  Main clone function 

def clone_repository(force: bool = False) -> Path:
    """
    Clone psf/requests into data/repos/requests.

    Args:
        force: If True, delete existing clone and re-clone fresh.

    Returns:
        Path to the cloned repository root.

    Raises:
        SystemExit on unrecoverable errors.
    """

    repo_path = Path(REPO_LOCAL_PATH)

    #  Handle force re-clone 
    if force and repo_path.exists():
        console.print(f"[yellow]⚠ Force flag set. Removing existing clone at {repo_path}[/]")
        try:
            shutil.rmtree(repo_path)
            console.print("[green]✓ Existing clone removed.[/]")
        except OSError as e:
            console.print(f"[bold red]Error removing existing repo:[/] {e}")
            sys.exit(1)

    #  Skip if already cloned 
    if repo_path.exists():
        if _is_valid_git_repo(repo_path):
            console.print(
                f"[green]✓ Repository already cloned[/] at "
                f"[dim]{repo_path}[/]  [dim](skipping clone)[/]"
            )
            _print_repo_info(repo_path)
            return repo_path
        else:
            # Directory exists but is not a valid git repo — remove and re-clone
            console.print(
                f"[yellow]⚠ Found directory at {repo_path} but it's not a valid git repo. "
                f"Removing and re-cloning...[/]"
            )
            shutil.rmtree(repo_path)

    #  Ensure parent directory exists 
    repo_path.parent.mkdir(parents=True, exist_ok=True)

    #  Clone with progress display 
    console.print(f"\n[bold]Cloning[/] [green]{REPO_URL}[/]")
    console.print(f"[dim]Destination:[/] {repo_path}\n")

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Cloning psf/requests...[/]", total=None)
            clone_progress = CloneProgress(progress, task)

            Repo.clone_from(
                url=REPO_URL,
                to_path=str(repo_path),
                progress=clone_progress,
                depth=1,        # shallow clone — we don't need full history
            )

        console.print(f"\n[bold green]✓ Clone complete![/]")
        _print_repo_info(repo_path)
        return repo_path

    except GitCommandError as e:
        _handle_clone_error(e)

    except Exception as e:
        console.print(f"\n[bold red]Unexpected error during clone:[/] {e}")
        console.print("[dim]Check your internet connection and try again.[/]")
        sys.exit(1)


#  Helpers 

def _is_valid_git_repo(path: Path) -> bool:
    """Check if a path contains a valid git repository."""
    try:
        Repo(str(path))
        return True
    except InvalidGitRepositoryError:
        return False
    except Exception:
        return False


def _print_repo_info(repo_path: Path):
    """Print basic info about the cloned repository."""
    try:
        repo = Repo(str(repo_path))
        branch = repo.active_branch.name
        commit = repo.head.commit
        console.print(
            f"  [dim]Branch:[/] {branch}  "
            f"[dim]Last commit:[/] {str(commit.hexsha)[:7]}  "
            f"[dim]by[/] {commit.author.name}"
        )
    except Exception:
        # Non-critical — just skip info display
        pass


def _handle_clone_error(e: GitCommandError):
    """Parse GitCommandError and show a helpful message."""
    error_str = str(e).lower()

    if "repository not found" in error_str or "does not exist" in error_str:
        console.print(f"\n[bold red]Error:[/] Repository not found at {REPO_URL}")
        console.print("[dim]Check the REPO_URL in core/config.py[/]")

    elif "unable to connect" in error_str or "could not resolve" in error_str:
        console.print("\n[bold red]Error:[/] Cannot reach GitHub.")
        console.print("[dim]Check your internet connection and try again.[/]")

    elif "already exists" in error_str:
        console.print("\n[bold red]Error:[/] Destination directory already exists.")
        console.print("[dim]Run with --reindex flag or delete data/repos/ manually.[/]")

    else:
        console.print(f"\n[bold red]Git error during clone:[/] {e}")
        console.print("[dim]Try deleting data/repos/ and running again.[/]")

    sys.exit(1)


#  File inventory helper (used by indexer) 

def get_indexable_files(repo_path: Path, allowed_extensions: set) -> list[Path]:
    """
    Walk the cloned repo and return all files eligible for indexing.

    Rules:
    - Only files with extensions in allowed_extensions are included
    - Hidden directories (starting with '.') are skipped
    - __pycache__ directories are skipped
    - node_modules, .git directories are skipped

    Args:
        repo_path: Root path of the cloned repository
        allowed_extensions: Set of file extensions to include e.g. {'.py', '.md'}

    Returns:
        Sorted list of Path objects for all indexable files
    """
    SKIP_DIRS = {".git", "__pycache__", "node_modules", ".tox", ".eggs", "dist", "build"}

    indexable = []

    for file_path in repo_path.rglob("*"):
        # Skip directories
        if not file_path.is_file():
            continue

        # Skip files inside ignored directories
        if any(part in SKIP_DIRS or part.startswith(".") for part in file_path.parts):
            continue

        # Only allow whitelisted extensions
        if file_path.suffix.lower() in allowed_extensions:
            indexable.append(file_path)

    return sorted(indexable)