"""
list_directory.py

Tool: list_directory(path, recursive, max_depth)

Explore the repository folder structure.
Returns a tree representation of directories and files.

Used when the agent needs to understand:
- What files exist in a given directory
- The overall package structure
- Where to look for a specific module
"""

from pathlib import Path
from core.config import REPO_LOCAL_PATH, ALLOWED_EXTENSIONS, LIST_DIR_MAX_DEPTH


def list_directory(
    path: str = ".",
    recursive: bool = True,
    max_depth: int = LIST_DIR_MAX_DEPTH,
) -> dict:
    """
    List the contents of a directory in the cloned repository.

    Args:
        path:      Relative path from repo root to explore.
                   Use "." or "" for the repo root.
                   e.g. "requests", "requests/packages"
        recursive: If True, recurse into subdirectories (default True).
        max_depth: Maximum depth to recurse (default from config).

    Returns:
        Dict with keys:
          path      : directory path explored
          tree      : text tree representation
          files     : flat list of all file paths found
          dirs      : flat list of all directory paths found
          total     : total file count
          tool      : "list_directory"
    """
    repo_root = Path(REPO_LOCAL_PATH)

    # Resolve target directory
    clean_path = (path or ".").strip().lstrip("/").lstrip("\\")
    if clean_path in (".", ""):
        target = repo_root
    else:
        target = repo_root / clean_path

    # Security: stay within repo
    try:
        target = target.resolve()
        target.relative_to(repo_root.resolve())
    except ValueError:
        return _error(f"Path '{path}' is outside the repository.", path)

    if not target.exists():
        return _error(f"Directory not found: '{clean_path}'", path)

    if not target.is_dir():
        return _error(f"'{clean_path}' is a file, not a directory. Use read_file instead.", path)

    # Build tree
    tree_lines = [f"{target.name}/"]
    all_files = []
    all_dirs = []

    _walk_tree(
        path=target,
        repo_root=repo_root,
        tree_lines=tree_lines,
        all_files=all_files,
        all_dirs=all_dirs,
        prefix="",
        depth=0,
        max_depth=max_depth if recursive else 1,
    )

    return {
        "tool":  "list_directory",
        "path":  clean_path or ".",
        "tree":  "\n".join(tree_lines),
        "files": all_files,
        "dirs":  all_dirs,
        "total": len(all_files),
    }


#  Helpers 

SKIP_DIRS = {
    ".git", "__pycache__", "node_modules",
    ".tox", ".eggs", "dist", "build", ".mypy_cache",
}


def _walk_tree(
    path: Path,
    repo_root: Path,
    tree_lines: list,
    all_files: list,
    all_dirs: list,
    prefix: str,
    depth: int,
    max_depth: int,
) -> None:
    if depth >= max_depth:
        return

    try:
        # Dirs first, then files — alphabetical within each group
        entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    except PermissionError:
        return

    # Filter out skip dirs and hidden entries
    entries = [
        e for e in entries
        if not (e.name in SKIP_DIRS or e.name.startswith("."))
    ]

    for i, entry in enumerate(entries):
        is_last = (i == len(entries) - 1)
        connector = "└ " if is_last else "├ "
        extension = "    " if is_last else "│   "
        rel = str(entry.relative_to(repo_root))

        if entry.is_dir():
            tree_lines.append(f"{prefix}{connector}{entry.name}/")
            all_dirs.append(rel)
            _walk_tree(
                entry, repo_root, tree_lines, all_files, all_dirs,
                prefix + extension, depth + 1, max_depth,
            )
        elif entry.is_file() and entry.suffix.lower() in ALLOWED_EXTENSIONS:
            size_kb = round(entry.stat().st_size / 1024, 1)
            tree_lines.append(f"{prefix}{connector}{entry.name}  [{size_kb}KB]")
            all_files.append(rel)


def _error(msg: str, path: str) -> dict:
    return {
        "tool":  "list_directory",
        "path":  path,
        "error": msg,
        "tree":  "",
        "files": [],
        "dirs":  [],
        "total": 0,
    }