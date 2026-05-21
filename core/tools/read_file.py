"""
read_file.py

Tool: read_file(path, start_line, end_line)

Read raw file content from the cloned repository.
Optionally limited to a line range for targeted reading.

Used when the agent needs the exact source of a specific file or function
rather than a semantically retrieved chunk — e.g. when the user asks
"show me the full Session class" or "read requests/adapters.py lines 40-80".
"""

from pathlib import Path
from core.config import REPO_LOCAL_PATH, READ_FILE_MAX_LINES


def read_file(
    path: str,
    start_line: int | None = None,
    end_line: int | None = None,
) -> dict:
    """
    Read a file from the cloned repository.

    Args:
        path:       Relative path from repo root.
                    e.g. "requests/sessions.py"
                    e.g. "requests/adapters.py"
        start_line: First line to return (1-indexed, inclusive). Optional.
        end_line:   Last line to return (1-indexed, inclusive). Optional.
                    If omitted, reads to end of file (up to READ_FILE_MAX_LINES).

    Returns:
        Dict with keys:
          path        : echoed relative path
          total_lines : total lines in file
          start_line  : actual start line returned
          end_line    : actual end line returned
          content     : the file content string
          truncated   : True if content was cut at READ_FILE_MAX_LINES
          tool        : "read_file"
    """
    if not path or not path.strip():
        return _error("Path cannot be empty.", path)

    # Resolve and validate path
    repo_root = Path(REPO_LOCAL_PATH)
    # Strip leading slashes so Path doesn't treat as absolute
    clean_path = path.strip().lstrip("/").lstrip("\\")
    full_path = repo_root / clean_path

    # Security: prevent path traversal outside repo
    try:
        full_path = full_path.resolve()
        repo_root.resolve()
        full_path.relative_to(repo_root.resolve())
    except ValueError:
        return _error(
            f"Path '{path}' is outside the repository root. "
            "Only paths within the cloned repo are accessible.",
            path,
        )

    if not full_path.exists():
        # Try to give a helpful suggestion
        suggestions = _find_similar(clean_path, repo_root)
        msg = f"File not found: '{clean_path}'."
        if suggestions:
            msg += f" Did you mean: {', '.join(suggestions)}?"
        return _error(msg, path)

    if not full_path.is_file():
        return _error(f"'{clean_path}' is a directory, not a file. Use list_directory instead.", path)

    # Read file
    try:
        raw_content = full_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return _error(f"Could not read file: {e}", path)

    all_lines = raw_content.splitlines()
    total_lines = len(all_lines)

    # Resolve line range (1-indexed from user, 0-indexed internally)
    s = max(1, start_line or 1) - 1              # 0-indexed start
    e = min(total_lines, end_line or total_lines) # 0-indexed end (inclusive)

    # Enforce max lines cap
    truncated = False
    if (e - s) > READ_FILE_MAX_LINES:
        e = s + READ_FILE_MAX_LINES
        truncated = True

    selected = all_lines[s:e]
    content = "\n".join(selected)

    return {
        "tool":        "read_file",
        "path":        clean_path,
        "total_lines": total_lines,
        "start_line":  s + 1,       # back to 1-indexed for output
        "end_line":    e,
        "content":     content,
        "truncated":   truncated,
        "truncated_msg": (
            f"Output capped at {READ_FILE_MAX_LINES} lines. "
            f"Use start_line/end_line to read further sections."
        ) if truncated else "",
    }


def _find_similar(target: str, repo_root: Path) -> list[str]:
    """Find files with similar names to help the user correct typos."""
    target_name = Path(target).name.lower()
    matches = []
    try:
        for f in repo_root.rglob("*.py"):
            if target_name in f.name.lower() or f.name.lower() in target_name:
                rel = str(f.relative_to(repo_root))
                matches.append(rel)
                if len(matches) >= 3:
                    break
    except Exception:
        pass
    return matches


def _error(msg: str, path: str) -> dict:
    return {
        "tool":    "read_file",
        "path":    path,
        "error":   msg,
        "content": "",
    }