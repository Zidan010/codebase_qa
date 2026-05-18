"""
parser.py

Code-aware parser and chunker for the psf/requests codebase.

Strategy:
  .py  files → AST-based chunking: each function/class becomes one chunk.
               Module-level code (imports, constants) becomes its own chunk.
               Preserves exact line numbers, docstrings, and nesting info.

  .md / .rst / .txt / .cfg / .toml / .yml files
             → Plain-text sliding window chunks with line-range metadata.

  Everything else → skipped (binaries, compiled, etc.)

Special:
  build_structure_map() → walks the full repo tree and produces a text
  representation of the directory/file structure stored as a special chunk
  in ChromaDB so structure questions always have a pre-built answer.

Each chunk is a dict with these keys (all metadata stored in ChromaDB):
  {
    "id":          unique string id for this chunk,
    "text":        the actual text content to embed,
    "metadata": {
      "source":       relative file path from repo root,
      "file_name":    basename,
      "extension":    .py / .md / etc,
      "chunk_type":   "function" | "class" | "module" | "plaintext" | "structure_map",
      "symbol_name":  function/class name (or "" for plain text),
      "parent_class": parent class name if method (or ""),
      "module":       dotted module name e.g. "requests.adapters",
      "start_line":   int,
      "end_line":     int,
      "docstring":    first line of docstring (or ""),
      "language":     "python" | "markdown" | "text" | "config",
    }
  }
"""

import ast
import hashlib
import textwrap
from pathlib import Path
from typing import Any

from rich.console import Console

from core.config import (
    ALLOWED_EXTENSIONS,
    BINARY_EXTENSIONS,
    MAX_CHUNK_LINES,
    MIN_CHUNK_LINES,
    CHUNK_OVERLAP_LINES,
    REPO_LOCAL_PATH,
)

console = Console()

#  Extension → language label 
EXT_LANGUAGE = {
    ".py":   "python",
    ".md":   "markdown",
    ".rst":  "restructuredtext",
    ".txt":  "text",
    ".cfg":  "config",
    ".toml": "config",
    ".yml":  "yaml",
    ".yaml": "yaml",
}


# ═══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

def parse_file(file_path: Path, repo_root: Path) -> list[dict]:
    """
    Parse a single file into a list of chunks.

    Args:
        file_path: Absolute path to the file.
        repo_root: Absolute path to the repo root (for relative path calculation).

    Returns:
        List of chunk dicts. Empty list if file should be skipped.
    """
    # Safety: skip binaries explicitly
    if file_path.suffix.lower() in BINARY_EXTENSIONS:
        return []

    if file_path.suffix.lower() not in ALLOWED_EXTENSIONS:
        return []

    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        console.print(f"[yellow]⚠ Could not read {file_path.name}: {e}[/]")
        return []

    # Skip empty or near-empty files
    if len(content.strip()) < 20:
        return []

    relative_path = str(file_path.relative_to(repo_root))
    ext = file_path.suffix.lower()

    if ext == ".py":
        return _parse_python(content, file_path, relative_path, repo_root)
    else:
        return _parse_plaintext(content, file_path, relative_path, ext)


def build_structure_map(repo_root: Path) -> dict:
    """
    Build a full directory tree of the repository as a single special chunk.
    This is stored in ChromaDB so questions like:
      "What is the folder structure?" or "What modules does requests have?"
    always have a pre-built, accurate answer.

    Returns:
        A single chunk dict with chunk_type="structure_map".
    """
    SKIP_DIRS = {
        ".git", "__pycache__", "node_modules",
        ".tox", ".eggs", "dist", "build", ".mypy_cache"
    }

    lines = [f"Repository structure: {repo_root.name}\n"]
    lines.append("=" * 60)

    def _walk(path: Path, prefix: str = "", depth: int = 0):
        if depth > 8:   # prevent absurd nesting
            return
        try:
            entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except PermissionError:
            return

        entries = [
            e for e in entries
            if not (e.name in SKIP_DIRS or e.name.startswith("."))
        ]

        for i, entry in enumerate(entries):
            is_last = (i == len(entries) - 1)
            connector = "└ " if is_last else "├ "
            extension = "    " if is_last else "│   "

            if entry.is_dir():
                lines.append(f"{prefix}{connector}{entry.name}/")
                _walk(entry, prefix + extension, depth + 1)
            else:
                # Only show indexable files in the map
                if entry.suffix.lower() in ALLOWED_EXTENSIONS:
                    lines.append(f"{prefix}{connector}{entry.name}")

    _walk(repo_root)

    # Also append a flat list of all Python modules for easy lookup
    lines.append("\n" + "=" * 60)
    lines.append("Python modules (dotted paths):")
    py_files = sorted(repo_root.rglob("*.py"))
    for py_file in py_files:
        skip = False
        for part in py_file.parts:
            if part in SKIP_DIRS or part.startswith("."):
                skip = True
                break
        if skip:
            continue
        module = _path_to_module(py_file, repo_root)
        if module:
            lines.append(f"  {module}")

    structure_text = "\n".join(lines)
    chunk_id = _make_id("structure_map", "structure_map", 0)

    return {
        "id": chunk_id,
        "text": structure_text,
        "metadata": {
            "source":       "structure_map",
            "file_name":    "structure_map",
            "extension":    "",
            "chunk_type":   "structure_map",
            "symbol_name":  "repository_structure",
            "parent_class": "",
            "module":       "",
            "start_line":   0,
            "end_line":     len(lines),
            "docstring":    "Full directory and module structure of the repository",
            "language":     "text",
        }
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  PYTHON AST PARSER
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_python(
    content: str,
    file_path: Path,
    relative_path: str,
    repo_root: Path,
) -> list[dict]:
    """
    Parse a .py file using Python's built-in ast module.

    Extraction order:
      1. Module-level docstring + imports block → one "module" chunk
      2. Each top-level function → one "function" chunk
      3. Each top-level class → one "class" chunk (header + class docstring)
      4. Each method inside a class → one "function" chunk with parent_class set
      5. Any remaining module-level code → one "module" chunk

    If AST parsing fails (e.g. syntax error in the file being parsed),
    falls back to plain-text chunking so we never lose content.
    """
    lines = content.splitlines()
    module_name = _path_to_module(file_path, repo_root)
    chunks = []

    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        console.print(
            f"[yellow]⚠ AST parse failed for {file_path.name} "
            f"(line {e.lineno}): {e.msg}. Using plain-text fallback.[/]"
        )
        return _parse_plaintext(content, file_path, relative_path, ".py")

    #  Module-level chunk (imports + module docstring) 
    module_chunk = _extract_module_chunk(tree, lines, relative_path, module_name, file_path.name)
    if module_chunk:
        chunks.append(module_chunk)

    #  Top-level functions and classes 
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            chunk = _extract_function_chunk(
                node, lines, relative_path, module_name, file_path.name,
                parent_class=""
            )
            if chunk:
                chunks.append(chunk)

        elif isinstance(node, ast.ClassDef):
            # Class-level chunk (header + docstring)
            class_chunk = _extract_class_chunk(
                node, lines, relative_path, module_name, file_path.name
            )
            if class_chunk:
                chunks.append(class_chunk)

            # Each method as its own chunk
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                    method_chunk = _extract_function_chunk(
                        child, lines, relative_path, module_name, file_path.name,
                        parent_class=node.name
                    )
                    if method_chunk:
                        chunks.append(method_chunk)

    return [c for c in chunks if c is not None]


def _extract_module_chunk(
    tree: ast.Module,
    lines: list[str],
    relative_path: str,
    module_name: str,
    file_name: str,
) -> dict | None:
    """
    Extract module-level content: docstring + import block.
    This gives the agent context about what a module does overall.
    """
    # Collect import lines and module docstring
    import_lines = []
    module_docstring = ast.get_docstring(tree) or ""

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import | ast.ImportFrom):
            start = node.lineno - 1
            end = getattr(node, "end_lineno", node.lineno) - 1
            import_lines.extend(lines[start:end + 1])

    # Combine docstring + imports
    parts = []
    if module_docstring:
        parts.append(f'"""{module_docstring}"""')
    if import_lines:
        parts.append("\n".join(import_lines))

    text = "\n".join(parts).strip()
    if len(text.splitlines()) < MIN_CHUNK_LINES:
        return None

    return {
        "id": _make_id(relative_path, "module", 0),
        "text": f"# Module: {module_name or file_name}\n{text}",
        "metadata": {
            "source":       relative_path,
            "file_name":    file_name,
            "extension":    ".py",
            "chunk_type":   "module",
            "symbol_name":  module_name or file_name,
            "parent_class": "",
            "module":       module_name,
            "start_line":   1,
            "end_line":     len(import_lines) + (len(module_docstring.splitlines()) if module_docstring else 0),
            "docstring":    module_docstring.split("\n")[0][:200] if module_docstring else "",
            "language":     "python",
        }
    }


def _extract_function_chunk(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    lines: list[str],
    relative_path: str,
    module_name: str,
    file_name: str,
    parent_class: str,
) -> dict | None:
    """Extract a single function or method as a chunk."""
    start = node.lineno - 1          # 0-indexed
    end = getattr(node, "end_lineno", node.lineno) - 1

    chunk_lines = lines[start:end + 1]
    if len(chunk_lines) < MIN_CHUNK_LINES:
        return None

    # If function is very long, keep full text but note it
    text = "\n".join(chunk_lines)
    docstring = ast.get_docstring(node) or ""

    symbol = f"{parent_class}.{node.name}" if parent_class else node.name
    chunk_type = "function"

    return {
        "id": _make_id(relative_path, symbol, start),
        "text": text,
        "metadata": {
            "source":       relative_path,
            "file_name":    file_name,
            "extension":    ".py",
            "chunk_type":   chunk_type,
            "symbol_name":  symbol,
            "parent_class": parent_class,
            "module":       module_name,
            "start_line":   node.lineno,
            "end_line":     getattr(node, "end_lineno", node.lineno),
            "docstring":    docstring.split("\n")[0][:200] if docstring else "",
            "language":     "python",
        }
    }


def _extract_class_chunk(
    node: ast.ClassDef,
    lines: list[str],
    relative_path: str,
    module_name: str,
    file_name: str,
) -> dict | None:
    """
    Extract a class as a chunk.
    We include the class header + docstring + class-level attributes,
    but NOT the method bodies (those are separate chunks).
    """
    start = node.lineno - 1
    end = getattr(node, "end_lineno", node.lineno) - 1

    # Find where the first method starts
    first_method_line = end  # default: end of class
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
            first_method_line = child.lineno - 2  # line before first method
            break

    chunk_lines = lines[start:first_method_line + 1]
    text = "\n".join(chunk_lines).strip()

    if len(text.splitlines()) < 1:
        return None

    docstring = ast.get_docstring(node) or ""

    # Build base list info for class
    bases = []
    for base in node.bases:
        if isinstance(base, ast.Name):
            bases.append(base.id)
        elif isinstance(base, ast.Attribute):
            bases.append(f"{base.value.id}.{base.attr}" if isinstance(base.value, ast.Name) else base.attr)

    header = f"# Class: {node.name}"
    if bases:
        header += f" (inherits: {', '.join(bases)})"
    header += f"\n# Module: {module_name}"

    return {
        "id": _make_id(relative_path, node.name, start),
        "text": f"{header}\n{text}",
        "metadata": {
            "source":       relative_path,
            "file_name":    file_name,
            "extension":    ".py",
            "chunk_type":   "class",
            "symbol_name":  node.name,
            "parent_class": "",
            "module":       module_name,
            "start_line":   node.lineno,
            "end_line":     getattr(node, "end_lineno", node.lineno),
            "docstring":    docstring.split("\n")[0][:200] if docstring else "",
            "language":     "python",
        }
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  PLAIN TEXT PARSER (md, rst, txt, cfg, toml, yml)
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_plaintext(
    content: str,
    file_path: Path,
    relative_path: str,
    ext: str,
) -> list[dict]:
    """
    Chunk plain-text files using a sliding window of MAX_CHUNK_LINES lines
    with CHUNK_OVERLAP_LINES overlap between consecutive chunks.

    This ensures no content is lost at chunk boundaries.
    """
    lines = content.splitlines()
    language = EXT_LANGUAGE.get(ext, "text")
    file_name = file_path.name
    chunks = []

    step = MAX_CHUNK_LINES - CHUNK_OVERLAP_LINES
    total_lines = len(lines)
    chunk_index = 0

    for start in range(0, total_lines, step):
        end = min(start + MAX_CHUNK_LINES, total_lines)
        chunk_lines = lines[start:end]
        text = "\n".join(chunk_lines).strip()

        if len(text.splitlines()) < MIN_CHUNK_LINES:
            continue

        chunks.append({
            "id": _make_id(relative_path, "plaintext", start),
            "text": text,
            "metadata": {
                "source":       relative_path,
                "file_name":    file_name,
                "extension":    ext,
                "chunk_type":   "plaintext",
                "symbol_name":  "",
                "parent_class": "",
                "module":       "",
                "start_line":   start + 1,
                "end_line":     end,
                "docstring":    "",
                "language":     language,
            }
        })
        chunk_index += 1

        if end >= total_lines:
            break

    return chunks


# ═══════════════════════════════════════════════════════════════════════════════
#  UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def _make_id(source: str, symbol: str, start_line: int) -> str:
    """
    Generate a stable, unique chunk ID.
    Using a hash ensures IDs are safe for ChromaDB (no special chars).
    """
    raw = f"{source}::{symbol}::{start_line}"
    return hashlib.md5(raw.encode()).hexdigest()


def _path_to_module(file_path: Path, repo_root: Path) -> str:
    """
    Convert a file path to a Python dotted module name.
    e.g. data/repos/requests/requests/adapters.py → requests.adapters

    Returns empty string if path cannot be converted.
    """
    try:
        relative = file_path.relative_to(repo_root)
        parts = list(relative.parts)

        # Remove .py extension from last part
        if parts[-1].endswith(".py"):
            parts[-1] = parts[-1][:-3]

        # Remove __init__ suffix (it's just the package name)
        if parts[-1] == "__init__":
            parts = parts[:-1]

        return ".".join(parts) if parts else ""
    except ValueError:
        return ""


def get_parser_stats(chunks: list[dict]) -> dict[str, Any]:
    """
    Return summary statistics about a list of parsed chunks.
    Useful for logging and debugging during indexing.
    """
    stats: dict[str, Any] = {
        "total": len(chunks),
        "by_type": {},
        "by_language": {},
        "files_parsed": set(),
    }

    for chunk in chunks:
        meta = chunk["metadata"]
        ctype = meta["chunk_type"]
        lang = meta["language"]
        source = meta["source"]

        stats["by_type"][ctype] = stats["by_type"].get(ctype, 0) + 1
        stats["by_language"][lang] = stats["by_language"].get(lang, 0) + 1
        stats["files_parsed"].add(source)

    stats["files_parsed"] = len(stats["files_parsed"])
    return stats