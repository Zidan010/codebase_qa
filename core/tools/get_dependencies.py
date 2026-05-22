"""
get_dependencies.py

Tool: get_dependencies(module_name)

Return the direct import dependencies of a given module —
what it imports, from where, and whether those are internal
(within the repo) or external (third-party/stdlib).

Used when the agent needs to answer:
  - "What does requests.sessions depend on?"
  - "What external libraries does adapters.py use?"
  - "What is the dependency chain for auth.py?"
"""

import ast
import sys
from pathlib import Path
from core.config import REPO_LOCAL_PATH

# Python stdlib module names (top-level)
# Used to classify imports as stdlib vs third-party
STDLIB_TOP_LEVEL = set(sys.stdlib_module_names) if hasattr(sys, "stdlib_module_names") else {
    "os", "sys", "re", "io", "abc", "ast", "copy", "time", "json",
    "math", "enum", "uuid", "socket", "struct", "hashlib", "logging",
    "typing", "pathlib", "datetime", "functools", "itertools",
    "collections", "contextlib", "threading", "urllib", "http",
    "email", "base64", "codecs", "textwrap", "warnings", "weakref",
    "traceback", "inspect", "importlib", "string", "random",
}


def get_dependencies(module_name: str) -> dict:
    """
    Return the import dependencies of a module.

    Args:
        module_name: Module in dotted notation, filename, or partial name.
                     e.g. "requests.sessions"
                     e.g. "sessions.py"
                     e.g. "adapters"

    Returns:
        Dict with keys:
          module         : resolved module name
          file           : relative file path
          internal       : list of imports from within the repo
          stdlib         : list of Python standard library imports
          third_party    : list of third-party package imports
          all_imports    : flat list of all import statements
          dependency_tree: text representation of imports grouped by kind
          tool           : "get_dependencies"
    """
    if not module_name or not module_name.strip():
        return _error("Module name cannot be empty.", "")

    repo_root = Path(REPO_LOCAL_PATH)
    file_path = _resolve_module(module_name.strip(), repo_root)

    if file_path is None:
        suggestions = _suggest(module_name.strip(), repo_root)
        msg = f"Module '{module_name}' not found."
        if suggestions:
            msg += f" Did you mean: {', '.join(suggestions)}?"
        return _error(msg, module_name)

    # Parse
    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(content)
    except SyntaxError as e:
        return _error(f"Syntax error: {e}", module_name)
    except Exception as e:
        return _error(f"Could not read: {e}", module_name)

    rel_path = str(file_path.relative_to(repo_root))
    resolved_module = _path_to_module(file_path, repo_root)

    # Extract all imports
    raw_imports = _extract_imports(tree)

    # Classify each import
    internal    = []
    stdlib      = []
    third_party = []
    all_imports = []

    for imp in raw_imports:
        top = imp["top_level"]
        stmt = imp["statement"]
        all_imports.append(stmt)

        if _is_internal(top, repo_root):
            internal.append({"module": top, "statement": stmt, "kind": "internal"})
        elif _is_stdlib(top):
            stdlib.append({"module": top, "statement": stmt, "kind": "stdlib"})
        else:
            third_party.append({"module": top, "statement": stmt, "kind": "third_party"})

    # Build readable dependency tree
    tree_text = _build_tree(resolved_module, internal, stdlib, third_party)

    return {
        "tool":            "get_dependencies",
        "module":          resolved_module,
        "file":            rel_path,
        "internal":        internal,
        "stdlib":          stdlib,
        "third_party":     third_party,
        "all_imports":     all_imports,
        "dependency_tree": tree_text,
        "totals": {
            "internal":    len(internal),
            "stdlib":      len(stdlib),
            "third_party": len(third_party),
        }
    }


#  Import extraction 

def _extract_imports(tree: ast.Module) -> list[dict]:
    """Extract all import statements from the AST."""
    imports = []

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append({
                    "top_level": alias.name.split(".")[0],
                    "statement": f"import {alias.name}"
                                 + (f" as {alias.asname}" if alias.asname else ""),
                })

        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            top = module.split(".")[0] if module else ""
            names = [
                a.name + (f" as {a.asname}" if a.asname else "")
                for a in node.names
            ]
            # Relative imports (level > 0) are internal
            if node.level and node.level > 0:
                top = "__relative__"
            imports.append({
                "top_level": top,
                "statement": f"from {'.' * (node.level or 0)}{module} import "
                             + ", ".join(names),
            })

    return imports


def _is_internal(top_level: str, repo_root: Path) -> bool:
    """Check if a top-level module name exists within the repo."""
    if top_level in ("__relative__", ""):
        return True
    # Check if a directory or .py file with this name exists in repo
    return (
        (repo_root / top_level).is_dir() or
        (repo_root / f"{top_level}.py").exists() or
        any(p.name == top_level for p in repo_root.iterdir() if p.is_dir())
    )


def _is_stdlib(top_level: str) -> bool:
    return top_level in STDLIB_TOP_LEVEL


def _build_tree(module: str, internal: list, stdlib: list, third_party: list) -> str:
    """Build a human-readable dependency tree string."""
    lines = [f"Dependencies of: {module}", "=" * 50]

    if internal:
        lines.append("\n[Internal (repo)]")
        for i in internal:
            lines.append(f"  ├ {i['statement']}")

    if stdlib:
        lines.append("\n[Standard Library]")
        for i in stdlib:
            lines.append(f"  ├ {i['statement']}")

    if third_party:
        lines.append("\n[Third-Party]")
        for i in third_party:
            lines.append(f"  ├ {i['statement']}")

    if not any([internal, stdlib, third_party]):
        lines.append("  (no imports found)")

    return "\n".join(lines)


#  Path helpers 

def _resolve_module(name: str, repo_root: Path) -> Path | None:
    """
    Resolve a module name to its file path.
    Handles both flat layout (requests/sessions.py) and
    src layout (src/requests/sessions.py). Prefers src/ over tests/.
    """
    as_path = name.replace(".", "/")

    # Strategy 1: exact dotted path from repo root
    for candidate in [
        repo_root / f"{as_path}.py",
        repo_root / as_path / "__init__.py",
    ]:
        if candidate.exists():
            return candidate

    # Strategy 2: src-layout
    for candidate in [
        repo_root / "src" / f"{as_path}.py",
        repo_root / "src" / as_path / "__init__.py",
    ]:
        if candidate.exists():
            return candidate

    # Strategy 3: stem match — use last segment only, rank src/ first
    stem = Path(name).stem.lower().split(".")[-1]
    matches = [f for f in repo_root.rglob("*.py") if f.stem.lower() == stem]
    if matches:
        return _rank_matches(matches, repo_root)[0]

    # Strategy 4: partial stem match
    partial = [f for f in repo_root.rglob("*.py") if stem in f.stem.lower()]
    if partial:
        return _rank_matches(partial, repo_root)[0]

    return None


def _rank_matches(matches: list[Path], repo_root: Path) -> list[Path]:
    """Sort candidates: src/ first, test dirs last."""
    def priority(p: Path) -> int:
        parts = p.relative_to(repo_root).parts
        if "src" in parts:
            return 0
        if parts[0].startswith("test"):
            return 2
        return 1
    return sorted(matches, key=priority)


def _suggest(name: str, repo_root: Path) -> list[str]:
    stem = name.split(".")[-1].lower()
    candidates = [f for f in repo_root.rglob("*.py") if stem[:4].lower() in f.stem.lower()]
    ranked = _rank_matches(candidates, repo_root)[:3]
    return [str(f.relative_to(repo_root)) for f in ranked]


def _path_to_module(file_path: Path, repo_root: Path) -> str:
    try:
        parts = list(file_path.relative_to(repo_root).parts)
        if parts[-1].endswith(".py"):
            parts[-1] = parts[-1][:-3]
        if parts[-1] == "__init__":
            parts = parts[:-1]
        return ".".join(parts)
    except ValueError:
        return file_path.stem


def _error(msg: str, module: str) -> dict:
    return {
        "tool":   "get_dependencies",
        "module": module,
        "error":  msg,
    }