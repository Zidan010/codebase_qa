"""
summarize_module.py

Tool: summarize_module(module_name)

Generate a structured summary of a Python module including:
- Purpose (from module docstring)
- Public API (exported functions and classes)
- Dependencies (what it imports)
- Key classes and their methods
- File location

This tool gives the agent a high-level overview of a module without
reading the entire file — useful for architecture questions and
understanding what a module provides before diving deeper.
"""

import ast
from pathlib import Path
from core.config import REPO_LOCAL_PATH


def summarize_module(module_name: str) -> dict:
    """
    Generate a structured summary of a module in the codebase.

    Args:
        module_name: Module name in dotted notation or filename.
                     e.g. "requests.sessions"
                     e.g. "requests.adapters"
                     e.g. "sessions"          (partial name also works)
                     e.g. "sessions.py"       (filename also works)

    Returns:
        Dict with keys:
          module       : resolved module name
          file         : relative file path
          purpose      : module docstring (first paragraph)
          public_api   : list of exported names (__all__ if defined, else public names)
          classes      : list of class summaries (name, bases, docstring, methods)
          functions    : list of top-level function summaries
          dependencies : list of imported modules
          constants    : list of module-level constants
          tool         : "summarize_module"
    """
    if not module_name or not module_name.strip():
        return _error("Module name cannot be empty.", "")

    repo_root = Path(REPO_LOCAL_PATH)
    file_path = _resolve_module_path(module_name.strip(), repo_root)

    if file_path is None:
        suggestions = _suggest_modules(module_name.strip(), repo_root)
        msg = f"Module '{module_name}' not found."
        if suggestions:
            msg += f" Did you mean: {', '.join(suggestions)}?"
        return _error(msg, module_name)

    # Read and parse file
    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(content)
    except SyntaxError as e:
        return _error(f"Syntax error in {file_path.name}: {e}", module_name)
    except Exception as e:
        return _error(f"Could not read module: {e}", module_name)

    rel_path = str(file_path.relative_to(repo_root))
    lines = content.splitlines()

    #  Extract components 
    purpose      = ast.get_docstring(tree) or ""
    dependencies = _extract_imports(tree)
    constants    = _extract_constants(tree, lines)
    classes      = _extract_classes(tree, lines)
    functions    = _extract_functions(tree)
    public_api   = _extract_public_api(tree, classes, functions)

    return {
        "tool":         "summarize_module",
        "module":       _path_to_module(file_path, repo_root),
        "file":         rel_path,
        "purpose":      purpose.split("\n\n")[0].strip() if purpose else "No module docstring found.",
        "public_api":   public_api,
        "classes":      classes,
        "functions":    functions,
        "dependencies": dependencies,
        "constants":    constants,
    }


#  AST Extractors 

def _extract_imports(tree: ast.Module) -> list[str]:
    """Extract all imported module names."""
    imports = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            names = [a.name for a in node.names]
            if len(names) <= 3:
                imports.append(f"{module} → {', '.join(names)}")
            else:
                imports.append(f"{module} → {', '.join(names[:3])} (+{len(names)-3} more)")
    return imports


def _extract_constants(tree: ast.Module, lines: list[str]) -> list[str]:
    """Extract module-level CONSTANT = value assignments."""
    constants = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    name = target.id
                    # Only include UPPER_CASE constants
                    if name.isupper() or name.startswith("DEFAULT_"):
                        line_idx = node.lineno - 1
                        raw = lines[line_idx].strip() if line_idx < len(lines) else ""
                        constants.append(raw[:100])
    return constants


def _extract_classes(tree: ast.Module, lines: list[str]) -> list[dict]:
    """Extract class summaries: name, bases, docstring, methods."""
    classes = []
    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, ast.ClassDef):
            continue

        # Base class names
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                parts = []
                n = base
                while isinstance(n, ast.Attribute):
                    parts.append(n.attr)
                    n = n.value
                if isinstance(n, ast.Name):
                    parts.append(n.id)
                bases.append(".".join(reversed(parts)))

        # Methods
        methods = []
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                doc = ast.get_docstring(child) or ""
                args = [a.arg for a in child.args.args if a.arg != "self"]
                methods.append({
                    "name":      child.name,
                    "args":      args,
                    "docstring": doc.split("\n")[0][:120] if doc else "",
                    "is_async":  isinstance(child, ast.AsyncFunctionDef),
                    "line":      child.lineno,
                })

        classes.append({
            "name":      node.name,
            "bases":     bases,
            "docstring": (ast.get_docstring(node) or "").split("\n")[0][:200],
            "methods":   methods,
            "line":      node.lineno,
        })
    return classes


def _extract_functions(tree: ast.Module) -> list[dict]:
    """Extract top-level function summaries."""
    funcs = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            doc = ast.get_docstring(node) or ""
            args = [a.arg for a in node.args.args]
            funcs.append({
                "name":      node.name,
                "args":      args,
                "docstring": doc.split("\n")[0][:120] if doc else "",
                "is_async":  isinstance(node, ast.AsyncFunctionDef),
                "line":      node.lineno,
            })
    return funcs


def _extract_public_api(
    tree: ast.Module,
    classes: list[dict],
    functions: list[dict],
) -> list[str]:
    """Return public API: __all__ if defined, else all public names."""
    # Check for __all__
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    if isinstance(node.value, ast.List | ast.Tuple):
                        return [
                            elt.s if isinstance(elt, ast.Constant) else ast.unparse(elt)
                            for elt in node.value.elts
                        ]

    # No __all__: return non-private names
    public = []
    for c in classes:
        if not c["name"].startswith("_"):
            public.append(f"class {c['name']}")
    for f in functions:
        if not f["name"].startswith("_"):
            public.append(f"def {f['name']}")
    return public


#  Path helpers 

def _resolve_module_path(module_name: str, repo_root: Path) -> Path | None:
    """
    Convert a module name to a file path. Tries multiple strategies:
    1. Exact dotted path: requests.sessions → requests/sessions.py
    2. Filename match: sessions.py or sessions
    3. Partial name match (last segment)
    """
    # Strategy 1: dotted module path
    as_path = module_name.replace(".", "/")
    candidates = [
        repo_root / f"{as_path}.py",
        repo_root / as_path / "__init__.py",
    ]
    for c in candidates:
        if c.exists():
            return c

    # Strategy 2: filename match (strip .py if present)
    name = module_name.rstrip(".py") if module_name.endswith(".py") else module_name
    name_lower = name.lower()

    for py_file in repo_root.rglob("*.py"):
        if py_file.stem.lower() == name_lower:
            return py_file

    # Strategy 3: partial name match
    for py_file in repo_root.rglob("*.py"):
        if name_lower in py_file.stem.lower():
            return py_file

    return None


def _suggest_modules(name: str, repo_root: Path) -> list[str]:
    """Suggest similar module names."""
    suggestions = []
    name_lower = name.lower()
    for py_file in repo_root.rglob("*.py"):
        if name_lower[:4] in py_file.stem.lower():
            rel = str(py_file.relative_to(repo_root))
            suggestions.append(rel)
            if len(suggestions) >= 3:
                break
    return suggestions


def _path_to_module(file_path: Path, repo_root: Path) -> str:
    try:
        rel = file_path.relative_to(repo_root)
        parts = list(rel.parts)
        if parts[-1].endswith(".py"):
            parts[-1] = parts[-1][:-3]
        if parts[-1] == "__init__":
            parts = parts[:-1]
        return ".".join(parts)
    except ValueError:
        return file_path.stem


def _error(msg: str, module: str) -> dict:
    return {
        "tool":    "summarize_module",
        "module":  module,
        "error":   msg,
    }