"""
find_usages.py

Tool: find_usages(symbol_name)

Find all usages of a function, class, method, or variable
across the entire codebase using AST-based static analysis.

Unlike semantic search (which finds semantically similar text),
this tool finds EXACT symbol references — imports, calls,
instantiations, decorators, type hints, and inheritance.

Used when the agent needs to answer:
  - "Where is Session used?"
  - "What calls resolve_redirects?"
  - "Which modules import HTTPAdapter?"
"""

import ast
from pathlib import Path
from dataclasses import dataclass
from core.config import REPO_LOCAL_PATH, FIND_USAGES_MAX_RESULTS, ALLOWED_EXTENSIONS


@dataclass
class Usage:
    file:        str
    line:        int
    usage_type:  str   # "import" | "call" | "instantiation" | "inheritance" | "decorator" | "attribute" | "annotation"
    context:     str   # the actual source line


def find_usages(symbol_name: str) -> dict:
    """
    Find all usages of a symbol across the codebase.

    Args:
        symbol_name: Name to search for.
                     Can be a simple name: "Session"
                     Or dotted:            "requests.Session"
                     Or a method:          "Session.get"

    Returns:
        Dict with keys:
          symbol       : the symbol searched
          total        : total usages found
          usages       : list of usage dicts (file, line, type, context)
          summary      : grouped count by usage_type
          tool         : "find_usages"
    """
    if not symbol_name or not symbol_name.strip():
        return _error("Symbol name cannot be empty.", "")

    symbol = symbol_name.strip()
    # For dotted names like "Session.get", search for the leaf name
    # but filter results by context
    search_name = symbol.split(".")[-1]
    parent_name = symbol.split(".")[0] if "." in symbol else None

    repo_root = Path(REPO_LOCAL_PATH)
    usages: list[Usage] = []

    for file_path in _get_python_files(repo_root):
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
            lines = content.splitlines()
            tree = ast.parse(content)
        except Exception:
            continue

        rel_path = str(file_path.relative_to(repo_root))
        file_usages = _find_in_file(tree, lines, rel_path, search_name, parent_name)
        usages.extend(file_usages)

        if len(usages) >= FIND_USAGES_MAX_RESULTS:
            break

    usages = usages[:FIND_USAGES_MAX_RESULTS]

    # Group by usage type for summary
    summary: dict[str, int] = {}
    for u in usages:
        summary[u.usage_type] = summary.get(u.usage_type, 0) + 1

    if not usages:
        return {
            "tool":    "find_usages",
            "symbol":  symbol,
            "total":   0,
            "usages":  [],
            "summary": {},
            "message": (
                f"No usages of '{symbol}' found in the codebase. "
                "Check spelling or try a partial name."
            ),
        }

    return {
        "tool":    "find_usages",
        "symbol":  symbol,
        "total":   len(usages),
        "capped":  len(usages) >= FIND_USAGES_MAX_RESULTS,
        "summary": summary,
        "usages":  [
            {
                "file":    u.file,
                "line":    u.line,
                "type":    u.usage_type,
                "context": u.context.strip(),
            }
            for u in usages
        ],
    }


#  AST visitor 

def _find_in_file(
    tree: ast.Module,
    lines: list[str],
    rel_path: str,
    name: str,
    parent_name: str | None,
) -> list[Usage]:
    """Walk AST of one file and collect all usages of `name`."""
    usages = []

    def get_line(lineno: int) -> str:
        idx = lineno - 1
        return lines[idx].strip() if 0 <= idx < len(lines) else ""

    for node in ast.walk(tree):

        #  Import: from x import Session 
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == name or alias.asname == name:
                    usages.append(Usage(
                        file=rel_path, line=node.lineno,
                        usage_type="import",
                        context=get_line(node.lineno),
                    ))

        #  Import: import requests.Session 
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if name in alias.name:
                    usages.append(Usage(
                        file=rel_path, line=node.lineno,
                        usage_type="import",
                        context=get_line(node.lineno),
                    ))

        #  Function / method call: Session() or obj.get() 
        elif isinstance(node, ast.Call):
            func = node.func
            # Direct call: Session(...)
            if isinstance(func, ast.Name) and func.id == name:
                ctx = get_line(node.lineno)
                utype = "instantiation" if name[0].isupper() else "call"
                usages.append(Usage(
                    file=rel_path, line=node.lineno,
                    usage_type=utype, context=ctx,
                ))
            # Attribute call: session.get(...) or HTTPAdapter.send(...)
            elif isinstance(func, ast.Attribute) and func.attr == name:
                if parent_name is None or (
                    isinstance(func.value, ast.Name) and
                    func.value.id.lower() in (parent_name.lower(), name.lower())
                ):
                    usages.append(Usage(
                        file=rel_path, line=node.lineno,
                        usage_type="call",
                        context=get_line(node.lineno),
                    ))

        #  Class inheritance: class MySession(Session) 
        elif isinstance(node, ast.ClassDef):
            for base in node.bases:
                if isinstance(base, ast.Name) and base.id == name:
                    usages.append(Usage(
                        file=rel_path, line=node.lineno,
                        usage_type="inheritance",
                        context=get_line(node.lineno),
                    ))
                elif isinstance(base, ast.Attribute) and base.attr == name:
                    usages.append(Usage(
                        file=rel_path, line=node.lineno,
                        usage_type="inheritance",
                        context=get_line(node.lineno),
                    ))

        #  Decorator: @session_required 
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            for dec in node.decorator_list:
                if isinstance(dec, ast.Name) and dec.id == name:
                    usages.append(Usage(
                        file=rel_path, line=dec.lineno,
                        usage_type="decorator",
                        context=get_line(dec.lineno),
                    ))
                elif isinstance(dec, ast.Attribute) and dec.attr == name:
                    usages.append(Usage(
                        file=rel_path, line=dec.lineno,
                        usage_type="decorator",
                        context=get_line(dec.lineno),
                    ))

        #  Attribute access: self.session or obj.Session 
        elif isinstance(node, ast.Attribute) and node.attr == name:
            # Avoid double-counting calls (already caught above)
            if not isinstance(node.ctx, ast.Load):
                continue
            parent = node
            # Check it's not already part of a Call node
            usages.append(Usage(
                file=rel_path, line=node.lineno if hasattr(node, "lineno") else 0,
                usage_type="attribute",
                context=get_line(node.lineno) if hasattr(node, "lineno") else "",
            ))

    # Deduplicate by (file, line, type)
    seen = set()
    deduped = []
    for u in usages:
        key = (u.file, u.line, u.usage_type)
        if key not in seen:
            seen.add(key)
            deduped.append(u)

    return deduped


def _get_python_files(repo_root: Path) -> list[Path]:
    SKIP_DIRS = {".git", "__pycache__", "node_modules", ".tox", "dist", "build"}
    files = []
    for f in repo_root.rglob("*.py"):
        if not any(p in SKIP_DIRS or p.startswith(".") for p in f.parts):
            files.append(f)
    return sorted(files)


def _error(msg: str, symbol: str) -> dict:
    return {
        "tool":   "find_usages",
        "symbol": symbol,
        "error":  msg,
        "usages": [],
    }