"""
trace_call_flow.py

Tool: trace_call_flow(symbol_name, depth)

Trace the execution flow starting from a given function or method.
Follows direct function calls N levels deep using AST analysis.

This is the "Code Navigation Agent" concept implemented as a tool.
Instead of building a full call graph (expensive), we trace forward
from a starting symbol and map what it calls, what those call, etc.

Used when the agent needs to answer:
  - "How does a request flow through the requests library?"
  - "What happens when Session.get() is called?"
  - "Walk me through the execution of send()"

Output is a readable call flow diagram showing the chain.
"""

import ast
from pathlib import Path
from core.config import REPO_LOCAL_PATH, TRACE_CALL_DEPTH


def trace_call_flow(
    symbol_name: str,
    depth: int = TRACE_CALL_DEPTH,
) -> dict:
    """
    Trace the call flow starting from a symbol.

    Args:
        symbol_name: Starting function or method.
                     e.g. "Session.request"
                     e.g. "send"
                     e.g. "resolve_redirects"
        depth:       How many levels deep to follow calls (default 2, max 4).
                     Higher depth = more complete but slower.

    Returns:
        Dict with keys:
          symbol      : starting symbol
          depth       : depth traced
          flow        : list of call levels, each a list of call dicts
          diagram     : readable text diagram of the call chain
          entry_file  : where the starting symbol was found
          entry_line  : line number of starting symbol
          tool        : "trace_call_flow"
    """
    if not symbol_name or not symbol_name.strip():
        return _error("Symbol name cannot be empty.", "")

    symbol = symbol_name.strip()
    depth = max(1, min(depth, 4))   # clamp 1–4

    repo_root = Path(REPO_LOCAL_PATH)
    all_definitions = _build_definition_index(repo_root)

    # Find the starting symbol
    start = _find_definition(symbol, all_definitions)
    if start is None:
        suggestions = [k for k in all_definitions if symbol.split(".")[-1].lower() in k.lower()][:3]
        msg = f"Symbol '{symbol}' not found in codebase."
        if suggestions:
            msg += f" Did you mean: {', '.join(suggestions)}?"
        return _error(msg, symbol)

    # Trace call flow
    flow_levels = []
    visited = set()
    current_level = [start]

    for level_idx in range(depth):
        next_level = []
        level_calls = []

        for sym_info in current_level:
            if sym_info["key"] in visited:
                continue
            visited.add(sym_info["key"])

            # Get calls made by this symbol
            calls = _get_calls_from(sym_info, all_definitions, repo_root)
            sym_info["calls_to"] = [c["key"] for c in calls]
            level_calls.append(sym_info)
            next_level.extend(calls)

        flow_levels.append(level_calls)
        if not next_level:
            break
        current_level = next_level

    diagram = _build_diagram(symbol, flow_levels)

    return {
        "tool":       "trace_call_flow",
        "symbol":     symbol,
        "depth":      depth,
        "entry_file": start["file"],
        "entry_line": start["line"],
        "diagram":    diagram,
        "flow":       [
            [
                {
                    "symbol":   s["key"],
                    "file":     s["file"],
                    "line":     s["line"],
                    "calls_to": s.get("calls_to", []),
                }
                for s in level
            ]
            for level in flow_levels
        ],
    }


#  Definition index 

def _build_definition_index(repo_root: Path) -> dict[str, dict]:
    """
    Build a mapping of symbol_key → {key, name, class, file, line, node}
    for all functions and methods in the repo.

    symbol_key format:
      top-level function: "send"
      method:             "Session.request"
    """
    index = {}
    SKIP_DIRS = {".git", "__pycache__", "node_modules", ".tox", "dist", "build"}

    for file_path in sorted(repo_root.rglob("*.py")):
        if any(p in SKIP_DIRS or p.startswith(".") for p in file_path.parts):
            continue
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(content)
        except Exception:
            continue

        rel = str(file_path.relative_to(repo_root))

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                key = node.name
                index[key] = {
                    "key": key, "name": node.name, "class": None,
                    "file": rel, "line": node.lineno, "node": node,
                    "content": content,
                }

            elif isinstance(node, ast.ClassDef):
                class_name = node.name
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                        key = f"{class_name}.{child.name}"
                        index[key] = {
                            "key": key, "name": child.name, "class": class_name,
                            "file": rel, "line": child.lineno, "node": child,
                            "content": content,
                        }

    return index


def _find_definition(symbol: str, index: dict) -> dict | None:
    """Find a symbol definition — exact match first, then partial."""
    # Exact match
    if symbol in index:
        return index[symbol]
    # Case-insensitive
    for key, val in index.items():
        if key.lower() == symbol.lower():
            return val
    # Partial: "request" matches "Session.request"
    leaf = symbol.split(".")[-1]
    for key, val in index.items():
        if val["name"] == leaf:
            return val
    return None


def _get_calls_from(sym_info: dict, index: dict, repo_root: Path) -> list[dict]:
    """
    Extract all function calls made within a function/method body
    and resolve them against the definition index.
    """
    node = sym_info.get("node")
    if node is None:
        return []

    called_names = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            func = child.func
            if isinstance(func, ast.Name):
                called_names.add(func.id)
            elif isinstance(func, ast.Attribute):
                called_names.add(func.attr)
                # Also try class.method form
                if isinstance(func.value, ast.Name):
                    called_names.add(f"{func.value.id}.{func.attr}")

    # Resolve against index (only calls we can find definitions for)
    resolved = []
    seen_keys = set()
    for name in called_names:
        match = None
        if name in index:
            match = index[name]
        else:
            # Try partial
            for key, val in index.items():
                if val["name"] == name and key not in seen_keys:
                    match = val
                    break
        if match and match["key"] not in seen_keys:
            seen_keys.add(match["key"])
            resolved.append(match)

    return resolved


def _build_diagram(start_symbol: str, flow_levels: list) -> str:
    """Build a human-readable call flow diagram."""
    lines = [
        f"Call Flow: {start_symbol}",
        "=" * 60,
    ]

    for level_idx, level in enumerate(flow_levels):
        indent = "  " * level_idx
        connector = "→" if level_idx > 0 else "►"

        for sym in level:
            loc = f"{sym['file']}:{sym['line']}"
            calls_to = sym.get("calls_to", [])
            lines.append(f"{indent}{connector} {sym['key']}  [{loc}]")
            if calls_to and level_idx < len(flow_levels) - 1:
                for c in calls_to[:5]:   # show max 5 outbound calls
                    lines.append(f"{indent}    └ calls: {c}")

        if level_idx < len(flow_levels) - 1:
            lines.append(f"{'  ' * (level_idx + 1)}↓")

    if len(flow_levels) == 1 and not flow_levels[0][0].get("calls_to"):
        lines.append("\n  (No further internal calls found at this depth)")

    return "\n".join(lines)


def _error(msg: str, symbol: str) -> dict:
    return {
        "tool":    "trace_call_flow",
        "symbol":  symbol,
        "error":   msg,
        "diagram": "",
        "flow":    [],
    }