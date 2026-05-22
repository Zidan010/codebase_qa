"""
formatter.py

Converts raw AgentState output into beautiful, structured Rich terminal output.

Responsibilities:
  - Render the final answer with proper markdown-to-Rich conversion
  - Display the reasoning trace as a collapsible panel
  - Show tools used with icons
  - Display source citations as a formatted table
  - Render out-of-scope and error states cleanly
  - Provide plain-text output mode for piping/logging

Output structure:
  ┌─ Answer ──────────────────────────────────────┐
  │  ## Answer                                    │
  │  ## How It Works                              │
  │  ## Relevant Code                             │
  └───────────────────────────────────────────────┘
  ┌─ Sources ─────────────────────────────────────┐
  │  • file.py  lines 10-20  Session.get          │
  └───────────────────────────────────────────────┘
  ┌─ Reasoning Trace ─────────────────────────────┐
  │  [Intent] implementation | tools=[...]        │
  │  [Tools]  Called search_code → 5 results      │
  │  [Critic] sufficient=True | HIGH              │
  └───────────────────────────────────────────────┘
"""

import re
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
from rich.markdown import Markdown
from rich.text import Text
from rich.rule import Rule
from rich import box

console = Console()

# Intent → emoji icon 
INTENT_ICONS = {
    "architecture":   "🏗️",
    "api_usage":      "📖",
    "implementation": "⚙️",
    "debugging":      "🐛",
    "dependency":     "🔗",
    "call_flow":      "🔄",
    "file_structure": "📁",
    "out_of_scope":   "🚫",
}

# Tool → emoji icon 
TOOL_ICONS = {
    "search_code":      "🔍",
    "read_file":        "📄",
    "list_directory":   "📂",
    "summarize_module": "📋",
    "find_usages":      "🔎",
    "get_dependencies": "🔗",
    "trace_call_flow":  "🔄",
}


# ══════════════════════════════════════════════════════════════════════════════
#  Main render function
# ══════════════════════════════════════════════════════════════════════════════

def render_response(state: dict, show_trace: bool = True, show_tools: bool = True, skip_trace: bool = False) -> None:
    """
    Render the full agent response to the terminal.

    Args:
        state:      Final AgentState dict from graph.run_query()
        show_trace: Whether to show the reasoning trace panel
        show_tools: Whether to show the tools-used summary
    """
    console.print()

    # Query echo    
    query = state.get("query", "")
    console.print(Rule(f"[bold cyan]Q: {query[:80]}[/]", style="cyan"))
    console.print()

    # Intent badge  
    intent     = state.get("intent", "")
    is_scope   = state.get("is_in_scope", True)
    calls_made = state.get("tool_calls_made", 0)
 
    if intent:
        icon = INTENT_ICONS.get(intent, "💡")
        scope_badge = "[green]IN SCOPE[/]" if is_scope else "[red]OUT OF SCOPE[/]"
        console.print(
            f"  {icon} Intent: [bold]{intent}[/]  {scope_badge}  "
            f"[dim]Tools called: {calls_made}[/]"
        )
        console.print()

    # Tools used  
    if show_tools:
        _render_tools_used(state)

    # Reasoning trace (before answer so user sees the working first)  
    if show_trace and not skip_trace:
        _render_trace(state)

    # Main answer  
    _render_answer(state)

    # Sources  
    _render_sources(state)

    console.print()


# ══════════════════════════════════════════════════════════════════════════════
#  Section renderers
# ══════════════════════════════════════════════════════════════════════════════
 
def _render_tools_used(state: dict) -> None:
    """Show which tools were called and what they returned."""
    tool_results = state.get("tool_results", [])
    if not tool_results:
        return
 
    # Collect unique tool calls
    tool_summary = []
    for r in tool_results:
        tool = r.get("tool", "unknown")
        icon = TOOL_ICONS.get(tool, "🔧")
 
        if "error" in r:
            desc = f"[red]error: {r['error'][:50]}[/]"
        elif tool == "search_code":
            desc = f"[green]{r.get('total', 0)} results[/] for '{r.get('query', '')[:40]}'"
        elif tool == "read_file":
            desc = f"[green]{r.get('total_lines', 0)} lines[/] from [dim]{r.get('path', '')}[/]"
        elif tool == "list_directory":
            desc = f"[green]{r.get('total', 0)} files[/] in [dim]{r.get('path', '')}[/]"
        elif tool == "summarize_module":
            desc = f"module [dim]{r.get('module', '')}[/]"
        elif tool == "find_usages":
            desc = f"[green]{r.get('total', 0)} usages[/] of [dim]{r.get('symbol', '')}[/]"
        elif tool == "get_dependencies":
            t = r.get("totals", {})
            desc = (f"[dim]{r.get('module', '')}[/] → "
                    f"internal={t.get('internal',0)} "
                    f"stdlib={t.get('stdlib',0)} "
                    f"3rd-party={t.get('third_party',0)}")
        elif tool == "trace_call_flow":
            desc = f"traced [dim]{r.get('symbol', '')}[/] ({len(r.get('flow', []))} levels)"
        else:
            desc = "completed"
 
        tool_summary.append(f"  {icon} [bold]{tool}[/]  {desc}")
 
    if tool_summary:
        tools_text = "\n".join(tool_summary)
        console.print(Panel(
            tools_text,
            title="[bold dim]Tools Used[/]",
            border_style="dim",
            padding=(0, 1),
        ))
        console.print()
 
 
def _render_answer(state: dict) -> None:
    """Render the main answer panel with markdown formatting."""
    answer = state.get("final_answer", "")
    if not answer:
        console.print(Panel("[yellow]No answer generated.[/]", border_style="yellow"))
        return
 
    # Convert answer markdown to Rich-compatible output
    # Extract code blocks for syntax highlighting
    rendered = _render_markdown_with_code(answer)
 
    console.print(Panel(
        rendered,
        title="[bold green]Answer[/]",
        border_style="green",
        padding=(1, 2),
    ))
 
 
def _render_sources(state: dict) -> None:
    """Render sources as a clean table."""
    sources = state.get("sources", [])
    if not sources:
        return
 
    table = Table(
        box=box.SIMPLE,
        show_header=True,
        header_style="bold dim",
        padding=(0, 1),
    )
    table.add_column("File", style="cyan", no_wrap=True)
    table.add_column("Lines", style="yellow", justify="right")
    table.add_column("Symbol", style="green")
    table.add_column("Type", style="dim")
 
    seen = set()
    for s in sources:
        key = (s.get("file", ""), s.get("lines", ""))
        if key in seen:
            continue
        seen.add(key)
        table.add_row(
            s.get("file", ""),
            str(s.get("lines", "")),
            s.get("symbol", "—"),
            s.get("type", ""),
        )
 
    console.print(Panel(
        table,
        title="[bold dim]Sources[/]",
        border_style="dim",
        padding=(0, 1),
    ))
    console.print()
 
 
def _render_trace(state: dict) -> None:
    """Render the agent reasoning trace."""
    trace = state.get("reasoning_trace", [])
    if not trace:
        return
 
    # Colour-code trace steps by node type
    trace_lines = []
    for step in trace:
        if step.startswith("[Repo Context]"):
            trace_lines.append(f"[blue]{step}[/]")
        elif step.startswith("[Intent]"):
            trace_lines.append(f"[cyan]{step}[/]")
        elif step.startswith("[Tools]"):
            trace_lines.append(f"[yellow]{step}[/]")
        elif step.startswith("[Critic]"):
            trace_lines.append(f"[magenta]{step}[/]")
        elif step.startswith("[Answer]"):
            trace_lines.append(f"[green]{step}[/]")
        else:
            trace_lines.append(f"[dim]{step}[/]")
 
    trace_text = "\n".join(trace_lines)
    console.print(Panel(
        trace_text,
        title="[bold dim]Reasoning Trace[/]",
        border_style="dim",
        padding=(0, 1),
        expand=False,
    ))
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  Markdown + code block renderer
# ══════════════════════════════════════════════════════════════════════════════
 
def _render_markdown_with_code(text: str) -> Text | str:
    """
    Convert markdown answer text to Rich-renderable content.
    Extracts fenced code blocks and renders them with syntax highlighting.
    Returns a renderable object.
    """
    # Split on code blocks: ```lang\ncode\n```
    # We'll return a list of Rich renderables
    parts = re.split(r"(```[\w]*\n.*?```)", text, flags=re.DOTALL)
 
    if len(parts) == 1:
        # No code blocks — render as plain markdown
        try:
            return Markdown(text)
        except Exception:
            return text
 
    # Build a combined renderable by joining with newlines
    # For simplicity in a Panel, convert to a single string with Rich markup
    result_parts = []
    for part in parts:
        if part.startswith("```"):
            # Extract language and code
            lines = part.split("\n")
            lang = lines[0].replace("```", "").strip() or "python"
            code = "\n".join(lines[1:]).rstrip("`").strip()
            # Render inline as indented block
            result_parts.append(f"\n[bold dim]── code ({lang}) ──[/]")
            result_parts.append(f"[green]{_escape_markup(code)}[/]")
            result_parts.append("[bold dim]────────────────[/]\n")
        else:
            # Convert markdown headers and bold to Rich markup
            converted = _md_to_rich(part)
            result_parts.append(converted)
 
    return "\n".join(result_parts)
 
 
def _md_to_rich(text: str) -> str:
    """Convert basic markdown to Rich markup."""
    lines = text.split("\n")
    out = []
    for line in lines:
        # ## Header → bold cyan
        if line.startswith("## "):
            out.append(f"\n[bold cyan]{line[3:]}[/]")
        elif line.startswith("# "):
            out.append(f"\n[bold cyan underline]{line[2:]}[/]")
        # **bold** → bold
        elif "**" in line:
            line = re.sub(r"\*\*(.+?)\*\*", r"[bold]\1[/]", line)
            out.append(line)
        # `inline code` → dim green
        elif "`" in line:
            line = re.sub(r"`([^`]+)`", r"[green]\1[/]", line)
            out.append(line)
        # Bullet points
        elif line.startswith("- ") or line.startswith("• "):
            out.append(f"  • {line[2:]}")
        else:
            out.append(line)
    return "\n".join(out)
 
 
def _escape_markup(text: str) -> str:
    """Escape Rich markup special characters in code blocks."""
    return text.replace("[", r"\[").replace("]", r"\]")
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  Plain text output (for piping / logging)
# ══════════════════════════════════════════════════════════════════════════════
 
def format_plain_text(state: dict) -> str:
    """
    Return the full response as plain text (no Rich markup).
    Used when output is piped to a file or another process.
    """
    lines = []
    lines.append(f"QUERY: {state.get('query', '')}")
    lines.append(f"INTENT: {state.get('intent', '')} | IN_SCOPE: {state.get('is_in_scope', True)}")
    lines.append("")
    lines.append("=" * 60)
    lines.append("ANSWER:")
    lines.append("=" * 60)
    lines.append(state.get("final_answer", "No answer generated."))
    lines.append("")
 
    sources = state.get("sources", [])
    if sources:
        lines.append("SOURCES:")
        for s in sources:
            lines.append(f"  • {s.get('file', '')} lines {s.get('lines', '')} — {s.get('symbol', '')}")
        lines.append("")
 
    trace = state.get("reasoning_trace", [])
    if trace:
        lines.append("REASONING TRACE:")
        for step in trace:
            lines.append(f"  {step}")
 
    return "\n".join(lines)
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  Welcome / status messages
# ══════════════════════════════════════════════════════════════════════════════
 
def print_welcome() -> None:
    """Print the welcome message when CLI starts."""
    console.print()
    console.print(Panel("",
        title="[bold cyan]CODEBASE Q&A[/]  [dim]•[/]  [green]Agentic Q&A on psf/requests[/]\n\n",
        subtitle="[dim]Type your question and press Enter.\n"
        "Commands:  [bold]/trace[/] toggle trace  [bold]/tools[/] toggle tools  "
        "[bold]/clear[/] clear history  [bold]/quit[/] exit[/]",
        border_style="cyan",
        padding=(1, 3), 
        subtitle_align="center",
        title_align="center",
    ))
    console.print()
 
 
def print_thinking() -> None:
    """Show a 'thinking' indicator while agent runs."""
    console.print("\n[dim cyan]⟳ Agent thinking...[/]\n")
 
 
def print_error(msg: str) -> None:
    """Print a formatted error message."""
    console.print(Panel(
        f"[bold red]Error:[/] {msg}",
        border_style="red",
        padding=(0, 2),
    ))