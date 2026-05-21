"""
cli.py

Conversational CLI interface for codebase_qa.

Features:
  - Multi-turn conversation with history (follow-up questions work)
  - Repo context cached across turns (no repeated LLM calls)
  - Toggle reasoning trace and tools display on the fly
  - Graceful handling of all edge cases (empty input, Ctrl+C, Ctrl+D)
  - Commands: /trace /tools /clear /help /quit
  - Never crashes on unexpected input

Usage:
    python cli.py
    python main.py          (preferred — runs setup first)
"""

import sys
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

console = Console()


#  Commands 
COMMANDS = {
    "/quit":  "Exit the program",
    "/exit":  "Exit the program",
    "/q":     "Exit the program",
    "/clear": "Clear conversation history (start fresh)",
    "/trace": "Toggle reasoning trace display on/off",
    "/tools": "Toggle tools-used display on/off",
    "/help":  "Show this help message",
    "/status":"Show current session status",
}


def start_cli() -> None:
    """
    Main CLI entry point. Starts the conversational loop.
    Called from main.py after setup completes.
    """
    from core.utils.formatter import print_welcome, print_thinking, render_response, print_error
    from core.agent.graph import run_query

    #  Session state 
    conversation_history: list[dict] = []
    repo_context: str = ""           # cached after first query
    show_trace: bool  = True
    show_tools: bool  = True
    query_count: int  = 0

    print_welcome()
    _print_help()

    #  Main loop 
    while True:
        try:
            # Prompt
            turn_label = f"[dim]({query_count + 1})[/] " if query_count > 0 else ""
            raw = Prompt.ask(f"\n{turn_label}[bold cyan]You[/]")

        except KeyboardInterrupt:
            console.print("\n\n[dim]Use [bold]/quit[/] to exit.[/]")
            continue
        except EOFError:
            # Ctrl+D — graceful exit
            console.print("\n[dim]Goodbye.[/]")
            break

        query = raw.strip()

        #  Skip empty input 
        if not query:
            continue

        #  Handle commands 
        if query.startswith("/"):
            cmd = query.lower().split()[0]
            handled = _handle_command(
                cmd, conversation_history,
                show_trace, show_tools, query_count, repo_context,
            )
            if handled == "quit":
                console.print("\n[dim]Goodbye. Happy coding! 🐍[/]\n")
                break
            elif handled == "clear":
                conversation_history = []
                repo_context = ""
                query_count  = 0
                console.print("[green]✓ Conversation history cleared.[/]")
            elif isinstance(handled, dict):
                show_trace = handled.get("show_trace", show_trace)
                show_tools = handled.get("show_tools", show_tools)
            continue

        #  Run agent 
        print_thinking()

        try:
            result = run_query(
                query=query,
                conversation_history=conversation_history,
                repo_context=repo_context,
            )
        except KeyboardInterrupt:
            console.print("\n[yellow]⚠ Query interrupted.[/]")
            continue
        except Exception as e:
            print_error(f"Unexpected error: {e}")
            continue

        #  Cache repo context for subsequent turns 
        if not repo_context and result.get("repo_context"):
            repo_context = result["repo_context"]

        #  Render response 
        render_response(result, show_trace=show_trace, show_tools=show_tools)

        #  Update conversation history 
        # Only update history for in-scope questions
        if result.get("is_in_scope", True):
            conversation_history.append({
                "role":    "user",
                "content": query,
            })
            answer = result.get("final_answer", "")
            if answer:
                # Store a compact version (first 500 chars) to avoid huge context
                conversation_history.append({
                    "role":    "assistant",
                    "content": answer[:500],
                })

            # Keep last 10 turns (5 exchanges) to stay within token limits
            if len(conversation_history) > 10:
                conversation_history = conversation_history[-10:]

        query_count += 1

        #  Follow-up hint 
        if result.get("is_in_scope", True) and query_count == 1:
            console.print(
                "\n[dim]💡 Tip: You can ask follow-up questions! "
                "e.g. 'Show me the code for that' or 'What calls that function?'[/]"
            )


#  Command handlers 

def _handle_command(
    cmd: str,
    history: list,
    show_trace: bool,
    show_tools: bool,
    query_count: int,
    repo_context: str,
) -> str | dict:
    """
    Handle a slash command.

    Returns:
        "quit"   → exit the loop
        "clear"  → clear history
        dict     → updated toggle state
        "ok"     → command handled, continue
    """
    if cmd in ("/quit", "/exit", "/q"):
        return "quit"

    elif cmd == "/clear":
        return "clear"

    elif cmd == "/trace":
        new_val = not show_trace
        state = "ON" if new_val else "OFF"
        console.print(f"[cyan]Reasoning trace: [bold]{state}[/][/]")
        return {"show_trace": new_val, "show_tools": show_tools}

    elif cmd == "/tools":
        new_val = not show_tools
        state = "ON" if new_val else "OFF"
        console.print(f"[cyan]Tools display: [bold]{state}[/][/]")
        return {"show_trace": show_trace, "show_tools": new_val}

    elif cmd == "/help":
        _print_help()
        return "ok"

    elif cmd == "/status":
        _print_status(history, show_trace, show_tools, query_count, repo_context)
        return "ok"

    else:
        console.print(
            f"[yellow]Unknown command: {cmd}[/]  "
            f"[dim]Type [bold]/help[/] for available commands.[/]"
        )
        return "ok"


def _print_help() -> None:
    """Print available commands."""
    lines = ["[bold]Available commands:[/]\n"]
    for cmd, desc in COMMANDS.items():
        lines.append(f"  [bold cyan]{cmd:<10}[/]  [dim]{desc}[/]")
    lines.append(
        "\n[dim]Ask any question about the [bold]psf/requests[/] codebase.\n"
        "Examples:\n"
        "  • How does Session handle HTTP redirects?\n"
        "  • What does HTTPAdapter.send do?\n"
        "  • Show me the folder structure\n"
        "  • What does requests.sessions depend on?\n"
        "  • Trace the call flow of Session.request[/]"
    )
    console.print(Panel(
        "\n".join(lines),
        title="[bold dim]Help[/]",
        border_style="dim",
        padding=(0, 2),
    ))


def _print_status(
    history: list,
    show_trace: bool,
    show_tools: bool,
    query_count: int,
    repo_context: str,
) -> None:
    """Print current session status."""
    ctx_status = (
        f"[green]cached ({len(repo_context)} chars)[/]"
        if repo_context else "[yellow]not built yet[/]"
    )
    console.print(Panel(
        f"[bold]Session Status[/]\n\n"
        f"  Queries asked:      [cyan]{query_count}[/]\n"
        f"  History turns:      [cyan]{len(history)}[/]\n"
        f"  Repo context:       {ctx_status}\n"
        f"  Show trace:         [cyan]{'ON' if show_trace else 'OFF'}[/]\n"
        f"  Show tools:         [cyan]{'ON' if show_tools else 'OFF'}[/]\n"
        f"  Target repo:        [green]psf/requests[/]",
        border_style="cyan",
        padding=(0, 2),
    ))