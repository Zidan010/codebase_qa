"""
nodes.py

LangGraph node functions. Each node:
  - Receives the full AgentState
  - Does one focused job
  - Returns a dict of state updates

Node pipeline:
  repo_context_node → intent_node → tool_node (loop) → critic_node → answer_node

Special paths:
  intent_node → answer_node  (if out of scope)
  critic_node → answer_node  (if evidence insufficient)
  critic_node → tool_node    (if need more evidence, under call limit)
"""

import json
from core.agent.state import AgentState
from core.agent.prompts import (
    SYSTEM_BASE, INTENT_CLASSIFIER_PROMPT, REPO_CONTEXT_PROMPT,
    TOOL_SELECTOR_PROMPT, CRITIC_PROMPT, ANSWER_PROMPT,
    OUT_OF_SCOPE_RESPONSE, INSUFFICIENT_EVIDENCE_RESPONSE,
)
from core.agent.groq_client import call_llm, call_llm_json
from core.vectorstore.chroma_store import get_store
from core.config import MAX_TOOL_CALLS_PER_QUERY, ENABLE_CRITIC

#  Tool registry 
from core.tools.search_code      import search_code
from core.tools.read_file        import read_file
from core.tools.list_directory   import list_directory
from core.tools.summarize_module import summarize_module
from core.tools.find_usages      import find_usages
from core.tools.get_dependencies import get_dependencies
from core.tools.trace_call_flow  import trace_call_flow

TOOL_REGISTRY = {
    "search_code":      search_code,
    "read_file":        read_file,
    "list_directory":   list_directory,
    "summarize_module": summarize_module,
    "find_usages":      find_usages,
    "get_dependencies": get_dependencies,
    "trace_call_flow":  trace_call_flow,
}


# ══════════════════════════════════════════════════════════════════════════════
#  NODE 1: Repo Context Builder
#  Runs once per session. Builds a high-level summary of the repo.
# ══════════════════════════════════════════════════════════════════════════════

def repo_context_node(state: AgentState) -> dict:
    """
    Build a cached high-level summary of the repository.
    If already built (from prior turn), skip.
    """
    # If already built in this session, reuse it
    if state.get("repo_context"):
        return {}

    store = get_store()
    smap = store.get_structure_map()

    if not smap:
        # Fallback: use a generic description
        return {
            "repo_context": (
                "psf/requests is a Python HTTP library. "
                "Key modules: sessions (Session class), adapters (HTTPAdapter), "
                "auth (authentication), models (Request/Response), "
                "exceptions (error hierarchy), utils (helpers)."
            ),
            "reasoning_trace": ["[Repo Context] Used fallback context (structure map not found)"],
        }

    structure_text = smap["text"][:3000]   # cap to avoid huge prompts
    prompt = REPO_CONTEXT_PROMPT.format(structure_map=structure_text)
    context = call_llm(prompt, system=SYSTEM_BASE)

    if context.startswith("ERROR:"):
        context = (
            "psf/requests is a Python HTTP library with modules for "
            "session management, HTTP adapters, authentication, and utilities."
        )

    return {
        "repo_context": context,
        "reasoning_trace": [f"[Repo Context] Built repo summary ({len(context)} chars)"],
    }


# ══════════════════════════════════════════════════════════════════════════════
#  NODE 2: Intent Classifier
#  Classifies query intent and scope. Routes out-of-scope queries immediately.
# ══════════════════════════════════════════════════════════════════════════════

def intent_node(state: AgentState) -> dict:
    """Classify query intent and determine if it's in scope."""
    query   = state["query"]
    history = state.get("conversation_history", [])

    # Pass history natively to Groq — the model reads prior turns as actual
    # conversation context, so it correctly resolves "that function", "it", etc.
    # The prompt only contains the current query — no flat text history dump.
    prompt = INTENT_CLASSIFIER_PROMPT.format(query=query)

    result = call_llm_json(prompt, system=SYSTEM_BASE, history=history)

    # Handle LLM errors gracefully
    if "error" in result:
        return {
            "intent":       "implementation",
            "is_in_scope":  True,
            "scope_reason": "Could not classify — defaulting to in-scope",
            "tools_to_use": ["search_code"],
            "reasoning_trace": [
                f"[Intent] Classification failed: {result['error']}. Defaulting to in-scope."
            ],
        }

    intent      = result.get("intent", "implementation")
    is_in_scope = result.get("is_in_scope", True)
    scope_reason = result.get("scope_reason", "")
    tools       = result.get("tools_recommended", ["search_code"])
    reasoning   = result.get("reasoning", "")

    trace_msg = (
        f"[Intent] '{intent}' | in_scope={is_in_scope} | "
        f"tools={tools} | reason: {reasoning}"
    )

    return {
        "intent":       intent,
        "is_in_scope":  is_in_scope,
        "scope_reason": scope_reason,
        "tools_to_use": tools,
        "reasoning_trace": [trace_msg],
    }


# ══════════════════════════════════════════════════════════════════════════════
#  NODE 3: Tool Executor
#  Asks the LLM which tools to call, executes them, accumulates results.
# ══════════════════════════════════════════════════════════════════════════════

def tool_node(state: AgentState) -> dict:
    """Select and execute tools. May be called multiple times in a loop."""
    query          = state["query"]
    intent         = state.get("intent", "implementation")
    repo_context   = state.get("repo_context", "")
    tool_results   = state.get("tool_results", [])
    calls_made     = state.get("tool_calls_made", 0)
    history        = state.get("conversation_history", [])

    # Safety: enforce max tool calls
    if calls_made >= MAX_TOOL_CALLS_PER_QUERY:
        return {
            "tool_results": [],
            "reasoning_trace": [
                f"[Tools] Max tool calls ({MAX_TOOL_CALLS_PER_QUERY}) reached. Stopping."
            ],
        }

    # Summarise prior results for the selector prompt
    results_summary = _summarize_tool_results(tool_results)

    # History is passed natively to Groq — no need to embed as text.
    # The model sees prior turns as actual conversation messages,
    # so "What calls that function?" correctly resolves "that function"
    # from the prior assistant answer.
    prompt = TOOL_SELECTOR_PROMPT.format(
        query=query,
        intent=intent,
        repo_context=repo_context[:500],
        tool_results_so_far=results_summary,
    )

    selection = call_llm_json(prompt, system=SYSTEM_BASE, history=history)

    if "error" in selection:
        # Fallback: do a basic semantic search
        fallback_result = search_code(query, top_k=5)
        return {
            "tool_results":  [fallback_result],
            "tool_calls_made": calls_made + 1,
            "reasoning_trace": [
                f"[Tools] Selector failed ({selection['error']}). "
                f"Fallback: search_code returned {fallback_result.get('total', 0)} results."
            ],
        }

    should_continue = selection.get("should_continue", True)
    tool_calls      = selection.get("tool_calls", [])
    reasoning       = selection.get("reasoning", "")

    if not should_continue or not tool_calls:
        return {
            "tool_results": [],
            "reasoning_trace": [
                f"[Tools] Selector decided no more tools needed. Reason: {reasoning}"
            ],
        }

    # Execute each selected tool
    new_results = []
    trace_msgs  = [f"[Tools] Reasoning: {reasoning}"]

    for call in tool_calls[:3]:   # max 3 tools per turn
        tool_name = call.get("tool", "")
        tool_args = call.get("args", {})

        if tool_name not in TOOL_REGISTRY:
            trace_msgs.append(f"[Tools] Unknown tool '{tool_name}' — skipped")
            continue

        try:
            tool_fn = TOOL_REGISTRY[tool_name]
            result  = tool_fn(**tool_args)
            new_results.append(result)
            trace_msgs.append(
                f"[Tools] Called {tool_name}({_format_args(tool_args)}) "
                f"→ {_summarize_result(result)}"
            )
        except TypeError as e:
            # Wrong args — don't crash, report and continue
            trace_msgs.append(f"[Tools] {tool_name} called with bad args: {e}")
        except Exception as e:
            trace_msgs.append(f"[Tools] {tool_name} error: {e}")

    return {
        "tool_results":    new_results,
        "tool_calls_made": calls_made + len(new_results),
        "reasoning_trace": trace_msgs,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  NODE 4: Retrieval Critic
#  Validates whether evidence is sufficient before generating an answer.
# ══════════════════════════════════════════════════════════════════════════════

def critic_node(state: AgentState) -> dict:
    """Validate retrieved evidence quality."""
    if not ENABLE_CRITIC:
        return {
            "evidence_sufficient": True,
            "critic_notes": "Critic disabled via config.",
            "reasoning_trace": ["[Critic] Skipped (ENABLE_CRITIC=False)"],
        }

    query        = state["query"]
    tool_results = state.get("tool_results", [])

    if not tool_results:
        return {
            "evidence_sufficient": False,
            "critic_notes": "No tool results available.",
            "reasoning_trace": ["[Critic] No evidence found — marking insufficient."],
        }

    evidence_text = _format_evidence_for_critic(tool_results)

    prompt = CRITIC_PROMPT.format(
        query=query,
        evidence=evidence_text[:4000],   # cap to avoid token overflow
    )

    result = call_llm_json(prompt, system=SYSTEM_BASE)

    if "error" in result:
        # If critic itself fails, be conservative and allow answering
        return {
            "evidence_sufficient": True,
            "critic_notes": f"Critic LLM call failed: {result['error']}",
            "reasoning_trace": [
                f"[Critic] Failed to run critic ({result['error']}). Allowing answer."
            ],
        }

    sufficient    = result.get("evidence_sufficient", True)
    confidence    = result.get("confidence", "medium")
    notes         = result.get("notes", "")
    can_partial   = result.get("can_answer_partially", False)

    # Allow partial answers through
    if can_partial and not sufficient:
        sufficient = True

    return {
        "evidence_sufficient": sufficient,
        "critic_notes": f"[{confidence.upper()}] {notes}",
        "reasoning_trace": [
            f"[Critic] sufficient={sufficient} | confidence={confidence} | {notes}"
        ],
    }


# ══════════════════════════════════════════════════════════════════════════════
#  NODE 5: Answer Generator
#  Generates the final structured answer from evidence.
# ══════════════════════════════════════════════════════════════════════════════

def answer_node(state: AgentState) -> dict:
    """Generate the final structured answer."""
    query          = state["query"]
    is_in_scope    = state.get("is_in_scope", True)
    scope_reason   = state.get("scope_reason", "")
    sufficient     = state.get("evidence_sufficient", True)
    critic_notes   = state.get("critic_notes", "")
    tool_results   = state.get("tool_results", [])
    repo_context   = state.get("repo_context", "")
    reasoning_trace = state.get("reasoning_trace", [])
    history        = state.get("conversation_history", [])

    #  Out of scope 
    if not is_in_scope:
        answer = OUT_OF_SCOPE_RESPONSE.format(reason=scope_reason)
        return {
            "final_answer":   answer,
            "sources":        [],
            "reasoning_trace": ["[Answer] Out-of-scope response generated."],
        }

    #  Insufficient evidence 
    if not sufficient and not tool_results:
        answer = INSUFFICIENT_EVIDENCE_RESPONSE.format(critic_notes=critic_notes)
        return {
            "final_answer":   answer,
            "sources":        [],
            "reasoning_trace": ["[Answer] Insufficient evidence response generated."],
        }

    #  Build evidence text 
    evidence_text = _format_evidence_for_answer(tool_results)
    trace_text    = "\n".join(reasoning_trace[-10:])  # last 10 trace steps

    # History passed natively to Groq — model understands conversation
    # context directly. Prompt only contains current query + evidence.
    prompt = ANSWER_PROMPT.format(
        system=SYSTEM_BASE,
        query=query,
        repo_context=repo_context[:600],
        evidence=evidence_text[:5000],
        trace=trace_text,
    )

    answer = call_llm(prompt, system=SYSTEM_BASE, history=history)

    if answer.startswith("ERROR:"):
        answer = (
            "## Answer\nI encountered an error generating the response.\n\n"
            f"## Error\n{answer}\n\n"
            "## Raw Evidence\nPlease check the tool results in the reasoning trace."
        )

    # Extract sources from tool results
    sources = _extract_sources(tool_results)

    return {
        "final_answer":    answer,
        "sources":         sources,
        "reasoning_trace": ["[Answer] Final answer generated."],
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _summarize_tool_results(results: list[dict]) -> str:
    """Compact summary of tool results for the selector prompt."""
    if not results:
        return "No tools called yet."
    lines = []
    for r in results[-3:]:   # last 3 results only
        tool = r.get("tool", "unknown")
        if "error" in r:
            lines.append(f"- {tool}: ERROR — {r['error']}")
        elif tool == "search_code":
            lines.append(f"- {tool}: {r.get('total', 0)} results for '{r.get('query', '')}'")
        elif tool == "read_file":
            lines.append(f"- {tool}: read {r.get('path', '')} ({r.get('total_lines', 0)} lines)")
        elif tool == "list_directory":
            lines.append(f"- {tool}: {r.get('total', 0)} files in '{r.get('path', '')}'")
        elif tool == "summarize_module":
            lines.append(f"- {tool}: summarized '{r.get('module', '')}'")
        elif tool == "find_usages":
            lines.append(f"- {tool}: {r.get('total', 0)} usages of '{r.get('symbol', '')}'")
        elif tool == "get_dependencies":
            lines.append(f"- {tool}: deps of '{r.get('module', '')}' — {r.get('totals', {})}")
        elif tool == "trace_call_flow":
            lines.append(f"- {tool}: traced '{r.get('symbol', '')}' ({len(r.get('flow', []))} levels)")
        else:
            lines.append(f"- {tool}: completed")
    return "\n".join(lines)


def _format_evidence_for_critic(results: list[dict]) -> str:
    """Format tool results as evidence text for the critic."""
    parts = []
    for r in results:
        tool = r.get("tool", "")
        if "error" in r:
            continue
        if tool == "search_code":
            for item in r.get("results", [])[:3]:
                parts.append(
                    f"[{item['file']} lines {item['lines']}]\n{item['snippet']}"
                )
        elif tool == "read_file":
            content = r.get("content", "")[:1000]
            parts.append(f"[{r.get('path')} lines {r.get('start_line')}-{r.get('end_line')}]\n{content}")
        elif tool == "summarize_module":
            parts.append(f"[Module: {r.get('module')}]\nPurpose: {r.get('purpose', '')}")
        elif tool == "find_usages":
            usages = r.get("usages", [])[:5]
            parts.append(f"[Usages of {r.get('symbol')}]\n" +
                         "\n".join(f"  {u['file']}:{u['line']} ({u['type']}) — {u['context']}" for u in usages))
        elif tool == "get_dependencies":
            parts.append(f"[Dependencies of {r.get('module')}]\n{r.get('dependency_tree', '')[:500]}")
        elif tool == "trace_call_flow":
            parts.append(f"[Call Flow: {r.get('symbol')}]\n{r.get('diagram', '')[:800]}")
        elif tool == "list_directory":
            parts.append(f"[Directory: {r.get('path')}]\n{r.get('tree', '')[:500]}")
    return "\n\n---\n\n".join(parts) if parts else "No evidence retrieved."


def _format_evidence_for_answer(results: list[dict]) -> str:
    """Format all tool results as rich evidence for the answer prompt."""
    parts = []
    for r in results:
        tool = r.get("tool", "")
        if "error" in r:
            parts.append(f"[{tool} ERROR]: {r['error']}")
            continue

        if tool == "search_code":
            for item in r.get("results", [])[:5]:
                parts.append(
                    f"FILE: {item['file']} | Lines: {item['lines']} | "
                    f"Symbol: {item['symbol']} | Type: {item['chunk_type']}\n"
                    f"Docstring: {item['docstring']}\n"
                    f"Code snippet:\n{item['snippet']}"
                )
        elif tool == "read_file":
            parts.append(
                f"FILE CONTENT: {r.get('path')} "
                f"(lines {r.get('start_line')}–{r.get('end_line')}):\n"
                f"{r.get('content', '')[:2000]}"
            )
        elif tool == "summarize_module":
            m = r
            classes_info = "\n".join(
                f"  class {c['name']}({', '.join(c['bases'])}): {c['docstring']}"
                for c in m.get("classes", [])
            )
            funcs_info = "\n".join(
                f"  def {f['name']}({', '.join(f['args'])}): {f['docstring']}"
                for f in m.get("functions", [])
            )
            parts.append(
                f"MODULE SUMMARY: {m.get('module')} [{m.get('file')}]\n"
                f"Purpose: {m.get('purpose', '')}\n"
                f"Public API: {m.get('public_api', [])}\n"
                f"Classes:\n{classes_info}\n"
                f"Functions:\n{funcs_info}\n"
                f"Dependencies: {m.get('all_imports', [])[:8]}"
            )
        elif tool == "find_usages":
            usages = r.get("usages", [])
            parts.append(
                f"USAGES OF '{r.get('symbol')}' ({r.get('total')} found):\n"
                + "\n".join(
                    f"  {u['file']}:{u['line']} [{u['type']}] — {u['context']}"
                    for u in usages[:10]
                )
            )
        elif tool == "get_dependencies":
            parts.append(
                f"DEPENDENCIES OF '{r.get('module')}':\n{r.get('dependency_tree', '')}"
            )
        elif tool == "trace_call_flow":
            parts.append(
                f"CALL FLOW FROM '{r.get('symbol')}':\n{r.get('diagram', '')}"
            )
        elif tool == "list_directory":
            parts.append(
                f"DIRECTORY LISTING: {r.get('path')}\n{r.get('tree', '')}"
            )

    return "\n\n" + ("=" * 50) + "\n\n".join(parts) if parts else "No evidence available."


def _extract_sources(results: list[dict]) -> list[dict]:
    """Extract citation sources from tool results."""
    sources = []
    seen = set()

    for r in results:
        tool = r.get("tool", "")
        if tool == "search_code":
            for item in r.get("results", [])[:5]:
                key = (item["file"], item["lines"])
                if key not in seen:
                    seen.add(key)
                    sources.append({
                        "file":   item["file"],
                        "lines":  item["lines"],
                        "symbol": item["symbol"],
                        "type":   item["chunk_type"],
                    })
        elif tool == "read_file":
            key = (r.get("path"), f"{r.get('start_line')}-{r.get('end_line')}")
            if key not in seen:
                seen.add(key)
                sources.append({
                    "file":   r.get("path", ""),
                    "lines":  f"{r.get('start_line')}–{r.get('end_line')}",
                    "symbol": "",
                    "type":   "file_read",
                })
        elif tool == "summarize_module":
            key = r.get("file", "")
            if key not in seen:
                seen.add(key)
                sources.append({
                    "file":   r.get("file", ""),
                    "lines":  "full module",
                    "symbol": r.get("module", ""),
                    "type":   "module_summary",
                })

    return sources


def _format_args(args: dict) -> str:
    """Format tool args for display in trace."""
    return ", ".join(f"{k}={repr(v)[:30]}" for k, v in args.items())


def _summarize_result(result: dict) -> str:
    """One-line summary of a tool result for trace display."""
    tool = result.get("tool", "")
    if "error" in result:
        return f"ERROR: {result['error'][:60]}"
    if tool == "search_code":
        return f"{result.get('total', 0)} results"
    if tool == "read_file":
        return f"{result.get('total_lines', 0)} lines"
    if tool == "list_directory":
        return f"{result.get('total', 0)} files"
    if tool == "find_usages":
        return f"{result.get('total', 0)} usages"
    if tool == "summarize_module":
        return f"module '{result.get('module', '')}'"
    if tool == "get_dependencies":
        t = result.get("totals", {})
        return f"internal={t.get('internal',0)} stdlib={t.get('stdlib',0)} third_party={t.get('third_party',0)}"
    if tool == "trace_call_flow":
        return f"{len(result.get('flow', []))} levels traced"
    return "ok"