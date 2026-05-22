"""
graph.py
 
LangGraph agent graph definition.

Graph topology:
                    ┌─────────────────┐
                    │  repo_context   │  (builds once, cached in state)
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │     intent      │  (classifies query, checks scope)
                    └────────┬────────┘
                             │
               ┌─────────────┴──────────────┐
          out_of_scope                  in_scope
               │                            │
               ▼                   ┌────────▼────────┐
           [answer]                │      tool        │  ◄──────┐
                                   └────────┬─────────┘         │
                                            │                    │
                                   ┌────────▼────────┐    need_more=True
                                   │     critic       │          │
                                   └────────┬─────────┘    ──────┘
                                            │
                          ┌─────────────────┴─────────────────┐
                     sufficient                          insufficient
                          │                                    │
                          ▼                                    ▼
                       [answer]                            [answer]
                                                    (insufficient response)
"""

from langgraph.graph import StateGraph, END
from core.agent.state import AgentState
from core.agent.nodes import (
    repo_context_node,
    intent_node,
    tool_node,
    critic_node,
    answer_node,
)
from core.config import MAX_TOOL_CALLS_PER_QUERY


# Routing functions 

def route_after_intent(state: AgentState) -> str:
    """After intent classification: go to tools or answer directly."""
    if not state.get("is_in_scope", True):
        return "answer"      # Out of scope → skip tools, go to answer
    return "tool"


def route_after_critic(state: AgentState) -> str:
    """After critic: generate answer or loop back for more tools."""
    sufficient    = state.get("evidence_sufficient", True)
    calls_made    = state.get("tool_calls_made", 0)

    if sufficient:
        return "answer"

    # If evidence isn't sufficient but we have budget → get more
    if calls_made < MAX_TOOL_CALLS_PER_QUERY:
        return "tool"

    # Out of budget → answer anyway (will be marked low-confidence)
    return "answer"


def route_after_tool(state: AgentState) -> str:
    """After tool execution: always go to critic."""
    return "critic"


# Graph builder  

def build_graph() -> StateGraph:
    """
    Build and compile the LangGraph agent graph.

    Returns:
        Compiled LangGraph graph ready for .invoke() or .stream()
    """
    graph = StateGraph(AgentState)

    # Register nodes  
    graph.add_node("repo_context", repo_context_node)
    graph.add_node("intent",       intent_node)
    graph.add_node("tool",         tool_node)
    graph.add_node("critic",       critic_node)
    graph.add_node("answer",       answer_node)

    # Entry point  
    graph.set_entry_point("repo_context")

    # Edges  
    graph.add_edge("repo_context", "intent")

    graph.add_conditional_edges(
        "intent",
        route_after_intent,
        {
            "tool":   "tool",
            "answer": "answer",
        }
    )

    graph.add_conditional_edges(
        "tool",
        route_after_tool,
        {
            "critic": "critic",
        }
    )

    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "answer": "answer",
            "tool":   "tool",
        }
    )

    graph.add_edge("answer", END)

    return graph.compile()


# Module-level compiled graph singleton  
_graph = None

def get_graph():
    """Return the compiled graph singleton."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


# ─── Main query function 

def run_query(
    query: str,
    conversation_history: list[dict] | None = None,
    repo_context: str = "",
) -> dict:
    """
    Run a user query through the agent graph.
    Returns the final state dict after all nodes complete.
    For live trace output use run_query_stream() instead.
    """
    graph = get_graph()
    initial_state = _build_initial_state(query, conversation_history, repo_context)

    try:
        final_state = graph.invoke(initial_state)
        return final_state
    except Exception as e:
        return {
            **initial_state,
            "final_answer": (
                f"## Answer\nAn unexpected error occurred.\n\n"
                f"## Error\n{e}\n\n"
                f"## Suggestion\nPlease try rephrasing your question."
            ),
            "error": str(e),
        }


def run_query_stream(
    query: str,
    conversation_history: list[dict] | None = None,
    repo_context: str = "",
    on_trace: callable = None,
):
    """
    Run a query with live streaming — calls on_trace(step) after each
    node completes so the CLI can print reasoning steps in real time.

    Args:
        query:                The user's question.
        conversation_history: Prior turns [{role, content}].
        repo_context:         Cached repo context.
        on_trace:             Callback(step: str) called for each new
                              trace line as it arrives. Use to print live.

    Returns:
        Final AgentState dict (same as run_query).
    """
    graph = get_graph()
    initial_state = _build_initial_state(query, conversation_history, repo_context)
    final_state = initial_state.copy()
    seen_trace = set()

    try:
        # stream_mode="updates" yields {node_name: state_update} after each node
        for chunk in graph.stream(initial_state, stream_mode="updates"):
            for node_name, update in chunk.items():
                # LangGraph yields None when a node returns {} (empty dict)
                if update is None:
                    continue

                # Extract new trace lines from this node's update
                new_traces = update.get("reasoning_trace", [])
                for step in new_traces:
                    if step not in seen_trace:
                        seen_trace.add(step)
                        if on_trace:
                            on_trace(node_name, step)

                # Merge update into final_state
                for key, value in update.items():
                    if value is None:
                        # Never overwrite existing state with None — skip
                        continue
                    if isinstance(value, list) and isinstance(final_state.get(key), list):
                        # LangGraph Annotated lists — extend, don't overwrite
                        final_state[key] = final_state.get(key, []) + value
                    else:
                        final_state[key] = value

        return final_state

    except Exception as e:
        return {
            **initial_state,
            "final_answer": (
                f"## Answer\nAn unexpected error occurred.\n\n"
                f"## Error\n{e}\n\n"
                f"## Suggestion\nPlease try rephrasing your question."
            ),
            "error": str(e),
        }


def _build_initial_state(
    query: str,
    conversation_history: list[dict] | None,
    repo_context: str,
) -> dict:
    """Build the initial AgentState dict."""
    return {
        "query":                query,
        "conversation_history": conversation_history or [],
        "intent":               "",
        "is_in_scope":          True,
        "scope_reason":         "",
        "repo_context":         repo_context,
        "tools_to_use":         [],
        "tool_results":         [],
        "tool_calls_made":      0,
        "reasoning_trace":      [],
        "evidence_sufficient":  True,
        "critic_notes":         "",
        "final_answer":         "",
        "sources":              [],
        "error":                "",
    }