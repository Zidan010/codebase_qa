"""
state.py

LangGraph agent state schema.

The state is a TypedDict that flows through every node in the graph.
Each node reads from it and writes back updated values.
LangGraph merges list fields automatically using Annotated + operator.add.
"""

from typing import Any, Annotated
from typing_extensions import TypedDict
import operator


class AgentState(TypedDict):
    #  Input 
    query:            str           # original user question
    conversation_history: list[dict]  # prior turns [{role, content}]

    #  Intent classification 
    intent:           str           # classified intent label
    is_in_scope:      bool          # True if query is about the repo
    scope_reason:     str           # why it's in/out of scope

    #  Repo context 
    repo_context:     str           # high-level repo summary (built once)

    #  Tool execution 
    tools_to_use:     list[str]     # tools selected by reasoner
    # Accumulated across multiple tool calls (LangGraph merges lists)
    tool_results:     Annotated[list[dict], operator.add]
    tool_calls_made:  int           # counter to prevent infinite loops
    reasoning_trace:  Annotated[list[str], operator.add]  # step-by-step trace

    #  Critic 
    evidence_sufficient: bool       # critic verdict
    critic_notes:     str           # critic reasoning

    #  Final output 
    final_answer:     str           # structured final answer
    sources:          list[dict]    # citations [{file, lines, symbol}]
    error:            str           # error message if something went wrong