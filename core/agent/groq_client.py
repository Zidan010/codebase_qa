"""
groq_client.py

Groq API client wrapper with:
- Automatic retry on rate limit (429) with user-visible wait message
- Structured JSON response parsing
- Conversation history support
- Consistent error handling
"""

import json
import time
import re
from groq import Groq, RateLimitError, APIError
from rich.console import Console
from core.config import (
    GROQ_API_KEY, GROQ_MODEL, GROQ_MAX_TOKENS,
    GROQ_TEMPERATURE, GROQ_MAX_RETRIES, GROQ_RETRY_WAIT_SECONDS,
)

console = Console()
_client: Groq | None = None

# Max history messages to pass per call (3 exchanges = 6 messages)
HISTORY_WINDOW = 6


def get_client() -> Groq:
    """Return the global Groq client singleton."""
    global _client
    if _client is None:
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


def prepare_history(history: list[dict] | None) -> list[dict]:
    """
    Sanitize and window conversation history before passing to Groq API.

    - Takes last HISTORY_WINDOW messages (3 exchanges = 6 messages)
    - Ensures roles are valid: "user" or "assistant" only
    - Ensures content is a non-empty string
    - Truncates individual messages to 800 chars to control token usage
      (full answers can be long — we only need enough for context)
    - Ensures history alternates user/assistant correctly
      (Groq rejects consecutive same-role messages)

    Args:
        history: Raw conversation history list [{role, content}]

    Returns:
        Cleaned list ready for Groq messages array.
    """
    if not history:
        return []

    # Take last HISTORY_WINDOW messages
    windowed = history[-HISTORY_WINDOW:]

    cleaned = []
    for turn in windowed:
        role    = turn.get("role", "")
        content = turn.get("content", "")

        # Only valid Groq roles
        if role not in ("user", "assistant"):
            continue

        # Must have content
        if not isinstance(content, str) or not content.strip():
            continue

        # Truncate to avoid token bloat — enough for context, not full answer
        content = content[:800].strip()
        cleaned.append({"role": role, "content": content})

    # Groq rejects consecutive same-role messages
    # Deduplicate by removing consecutive duplicates
    deduped = []
    for turn in cleaned:
        if deduped and deduped[-1]["role"] == turn["role"]:
            # Replace previous with latest same-role message
            deduped[-1] = turn
        else:
            deduped.append(turn)

    return deduped


def call_llm(
    prompt: str,
    system: str = "",
    history: list[dict] | None = None,
    temperature: float = GROQ_TEMPERATURE,
    max_tokens: int = GROQ_MAX_TOKENS,
) -> str:
    """
    Call Groq LLM with retry on rate limit.

    Args:
        prompt:      The current user message / prompt text.
        system:      System prompt (prepended as system role message).
        history:     Prior conversation turns [{role, content}].
                     Passed natively as message array to Groq —
                     LLM understands conversation context properly.
        temperature: Sampling temperature.
        max_tokens:  Max tokens in response.

    Returns:
        Raw string response from the LLM.
        On unrecoverable error, returns an error string starting with "ERROR:".
    """
    # Build messages: [history...] + [current user message]
    prior = prepare_history(history)
    messages = prior + [{"role": "user", "content": prompt}]

    for attempt in range(1, GROQ_MAX_RETRIES + 1):
        try:
            client = get_client()
            # System message goes first if provided
            final_messages = (
                [{"role": "system", "content": system}] + messages
                if system else messages
            )

            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=final_messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or ""

        except RateLimitError:
            if attempt < GROQ_MAX_RETRIES:
                console.print(
                    f"\n[yellow]⏳ Groq rate limit reached. "
                    f"Waiting {GROQ_RETRY_WAIT_SECONDS}s before retrying "
                    f"(attempt {attempt}/{GROQ_MAX_RETRIES})...[/]"
                )
                time.sleep(GROQ_RETRY_WAIT_SECONDS)
            else:
                console.print(
                    "[bold red]Rate limit exceeded after all retries. "
                    "Please wait a minute and try again.[/]"
                )
                return "ERROR: Rate limit exceeded. Please wait and retry."

        except APIError as e:
            console.print(f"[bold red]Groq API error:[/] {e}")
            return f"ERROR: Groq API error: {e}"

        except Exception as e:
            console.print(f"[bold red]Unexpected LLM error:[/] {e}")
            return f"ERROR: {e}"

    return "ERROR: All retry attempts failed."


def call_llm_json(
    prompt: str,
    system: str = "",
    history: list[dict] | None = None,
) -> dict:
    """
    Call LLM and parse the response as JSON.
    Strips markdown fences if present.

    Returns:
        Parsed dict, or {"error": "..."} on failure.
    """
    raw = call_llm(prompt, system=system, history=history, temperature=0.0)

    if raw.startswith("ERROR:"):
        return {"error": raw}

    # Strip markdown code fences: ```json ... ``` or ``` ... ```
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip()
    cleaned = cleaned.replace("```", "").strip()

    # Find the first { ... } block (handles extra text before/after JSON)
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return {"error": f"No JSON found in response: {cleaned[:200]}"}

    try:
        return json.loads(match.group())
    except json.JSONDecodeError as e:
        # Last resort: try the whole cleaned string
        try:
            return json.loads(cleaned)
        except Exception:
            return {"error": f"JSON parse failed: {e}. Raw: {cleaned[:200]}"}