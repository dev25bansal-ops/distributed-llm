"""Local token counting for the DistLLM SDK.

Uses ``tiktoken`` when available to estimate token counts
without a round-trip to the API.  Gracefully falls back to
a character-based heuristic when tiktoken is not installed.

Usage::

    from distllm_sdk.tokenizer import count_chat_tokens

    tokens = count_chat_tokens([{"role": "user", "content": "Hello!"}])
    print(tokens)  # 10
"""

from __future__ import annotations

from typing import Any

# Lazy import with graceful fallback
_TIKTOKEN_AVAILABLE = False
try:
    import tiktoken

    _TIKTOKEN_AVAILABLE = True
except ImportError:
    tiktoken = None  # type: ignore[assignment]


def _get_encoding(model: str = "distributed-llm"):
    """Get the best available encoding for *model*.

    Falls back to ``cl100k_base`` (the encoding used by GPT-4 / gpt-3.5-turbo)
    for unknown models, then to ``None`` if tiktoken is not installed.
    """
    if not _TIKTOKEN_AVAILABLE:
        return None
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        try:
            return tiktoken.get_encoding("cl100k_base")
        except Exception:
            return None


_CHAR_RATIO = 0.25  # ~4 chars per token for heuristic


def _heuristic_count(text: str) -> int:
    """Fallback token estimate based on character count."""
    return max(1, int(len(text) * _CHAR_RATIO))


def count_tokens(text: str, model: str = "distributed-llm") -> int:
    """Count tokens in *text* using the best available method.

    Uses tiktoken when installed, falls back to character-based heuristic.
    """
    enc = _get_encoding(model)
    if enc is not None:
        return len(enc.encode(text, disallowed_special=()))
    return _heuristic_count(text)


def count_messages_tokens(
    messages: list[dict[str, str]],
    model: str = "distributed-llm",
) -> int:
    """Count tokens in a chat message list, matching OpenAI's chat format.

    Approximates the per-message overhead (``<|start|>role\\ncontent<|end|>``).
    """
    total = 0
    per_message = 3  # <|start|> role \n content
    for msg in messages:
        total += per_message
        total += count_tokens(msg.get("content", ""), model)
        total += count_tokens(msg.get("role", ""), model)
    total += 3  # final <|start|> assistant
    return total


def estimate_cost(
    prompt_tokens: int,
    completion_tokens: int,
    price_per_million_input: float = 0.50,
    price_per_million_output: float = 1.50,
) -> float:
    """Estimate USD cost for token usage.

    Args:
        prompt_tokens: Number of input (prompt) tokens.
        completion_tokens: Number of output (completion) tokens.
        price_per_million_input: Cost per million prompt tokens.
        price_per_million_output: Cost per million completion tokens.

    Returns:
        Estimated cost in USD.
    """
    input_cost = (prompt_tokens / 1_000_000) * price_per_million_input
    output_cost = (completion_tokens / 1_000_000) * price_per_million_output
    return round(input_cost + output_cost, 8)
