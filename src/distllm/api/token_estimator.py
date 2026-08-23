"""Shared token estimation utilities for the API layer.

Consolidates the duplicate ``_estimate_tokens()`` logic that existed in
both ``cost_middleware.py`` and ``quota_middleware.py``.
"""

from __future__ import annotations

from loguru import logger

# Try to import tiktoken for accurate token counting (~95% for English)
_tiktoken_encoding = None
try:
    import tiktoken
    _tiktoken_encoding = tiktoken.get_encoding("cl100k_base")
except (ImportError, Exception):
    pass


def estimate_tokens(text: str) -> int:
    """Estimate token count using tiktoken if available, else heuristic.

    tiktoken is ~95% accurate for English text. The ``len // 4`` heuristic
    is ~60-70% accurate for non-English, code, and JSON.

    Args:
        text: Input text to estimate.

    Returns:
        Estimated token count (always >= 0).
    """
    if not text:
        return 0

    if _tiktoken_encoding is not None:
        try:
            return len(_tiktoken_encoding.encode(text, disallowed_special=()))
        except Exception as e:
            logger.debug(f"tiktoken encode failed, using heuristic: {e}")

    # Fallback: ~4 chars per token (conservative for English)
    return max(1, len(text) // 4)


def estimate_tokens_from_messages(messages: list[dict]) -> int:
    """Estimate total tokens from an OpenAI-format messages list.

    Sums token estimates for each message's content field.
    """
    total = 0
    for msg in messages:
        content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
        if isinstance(content, str):
            total += estimate_tokens(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text", "")
                    if text:
                        total += estimate_tokens(text)
                elif hasattr(item, "type") and hasattr(item, "text"):
                    if item.type == "text" and item.text:
                        total += estimate_tokens(item.text)
    return total


def estimate_tokens_from_response(raw_body: str | bytes) -> int:
    """Estimate output tokens from a JSON response body.

    Parses the response and extracts the generated text from
    OpenAI-compatible ``choices[].text`` or ``choices[].message.content``.
    """
    if not raw_body:
        return 0
    try:
        import json
        data = json.loads(raw_body) if isinstance(raw_body, (str, bytes)) else raw_body
        if not isinstance(data, dict):
            return 0
        choices = data.get("choices", [])
        if not choices:
            return 0
        text = choices[0].get("text", "") or choices[0].get("message", {}).get("content", "")
        return estimate_tokens(str(text))
    except (json.JSONDecodeError, TypeError, AttributeError):
        return 0
