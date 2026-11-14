"""Context-window resolution and request-size guard.

A prompt larger than the model's window should be rejected at the API
boundary with a clean 400 ``context_length_exceeded`` error — not passed down
to the backend, where it fails as an opaque 500 (or, on the distributed path,
silently truncates the generation budget).

The window is resolved from the most trustworthy source available, in order:

1. ``coordinator.model_info["max_position_embeddings"]`` — read from the
   model's own config, the same value the generation pipeline uses to cap
   ``max_new_tokens``.
2. ``coordinator.model_info["max_model_len"]`` — vLLM-style spelling.
3. ``settings.vllm.max_model_len`` — explicit operator override.
4. ``settings.llamacpp.n_ctx`` — llama.cpp context size, when that backend
   is what is serving.

If none is available the function returns ``None`` and the guard is skipped
(the request proceeds unchanged, preserving today's behaviour for backends
that do not advertise a window).
"""

from __future__ import annotations

from typing import Any

from .errors import error_response_from_request


def resolve_context_window(coordinator: Any) -> int | None:
    """Return the model's context window in tokens, or ``None`` if unknown."""
    if coordinator is None:
        return None

    info = getattr(coordinator, "model_info", None)
    if isinstance(info, dict):
        for key in ("max_position_embeddings", "max_model_len"):
            value = info.get(key)
            if isinstance(value, int) and value > 0:
                return value

    # Operator overrides from the typed settings.
    try:
        from distllm.config.settings import DistLLMSettings

        settings = DistLLMSettings()
        vllm_len = getattr(settings.vllm, "max_model_len", None)
        if isinstance(vllm_len, int) and vllm_len > 0:
            return vllm_len
        if getattr(settings.llamacpp, "enabled", False):
            n_ctx = getattr(settings.llamacpp, "n_ctx", None)
            if isinstance(n_ctx, int) and n_ctx > 0:
                return n_ctx
    except Exception:
        pass

    return None


def count_prompt_tokens(coordinator: Any, prompt: str) -> int | None:
    """Token-count *prompt* with the coordinator's tokenizer.

    Returns ``None`` when no tokenizer is available, in which case the caller
    should fall back to a conservative character-based estimate.
    """
    tokenizer = getattr(coordinator, "tokenizer", None)
    if tokenizer is None:
        return None
    try:
        return len(tokenizer.encode(prompt))
    except Exception:
        return None


def estimate_tokens_without_tokenizer(prompt: str) -> int:
    """Rough lower bound used only when the tokenizer is unavailable.

    Four characters per token is the common English approximation.  It is
    intentionally conservative: the guard should not reject a request that
    might fit, so this is used only as a last resort.
    """
    return max(1, len(prompt) // 4)


def check_context_length(
    coordinator: Any,
    prompt: str,
    max_tokens: int,
    request: Any = None,
):
    """Return an error ``JSONResponse`` if the request overflows the window.

    Returns ``None`` when the request fits, or when the window is unknown.

    ``max_tokens`` is the generation budget the client asked for.  The check
    is ``prompt_tokens + max_tokens > window``: the backend must hold both the
    prompt and the tokens it is about to generate.
    """
    window = resolve_context_window(coordinator)
    if window is None:
        return None

    prompt_tokens = count_prompt_tokens(coordinator, prompt)
    estimated = prompt_tokens is None
    if prompt_tokens is None:
        prompt_tokens = estimate_tokens_without_tokenizer(prompt)

    if prompt_tokens + max_tokens <= window:
        return None

    # Only the real tokenizer count is authoritative enough to reject on a
    # "prompt alone" basis; with the char heuristic, report the combined
    # overflow either way.
    message = (
        f"This model's maximum context length is {window} tokens. "
        f"However, your request needs {prompt_tokens} prompt tokens"
    )
    if max_tokens:
        message += f" plus {max_tokens} completion tokens"
    message += (
        f" for a total of {prompt_tokens + max_tokens} tokens. "
        f"Please shorten the messages or set a smaller max_tokens."
    )

    return error_response_from_request(
        status_code=400,
        error="Bad Request",
        message=message,
        type="invalid_request_error",
        request=request,
        code="context_length_exceeded",
    )
