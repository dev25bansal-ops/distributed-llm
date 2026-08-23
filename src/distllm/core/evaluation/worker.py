"""Parallel worker pool and token counting helpers for LLM evaluation.

Extracted from :mod:`distllm.core.evaluation_harness`.
"""

from __future__ import annotations

import concurrent.futures
import re
import threading
from collections.abc import Callable
from typing import Any

from loguru import logger

from distllm.core.evaluation.constants import _MAX_WORKERS
from distllm.core.evaluation.models import EvalResult, EvalSample


class _WorkerPool:
    """Manages parallel evaluation across worker threads."""

    def __init__(self, max_workers: int = _MAX_WORKERS) -> None:
        self._max_workers = max_workers

    def run(
        self,
        samples: list[EvalSample],
        generate_fn: Callable[[str], tuple[str, float, int, int]],  # (prediction, latency, prompt_tokens, gen_tokens)
        progress_cb: Callable[[int, int], None] | None = None,
    ) -> list[EvalResult]:
        """Evaluate samples in parallel using a thread pool."""
        results: list[EvalResult | None] = [None] * len(samples)
        completed = [0]
        lock = threading.RLock()

        def _evaluate(idx: int, sample: EvalSample) -> None:
            try:
                prediction, latency, ptokens, gtokens = generate_fn(sample.question)
                with lock:
                    results[idx] = EvalResult(
                        sample=sample,
                        prediction=prediction,
                        latency_ms=latency,
                        prompt_tokens=ptokens,
                        generated_tokens=gtokens,
                    )
                    completed[0] += 1
                    if progress_cb:
                        progress_cb(completed[0], len(samples))
            except Exception as exc:
                logger.error("Eval failed for sample {}: {}", idx, exc)
                with lock:
                    results[idx] = EvalResult(
                        sample=sample,
                        prediction="",
                        error=str(exc),
                    )
                    completed[0] += 1
                    if progress_cb:
                        progress_cb(completed[0], len(samples))

        with concurrent.futures.ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = [
                pool.submit(_evaluate, i, sample)
                for i, sample in enumerate(samples)
            ]
            concurrent.futures.wait(futures)

        return [r for r in results if r is not None]


# ---------------------------------------------------------------------------
# Token count helper
# ---------------------------------------------------------------------------


def _count_tokens(text: str, language: str = "en") -> int:
    """Rough token count estimate.

    Different languages/tasks have different tokenization ratios:
    - English prose: ~4 chars/token
    - Code: ~3.5 chars/token (operators, punctuation)
    - CJK (Chinese/Japanese/Korean): ~1.5 chars/token
    - Python/JSON: ~3 chars/token

    These are all estimates. Pass a real tokenizer for accuracy.

    Args:
        text: The text to estimate.
        language: "en", "code", "cjk", "py", or "auto" (tries to detect).

    Returns:
        Estimated token count.
    """
    chars = len(text)
    words = len(text.split())

    if language == "auto":
        # Simple heuristic: if >30% chars are non-ASCII, treat as CJK
        non_ascii = sum(1 for c in text if ord(c) > 127)
        if non_ascii > chars * 0.3:
            language = "cjk"
        # If lots of punctuation/operators, treat as code
        special = sum(1 for c in text if c in "{}[]()=+-*/&|!<>;:#@")
        if special > chars * 0.05:
            language = "code"

    ratios = {
        "en": 4.0,
        "code": 3.5,
        "cjk": 1.5,
        "py": 3.0,
    }
    ratio = ratios.get(language, 4.0)
    # Blend word-count and char-count heuristics
    return max(words, int(chars / ratio))


__all__ = [
    "_WorkerPool",
    "_count_tokens",
]
