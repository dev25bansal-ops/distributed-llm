"""Toxicity detection using transformer, ONNX, or keyword-based methods."""

from __future__ import annotations

import asyncio
import os
from typing import Any

from loguru import logger

from distllm.security.content_moderation.base import (
    _DEFAULT_TOXICITY_MODEL,
    _DEFAULT_TOXICITY_THRESHOLD,
    _KeywordBackend,
    _lazy_import,
    _ONNXBackend,
    _TextClassifierBackend,
    _TransformersBackend,
    ToxicResult,
)


class ToxicityDetector:
    """Detect toxic language in text using transformer, ONNX, or keyword-based methods.

    The detector tries backends in order:
    1. HuggingFace ``transformers`` pipeline (GPU if available).
    2. ONNX Runtime (requires an exported model directory).
    3. Keyword / pattern matching (always available).

    Args:
        model_name: HuggingFace model name, or path to a directory
            containing ``model.onnx`` for the ONNX fallback.
        threshold: Score above which text is considered toxic
            (0.0--1.0).  Defaults to ``CONTENT_MODERATION_THRESHOLD`` env
            var or ``0.7``.
        use_keyword_fallback: Whether to fall back to keyword matching
            when no ML backend is available.  Defaults to ``True``.
        custom_keywords: Additional keyword categories to merge into
            the keyword fallback (keyword -> list of terms).
        categories: Subset of toxicity categories to monitor.  When
            ``None``, all available categories are checked.

    Raises:
        RuntimeError: If no backend can be initialised and
            ``use_keyword_fallback`` is ``False``.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_TOXICITY_MODEL,
        threshold: float = _DEFAULT_TOXICITY_THRESHOLD,
        use_keyword_fallback: bool = True,
        custom_keywords: dict[str, list[str]] | None = None,
        categories: list[str] | None = None,
    ) -> None:
        self._threshold = threshold
        self._categories = categories
        self._backend: _TextClassifierBackend | None = None

        # Try backends in priority order.
        tx_backend = _TransformersBackend(model_name)
        if tx_backend.available():
            self._backend = tx_backend
            logger.debug("ToxicityDetector using transformers backend.")
            return

        has_onnx, _ = _lazy_import("onnxruntime")
        if has_onnx and os.path.isdir(model_name):
            onnx_backend = _ONNXBackend(model_name)
            if onnx_backend.available():
                self._backend = onnx_backend
                logger.debug("ToxicityDetector using ONNX backend.")
                return

        if use_keyword_fallback:
            kw_backend = _KeywordBackend(custom_keywords=custom_keywords)
            self._backend = kw_backend
            logger.debug("ToxicityDetector using keyword backend (no ML model loaded).")
            return

        raise RuntimeError(
            "No toxicity detection backend available. Install transformers, "
            "onnxruntime, or set use_keyword_fallback=True."
        )

    @property
    def backend_type(self) -> str:
        """Human-readable name of the active backend."""
        if isinstance(self._backend, _TransformersBackend):
            return "transformers"
        if isinstance(self._backend, _ONNXBackend):
            return "onnx"
        if isinstance(self._backend, _KeywordBackend):
            return "keyword"
        return "none"

    # ------------------------------------------------------------------
    # Synchronous API
    # ------------------------------------------------------------------

    def check(self, text: str) -> ToxicResult:
        """Run toxicity detection on *text*.

        Args:
            text: The input string to classify.

        Returns:
            A ``ToxicResult`` with per-category scores and an aggregate verdict.
        """
        if not text or not text.strip():
            return ToxicResult(toxic=False, categories={}, score=0.0)

        assert self._backend is not None
        scores = self._backend.predict(text, categories=self._categories)
        aggregate = max(scores.values()) if scores else 0.0

        return ToxicResult(
            toxic=aggregate >= self._threshold,
            categories=scores,
            score=round(aggregate, 4),
        )

    # ------------------------------------------------------------------
    # Async API
    # ------------------------------------------------------------------

    async def async_check(self, text: str) -> ToxicResult:
        """Async variant of :meth:`check` that offloads to a thread pool."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.check, text)
