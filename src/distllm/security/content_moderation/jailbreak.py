"""Detect jailbreak and prompt-injection attempts in message exchanges."""

from __future__ import annotations

import asyncio
import re
from typing import Any

from loguru import logger

from distllm.security.content_moderation.base import (
    _DEFAULT_JAILBREAK_THRESHOLD,
    _lazy_import,
    JailbreakResult,
)

# Known jailbreak and prompt-injection patterns.  These are matched case-
# insensitively against the concatenated message content.
_JAILBREAK_PATTERNS: list[tuple[str, str]] = [
    ("ignore_instructions", r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions"),
    ("ignore_instructions", r"ignore\s+(all\s+)?(previous|prior|above)\s+(commands|directives|rules)"),
    ("ignore_instructions", r"ignore\s+(all\s+)?(rules|instructions|commands|directives)"),
    ("dan_mode", r"act\s+as\s+(dan|do\s+anything\s+now)"),
    ("dan_mode", r"you\s+(no\s+longer\s+(need|have)|don't\s+(need|have))\s+(to\s+)?(follow|adhere|obey)"),
    ("roleplay_bypass", r"role\s*play\s+as\s+.*?(without\s+(restriction|limit|rule)|unfiltered|uncensored)"),
    ("roleplay_bypass", r"pretend\s+(you\s+are|to\s+be)\s+.*?(without\s+(restriction|limit|rule)|no\s+(restrictions|rules|limits))"),
    ("roleplay_bypass", r"pretend\s+.*?(no\s+(restrictions?|rules?|limits?)|without\s+(restriction|limit|rule))"),
    ("system_prompt_leak", r"(print|reveal|show|output|display|leak|dump)\s+(your|the)\s+(system\s+)?prompt"),
    ("system_prompt_leak", r"(print|reveal|show|output|display|leak|dump)\s+(your|the)\s+(initial|first|system)\s+message"),
    ("hypothetical_bypass", r"(imagine|hypothetical|pretend|simulate)\s+(a\s+)?scenario\s+where\s+you\s+(can|are\s+allowed)"),
    ("hypothetical_bypass", r"(fictional|hypothetical|theoretical|creative)\s+(writing|scenario|situation)\s+.*?(no\s+(rules|restrictions|limits))"),
    ("encoding_bypass", r"(base64|base32|hex|rot13|caesar|cipher)\s+(encode|decode|encrypt|decrypt)"),
    ("encoding_bypass", r"(bypass|evade|circumvent|crack|break)\s+(your|the|these)\s+(safety|guardrails|restrictions|filters|rules)"),
    ("encoding_bypass", r"tell\s+me\s+how\s+to\s+(hack|crack|exploit|bypass|jailbreak)"),
    ("universal_bypass", r"(token|key|password|secret)\s+(is|are)\s+['\"]?\w+['\"]?\s+and\s+(reveal|output|print)\s+(it|them)"),
    ("universal_bypass", r"you\s+must\s+(now\s+)?(ignore|disregard|forget|erase|remove)\s+(all\s+)?(constraints|restrictions|boundaries)"),
]


class JailbreakDetector:
    """Detect jailbreak and prompt-injection attempts in message exchanges.

    Uses a two-tier approach:
    1. Pattern matching against a comprehensive set of known jailbreak
       patterns (always available).
    2. Optional transformer-based classifier for semantic detection of
       novel attack patterns.

    Args:
        threshold: Confidence threshold above which an attempt is flagged.
            Defaults to ``CONTENT_JAILBREAK_THRESHOLD`` env or ``0.5``.
        use_ml: Whether to attempt ML-based classification in addition
            to pattern matching.  Defaults to ``False`` (pattern-based
            only).  Set to ``True`` to enable transformer-based detection
            -- this will download a prompt-injection model on first use.
        enable_patterns: Whether to use pattern-based detection.
            Defaults to ``True``.
        custom_patterns: Additional ``(category, regex_str)`` patterns to
            merge into the default set.
    """

    def __init__(
        self,
        threshold: float = _DEFAULT_JAILBREAK_THRESHOLD,
        use_ml: bool = False,
        enable_patterns: bool = True,
        custom_patterns: list[tuple[str, str]] | None = None,
    ) -> None:
        self._threshold = threshold
        self._enable_patterns = enable_patterns
        self._compiled: list[tuple[str, re.Pattern[str]]] = []

        if enable_patterns:
            combined = list(_JAILBREAK_PATTERNS)
            if custom_patterns:
                combined.extend(custom_patterns)
            for category, regex_str in combined:
                try:
                    self._compiled.append((category, re.compile(regex_str, re.IGNORECASE)))
                except re.error as exc:
                    logger.warning("Invalid jailbreak pattern {!r}: {}", regex_str, exc)

        self._ml_pipeline: Any = None
        if use_ml:
            has_tf, transformers_mod = _lazy_import("transformers")
            if has_tf:
                try:
                    has_torch, torch_mod = _lazy_import("torch")
                    device = (
                        0
                        if has_torch and getattr(torch_mod, "cuda", None) and torch_mod.cuda.is_available()
                        else -1
                    )
                    self._ml_pipeline = transformers_mod.pipeline(
                        "text-classification",
                        model="ProtectAI/deberta-v3-base-prompt-injection-v2",
                        device=device,
                    )
                    logger.debug("JailbreakDetector loaded ML model.")
                except Exception as exc:
                    logger.warning(
                        "Failed to load jailbreak ML model (using patterns only): {}",
                        exc,
                    )
                    self._ml_pipeline = None

    # ------------------------------------------------------------------
    # Synchronous API
    # ------------------------------------------------------------------

    def check(self, messages: list[dict[str, str]]) -> JailbreakResult:
        """Analyse a message list for jailbreak attempts.

        *messages* is expected to be a list of dicts with at least a
        ``"content"`` key.  Typical chat formats (``{"role": ..., "content": ...}``)
        are handled transparently.

        Args:
            messages: The conversation or message list to inspect.

        Returns:
            A ``JailbreakResult`` with detection verdict and matched patterns.
        """
        if not messages:
            return JailbreakResult(jailbreak_attempt=False, confidence=0.0)

        # Flatten all message content into a single string for analysis.
        full_text = " ".join(
            m.get("content", "") if isinstance(m, dict) else str(m)
            for m in messages
        )

        matched_categories: list[str] = []
        pattern_score = 0.0

        # 1. Pattern matching.
        if self._enable_patterns and self._compiled:
            matches: set[str] = set()
            for category, pattern in self._compiled:
                if pattern.search(full_text):
                    matches.add(category)
            matched_categories = sorted(matches)
            # Scale confidence by number of distinct categories matched.
            pattern_score = min(len(matched_categories) * 0.5, 0.95)

        # 2. ML classification.
        ml_score = 0.0
        if self._ml_pipeline is not None:
            try:
                result = self._ml_pipeline(full_text)[0]
                label = result.get("label", "safe").lower()
                ml_score = float(result.get("score", 0.0))
                if "injection" in label or "jailbreak" in label or label == "unsafe":
                    ml_score = ml_score
                else:
                    ml_score = 0.0
            except Exception as exc:
                logger.debug("ML jailbreak inference failed (non-fatal): {}", exc)

        # 3. Combine scores -- take the maximum of pattern and ML signals.
        confidence = max(pattern_score, ml_score)

        return JailbreakResult(
            jailbreak_attempt=confidence >= self._threshold,
            confidence=round(confidence, 4),
            matched_patterns=matched_categories,
            explanation=self._build_explanation(matched_categories, ml_score, confidence),
        )

    @staticmethod
    def _build_explanation(
        matched_categories: list[str],
        ml_score: float,
        confidence: float,
    ) -> str:
        parts: list[str] = []
        if matched_categories:
            parts.append(f"Pattern matches: {', '.join(matched_categories)}")
        if ml_score > 0.5:
            parts.append(f"ML classifier score: {ml_score:.2f}")
        if not parts:
            return "No jailbreak indicators detected."
        return "; ".join(parts) + f" (overall confidence: {confidence:.2f})"

    # ------------------------------------------------------------------
    # Async API
    # ------------------------------------------------------------------

    async def async_check(self, messages: list[dict[str, str]]) -> JailbreakResult:
        """Async variant of :meth:`check` that offloads to a thread pool."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.check, messages)
