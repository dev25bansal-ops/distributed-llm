"""Detect and redact personally identifiable information in text.

Supports regex-based detection for emails, phone numbers, SSNs, credit
cards, and IP addresses, plus optional transformer-based NER for
persons, organisations, and locations.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

from loguru import logger

from distllm.security.content_moderation.base import (
    _lazy_import,
    PIEEntity,
    PIIResult,
)

# Regex patterns for common PII types.  Each entry maps a human-readable
# label to a compiled regex and a redaction template.
_PII_PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    (
        "email",
        re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
        "[EMAIL]",
    ),
    (
        "ip_address",
        re.compile(
            r"\b(?:(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)\.){3}"
            r"(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)\b"
        ),
        "[IP_ADDRESS]",
    ),
    (
        "ssn",
        re.compile(r"\b(?!000|666|9\d{2})\d{3}[-](?!00)\d{2}[-](?!0000)\d{4}\b"),
        "[SSN]",
    ),
    (
        "credit_card",
        re.compile(
            r"\b(?:\d{4}[-\s]?){3}\d{4}\b"
            r"|\b\d{16}\b"
        ),
        "[CREDIT_CARD]",
    ),
    (
        "phone",
        re.compile(
            r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?"
            r"\d{3,4}[-.\s]?\d{3,4}(?:\s?(?:ext|x\.?)\s?\d{1,5})?\b"
        ),
        "[PHONE]",
    ),
    # NOTE: Phone regex intentionally includes '.' as separator (common in
    # US numbers like 555.123.4567).  Overlapping matches with IP addresses
    # and credit cards are resolved by the overlap-resolution logic --
    # the longer / more specific match always wins.
]


class _Replacement:
    """A single text replacement with original-text offsets."""

    __slots__ = ("start", "end", "replacement", "entity")

    def __init__(
        self, start: int, end: int, replacement: str, entity: PIEEntity
    ) -> None:
        self.start = start
        self.end = end
        self.replacement = replacement
        self.entity = entity


class PIIRedactor:
    """Detect and redact personally identifiable information in text.

    Uses regex-based detection for email addresses, phone numbers, SSNs,
    credit card numbers, and IP addresses.  When ``transformers`` is
    available with a token-classification (NER) pipeline, the redactor
    also detects named entities (persons, organisations, locations) as
    additional PII signals.

    **Offset bug fix**: All regex and NER detections are collected
    against the *original* text offsets, then applied in reverse order
    so that earlier replacement positions remain valid.  This prevents
    the previous bug where NER offsets from the original text were
    applied to a string that had already been shortened by regex
    substitutions.

    Args:
        patterns: Additional PII patterns as ``(label, regex_str, replacement)``
            tuples to merge into the default pattern set.
        redact_with_label: When ``True`` (default), replace detected PII
            with descriptive labels like ``[EMAIL]``.  When ``False``,
            replace with ``[REDACTED]``.
        enable_ner: Whether to supplement regex detection with a
            transformer NER model.  Defaults to ``False`` (pattern-based
            only).  Set to ``True`` to enable transformer-based entity
            recognition -- this will download the model on first use.
        ner_model: HuggingFace model name for NER.  Defaults to
            ``"dslim/bert-base-NER"``.
    """

    def __init__(
        self,
        patterns: list[tuple[str, str, str]] | None = None,
        redact_with_label: bool = True,
        enable_ner: bool = False,
        ner_model: str = "dslim/bert-base-NER",
    ) -> None:
        self._redact_with_label = redact_with_label
        self._patterns: list[tuple[str, re.Pattern[str], str]] = list(_PII_PATTERNS)

        if patterns:
            for label, regex_str, replacement in patterns:
                self._patterns.append((label, re.compile(regex_str), replacement))

        self._ner_pipeline: Any = None
        if enable_ner:
            has_tf, transformers_mod = _lazy_import("transformers")
            if has_tf:
                try:
                    has_torch, torch_mod = _lazy_import("torch")
                    device = (
                        0
                        if has_torch and getattr(torch_mod, "cuda", None) and torch_mod.cuda.is_available()
                        else -1
                    )
                    self._ner_pipeline = transformers_mod.pipeline(
                        "token-classification",
                        model=ner_model,
                        aggregation_strategy="simple",
                        device=device,
                    )
                    logger.debug("PIIRedactor loaded NER model: {}", ner_model)
                except Exception as exc:
                    logger.warning("Failed to load NER model {}: {}", ner_model, exc)
                    self._ner_pipeline = None

    # ------------------------------------------------------------------
    # Synchronous API
    # ------------------------------------------------------------------

    def redact(self, text: str) -> PIIResult:
        """Detect and redact PII in *text*.

        All detections are measured against the *original* text offsets
        for correctness.  Replacements are applied in reverse end-position
        order so that positions do not shift as the string is modified.

        Args:
            text: The input string to scan.

        Returns:
            A ``PIIResult`` containing the redacted text and a list of
            detected entities (sorted by start position).
        """
        if not text:
            return PIIResult(redacted_text=text, entities_found=[])

        entities: list[PIEEntity] = []
        replacements: list[_Replacement] = []

        # ---- 1. Regex-based detection on the ORIGINAL text ----
        for label, pattern, replacement in self._patterns:
            for match in pattern.finditer(text):
                rep = replacement if self._redact_with_label else "[REDACTED]"
                entity = PIEEntity(
                    type=label,
                    value=match.group(),
                    start=match.start(),
                    end=match.end(),
                    redacted=rep,
                )
                entities.append(entity)
                replacements.append(
                    _Replacement(match.start(), match.end(), rep, entity)
                )

        # ---- 2. NER-based detection on the ORIGINAL text ----
        if self._ner_pipeline is not None:
            try:
                ner_results = self._ner_pipeline(text)  # type: ignore[misc]
                for ner_entity in ner_results:
                    label = ner_entity.get("entity_group", ner_entity.get("entity", "unknown")).lower()
                    score = ner_entity.get("score", 0.0)
                    if score < 0.7:
                        continue
                    word = ner_entity.get("word", "")
                    ner_start = ner_entity.get("start", 0)
                    ner_end = ner_entity.get("end", 0)

                    # Avoid double-reporting entities already caught by regex.
                    already_found = any(
                        e.start == ner_start and e.end == ner_end for e in entities
                    )
                    if already_found:
                        continue

                    rep = f"[{label.upper()}]" if self._redact_with_label else "[REDACTED]"
                    entity = PIEEntity(
                        type=f"ner:{label}",
                        value=word,
                        start=ner_start,
                        end=ner_end,
                        redacted=rep,
                    )
                    entities.append(entity)
                    replacements.append(_Replacement(ner_start, ner_end, rep, entity))

            except Exception as exc:
                logger.debug("NER inference failed (non-fatal): {}", exc)

        # ---- 3. Resolve overlapping replacements ----
        # When multiple regexes match overlapping spans (e.g. an IP address
        # that also looks like a partial phone number), keep the longest
        # match as the most specific one.
        if replacements:
            # Sort by start ascending, then by end descending (longer span first).
            replacements.sort(key=lambda r: (r.start, -r.end))
            non_overlapping: list[_Replacement] = [replacements[0]]
            for r in replacements[1:]:
                last = non_overlapping[-1]
                if r.start >= last.end:
                    # No overlap -- keep it.
                    non_overlapping.append(r)
                elif (r.end - r.start) > (last.end - last.start):
                    # Overlaps but is longer -- replace the last entry.
                    non_overlapping[-1] = r
                # Otherwise the existing entry (longer) wins -- skip this one.

            # Rebuild entities from the winning replacements.
            entities = [r.entity for r in non_overlapping]

            # ---- 4. Apply replacements in REVERSE order (end to start) ----
            non_overlapping.sort(key=lambda r: r.end, reverse=True)
            redacted = text
            for r in non_overlapping:
                redacted = redacted[: r.start] + r.replacement + redacted[r.end :]
        else:
            redacted = text

        entities.sort(key=lambda e: e.start)
        return PIIResult(
            redacted_text=redacted,
            entities_found=entities,
        )

    # ------------------------------------------------------------------
    # Async API
    # ------------------------------------------------------------------

    async def async_redact(self, text: str) -> PIIResult:
        """Async variant of :meth:`redact` that offloads to a thread pool."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.redact, text)
