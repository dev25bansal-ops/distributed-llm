"""Orchestrate multiple content detectors in a single pass."""

from __future__ import annotations

import asyncio
from typing import Any

from distllm.security.content_moderation.base import (
    ModerationResult,
    ToxicResult,
    PIIResult,
    JailbreakResult,
    TopicFilterResult,
)
from distllm.security.content_moderation.toxicity import ToxicityDetector
from distllm.security.content_moderation.pii import PIIRedactor
from distllm.security.content_moderation.jailbreak import JailbreakDetector
from distllm.security.content_moderation.topics import TopicFilter

# Mapping from detector names to their result attribute keys.
_DETECTOR_RESULT_ATTRS: dict[str, str] = {
    "toxicity": "toxicity",
    "pii": "pii",
    "jailbreak": "jailbreak",
    "topic_filter": "topic_filter",
}


class ContentModerationPipeline:
    """Orchestrate multiple content detectors in a single pass.

    Runs each enabled detector on the input and aggregates the results
    into a single ``ModerationResult``.  The pipeline is considered to
    have *passed* when all enabled checks pass (no toxicity, no jailbreak,
    topics allowed, ...).

    Both synchronous (``process``) and asynchronous (``async_process``)
    entry points are provided.

    Args:
        detect_toxicity: Enable the toxicity detector.  Defaults to
            ``True``.
        redact_pii: Enable PII redaction.  Defaults to ``True``.
        detect_jailbreak: Enable jailbreak detection.  Defaults to
            ``True``.
        filter_topics: Enable topic filtering.  Defaults to ``False``.
        toxicity_kwargs: Keyword arguments forwarded to
            ``ToxicityDetector``.
        pii_kwargs: Keyword arguments forwarded to ``PIIRedactor``.
        jailbreak_kwargs: Keyword arguments forwarded to
            ``JailbreakDetector``.
        topic_filter_kwargs: Keyword arguments forwarded to
            ``TopicFilter``.
        default_policies: Default topic policies used when no policies
            are provided to ``process()``.
        fail_on: Set of checks that cause the pipeline verdict to be
            ``failed`` when triggered.  Defaults to
            ``{"toxicity", "jailbreak"}``.
    """

    def __init__(
        self,
        detect_toxicity: bool = True,
        redact_pii: bool = True,
        detect_jailbreak: bool = True,
        filter_topics: bool = False,
        toxicity_kwargs: dict[str, Any] | None = None,
        pii_kwargs: dict[str, Any] | None = None,
        jailbreak_kwargs: dict[str, Any] | None = None,
        topic_filter_kwargs: dict[str, Any] | None = None,
        default_policies: dict[str, list[str]] | None = None,
        fail_on: set[str] | None = None,
    ) -> None:
        self._toxicity_detector: ToxicityDetector | None = None
        self._pii_redactor: PIIRedactor | None = None
        self._jailbreak_detector: JailbreakDetector | None = None
        self._topic_filter: TopicFilter | None = None
        self._default_policies = default_policies or {}
        self._fail_on = fail_on or {"toxicity", "jailbreak"}

        if detect_toxicity:
            self._toxicity_detector = ToxicityDetector(**(toxicity_kwargs or {}))
        if redact_pii:
            self._pii_redactor = PIIRedactor(**(pii_kwargs or {}))
        if detect_jailbreak:
            self._jailbreak_detector = JailbreakDetector(**(jailbreak_kwargs or {}))
        if filter_topics:
            self._topic_filter = TopicFilter(**(topic_filter_kwargs or {}))

    # ------------------------------------------------------------------
    # Synchronous API
    # ------------------------------------------------------------------

    def process(
        self,
        input_data: str | dict[str, Any],
        topic_policies: dict[str, list[str]] | None = None,
    ) -> ModerationResult:
        """Run all enabled detectors on *input_data*.

        *input_data* can be a plain string or a dict with a ``"content"``
        key (matching typical LLM message formats).  For jailbreak detection,
        the dict may contain a ``"messages"`` key with a list of messages.

        Args:
            input_data: The content to moderate.
            topic_policies: Topic policies to apply.  Falls back to
                ``default_policies`` if not provided.

        Returns:
            A ``ModerationResult`` aggregating all detector outputs.
        """
        # Normalise the input.
        text: str = ""
        messages: list[dict[str, str]] = []

        if isinstance(input_data, str):
            text = input_data
            messages = [{"role": "user", "content": input_data}]
        elif isinstance(input_data, dict):
            text = input_data.get("content", "") or ""
            msgs = input_data.get("messages", None)
            if msgs is not None:
                messages = list(msgs)
            else:
                messages = [{"role": input_data.get("role", "user"), "content": text}]
        else:
            text = str(input_data)
            messages = [{"role": "user", "content": text}]

        # Run each enabled detector.
        tox_result: ToxicResult | None = None
        pii_result: PIIResult | None = None
        jb_result: JailbreakResult | None = None
        topic_result: TopicFilterResult | None = None
        redacted = text

        if self._pii_redactor is not None:
            pii_result = self._pii_redactor.redact(text)
            redacted = pii_result.redacted_text

        if self._toxicity_detector is not None:
            tox_result = self._toxicity_detector.check(redacted)

        if self._jailbreak_detector is not None:
            jb_result = self._jailbreak_detector.check(messages)

        if self._topic_filter is not None:
            policies = topic_policies if topic_policies is not None else self._default_policies
            topic_result = self._topic_filter.check(text, policies=policies)

        # Compute aggregate verdict.
        failures: list[str] = []

        if tox_result is not None and tox_result.toxic and "toxicity" in self._fail_on:
            failures.append("toxicity")
        if jb_result is not None and jb_result.jailbreak_attempt and "jailbreak" in self._fail_on:
            failures.append("jailbreak")
        if topic_result is not None and not topic_result.allowed and "topic_filter" in self._fail_on:
            failures.append("topic_filter")
        if pii_result is not None and pii_result.entities_found and "pii" in self._fail_on:
            failures.append("pii")

        passed = len(failures) == 0

        return ModerationResult(
            passed=passed,
            toxicity=tox_result,
            pii=pii_result,
            jailbreak=jb_result,
            topic_filter=topic_result,
            redacted_text=redacted,
        )

    # ------------------------------------------------------------------
    # Async API
    # ------------------------------------------------------------------

    async def async_process(
        self,
        input_data: str | dict[str, Any],
        topic_policies: dict[str, list[str]] | None = None,
    ) -> ModerationResult:
        """Async variant of :meth:`process` using per-detector async methods."""
        # Normalise the input.
        text: str = ""
        messages: list[dict[str, str]] = []

        if isinstance(input_data, str):
            text = input_data
            messages = [{"role": "user", "content": input_data}]
        elif isinstance(input_data, dict):
            text = input_data.get("content", "") or ""
            msgs = input_data.get("messages", None)
            if msgs is not None:
                messages = list(msgs)
            else:
                messages = [{"role": input_data.get("role", "user"), "content": text}]
        else:
            text = str(input_data)
            messages = [{"role": "user", "content": text}]

        tox_result: ToxicResult | None = None
        pii_result: PIIResult | None = None
        jb_result: JailbreakResult | None = None
        topic_result: TopicFilterResult | None = None
        redacted = text

        # Run PII redaction async (if enabled).
        if self._pii_redactor is not None:
            pii_result = await self._pii_redactor.async_redact(text)
            redacted = pii_result.redacted_text

        # Prepare coroutines for parallel async execution.
        coros: list[Any] = []

        if self._toxicity_detector is not None:
            coros.append(self._toxicity_detector.async_check(redacted))
        else:
            coros.append(None)

        if self._jailbreak_detector is not None:
            coros.append(self._jailbreak_detector.async_check(messages))
        else:
            coros.append(None)

        if self._topic_filter is not None:
            policies = topic_policies if topic_policies is not None else self._default_policies
            coros.append(self._topic_filter.async_check(text, policies=policies))
        else:
            coros.append(None)

        # Run toxicity, jailbreak, and topic checks concurrently.
        if any(c is not None for c in coros):
            loop = asyncio.get_running_loop()
            # Gather only non-None coroutines.
            valid_coros = [c for c in coros if c is not None]
            results = await asyncio.gather(*valid_coros)

            idx = 0
            if self._toxicity_detector is not None:
                tox_result = results[idx]
                idx += 1
            if self._jailbreak_detector is not None:
                jb_result = results[idx]
                idx += 1
            if self._topic_filter is not None:
                topic_result = results[idx]
                idx += 1

        # Compute aggregate verdict.
        failures: list[str] = []

        if tox_result is not None and tox_result.toxic and "toxicity" in self._fail_on:
            failures.append("toxicity")
        if jb_result is not None and jb_result.jailbreak_attempt and "jailbreak" in self._fail_on:
            failures.append("jailbreak")
        if topic_result is not None and not topic_result.allowed and "topic_filter" in self._fail_on:
            failures.append("topic_filter")
        if pii_result is not None and pii_result.entities_found and "pii" in self._fail_on:
            failures.append("pii")

        passed = len(failures) == 0

        return ModerationResult(
            passed=passed,
            toxicity=tox_result,
            pii=pii_result,
            jailbreak=jb_result,
            topic_filter=topic_result,
            redacted_text=redacted,
        )
