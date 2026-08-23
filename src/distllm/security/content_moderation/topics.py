"""Filter content based on configurable topic policies."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field

from loguru import logger

from distllm.security.content_moderation.base import TopicFilterResult


@dataclass(frozen=True)
class _TopicPolicy:
    """Internal representation of a single topic policy."""

    name: str
    terms: list[str]
    mode: str  # "allow" or "block"


class TopicFilter:
    """Filter content based on configurable topic policies.

    Policies are provided as dictionaries with ``"allow"`` and/or ``"block"``
    keys, each containing a list of terms, regex patterns, or phrases.
    When a block-list term matches, the content is flagged; when an
    allow-list term matches, it can override block-list matches (if
    ``allow_overrides`` is enabled).

    Args:
        allow_overrides: When ``True``, allow-list matches can override
            block-list violations for the same content.  Defaults to
            ``False``.
        case_sensitive: Whether keyword matching is case-sensitive.
            Defaults to ``False``.
    """

    def __init__(
        self,
        allow_overrides: bool = False,
        case_sensitive: bool = False,
    ) -> None:
        self._allow_overrides = allow_overrides
        self._case_sensitive = case_sensitive

    # ------------------------------------------------------------------
    # Synchronous API
    # ------------------------------------------------------------------

    def check(
        self,
        text: str,
        policies: dict[str, list[str]] | None = None,
    ) -> TopicFilterResult:
        """Evaluate *text* against *policies*.

        The *policies* dict can contain ``"allow"`` and/or ``"block"``
        keys.  The value for each is a list of strings; each string is
        either a literal keyword or a regex pattern enclosed in ``/``
        slashes (e.g. ``"/\\bviolence\\b/i"``).

        Args:
            text: The content to check.
            policies: A policy dict.  If ``None``, an empty policy set
                is used (everything is allowed).

        Returns:
            A ``TopicFilterResult`` indicating whether the content is
            allowed and which policies (if any) were violated.
        """
        if not text:
            return TopicFilterResult(allowed=True, violated_policies=[], matched_terms={})

        policies = policies or {}
        matched_terms: dict[str, list[str]] = {}
        violations: list[str] = []

        block_terms = policies.get("block", [])
        allow_terms = policies.get("allow", [])

        # Check block list.
        if block_terms:
            block_matches = self._match_terms(text, block_terms)
            if block_matches:
                matched_terms["block"] = block_matches
                violations.append("block")

        # Check allow list (only relevant if there were violations).
        allow_matches: list[str] = []
        if violations and allow_terms:
            allow_matches = self._match_terms(text, allow_terms)
            if allow_matches:
                matched_terms["allow"] = allow_matches

        # Determine verdict.
        if not violations:
            allowed = True
        elif self._allow_overrides and allow_matches:
            # Allow-list match overrides block-list violation.
            allowed = True
            violations = []
        else:
            allowed = False

        return TopicFilterResult(
            allowed=allowed,
            violated_policies=violations,
            matched_terms=matched_terms,
        )

    def _match_terms(self, text: str, terms: list[str]) -> list[str]:
        """Return the subset of *terms* that match in *text*.

        Each term is either a literal keyword or a regex pattern
        enclosed in ``/`` delimiters (e.g. ``"/\\bpython\\b/i"``).
        """
        if not self._case_sensitive:
            search_text = text.lower()
        else:
            search_text = text

        matches: list[str] = []
        for term in terms:
            if term.startswith("/") and term.endswith("/"):
                # Raw regex.
                raw = term[1:-1]
                flags = re.IGNORECASE if not self._case_sensitive else 0
                try:
                    if re.search(raw, search_text, flags=flags):
                        matches.append(term)
                except re.error as exc:
                    logger.warning("Invalid regex in topic policy {!r}: {}", term, exc)
            else:
                # Literal keyword.
                needle = term if self._case_sensitive else term.lower()
                if needle in search_text:
                    matches.append(term)
        return matches

    # ------------------------------------------------------------------
    # Async API
    # ------------------------------------------------------------------

    async def async_check(
        self,
        text: str,
        policies: dict[str, list[str]] | None = None,
    ) -> TopicFilterResult:
        """Async variant of :meth:`check` that offloads to a thread pool."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.check, text, policies)
