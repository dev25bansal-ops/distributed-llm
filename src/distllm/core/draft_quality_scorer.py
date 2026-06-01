"""Draft model quality scoring and auto-selection.

Tracks acceptance rate per draft model per target model, and
automatically selects the best draft model for each request.

Usage::

    scorer = DraftQualityScorer()
    scorer.record("draft-7b", "target-70b", accepted=4, total=8)
    best = scorer.select_best_draft("target-70b", available_drafts)
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class DraftStats:
    """Statistics for a draft-target model pair."""
    draft_model: str
    target_model: str
    total_accepted: int = 0
    total_proposed: int = 0
    total_calls: int = 0
    avg_acceptance_rate: float = 0.0
    last_updated: float = field(default_factory=time.time)


class DraftQualityScorer:
    """Tracks and scores draft model quality for speculative decoding.

    Maintains per-pair acceptance statistics and selects the best
    draft model for a given target model.
    """

    def __init__(self, decay_factor: float = 0.95):
        self._decay_factor = decay_factor
        self._stats: dict[tuple[str, str], DraftStats] = {}
        self._lock = threading.Lock()

    def record(
        self,
        draft_model: str,
        target_model: str,
        accepted: int,
        total: int,
    ) -> None:
        """Record a speculative decoding result.

        Args:
            draft_model: Name/ID of the draft model.
            target_model: Name/ID of the target model.
            accepted: Number of tokens accepted.
            total: Number of tokens proposed.
        """
        key = (draft_model, target_model)
        with self._lock:
            if key not in self._stats:
                self._stats[key] = DraftStats(
                    draft_model=draft_model,
                    target_model=target_model,
                )
            stats = self._stats[key]

            # Exponential moving average
            rate = accepted / max(total, 1)
            if stats.total_calls == 0:
                stats.avg_acceptance_rate = rate
            else:
                stats.avg_acceptance_rate = (
                    self._decay_factor * stats.avg_acceptance_rate
                    + (1 - self._decay_factor) * rate
                )

            stats.total_accepted += accepted
            stats.total_proposed += total
            stats.total_calls += 1
            stats.last_updated = time.time()

    def get_acceptance_rate(
        self, draft_model: str, target_model: str
    ) -> float | None:
        """Get the acceptance rate for a draft-target pair."""
        key = (draft_model, target_model)
        with self._lock:
            stats = self._stats.get(key)
            return stats.avg_acceptance_rate if stats else None

    def select_best_draft(
        self,
        target_model: str,
        available_drafts: list[str],
        min_calls: int = 3,
    ) -> str | None:
        """Select the best draft model for a target model.

        Args:
            target_model: The target model to find a draft for.
            available_drafts: List of available draft model names.
            min_calls: Minimum calls before trusting a draft model.

        Returns:
            Name of the best draft model, or None if no data.
        """
        with self._lock:
            best_draft = None
            best_rate = -1.0

            for draft in available_drafts:
                key = (draft, target_model)
                stats = self._stats.get(key)
                if stats is None or stats.total_calls < min_calls:
                    continue
                if stats.avg_acceptance_rate > best_rate:
                    best_rate = stats.avg_acceptance_rate
                    best_draft = draft

            return best_draft

    def get_all_stats(self) -> list[dict]:
        """Return all draft model statistics."""
        with self._lock:
            return [
                {
                    "draft_model": s.draft_model,
                    "target_model": s.target_model,
                    "acceptance_rate": round(s.avg_acceptance_rate, 3),
                    "total_accepted": s.total_accepted,
                    "total_proposed": s.total_proposed,
                    "total_calls": s.total_calls,
                }
                for s in self._stats.values()
            ]

    def get_leaderboard(self, target_model: str) -> list[dict]:
        """Get draft models ranked by acceptance rate for a target."""
        with self._lock:
            entries = [
                s for s in self._stats.values()
                if s.target_model == target_model
            ]
            entries.sort(key=lambda s: s.avg_acceptance_rate, reverse=True)
            return [
                {
                    "draft_model": s.draft_model,
                    "acceptance_rate": round(s.avg_acceptance_rate, 3),
                    "total_calls": s.total_calls,
                }
                for s in entries
            ]
