"""Adaptive speculative decoding — selects draft model and candidate count dynamically.

Tracks acceptance rates per draft model and per workload class, then
adjusts the number of speculative candidates accordingly.  High-quality
drafts receive more candidates (more speculation per forward pass) while
low-quality drafts are limited to prevent wasted computation.

Also decides whether to use a remote draft or fall back to self-speculation
(embedded draft within the target model) based on observed quality.

Integration with :mod:`distllm.dist.speculative.draft_registry` and
:mod:`distllm.dist.adaptive_speculator` (the existing core-level adaptive
decoder) is supported: this module provides the per-model tracking layer
while the latter handles the full generation loop.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class AcceptanceStats:
    """Per-draft-model acceptance statistics with exponential moving average."""

    ema_acceptance: float = 0.0
    observation_count: int = 0
    total_speculated: int = 0
    total_accepted: int = 0

    @property
    def acceptance_rate(self) -> float:
        return self.total_accepted / max(self.total_speculated, 1)

    def update(self, rate: float, alpha: float = 0.3) -> None:
        """Update EMA and raw counts with a new observation.

        Args:
            rate: Acceptance rate observation (0.0 — 1.0).
            alpha: EMA smoothing factor (higher = more weight to recent).
        """
        self.observation_count += 1
        if self.observation_count == 1:
            self.ema_acceptance = rate
        else:
            self.ema_acceptance = (
                (1 - alpha) * self.ema_acceptance + alpha * rate
            )


def _default_candidate_range() -> tuple[int, int]:
    return (1, 16)


@dataclass
class AdaptiveSpecConfig:
    """Configuration for :class:`AdaptiveSpeculator`.

    Attributes:
        min_candidates: Minimum speculation depth for any draft.
        max_candidates: Maximum speculation depth for any draft.
        target_acceptance: Acceptance rate above which candidates are
            incremented (up to ``max_candidates``).
        low_acceptance_threshold: Below this threshold candidates are
            aggressively reduced toward ``min_candidates``.
        ema_alpha: EMA smoothing factor for acceptance tracking.
        cooldown_s: Minimum seconds between candidate adjustments for a
            given draft, preventing thrashing.
        warmup_observations: Number of observations before the adaptive
            logic trusts the EMA value.
        self_spec_threshold: Acceptance rate below which the speculator
            prefers self-speculation over the remote draft.
        activation_window: Number of recent observations to keep per
            draft for trend detection.  0 means unlimited.
    """
    min_candidates: int = 1
    max_candidates: int = 16
    target_acceptance: float = 0.7
    low_acceptance_threshold: float = 0.3
    ema_alpha: float = 0.3
    cooldown_s: float = 5.0
    warmup_observations: int = 5
    self_spec_threshold: float = 0.2
    activation_window: int = 0


class AdaptiveSpeculator:
    """Per-draft-model adaptive candidate count and speculation strategy.

    Usage::

        spec = AdaptiveSpeculator(config=AdaptiveSpecConfig())

        # After each draft inference call:
        spec.record_acceptance("draft-llama-68m", 0.75, num_candidates=8)

        # Before the next call:
        n = spec.get_num_candidates("draft-llama-68m")  # dynamically tuned
        strategy = spec.choose_strategy("draft-llama-68m")
        # -> "remote" if quality is good, "self" if poor
    """

    def __init__(self, config: AdaptiveSpecConfig | None = None):
        self._config = config or AdaptiveSpecConfig()
        self._stats: dict[str, AcceptanceStats] = defaultdict(AcceptanceStats)
        self._candidate_counts: dict[str, int] = defaultdict(
            lambda: max(self._config.min_candidates, 4),
        )
        self._last_adapt: dict[str, float] = defaultdict(float)
        self._lock = threading.Lock()

        logger.info(
            "AdaptiveSpeculator initialized: min={}, max={}, "
            "target_acceptance={}, low_threshold={}",
            self._config.min_candidates,
            self._config.max_candidates,
            self._config.target_acceptance,
            self._config.low_acceptance_threshold,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_acceptance(
        self,
        draft_id: str,
        rate: float,
        num_candidates: int = 0,
    ) -> None:
        """Record an acceptance-rate observation for *draft_id*.

        Args:
            draft_id: Identifier of the draft model.
            rate: Observed acceptance rate (0.0 — 1.0).
            num_candidates: Number of candidates that were speculated.
                Used to update the ``total_speculated`` / ``total_accepted``
                counts when > 0.
        """
        with self._lock:
            stats = self._stats[draft_id]
            stats.update(rate, alpha=self._config.ema_alpha)
            if num_candidates > 0:
                stats.total_speculated += num_candidates
                stats.total_accepted += int(num_candidates * rate)

        self._adapt_candidates(draft_id, rate)

    def get_num_candidates(self, draft_id: str) -> int:
        """Return the dynamically-adjusted candidate count for *draft_id*.

        Returns the minimum count for unknown drafts until enough
        observations have been collected.
        """
        with self._lock:
            return self._candidate_counts.get(
                draft_id, max(self._config.min_candidates, 4),
            )

    def get_stats(self, draft_id: str) -> AcceptanceStats | None:
        """Return the current acceptance stats for *draft_id*.

        Returns ``None`` if the draft has not been observed yet.
        """
        with self._lock:
            raw = self._stats.get(draft_id)
            if raw is None:
                return None
            return AcceptanceStats(
                ema_acceptance=raw.ema_acceptance,
                observation_count=raw.observation_count,
                total_speculated=raw.total_speculated,
                total_accepted=raw.total_accepted,
            )

    def get_all_stats(self) -> dict[str, dict[str, Any]]:
        """Return acceptance stats for all known drafts.

        Returns a dict keyed by ``draft_id`` with summary values.
        """
        with self._lock:
            return {
                did: {
                    "ema_acceptance": round(s.ema_acceptance, 4),
                    "observation_count": s.observation_count,
                    "total_speculated": s.total_speculated,
                    "total_accepted": s.total_accepted,
                    "acceptance_rate": round(s.acceptance_rate, 4),
                    "num_candidates": self._candidate_counts.get(
                        did, self._config.min_candidates,
                    ),
                }
                for did, s in self._stats.items()
            }

    def choose_strategy(self, draft_id: str) -> str:
        """Decide whether to use the remote draft or self-speculation.

        Returns ``"remote"`` if the draft's EMA acceptance is above
        ``self_spec_threshold`` (or if no stats exist yet), otherwise
        ``"self"``.
        """
        with self._lock:
            stats = self._stats.get(draft_id)
            if stats is None:
                return "remote"  # untrusted — give it a chance
            if stats.observation_count < 2:
                return "remote"
            if (
                stats.ema_acceptance >= self._config.self_spec_threshold
                or stats.observation_count < self._config.warmup_observations
            ):
                return "remote"
            return "self"

    def reset(self, draft_id: str) -> None:
        """Reset all tracked state for *draft_id*."""
        with self._lock:
            self._stats.pop(draft_id, None)
            self._candidate_counts.pop(draft_id, None)
            self._last_adapt.pop(draft_id, None)
            logger.debug("Reset adaptive state for draft {!r}", draft_id)

    def reset_all(self) -> None:
        """Reset tracked state for all drafts."""
        with self._lock:
            self._stats.clear()
            self._candidate_counts.clear()
            self._last_adapt.clear()
        logger.debug("Reset all adaptive state")

    # ------------------------------------------------------------------
    # Internal: adaptive candidate tuning
    # ------------------------------------------------------------------

    def _adapt_candidates(self, draft_id: str, rate: float) -> None:
        """Dynamically adjust the candidate count for *draft_id*.

        High acceptance → speculate deeper (more candidates).
        Low acceptance → speculate shallower (less wasted compute).
        """
        now = time.time()
        with self._lock:
            # Cooldown: prevent thrashing.
            if now - self._last_adapt[draft_id] < self._config.cooldown_s:
                return
            self._last_adapt[draft_id] = now

            stats = self._stats.get(draft_id)
            if stats is None:
                return

            # Wait for warmup.
            if stats.observation_count < self._config.warmup_observations:
                return

            current = self._candidate_counts[draft_id]
            ema = stats.ema_acceptance

            if ema >= self._config.target_acceptance:
                # High acceptance: increase candidates exponentially.
                new_count = min(
                    current * 2,
                    self._config.max_candidates,
                )
            elif ema <= self._config.low_acceptance_threshold:
                # Low acceptance: reduce aggressively.
                new_count = max(
                    current // 2,
                    self._config.min_candidates,
                )
            else:
                # Moderate: nudge by ±1.
                if rate > ema:
                    new_count = min(current + 1, self._config.max_candidates)
                else:
                    new_count = max(current - 1, self._config.min_candidates)

            if new_count != current:
                self._candidate_counts[draft_id] = new_count
                logger.debug(
                    "Adjusted candidates for {!r}: {} -> {} "
                    "(ema_acceptance={:.2f})",
                    draft_id, current, new_count, ema,
                )
