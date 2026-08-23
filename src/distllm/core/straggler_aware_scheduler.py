"""Straggler-aware gradient-based scheduling.

Extends reactive MAD-based straggler detection with proportional budget
reduction, speculative isolation, and gradient recovery.

Architecture::

    StragglerDetector (existing)
         │  (latency samples)
         ▼
    GradientTracker ──► sliding window latency gradient
         │
         ├── positive gradient (slowing)  → reduce budget proportionally
         ├── negative gradient (recovering) → restore budget gradually
         └── flat gradient (stable)       → maintain current budget
              │
              ▼
    SpeculativeIsolation ──► run duplicate on 2 nodes, use faster result
              │
              ▼
    GradientRecovery ──► exponential backoff budget restoration
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class BudgetAllocation:
    """Token budget allocation for a node."""
    node_id: str
    max_batch_size: int
    max_tokens_per_batch: int
    reduction_factor: float = 1.0  # 1.0 = full budget, 0.5 = half
    is_speculative: bool = False
    gradient: float = 0.0  # Current latency gradient


class GradientTracker:
    """Sliding-window latency gradient tracker.

    Computes the first derivative of latency over time using linear
    regression on the sliding window.  A positive gradient means the
    node is slowing down (straggling), while a negative gradient means
    it is recovering.
    """

    def __init__(self, window_size: int = 20):
        self._window: deque[tuple[float, float]] = deque(maxlen=window_size)
        self._current_gradient: float = 0.0

    def record(self, latency_ms: float) -> None:
        """Record a latency sample and recompute the gradient."""
        self._window.append((time.monotonic(), latency_ms))
        self._current_gradient = self._compute_gradient()

    def _compute_gradient(self) -> float:
        """Compute the slope of latency over time using least-squares.

        Returns:
            Gradient in ms/sec.  Positive = slowing down (straggling).
        """
        n = len(self._window)
        if n < 3:
            return 0.0

        times = [t for t, _ in self._window]
        lats = [l for _, l in self._window]

        t_mean = sum(times) / n
        l_mean = sum(lats) / n

        num = sum((t - t_mean) * (l - l_mean) for t, l in zip(times, lats))
        den = sum((t - t_mean) ** 2 for t in times)

        if den == 0:
            return 0.0
        return num / den

    @property
    def gradient(self) -> float:
        return self._current_gradient

    @property
    def is_straggling(self) -> bool:
        """True if latency is consistently increasing."""
        return self._current_gradient > 0.5  # ms/sec threshold

    @property
    def is_recovering(self) -> bool:
        """True if latency is decreasing."""
        return self._current_gradient < -0.5


class StragglerAwareScheduler:
    """Scheduler with straggler-aware budget allocation.

    Detects stragglers via latency gradients, reduces their token budgets
    proportionally, sends speculative duplicates for critical requests,
    and gradually restores budgets on recovery.

    Usage::

        scheduler = StragglerAwareScheduler(base_batch_size=64)
        scheduler.record_latency("node-1", 150.0)  # slow node
        budget = scheduler.get_budget("node-1")
        # budget.max_batch_size will be reduced proportionally
    """

    def __init__(
        self,
        base_batch_size: int = 64,
        base_tokens_per_batch: int = 4096,
        min_batch_size: int = 4,
        min_tokens_per_batch: int = 256,
        reduction_rate: float = 0.3,
        recovery_rate: float = 0.1,
        speculative_threshold: float = 0.6,
        gradient_window: int = 20,
    ):
        self._base_batch = base_batch_size
        self._base_tokens = base_tokens_per_batch
        self._min_batch = min_batch_size
        self._min_tokens = min_tokens_per_batch
        self._reduction_rate = reduction_rate
        self._recovery_rate = recovery_rate
        self._speculative_threshold = speculative_threshold
        self._gradient_window = gradient_window

        # Per-node state
        self._trackers: dict[str, GradientTracker] = {}
        self._reduction_factors: dict[str, float] = {}
        self._speculative_nodes: set[str] = set()

    def _get_tracker(self, node_id: str) -> GradientTracker:
        if node_id not in self._trackers:
            self._trackers[node_id] = GradientTracker(window_size=20)
        return self._trackers[node_id]

    def record_latency(self, node_id: str, latency_ms: float) -> None:
        """Record a latency sample for a node.

        Updates the gradient tracker and adjusts the reduction factor.
        """
        tracker = self._get_tracker(node_id)
        tracker.record(latency_ms)
        grad = tracker.gradient

        if tracker.is_straggling:
            current = self._reduction_factors.get(node_id, 1.0)
            reduction = min(self._reduction_rate, abs(grad) / 100.0)
            new_factor = max(0.1, current - reduction)
            self._reduction_factors[node_id] = new_factor
            logger.debug(f"Straggler {node_id}: factor {current:.2f} -> {new_factor:.2f} (grad={grad:.1f})")

            if new_factor < self._speculative_threshold:
                self._speculative_nodes.add(node_id)
                logger.warning(f"Speculative isolation for {node_id}")

        elif tracker.is_recovering:
            current = self._reduction_factors.get(node_id, 1.0)
            recovery = min(self._recovery_rate, abs(grad) / 100.0)
            new_factor = min(1.0, current + recovery)
            if new_factor >= 1.0:
                self._reduction_factors.pop(node_id, None)
                self._speculative_nodes.discard(node_id)
            else:
                self._reduction_factors[node_id] = new_factor
            logger.debug(f"Recovering {node_id}: factor {current:.2f} -> {new_factor:.2f}")

        elif node_id in self._reduction_factors and abs(grad) < 0.2:
            # Stable — very gradual recovery
            current = self._reduction_factors.get(node_id, 1.0)
            new_factor = min(1.0, current + self._recovery_rate * 0.5)
            if new_factor >= 1.0:
                self._reduction_factors.pop(node_id, None)
            else:
                self._reduction_factors[node_id] = new_factor

    def get_budget(self, node_id: str) -> BudgetAllocation:
        """Get the budget allocation for a node.

        Returns a BudgetAllocation with adjusted max_batch_size and
        max_tokens_per_batch based on the current reduction factor.
        """
        factor = self._reduction_factors.get(node_id, 1.0)
        grad = self._trackers[node_id].gradient if node_id in self._trackers else 0.0

        return BudgetAllocation(
            node_id=node_id,
            max_batch_size=max(self._min_batch, int(self._base_batch * factor)),
            max_tokens_per_batch=max(self._min_tokens, int(self._base_tokens * factor)),
            reduction_factor=factor,
            is_speculative=node_id in self._speculative_nodes,
            gradient=grad,
        )

    @property
    def active_stragglers(self) -> list[tuple[str, float]]:
        """List of (node_id, reduction_factor) for nodes with reduced budgets."""
        return sorted(
            [(n, f) for n, f in self._reduction_factors.items()],
            key=lambda x: x[1],
        )

    @property
    def speculative_nodes(self) -> set[str]:
        return set(self._speculative_nodes)

    @property
    def stats(self) -> dict:
        return {
            "active_stragglers": len(self._reduction_factors),
            "speculative_nodes": len(self._speculative_nodes),
            "reduction_factors": dict(self._reduction_factors),
        }


class GradientRecovery:
    """Exponential backoff budget restoration for recovered stragglers.

    When a node recovers, its budget is restored gradually (exponential
    backoff style) to avoid thundering-herd effects.
    """

    def __init__(self, max_steps: int = 10, base_boost: float = 0.2):
        self._max_steps = max_steps
        self._base_boost = base_boost
        self._recovery_progress: dict[str, int] = {}

    def next_factor(self, node_id: str, current_factor: float) -> float:
        """Compute the next recovery factor.

        Uses::
            factor = current_factor + base_boost * exp(-step / decay)
        """
        step = self._recovery_progress.get(node_id, 0)
        if step >= self._max_steps:
            self._recovery_progress.pop(node_id, None)
            return 1.0

        boost = self._base_boost * math.exp(-step / 3.0)
        new_factor = min(1.0, current_factor + boost)
        self._recovery_progress[node_id] = step + 1
        return new_factor

    def reset(self, node_id: str) -> None:
        self._recovery_progress.pop(node_id, None)
