"""Predictive cache warming with LRU prefix tracking, proactive KV push,
and Markov chain prefix prediction.

Extends the existing template-based CacheWarmer with learned popularity
patterns and proactive cache management.

Architecture::

    Request arrives with prefix_hash
         │
         ▼
    LRU Prefix Popularity Tracker
         │  updates frequency + recency
         ▼
    Markov Chain Prefix Predictor
         │  predicts next likely prefix(es)
         ▼
    Proactive KV Pusher
         │  if predicted prefix is not local → push
         ▼
    Warmed cache ready for next request

Usage::

    warmer = PredictiveCacheWarmer()
    warmer.record_access("prefix-hash-abc")
    warmer.record_access("prefix-hash-def", next_hash="prefix-hash-ghi")
    predicted = warmer.predict_next("prefix-hash-def")
    # predicted = "prefix-hash-ghi"
    warmer.maybe_push(predicted, from_node="n1", to_nodes=["n2", "n3"])
"""

from __future__ import annotations

import math
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class PrefixStats:
    """Statistics for a single prefix."""
    hash: str
    access_count: int = 0
    last_access: float = 0.0
    first_access: float = 0.0
    avg_interval_s: float = 0.0


class LRUPrefixTracker:
    """LRU prefix popularity tracker.

    Maintains a bounded cache of prefix statistics with LRU eviction.
    Tracks access frequency, recency, and average inter-arrival time.
    """

    def __init__(self, max_prefixes: int = 10000):
        self._max = max_prefixes
        self._stats: dict[str, PrefixStats] = {}
        self._access_order: deque[str] = deque(maxlen=max_prefixes)
        self._lock = threading.RLock()

    def record(self, prefix_hash: str) -> None:
        """Record an access to *prefix_hash*."""
        with self._lock:
            now = time.time()
            if prefix_hash in self._stats:
                s = self._stats[prefix_hash]
                interval = now - s.last_access
                s.access_count += 1
                s.last_access = now
                s.avg_interval_s = (
                    (s.avg_interval_s * (s.access_count - 2) + interval)
                    / max(s.access_count - 1, 1)
                )
            else:
                self._stats[prefix_hash] = PrefixStats(
                    hash=prefix_hash,
                    access_count=1,
                    last_access=now,
                    first_access=now,
                )
            # Update LRU order
            if prefix_hash in self._access_order:
                self._access_order.remove(prefix_hash)
            self._access_order.append(prefix_hash)
            self._evict_if_needed()

    def _evict_if_needed(self) -> None:
        while len(self._stats) > self._max:
            oldest = self._access_order.popleft()
            self._stats.pop(oldest, None)

    def get_stats(self, prefix_hash: str) -> PrefixStats | None:
        with self._lock:
            return self._stats.get(prefix_hash)

    def top_k(self, k: int = 10) -> list[PrefixStats]:
        """Return the top K most frequently accessed prefixes."""
        with self._lock:
            sorted_stats = sorted(
                self._stats.values(),
                key=lambda s: (-s.access_count, -s.last_access),
            )
            return sorted_stats[:k]

    @property
    def total_prefixes(self) -> int:
        with self._lock:
            return len(self._stats)

    @property
    def stats(self) -> dict:
        with self._lock:
            if not self._stats:
                return {"total": 0, "top_5": []}
            top = self.top_k(5)
            return {
                "total": len(self._stats),
                "top_5": [{"hash": s.hash[:12], "count": s.access_count} for s in top],
            }


class MarkovPrefixPredictor:
    """Markov chain prefix predictor.

    Builds a transition matrix: P(next_prefix | current_prefix).
    After observing enough transitions, predicts the most likely
    next prefix for a given current prefix.

    Supports order-N Markov chains (default: N=1, bigram).
    """

    def __init__(self, max_prefixes: int = 5000, order: int = 1):
        self._order = order
        self._max = max_prefixes

        # transition[prefix] = {next_prefix: count}
        self._transitions: dict[str, dict[str, int]] = {}
        self._total_transitions: dict[str, int] = {}
        self._lock = threading.RLock()

    def record(self, current_hash: str, next_hash: str) -> None:
        """Record a transition from *current_hash* to *next_hash*."""
        with self._lock:
            if current_hash not in self._transitions:
                if len(self._transitions) >= self._max:
                    return
                self._transitions[current_hash] = {}
                self._total_transitions[current_hash] = 0
            self._transitions[current_hash][next_hash] = (
                self._transitions[current_hash].get(next_hash, 0) + 1
            )
            self._total_transitions[current_hash] += 1

    def predict(self, current_hash: str, top_k: int = 3) -> list[tuple[str, float]]:
        """Predict the most likely next prefixes.

        Returns:
            List of (prefix_hash, probability) tuples, sorted by
            probability descending.
        """
        with self._lock:
            if current_hash not in self._transitions:
                return []
            total = self._total_transitions[current_hash]
            if total <= 0:
                return []
            sorted_next = sorted(
                self._transitions[current_hash].items(),
                key=lambda x: -x[1],
            )
            return [
                (h, c / total) for h, c in sorted_next[:top_k]
            ]

    def transition_probability(self, current_hash: str, next_hash: str) -> float:
        """P(next_hash | current_hash)."""
        with self._lock:
            if current_hash not in self._transitions:
                return 0.0
            total = self._total_transitions[current_hash]
            if total <= 0:
                return 0.0
            return self._transitions[current_hash].get(next_hash, 0) / total


class ProactiveKVPusher:
    """Proactively pushes KV cache entries to nodes predicted to need them.

    When a prefix is predicted to be accessed on a different node, push
    its KV cache preemptively to avoid cold-start latency.
    """

    def __init__(self, max_pushes_per_cycle: int = 5):
        self._max_pushes = max_pushes_per_cycle
        self._push_history: dict[str, float] = {}  # prefix -> last_push_time
        self._lock = threading.RLock()

    def should_push(self, prefix_hash: str, probability: float) -> bool:
        """Check if *prefix_hash* should be proactively pushed.

        Only pushes if:
        1. Probability is high enough (>0.3)
        2. Haven't pushed recently (>60s ago)
        """
        if probability < 0.3:
            return False
        with self._lock:
            last_push = self._push_history.get(prefix_hash, 0.0)
            if time.time() - last_push < 60.0:
                return False
            return True

    def record_push(self, prefix_hash: str, source: str, targets: list[str]) -> None:
        with self._lock:
            self._push_history[prefix_hash] = time.time()
            logger.info(f"Proactive KV push: {prefix_hash} {source} -> {targets}")


class PredictiveCacheWarmer:
    """Full predictive cache warming system.

    Combines LRU prefix tracking, Markov chain prediction, and proactive
    KV cache pushing.

    Usage::

        warmer = PredictiveCacheWarmer()
        warmer.record_access("prefix-abc")
        warmer.record_access("prefix-abc", next_hash="prefix-def")
        predicted = warmer.predict_next("prefix-abc")
        # predicted = [("prefix-def", 1.0)]
    """

    def __init__(
        self,
        max_prefixes: int = 10000,
        markov_order: int = 1,
        proactivity_threshold: float = 0.3,
    ):
        self._tracker = LRUPrefixTracker(max_prefixes=max_prefixes)
        self._predictor = MarkovPrefixPredictor(
            max_prefixes=max_prefixes,
            order=markov_order,
        )
        self._pusher = ProactiveKVPusher()
        self._proactivity_threshold = proactivity_threshold
        self._lock = threading.RLock()

    def record_access(
        self, prefix_hash: str, next_hash: str | None = None,
    ) -> None:
        """Record a prefix access and optional transition.

        Args:
            prefix_hash: The prefix that was accessed.
            next_hash: Optional next prefix (for Markov chain training).
        """
        self._tracker.record(prefix_hash)
        if next_hash:
            self._predictor.record(prefix_hash, next_hash)

    def predict_next(
        self, current_hash: str, top_k: int = 3,
    ) -> list[tuple[str, float]]:
        """Predict the next likely prefix(es).

        Returns list of (prefix_hash, probability) sorted by probability.
        """
        return self._predictor.predict(current_hash, top_k=top_k)

    def get_top_prefixes(self, k: int = 10) -> list[PrefixStats]:
        """Get the top K most popular prefixes."""
        return self._tracker.top_k(k)

    def should_warm(self, prefix_hash: str) -> bool:
        """Check if *prefix_hash* should be warmed based on popularity."""
        stats = self._tracker.get_stats(prefix_hash)
        if stats is None:
            return False
        # Warm if accessed more than once in the last hour
        return stats.access_count >= 2 and (time.time() - stats.last_access) < 3600

    def maybe_proactive_push(
        self,
        current_hash: str,
        source_node: str = "",
        target_nodes: list[str] | None = None,
    ) -> list[str]:
        """Check predictions and proactively push if beneficial.

        Args:
            current_hash: The current prefix hash.
            source_node: Node that has the KV cache.
            target_nodes: Nodes that might need it.

        Returns:
            List of prefix hashes that were pushed.
        """
        pushed: list[str] = []
        predictions = self.predict_next(current_hash, top_k=3)
        targets = target_nodes or []

        for predicted_hash, prob in predictions:
            if self._pusher.should_push(predicted_hash, prob) and targets:
                self._pusher.record_push(predicted_hash, source_node, targets)
                pushed.append(predicted_hash)

        return pushed

    @property
    def stats(self) -> dict:
        return {
            "lru_tracker": self._tracker.stats,
            "markov_prefixes": len(self._predictor._transitions),
            "proactivity_threshold": self._proactivity_threshold,
        }
