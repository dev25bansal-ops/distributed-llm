"""Composite router combining consistent-hash, latency, and load signals.

Each candidate node is scored as a weighted sum of three sub-scores:

    score = w1 * hash_score + w2 * (1 - norm_latency)
          + w3 * (1 - norm_load)

The node with the **highest** score is selected.
"""

from __future__ import annotations

import sys
from typing import Final

from .consistent_hash import ConsistentHashRouter
from .latency_aware import LatencyAwareRouter
from .load_aware import LoadAwareRouter

_DEFAULT_W1: Final[float] = 0.4
_DEFAULT_W2: Final[float] = 0.3
_DEFAULT_W3: Final[float] = 0.3


class CompositeRouter:
    """Combines consistent-hash affinity, EWMA latency, and active-load
    signals into a single routing decision.

    Parameters
    ----------
    hash_router :
        Pre-configured ``ConsistentHashRouter`` instance.
    latency_router :
        Pre-configured ``LatencyAwareRouter`` instance.
    load_router :
        Pre-configured ``LoadAwareRouter`` instance.
    w1 :
        Weight for the consistent-hash score (default 0.4).
    w2 :
        Weight for the inverse-latency score (default 0.3).
    w3 :
        Weight for the inverse-load score (default 0.3).
    """

    def __init__(
        self,
        hash_router: ConsistentHashRouter,
        latency_router: LatencyAwareRouter,
        load_router: LoadAwareRouter,
        w1: float = _DEFAULT_W1,
        w2: float = _DEFAULT_W2,
        w3: float = _DEFAULT_W3,
    ) -> None:
        self._hash = hash_router
        self._latency = latency_router
        self._load = load_router
        self._validate_weights(w1, w2, w3)
        self._w1 = w1
        self._w2 = w2
        self._w3 = w3

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_node(self, key: str, candidates: list[str]) -> str:
        """Score each candidate and return the best node for *key*.

        The consistent-hash sub-score is ``1.0`` if *candidates* contains
        the primary hash node for *key*, else ``0.0``.  Latency and load
        are normalised to ``[0, 1]`` across the candidate set before
        weighting.

        Raises ``ValueError`` when *candidates* is empty.
        """
        if not candidates:
            raise ValueError("candidates list must not be empty")
        if len(candidates) == 1:
            return candidates[0]

        # --- Sub-scores ---------------------------------------------------

        # 1. Consistent-hash affinity.
        try:
            primary = self._hash.get_node(key)
        except RuntimeError:
            primary = None
        hash_scores = {
            nid: (1.0 if nid == primary else 0.0) for nid in candidates
        }

        # 2. Normalised inverse latency.
        raw_lat = {
            nid: self._latency.latencies.get(nid)
            for nid in candidates
        }
        known = [v for v in raw_lat.values() if v is not None]
        lat_max = max(known) if known else 0.0
        lat_min = min(known) if known else 0.0
        lat_range = lat_max - lat_min
        lat_scores: dict[str, float] = {}
        for nid in candidates:
            v = raw_lat[nid]
            if v is None:
                lat_scores[nid] = 0.5  # neutral for unknown nodes
            elif lat_range == 0.0:
                lat_scores[nid] = 0.5  # all known values equal
            else:
                lat_scores[nid] = 1.0 - (v - lat_min) / lat_range

        # 3. Normalised inverse load.
        raw_load = {
            nid: self._load.get_load(nid) for nid in candidates
        }
        load_max = max(raw_load.values()) if raw_load else 0
        load_scores: dict[str, float] = {}
        for nid in candidates:
            if load_max == 0:
                load_scores[nid] = 0.5  # all zero → neutral
            else:
                load_scores[nid] = 1.0 - raw_load[nid] / load_max

        # --- Weighted sum -------------------------------------------------

        best = candidates[0]
        best_score: float = -1.0

        for nid in candidates:
            score = (
                self._w1 * hash_scores[nid]
                + self._w2 * lat_scores[nid]
                + self._w3 * load_scores[nid]
            )
            if score > best_score:
                best_score = score
                best = nid

        return best

    def set_weights(self, w1: float, w2: float, w3: float) -> None:
        """Update the three scoring weights."""
        self._validate_weights(w1, w2, w3)
        self._w1 = w1
        self._w2 = w2
        self._w3 = w3

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_weights(w1: float, w2: float, w3: float) -> None:
        """Ensure weights are non-negative (they need not sum to 1)."""
        for name, val in (("w1", w1), ("w2", w2), ("w3", w3)):
            if val < 0:
                raise ValueError(
                    f"Weight {name} must be non-negative, got {val}"
                )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def weights(self) -> tuple[float, float, float]:
        """Return the current weight triple ``(w1, w2, w3)``."""
        return (self._w1, self._w2, self._w3)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(w1={self._w1}, w2={self._w2}, "
            f"w3={self._w3})"
        )


if sys.version_info >= (3, 9):
    __all__ = ("CompositeRouter",)
else:
    __all__ = ["CompositeRouter"]
