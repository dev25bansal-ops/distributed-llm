"""Latency-aware router using EWMA-based RTT tracking.

Selects the candidate node with the lowest exponentially-weighted
moving-average round-trip time.
"""

from __future__ import annotations

import sys
from typing import Final

_ALPHA: Final[float] = 0.3


class LatencyAwareRouter:
    """Tracks per-node RTT via an EWMA filter and selects the lowest-latency
    candidate.

    The smoothed RTT is updated on each ``update_latency`` call:

        ewma = alpha * rtt + (1 - alpha) * ewma

    Parameters
    ----------
    alpha :
        Decay factor for the EWMA filter.  Defaults to 0.3 (weighs recent
        samples more heavily).
    """

    def __init__(self, alpha: float = _ALPHA) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")

        self._alpha = alpha
        # node_id -> smoothed RTT in milliseconds (None = no data yet).
        self._latencies: dict[str, float | None] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_node(self, candidates: list[str]) -> str:
        """Return the candidate with the lowest EWMA RTT.

        Candidates with no recorded RTT are ranked first (treated as
        unknown / potentially fast).  If multiple candidates share the
        same latency the first encountered in *candidates* order is
        returned.

        Raises ``ValueError`` when *candidates* is empty.
        """
        if not candidates:
            raise ValueError("candidates list must not be empty")

        best = candidates[0]
        best_latency = self._latencies.get(best)  # None if unknown

        for node_id in candidates[1:]:
            lat = self._latencies.get(node_id)
            # Prefer nodes with no data, then lowest known latency.
            if best_latency is not None and (
                lat is None or lat < best_latency
            ):
                best = node_id
                best_latency = lat

        return best

    def update_latency(self, node_id: str, rtt_ms: float) -> None:
        """Update the EWMA-smoothed RTT for *node_id* with a new sample.

        The first sample initialises the EWMA directly; subsequent
        samples are blended with the current estimate.
        """
        if rtt_ms < 0:
            raise ValueError(f"RTT must be non-negative, got {rtt_ms}")

        current = self._latencies.get(node_id)
        if current is None:
            self._latencies[node_id] = rtt_ms
        else:
            self._latencies[node_id] = (
                self._alpha * rtt_ms + (1.0 - self._alpha) * current
            )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def alpha(self) -> float:
        """Return the EWMA decay factor."""
        return self._alpha

    @property
    def latencies(self) -> dict[str, float | None]:
        """Return a copy of the node -> smoothed-RTT map."""
        return dict(self._latencies)

    @property
    def size(self) -> int:
        """Return the number of tracked nodes."""
        return len(self._latencies)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(alpha={self._alpha}, "
            f"nodes={self.size})"
        )


# Keep ``__all__`` in sync with the public class.
if sys.version_info >= (3, 9):
    __all__ = ("LatencyAwareRouter",)
else:
    __all__ = ["LatencyAwareRouter"]
