"""RDP (Renyi Differential Privacy) accounting for privacy budget tracking.

Provides the RDPAccounting class which tracks cumulative privacy spend
across multiple queries using Renyi divergence composition, providing
tighter bounds than standard composition theorems.

Reference:
    Mironov, "Renyi Differential Privacy", 2017.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class RDPAccounting:
    """Renyi Differential Privacy (RDP) accountant.

    Tracks cumulative privacy spend across multiple queries using Renyi
    divergence composition, which provides tighter bounds than standard
    composition theorems.

    Reference:
        Mironov, "Renyi Differential Privacy", 2017.
    """

    orders: list[float] = field(default_factory=lambda: [])

    def __post_init__(self) -> None:
        if not self.orders:
            self.orders = [
                0.1,
                0.2,
                0.5,
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
                10.0,
                20.0,
                50.0,
                100.0,
                500.0,
                1000.0,
            ]
        self._rdp_per_query: dict[int, list[float]] = {}
        self._total_rdp: list[float] = [0.0] * len(self.orders)

    def compute_rdp(
        self,
        sigma: float,
        num_queries: int = 1,
    ) -> list[float]:
        """Compute per-query RDP for a Gaussian mechanism with given *sigma*.

        For subsampled Gaussian mechanism (sampling rate q),
        the RDP is approximately:
            RDP(alpha) <= (q^2 * alpha) / sigma^2   for alpha > 1

        For non-subsampled (q = 1):
            RDP(alpha) = alpha / (2 * sigma^2)

        Args:
            sigma: Noise scale (standard deviation).
            num_queries: Number of queries (steps) to compose.

        Returns:
            List of RDP values for each alpha order.
        """
        if sigma <= 0:
            return [0.0] * len(self.orders)

        rdp_values: list[float] = []
        for alpha in self.orders:
            if alpha == 0:
                rdp_values.append(0.0)
            else:
                # Non-subsampled Gaussian mechanism RDP
                rdp_alpha = alpha / (2.0 * sigma * sigma)
                rdp_values.append(rdp_alpha * num_queries)
        return rdp_values

    def add_query(self, sigma: float, query_id: int | None = None) -> None:
        """Record one or more queries into the accountant.

        Args:
            sigma: Noise scale used for the query.
            query_id: Optional identifier for the query.
        """
        delta_rdp = self.compute_rdp(sigma, num_queries=1)
        if query_id is not None:
            self._rdp_per_query[query_id] = delta_rdp
        self._total_rdp = [
            t + d for t, d in zip(self._total_rdp, delta_rdp)
        ]

    def get_epsilon(self, delta: float) -> float:
        """Convert the accumulated RDP to (epsilon, delta)-DP.

        Uses the standard conversion:
            epsilon = min_alpha ( RDP(alpha) - log(delta) / (alpha - 1) )

        Args:
            delta: Target delta for the conversion.

        Returns:
            The smallest epsilon achievable for the given delta.
        """
        if not self._total_rdp or not any(self._total_rdp):
            # No queries recorded: zero privacy spent.  (Skipping this makes
            # the log(1/delta)/(alpha-1) term surface as a phantom ~0.01
            # epsilon for an accountant that never saw a query.)
            return 0.0

        best_epsilon = float("inf")
        for alpha, rdp_alpha in zip(self.orders, self._total_rdp):
            if alpha <= 1:
                continue
            eps = rdp_alpha - math.log(delta) / (alpha - 1)
            if eps < best_epsilon:
                best_epsilon = eps
        return max(0.0, best_epsilon)

    def get_privacy_spent(self, delta: float) -> dict[str, float]:
        """Return a summary of total privacy spend.

        Args:
            delta: Target delta for epsilon conversion.

        Returns:
            Dict with 'epsilon', 'delta', and 'orders_used'.
        """
        return {
            "epsilon": round(self.get_epsilon(delta), 4),
            "delta": delta,
            "orders_used": len(self.orders),
        }
