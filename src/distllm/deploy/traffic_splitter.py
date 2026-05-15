"""Traffic splitter for canary deployments.

Routes requests between stable and canary versions based on
a configurable percentage, using consistent hashing on request_id
to ensure the same request always routes to the same version.
"""


class TrafficSplitter:
    """Splits traffic between stable and canary versions.

    Uses hash-based routing on request_id for consistency.
    """

    def __init__(
        self,
        stable_version: str = "stable",
        canary_version: str = "canary",
        canary_pct: float = 0.0,
    ):
        self.stable_version = stable_version
        self.canary_version = canary_version
        self.canary_pct = canary_pct  # 0.0 to 100.0

    def set_canary_pct(self, pct: float) -> None:
        """Update the canary traffic percentage."""
        self.canary_pct = max(0.0, min(100.0, pct))

    def select_version(self, request_id: str) -> str:
        """Select which version should handle a request.

        Args:
            request_id: Unique request identifier.

        Returns:
            Version string ("stable" or "canary").
        """
        if self.canary_pct <= 0:
            return self.stable_version
        if self.canary_pct >= 100:
            return self.canary_version

        # Consistent hash-based routing
        hash_val = hash(request_id) % 100
        if hash_val < self.canary_pct:
            return self.canary_version
        return self.stable_version

    def is_active(self) -> bool:
        """Check if canary is currently active."""
        return 0 < self.canary_pct < 100

    def get_distribution_stats(self, request_ids: list) -> dict:
        """Analyze traffic distribution for a set of request IDs.

        Returns:
            Dict with stable_count, canary_count, stable_pct, canary_pct.
        """
        stable_count = 0
        canary_count = 0
        for rid in request_ids:
            version = self.select_version(rid)
            if version == self.canary_version:
                canary_count += 1
            else:
                stable_count += 1

        total = stable_count + canary_count
        return {
            "stable_count": stable_count,
            "canary_count": canary_count,
            "stable_pct": stable_count / total * 100 if total > 0 else 100,
            "canary_pct": canary_count / total * 100 if total > 0 else 0,
        }
