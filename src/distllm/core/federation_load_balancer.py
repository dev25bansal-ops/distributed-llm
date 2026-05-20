"""Federation load balancer for cross-cluster load reporting.

Collects load metrics from remote clusters via heartbeat and
integrates with the existing LoadReporter in geo_router.py.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class RemoteClusterLoad:
    """Load metrics from a remote cluster."""
    cluster_id: str
    active_requests: int = 0
    pending_requests: int = 0
    gpu_utilization: float = 0.0
    queue_depth: int = 0
    tokens_per_sec: float = 0.0
    last_report: float = 0.0
    stale: bool = False

    @property
    def is_overloaded(self) -> bool:
        """Check if the cluster is overloaded (>85% GPU or queue >50)."""
        return self.gpu_utilization > 85.0 or self.queue_depth > 50

    @property
    def available_capacity(self) -> float:
        """Estimated available capacity (0-100)."""
        if self.is_overloaded:
            return 0.0
        return max(0.0, 100.0 - self.gpu_utilization)

    def is_stale(self, threshold_s: float = 30.0) -> bool:
        """Check if the load report is stale."""
        return (time.time() - self.last_report) > threshold_s


class FederationLoadBalancer:
    """Manages load reporting across federation peers.

    Remote coordinators periodically send load reports via heartbeat.
    This class aggregates and exposes load data for routing decisions.
    Integrates with GeoRouter.LoadReporter for unified load-aware routing.
    """

    def __init__(
        self,
        stale_threshold_s: float = 30.0,
        ema_alpha: float = 0.3,
    ) -> None:
        self.stale_threshold_s = stale_threshold_s
        self.ema_alpha = ema_alpha
        self._loads: dict[str, RemoteClusterLoad] = {}

    def report_load(
        self,
        cluster_id: str,
        active_requests: int,
        pending_requests: int,
        gpu_utilization: float,
        queue_depth: int,
        tokens_per_sec: float = 0.0,
    ) -> None:
        """Update load metrics for a remote cluster.

        Uses EMA smoothing for stable metrics.

        Args:
            cluster_id: Target cluster identifier.
            active_requests: Currently processing requests.
            pending_requests: Queued requests.
            gpu_utilization: Average GPU utilization (0-100).
            queue_depth: Number of requests in queue.
            tokens_per_sec: Current throughput.
        """
        now = time.time()

        if cluster_id not in self._loads:
            self._loads[cluster_id] = RemoteClusterLoad(
                cluster_id=cluster_id,
                active_requests=active_requests,
                pending_requests=pending_requests,
                gpu_utilization=gpu_utilization,
                queue_depth=queue_depth,
                tokens_per_sec=tokens_per_sec,
                last_report=now,
            )
            return

        load = self._loads[cluster_id]
        alpha = self.ema_alpha

        # EMA smoothing
        load.active_requests = int(alpha * active_requests + (1 - alpha) * load.active_requests)
        load.pending_requests = int(alpha * pending_requests + (1 - alpha) * load.pending_requests)
        load.gpu_utilization = alpha * gpu_utilization + (1 - alpha) * load.gpu_utilization
        load.queue_depth = int(alpha * queue_depth + (1 - alpha) * load.queue_depth)
        load.tokens_per_sec = alpha * tokens_per_sec + (1 - alpha) * load.tokens_per_sec
        load.last_report = now
        load.stale = False

    def get_remote_load(self, cluster_id: str) -> RemoteClusterLoad | None:
        """Get the latest load metrics for a cluster."""
        load = self._loads.get(cluster_id)
        if load and load.is_stale(self.stale_threshold_s):
            load.stale = True
            logger.debug(f"Load report for cluster {cluster_id} is stale")
        return load

    def get_all_loads(self) -> dict[str, RemoteClusterLoad]:
        """Get load metrics for all known clusters."""
        # Update stale flags
        for load in self._loads.values():
            load.stale = load.is_stale(self.stale_threshold_s)
        return dict(self._loads)

    def get_best_cluster(self, cluster_ids: list[str]) -> str | None:
        """Select the best cluster from a list based on load.

        Prefers clusters with lowest GPU utilization and queue depth.
        Skips stale or overloaded clusters.

        Args:
            cluster_ids: List of candidate cluster IDs.

        Returns:
            Best cluster ID, or None if all are overloaded/stale.
        """
        candidates = []
        for cid in cluster_ids:
            load = self._loads.get(cid)
            if load is None:
                # Unknown cluster, treat as available
                candidates.append((cid, 0.0))
            elif load.is_stale(self.stale_threshold_s):
                continue  # Skip stale
            elif load.is_overloaded:
                continue  # Skip overloaded
            else:
                # Score: lower is better
                score = load.gpu_utilization + load.queue_depth
                candidates.append((cid, score))

        if not candidates:
            return None

        return min(candidates, key=lambda x: x[1])[0]

    def remove_cluster(self, cluster_id: str) -> None:
        """Remove a cluster from load tracking."""
        self._loads.pop(cluster_id, None)
        logger.info(f"Removed cluster {cluster_id} from load balancer")

    def to_dict(self) -> dict[str, Any]:
        """Export load data for the GeoRouter LoadReporter."""
        return {
            cid: {
                "active": load.active_requests,
                "pending": load.pending_requests,
                "gpu_util": load.gpu_utilization,
                "queue_depth": load.queue_depth,
                "tokens_per_sec": load.tokens_per_sec,
                "stale": load.stale,
            }
            for cid, load in self._loads.items()
        }
