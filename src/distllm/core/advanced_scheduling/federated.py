"""Federated scheduling across multiple clusters."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ClusterStatus:
    """Status of a federated cluster."""
    cluster_id: str
    active_requests: int = 0
    pending_requests: int = 0
    gpu_utilization: float = 0.0
    last_heartbeat: float = field(default_factory=time.time)
    is_healthy: bool = True


@dataclass
class FederatedRoute:
    """A routing decision for federated inference."""
    target_cluster: str
    reason: str
    estimated_latency_ms: float = 0.0
    estimated_cost: float = 0.0


class FederatedScheduler:
    """Routes requests across federated clusters."""

    def __init__(self, spillover_threshold: float = 80.0):
        self._clusters: dict[str, ClusterStatus] = {}
        self._spillover_threshold = spillover_threshold
        self._lock = threading.Lock()

    def update_cluster(self, status: ClusterStatus) -> None:
        with self._lock:
            self._clusters[status.cluster_id] = status

    def should_spillover(self, local_utilization: float) -> bool:
        return local_utilization > self._spillover_threshold

    def select_target(self, exclude: str = "") -> FederatedRoute | None:
        with self._lock:
            candidates = [
                c for c in self._clusters.values()
                if c.cluster_id != exclude and c.is_healthy
            ]
            if not candidates:
                return None
            best = min(candidates, key=lambda c: c.gpu_utilization)
            return FederatedRoute(
                target_cluster=best.cluster_id,
                reason="lowest_utilization",
                estimated_latency_ms=50.0,
            )
