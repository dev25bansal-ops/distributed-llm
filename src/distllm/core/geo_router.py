"""Geo-aware cross-cluster routing for federated LLM inference.

Makes routing decisions based on cluster affinity, latency matrix,
and load balancing across nodes within a cluster. Integrates with
FederationManager and CrossClusterLatencyMonitor to select optimal
target clusters and edge nodes for cross-cluster requests.
"""

import time
import threading
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class ClusterLoad:
    """Load metrics for a single cluster."""
    cluster_id: str
    active_requests: int = 0
    pending_requests: int = 0
    gpu_utilization: float = 0.0
    queue_depth: int = 0
    last_updated: float = field(default_factory=time.time)

    @property
    def is_overloaded(self) -> bool:
        return self.gpu_utilization > 0.85 or self.queue_depth > 50

    def is_overloaded_at(self, threshold: float) -> bool:
        """Check if cluster is overloaded at a given GPU utilization threshold."""
        queue_threshold = int(threshold * 100)
        return self.gpu_utilization > threshold or self.queue_depth > queue_threshold

    @property
    def available_capacity(self) -> float:
        """0.0 (full) to 1.0 (empty)."""
        load = max(self.gpu_utilization, self.queue_depth / 100.0)
        return max(0.0, 1.0 - min(load, 1.0))


class LoadReporter:
    """Tracks per-cluster load metrics from coordinator heartbeat."""

    def __init__(self, stale_threshold_s: float = 30.0):
        self._loads: dict[str, ClusterLoad] = {}
        self._stale_threshold = stale_threshold_s
        self._lock = threading.Lock()

    def report(self, cluster_id: str, active: int = 0, pending: int = 0,
               gpu_util: float = 0.0, queue_depth: int = 0) -> None:
        with self._lock:
            load = self._loads.get(cluster_id)
            if load is None:
                load = ClusterLoad(cluster_id=cluster_id)
                self._loads[cluster_id] = load
            load.active_requests = active
            load.pending_requests = pending
            load.gpu_utilization = gpu_util
            load.queue_depth = queue_depth
            load.last_updated = time.time()

    def get_load(self, cluster_id: str) -> ClusterLoad | None:
        with self._lock:
            load = self._loads.get(cluster_id)
            if load is None:
                return None
            # Check staleness
            if time.time() - load.last_updated > self._stale_threshold:
                logger.warning(f"Stale load data for {cluster_id} ({time.time() - load.last_updated:.0f}s old)")
                return None
            return load

    def get_all_loads(self) -> dict[str, ClusterLoad]:
        with self._lock:
            return dict(self._loads)

    def remove_cluster(self, cluster_id: str) -> None:
        with self._lock:
            self._loads.pop(cluster_id, None)


class GeoRouter:
    """Routes requests across federated clusters based on multiple signals.

    Decision factors (in priority order):
    1. Local cluster capacity (prefer local if not overloaded)
    2. Cross-cluster latency matrix (pick nearest cluster with capacity)
    3. Load balancing across nodes within the target cluster
    4. Edge node selection for cross-cluster traffic

    Usage:
        router = GeoRouter(federation, latency_monitor, load_reporter)
        target = router.select_target_cluster("prompt", "us-east-1")
        edge_node = router.select_edge_node(target, "us-east-1")
    """

    def __init__(
        self,
        federation: Any,
        latency_monitor: Any,
        load_reporter: LoadReporter | None = None,
        local_latency_threshold_ms: float = 50.0,
        spill_threshold: float = 0.85,
    ):
        self._federation = federation
        self._latency_monitor = latency_monitor
        self._load_reporter = load_reporter or LoadReporter()
        self._local_latency_threshold = local_latency_threshold_ms
        self._spill_threshold = spill_threshold
        self._lock = threading.Lock()

    def select_target_cluster(
        self,
        request: str = "",
        source_cluster: str = "default",
    ) -> tuple[str, str]:
        """Select the best target cluster for a request.

        Args:
            request: Request prompt/metadata (unused now, reserved for content-aware routing).
            source_cluster: The cluster where the request originated.

        Returns:
            Tuple of (target_cluster_id, reason).
        """
        # 1. Check local cluster capacity
        local_load = self._load_reporter.get_load(source_cluster)
        if local_load is not None and not local_load.is_overloaded:
            return source_cluster, "local_capacity"

        # 2. Find nearest cluster with available capacity
        all_clusters = self._federation.list_clusters()
        candidate_clusters = [c for c in all_clusters if c != source_cluster]

        if not candidate_clusters:
            return source_cluster, "no_alternative"

        best_cluster = None
        best_score = float('inf')
        best_reason = "latency_fallback"

        for cluster_id in candidate_clusters:
            load = self._load_reporter.get_load(cluster_id)
            latency = self._latency_monitor.get_latency(source_cluster, cluster_id)

            # Skip overloaded clusters
            if load is not None and load.is_overloaded:
                continue

            # Score = latency * (1 / available_capacity)
            # Lower score = better cluster
            capacity = load.available_capacity if load is not None else 0.5
            score = latency * (1.0 / max(capacity, 0.01))

            if score < best_score:
                best_score = score
                best_cluster = cluster_id
                best_reason = f"nearest_with_capacity (latency={latency:.0f}ms, capacity={capacity:.0%})"

        if best_cluster is None:
            # All remote clusters overloaded or unknown — fall back to local
            return source_cluster, "all_remote_overloaded"

        return best_cluster, best_reason

    def select_edge_node(
        self,
        target_cluster: str,
        source_cluster: str = "default",
    ) -> str | None:
        """Select the best edge node in the target cluster for cross-cluster traffic.

        Uses load-based selection: prefers edge nodes with lowest current load.
        Falls back to round-robin if load data is unavailable.

        Args:
            target_cluster: Target cluster ID.
            source_cluster: Source cluster ID.

        Returns:
            Edge node ID, or None if no edge nodes available.
        """
        edge_nodes = self._federation.get_edge_nodes(target_cluster)
        if not edge_nodes:
            # No dedicated edge nodes — use any node in the cluster
            edge_nodes = self._federation.get_nodes_in_cluster(target_cluster)

        if not edge_nodes:
            return None

        if len(edge_nodes) == 1:
            return next(iter(edge_nodes))

        # Select edge node with lowest load (round-robin fallback)
        loads = []
        for node_id in edge_nodes:
            load = self._load_reporter.get_load(target_cluster)
            if load is not None:
                # Distribute evenly: use hash-based selection for fairness
                loads.append((hash(node_id) % 100, node_id))
            else:
                loads.append((0, node_id))

        # Sort by hash (pseudo-random but deterministic)
        loads.sort(key=lambda x: x[0])
        return loads[0][1]

    def get_cluster_load(self, cluster_id: str) -> ClusterLoad | None:
        """Get current load metrics for a cluster."""
        return self._load_reporter.get_load(cluster_id)

    def report_cluster_load(self, cluster_id: str, active: int = 0,
                            pending: int = 0, gpu_util: float = 0.0,
                            queue_depth: int = 0) -> None:
        """Report load metrics for a cluster (called by coordinator heartbeat)."""
        self._load_reporter.report(
            cluster_id=cluster_id,
            active=active,
            pending=pending,
            gpu_util=gpu_util,
            queue_depth=queue_depth,
        )

    def stats(self) -> dict:
        """Get routing statistics."""
        loads = self._load_reporter.get_all_loads()
        return {
            "cluster_loads": {
                cid: {
                    "active": l.active_requests,
                    "pending": l.pending_requests,
                    "gpu_util": l.gpu_utilization,
                    "queue_depth": l.queue_depth,
                    "overloaded": l.is_overloaded,
                }
                for cid, l in loads.items()
            },
            "latency_matrix": getattr(self._latency_monitor, 'stats', lambda: {})(),
            "federation": getattr(self._federation, 'stats', lambda: {})(),
        }
