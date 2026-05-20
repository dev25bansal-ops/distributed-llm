"""Multi-cluster router: routes user sessions to the best cluster + coordinator.

Uses a two-level routing strategy:

1. **Cluster-level** (which cluster?):
   - *Affinity ring*: consistent hashing on ``user_id`` pins users to a preferred
     cluster for KV cache locality
   - *Latency spillover*: if the affinity cluster is overloaded or unhealthy,
     spill to the nearest cluster with available capacity
   - *Region affinity*: ``user_region`` tag breaks ties

2. **Coordinator-level** (which coordinator within the cluster?):
   - Delegates to the existing ``RouterService`` (consistent hash ring)
   - Prefers the coordinator with lowest active request count

Usage::

    router = MultiClusterRouter(discovery=my_discovery)
    target = await router.route(user_id="user-abc123", user_region="us-east-1")
    # → ClusterCoordinator(url="http://coord-0.us-east:8000")
"""

import asyncio
import hashlib
import time
import threading
from typing import Any

from loguru import logger

from distllm.router.cluster_discovery import (
    ClusterCoordinator,
    ClusterDiscovery,
    ClusterInfo,
)


class ClusterAffinityRing:
    """Consistent hash ring that maps ``user_id`` → preferred cluster.

    Provides sticky session routing with minimal remapping when
    clusters join or leave the federation. Each cluster is represented
    by ``replicas`` virtual tokens on the ring.
    """

    def __init__(self, replicas: int = 64):
        self._replicas = replicas
        self._ring: list[int] = []
        self._token_map: dict[int, str] = {}  # hash → cluster_id
        self._lock = threading.Lock()

    def add_cluster(self, cluster_id: str) -> None:
        with self._lock:
            for i in range(self._replicas):
                key = self._hash(f"{cluster_id}:{i}")
                self._ring.append(key)
                self._token_map[key] = cluster_id
            self._ring.sort()

    def remove_cluster(self, cluster_id: str) -> None:
        with self._lock:
            keys = [k for k, v in self._token_map.items() if v == cluster_id]
            for k in keys:
                self._ring.remove(k)
                del self._token_map[k]

    def get_cluster(self, user_key: str) -> str | None:
        """Get the preferred cluster for *user_key*.

        Returns the cluster whose token is the first match on or after
        the hash of *user_key*.
        """
        if not self._ring:
            return None
        with self._lock:
            h = self._hash(user_key)
            for token in self._ring:
                if token >= h:
                    return self._token_map.get(token)
            return self._token_map.get(self._ring[0])

    def get_cluster_with_fallback(self, user_key: str,
                                  healthy_clusters: set[str]) -> str | None:
        """Walk the ring from the preferred cluster until a healthy one is found.

        This is the primary lookup used by ``MultiClusterRouter``.
        """
        if not self._ring:
            return None
        h = self._hash(user_key)
        with self._lock:
            start_idx = 0
            for i, token in enumerate(self._ring):
                if token >= h:
                    start_idx = i
                    break

            for i in range(len(self._ring)):
                cid = self._token_map[self._ring[(start_idx + i) % len(self._ring)]]
                if cid in healthy_clusters:
                    return cid
        return None

    def _hash(self, key: str) -> int:
        return int(hashlib.sha256(key.encode()).hexdigest(), 16)

    @property
    def cluster_count(self) -> int:
        with self._lock:
            return len(set(self._token_map.values()))


class LatencyRouter:
    """Selects a target cluster based on measured latency and capacity.

    Routes to the nearest cluster with available capacity when the
    preferred cluster is overloaded or unhealthy (spillover routing).
    """

    def __init__(self, discovery: ClusterDiscovery,
                 local_cluster_id: str = "default",
                 local_latency_threshold_ms: float = 5.0,
                 spill_threshold: float = 0.85):
        self._discovery = discovery
        self._local_cluster_id = local_cluster_id
        self._local_latency_threshold = local_latency_threshold_ms
        self._spill_threshold = spill_threshold
        self._latency_overrides: dict[str, float] = {}  # cluster_id → ms
        self._lock = threading.Lock()

    def set_latency(self, cluster_id: str, latency_ms: float) -> None:
        """Override measured latency for a cluster (manual probe result)."""
        with self._lock:
            self._latency_overrides[cluster_id] = latency_ms

    def get_latency(self, cluster_id: str) -> float:
        """Get the round-trip latency to a cluster.

        Returns an explicit override if set, otherwise looks up the
        cluster's ``measured_latency_ms``, defaulting to 1000ms for
        unknown clusters.
        """
        with self._lock:
            if cluster_id in self._latency_overrides:
                return self._latency_overrides[cluster_id]
        info = self._discovery.get_cluster(cluster_id)
        if info is not None and info.measured_latency_ms > 0:
            return info.measured_latency_ms
        return 1000.0

    def should_spill(self, cluster_id: str) -> bool:
        """Check if a cluster is overloaded enough to trigger spillover."""
        info = self._discovery.get_cluster(cluster_id)
        if info is None:
            return True
        return info.is_overloaded or not info.healthy

    def select_target(self, preferred_cluster: str | None = None,
                      user_key: str = "") -> tuple[str, str]:
        """Select the best target cluster.

        Args:
            preferred_cluster: The cluster preferred by the affinity ring.
                ``None`` means pick the best based purely on latency/capacity.
            user_key: User identifier (used for tie-breaking).

        Returns:
            ``(cluster_id, reason)`` tuple.
        """
        all_clusters = self._discovery.get_clusters()

        # 1. Try preferred cluster if it's healthy
        if preferred_cluster and preferred_cluster in all_clusters:
            info = all_clusters[preferred_cluster]
            local_latency = self.get_latency(self._local_cluster_id)
            pref_latency = self.get_latency(preferred_cluster)
            if not info.is_overloaded and info.healthy:
                return preferred_cluster, "affinity_healthy"

        # 2. Check local cluster as fast fallback
        local = all_clusters.get(self._local_cluster_id)
        if local and not local.is_overloaded and local.healthy:
            return self._local_cluster_id, "local_fallback"

        # 3. Find the nearest cluster with available capacity
        best_cluster = None
        best_reason = "no_capacity"
        best_score = float("inf")

        for cid, info in all_clusters.items():
            if cid == preferred_cluster or not info.healthy:
                continue
            latency = self.get_latency(cid)
            capacity = info.available_capacity

            if info.is_overloaded:
                continue

            # Score = latency / max(capacity, epsilon)
            score = latency / max(capacity, 0.01)
            if score < best_score:
                best_score = score
                best_cluster = cid
                best_reason = f"latency_spillover"

        if best_cluster is not None:
            return best_cluster, best_reason

        # 4. Everything is overloaded — return local as last resort
        return self._local_cluster_id, "all_overloaded_fallback"


class MultiClusterRouter:
    """Top-level router that routes user sessions across federated clusters.

    Combines consistent-hash affinity, latency-based spillover, and
    coordinator-level routing into a single ``route()`` call.

    Usage::

        discovery = ClusterDiscovery()
        await discovery.start("static", {...})
        router = MultiClusterRouter(discovery)
        target = await router.route("user-abc123")
        response = await router.chat_completion(target, body, headers)
    """

    def __init__(self, discovery: ClusterDiscovery,
                 local_cluster_id: str = "default"):
        self._discovery = discovery
        self._local_cluster_id = local_cluster_id
        self._affinity = ClusterAffinityRing()
        self._latency_router = LatencyRouter(discovery, local_cluster_id)
        self._lock = threading.Lock()

        # Wire discovery events to keep the affinity ring in sync
        discovery.on_cluster_add(self._affinity.add_cluster)
        discovery.on_cluster_remove(self._affinity.remove_cluster)

        # Seed the ring with already-discovered clusters
        for cid in discovery.get_cluster_ids():
            self._affinity.add_cluster(cid)

    async def start(self) -> None:
        logger.info("MultiClusterRouter started")

    async def stop(self) -> None:
        logger.info("MultiClusterRouter stopped")

    async def route(self, user_id: str, user_region: str | None = None,
                    spill_threshold: float | None = None) -> tuple[ClusterCoordinator | None, str]:
        """Route a user session to the best cluster + coordinator.

        .. code-block:: python

            coord, reason = await router.route("user-abc123")
            # coord.url → "http://coord-0.us-east:8000"
            # reason   → "affinity_healthy"

        Args:
            user_id: Unique user identifier for sticky routing.
            user_region: Optional region hint (e.g. ``"us-east-1"``).
            spill_threshold: Override default spill threshold (0.0-1.0).

        Returns:
            ``(ClusterCoordinator or None, reason_string)``.
        """
        # Step 1: Determine the preferred cluster
        healthy_ids = set()
        for cid, info in self._discovery.get_clusters().items():
            if info.healthy:
                healthy_ids.add(cid)

        preferred = self._affinity.get_cluster_with_fallback(user_id, healthy_ids)

        # Region hint overrides ring if the ring's pick is unhealthy
        if user_region:
            region_cluster = self._resolve_region(user_region)
            if region_cluster and region_cluster in healthy_ids:
                preferred = region_cluster

        # Step 2: Select target cluster (latency-aware spillover)
        target_id, reason = self._latency_router.select_target(
            preferred_cluster=preferred, user_key=user_id,
        )

        # Step 3: Get the best coordinator in the target cluster
        coord = self._discovery.get_coordinator(target_id)

        if coord is None:
            logger.warning(f"No healthy coordinator in cluster {target_id}")
            return None, f"no_coordinator_in_{target_id}"

        return coord, reason

    def _resolve_region(self, region: str) -> str | None:
        """Map a region name to a cluster ID.

        Checks for an exact match on region, then falls back to
        checking if the region is a prefix of a cluster ID.
        """
        for cid, info in self._discovery.get_clusters().items():
            if info.region == region:
                return cid
        for cid in self._discovery.get_cluster_ids():
            if cid.startswith(region):
                return cid
        return None

    def report_load(self, cluster_id: str, *,
                    gpu_util: float = 0.0, queue_depth: int = 0) -> None:
        """Report load metrics from a cluster heartbeat."""
        self._discovery.update_cluster_load(
            cluster_id, gpu_util=gpu_util, queue_depth=queue_depth,
        )

    def report_latency(self, cluster_id: str, latency_ms: float) -> None:
        """Report a measured latency to a cluster."""
        self._latency_router.set_latency(cluster_id, latency_ms)
        info = self._discovery.get_cluster(cluster_id)
        if info:
            info.measured_latency_ms = latency_ms

    @property
    def discovery(self) -> ClusterDiscovery:
        return self._discovery

    @property
    def affinity(self) -> ClusterAffinityRing:
        return self._affinity
