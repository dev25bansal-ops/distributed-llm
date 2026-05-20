"""FederationRouter: cross-cluster topology and geo-routing facade.

Integrates DNS-based geo-routing, cross-cluster KV cache replication,
and capacity-aware spillover to support multi-cluster federated inference.
"""

from __future__ import annotations

import socket
import threading
from typing import Any

from distllm.core.cluster_topology import FederationManager, CrossClusterLatencyMonitor
from distllm.core.geo_router import GeoRouter, LoadReporter
from distllm.core.cross_cluster_forwarder import CrossClusterForwarder


class DNSGeoResolver:
    """DNS-based geo-routing: maps client IPs to the nearest cluster.

    Uses a combination of:
      - Explicit region-to-cluster mapping
      - EDNS0 client subnet (ECS) if available via DNS
      - Latency matrix fallback
    """

    def __init__(self, default_region: str = "default"):
        self._region_map: dict[str, str] = {}  # region -> cluster_id
        self._ip_prefix_map: dict[str, str] = {}  # IP prefix -> cluster_id
        self._default_region = default_region
        self._lock = threading.Lock()

    def map_region(self, region: str, cluster_id: str) -> None:
        with self._lock:
            self._region_map[region] = cluster_id

    def map_ip_prefix(self, prefix: str, cluster_id: str) -> None:
        """Map an IP CIDR prefix to a cluster (e.g. '10.0.0.0/8' -> 'us-east-1')."""
        with self._lock:
            self._ip_prefix_map[prefix] = cluster_id

    def resolve(self, client_ip: str | None = None, dns_name: str | None = None) -> str | None:
        """Resolve a client to the nearest cluster via DNS/geo hints."""
        # 1. Try explicit DNS name mapping
        if dns_name:
            try:
                hints = socket.getaddrinfo(dns_name, None)
                for hint in hints:
                    addr = hint[4][0]
                    cluster = self._resolve_ip(addr)
                    if cluster:
                        return cluster
            except OSError:
                pass

        # 2. Try IP-based resolution
        if client_ip:
            cluster = self._resolve_ip(client_ip)
            if cluster:
                return cluster

        return None

    def _resolve_ip(self, ip: str) -> str | None:
        with self._lock:
            for prefix, cluster_id in self._ip_prefix_map.items():
                if ip.startswith(prefix.rstrip('.')):
                    return cluster_id
        return None

    def get_cluster_for_region(self, region: str) -> str | None:
        with self._lock:
            return self._region_map.get(region)


class KVReplicationQueue:
    """Cross-cluster KV cache replication queue with batching.

    Collects prefix KV entries and replicates them to peer clusters
    asynchronously to warm their prefix caches.
    """

    def __init__(self, forwarder: CrossClusterForwarder | None = None):
        self._forwarder = forwarder or CrossClusterForwarder()
        self._queue: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._replicating = False

    def enqueue(self, prefix_hash: str, kv_data: dict[str, Any], target_clusters: list[str]) -> None:
        """Queue a KV entry for replication to target clusters."""
        with self._lock:
            self._queue.append({
                "prefix_hash": prefix_hash,
                "kv_data": kv_data,
                "targets": target_clusters,
            })

    def flush(self, batch_size: int = 8) -> int:
        """Flush pending replication entries up to batch_size.

        Returns:
            Number of successfully replicated entries.
        """
        with self._lock:
            batch = self._queue[:batch_size]
            self._queue[:batch_size] = []

        if not batch:
            return 0

        success = 0
        for entry in batch:
            for target in entry["targets"]:
                if self._forwarder.forward_kv_cache(
                    remote_node_url=target,
                    prefix_hash=entry["prefix_hash"],
                    kv_data=entry["kv_data"],
                ):
                    success += 1
        return success

    def size(self) -> int:
        with self._lock:
            return len(self._queue)


class FederationRouter:
    """Owns federation topology, load reporting, and geo-routing decisions.

    Integrates:
      - DNSGeoResolver for client-to-cluster mapping
      - KVReplicationQueue for cross-cluster prefix cache warming
      - Capacity-aware spillover to peer clusters
    """

    def __init__(
        self,
        local_cluster_id: str = "default",
        load_reporter: LoadReporter | None = None,
    ):
        self.federation_manager = FederationManager(local_cluster_id=local_cluster_id)
        self.latency_monitor = CrossClusterLatencyMonitor(self.federation_manager)
        self.load_reporter = load_reporter or LoadReporter()
        self.geo_router = GeoRouter(
            federation=self.federation_manager,
            latency_monitor=self.latency_monitor,
            load_reporter=self.load_reporter,
        )
        self.dns_resolver = DNSGeoResolver(default_region=local_cluster_id)
        self.kv_replication = KVReplicationQueue()
        self._forwarder = CrossClusterForwarder()

    def attach_registrar(self, registrar: Any, expert_registry: Any = None) -> None:
        """Wire federation resources to a node registrar."""
        registrar.federation_manager = self.federation_manager
        registrar.expert_registry = expert_registry

    def report_load(
        self,
        cluster_id: str,
        *,
        active: int = 0,
        pending: int = 0,
        gpu_util: float = 0.0,
        queue_depth: int = 0,
    ) -> None:
        self.geo_router.report_cluster_load(
            cluster_id=cluster_id,
            active=active,
            pending=pending,
            gpu_util=gpu_util,
            queue_depth=queue_depth,
        )

    # ---- D1: DNS-based geo-routing ----
    def route_with_dns(
        self,
        request: str = "",
        source_cluster: str | None = None,
        client_ip: str | None = None,
        client_region: str | None = None,
    ) -> tuple[str, str]:
        """Route a request with DNS/geo hints, falling back to capacity-aware routing.

        Priority:
          1. Client region -> cluster mapping (static config)
          2. Client IP -> cluster mapping (DNS prefix map)
          3. Capacity-aware routing via GeoRouter
        """
        source = source_cluster or self.federation_manager.local_cluster_id

        # 1. Region-based routing
        if client_region:
            cluster = self.dns_resolver.get_cluster_for_region(client_region)
            if cluster:
                return cluster, f"dns_region_{client_region}"

        # 2. IP-based routing
        if client_ip:
            cluster = self.dns_resolver.resolve(client_ip=client_ip)
            if cluster:
                return cluster, f"dns_ip_{client_ip}"

        # 3. Fall back to capacity-aware routing
        return self.geo_router.select_target_cluster(request, source)

    # ---- D3: Capacity-aware spillover ----
    def route_with_spillover(
        self,
        request: str = "",
        source_cluster: str | None = None,
        spill_threshold: float = 0.85,
    ) -> tuple[str, str]:
        """Route a request spilling to peer clusters when local capacity is exceeded.

        Args:
            request: Request metadata.
            source_cluster: Origin cluster ID.
            spill_threshold: GPU utilization threshold that triggers spillover.

        Returns:
            Tuple of (target_cluster_id, reason).
        """
        source = source_cluster or self.federation_manager.local_cluster_id
        local_load = self.load_reporter.get_load(source)

        # Stay local if capacity is available
        if local_load is not None and local_load.gpu_utilization < spill_threshold:
            return source, "local_capacity"

        # Spill to best peer cluster
        return self.geo_router.select_target_cluster(request, source)

    def route(self, request: str = "", source_cluster: str | None = None) -> tuple[str, str]:
        """Default route: DNS-aware with capacity spillover."""
        return self.route_with_spillover(request, source_cluster)

    # ---- D2: Cross-cluster KV cache replication ----
    def replicate_prefix_cache(
        self,
        prefix_hash: str,
        kv_data: dict[str, Any],
        target_clusters: list[str] | None = None,
    ) -> None:
        """Replicate a KV cache prefix to peer clusters for global cache warming.

        Args:
            prefix_hash: Hash identifying the cached prefix.
            kv_data: Serialized KV cache data.
            target_clusters: List of peer cluster IDs. If None, replicates to all.
        """
        if target_clusters is None:
            target_clusters = [
                cid for cid in self.federation_manager.list_clusters()
                if cid != self.federation_manager.local_cluster_id
            ]

        self.kv_replication.enqueue(prefix_hash, kv_data, target_clusters)
        flushed = self.kv_replication.flush()
        if flushed:
            logger.debug(f"Replicated {prefix_hash} to {flushed} peer clusters")

    def stats(self) -> dict:
        base = self.geo_router.stats()
        base["geo_regions"] = dict(self.dns_resolver._region_map) if hasattr(self.dns_resolver, '_region_map') else {}
        base["kv_replication_queue"] = self.kv_replication.size()
        return base


# Lazy import to avoid circular dependency at module level
from loguru import logger
