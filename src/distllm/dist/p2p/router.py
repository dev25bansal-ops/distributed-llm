"""FederationRouter: cross-cluster topology and geo-routing facade.

Integrates DNS-based geo-routing, cross-cluster KV cache replication,
and capacity-aware spillover to support multi-cluster federated inference.
"""

from __future__ import annotations

import ipaddress
import socket
import threading
from typing import Any

import itertools

from distllm.dist.topology import FederationManager, CrossClusterLatencyMonitor
from distllm.dist.geo import GeoRouter, LoadReporter
from distllm.dist.cross_cluster import CrossClusterForwarder
from distllm.dist.p2p.load_balancer import FederationLoadBalancer


class DNSGeoResolver:
    def __init__(self, default_region: str = "default"):
        self._region_map: dict[str, str] = {}
        self._ip_prefix_map: dict[str, str] = {}
        self._default_region = default_region
        self._lock = threading.Lock()

    def map_region(self, region: str, cluster_id: str) -> None:
        with self._lock:
            self._region_map[region] = cluster_id

    def map_ip_prefix(self, prefix: str, cluster_id: str) -> None:
        with self._lock:
            self._ip_prefix_map[prefix] = cluster_id

    def resolve(self, client_ip: str | None = None, dns_name: str | None = None) -> str | None:
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

        if client_ip:
            cluster = self._resolve_ip(client_ip)
            if cluster:
                return cluster

        return None

    def _resolve_ip(self, ip: str) -> str | None:
        with self._lock:
            for prefix, cluster_id in self._ip_prefix_map.items():
                try:
                    if "/" in prefix:
                        network = ipaddress.ip_network(prefix, strict=False)
                        if ipaddress.ip_address(ip) in network:
                            return cluster_id
                    elif ip.startswith(prefix.rstrip(".")):
                        return cluster_id
                except ValueError:
                    if ip.startswith(prefix.rstrip(".")):
                        return cluster_id
        return None

    def get_cluster_for_region(self, region: str) -> str | None:
        with self._lock:
            return self._region_map.get(region)


class KVReplicationQueue:
    def __init__(self, forwarder: CrossClusterForwarder | None = None):
        self._forwarder = forwarder or CrossClusterForwarder()
        self._queue: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._replicating = False

    def enqueue(self, prefix_hash: str, kv_data: dict[str, Any], target_clusters: list[str]) -> None:
        with self._lock:
            self._queue.append({
                "prefix_hash": prefix_hash,
                "kv_data": kv_data,
                "targets": target_clusters,
            })

    def flush(self, batch_size: int = 8) -> int:
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
    def __init__(
        self,
        local_cluster_id: str = "default",
        load_reporter: LoadReporter | None = None,
    ):
        self.federation_manager = FederationManager(local_cluster_id=local_cluster_id)
        self.latency_monitor = CrossClusterLatencyMonitor(self.federation_manager)
        self.load_reporter = load_reporter or LoadReporter()
        self.load_balancer = FederationLoadBalancer()
        self.geo_router = GeoRouter(
            federation=self.federation_manager,
            latency_monitor=self.latency_monitor,
            load_reporter=self.load_reporter,
        )
        self.dns_resolver = DNSGeoResolver(default_region=local_cluster_id)
        self.kv_replication = KVReplicationQueue()
        self._forwarder = CrossClusterForwarder()
        self._node_rr_counters: dict[str, itertools.cycle] = {}

    def _next_node_url(self, cluster_id: str, nodes: list[str]) -> str:
        if cluster_id not in self._node_rr_counters:
            self._node_rr_counters[cluster_id] = itertools.cycle(nodes)
        else:
            current = self._node_rr_counters[cluster_id]
            node_list = list(nodes)
            if node_list:
                # Peek at the next value WITHOUT advancing the iterator,
                # then check membership.  Previously called next() for the
                # check (advancing by 1) then next() for the return
                # (advancing by 2), skipping every other node.
                peek = next(current)
                if peek not in node_list:
                    self._node_rr_counters[cluster_id] = itertools.cycle(nodes)
                else:
                    # Put peek back by re-chaining — itertools.cycle has no
                    # "un-advance", so rebuild the cycle starting from peek.
                    self._node_rr_counters[cluster_id] = itertools.cycle(
                        [peek] + [n for n in node_list if n != peek]
                    )
        return next(self._node_rr_counters[cluster_id])

    def attach_registrar(self, registrar: Any, expert_registry: Any = None) -> None:
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

    def route_with_dns(
        self,
        request: str = "",
        source_cluster: str | None = None,
        client_ip: str | None = None,
        client_region: str | None = None,
    ) -> tuple[str, str]:
        source = source_cluster or self.federation_manager.local_cluster_id

        if client_region:
            cluster = self.dns_resolver.get_cluster_for_region(client_region)
            if cluster:
                return cluster, f"dns_region_{client_region}"

        if client_ip:
            cluster = self.dns_resolver.resolve(client_ip=client_ip)
            if cluster:
                return cluster, f"dns_ip_{client_ip}"

        return self.geo_router.select_target_cluster(request, source)

    def route_with_spillover(
        self,
        request: str = "",
        source_cluster: str | None = None,
        spill_threshold: float = 0.85,
    ) -> tuple[str, str]:
        source = source_cluster or self.federation_manager.local_cluster_id
        local_load = self.load_reporter.get_load(source)

        if local_load is not None and local_load.gpu_utilization < spill_threshold:
            return source, "local_capacity"

        return self.geo_router.select_target_cluster(request, source)

    def route(self, request: str = "", source_cluster: str | None = None) -> tuple[str, str]:
        return self.route_with_spillover(request, source_cluster)

    def forward_to_best_cluster(
        self,
        request_dict: dict[str, Any],
        candidate_clusters: list[str] | None = None,
    ) -> dict[str, Any]:
        if candidate_clusters is None:
            candidate_clusters = [
                cid for cid in self.federation_manager.list_clusters()
                if cid != self.federation_manager.local_cluster_id
            ]

        best = self.load_balancer.get_best_cluster(candidate_clusters)
        if best is None:
            raise RuntimeError("No available cluster to forward request to")

        nodes = self.federation_manager.get_nodes_in_cluster(best)
        if not nodes:
            raise RuntimeError(f"Cluster {best} has no registered nodes")

        target_url = self._next_node_url(best, nodes)
        return self._forwarder.forward_request(target_url, request_dict)

    def forward_to_cluster(
        self,
        cluster_id: str,
        request_dict: dict[str, Any],
    ) -> dict[str, Any]:
        nodes = self.federation_manager.get_nodes_in_cluster(cluster_id)
        if not nodes:
            raise RuntimeError(f"Cluster {cluster_id} has no registered nodes")

        target_url = self._next_node_url(cluster_id, nodes)
        return self._forwarder.forward_request(target_url, request_dict)

    def replicate_prefix_cache(
        self,
        prefix_hash: str,
        kv_data: dict[str, Any],
        target_clusters: list[str] | None = None,
    ) -> None:
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


from loguru import logger
