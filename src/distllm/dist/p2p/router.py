"""FederationRouter: cross-cluster topology and geo-routing facade.

Integrates DNS-based geo-routing, cross-cluster KV cache replication,
and capacity-aware spillover to support multi-cluster federated inference.

Also provides BandwidthAwareRouter and PathHealthMonitor for per-peer
bandwidth-aware, latency-weighted, congestion-aware path selection.
"""

from __future__ import annotations

import ipaddress
import socket
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

import itertools

from distllm.dist.topology import FederationManager, CrossClusterLatencyMonitor
from distllm.dist.geo import GeoRouter, LoadReporter
from distllm.dist.cross_cluster import CrossClusterForwarder
from distllm.dist.p2p.load_balancer import FederationLoadBalancer


# ---------------------------------------------------------------------------
# Existing FederationRouter infrastructure
# ---------------------------------------------------------------------------


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
                    # Put peek back by re-chaining -- itertools.cycle has no
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


# ---------------------------------------------------------------------------
# Per-peer path-aware routing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PathStats:
    """Immutable snapshot of a peer's path statistics at a point in time.

    Attributes:
        bandwidth_bps: Measured bandwidth in bits per second.
        rtt_ms: Smoothed round-trip time in milliseconds (via EWMA).
        in_flight_bytes: Estimated bytes currently in flight (unacknowledged).
        success_rate: Exponentially weighted moving average of success (0-1).
        last_seen: Unix timestamp of the last update for this peer.
    """

    bandwidth_bps: float = 0.0
    rtt_ms: float = 0.0
    in_flight_bytes: int = 0
    success_rate: float = 1.0
    last_seen: float = 0.0


class BandwidthAwareRouter:
    """Bandwidth-aware, latency-weighted, congestion-aware path selection router.

    Tracks per-peer metrics -- bandwidth, EWMA-smoothed RTT, in-flight bytes,
    and success rate -- and selects optimal paths for data transfers. Supports
    multi-path splitting for large payloads by distributing across peers
    weighted by their measured bandwidth.

    Thread-safe for concurrent metric updates and path selection queries.
    """

    def __init__(self, ewma_alpha: float = 0.125) -> None:
        """Initialize the router.

        Args:
            ewma_alpha: Smoothing factor for EWMA latency and success rate.
                Default 0.125 (TCP-standard).  Closer to 1 = more responsive,
                closer to 0 = smoother.
        """
        self._ewma_alpha = ewma_alpha
        self._peers: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    # -- Metric updates -----------------------------------------------------

    def update_bandwidth(self, peer_id: str, bps: float) -> None:
        """Record or update a peer's measured bandwidth.

        Args:
            peer_id: Unique identifier for the peer.
            bps: Measured bandwidth in bits per second.
        """
        with self._lock:
            row = self._peers.setdefault(peer_id, {})
            row['bandwidth_bps'] = bps
            row['last_seen'] = time.time()

    def update_latency(self, peer_id: str, rtt_ms: float) -> None:
        """Update a peer's round-trip time using EWMA smoothing.

        Args:
            peer_id: Unique identifier for the peer.
            rtt_ms: Observed round-trip time in milliseconds.
        """
        with self._lock:
            row = self._peers.setdefault(peer_id, {})
            current = row.get('rtt_ms')
            alpha = self._ewma_alpha
            # If no prior value, initialise with the first measurement.
            row['rtt_ms'] = alpha * rtt_ms + (1 - alpha) * (current if current is not None else rtt_ms)
            row['last_seen'] = time.time()

    def update_congestion(self, peer_id: str, delta_bytes: int) -> None:
        """Adjust a peer's in-flight byte count.

        Call with positive delta when sending data and negative delta when
        an acknowledgement is received.  The counter is floored at zero.

        Args:
            peer_id: Unique identifier for the peer.
            delta_bytes: Signed change to the in-flight counter.
        """
        with self._lock:
            row = self._peers.setdefault(peer_id, {})
            current = row.get('in_flight_bytes', 0)
            row['in_flight_bytes'] = max(0, current + delta_bytes)

    def update_success(self, peer_id: str, success: bool) -> None:
        """Update a peer's success/failure rate via EWMA.

        Args:
            peer_id: Unique identifier for the peer.
            success: True if the last operation succeeded, False otherwise.
        """
        with self._lock:
            row = self._peers.setdefault(peer_id, {})
            current = row.get('success_rate', 1.0)
            alpha = self._ewma_alpha
            row['success_rate'] = alpha * (1.0 if success else 0.0) + (1 - alpha) * current
            row['last_seen'] = time.time()

    # -- Path selection -----------------------------------------------------

    def select_path(self, peers: list[str], size_bytes: int) -> str | None:
        """Select the best single peer for a transfer of *size_bytes*.

        Scores each candidate peer based on a composite of:
            - bandwidth (higher is better)
            - latency (lower is better, damped by 100 ms reference)
            - congestion  (fewer in-flight bytes relative to bandwidth is better)
            - success rate (higher is better)

        Args:
            peers: List of candidate peer IDs.
            size_bytes: Size of the data to transfer in bytes.

        Returns:
            The best peer ID, or None if *peers* is empty.
        """
        if not peers:
            return None
        if len(peers) == 1:
            return peers[0]

        with self._lock:
            scored = [(pid, self._score_peer(pid, size_bytes)) for pid in peers]

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0] if scored else None

    def select_multi_path(self, peers: list[str], size_bytes: int) -> list[tuple[str, float]]:
        """Split a large transfer across multiple peers.

        Returns a list of (peer_id, fraction) pairs where fractions sum to 1.0.
        The fractions are proportional to each peer's measured bandwidth, so
        higher-bandwidth peers receive a larger share.

        Args:
            peers: List of candidate peer IDs.
            size_bytes: Total transfer size in bytes (used to sanity-check
                congestion, though the primary weighting is bandwidth-based).

        Returns:
            List of (peer_id, fraction) pairs sorted by fraction descending.
            If *peers* is empty, returns an empty list.
        """
        if not peers:
            return []
        if len(peers) == 1:
            return [(peers[0], 1.0)]

        with self._lock:
            weighted: list[tuple[str, float]] = []
            total_weight = 0.0
            for pid in peers:
                row = self._peers.get(pid, {})
                bw = row.get('bandwidth_bps', 1.0)
                if bw <= 0.0:
                    bw = 1.0
                weighted.append((pid, bw))
                total_weight += bw

        result = [(pid, w / total_weight) for pid, w in weighted]
        result.sort(key=lambda x: x[1], reverse=True)
        return result

    def _score_peer(self, peer_id: str, size_bytes: int) -> float:
        """Compute a composite score for a single peer.

        Higher is better.  The formula:
            score = bandwidth * success_rate
                  / (1 + rtt / 100)
                  / (1 + (in_flight + size_bytes) / bandwidth)
        """
        row = self._peers.get(peer_id, {})
        bw = row.get('bandwidth_bps', 1.0)
        if bw <= 0.0:
            bw = 1.0

        rtt = row.get('rtt_ms', 0.0)
        in_flight = row.get('in_flight_bytes', 0)
        sr = row.get('success_rate', 1.0)

        # Latency penalty: if rtt is 100ms, the term is ~2x divisor.
        latency_penalty = 1.0 + rtt / 100.0
        # Congestion penalty: how long the new data would take to drain.
        congestion_penalty = 1.0 + (in_flight + size_bytes) / bw if bw > 0 else 1.0

        return (bw * sr) / (latency_penalty * congestion_penalty)

    # -- Introspection ------------------------------------------------------

    def peer_stats(self, peer_id: str) -> PathStats | None:
        """Return an immutable snapshot of a peer's current stats.

        Args:
            peer_id: Unique identifier for the peer.

        Returns:
            A PathStats namedtuple, or None if the peer is not tracked.
        """
        with self._lock:
            row = self._peers.get(peer_id)
            if row is None:
                return None
            return PathStats(
                bandwidth_bps=row.get('bandwidth_bps', 0.0),
                rtt_ms=row.get('rtt_ms', 0.0),
                in_flight_bytes=row.get('in_flight_bytes', 0),
                success_rate=row.get('success_rate', 1.0),
                last_seen=row.get('last_seen', 0.0),
            )

    def list_peers(self) -> list[str]:
        """Return a copy of the list of tracked peer IDs."""
        with self._lock:
            return list(self._peers.keys())

    def remove_peer(self, peer_id: str) -> None:
        """Remove a peer from tracking entirely.

        Args:
            peer_id: Unique identifier for the peer to remove.
        """
        with self._lock:
            self._peers.pop(peer_id, None)


class PathHealthMonitor:
    """Periodic health monitor for peer paths.

    Runs health checks on a timer, using a caller-supplied ping callback to
    determine liveness.  Peers that fail *max_consecutive_failures* checks in
    a row are evicted from the router.  Router weights are implicitly
    recalculated on the next ``select_path`` call because the removed peer's
    data is gone.

    Thread-safe.  Start and stop are idempotent.
    """

    def __init__(
        self,
        router: BandwidthAwareRouter,
        ping_callback: Callable[[str], bool],
        max_consecutive_failures: int = 3,
        check_interval: float = 5.0,
    ) -> None:
        """Initialize the health monitor.

        Args:
            router: The BandwidthAwareRouter instance to monitor.
            ping_callback: A callable that takes a peer ID and returns True
                if the peer is reachable, False otherwise.  This could be a
                gRPC ping, an HTTP HEAD, or a simple socket connect.
            max_consecutive_failures: Number of consecutive failed health
                checks before a peer is evicted.  Default 3.
            check_interval: Seconds between health-check rounds.  Default 5.0.
        """
        self._router = router
        self._ping = ping_callback
        self._max_fails = max_consecutive_failures
        self._check_interval = check_interval
        self._failures: dict[str, int] = {}
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None
        self._running = False

    def start(self) -> None:
        """Start periodic health checks.  Idempotent if already running."""
        if self._running:
            return
        self._running = True
        self._schedule_check()

    def stop(self) -> None:
        """Stop periodic health checks.  Idempotent if already stopped."""
        self._running = False
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def _schedule_check(self) -> None:
        if not self._running:
            return
        self._timer = threading.Timer(self._check_interval, self._run_health_check)
        self._timer.daemon = True
        self._timer.start()

    def _run_health_check(self) -> None:
        """Execute one round of health checks against all tracked peers."""
        if not self._running:
            return

        peers = self._router.list_peers()
        for peer_id in peers:
            alive = self._ping(peer_id)
            if alive:
                with self._lock:
                    self._failures.pop(peer_id, None)
                self._router.update_success(peer_id, True)
            else:
                with self._lock:
                    fails = self._failures.get(peer_id, 0) + 1
                    self._failures[peer_id] = fails
                self._router.update_success(peer_id, False)
                if fails >= self._max_fails:
                    self._router.remove_peer(peer_id)
                    with self._lock:
                        self._failures.pop(peer_id, None)

        self._schedule_check()

    @property
    def failure_counts(self) -> dict[str, int]:
        """Return a copy of the current per-peer consecutive failure counts."""
        with self._lock:
            return dict(self._failures)


from loguru import logger
