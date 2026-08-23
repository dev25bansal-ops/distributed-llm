"""Cross-Cluster Federation — link multiple clusters into one virtual cluster.

Each cluster runs its own coordinator and worker nodes. Federation
allows a coordinator to forward requests to another coordinator's
cluster when local resources are insufficient, or to combine
multiple clusters to run very large models.

Architecture:
    Cluster A (4 GPUs) ◄── Federated Gateway ──► Cluster B (3 GPUs)
         │                                                    │
    ┌────┴────┐                                         ┌────┴────┐
    │ Workers │                                         │ Workers │
    │ L0-L15  │                                         │ L16-L31 │
    └─────────┘                                         └─────────┘

Usage (on each coordinator):
    distllm-coordinator --federate --federation-seed 10.0.0.1:50050
"""

from __future__ import annotations

import asyncio
import enum
import os
import random
import time
import threading
from dataclasses import dataclass
from typing import Any

import httpx
from loguru import logger
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from distllm.dist.p2p.discovery import FederationPeerDiscovery, PeerInfo
from distllm.dist.p2p.router import FederationRouter
from distllm.dist.p2p.load_balancer import FederationLoadBalancer
from distllm.dist.cross_cluster import CrossClusterForwarder
from distllm.dist.cache_digest import ContentRouter, CacheDigestExchange, KVCacheDigest

# Default timeout for internal federation HTTP calls (heartbeat, gossip).
# Separate from the per-request timeout in forward_request / streaming.
_FEDERATION_HTTP_TIMEOUT: float = 5.0
_FEDERATION_MAX_RETRIES: int = 3


class FederationConfig(BaseSettings):
    """Configuration for cross-cluster federation.

    .. rubric:: 12-factor overrides

    Every field can be set via environment variable with the ``FEDERATION_``
    prefix.  Examples::

        export FEDERATION_ENABLED=true
        export FEDERATION_HEARTBEAT_INTERVAL=30
        export FEDERATION_SEED_NODES='["10.0.0.1:50050"]'
    """

    model_config = SettingsConfigDict(
        env_prefix="FEDERATION_",
        extra="ignore",
        frozen=True,
    )

    enabled: bool = False
    cluster_id: str = "default"
    listen_host: str = "0.0.0.0"
    listen_port: int = Field(default=50060, ge=1024, le=65535)
    seed_nodes: list[str] = Field(default_factory=list)
    discovery_interval_s: float = Field(default=30.0, gt=0)
    heartbeat_interval_s: float = Field(default=15.0, gt=0)
    spillover_enabled: bool = True
    spillover_threshold_gpu_util: float = Field(default=80.0, ge=0, le=100)
    # Circuit breaker
    circuit_breaker_threshold: int = Field(default=5, ge=1)
    circuit_breaker_reset_s: float = Field(default=60.0, gt=0)
    # Cache TTL
    cache_digest_ttl_s: float = Field(default=300.0, gt=0)
    # Gossip
    gossip_enabled: bool = False
    gossip_fanout: int = Field(default=3, ge=1)


class BreakerState(str, enum.Enum):
    CLOSED = "closed"          # Normal operation — requests pass through.
    HALF_OPEN = "half_open"    # Probe phase — a single request is allowed
                                 # to test if the peer has recovered.
    OPEN = "open"              # Requests are blocked — peer is presumed dead.


@dataclass
class PeerBreakerEntry:
    """Per-peer state within the circuit breaker."""
    state: BreakerState = BreakerState.CLOSED
    failure_count: int = 0
    open_until: float = 0.0
    half_open_probe_ok: bool = False
    consecutive_successes: int = 0

    def is_timed_out(self, now: float) -> bool:
        return self.open_until > 0 and now >= self.open_until


class PeerCircuitBreaker:
    """Per-peer circuit breaker with full Closed → Half-Open → Open state machine.

    State transitions::

        CLOSED ──(threshold failures)──► OPEN
        OPEN   ──(reset_s elapsed)──────► HALF_OPEN
        HALF_OPEN ──(probe succeeds)────► CLOSED
        HALF_OPEN ──(probe fails)───────► OPEN
    """

    def __init__(self, threshold: int = 5, reset_s: float = 60.0,
                 half_open_max: int = 3):
        self._threshold = threshold
        self._reset_s = reset_s
        self._half_open_max = half_open_max
        self._entries: dict[str, PeerBreakerEntry] = {}
        self._lock = threading.Lock()

    def _get_entry(self, peer_id: str) -> PeerBreakerEntry:
        if peer_id not in self._entries:
            self._entries[peer_id] = PeerBreakerEntry()
        return self._entries[peer_id]

    def record_failure(self, peer_id: str) -> None:
        with self._lock:
            entry = self._get_entry(peer_id)
            entry.failure_count += 1
            entry.consecutive_successes = 0

            if entry.state == BreakerState.HALF_OPEN:
                # Probe failed — back to OPEN for the full reset window.
                entry.state = BreakerState.OPEN
                entry.open_until = time.time() + self._reset_s
                logger.warning(
                    f"Circuit breaker HALF_OPEN→OPEN for {peer_id} "
                    f"(probe failed)"
                )
            elif entry.failure_count >= self._threshold:
                entry.state = BreakerState.OPEN
                entry.open_until = time.time() + self._reset_s
                logger.warning(
                    f"Circuit breaker CLOSED→OPEN for {peer_id} "
                    f"({entry.failure_count} failures)"
                )

    def record_success(self, peer_id: str) -> None:
        with self._lock:
            entry = self._get_entry(peer_id)
            if entry.state == BreakerState.HALF_OPEN:
                entry.consecutive_successes += 1
                if entry.consecutive_successes >= self._half_open_max:
                    entry.state = BreakerState.CLOSED
                    entry.failure_count = 0
                    entry.open_until = 0.0
                    entry.consecutive_successes = 0
                    logger.info(
                        f"Circuit breaker HALF_OPEN→CLOSED for {peer_id} "
                        f"({entry.consecutive_successes} probe successes)"
                    )
            elif entry.state == BreakerState.CLOSED:
                entry.failure_count = 0  # reset on success in closed state

    def is_open(self, peer_id: str) -> bool:
        with self._lock:
            entry = self._get_entry(peer_id)
            now = time.time()

            if entry.state == BreakerState.OPEN:
                # Check if it's time to transition to HALF_OPEN.
                if entry.is_timed_out(now):
                    entry.state = BreakerState.HALF_OPEN
                    entry.open_until = 0.0
                    logger.info(
                        f"Circuit breaker OPEN→HALF_OPEN for {peer_id} "
                        f"(reset window elapsed)"
                    )
                    return False  # Allow the probe request through.
                return True

            if entry.state == BreakerState.HALF_OPEN:
                # In HALF_OPEN only allow a limited number of probe requests.
                # Once consecutive_successes reaches half_open_max we close.
                return False  # Probe requests are allowed.

            return False  # CLOSED — normal operation.

    def force_open(self, peer_id: str) -> None:
        """Manually open the circuit for a peer (e.g. on unrecoverable error)."""
        with self._lock:
            entry = self._get_entry(peer_id)
            entry.state = BreakerState.OPEN
            entry.open_until = time.time() + self._reset_s
            entry.failure_count = self._threshold
            logger.warning(f"Circuit breaker FORCE_OPEN for {peer_id}")

    def get_state(self) -> dict[str, Any]:
        with self._lock:
            open_peers = []
            half_open_peers = []
            tracked = []
            for pid, entry in self._entries.items():
                tracked.append({
                    "peer_id": pid,
                    "state": entry.state.value,
                    "failures": entry.failure_count,
                })
                if entry.state == BreakerState.OPEN:
                    open_peers.append(pid)
                elif entry.state == BreakerState.HALF_OPEN:
                    half_open_peers.append(pid)
            return {
                "open_breakers": open_peers,
                "half_open_breakers": half_open_peers,
                "tracked_peers": tracked,
            }


class FederationCoordinator:
    """Manages cross-cluster federation for a coordinator.

    Integrates peer discovery, routing, load balancing, and
    request forwarding to create a unified federated cluster.

    Args:
        config: Federation configuration.
        local_cluster_id: Unique ID for this cluster.
        local_host: This coordinator's hostname/IP.
        local_port: This coordinator's API port.
        coordinator_ref: Reference to the Coordinator for status checks.
    """

    def __init__(
        self,
        config: FederationConfig,
        local_cluster_id: str,
        local_host: str,
        local_port: int,
        coordinator_ref=None,
    ):
        self.config = config
        self._coordinator = coordinator_ref
        self._local_cluster_id = local_cluster_id
        self._discovery = FederationPeerDiscovery(
            own_cluster_id=local_cluster_id,
            own_host=local_host,
            own_port=local_port,
            discovery_interval_s=config.discovery_interval_s,
        )
        self._router = FederationRouter()
        self._load_balancer = FederationLoadBalancer()
        self._forwarder = CrossClusterForwarder()
        self._peers: dict[str, PeerInfo] = {}
        self._evicted_peers: dict[str, dict] = {}  # peer_id -> {info, evicted_at, retry_after}
        self._running = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None

        # Circuit breaker for peer failures
        self._circuit_breaker = PeerCircuitBreaker(
            threshold=config.circuit_breaker_threshold,
            reset_s=config.circuit_breaker_reset_s,
        )

        # Split-brain detection
        from distllm.core.split_brain import SplitBrainDetector
        self._split_brain = SplitBrainDetector(
            cluster_id=local_cluster_id,
            heartbeat_timeout_s=config.heartbeat_interval_s * 3,
            failure_threshold=config.circuit_breaker_threshold,
        )

        # Content-based routing state
        self._content_router = ContentRouter()
        self._cache_digests: dict[str, dict[str, Any]] = {}
        self._cache_digest_timestamps: dict[str, float] = {}
        self._local_cache_digest: dict[str, Any] | None = None
        self._svid_pem: str | None = None

        # Gossip state
        self._gossip_thread: threading.Thread | None = None

        # Metrics
        self._metrics = {
            "total_forwards": 0,
            "forward_successes": 0,
            "forward_failures": 0,
            "spillovers": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "heartbeats_sent": 0,
            "heartbeats_failed": 0,
        }
        self._metrics_lock = threading.Lock()

        # Determine whether to use HTTPS based on coordinator's TLS config
        self._use_tls = False
        if coordinator_ref is not None and hasattr(coordinator_ref, 'config'):
            tls_enabled = os.environ.get("DISTLLM_TLS_ENABLED", "false").lower() == "true"
            if hasattr(coordinator_ref.config, 'tls'):
                tls_enabled = tls_enabled or getattr(coordinator_ref.config.tls, 'enabled', False)
            self._use_tls = tls_enabled

        # Persistent HTTP clients for connection pooling
        self._http_client = httpx.Client(timeout=_FEDERATION_HTTP_TIMEOUT)
        self._async_client = httpx.AsyncClient(timeout=30.0)

    def _scheme(self) -> str:
        return "https" if self._use_tls else "http"

    def start(self) -> None:
        """Start federation discovery and heartbeat."""
        if not self.config.enabled:
            return

        self._discovery.add_seed_nodes(self.config.seed_nodes)

        self._running.set()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True,
            name="federation-heartbeat",
        )
        self._heartbeat_thread.start()

        if self.config.gossip_enabled:
            self._gossip_thread = threading.Thread(
                target=self._gossip_loop,
                daemon=True,
                name="federation-gossip",
            )
            self._gossip_thread.start()

        logger.info(f"Federation started: cluster={self.config.cluster_id}, "
                     f"port={self.config.listen_port}, "
                     f"seeds={self.config.seed_nodes}")

    def stop(self) -> None:
        self._running.clear()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=5)
            self._heartbeat_thread = None
        if self._gossip_thread is not None:
            self._gossip_thread.join(timeout=5)
            self._gossip_thread = None
        self._http_client.close()
        logger.info("Federation stopped")

    async def close(self) -> None:
        """Async close: shut down and close the persistent HTTP clients."""
        self.stop()
        await self._async_client.aclose()

    def get_peers(self) -> list[dict[str, Any]]:
        """Get list of discovered peer clusters."""
        return [
            {
                "cluster_id": pid,
                "host": info.host,
                "port": info.port,
                "region": info.region,
                "last_seen": info.last_seen,
                "is_edge": info.is_edge,
            }
            for pid, info in self._peers.items()
        ]

    def get_status(self) -> dict[str, Any]:
        """Get federation status."""
        return {
            "enabled": self.config.enabled,
            "cluster_id": self.config.cluster_id,
            "peers": self.get_peers(),
            "spillover_enabled": self.config.spillover_enabled,
            "seed_nodes": self.config.seed_nodes,
        }

    def _heartbeat_loop(self) -> None:
        """Background loop: discover peers and exchange load metrics."""
        while self._running.is_set():
            self._running.wait(self.config.heartbeat_interval_s)
            if not self._running.is_set():
                break

            self._discover_peers()
            self._exchange_heartbeats()

    def _discover_peers(self) -> None:
        """Discover peer coordinators via seed nodes."""
        try:
            discovered = self._discovery.discover_peers()
            for peer in discovered:
                self._peers[peer.cluster_id] = peer
                if peer.cluster_id != self.config.cluster_id:
                    logger.debug(f"Discovered peer cluster: {peer.cluster_id} "
                                  f"at {peer.host}:{peer.port}")
        except Exception as e:
            logger.debug(f"Peer discovery failed: {e}")

    def _exchange_heartbeats(self) -> None:
        """Send heartbeat to all known peers with local load and cache digest."""
        headers = {}
        if self._coordinator is not None:
            cluster_key = getattr(self._coordinator.config, 'cluster_key', None)
            if cluster_key:
                headers["X-Cluster-Key"] = cluster_key

        # Zero-trust (A4): attach THIS node's SPIFFE SVID so receivers can
        # attribute the heartbeat to a specific workload.  The SVID is
        # issued once (dev CA scaffold) and cached for the process.
        if self._svid_pem is None:
            try:
                from distllm.security.spiffe import PeerIdentity, issue_svid

                svid = issue_svid(PeerIdentity(peer_id=self.config.cluster_id))
                self._svid_pem = svid.cert_pem
            except Exception as exc:  # scaffold unavailable — stay silent
                logger.debug(f"SVID issuance unavailable: {exc}")
        if self._svid_pem:
            headers["X-SVID-PEM"] = self._svid_pem

        for peer_id, peer in list(self._peers.items()):
            if peer_id == self.config.cluster_id:
                continue
            try:
                local_load = self._get_local_load()

                # Attach cache digest if available
                if self._local_cache_digest is not None:
                    local_load["cache_digest"] = self._local_cache_digest

                resp = self._http_client.post(
                    f"{self._scheme()}://{peer.host}:{peer.port}/v1/federation/heartbeat",
                    json=local_load,
                    headers=headers or None,
                    timeout=_FEDERATION_HTTP_TIMEOUT,
                )
                if resp.status_code == 200:
                    remote_load = resp.json()
                    self._load_balancer.report_load(
                        cluster_id=peer_id,
                        active_requests=remote_load.get("active_requests", 0),
                        pending_requests=remote_load.get("pending_requests", 0),
                        gpu_utilization=remote_load.get("gpu_utilization", 0.0),
                        queue_depth=remote_load.get("pending_requests", 0),
                    )

                    # Parse remote cache digest
                    remote_digest_raw = remote_load.get("cache_digest")
                    if remote_digest_raw is not None:
                        if isinstance(remote_digest_raw, dict):
                            self._cache_digests[peer_id] = remote_digest_raw
                            self._cache_digest_timestamps[peer_id] = time.time()
                        elif isinstance(remote_digest_raw, str):
                            parsed = CacheDigestExchange.deserialize(
                                remote_digest_raw.encode("latin1")
                            )
                            if parsed:
                                for cid, digest in parsed.items():
                                    self._cache_digests[cid] = digest
                                    self._cache_digest_timestamps[cid] = time.time()

                    self._evict_stale_cache_digests()

                    peer.last_seen = time.time()
                    self._circuit_breaker.record_success(peer_id)
                    self._split_brain.heartbeat(peer_id)
            except Exception as e:
                self._circuit_breaker.record_failure(peer_id)
                self._split_brain.record_failure(peer_id)
                if self._circuit_breaker.is_open(peer_id):
                    logger.warning(f"Removing peer {peer_id} (circuit breaker open)")
                    # Store peer info for potential re-discovery
                    peer_info = self._peers.pop(peer_id, None)
                    if peer_info:
                        self._evicted_peers[peer_id] = {
                            **peer_info,
                            "evicted_at": time.time(),
                            "retry_after": time.time() + 300,  # Try again in 5 min
                        }
                else:
                    logger.debug(f"Peer {peer_id} heartbeat failed: {e}")
            self._record_metric("heartbeats_sent")

        # Check for split-brain partition after all heartbeats
        if self._split_brain.check_partition():
            partitioned = self._split_brain.get_partitioned_peers()
            logger.warning(
                f"Split-brain detected! Partitioned peers: {partitioned}. "
                f"Incrementing fence token."
            )
            self._split_brain.increment_fence_token()

    def _get_local_load(self) -> dict[str, Any]:
        """Get local cluster load metrics from real system sources.

        Uses:
          - ``SystemMonitor`` with pynvml for GPU utilization
          - ``BatchScheduler.stats()`` for active/pending request counts
          - ``psutil`` for CPU and memory load
        """
        import psutil

        coord = self._coordinator
        if coord is None:
            return {
                "gpu_utilization": 0.0,
                "gpu_memory_percent": 0.0,
                "cpu_percent": 0.0,
                "memory_percent": 0.0,
                "active_requests": 0,
                "pending_requests": 0,
                "node_count": 0,
            }

        gpu_util = 0.0
        gpu_mem_pct = 0.0

        try:
            from distllm.core.monitor import SystemMonitor
            monitor = SystemMonitor()
            metrics = monitor.collect()
            gpu_info = metrics.get("gpu", {})
            if "utilization_gpu" in gpu_info:
                gpu_util = float(gpu_info["utilization_gpu"])
            if "memory_percent" in gpu_info:
                gpu_mem_pct = float(gpu_info["memory_percent"])
        except Exception:
            pass

        active = 0
        pending = 0
        try:
            if coord.scheduler is not None:
                stats = coord.scheduler.stats()
                active = stats.get("active_requests", 0)
                pending = stats.get("pending_requests", 0)
        except Exception:
            pass

        cpu_pct = psutil.cpu_percent(interval=0.1)
        mem_pct = psutil.virtual_memory().percent

        return {
            "gpu_utilization": round(gpu_util, 1),
            "gpu_memory_percent": round(gpu_mem_pct, 1),
            "cpu_percent": round(cpu_pct, 1),
            "memory_percent": round(mem_pct, 1),
            "active_requests": active,
            "pending_requests": pending,
            "node_count": len(coord.nodes),
        }

    def should_spillover(self) -> bool:
        """Check if we should spill over to a federated peer."""
        if not self.config.spillover_enabled or not self._peers:
            return False

        local_load = self._get_local_load()
        gpu_util = local_load.get("gpu_utilization", 0.0)
        return gpu_util > self.config.spillover_threshold_gpu_util

    async def forward_request(self, peer: dict[str, Any],
                              request: dict[str, Any],
                              timeout_s: float = 120.0) -> dict[str, Any]:
        """Forward a chat completion request to a peer cluster.

        Authenticates with the cluster key (X-Cluster-Key) and API key
        (Authorization: Bearer) so the peer's AuthMiddleware accepts the request.

        Uses :meth:`_forward_with_retry` for exponential backoff on
        transient failures.

        Args:
            peer: Peer dict with cluster_id, host, port.
            request: Chat completion request dict.
            timeout_s: Request timeout in seconds.

        Returns:
            Response dict with x-distllm-federated tag.
        """
        peer_id = peer["cluster_id"]

        if self._circuit_breaker.is_open(peer_id):
            raise RuntimeError(f"Circuit breaker open for peer {peer_id}")

        headers = self._build_auth_headers()

        url = f"{self._scheme()}://{peer['host']}:{peer['port']}/v1/chat/completions"
        try:
            resp = await self._async_client.post(url, json=request, headers=headers or None, timeout=timeout_s)
            resp.raise_for_status()
            result = resp.json()
            result["x-distllm-federated"] = peer_id
            self._circuit_breaker.record_success(peer_id)
            self._record_metric("forward_successes")
        except Exception as e:
            self._circuit_breaker.record_failure(peer_id)
            self._record_metric("forward_failures")
            logger.warning(f"Federated request to {peer_id} failed: {e}")
            raise
        finally:
            self._record_metric("total_forwards")
        return result

    async def forward_request_streaming(
        self,
        peer: dict[str, Any],
        request: dict[str, Any],
        timeout_s: float = 120.0,
    ):
        """Forward a request with streaming response.

        Yields chunks as they arrive from the peer cluster.
        """
        peer_id = peer["cluster_id"]

        if self._circuit_breaker.is_open(peer_id):
            raise RuntimeError(f"Circuit breaker open for peer {peer_id}")

        headers = self._build_auth_headers()

        request["stream"] = True
        url = f"{self._scheme()}://{peer['host']}:{peer['port']}/v1/chat/completions"

        try:
            async with self._async_client.stream("POST", url, json=request, headers=headers or None, timeout=timeout_s) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if line.startswith("data: "):
                            data = line[6:]
                            if data.strip() == "[DONE]":
                                break
                            yield data
            self._circuit_breaker.record_success(peer_id)
            self._record_metric("forward_successes")
        except Exception as e:
            self._circuit_breaker.record_failure(peer_id)
            self._record_metric("forward_failures")
            logger.warning(f"Streaming federated request to {peer_id} failed: {e}")
            raise
        finally:
            self._record_metric("total_forwards")

    # ── Exponential backoff retry ─────────────────────────────────────

    async def _forward_with_retry(
        self,
        peer: dict[str, Any],
        request: dict[str, Any],
        timeout_s: float = 120.0,
        max_retries: int = _FEDERATION_MAX_RETRIES,
    ) -> dict[str, Any]:
        """Forward a request with exponential backoff on transient failures.

        Retries on ``httpx`` transport errors (connection reset, DNS failure,
        timeout) and 5xx server errors.  Non-retriable errors (4xx, circuit
        breaker open) raise immediately.

        Backoff schedule: 0.5s, 1s, 2s (jittered ±25%).
        """
        last_exc: Exception | None = None
        peer_id = peer["cluster_id"]

        for attempt in range(1, max_retries + 1):
            try:
                return await self.forward_request(peer, request, timeout_s)
            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                if status >= 500 and status <= 599 and attempt < max_retries:
                    last_exc = e
                    delay = (0.5 * (2 ** (attempt - 1))) * random.uniform(0.75, 1.25)
                    logger.warning(
                        f"Retry {attempt}/{max_retries} for peer {peer_id} "
                        f"(HTTP {status}) in {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)
                else:
                    raise
            except (httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError) as e:
                if attempt < max_retries:
                    last_exc = e
                    delay = (0.5 * (2 ** (attempt - 1))) * random.uniform(0.75, 1.25)
                    logger.warning(
                        f"Retry {attempt}/{max_retries} for peer {peer_id} "
                        f"({type(e).__name__}) in {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)
                else:
                    raise

        # All retries exhausted — surface the last exception.
        raise RuntimeError(
            f"Federated request to {peer_id} failed after {max_retries} retries"
        ) from last_exc

    # ── Cluster health API ────────────────────────────────────────────

    async def check_peer_health(self, peer_id: str) -> dict[str, Any]:
        """Probe a single peer's health endpoint.

        GET ``/v1/federation/health`` on the peer coordinator.  Returns
        the peer's response body (load metrics, uptime, cluster ID) or
        raises on failure.

        This is an *active* health check (vs. passive heartbeat-based
        detection) and can be called on-demand to get a fresh view of
        a peer's liveness.
        """
        peer = self._peers.get(peer_id)
        if peer is None:
            raise ValueError(f"Unknown peer {peer_id}")

        headers = self._build_auth_headers()

        url = f"{self._scheme()}://{peer.host}:{peer.port}/v1/federation/health"
        resp = await self._async_client.get(url, headers=headers or None, timeout=10.0)
        resp.raise_for_status()
        body = resp.json()
        body["_peer_id"] = peer_id
        body["_reachable"] = True
        return body

    async def get_all_peers_health(
        self, timeout_per_peer: float = 5.0,
    ) -> list[dict[str, Any]]:
        """Probe all known peers concurrently.

        Returns a list of health dicts, each with ``_peer_id`` and
        ``_reachable`` fields.  Unreachable peers have ``_reachable=False``
        and an ``_error`` field.  This method never raises — the caller
        inspects ``_reachable`` per peer.
        """
        import asyncio

        async def probe(pid: str) -> dict[str, Any]:
            try:
                return await self.check_peer_health(pid)
            except Exception as e:
                return {"_peer_id": pid, "_reachable": False, "_error": str(e)}

        tasks = [probe(pid) for pid in self._peers if pid != self.config.cluster_id]
        return await asyncio.gather(*tasks)

    # ── HTTPS enforcement ─────────────────────────────────────────────

    def _require_tls(self, msg: str = "") -> None:
        """Raise if the federation link is not using TLS.

        Call this before sending any payload that contains authentication
        material (cluster key, API key) when TLS is expected.
        """
        if not self._use_tls:
            raise RuntimeError(
                f"TLS is required for this operation but federation is "
                f"configured with HTTP. {msg}Enable TLS or set the "
                f"DISTLLM_TLS_ENABLED environment variable. "
                f"This check prevents sending cluster credentials in plaintext."
            )

    def _build_auth_headers(self) -> dict[str, str]:
        """Build authentication headers for federation requests.

        Warns loudly when sending secrets over plaintext HTTP.
        """
        headers: dict[str, str] = {}
        # SECURITY: use dedicated federation API key with minimal permissions
        fed_api_key = os.environ.get("DISTLLM_FEDERATION_API_KEY", "")
        if not fed_api_key:
            fed_api_key = os.environ.get("API_KEY", "")
            if fed_api_key:
                logger.warning(
                    "Federation using admin API_KEY instead of dedicated "
                    "DISTLLM_FEDERATION_API_KEY"
                )
        if fed_api_key:
            headers["Authorization"] = f"Bearer {fed_api_key}"
        if self._coordinator is not None:
            cluster_key = getattr(self._coordinator.config, 'cluster_key', None)
            if cluster_key:
                headers["X-Cluster-Key"] = cluster_key
                if not self._use_tls:
                    logger.warning(
                        "SECURITY: Sending X-Cluster-Key over HTTP (plaintext). "
                        "Enable TLS for production federation links."
                    )
        return headers

    # ── SLA-aware federation ─────────────────────────────────────────

    def get_peer_slo_status(self) -> list[dict[str, Any]]:
        """Get SLO status for all peer clusters.

        Tracks latency, throughput, and error rate per peer to enable
        SLA-aware routing and automatic failover.

        Returns:
            List of peer SLO status dicts.
        """
        results = []
        for pid, peer in self._peers.items():
            if pid == self.config.cluster_id:
                continue
            load = self._load_balancer.get_load(pid)
            results.append({
                "cluster_id": pid,
                "host": peer.host,
                "port": peer.port,
                "available_capacity": load.available_capacity if load else 0.0,
                "is_overloaded": load.is_overloaded if load else False,
                "last_seen": peer.last_seen,
                "is_healthy": (time.time() - peer.last_seen) < 60 if peer.last_seen else False,
            })
        return results

    def select_peer_for_sla(
        self,
        quality_tier: str = "medium",
        max_latency_ms: float = 5000.0,
        min_capacity: float = 10.0,
    ) -> dict[str, Any] | None:
        """Select a peer cluster that meets SLA requirements.

        Combines capacity-based selection with SLO awareness to ensure
        the selected peer can meet the requested quality tier.

        Args:
            quality_tier: Quality tier ("high", "medium", "low").
            max_latency_ms: Maximum acceptable latency.
            min_capacity: Minimum available capacity percentage.

        Returns:
            Peer dict or None if no peer meets SLA.
        """
        candidates = []
        for pid, peer in self._peers.items():
            if pid == self.config.cluster_id:
                continue
            load = self._load_balancer.get_load(pid)
            if load is None or load.is_overloaded:
                continue
            if load.available_capacity < min_capacity:
                continue
            age_s = time.time() - peer.last_seen if peer.last_seen else float("inf")
            if age_s > 60:
                continue
            candidates.append({
                "cluster_id": pid,
                "host": peer.host,
                "port": peer.port,
                "available_capacity": load.available_capacity,
                "age_s": age_s,
            })

        if not candidates:
            return None

        candidates.sort(key=lambda c: (-c["available_capacity"], c["age_s"]))
        return candidates[0]

    # ── Content-based routing ──

    def update_cache_digest(self, prompt_token_ids: list[int] | None = None) -> None:
        """Update the local cluster's KV cache digest.

        Should be called after a new prefix is cached (e.g., on requests with
        long system prompts / few-shot examples).

        Args:
            prompt_token_ids: Token IDs of the cached prefix.  ``None`` to
                clear the local digest.
        """
        if prompt_token_ids is None:
            self._local_cache_digest = None
            return

        digest = KVCacheDigest(window_size=128).compute(prompt_token_ids)
        digest["cluster_id"] = self.config.cluster_id
        self._local_cache_digest = digest
        logger.debug(f"Updated local cache digest ({len(prompt_token_ids)} tokens)")

    def get_best_peer(self, prompt_digest: dict[str, Any] | None = None) -> dict[str, Any] | None:
        """Get the best peer cluster, optionally considering cache affinity.

        When *prompt_digest* is provided, content-based routing is used:
        peers with cached prefixes matching the prompt are preferred.
        Otherwise, falls back to capacity-based selection.

        Args:
            prompt_digest: Optional digest of the current prompt for cache-aware routing.

        Returns:
            Peer dict with ``cluster_id``, ``host``, ``port``, and optionally
            ``cache_affinity`` and ``matched_length``.
        """
        if not self._peers:
            return None

        # Build candidate list with load and cache scores
        candidates = []
        for pid, peer in self._peers.items():
            if pid == self.config.cluster_id:
                continue
            load = self._load_balancer.get_load(pid)
            if load is not None and load.is_overloaded:
                continue

            candidate: dict[str, Any] = {
                "cluster_id": pid,
                "host": peer.host,
                "port": peer.port,
                "available_capacity": load.available_capacity if load else 0.0,
                "cache_affinity": 0.0,
                "matched_length": 0,
            }

            if prompt_digest is not None:
                peer_digest = self._cache_digests.get(pid)
                if peer_digest is not None:
                    scores = self._content_router.score_cluster(
                        prompt_digest,
                        {pid: peer_digest},
                        {pid: 1.0 - (candidate["available_capacity"] / 100.0) if candidate["available_capacity"] > 0 else 1.0},
                    )
                    if scores:
                        candidate["cache_affinity"] = scores[0].cache_affinity
                        candidate["matched_length"] = scores[0].matched_length

            candidates.append(candidate)

        if not candidates:
            return None

        # Sort: by cache affinity descending, then by capacity descending
        candidates.sort(
            key=lambda c: (
                c["cache_affinity"],
                c["available_capacity"],
            ),
            reverse=True,
        )

        # Only use cache affinity if it provides meaningful signal
        if candidates[0]["cache_affinity"] > 0.3:
            return candidates[0]

        # Fall back to capacity-based selection
        candidates.sort(key=lambda c: -c["available_capacity"])
        return candidates[0]

    def get_peers_with_cache(
        self, prompt_digest: dict[str, Any], min_affinity: float = 0.3,
    ) -> list[dict[str, Any]]:
        """Find peer clusters that have cached prefixes relevant to the prompt.

        Args:
            prompt_digest: Digest of the current prompt.
            min_affinity: Minimum similarity score (0.0–1.0) to include.

        Returns:
            List of peer dicts sorted by affinity, each containing
            ``cluster_id``, ``host``, ``port``, ``cache_affinity``,
            ``matched_length``.
        """
        results = []
        for pid, peer in self._peers.items():
            if pid == self.config.cluster_id:
                continue
            peer_digest = self._cache_digests.get(pid)
            if peer_digest is None:
                continue

            scores = self._content_router.score_cluster(
                prompt_digest,
                {pid: peer_digest},
                {pid: 0.0},
            )
            if scores and scores[0].cache_affinity >= min_affinity:
                results.append({
                    "cluster_id": pid,
                    "host": peer.host,
                    "port": peer.port,
                    "cache_affinity": scores[0].cache_affinity,
                    "matched_length": scores[0].matched_length,
                })

        results.sort(key=lambda r: -r["cache_affinity"])
        return results

    async def forward_with_cache_affinity(
        self,
        request: dict[str, Any],
        prompt_token_ids: list[int] | None = None,
    ) -> dict[str, Any]:
        """Forward a request to the cluster with the best cache affinity.

        If the prompt matches a cached prefix on a peer cluster, the request
        is sent there to avoid redundant prefill computation.  Falls back
        to capacity-based routing if no cache match is found.

        This enables cluster-to-cluster speculative decoding: the peer
        cluster with the warm cache handles the prompt as a prefill step
        and returns the result.

        Args:
            request: Chat completion request dict.
            prompt_token_ids: Token IDs of the prompt for cache matching.

        Returns:
            Response dict from the peer, with ``x-distllm-federated`` set.
        """
        prompt_digest = None
        if prompt_token_ids:
            prompt_digest = KVCacheDigest(window_size=128).compute(prompt_token_ids)
            prompt_digest["cluster_id"] = self.config.cluster_id

        peer = self.get_best_peer(prompt_digest=prompt_digest)
        if peer is None:
            raise RuntimeError("No suitable peer available for cache-aware routing")

        result = await self.forward_request(peer, request)

        # Tag the response with routing metadata
        if isinstance(result, dict):
            result["x-distllm-cache-affinity"] = peer.get("cache_affinity", 0.0)
            result["x-distllm-matched-tokens"] = peer.get("matched_length", 0)

        return result

    # ── Cache TTL (clock-skew tolerant) ──────────────────────────────

    def _evict_stale_cache_digests(self) -> None:
        """Evict cache digests older than TTL.

        Uses per-entry timestamps recorded with *local* monotonic time
        when the digest was *received*, not the sender's wall-clock
        ``timestamp`` field.  This avoids premature eviction or stale
        retention due to clock skew between clusters.
        """
        now = time.time()
        ttl = self.config.cache_digest_ttl_s
        stale = [
            pid for pid, ts in self._cache_digest_timestamps.items()
            if now - ts > ttl
        ]
        for pid in stale:
            self._cache_digests.pop(pid, None)
            self._cache_digest_timestamps.pop(pid, None)

    # ── Gossip Protocol ────────────────────────────────────────────

    def _gossip_loop(self) -> None:
        """Background gossip: share cache digests with random peers."""
        while self._running.is_set():
            self._running.wait(10.0)
            if not self._running.is_set():
                break
            self._gossip_once()

    def _gossip_once(self) -> None:
        """Send local cache digest to a random subset of peers."""
        if self._local_cache_digest is None:
            return

        import random
        peers = [p for pid, p in self._peers.items() if pid != self.config.cluster_id]
        if not peers:
            return

        fanout = min(self.config.gossip_fanout, len(peers))
        targets = random.sample(peers, fanout)

        payload = {
            "sender_id": self.config.cluster_id,
            "cache_digest": self._local_cache_digest,
            "timestamp": time.time(),
        }

        for peer in targets:
            try:
                resp = self._http_client.post(
                    f"{self._scheme()}://{peer.host}:{peer.port}/v1/federation/gossip",
                    json=payload,
                    timeout=_FEDERATION_HTTP_TIMEOUT,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    sender = data.get("sender_id")
                    digest = data.get("cache_digest")
                    if sender and digest:
                        self._cache_digests[sender] = digest
                        self._cache_digest_timestamps[sender] = time.time()
            except Exception:
                logger.debug("Federation gossip cache digest exchange failed")

    # ── Metrics ──

    def _record_metric(self, name: str, value: float = 1.0) -> None:
        with self._metrics_lock:
            if name in self._metrics:
                self._metrics[name] += value

    def get_metrics(self) -> dict[str, Any]:
        """Get federation metrics."""
        with self._metrics_lock:
            metrics = dict(self._metrics)
        metrics["peer_count"] = len(self._peers)
        metrics["cache_digest_count"] = len(self._cache_digests)
        metrics["circuit_breaker"] = self._circuit_breaker.get_state()
        return metrics
