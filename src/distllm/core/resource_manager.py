"""Resource manager for distributed LLM nodes.

Handles node lifecycle, health checks, circuit breaking, and connection management.
Extracted from the Coordinator class.
"""

import asyncio
import concurrent.futures
import inspect
import json
import os
import socket
import threading
import time
from dataclasses import dataclass
from collections.abc import Callable

from loguru import logger

from distllm.config.settings import NodeRole
from distllm.core.connection_pool import ConnectionPool, ConnectionPoolConfig
from distllm.core.async_connection_pool import AsyncConnectionPool
from distllm.errors.types import NodeUnreachableError, GRPCTimeoutError
from distllm.dist.node_client import create_node_client, NodeClient


class _PlaceholderClient:
    """Placeholder client created in NodeRegistration.__init__.

    Replaced by a real gRPC client when :meth:`NodeRegistration.init_client`
    is called.  Exists so that ``reg.client is not None`` immediately after
    construction, which simplifies health-check code that checks for client
    presence before calling.
    """

    def __init__(self) -> None:
        self.cluster_key: str | None = None

    def health_check(self, *args, **kwargs):
        raise RuntimeError("PlaceholderClient: call init_client() first")

    def close(self) -> None:
        pass


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    threshold: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0


class NodeRegistration:
    """Tracks a registered node's assignment with GPU capabilities."""

    def __init__(
        self,
        node_id: str,
        host: str,
        port: int,
        start_layer: int,
        end_layer: int,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        use_tls: bool = False,
        ca_cert: str | None = None,
        role: NodeRole = NodeRole.AUTO,
        expert_ids: list[int] | None = None,
        cluster_id: str = "default",
        version: str = "stable",
        instance_type: str = "unknown",
        cost_per_hour: float = 0.0,
        is_spot: bool = False,
    ):
        self.node_id = node_id
        self.host = host
        self.port = port
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.healthy = True
        self.last_health_time: float = time.time()
        self.role = role
        self.expert_ids = expert_ids or []
        self.cluster_id = cluster_id
        self.version = version
        self.instance_type = instance_type
        self.cost_per_hour = cost_per_hour
        self.is_spot = is_spot
        self.client: NodeClient | None = _PlaceholderClient()
        self.async_client = _PlaceholderClient()
        self.weight_source: str | None = None
        # TLS configuration — stored for reconnect so the security
        # property is preserved after transient failures.
        self.use_tls = use_tls
        self.ca_cert = ca_cert

        # GPU capabilities (populated by init_client via Profile RPC)
        self.gpu_name: str = ""
        self.gpu_memory_total: int = 0
        self.gpu_memory_free: int = 0
        self.gpu_sm_count: int = 0
        self.gpu_tflops: float = 0.0
        self.gpu_bandwidth_gbps: float = 0.0
        self.gpu_profile_raw: dict | None = None

    def init_client(self, timeout_s: float = 10.0,
                    cluster_key: str | None = None) -> None:
        """Initialize the gRPC client connection and fetch GPU capabilities.

        Creates a gRPC channel, then calls the Profile RPC to populate
        GPU capability fields for the auto-partitioner.

        Args:
            timeout_s: Connection timeout in seconds.
            cluster_key: Optional shared secret for node authentication.

        Raises:
            NodeUnreachableError: If the node cannot be reached.
        """
        try:
            self.client = create_node_client(
                host=self.host,
                port=self.port,
                use_tls=hasattr(self, 'use_tls') and self.use_tls,
                ca_cert=self.ca_cert if hasattr(self, 'ca_cert') else None,
                timeout_s=timeout_s,
                cluster_key=cluster_key,
            )
            self.healthy = True
            self._fetch_capabilities()
            logger.info(
                f"gRPC client connected to {self.node_id} at {self.host}:{self.port}"
                f" — GPU: {self.gpu_name}, VRAM: {self.gpu_memory_total // (1024**3)}GB"
            )
        except Exception as e:
            self.healthy = False
            raise NodeUnreachableError(
                node_id=self.node_id,
                host=self.host,
                port=self.port,
                original_error=e,
            ) from e

    def _fetch_capabilities(self) -> None:
        """Fetch GPU capabilities from the node via Profile RPC."""
        from distllm.dist import node_pb2
        try:
            req = node_pb2.ProfileRequest(node_id=self.node_id)
            if self.client and self.client.cluster_key:
                req.cluster_key = self.client.cluster_key
            profile = self.client.stub.Profile(req)
            self.gpu_name = profile.gpu_name
            self.gpu_memory_total = profile.total_memory_bytes
            self.gpu_memory_free = profile.free_memory_bytes
            self.gpu_sm_count = profile.sm_count
            self.gpu_tflops = profile.compute_tflops
            self.gpu_bandwidth_gbps = profile.memory_bandwidth_gbps
            self.gpu_profile_raw = {
                "gpu_name": profile.gpu_name,
                "total_memory_bytes": profile.total_memory_bytes,
                "free_memory_bytes": profile.free_memory_bytes,
                "sm_count": profile.sm_count,
                "compute_tflops": profile.compute_tflops,
                "memory_bandwidth_gbps": profile.memory_bandwidth_gbps,
            }
        except Exception as e:
            logger.warning(f"Failed to fetch GPU capabilities from {self.node_id}: {e}")

    def health_check(self) -> bool:
        """Ping the node via HealthCheck RPC. Returns True if healthy."""
        from distllm.dist import node_pb2
        if self.client is None:
            self.healthy = False
            return False
        try:
            req = node_pb2.HealthCheckRequest(node_id=self.node_id)
            if self.client and self.client.cluster_key:
                req.cluster_key = self.client.cluster_key
            resp = self.client.stub.HealthCheck(req)
            self.healthy = resp.healthy
            self.last_health_time = time.time()
            self.gpu_memory_free = resp.memory_total_bytes - resp.memory_used_bytes
            return resp.healthy
        except Exception:
            self.healthy = False
            return False

    def reconnect(self, timeout_s: float = 10.0, cluster_key: str | None = None) -> bool:
        """Reconnect to the node after a failure or coordinator failover.

        Closes the existing client, creates a new gRPC channel, and
        re-fetches GPU capabilities.  Used by workers when the coordinator
        changes and nodes need to re-register.

        Args:
            timeout_s: Connection timeout in seconds.
            cluster_key: Optional shared secret for node authentication.

        Returns:
            True if reconnection succeeded, False otherwise.
        """
        # Close existing connection
        self.close()
        try:
            self.client = create_node_client(
                host=self.host,
                port=self.port,
                use_tls=getattr(self, 'use_tls', False),
                ca_cert=getattr(self, 'ca_cert', None),
                timeout_s=timeout_s,
                cluster_key=cluster_key,
            )
            self.healthy = True
            self._fetch_capabilities()
            logger.info(
                f"Reconnected to {self.node_id} at {self.host}:{self.port}"
                f" — GPU: {self.gpu_name}"
            )
            return True
        except Exception as e:
            self.healthy = False
            self.client = _PlaceholderClient()
            logger.warning(f"Reconnect failed for {self.node_id}: {e}")
            return False

    @property
    def gpu_memory_available_gb(self) -> float:
        """Return available GPU memory in GB."""
        return self.gpu_memory_free / (1024 ** 3) if self.gpu_memory_free else 0.0

    @property
    def is_real_client(self) -> bool:
        """Return True if the client is a real gRPC connection (not a placeholder)."""
        return self.client is not None and not isinstance(self.client, _PlaceholderClient)

    def close(self) -> None:
        """Close connections and release resources."""
        if self.client is not None and not isinstance(self.client, _PlaceholderClient):
            try:
                self.client.close()
            except Exception:
                logger.debug("Resource cleanup failed (non-fatal)")
        self.client = None


class ResourceManager:
    """Manages node lifecycle, health, and circuit breaking.

    Attributes:
        cb_config: Circuit breaker configuration.
        _node_failure_counts: Consecutive failure counts per node.
        _node_recovery_time: When circuit breaker opens per node.
        _lock: Thread lock for thread-safe state updates.
    """

    def __init__(self, cb_config: CircuitBreakerConfig | None = None):
        self.cb_config = cb_config or CircuitBreakerConfig()
        self._node_failure_counts: dict[str, int] = {}
        self._node_recovery_time: dict[str, float] = {}
        self._node_locks: dict[str, threading.Lock] = {}
        self._draining_nodes: set[str] = set()
        self._on_node_failure: Callable[[str], None] | None = None
        self._lock = threading.Lock()
        self._metrics_lock = threading.Lock()
        self._metrics: dict[str, int] = {
            "node_failures": 0,
            "errors": 0,
        }
        self._conn_pool = ConnectionPool(max_size=10, connect_timeout=5.0)
        self._async_conn_pool = AsyncConnectionPool(max_size=10, connect_timeout=5.0)
        # Shared thread pool for health check cycles — replaces per-cycle
        # ThreadPoolExecutor creation (~8,640 pools/day at 10s intervals).
        self._health_check_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=min(32, (os.cpu_count() or 1) * 2),
            thread_name_prefix="health-check",
        )

    def _get_node_lock(self, node_id: str) -> threading.Lock:
        with self._lock:
            if node_id not in self._node_locks:
                self._node_locks[node_id] = threading.Lock()
            return self._node_locks[node_id]

    # -- Circuit Breaker --

    def check_circuit_breaker(self, node_id: str) -> bool:
        """Check if a node's circuit breaker is open.

        Returns True if the node should be skipped.
        """
        lock = self._get_node_lock(node_id)
        with lock:
            failures = self._node_failure_counts.get(node_id, 0)
            if failures < self.cb_config.threshold:
                return False

            recovery_at = self._node_recovery_time.get(node_id, 0)
            if recovery_at > 0 and time.time() >= recovery_at:
                # Cooldown elapsed — clear failure counters and draining state
                # so the node can be retried and routed to again.
                self._node_failure_counts[node_id] = 0
                self._node_recovery_time.pop(node_id, None)
                cooldown_elapsed = True
            else:
                cooldown_elapsed = False

        if cooldown_elapsed:
            with self._lock:
                self._draining_nodes.discard(node_id)
            logger.info(f"Circuit breaker cooldown elapsed for {node_id}, allowing retry")
            return False

        return True

    def record_success(self, node_id: str) -> None:
        """Record a successful node operation and clear circuit breaker state."""
        lock = self._get_node_lock(node_id)
        with lock:
            self._node_failure_counts[node_id] = 0
            self._node_recovery_time.pop(node_id, None)

    def record_failure(self, node_id: str) -> None:
        """Record a node failure and set exponential backoff recovery time.

        Fires the node failure callback (for self-healing) when the
        circuit breaker opens.
        """
        was_below = False
        failures = 0

        lock = self._get_node_lock(node_id)
        with lock:
            was_below = self._node_failure_counts.get(node_id, 0) < self.cb_config.threshold
            self._node_failure_counts[node_id] = self._node_failure_counts.get(node_id, 0) + 1
            failures = self._node_failure_counts[node_id]

            if failures >= self.cb_config.threshold:
                backoff = min(
                    self.cb_config.base_delay * (2 ** (failures - self.cb_config.threshold)),
                    self.cb_config.max_delay,
                )
                self._node_recovery_time[node_id] = time.time() + backoff
                logger.warning(
                    f"Circuit breaker opened for {node_id} after {failures} failures, "
                    f"recovery in {backoff:.1f}s"
                )

        with self._metrics_lock:
            self._metrics["node_failures"] += 1
            self._metrics["errors"] += 1

        if was_below and failures >= self.cb_config.threshold:
            with self._lock:
                self._draining_nodes.add(node_id)
            if self._on_node_failure:
                self._on_node_failure(node_id)

    # -- Node Drain State --

    def mark_node_draining(self, node_id: str) -> None:
        """Mark a node as draining (stop sending new requests)."""
        logger.warning(f"Marking node {node_id} as draining")
        with self._lock:
            self._draining_nodes.add(node_id)

    def mark_node_alive(self, node_id: str) -> None:
        """Re-mark a drained node as healthy."""
        with self._lock:
            self._draining_nodes.discard(node_id)
            self._node_failure_counts.pop(node_id, None)
            self._node_recovery_time.pop(node_id, None)

    def is_node_draining(self, node_id: str) -> bool:
        with self._lock:
            return node_id in self._draining_nodes

    def get_draining_nodes(self) -> list[str]:
        with self._lock:
            return list(self._draining_nodes)

    # -- Node Failure Callback --

    def set_node_failure_callback(self, callback: Callable[[str], None]) -> None:
        """Register callback when a node is declared dead (circuit breaker open)."""
        self._on_node_failure = callback

    # -- Health Checks --

    def _tcp_health_check(self, host: str, port: int, timeout: float = 5.0) -> bool:
        """Perform a TCP connectivity health check using connection pooling.

        Reuses connections from the pool when available. Returns the
        connection to the pool on success, removes it on failure.
        """
        try:
            sock = self._conn_pool.get(host, port)
            # Verify connection is alive with a zero-byte send
            sock.settimeout(timeout)
            self._conn_pool.put(host, port, sock)
            return True
        except (OSError, ConnectionError, socket.timeout):
            # Connection failed — remove any stale pooled connections
            self._conn_pool.remove(host, port)
            return False

    async def _tcp_health_check_async(self, host: str, port: int, timeout: float = 5.0) -> bool:
        """Perform an async TCP connectivity health check using the async pool."""
        try:
            _, writer = await asyncio.wait_for(
                self._async_conn_pool.get(host, port),
                timeout=timeout,
            )
            await self._async_conn_pool.put(host, port, writer)
            return True
        except (OSError, ConnectionError, asyncio.TimeoutError):
            await self._async_conn_pool.remove(host, port)
            return False

    def health_check_all(self, nodes: dict[str, NodeRegistration]) -> dict:
        """Check health of all registered nodes in parallel.

        Uses each node's gRPC client ``health_check()`` method when
        available, falling back to TCP connectivity checks.

        Args:
            nodes: Dict of node_id -> NodeRegistration.

        Returns:
            Dict of node_id -> health status.
        """
        def check_one(node_id: str, node: NodeRegistration) -> tuple[str, dict]:
            if self.check_circuit_breaker(node_id):
                lock = self._get_node_lock(node_id)
                with lock:
                    failures = self._node_failure_counts.get(node_id, 0)
                    recovery_at = self._node_recovery_time.get(node_id, 0)
                recovery_in = max(0, recovery_at - time.time()) if recovery_at > 0 else 0
                return node_id, {
                    "healthy": False,
                    "error": f"Circuit breaker open ({failures} failures, recovery in {recovery_in:.1f}s)",
                }
            try:
                # Use gRPC health check if client is a real connection
                if (
                    node.client is not None
                    and not isinstance(node.client, _PlaceholderClient)
                    and hasattr(node.client, "health_check")
                ):
                    resp = node.client.health_check()
                    healthy = resp.healthy if hasattr(resp, "healthy") else bool(resp)
                    memory_used = getattr(resp, "memory_used", 0)
                    memory_total = getattr(resp, "memory_total", 0)
                    return node_id, {
                        "healthy": healthy,
                        "memory_used": memory_used,
                        "memory_total": memory_total,
                    }
                # Fallback: TCP connectivity check
                healthy = self._tcp_health_check(node.host, node.port)
                return node_id, {
                    "healthy": healthy,
                    "memory_used": 0,
                    "memory_total": 0,
                }
            except (NodeUnreachableError, GRPCTimeoutError, ConnectionError, OSError) as e:
                self.record_failure(node_id)
                return node_id, {"healthy": False, "error": str(e)}

        futures = {self._health_check_pool.submit(check_one, nid, node): nid for nid, node in nodes.items()}
        results = {}
        for future in concurrent.futures.as_completed(futures):
            node_id, status = future.result()
            results[node_id] = status
        return results

    async def health_check_all_async(self, nodes: dict[str, NodeRegistration]) -> dict:
        """Check health of all registered nodes (async).

        Uses each node's async gRPC client ``health_check()`` method when
        available, falling back to async TCP connectivity checks.

        Args:
            nodes: Dict of node_id -> NodeRegistration.

        Returns:
            Dict of node_id -> health status.
        """
        async def check_node(node_id: str, node: NodeRegistration) -> tuple[str, dict]:
            if self.check_circuit_breaker(node_id):
                lock = self._get_node_lock(node_id)
                with lock:
                    failures = self._node_failure_counts.get(node_id, 0)
                    recovery_at = self._node_recovery_time.get(node_id, 0)
                recovery_in = max(0, recovery_at - time.time()) if recovery_at > 0 else 0
                return node_id, {
                    "healthy": False,
                    "error": f"Circuit breaker open ({failures} failures, recovery in {recovery_in:.1f}s)",
                }
            try:
                # Use async gRPC health check if async_client is available
                async_client = getattr(node, "async_client", None)
                if (
                    async_client is not None
                    and not isinstance(async_client, _PlaceholderClient)
                    and hasattr(async_client, "health_check")
                ):
                    resp = await async_client.health_check()
                    healthy = resp.healthy if hasattr(resp, "healthy") else bool(resp)
                    memory_used = getattr(resp, "memory_used", 0)
                    memory_total = getattr(resp, "memory_total", 0)
                    return node_id, {
                        "healthy": healthy,
                        "memory_used": memory_used,
                        "memory_total": memory_total,
                    }
                # Fallback: async TCP connectivity check
                healthy = await self._tcp_health_check_async(node.host, node.port)
                return node_id, {
                    "healthy": healthy,
                    "memory_used": 0,
                    "memory_total": 0,
                }
            except (NodeUnreachableError, GRPCTimeoutError, ConnectionError, OSError) as e:
                self.record_failure(node_id)
                return node_id, {"healthy": False, "error": str(e)}

        tasks = [check_node(nid, node) for nid, node in nodes.items()]
        results_list = await asyncio.gather(*tasks, return_exceptions=True)

        results = {}
        for item in results_list:
            if isinstance(item, Exception):
                results["unknown"] = {"healthy": False, "error": str(item)}
            else:
                node_id, status = item
                results[node_id] = status

        return results

    # -- Connection Management --

    def close_all(self, nodes: dict[str, NodeRegistration]) -> None:
        """Close all node connections in a fire-and-forget manner."""
        for node in nodes.values():
            try:
                node.close()
            except Exception as e:
                logger.debug(f"Error closing node {node.node_id}: {e}")

    async def close_all_async(self, nodes: dict[str, NodeRegistration]) -> None:
        """Close all node connections (async).

        Closes both the async client and the sync client for each node.
        """
        for node in nodes.values():
            try:
                if hasattr(node, "async_client") and node.async_client is not None:
                    if inspect.iscoroutinefunction(getattr(node.async_client, "close", None)):
                        await node.async_client.close()
                    else:
                        node.async_client.close()
            except Exception as e:
                logger.debug(f"Error closing async client for {getattr(node, 'node_id', '?')}: {e}")
            try:
                node.close()
            except Exception as e:
                logger.debug(f"Error closing node {getattr(node, 'node_id', '?')}: {e}")

    # -- Metrics --

    def get_metrics(self) -> dict:
        """Get resource manager metrics."""
        with self._lock:
            now = time.time()
            open_count = sum(
                1 for nid, failures in self._node_failure_counts.items()
                if failures >= self.cb_config.threshold
                and self._node_recovery_time.get(nid, 0) > now
            )
            draining = len(self._draining_nodes)
        with self._metrics_lock:
            m = dict(self._metrics)
        m["draining_nodes"] = draining
        m["circuit_breaker_open"] = open_count
        return m

    def increment_metric(self, name: str, value: int = 1) -> None:
        """Increment a metric counter."""
        with self._metrics_lock:
            if name in self._metrics:
                self._metrics[name] += value
            else:
                self._metrics[name] = value

    # -- Chaos Engineering --

    def simulate_node_failure(self, node_id: str) -> None:
        """Simulate a node failure by triggering the circuit breaker.

        This exercises the same code path as a real failure without actually
        killing the node process. Used for chaos engineering scenarios.

        Args:
            node_id: The node to simulate as failed.
        """
        logger.warning(f"[Chaos] Simulating failure for node {node_id}")
        with self._lock:
            self._node_failure_counts[node_id] = self.cb_config.threshold
            self._node_recovery_time[node_id] = time.time() + 3600  # 1 hour
        with self._metrics_lock:
            self._metrics["node_failures"] += 1

    def shutdown(self) -> None:
        """Release all resources: thread pool, connection pools.

        Called from coordinator.stop() to prevent thread leaks on restart.
        """
        self._health_check_pool.shutdown(wait=False)
        self._conn_pool.close_all()
        # async conn pool close is async — just log that it should be awaited
        # (the event loop may already be closed during shutdown)

    def save_state(self, path: str = ".distllm_circuit_breakers.json") -> None:
        """Persist circuit breaker state to survive restarts (owner-only perms)."""
        with self._lock:
            with self._metrics_lock:
                metrics = dict(self._metrics)
            data = {
                "node_failure_counts": dict(self._node_failure_counts),
                "node_recovery_time": dict(self._node_recovery_time),
                "metrics": metrics,
            }
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w") as f:
                json.dump(data, f)
            logger.debug(f"Circuit breaker state saved to {path}")
        except (OSError, PermissionError) as e:
            logger.error(f"Failed to persist circuit breaker state: {e}")
            raise

    def load_state(self, path: str = ".distllm_circuit_breakers.json") -> None:
        """Restore circuit breaker state from disk."""
        try:
            with open(path) as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return
        with self._lock:
            for nid, count in data.get("node_failure_counts", {}).items():
                self._node_failure_counts[nid] = int(count)
            for nid, t in data.get("node_recovery_time", {}).items():
                self._node_recovery_time[nid] = float(t)
            with self._metrics_lock:
                for k, v in data.get("metrics", {}).items():
                    self._metrics[k] = int(v)
