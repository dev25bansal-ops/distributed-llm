"""Resource manager for distributed LLM nodes.

Handles node lifecycle, health checks, circuit breaking, and connection management.
Extracted from the Coordinator class.
"""

import asyncio
import concurrent.futures
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from distllm.communication.grpc import NodeClient, AsyncNodeClient
from distllm.config.loader import NodeRole
from distllm.errors.types import NodeUnreachableError, GRPCTimeoutError, CircuitBreakerError


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    threshold: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0


class NodeRegistration:
    """Tracks a registered node's assignment."""

    def __init__(
        self,
        node_id: str,
        host: str,
        port: int,
        start_layer: int,
        end_layer: int,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        use_tls: bool = True,
        ca_cert: Optional[str] = None,
        role: NodeRole = NodeRole.AUTO,
        expert_ids: Optional[List[int]] = None,
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
        self.role = role
        self.expert_ids = expert_ids or []
        self.cluster_id = cluster_id
        self.version = version
        self.instance_type = instance_type
        self.cost_per_hour = cost_per_hour
        self.is_spot = is_spot
        self.client = NodeClient(
            host, port,
            max_retries=max_retries,
            retry_delay=retry_delay,
            use_tls=use_tls,
            ca_cert=ca_cert,
        )
        self.async_client = AsyncNodeClient(
            host, port,
            max_retries=max_retries,
            retry_delay=retry_delay,
            use_tls=use_tls,
            ca_cert=ca_cert,
        )


class ResourceManager:
    """Manages node lifecycle, health, and circuit breaking.

    Attributes:
        cb_config: Circuit breaker configuration.
        _node_failure_counts: Consecutive failure counts per node.
        _node_recovery_time: When circuit breaker opens per node.
        _lock: Thread lock for thread-safe state updates.
    """

    def __init__(self, cb_config: Optional[CircuitBreakerConfig] = None):
        self.cb_config = cb_config or CircuitBreakerConfig()
        self._node_failure_counts: Dict[str, int] = {}
        self._node_recovery_time: Dict[str, float] = {}
        self._lock = threading.Lock()
        self._metrics: Dict[str, int] = {
            "node_failures": 0,
            "errors": 0,
        }

    # -- Circuit Breaker --

    def check_circuit_breaker(self, node_id: str) -> bool:
        """Check if a node's circuit breaker is open.

        Returns True if the node should be skipped.
        """
        with self._lock:
            failures = self._node_failure_counts.get(node_id, 0)
            if failures < self.cb_config.threshold:
                return False

            recovery_at = self._node_recovery_time.get(node_id, 0)
            if recovery_at > 0 and time.time() >= recovery_at:
                logger.info(f"Circuit breaker cooldown elapsed for {node_id}, allowing retry")
                return False

            return True

    def record_success(self, node_id: str) -> None:
        """Record a successful node operation and clear circuit breaker state."""
        with self._lock:
            self._node_failure_counts[node_id] = 0
            self._node_recovery_time.pop(node_id, None)

    def record_failure(self, node_id: str) -> None:
        """Record a node failure and set exponential backoff recovery time."""
        with self._lock:
            self._node_failure_counts[node_id] = self._node_failure_counts.get(node_id, 0) + 1
            failures = self._node_failure_counts[node_id]

            self._metrics["node_failures"] += 1
            self._metrics["errors"] += 1

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

    # -- Health Checks --

    def health_check_all(self, nodes: Dict[str, NodeRegistration]) -> dict:
        """Check health of all registered nodes concurrently.

        Args:
            nodes: Dict of node_id -> NodeRegistration.

        Returns:
            Dict of node_id -> health status.
        """
        results = {}
        nodes_to_check = {}

        for node_id, node in nodes.items():
            if self.check_circuit_breaker(node_id):
                failures = self._node_failure_counts.get(node_id, 0)
                recovery_at = self._node_recovery_time.get(node_id, 0)
                recovery_in = max(0, recovery_at - time.time()) if recovery_at > 0 else 0
                results[node_id] = {
                    "healthy": False,
                    "error": f"Circuit breaker open ({failures} failures, recovery in {recovery_in:.1f}s)",
                }
            else:
                nodes_to_check[node_id] = node

        def check_one(nid: str, nd: NodeRegistration) -> Tuple[str, dict]:
            try:
                health = nd.client.health_check()
                return nid, {"healthy": health.healthy, "memory_used": health.memory_used, "memory_total": health.memory_total}
            except (NodeUnreachableError, GRPCTimeoutError, ConnectionError, OSError) as e:
                self.record_failure(nid)
                return nid, {"healthy": False, "error": str(e)}

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(nodes_to_check) or 1) as executor:
            futures = {executor.submit(check_one, nid, nd): nid for nid, nd in nodes_to_check.items()}
            for future in concurrent.futures.as_completed(futures):
                nid, status = future.result()
                results[nid] = status

        return results

    async def health_check_all_async(self, nodes: Dict[str, NodeRegistration]) -> dict:
        """Check health of all registered nodes (async).

        Args:
            nodes: Dict of node_id -> NodeRegistration.

        Returns:
            Dict of node_id -> health status.
        """
        async def check_node(node_id: str, node: NodeRegistration) -> Tuple[str, dict]:
            if self.check_circuit_breaker(node_id):
                failures = self._node_failure_counts.get(node_id, 0)
                recovery_at = self._node_recovery_time.get(node_id, 0)
                recovery_in = max(0, recovery_at - time.time()) if recovery_at > 0 else 0
                return node_id, {
                    "healthy": False,
                    "error": f"Circuit breaker open ({failures} failures, recovery in {recovery_in:.1f}s)",
                }
            try:
                health = await node.async_client.health_check()
                return node_id, {
                    "healthy": health.healthy,
                    "memory_used": health.memory_used,
                    "memory_total": health.memory_total,
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

    def close_all(self, nodes: Dict[str, NodeRegistration]) -> None:
        """Close all node connections."""
        for node in nodes.values():
            try:
                node.client.close()
            except Exception as e:
                logger.debug(f"Error closing node connection: {e}")

    async def close_all_async(self, nodes: Dict[str, NodeRegistration]) -> None:
        """Close all node connections (async)."""
        async def close_node(node: NodeRegistration):
            try:
                await node.async_client.close()
            except Exception as e:
                logger.debug(f"Error closing node connection: {e}")

        tasks = [close_node(node) for node in nodes.values()]
        await asyncio.gather(*tasks)

    # -- Metrics --

    def get_metrics(self) -> Dict[str, int]:
        """Get resource manager metrics."""
        return dict(self._metrics)

    def increment_metric(self, name: str, value: int = 1) -> None:
        """Increment a metric counter."""
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
