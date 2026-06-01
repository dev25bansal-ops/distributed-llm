"""Health checking for coordinator and worker nodes.

Provides synchronous and async health check dispatch with gRPC
deadline propagation for bounded probe latency.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from loguru import logger

from distllm.core.resource_manager import ResourceManager


class HealthChecker:
    """Dispatches health checks across registered nodes.

    Args:
        resource_mgr: ResourceManager for circuit breaker state.
        timeout_s: Per-node health check timeout in seconds.
            Propagated as gRPC deadline to all health probes.
    """

    def __init__(
        self,
        resource_mgr: ResourceManager,
        timeout_s: float = 5.0,
    ) -> None:
        self._resource_mgr = resource_mgr
        self._timeout_s = timeout_s

    @property
    def timeout_s(self) -> float:
        """Per-node health check timeout."""
        return self._timeout_s

    def check_all(
        self,
        nodes: dict[str, Any],
        node_order: list[str],
        is_healthy_fn: Callable[[str], bool],
    ) -> dict[str, dict]:
        """Run synchronous health checks on all nodes.

        Args:
            nodes: Dict of node_id -> node object.
            node_order: Ordered list of node IDs.
            is_healthy_fn: Callable that returns True if a node is healthy.

        Returns:
            Dict of node_id -> health status dict.
        """
        results: dict[str, dict] = {}
        for node_id in node_order:
            node = nodes.get(node_id)
            if node is None:
                results[node_id] = {"healthy": False, "reason": "not_found"}
                continue

            # Check circuit breaker first
            if self._resource_mgr.check_circuit_breaker(node_id):
                results[node_id] = {
                    "healthy": False,
                    "reason": "circuit_breaker_open",
                }
                continue

            try:
                # Propagate timeout as deadline hint
                if hasattr(node, 'set_deadline'):
                    node.set_deadline(self._timeout_s)
                healthy = is_healthy_fn(node_id)
                results[node_id] = {"healthy": healthy, "timeout_s": self._timeout_s}
                if healthy:
                    self._resource_mgr.record_success(node_id)
                else:
                    self._resource_mgr.record_failure(node_id)
            except Exception as e:
                logger.warning(f"Health check failed for {node_id}: {e}")
                results[node_id] = {"healthy": False, "reason": str(e)}
                self._resource_mgr.record_failure(node_id)

        return results

    async def check_all_async(
        self,
        nodes: dict[str, Any],
        node_order: list[str],
        is_healthy_fn: Callable[[str], bool],
    ) -> dict[str, dict]:
        """Run async health checks on all nodes concurrently.

        Each check is bounded by ``timeout_s`` via ``asyncio.wait_for``.
        """
        results: dict[str, dict] = {}

        async def _check_one(node_id: str) -> None:
            node = nodes.get(node_id)
            if node is None:
                results[node_id] = {"healthy": False, "reason": "not_found"}
                return

            if self._resource_mgr.check_circuit_breaker(node_id):
                results[node_id] = {
                    "healthy": False,
                    "reason": "circuit_breaker_open",
                }
                return

            try:
                # Enforce deadline via asyncio
                healthy = await asyncio.wait_for(
                    asyncio.to_thread(is_healthy_fn, node_id),
                    timeout=self._timeout_s,
                )
                results[node_id] = {"healthy": healthy, "timeout_s": self._timeout_s}
                if healthy:
                    self._resource_mgr.record_success(node_id)
                else:
                    self._resource_mgr.record_failure(node_id)
            except asyncio.TimeoutError:
                logger.warning(f"Health check timed out for {node_id} ({self._timeout_s}s)")
                results[node_id] = {"healthy": False, "reason": "timeout", "timeout_s": self._timeout_s}
                self._resource_mgr.record_failure(node_id)
            except Exception as e:
                logger.warning(f"Health check failed for {node_id}: {e}")
                results[node_id] = {"healthy": False, "reason": str(e)}
                self._resource_mgr.record_failure(node_id)

        tasks = [_check_one(nid) for nid in node_order]
        await asyncio.gather(*tasks)
        return results
