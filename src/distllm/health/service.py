"""Health Check Service for distributed-llm.

Proactively monitors all registered nodes with periodic gRPC probes,
tracks health state, and triggers failover callbacks on state transitions.
"""

import asyncio
from typing import Callable, Dict, Optional

from loguru import logger

from distllm.health.state import HealthRecord, HealthStateStore, NodeState
from distllm.health.prober import probe_node
from distllm.health.failover import FailoverEngine


class HealthCheckService:
    """Orchestrates periodic health probing and state management."""

    def __init__(
        self,
        probe_interval: float = 5.0,
        probe_timeout: float = 10.0,
        failure_threshold: int = 3,
        degraded_latency_ms: float = 2000.0,
        recovery_threshold: int = 2,
    ):
        self._store = HealthStateStore()
        self._failover = FailoverEngine(
            failure_threshold=failure_threshold,
            degraded_latency_ms=degraded_latency_ms,
            recovery_threshold=recovery_threshold,
        )
        self._probe_interval = probe_interval
        self._probe_timeout = probe_timeout
        self._running = False
        self._get_client: Optional[Callable[[str], object]] = None
        self._task: Optional[asyncio.Task] = None

    def on_state_change(
        self, callback: Callable[[str, NodeState, NodeState], None]
    ) -> None:
        """Register callback for health state transitions."""
        self._failover.on_state_change(callback)

    def register_node(
        self,
        node_id: str,
        client,
        layer_range: str = "",
    ) -> None:
        """Register a node for health monitoring."""
        record = HealthRecord(node_id=node_id, layer_range=layer_range)
        self._store.set(node_id, record)

    def unregister_node(self, node_id: str) -> None:
        """Remove a node from monitoring."""
        self._store.remove(node_id)

    def get_node(self, node_id: str) -> Optional[HealthRecord]:
        return self._store.get(node_id)

    def get_all(self) -> Dict[str, HealthRecord]:
        return self._store.get_all()

    def healthy_nodes(self) -> list[str]:
        return self._store.healthy_nodes()

    async def start(self, get_client: Callable[[str], object]) -> None:
        """Start the periodic probe loop.

        Args:
            get_client: Function that returns a NodeClient for a given node_id.
        """
        self._get_client = get_client
        self._running = True
        self._task = asyncio.create_task(self._probe_loop())
        logger.info("HealthCheckService started")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("HealthCheckService stopped")

    async def _probe_loop(self) -> None:
        while self._running:
            for node_id in list(self._store.get_all().keys()):
                if not self._running:
                    break
                await self._probe_once(node_id)
            await asyncio.sleep(self._probe_interval)

    async def _probe_once(self, node_id: str) -> None:
        record = self._store.get(node_id)
        if record is None or record.state == NodeState.DRAINING:
            return

        if self._get_client is None:
            return

        client = self._get_client(node_id)
        if client is None:
            return

        success, latency_ms, data = await probe_node(client, timeout=self._probe_timeout)

        # Update record with probe data
        record.last_probe_time = __import__("time").time()
        if success:
            record.gpu_utilization = data.get("gpu_utilization", 0.0)
            record.memory_used = data.get("memory_used", 0)
            record.memory_total = data.get("memory_total", 0)

        new_state = self._failover.evaluate(record, success, latency_ms)
        if record.state != new_state:
            logger.info(
                f"Node {node_id} state changed: {record.state.value} -> {new_state.value}"
            )
            record.state = new_state
