"""Health Check Service for distributed-llm.

Proactively monitors all registered nodes with periodic gRPC probes,
tracks health state, and triggers failover callbacks on state transitions.
"""

import asyncio
import time
from typing import Callable

from loguru import logger

from distllm.health.failover import FailoverEngine
from distllm.health.prober import probe_node
from distllm.health.state import HealthRecord, HealthStateStore, NodeState

# Exponential backoff constants for permanently dead nodes.
_BACKOFF_FAILURE_THRESHOLD = 10  # consecutive failures before backoff kicks in
_BACKOFF_BASE_SECONDS = 5.0  # base interval multiplied by 2^(failures - threshold)
_BACKOFF_MAX_SECONDS = 300.0  # cap at 5 minutes


class HealthCheckService:
    """Orchestrates periodic health probing and state management.

    Triggers ``on_node_death`` callback when a node transitions to OFFLINE
    after exceeding the failure threshold, so the self-healing cluster can
    recover in-flight sequences and redistribute layers.
    """

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
        self._get_client: Callable[[str], object] | None = None
        self._task: asyncio.Task | None = None
        self._on_node_death: Callable[[str], None] | None = None
        self._next_probe_time: dict[str, float] = {}

    def on_node_death(self, callback: Callable[[str], None]) -> None:
        """Register callback when a node is declared dead (UNHEALTHY -> OFFLINE).

        The coordinator's ``NodeRecoveryManager`` subscribes to this to
        trigger the self-healing protocol.
        """
        self._on_node_death = callback

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
        self._next_probe_time.pop(node_id, None)

    def get_node(self, node_id: str) -> HealthRecord | None:
        return self._store.get(node_id)

    def get_all(self) -> dict[str, HealthRecord]:
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
            now = time.monotonic()
            tasks = []
            for node_id in list(self._store.get_all().keys()):
                next_time = self._next_probe_time.get(node_id, 0.0)
                if now >= next_time:
                    tasks.append(self._probe_once(node_id))
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for result in results:
                    if isinstance(result, Exception):
                        logger.error(f"Unhandled error in health probe: {result}")
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
        record.last_probe_time = time.time()
        if success:
            record.gpu_utilization = data.get("gpu_utilization", 0.0)
            record.memory_used = data.get("memory_used", 0)
            record.memory_total = data.get("memory_total", 0)

        # Capture state BEFORE evaluate: evaluate() mutates record.state in
        # place, so comparing record.state != new_state *after* the call is
        # always False and the OFFLINE/self-healing branch is unreachable.
        old_state = record.state
        new_state = self._failover.evaluate(record, success, latency_ms)
        if old_state != new_state:
            logger.info(
                f"Node {node_id} state changed: {old_state.value} -> {new_state.value}"
            )
            record.state = new_state

            # Trigger self-healing when a node goes OFFLINE
            if new_state == NodeState.OFFLINE and self._on_node_death:
                logger.warning(f"Node {node_id} is OFFLINE, triggering recovery")
                self._on_node_death(node_id)

        # Schedule next probe with exponential backoff for dead nodes
        if record.consecutive_failures >= _BACKOFF_FAILURE_THRESHOLD:
            exp = record.consecutive_failures - _BACKOFF_FAILURE_THRESHOLD
            backoff = min(
                _BACKOFF_BASE_SECONDS * (2 ** exp),
                _BACKOFF_MAX_SECONDS,
            )
            self._next_probe_time[node_id] = time.monotonic() + backoff
        else:
            self._next_probe_time[node_id] = time.monotonic() + self._probe_interval
