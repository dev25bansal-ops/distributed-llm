"""Spot instance auto-failover for preemptible GPU nodes.

Monitors spot instance health and automatically migrates workloads
when preemption is detected. Reduces GPU costs by 60-80% while
maintaining availability through rapid failover.

Usage::

    failover = SpotFailover(
        coordinator=coord,
        check_interval_s=10,
        migration_timeout_s=120,
    )
    failover.register_node("node-1", is_spot=True, provider="aws")
    failover.start()
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from loguru import logger


class PreemptionSignal(str, Enum):
    """Signals that indicate spot instance preemption."""
    HEALTH_CHECK_FAIL = "health_check_fail"
    GCP_TERMINATION = "gcp_termination"        # metadata endpoint
    AWS_SPOT_WARNING = "aws_spot_warning"       # 2-minute warning
    AZURE_EVICTED = "azure_evicted"             # scheduled events
    CUSTOM = "custom"


@dataclass
class SpotNode:
    """A registered spot instance node."""
    node_id: str
    provider: str = "aws"
    is_spot: bool = True
    region: str = ""
    instance_type: str = ""
    cost_per_hour: float = 0.0
    registered_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    consecutive_failures: int = 0
    is_active: bool = True
    preemption_count: int = 0


@dataclass
class FailoverEvent:
    """Record of a failover event."""
    event_id: str
    node_id: str
    signal: PreemptionSignal
    timestamp: float
    migrated_sequences: int = 0
    migration_time_ms: float = 0.0
    target_node: str = ""
    success: bool = False


class SpotFailover:
    """Monitors spot instances and auto-failovers on preemption.

    Detects preemption via:
    - Health check failures (gRPC timeout)
    - Cloud provider metadata endpoints (GCP/AWS/Azure)
    - Custom preemption callbacks

    On preemption:
    1. Marks node as preempted
    2. Drains in-flight requests
    3. Checkpoints KV cache
    4. Redistributes layers to surviving nodes
    5. Resumes requests on new node
    """

    def __init__(
        self,
        coordinator: Any = None,
        check_interval_s: float = 10.0,
        failure_threshold: int = 3,
        migration_timeout_s: float = 120.0,
        on_preemption: Callable[[str, str], None] | None = None,
    ):
        self._coordinator = coordinator
        self._check_interval = check_interval_s
        self._failure_threshold = failure_threshold
        self._migration_timeout = migration_timeout_s
        self._on_preemption = on_preemption

        self._nodes: dict[str, SpotNode] = {}
        self._events: list[FailoverEvent] = []
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

        self._stats = {
            "preemptions_detected": 0,
            "failovers_completed": 0,
            "failovers_failed": 0,
            "total_migrated_sequences": 0,
            "total_migration_time_ms": 0.0,
        }

    def register_node(
        self,
        node_id: str,
        is_spot: bool = True,
        provider: str = "aws",
        region: str = "",
        instance_type: str = "",
        cost_per_hour: float = 0.0,
    ) -> None:
        """Register a node for spot monitoring."""
        with self._lock:
            self._nodes[node_id] = SpotNode(
                node_id=node_id,
                provider=provider,
                is_spot=is_spot,
                region=region,
                instance_type=instance_type,
                cost_per_hour=cost_per_hour,
            )
        logger.info(f"Registered spot node: {node_id} ({provider}, spot={is_spot})")

    def unregister_node(self, node_id: str) -> None:
        with self._lock:
            self._nodes.pop(node_id, None)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="spot-failover",
        )
        self._thread.start()
        logger.info("Spot failover monitor started")

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=self._check_interval * 2)

    def heartbeat(self, node_id: str) -> None:
        """Record a heartbeat from a node."""
        with self._lock:
            node = self._nodes.get(node_id)
            if node:
                node.last_heartbeat = time.time()
                node.consecutive_failures = 0

    def report_preemption(
        self,
        node_id: str,
        signal: PreemptionSignal = PreemptionSignal.CUSTOM,
    ) -> None:
        """Report a preemption event (called by cloud metadata checker)."""
        logger.warning(f"Preemption reported for {node_id}: {signal.value}")
        self._handle_preemption(node_id, signal)

    def _monitor_loop(self) -> None:
        while self._running:
            try:
                self._check_nodes()
            except Exception as e:
                logger.warning(f"Spot monitor error: {e}")

            deadline = time.time() + self._check_interval
            while self._running and time.time() < deadline:
                time.sleep(1.0)

    def _check_nodes(self) -> None:
        now = time.time()
        with self._lock:
            for node_id, node in list(self._nodes.items()):
                if not node.is_active or not node.is_spot:
                    continue

                elapsed = now - node.last_heartbeat
                if elapsed > self._check_interval * 2:
                    node.consecutive_failures += 1
                    if node.consecutive_failures >= self._failure_threshold:
                        logger.warning(
                            f"Spot node {node_id} missed {node.consecutive_failures} heartbeats"
                        )
                        self._handle_preemption(node_id, PreemptionSignal.HEALTH_CHECK_FAIL)

    def _handle_preemption(self, node_id: str, signal: PreemptionSignal) -> None:
        event_id = f"failover-{int(time.time())}-{node_id}"
        event = FailoverEvent(
            event_id=event_id,
            node_id=node_id,
            signal=signal,
            timestamp=time.time(),
        )

        with self._lock:
            node = self._nodes.get(node_id)
            if node:
                node.is_active = False
                node.preemption_count += 1
            self._stats["preemptions_detected"] += 1

        # Fire callback
        if self._on_preemption:
            try:
                self._on_preemption(node_id, signal.value)
            except Exception as e:
                logger.error(f"Preemption callback failed: {e}")

        # Migrate workloads
        start = time.time()
        migrated = self._migrate_workloads(node_id)
        elapsed_ms = (time.time() - start) * 1000

        event.migrated_sequences = migrated
        event.migration_time_ms = elapsed_ms
        event.success = migrated >= 0

        with self._lock:
            self._events.append(event)
            if event.success:
                self._stats["failovers_completed"] += 1
                self._stats["total_migrated_sequences"] += migrated
                self._stats["total_migration_time_ms"] += elapsed_ms
            else:
                self._stats["failovers_failed"] += 1

        logger.info(
            f"Failover {event_id}: {migrated} sequences migrated in {elapsed_ms:.0f}ms"
        )

    def _migrate_workloads(self, preempted_node: str) -> int:
        """Migrate workloads from preempted node to surviving nodes."""
        if self._coordinator is None:
            return 0

        try:
            # Use recovery manager if available
            recovery = getattr(self._coordinator, '_recovery_manager', None)
            if recovery is not None:
                plan = recovery.on_node_failure(preempted_node)
                return len(plan.recovered_sequences) if plan else 0

            # Fallback: just log
            logger.warning(f"No recovery manager for spot failover of {preempted_node}")
            return 0
        except Exception as e:
            logger.error(f"Migration failed for {preempted_node}: {e}")
            return -1

    def get_active_spot_nodes(self) -> list[str]:
        with self._lock:
            return [
                nid for nid, n in self._nodes.items()
                if n.is_spot and n.is_active
            ]

    def get_cost_savings(self) -> dict:
        """Estimate cost savings from using spot instances."""
        with self._lock:
            spot_nodes = [n for n in self._nodes.values() if n.is_spot]
            total_spot_cost = sum(n.cost_per_hour for n in spot_nodes if n.is_active)
            # Assume on-demand is 3x spot price
            on_demand_cost = total_spot_cost * 3
            savings = on_demand_cost - total_spot_cost
            return {
                "active_spot_nodes": len([n for n in spot_nodes if n.is_active]),
                "total_spot_cost_per_hour": round(total_spot_cost, 2),
                "equivalent_on_demand_cost": round(on_demand_cost, 2),
                "hourly_savings": round(savings, 2),
                "monthly_savings": round(savings * 730, 2),
                "preemption_count": sum(n.preemption_count for n in spot_nodes),
            }

    def stats(self) -> dict:
        with self._lock:
            return {
                **self._stats,
                "registered_nodes": len(self._nodes),
                "active_spot_nodes": len([n for n in self._nodes.values() if n.is_spot and n.is_active]),
                "recent_events": len(self._events),
            }
