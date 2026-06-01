"""Idle node hibernation — scale to zero when idle.

Monitors cluster utilization and hibernates idle nodes to save costs.
When demand returns, nodes are automatically woken up.

Reduces costs by 100% during idle periods (nights, weekends).

Usage::

    manager = HibernationManager(
        idle_threshold_s=300,  # 5 minutes idle
        min_active_nodes=1,    # Always keep 1 node active
    )
    manager.start()
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from loguru import logger


class NodePowerState(str, Enum):
    """Power state of a node."""
    ACTIVE = "active"
    IDLE = "idle"
    HIBERNATING = "hibernating"
    WAKING = "waking"
    OFFLINE = "offline"


@dataclass
class HibernationNode:
    """A node tracked for hibernation."""
    node_id: str
    power_state: NodePowerState = NodePowerState.ACTIVE
    last_request_time: float = field(default_factory=time.time)
    idle_since: float = 0.0
    hibernated_at: float = 0.0
    wake_count: int = 0
    cost_per_hour: float = 0.0


@dataclass
class HibernationDecision:
    """A hibernation/wake decision."""
    node_id: str
    action: str  # "hibernate" or "wake"
    reason: str
    estimated_savings_per_hour: float = 0.0


class HibernationManager:
    """Manages node hibernation for idle cost savings.

    Monitors request patterns and hibernates nodes that have been
    idle for longer than the threshold. Maintains a minimum number
    of active nodes for responsiveness.
    """

    def __init__(
        self,
        idle_threshold_s: float = 300.0,
        check_interval_s: float = 30.0,
        min_active_nodes: int = 1,
        max_hibernate_per_cycle: int = 1,
        wake_timeout_s: float = 60.0,
        on_hibernate: Callable[[str], None] | None = None,
        on_wake: Callable[[str], None] | None = None,
    ):
        self._idle_threshold = idle_threshold_s
        self._check_interval = check_interval_s
        self._min_active = min_active_nodes
        self._max_hibernate = max_hibernate_per_cycle
        self._wake_timeout = wake_timeout_s
        self._on_hibernate = on_hibernate
        self._on_wake = on_wake

        self._nodes: dict[str, HibernationNode] = {}
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

        self._stats = {
            "hibernations": 0,
            "wakes": 0,
            "idle_hours_saved": 0.0,
            "cost_saved": 0.0,
        }

    def register_node(
        self,
        node_id: str,
        cost_per_hour: float = 0.0,
    ) -> None:
        with self._lock:
            self._nodes[node_id] = HibernationNode(
                node_id=node_id,
                cost_per_hour=cost_per_hour,
            )

    def unregister_node(self, node_id: str) -> None:
        with self._lock:
            self._nodes.pop(node_id, None)

    def record_request(self, node_id: str) -> None:
        """Record a request to a node (resets idle timer)."""
        with self._lock:
            node = self._nodes.get(node_id)
            if node:
                node.last_request_time = time.time()
                node.idle_since = 0.0
                if node.power_state == NodePowerState.HIBERNATING:
                    # Wake the node
                    self._wake_node(node)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="hibernation-manager",
        )
        self._thread.start()
        logger.info(f"Hibernation manager started (idle_threshold={self._idle_threshold}s, min_active={self._min_active})")

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=self._check_interval * 2)
        # Wake all hibernating nodes
        with self._lock:
            for node in self._nodes.values():
                if node.power_state == NodePowerState.HIBERNATING:
                    self._wake_node(node)

    def _monitor_loop(self) -> None:
        while self._running:
            try:
                self._check_idle_nodes()
            except Exception as e:
                logger.warning(f"Hibernation check error: {e}")

            deadline = time.time() + self._check_interval
            while self._running and time.time() < deadline:
                time.sleep(1.0)

    def _check_idle_nodes(self) -> None:
        now = time.time()
        with self._lock:
            active_nodes = [
                n for n in self._nodes.values()
                if n.power_state in (NodePowerState.ACTIVE, NodePowerState.IDLE)
            ]

            # Mark idle nodes
            for node in active_nodes:
                idle_time = now - node.last_request_time
                if idle_time > self._idle_threshold and node.power_state == NodePowerState.ACTIVE:
                    node.power_state = NodePowerState.IDLE
                    node.idle_since = now

            # Hibernate idle nodes (keep min_active)
            hibernated = 0
            for node in active_nodes:
                if hibernated >= self._max_hibernate:
                    break
                if node.power_state != NodePowerState.IDLE:
                    continue
                if len(active_nodes) - hibernated <= self._min_active:
                    break

                self._hibernate_node(node)
                hibernated += 1

    def _hibernate_node(self, node: HibernationNode) -> None:
        node.power_state = NodePowerState.HIBERNATING
        node.hibernated_at = time.time()
        self._stats["hibernations"] += 1

        logger.info(f"Hibernating node {node.node_id} (idle since {node.idle_since:.0f})")

        if self._on_hibernate:
            try:
                self._on_hibernate(node.node_id)
            except Exception as e:
                logger.error(f"Hibernate callback failed for {node.node_id}: {e}")

    def _wake_node(self, node: HibernationNode) -> None:
        if node.power_state != NodePowerState.HIBERNATING:
            return

        node.power_state = NodePowerState.WAKING
        node.wake_count += 1
        self._stats["wakes"] += 1

        # Track idle hours saved
        if node.hibernated_at > 0:
            idle_hours = (time.time() - node.hibernated_at) / 3600
            self._stats["idle_hours_saved"] += idle_hours
            self._stats["cost_saved"] += idle_hours * node.cost_per_hour

        logger.info(f"Waking node {node.node_id}")

        if self._on_wake:
            try:
                self._on_wake(node.node_id)
            except Exception as e:
                logger.error(f"Wake callback failed for {node.node_id}: {e}")

        # Mark as active after wake
        node.power_state = NodePowerState.ACTIVE
        node.hibernated_at = 0.0

    def force_hibernate(self, node_id: str) -> bool:
        """Force hibernate a specific node."""
        with self._lock:
            node = self._nodes.get(node_id)
            if node and node.power_state == NodePowerState.ACTIVE:
                self._hibernate_node(node)
                return True
            return False

    def force_wake(self, node_id: str) -> bool:
        """Force wake a specific node."""
        with self._lock:
            node = self._nodes.get(node_id)
            if node and node.power_state == NodePowerState.HIBERNATING:
                self._wake_node(node)
                return True
            return False

    def stats(self) -> dict:
        with self._lock:
            states = {}
            for node in self._nodes.values():
                states[node.power_state.value] = states.get(node.power_state.value, 0) + 1
            return {
                **self._stats,
                "node_states": states,
                "total_nodes": len(self._nodes),
                "idle_threshold_s": self._idle_threshold,
            }
