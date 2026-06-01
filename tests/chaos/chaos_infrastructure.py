"""Chaos engineering infrastructure for real fault injection.

Provides concrete fault injectors that actually modify system behavior,
not just mocks. Used by chaos test scenarios.

Usage::

    injector = NetworkLatencyInjector(latency_ms=200)
    injector.start()
    # ... run test ...
    injector.stop()
"""

from __future__ import annotations

import asyncio
import random
import threading
import time
from typing import Any

from loguru import logger


class NetworkLatencyInjector:
    """Injects artificial network latency into function calls.

    Wraps a callable and adds random latency drawn from a
    configurable distribution.
    """

    def __init__(
        self,
        latency_ms: float = 100.0,
        jitter_ms: float = 50.0,
        target_fn: Any = None,
    ):
        self._base_latency = latency_ms / 1000.0
        self._jitter = jitter_ms / 1000.0
        self._target_fn = target_fn
        self._active = False
        self._injected_count = 0

    def start(self) -> None:
        self._active = True

    def stop(self) -> None:
        self._active = False

    def wrap(self, fn):
        """Wrap a function with latency injection."""
        def wrapper(*args, **kwargs):
            if self._active:
                delay = self._base_latency + random.uniform(0, self._jitter)
                time.sleep(delay)
                self._injected_count += 1
            return fn(*args, **kwargs)
        return wrapper

    async def wrap_async(self, fn):
        """Wrap an async function with latency injection."""
        async def wrapper(*args, **kwargs):
            if self._active:
                delay = self._base_latency + random.uniform(0, self._jitter)
                await asyncio.sleep(delay)
                self._injected_count += 1
            return await fn(*args, **kwargs)
        return wrapper

    @property
    def injected_count(self) -> int:
        return self._injected_count


class NetworkPartitionSimulator:
    """Simulates network partitions by blocking communication between nodes.

    Maintains a partition map — nodes in different partitions cannot
    communicate.
    """

    def __init__(self):
        self._partitions: list[set[str]] = []
        self._active = False

    def create_partition(self, group_a: list[str], group_b: list[str]) -> None:
        """Create a network partition between two groups of nodes."""
        self._partitions = [set(group_a), set(group_b)]
        self._active = True
        logger.warning(
            f"Network partition created: {group_a} | {group_b}"
        )

    def heal(self) -> None:
        """Heal all partitions."""
        self._partitions = []
        self._active = False
        logger.info("Network partition healed")

    def can_communicate(self, node_a: str, node_b: str) -> bool:
        """Check if two nodes can communicate."""
        if not self._active:
            return True
        for partition in self._partitions:
            if node_a in partition and node_b in partition:
                return True
        return False

    @property
    def is_active(self) -> bool:
        return self._active


class MemoryPressureInjector:
    """Simulates GPU memory pressure by tracking allocations."""

    def __init__(self, max_memory_gb: float = 80.0):
        self._max_memory = max_memory_gb * 1024 ** 3
        self._allocated = 0
        self._pressure_events: list[dict] = []

    def allocate(self, bytes_count: int) -> bool:
        """Simulate memory allocation. Returns False if OOM."""
        if self._allocated + bytes_count > self._max_memory:
            self._pressure_events.append({
                "type": "oom",
                "requested": bytes_count,
                "available": self._max_memory - self._allocated,
                "timestamp": time.time(),
            })
            return False
        self._allocated += bytes_count
        return True

    def free(self, bytes_count: int) -> None:
        self._allocated = max(0, self._allocated - bytes_count)

    @property
    def utilization(self) -> float:
        return self._allocated / self._max_memory if self._max_memory > 0 else 0.0

    @property
    def is_under_pressure(self) -> bool:
        return self.utilization > 0.85


class NodeCrashSimulator:
    """Simulates node crashes by marking nodes as unavailable."""

    def __init__(self):
        self._crashed_nodes: set[str] = set()
        self._crash_history: list[dict] = []

    def crash_node(self, node_id: str) -> None:
        """Simulate a node crash."""
        self._crashed_nodes.add(node_id)
        self._crash_history.append({
            "node_id": node_id,
            "action": "crash",
            "timestamp": time.time(),
        })
        logger.warning(f"Node {node_id} crashed (simulated)")

    def recover_node(self, node_id: str) -> None:
        """Simulate a node recovery."""
        self._crashed_nodes.discard(node_id)
        self._crash_history.append({
            "node_id": node_id,
            "action": "recover",
            "timestamp": time.time(),
        })
        logger.info(f"Node {node_id} recovered (simulated)")

    def is_alive(self, node_id: str) -> bool:
        return node_id not in self._crashed_nodes

    @property
    def crashed_nodes(self) -> set[str]:
        return set(self._crashed_nodes)


class ClockSkewSimulator:
    """Simulates clock skew between nodes."""

    def __init__(self):
        self._offsets: dict[str, float] = {}  # node_id -> offset seconds

    def set_skew(self, node_id: str, offset_seconds: float) -> None:
        self._offsets[node_id] = offset_seconds

    def get_time(self, node_id: str) -> float:
        """Get the current time as seen by a node (with skew)."""
        return time.time() + self._offsets.get(node_id, 0.0)

    def clear(self) -> None:
        self._offsets.clear()
