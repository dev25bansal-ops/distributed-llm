"""Chaos scenario: network partition.

Simulates a network partition where a subset of nodes become unreachable.
Verifies that the system continues to serve requests with reduced capacity
and recovers when the partition heals.
"""

import threading
import time
from typing import Callable, Optional

from loguru import logger


class NetworkPartitionSimulator:
    """Simulates network partitions by blocking specific node communication.

    Usage:
        sim = NetworkPartitionSimulator()
        sim.isolate_node("node-1")  # Block all communication to node-1
        # ... run tests ...
        sim.heal_partition()  # Restore communication
    """

    def __init__(self):
        self._isolated: set[str] = set()
        self._lock = threading.Lock()
        self._block_hooks: list[Callable] = []

    def register_block_hook(self, hook: Callable[[str], bool]) -> None:
        """Register a hook that checks if a node should be blocked.

        The hook receives a node_id and returns True if the message should
        be blocked (node is partitioned).
        """
        self._block_hooks.append(hook)

    def isolate_node(self, node_id: str) -> None:
        """Isolate a single node."""
        with self._lock:
            self._isolated.add(node_id)
        logger.warning(f"Network partition: isolated node {node_id}")

    def isolate_nodes(self, node_ids: list[str]) -> None:
        """Isolate multiple nodes."""
        with self._lock:
            self._isolated.update(node_ids)
        logger.warning(f"Network partition: isolated nodes {node_ids}")

    def heal_partition(self) -> None:
        """Restore communication to all previously isolated nodes."""
        with self._lock:
            healed = list(self._isolated)
            self._isolated.clear()
        for node_id in healed:
            logger.warning(f"Network partition healed: {node_id} restored")

    def is_isolated(self, node_id: str) -> bool:
        with self._lock:
            return node_id in self._isolated

    def should_block(self, node_id: str) -> bool:
        """Check if a message to node_id should be blocked."""
        if self.is_isolated(node_id):
            return True
        for hook in self._block_hooks:
            if hook(node_id):
                return True
        return False

    def partition_half(self, nodes: list[str]) -> None:
        """Isolate the second half of the node list (simulates rack failure)."""
        mid = len(nodes) // 2
        self.isolate_nodes(nodes[mid:])

    @property
    def isolated_count(self) -> int:
        with self._lock:
            return len(self._isolated)


def test_partition_isolation():
    sim = NetworkPartitionSimulator()
    sim.isolate_node("node-1")
    assert sim.should_block("node-1")
    assert not sim.should_block("node-0")
    sim.heal_partition()
    assert not sim.should_block("node-1")


def test_partition_half():
    sim = NetworkPartitionSimulator()
    sim.partition_half(["node-0", "node-1", "node-2", "node-3"])
    assert not sim.should_block("node-0")
    assert not sim.should_block("node-1")
    assert sim.should_block("node-2")
    assert sim.should_block("node-3")
    assert sim.isolated_count == 2
