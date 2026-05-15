"""Consistent hashing ring for the distributed-llm router.

Provides sticky session routing with minimal remapping
when nodes join or leave the cluster.
"""

import hashlib
from bisect import bisect_left
from typing import Dict, List, Optional


class ConsistentHashRing:
    """Consistent hash ring with virtual nodes.

    Each physical node is represented by `replicas` virtual nodes
    distributed around the ring for balanced load distribution.
    """

    def __init__(self, replicas: int = 150):
        self._replicas = replicas
        self._ring: List[int] = []
        self._node_map: Dict[int, str] = {}
        self._nodes: set = set()

    def add_node(self, node_id: str) -> None:
        if node_id in self._nodes:
            return
        self._nodes.add(node_id)
        for i in range(self._replicas):
            key = self._hash(f"{node_id}:{i}")
            self._ring.append(key)
            self._node_map[key] = node_id
        self._ring.sort()

    def remove_node(self, node_id: str) -> None:
        if node_id not in self._nodes:
            return
        self._nodes.discard(node_id)
        keys_to_remove = [k for k, v in self._node_map.items() if v == node_id]
        for k in keys_to_remove:
            self._ring.remove(k)
            del self._node_map[k]

    def get_node(self, key: str) -> Optional[str]:
        if not self._ring:
            return None
        hash_key = self._hash(key)
        idx = bisect_left(self._ring, hash_key) % len(self._ring)
        return self._node_map[self._ring[idx]]

    def get_node_with_fallback(
        self, key: str, healthy_nodes: Optional[set] = None
    ) -> Optional[str]:
        """Get node for key, walking the ring until a healthy node is found."""
        if not self._ring:
            return None

        hash_key = self._hash(key)
        idx = bisect_left(self._ring, hash_key)

        for _ in range(len(self._ring)):
            node = self._node_map[self._ring[idx % len(self._ring)]]
            if healthy_nodes is None or node in healthy_nodes:
                return node
            idx += 1

        return None

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    def _hash(self, key: str) -> int:
        return int(hashlib.sha256(key.encode()).hexdigest(), 16)
