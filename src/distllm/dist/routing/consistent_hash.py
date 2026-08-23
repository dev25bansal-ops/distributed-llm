"""Consistent hash router using the Ketama (MD5-based) algorithm.

Provides deterministic node assignment with minimal remapping when the
node set changes, using virtual nodes for balanced distribution.
"""

from __future__ import annotations

import hashlib
import struct
from typing import Final

_DEFAULT_VNODES: Final[int] = 160


class ConsistentHashRouter:
    """Maps keys to nodes using MD5-based consistent hashing (Ketama).

    Each real node is represented by *weighted* virtual nodes on a 2**32-1
    ring.  ``get_node(key)`` returns the first node encountered when walking
    clockwise from the key's hash position.

    Parameters
    ----------
    virtual_node_count :
        Number of virtual replicas per unit of weight.  Defaults to 160,
        matching the libketama convention.
    """

    def __init__(self, virtual_node_count: int = _DEFAULT_VNODES) -> None:
        self._vnode_count = virtual_node_count
        # Sorted list of (hash, node_id) pairs forming the ring.
        self._ring: list[tuple[int, str]] = []
        self._nodes: dict[str, float] = {}  # node_id -> weight

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_node(self, node_id: str, weight: float = 1.0) -> None:
        """Add a node to the ring with the given *weight*.

        Higher weight values create proportionally more virtual nodes,
        giving the node a larger share of the hash space.
        """
        if node_id in self._nodes:
            raise ValueError(f"Node {node_id!r} already exists")
        if weight <= 0:
            raise ValueError(f"Weight must be positive, got {weight}")

        self._nodes[node_id] = weight
        vnodes = max(1, int(self._vnode_count * weight))

        for v_idx in range(vnodes):
            key = self._hash(f"{node_id}:{v_idx}")
            self._ring.append((key, node_id))

        self._ring.sort(key=lambda x: x[0])

    def remove_node(self, node_id: str) -> None:
        """Remove *node_id* and all of its virtual nodes from the ring."""
        if node_id not in self._nodes:
            raise ValueError(f"Node {node_id!r} not found")

        del self._nodes[node_id]
        self._ring = [(h, n) for h, n in self._ring if n != node_id]

    def get_node(self, key: str) -> str:
        """Return the node responsible for *key*.

        Raises ``RuntimeError`` when no nodes are registered.
        """
        if not self._ring:
            raise RuntimeError("No nodes registered in the hash ring")

        h = self._hash(key)
        # Binary search for the first hash >= h.
        idx = self._find_ge(h)
        return self._ring[idx][1]

    def get_nodes(self, key: str, count: int = 1) -> list[str]:
        """Return up to *count* distinct nodes responsible for *key*.

        Nodes are returned in clockwise order starting from the primary
        replica.  If *count* exceeds the number of available nodes every
        node is returned (no duplicates).
        """
        if not self._ring:
            raise RuntimeError("No nodes registered in the hash ring")

        h = self._hash(key)
        start = self._find_ge(h)
        seen: set[str] = set()
        result: list[str] = []

        # Walk the ring clockwise collecting distinct nodes.
        for i in range(len(self._ring)):
            node_id = self._ring[(start + i) % len(self._ring)][1]
            if node_id not in seen:
                seen.add(node_id)
                result.append(node_id)
                if len(result) >= count:
                    break

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hash(key: str) -> int:
        """Compute a 32-bit MD5 hash (Ketama-style)."""
        md5 = hashlib.md5(key.encode("utf-8"), usedforsecurity=False)
        # Take the first 4 bytes as an unsigned little-endian int.
        return struct.unpack("<I", md5.digest()[:4])[0]

    def _find_ge(self, h: int) -> int:
        """Return the index of the first ring entry with hash >= *h*.

        Wraps around to 0 if *h* is greater than all entries.
        """
        lo, hi = 0, len(self._ring)
        while lo < hi:
            mid = (lo + hi) // 2
            if self._ring[mid][0] < h:
                lo = mid + 1
            else:
                hi = mid
        return lo % len(self._ring)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def nodes(self) -> dict[str, float]:
        """Return a copy of the registered node -> weight mapping."""
        return dict(self._nodes)

    @property
    def ring_size(self) -> int:
        """Return the total number of virtual entries on the ring."""
        return len(self._ring)
