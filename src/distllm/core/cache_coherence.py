"""F3: Distributed cache coherence protocol.

Vector-clock-based coherence for P2P cache invalidation.
Ensures eventual consistency and prevents serving stale KV caches.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from loguru import logger


class CacheCoherenceProtocol:
    """Vector-clock-based cache coherence for distributed P2P caching.

    Tracks version vectors for each cached prefix to detect and
    invalidate stale entries across the gossip ring.
    """

    def __init__(self, node_id: str):
        self._node_id = node_id
        # prefix_hash -> {node_id -> vector_clock_value}
        self._vector_clocks: dict[str, dict[str, int]] = {}
        # prefix_hash -> timestamp of last invalidation
        self._invalidated: dict[str, float] = {}
        self._lock = threading.Lock()

    def on_store(self, prefix_hash: str) -> int:
        """Record a store operation and return the new vector clock value.

        Args:
            prefix_hash: Hash of the cached prefix.

        Returns:
            The new vector clock value for this node.
        """
        with self._lock:
            if prefix_hash not in self._vector_clocks:
                self._vector_clocks[prefix_hash] = {}
            current = self._vector_clocks[prefix_hash].get(self._node_id, 0)
            new_clock = current + 1
            self._vector_clocks[prefix_hash][self._node_id] = new_clock
            return new_clock

    def on_receive(self, prefix_hash: str, remote_node_id: str, remote_clock: int) -> bool:
        """Process a remote clock update.

        Args:
            prefix_hash: Hash of the cached prefix.
            remote_node_id: ID of the remote node.
            remote_clock: Remote node's vector clock value.

        Returns:
            True if the remote entry is newer (should update local cache).
        """
        with self._lock:
            if prefix_hash not in self._vector_clocks:
                self._vector_clocks[prefix_hash] = {}

            local_clock = self._vector_clocks[prefix_hash].get(remote_node_id, 0)
            if remote_clock > local_clock:
                self._vector_clocks[prefix_hash][remote_node_id] = remote_clock
                return True
            return False

    def is_stale(self, prefix_hash: str, remote_clocks: dict[str, int]) -> bool:
        """Check if a remote entry is stale compared to local knowledge.

        Args:
            prefix_hash: Hash of the cached prefix.
            remote_clocks: Remote node's full vector clock for this prefix.

        Returns:
            True if the remote entry is stale.
        """
        with self._lock:
            local_clocks = self._vector_clocks.get(prefix_hash, {})
            # Check if any local clock is ahead of the remote clock
            for node_id, local_val in local_clocks.items():
                remote_val = remote_clocks.get(node_id, 0)
                if local_val > remote_val:
                    return True
            return False

    def invalidate(self, prefix_hash: str) -> None:
        """Invalidate a prefix across all nodes.

        Args:
            prefix_hash: Hash of the prefix to invalidate.
        """
        with self._lock:
            # Increment our clock to mark this as the latest version
            if prefix_hash not in self._vector_clocks:
                self._vector_clocks[prefix_hash] = {}
            current = self._vector_clocks[prefix_hash].get(self._node_id, 0)
            self._vector_clocks[prefix_hash][self._node_id] = current + 1
            self._invalidated[prefix_hash] = time.time()

    def get_clocks(self, prefix_hash: str) -> dict[str, int]:
        """Get the vector clock for a prefix."""
        with self._lock:
            return dict(self._vector_clocks.get(prefix_hash, {}))

    def get_invalidated_since(self, since: float) -> list[str]:
        """Get prefixes invalidated since a timestamp."""
        with self._lock:
            return [
                h for h, t in self._invalidated.items()
                if t > since
            ]

    def cleanup_old_entries(self, max_age_seconds: float = 3600.0) -> int:
        """Remove old vector clock entries.

        Returns:
            Number of entries removed.
        """
        with self._lock:
            cutoff = time.time() - max_age_seconds
            to_remove = [
                h for h, t in self._invalidated.items()
                if t < cutoff
            ]
            for h in to_remove:
                del self._invalidated[h]
                self._vector_clocks.pop(h, None)
            return len(to_remove)

    def stats(self) -> dict:
        """Return coherence protocol statistics."""
        with self._lock:
            return {
                "tracked_prefixes": len(self._vector_clocks),
                "invalidated_prefixes": len(self._invalidated),
                "node_id": self._node_id,
            }
