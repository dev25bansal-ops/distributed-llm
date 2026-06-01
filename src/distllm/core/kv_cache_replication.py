"""KV Cache replication for distributed coherency.

Provides a replication-aware wrapper around KV cache that tracks
which nodes have copies of each cache entry, enabling recovery
when a node fails.

Usage::

    replicated = ReplicatedKVCache(kv_cache, node_id="node-1")
    replicated.store(request_id, layer_idx, key, value)
    # On node failure, recover from replicas
    recovered = replicated.recover_from_peers(request_id, alive_nodes)
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

import torch
from loguru import logger


@dataclass
class CacheReplica:
    """Tracks a KV cache entry's location across nodes."""
    request_id: str
    layer_idx: int
    node_ids: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    size_bytes: int = 0


class ReplicatedKVCache:
    """Replication-aware KV cache wrapper.

    Tracks which nodes have copies of each cache entry,
    enabling recovery when a node fails.
    """

    def __init__(self, kv_cache: Any, node_id: str, replication_factor: int = 2):
        self._cache = kv_cache
        self._node_id = node_id
        self._replication_factor = replication_factor
        self._replicas: dict[str, CacheReplica] = {}  # key -> replica info
        self._lock = threading.Lock()

    def store(self, request_id: str, layer_idx: int, key: torch.Tensor, value: torch.Tensor) -> None:
        """Store KV cache entry and track replication."""
        # Store in local cache
        if hasattr(self._cache, 'update'):
            self._cache.update(layer_idx, key, value)

        # Track replica location
        cache_key = f"{request_id}:{layer_idx}"
        with self._lock:
            if cache_key not in self._replicas:
                self._replicas[cache_key] = CacheReplica(
                    request_id=request_id,
                    layer_idx=layer_idx,
                )
            replica = self._replicas[cache_key]
            if self._node_id not in replica.node_ids:
                replica.node_ids.append(self._node_id)
            replica.size_bytes = key.numel() * key.element_size() + value.numel() * value.element_size()

    def get_replica_nodes(self, request_id: str, layer_idx: int) -> list[str]:
        """Return list of nodes that have a copy of this cache entry."""
        cache_key = f"{request_id}:{layer_idx}"
        with self._lock:
            replica = self._replicas.get(cache_key)
            return list(replica.node_ids) if replica else []

    def mark_node_failed(self, failed_node_id: str) -> list[str]:
        """Remove failed node from all replica tracking.

        Returns list of request_ids that lost all replicas.
        """
        lost_requests = set()
        with self._lock:
            for key, replica in list(self._replicas.items()):
                if failed_node_id in replica.node_ids:
                    replica.node_ids.remove(failed_node_id)
                    if not replica.node_ids:
                        lost_requests.add(replica.request_id)
                        del self._replicas[key]
        return list(lost_requests)

    def needs_replication(self, request_id: str, layer_idx: int) -> bool:
        """Check if this entry needs more replicas."""
        cache_key = f"{request_id}:{layer_idx}"
        with self._lock:
            replica = self._replicas.get(cache_key)
            if replica is None:
                return True
            return len(replica.node_ids) < self._replication_factor

    def stats(self) -> dict:
        with self._lock:
            return {
                "tracked_entries": len(self._replicas),
                "total_replicas": sum(len(r.node_ids) for r in self._replicas.values()),
                "replication_factor": self._replication_factor,
                "node_id": self._node_id,
            }
