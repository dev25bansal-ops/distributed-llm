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

    def __init__(
        self,
        kv_cache: Any,
        node_id: str,
        replication_factor: int = 2,
        transport: Any = None,
        peer_nodes: list[str] | None = None,
    ):
        """....

        Args:
            kv_cache: The local cache (anything with ``update``).
            node_id: This node's ID.
            replication_factor: Total copies to maintain (self included).
            transport: M12 — callable ``transport(node_id, request_id,
                layer_idx, key, value) -> bool`` used to push replicas to
                peers. Without it, replication is tracking-only.
            peer_nodes: Cluster membership used to pick replica targets.
        """
        self._cache = kv_cache
        self._node_id = node_id
        self._replication_factor = max(replication_factor, 1)
        self._transport = transport
        self._peer_nodes = list(peer_nodes or [])
        self._replicas: dict[str, CacheReplica] = {}  # key -> replica info
        self._lock = threading.Lock()

    def store(self, request_id: str, layer_idx: int, key: torch.Tensor, value: torch.Tensor) -> None:
        """Store KV cache entry, replicate to peers, and track locations."""
        # Store in local cache
        if hasattr(self._cache, 'update'):
            self._cache.update(layer_idx, key, value)

        # M12: actually push replicas to peers (tracking-only before).
        pushed: list[str] = []
        if self._transport is not None:
            for peer in self._replica_targets():
                try:
                    if self._transport(peer, request_id, layer_idx, key, value):
                        pushed.append(peer)
                except Exception as e:
                    logger.warning(f"Replica push to {peer} failed: {e}")

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
            for peer in pushed:
                if peer not in replica.node_ids:
                    replica.node_ids.append(peer)
            replica.size_bytes = key.numel() * key.element_size() + value.numel() * value.element_size()

    def _replica_targets(self) -> list[str]:
        """Peers to push to: self excluded, factor-1 targets."""
        peers = [p for p in self._peer_nodes if p != self._node_id]
        return peers[: max(self._replication_factor - 1, 0)]

    def recover_from_peers(
        self,
        request_id: str,
        layer_idx: int,
        fetcher: Any = None,
    ) -> bool:
        """Restore a lost local entry from a replica peer.

        Args:
            request_id / layer_idx: The lost entry.
            fetcher: ``fetcher(node_id, request_id, layer_idx) ->
                (key, value) | None``. Defaults to a ``fetch_from_peer``
                attribute on the transport when present.

        Returns:
            True if the entry was found on a peer and restored locally.
        """
        if fetcher is None and self._transport is not None:
            fetcher = getattr(self._transport, "fetch_from_peer", None)
        if fetcher is None:
            logger.warning("recover_from_peers: no fetcher available")
            return False

        cache_key = f"{request_id}:{layer_idx}"
        with self._lock:
            replica = self._replicas.get(cache_key)
            candidates = [
                n for n in (list(replica.node_ids) if replica else [])
                if n != self._node_id
            ] or [p for p in self._peer_nodes if p != self._node_id]

        for peer in candidates:
            try:
                result = fetcher(peer, request_id, layer_idx)
            except Exception as e:
                logger.warning(f"Peer fetch from {peer} failed: {e}")
                continue
            if result is None:
                continue
            key, value = result
            if hasattr(self._cache, "update"):
                self._cache.update(layer_idx, key, value)
            logger.info(f"Recovered cache entry {cache_key} from peer {peer}")
            return True
        return False

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
