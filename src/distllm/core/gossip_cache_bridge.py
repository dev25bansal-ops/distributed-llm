"""Gossip-KV Cache integration bridge with distributed replication.

Connects the P2P gossip protocol with the KV cache system to enable
distributed cache discovery, sharing, and replication across nodes.

Features:
- Automatic cache entry advertisement via gossip
- Cross-node prefix discovery for request routing
- Replication factor tracking (ensures N copies exist)
- Eviction propagation (tombstones)

Usage::

    bridge = GossipCacheBridge(
        gossip=GossipProtocol("node-1"),
        cache_manager=CacheManager(),
        replication_factor=2,
    )
    bridge.start()
    # KV cache entries are automatically advertised and replicated
"""

from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from loguru import logger


@dataclass
class CacheReplicaInfo:
    """Tracks replication state for a cache entry."""
    prefix_hash: str
    source_node: str
    replica_nodes: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    size_bytes: int = 0


class GossipCacheBridge:
    """Bridges gossip protocol with KV cache for distributed cache discovery.

    When a node stores a KV cache entry, it advertises the prefix hash
    via gossip. Other nodes can then discover where cached prefixes
    are located and route requests accordingly.
    """

    def __init__(
        self,
        gossip: Any = None,
        cache_manager: Any = None,
        advertise_interval_s: float = 10.0,
        replication_factor: int = 2,
        on_replicate: Callable[[str, str], None] | None = None,
    ):
        self._gossip = gossip
        self._cache_manager = cache_manager
        self._advertise_interval = advertise_interval_s
        self._replication_factor = replication_factor
        self._on_replicate = on_replicate
        self._running = False
        self._thread: threading.Thread | None = None
        self._advertised_prefixes: set[str] = set()
        self._replicas: dict[str, CacheReplicaInfo] = {}  # prefix_hash -> replica info
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start the gossip-cache bridge."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._advertise_loop,
            daemon=True,
            name="gossip-cache-bridge",
        )
        self._thread.start()
        logger.info(f"Gossip-Cache bridge started (replication_factor={self._replication_factor})")

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

    def on_cache_store(self, prefix_hash: str, node_id: str, size_bytes: int = 0) -> None:
        """Called when a KV cache entry is stored locally.

        Advertises via gossip and tracks replication.
        """
        if self._gossip is None:
            return
        with self._lock:
            if prefix_hash not in self._advertised_prefixes:
                self._gossip.store_local(prefix_hash, node_id)
                self._advertised_prefixes.add(prefix_hash)

            # Track replica
            if prefix_hash not in self._replicas:
                self._replicas[prefix_hash] = CacheReplicaInfo(
                    prefix_hash=prefix_hash,
                    source_node=node_id,
                    size_bytes=size_bytes,
                )
            info = self._replicas[prefix_hash]
            if node_id not in info.replica_nodes:
                info.replica_nodes.append(node_id)

    def on_cache_evict(self, prefix_hash: str) -> None:
        """Called when a KV cache entry is evicted.

        Removes from advertised set and updates replica tracking.
        """
        with self._lock:
            self._advertised_prefixes.discard(prefix_hash)
            # Don't delete replica info — other nodes may still have it

    def discover_prefix(self, prefix_hash: str) -> list[str]:
        """Discover which nodes have a cached prefix via gossip."""
        if self._gossip is None:
            return []
        with self._lock:
            entries = self._gossip.state.cache_index.get(prefix_hash, [])
            return [nid for nid, _, _ in entries]

    def get_replica_count(self, prefix_hash: str) -> int:
        """Return the number of replicas for a cache entry."""
        with self._lock:
            info = self._replicas.get(prefix_hash)
            return len(info.replica_nodes) if info else 0

    def needs_replication(self, prefix_hash: str) -> bool:
        """Check if a cache entry needs more replicas."""
        with self._lock:
            info = self._replicas.get(prefix_hash)
            if info is None:
                return True
            return len(info.replica_nodes) < self._replication_factor

    def get_under_replicated(self) -> list[str]:
        """Return list of prefix hashes that need more replicas."""
        with self._lock:
            return [
                h for h, info in self._replicas.items()
                if len(info.replica_nodes) < self._replication_factor
            ]

    def mark_node_failed(self, failed_node_id: str) -> list[str]:
        """Remove failed node from all replica tracking.

        Returns list of prefix hashes that lost all replicas.
        """
        lost = []
        with self._lock:
            for prefix_hash, info in list(self._replicas.items()):
                if failed_node_id in info.replica_nodes:
                    info.replica_nodes.remove(failed_node_id)
                    if not info.replica_nodes:
                        lost.append(prefix_hash)
                        del self._replicas[prefix_hash]
        return lost

    def _advertise_loop(self) -> None:
        """Periodically advertise local cache entries via gossip."""
        while self._running:
            try:
                if self._gossip is not None:
                    self._gossip.advertise(delta_only=True)

                # Check for under-replicated entries
                under_replicated = self.get_under_replicated()
                if under_replicated and self._on_replicate:
                    for prefix_hash in under_replicated[:5]:  # Limit per cycle
                        try:
                            self._on_replicate(prefix_hash, self._replicas[prefix_hash].source_node)
                        except Exception as e:
                            logger.debug(f"Replication callback failed: {e}")

            except Exception as e:
                logger.debug(f"Gossip advertise failed: {e}")

            deadline = time.time() + self._advertise_interval
            while self._running and time.time() < deadline:
                time.sleep(1.0)

    def stats(self) -> dict:
        with self._lock:
            under_rep = sum(
                1 for info in self._replicas.values()
                if len(info.replica_nodes) < self._replication_factor
            )
            return {
                "running": self._running,
                "advertised_prefixes": len(self._advertised_prefixes),
                "tracked_replicas": len(self._replicas),
                "under_replicated": under_rep,
                "replication_factor": self._replication_factor,
                "gossip_connected": self._gossip is not None,
            }
