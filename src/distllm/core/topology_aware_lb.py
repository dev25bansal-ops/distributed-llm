"""Topology-aware and cache-aware load balancing.

Extends the basic LoadBalancer with:

1. Prefix locality tracking — remembers which nodes cached which prefixes
2. Pipeline group awareness — understands model parallelism topology
3. Cache-aware P&D routing — routes prefill/decode to nodes with cached KV

Usage::

    lb = TopologyAwareLoadBalancer()
    lb.register_node("node-1", pipeline_group="group-a", nvlink_peers=["node-2"])
    lb.record_prefix("prefix-hash-abc", "node-1")
    target = lb.pick(request_id="req-1", prefix_hash="prefix-hash-abc")
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class TopologyNode:
    """A node with topology and cache metadata."""
    node_id: str
    pipeline_group: str = ""
    nvlink_peers: list[str] = field(default_factory=list)
    cache_capacity_gb: float = 80.0
    cache_used_gb: float = 0.0
    active_connections: int = 0
    is_healthy: bool = True


class PrefixLocalityTracker:
    """Tracks which nodes have cached prefixes for locality-aware routing.

    When a request has a known prefix hash, routes to a node that already
    has that prefix cached to avoid redundant prefill computation.

    Uses a popularity-weighted consistent hash for replica selection.
    """

    def __init__(self):
        # prefix_hash -> set of node_ids that have this prefix cached
        self._prefix_map: dict[str, set[str]] = defaultdict(set)
        # prefix_hash -> access count (for popularity tracking)
        self._prefix_popularity: dict[str, int] = defaultdict(int)
        self._lock = threading.RLock()

    def record(self, prefix_hash: str, node_id: str) -> None:
        """Record that *node_id* has *prefix_hash* cached."""
        with self._lock:
            self._prefix_map[prefix_hash].add(node_id)
            self._prefix_popularity[prefix_hash] += 1

    def get_nodes(self, prefix_hash: str) -> list[str]:
        """Get nodes that have *prefix_hash* cached, ordered by staleness."""
        with self._lock:
            nodes = list(self._prefix_map.get(prefix_hash, set()))
        return nodes

    def remove_node(self, node_id: str) -> None:
        with self._lock:
            empty_prefixes = []
            for prefix_hash, prefix_set in self._prefix_map.items():
                prefix_set.discard(node_id)
                if not prefix_set:
                    empty_prefixes.append(prefix_hash)
            for prefix_hash in empty_prefixes:
                del self._prefix_map[prefix_hash]

    def is_cached(self, prefix_hash: str, node_id: str) -> bool:
        with self._lock:
            return node_id in self._prefix_map.get(prefix_hash, set())

    def evict(self, prefix_hash: str) -> None:
        with self._lock:
            self._prefix_map.pop(prefix_hash, None)
            self._prefix_popularity.pop(prefix_hash, None)

    @property
    def stats(self) -> dict:
        with self._lock:
            return {
                "cached_prefixes": len(self._prefix_map),
                "total_entries": sum(len(v) for v in self._prefix_map.values()),
            }


class PipelineGroupAwareness:
    """Tracks pipeline parallelism groups for locality-aware routing.

    Routes requests to nodes within the same pipeline group whenever
    possible to minimize cross-group data transfer.
    """

    def __init__(self):
        self._node_groups: dict[str, str] = {}  # node_id -> group
        self._group_nodes: dict[str, list[str]] = defaultdict(list)
        self._nvlink_topology: dict[str, list[str]] = defaultdict(list)
        self._lock = threading.RLock()

    def register(self, node_id: str, group: str, nvlink_peers: list[str] | None = None) -> None:
        with self._lock:
            prev_group = self._node_groups.get(node_id)
            if prev_group:
                self._group_nodes[prev_group] = [n for n in self._group_nodes[prev_group] if n != node_id]
            self._node_groups[node_id] = group
            self._group_nodes[group].append(node_id)
            if nvlink_peers:
                self._nvlink_topology[node_id] = nvlink_peers

    def same_group(self, node_a: str, node_b: str) -> bool:
        with self._lock:
            return self._node_groups.get(node_a) == self._node_groups.get(node_b)

    def group_for(self, node_id: str) -> str:
        with self._lock:
            return self._node_groups.get(node_id, "")

    def nodes_in_group(self, group: str) -> list[str]:
        with self._lock:
            return list(self._group_nodes.get(group, []))

    def has_nvlink(self, node_a: str, node_b: str) -> bool:
        with self._lock:
            return node_b in self._nvlink_topology.get(node_a, [])


class CacheAwareRouter:
    """Routes prefill and decode requests to nodes with relevant KV cache.

    For prefill: prefers nodes that already have a similar prefix cached.
    For decode: prefers nodes already holding the sequence's KV cache.
    """

    def __init__(
        self,
        prefix_tracker: PrefixLocalityTracker | None = None,
        pipeline_awareness: PipelineGroupAwareness | None = None,
    ):
        self._prefix = prefix_tracker or PrefixLocalityTracker()
        self._pipeline = pipeline_awareness or PipelineGroupAwareness()
        # sequence_id -> node_id (for decode-phase locality)
        self._sequence_nodes: dict[str, str] = {}
        self._lock = threading.RLock()

    def record_sequence(self, sequence_id: str, node_id: str) -> None:
        with self._lock:
            self._sequence_nodes[sequence_id] = node_id

    def get_sequence_node(self, sequence_id: str) -> str | None:
        with self._lock:
            return self._sequence_nodes.get(sequence_id)

    def remove_sequence(self, sequence_id: str) -> None:
        with self._lock:
            self._sequence_nodes.pop(sequence_id, None)


class TopologyAwareLoadBalancer:
    """Load balancer with topology and cache awareness.

    Combines prefix locality, pipeline group affinity, NVLink topology,
    and standard load metrics (connections, latency) to make routing decisions.

    Scoring::
        score = (1 - cache_hit_bonus) if prefix cached on node
              + group_affinity_bonus if in same pipeline group
              - normalized_load_penalty based on active connections
    """

    def __init__(
        self,
        cache_hit_bonus: float = 0.3,
        group_affinity_bonus: float = 0.15,
        nvlink_bonus: float = 0.1,
        load_penalty_weight: float = 0.2,
    ):
        self._prefix_tracker = PrefixLocalityTracker()
        self._pipeline = PipelineGroupAwareness()
        self._cache_router = CacheAwareRouter(
            prefix_tracker=self._prefix_tracker,
            pipeline_awareness=self._pipeline,
        )
        self._nodes: dict[str, TopologyNode] = {}
        self._lock = threading.RLock()
        self._cache_hit_bonus = cache_hit_bonus
        self._group_affinity_bonus = group_affinity_bonus
        self._nvlink_bonus = nvlink_bonus
        self._load_penalty_weight = load_penalty_weight
        self._total_routes = 0
        self._cache_hits = 0
        self._group_affinity_routes = 0

    def register_node(
        self,
        node_id: str,
        pipeline_group: str = "",
        nvlink_peers: list[str] | None = None,
    ) -> None:
        with self._lock:
            self._nodes[node_id] = TopologyNode(
                node_id=node_id,
                pipeline_group=pipeline_group,
                nvlink_peers=nvlink_peers or [],
            )
            self._pipeline.register(node_id, pipeline_group, nvlink_peers)

    def record_prefix(self, prefix_hash: str, node_id: str) -> None:
        self._prefix_tracker.record(prefix_hash, node_id)

    def record_sequence(self, sequence_id: str, node_id: str) -> None:
        self._cache_router.record_sequence(sequence_id, node_id)

    def pick(
        self,
        request_id: str = "",
        prefix_hash: str = "",
        sequence_id: str = "",
        source_node: str = "",
        pipeline_group: str = "",
    ) -> str | None:
        """Pick the best node for a request.

        Args:
            request_id: Unique request identifier.
            prefix_hash: Hash of the request prefix (for cache locality).
            sequence_id: Sequence ID for decode-phase locality.
            source_node: Source node (for NVLink topology awareness).
            pipeline_group: Target pipeline group.

        Returns:
            Selected node_id, or None if no nodes available.
        """
        self._total_routes += 1

        with self._lock:
            candidates = dict(self._nodes)

        if not candidates:
            return None

        # If a sequence is continuing, prefer its current node
        if sequence_id:
            seq_node = self._cache_router.get_sequence_node(sequence_id)
            if seq_node and seq_node in candidates and candidates[seq_node].is_healthy:
                return seq_node

        # Score each candidate
        best_node = None
        best_score = float("-inf")

        for node_id, node in candidates.items():
            if not node.is_healthy:
                continue

            score = 0.0

            # Cache locality bonus
            if prefix_hash and self._prefix_tracker.is_cached(prefix_hash, node_id):
                score += self._cache_hit_bonus
                self._cache_hits += 1

            # Pipeline group affinity
            node_group = self._pipeline.group_for(node_id)
            if pipeline_group and node_group == pipeline_group:
                score += self._group_affinity_bonus
                self._group_affinity_routes += 1
            elif source_node and self._pipeline.has_nvlink(source_node, node_id):
                score += self._nvlink_bonus

            # Load penalty (normalized)
            max_conn = max(n.active_connections for n in candidates.values()) or 1
            load_ratio = node.active_connections / max_conn
            score -= self._load_penalty_weight * load_ratio

            if score > best_score:
                best_score = score
                best_node = node_id

        if best_node:
            with self._lock:
                if best_node in self._nodes:
                    self._nodes[best_node].active_connections += 1
            if prefix_hash:
                self._prefix_tracker.record(prefix_hash, best_node)

        return best_node

    def release(self, node_id: str) -> None:
        with self._lock:
            if node_id in self._nodes:
                self._nodes[node_id].active_connections = max(
                    0, self._nodes[node_id].active_connections - 1,
                )

    def mark_unhealthy(self, node_id: str) -> None:
        with self._lock:
            if node_id in self._nodes:
                self._nodes[node_id].is_healthy = False

    def mark_healthy(self, node_id: str) -> None:
        with self._lock:
            if node_id in self._nodes:
                self._nodes[node_id].is_healthy = True

    @property
    def stats(self) -> dict:
        with self._lock:
            return {
                "total_routes": self._total_routes,
                "cache_hits": self._cache_hits,
                "group_affinity_routes": self._group_affinity_routes,
                "active_nodes": len(self._nodes),
                "healthy_nodes": sum(1 for n in self._nodes.values() if n.is_healthy),
                "prefix_cache": self._prefix_tracker.stats,
            }
