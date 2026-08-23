"""Cross-cluster federation topology and latency monitoring.

Manages cluster topology, cross-cluster gossip, and latency-aware
routing for distributed LLM inference across data centers.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


@dataclass
class ClusterInfo:
    cluster_id: str
    region: str = "unknown"
    nodes: Set[str] = field(default_factory=set)
    base_latency_ms: float = 1.0
    edge_nodes: Set[str] = field(default_factory=set)

    @property
    def size(self) -> int:
        return len(self.nodes)


class FederationManager:
    def __init__(self, local_cluster_id: str = "default"):
        self.clusters: Dict[str, ClusterInfo] = {}
        self.node_to_cluster: Dict[str, str] = {}
        self.local_cluster_id = local_cluster_id
        self._lock = threading.Lock()

        self.register_cluster(ClusterInfo(
            cluster_id=local_cluster_id,
            region="local",
        ))

    def register_cluster(self, cluster: ClusterInfo) -> None:
        with self._lock:
            self.clusters[cluster.cluster_id] = cluster

    def register_node(self, node_id: str, cluster_id: str, is_edge: bool = False) -> None:
        with self._lock:
            self.node_to_cluster[node_id] = cluster_id
            if cluster_id in self.clusters:
                self.clusters[cluster_id].nodes.add(node_id)
                if is_edge:
                    self.clusters[cluster_id].edge_nodes.add(node_id)

    def unregister_node(self, node_id: str) -> None:
        with self._lock:
            cluster_id = self.node_to_cluster.pop(node_id, None)
            if cluster_id and cluster_id in self.clusters:
                self.clusters[cluster_id].nodes.discard(node_id)
                self.clusters[cluster_id].edge_nodes.discard(node_id)

    def get_cluster(self, node_id: str) -> Optional[str]:
        with self._lock:
            return self.node_to_cluster.get(node_id)

    def get_nodes_in_cluster(self, cluster_id: str) -> Set[str]:
        with self._lock:
            if cluster_id in self.clusters:
                return set(self.clusters[cluster_id].nodes)
            return set()

    def get_edge_nodes(self, cluster_id: str) -> Set[str]:
        with self._lock:
            if cluster_id in self.clusters:
                return set(self.clusters[cluster_id].edge_nodes)
            return set()

    def list_clusters(self) -> List[str]:
        with self._lock:
            return list(self.clusters.keys())

    def is_local(self, node_id: str) -> bool:
        cluster = self.get_cluster(node_id)
        return cluster == self.local_cluster_id

    def stats(self) -> dict:
        with self._lock:
            return {
                "cluster_count": len(self.clusters),
                "node_count": len(self.node_to_cluster),
                "clusters": {
                    cid: {
                        "region": c.region,
                        "nodes": len(c.nodes),
                        "edge_nodes": len(c.edge_nodes),
                    }
                    for cid, c in self.clusters.items()
                },
            }


class CrossClusterLatencyMonitor:
    def __init__(self, federation: FederationManager):
        self.federation = federation
        self.latency_matrix: Dict[tuple, float] = {}
        self._lock = threading.Lock()

    def record_latency(self, src_cluster: str, dst_cluster: str, latency_ms: float) -> None:
        with self._lock:
            key = (src_cluster, dst_cluster)
            if key in self.latency_matrix:
                old = self.latency_matrix[key]
                self.latency_matrix[key] = old * 0.7 + latency_ms * 0.3
            else:
                self.latency_matrix[key] = latency_ms
            self.latency_matrix[(dst_cluster, src_cluster)] = self.latency_matrix[key]

    def get_latency(self, src_cluster: str, dst_cluster: str) -> float:
        if src_cluster == dst_cluster:
            return 0.0
        with self._lock:
            return self.latency_matrix.get((src_cluster, dst_cluster), 1000.0)

    def get_latency_to_node(self, node_id: str) -> float:
        local = self.federation.local_cluster_id
        remote_cluster = self.federation.get_cluster(node_id)
        if remote_cluster is None:
            return 1000.0
        return self.get_latency(local, remote_cluster)

    def get_closest_cluster(self, candidate_clusters: List[str]) -> Optional[str]:
        if not candidate_clusters:
            return None
        local = self.federation.local_cluster_id
        return min(
            candidate_clusters,
            key=lambda c: self.get_latency(local, c),
        )

    def stats(self) -> dict:
        with self._lock:
            return {
                f"{src}->{dst}": ms
                for (src, dst), ms in sorted(self.latency_matrix.items())
            }
