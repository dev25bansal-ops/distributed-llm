"""Cross-cluster federation topology and latency monitoring.

Manages cluster topology, cross-cluster gossip, and latency-aware
routing for distributed LLM inference across data centers.
"""

import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


@dataclass
class ClusterInfo:
    """Represents a cluster/region in the federation.

    Attributes:
        cluster_id: Unique cluster identifier.
        region: Geographic region name.
        nodes: Set of node IDs in this cluster.
        base_latency_ms: Expected intra-cluster latency.
        edge_nodes: Node IDs that handle cross-cluster communication.
    """

    cluster_id: str
    region: str = "unknown"
    nodes: Set[str] = field(default_factory=set)
    base_latency_ms: float = 1.0
    edge_nodes: Set[str] = field(default_factory=set)

    @property
    def size(self) -> int:
        return len(self.nodes)


class FederationManager:
    """Manages multi-cluster topology and node assignments.

    Tracks which nodes belong to which clusters, manages cross-cluster
    discovery, and provides topology-aware routing hints.

    Attributes:
        clusters: Dict mapping cluster_id to ClusterInfo.
        node_to_cluster: Dict mapping node_id to cluster_id.
        local_cluster_id: ID of the local cluster.
        _lock: Threading lock for concurrent access.
    """

    def __init__(self, local_cluster_id: str = "default"):
        self.clusters: Dict[str, ClusterInfo] = {}
        self.node_to_cluster: Dict[str, str] = {}
        self.local_cluster_id = local_cluster_id
        self._lock = threading.Lock()

        # Register local cluster
        self.register_cluster(ClusterInfo(
            cluster_id=local_cluster_id,
            region="local",
        ))

    def register_cluster(self, cluster: ClusterInfo) -> None:
        """Register a new cluster.

        Args:
            cluster: ClusterInfo to register.
        """
        with self._lock:
            self.clusters[cluster.cluster_id] = cluster

    def register_node(self, node_id: str, cluster_id: str, is_edge: bool = False) -> None:
        """Register a node in a cluster.

        Args:
            node_id: Node identifier.
            cluster_id: Cluster this node belongs to.
            is_edge: Whether this node is an edge node for cross-cluster traffic.
        """
        with self._lock:
            self.node_to_cluster[node_id] = cluster_id
            if cluster_id in self.clusters:
                self.clusters[cluster_id].nodes.add(node_id)
                if is_edge:
                    self.clusters[cluster_id].edge_nodes.add(node_id)

    def unregister_node(self, node_id: str) -> None:
        """Remove a node from federation.

        Args:
            node_id: Node to remove.
        """
        with self._lock:
            cluster_id = self.node_to_cluster.pop(node_id, None)
            if cluster_id and cluster_id in self.clusters:
                self.clusters[cluster_id].nodes.discard(node_id)
                self.clusters[cluster_id].edge_nodes.discard(node_id)

    def get_cluster(self, node_id: str) -> Optional[str]:
        """Get the cluster ID for a node.

        Args:
            node_id: Node identifier.

        Returns:
            Cluster ID, or None if not found.
        """
        with self._lock:
            return self.node_to_cluster.get(node_id)

    def get_nodes_in_cluster(self, cluster_id: str) -> Set[str]:
        """Get all node IDs in a cluster.

        Args:
            cluster_id: Cluster identifier.

        Returns:
            Set of node IDs.
        """
        with self._lock:
            if cluster_id in self.clusters:
                return set(self.clusters[cluster_id].nodes)
            return set()

    def get_edge_nodes(self, cluster_id: str) -> Set[str]:
        """Get edge nodes for a cluster.

        Args:
            cluster_id: Cluster identifier.

        Returns:
            Set of edge node IDs.
        """
        with self._lock:
            if cluster_id in self.clusters:
                return set(self.clusters[cluster_id].edge_nodes)
            return set()

    def list_clusters(self) -> List[str]:
        """List all registered cluster IDs.

        Returns:
            List of cluster_id strings.
        """
        with self._lock:
            return list(self.clusters.keys())

    def is_local(self, node_id: str) -> bool:
        """Check if a node is in the local cluster.

        Args:
            node_id: Node identifier.

        Returns:
            True if node is in local cluster.
        """
        cluster = self.get_cluster(node_id)
        return cluster == self.local_cluster_id

    def stats(self) -> dict:
        """Get federation statistics.

        Returns:
            Dict with cluster count, node count, and per-cluster breakdown.
        """
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
    """Measures and tracks cross-cluster network latency.

    Sends periodic pings between cluster edge nodes to measure
    inter-cluster RTT. Provides latency matrix for routing decisions.

    Attributes:
        federation: FederationManager for cluster topology.
        latency_matrix: Dict mapping (src_cluster, dst_cluster) to latency_ms.
        _lock: Threading lock for concurrent access.
    """

    def __init__(self, federation: FederationManager):
        self.federation = federation
        self.latency_matrix: Dict[tuple, float] = {}
        self._lock = threading.Lock()

    def record_latency(self, src_cluster: str, dst_cluster: str, latency_ms: float) -> None:
        """Record a latency measurement.

        Uses exponential moving average to smooth measurements.

        Args:
            src_cluster: Source cluster ID.
            dst_cluster: Destination cluster ID.
            latency_ms: Measured latency in milliseconds.
        """
        with self._lock:
            key = (src_cluster, dst_cluster)
            if key in self.latency_matrix:
                # EMA with alpha=0.3
                old = self.latency_matrix[key]
                self.latency_matrix[key] = old * 0.7 + latency_ms * 0.3
            else:
                self.latency_matrix[key] = latency_ms
            # Symmetric
            self.latency_matrix[(dst_cluster, src_cluster)] = self.latency_matrix[key]

    def get_latency(self, src_cluster: str, dst_cluster: str) -> float:
        """Get current estimated latency between clusters.

        Args:
            src_cluster: Source cluster ID.
            dst_cluster: Destination cluster ID.

        Returns:
            Estimated latency in ms. 0.0 if same cluster, 1000.0 if unknown.
        """
        if src_cluster == dst_cluster:
            return 0.0
        with self._lock:
            return self.latency_matrix.get((src_cluster, dst_cluster), 1000.0)

    def get_latency_to_node(self, node_id: str) -> float:
        """Get estimated latency from local cluster to a node.

        Args:
            node_id: Target node ID.

        Returns:
            Estimated latency in ms.
        """
        local = self.federation.local_cluster_id
        remote_cluster = self.federation.get_cluster(node_id)
        if remote_cluster is None:
            return 1000.0
        return self.get_latency(local, remote_cluster)

    def get_closest_cluster(self, candidate_clusters: List[str]) -> Optional[str]:
        """Find the cluster with lowest latency from local cluster.

        Args:
            candidate_clusters: List of cluster IDs to compare.

        Returns:
            Cluster ID with lowest latency, or None if empty.
        """
        if not candidate_clusters:
            return None
        local = self.federation.local_cluster_id
        return min(
            candidate_clusters,
            key=lambda c: self.get_latency(local, c),
        )

    def stats(self) -> dict:
        """Get latency statistics.

        Returns:
            Dict with latency matrix entries.
        """
        with self._lock:
            return {
                f"{src}->{dst}": ms
                for (src, dst), ms in sorted(self.latency_matrix.items())
            }
