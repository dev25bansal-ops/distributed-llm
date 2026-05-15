"""Thread-safe expert-to-node mapping for distributed MoE inference."""

import threading
from typing import Dict, List, Optional


class ExpertRegistry:
    """Thread-safe registry mapping experts to worker nodes.

    Tracks which nodes hold which experts and supports load-aware
    node selection for expert routing.
    """

    def __init__(self):
        self._lock = threading.Lock()
        # expert_id -> list of (node_id, layer_idx)
        self._expert_to_nodes: Dict[int, List[tuple]] = {}
        # node_id -> list of (expert_id, layer_idx)
        self._node_to_experts: Dict[str, List[tuple]] = {}
        # node_id -> request count (for load balancing)
        self._node_load: Dict[str, int] = {}

    def register_expert(self, expert_id: int, node_id: str, layer_idx: int) -> None:
        """Register an expert on a node.

        Args:
            expert_id: Expert identifier.
            node_id: Node that holds this expert.
            layer_idx: Layer index the expert belongs to.
        """
        with self._lock:
            if expert_id not in self._expert_to_nodes:
                self._expert_to_nodes[expert_id] = []
            self._expert_to_nodes[expert_id].append((node_id, layer_idx))

            if node_id not in self._node_to_experts:
                self._node_to_experts[node_id] = []
            self._node_to_experts[node_id].append((expert_id, layer_idx))

            if node_id not in self._node_load:
                self._node_load[node_id] = 0

    def unregister_node(self, node_id: str) -> None:
        """Remove all experts registered on a node.

        Args:
            node_id: Node to remove.
        """
        with self._lock:
            # Remove from expert_to_nodes
            for expert_id in list(self._expert_to_nodes.keys()):
                self._expert_to_nodes[expert_id] = [
                    (nid, lidx)
                    for nid, lidx in self._expert_to_nodes[expert_id]
                    if nid != node_id
                ]
                if not self._expert_to_nodes[expert_id]:
                    del self._expert_to_nodes[expert_id]

            # Remove from node_to_experts
            self._node_to_experts.pop(node_id, None)
            self._node_load.pop(node_id, None)

    def get_expert_nodes(self, expert_id: int) -> List[str]:
        """Get all nodes that hold a specific expert.

        Args:
            expert_id: Expert identifier.

        Returns:
            List of node IDs (empty if expert not found).
        """
        with self._lock:
            entries = self._expert_to_nodes.get(expert_id, [])
            return list(set(nid for nid, _ in entries))

    def get_node_experts(self, node_id: str) -> List[int]:
        """Get all experts hosted on a node.

        Args:
            node_id: Node identifier.

        Returns:
            List of expert IDs.
        """
        with self._lock:
            entries = self._node_to_experts.get(node_id, [])
            return [eid for eid, _ in entries]

    def list_all(self) -> Dict[int, List[str]]:
        """Get full expert-to-node mapping.

        Returns:
            Dict mapping expert_id to list of node IDs.
        """
        with self._lock:
            return {
                eid: list(set(nid for nid, _ in nodes))
                for eid, nodes in self._expert_to_nodes.items()
            }

    def select_best_node(self, expert_id: int) -> Optional[str]:
        """Select the least-loaded node for an expert.

        Args:
            expert_id: Expert identifier.

        Returns:
            Best node ID, or None if expert not found.
        """
        with self._lock:
            nodes = self._expert_to_nodes.get(expert_id, [])
            if not nodes:
                return None

            # Pick node with lowest load
            node_ids = list(set(nid for nid, _ in nodes))
            return min(node_ids, key=lambda n: self._node_load.get(n, 0))

    def record_request(self, node_id: str) -> None:
        """Increment load counter for a node.

        Args:
            node_id: Node that received a request.
        """
        with self._lock:
            self._node_load[node_id] = self._node_load.get(node_id, 0) + 1

    def release_request(self, node_id: str) -> None:
        """Decrement load counter for a node.

        Args:
            node_id: Node that completed a request.
        """
        with self._lock:
            if node_id in self._node_load:
                self._node_load[node_id] = max(0, self._node_load[node_id] - 1)

    def stats(self) -> dict:
        """Return registry statistics.

        Returns:
            Dict with total_experts, total_nodes, load_distribution.
        """
        with self._lock:
            return {
                "total_experts": len(self._expert_to_nodes),
                "total_nodes": len(self._node_to_experts),
                "load_distribution": dict(self._node_load),
            }
