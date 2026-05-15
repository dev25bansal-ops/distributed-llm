"""Cost tracking for distributed-llm nodes."""

import time
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class NodeCostInfo:
    """Cost information for a single node."""
    node_id: str
    cost_per_hour: float
    is_spot: bool
    instance_type: str
    total_cost: float = 0.0
    start_time: float = 0.0


class CostTracker:
    """Tracks per-node and aggregate cost for the distributed system.

    Attributes:
        budget_per_hour: Maximum allowed spend per hour.
        _nodes: Dict of node_id -> NodeCostInfo.
        _start_times: When each node started accruing cost.
    """

    def __init__(self, budget_per_hour: float = 0.0):
        self.budget_per_hour = budget_per_hour
        self._nodes: Dict[str, NodeCostInfo] = {}
        self._interruption_count = 0

    def register_node(
        self,
        node_id: str,
        cost_per_hour: float = 0.0,
        is_spot: bool = False,
        instance_type: str = "unknown",
    ) -> None:
        """Register a node for cost tracking.

        Args:
            node_id: Unique node identifier.
            cost_per_hour: Hourly cost of the node.
            is_spot: Whether this is a spot instance.
            instance_type: Cloud instance type (e.g., "g5.xlarge").
        """
        self._nodes[node_id] = NodeCostInfo(
            node_id=node_id,
            cost_per_hour=cost_per_hour,
            is_spot=is_spot,
            instance_type=instance_type,
            start_time=time.time(),
        )

    def unregister_node(self, node_id: str) -> Optional[NodeCostInfo]:
        """Unregister a node and return its final cost info."""
        node = self._nodes.pop(node_id, None)
        if node:
            node.total_cost = self._calculate_node_cost(node)
        return node

    def _calculate_node_cost(self, node: NodeCostInfo) -> float:
        """Calculate total cost accrued by a node."""
        elapsed_hours = (time.time() - node.start_time) / 3600.0
        return node.cost_per_hour * elapsed_hours

    def get_node_cost(self, node_id: str) -> float:
        """Get total cost accrued by a specific node."""
        node = self._nodes.get(node_id)
        if node is None:
            return 0.0
        return self._calculate_node_cost(node)

    def get_current_hourly_spend(self) -> float:
        """Get the current total hourly spend across all nodes."""
        return sum(n.cost_per_hour for n in self._nodes.values())

    def get_budget_remaining(self) -> float:
        """Get remaining budget per hour."""
        if self.budget_per_hour <= 0:
            return 0.0
        return max(0.0, self.budget_per_hour - self.get_current_hourly_spend())

    def is_within_budget(self) -> bool:
        """Check if current spend is within budget."""
        if self.budget_per_hour <= 0:
            return True  # No budget set
        return self.get_current_hourly_spend() <= self.budget_per_hour

    def get_total_accrued_cost(self) -> float:
        """Get total cost accrued across all nodes."""
        return sum(self._calculate_node_cost(n) for n in self._nodes.values())

    def record_spot_interruption(self) -> None:
        """Record a spot instance interruption event."""
        self._interruption_count += 1

    @property
    def interruption_count(self) -> int:
        return self._interruption_count

    def get_nodes_by_cost(self, ascending: bool = True) -> list:
        """Get all nodes sorted by cost."""
        nodes = list(self._nodes.values())
        nodes.sort(key=lambda n: n.cost_per_hour, reverse=not ascending)
        return nodes

    def get_spot_nodes(self) -> list:
        """Get all spot nodes."""
        return [n for n in self._nodes.values() if n.is_spot]

    def get_on_demand_nodes(self) -> list:
        """Get all on-demand nodes."""
        return [n for n in self._nodes.values() if not n.is_spot]

    def get_stats(self) -> dict:
        """Get cost tracking statistics."""
        return {
            "total_nodes": len(self._nodes),
            "spot_nodes": len(self.get_spot_nodes()),
            "on_demand_nodes": len(self.get_on_demand_nodes()),
            "hourly_spend": self.get_current_hourly_spend(),
            "budget_remaining": self.get_budget_remaining(),
            "total_accrued": self.get_total_accrued_cost(),
            "interruptions": self._interruption_count,
            "within_budget": self.is_within_budget(),
        }
