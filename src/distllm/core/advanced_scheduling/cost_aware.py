"""Cost-aware scheduling — adjust priorities based on node cost."""

from __future__ import annotations

from typing import Any


class CostAwarePriorityAdjuster:
    """Adjusts sequence priorities based on node cost.

    Routes low-priority requests to cheaper nodes and high-priority
    requests to faster nodes.
    """

    def __init__(self, cost_weight: float = 0.3):
        self._cost_weight = cost_weight
        self._node_costs: dict[str, float] = {}

    def update_node_cost(self, node_id: str, cost_per_hour: float) -> None:
        self._node_costs[node_id] = cost_per_hour

    def adjust_priority(
        self,
        base_priority: int,
        est_tokens: int,
        node_id: str | None = None,
    ) -> tuple[int, float]:
        """Adjust priority and return (adjusted_priority, estimated_cost)."""
        if not node_id or node_id not in self._node_costs:
            return base_priority, 0.0

        cost_per_hour = self._node_costs[node_id]
        est_cost = (est_tokens / 1_000_000) * cost_per_hour

        # Higher cost = slightly lower priority (prefer cheaper nodes)
        adjustment = int(self._cost_weight * (cost_per_hour / 10.0))
        adjusted = max(0, base_priority + adjustment)

        return adjusted, est_cost
