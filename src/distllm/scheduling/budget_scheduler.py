"""Budget-aware scheduler for cost-optimized node selection."""

from loguru import logger


class BudgetScheduler:
    """Selects nodes based on cost constraints and spot availability.

    When cost tracking is enabled, prefers spot instances within budget.
    Falls back to on-demand when spot is unavailable or budget is exceeded.
    """

    def __init__(
        self,
        budget_per_hour: float = 0.0,
        spot_preference: float = 0.8,
        enabled: bool = False,
    ):
        self.enabled = enabled
        self.budget_per_hour = budget_per_hour
        self.spot_preference = spot_preference

    def select_node(
        self,
        candidates: list,
        current_spend: float = 0.0,
    ) -> object | None:
        """Select the best node from candidates based on cost.

        Args:
            candidates: List of node-like objects with is_spot and cost_per_hour attrs.
            current_spend: Current hourly spend across all active nodes.

        Returns:
            The selected node, or None if no candidates.
        """
        if not candidates:
            return None

        if not self.enabled:
            return candidates[0]  # Default: pick first candidate

        # Filter candidates that fit within budget
        if self.budget_per_hour > 0:
            remaining = self.budget_per_hour - current_spend
            within_budget = [n for n in candidates if n.cost_per_hour <= remaining]
            if not within_budget:
                # All candidates exceed budget, pick cheapest
                within_budget = candidates

            candidates = within_budget

        # Prefer spot nodes based on spot_preference probability
        spot_nodes = [n for n in candidates if n.is_spot]
        on_demand_nodes = [n for n in candidates if not n.is_spot]

        if spot_nodes and on_demand_nodes:
            import random
            if random.random() < self.spot_preference:
                candidates = spot_nodes
            else:
                candidates = on_demand_nodes
        elif spot_nodes:
            candidates = spot_nodes
        # else: use on-demand candidates as-is

        # Pick cheapest from the filtered set
        return min(candidates, key=lambda n: n.cost_per_hour)

    def select_nodes_batch(
        self,
        candidates: list,
        count: int,
        current_spend: float = 0.0,
    ) -> list:
        """Select multiple nodes from candidates based on cost.

        Args:
            candidates: List of node-like objects.
            count: Number of nodes to select.
            current_spend: Current hourly spend.

        Returns:
            List of selected nodes (up to count).
        """
        selected = []
        remaining_candidates = list(candidates)
        remaining_spend = current_spend

        for _ in range(min(count, len(candidates))):
            node = self.select_node(remaining_candidates, remaining_spend)
            if node is None:
                break
            selected.append(node)
            remaining_candidates.remove(node)
            remaining_spend += node.cost_per_hour

        return selected
