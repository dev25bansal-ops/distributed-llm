"""Tests for CostAwarePriorityAdjuster."""

from __future__ import annotations

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_cost_aware = load_module("distllm/core/advanced_scheduling/cost_aware.py")
CostAwarePriorityAdjuster = _cost_aware.CostAwarePriorityAdjuster


class TestCostAwarePriorityAdjuster:
    """Test suite for CostAwarePriorityAdjuster."""

    def test_default_construction(self) -> None:
        """Default cost_weight is 0.3 and no node costs pre-populated."""
        adjuster = CostAwarePriorityAdjuster()
        assert adjuster._cost_weight == 0.3
        assert adjuster._node_costs == {}

    def test_custom_cost_weight(self) -> None:
        """Custom cost_weight is accepted."""
        adjuster = CostAwarePriorityAdjuster(cost_weight=0.7)
        assert adjuster._cost_weight == 0.7

    def test_update_node_cost(self) -> None:
        """update_node_cost stores cost per hour for a node."""
        adjuster = CostAwarePriorityAdjuster()
        adjuster.update_node_cost("node-a", 12.50)
        assert adjuster._node_costs["node-a"] == 12.50

    def test_update_node_cost_overwrites(self) -> None:
        """Updating an existing node cost replaces the previous value."""
        adjuster = CostAwarePriorityAdjuster()
        adjuster.update_node_cost("node-a", 10.0)
        adjuster.update_node_cost("node-a", 20.0)
        assert adjuster._node_costs["node-a"] == 20.0

    def test_adjust_priority_no_node_id_returns_base(self) -> None:
        """With no node_id, priority and cost are unchanged."""
        adjuster = CostAwarePriorityAdjuster()
        adj, cost = adjuster.adjust_priority(base_priority=5, est_tokens=1000)
        assert adj == 5
        assert cost == 0.0

    def test_adjust_priority_unknown_node_id_returns_base(self) -> None:
        """With an unknown node_id, priority and cost are unchanged."""
        adjuster = CostAwarePriorityAdjuster()
        adjuster.update_node_cost("node-a", 10.0)
        adj, cost = adjuster.adjust_priority(
            base_priority=5, est_tokens=1000, node_id="unknown"
        )
        assert adj == 5
        assert cost == 0.0

    def test_adjust_priority_computes_cost_and_adjustment(self) -> None:
        """Priority is adjusted upward (lowered) on expensive nodes."""
        adjuster = CostAwarePriorityAdjuster(cost_weight=1.0)
        adjuster.update_node_cost("node-a", 50.0)

        adj, cost = adjuster.adjust_priority(base_priority=10, est_tokens=1_000_000, node_id="node-a")

        # cost = (1M / 1M) * 50 = 50.0
        assert cost == 50.0
        # adjustment = 1.0 * (50/10) = 5 -> adjusted = 10 + 5 = 15
        assert adj == 15

    def test_adjust_priority_never_negative(self) -> None:
        """Adjusted priority is clamped to zero if the result would be negative.

        Setting a very high cost_weight with a small cost_per_hour still
        produces a non-negative priority because the adjustment adds to the
        base (it does not subtract).  Clamping to max(0, ...) is a safety
        net for edge cases.
        """
        adjuster = CostAwarePriorityAdjuster(cost_weight=100.0)
        adjuster.update_node_cost("node-a", 100.0)

        # est_tokens=0 means estimated_cost=0, adjustment=cost_weight*(100/10)=1000
        adj, cost = adjuster.adjust_priority(base_priority=1, est_tokens=0, node_id="node-a")
        assert adj == 1001  # 1 + 1000, clamped to max(0, 1001)
        assert cost == 0.0

    def test_adjust_priority_zero_base_priority(self) -> None:
        """Zero base priority works correctly."""
        adjuster = CostAwarePriorityAdjuster()
        adjuster.update_node_cost("node-a", 10.0)

        adj, cost = adjuster.adjust_priority(base_priority=0, est_tokens=100_000, node_id="node-a")
        # adjustment = 0.3 * (10/10) = 0 (floor of int())
        assert adj == 0
        assert cost == 1.0  # (100k/1M) * 10 = 1.0

    def test_adjust_priority_low_cost_node(self) -> None:
        """Cheap nodes get near-original priority."""
        adjuster = CostAwarePriorityAdjuster(cost_weight=0.3)
        adjuster.update_node_cost("node-a", 1.0)

        adj, cost = adjuster.adjust_priority(base_priority=10, est_tokens=500_000, node_id="node-a")
        # adjustment = 0.3 * (1/10) = 0 -> adj = 10
        assert adj == 10
        # cost = (500k/1M) * 1 = 0.5
        assert cost == 0.5

    def test_multiple_nodes_independent_costs(self) -> None:
        """Multiple nodes each have independent cost tracking."""
        adjuster = CostAwarePriorityAdjuster()
        adjuster.update_node_cost("cheap", 5.0)
        adjuster.update_node_cost("expensive", 100.0)

        _, cost_cheap = adjuster.adjust_priority(0, 1_000_000, node_id="cheap")
        _, cost_expensive = adjuster.adjust_priority(0, 1_000_000, node_id="expensive")

        assert cost_cheap == 5.0
        assert cost_expensive == 100.0
