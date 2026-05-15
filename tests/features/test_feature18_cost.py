"""Tests for Feature 18: Cost-Aware Scheduling."""

import time
from unittest.mock import MagicMock, patch

import pytest

from distllm.scheduling.cost_tracker import CostTracker, NodeCostInfo
from distllm.scheduling.budget_scheduler import BudgetScheduler
from distllm.scheduling.spot_handler import SpotHandler


class MockNode:
    """Mock node for testing budget scheduler."""
    def __init__(self, node_id, cost_per_hour=0.0, is_spot=False, instance_type="unknown"):
        self.node_id = node_id
        self.cost_per_hour = cost_per_hour
        self.is_spot = is_spot
        self.instance_type = instance_type


class TestCostTracker:
    @pytest.fixture
    def tracker(self):
        return CostTracker(budget_per_hour=10.0)

    def test_register_node(self, tracker):
        tracker.register_node("node-0", cost_per_hour=2.5, is_spot=True, instance_type="g5.xlarge")
        assert "node-0" in tracker._nodes
        assert tracker._nodes["node-0"].is_spot is True

    def test_unregister_node(self, tracker):
        tracker.register_node("node-0", cost_per_hour=2.5)
        node = tracker.unregister_node("node-0")
        assert node is not None
        assert node.node_id == "node-0"
        assert "node-0" not in tracker._nodes

    def test_unregister_nonexistent(self, tracker):
        assert tracker.unregister_node("nonexistent") is None

    def test_get_node_cost_zero(self, tracker):
        assert tracker.get_node_cost("nonexistent") == 0.0

    def test_current_hourly_spend(self, tracker):
        tracker.register_node("n1", cost_per_hour=2.0)
        tracker.register_node("n2", cost_per_hour=3.0)
        assert tracker.get_current_hourly_spend() == 5.0

    def test_budget_remaining(self, tracker):
        tracker.register_node("n1", cost_per_hour=2.0)
        assert tracker.get_budget_remaining() == 8.0  # 10 - 2

    def test_budget_remaining_no_budget(self):
        tracker = CostTracker(budget_per_hour=0.0)
        tracker.register_node("n1", cost_per_hour=2.0)
        assert tracker.get_budget_remaining() == 0.0

    def test_is_within_budget(self, tracker):
        tracker.register_node("n1", cost_per_hour=2.0)
        assert tracker.is_within_budget() is True

    def test_is_over_budget(self, tracker):
        tracker.register_node("n1", cost_per_hour=12.0)
        assert tracker.is_within_budget() is False

    def test_no_budget_always_within(self):
        tracker = CostTracker(budget_per_hour=0.0)
        tracker.register_node("n1", cost_per_hour=100.0)
        assert tracker.is_within_budget() is True

    def test_get_nodes_by_cost(self, tracker):
        tracker.register_node("expensive", cost_per_hour=10.0)
        tracker.register_node("cheap", cost_per_hour=1.0)
        nodes = tracker.get_nodes_by_cost(ascending=True)
        assert nodes[0].node_id == "cheap"
        assert nodes[1].node_id == "expensive"

    def test_get_spot_nodes(self, tracker):
        tracker.register_node("spot1", is_spot=True)
        tracker.register_node("od1", is_spot=False)
        assert len(tracker.get_spot_nodes()) == 1
        assert tracker.get_spot_nodes()[0].node_id == "spot1"

    def test_get_on_demand_nodes(self, tracker):
        tracker.register_node("spot1", is_spot=True)
        tracker.register_node("od1", is_spot=False)
        assert len(tracker.get_on_demand_nodes()) == 1

    def test_record_spot_interruption(self, tracker):
        tracker.record_spot_interruption()
        tracker.record_spot_interruption()
        assert tracker.interruption_count == 2

    def test_get_stats(self, tracker):
        tracker.register_node("n1", cost_per_hour=2.0, is_spot=True)
        stats = tracker.get_stats()
        assert stats["total_nodes"] == 1
        assert stats["spot_nodes"] == 1
        assert stats["hourly_spend"] == 2.0


class TestBudgetScheduler:
    def test_disabled_returns_first(self):
        scheduler = BudgetScheduler(enabled=False)
        nodes = [MockNode("n1", cost_per_hour=10), MockNode("n2", cost_per_hour=1)]
        result = scheduler.select_node(nodes)
        assert result.node_id == "n1"  # First, regardless of cost

    def test_empty_candidates(self):
        scheduler = BudgetScheduler(enabled=True)
        assert scheduler.select_node([]) is None

    def test_select_cheapest_within_budget(self):
        scheduler = BudgetScheduler(enabled=True, budget_per_hour=5.0)
        nodes = [
            MockNode("expensive", cost_per_hour=4.0),
            MockNode("cheap", cost_per_hour=1.0),
        ]
        result = scheduler.select_node(nodes, current_spend=0.0)
        assert result.node_id == "cheap"

    def test_prefers_spot_when_available(self):
        scheduler = BudgetScheduler(enabled=True, spot_preference=1.0)
        nodes = [
            MockNode("spot", cost_per_hour=1.0, is_spot=True),
            MockNode("od", cost_per_hour=0.5, is_spot=False),
        ]
        result = scheduler.select_node(nodes)
        assert result.node_id == "spot"

    def test_select_nodes_batch(self):
        scheduler = BudgetScheduler(enabled=True)
        nodes = [
            MockNode("n1", cost_per_hour=1.0),
            MockNode("n2", cost_per_hour=2.0),
            MockNode("n3", cost_per_hour=3.0),
        ]
        selected = scheduler.select_nodes_batch(nodes, count=2)
        assert len(selected) == 2
        assert selected[0].node_id == "n1"
        assert selected[1].node_id == "n2"


class TestSpotHandler:
    @pytest.fixture
    def handler(self):
        tracker = CostTracker()
        return SpotHandler(cost_tracker=tracker)

    def test_handle_interruption_notice(self, handler):
        drain_called = []
        handler.set_drain_callback(lambda nid: drain_called.append(nid))
        handler.set_fallback_callback(lambda nid: ["fallback-1"])

        fallbacks = handler.handle_interruption_notice("spot-0")
        assert "spot-0" in handler._interrupted_nodes
        assert drain_called == ["spot-0"]
        assert fallbacks == ["fallback-1"]

    def test_interruption_recorded_in_cost_tracker(self, handler):
        handler.cost_tracker.record_spot_interruption = MagicMock()
        handler.handle_interruption_notice("spot-0")
        handler.cost_tracker.record_spot_interruption.assert_called_once()

    def test_is_interrupted(self, handler):
        assert not handler.is_interrupted("spot-0")
        handler.handle_interruption_notice("spot-0")
        assert handler.is_interrupted("spot-0")

    def test_get_interrupted_nodes(self, handler):
        handler.handle_interruption_notice("n1")
        handler.handle_interruption_notice("n2")
        assert set(handler.get_interrupted_nodes()) == {"n1", "n2"}

    def test_clear_interruption(self, handler):
        handler.handle_interruption_notice("spot-0")
        handler.clear_interruption("spot-0")
        assert not handler.is_interrupted("spot-0")

    def test_check_interruption_metadata_unreachable(self, handler):
        """Should return None when metadata endpoint is unreachable."""
        result = handler.check_interruption_metadata()
        assert result is None

    def test_poll_interruptions_no_interruption(self, handler):
        result = handler.poll_interruptions(["n1", "n2"])
        assert result == {}


class TestNodeRegistrationBackwardCompat:
    def test_node_registration_default_cost_fields(self):
        from distllm.core.resource_manager import NodeRegistration
        from distllm.config.loader import NodeRole

        reg = NodeRegistration("n1", "localhost", 50051, 0, 3)
        assert reg.version == "stable"
        assert reg.instance_type == "unknown"
        assert reg.cost_per_hour == 0.0
        assert reg.is_spot is False
