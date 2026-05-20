"""Tests for Rebalancer."""

import time
import pytest

from distllm.core.latency_tracker import LatencyTracker
from distllm.core.rebalancer import Rebalancer, PartitionRecommendation
from distllm.config.settings import RebalancerSettings


def make_rebalancer(enabled=True, straggler_threshold=1.5, min_improvement_pct=0.1, cooldown_seconds=0, grace_period_steps=1):
    settings = RebalancerSettings(
        enabled=enabled,
        straggler_threshold=straggler_threshold,
        min_improvement_pct=min_improvement_pct,
        cooldown_seconds=cooldown_seconds,
        grace_period_steps=grace_period_steps,
    )
    tracker = LatencyTracker()
    return Rebalancer(tracker, settings), tracker


class TestRebalancer:
    """Tests for Rebalancer class."""

    def test_detect_stragglers_yes(self):
        """One node 2x slower, detected."""
        reb, tracker = make_rebalancer(straggler_threshold=1.5)
        tracker.record("fast", 10.0)
        tracker.record("slow", 50.0)  # 5x median (median=30, threshold=45)

        stragglers = reb.detect_stragglers()
        assert "slow" in stragglers
        assert "fast" not in stragglers

    def test_detect_stragglers_no(self):
        """All nodes similar, none detected."""
        reb, tracker = make_rebalancer(straggler_threshold=1.5)
        tracker.record("node-1", 10.0)
        tracker.record("node-2", 11.0)

        stragglers = reb.detect_stragglers()
        assert stragglers == []

    def test_detect_stragglers_insufficient_data(self):
        """Only 1 node, returns empty."""
        reb, tracker = make_rebalancer()
        tracker.record("node-1", 10.0)

        assert reb.detect_stragglers() == []

    def test_compute_new_partition_equal(self):
        """Equal latencies -> roughly equal partition."""
        reb, _ = make_rebalancer()
        result = reb.compute_new_partition(12, {"node-1": 10.0, "node-2": 10.0})

        assert len(result) == 2
        total_layers = sum(p.end_layer - p.start_layer + 1 for p in result)
        assert total_layers == 12

    def test_compute_new_partition_weighted(self):
        """Faster node gets more layers."""
        reb, _ = make_rebalancer()
        result = reb.compute_new_partition(20, {"fast": 5.0, "slow": 15.0})

        fast_layers = None
        slow_layers = None
        for p in result:
            if p.node_id == "fast":
                fast_layers = p.end_layer - p.start_layer + 1
            else:
                slow_layers = p.end_layer - p.start_layer + 1

        assert fast_layers is not None
        assert slow_layers is not None
        assert fast_layers > slow_layers  # Faster node should get more layers

    def test_compute_new_partition_empty(self):
        """Empty node_latencies returns empty."""
        reb, _ = make_rebalancer()
        assert reb.compute_new_partition(10, {}) == []

    def test_should_rebalance_disabled(self):
        """Returns False when disabled."""
        reb, tracker = make_rebalancer(enabled=False)
        tracker.record("node-1", 10.0)
        tracker.record("node-2", 50.0)

        should, reason = reb.should_rebalance()
        assert should is False
        assert "disabled" in reason

    def test_should_rebalance_cooldown(self):
        """Returns False during cooldown."""
        reb, tracker = make_rebalancer(cooldown_seconds=300)
        tracker.record("node-1", 10.0)
        tracker.record("node-2", 50.0)

        # Record a rebalance to start the cooldown
        reb.record_rebalance()

        should, reason = reb.should_rebalance()
        assert should is False
        assert "cooldown" in reason

    def test_should_rebalance_no_stragglers(self):
        """Returns False when no stragglers."""
        reb, tracker = make_rebalancer(cooldown_seconds=0)
        tracker.record("node-1", 10.0)
        tracker.record("node-2", 11.0)

        should, reason = reb.should_rebalance()
        assert should is False
        assert "no stragglers" in reason

    def test_should_rebalance_yes(self):
        """Returns True when conditions met."""
        reb, tracker = make_rebalancer(
            enabled=True,
            straggler_threshold=1.5,
            min_improvement_pct=0.1,
            cooldown_seconds=0,
        )
        tracker.record("node-1", 10.0)
        tracker.record("node-2", 100.0)  # Large straggler

        should, reason = reb.should_rebalance()
        assert should is True
        assert "node-2" in reason

    def test_record_rebalance_updates_cooldown(self):
        """After recording, cooldown active."""
        reb, tracker = make_rebalancer(cooldown_seconds=300)
        reb.record_rebalance()

        should, reason = reb.should_rebalance()
        assert should is False
        assert "cooldown" in reason

    def test_partition_recommendation_dataclass(self):
        """PartitionRecommendation has correct fields."""
        rec = PartitionRecommendation("node-1", 0, 5)
        assert rec.node_id == "node-1"
        assert rec.start_layer == 0
        assert rec.end_layer == 5
