"""Tests for the dynamic pipeline rebalancer module.

Tests use ONLY real objects (no mocks) and cover the full public API surface.
"""

from __future__ import annotations

import pytest

from distllm.dist.latency import LatencyTracker
from distllm.dist.rebalancer import (
    PartitionRecommendation,
    Rebalancer,
    StragglerAction,
)
from distllm.config.settings import RebalancerSettings


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------


class TestPartitionRecommendation:
    """Unit tests for the PartitionRecommendation dataclass."""

    def test_fields(self) -> None:
        rec = PartitionRecommendation("node-0", 0, 3)
        assert rec.node_id == "node-0"
        assert rec.start_layer == 0
        assert rec.end_layer == 3

    def test_default_creation(self) -> None:
        rec = PartitionRecommendation(node_id="n1", start_layer=2, end_layer=5)
        assert rec.node_id == "n1"
        assert rec.start_layer == 2
        assert rec.end_layer == 5

    def test_single_layer(self) -> None:
        """A partition covering exactly one layer."""
        rec = PartitionRecommendation("n1", 7, 7)
        assert rec.start_layer == rec.end_layer == 7

    def test_equality_by_value(self) -> None:
        a = PartitionRecommendation("n1", 0, 3)
        b = PartitionRecommendation("n1", 0, 3)
        assert a == b


class TestStragglerAction:
    """Unit tests for the StragglerAction dataclass."""

    def test_minimal_action(self) -> None:
        action = StragglerAction(node_id="n1", action="none")
        assert action.node_id == "n1"
        assert action.action == "none"
        assert action.layer_count_change == 0
        assert action.batch_size_reduction == 0
        assert action.reason == ""

    def test_reassign_action(self) -> None:
        action = StragglerAction(
            node_id="n1",
            action="reassign",
            layer_count_change=-1,
            reason="slowdown 3.0x > 2.0x",
        )
        assert action.action == "reassign"
        assert action.layer_count_change == -1
        assert "3.0x" in action.reason

    def test_reduce_batch_action(self) -> None:
        action = StragglerAction(
            node_id="n1",
            action="reduce_batch",
            batch_size_reduction=2,
            reason="slowdown 1.8x",
        )
        assert action.action == "reduce_batch"
        assert action.batch_size_reduction == 2

    def test_equality_by_value(self) -> None:
        a = StragglerAction("n1", "none")
        b = StragglerAction("n1", "none")
        assert a == b



# ---------------------------------------------------------------------------
# Rebalancer tests
# ---------------------------------------------------------------------------

# Use a short grace period so stragglers are flagged immediately.
_DEFAULT_SETTINGS = RebalancerSettings(
    enabled=True,
    straggler_threshold=1.5,
    min_improvement_pct=0.1,
    cooldown_seconds=0,
    grace_period_steps=1,
    auto_mitigate=False,
)

_AUTO_SETTINGS = _DEFAULT_SETTINGS.model_copy(update={"auto_mitigate": True})

_DISABLED_SETTINGS = _DEFAULT_SETTINGS.model_copy(update={"enabled": False})


def _make_tracker(
    latencies: dict[str, list[float]],
) -> LatencyTracker:
    """Build a LatencyTracker pre-populated with measurements."""
    tracker = LatencyTracker(window_size=100)
    for node_id, values in latencies.items():
        for v in values:
            tracker.record(node_id, v)
    return tracker


# -- detect_stragglers -----------------------------------------------------


class TestDetectStragglers:
    """Tests for Rebalancer.detect_stragglers()."""

    def test_no_stragglers_when_all_equal(self) -> None:
        tracker = _make_tracker({"n1": [1.0, 1.0], "n2": [1.0, 1.0]})
        balancer = Rebalancer(tracker, _DEFAULT_SETTINGS)
        assert balancer.detect_stragglers() == []

    def test_straggler_detected(self) -> None:
        """Node with latency well above median is a straggler."""
        tracker = _make_tracker({"n1": [1.0], "n2": [5.0], "n3": [1.0]})
        balancer = Rebalancer(tracker, _DEFAULT_SETTINGS)
        stragglers = balancer.detect_stragglers()
        assert "n2" in stragglers
        assert "n1" not in stragglers

    def test_custom_threshold(self) -> None:
        """Higher threshold can exclude a borderline node."""
        tracker = _make_tracker({"n1": [1.0], "n2": [1.8], "n3": [1.0]})
        balancer = Rebalancer(tracker, _DEFAULT_SETTINGS)
        # threshold=2.0 means 1.8 is not > 2.0 * median(1.0) = 2.0
        assert balancer.detect_stragglers(threshold=2.0) == []

    def test_returns_empty_for_single_node(self) -> None:
        """Need at least 2 nodes with data to compare."""
        tracker = _make_tracker({"n1": [100.0]})
        balancer = Rebalancer(tracker, _DEFAULT_SETTINGS)
        assert balancer.detect_stragglers() == []

    def test_returns_empty_for_no_data(self) -> None:
        tracker = LatencyTracker()
        balancer = Rebalancer(tracker, _DEFAULT_SETTINGS)
        assert balancer.detect_stragglers() == []

    def test_returns_empty_when_median_is_zero(self) -> None:
        """If all latencies are 0, median=0 and we return empty."""
        tracker = _make_tracker({"n1": [0.0], "n2": [0.0]})
        balancer = Rebalancer(tracker, _DEFAULT_SETTINGS)
        assert balancer.detect_stragglers() == []

    def test_straggler_history_builds_up(self) -> None:
        """A node must exceed grace_period_steps before being reported."""
        tracker = _make_tracker({"n1": [1.0], "n2": [3.0], "n3": [1.0]})
        settings = RebalancerSettings(
            enabled=True,
            straggler_threshold=1.5,
            grace_period_steps=3,
            cooldown_seconds=0,
            min_improvement_pct=0.1,
            auto_mitigate=False,
        )
        balancer = Rebalancer(tracker, settings)
        # First call: counter goes to 1, but grace is 3
        assert balancer.detect_stragglers() == []
        # Second call: counter goes to 2
        assert balancer.detect_stragglers() == []
        # Third call: counter goes to 3, now flagged
        stragglers = balancer.detect_stragglers()
        assert "n2" in stragglers

    def test_recovered_node_cleared_from_history(self) -> None:
        """Node that was a straggler but recovers gets removed from history."""
        tracker = _make_tracker({"n1": [1.0], "n2": [5.0], "n3": [1.0]})
        settings = RebalancerSettings(
            enabled=True,
            straggler_threshold=1.5,
            grace_period_steps=2,
            cooldown_seconds=0,
            min_improvement_pct=0.1,
            auto_mitigate=False,
        )
        balancer = Rebalancer(tracker, settings)
        assert balancer.detect_stragglers() == []  # counter=1
        # Now n2 recovers (latency drops to normal)
        tracker.reset("n2")
        for _ in range(5):
            tracker.record("n2", 1.0)
        assert balancer.detect_stragglers() == []
        # n2 should no longer be in the history
        assert "n2" not in balancer._straggler_history


# -- compute_new_partition --------------------------------------------------


class TestComputeNewPartition:
    """Tests for Rebalancer.compute_new_partition()."""

    def test_balanced_partition(self) -> None:
        """Equal latencies produce equal partitions."""
        tracker = _make_tracker({"n1": [1.0], "n2": [1.0]})
        balancer = Rebalancer(tracker, _DEFAULT_SETTINGS)
        result = balancer.compute_new_partition(10, {"n1": 1.0, "n2": 1.0})
        assert len(result) == 2
        assert result[0].node_id == "n1"
        assert result[0].start_layer == 0
        assert result[1].node_id == "n2"
        assert result[1].end_layer == 9

    def test_returns_empty_for_empty_latencies(self) -> None:
        tracker = _make_tracker({"n1": [1.0]})
        balancer = Rebalancer(tracker, _DEFAULT_SETTINGS)
        assert balancer.compute_new_partition(10, {}) == []

    def test_returns_empty_for_zero_total_layers(self) -> None:
        tracker = _make_tracker({"n1": [1.0]})
        balancer = Rebalancer(tracker, _DEFAULT_SETTINGS)
        assert balancer.compute_new_partition(0, {"n1": 1.0}) == []

    def test_returns_empty_for_negative_layers(self) -> None:
        tracker = _make_tracker({"n1": [1.0]})
        balancer = Rebalancer(tracker, _DEFAULT_SETTINGS)
        assert balancer.compute_new_partition(-1, {"n1": 1.0}) == []

    def test_allocates_more_to_faster_node(self) -> None:
        """Faster node (lower latency = higher inverse) gets more layers."""
        tracker = _make_tracker({"slow": [10.0], "fast": [1.0]})
        balancer = Rebalancer(tracker, _DEFAULT_SETTINGS)
        result = balancer.compute_new_partition(100, {"slow": 10.0, "fast": 1.0})
        fast_rec = [r for r in result if r.node_id == "fast"][0]
        slow_rec = [r for r in result if r.node_id == "slow"][0]
        fast_layers = fast_rec.end_layer - fast_rec.start_layer + 1
        slow_layers = slow_rec.end_layer - slow_rec.start_layer + 1
        assert fast_layers > slow_layers

    def test_allocates_at_least_one_layer_each(self) -> None:
        """Each node gets at least 1 layer even when very slow."""
        tracker = _make_tracker({"n1": [1.0], "n2": [100.0]})
        balancer = Rebalancer(tracker, _DEFAULT_SETTINGS)
        result = balancer.compute_new_partition(2, {"n1": 1.0, "n2": 100.0})
        assert len(result) == 2
        for rec in result:
            layers = rec.end_layer - rec.start_layer + 1
            assert layers >= 1

    def test_partitions_cover_all_layers(self) -> None:
        """All layers from 0 to total_layers-1 are covered."""
        tracker = _make_tracker({"n1": [1.0], "n2": [2.0], "n3": [3.0]})
        balancer = Rebalancer(tracker, _DEFAULT_SETTINGS)
        total = 50
        result = balancer.compute_new_partition(total, {"n1": 1.0, "n2": 2.0, "n3": 3.0})
        assert len(result) == 3
        # Start at 0, contiguous, no gaps
        assert result[0].start_layer == 0
        assert result[-1].end_layer == total - 1
        for i in range(len(result) - 1):
            assert result[i].end_layer + 1 == result[i + 1].start_layer

    def test_partition_with_zeros(self) -> None:
        """Zero latencies are skipped (inverse undefined)."""
        tracker = _make_tracker({"n1": [1.0], "n2": [0.0]})
        balancer = Rebalancer(tracker, _DEFAULT_SETTINGS)
        result = balancer.compute_new_partition(10, {"n1": 1.0, "n2": 0.0})
        # n2 is zero-latency, treated as having no weight, so partition
        # includes n2 but with 0 share -- but latency 0 gets filtered.
        # Only n1 contributes.
        assert len(result) >= 1
        # All layers still covered
        assert result[-1].end_layer == 9

    def test_returns_empty_when_all_latencies_zero(self) -> None:
        """All zero latencies mean total_inverse=0, returns []."""
        tracker = _make_tracker({"n1": [1.0], "n2": [1.0]})
        balancer = Rebalancer(tracker, _DEFAULT_SETTINGS)
        assert balancer.compute_new_partition(10, {"n1": 0.0, "n2": 0.0}) == []

    def test_result_types(self) -> None:
        """Each element is a PartitionRecommendation."""
        tracker = _make_tracker({"n1": [1.0], "n2": [2.0]})
        balancer = Rebalancer(tracker, _DEFAULT_SETTINGS)
        result = balancer.compute_new_partition(10, {"n1": 1.0, "n2": 2.0})
        for rec in result:
            assert isinstance(rec, PartitionRecommendation)


# -- compute_mitigation_actions ---------------------------------------------


class TestComputeMitigationActions:
    """Tests for Rebalancer.compute_mitigation_actions()."""

    def test_empty_input(self) -> None:
        tracker = _make_tracker({"n1": [1.0]})
        balancer = Rebalancer(tracker, _DEFAULT_SETTINGS)
        assert balancer.compute_mitigation_actions([]) == []

    def test_insufficient_data(self) -> None:
        """Needs at least 2 nodes with recorded data."""
        tracker = _make_tracker({"n1": [1.0]})
        balancer = Rebalancer(tracker, _DEFAULT_SETTINGS)
        assert balancer.compute_mitigation_actions(["n1"]) == []

    def test_action_non_straggler_skipped(self) -> None:
        """Node with latency at or below median gets no action."""
        tracker = _make_tracker({"n1": [1.0], "n2": [1.0], "n3": [1.0]})
        balancer = Rebalancer(tracker, _DEFAULT_SETTINGS)
        actions = balancer.compute_mitigation_actions(["n1"])
        assert len(actions) == 0

    def test_mild_slowdown_reduces_batch(self) -> None:
        """Slowdown between 1.5x and 2.0x produces reduce_batch."""
        tracker = _make_tracker({"n1": [1.0], "n2": [1.8], "n3": [1.0]})
        balancer = Rebalancer(tracker, _DEFAULT_SETTINGS)
        actions = balancer.compute_mitigation_actions(["n2"])
        assert len(actions) == 1
        assert actions[0].action == "reduce_batch"
        assert actions[0].batch_size_reduction >= 1
        assert actions[0].node_id == "n2"

    def test_severe_slowdown_reassigns(self) -> None:
        """Slowdown > 2.0x with auto_mitigate enabled produces reassign."""
        tracker = _make_tracker({"n1": [1.0], "n2": [5.0], "n3": [1.0]})
        balancer = Rebalancer(tracker, _AUTO_SETTINGS)
        actions = balancer.compute_mitigation_actions(["n2"])
        assert len(actions) == 1
        assert actions[0].action == "reassign"
        assert actions[0].layer_count_change == -1

    def test_severe_slowdown_no_auto_mitigate(self) -> None:
        """Without auto_mitigate, severe slowdown falls back to reduce_batch (not reassign)."""
        tracker = _make_tracker({"n1": [1.0], "n2": [5.0], "n3": [1.0]})
        balancer = Rebalancer(tracker, _DEFAULT_SETTINGS)
        actions = balancer.compute_mitigation_actions(["n2"])
        # The >2.0x check requires auto_mitigate, so it falls to the >1.5x
        # elif branch which produces reduce_batch.
        assert len(actions) == 1
        assert actions[0].action == "reduce_batch"
        assert actions[0].action != "reassign"

    def test_batch_reduction_caps_at_four(self) -> None:
        """Repeated batch reductions cap at 4 steps."""
        tracker = _make_tracker({"n1": [1.0], "n2": [1.8], "n3": [1.0]})
        balancer = Rebalancer(tracker, _DEFAULT_SETTINGS)

        for _ in range(5):
            balancer.compute_mitigation_actions(["n2"])

        actions = balancer.compute_mitigation_actions(["n2"])
        assert len(actions) == 1
        assert actions[0].batch_size_reduction <= 4

    def test_within_tolerance(self) -> None:
        """Slowdown <= 1.5x produces a 'none' action."""
        tracker = _make_tracker({"n1": [1.0], "n2": [1.4], "n3": [1.0]})
        balancer = Rebalancer(tracker, _AUTO_SETTINGS)
        actions = balancer.compute_mitigation_actions(["n2"])
        assert len(actions) == 1
        assert actions[0].action == "none"

    def test_ratio_just_above_two_with_auto(self) -> None:
        """Borderline > 2.0x, average is above median."""
        tracker = _make_tracker({"n1": [1.0], "n2": [2.1], "n3": [1.0]})
        balancer = Rebalancer(tracker, _AUTO_SETTINGS)
        actions = balancer.compute_mitigation_actions(["n2"])
        assert len(actions) == 1
        assert actions[0].action == "reassign"


# -- apply_mitigation_actions -----------------------------------------------


class TestApplyMitigationActions:
    """Tests for Rebalancer.apply_mitigation_actions()."""

    def test_empty_actions_does_nothing(self) -> None:
        tracker = _make_tracker({"n1": [1.0]})
        callback_called = False

        def callback(node_id: str, change: int) -> None:
            nonlocal callback_called
            callback_called = True

        balancer = Rebalancer(tracker, _DEFAULT_SETTINGS, on_reassign=callback)
        balancer.apply_mitigation_actions([])
        assert not callback_called

    def test_reassign_triggers_callback(self) -> None:
        """reassign action invokes on_reassign callback."""
        tracker = _make_tracker({"n1": [1.0]})
        captured: list[tuple[str, int]] = []

        def callback(node_id: str, change: int) -> None:
            captured.append((node_id, change))

        balancer = Rebalancer(tracker, _DEFAULT_SETTINGS, on_reassign=callback)
        actions = [StragglerAction("n1", "reassign", layer_count_change=-1)]
        balancer.apply_mitigation_actions(actions)
        assert captured == [("n1", -1)]

    def test_callback_exception_is_caught(self) -> None:
        """Exception in on_reassign is logged, not propagated."""
        tracker = _make_tracker({"n1": [1.0]})

        def failing_callback(node_id: str, change: int) -> None:
            raise RuntimeError("simulated failure")

        balancer = Rebalancer(tracker, _DEFAULT_SETTINGS, on_reassign=failing_callback)
        actions = [StragglerAction("n1", "reassign", layer_count_change=-1)]
        # Should not raise
        balancer.apply_mitigation_actions(actions)

    def test_reduce_batch_no_callback(self) -> None:
        """reduce_batch action does not invoke callback."""
        tracker = _make_tracker({"n1": [1.0]})
        callback_called = False

        def callback(node_id: str, change: int) -> None:
            nonlocal callback_called
            callback_called = True

        balancer = Rebalancer(tracker, _DEFAULT_SETTINGS, on_reassign=callback)
        actions = [StragglerAction("n1", "reduce_batch", batch_size_reduction=2)]
        balancer.apply_mitigation_actions(actions)
        assert not callback_called

    def test_none_action_noop(self) -> None:
        """Action type 'none' does nothing."""
        tracker = _make_tracker({"n1": [1.0]})
        callback_called = False

        def callback(node_id: str, change: int) -> None:
            nonlocal callback_called
            callback_called = True

        balancer = Rebalancer(tracker, _DEFAULT_SETTINGS, on_reassign=callback)
        actions = [StragglerAction("n1", "none")]
        balancer.apply_mitigation_actions(actions)
        assert not callback_called

    def test_no_callback_set_does_not_crash(self) -> None:
        """apply_mitigation with on_reassign=None is safe."""
        tracker = _make_tracker({"n1": [1.0]})
        balancer = Rebalancer(tracker, _DEFAULT_SETTINGS, on_reassign=None)
        actions = [StragglerAction("n1", "reassign", layer_count_change=-1)]
        balancer.apply_mitigation_actions(actions)


# -- get_batch_size_adjustment / clear_batch_adjustments --------------------


class TestBatchSizeAdjustments:
    """Tests for batch size adjustment getter/setter/clear."""

    def test_default_is_1_0(self) -> None:
        tracker = _make_tracker({"n1": [1.0]})
        balancer = Rebalancer(tracker, _DEFAULT_SETTINGS)
        assert balancer.get_batch_size_adjustment("nonexistent") == 1.0

    def test_single_reduction(self) -> None:
        tracker = _make_tracker({"n1": [1.0]})
        balancer = Rebalancer(tracker, _DEFAULT_SETTINGS)
        # Simulate a batch reduction being registered
        balancer._batch_size_adjustments["n1"] = 1
        assert balancer.get_batch_size_adjustment("n1") == 0.75

    def test_max_reduction(self) -> None:
        """At cap of 4, factor is clamped to 0.25 (minimum)."""
        tracker = _make_tracker({"n1": [1.0]})
        balancer = Rebalancer(tracker, _DEFAULT_SETTINGS)
        balancer._batch_size_adjustments["n1"] = 4
        assert balancer.get_batch_size_adjustment("n1") == 0.25

    def test_clear_restores_default(self) -> None:
        tracker = _make_tracker({"n1": [1.0]})
        balancer = Rebalancer(tracker, _DEFAULT_SETTINGS)
        balancer._batch_size_adjustments["n1"] = 2
        balancer.clear_batch_adjustments("n1")
        assert balancer.get_batch_size_adjustment("n1") == 1.0

    def test_clear_nonexistent_no_error(self) -> None:
        tracker = _make_tracker({"n1": [1.0]})
        balancer = Rebalancer(tracker, _DEFAULT_SETTINGS)
        balancer.clear_batch_adjustments("ghost")  # should not raise


# -- should_rebalance -------------------------------------------------------


class TestShouldRebalance:
    """Tests for Rebalancer.should_rebalance()."""

    def test_disabled_returns_false(self) -> None:
        tracker = _make_tracker({"n1": [1.0], "n2": [1.0]})
        balancer = Rebalancer(tracker, _DISABLED_SETTINGS)
        ok, reason = balancer.should_rebalance()
        assert not ok
        assert "disabled" in reason

    def test_no_stragglers_returns_false(self) -> None:
        tracker = _make_tracker({"n1": [1.0], "n2": [1.0]})
        balancer = Rebalancer(tracker, _DEFAULT_SETTINGS)
        ok, reason = balancer.should_rebalance()
        assert not ok
        assert "no stragglers" in reason

    def test_insufficient_data_returns_false(self) -> None:
        tracker = LatencyTracker()
        tracker.record("n1", 1.0)
        balancer = Rebalancer(tracker, _DEFAULT_SETTINGS)
        ok, reason = balancer.should_rebalance()
        assert not ok
        assert "insufficient" in reason or "no stragglers" in reason

    def test_should_rebalance_when_straggler_present(self) -> None:
        """With a straggler and improvement above threshold, returns True."""
        tracker = _make_tracker({"n1": [1.0], "n2": [5.0], "n3": [1.0]})
        settings = RebalancerSettings(
            enabled=True,
            straggler_threshold=1.5,
            min_improvement_pct=0.1,
            cooldown_seconds=0,
            grace_period_steps=1,
            auto_mitigate=False,
        )
        balancer = Rebalancer(tracker, settings)
        ok, reason = balancer.should_rebalance()
        assert ok
        assert "stragglers" in reason

    def test_cooldown_respected(self) -> None:
        """Within cooldown, should_rebalance returns False."""
        tracker = _make_tracker({"n1": [1.0], "n2": [5.0], "n3": [1.0]})
        settings = RebalancerSettings(
            enabled=True,
            straggler_threshold=1.5,
            min_improvement_pct=0.1,
            cooldown_seconds=3600,
            grace_period_steps=1,
            auto_mitigate=False,
        )
        balancer = Rebalancer(tracker, settings)
        balancer.record_rebalance()
        ok, reason = balancer.should_rebalance()
        assert not ok
        assert "cooldown" in reason


# -- record_rebalance / set_current_partition ------------------------------


class TestRecordAndSetPartition:
    """Tests for record_rebalance and set_current_partition."""

    def test_record_rebalance_updates_time(self) -> None:
        tracker = LatencyTracker()
        balancer = Rebalancer(tracker, _DEFAULT_SETTINGS)
        before = balancer._last_rebalance_time
        balancer.record_rebalance()
        assert balancer._last_rebalance_time >= before

    def test_set_current_partition(self) -> None:
        tracker = LatencyTracker()
        balancer = Rebalancer(tracker, _DEFAULT_SETTINGS)
        partition = [("n1", 0, 4), ("n2", 5, 9)]
        balancer.set_current_partition(partition)
        # Internal state should match. Avoid accessing private field
        # in prod code, but for a zero-mock test this is acceptable.
        assert balancer._current_partition == partition

    def test_set_current_partition_overwrites(self) -> None:
        tracker = LatencyTracker()
        balancer = Rebalancer(tracker, _DEFAULT_SETTINGS)
        balancer.set_current_partition([("n1", 0, 9)])
        balancer.set_current_partition([("n2", 0, 4)])
        assert balancer._current_partition == [("n2", 0, 4)]

    def test_set_current_partition_empty(self) -> None:
        tracker = LatencyTracker()
        balancer = Rebalancer(tracker, _DEFAULT_SETTINGS)
        balancer.set_current_partition([])
        assert balancer._current_partition == []


# -- Integration-style: end-to-end flows ------------------------------------


class TestExtendedWorkflow:
    """Larger flows that exercise multiple methods in sequence.

    These are not true integration tests (no I/O), but they combine
    multiple public methods with real objects to exercise the full
    detection-mitigation loop.
    """

    def test_detect_then_mitigate(self) -> None:
        """End-to-end: detect straggler -> compute action -> apply."""
        tracker = _make_tracker({"n1": [1.0], "n2": [4.0], "n3": [1.0]})
        settings = RebalancerSettings(
            enabled=True,
            straggler_threshold=1.5,
            min_improvement_pct=0.1,
            cooldown_seconds=0,
            grace_period_steps=1,
            auto_mitigate=True,
        )
        callback_log: list[tuple[str, int]] = []
        balancer = Rebalancer(tracker, settings, on_reassign=lambda n, c: callback_log.append((n, c)))

        stragglers = balancer.detect_stragglers()
        assert "n2" in stragglers

        actions = balancer.compute_mitigation_actions(stragglers)
        assert len(actions) >= 1
        reassign_actions = [a for a in actions if a.action == "reassign"]
        assert len(reassign_actions) >= 1

        balancer.apply_mitigation_actions(actions)
        assert len(callback_log) >= 1
        assert callback_log[0][0] == "n2"

    def test_rebalance_check_and_perform(self) -> None:
        """Full should_rebalance -> compute_new_partition cycle."""
        tracker = _make_tracker({"n1": [1.0], "n2": [5.0], "n3": [1.0]})
        settings = RebalancerSettings(
            enabled=True,
            straggler_threshold=1.5,
            min_improvement_pct=1.0,
            cooldown_seconds=0,
            grace_period_steps=1,
            auto_mitigate=False,
        )
        balancer = Rebalancer(tracker, settings)

        ok, _ = balancer.should_rebalance()
        assert ok

        latencies = {"n1": 1.0, "n2": 5.0, "n3": 1.0}
        partition = balancer.compute_new_partition(30, latencies)
        assert len(partition) == 3
        balancer.set_current_partition(
            [(rec.node_id, rec.start_layer, rec.end_layer) for rec in partition]
        )
        balancer.record_rebalance()

        # After record_rebalance, cooldown should be active
        settings_with_cooldown = settings.model_copy(update={"cooldown_seconds": 3600})
        balancer2 = Rebalancer(tracker, settings_with_cooldown)
        balancer2.record_rebalance()
        ok2, reason2 = balancer2.should_rebalance()
        assert not ok2
        assert "cooldown" in reason2

    def test_multiple_stragglers(self) -> None:
        """Multiple stragglers all get mitigation actions."""
        tracker = _make_tracker({"n1": [1.0], "n2": [1.0], "n3": [10.0], "n4": [10.0]})
        settings = _AUTO_SETTINGS
        balancer = Rebalancer(tracker, settings)

        stragglers = balancer.detect_stragglers()
        assert len(stragglers) >= 2
        assert "n3" in stragglers
        assert "n4" in stragglers

        actions = balancer.compute_mitigation_actions(stragglers)
        assert len(actions) >= 1
        # Each action should be for a node in stragglers
        for action in actions:
            assert action.node_id in stragglers
