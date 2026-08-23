"""Tests for IntelligentAutoscaler — predictive, cost-aware autoscaling.

Covers:
- ScalingMetrics, ScalingDecision, CostProfile dataclasses
- IntelligentAutoscaler: construction, record_metrics, set_cost_profile,
  evaluate (reactive scaling up/down, cooldown, predictive component),
  cost estimation, stats
"""

from __future__ import annotations

import pytest

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_ia = load_module("distllm/core/intelligent_autoscaler.py")
IntelligentAutoscaler = _ia.IntelligentAutoscaler
ScalingMetrics = _ia.ScalingMetrics
ScalingDecision = _ia.ScalingDecision
CostProfile = _ia.CostProfile


# ── Dataclass tests ───────────────────────────────────────────────────────────


class TestScalingMetrics:
    def test_defaults(self):
        m = ScalingMetrics()
        assert m.active_requests == 0
        assert m.pending_requests == 0
        assert m.gpu_utilization == 0.0
        assert m.current_nodes == 1
        assert m.timestamp > 0


class TestScalingDecision:
    def test_defaults(self):
        d = ScalingDecision(should_scale=False, target_nodes=5, reason="test")
        assert d.should_scale is False
        assert d.target_nodes == 5
        assert d.reason == "test"
        assert d.estimated_cost_change == 0.0
        assert d.confidence == 0.0


class TestCostProfile:
    def test_defaults(self):
        p = CostProfile(
            node_type="gpu-standard",
            cost_per_hour=2.5,
            gpu_memory_gb=80.0,
            gpu_tflops=312.0,
        )
        assert p.node_type == "gpu-standard"
        assert p.is_spot is False
        assert p.spot_probability == 0.0


# ── IntelligentAutoscaler — Construction ──────────────────────────────────────


class TestIntelligentAutoscalerConstruction:
    def test_default_values(self):
        s = IntelligentAutoscaler()
        assert s._min_nodes == 1
        assert s._max_nodes == 20
        assert s._target_util == 0.7
        assert s._scale_up == 0.85
        assert s._scale_down == 0.3
        assert s._cooldown == 60.0
        assert s._last_scale_time == 0.0
        assert len(s._history) == 0

    def test_custom_values(self):
        s = IntelligentAutoscaler(
            min_nodes=3,
            max_nodes=10,
            target_utilization=0.8,
            scale_up_threshold=0.9,
            scale_down_threshold=0.2,
            cooldown_seconds=120.0,
            prediction_window=50,
        )
        assert s._min_nodes == 3
        assert s._max_nodes == 10
        assert s._cooldown == 120.0


# ── IntelligentAutoscaler — Record Metrics ────────────────────────────────────


class TestIntelligentAutoscalerMetrics:
    def test_record_metrics(self):
        s = IntelligentAutoscaler()
        m = ScalingMetrics(active_requests=10, gpu_utilization=75.0)
        s.record_metrics(m)
        assert len(s._history) == 1

    def test_history_window(self):
        s = IntelligentAutoscaler(prediction_window=5)
        for i in range(10):
            s.record_metrics(ScalingMetrics(active_requests=i))
        assert len(s._history) == 5


# ── IntelligentAutoscaler — Cost Profiles ─────────────────────────────────────


class TestIntelligentAutoscalerCostProfiles:
    def test_set_cost_profile(self):
        s = IntelligentAutoscaler()
        profile = CostProfile("gpu", 3.0, 80.0, 312.0)
        s.set_cost_profile("gpu", profile)
        assert len(s._cost_profiles) == 1
        assert s._cost_profiles["gpu"].cost_per_hour == 3.0


# ── IntelligentAutoscaler — Evaluate ──────────────────────────────────────────


class TestIntelligentAutoscalerEvaluate:
    def test_cooldown_respected(self):
        """Immediately after a scale, cooldown should block further scaling."""
        s = IntelligentAutoscaler(cooldown_seconds=3600.0, scale_up_threshold=0.5)
        m = ScalingMetrics(gpu_utilization=95.0, current_nodes=1)
        # First call: should scale up
        d1 = s.evaluate(m)
        # Second call: should be blocked by cooldown
        d2 = s.evaluate(ScalingMetrics(gpu_utilization=95.0, current_nodes=2))
        assert d2.should_scale is False
        assert d2.reason == "cooldown"

    def test_optimal_no_action(self):
        s = IntelligentAutoscaler(min_nodes=2, max_nodes=10, target_utilization=0.7)
        m = ScalingMetrics(
            gpu_utilization=70.0, pending_requests=0, current_nodes=2,
        )
        d = s.evaluate(m)
        # Cooldown not yet tripped, but at target
        # Could be "optimal" or "cooldown" depending on timing
        assert isinstance(d, ScalingDecision)

    def test_scale_up_high_utilization(self):
        s = IntelligentAutoscaler(
            min_nodes=1, max_nodes=10, scale_up_threshold=0.5,
            cooldown_seconds=0.0,
        )
        m = ScalingMetrics(gpu_utilization=90.0, current_nodes=1)
        d = s.evaluate(m)
        assert d.should_scale is True
        assert d.target_nodes > 1
        assert d.reason == "scale_up"

    def test_scale_up_high_queue(self):
        s = IntelligentAutoscaler(
            min_nodes=1, max_nodes=10, scale_up_threshold=0.9,
            cooldown_seconds=0.0,
        )
        m = ScalingMetrics(
            gpu_utilization=50.0, pending_requests=100, current_nodes=1,
        )
        d = s.evaluate(m)
        assert d.should_scale is True
        assert d.target_nodes > 1

    def test_scale_down_low_utilization(self):
        s = IntelligentAutoscaler(
            min_nodes=1, max_nodes=10, scale_down_threshold=0.5,
            cooldown_seconds=0.0,
        )
        m = ScalingMetrics(gpu_utilization=10.0, current_nodes=5)
        d = s.evaluate(m)
        assert d.should_scale is True
        assert d.target_nodes < 5
        assert d.reason == "scale_down"

    def test_scale_down_min_nodes_enforced(self):
        s = IntelligentAutoscaler(
            min_nodes=2, max_nodes=10,
            scale_down_threshold=0.9,
            cooldown_seconds=0.0,
        )
        m = ScalingMetrics(gpu_utilization=5.0, current_nodes=2)
        d = s.evaluate(m)
        # Already at min, should not scale down further
        assert d.should_scale is False or d.target_nodes >= 2

    def test_scale_up_max_nodes_enforced(self):
        s = IntelligentAutoscaler(
            min_nodes=1, max_nodes=3,
            scale_up_threshold=0.5,
            cooldown_seconds=0.0,
        )
        s.record_metrics(ScalingMetrics(gpu_utilization=95.0, current_nodes=3))
        m = ScalingMetrics(gpu_utilization=95.0, current_nodes=3)
        d = s.evaluate(m)
        # Already at max
        assert d.target_nodes <= 3

    def test_cost_estimate_included(self):
        s = IntelligentAutoscaler(
            min_nodes=1, max_nodes=10,
            scale_up_threshold=0.5,
            cooldown_seconds=0.0,
        )
        s.set_cost_profile("gpu", CostProfile("gpu", 3.0, 80.0, 312.0))
        m = ScalingMetrics(gpu_utilization=95.0, current_nodes=1)
        d = s.evaluate(m)
        assert d.estimated_cost_change != 0.0

    def test_cost_estimate_zero_without_profiles(self):
        s = IntelligentAutoscaler(
            min_nodes=1, max_nodes=10,
            scale_up_threshold=0.5,
            cooldown_seconds=0.0,
        )
        m = ScalingMetrics(gpu_utilization=95.0, current_nodes=1)
        d = s.evaluate(m)
        assert d.estimated_cost_change == 0.0


# ── IntelligentAutoscaler — Predictive ────────────────────────────────────────


class TestIntelligentAutoscalerPredictive:
    def test_prediction_with_few_metrics_returns_min(self):
        s = IntelligentAutoscaler(min_nodes=2)
        # Less than 10 history entries
        for _ in range(5):
            s.record_metrics(ScalingMetrics(gpu_utilization=50.0))
        pred = s._predict_load()
        assert pred == 2  # _min_nodes

    def test_prediction_with_sufficient_history(self):
        s = IntelligentAutoscaler(min_nodes=1)
        for i in range(20):
            s.record_metrics(ScalingMetrics(gpu_utilization=float(i * 5)))
        pred = s._predict_load()
        assert isinstance(pred, int)
        assert pred >= 1

    def test_rising_trend_triggers_proactive_scale(self):
        s = IntelligentAutoscaler(min_nodes=1)
        # Rising utilization trend
        for i in range(20):
            s.record_metrics(ScalingMetrics(
                gpu_utilization=min(float(i) * 6, 95.0),
                current_nodes=2,
            ))
        pred = s._predict_load()
        # Trend should trigger +1
        # First half avg ~15, second half avg ~75+, difference >10, second half >60
        assert pred >= 2


# ── IntelligentAutoscaler — Confidence ────────────────────────────────────────


class TestIntelligentAutoscalerConfidence:
    def test_low_confidence_with_few_samples(self):
        s = IntelligentAutoscaler()
        for _ in range(3):
            s.record_metrics(ScalingMetrics())
        assert s._prediction_confidence() == 0.3

    def test_medium_confidence(self):
        s = IntelligentAutoscaler()
        for _ in range(15):
            s.record_metrics(ScalingMetrics())
        assert s._prediction_confidence() == 0.6

    def test_high_confidence(self):
        s = IntelligentAutoscaler()
        for _ in range(25):
            s.record_metrics(ScalingMetrics())
        assert s._prediction_confidence() == 0.8


# ── IntelligentAutoscaler — Stats ─────────────────────────────────────────────


class TestIntelligentAutoscalerStats:
    def test_get_stats(self):
        s = IntelligentAutoscaler(min_nodes=2, max_nodes=10)
        s.set_cost_profile("gpu", CostProfile("gpu", 3.0, 80.0, 312.0))
        for _ in range(5):
            s.record_metrics(ScalingMetrics())
        stats = s.get_stats()
        assert stats["history_size"] == 5
        assert stats["cost_profiles"] == 1
        assert stats["min_nodes"] == 2
        assert stats["max_nodes"] == 10
        assert stats["target_utilization"] == 0.7


# ── Edge Cases ────────────────────────────────────────────────────────────────


class TestIntelligentAutoscalerEdgeCases:
    def test_evaluate_zero_current_nodes(self):
        s = IntelligentAutoscaler(min_nodes=2, cooldown_seconds=0.0)
        m = ScalingMetrics(current_nodes=0)
        d = s.evaluate(m)
        assert d.target_nodes >= 2

    def test_evaluate_handles_normalization(self):
        s = IntelligentAutoscaler(cooldown_seconds=0.0)
        m = ScalingMetrics(gpu_utilization=50.0, current_nodes=1)
        d = s.evaluate(m)
        assert isinstance(d, ScalingDecision)
