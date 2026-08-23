"""Tests for GracefulDegradation: node failure → degraded throughput,
progressive degradation levels (0.3→0.5→0.7→0.85), and recovery.

Tests GracefulDegradation, LoadSnapshot, and DegradationPlan directly.
"""

import pytest

from distllm.core.graceful_degradation import (
    GracefulDegradation,
    DegradationLevel,
    DegradationPlan,
    LoadSnapshot,
)


# =========================================================================
# LoadSnapshot — composite score computation
# =========================================================================

class TestLoadSnapshot:
    """LoadSnapshot.score() composite load metric."""

    def test_idle_score_is_zero(self):
        snap = LoadSnapshot()
        assert snap.score() == 0.0

    def test_full_queue_depth(self):
        snap = LoadSnapshot(queue_depth=50)
        assert snap.score() == pytest.approx(0.3, abs=0.01)

    def test_full_latency(self):
        snap = LoadSnapshot(avg_latency_ms=5000)
        assert snap.score() == pytest.approx(0.3, abs=0.01)

    def test_full_memory(self):
        snap = LoadSnapshot(memory_util_pct=90)
        assert snap.score() == pytest.approx(0.2, abs=0.01)

    def test_full_request_rate(self):
        snap = LoadSnapshot(request_rate=100)
        assert snap.score() == pytest.approx(0.2, abs=0.01)

    def test_half_load(self):
        snap = LoadSnapshot(queue_depth=25, avg_latency_ms=2500,
                            memory_util_pct=45, request_rate=50)
        assert snap.score() == pytest.approx(0.5, abs=0.01)

    def test_overload_capped_at_one(self):
        snap = LoadSnapshot(queue_depth=500, avg_latency_ms=50000,
                            memory_util_pct=200, request_rate=1000)
        assert snap.score() <= 1.0

    def test_partial_weight(self):
        snap = LoadSnapshot(queue_depth=25, avg_latency_ms=2500)
        score = snap.score()
        assert 0.2 < score < 0.4


# =========================================================================
# Graceful degradation — node failure → reduced throughput
# =========================================================================

class TestGracefulDegradation:
    """High load → degradation plan with reduced throughput."""

    def test_disabled_returns_none(self):
        gd = GracefulDegradation(enabled=False)
        plan = gd.evaluate(LoadSnapshot(queue_depth=100, avg_latency_ms=10000))
        assert plan.level == DegradationLevel.NONE

    def test_light_load_reduces_tokens(self):
        gd = GracefulDegradation()
        plan = gd.evaluate(LoadSnapshot(queue_depth=30, avg_latency_ms=2000))
        assert plan.level == DegradationLevel.LIGHT
        assert plan.max_tokens == 1024

    def test_moderate_load_reduces_tokens_and_model(self):
        gd = GracefulDegradation(fallback_model="small-model")
        plan = gd.evaluate(LoadSnapshot(queue_depth=45, avg_latency_ms=4000))
        assert plan.level == DegradationLevel.MODERATE
        assert plan.max_tokens == 512
        assert plan.model_override == "small-model"

    def test_severe_load_uses_stale(self):
        gd = GracefulDegradation()
        plan = gd.evaluate(LoadSnapshot(queue_depth=45, avg_latency_ms=4000,
                                        memory_util_pct=70, request_rate=80))
        assert plan.level == DegradationLevel.SEVERE
        assert plan.max_tokens == 256
        assert plan.use_stale is True
        assert plan.truncate_prompt == 1024

    def test_critical_load_allows_partial(self):
        gd = GracefulDegradation()
        plan = gd.evaluate(LoadSnapshot(queue_depth=50, avg_latency_ms=5000,
                                        memory_util_pct=90, request_rate=100))
        assert plan.level == DegradationLevel.CRITICAL
        assert plan.max_tokens == 64
        assert plan.use_stale is True
        assert plan.partial_ok is True
        assert plan.truncate_prompt == 512

    def test_apply_to_params_caps_tokens(self):
        plan = DegradationPlan(level=DegradationLevel.LIGHT, max_tokens=1024)
        params = {"max_new_tokens": 4096}
        # apply_to_params is non-mutating: it returns the adjusted copy.
        result = plan.apply_to_params(params)
        assert result["max_new_tokens"] == 1024
        assert params["max_new_tokens"] == 4096  # original untouched

    def test_apply_to_params_truncates_prompt(self):
        plan = DegradationPlan(level=DegradationLevel.SEVERE, truncate_prompt=100)
        params = {"prompt": "x" * 500}
        result = plan.apply_to_params(params)
        assert len(result["prompt"]) == 100
        assert len(params["prompt"]) == 500  # original untouched

    def test_partial_response_format(self):
        gd = GracefulDegradation(partial_response="try again later")
        resp = gd.get_partial_response("req-123")
        assert resp["request_id"] == "req-123"
        assert resp["degraded"] is True
        assert resp["choices"][0]["finish_reason"] == "degraded"
        assert resp["choices"][0]["text"] == "try again later"

    def test_evaluate_tracks_history(self):
        gd = GracefulDegradation()
        gd.evaluate(LoadSnapshot(queue_depth=10))
        gd.evaluate(LoadSnapshot(queue_depth=20))
        stats = gd.stats()
        assert stats["history_size"] == 2


# =========================================================================
# Degradation levels — progressive thresholds (0.3→0.5→0.7→0.85)
# =========================================================================

class TestDegradationLevels:
    """Progressive thresholds map to correct degradation level."""

    def test_below_light_is_none(self):
        gd = GracefulDegradation()
        plan = gd.evaluate(LoadSnapshot(queue_depth=10, avg_latency_ms=500))
        assert plan.level == DegradationLevel.NONE

    def test_light_threshold_exact(self):
        gd = GracefulDegradation(light_threshold=0.3,
                                 moderate_threshold=0.5,
                                 severe_threshold=0.7,
                                 critical_threshold=0.85)
        plan = gd.evaluate(LoadSnapshot(queue_depth=30, avg_latency_ms=2000))
        assert plan.level == DegradationLevel.LIGHT
        assert plan.max_tokens == 1024

    def test_moderate_threshold(self):
        gd = GracefulDegradation(moderate_threshold=0.5)
        plan = gd.evaluate(LoadSnapshot(queue_depth=45, avg_latency_ms=4000))
        assert plan.level == DegradationLevel.MODERATE

    def test_moderate_not_yet_severe(self):
        gd = GracefulDegradation(moderate_threshold=0.5, severe_threshold=0.7)
        plan = gd.evaluate(LoadSnapshot(queue_depth=50, avg_latency_ms=3500,
                                        memory_util_pct=30))
        assert plan.level == DegradationLevel.MODERATE

    def test_severe_threshold(self):
        gd = GracefulDegradation(severe_threshold=0.7)
        plan = gd.evaluate(LoadSnapshot(queue_depth=50, avg_latency_ms=5000,
                                        memory_util_pct=50, request_rate=50))
        assert plan.level == DegradationLevel.SEVERE

    def test_severe_not_yet_critical(self):
        gd = GracefulDegradation(severe_threshold=0.7, critical_threshold=0.85)
        plan = gd.evaluate(LoadSnapshot(queue_depth=50, avg_latency_ms=5000,
                                        memory_util_pct=50, request_rate=50))
        assert plan.level == DegradationLevel.SEVERE

    def test_critical_threshold(self):
        gd = GracefulDegradation(critical_threshold=0.85)
        plan = gd.evaluate(LoadSnapshot(queue_depth=50, avg_latency_ms=5000,
                                        memory_util_pct=90, request_rate=100))
        assert plan.level == DegradationLevel.CRITICAL

    def test_custom_thresholds(self):
        gd = GracefulDegradation(light_threshold=0.1, moderate_threshold=0.2,
                                 severe_threshold=0.3, critical_threshold=0.4)
        plan = gd.evaluate(LoadSnapshot(queue_depth=25, avg_latency_ms=2000))
        assert plan.level == DegradationLevel.MODERATE

    def test_each_level_has_unique_max_tokens(self):
        gd = GracefulDegradation()
        plan_none = gd.evaluate(LoadSnapshot())
        plan_light = gd.evaluate(LoadSnapshot(queue_depth=30, avg_latency_ms=2000))
        plan_mod = gd.evaluate(LoadSnapshot(queue_depth=45, avg_latency_ms=4000))
        plan_sev = gd.evaluate(LoadSnapshot(queue_depth=50, avg_latency_ms=5000,
                                            memory_util_pct=50, request_rate=50))
        plan_crit = gd.evaluate(LoadSnapshot(queue_depth=50, avg_latency_ms=5000,
                                             memory_util_pct=90, request_rate=100))
        assert plan_none.max_tokens is None
        assert plan_light.max_tokens == 1024
        assert plan_mod.max_tokens == 512
        assert plan_sev.max_tokens == 256
        assert plan_crit.max_tokens == 64

    def test_current_level_tracks_average(self):
        gd = GracefulDegradation()
        for _ in range(10):
            gd.evaluate(LoadSnapshot(queue_depth=50, avg_latency_ms=5000,
                                     memory_util_pct=50, request_rate=50))
        assert gd.current_level in (DegradationLevel.SEVERE, DegradationLevel.CRITICAL)


# =========================================================================
# Degradation recovery — load drops → full capacity restored
# =========================================================================

class TestDegradationRecovery:
    """Node restored → load drops → evaluate returns NONE."""

    def test_idle_after_overload_returns_none(self):
        gd = GracefulDegradation()
        gd.evaluate(LoadSnapshot(queue_depth=50, avg_latency_ms=5000,
                                 memory_util_pct=90, request_rate=100))
        plan = gd.evaluate(LoadSnapshot())
        assert plan.level == DegradationLevel.NONE

    def test_full_cycle_critical_to_none(self):
        gd = GracefulDegradation()
        plan1 = gd.evaluate(LoadSnapshot(queue_depth=50, avg_latency_ms=5000,
                                         memory_util_pct=90, request_rate=100))
        assert plan1.level == DegradationLevel.CRITICAL
        plan2 = gd.evaluate(LoadSnapshot(queue_depth=0, avg_latency_ms=50,
                                         memory_util_pct=30, request_rate=5))
        assert plan2.level == DegradationLevel.NONE
        assert plan2.max_tokens is None

    def test_recovery_preserves_max_tokens(self):
        params = {"max_new_tokens": 2048}
        gd = GracefulDegradation()
        plan = gd.evaluate(LoadSnapshot())
        updated = plan.apply_to_params(params)
        assert updated["max_new_tokens"] == 2048

    def test_disabled_after_enabled(self):
        gd = GracefulDegradation(enabled=True)
        gd.evaluate(LoadSnapshot(queue_depth=50, avg_latency_ms=5000))
        gd._enabled = False
        plan = gd.evaluate(LoadSnapshot(queue_depth=50, avg_latency_ms=5000))
        assert plan.level == DegradationLevel.NONE

    def test_stats_reflects_current_level(self):
        gd = GracefulDegradation()
        gd.evaluate(LoadSnapshot(queue_depth=50, avg_latency_ms=5000,
                                 memory_util_pct=90, request_rate=100))
        gd.evaluate(LoadSnapshot())
        stats = gd.stats()
        assert stats["current_level"] >= 0
        assert stats["enabled"] is True

    def test_multiple_degradation_recovery_cycles(self):
        gd = GracefulDegradation()
        for _ in range(3):
            plan = gd.evaluate(LoadSnapshot(queue_depth=50, avg_latency_ms=5000,
                                            memory_util_pct=90, request_rate=100))
            assert plan.level == DegradationLevel.CRITICAL
            plan = gd.evaluate(LoadSnapshot())
            assert plan.level == DegradationLevel.NONE

    def test_gradual_recovery(self):
        gd = GracefulDegradation()
        gd.evaluate(LoadSnapshot(queue_depth=50, avg_latency_ms=5000,
                                 memory_util_pct=90, request_rate=100))
        plan = gd.evaluate(LoadSnapshot(queue_depth=30, avg_latency_ms=2000))
        assert plan.level == DegradationLevel.LIGHT
        plan = gd.evaluate(LoadSnapshot())
        assert plan.level == DegradationLevel.NONE
