"""Tests for straggler-aware gradient-based scheduling."""

from __future__ import annotations

from distllm.core.straggler_aware_scheduler import (
    BudgetAllocation,
    GradientRecovery,
    GradientTracker,
    StragglerAwareScheduler,
)


class TestGradientTracker:
    def test_init(self):
        gt = GradientTracker(window_size=10)
        assert gt.gradient == 0.0
        assert not gt.is_straggling
        assert not gt.is_recovering

    def test_rising_latency_detected(self):
        gt = GradientTracker(window_size=10)
        for i in range(20):
            gt.record(100.0 + i * 2)
        assert gt.is_straggling
        assert gt.gradient > 0

    def test_falling_latency_detected(self):
        gt = GradientTracker(window_size=10)
        for i in range(20):
            gt.record(200.0 - i * 3)
        assert gt.is_recovering
        assert gt.gradient < 0

    def test_flat_regular(self):
        gt = GradientTracker(window_size=10)
        for i in range(20):
            gt.record(100.0)
        assert not gt.is_straggling
        assert not gt.is_recovering

    def test_insufficient_samples(self):
        gt = GradientTracker(window_size=10)
        gt.record(100.0)
        assert gt.gradient == 0.0


class TestStragglerAwareScheduler:
    def test_init(self):
        s = StragglerAwareScheduler(base_batch_size=64)
        assert s.active_stragglers == []

    def test_get_budget_full(self):
        s = StragglerAwareScheduler()
        b = s.get_budget("node-1")
        assert b.max_batch_size == 64
        assert b.reduction_factor == 1.0

    def test_budget_reduced_on_straggler(self):
        s = StragglerAwareScheduler(base_batch_size=64)
        for i in range(20):
            s.record_latency("node-1", 100.0 + i * 5)
        b = s.get_budget("node-1")
        assert b.max_batch_size < 64
        assert b.reduction_factor < 1.0

    def test_budget_recovers(self):
        s = StragglerAwareScheduler(base_batch_size=64)
        for i in range(20):
            s.record_latency("node-1", 100.0 + i * 5)
        for i in range(30):
            s.record_latency("node-1", 200.0 - i * 5)
        b = s.get_budget("node-1")
        assert b.reduction_factor == 1.0

    def test_speculative_isolation(self):
        s = StragglerAwareScheduler(base_batch_size=64, speculative_threshold=0.5)
        for i in range(15):
            s.record_latency("node-1", 100.0 + i * 5)
        assert "node-1" in s.speculative_nodes

    def test_multiple_nodes(self):
        s = StragglerAwareScheduler()
        for i in range(15):
            s.record_latency("fast", 50.0)
            s.record_latency("slow", 100.0 + i * 10)
        fast_b = s.get_budget("fast")
        slow_b = s.get_budget("slow")
        assert fast_b.reduction_factor > slow_b.reduction_factor

    def test_stats(self):
        s = StragglerAwareScheduler()
        for i in range(15):
            s.record_latency("n1", 100.0 + i * 5)
        stats = s.stats
        assert "active_stragglers" in stats
        assert "reduction_factors" in stats
        assert stats["active_stragglers"] >= 1

    def test_min_batch_respected(self):
        s = StragglerAwareScheduler(base_batch_size=64, min_batch_size=2)
        for i in range(50):
            s.record_latency("n1", 100.0 + i * 20)
        b = s.get_budget("n1")
        assert b.max_batch_size >= 2


class TestGradientRecovery:
    def test_init(self):
        gr = GradientRecovery(max_steps=5)
        assert gr._recovery_progress == {}

    def test_exponential_recovery(self):
        gr = GradientRecovery(max_steps=10, base_boost=0.2)
        f = 0.3
        for i in range(10):
            f = gr.next_factor("n1", f)
        assert f >= 0.97  # Should approach 1.0 asymptotically

    def test_reset(self):
        gr = GradientRecovery(max_steps=5)
        gr.next_factor("n1", 0.5)
        assert "n1" in gr._recovery_progress
        gr.reset("n1")
        assert "n1" not in gr._recovery_progress

    def test_convergence_speed(self):
        gr = GradientRecovery(max_steps=3, base_boost=0.5)
        f = 0.0
        for i in range(5):
            f = gr.next_factor("n1", f)
        assert f >= 0.99, f"Should converge quickly, got {f}"
