"""Tests for Pareto-optimal multi-objective scheduler."""

from __future__ import annotations

from distllm.core.advanced_scheduling.multi_objective import (
    MultiObjectiveScheduler,
    ObjectiveWeights,
    SystemState,
)


class _Req:
    """Stub request for testing."""
    def __init__(self, latency_ms=100, node_id="", priority=2):
        self.estimated_latency_ms = latency_ms
        self.node_id = node_id
        self.priority = priority


class TestObjectiveWeights:
    def test_normalize(self):
        w = ObjectiveWeights(latency=0.5, cost=0.5, energy=0.5, throughput=0.5)
        n = w.normalize()
        assert abs(n.latency - 0.25) < 0.01

    def test_zero_total(self):
        w = ObjectiveWeights(latency=0, cost=0, energy=0, throughput=0)
        n = w.normalize()
        assert abs(n.latency - 0.25) < 0.01


class TestMultiObjectiveScheduler:
    def test_init(self):
        s = MultiObjectiveScheduler()
        assert abs(sum([s._weights.latency, s._weights.cost, s._weights.energy, s._weights.throughput]) - 1.0) < 0.01

    def test_score_in_range(self):
        s = MultiObjectiveScheduler()
        s.update_cost_profile("n1", 1.0)
        s.update_energy_profile("n1", 50.0)
        score = s.score_request(_Req(node_id="n1"))
        assert 0 <= score <= 1.0

    def test_score_honors_latency(self):
        s = MultiObjectiveScheduler()
        fast = s.score_request(_Req(latency_ms=10))
        slow = s.score_request(_Req(latency_ms=5000))
        assert fast >= slow

    def test_select_best_returns_highest_score(self):
        s = MultiObjectiveScheduler()
        s.update_cost_profile("n1", 1.0)
        s.update_cost_profile("n2", 10.0)
        r1 = _Req(node_id="n1")
        r2 = _Req(node_id="n2")
        best = s.select_best([r1, r2])
        assert best is not None

    def test_select_best_empty(self):
        s = MultiObjectiveScheduler()
        assert s.select_best([]) is None

    def test_weight_adaptation_thermal(self):
        s = MultiObjectiveScheduler(energy_weight=0.1)
        hot = SystemState(gpu_temperature_c=95.0, gpu_thermal_threshold_c=83.0)
        old_energy = s._weights.energy
        for _ in range(20):
            s.score_request(_Req(), hot)
        assert s._weights.energy > old_energy

    def test_weight_adaptation_latency(self):
        s = MultiObjectiveScheduler(latency_weight=0.1)
        slow = SystemState(avg_latency_p99_ms=2000, latency_slo_ms=1000)
        old_latency = s._weights.latency
        for _ in range(20):
            s.score_request(_Req(), slow)
        assert s._weights.latency > old_latency

    def test_weight_adaptation_slo(self):
        s = MultiObjectiveScheduler(latency_weight=0.1)
        bad_slo = SystemState(slo_attainment_rate=0.5)
        old_latency = s._weights.latency
        for _ in range(20):
            s.score_request(_Req(), bad_slo)
        assert s._weights.latency > old_latency

    def test_all_objectives_minimum_weight(self):
        s = MultiObjectiveScheduler()
        state = SystemState(
            gpu_temperature_c=95.0, gpu_thermal_threshold_c=83.0,
            avg_latency_p99_ms=5000, latency_slo_ms=100,
            node_cost_per_hour=100.0, cluster_utilization=0.95,
        )
        for _ in range(50):
            s.score_request(_Req(), state)
        w = s._weights
        assert w.latency >= 0.05
        assert w.cost >= 0.05
        assert w.energy >= 0.05
        assert w.throughput >= 0.05

    def test_stats(self):
        s = MultiObjectiveScheduler()
        s.score_request(_Req())
        stats = s.stats
        assert "weights" in stats
        assert "total_requests" in stats
        assert stats["total_requests"] >= 1
