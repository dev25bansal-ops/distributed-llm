"""Tests for chaos engineering orchestrator."""

from distllm.dist.chaos.chaos_orchestrator import (
    ChaosOrchestrator,
    FaultScenario,
    FaultType,
    FaultTarget,
    ScenarioGenerator,
    ExperimentResult,
)


class TestFaultScenario:
    def test_scenario_creation(self):
        s = FaultScenario(FaultType.NODE_KILL, FaultTarget.WORKER, 30, 0.8)
        assert s.fault_type == FaultType.NODE_KILL
        assert s.target == FaultTarget.WORKER
        assert s.duration_s == 30
        assert s.intensity == 0.8

    def test_random_scenario(self):
        s = ScenarioGenerator.random_scenario(seed=42)
        assert isinstance(s.fault_type, FaultType)
        assert isinstance(s.target, FaultTarget)
        assert 0.0 < s.intensity <= 1.0

    def test_predefined_scenarios(self):
        scenarios = ScenarioGenerator.predefined_scenarios()
        assert len(scenarios) >= 5
        assert all(isinstance(s, FaultScenario) for s in scenarios)


class TestChaosOrchestrator:
    def test_run_scenario(self):
        chaos = ChaosOrchestrator(staging=True)
        scenario = FaultScenario(FaultType.NODE_KILL, FaultTarget.WORKER, 5)
        result = chaos.run_scenario(scenario, timeout_s=30)
        assert isinstance(result, ExperimentResult)
        assert result.success is True
        assert result.recovery_time_s > 0

    def test_metrics(self):
        chaos = ChaosOrchestrator(staging=True)
        assert chaos.metrics["experiments_run"] == 0
        chaos.run_scenario(FaultScenario(FaultType.NODE_KILL, FaultTarget.WORKER, 5))
        assert chaos.metrics["experiments_run"] == 1
        assert 0 <= chaos.metrics["recovery_success_rate"] <= 1.0

    def test_run_suite(self):
        chaos = ChaosOrchestrator(staging=True)
        results = chaos.run_suite()
        assert len(results) >= 5
        assert all(r.success for r in results)
