"""Tests for Feature 16: Chaos Engineering Dashboard."""

import time
from unittest.mock import MagicMock, patch

import pytest

from distllm.chaos.injector import ChaosInjector, ChaosEvent
from distllm.chaos.scenario import ChaosScenario, ChaosStep, ScenarioRunner, ScenarioResult
from distllm.chaos.resilience import ResilienceScorer, ResilienceScore
from distllm.chaos.dashboard import render_scenario_summary, render_events, render_resilience_summary, _render_table
from distllm.chaos.dashboard import render_scenario_summary, render_events, render_resilience_summary


class MockResourceManager:
    """Mock resource manager for testing."""

    def __init__(self):
        self.failures = []
        self.failure_count = 0
        self.recovery_time = {}

    def simulate_node_failure(self, node_id: str):
        self.failures.append(node_id)
        self.failure_count += 1


class TestChaosInjector:
    @pytest.fixture
    def injector(self):
        rm = MockResourceManager()
        return ChaosInjector(resource_manager=rm, max_latency_ms=3000)

    def test_kill_node(self, injector):
        event = injector.kill_node("node-0")
        assert event.event_type == "kill_node"
        assert event.node_id == "node-0"
        assert event.result == "success"
        assert injector.resource_manager.failures == ["node-0"]

    def test_kill_node_records_event(self, injector):
        injector.kill_node("node-0")
        assert len(injector.events) == 1

    def test_add_latency(self, injector):
        event = injector.add_latency("node-0", 500)
        assert event.event_type == "add_latency"
        assert event.params["delay_ms"] == 500
        assert injector.get_latency_for_node("node-0") == 500

    def test_add_latency_capped_at_max(self, injector):
        injector.add_latency("node-0", 99999)
        assert injector.get_latency_for_node("node-0") == 3000  # max_latency_ms

    def test_clear_latency(self, injector):
        injector.add_latency("node-0", 200)
        injector.clear_latency("node-0")
        assert injector.get_latency_for_node("node-0") == 0.0

    def test_drop_message(self, injector):
        event = injector.drop_message("node-0", "tensor.*")
        assert event.event_type == "drop_message"
        assert event.params["pattern"] == "tensor.*"
        assert injector.should_drop_message("node-0", "tensor_data_v1")
        assert not injector.should_drop_message("node-0", "health_check")

    def test_clear_drop_pattern(self, injector):
        injector.drop_message("node-0", "tensor.*")
        injector.clear_drop_pattern("node-0")
        assert not injector.should_drop_message("node-0", "tensor_data")

    def test_no_drop_pattern_returns_false(self, injector):
        assert not injector.should_drop_message("node-0", "anything")

    def test_corrupt_data(self, injector):
        event = injector.corrupt_data("node-0", 0.5)
        assert event.event_type == "corrupt_data"
        assert event.params["corruption_rate"] == 0.5

    def test_corrupt_data_rate_zero(self, injector):
        injector.corrupt_data("node-0", 0.0)
        assert not injector.should_corrupt("node-0")

    def test_clear_corruption_rate(self, injector):
        injector.corrupt_data("node-0", 1.0)
        injector.clear_corruption_rate("node-0")
        assert not injector.should_corrupt("node-0")

    def test_corrupt_tensor(self, injector):
        data = b"\x00" * 100
        corrupted = injector.corrupt_tensor(data)
        assert len(corrupted) == len(data)
        assert corrupted != data  # At least one bit flipped

    def test_corrupt_empty_data(self, injector):
        assert injector.corrupt_tensor(b"") == b""

    def test_reset(self, injector):
        injector.kill_node("node-0")
        injector.add_latency("node-1", 100)
        injector.reset()
        assert len(injector.events) == 0
        assert injector.get_latency_for_node("node-1") == 0.0


class TestResourceManagerSimulateFailure:
    def test_simulate_failure_opens_circuit_breaker(self):
        from distllm.core.resource_manager import ResourceManager, CircuitBreakerConfig

        rm = ResourceManager(cb_config=CircuitBreakerConfig(threshold=2))
        rm.simulate_node_failure("node-0")

        # Circuit breaker should be open
        assert rm.check_circuit_breaker("node-0") is True
        assert rm._node_failure_counts["node-0"] >= 2

    def test_simulate_failure_records_event(self):
        from distllm.core.resource_manager import ResourceManager

        rm = ResourceManager()
        initial_failures = rm._metrics["node_failures"]
        rm.simulate_node_failure("node-0")
        assert rm._metrics["node_failures"] > initial_failures


class TestScenarioRunner:
    @pytest.fixture
    def injector(self):
        rm = MockResourceManager()
        return ChaosInjector(resource_manager=rm, max_latency_ms=3000)

    def test_run_single_step_scenario(self, injector):
        scenario = ChaosScenario(
            name="kill_one_node",
            steps=[ChaosStep(action="kill_node", params={"node_id": "node-0"})],
            expected_recovery_time_s=10.0,
        )
        runner = ScenarioRunner(injector)
        result = runner.run_scenario(scenario)
        assert result.scenario_name == "kill_one_node"
        assert result.steps_executed == 1
        assert result.steps_failed == 0

    def test_run_multi_step_scenario(self, injector):
        scenario = ChaosScenario(
            name="latency_and_kill",
            steps=[
                ChaosStep(action="add_latency", params={"node_id": "node-0", "delay_ms": 100}),
                ChaosStep(action="kill_node", params={"node_id": "node-1"}, delay_after=0.01),
            ],
        )
        runner = ScenarioRunner(injector)
        result = runner.run_scenario(scenario)
        assert result.steps_executed == 2
        assert result.steps_failed == 0

    def test_run_scenario_with_unknown_action(self, injector):
        scenario = ChaosScenario(
            name="bad_action",
            steps=[ChaosStep(action="unknown_action", params={})],
        )
        runner = ScenarioRunner(injector)
        result = runner.run_scenario(scenario)
        assert result.steps_failed == 1

    def test_results_list(self, injector):
        scenario = ChaosScenario(name="test", steps=[])
        runner = ScenarioRunner(injector)
        runner.run_scenario(scenario)
        assert len(runner.results) == 1


class TestResilienceScorer:
    def test_perfect_score(self):
        score = ResilienceScorer.compute_score(
            actual_recovery_time_s=5.0,
            expected_recovery_time_s=10.0,
            has_data_loss=False,
            actual_error_rate=0.0,
            max_acceptable_error_rate=0.05,
        )
        assert score.recovery_time_score == 100.0
        assert score.data_loss_score == 100.0
        assert score.error_rate_score == 100.0
        assert score.overall == 100.0
        assert score.grade == "A"

    def test_zero_recovery_time(self):
        score = ResilienceScorer.compute_score(
            actual_recovery_time_s=0.0,
            expected_recovery_time_s=10.0,
            has_data_loss=False,
            actual_error_rate=0.0,
            max_acceptable_error_rate=0.05,
        )
        assert score.recovery_time_score == 100.0

    def test_data_loss_zeroes_component(self):
        score = ResilienceScorer.compute_score(
            actual_recovery_time_s=5.0,
            expected_recovery_time_s=10.0,
            has_data_loss=True,
            actual_error_rate=0.0,
            max_acceptable_error_rate=0.05,
        )
        assert score.data_loss_score == 0.0
        assert score.overall < 100.0

    def test_high_error_rate(self):
        score = ResilienceScorer.compute_score(
            actual_recovery_time_s=5.0,
            expected_recovery_time_s=10.0,
            has_data_loss=False,
            actual_error_rate=0.1,
            max_acceptable_error_rate=0.05,
        )
        assert score.error_rate_score == 0.0

    def test_grade_f(self):
        score = ResilienceScorer.compute_score(
            actual_recovery_time_s=100.0,
            expected_recovery_time_s=10.0,
            has_data_loss=True,
            actual_error_rate=0.5,
            max_acceptable_error_rate=0.05,
        )
        assert score.grade == "F"
        assert score.overall == 0.0


class TestDashboard:
    def test_render_empty_summary(self):
        output = render_resilience_summary([])
        assert "No chaos scenarios" in output

    def test_render_scenario_summary(self):
        results = [ScenarioResult(
            scenario_name="test",
            steps_executed=3,
            steps_failed=0,
            total_duration_s=5.0,
            actual_recovery_time_s=3.0,
            actual_error_rate=0.0,
        )]
        scores = [ResilienceScore(overall=95.0, recovery_time_score=100, data_loss_score=100, error_rate_score=85, scenario_name="test")]
        output = render_scenario_summary(results, scores)
        assert "test" in output
        assert "95" in output

    def test_render_events(self):
        events = [ChaosEvent(
            event_type="kill_node",
            node_id="node-0",
            timestamp=time.time(),
            params={},
            result="success",
            duration_s=0.01,
        )]
        output = render_events(events)
        assert "kill_node" in output
        assert "node-0" in output

    def test_render_resilience_summary(self):
        scores = [ResilienceScore(overall=90, recovery_time_score=100, data_loss_score=100, error_rate_score=70, scenario_name="good")]
        output = render_resilience_summary(scores)
        assert "Average resilience score: 90.0" in output
        assert "Best scenario: good" in output

    def test_render_table_empty(self):
        output = _render_table(["A", "B"], [])
        assert "A" in output

    def test_render_scenario_summary_empty(self):
        output = render_scenario_summary([], [])
        assert "Scenario" in output

    def test_render_events_single(self):
        event = ChaosEvent(event_type="kill", node_id="n1", timestamp=time.time(), params={}, result="success", duration_s=0.5)
        output = render_events([event])
        assert "kill" in output
        assert "n1" in output
        assert "success" in output

    def test_render_events_empty(self):
        output = render_events([])
        assert "Type" in output

    def test_render_resilience_summary_no_scores(self):
        output = render_resilience_summary([])
        assert "No chaos scenarios have been executed yet" in output


# ===========================================================================
# Injector — delay injection
# ===========================================================================


class TestChaosInjectorDelay:
    def test_add_latency_stores_delay(self):
        injector = ChaosInjector(MagicMock(), max_latency_ms=5000)
        injector.add_latency("node-1", delay_ms=200)
        assert injector.get_latency_for_node("node-1") == 200

    def test_add_latency_capped_at_max(self):
        injector = ChaosInjector(MagicMock(), max_latency_ms=1000)
        injector.add_latency("node-1", delay_ms=5000)
        assert injector.get_latency_for_node("node-1") == 1000

    def test_clear_latency_removes(self):
        injector = ChaosInjector(MagicMock())
        injector.add_latency("node-1", delay_ms=200)
        injector.clear_latency("node-1")
        assert injector.get_latency_for_node("node-1") == 0.0

    def test_default_latency_zero(self):
        injector = ChaosInjector(MagicMock())
        assert injector.get_latency_for_node("nonexistent") == 0.0

    def test_latency_records_event(self):
        injector = ChaosInjector(MagicMock())
        event = injector.add_latency("node-1", delay_ms=150)
        assert event.event_type == "add_latency"
        assert event.node_id == "node-1"
        assert event.result == "success"


# ===========================================================================
# Injector — error injection (fake exception via kill_node)
# ===========================================================================


class TestChaosInjectorError:
    def test_kill_node_records_event(self):
        rm = MagicMock()
        injector = ChaosInjector(rm)
        event = injector.kill_node("node-fail")
        assert event.event_type == "kill_node"
        assert event.node_id == "node-fail"
        rm.simulate_node_failure.assert_called_with("node-fail")

    def test_kill_triggers_circuit_breaker(self):
        rm = MagicMock()
        injector = ChaosInjector(rm)
        injector.kill_node("node-bad")
        assert len(injector.events) == 1

    def test_drop_message_records_event(self):
        injector = ChaosInjector(MagicMock())
        event = injector.drop_message("node-1", ".*error.*")
        assert event.event_type == "drop_message"
        assert event.result == "success"

    def test_should_drop_message_matches_pattern(self):
        injector = ChaosInjector(MagicMock())
        injector.drop_message("node-1", ".*timeout.*")
        assert injector.should_drop_message("node-1", "request timeout") is True
        assert injector.should_drop_message("node-1", "normal message") is False

    def test_clear_drop_pattern_removes(self):
        injector = ChaosInjector(MagicMock())
        injector.drop_message("node-1", ".*drop.*")
        injector.clear_drop_pattern("node-1")
        assert injector.should_drop_message("node-1", "this should drop") is False

    def test_drop_without_pattern_returns_false(self):
        injector = ChaosInjector(MagicMock())
        assert injector.should_drop_message("node-x", "any message") is False

    def test_corrupt_data_records_event(self):
        injector = ChaosInjector(MagicMock())
        event = injector.corrupt_data("node-1", corruption_rate=0.5)
        assert event.event_type == "corrupt_data"
        assert event.result == "success"

    def test_corrupt_data_zero_rate_no_corruption(self):
        injector = ChaosInjector(MagicMock())
        injector.corrupt_data("node-1", corruption_rate=0.0)
        assert injector.should_corrupt("node-1") is False

    def test_clear_corruption_rate(self):
        injector = ChaosInjector(MagicMock())
        injector.corrupt_data("node-1", corruption_rate=0.5)
        injector.clear_corruption_rate("node-1")
        assert injector.should_corrupt("node-1") is False

    def test_corrupt_tensor_flips_bits(self):
        injector = ChaosInjector(MagicMock())
        original = b"hello world"
        corrupted = injector.corrupt_tensor(original)
        assert corrupted != original
        assert len(corrupted) == len(original)

    def test_corrupt_empty_data(self):
        injector = ChaosInjector(MagicMock())
        result = injector.corrupt_tensor(b"")
        assert result == b""

    def test_multiple_events_recorded(self):
        injector = ChaosInjector(MagicMock())
        injector.add_latency("n1", 100)
        injector.drop_message("n2", "pattern")
        injector.corrupt_data("n3", 0.1)
        assert len(injector.events) == 3
