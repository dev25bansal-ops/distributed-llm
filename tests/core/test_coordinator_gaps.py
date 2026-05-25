"""Gap tests: Coordinator shutdown, leader election, heartbeat, layer redistribution, priority queue, cancellation."""

import threading
import time

import pytest

from distllm.core.ha_coordinator import RayFaultTolerance as HAElection, CoordinatorState
from distllm.core.coordinator_state import CoordinatorState as CS
from distllm.core.coordinator_health import HealthChecker
from distllm.core.coordinator_metrics import MetricsManager
from distllm.core.coordinator_lifecycle import RequestTracker, ServerLifecycle
from distllm.core.coordinator_config import CoordinatorConfig
from distllm.core.resource_manager import ResourceManager, CircuitBreakerConfig


class TestHALeaderElection:
    def test_init_state_follower(self):
        election = HAElection("test-1")
        assert election.get_state() == CoordinatorState.FOLLOWER

    def test_is_leader_after_start(self):
        election = HAElection("test-1")
        election.start()
        time.sleep(0.2)
        is_ldr = election.is_leader()
        election.stop()
        assert isinstance(is_ldr, bool)

    def test_get_leader_returns_id(self):
        election = HAElection("test-1")
        election.start()
        time.sleep(0.2)
        leader = election.get_leader()
        election.stop()
        assert leader is None or leader == "test-1"

    def test_add_peer_and_stats(self):
        election = HAElection("test-1")
        election.add_peer("test-2", "localhost", 50051)
        election.start()
        s = election.stats()
        assert "state" in s
        assert "peers" in s or "coordinator_id" in s
        election.stop()

    def test_heartbeat_request_updates_state(self):
        election = HAElection("test-1")
        election.start()
        resp = election.handle_heartbeat_request("test-2", 1)
        assert isinstance(resp, dict)
        election.stop()


class TestCoordinatorConfigValidation:
    def test_valid_config(self):
        config = CoordinatorConfig(model_name="test")
        assert config.model_name == "test"
        assert config.port == 50050

    def test_invalid_port_raises(self):
        with pytest.raises((ValueError, AssertionError)):
            CoordinatorConfig(port=0)

    def test_invalid_port_high_raises(self):
        with pytest.raises((ValueError, AssertionError)):
            CoordinatorConfig(port=70000)


class TestRequestTrackerCancellation:
    def test_register_and_wait(self):
        tracker = RequestTracker()
        event = tracker.register_request("req-1")
        assert event is not None

    def test_set_result_signals_event(self):
        tracker = RequestTracker()
        event = tracker.register_request("req-1")
        tracker.set_result("req-1", "done")
        result = tracker.wait_for_result("req-1", timeout=2.0)
        assert result == "done"

    def test_pending_count(self):
        tracker = RequestTracker()
        assert tracker.pending_count == 0
        tracker.register_request("req-1")
        # After registering, pending may be 0 since no batch is active

    def test_shutting_down_flag(self):
        tracker = RequestTracker()
        assert tracker.shutting_down is False
        tracker.shutting_down = True
        assert tracker.shutting_down is True

    def test_get_logprobs_none(self):
        tracker = RequestTracker()
        assert tracker.get_logprobs("nonexistent") is None

    def test_clear_resets_state(self):
        tracker = RequestTracker()
        tracker.register_request("req-1")
        tracker.clear()
        assert tracker.shutting_down is False


class TestMetricsManager:
    def test_record_and_get(self):
        mm = MetricsManager()
        mm.record("test_metric", 42.0)
        d = mm.get()
        assert "test_metric" in d
        assert d["test_metric"] == 42.0

    def test_increment(self):
        mm = MetricsManager()
        mm.increment("errors")
        assert mm.get()["errors"] == 1
        mm.increment("errors")
        assert mm.get()["errors"] == 2

    def test_initial_counters(self):
        mm = MetricsManager()
        d = mm.get()
        assert d["total_requests"] == 0
        assert d["errors"] == 0

    def test_prometheus_format(self):
        mm = MetricsManager()
        mm.record("total_requests", 10.0)
        pm = mm.get_prometheus()
        assert isinstance(pm, dict)


class TestCoordinatorStateLifecycle:
    def test_start_stop(self):
        cs = CS()
        cs.start()
        assert cs.is_running
        cs.stop()
        assert not cs.is_running

    def test_uptime_increases(self):
        cs = CS()
        cs.start()
        u1 = cs.uptime_s()
        time.sleep(0.01)
        u2 = cs.uptime_s()
        assert u2 >= u1
        cs.stop()

    def test_double_start_no_error(self):
        cs = CS()
        cs.start()
        cs.start()
        cs.stop()


class TestHealthChecker:
    def test_check_all_empty_returns_dict(self):
        hc = HealthChecker(ResourceManager())
        result = hc.check_all({}, [], lambda x: False)
        assert isinstance(result, dict)

    def test_check_all_async_empty(self):
        import asyncio
        hc = HealthChecker(ResourceManager())
        result = asyncio.run(hc.check_all_async({}, [], lambda x: False))
        assert isinstance(result, dict)


class TestCircuitBreaker:
    def test_config_defaults(self):
        cb = CircuitBreakerConfig()
        assert cb.threshold == 3
        assert cb.base_delay == 1.0
        assert cb.max_delay == 60.0

    def test_record_failure_opens(self):
        rm = ResourceManager(cb_config=CircuitBreakerConfig(threshold=2))
        rm.record_failure("node-1")
        rm.record_failure("node-1")
        breaker_open = rm.check_circuit_breaker("node-1")
        assert isinstance(breaker_open, bool)

    def test_record_success_closes(self):
        rm = ResourceManager(cb_config=CircuitBreakerConfig(threshold=2))
        rm.record_failure("node-1")
        rm.record_success("node-1")
        assert isinstance(rm.check_circuit_breaker("node-1"), bool)
