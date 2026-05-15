"""Tests for Feature 17: Automated Canary Deployments."""

import time
from unittest.mock import MagicMock, patch

import pytest

from distllm.deploy.traffic_splitter import TrafficSplitter
from distllm.deploy.rollout_strategy import RolloutStage, RolloutStrategy, RolloutState
from distllm.deploy.canary_controller import CanaryController
from distllm.deploy.canary_crd import CanarySpec, NodePoolWithCanary


class TestTrafficSplitter:
    def test_all_stable(self):
        splitter = TrafficSplitter(canary_pct=0.0)
        assert splitter.select_version("req-1") == "stable"
        assert splitter.select_version("req-2") == "stable"

    def test_all_canary(self):
        splitter = TrafficSplitter(canary_pct=100.0)
        assert splitter.select_version("req-1") == "canary"

    def test_50_50_distribution(self):
        splitter = TrafficSplitter(canary_pct=50.0)
        request_ids = [f"req-{i}" for i in range(10000)]
        stats = splitter.get_distribution_stats(request_ids)
        # Should be approximately 50/50 (within 5% tolerance for randomness)
        assert 40 < stats["canary_pct"] < 60

    def test_5_pct_distribution(self):
        splitter = TrafficSplitter(canary_pct=5.0)
        request_ids = [f"req-{i}" for i in range(10000)]
        stats = splitter.get_distribution_stats(request_ids)
        assert 2 < stats["canary_pct"] < 8

    def test_consistent_routing(self):
        splitter = TrafficSplitter(canary_pct=50.0)
        # Same request_id should always route to same version
        version1 = splitter.select_version("consistent-req")
        for _ in range(100):
            assert splitter.select_version("consistent-req") == version1

    def test_set_canary_pct(self):
        splitter = TrafficSplitter(canary_pct=5.0)
        splitter.set_canary_pct(25.0)
        assert splitter.canary_pct == 25.0

    def test_set_canary_pct_clamped(self):
        splitter = TrafficSplitter()
        splitter.set_canary_pct(150.0)
        assert splitter.canary_pct == 100.0
        splitter.set_canary_pct(-10.0)
        assert splitter.canary_pct == 0.0

    def test_is_active(self):
        splitter = TrafficSplitter(canary_pct=5.0)
        assert splitter.is_active() is True
        splitter.set_canary_pct(0.0)
        assert splitter.is_active() is False
        splitter.set_canary_pct(100.0)
        assert splitter.is_active() is False


class TestRolloutStrategy:
    def test_create_rollout(self):
        strategy = RolloutStrategy()
        state = strategy.create_rollout("v2")
        assert state.canary_version == "v2"
        assert state.current_stage_index == 0
        assert state.current_weight_pct == 0.0

    def test_get_current_stage(self):
        strategy = RolloutStrategy()
        state = strategy.create_rollout("v2")
        stage = strategy.get_current_stage(state)
        assert stage is not None
        assert stage.weight_pct == 5

    def test_advance_stage(self):
        strategy = RolloutStrategy()
        state = strategy.create_rollout("v2")
        assert strategy.advance_stage(state)
        assert state.current_stage_index == 1
        assert state.current_weight_pct == 25

    def test_advance_past_last_stage(self):
        strategy = RolloutStrategy()
        state = strategy.create_rollout("v2")
        # Advance through all 5 stages
        for _ in range(5):
            strategy.advance_stage(state)
        assert state.is_complete is True
        assert state.current_weight_pct == 100.0

    def test_rollback_on_error_rate(self):
        strategy = RolloutStrategy()
        state = strategy.create_rollout("v2")
        should = strategy.should_rollback(
            state,
            current_error_rate=0.1,  # Above 0.05 threshold
            current_p99_latency_ms=100,
            stable_p99_latency_ms=100,
        )
        assert should is True

    def test_rollback_on_latency(self):
        strategy = RolloutStrategy()
        state = strategy.create_rollout("v2")
        should = strategy.should_rollback(
            state,
            current_error_rate=0.0,
            current_p99_latency_ms=500,  # 5x stable
            stable_p99_latency_ms=100,
        )
        assert should is True

    def test_no_rollback_within_thresholds(self):
        strategy = RolloutStrategy()
        state = strategy.create_rollout("v2")
        should = strategy.should_rollback(
            state,
            current_error_rate=0.01,
            current_p99_latency_ms=150,
            stable_p99_latency_ms=100,
        )
        assert should is False

    def test_trigger_rollback(self):
        strategy = RolloutStrategy()
        state = strategy.create_rollout("v2")
        strategy.trigger_rollback(state)
        assert state.is_rolling_back is True
        assert state.current_weight_pct == 0.0

    def test_record_request(self):
        strategy = RolloutStrategy()
        state = strategy.create_rollout("v2")
        strategy.record_request(state, was_error=False, latency_ms=100)
        strategy.record_request(state, was_error=True, latency_ms=200)
        assert state.total_requests == 2
        assert state.error_count == 1
        assert state.avg_latency_ms == 150.0

    def test_get_analysis(self):
        strategy = RolloutStrategy()
        state = strategy.create_rollout("v2")
        strategy.record_request(state, was_error=True, latency_ms=100)
        analysis = strategy.get_analysis(state)
        assert analysis["error_rate"] == 1.0
        assert analysis["total_requests"] == 1


class TestCanaryController:
    @pytest.fixture
    def controller(self):
        return CanaryController(
            stable_version="v1",
            canary_version="v2",
            rollback_threshold=0.05,
        )

    def test_start_canary(self, controller):
        state = controller.start_canary("v2")
        assert controller.is_active
        assert state.canary_version == "v2"

    def test_cannot_start_duplicate(self, controller):
        controller.start_canary("v2")
        with pytest.raises(RuntimeError, match="already active"):
            controller.start_canary("v3")

    def test_get_canary_version_when_inactive(self, controller):
        assert controller.get_canary_version("req-1") == "v1"

    def test_get_canary_version_when_active(self, controller):
        controller.start_canary("v2")
        version = controller.get_canary_version("req-1")
        assert version in ("v1", "v2")

    def test_advance_on_good_metrics(self, controller):
        controller.start_canary("v2")
        # Mock time to simulate analysis window elapsed
        controller._rollout_state.stage_start_time = time.time() - 9999
        result = controller.check_and_advance(
            current_error_rate=0.0,
            current_p99_ms=100,
            stable_p99_ms=100,
        )
        assert result == "advanced"
        assert controller._rollout_state.current_weight_pct == 25

    def test_rollback_on_bad_error_rate(self, controller):
        controller.start_canary("v2")
        controller._rollout_state.stage_start_time = time.time() - 9999
        result = controller.check_and_advance(
            current_error_rate=0.5,  # Very high error rate
            current_p99_ms=100,
            stable_p99_ms=100,
        )
        assert result == "rolled_back"
        assert controller.splitter.canary_pct == 0.0

    def test_rollback_on_high_latency(self, controller):
        controller.start_canary("v2")
        controller._rollout_state.stage_start_time = time.time() - 9999
        result = controller.check_and_advance(
            current_error_rate=0.0,
            current_p99_ms=1000,  # 10x stable
            stable_p99_ms=100,
        )
        assert result == "rolled_back"

    def test_abort_canary(self, controller):
        controller.start_canary("v2")
        controller.abort_canary()
        assert not controller.is_active


class TestCanaryCRD:
    def test_canary_spec_defaults(self):
        spec = CanarySpec()
        assert spec.enabled is False
        assert len(spec.stages) == 5

    def test_canary_spec_to_dict(self):
        spec = CanarySpec(enabled=True, canary_version="v2", canary_weight=10)
        d = spec.to_dict()
        assert d["enabled"] is True
        assert d["canary_version"] == "v2"
        assert d["canary_weight"] == 10

    def test_canary_spec_from_dict(self):
        data = {"enabled": True, "canary_version": "v3", "rollback_threshold": 0.02}
        spec = CanarySpec.from_dict(data)
        assert spec.enabled is True
        assert spec.rollback_threshold == 0.02

    def test_node_pool_with_canary_fields(self):
        pool = NodePoolWithCanary(
            host="localhost",
            port_range="50051-50060",
            start_layer=0,
            end_layer=5,
            canary_weight=10,
            canary_version="v2",
        )
        d = pool.to_dict()
        assert d["canary_weight"] == 10
        assert d["canary_version"] == "v2"
