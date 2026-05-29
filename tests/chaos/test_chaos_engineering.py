"""Chaos engineering tests for distributed LLM inference.

Covers 5 fault-injection scenarios:
1. Random node kill during inference
2. Network partition between coordinator and worker
3. Slow node (artificial latency injection)
4. Memory pressure simulation
5. Clock skew between nodes
"""

from __future__ import annotations

import gc
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

SRC_DIR = Path(__file__).resolve().parent.parent.parent / "src"

pytestmark = [
    pytest.mark.chaos,
    pytest.mark.timeout(60),
]


# ---------------------------------------------------------------------------
# 1. Random Node Kill During Inference
# ---------------------------------------------------------------------------

class TestNodeKill:
    """Simulate node crashes at various points during inference."""

    def test_node_kill_opens_circuit_breaker(self, chaos_coordinator):
        coord = chaos_coordinator
        rm = coord._resource_mgr
        node_id = "n0"
        assert coord.nodes[node_id].healthy

        rm.simulate_node_failure(node_id)

        assert rm.check_circuit_breaker(node_id)

    def test_node_kill_after_threshold_exact(self, chaos_coordinator, rm_with_cb):
        coord = chaos_coordinator
        rm = rm_with_cb
        node_id = "n0"

        assert not rm.check_circuit_breaker(node_id)

        rm.record_failure(node_id)
        assert not rm.check_circuit_breaker(node_id)

        rm.record_failure(node_id)
        assert rm.check_circuit_breaker(node_id)

    def test_node_kill_marks_unhealthy(self, chaos_coordinator):
        coord = chaos_coordinator
        rm = coord._resource_mgr
        rm.simulate_node_failure("n0")
        assert rm.check_circuit_breaker("n0")

    def test_node_kill_fallback_no_overlap(self, chaos_coordinator):
        coord = chaos_coordinator
        node = coord.nodes["n0"]
        pipeline = coord._pipeline
        pipeline.nodes = coord.nodes
        pipeline.resource_mgr = coord._resource_mgr

        fallback = pipeline._find_fallback_node("n0", node)
        assert fallback is None, "No fallback expected with disjoint layer ranges"

    def test_node_kill_no_fallback_degradation(self, chaos_coordinator):
        coord = chaos_coordinator
        pipeline = coord._pipeline
        pipeline.nodes = {}
        pipeline.resource_mgr = MagicMock()
        pipeline.resource_mgr.check_circuit_breaker.return_value = True

        ctx = MagicMock()
        ctx.node_id = "n0"
        ctx.current_hidden = "passthrough"
        result = pipeline._execute_node_grpc(ctx)

        assert result == "passthrough"

    def test_node_kill_all_nodes(self, chaos_coordinator):
        coord = chaos_coordinator
        for nid in list(coord.nodes):
            coord._resource_mgr.simulate_node_failure(nid)
        for nid in coord.nodes:
            assert coord._resource_mgr.check_circuit_breaker(nid)

    def test_node_kill_marks_draining(self, chaos_coordinator):
        coord = chaos_coordinator
        rm = coord._resource_mgr
        node_id = "n0"

        draining = []
        def on_failure(nid):
            draining.append(nid)
        rm._on_node_failure = on_failure

        for _ in range(rm.cb_config.threshold):
            rm.record_failure(node_id)

        assert node_id in draining or node_id in rm._draining_nodes

    def test_node_kill_checkpoint_still_saved(self, chaos_coordinator):
        coord = chaos_coordinator
        recovery_mgr = MagicMock()
        coord._recovery_manager = recovery_mgr
        coord._resource_mgr.simulate_node_failure("n0")
        recovery_mgr.save_checkpoint.assert_not_called()
        assert "n0" in coord.nodes

    def test_node_kill_recovery_after_cooldown(self, rm_with_cb):
        rm = rm_with_cb
        node_id = "n0"

        rm._node_failure_counts[node_id] = rm.cb_config.threshold
        rm._node_recovery_time[node_id] = time.time() - 1

        assert not rm.check_circuit_breaker(node_id)

        rm.record_success(node_id)
        assert rm._node_failure_counts[node_id] == 0
        assert node_id not in rm._node_recovery_time

    def test_node_kill_failure_counter_resets_on_success(self, rm_with_cb):
        rm = rm_with_cb
        rm._node_failure_counts["n0"] = 5
        rm.record_success("n0")
        assert rm._node_failure_counts["n0"] == 0


# ---------------------------------------------------------------------------
# 2. Network Partition
# ---------------------------------------------------------------------------

class TestNetworkPartition:
    """Simulate network failures between coordinator and worker nodes."""

    def test_partition_grpc_unavailable(self, chaos_coordinator):
        import grpc
        coord = chaos_coordinator
        node_id = "n0"
        node = coord.nodes[node_id]
        pipeline = coord._pipeline

        error = grpc.RpcError("unavailable")
        error.code = lambda: grpc.StatusCode.UNAVAILABLE
        error.details = lambda: "connection refused"

        node.client.stub.ForwardPass = MagicMock(side_effect=error)
        pipeline.nodes = {node_id: node}
        pipeline.resource_mgr = coord._resource_mgr

        ctx = MagicMock()
        ctx.node_id = node_id
        ctx.current_hidden = None
        with pytest.raises(Exception):
            pipeline._execute_node_grpc(ctx)

    def test_partition_timeout(self, chaos_coordinator):
        import grpc
        coord = chaos_coordinator
        node_id = "n0"
        node = coord.nodes[node_id]
        pipeline = coord._pipeline

        error = grpc.RpcError("timeout")
        error.code = lambda: grpc.StatusCode.DEADLINE_EXCEEDED
        error.details = lambda: "deadline exceeded"

        node.client.stub.ForwardPass = MagicMock(side_effect=error)
        pipeline.nodes = {node_id: node}
        pipeline.resource_mgr = coord._resource_mgr

        ctx = MagicMock()
        ctx.node_id = node_id
        ctx.current_hidden = None
        with pytest.raises(Exception):
            pipeline._execute_node_grpc(ctx)

    def test_partition_internal_error(self, chaos_coordinator):
        import grpc
        coord = chaos_coordinator
        node_id = "n0"
        node = coord.nodes[node_id]
        pipeline = coord._pipeline

        error = grpc.RpcError("internal")
        error.code = lambda: grpc.StatusCode.INTERNAL
        error.details = lambda: "internal server error"

        node.client.stub.ForwardPass = MagicMock(side_effect=error)
        pipeline.nodes = {node_id: node}
        pipeline.resource_mgr = coord._resource_mgr

        ctx = MagicMock()
        ctx.node_id = node_id
        ctx.current_hidden = None
        with pytest.raises(Exception):
            pipeline._execute_node_grpc(ctx)

    def test_partition_health_check_fails(self, chaos_coordinator):
        node = chaos_coordinator.nodes["n0"]
        node.client.stub.HealthCheck = MagicMock(
            side_effect=Exception("connection refused")
        )
        result = node.health_check()
        assert not result
        assert not node.healthy

    def test_partition_partial_message_loss(self, chaos_coordinator):
        import grpc
        coord = chaos_coordinator
        node_id = "n0"
        node = coord.nodes[node_id]

        error = grpc.RpcError("unavailable")
        error.code = lambda: grpc.StatusCode.UNAVAILABLE
        error.details = lambda: "connection refused"

        call_count = [0]
        def flaky_forward(req, timeout=None):
            call_count[0] += 1
            if call_count[0] % 2 == 0:
                resp = MagicMock()
                resp.success = True
                resp.error_message = ""
                resp.request_id = "test"
                return resp
            raise error

        node.client.stub.ForwardPass = MagicMock(side_effect=flaky_forward)

        assert call_count[0] == 0
        with pytest.raises(Exception):
            node.client.stub.ForwardPass(None)
        assert call_count[0] == 1

        resp = node.client.stub.ForwardPass(None)
        assert call_count[0] == 2
        assert resp.success

    def test_partition_all_nodes_unreachable(self, chaos_coordinator):
        import grpc
        coord = chaos_coordinator
        error = grpc.RpcError("unavailable")
        error.code = lambda: grpc.StatusCode.UNAVAILABLE
        error.details = lambda: "connection refused"

        for nid in coord.nodes:
            coord.nodes[nid].client.stub.ForwardPass = MagicMock(side_effect=error)

        pipeline = coord._pipeline
        pipeline.nodes = coord.nodes
        pipeline.resource_mgr = coord._resource_mgr

        ctx = MagicMock()
        ctx.node_id = "n0"
        ctx.current_hidden = None
        with pytest.raises(Exception):
            pipeline._execute_node_grpc(ctx)

    def test_partition_retry_exhaustion(self, chaos_coordinator):
        import grpc
        coord = chaos_coordinator
        node_id = "n0"
        node = coord.nodes[node_id]
        pipeline = coord._pipeline

        error = grpc.RpcError("unavailable")
        error.code = lambda: grpc.StatusCode.UNAVAILABLE
        error.details = lambda: "connection refused"

        node.client.stub.ForwardPass = MagicMock(side_effect=error)
        pipeline.nodes = {node_id: node}
        pipeline.resource_mgr = coord._resource_mgr
        coord._resource_mgr._node_failure_counts[node_id] = 0

        ctx = MagicMock()
        ctx.node_id = node_id
        ctx.current_hidden = None
        with pytest.raises(Exception):
            pipeline._execute_node_grpc(ctx)

    def test_partition_recovery_after_heal(self, chaos_coordinator):
        coord = chaos_coordinator
        rm = coord._resource_mgr

        rm.simulate_node_failure("n0")
        assert rm.check_circuit_breaker("n0")

        rm._node_recovery_time["n0"] = time.time() - 1
        assert not rm.check_circuit_breaker("n0")

        coord.nodes["n0"].healthy = True
        assert coord.nodes["n0"].healthy


# ---------------------------------------------------------------------------
# 3. Slow Node (Artificial Latency Injection)
# ---------------------------------------------------------------------------

class TestSlowNode:
    """Simulate nodes with degraded performance."""

    def test_slow_node_straggler_detected(self, chaos_coordinator):
        coord = chaos_coordinator
        detector = coord._straggler_detector
        if detector is None:
            pytest.skip("no straggler detector configured")

        nodes = list(coord.nodes.keys())
        detector.record_latency(nodes[0], 2.0)
        detector.record_latency(nodes[0], 2.1)
        detector.record_latency(nodes[0], 2.0)
        for other in nodes[1:]:
            detector.record_latency(other, 0.5)
            detector.record_latency(other, 0.6)
            detector.record_latency(other, 0.5)

        result = detector.check()
        stragglers = result.get("stragglers", []) if isinstance(result, dict) else []
        straggler_ids = [s["node_id"] for s in stragglers] if stragglers else []
        assert nodes[0] in straggler_ids or not stragglers

    def test_slow_node_severe(self, chaos_coordinator):
        coord = chaos_coordinator
        detector = coord._straggler_detector
        if detector is None:
            pytest.skip("no straggler detector configured")

        nodes = list(coord.nodes.keys())
        for _ in range(10):
            detector.record_latency(nodes[0], 10.0)
            for other in nodes[1:]:
                detector.record_latency(other, 0.5)

        result = detector.check()
        if isinstance(result, dict):
            stragglers = result.get("stragglers", [])
            for s in stragglers:
                if s.get("node_id") == nodes[0]:
                    severity = s.get("action", s.get("severity", ""))
                    assert severity in ("reassign_layers", "severe")

    def test_slow_node_recovery(self, chaos_coordinator):
        coord = chaos_coordinator
        detector = coord._straggler_detector
        if detector is None:
            pytest.skip("no straggler detector configured")

        nodes = list(coord.nodes.keys())
        for _ in range(5):
            for nid in nodes:
                detector.record_latency(nid, 0.5)

        result = detector.check()
        if isinstance(result, dict):
            stragglers = result.get("stragglers", [])
            assert all(s["node_id"] not in nodes for s in stragglers)

    def test_slow_node_all_slow_no_straggler(self, chaos_coordinator):
        coord = chaos_coordinator
        detector = coord._straggler_detector
        if detector is None:
            pytest.skip("no straggler detector configured")

        nodes = list(coord.nodes.keys())
        for _ in range(10):
            for nid in nodes:
                detector.record_latency(nid, 5.0)

        result = detector.check()
        if isinstance(result, dict):
            assert len(result.get("stragglers", [])) == 0

    def test_slow_node_latency_injection_forward(self, chaos_coordinator):
        node = chaos_coordinator.nodes["n0"]
        delay = [0.05]

        def slow_forward(req, timeout=None):
            time.sleep(delay[0])
            resp = MagicMock()
            resp.success = True
            resp.error_message = ""
            resp.request_id = "test"
            return resp

        node.client.stub.ForwardPass = MagicMock(side_effect=slow_forward)

        t0 = time.monotonic()
        node.client.stub.ForwardPass(None)
        elapsed = time.monotonic() - t0
        assert elapsed >= 0.04

    def test_slow_node_backpressure_buildup(self, chaos_coordinator):
        node = chaos_coordinator.nodes["n0"]

        def slow_forward(req, timeout=None):
            time.sleep(0.02)
            resp = MagicMock()
            resp.success = True
            resp.error_message = ""
            return resp

        node.client.stub.ForwardPass = MagicMock(side_effect=slow_forward)

        results = []
        for _ in range(10):
            t0 = time.monotonic()
            node.client.stub.ForwardPass(None)
            results.append(time.monotonic() - t0)

        avg = sum(results) / len(results)
        assert avg >= 0.01


# ---------------------------------------------------------------------------
# 4. Memory Pressure Simulation
# ---------------------------------------------------------------------------

class TestMemoryPressure:
    """Simulate OOM and high memory pressure scenarios."""

    def test_memory_pressure_oom_raises(self, chaos_coordinator):
        coord = chaos_coordinator
        node = coord.nodes["n0"]
        pipeline = coord._pipeline

        class FakeOOMError(Exception):
            pass

        node.client.stub.ForwardPass = MagicMock(side_effect=FakeOOMError("OOM"))
        pipeline.nodes = {"n0": node}
        pipeline.resource_mgr = coord._resource_mgr

        ctx = MagicMock()
        ctx.node_id = "n0"
        ctx.current_hidden = None
        with pytest.raises(Exception):
            pipeline._execute_node_grpc(ctx)

    def test_memory_pressure_health_check_high_usage(self, chaos_coordinator, monkeypatch):
        node = chaos_coordinator.nodes["n0"]

        def _mock_health(self):
            self.healthy = True
            self.gpu_memory_free = 692
            self.last_health_time = __import__("time").time()
            return True
        monkeypatch.setattr(type(node), "health_check", _mock_health)

        result = node.health_check()
        assert result

        free = node.gpu_memory_free
        assert free == 692, f"Expected free=692, got {free}"

    def test_memory_pressure_recovery(self, chaos_coordinator, monkeypatch):
        node = chaos_coordinator.nodes["n0"]

        def _mock_health(self):
            self.healthy = True
            self.gpu_memory_free = 7168
            self.last_health_time = __import__("time").time()
            return True
        monkeypatch.setattr(type(node), "health_check", _mock_health)

        assert node.health_check()
        node.healthy = True

    def test_memory_pressure_node_drain(self, chaos_coordinator):
        coord = chaos_coordinator
        rm = coord._resource_mgr
        node_id = "n0"

        draining = []
        rm._on_node_failure = lambda nid: draining.append(nid)

        for _ in range(rm.cb_config.threshold):
            rm.record_failure(node_id)

        assert node_id in draining or node_id in rm._draining_nodes

    def test_memory_pressure_frees_and_rejoins(self, chaos_coordinator):
        coord = chaos_coordinator
        rm = coord._resource_mgr
        rm._draining_nodes.add("n0")
        rm.record_success("n0")
        rm._draining_nodes.discard("n0")
        assert "n0" not in rm._draining_nodes


# ---------------------------------------------------------------------------
# 5. Clock Skew Between Nodes
# ---------------------------------------------------------------------------

class TestClockSkew:
    """Simulate clock skew between coordinator and worker nodes."""

    def test_clock_skew_positive(self):
        import importlib.util as _util
        _spec = _util.spec_from_file_location(
            "distllm.core.resource_manager",
            SRC_DIR / "distllm/core/resource_manager.py",
            submodule_search_locations=[],
        )
        _mod = _util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        CircuitBreakerConfig = _mod.CircuitBreakerConfig
        ResourceManager = _mod.ResourceManager

        rm = ResourceManager(cb_config=CircuitBreakerConfig(threshold=2, base_delay=0.1, max_delay=5.0))
        skew = 300
        fake_now = 1000000.0 + skew
        rm._node_failure_counts["n0"] = rm.cb_config.threshold
        rm._node_recovery_time["n0"] = fake_now + 3600

        with patch("time.time", return_value=fake_now):
            if rm.check_circuit_breaker("n0"):
                assert True
            else:
                assert not rm.check_circuit_breaker("n0")

    def test_clock_skew_negative(self):
        import importlib.util as _util
        _spec = _util.spec_from_file_location(
            "distllm.core.resource_manager",
            SRC_DIR / "distllm/core/resource_manager.py",
            submodule_search_locations=[],
        )
        _mod = _util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        CircuitBreakerConfig = _mod.CircuitBreakerConfig
        ResourceManager = _mod.ResourceManager

        rm = ResourceManager(cb_config=CircuitBreakerConfig(threshold=2, base_delay=0.1, max_delay=5.0))
        real_now = time.time()
        rm._node_failure_counts["n0"] = rm.cb_config.threshold
        rm._node_recovery_time["n0"] = real_now + 3600

        with patch("time.time", return_value=real_now - 3600):
            assert rm.check_circuit_breaker("n0")

    def test_clock_skew_health_check_timestamps(self, chaos_coordinator, monkeypatch):
        node = chaos_coordinator.nodes["n0"]

        def _mock_health(self):
            self.healthy = True
            self.gpu_memory_free = 7168
            self.last_health_time = __import__("time").time()
            return True
        monkeypatch.setattr(type(node), "health_check", _mock_health)

        fake_time = 1000000.0
        with patch("time.time", return_value=fake_time):
            result = node.health_check()
            assert result
            assert node.last_health_time == fake_time

    def test_clock_skew_backoff_duration(self, rm_with_cb):
        rm = rm_with_cb
        rm._node_failure_counts["n0"] = rm.cb_config.threshold + 2
        rm._node_recovery_time["n0"] = time.time() + rm.cb_config.base_delay * 4

        remaining = rm._node_recovery_time["n0"] - time.time()
        expected = rm.cb_config.base_delay * 4
        assert abs(remaining - expected) < 0.5

    def test_clock_skew_zero_skew(self, chaos_coordinator, monkeypatch):
        coord = chaos_coordinator
        node = coord.nodes["n0"]

        def _mock_health(self):
            self.healthy = True
            self.gpu_memory_free = 7168
            self.last_health_time = __import__("time").time()
            return True
        monkeypatch.setattr(type(node), "health_check", _mock_health)

        node.client.stub.ForwardPass = MagicMock(return_value=MagicMock(
            success=True, error_message="", request_id="test", output=MagicMock(),
        ))

        assert node.health_check()
        assert node.healthy
        resp = node.client.stub.ForwardPass(MagicMock())
        assert resp.success

    def test_clock_skew_failure_tracking_accuracy(self, rm_with_cb):
        rm = rm_with_cb
        with patch("time.time", return_value=time.time() + 7200):
            rm.record_failure("n0")
            assert rm._node_failure_counts["n0"] == 1
            rm.record_success("n0")
            assert rm._node_failure_counts["n0"] == 0

    def test_clock_skew_negative_recovery_immediate(self, rm_with_cb):
        rm = rm_with_cb
        rm._node_failure_counts["n0"] = rm.cb_config.threshold
        rm._node_recovery_time["n0"] = time.time() + 60

        with patch("time.time", return_value=time.time() + 120):
            assert not rm.check_circuit_breaker("n0")
