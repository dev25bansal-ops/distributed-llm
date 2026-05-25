"""Unit tests for disaggregated serving core components."""

import time

import pytest

from distllm.core.disagg.pool import PrefillPool, DecodePool
from distllm.core.disagg.kv_cache import KVCacheStore
from distllm.core.disagg.metrics import RollingWindow, PoolMetricsCollector, DisaggMetrics
from distllm.core.disagg.scaler import (
    PrefillScaler,
    DecodeScaler,
    PoolTelemetry,
    ScalingDirection,
)
from distllm.core.disagg.types import PoolNode, PoolStatus


# ===================================================================
# Pool unit tests
# ===================================================================

class TestPrefillPool:
    @pytest.mark.asyncio
    async def test_register_node(self):
        pool = PrefillPool()
        await pool.register_node("node-1", "localhost", 50051, capacity=4)
        assert "node-1" in pool._nodes
        assert pool._nodes["node-1"].capacity == 4

    @pytest.mark.asyncio
    async def test_unregister_node(self):
        pool = PrefillPool()
        await pool.register_node("node-1", "localhost", 50051)
        await pool.unregister_node("node-1")
        assert "node-1" not in pool._nodes

    @pytest.mark.asyncio
    async def test_unregister_unknown_node(self):
        pool = PrefillPool()
        await pool.unregister_node("nonexistent")  # should not raise

    @pytest.mark.asyncio
    async def test_select_node_returns_least_loaded(self):
        pool = PrefillPool()
        await pool.register_node("node-a", "h1", 5001, capacity=4)
        await pool.register_node("node-b", "h2", 5002, capacity=4)
        pool._nodes["node-a"].current_load = 3
        pool._nodes["node-b"].current_load = 1

        selected = await pool.select_node()
        assert selected is not None
        assert selected.node_id == "node-b"  # least loaded

    @pytest.mark.asyncio
    async def test_select_node_all_full(self):
        pool = PrefillPool()
        await pool.register_node("node-a", "h1", 5001, capacity=2)
        await pool.register_node("node-b", "h2", 5002, capacity=2)
        pool._nodes["node-a"].current_load = 2
        pool._nodes["node-b"].current_load = 2

        selected = await pool.select_node()
        assert selected is None

    @pytest.mark.asyncio
    async def test_select_node_degraded_node_skipped(self):
        pool = PrefillPool()
        await pool.register_node("node-a", "h1", 5001, capacity=4)
        await pool.register_node("node-b", "h2", 5002, capacity=4)
        pool._nodes["node-b"].status = PoolStatus.DEGRADED

        selected = await pool.select_node()
        assert selected is not None
        assert selected.node_id == "node-a"

    @pytest.mark.asyncio
    async def test_select_node_empty_pool(self):
        pool = PrefillPool()
        selected = await pool.select_node()
        assert selected is None

    @pytest.mark.asyncio
    async def test_select_node_increments_load(self):
        pool = PrefillPool()
        await pool.register_node("node-1", "h1", 5001, capacity=4)
        selected = await pool.select_node()
        assert selected.current_load == 1

    @pytest.mark.asyncio
    async def test_release_node(self):
        pool = PrefillPool()
        await pool.register_node("node-1", "h1", 5001, capacity=4)
        pool._nodes["node-1"].current_load = 2
        await pool.release_node("node-1")
        assert pool._nodes["node-1"].current_load == 1

    @pytest.mark.asyncio
    async def test_release_node_unknown(self):
        pool = PrefillPool()
        await pool.release_node("nonexistent")  # should not raise

    @pytest.mark.asyncio
    async def test_release_node_at_zero(self):
        pool = PrefillPool()
        await pool.register_node("node-1", "h1", 5001, capacity=4)
        await pool.release_node("node-1")
        assert pool._nodes["node-1"].current_load == 0

    @pytest.mark.asyncio
    async def test_get_stats_empty(self):
        pool = PrefillPool()
        stats = pool.get_stats()
        assert stats["total_nodes"] == 0
        assert stats["active_nodes"] == 0
        assert stats["utilization_pct"] == 0.0

    @pytest.mark.asyncio
    async def test_get_stats_with_nodes(self):
        pool = PrefillPool()
        await pool.register_node("node-1", "h1", 5001, capacity=4)
        await pool.register_node("node-2", "h2", 5002, capacity=8)
        pool._nodes["node-1"].current_load = 2
        pool._nodes["node-2"].current_load = 4
        stats = pool.get_stats()
        assert stats["total_nodes"] == 2
        assert stats["active_nodes"] == 2
        assert stats["total_load"] == 6
        assert stats["total_capacity"] == 12
        assert stats["utilization_pct"] == 50.0

    @pytest.mark.asyncio
    async def test_multiple_selections_round_robin_by_load(self):
        pool = PrefillPool()
        await pool.register_node("node-a", "h1", 5001, capacity=4)
        await pool.register_node("node-b", "h2", 5002, capacity=4)

        s1 = await pool.select_node()
        s2 = await pool.select_node()
        # both have 0 load initially, sort is stable so both get selected
        assert s1.node_id in ("node-a", "node-b")
        assert s2.node_id in ("node-a", "node-b")


class TestDecodePool:
    @pytest.mark.asyncio
    async def test_register_node(self):
        pool = DecodePool()
        await pool.register_node("node-1", "localhost", 50052, capacity=8)
        assert "node-1" in pool._nodes

    @pytest.mark.asyncio
    async def test_unregister_node(self):
        pool = DecodePool()
        await pool.register_node("node-1", "h1", 5001)
        await pool.unregister_node("node-1")
        assert "node-1" not in pool._nodes

    @pytest.mark.asyncio
    async def test_assign_request_to_specific_node(self):
        pool = DecodePool()
        await pool.register_node("node-1", "h1", 5001, capacity=4)
        result = await pool.assign_request("req-1", node_id="node-1")
        assert result == "node-1"
        assert pool._nodes["node-1"].current_load == 1
        assert pool.get_node_for_request("req-1") == "node-1"

    @pytest.mark.asyncio
    async def test_assign_request_auto_select(self):
        pool = DecodePool()
        await pool.register_node("node-a", "h1", 5001, capacity=4)
        await pool.register_node("node-b", "h2", 5002, capacity=4)
        result = await pool.assign_request("req-1")
        assert result in ("node-a", "node-b")

    @pytest.mark.asyncio
    async def test_assign_to_full_node_falls_back(self):
        pool = DecodePool()
        await pool.register_node("node-a", "h1", 5001, capacity=1)
        await pool.register_node("node-b", "h2", 5002, capacity=1)
        # Fill node-a
        pool._nodes["node-a"].current_load = 1
        # Try to assign to node-a specifically
        result = await pool.assign_request("req-1", node_id="node-a")
        # Should fall back to an available node
        assert result == "node-b"

    @pytest.mark.asyncio
    async def test_assign_when_all_full(self):
        pool = DecodePool()
        await pool.register_node("node-a", "h1", 5001, capacity=1)
        pool._nodes["node-a"].current_load = 1
        result = await pool.assign_request("req-1")
        assert result is None

    @pytest.mark.asyncio
    async def test_release_request(self):
        pool = DecodePool()
        await pool.register_node("node-1", "h1", 5001, capacity=4)
        await pool.assign_request("req-1", node_id="node-1")
        assert pool._nodes["node-1"].current_load == 1
        await pool.release_request("req-1")
        assert pool._nodes["node-1"].current_load == 0
        assert pool.get_node_for_request("req-1") is None

    @pytest.mark.asyncio
    async def test_release_unknown_request(self):
        pool = DecodePool()
        await pool.release_request("nonexistent")  # should not raise

    @pytest.mark.asyncio
    async def test_get_node_for_request(self):
        pool = DecodePool()
        await pool.register_node("node-1", "h1", 5001, capacity=4)
        await pool.assign_request("req-1", node_id="node-1")
        assert pool.get_node_for_request("req-1") == "node-1"

    @pytest.mark.asyncio
    async def test_get_stats(self):
        pool = DecodePool()
        await pool.register_node("node-1", "h1", 5001, capacity=8)
        await pool.register_node("node-2", "h2", 5002, capacity=4)
        pool._nodes["node-1"].current_load = 2
        stats = pool.get_stats()
        assert stats["total_nodes"] == 2
        assert stats["assigned_requests"] == 0  # _request_node_map is empty
        assert stats["total_load"] == 2

    @pytest.mark.asyncio
    async def test_nodes_and_request_node_map_properties(self):
        pool = DecodePool()
        await pool.register_node("node-1", "h1", 5001, capacity=4)
        await pool.assign_request("req-1", node_id="node-1")
        assert "node-1" in pool.nodes()
        assert pool.request_node_map() == {"req-1": "node-1"}


# ===================================================================
# KVCacheStore unit tests
# ===================================================================

class TestKVCacheStore:
    def test_store_and_get(self):
        store = KVCacheStore(default_ttl_secs=300)
        store.store("req-1", {"key": "value"})
        assert store.get("req-1") == {"key": "value"}

    def test_get_nonexistent(self):
        store = KVCacheStore()
        assert store.get("missing") is None

    def test_remove(self):
        store = KVCacheStore()
        store.store("req-1", "data")
        store.remove("req-1")
        assert store.get("req-1") is None

    def test_remove_nonexistent(self):
        store = KVCacheStore()
        store.remove("missing")  # should not raise

    def test_size(self):
        store = KVCacheStore()
        assert store.size() == 0
        store.store("req-1", "data")
        assert store.size() == 1
        store.store("req-2", "data2")
        assert store.size() == 2

    def test_clear(self):
        store = KVCacheStore()
        store.store("req-1", "data")
        store.store("req-2", "data2")
        store.clear()
        assert store.size() == 0

    def test_expired_entry(self):
        store = KVCacheStore(default_ttl_secs=-1)  # already expired
        store.store("req-1", "data")
        assert store.get("req-1") is None

    def test_sweep_expired(self):
        store = KVCacheStore()
        store.store("req-fresh", "data", ttl_secs=3600)
        store.store("req-stale", "data2", ttl_secs=-1)
        swept = store.sweep_expired()
        assert swept >= 1
        assert store.get("req-fresh") is not None

    def test_custom_ttl(self):
        store = KVCacheStore(default_ttl_secs=300)
        store.store("req-1", "data", ttl_secs=0.001)
        import time
        time.sleep(0.01)
        assert store.get("req-1") is None

    def test_store_overwrite(self):
        store = KVCacheStore()
        store.store("req-1", "data1")
        store.store("req-1", "data2")
        assert store.get("req-1") == "data2"


# ===================================================================
# RollingWindow unit tests
# ===================================================================

class TestRollingWindow:
    def test_empty_window(self):
        w = RollingWindow(maxlen=10)
        assert w.avg() == 0.0
        assert w.p50() == 0.0
        assert w.p99() == 0.0
        assert w.latest() is None
        assert w.count() == 0

    def test_add_and_count(self):
        w = RollingWindow(maxlen=10)
        w.add(1.0)
        w.add(2.0)
        assert w.count() == 2

    def test_avg(self):
        w = RollingWindow(maxlen=10)
        w.add(10.0)
        w.add(20.0)
        assert w.avg() == 15.0

    def test_p50_even(self):
        w = RollingWindow(maxlen=10)
        for v in [1, 3, 2, 4]:
            w.add(float(v))
        # sorted: [1, 2, 3, 4]; mid = 4//2 = 2 => sorted_vals[2] = 3
        assert w.p50() == 3.0

    def test_p50_odd(self):
        w = RollingWindow(maxlen=10)
        for v in [1, 3, 2]:
            w.add(float(v))
        assert w.p50() == 2.0

    def test_p99(self):
        w = RollingWindow(maxlen=100)
        for v in range(1, 101):
            w.add(float(v))
        # sorted: [1..100]; idx = int(100 * 0.99) = 99 => sorted_vals[99] = 100
        assert w.p99() == 100.0

    def test_latest(self):
        w = RollingWindow(maxlen=10)
        w.add(5.0)
        w.add(10.0)
        assert w.latest() == 10.0

    def test_reset(self):
        w = RollingWindow(maxlen=10)
        w.add(1.0)
        w.add(2.0)
        w.reset()
        assert w.count() == 0
        assert w.avg() == 0.0

    def test_maxlen_bounds(self):
        w = RollingWindow(maxlen=3)
        for v in [1, 2, 3, 4, 5]:
            w.add(float(v))
        assert w.count() == 3
        assert w.latest() == 5.0


# ===================================================================
# PoolMetricsCollector unit tests
# ===================================================================

class TestPoolMetricsCollector:
    def test_initial_state(self):
        c = PoolMetricsCollector("test")
        assert c.pool_name == "test"
        assert c.request_count == 0
        assert c.error_count == 0
        assert c.total_tokens == 0

    def test_record_request_success(self):
        c = PoolMetricsCollector("test")
        c.record_request(latency_ms=10.0, tokens=100, success=True)
        assert c.request_count == 1
        assert c.error_count == 0
        assert c.total_tokens == 100

    def test_record_request_failure(self):
        c = PoolMetricsCollector("test")
        c.record_request(latency_ms=5.0, tokens=0, success=False)
        assert c.request_count == 1
        assert c.error_count == 1

    def test_error_rate(self):
        c = PoolMetricsCollector("test")
        assert c.error_rate == 0.0
        c.record_request(1.0, success=True)
        c.record_request(1.0, success=False)
        assert c.error_rate == 0.5

    def test_error_rate_no_requests(self):
        c = PoolMetricsCollector("test")
        assert c.error_rate == 0.0

    def test_throughput_rps(self):
        c = PoolMetricsCollector("test")
        # Direct access: should be 0 until at least 1 second of uptime
        c.request_count = 10
        c._start_time -= 10  # pretend 10 seconds elapsed
        assert c.throughput_rps == pytest.approx(1.0)

    def test_throughput_tps(self):
        c = PoolMetricsCollector("test")
        c.total_tokens = 100
        c._start_time -= 10
        assert c.throughput_tps == pytest.approx(10.0)

    def test_throughput_zero_before_one_sec(self):
        c = PoolMetricsCollector("test")
        c.request_count = 100
        c._start_time = time.time()  # just created
        assert c.throughput_rps == 0.0

    def test_snapshot_shape(self):
        c = PoolMetricsCollector("test")
        c.record_request(10.0, tokens=50, success=True)
        snap = c.snapshot()
        assert snap["pool"] == "test"
        assert "latency" in snap
        assert "batch_latency" in snap
        assert snap["latency"]["avg_ms"] == 10.0

    def test_reset(self):
        c = PoolMetricsCollector("test")
        c.record_request(10.0, tokens=50, success=True)
        c.reset()
        assert c.request_count == 0
        assert c.error_count == 0
        assert c.total_tokens == 0

    def test_record_batch(self):
        c = PoolMetricsCollector("test")
        c.record_batch(batch_size=4, latency_ms=20.0)
        snap = c.snapshot()
        assert snap["batch_latency"]["samples"] == 1


# ===================================================================
# DisaggMetrics unit tests
# ===================================================================

class TestDisaggMetrics:
    def test_snapshot_contains_both_pools(self):
        dm = DisaggMetrics()
        snap = dm.snapshot()
        assert "prefill" in snap
        assert "decode" in snap

    def test_summary_format(self):
        dm = DisaggMetrics()
        summary = dm.summary()
        assert "Prefill:" in summary
        assert "Decode:" in summary

    def test_reset(self):
        dm = DisaggMetrics()
        dm.prefill.record_request(10.0, success=True)
        dm.reset()
        assert dm.prefill.request_count == 0
        assert dm.decode.request_count == 0


# ===================================================================
# Scaler unit tests (PrefillScaler and DecodeScaler)
# ===================================================================

class TestPrefillScaler:
    def test_initial_state(self):
        s = PrefillScaler(min_nodes=1, max_nodes=16)
        assert s.min_nodes == 1
        assert s._current_nodes == 1

    def test_hold_within_bounds(self):
        s = PrefillScaler(min_nodes=1, max_nodes=16)
        s._last_scale_time = 0  # ensure no cooldown
        telemetry = PoolTelemetry(total_capacity=4, total_load=2)
        decision = s.evaluate(telemetry)
        assert decision.direction == ScalingDirection.HOLD

    def test_scale_up_high_utilization(self):
        s = PrefillScaler(min_nodes=1, max_nodes=16, scale_up_threshold=0.5, cooldown_seconds=0)
        s._last_scale_time = 0
        telemetry = PoolTelemetry(total_capacity=4, total_load=3)  # 75% > 50%
        decision = s.evaluate(telemetry)
        assert decision.direction == ScalingDirection.SCALE_UP
        assert decision.count > 0

    def test_scale_up_latency(self):
        s = PrefillScaler(min_nodes=1, max_nodes=16, target_latency_ms=100.0, cooldown_seconds=0)
        s._last_scale_time = 0
        telemetry = PoolTelemetry(total_capacity=10, total_load=5, avg_latency_ms=200.0)
        decision = s.evaluate(telemetry)
        assert decision.direction == ScalingDirection.SCALE_UP

    def test_scale_down_low_utilization(self):
        s = PrefillScaler(min_nodes=1, max_nodes=16, scale_down_threshold=0.3, cooldown_seconds=0)
        s._current_nodes = 4
        s._last_scale_time = 0
        telemetry = PoolTelemetry(total_capacity=10, total_load=1)  # 10% < 30%
        decision = s.evaluate(telemetry)
        assert decision.direction == ScalingDirection.SCALE_DOWN
        assert decision.count == 1

    def test_scale_down_at_min_nodes(self):
        s = PrefillScaler(min_nodes=2, max_nodes=16, scale_down_threshold=0.3, cooldown_seconds=0)
        s._current_nodes = 2
        s._last_scale_time = 0
        telemetry = PoolTelemetry(total_capacity=10, total_load=1)
        decision = s.evaluate(telemetry)
        assert decision.direction == ScalingDirection.HOLD

    def test_cooldown_honored(self):
        s = PrefillScaler(min_nodes=1, max_nodes=16, cooldown_seconds=3600)
        s._last_scale_time = time.time()
        telemetry = PoolTelemetry(total_capacity=4, total_load=4)  # 100% utilization
        decision = s.evaluate(telemetry)
        assert decision.direction == ScalingDirection.HOLD
        assert "cooldown" in decision.reason

    def test_no_divide_by_zero(self):
        s = PrefillScaler(min_nodes=1, max_nodes=16, cooldown_seconds=0)
        s._last_scale_time = 0
        telemetry = PoolTelemetry(total_capacity=0, total_load=0)
        decision = s.evaluate(telemetry)
        assert decision.direction == ScalingDirection.HOLD

    def test_scale_up_caps_at_max_nodes(self):
        s = PrefillScaler(min_nodes=1, max_nodes=2, scale_up_threshold=0.5, cooldown_seconds=0)
        s._current_nodes = 2
        s._last_scale_time = 0
        telemetry = PoolTelemetry(total_capacity=4, total_load=4)
        decision = s.evaluate(telemetry)
        assert decision.direction == ScalingDirection.HOLD

    def test_scale_up_reason(self):
        s = PrefillScaler(min_nodes=1, max_nodes=16, scale_up_threshold=0.5, cooldown_seconds=0)
        s._last_scale_time = 0
        telemetry = PoolTelemetry(total_capacity=4, total_load=3)
        decision = s.evaluate(telemetry)
        assert "Utilization" in decision.reason


class TestDecodeScaler:
    def test_initial_state(self):
        s = DecodeScaler(min_nodes=1, max_nodes=32)
        assert s.min_nodes == 1

    def test_hold_within_bounds(self):
        s = DecodeScaler(min_nodes=1, max_nodes=32, cooldown_seconds=0)
        s._last_scale_time = 0
        telemetry = PoolTelemetry(total_capacity=10, total_load=3)
        decision = s.evaluate(telemetry)
        assert decision.direction == ScalingDirection.HOLD

    def test_scale_up_high_utilization(self):
        s = DecodeScaler(min_nodes=1, max_nodes=32, scale_up_threshold=0.5, cooldown_seconds=0)
        s._last_scale_time = 0
        telemetry = PoolTelemetry(total_capacity=4, total_load=3)  # 75% > 50%
        decision = s.evaluate(telemetry)
        assert decision.direction == ScalingDirection.SCALE_UP
        assert decision.count > 0

    def test_scale_down_low_utilization(self):
        s = DecodeScaler(min_nodes=1, max_nodes=32, scale_down_threshold=0.3, cooldown_seconds=0)
        s._current_nodes = 3
        s._last_scale_time = 0
        telemetry = PoolTelemetry(total_capacity=10, total_load=1)  # 10% < 30%
        decision = s.evaluate(telemetry)
        assert decision.direction == ScalingDirection.SCALE_DOWN

    def test_cooldown_honored(self):
        s = DecodeScaler(min_nodes=1, max_nodes=32, cooldown_seconds=3600)
        s._last_scale_time = time.time()
        telemetry = PoolTelemetry(total_capacity=4, total_load=4)
        decision = s.evaluate(telemetry)
        assert decision.direction == ScalingDirection.HOLD

    def test_no_divide_by_zero(self):
        s = DecodeScaler(min_nodes=1, max_nodes=32, cooldown_seconds=0)
        s._last_scale_time = 0
        telemetry = PoolTelemetry(total_capacity=0, total_load=0)
        decision = s.evaluate(telemetry)
        assert decision.direction == ScalingDirection.HOLD


# ===================================================================
# PoolNode / Types
# ===================================================================

class TestPoolNode:
    def test_defaults(self):
        n = PoolNode(node_id="n1", host="h1", port=5001)
        assert n.capacity == 0
        assert n.current_load == 0
        assert n.status == PoolStatus.ACTIVE
        assert n.metrics == {}

    def test_custom_values(self):
        n = PoolNode(node_id="n1", host="h1", port=5001, capacity=8, status=PoolStatus.DEGRADED)
        assert n.capacity == 8
        assert n.status == PoolStatus.DEGRADED
