"""F-014 regression: IntelligentAutoscaler must be fed and actuated.

Before the fix, ``Coordinator.start()`` instantiated the
``IntelligentAutoscaler``, called ``record_metrics`` exactly once with a
``ScalingMetrics`` that left ``gpu_utilization`` at its 0.0 default, and never
called ``evaluate()`` again — no periodic loop, no actuation path for
``ScalingDecision.target_nodes`` (same dead wiring in coordinator_subsystem).
The autoscaler could neither observe load nor scale anything.

These tests pin the corrected behaviour:
- metrics are collected from live sources (batch scheduler + GPU telemetry
  with a batch-occupancy fallback) so gpu_utilization is populated;
- a background loop feeds the autoscaler repeatedly and can be stopped;
- should-scale decisions reach the registered provisioning callback.
"""

from __future__ import annotations

import time

import pytest

from distllm.core.coordinator import Coordinator
from distllm.core.intelligent_autoscaler import (
    IntelligentAutoscaler,
    ScalingMetrics,
)


def _make_coord() -> Coordinator:
    return Coordinator(
        model_name="test-model",
        dtype="float32",
        max_batch_size=4,
        max_tokens_per_batch=512,
    )


class _StubScheduler:
    """Batch scheduler stand-in returning fixed stats."""

    def __init__(self, stats: dict):
        self._stats = stats

    def stats(self) -> dict:
        return dict(self._stats)


class _FakeMonitor:
    """SystemMonitor stand-in."""

    def __init__(self, gpu: dict | None = None, error: bool = False):
        self._gpu = gpu
        self._error = error
        self.calls = 0

    def collect(self) -> dict:
        self.calls += 1
        if self._error:
            raise RuntimeError("monitor boom")
        return {"gpu": self._gpu} if self._gpu is not None else {}


HIGH_LOAD_STATS = {
    "active_requests": 0,
    "pending_requests": 100,  # queue_ratio >> 5 -> reactive scale-up
    "max_batch_size": 4,
}
IDLE_STATS = {
    "active_requests": 0,
    "pending_requests": 0,
    "max_batch_size": 4,
}


class TestCollectScalingMetrics:
    def test_real_gpu_utilization_used_when_monitor_reports_it(self):
        coord = _make_coord()
        coord._batch_scheduler = _StubScheduler(IDLE_STATS)
        coord._system_monitor = _FakeMonitor(gpu={"utilization_gpu": 87.0})

        m = coord._collect_scaling_metrics()

        assert isinstance(m, ScalingMetrics)
        assert m.gpu_utilization == pytest.approx(87.0)
        assert m.active_requests == 0
        assert m.pending_requests == 0
        assert m.queue_depth == 0
        assert m.current_nodes == len(coord.nodes)

    def test_utilization_falls_back_to_batch_occupancy_without_telemetry(self):
        """Without GPU telemetry utilization must reflect load, not stay 0.0."""
        coord = _make_coord()
        coord._batch_scheduler = _StubScheduler(
            {"active_requests": 1, "pending_requests": 1, "max_batch_size": 4}
        )
        coord._system_monitor = False  # sentinel: telemetry unavailable

        m = coord._collect_scaling_metrics()

        assert m.gpu_utilization == pytest.approx((1 + 1) / 4 * 100)

    def test_utilization_fallback_clamped_to_100(self):
        coord = _make_coord()
        coord._batch_scheduler = _StubScheduler(
            {"active_requests": 90, "pending_requests": 90, "max_batch_size": 4}
        )
        coord._system_monitor = False

        m = coord._collect_scaling_metrics()

        assert m.gpu_utilization == pytest.approx(100.0)

    def test_failing_monitor_falls_back_to_batch_occupancy(self):
        coord = _make_coord()
        coord._batch_scheduler = _StubScheduler(HIGH_LOAD_STATS)
        coord._system_monitor = _FakeMonitor(error=True)

        m = coord._collect_scaling_metrics()

        # (0 + 100) / 4 * 100 -> clamped to 100
        assert m.gpu_utilization == pytest.approx(100.0)


class TestTickActuates:
    def test_should_scale_decision_reaches_provisioning_callback(self):
        coord = _make_coord()
        coord._autoscaler = IntelligentAutoscaler(
            min_nodes=1, max_nodes=20, cooldown_seconds=0.0
        )
        coord._batch_scheduler = _StubScheduler(HIGH_LOAD_STATS)
        coord._system_monitor = False
        seen: list = []
        coord.set_scale_callback(seen.append)

        coord._autoscaler_tick()

        # The autoscaler was actually fed...
        assert len(coord._autoscaler._history) == 1
        # ...and produced an actuated scale-up decision.
        assert len(seen) == 1
        decision = seen[0]
        assert decision.should_scale is True
        assert decision.reason == "scale_up"
        assert decision.target_nodes > len(coord.nodes)

    def test_idle_cluster_at_min_capacity_does_not_invoke_callback(self):
        """A 1-node idle cluster is already at target -> 'optimal', no actuation."""
        from unittest.mock import patch

        coord = _make_coord()
        coord._autoscaler = IntelligentAutoscaler(min_nodes=1, cooldown_seconds=0.0)
        seen: list = []
        coord.set_scale_callback(seen.append)

        idle_one_node = ScalingMetrics(
            active_requests=0,
            pending_requests=0,
            gpu_utilization=5.0,
            queue_depth=0,
            current_nodes=1,
        )
        with patch.object(
            coord, "_collect_scaling_metrics", return_value=idle_one_node
        ):
            coord._autoscaler_tick()

        assert seen == []
        # Metrics were still fed into the autoscaler history.
        assert len(coord._autoscaler._history) == 1

    def test_no_scale_callback_still_feeds_and_does_not_raise(self):
        """Default deployment: decisions are logged, metrics still recorded."""
        coord = _make_coord()
        coord._autoscaler = IntelligentAutoscaler(cooldown_seconds=0.0)
        coord._batch_scheduler = _StubScheduler(HIGH_LOAD_STATS)
        coord._system_monitor = False

        coord._autoscaler_tick()  # must not raise without a callback

        assert len(coord._autoscaler._history) == 1

    def test_raising_callback_is_contained(self):
        coord = _make_coord()
        coord._autoscaler = IntelligentAutoscaler(cooldown_seconds=0.0)
        coord._batch_scheduler = _StubScheduler(HIGH_LOAD_STATS)
        coord._system_monitor = False

        def _boom(decision):
            raise RuntimeError("provisioner down")

        coord.set_scale_callback(_boom)

        coord._autoscaler_tick()  # exception must not escape the tick

        assert len(coord._autoscaler._history) == 1

    def test_tick_is_noop_without_autoscaler(self):
        coord = _make_coord()
        coord._autoscaler = None
        coord._system_monitor = False

        coord._autoscaler_tick()  # must not raise


class TestLoopLifecycle:
    def test_loop_feeds_metrics_periodically_then_stops(self):
        coord = _make_coord()
        coord._autoscaler = IntelligentAutoscaler(cooldown_seconds=0.0)
        coord._batch_scheduler = _StubScheduler(IDLE_STATS)
        coord._system_monitor = False

        coord._start_autoscaler_loop(interval_s=0.05)
        try:
            thread = coord._autoscaler_thread
            assert thread is not None and thread.is_alive()
            assert thread.daemon is True
            assert thread.name == "autoscaler-loop"
            time.sleep(0.35)
            # Before the fix the history never grew past the single startup
            # record_metrics entry; the loop must keep feeding it.
            assert len(coord._autoscaler._history) >= 2
        finally:
            coord._stop_autoscaler_loop()

        assert coord._autoscaler_thread is None
        count_after_stop = len(coord._autoscaler._history)
        time.sleep(0.15)
        assert len(coord._autoscaler._history) == count_after_stop

    def test_double_start_keeps_single_thread(self):
        coord = _make_coord()
        coord._autoscaler = IntelligentAutoscaler(cooldown_seconds=0.0)
        coord._system_monitor = False
        try:
            coord._start_autoscaler_loop(interval_s=0.05)
            first = coord._autoscaler_thread
            coord._start_autoscaler_loop(interval_s=0.05)
            assert coord._autoscaler_thread is first
        finally:
            coord._stop_autoscaler_loop()

    def test_start_is_noop_without_autoscaler(self):
        coord = _make_coord()  # autoscaler subsystem not started -> None
        coord._start_autoscaler_loop(interval_s=0.05)
        assert coord._autoscaler_thread is None

    def test_stop_safe_when_never_started(self):
        coord = _make_coord()
        coord._stop_autoscaler_loop()  # must not raise
        assert coord._autoscale_stop.is_set()


class TestInitState:
    def test_autoscaler_wiring_attributes_exist(self):
        coord = _make_coord()
        try:
            assert coord._autoscaler is None
            assert coord._autoscaler_thread is None
            assert not coord._autoscale_stop.is_set()
            assert coord._scale_callback is None
        finally:
            if hasattr(coord, "close"):
                coord.close()
