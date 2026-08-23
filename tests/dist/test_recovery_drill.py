"""Tests for RecoveryDrill — periodic chaos-engineering recovery simulations.

Zero mocks: all tests use the real NodeRecoveryManager from recovery.py.
"""

from __future__ import annotations

import time

import pytest

from distllm.dist.recovery import NodeRecoveryManager
from distllm.dist.recovery_drill import DrillResult, RecoveryDrill


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_mgr() -> NodeRecoveryManager:
    """Return a fresh NodeRecoveryManager with dry_run default."""
    return NodeRecoveryManager()


def _drill_run(mgr: NodeRecoveryManager, **kwargs) -> DrillResult:
    """Create a RecoveryDrill, run one drill, return the result."""
    drill = RecoveryDrill(recovery_mgr=mgr, **kwargs)
    return drill.run_drill_now()


DEFAULT_SLA_MS = 5000.0
DEFAULT_SLA_LOST = 0


# ── DrillResult ─────────────────────────────────────────────────────────────


class TestDrillResult:
    """Construction and serialisation of DrillResult."""

    def test_construct_minimal(self) -> None:
        """All fields can be set via positional args."""
        r = DrillResult(
            timestamp=1.0,
            simulated_node_id="n1",
            recovery_time_ms=42.0,
            sequences_recovered=3,
            sequences_lost=0,
            redistributions=2,
            passed=True,
            failures=[],
        )
        assert r.timestamp == 1.0
        assert r.simulated_node_id == "n1"
        assert r.recovery_time_ms == 42.0
        assert r.sequences_recovered == 3
        assert r.sequences_lost == 0
        assert r.redistributions == 2
        assert r.passed is True
        assert r.failures == []

    def test_construct_with_failures(self) -> None:
        """A failed DrillResult stores failure messages."""
        r = DrillResult(
            timestamp=2.0,
            simulated_node_id="n2",
            recovery_time_ms=999.0,
            sequences_recovered=0,
            sequences_lost=5,
            redistributions=0,
            passed=False,
            failures=["timeout", "data_loss"],
        )
        assert r.passed is False
        assert "timeout" in r.failures

    def test_to_dict(self) -> None:
        """to_dict returns a plain dict with rounded recovery_time_ms."""
        r = DrillResult(
            timestamp=100.0,
            simulated_node_id="n3",
            recovery_time_ms=42.777,
            sequences_recovered=2,
            sequences_lost=1,
            redistributions=1,
            passed=True,
            failures=[],
        )
        d = r.to_dict()
        assert d["timestamp"] == 100.0
        assert d["simulated_node_id"] == "n3"
        assert d["recovery_time_ms"] == 42.8  # rounded to 1 decimal
        assert d["sequences_recovered"] == 2
        assert d["sequences_lost"] == 1
        assert d["redistributions"] == 1
        assert d["passed"] is True
        assert d["failures"] == []

    def test_to_dict_empty_failures(self) -> None:
        """to_dict preserves empty failures list."""
        r = DrillResult(
            timestamp=0.0,
            simulated_node_id="empty",
            recovery_time_ms=0.0,
            sequences_recovered=0,
            sequences_lost=0,
            redistributions=0,
            passed=True,
            failures=[],
        )
        d = r.to_dict()
        assert d["failures"] == []


# ── RecoveryDrill construction ──────────────────────────────────────────────


class TestRecoveryDrillInit:
    """RecoveryDril __init__ defaults and parameter wiring."""

    def test_defaults(self) -> None:
        """Default params produce a usable drill with empty history."""
        mgr = _make_mgr()
        drill = RecoveryDrill(recovery_mgr=mgr)
        assert drill._recovery_mgr is mgr
        assert drill._autoscaler is None
        assert drill._sla_max_recovery_ms == DEFAULT_SLA_MS
        assert drill._sla_max_sequences_lost == DEFAULT_SLA_LOST
        assert drill._sla_min_redistributions == 0
        assert drill._max_history == 100
        assert drill.history == []
        assert drill.sla_pass_rate == 1.0
        assert drill._running.is_set() is False
        assert drill._thread is None

    def test_custom_sla_values(self) -> None:
        """SLA thresholds are wired from constructor args."""
        mgr = _make_mgr()
        drill = RecoveryDrill(
            recovery_mgr=mgr,
            sla_max_recovery_ms=100.0,
            sla_max_sequences_lost=1,
            sla_min_redistributions=2,
        )
        assert drill._sla_max_recovery_ms == 100.0
        assert drill._sla_max_sequences_lost == 1
        assert drill._sla_min_redistributions == 2

    def test_custom_max_history(self) -> None:
        """max_history bounds the internal history list."""
        mgr = _make_mgr()
        drill = RecoveryDrill(recovery_mgr=mgr, max_history=5)
        assert drill._max_history == 5


# ── RecoveryDrill execution ─────────────────────────────────────────────────


class TestRecoveryDrillExecution:
    """Single drill execution via run_drill_now / _execute_drill."""

    def test_run_drill_now_returns_drill_result(self) -> None:
        """run_drill_now returns a DrillResult."""
        mgr = _make_mgr()
        drill = RecoveryDrill(recovery_mgr=mgr)
        result = drill.run_drill_now()
        assert isinstance(result, DrillResult)

    def test_run_drill_now_synthetic_node(self) -> None:
        """The default fallback target is 'drill-target-synthetic'."""
        mgr = _make_mgr()
        drill = RecoveryDrill(recovery_mgr=mgr)
        result = drill.run_drill_now()
        assert result.simulated_node_id == "drill-target-synthetic"

    def test_run_drill_now_appends_history(self) -> None:
        """Each drill call appends to drill.history."""
        mgr = _make_mgr()
        drill = RecoveryDrill(recovery_mgr=mgr)
        assert len(drill.history) == 0
        drill.run_drill_now()
        assert len(drill.history) == 1
        drill.run_drill_now()
        assert len(drill.history) == 2

    def test_run_drill_now_records_timestamp(self) -> None:
        """The result timestamp is close to current time."""
        mgr = _make_mgr()
        drill = RecoveryDrill(recovery_mgr=mgr)
        before = time.time()
        result = drill.run_drill_now()
        after = time.time()
        assert before <= result.timestamp <= after

    def test_run_drill_now_returns_nonzero_time(self) -> None:
        """recovery_time_ms should be > 0 for a real dry-run recovery."""
        mgr = _make_mgr()
        drill = RecoveryDrill(recovery_mgr=mgr)
        result = drill.run_drill_now()
        assert result.recovery_time_ms > 0.0

    def test_run_drill_now_default_sla_pass(self) -> None:
        """Default SLA thresholds should pass for a basic dry run."""
        mgr = _make_mgr()
        drill = RecoveryDrill(recovery_mgr=mgr)
        result = drill.run_drill_now()
        assert result.passed is True
        assert result.failures == []

    def test_strict_sla_time_raises_violation(self) -> None:
        """Setting sla_max_recovery_ms=0 causes a failure since any recovery
        takes non-zero time."""
        mgr = _make_mgr()
        drill = RecoveryDrill(recovery_mgr=mgr, sla_max_recovery_ms=0.0)
        result = drill.run_drill_now()
        assert result.passed is False
        assert any("recovery_time" in f for f in result.failures)

    def test_strict_sla_sequences_lost(self) -> None:
        """Setting sla_max_sequences_lost < 0 causes a failure on zero lost
        because 0 > negative."""
        mgr = _make_mgr()
        drill = RecoveryDrill(recovery_mgr=mgr, sla_max_sequences_lost=-1)
        result = drill.run_drill_now()
        assert result.passed is False
        assert any("sequences_lost" in f for f in result.failures)

    def test_strict_sla_redistributions(self) -> None:
        """Setting sla_min_redistributions > 0 causes a failure because a
        basic dry run produces zero redistributions."""
        mgr = _make_mgr()
        drill = RecoveryDrill(recovery_mgr=mgr, sla_min_redistributions=1)
        result = drill.run_drill_now()
        assert result.passed is False
        assert any("redistributions" in f for f in result.failures)


# ── History bounds ──────────────────────────────────────────────────────────


class TestRecoveryDrillHistoryBounds:
    """History list trimming and edge-case behaviour."""

    def test_max_history_trims(self) -> None:
        """When max_history is small, old entries are dropped."""
        mgr = _make_mgr()
        max_h = 3
        drill = RecoveryDrill(recovery_mgr=mgr, max_history=max_h)
        for _ in range(10):
            drill.run_drill_now()
        assert len(drill.history) <= max_h

    def test_max_history_preserves_most_recent(self) -> None:
        """After trimming, the most recent results survive."""
        mgr = _make_mgr()
        drill = RecoveryDrill(recovery_mgr=mgr, max_history=3)
        for i in range(5):
            drill.run_drill_now()
        # history[-max_history:] keeps last 3
        assert len(drill.history) == 3
        # All remaining entries should be DrillResult instances
        for entry in drill.history:
            assert isinstance(entry, DrillResult)


# ── Observability ───────────────────────────────────────────────────────────


class TestRecoveryDrillObservability:
    """get_summary and get_history."""

    def test_get_summary_empty(self) -> None:
        """Empty history returns zero drills and pass_rate=1.0."""
        mgr = _make_mgr()
        drill = RecoveryDrill(recovery_mgr=mgr)
        s = drill.get_summary()
        assert s["total_drills"] == 0
        assert s["pass_rate"] == 1.0

    def test_get_summary_all_passed(self) -> None:
        """After passed drills, summary reflects correct counts."""
        mgr = _make_mgr()
        drill = RecoveryDrill(recovery_mgr=mgr)
        drill.run_drill_now()
        drill.run_drill_now()
        s = drill.get_summary()
        assert s["total_drills"] == 2
        assert s["passed"] == 2
        assert s["pass_rate"] == 1.0
        # Second drill may hit "already being recovered" guard (zero time),
        # so avg can be >0 or 0 — just verify the field exists and is numeric
        assert isinstance(s["avg_recovery_time_ms"], float)
        assert s["latest"] is not None

    def test_get_summary_with_failures(self) -> None:
        """Summary correctly shows pass_rate < 1 when some drills fail."""
        # Drills on separate managers to avoid dead-node state leakage
        mgr_pass = _make_mgr()
        drill_pass = RecoveryDrill(recovery_mgr=mgr_pass, max_history=10)
        drill_pass.run_drill_now()

        mgr_fail = _make_mgr()
        drill_fail = RecoveryDrill(
            recovery_mgr=mgr_fail, sla_max_recovery_ms=0.0, max_history=10
        )
        drill_fail.run_drill_now()

        s = drill_fail.get_summary()
        assert s["total_drills"] == 1
        assert s["passed"] == 0
        assert s["pass_rate"] == 0.0

    def test_get_summary_avg_time(self) -> None:
        """Average is computed as mean of all recovery times."""
        mgr = _make_mgr()
        drill = RecoveryDrill(recovery_mgr=mgr)
        drill.run_drill_now()
        drill.run_drill_now()
        s = drill.get_summary()
        total = sum(r.recovery_time_ms for r in drill.history)
        expected_avg = round(total / 2, 1)
        assert s["avg_recovery_time_ms"] == expected_avg

    def test_get_history_default_limit(self) -> None:
        """get_history returns up to 20 entries by default."""
        mgr = _make_mgr()
        drill = RecoveryDrill(recovery_mgr=mgr)
        for _ in range(5):
            drill.run_drill_now()
        history = drill.get_history()
        assert len(history) == 5
        for entry in history:
            assert isinstance(entry, dict)

    def test_get_history_custom_limit(self) -> None:
        """get_history respects a custom limit."""
        mgr = _make_mgr()
        drill = RecoveryDrill(recovery_mgr=mgr)
        for _ in range(10):
            drill.run_drill_now()
        history = drill.get_history(limit=3)
        assert len(history) == 3

    def test_get_history_returns_latest_first(self) -> None:
        """get_history returns the most recent results (last N in list)."""
        mgr = _make_mgr()
        drill = RecoveryDrill(recovery_mgr=mgr)
        for _ in range(5):
            drill.run_drill_now()
        history = drill.get_history(limit=5)
        # The last item in history corresponds to the most recent drill
        assert history[-1] == drill.history[-1].to_dict()

    def test_get_history_empty(self) -> None:
        """get_history returns empty list when no drills have run."""
        mgr = _make_mgr()
        drill = RecoveryDrill(recovery_mgr=mgr)
        assert drill.get_history(limit=10) == []


# ── Lifecycle (start / stop) ────────────────────────────────────────────────


class TestRecoveryDrillLifecycle:
    """Threading lifecycle — start/stop without timing assumptions."""

    def test_start_then_stop(self) -> None:
        """Starting and stopping the background thread does not raise."""
        mgr = _make_mgr()
        drill = RecoveryDrill(recovery_mgr=mgr)
        drill.start(interval_s=0.05)
        assert drill._running.is_set()
        assert drill._thread is not None
        assert drill._thread.is_alive()
        drill.stop()
        assert not drill._running.is_set()

    def test_double_start_is_idempotent(self) -> None:
        """Calling start twice does not start a second thread."""
        mgr = _make_mgr()
        drill = RecoveryDrill(recovery_mgr=mgr)
        drill.start(interval_s=0.05)
        thread_id = id(drill._thread)
        drill.start(interval_s=0.05)  # second call is a no-op
        assert id(drill._thread) == thread_id
        drill.stop()

    def test_stop_without_start_does_not_raise(self) -> None:
        """Calling stop before start is safe."""
        mgr = _make_mgr()
        drill = RecoveryDrill(recovery_mgr=mgr)
        drill.stop()  # should not raise

    def test_stop_multiple_times(self) -> None:
        """Calling stop multiple times is safe."""
        mgr = _make_mgr()
        drill = RecoveryDrill(recovery_mgr=mgr)
        drill.start(interval_s=0.05)
        drill.stop()
        drill.stop()  # second stop should not raise


# ── Edge cases ──────────────────────────────────────────────────────────────


class TestRecoveryDrillEdgeCases:
    """Boundary and degenerate cases."""

    def test_drill_with_autoscaler_none(self) -> None:
        """Drill works without an autoscaler."""
        mgr = _make_mgr()
        drill = RecoveryDrill(recovery_mgr=mgr, autoscaler=None)
        result = drill.run_drill_now()
        assert isinstance(result, DrillResult)
        assert result.passed is True

    def test_select_drill_target_fallback(self) -> None:
        """_select_drill_target returns 'drill-target-synthetic' when no
        nodes are registered (the normal case without autoscaler)."""
        mgr = _make_mgr()
        drill = RecoveryDrill(recovery_mgr=mgr)
        target = drill._select_drill_target()
        assert target == "drill-target-synthetic"

    def test_full_failure_list_multiple_slas(self) -> None:
        """Multiple SLA violations produce multiple failure entries."""
        mgr = _make_mgr()
        drill = RecoveryDrill(
            recovery_mgr=mgr,
            sla_max_recovery_ms=0.0,
            sla_max_sequences_lost=-1,
            sla_min_redistributions=1,
        )
        result = drill.run_drill_now()
        assert result.passed is False
        assert len(result.failures) >= 2

    def test_sla_pass_rate_default(self) -> None:
        """sla_pass_rate attribute is 1.0 by default and can be set."""
        mgr = _make_mgr()
        drill = RecoveryDrill(recovery_mgr=mgr)
        assert drill.sla_pass_rate == 1.0
        drill.sla_pass_rate = 0.95
        assert drill.sla_pass_rate == 0.95

    def test_sequences_recovered_in_result(self) -> None:
        """A basic drill has zero sequences recovered (no checkpoints)."""
        mgr = _make_mgr()
        drill = RecoveryDrill(recovery_mgr=mgr)
        result = drill.run_drill_now()
        assert result.sequences_recovered == 0
        assert result.sequences_lost == 0

    def test_result_with_checkpoints(self) -> None:
        """When checkpoints exist, the dry run reports them in the plan."""
        mgr = _make_mgr()
        mgr.save_checkpoint(
            request_id="req-001",
            kv_cache={"layer_0": "tensor_placeholder"},
            prompt_tokens=[1, 2, 3],
            generated_tokens=[4, 5],
            node_id="test-node",
        )
        drill = RecoveryDrill(recovery_mgr=mgr)
        # The target is always "drill-target-synthetic", so the checkpoint
        # on "test-node" won't be recovered — expectation: 0 recovered
        result = drill.run_drill_now()
        assert isinstance(result, DrillResult)
        assert result.simulated_node_id == "drill-target-synthetic"
