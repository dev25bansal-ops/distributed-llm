"""Regression test for HIGH fix H12: autoscaler decision race.

``IntelligentAutoscaler.evaluate`` reads *and* mutates shared mutable state
(``_last_scale_time`` cooldown timestamp, the ``_pending_dir`` / ``_pending_count``
flap-guard counters, the committed scale action).  Before the fix only the
history append inside ``record_metrics`` took ``self._lock``; the whole
decision body ran *outside* the lock.  Under concurrency two threads could
both observe "no cooldown active" (and the same pending-direction window) and
both emit a scale action before either updated ``_last_scale_time`` — producing
scale thrash / lost updates.

The fix moves the entire read-compute-mutate decision body under ``self._lock``
so the cooldown read, the flap-guard bookkeeping and the ``_last_scale_time``
write are atomic with respect to each other.

This test drives ``evaluate`` from many threads simultaneously, in a scenario
where a cooldown must block every scale action after the first one.  To make
the previously-racy (unlocked) read-modify-write window *deterministic* and
reliably exposable, we inject a slow ``_predict_load``.  In the buggy code that
helper runs *outside* the lock, so all threads pile into the decision body at
once, each sees "no cooldown", and many emit a scale in a single round.  In the
fixed code the helper runs *inside* the lock, so the first thread to commit a
scale sets the cooldown and every other thread is blocked.

We also inject a frozen fake clock so the cooldown math is fully deterministic
(every ``time.time()`` call sees the same value).
"""

from __future__ import annotations

import sys
import threading
import time as _real_time
from pathlib import Path

import pytest

_REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

import distllm.core.intelligent_autoscaler as autoscaler_mod  # noqa: E402
from distllm.core.intelligent_autoscaler import (  # noqa: E402
    IntelligentAutoscaler,
    ScalingMetrics,
)


class _FakeClock:
    """A controllable, frozen wall clock standing in for ``time``.

    The module calls ``time.time()``; we replace the module's ``time`` binding
    with this object whose ``.time()`` returns a frozen value.  The value is
    held constant across a round so the decision is deterministic.
    """

    def __init__(self, now: float = 100.0):
        self.now = now

    def time(self) -> float:
        return self.now


@pytest.fixture()
def fake_clock(monkeypatch):
    clock = _FakeClock(now=100.0)
    monkeypatch.setattr(autoscaler_mod, "time", clock)
    return clock


def _make_scaler() -> IntelligentAutoscaler:
    # stable_cycles=1 so a single out-of-band evaluation emits a scale;
    # hysteresis_band=1 so a genuine +1 step is not absorbed by the dead-band.
    return IntelligentAutoscaler(
        min_nodes=1,
        max_nodes=20,
        cooldown_seconds=60.0,
        hysteresis_band=1,
        stable_cycles=1,
    )


def _scale_metrics() -> ScalingMetrics:
    # High GPU utilisation -> reactive target = current + 1, which (with band 1)
    # yields a scale-up decision when not in cooldown.
    return ScalingMetrics(
        active_requests=10,
        pending_requests=0,
        avg_latency_ms=1.0,
        gpu_utilization=95.0,
        queue_depth=0,
        current_nodes=5,
    )


def _install_slow_predict(scaler, sleep_s=0.02):
    """Replace ``_predict_load`` with one that sleeps.

    In the buggy (unlocked) code this runs inside the decision body *outside*
    the lock, so all threads spend real wall-clock time there simultaneously
    and the cooldown-read / ``_last_scale_time``-write race is exposed.  In the
    fixed code it runs inside ``with self._lock:`` so threads are serialised
    and only the first commits a scale.
    """

    orig = scaler._predict_load

    def _slow(*a, **k):
        _real_time.sleep(sleep_s)
        return orig(*a, **k)

    scaler._predict_load = _slow


def _run_concurrent_round(scaler, metrics, n_threads=16):
    """Launch ``n_threads`` threads that each call ``evaluate`` concurrently.

    Returns the number of should_scale==True decisions produced this round.
    """
    results = []
    results_lock = threading.Lock()
    # Barrier so all threads start the decision body at (roughly) the same time,
    # maximising overlap inside the (previously) unlocked decision body.
    barrier = threading.Barrier(n_threads)

    def worker():
        barrier.wait()
        decision = scaler.evaluate(metrics)
        if decision.should_scale:
            with results_lock:
                results.append(decision)

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return len(results)


def test_concurrent_decisions_emit_exactly_one_scale_per_round(fake_clock):
    """With a cooldown in force, concurrent evaluates must yield exactly 1 scale.

    On the buggy code ``_predict_load`` runs unlocked, so all threads are in the
    decision body at once, all observe "no cooldown" and many emit a scale in a
    single round.  After the fix the decision body is atomic, so exactly one
    scale is emitted and every other thread in the same round is blocked by the
    cooldown.  We run many rounds to defeat residual timing luck.
    """
    scaler = _make_scaler()
    _install_slow_predict(scaler, sleep_s=0.02)
    metrics = _scale_metrics()

    rounds = 20
    max_scales_in_any_round = 0
    for _ in range(rounds):
        # Clean round: last scale happened long ago (0); "now" is far in the
        # future (100) so the round begins with no active cooldown.
        scaler._last_scale_time = 0.0
        fake_clock.now = 100.0
        n_scales = _run_concurrent_round(scaler, metrics, n_threads=16)
        max_scales_in_any_round = max(max_scales_in_any_round, n_scales)

    assert max_scales_in_any_round == 1, (
        f"concurrent evaluates produced up to {max_scales_in_any_round} "
        f"scale actions in a single round; expected exactly 1 (cooldown must "
        f"block the rest) — decision body was not atomic under the lock"
    )


def test_shared_state_consistent_after_concurrent_decisions(fake_clock):
    """After concurrent decisions the cooldown/flap state matches the single-
    threaded expectation: one scale committed, cooldown active, pending reset.

    Sequentially (single-threaded) a scale round would: emit 1 scale, set
    ``_last_scale_time = now``, and reset ``_pending_dir``/``_pending_count``
    to 0.  This asserts the concurrent execution leaves the same state and does
    not lose the cooldown timestamp nor corrupt the flap-guard counters.
    """
    scaler = _make_scaler()
    _install_slow_predict(scaler, sleep_s=0.02)
    metrics = _scale_metrics()

    scaler._last_scale_time = 0.0
    fake_clock.now = 100.0
    n_scales = _run_concurrent_round(scaler, metrics, n_threads=16)

    assert n_scales == 1
    # Cooldown must be live: the latest scale timestamp equals our frozen clock.
    assert scaler._last_scale_time == 100.0
    # Flap-guard counters must be reset after committing a scale.
    assert scaler._pending_dir == 0
    assert scaler._pending_count == 0
    # And a follow-up decision must be blocked by the cooldown, not re-scale.
    followup = scaler.evaluate(metrics)
    assert followup.should_scale is False
    assert followup.reason == "cooldown"


def test_sequential_behavior_unchanged():
    """Sanity: the fix is behaviour-preserving for single-threaded callers.

    Confirms the flap-guard still requires ``stable_cycles`` consecutive
    same-direction evaluations and that a committed scale is followed by a
    cooldown-blocked decision.
    """
    scaler = IntelligentAutoscaler(
        min_nodes=1,
        max_nodes=20,
        cooldown_seconds=60.0,
        hysteresis_band=1,
        stable_cycles=2,  # require 2 consecutive same-direction evals
    )
    metrics = _scale_metrics()

    # Round 1: first eval sets pending_dir=+1, count=1 < stable_cycles(2) -> no.
    d1 = scaler.evaluate(metrics)
    assert d1.should_scale is False
    assert d1.reason == "hysteresis"
    # Round 2: second consecutive same-direction eval -> scale.
    d2 = scaler.evaluate(metrics)
    assert d2.should_scale is True
    assert d2.target_nodes == 6
    # Immediately after, cooldown blocks further scales.
    d3 = scaler.evaluate(metrics)
    assert d3.should_scale is False
    assert d3.reason == "cooldown"
