"""Regression tests for High-severity findings H9 (load-balancer thundering
herd) and H10 (coordinator_state self-deadlock).

Both are torch-free and run fast.

H10: ``CoordinatorStateMachine.stats()`` held a non-reentrant ``Lock`` and then
called ``uptime_s()`` / ``time_in_role_s()`` which re-acquire it → self-deadlock
(hang). Fixed by switching to ``RLock``. The test calls ``stats()`` from a
watchdog-guarded thread and asserts it returns quickly.

H9: connection-sensitive strategies (LEAST_CONNECTIONS / POWER_OF_TWO /
LATENCY_WEIGHTED) selected on a stale snapshot and incremented
``active_connections`` only *after* selection, so N concurrent picks all chose
the same node (thundering herd). Fixed by selecting + incrementing atomically
under one lock. The test fires many concurrent picks and asserts the load is
spread rather than piled onto a single node.
"""

from __future__ import annotations

import threading

from distllm.core.coordinator_state import (
    CoordinatorRole,
    CoordinatorStateMachine,
)
from distllm.core.load_balancer import LBStrategy, LoadBalancer


# ── H10: stats() must not self-deadlock ───────────────────────────────────

def test_coordinator_stats_no_deadlock():
    sm = CoordinatorStateMachine()
    sm.transition_to(CoordinatorRole.FOLLOWER)
    sm.transition_to(CoordinatorRole.LEADER)

    result: dict = {}
    err: list = []

    def _call():
        try:
            result["stats"] = sm.stats()
        except Exception as e:  # pragma: no cover
            err.append(e)

    t = threading.Thread(target=_call, daemon=True)
    t.start()
    t.join(timeout=5.0)

    assert not t.is_alive(), "stats() deadlocked (thread still running after 5s)"
    assert not err, f"stats() raised: {err}"
    stats = result["stats"]
    assert stats["role"] == "leader"
    assert stats["uptime_s"] >= 0.0
    assert stats["time_in_role_s"] >= 0.0


def test_coordinator_stats_repeated_calls():
    # Repeated calls must also not deadlock (RLock re-acquire each time).
    sm = CoordinatorStateMachine()
    sm.transition_to(CoordinatorRole.LEADER)
    for _ in range(50):
        s = sm.stats()
        assert s["role"] == "leader"


# ── H9: concurrent picks must spread across nodes ─────────────────────────

def _run_concurrent_picks(strategy: LBStrategy, n_nodes: int = 4, n_req: int = 40):
    lb = LoadBalancer(strategy=strategy)
    for i in range(n_nodes):
        lb.add_target(f"10.0.0.{i}", 50050, node_id=f"coord-{i}")

    picked: list = []
    lock = threading.Lock()
    barrier = threading.Barrier(n_req)

    def _worker():
        # Synchronize all threads to maximize contention on the same snapshot.
        barrier.wait()
        target = lb.pick("req")
        with lock:
            picked.append(target.node_id)

    threads = [threading.Thread(target=_worker) for _ in range(n_req)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)

    assert all(not t.is_alive() for t in threads), "pick() hung under concurrency"
    return picked


def test_least_connections_spreads_under_concurrency():
    picked = _run_concurrent_picks(LBStrategy.LEAST_CONNECTIONS, n_nodes=4, n_req=40)
    distinct = set(picked)
    # With atomic pick+increment, 40 requests across 4 empty nodes must not all
    # land on one node. A stale-snapshot bug piles them all onto a single node.
    assert len(distinct) >= 2, f"thundering herd: all requests went to {distinct}"
    # Load should be reasonably balanced: no node should absorb everything.
    counts = {n: picked.count(n) for n in distinct}
    assert max(counts.values()) < len(picked), f"one node took all: {counts}"


def test_power_of_two_spreads_under_concurrency():
    picked = _run_concurrent_picks(LBStrategy.POWER_OF_TWO, n_nodes=4, n_req=40)
    assert len(set(picked)) >= 2, f"herd on {set(picked)}"


def test_least_connections_sequential_spread():
    # Sequential picks with no completions should each land on a distinct
    # least-loaded node (each pick reserves a slot). NOTE: this property holds
    # for the fixed code; it is a sanity check, not a discriminating regression
    # guard for the concurrency race (see module docstring / H9 notes). The
    # true herd bug only manifests under concurrency with a wide snapshot->commit
    # window, which CPython's GIL masks in-process.
    lb = LoadBalancer(strategy=LBStrategy.LEAST_CONNECTIONS)
    for i in range(3):
        lb.add_target(f"10.0.1.{i}", 50050, node_id=f"c-{i}")
    picks = [lb.pick("r").node_id for _ in range(3)]
    assert len(set(picks)) == 3, f"expected even spread, got {picks}"


# ── H9 (deterministic): source-level hook forces the race window ───────────

def test_least_connections_no_herd_deterministic():
    """Deterministically expose the thundering-herd race using the built-in
    ``_post_snapshot_hook``.

    A barrier in the hook forces every concurrent pick() to finish its Phase-1
    snapshot (all observing identical, all-zero connection counts) BEFORE any
    of them commits an increment. Under this schedule:

      * A buggy impl (select on the stale snapshot, increment afterwards) makes
        every thread choose the same least-loaded node -> full herd.
      * The fixed impl re-reads live counters and increments atomically inside
        one lock, so the reservations serialize and spread across nodes.

    This is GIL-independent: the barrier guarantees the interleaving the real
    bug needs, rather than relying on lucky timing.
    """
    n_nodes = 4
    n_req = n_nodes  # exactly one request per node if correctly balanced

    lb = LoadBalancer(strategy=LBStrategy.LEAST_CONNECTIONS)
    for i in range(n_nodes):
        lb.add_target(f"10.0.2.{i}", 50050, node_id=f"c-{i}")

    # All workers rendezvous inside the post-snapshot hook, guaranteeing every
    # pick has taken its snapshot before any proceeds to select/commit.
    barrier = threading.Barrier(n_req)
    lb._post_snapshot_hook = lambda: barrier.wait(timeout=5.0)

    picked: list = []
    plock = threading.Lock()

    def _worker():
        target = lb.pick("req")
        with plock:
            picked.append(target.node_id)

    threads = [threading.Thread(target=_worker) for _ in range(n_req)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)

    assert all(not t.is_alive() for t in threads), "pick() hung"
    # The fixed, atomic implementation must spread one request to each node.
    counts = {n: picked.count(n) for n in set(picked)}
    assert len(set(picked)) == n_nodes, (
        f"thundering herd: {n_req} simultaneous picks did not spread across "
        f"all {n_nodes} nodes -> {counts}"
    )


# ── H10 (deterministic): lock must be reentrant ───────────────────────────

def test_coordinator_lock_is_reentrant():
    """The state-machine lock must be an RLock so stats() can call the
    lock-holding helpers without self-deadlock. Assert the concrete type is a
    reentrant lock (RLock instances are of threading._RLock / support nested
    acquire)."""
    sm = CoordinatorStateMachine()
    lock = sm._lock
    # An RLock can be acquired twice by the same thread; a plain Lock cannot.
    acquired_first = lock.acquire(timeout=1.0)
    acquired_second = lock.acquire(timeout=1.0)
    try:
        assert acquired_first and acquired_second, (
            "lock is not reentrant (plain Lock) -> stats() would self-deadlock"
        )
    finally:
        if acquired_second:
            lock.release()
        if acquired_first:
            lock.release()
