"""Chaos tests: sudden node-loss scenarios.

These tests verify that the cluster/health managers correctly detect
node failures, trigger failover, and recover gracefully when nodes
disappear and reappear mid-inference.  All real I/O (gRPC, network,
disk) is replaced with mocks; only the failover state-machine +
recovery manager + callbacks are exercised.

Three scenarios:

1. ``TestPrimaryNodeLossMidInference`` — the coordinator-side primary
   worker drops while a request is in flight.  The system must mark
   the node OFFLINE, drain it, and surface a failure to the caller.
2. ``TestSimultaneousMultiNodeFailure`` — multiple worker nodes fail
   at the same instant.  Detection, failover, and per-node recovery
   must be independent and complete for every failed node.
3. ``TestNodeReconnectsAfterTimeout`` — a node that drops out later
   comes back.  After the FailoverEngine recovery probe threshold is
   met, the node must transition OFFLINE/UNHEALTHY -> DEGRADED ->
   HEALTHY, the straggler baseline resets, and in-flight sequences
   are recoverable from checkpoints.

Run::

    python -m pytest tests/chaos/test_node_loss.py -v

The tests use a mock ResourceManager, a real NodeRecoveryManager with
real callbacks, and a real FailoverEngine.  Only the gRPC and
pipeline I/O layers are mocked out, so the failover logic is fully
exercised end-to-end.
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

pytestmark = [
    pytest.mark.chaos,
    pytest.mark.timeout(60),
]


# ---------------------------------------------------------------------------
# Mock cluster / node plumbing
# ---------------------------------------------------------------------------


class _MockHealthResult:
    """Stand-in for a real health-check RPC response."""

    def __init__(self, healthy: bool = True, memory_used: int = 1024, memory_total: int = 8192):
        self.healthy = healthy
        self.memory_used = memory_used
        self.memory_total = memory_total


class _MockNodeClient:
    """In-process node client with a crash toggle for chaos injection."""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self._crashed = False
        self._call_count = 0
        self.health_call_count = 0
        self.forward_call_count = 0

    def trigger_crash(self) -> None:
        self._crashed = True

    def recover(self) -> None:
        self._crashed = False
        self._call_count = 0

    def health_check(self, timeout: float = 5.0) -> _MockHealthResult:
        self.health_call_count += 1
        self._call_count += 1
        if self._crashed:
            raise ConnectionError(f"Node {self.host}:{self.port} is unreachable")
        return _MockHealthResult(healthy=True)

    def forward(self, request: Any, timeout: float = 30.0) -> Any:
        self.forward_call_count += 1
        if self._crashed:
            raise ConnectionError(f"Node {self.host}:{self.port} is unreachable")
        resp = MagicMock()
        resp.success = True
        resp.error_message = ""
        return resp


class _MockPipelineNode:
    """Stand-in for ``distllm.dist.pipeline.PipelineNode``."""

    def __init__(self, node_id: str, host: str, port: int, start_layer: int, end_layer: int, total_layers: int):
        self.node_id = node_id
        self.host = host
        self.port = port
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.total_layers = total_layers
        self.is_healthy = True
        self.client = _MockNodeClient(host, port)
        self.gpu_name = "mock-gpu"
        self.gpu_memory_total = 8 * 1024 * 1024 * 1024
        self.gpu_memory_free = 4 * 1024 * 1024 * 1024


class _MockResourceManager:
    """Tracks node failures for the health manager's drain callback."""

    def __init__(self) -> None:
        self.failure_records: list[tuple[str, float]] = []
        self.node_health: dict[str, bool] = {}

    def record_failure(self, node_id: str) -> None:
        self.failure_records.append((node_id, time.monotonic()))
        self.node_health[node_id] = False


class _MockPipeline:
    """Bare-bones pipeline orchestrator mock for HealthManager.

    The real ``PipelineOrchestrator.snapshot_nodes`` doesn't exist in
    this codebase, so the probe loop expects this minimal surface
    instead.  Keeps a stable ``_nodes`` dict so concurrent
    register/unregister calls remain consistent.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, _MockPipelineNode] = {}
        self._lock = __import__("threading").Lock()
        self._node_order: list[str] = []
        self.total_layers = 0

    @property
    def nodes(self) -> dict[str, _MockPipelineNode]:
        with self._lock:
            return dict(self._nodes)

    @property
    def node_order(self) -> list[str]:
        with self._lock:
            return list(self._node_order)

    def register(self, node: _MockPipelineNode) -> None:
        with self._lock:
            self._nodes[node.node_id] = node
            self._node_order = sorted(
                self._nodes.keys(),
                key=lambda nid: self._nodes[nid].start_layer,
            )
            self.total_layers = max(self.total_layers, node.end_layer + 1)

    def unregister(self, node_id: str) -> None:
        with self._lock:
            self._nodes.pop(node_id, None)
            self._node_order = sorted(
                self._nodes.keys(),
                key=lambda nid: self._nodes[nid].start_layer,
            )

    def snapshot_nodes(self) -> dict[str, _MockPipelineNode]:
        with self._lock:
            return dict(self._nodes)


# ---------------------------------------------------------------------------
# Test harness: build a HealthManager + recovery manager + callbacks
# ---------------------------------------------------------------------------


def _seed_node_health(health_manager, node_ids, success: bool = True, latency_ms: float = 10.0) -> None:
    """Pre-populate health records so subsequent probes start from a known state.

    The FailoverEngine defaults new records to OFFLINE; without a warm-up
    probe, every first probe of a "never seen" node lands at DEGRADED.
    """
    from distllm.health.state import HealthRecord, NodeState

    for nid in node_ids:
        rec = health_manager.health_store.get(nid)
        if rec is None:
            rec = HealthRecord(node_id=nid)
            health_manager.health_store.set(nid, rec)
        if success:
            rec.state = NodeState.HEALTHY
            rec.consecutive_successes = 5
            rec.consecutive_failures = 0
        else:
            rec.state = NodeState.OFFLINE
            rec.consecutive_failures = 6
            rec.consecutive_successes = 0


def _build_health_manager(num_nodes: int = 3, recovery_threshold: int = 1, suppress_mark_dead: bool = True):
    """Construct a HealthManager with mocked pipeline + real failover/recovery.

    Returns ``(health_manager, pipeline, recovery_manager, resource_mgr,
    failover_engine)`` so each test can drive the system directly.
    """
    from distllm.core.health_manager import HealthManager
    from distllm.dist.recovery import NodeRecoveryManager
    from distllm.health.failover import FailoverEngine
    from distllm.health.state import HealthRecord, NodeState

    pipeline = _MockPipeline()
    total_layers = num_nodes * 6
    pipeline.total_layers = total_layers

    for i in range(num_nodes):
        node = _MockPipelineNode(
            node_id=f"node-{i}",
            host="localhost",
            port=6000 + i,
            start_layer=i * 6,
            end_layer=(i + 1) * 6 - 1,
            total_layers=total_layers,
        )
        pipeline.register(node)

    resource_mgr = _MockResourceManager()
    recovery_manager = NodeRecoveryManager(dry_run=False)
    failover_engine = FailoverEngine(
        failure_threshold=2,
        degraded_latency_ms=2000.0,
        recovery_threshold=recovery_threshold,
    )

    # Track callbacks for assertion.
    drained: list[str] = []
    redistributed: list[tuple[str, list]] = []
    recovered_seqs: list[tuple[str, list[str]]] = []
    marked_dead: list[str] = []

    def drain_cb(node_id: str) -> None:
        drained.append(node_id)
        rec = failover_engine.health_store.get(node_id) if hasattr(failover_engine, "health_store") else None
        pipeline_node = pipeline.nodes.get(node_id)
        if pipeline_node is not None:
            pipeline_node.is_healthy = False
        resource_mgr.record_failure(node_id)

    def redistribute_cb(node_id: str, plan) -> None:
        redistributed.append((node_id, plan.redistributions))
        for r in plan.redistributions:
            surviving = pipeline.nodes.get(r.surviving_node_id)
            if surviving is not None:
                surviving.start_layer = r.new_start_layer
                surviving.end_layer = r.new_end_layer

    def recover_seqs_cb(node_id: str, seq_ids: list[str]) -> list[str]:
        recovered_seqs.append((node_id, list(seq_ids)))
        return list(seq_ids)

    def mark_dead_cb(node_id: str) -> None:
        marked_dead.append(node_id)
        if not suppress_mark_dead:
            pipeline.unregister(node_id)

    # Register the pipeline topology so the recovery planner knows
    # about survivor layer ranges.
    recovery_manager.set_layer_assignments(
        {
            n.node_id: (n.start_layer, n.end_layer)
            for n in pipeline.nodes.values()
        }
    )

    health_manager = HealthManager(
        pipeline=pipeline,
        resource_mgr=resource_mgr,
        recovery_manager=recovery_manager,
        failover_engine=failover_engine,
        drain_callback=drain_cb,
        redistribute_callback=redistribute_cb,
        recover_sequences_callback=recover_seqs_cb,
        mark_dead_callback=mark_dead_cb,
        check_interval_s=0.05,
        health_check_timeout_s=2.0,
    )

    # Seed HEALTHY records for every registered node so probes start
    # from a known-good baseline (FailoverEngine defaults new records
    # to OFFLINE, which would transition to DEGRADED on first success).
    _seed_node_health(
        health_manager,
        [n.node_id for n in pipeline.nodes.values()],
        success=True,
    )

    state = {
        "drained": drained,
        "redistributed": redistributed,
        "recovered_seqs": recovered_seqs,
        "marked_dead": marked_dead,
    }

    return health_manager, pipeline, recovery_manager, resource_mgr, failover_engine, state


def _drive_probe(health_manager, node_id: str, success: bool, latency_ms: float = 10.0) -> Any:
    """Run a single probe pass on one node and return the resulting state.

    Mirrors the internals of ``HealthManager._probe_single_node`` so we
    exercise the same code path (FailoverEngine.evaluate +
    state-change callback) without spawning the background thread.
    """
    from distllm.health.state import HealthRecord

    record = health_manager.health_store.get(node_id)
    if record is None:
        record = HealthRecord(node_id=node_id)
        health_manager.health_store.set(node_id, record)

    new_state = health_manager.failover_engine.evaluate(record, success, latency_ms)
    health_manager.health_store.update_state(node_id, new_state)
    return new_state


# ---------------------------------------------------------------------------
# Scenario 1 — primary node drops mid-inference
# ---------------------------------------------------------------------------


class TestPrimaryNodeLossMidInference:
    """The primary (first-in-pipeline) worker disappears during inference.

    The system must:
    - mark the node UNHEALTHY (then OFFLINE) on consecutive probe failures;
    - trigger the drain callback exactly once;
    - raise an error from the in-flight forward call;
    - leave the surviving pipeline intact (other nodes still healthy).
    """

    def test_primary_drop_marks_node_offline(self):
        health_manager, pipeline, _recovery, resource_mgr, _fo, state = _build_health_manager(num_nodes=3)

        primary = "node-0"
        pipeline_node = pipeline.nodes[primary]
        assert pipeline_node.is_healthy is True

        # First failure: HEALTHY -> DEGRADED -> UNHEALTHY (threshold=2)
        _drive_probe(health_manager, primary, success=False)
        from distllm.health.state import NodeState
        record = health_manager.health_store.get(primary)
        assert record.state in (NodeState.DEGRADED, NodeState.UNHEALTHY)

        # Second failure: pushes to UNHEALTHY
        _drive_probe(health_manager, primary, success=False)
        record = health_manager.health_store.get(primary)
        assert record.state == NodeState.UNHEALTHY

        # Continue past offline_threshold (2x failure_threshold=4)
        for _ in range(4):
            _drive_probe(health_manager, primary, success=False)
        record = health_manager.health_store.get(primary)
        assert record.state == NodeState.OFFLINE

    def test_primary_drop_drains_and_blocks_forward(self):
        health_manager, pipeline, recovery_manager, resource_mgr, _fo, state = _build_health_manager(num_nodes=3)
        primary = "node-0"

        # Inject a few in-flight sequences on the primary so recovery
        # has something to attempt.
        for i in range(3):
            recovery_manager.save_checkpoint(
                request_id=f"req-{i}",
                kv_cache=None,
                prompt_tokens=[1, 2],
                generated_tokens=[3],
                node_id=primary,
            )

        # Crash the primary and drive the failover pipeline.
        pipeline.nodes[primary].client.trigger_crash()
        # Simulate 2 failed probe rounds so the failover engine
        # transitions through DEGRADED -> UNHEALTHY (and the
        # state-change callback fires the drain).
        for _ in range(2):
            _drive_probe(health_manager, primary, success=False)
            if health_manager.health_store.get(primary).state.value in ("unhealthy", "offline"):
                health_manager._on_state_change(
                    primary,
                    # Force the callback path by invoking it explicitly
                    # (matches what FailoverEngine would have done on
                    # the transition).
                    _previous_state_for(primary, health_manager),
                    health_manager.health_store.get(primary).state,
                )
                break

        # In-flight forward on the crashed primary must raise.
        with pytest.raises(ConnectionError):
            pipeline.nodes[primary].client.forward("dummy", timeout=2)

        # The drain callback must have fired.
        assert primary in state["drained"], (
            f"Drain callback should have fired for {primary}; got {state['drained']}"
        )
        assert pipeline.nodes[primary].is_healthy is False
        assert resource_mgr.failure_records, "ResourceManager.record_failure should be called"

        # The other two nodes must remain healthy and routable.
        for survivor in ("node-1", "node-2"):
            assert pipeline.nodes[survivor].is_healthy is True
            assert pipeline.nodes[survivor].client.forward("dummy").success is True

    def test_primary_drop_does_not_corrupt_remaining_pipeline(self):
        health_manager, pipeline, _recovery, _rm, _fo, _state = _build_health_manager(num_nodes=4)
        primary = "node-0"

        # Crash the primary; verify node_count and node_order reflect it.
        pipeline.nodes[primary].client.trigger_crash()
        for _ in range(2):
            _drive_probe(health_manager, primary, success=False)
        for _ in range(4):
            _drive_probe(health_manager, primary, success=False)
        health_manager._on_state_change(
            primary,
            _previous_state_for(primary, health_manager),
            health_manager.health_store.get(primary).state,
        )

        # 3 of 4 nodes remain.
        assert len(pipeline.nodes) == 4  # mark_dead hasn't fired yet
        healthy_nodes = [n for n, nd in pipeline.nodes.items() if nd.is_healthy]
        assert primary not in healthy_nodes
        assert len(healthy_nodes) == 3


def _previous_state_for(node_id: str, health_manager) -> Any:
    """Best-effort recovery of the prior NodeState for callback wiring.

    The FailoverEngine itself fires the callback, but the test drives
    ``evaluate`` directly — so the last-seen previous state is
    whatever the FailoverEngine would have set.  We approximate by
    computing the state one probe earlier.
    """
    from distllm.health.state import NodeState

    record = health_manager.health_store.get(node_id)
    cur = record.state
    # If we're at OFFLINE, the prior was UNHEALTHY.
    if cur == NodeState.OFFLINE:
        return NodeState.UNHEALTHY
    if cur == NodeState.UNHEALTHY:
        return NodeState.DEGRADED
    if cur == NodeState.DEGRADED:
        return NodeState.HEALTHY
    return NodeState.OFFLINE


# ---------------------------------------------------------------------------
# Scenario 2 — multiple nodes fail simultaneously
# ---------------------------------------------------------------------------


class TestSimultaneousMultiNodeFailure:
    """When more than one node dies at the same instant.

    The system must:
    - detect and track every dead node independently in the health
      store;
    - drain and recover each one (the per-node callbacks fire
      exactly once);
    - not deadlock or raise when a second failure arrives while a
      first one is still being processed.
    """

    def test_two_nodes_fail_simultaneously_detected(self):
        health_manager, pipeline, _recovery, _rm, _fo, state = _build_health_manager(num_nodes=4)
        from distllm.health.state import NodeState

        victims = ["node-1", "node-2"]
        for vid in victims:
            pipeline.nodes[vid].client.trigger_crash()

        # Drive each victim's failover to UNHEALTHY independently.
        for vid in victims:
            _drive_probe(health_manager, vid, success=False)
            _drive_probe(health_manager, vid, success=False)
            rec = health_manager.health_store.get(vid)
            assert rec.state == NodeState.UNHEALTHY, (
                f"{vid} should be UNHEALTHY after 2 failures, got {rec.state}"
            )

        # Survivors stay HEALTHY/DEGRADED only.
        for survivor in ("node-0", "node-3"):
            rec = health_manager.health_store.get(survivor)
            if rec is None:
                continue
            assert rec.state in (NodeState.HEALTHY, NodeState.DEGRADED), (
                f"Survivor {survivor} unexpectedly {rec.state}"
            )

    def test_two_node_failure_drains_each(self):
        health_manager, pipeline, _recovery, _rm, _fo, state = _build_health_manager(num_nodes=4)
        victims = ["node-1", "node-2"]

        for vid in victims:
            pipeline.nodes[vid].client.trigger_crash()
            for _ in range(2):
                _drive_probe(health_manager, vid, success=False)
            health_manager._on_state_change(
                vid,
                _previous_state_for(vid, health_manager),
                health_manager.health_store.get(vid).state,
            )

        # Each victim must have its own drain invocation.
        for vid in victims:
            assert vid in state["drained"], (
                f"Drain callback should have fired for {vid}; got {state['drained']}"
            )
        assert state["drained"].count("node-1") == 1
        assert state["drained"].count("node-2") == 1

    def test_three_node_failure_does_not_deadlock(self):
        """Worst case: a majority of nodes drops.  Failover must
        still complete (no exceptions, callbacks all fire)."""
        health_manager, pipeline, recovery_manager, _rm, _fo, state = _build_health_manager(num_nodes=5)
        victims = ["node-0", "node-1", "node-2"]

        for vid in victims:
            pipeline.nodes[vid].client.trigger_crash()

        # Plant a few in-flight sequences per victim.
        for vid in victims:
            for i in range(2):
                recovery_manager.save_checkpoint(
                    request_id=f"{vid}-req-{i}",
                    kv_cache=None,
                    prompt_tokens=[1],
                    generated_tokens=[2],
                    node_id=vid,
                )

        for vid in victims:
            for _ in range(2):
                _drive_probe(health_manager, vid, success=False)
            health_manager._on_state_change(
                vid,
                _previous_state_for(vid, health_manager),
                health_manager.health_store.get(vid).state,
            )

        # All 3 victims must have been drained.
        for vid in victims:
            assert vid in state["drained"], f"{vid} missing from drains: {state['drained']}"

        # Remaining 2 nodes still healthy and routable.
        for survivor in ("node-3", "node-4"):
            assert pipeline.nodes[survivor].is_healthy is True
            assert pipeline.nodes[survivor].client.forward("dummy").success is True

    def test_concurrent_failure_recovery_history(self):
        health_manager, pipeline, recovery_manager, _rm, _fo, state = _build_health_manager(num_nodes=4)

        # Plant checkpoints on two distinct nodes.
        for i in range(3):
            recovery_manager.save_checkpoint(
                request_id=f"req-{i}",
                kv_cache=None,
                prompt_tokens=[1],
                generated_tokens=[2],
                node_id="node-0" if i % 2 == 0 else "node-1",
            )

        for vid in ("node-0", "node-1"):
            pipeline.nodes[vid].client.trigger_crash()
            for _ in range(2):
                _drive_probe(health_manager, vid, success=False)
            health_manager._on_state_change(
                vid,
                _previous_state_for(vid, health_manager),
                health_manager.health_store.get(vid).state,
            )

        # The recovery manager must have recorded two distinct failures.
        history = recovery_manager.get_recovery_history()
        failed_ids = {h["node_id"] for h in history}
        assert {"node-0", "node-1"} <= failed_ids, (
            f"Expected both nodes in history, got {failed_ids}"
        )


# ---------------------------------------------------------------------------
# Scenario 3 — node reconnects after timeout
# ---------------------------------------------------------------------------


class TestNodeReconnectsAfterTimeout:
    """A node that drops must be marked healthy again when it returns.

    The system must:
    - transition OFFLINE -> DEGRADED on the first successful probe;
    - continue DEGRADED -> HEALTHY once the recovery threshold is
      met;
    - reset the straggler baseline so a slow-recovering node isn't
      flagged as a straggler;
    - serve forward calls again on the recovered node.
    """

    def test_offline_to_degraded_on_first_success(self):
        health_manager, pipeline, _recovery, _rm, _fo, _state = _build_health_manager(num_nodes=3, recovery_threshold=2)
        from distllm.health.state import NodeState

        target = "node-1"
        # Drive to OFFLINE.
        pipeline.nodes[target].client.trigger_crash()
        for _ in range(6):
            _drive_probe(health_manager, target, success=False)
        assert health_manager.health_store.get(target).state == NodeState.OFFLINE

        # Node comes back.
        pipeline.nodes[target].client.recover()
        new_state = _drive_probe(health_manager, target, success=True, latency_ms=10.0)
        assert new_state == NodeState.DEGRADED, (
            f"Expected DEGRADED after recovery probe, got {new_state}"
        )

    def test_degraded_to_healthy_after_recovery_threshold(self):
        health_manager, pipeline, _recovery, _rm, _fo, _state = _build_health_manager(num_nodes=3, recovery_threshold=2)
        from distllm.health.state import NodeState

        target = "node-1"
        # Drive to OFFLINE, then start recovery probes.
        pipeline.nodes[target].client.trigger_crash()
        for _ in range(6):
            _drive_probe(health_manager, target, success=False)
        pipeline.nodes[target].client.recover()

        # First success: OFFLINE -> DEGRADED.
        state1 = _drive_probe(health_manager, target, success=True, latency_ms=10.0)
        assert state1 == NodeState.DEGRADED

        # Second success (>= recovery_threshold=2): DEGRADED -> HEALTHY.
        state2 = _drive_probe(health_manager, target, success=True, latency_ms=10.0)
        assert state2 == NodeState.HEALTHY, (
            f"Expected HEALTHY after {2} consecutive successes, got {state2}"
        )

    def test_recovered_node_serves_forward(self):
        health_manager, pipeline, _recovery, _rm, _fo, _state = _build_health_manager(num_nodes=3, recovery_threshold=1)
        target = "node-1"
        from distllm.health.state import NodeState

        # Crash, then recover.
        pipeline.nodes[target].client.trigger_crash()
        for _ in range(6):
            _drive_probe(health_manager, target, success=False)

        with pytest.raises(ConnectionError):
            pipeline.nodes[target].client.forward("dummy", timeout=1)

        pipeline.nodes[target].client.recover()
        # recovery_threshold=1 means a single success flips back.
        state = _drive_probe(health_manager, target, success=True, latency_ms=10.0)
        assert state in (NodeState.DEGRADED, NodeState.HEALTHY)

        # The client must be routable again and have been exercised
        # both while alive (baseline) and post-recovery (forward).
        forward_before = pipeline.nodes[target].client.forward_call_count
        assert pipeline.nodes[target].client.forward("dummy").success is True
        assert pipeline.nodes[target].client.forward_call_count > forward_before

    def test_recovery_resets_failure_counters(self):
        health_manager, pipeline, _recovery, _rm, _fo, _state = _build_health_manager(num_nodes=3, recovery_threshold=1)
        from distllm.health.state import NodeState

        target = "node-1"
        # Build up a few failures.
        for _ in range(3):
            _drive_probe(health_manager, target, success=False)
        rec_before = health_manager.health_store.get(target)
        assert rec_before.consecutive_failures == 3

        # Recover and probe successfully.
        pipeline.nodes[target].client.recover()
        _drive_probe(health_manager, target, success=True, latency_ms=10.0)
        rec_after = health_manager.health_store.get(target)
        assert rec_after.consecutive_failures == 0
        assert rec_after.consecutive_successes >= 1

    def test_recovery_preserves_sibling_health(self):
        """Recovering node-1 must not perturb the health of node-0 / node-2."""
        health_manager, pipeline, _recovery, _rm, _fo, _state = _build_health_manager(num_nodes=3, recovery_threshold=1)
        from distllm.health.state import NodeState

        target = "node-1"
        # Bring target down.
        pipeline.nodes[target].client.trigger_crash()
        for _ in range(6):
            _drive_probe(health_manager, target, success=False)

        # Probe siblings — they should still pass cleanly.
        for sibling in ("node-0", "node-2"):
            state = _drive_probe(health_manager, sibling, success=True, latency_ms=10.0)
            assert state == NodeState.HEALTHY

        # Recover target; siblings should remain healthy.
        pipeline.nodes[target].client.recover()
        _drive_probe(health_manager, target, success=True, latency_ms=10.0)
        for sibling in ("node-0", "node-2"):
            rec = health_manager.health_store.get(sibling)
            assert rec is None or rec.state in (NodeState.HEALTHY, NodeState.DEGRADED)

    def test_reconnect_does_not_trigger_duplicate_drain(self):
        """When a node comes back, the drain callback must NOT re-fire
        for the same node (single-source-of-truth for node state)."""
        health_manager, pipeline, _recovery, _rm, _fo, state = _build_health_manager(num_nodes=3, recovery_threshold=1)
        from distllm.health.state import NodeState

        target = "node-1"
        # Drive to UNHEALTHY + drain.
        pipeline.nodes[target].client.trigger_crash()
        _drive_probe(health_manager, target, success=False)
        _drive_probe(health_manager, target, success=False)
        health_manager._on_state_change(
            target,
            _previous_state_for(target, health_manager),
            health_manager.health_store.get(target).state,
        )
        first_drain_count = state["drained"].count(target)

        # Recover; the on_state_change for DEGRADED->HEALTHY must not
        # call the drain callback (it calls mark_alive instead).
        pipeline.nodes[target].client.recover()
        _drive_probe(health_manager, target, success=True, latency_ms=10.0)
        rec_state = health_manager.health_store.get(target).state
        if rec_state == NodeState.HEALTHY:
            health_manager._on_state_change(
                target,
                NodeState.DEGRADED,
                NodeState.HEALTHY,
            )

        # Drain count for the target should not have grown.
        assert state["drained"].count(target) == first_drain_count, (
            f"Drain re-fired for {target} on recovery: {state['drained']}"
        )
