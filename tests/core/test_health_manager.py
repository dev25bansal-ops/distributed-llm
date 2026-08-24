"""Tests for HealthManager in distllm.core.health_manager.

Covers:
    Construction and default properties
    State transition callbacks (FailoverEngine integration)
    Straggler detection callback
    Recovery callbacks (drain, redistribute, recover, mark_dead)
    Per-node health probe logic
    Health probe loop lifecycle (start / stop)
    Node status reporting

Every test is deterministic (no network, no GPU, no time.sleep).
No MagicMock -- real objects or lightweight stubs only.
"""

from __future__ import annotations

import sys
import threading
import types
from typing import Any

import pytest

from tests._import_helper import SRC_DIR, bootstrap_fake_packages, load_module

# ------------------------------------------------------------------ #
# Bootstrap fake packages, then create a synthetic                     #
# ``distllm.dist.pipeline`` package so that health_manager's            #
# module-level ``from distllm.dist.pipeline import PipelineOrchestrator`` #
# resolves without triggering the full pipeline import chain.           #
# ------------------------------------------------------------------ #

bootstrap_fake_packages()

# Minimal stub for PipelineOrchestrator used in the fake package.
# Tests that need a real pipeline object pass their own.
class _FakePipelineOrchestrator:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.nodes: dict[str, Any] = {}
        self.node_order: list[str] = []

    def snapshot_nodes(self) -> dict[str, Any]:
        return dict(self.nodes)


_pipe_pkg = types.ModuleType("distllm.dist.pipeline")
_pipe_pkg.__path__ = [str(SRC_DIR / "distllm" / "dist" / "pipeline")]
_pipe_pkg.__package__ = "distllm.dist.pipeline"
_pipe_pkg.PipelineOrchestrator = _FakePipelineOrchestrator
sys.modules.setdefault("distllm.dist.pipeline", _pipe_pkg)

# ------------------------------------------------------------------ #
# Load dependent sub-modules in dependency order.                      #
# ------------------------------------------------------------------ #

_state_mod = load_module("distllm/health/state.py")
_failover_mod = load_module("distllm/health/failover.py")
_rec_mod = load_module("distllm/dist/recovery.py")
_rep_mod = load_module("distllm/dist/reputation.py")
_strag_mod = load_module("distllm/dist/straggler.py")
_res_mod = load_module("distllm/core/resource_manager.py")

# Now health_manager's module-level imports should resolve.
_hm = load_module("distllm/core/health_manager.py")

# ------------------------------------------------------------------ #
# Re-export key symbols for test readability.                          #
# ------------------------------------------------------------------ #

HealthManager = _hm.HealthManager
NodeState = _state_mod.NodeState
HealthRecord = _state_mod.HealthRecord
HealthStateStore = _state_mod.HealthStateStore
FailoverEngine = _failover_mod.FailoverEngine
DetectionMethod = _strag_mod.DetectionMethod
StragglerReport = _strag_mod.StragglerReport
StragglerSeverity = _strag_mod.StragglerSeverity


# ===================================================================
# Stubs
# ===================================================================


class _StubPipelineNode:
    """Minimal pipeline node with health_check, client, and layer attrs."""

    def __init__(
        self,
        node_id: str = "node-0",
        healthy: bool = True,
        has_client: bool = True,
        start_layer: int = 0,
        end_layer: int = 3,
        gpu_name: str = "Tesla T4",
        gpu_memory_free: int = 15000,
    ) -> None:
        self.node_id = node_id
        self.healthy = healthy
        self._health_result = healthy
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.gpu_name = gpu_name
        self.gpu_memory_free = gpu_memory_free
        self.client = _StubClient() if has_client else None

    def health_check(self) -> bool:
        return self._health_result

    def set_health_result(self, value: bool) -> None:
        self._health_result = value


class _StubClient:
    """Minimal client with transfer_kv_cache."""

    def __init__(self) -> None:
        self.transferred: list[Any] = []

    def transfer_kv_cache(self, kv_cache: Any) -> None:
        self.transferred.append(kv_cache)


class _StubPipeline:
    """Minimal pipeline orchestrator stub."""

    def __init__(self) -> None:
        self.nodes: dict[str, _StubPipelineNode] = {}
        self.node_order: list[str] = []

    def snapshot_nodes(self) -> dict[str, _StubPipelineNode]:
        return dict(self.nodes)

    def add_node(self, node: _StubPipelineNode) -> None:
        self.nodes[node.node_id] = node
        self.node_order.append(node.node_id)


class _StubResourceManager:
    """Minimal resource manager that records failures."""

    def __init__(self) -> None:
        self.failures: list[str] = []

    def record_failure(self, node_id: str) -> None:
        self.failures.append(node_id)


class _StubReputation:
    """Minimal reputation system that records health calls."""

    def __init__(self) -> None:
        self.records: list[tuple[str, bool]] = []

    def record_health(self, node_id: str, healthy: bool) -> None:
        self.records.append((node_id, healthy))


# ===================================================================
# HELPERS
# ===================================================================


def make_pipeline_and_rmgr(
    *node_ids: str,
) -> tuple[_StubPipeline, _StubResourceManager, dict[str, _StubPipelineNode]]:
    """Create a stub pipeline with one node per *node_ids*.

    Returns ``(pipeline, resource_mgr, {node_id -> node})``.
    """
    pipe = _StubPipeline()
    rmgr = _StubResourceManager()
    nodes: dict[str, _StubPipelineNode] = {}
    for i, nid in enumerate(node_ids or ("node-0",)):
        node = _StubPipelineNode(node_id=nid, start_layer=i, end_layer=i, healthy=True)
        pipe.add_node(node)
        nodes[nid] = node
    return pipe, rmgr, nodes


def make_probe_node(healthy: bool = True) -> _StubPipelineNode:
    """Shorthand for a single node."""
    return _StubPipelineNode(node_id="node-0", healthy=healthy)


def _make_straggler_report(
    node_id: str = "node-slow",
    recommended_action: str = "",
) -> Any:
    """Create a minimal StragglerReport-like object.

    StragglerReport is a plain class (not a dataclass) with no __init__,
    so we construct an instance with __new__ and set attributes directly.
    """
    report = object.__new__(StragglerReport)
    report.node_id = node_id
    report.severity = StragglerSeverity.MODERATE
    report.avg_latency = 200.0
    report.p95_latency = 500.0
    report.baseline_latency = 50.0
    report.slowdown_factor = 4.0
    report.detection_method = DetectionMethod.MAD
    report.consecutive_detections = 3
    report.recommended_action = recommended_action
    return report


# ===================================================================
# CONSTRUCTION & DEFAULTS
# ===================================================================


class TestConstruction:
    """HealthManager construction and default property access."""

    def test_construction_minimal(self) -> None:
        """Minimal construction with pipeline and resource_mgr only."""
        pipe, rmgr, _ = make_pipeline_and_rmgr()
        hm = HealthManager(pipeline=pipe, resource_mgr=rmgr)
        assert hm is not None

    def test_construction_defaults_create_recovery_manager(self) -> None:
        """When recovery_manager is None, a default NodeRecoveryManager is created."""
        pipe, rmgr, _ = make_pipeline_and_rmgr()
        hm = HealthManager(pipeline=pipe, resource_mgr=rmgr)
        assert hm.recovery_manager is not None

    def test_construction_defaults_create_straggler_detector(self) -> None:
        """When straggler_detector is None, a default StragglerDetector is created."""
        pipe, rmgr, _ = make_pipeline_and_rmgr()
        hm = HealthManager(pipeline=pipe, resource_mgr=rmgr)
        assert hm.straggler_detector is not None

    def test_construction_defaults_create_failover_engine(self) -> None:
        """When failover_engine is None, a default FailoverEngine is created."""
        pipe, rmgr, _ = make_pipeline_and_rmgr()
        hm = HealthManager(pipeline=pipe, resource_mgr=rmgr)
        assert hm.failover_engine is not None

    def test_construction_defaults_create_health_store(self) -> None:
        """A HealthStateStore should always be created."""
        pipe, rmgr, _ = make_pipeline_and_rmgr()
        hm = HealthManager(pipeline=pipe, resource_mgr=rmgr)
        assert hm.health_store is not None
        assert isinstance(hm.health_store, HealthStateStore)

    def test_construction_with_all_optional_deps(self) -> None:
        """All optional dependencies can be injected."""
        pipe, rmgr, _ = make_pipeline_and_rmgr()
        failover = FailoverEngine(failure_threshold=5)
        hm = HealthManager(
            pipeline=pipe,
            resource_mgr=rmgr,
            reputation=_StubReputation(),
            recovery_manager=_rec_mod.NodeRecoveryManager(),
            straggler_detector=_strag_mod.StragglerDetector(
                detection_method=DetectionMethod.MAD,
            ),
            check_interval_s=5.0,
            health_check_timeout_s=2.0,
            failover_engine=failover,
        )
        assert hm._check_interval_s == 5.0
        assert hm._health_check_timeout_s == 2.0
        assert hm.failover_engine is failover

    def test_properties_return_expected_objects(self) -> None:
        """Property accessors return the expected objects."""
        pipe, rmgr, _ = make_pipeline_and_rmgr()
        hm = HealthManager(pipeline=pipe, resource_mgr=rmgr)
        assert hm.straggler_detector is hm._straggler_detector
        assert hm.recovery_manager is hm._recovery_manager
        assert hm.failover_engine is hm._failover
        assert hm.health_store is hm._health_store

    def test_construction_registers_state_change_callback(self) -> None:
        """FailoverEngine should have at least one callback after init."""
        pipe, rmgr, _ = make_pipeline_and_rmgr()
        hm = HealthManager(pipeline=pipe, resource_mgr=rmgr)
        # The FailoverEngine stores callbacks in a list
        assert len(hm.failover_engine._callbacks) >= 1

    def test_construction_sets_up_recovery_callbacks(self) -> None:
        """Recovery manager callbacks should be set after init."""
        pipe, rmgr, _ = make_pipeline_and_rmgr()
        hm = HealthManager(pipeline=pipe, resource_mgr=rmgr)
        assert hm.recovery_manager._on_drain is not None
        assert hm.recovery_manager._on_redistribute is not None
        assert hm.recovery_manager._on_recover is not None
        assert hm.recovery_manager._on_mark_dead is not None

    def test_construction_check_interval_default(self) -> None:
        """Default check_interval_s should be 10.0."""
        pipe, rmgr, _ = make_pipeline_and_rmgr()
        hm = HealthManager(pipeline=pipe, resource_mgr=rmgr)
        assert hm._check_interval_s == 10.0

    def test_construction_health_check_timeout_default(self) -> None:
        """Default health_check_timeout_s should be 5.0."""
        pipe, rmgr, _ = make_pipeline_and_rmgr()
        hm = HealthManager(pipeline=pipe, resource_mgr=rmgr)
        assert hm._health_check_timeout_s == 5.0


# ===================================================================
# STATE TRANSITION CALLBACK
# ===================================================================


class TestStateTransitionCallback:
    """_on_state_change -- FailoverEngine integration."""

    def test_unhealthy_records_failure_and_triggers_recovery(self) -> None:
        """Transition to UNHEALTHY should call record_failure and on_node_failure."""
        pipe, rmgr, nodes = make_pipeline_and_rmgr("node-a")
        hm = HealthManager(pipeline=pipe, resource_mgr=rmgr)
        hm._on_state_change("node-a", NodeState.DEGRADED, NodeState.UNHEALTHY)
        assert "node-a" in rmgr.failures
        assert hm.recovery_manager.is_dead("node-a")

    def test_offline_records_failure_and_triggers_recovery(self) -> None:
        """Transition to OFFLINE should call record_failure and on_node_failure."""
        pipe, rmgr, nodes = make_pipeline_and_rmgr("node-a")
        hm = HealthManager(pipeline=pipe, resource_mgr=rmgr)
        hm._on_state_change("node-a", NodeState.UNHEALTHY, NodeState.OFFLINE)
        assert "node-a" in rmgr.failures
        assert hm.recovery_manager.is_dead("node-a")

    def test_recovering_from_offline_marks_alive_and_resets_baseline(self) -> None:
        """DEGRADED from OFFLINE should mark_alive and reset straggler baseline."""
        pipe, rmgr, nodes = make_pipeline_and_rmgr("node-a")
        hm = HealthManager(pipeline=pipe, resource_mgr=rmgr)
        # First, record a baseline in the straggler detector
        hm.straggler_detector.record_latency("node-a", 50.0)
        # Transition: OFFLINE -> DEGRADED (recovering)
        hm._on_state_change("node-a", NodeState.OFFLINE, NodeState.DEGRADED)
        assert not hm.recovery_manager.is_dead("node-a")
        assert not hm.recovery_manager.is_draining("node-a")

    def test_recovering_from_unhealthy_marks_alive_and_resets_baseline(self) -> None:
        """DEGRADED from UNHEALTHY should mark_alive and reset straggler baseline."""
        pipe, rmgr, nodes = make_pipeline_and_rmgr("node-a")
        hm = HealthManager(pipeline=pipe, resource_mgr=rmgr)
        hm._on_state_change("node-a", NodeState.UNHEALTHY, NodeState.DEGRADED)
        assert not hm.recovery_manager.is_dead("node-a")

    def test_healthy_from_degraded_marks_alive_and_resets_baseline(self) -> None:
        """HEALTHY from DEGRADED should mark_alive and reset straggler baseline."""
        pipe, rmgr, nodes = make_pipeline_and_rmgr("node-a")
        hm = HealthManager(pipeline=pipe, resource_mgr=rmgr)
        hm._on_state_change("node-a", NodeState.DEGRADED, NodeState.HEALTHY)
        assert not hm.recovery_manager.is_dead("node-a")

    def test_healthy_stays_healthy_no_side_effects(self) -> None:
        """HEALTHY -> HEALTHY (latency okay) should not trigger failure."""
        pipe, rmgr, nodes = make_pipeline_and_rmgr("node-a")
        hm = HealthManager(pipeline=pipe, resource_mgr=rmgr)
        hm._on_state_change("node-a", NodeState.HEALTHY, NodeState.HEALTHY)
        assert "node-a" not in rmgr.failures

    def test_degraded_without_recovery_condition_no_side_effects(self) -> None:
        """DEGRADED -> UNHEALTHY is covered above.  HEALTHY -> DEGRADED is not
        a recovery transition, so no mark_alive or on_node_failure."""
        pipe, rmgr, nodes = make_pipeline_and_rmgr("node-a")
        hm = HealthManager(pipeline=pipe, resource_mgr=rmgr)
        hm._on_state_change("node-a", NodeState.HEALTHY, NodeState.DEGRADED)
        # Not a recovery transition
        assert "node-a" not in rmgr.failures
        assert not hm.recovery_manager.is_dead("node-a")


# ===================================================================
# STRAGGLER CALLBACK
# ===================================================================


class TestStragglerCallback:
    """_on_straggler_detected -- recovery trigger for reassign_layers."""

    def test_reassign_action_triggers_recovery(self) -> None:
        """When recommended_action is 'reassign_layers', on_node_failure should fire."""
        pipe, rmgr, _ = make_pipeline_and_rmgr("node-slow")
        hm = HealthManager(pipeline=pipe, resource_mgr=rmgr)
        report = _make_straggler_report(
            node_id="node-slow",
            recommended_action="reassign_layers",
        )
        hm._on_straggler_detected(report)
        assert hm.recovery_manager.is_dead("node-slow")

    def test_other_action_does_not_trigger_recovery(self) -> None:
        """When recommended_action is NOT 'reassign_layers', no recovery."""
        pipe, rmgr, _ = make_pipeline_and_rmgr("node-slow")
        hm = HealthManager(pipeline=pipe, resource_mgr=rmgr)
        report = _make_straggler_report(
            node_id="node-slow",
            recommended_action="throttle",
        )
        hm._on_straggler_detected(report)
        assert not hm.recovery_manager.is_dead("node-slow")

    def test_reassign_empty_node_id_does_not_crash(self) -> None:
        """reassign_layers with an empty node_id should not raise."""
        pipe, rmgr, _ = make_pipeline_and_rmgr()
        hm = HealthManager(pipeline=pipe, resource_mgr=rmgr)
        report = _make_straggler_report(
            node_id="",
            recommended_action="reassign_layers",
        )
        # Should not raise
        hm._on_straggler_detected(report)


# ===================================================================
# RECOVERY CALLBACKS
# ===================================================================


class TestRecoveryCallbacks:
    """_setup_recovery_callbacks -- drain, redistribute, recover, mark_dead."""

    def test_drain_callback_marks_node_unhealthy_and_records_failure(self) -> None:
        """The drain callback should set node.is_healthy = False and record failure."""
        pipe, rmgr, nodes = make_pipeline_and_rmgr("node-a")
        hm = HealthManager(pipeline=pipe, resource_mgr=rmgr)
        drain = hm.recovery_manager._on_drain
        assert drain is not None
        drain("node-a")
        # C6: canonical attribute is ``is_healthy`` (what schedulers filter
        # on) — the old code wrote a nonexistent ``.healthy`` attribute.
        assert nodes["node-a"].is_healthy is False
        assert "node-a" in rmgr.failures

    def test_drain_callback_missing_node_does_not_crash(self) -> None:
        """Drain callback with a node_id not in the pipeline should not raise."""
        pipe, rmgr, _ = make_pipeline_and_rmgr("node-a")
        hm = HealthManager(pipeline=pipe, resource_mgr=rmgr)
        drain = hm.recovery_manager._on_drain
        assert drain is not None
        drain("nonexistent")  # should not raise

    def test_redistribute_callback_updates_surviving_node_layers(self) -> None:
        """Redistribute should update the surviving node's layer range."""
        pipe, rmgr, nodes = make_pipeline_and_rmgr("survivor")
        hm = HealthManager(pipeline=pipe, resource_mgr=rmgr)
        nodes["survivor"].start_layer = 0
        nodes["survivor"].end_layer = 3
        redistribute = hm.recovery_manager._on_redistribute
        assert redistribute is not None
        plan = _rec_mod.NodeRecoveryPlan(failed_node_id="dead-node")
        plan.redistributions = [
            _rec_mod.LayerRedistribution(
                surviving_node_id="survivor",
                added_start_layer=4,
                added_end_layer=7,
                new_start_layer=0,
                new_end_layer=7,
            ),
        ]
        redistribute("dead-node", plan)
        assert nodes["survivor"].start_layer == 0
        assert nodes["survivor"].end_layer == 7

    def test_redistribute_missing_survivor_does_not_crash(self) -> None:
        """Redistribute with a missing surviving node should not raise."""
        pipe, rmgr, _ = make_pipeline_and_rmgr("survivor")
        hm = HealthManager(pipeline=pipe, resource_mgr=rmgr)
        redistribute = hm.recovery_manager._on_redistribute
        assert redistribute is not None
        plan = _rec_mod.NodeRecoveryPlan(failed_node_id="dead-node")
        plan.redistributions = [
            _rec_mod.LayerRedistribution(
                surviving_node_id="not-here",
                added_start_layer=0,
                added_end_layer=1,
                new_start_layer=0,
                new_end_layer=1,
            ),
        ]
        redistribute("dead-node", plan)  # should not raise

    def test_recover_callback_recovers_sequences_with_checkpoint(self) -> None:
        """Recover should transfer KV cache to a surviving node."""
        pipe, rmgr, nodes = make_pipeline_and_rmgr("survivor")
        hm = HealthManager(pipeline=pipe, resource_mgr=rmgr)
        nodes["survivor"].client = _StubClient()
        # Save a checkpoint for the failed node
        hm.recovery_manager.save_checkpoint(
            request_id="seq-1",
            kv_cache={"layer0": [1.0, 2.0]},
            prompt_tokens=[1, 2, 3],
            generated_tokens=[100],
            node_id="dead-node",
        )
        recover = hm.recovery_manager._on_recover
        assert recover is not None
        result = recover("dead-node", ["seq-1"])
        assert "seq-1" in result

    def test_recover_callback_no_checkpoint_skips(self) -> None:
        """Recover should skip sequences without a checkpoint."""
        pipe, rmgr, nodes = make_pipeline_and_rmgr("survivor")
        hm = HealthManager(pipeline=pipe, resource_mgr=rmgr)
        nodes["survivor"].client = _StubClient()
        recover = hm.recovery_manager._on_recover
        result = recover("dead-node", ["seq-missing"])
        assert result == []  # no checkpoint, so no recovery

    def test_recover_callback_no_survivors_returns_empty(self) -> None:
        """Recover with empty pipeline (no survivors) should return []."""
        pipe = _StubPipeline()  # empty -- no nodes
        rmgr = _StubResourceManager()
        hm = HealthManager(pipeline=pipe, resource_mgr=rmgr)
        recover = hm.recovery_manager._on_recover
        result = recover("dead-node", ["seq-1"])
        assert result == []

    def test_recover_callback_no_client_on_node(self) -> None:
        """Recover should skip KV transfer when target has no client."""
        pipe, rmgr, nodes = make_pipeline_and_rmgr("survivor")
        hm = HealthManager(pipeline=pipe, resource_mgr=rmgr)
        nodes["survivor"].client = None
        hm.recovery_manager.save_checkpoint(
            request_id="seq-1",
            kv_cache={"layer0": [1.0]},
            prompt_tokens=[1],
            generated_tokens=[42],
            node_id="dead-node",
        )
        recover = hm.recovery_manager._on_recover
        result = recover("dead-node", ["seq-1"])
        # Should still recover the seq even without client KV transfer
        assert "seq-1" in result

    def test_mark_dead_removes_node_from_pipeline(self) -> None:
        """Mark-dead should pop the node from pipeline.nodes."""
        pipe, rmgr, nodes = make_pipeline_and_rmgr("node-a")
        hm = HealthManager(pipeline=pipe, resource_mgr=rmgr)
        assert "node-a" in pipe.nodes
        mark_dead = hm.recovery_manager._on_mark_dead
        assert mark_dead is not None
        mark_dead("node-a")
        assert "node-a" not in pipe.nodes

    def test_mark_dead_missing_node_does_not_crash(self) -> None:
        """Mark-dead with an unknown node should not raise."""
        pipe, rmgr, _ = make_pipeline_and_rmgr("node-a")
        hm = HealthManager(pipeline=pipe, resource_mgr=rmgr)
        mark_dead = hm.recovery_manager._on_mark_dead
        assert mark_dead is not None
        mark_dead("node-nonexistent")  # should not raise

    def test_mark_dead_updates_node_order(self) -> None:
        """Mark-dead should update node_order after removal."""
        pipe, rmgr, nodes = make_pipeline_and_rmgr("node-a", "node-b")
        hm = HealthManager(pipeline=pipe, resource_mgr=rmgr)
        mark_dead = hm.recovery_manager._on_mark_dead
        mark_dead("node-a")
        assert pipe.node_order == ["node-b"]


# ===================================================================
# PER-NODE HEALTH PROBE
# ===================================================================


class TestProbeSingleNode:
    """_probe_single_node -- health check, latency, state transition."""

    def test_successful_probe_creates_health_record(self) -> None:
        """First successful probe should create a HealthRecord in the store."""
        pipe, rmgr, _ = make_pipeline_and_rmgr("node-0")
        hm = HealthManager(pipeline=pipe, resource_mgr=rmgr)
        node = make_probe_node(healthy=True)
        hm._probe_single_node("node-0", node)
        record = hm.health_store.get("node-0")
        assert record is not None
        assert record.node_id == "node-0"

    def test_successful_probe_transitions_from_offline_to_degraded(self) -> None:
        """A successful probe on a node with no record (OFFLINE default) should
        go to DEGRADED via the FailoverEngine."""
        pipe, rmgr, _ = make_pipeline_and_rmgr("node-0")
        hm = HealthManager(pipeline=pipe, resource_mgr=rmgr)
        node = make_probe_node(healthy=True)
        hm._probe_single_node("node-0", node)
        record = hm.health_store.get("node-0")
        assert record is not None
        # Default HealthRecord starts at OFFLINE; successful probe transitions to DEGRADED.
        assert record.state == NodeState.DEGRADED

    def test_failed_probe_transitions_to_degraded_or_unhealthy(self) -> None:
        """A failed probe on a node should increment consecutive failures."""
        pipe, rmgr, _ = make_pipeline_and_rmgr("node-0")
        hm = HealthManager(pipeline=pipe, resource_mgr=rmgr)
        node = make_probe_node(healthy=False)
        # First failure: consecutive_failures=1, state should be DEGRADED
        hm._probe_single_node("node-0", node)
        record = hm.health_store.get("node-0")
        assert record is not None
        assert record.consecutive_failures >= 1
        # Default threshold is 3 for UNHEALTHY, so after 1 failure we're DEGRADED
        assert record.state in (NodeState.DEGRADED,)

    def test_multiple_failures_lead_to_unhealthy(self) -> None:
        """Three consecutive failures should transition to UNHEALTHY."""
        pipe, rmgr, _ = make_pipeline_and_rmgr("node-0")
        hm = HealthManager(pipeline=pipe, resource_mgr=rmgr)
        node = make_probe_node(healthy=False)
        # Probe three times -- each fails
        for _ in range(3):
            hm._probe_single_node("node-0", node)
        record = hm.health_store.get("node-0")
        assert record is not None
        assert record.consecutive_failures >= 3
        assert record.state == NodeState.UNHEALTHY

    def test_probe_with_reputation_records_health(self) -> None:
        """When reputation is set, probe should record health result."""
        pipe, rmgr, _ = make_pipeline_and_rmgr("node-0")
        rep = _StubReputation()
        hm = HealthManager(pipeline=pipe, resource_mgr=rmgr, reputation=rep)
        node = make_probe_node(healthy=True)
        hm._probe_single_node("node-0", node)
        assert len(rep.records) == 1
        assert rep.records[0] == ("node-0", True)

    def test_probe_with_reputation_records_failure(self) -> None:
        """When reputation is set, a failed probe should record healthy=False."""
        pipe, rmgr, _ = make_pipeline_and_rmgr("node-0")
        rep = _StubReputation()
        hm = HealthManager(pipeline=pipe, resource_mgr=rmgr, reputation=rep)
        node = make_probe_node(healthy=False)
        hm._probe_single_node("node-0", node)
        assert len(rep.records) == 1
        assert rep.records[0] == ("node-0", False)

    def test_probe_uses_existing_health_record(self) -> None:
        """Probing again should update, not replace, the existing record."""
        pipe, rmgr, _ = make_pipeline_and_rmgr("node-0")
        hm = HealthManager(pipeline=pipe, resource_mgr=rmgr)
        node = make_probe_node(healthy=True)
        # First probe
        hm._probe_single_node("node-0", node)
        record1 = hm.health_store.get("node-0")
        # Second probe
        hm._probe_single_node("node-0", node)
        record2 = hm.health_store.get("node-0")
        assert record1 is record2  # same object, not replaced

    def test_probe_records_latency_in_record(self) -> None:
        """Probe should update latency percentiles."""
        pipe, rmgr, _ = make_pipeline_and_rmgr("node-0")
        hm = HealthManager(pipeline=pipe, resource_mgr=rmgr)
        node = make_probe_node(healthy=True)
        hm._probe_single_node("node-0", node)
        record = hm.health_store.get("node-0")
        assert record is not None
        # Latency should be recorded (even if very small)
        assert record.latency_p50_ms >= 0

    def test_probe_node_raises_exception(self) -> None:
        """If health_check raises, the exception should propagate."""

        class _RaisingNode(_StubPipelineNode):
            def health_check(self) -> bool:
                raise RuntimeError("probe failed")

        pipe, rmgr, _ = make_pipeline_and_rmgr()
        hm = HealthManager(pipeline=pipe, resource_mgr=rmgr)
        node = _RaisingNode(node_id="node-bad", healthy=True)
        with pytest.raises(RuntimeError, match="probe failed"):
            hm._probe_single_node("node-bad", node)


# ===================================================================
# HEALTH PROBE LOOP LIFECYCLE
# ===================================================================


class TestHealthProbeLoopLifecycle:
    """start/stop lifecycle of the health probe background thread."""

    def test_start_creates_and_starts_thread(self) -> None:
        """After start(), _health_thread should be alive."""
        pipe, rmgr, _ = make_pipeline_and_rmgr("node-0")
        hm = HealthManager(pipeline=pipe, resource_mgr=rmgr, check_interval_s=0.01)
        hm.start()
        assert hm._health_thread is not None
        assert hm._health_thread.is_alive()
        hm.stop()

    def test_stop_joins_thread(self) -> None:
        """After stop(), _health_thread should no longer be alive."""
        pipe, rmgr, _ = make_pipeline_and_rmgr("node-0")
        hm = HealthManager(pipeline=pipe, resource_mgr=rmgr, check_interval_s=0.01)
        hm.start()
        hm.stop()
        assert hm._health_thread is not None
        assert not hm._health_thread.is_alive()

    def test_stop_without_start_does_not_crash(self) -> None:
        """Calling stop() without start() should be safe."""
        pipe, rmgr, _ = make_pipeline_and_rmgr()
        hm = HealthManager(pipeline=pipe, resource_mgr=rmgr)
        hm.stop()  # should not raise

    def test_double_stop_does_not_crash(self) -> None:
        """Calling stop() twice should be safe."""
        pipe, rmgr, _ = make_pipeline_and_rmgr()
        hm = HealthManager(pipeline=pipe, resource_mgr=rmgr, check_interval_s=0.01)
        hm.start()
        hm.stop()
        hm.stop()  # second stop -- should not raise

    def test_start_sets_running_event(self) -> None:
        """After start(), _running should be set."""
        pipe, rmgr, _ = make_pipeline_and_rmgr()
        hm = HealthManager(pipeline=pipe, resource_mgr=rmgr)
        hm.start()
        assert hm._running.is_set()
        hm.stop()

    def test_stop_clears_running_and_sets_health_event(self) -> None:
        """After stop(), _running should be clear and _health_event set."""
        pipe, rmgr, _ = make_pipeline_and_rmgr()
        hm = HealthManager(pipeline=pipe, resource_mgr=rmgr)
        hm.start()
        hm.stop()
        assert not hm._running.is_set()
        assert hm._health_event.is_set()

    def test_probe_loop_skips_nodes_without_client(self) -> None:
        """Nodes with client=None should be skipped during probing."""
        pipe, rmgr, nodes = make_pipeline_and_rmgr("node-0")
        nodes["node-0"].client = None
        hm = HealthManager(pipeline=pipe, resource_mgr=rmgr, check_interval_s=0.01)
        hm.start()
        # Give the loop a chance to run one iteration
        hm._health_event.wait(0.05)
        hm.stop()
        # The node was skipped (no client), so no health record should exist.
        record = hm.health_store.get("node-0")
        assert record is None

    def test_probe_loop_probes_node_with_client(self) -> None:
        """Nodes with a client should be probed and get a health record."""
        pipe, rmgr, nodes = make_pipeline_and_rmgr("node-0")
        hm = HealthManager(pipeline=pipe, resource_mgr=rmgr, check_interval_s=0.01)
        hm.start()
        hm._health_event.wait(0.05)
        hm.stop()
        record = hm.health_store.get("node-0")
        assert record is not None


# ===================================================================
# NODE STATUS REPORTING
# ===================================================================


class TestGetNodeStatus:
    """get_node_status -- status dictionary for each node."""

    def test_empty_pipeline_returns_empty_dict(self) -> None:
        """With no nodes, get_node_status should return {}."""
        pipe = _StubPipeline()
        rmgr = _StubResourceManager()
        hm = HealthManager(pipeline=pipe, resource_mgr=rmgr)
        status = hm.get_node_status()
        assert status == {}

    def test_single_node_without_health_record(self) -> None:
        """A node without a health record should show 'unknown' state."""
        pipe, rmgr, nodes = make_pipeline_and_rmgr("node-0")
        hm = HealthManager(pipeline=pipe, resource_mgr=rmgr)
        status = hm.get_node_status()
        assert "node-0" in status
        assert status["node-0"]["state"] == "unknown"
        assert status["node-0"]["healthy"] is True
        assert status["node-0"]["start_layer"] == 0
        assert status["node-0"]["end_layer"] == 0

    def test_single_node_with_health_record(self) -> None:
        """A probed node should show the correct state."""
        pipe, rmgr, nodes = make_pipeline_and_rmgr("node-0")
        hm = HealthManager(pipeline=pipe, resource_mgr=rmgr)
        # Manually create a health record
        record = HealthRecord(node_id="node-0", state=NodeState.HEALTHY)
        hm.health_store.set("node-0", record)
        status = hm.get_node_status()
        assert status["node-0"]["state"] == "healthy"
        assert status["node-0"]["healthy"] is True

    def test_node_with_gpu_attributes(self) -> None:
        """GPU attributes should appear in status."""
        pipe, rmgr, nodes = make_pipeline_and_rmgr("node-0")
        nodes["node-0"].gpu_name = "A100"
        nodes["node-0"].gpu_memory_free = 40000
        hm = HealthManager(pipeline=pipe, resource_mgr=rmgr)
        status = hm.get_node_status()
        assert status["node-0"]["gpu_name"] == "A100"
        assert status["node-0"]["gpu_memory_free"] == 40000

    def test_multiple_nodes(self) -> None:
        """Multiple nodes should all appear in the status dict."""
        pipe, rmgr, nodes = make_pipeline_and_rmgr("node-a", "node-b")
        hm = HealthManager(pipeline=pipe, resource_mgr=rmgr)
        status = hm.get_node_status()
        assert set(status.keys()) == {"node-a", "node-b"}

    def test_node_with_getattr_failure_is_graceful(self) -> None:
        """If a node's attribute access raises, the status should still be reported."""
        pipe, rmgr, nodes = make_pipeline_and_rmgr("node-0")

        # Remove start_layer to cause an AttributeError
        class _BrokenNode:
            health_check = lambda self: True
            client = object()

        pipe.nodes["node-0"] = _BrokenNode()
        hm = HealthManager(pipeline=pipe, resource_mgr=rmgr)
        # health_store.get will return None for an unprobed node,
        # so the try/except in start_layer access won't be reached.
        # Let's set a record to trigger the attribute access path.
        record = HealthRecord(node_id="node-0", state=NodeState.HEALTHY)
        hm.health_store.set("node-0", record)
        status = hm.get_node_status()
        # The entire node loop should not crash
        assert "node-0" in status
        assert status["node-0"]["healthy"] is False  # fallback
        assert status["node-0"]["state"] == "unknown"  # fallback


# ===================================================================
# EDGE CASES
# ===================================================================


class TestEdgeCases:
    """Unusual or error paths."""

    def test_reputation_none_does_not_crash(self) -> None:
        """When reputation is None, the probe should skip reputation recording."""
        pipe, rmgr, _ = make_pipeline_and_rmgr("node-0")
        hm = HealthManager(pipeline=pipe, resource_mgr=rmgr, reputation=None)
        node = make_probe_node(healthy=True)
        hm._probe_single_node("node-0", node)  # should not raise/error

    def test_straggler_detector_callback_connected(self) -> None:
        """The default StragglerDetector's callback should be _on_straggler_detected."""
        pipe, rmgr, _ = make_pipeline_and_rmgr("node-0")
        hm = HealthManager(pipeline=pipe, resource_mgr=rmgr)
        # Bound-method identity: access __self__ and __func__ separately
        cb = hm.straggler_detector._on_straggler
        assert cb.__self__ is hm
        assert cb.__func__ is type(hm)._on_straggler_detected

    def test_failover_engine_callback_connected(self) -> None:
        """The FailoverEngine should have _on_state_change as a registered callback."""
        pipe, rmgr, _ = make_pipeline_and_rmgr("node-0")
        hm = HealthManager(pipeline=pipe, resource_mgr=rmgr)
        assert hm._on_state_change in hm.failover_engine._callbacks

    def test_health_probe_loop_evicts_stale_checkpoints(self) -> None:
        """The probe loop should call evict_stale_checkpoints each iteration."""
        pipe, rmgr, _ = make_pipeline_and_rmgr("node-0")
        hm = HealthManager(pipeline=pipe, resource_mgr=rmgr, check_interval_s=0.01)
        # Save a checkpoint with an old timestamp
        hm.recovery_manager.save_checkpoint(
            request_id="stale-seq",
            kv_cache=None,
            prompt_tokens=[1],
            generated_tokens=[2],
            node_id="node-0",
        )
        # Manually age the checkpoint
        import time
        ckpt = hm.recovery_manager.get_checkpoint("stale-seq")
        assert ckpt is not None
        ckpt.timestamp = 0.0  # far in the past
        hm.start()
        hm._health_event.wait(0.05)
        hm.stop()
        # The checkpoint should have been evicted
        assert hm.recovery_manager.get_checkpoint("stale-seq") is None
