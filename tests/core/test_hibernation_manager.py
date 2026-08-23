"""Tests for HibernationManager — idle node hibernation.

Covers:
- HibernationNode and HibernationDecision dataclasses
- HibernationManager: construction, node registration, request tracking,
  idle detection, hibernation, wake, force operations, stats
- Edge cases: unknown nodes, double operations, min_active enforcement
"""

from __future__ import annotations

import pytest

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_hm = load_module("distllm/core/hibernation_manager.py")
HibernationManager = _hm.HibernationManager
HibernationNode = _hm.HibernationNode
HibernationDecision = _hm.HibernationDecision
NodePowerState = _hm.NodePowerState


# ── Dataclass tests ───────────────────────────────────────────────────────────


class TestHibernationNode:
    def test_default_values(self):
        node = HibernationNode(node_id="node-1")
        assert node.node_id == "node-1"
        assert node.power_state == NodePowerState.ACTIVE
        assert node.wake_count == 0
        assert node.cost_per_hour == 0.0
        assert node.idle_since == 0.0
        assert node.hibernated_at == 0.0


class TestHibernationDecision:
    def test_dataclass_fields(self):
        d = HibernationDecision(
            node_id="node-1",
            action="hibernate",
            reason="idle for 5 minutes",
            estimated_savings_per_hour=10.0,
        )
        assert d.node_id == "node-1"
        assert d.action == "hibernate"
        assert d.estimated_savings_per_hour == 10.0


# ── HibernationManager — Construction ─────────────────────────────────────────


class TestHibernationManagerConstruction:
    def test_default_values(self):
        mgr = HibernationManager()
        assert mgr._idle_threshold == 300.0
        assert mgr._check_interval == 30.0
        assert mgr._min_active == 1
        assert mgr._max_hibernate == 1
        assert mgr._wake_timeout == 60.0
        assert mgr._on_hibernate is None
        assert mgr._on_wake is None
        assert mgr._running is False

    def test_custom_values(self):
        cb = lambda n: None
        mgr = HibernationManager(
            idle_threshold_s=60.0,
            check_interval_s=10.0,
            min_active_nodes=2,
            max_hibernate_per_cycle=3,
            wake_timeout_s=30.0,
            on_hibernate=cb,
            on_wake=cb,
        )
        assert mgr._idle_threshold == 60.0
        assert mgr._min_active == 2
        assert mgr._max_hibernate == 3
        assert mgr._on_hibernate is cb
        assert mgr._on_wake is cb


# ── HibernationManager — Node Management ──────────────────────────────────────


class TestHibernationManagerNodeLifecycle:
    def test_register_node(self):
        mgr = HibernationManager()
        mgr.register_node("node-1", cost_per_hour=5.0)
        assert "node-1" in mgr._nodes
        assert mgr._nodes["node-1"].cost_per_hour == 5.0

    def test_unregister_node(self):
        mgr = HibernationManager()
        mgr.register_node("node-1")
        mgr.unregister_node("node-1")
        assert "node-1" not in mgr._nodes

    def test_unregister_unknown_node_safe(self):
        mgr = HibernationManager()
        mgr.unregister_node("nonexistent")  # Should not raise

    def test_register_multiple_nodes(self):
        mgr = HibernationManager()
        mgr.register_node("n1")
        mgr.register_node("n2")
        mgr.register_node("n3")
        assert len(mgr._nodes) == 3


# ── HibernationManager — Request Tracking ─────────────────────────────────────


class TestHibernationManagerRequestTracking:
    def test_record_request_unknown_node_safe(self):
        mgr = HibernationManager()
        mgr.record_request("nonexistent")  # Should not raise

    def test_record_request_resets_idle(self):
        mgr = HibernationManager()
        mgr.register_node("node-1")
        mgr._nodes["node-1"].idle_since = 100.0
        mgr.record_request("node-1")
        assert mgr._nodes["node-1"].idle_since == 0.0

    def test_record_request_resets_last_request_time(self):
        mgr = HibernationManager()
        mgr.register_node("node-1")
        mgr.record_request("node-1")
        assert mgr._nodes["node-1"].last_request_time > 0

    def test_record_request_wakes_hibernating_node(self):
        mgr = HibernationManager()
        mgr.register_node("node-1")
        # Manually hibernate
        node = mgr._nodes["node-1"]
        node.power_state = NodePowerState.HIBERNATING
        mgr.record_request("node-1")
        assert mgr._nodes["node-1"].power_state == NodePowerState.ACTIVE


# ── HibernationManager — Hibernation Logic ────────────────────────────────────


class TestHibernationManagerHibernation:
    def test_force_hibernate_unknown_node(self):
        mgr = HibernationManager()
        assert mgr.force_hibernate("nonexistent") is False

    def test_force_hibernate_active_node(self):
        mgr = HibernationManager()
        mgr.register_node("node-1")
        assert mgr.force_hibernate("node-1") is True
        assert mgr._nodes["node-1"].power_state == NodePowerState.HIBERNATING
        assert mgr._stats["hibernations"] == 1

    def test_force_hibernate_already_hibernating(self):
        mgr = HibernationManager()
        mgr.register_node("node-1")
        mgr._nodes["node-1"].power_state = NodePowerState.HIBERNATING
        assert mgr.force_hibernate("node-1") is False  # Not ACTIVE

    def test_force_wake_unknown_node(self):
        mgr = HibernationManager()
        assert mgr.force_wake("nonexistent") is False

    def test_force_wake_hibernating_node(self):
        mgr = HibernationManager()
        mgr.register_node("node-1", cost_per_hour=10.0)
        mgr.force_hibernate("node-1")
        assert mgr.force_wake("node-1") is True
        assert mgr._nodes["node-1"].power_state == NodePowerState.ACTIVE
        assert mgr._stats["wakes"] == 1
        # Cost savings should be tracked
        assert mgr._stats["idle_hours_saved"] >= 0
        assert mgr._stats["cost_saved"] >= 0

    def test_force_wake_active_node(self):
        mgr = HibernationManager()
        mgr.register_node("node-1")
        assert mgr.force_wake("node-1") is False  # Not HIBERNATING

    def test_hibernate_callback_invoked(self):
        calls = []

        def cb(node_id):
            calls.append(node_id)

        mgr = HibernationManager(on_hibernate=cb)
        mgr.register_node("node-1")
        mgr.force_hibernate("node-1")
        assert len(calls) == 1
        assert calls[0] == "node-1"

    def test_hibernate_callback_exception_handled(self):
        def cb(node_id):
            raise ValueError("boom")

        mgr = HibernationManager(on_hibernate=cb)
        mgr.register_node("node-1")
        mgr.force_hibernate("node-1")  # Should not raise

    def test_wake_callback_invoked(self):
        calls = []

        def cb(node_id):
            calls.append(node_id)

        mgr = HibernationManager(on_wake=cb)
        mgr.register_node("node-1")
        mgr.force_hibernate("node-1")
        mgr.force_wake("node-1")
        assert len(calls) == 1
        assert calls[0] == "node-1"

    def test_wake_callback_exception_handled(self):
        def cb(node_id):
            raise ValueError("boom")

        mgr = HibernationManager(on_wake=cb)
        mgr.register_node("node-1")
        mgr.force_hibernate("node-1")
        mgr.force_wake("node-1")  # Should not raise


# ── HibernationManager — Idle Detection ───────────────────────────────────────


class TestHibernationManagerIdleDetection:
    def test_check_idle_nodes_marks_idle(self):
        mgr = HibernationManager(
            idle_threshold_s=0.0,  # Immediate idle
        )
        mgr.register_node("node-1")
        mgr._check_idle_nodes()

        node = mgr._nodes["node-1"]
        # After idle_threshold_s=0, should be marked IDLE (not yet hibernated,
        # because ACTIVE->IDLE happens first, then IDLE->HIBERNATING)
        assert node.power_state == NodePowerState.IDLE

    def test_min_active_nodes_enforced(self):
        """With 1 node registered and min_active=1, it should not hibernate."""
        mgr = HibernationManager(
            idle_threshold_s=0.0,
            min_active_nodes=1,
        )
        mgr.register_node("node-1")
        # Manually set to IDLE
        mgr._nodes["node-1"].power_state = NodePowerState.IDLE
        mgr._check_idle_nodes()
        # Should NOT hibernate because we'd go below min_active
        assert mgr._nodes["node-1"].power_state != NodePowerState.HIBERNATING

    def test_hibernates_when_spares_available(self):
        """With 2 nodes and min_active=1, one can hibernate."""
        mgr = HibernationManager(
            idle_threshold_s=0.0,
            min_active_nodes=1,
        )
        mgr.register_node("node-1")
        mgr.register_node("node-2")
        mgr._check_idle_nodes()
        # At least one node should be IDLE or HIBERNATING
        states = [n.power_state for n in mgr._nodes.values()]
        idle_or_hibernating = [s for s in states if s in (
            NodePowerState.IDLE,
            NodePowerState.HIBERNATING,
        )]
        assert len(idle_or_hibernating) >= 1


# ── HibernationManager — Start/Stop ───────────────────────────────────────────


class TestHibernationManagerStartStop:
    def test_start_and_stop(self):
        mgr = HibernationManager(check_interval_s=0.01)
        mgr.start()
        assert mgr._running is True
        mgr.stop()
        assert mgr._running is False

    def test_start_twice_is_idempotent(self):
        mgr = HibernationManager(check_interval_s=0.01)
        mgr.start()
        thread_id = id(mgr._thread)
        mgr.start()  # Should be no-op
        assert id(mgr._thread) == thread_id
        mgr.stop()

    def test_stop_wakes_all_hibernating(self):
        mgr = HibernationManager()
        mgr.register_node("node-1")
        mgr.register_node("node-2")
        mgr._nodes["node-1"].power_state = NodePowerState.HIBERNATING
        mgr._nodes["node-2"].power_state = NodePowerState.HIBERNATING
        mgr.stop()  # Should wake both
        assert mgr._nodes["node-1"].power_state == NodePowerState.ACTIVE
        assert mgr._nodes["node-2"].power_state == NodePowerState.ACTIVE


# ── HibernationManager — Stats ────────────────────────────────────────────────


class TestHibernationManagerStats:
    def test_stats_initial(self):
        mgr = HibernationManager()
        s = mgr.stats()
        assert s["hibernations"] == 0
        assert s["wakes"] == 0
        assert s["total_nodes"] == 0
        assert s["node_states"] == {}

    def test_stats_after_operations(self):
        mgr = HibernationManager()
        mgr.register_node("node-1")
        mgr.register_node("node-2")
        mgr.force_hibernate("node-1")
        s = mgr.stats()
        assert s["total_nodes"] == 2
        assert s["hibernations"] == 1
        assert "hibernating" in s["node_states"]
