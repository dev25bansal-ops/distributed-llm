"""Comprehensive tests for NodeRecoveryManager.

Tests layer redistribution algorithms (basic + capacity-aware),
checkpoint CRUD with thread safety, disk persistence roundtrip,
concurrent failure handling, dry-run mode, and Prometheus metrics
integration.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time

import pytest
import torch

from distllm.dist.recovery import (
    NodeRecoveryManager,
    NodeRecoveryPlan,
    LayerRedistribution,
    RecoveryState,
    SequenceCheckpoint,
    compute_redistributions,
    compute_redistributions_capacity_aware,
    _tensor_size_bytes,
)


# ── _tensor_size_bytes ────────────────────────────────────────────────


class TestTensorSizeBytes:
    def test_tensor(self):
        t = torch.zeros(100, dtype=torch.float32)
        assert _tensor_size_bytes(t) == 100 * 4

    def test_parameter(self):
        p = torch.nn.Parameter(torch.zeros(50, dtype=torch.float16))
        assert _tensor_size_bytes(p) == 50 * 2

    def test_dict(self):
        d = {"a": torch.zeros(10, dtype=torch.float32), "b": torch.zeros(5, dtype=torch.float32)}
        assert _tensor_size_bytes(d) == 15 * 4

    def test_list(self):
        lst = [torch.zeros(8, dtype=torch.float32)]
        assert _tensor_size_bytes(lst) == 8 * 4

    def test_nested(self):
        nested = {"layers": [{"k": torch.zeros(4, dtype=torch.float32)}]}
        assert _tensor_size_bytes(nested) == 4 * 4

    def test_none(self):
        assert _tensor_size_bytes(None) == 0

    def test_scalar(self):
        assert _tensor_size_bytes(42) == 0


# ── compute_redistributions ───────────────────────────────────────────


class TestComputeRedistributions:
    def test_empty_survivors(self):
        result = compute_redistributions(0, 3, {})
        assert result == []

    def test_single_survivor(self):
        result = compute_redistributions(4, 7, {"n1": (0, 3)})
        assert len(result) == 1
        assert result[0].surviving_node_id == "n1"
        assert result[0].added_start_layer == 4
        assert result[0].added_end_layer == 7

    def test_two_survivors_even_split(self):
        result = compute_redistributions(4, 7, {"n1": (0, 3), "n2": (8, 11)})
        assert len(result) == 2
        # 4 layers across 2 nodes = 2 each
        assert result[0].added_end_layer - result[0].added_start_layer + 1 == 2
        assert result[1].added_end_layer - result[1].added_start_layer + 1 == 2

    def test_three_survivors_uneven(self):
        """Head-edge failure (0,3) with survivors to the right.

        C3 fix: only the ADJACENT survivor (n1) absorbs the orphan;
        distant survivors are untouched.  (Pre-fix the orphan was split
        across all three, producing overlapping ranges like n1=(0,7)
        vs n2=(1,11).)
        """
        result = compute_redistributions(
            0, 3,
            {"n1": (4, 7), "n2": (8, 11), "n3": (12, 15)},
        )
        # Only the adjacent neighbor receives a redistribution.
        assert [r.surviving_node_id for r in result] == ["n1"]
        r = result[0]
        assert (r.added_start_layer, r.added_end_layer) == (0, 3)
        assert (r.new_start_layer, r.new_end_layer) == (0, 7)
        assert r.requires_weight_load is True

    def test_dead_count_zero(self):
        result = compute_redistributions(5, 3, {"n1": (0, 3)})
        assert result == []


# ── compute_redistributions_capacity_aware ────────────────────────────


class TestCapacityAware:
    def test_fallback_when_no_memory_info(self):
        result = compute_redistributions_capacity_aware(
            4, 7, {"n1": (0, 3)}, survivor_memory_gb=None,
        )
        assert len(result) == 1

    def test_proportional_allocation(self):
        """Capacity-aware: the orphan goes to the adjacent ELIGIBLE survivor.

        C3 fix: only adjacency slots may absorb (a farther-out absorber
        would swallow intermediate survivors' layers).  n2 is adjacent to
        the failed range and has 3x free memory, but n1 sits between it
        and the orphan, so n2 can never absorb without overlapping n1.
        """
        result = compute_redistributions_capacity_aware(
            0, 5,  # 6 layers to redistribute
            {"n1": (6, 7), "n2": (8, 9)},
            survivor_memory_gb={"n1": 10.0, "n2": 30.0},
            min_memory_per_layer_gb=1.0,
        )
        # Only the adjacent eligible survivor absorbs.
        assert [r.surviving_node_id for r in result] == ["n1"]
        r = result[0]
        assert (r.added_start_layer, r.added_end_layer) == (0, 5)
        assert (r.new_start_layer, r.new_end_layer) == (0, 7)

    def test_skip_insufficient_memory(self):
        """Node with no free memory is skipped."""
        result = compute_redistributions_capacity_aware(
            0, 3,
            {"n1": (4, 7)},
            survivor_memory_gb={"n1": 0.0},
            min_memory_per_layer_gb=1.0,
        )
        # Falls back to even distribution
        assert len(result) == 1

    def test_empty_survivors(self):
        result = compute_redistributions_capacity_aware(0, 3, {}, {"n1": 10.0})
        assert result == []


# ── SequenceCheckpoint ────────────────────────────────────────────────


class TestSequenceCheckpoint:
    def test_size_bytes_tensor(self):
        ckpt = SequenceCheckpoint(
            request_id="r1", kv_cache=torch.zeros(100, dtype=torch.float32),
            prompt_tokens=[], generated_tokens=[], node_id="n1",
        )
        assert ckpt.size_bytes() == 100 * 4

    def test_size_bytes_dict(self):
        ckpt = SequenceCheckpoint(
            request_id="r1",
            kv_cache={"key": torch.zeros(50, dtype=torch.float16)},
            prompt_tokens=[], generated_tokens=[], node_id="n1",
        )
        assert ckpt.size_bytes() == 50 * 2

    def test_size_bytes_empty(self):
        ckpt = SequenceCheckpoint(
            request_id="r1", kv_cache=None,
            prompt_tokens=[], generated_tokens=[], node_id="n1",
        )
        assert ckpt.size_bytes() == 0


# ── NodeRecoveryManager ───────────────────────────────────────────────


@pytest.fixture
def mgr():
    return NodeRecoveryManager(node_id="test-coord", persist_path=None)


class TestCheckpointCRUD:
    def test_save_checkpoint(self, mgr):
        mgr.save_checkpoint("r1", torch.zeros(10), [1, 2], [3], "n1")
        ckpt = mgr.get_checkpoint("r1")
        assert ckpt is not None
        assert ckpt.request_id == "r1"
        assert ckpt.node_id == "n1"

    def test_save_checkpoint_increments_count(self, mgr):
        initial = mgr.get_metrics()["checkpoint_count"]
        mgr.save_checkpoint("r1", None, [1], [2], "n1")
        assert mgr.get_metrics()["checkpoint_count"] == initial + 1

    def test_drop_checkpoint(self, mgr):
        mgr.save_checkpoint("r1", None, [1], [2], "n1")
        mgr.drop_checkpoint("r1")
        assert mgr.get_checkpoint("r1") is None

    def test_get_checkpoints_for_node(self, mgr):
        mgr.save_checkpoint("r1", None, [1], [2], "n1")
        mgr.save_checkpoint("r2", None, [3], [4], "n1")
        mgr.save_checkpoint("r3", None, [5], [6], "n2")
        checkpoints = mgr.get_checkpoints_for_node("n1")
        assert len(checkpoints) == 2

    def test_evict_stale_checkpoints(self, mgr):
        mgr.save_checkpoint("r1", None, [1], [2], "n1")
        # No staleness expected: less than 300s TTL
        evicted = mgr.evict_stale_checkpoints()
        assert evicted == 0


class TestNodeState:
    def test_mark_dead(self, mgr):
        assert not mgr.is_dead("n1")
        mgr._dead_nodes.add("n1")
        assert mgr.is_dead("n1")

    def test_mark_draining(self, mgr):
        assert not mgr.is_draining("n1")
        mgr._draining.add("n1")
        assert mgr.is_draining("n1")

    def test_mark_alive(self, mgr):
        mgr._dead_nodes.add("n1")
        mgr.mark_alive("n1")
        assert not mgr.is_dead("n1")

    def test_draining_nodes_property(self, mgr):
        mgr._draining.add("n1")
        assert "n1" in mgr.draining_nodes


class TestDryRun:
    def test_dry_run_does_not_mark_dead(self, mgr):
        """dry_run_recovery should not fire the mark-dead callback."""
        was_called = False

        def mark_dead(node_id: str) -> None:
            nonlocal was_called
            was_called = True

        mgr.set_mark_dead_callback(mark_dead)
        mgr.dry_run_recovery("n1")
        assert not was_called

    def test_dry_run_does_not_drain(self, mgr):
        """dry_run_recovery should not fire the drain callback."""
        was_called = False

        def drain(node_id: str) -> None:
            nonlocal was_called
            was_called = True

        mgr.set_drain_callback(drain)
        mgr.dry_run_recovery("n1")
        assert not was_called

    def test_dry_run_returns_plan(self, mgr):
        plan = mgr.dry_run_recovery("n1")
        assert isinstance(plan, NodeRecoveryPlan)
        assert plan.failed_node_id == "n1"

    def test_dry_run_does_not_remove_checkpoints(self, mgr):
        mgr.save_checkpoint("r1", None, [1], [2], "n1")
        mgr.dry_run_recovery("n1")
        assert mgr.get_checkpoint("r1") is not None  # should still exist


class TestMetrics:
    def test_get_metrics(self, mgr):
        m = mgr.get_metrics()
        assert "recoveries" in m
        assert "failed_nodes" in m
        assert "sequences_recovered" in m
        assert "sequences_lost" in m
        assert "draining_nodes" in m
        assert "dead_nodes" in m
        assert "active_checkpoints" in m

    def test_recovery_history(self, mgr):
        # Should start empty
        assert mgr.get_recovery_history() == []


class TestConcurrentAccess:
    def test_save_checkpoint_concurrent(self, mgr):
        """Multiple threads should be able to save checkpoints safely."""
        errors = []

        def _save(rid: str):
            try:
                mgr.save_checkpoint(rid, torch.zeros(4), [1, 2], [3], "n1")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_save, args=(f"r{i}",)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_concurrent_save_and_drop(self, mgr):
        """Concurrent save and drop should not cause deadlocks."""

        def _save():
            for i in range(50):
                mgr.save_checkpoint(f"r{i}", None, [], [], "n1")

        def _drop():
            for i in range(50):
                mgr.drop_checkpoint(f"r{i}")

        t1 = threading.Thread(target=_save)
        t2 = threading.Thread(target=_drop)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)
        # Should not deadlock


# ── Disk persistence ──────────────────────────────────────────────────


class TestDiskPersistence:
    def test_save_and_load(self, mgr):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            mgr.save_checkpoint("r1", torch.zeros(10), [1, 2], [3], "n1")
            mgr.save_checkpoint("r2", torch.ones(10), [4, 5], [6], "n2")
            assert mgr.save_to_disk(path=path)

            # Create a new manager and load from disk
            mgr2 = NodeRecoveryManager(node_id="test-coord", persist_path=None)
            assert mgr2.load_from_disk(path=path)
            assert mgr2.get_checkpoint("r1") is not None
            assert mgr2.get_checkpoint("r2") is not None
        finally:
            os.unlink(path)
            # Clean up companion files
            for ext in [".kv.pt"]:
                companion = path + ext
                if os.path.exists(companion):
                    os.unlink(companion)

    def test_save_no_path_returns_false(self, mgr):
        assert not mgr.save_to_disk(path=None)

    def test_load_nonexistent_path(self, mgr):
        assert not mgr.load_from_disk(path="/nonexistent/checkpoints.json")

    def test_save_and_load_roundtrip_metrics(self, mgr):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            mgr.save_checkpoint("r1", None, [1], [2], "n1")
            mgr.save_to_disk(path=path)
            prev_count = mgr.get_metrics()["checkpoint_count"]

            mgr2 = NodeRecoveryManager()
            mgr2.load_from_disk(path=path)
            assert mgr2.get_metrics()["checkpoint_count"] >= 1
        finally:
            os.unlink(path)
            for ext in [".kv.pt"]:
                companion = path + ext
                if os.path.exists(companion):
                    os.unlink(companion)
