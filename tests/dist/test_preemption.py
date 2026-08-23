"""Tests for distllm.dist.preemption -- SLA-aware preemption with checkpoint/resume.

Uses only real objects (zero mocks). Exercises the public API surface
(GPUMemoryMonitor, SLATracker, CheckpointState, PreemptionPolicy).
"""

from __future__ import annotations

import time

import pytest
import torch

from distllm.dist.preemption import (
    CheckpointState,
    GPUMemoryMonitor,
    PreemptionPolicy,
    SLATracker,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeSequence:
    """Minimal sequence-like object for checkpoint creation tests."""

    def __init__(
        self,
        prompt_tokens: list[int] | None = None,
        generated_tokens: list[int] | None = None,
        priority: int = 0,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
    ) -> None:
        self.prompt_tokens = prompt_tokens or [1, 2, 3]
        self.generated_tokens = generated_tokens or [4, 5]
        self.priority = priority
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k


def _make_kv(seq_len: int = 1, n_heads: int = 1, dim: int = 8) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Create a tiny KV-cache pair for testing (float32)."""
    return [(torch.zeros(seq_len, n_heads, dim), torch.zeros(seq_len, n_heads, dim))]


# ---------------------------------------------------------------------------
# TestGPUMemoryMonitor
# ---------------------------------------------------------------------------


class TestGPUMemoryMonitor:
    """Construction, utilization queries, and preemption signal."""

    def test_default_construction(self) -> None:
        mon = GPUMemoryMonitor()
        assert mon.device == 0
        assert mon.warn_threshold == 0.85
        assert mon.preempt_threshold == 0.92
        assert mon._history == []

    def test_custom_construction(self) -> None:
        mon = GPUMemoryMonitor(device=1, warn_threshold=0.90, preempt_threshold=0.95)
        assert mon.device == 1
        assert mon.warn_threshold == 0.90
        assert mon.preempt_threshold == 0.95

    def test_get_utilization_returns_bound_float(self) -> None:
        mon = GPUMemoryMonitor()
        util = mon.get_utilization()
        assert isinstance(util, float)
        assert 0.0 <= util <= 1.0

    def test_get_utilization_appends_to_history(self) -> None:
        mon = GPUMemoryMonitor()
        mon.get_utilization()
        assert len(mon._history) == 1

    def test_get_smoothed_utilization_empty_history_falls_back(self) -> None:
        mon = GPUMemoryMonitor()
        smoothed = mon.get_smoothed_utilization()
        assert isinstance(smoothed, float)

    def test_get_smoothed_utilization_computes_ema(self) -> None:
        mon = GPUMemoryMonitor()
        mon._history = [0.1, 0.2, 0.3]
        # alpha = 0.3
        # ema(0.1) = 0.1
        # ema(0.1, 0.2) = 0.3*0.2 + 0.7*0.1 = 0.06 + 0.07 = 0.13
        # ema(0.1, 0.2, 0.3) = 0.3*0.3 + 0.7*0.13 = 0.09 + 0.091 = 0.181
        assert mon.get_smoothed_utilization() == pytest.approx(0.181, abs=1e-3)

    def test_should_preempt_below_threshold(self) -> None:
        mon = GPUMemoryMonitor(preempt_threshold=0.99)
        mon._history = [0.1]
        assert mon.should_preempt() is False

    def test_should_preempt_above_threshold(self) -> None:
        mon = GPUMemoryMonitor(preempt_threshold=0.01)
        mon._history = [0.5]
        assert mon.should_preempt() is True

    def test_history_capped_at_ten(self) -> None:
        mon = GPUMemoryMonitor()
        mon._history = [0.1] * 15
        mon.get_utilization()
        assert len(mon._history) <= 10


# ---------------------------------------------------------------------------
# TestSLATracker
# ---------------------------------------------------------------------------


class TestSLATracker:
    """Request tracking, SLA deadline checks, violation accounting."""

    def test_default_construction(self) -> None:
        tracker = SLATracker()
        assert tracker.max_violations == 3
        assert tracker.sla_deadline_ms == 5000.0
        assert tracker.total_violations == 0

    def test_custom_construction(self) -> None:
        tracker = SLATracker(max_violations=5, sla_deadline_ms=1000.0)
        assert tracker.max_violations == 5
        assert tracker.sla_deadline_ms == 1000.0

    def test_start_request_records_timestamp(self) -> None:
        tracker = SLATracker()
        tracker.start_request("req-1")
        assert "req-1" in tracker._request_start
        assert isinstance(tracker._request_start["req-1"], float)

    def test_check_sla_unknown_request_returns_true(self) -> None:
        tracker = SLATracker()
        assert tracker.check_sla("unknown") is True

    def test_check_sla_within_deadline_returns_true(self) -> None:
        tracker = SLATracker(sla_deadline_ms=1e9)
        tracker.start_request("req-1")
        assert tracker.check_sla("req-1") is True

    def test_check_sla_exceeds_deadline_violates(self) -> None:
        tracker = SLATracker(sla_deadline_ms=0.0)
        tracker.start_request("req-1")
        assert tracker.check_sla("req-1") is False

    def test_check_sla_increments_total_violations(self) -> None:
        tracker = SLATracker(sla_deadline_ms=0.0)
        tracker.start_request("req-1")
        tracker.check_sla("req-1")
        assert tracker.total_violations == 1

    def test_multiple_checks_accumulate_violations(self) -> None:
        tracker = SLATracker(sla_deadline_ms=0.0)
        tracker.start_request("req-1")
        tracker.check_sla("req-1")
        tracker.check_sla("req-1")
        assert tracker.total_violations == 2

    def test_get_violation_count_tracks_per_request(self) -> None:
        tracker = SLATracker(sla_deadline_ms=0.0)
        tracker.start_request("req-1")
        tracker.check_sla("req-1")
        assert tracker.get_violation_count("req-1") == 1

    def test_get_violation_count_unknown_returns_zero(self) -> None:
        tracker = SLATracker()
        assert tracker.get_violation_count("nonexistent") == 0

    def test_complete_request_cleans_up_state(self) -> None:
        tracker = SLATracker(sla_deadline_ms=0.0)
        tracker.start_request("req-1")
        tracker.check_sla("req-1")
        tracker.complete_request("req-1")
        assert "req-1" not in tracker._request_start
        assert "req-1" not in tracker._violations

    def test_complete_request_unknown_does_not_raise(self) -> None:
        tracker = SLATracker()
        tracker.complete_request("unknown")

    def test_check_sla_after_complete_returns_true(self) -> None:
        tracker = SLATracker(sla_deadline_ms=0.0)
        tracker.start_request("req-1")
        tracker.complete_request("req-1")
        assert tracker.check_sla("req-1") is True

    def test_total_violations_after_complete_preserved(self) -> None:
        tracker = SLATracker(sla_deadline_ms=0.0)
        tracker.start_request("req-1")
        tracker.check_sla("req-1")
        tracker.complete_request("req-1")
        assert tracker.total_violations == 1


# ---------------------------------------------------------------------------
# TestCheckpointState
# ---------------------------------------------------------------------------


class TestCheckpointState:
    """Dataclass construction and memory_bytes calculation."""

    def test_minimal_construction(self) -> None:
        k = torch.zeros(1, 1, 1)
        v = torch.zeros(1, 1, 1)
        cp = CheckpointState(
            request_id="req-1",
            prompt_tokens=[1, 2, 3],
            generated_tokens=[4],
            kv_cache=[(k, v)],
            priority=5,
            temperature=0.8,
            top_p=0.95,
            top_k=40,
        )
        assert cp.request_id == "req-1"
        assert cp.prompt_tokens == [1, 2, 3]
        assert cp.generated_tokens == [4]
        assert cp.priority == 5
        assert cp.temperature == 0.8
        assert cp.top_p == 0.95
        assert cp.top_k == 40
        assert isinstance(cp.preempted_at, float)

    def test_empty_prompt_tokens(self) -> None:
        k = torch.zeros(1, 1, 1)
        v = torch.zeros(1, 1, 1)
        cp = CheckpointState(
            request_id="req-empty",
            prompt_tokens=[],
            generated_tokens=[],
            kv_cache=[(k, v)],
            priority=0,
            temperature=1.0,
            top_p=0.9,
            top_k=50,
        )
        assert cp.prompt_tokens == []
        assert cp.generated_tokens == []

    def test_memory_bytes_single_pair(self) -> None:
        k = torch.zeros(2, 4, 8)
        v = torch.zeros(2, 4, 8)
        cp = CheckpointState(
            request_id="req-1",
            prompt_tokens=[1],
            generated_tokens=[2],
            kv_cache=[(k, v)],
            priority=0,
            temperature=1.0,
            top_p=0.9,
            top_k=50,
        )
        expected = k.element_size() * k.numel() + v.element_size() * v.numel()
        assert cp.memory_bytes() == expected

    def test_memory_bytes_multiple_pairs(self) -> None:
        k1, v1 = torch.zeros(1, 1, 4), torch.zeros(1, 1, 4)
        k2, v2 = torch.zeros(2, 2, 4), torch.zeros(2, 2, 4)
        cp = CheckpointState(
            request_id="req-2",
            prompt_tokens=[1],
            generated_tokens=[2],
            kv_cache=[(k1, v1), (k2, v2)],
            priority=0,
            temperature=1.0,
            top_p=0.9,
            top_k=50,
        )
        expected = (
            k1.element_size() * k1.numel() + v1.element_size() * v1.numel()
            + k2.element_size() * k2.numel() + v2.element_size() * v2.numel()
        )
        assert cp.memory_bytes() == expected

    def test_memory_bytes_empty_kv_cache(self) -> None:
        cp = CheckpointState(
            request_id="req-nokv",
            prompt_tokens=[1],
            generated_tokens=[2],
            kv_cache=[],
            priority=0,
            temperature=1.0,
            top_p=0.9,
            top_k=50,
        )
        assert cp.memory_bytes() == 0

    def test_preempted_at_auto_generated(self) -> None:
        t0 = time.time()
        cp = CheckpointState(
            request_id="req-1",
            prompt_tokens=[1],
            generated_tokens=[2],
            kv_cache=[],
            priority=0,
            temperature=1.0,
            top_p=0.9,
            top_k=50,
        )
        t1 = time.time()
        assert t0 <= cp.preempted_at <= t1


# ---------------------------------------------------------------------------
# TestPreemptionPolicy
# ---------------------------------------------------------------------------


class TestPreemptionPolicy:
    """Policy orchestration: preemption triggers, checkpoint lifecycle, eviction."""

    # -- Construction -----------------------------------------------------

    def test_default_construction(self) -> None:
        policy = PreemptionPolicy()
        assert policy.max_queue_depth == 100
        assert policy.max_checkpoints == 10
        assert policy.checkpoint_memory_limit_mb == 4096
        assert policy._checkpoints == {}
        assert policy._total_checkpoint_memory == 0

    def test_custom_construction(self) -> None:
        gpu = GPUMemoryMonitor(preempt_threshold=0.99)
        sla = SLATracker(max_violations=5)
        policy = PreemptionPolicy(
            gpu_monitor=gpu,
            sla_tracker=sla,
            max_queue_depth=50,
            max_checkpoints=5,
            checkpoint_memory_limit_mb=1024,
        )
        assert policy.max_queue_depth == 50
        assert policy.max_checkpoints == 5
        assert policy.checkpoint_memory_limit_mb == 1024
        assert policy.gpu_monitor is gpu
        assert policy.sla_tracker is sla

    # -- should_preempt ---------------------------------------------------

    def test_should_preempt_gpu_pressure(self) -> None:
        gpu = GPUMemoryMonitor(preempt_threshold=0.01)
        policy = PreemptionPolicy(gpu_monitor=gpu)
        gpu._history = [0.5]
        assert policy.should_preempt(pending_count=0) is True

    def test_should_preempt_sla_violations(self) -> None:
        sla = SLATracker(max_violations=0, sla_deadline_ms=0.0)
        policy = PreemptionPolicy(sla_tracker=sla)
        sla.start_request("req-1")
        sla.check_sla("req-1")
        assert policy.should_preempt(pending_count=0, request_id="req-1") is True

    def test_should_preempt_queue_depth(self) -> None:
        policy = PreemptionPolicy(max_queue_depth=5)
        assert policy.should_preempt(pending_count=10) is True

    def test_should_preempt_all_clear(self) -> None:
        gpu = GPUMemoryMonitor(preempt_threshold=0.99)
        sla = SLATracker(sla_deadline_ms=1e9)
        policy = PreemptionPolicy(gpu_monitor=gpu, sla_tracker=sla)
        gpu._history = [0.1]
        assert policy.should_preempt(pending_count=0) is False

    def test_should_preempt_no_request_id_skips_sla_check(self) -> None:
        gpu = GPUMemoryMonitor(preempt_threshold=0.99)
        policy = PreemptionPolicy(gpu_monitor=gpu)
        gpu._history = [0.1]
        assert policy.should_preempt(pending_count=0) is False

    def test_should_preempt_accepts_min_priority(self) -> None:
        policy = PreemptionPolicy(max_queue_depth=5)
        assert policy.should_preempt(pending_count=10, min_priority=5) is True

    def test_should_preempt_sla_not_exceeded_returns_false(self) -> None:
        sla = SLATracker(max_violations=3, sla_deadline_ms=1e9)
        policy = PreemptionPolicy(sla_tracker=sla)
        sla.start_request("req-1")
        assert policy.should_preempt(pending_count=0, request_id="req-1") is False

    # -- create_checkpoint ------------------------------------------------

    def test_create_checkpoint_success(self) -> None:
        policy = PreemptionPolicy(max_checkpoints=5)
        seq = _FakeSequence(priority=2, temperature=0.7, top_p=0.8, top_k=30)
        kv = _make_kv(seq_len=1, n_heads=1, dim=4)
        cp = policy.create_checkpoint("req-1", kv, seq)
        assert cp is not None
        assert cp.request_id == "req-1"
        assert cp.prompt_tokens == [1, 2, 3]
        assert cp.generated_tokens == [4, 5]
        assert cp.priority == 2
        assert cp.temperature == 0.7
        assert cp.top_p == 0.8
        assert cp.top_k == 30
        assert "req-1" in policy._checkpoints
        assert policy._total_checkpoint_memory > 0

    def test_create_checkpoint_empty_kv_cache(self) -> None:
        policy = PreemptionPolicy()
        seq = _FakeSequence()
        cp = policy.create_checkpoint("req-nokv", [], seq)
        assert cp is not None
        assert cp.kv_cache == []
        assert policy._total_checkpoint_memory == 0.0

    def test_create_checkpoint_limit_exceeded(self) -> None:
        policy = PreemptionPolicy(max_checkpoints=2)
        seq = _FakeSequence()
        kv = _make_kv()
        policy.create_checkpoint("req-1", kv, seq)
        policy.create_checkpoint("req-2", kv, seq)
        assert policy.create_checkpoint("req-3", kv, seq) is None

    def test_create_checkpoint_memory_limit(self) -> None:
        policy = PreemptionPolicy(checkpoint_memory_limit_mb=0)
        seq = _FakeSequence()
        kv = _make_kv(seq_len=1, n_heads=1, dim=4)
        assert policy.create_checkpoint("req-1", kv, seq) is None

    def test_create_checkpoint_clones_tensors(self) -> None:
        policy = PreemptionPolicy()
        seq = _FakeSequence()
        k_orig = torch.randn(1, 1, 4)
        v_orig = torch.randn(1, 1, 4)
        kv = [(k_orig, v_orig)]
        cp = policy.create_checkpoint("req-1", kv, seq)
        assert cp is not None
        ck, cv = cp.kv_cache[0]
        assert ck.data_ptr() != k_orig.data_ptr()
        assert cv.data_ptr() != v_orig.data_ptr()

    # -- restore_checkpoint -----------------------------------------------

    def test_restore_checkpoint_returns_checkpoint(self) -> None:
        policy = PreemptionPolicy()
        seq = _FakeSequence()
        kv = _make_kv()
        policy.create_checkpoint("req-1", kv, seq)
        cp = policy.restore_checkpoint("req-1")
        assert cp is not None
        assert cp.request_id == "req-1"

    def test_restore_checkpoint_removes_from_internal_dict(self) -> None:
        policy = PreemptionPolicy()
        seq = _FakeSequence()
        kv = _make_kv()
        policy.create_checkpoint("req-1", kv, seq)
        policy.restore_checkpoint("req-1")
        assert "req-1" not in policy._checkpoints

    def test_restore_checkpoint_unknown_returns_none(self) -> None:
        policy = PreemptionPolicy()
        assert policy.restore_checkpoint("unknown") is None

    def test_restore_adjusts_total_memory_down(self) -> None:
        policy = PreemptionPolicy()
        seq = _FakeSequence()
        kv = _make_kv(seq_len=2, n_heads=2, dim=4)
        policy.create_checkpoint("req-1", kv, seq)
        assert policy._total_checkpoint_memory > 0
        policy.restore_checkpoint("req-1")
        assert policy._total_checkpoint_memory == 0.0

    # -- get_checkpoint_stats ---------------------------------------------

    def test_get_checkpoint_stats_empty(self) -> None:
        policy = PreemptionPolicy(
            max_checkpoints=10, checkpoint_memory_limit_mb=4096
        )
        stats = policy.get_checkpoint_stats()
        assert stats == {
            "checkpoint_count": 0,
            "total_memory_mb": 0.0,
            "memory_limit_mb": 4096,
            "max_checkpoints": 10,
        }

    def test_get_checkpoint_stats_after_create(self) -> None:
        policy = PreemptionPolicy()
        seq = _FakeSequence()
        # Use large-enough tensors so memory rounds to >0 MB (needs >52428 bytes)
        kv = [(torch.zeros(32, 8, 128), torch.zeros(32, 8, 128))]
        policy.create_checkpoint("req-1", kv, seq)
        stats = policy.get_checkpoint_stats()
        assert stats["checkpoint_count"] == 1
        assert stats["total_memory_mb"] > 0

    # -- evict_oldest_checkpoint ------------------------------------------

    def test_evict_oldest_checkpoint_empty_returns_none(self) -> None:
        policy = PreemptionPolicy()
        assert policy.evict_oldest_checkpoint() is None

    def test_evict_oldest_checkpoint_single(self) -> None:
        policy = PreemptionPolicy()
        seq = _FakeSequence()
        kv = _make_kv()
        policy.create_checkpoint("req-1", kv, seq)
        assert policy.evict_oldest_checkpoint() == "req-1"

    def test_evict_oldest_checkpoint_removes_oldest(self) -> None:
        policy = PreemptionPolicy(max_checkpoints=5)
        seq = _FakeSequence()
        kv = _make_kv()
        policy.create_checkpoint("req-1", kv, seq)
        time.sleep(0.05)
        policy.create_checkpoint("req-2", kv, seq)
        assert policy.evict_oldest_checkpoint() == "req-1"

    def test_evict_oldest_checkpoint_adjusts_memory(self) -> None:
        policy = PreemptionPolicy()
        seq = _FakeSequence()
        kv = _make_kv(seq_len=1, n_heads=1, dim=4)
        policy.create_checkpoint("req-1", kv, seq)
        assert policy._total_checkpoint_memory > 0
        policy.evict_oldest_checkpoint()
        assert policy._total_checkpoint_memory == 0.0
        assert len(policy._checkpoints) == 0
