"""Comprehensive preemption lifecycle tests for BatchScheduler.

Tests the full preemption cycle:
- Preempt while pending
- Preempt max sequences
- Concurrent preemption
- Restore after complete
- KV state round-trip
- Multiple preemptions
- Preempt high priority
- Tiered preemption
- Policy-driven preemption

Run: pytest tests/core/test_preemption.py -v
"""

import time
import threading
from unittest.mock import MagicMock

import pytest
import torch

from distllm.core.batch_scheduler import (
    BatchScheduler,
    ScheduledBatch,
    Sequence,
    SequenceStatus,
)


def _make_seq(
    request_id: str = "req-1",
    prompt_len: int = 50,
    max_new: int = 128,
    priority: int = 2,
) -> Sequence:
    return Sequence(
        request_id=request_id,
        prompt_tokens=[1] * prompt_len,
        max_new_tokens=max_new,
        priority=priority,
    )


# ===================================================================
# Preemption Lifecycle Tests
# ===================================================================


class TestPreemptionLifecycle:
    """End-to-end preemption lifecycle tests."""

    def test_preempt_then_restore_basic(self):
        """Preempt a sequence, advance others, restore preempted."""
        sched = BatchScheduler(max_batch_size=2)
        sched.add(_make_seq("a", priority=2))
        sched.add(_make_seq("b", priority=3))
        sched.add(_make_seq("c", priority=1))

        batch = sched.schedule()
        batch_ids = {s.request_id for s in batch.sequences}
        assert "a" in batch_ids
        assert "c" in batch_ids  # higher priority than b

        kv = {"a": "kv_data_a"}
        preempted = sched.preempt_lowest(min_priority=2, kv_cache_state=kv)
        assert preempted is not None
        assert preempted.request_id == "a"
        assert sched.get_preempted_count() == 1

        # Advance remaining
        batch2 = sched.schedule()
        if batch2 is not None:
            sched.step(batch2, torch.tensor([42] * len(batch2.sequences)))

        # Restore
        restored_kv = {}
        restored = sched.restore_preempted(kv_cache_state=restored_kv)
        assert len(restored) == 1
        assert restored[0].request_id == "a"
        assert restored[0].status == SequenceStatus.DECODING
        assert restored_kv["a"] == "kv_data_a"
        assert sched.get_preempted_count() == 0
        assert "a" in sched.active

    def test_preempt_no_candidates(self):
        """No preemption when all active priorities < min_priority."""
        sched = BatchScheduler(max_batch_size=2)
        sched.add(_make_seq("a", priority=0))
        sched.add(_make_seq("b", priority=1))
        sched.schedule()

        preempted = sched.preempt_lowest(min_priority=3)
        assert preempted is None

    def test_preempt_empty_active(self):
        """No preemption when no active sequences."""
        sched = BatchScheduler(max_batch_size=2)
        preempted = sched.preempt_lowest(min_priority=1)
        assert preempted is None

    def test_preempt_respects_max_preempted(self):
        """Cannot preempt more than max_preempted sequences."""
        sched = BatchScheduler(max_batch_size=4)
        sched.set_max_preempted(1)

        sched.add(_make_seq("a", priority=3))
        sched.add(_make_seq("b", priority=3))
        sched.add(_make_seq("c", priority=3))
        sched.add(_make_seq("d", priority=3))
        sched.schedule()

        p1 = sched.preempt_lowest(min_priority=1)
        assert p1 is not None

        p2 = sched.preempt_lowest(min_priority=1)
        assert p2 is None  # max_preempted reached

    def test_preempt_highest_numeric_priority(self):
        """Preempts the sequence with highest numeric priority (least important)."""
        sched = BatchScheduler(max_batch_size=3)
        sched.add(_make_seq("critical", priority=0))
        sched.add(_make_seq("normal", priority=2))
        sched.add(_make_seq("low", priority=3))
        sched.schedule()

        preempted = sched.preempt_lowest(min_priority=2)
        assert preempted is not None
        assert preempted.request_id == "low"

    def test_preempt_with_paged_attention(self):
        """Preemption calls PagedAttention swap_out_sequence."""
        mock_pool = MagicMock()
        mock_pool.total_blocks = 1000
        mock_pool.utilization = 0.5

        mock_pa = MagicMock()
        mock_pa.swap_out_sequence = MagicMock()
        mock_pa.pool_utilization = 0.5
        mock_pa.pool = mock_pool
        mock_pa.block_size = 16

        sched = BatchScheduler(max_batch_size=2, paged_attention_mgr=mock_pa)

        sched.add(_make_seq("a", priority=3))
        sched.schedule()

        sched.preempt_lowest(min_priority=1)
        mock_pa.swap_out_sequence.assert_called_once_with("a")

    def test_preempt_removes_from_active(self):
        """Preempted sequence is removed from active set."""
        sched = BatchScheduler(max_batch_size=2)
        sched.add(_make_seq("a", priority=3))
        sched.schedule()

        assert "a" in sched.active
        sched.preempt_lowest(min_priority=1)
        assert "a" not in sched.active

    def test_preempt_updates_token_count(self):
        """Preempting a sequence reduces total token count."""
        sched = BatchScheduler(max_batch_size=2)
        sched.add(_make_seq("a", prompt_len=100, priority=3))
        sched.schedule()

        initial_tokens = sched._total_tokens
        sched.preempt_lowest(min_priority=1)
        assert sched._total_tokens < initial_tokens

    def test_preempt_adds_to_pending_heap(self):
        """Preempted sequence is re-added to pending heap."""
        sched = BatchScheduler(max_batch_size=2)
        sched.add(_make_seq("a", priority=3))
        sched.schedule()

        sched.preempt_lowest(min_priority=1)
        assert sched.pending_count >= 1


# ===================================================================
# KV State Round-Trip Tests
# ===================================================================


class TestKVStateRoundTrip:
    """Tests for KV state save/restore during preemption."""

    def test_save_restore_kv_state(self):
        """KV state is preserved through preemption/restore cycle."""
        sched = BatchScheduler(max_batch_size=2)
        sched.add(_make_seq("a", priority=3))
        sched.schedule()

        kv_data = {"layers": [torch.randn(10, 10) for _ in range(4)]}
        kv = {"a": kv_data}

        sched.preempt_lowest(min_priority=1, kv_cache_state=kv)

        restored_kv = {}
        sched.restore_preempted(kv_cache_state=restored_kv)

        assert "a" in restored_kv
        assert restored_kv["a"]["layers"][0].shape == (10, 10)

    def test_save_kv_state_none(self):
        """Saving with None kv_cache_state does nothing."""
        sched = BatchScheduler(max_batch_size=2)
        sched.add(_make_seq("a", priority=3))
        sched.schedule()

        sched.preempt_lowest(min_priority=1, kv_cache_state=None)
        assert sched.get_preempted_count() == 1

    def test_restore_kv_state_none(self):
        """Restoring with None kv_cache_state doesn't crash."""
        sched = BatchScheduler(max_batch_size=2)
        sched.add(_make_seq("a", priority=3))
        sched.schedule()

        sched.preempt_lowest(min_priority=1, kv_cache_state={"a": "data"})
        restored = sched.restore_preempted(kv_cache_state=None)
        assert len(restored) == 1

    def test_multiple_kv_state_round_trip(self):
        """Multiple sequences' KV states are preserved independently."""
        sched = BatchScheduler(max_batch_size=4)
        sched.set_max_preempted(4)

        for i in range(4):
            sched.add(_make_seq(f"seq-{i}", priority=3))
        sched.schedule()

        kv = {f"seq-{i}": f"data_{i}" for i in range(4)}
        for _ in range(2):
            sched.preempt_lowest(min_priority=1, kv_cache_state=kv)

        restored_kv = {}
        sched.restore_preempted(kv_cache_state=restored_kv)
        assert len(restored_kv) == 2


# ===================================================================
# Concurrent Preemption Tests
# ===================================================================


class TestConcurrentPreemption:
    """Thread-safety tests for preemption."""

    def test_concurrent_preempt_and_schedule(self):
        """Concurrent preemption and scheduling don't crash."""
        sched = BatchScheduler(max_batch_size=4)
        for i in range(10):
            sched.add(_make_seq(f"seq-{i}", priority=3))

        errors = []

        def preempt_loop():
            try:
                for _ in range(5):
                    sched.preempt_lowest(min_priority=1, kv_cache_state={})
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def schedule_loop():
            try:
                for _ in range(10):
                    sched.schedule()
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=preempt_loop)
        t2 = threading.Thread(target=schedule_loop)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        assert len(errors) == 0

    def test_concurrent_add_and_preempt(self):
        """Concurrent adds and preemptions don't crash."""
        sched = BatchScheduler(max_batch_size=4)
        errors = []

        def add_loop():
            try:
                for i in range(20):
                    sched.add(_make_seq(f"add-{i}", priority=3))
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def preempt_loop():
            try:
                for _ in range(10):
                    sched.preempt_lowest(min_priority=1)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=add_loop)
        t2 = threading.Thread(target=preempt_loop)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        assert len(errors) == 0


# ===================================================================
# Multiple Preemption Tests
# ===================================================================


class TestMultiplePreemption:
    """Tests for multiple sequential preemptions."""

    def test_preempt_multiple_sequential(self):
        """Can preempt multiple sequences one after another."""
        sched = BatchScheduler(max_batch_size=4)
        sched.set_max_preempted(4)

        for i in range(4):
            sched.add(_make_seq(f"seq-{i}", priority=3))
        sched.schedule()

        preempted = []
        for _ in range(3):
            p = sched.preempt_lowest(min_priority=1, kv_cache_state={})
            if p:
                preempted.append(p)

        assert len(preempted) == 3
        assert sched.get_preempted_count() == 3

    def test_preempt_restore_preempt_again(self):
        """Can preempt, restore, and preempt again."""
        sched = BatchScheduler(max_batch_size=2)
        sched.add(_make_seq("a", priority=3))
        sched.add(_make_seq("b", priority=3))
        sched.schedule()

        # First preemption
        sched.preempt_lowest(min_priority=1, kv_cache_state={"a": "data1"})
        assert sched.get_preempted_count() == 1

        # Restore
        sched.restore_preempted(kv_cache_state={})
        assert sched.get_preempted_count() == 0

        # Second preemption
        sched.preempt_lowest(min_priority=1, kv_cache_state={"a": "data2"})
        assert sched.get_preempted_count() == 1

    def test_preempt_all_then_restore_all(self):
        """Preempt all active sequences, then restore all."""
        sched = BatchScheduler(max_batch_size=3)
        sched.set_max_preempted(3)

        for i in range(3):
            sched.add(_make_seq(f"seq-{i}", priority=3))
        sched.schedule()

        for _ in range(3):
            sched.preempt_lowest(min_priority=1, kv_cache_state={})

        assert sched.get_preempted_count() == 3
        assert sched.active_count == 0

        restored = sched.restore_preempted(kv_cache_state={})
        assert len(restored) == 3
        assert sched.get_preempted_count() == 0


# ===================================================================
# Policy-Driven Preemption Tests
# ===================================================================


class TestPolicyDrivenPreemption:
    """Tests for PreemptionPolicy integration."""

    def test_preempt_if_needed_no_policy(self):
        """preempt_if_needed returns None when no policy is set."""
        sched = BatchScheduler(max_batch_size=2)
        result = sched.preempt_if_needed()
        assert result is None

    def test_preempt_if_needed_policy_triggers(self):
        """preempt_if_needed preempts when policy says yes."""
        from unittest.mock import MagicMock

        sched = BatchScheduler(max_batch_size=2)
        sched.add(_make_seq("a", priority=3))
        sched.schedule()

        policy = MagicMock()
        policy.should_preempt.return_value = True
        sched.set_preemption_policy(policy)

        result = sched.preempt_if_needed()
        assert result is not None
        policy.should_preempt.assert_called_once()

    def test_preempt_if_needed_policy_no_trigger(self):
        """preempt_if_needed returns None when policy says no."""
        from unittest.mock import MagicMock

        sched = BatchScheduler(max_batch_size=2)
        sched.add(_make_seq("a", priority=3))
        sched.schedule()

        policy = MagicMock()
        policy.should_preempt.return_value = False
        sched.set_preemption_policy(policy)

        result = sched.preempt_if_needed()
        assert result is None

    def test_preempt_if_needed_integrated_in_schedule(self):
        """preempt_if_needed is called during schedule()."""
        from unittest.mock import MagicMock

        sched = BatchScheduler(max_batch_size=2)
        sched.add(_make_seq("a", priority=3))
        sched.add(_make_seq("b", priority=3))
        sched.schedule()

        policy = MagicMock()
        policy.should_preempt.return_value = True
        sched.set_preemption_policy(policy)

        # schedule() should call preempt_if_needed internally
        batch = sched.schedule()
        policy.should_preempt.assert_called()


# ===================================================================
# Preemption Stats Tests
# ===================================================================


class TestPreemptionStats:
    """Tests for preemption statistics."""

    def test_stats_includes_preempted(self):
        """stats() includes preempted_requests count."""
        sched = BatchScheduler(max_batch_size=2)
        stats = sched.stats()
        assert stats["preempted_requests"] == 0

        sched.add(_make_seq("a", priority=3))
        sched.schedule()
        sched.preempt_lowest(min_priority=1)

        stats = sched.stats()
        assert stats["preempted_requests"] == 1

    def test_set_max_preempted(self):
        """set_max_preempted updates the limit."""
        sched = BatchScheduler(max_batch_size=4)
        sched.set_max_preempted(10)
        assert sched._max_preempted == 10

    def test_set_max_preempted_clamps_negative(self):
        """set_max_preempted clamps negative values to 0."""
        sched = BatchScheduler(max_batch_size=4)
        sched.set_max_preempted(-5)
        assert sched._max_preempted == 0
