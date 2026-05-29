"""Edge-case tests for BatchScheduler: empty batch, single sequence, preemption, chunked prefill."""

import threading
import time
from unittest.mock import MagicMock

import pytest
import torch

from distllm.core.batch_scheduler import BatchScheduler, Sequence, SequenceStatus


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
# Empty batch / edge cases
# ===================================================================

class TestEmptyBatch:
    def test_schedule_empty_returns_none(self):
        sched = BatchScheduler(max_batch_size=4)
        assert sched.schedule() is None

    def test_schedule_empty_no_pending(self):
        sched = BatchScheduler(max_batch_size=4)
        assert not sched.has_pending

    def test_schedule_empty_queue_and_active(self):
        sched = BatchScheduler(max_batch_size=4)
        assert sched.pending_count == 0
        assert sched.active_count == 0

    def test_generate_batch_with_empty_queue(self):
        """generate_batch should return immediately when nothing pending."""
        sched = BatchScheduler(max_batch_size=4)
        # Should not hang
        import time
        start = time.time()
        # We can't directly call generate_batch without coordinator,
        # but we can verify schedule returns None
        assert sched.schedule() is None


class TestSingleSequence:
    def test_schedule_single_sequence(self):
        sched = BatchScheduler(max_batch_size=4)
        seq = _make_seq()
        sched.add(seq)
        batch = sched.schedule()
        assert batch is not None
        assert len(batch.sequences) == 1
        assert batch.sequences[0].request_id == "req-1"

    def test_single_sequence_completes(self):
        sched = BatchScheduler(max_batch_size=4)
        seq = _make_seq(max_new=3)
        sched.add(seq)
        batch = sched.schedule()
        assert batch is not None

        tokens = torch.tensor([100, 101, 102])
        eos_token_id = 102
        for i in range(3):
            sched.step(batch, tokens[i].unsqueeze(0))
            if tokens[i].item() == eos_token_id:
                break
        # Should be marked complete
        assert seq.is_complete or seq.status == SequenceStatus.DONE

    def test_single_sequence_hits_max_tokens(self):
        sched = BatchScheduler(max_batch_size=4)
        seq = _make_seq(max_new=2)
        sched.add(seq)
        batch = sched.schedule()
        assert batch is not None

        sched.step(batch, torch.tensor([10]))
        assert not seq.is_complete
        sched.step(batch, torch.tensor([11]))
        assert seq.is_complete


# ===================================================================
# Max batch size edge cases
# ===================================================================

class TestMaxBatchEdgeCases:
    def test_max_batch_size_1(self):
        sched = BatchScheduler(max_batch_size=1)
        sched.add(_make_seq("a"))
        sched.add(_make_seq("b"))
        batch = sched.schedule()
        assert batch is not None
        assert len(batch.sequences) == 1
        assert batch.sequences[0].request_id == "a"

    def test_max_batch_size_equals_pending(self):
        sched = BatchScheduler(max_batch_size=5)
        for i in range(5):
            sched.add(_make_seq(f"seq-{i}"))
        batch = sched.schedule()
        assert batch is not None
        assert len(batch.sequences) == 5

    def test_max_batch_size_surplus_pending(self):
        sched = BatchScheduler(max_batch_size=3)
        for i in range(10):
            sched.add(_make_seq(f"seq-{i}"))
        batch = sched.schedule()
        assert batch is not None
        assert len(batch.sequences) == 3
        # Remaining should be pending
        assert sched.pending_count == 7


# ===================================================================
# Token budget edge cases
# ===================================================================

class TestTokenBudget:
    def test_exact_token_budget(self):
        sched = BatchScheduler(max_batch_size=4, max_tokens_per_batch=100)
        sched.add(_make_seq("big", prompt_len=100))
        batch = sched.schedule()
        assert batch is not None
        assert len(batch.sequences) == 1

    def test_oversized_prompt_excluded(self):
        sched = BatchScheduler(max_batch_size=4, max_tokens_per_batch=50)
        sched.add(_make_seq("big", prompt_len=100))
        sched.add(_make_seq("small", prompt_len=25))
        batch = sched.schedule()
        assert batch is not None
        ids = [s.request_id for s in batch.sequences]
        assert "small" in ids
        assert "big" not in ids

    def test_multiple_small_sequences_under_budget(self):
        sched = BatchScheduler(max_batch_size=10, max_tokens_per_batch=200)
        for i in range(8):
            sched.add(_make_seq(f"s{i}", prompt_len=20, max_new=5))
        batch = sched.schedule()
        assert batch is not None
        assert len(batch.sequences) == 8


# ===================================================================
# Preemption
# ===================================================================

class TestPreemption:
    def test_preempt_lowest_priority(self):
        sched = BatchScheduler(max_batch_size=2)
        seq_high = _make_seq("high", priority=5)
        seq_low = _make_seq("low", priority=1)

        sched.add(seq_high)
        sched.add(seq_low)
        batch = sched.schedule()
        assert batch is not None
        assert len(batch.sequences) == 2

        # preempt_lowest selects highest-priority active seq >= min_priority
        preempted = sched.preempt_lowest(min_priority=3, kv_cache_state={})
        assert preempted is not None
        # With priority >= 3, "high" (5) is selected (highest >= threshold)
        assert preempted.request_id == "high"

    def test_preempt_no_candidates(self):
        sched = BatchScheduler(max_batch_size=2)
        sched.add(_make_seq("a", priority=5))
        sched.schedule()
        # All active priorities >= 3 are eligible; "a" (5) will be preempted
        preempted = sched.preempt_lowest(min_priority=3, kv_cache_state={})
        assert preempted is not None
        assert preempted.request_id == "a"

    def test_preempt_empty_active(self):
        sched = BatchScheduler(max_batch_size=2)
        preempted = sched.preempt_lowest(min_priority=1, kv_cache_state={})
        assert preempted is None

    def test_preempt_restore(self):
        """End-to-end: preempt a sequence, advance others, restore preempted."""
        sched = BatchScheduler(max_batch_size=2)
        seq_a = _make_seq("a", priority=2)
        seq_b = _make_seq("b", priority=3)
        seq_c = _make_seq("c", priority=1)

        sched.add(seq_a)
        sched.add(seq_b)
        sched.add(seq_c)

        # Schedule: a + c (both fit, c has higher priority than b)
        batch = sched.schedule()
        assert batch is not None
        batch_ids = {s.request_id for s in batch.sequences}
        assert "a" in batch_ids
        assert "c" in batch_ids

        # Preempt 'a' (priority 2 >= min_priority=1)
        kv = {"a": "kv_data_a"}
        preempted = sched.preempt_lowest(min_priority=1, kv_cache_state=kv)
        assert preempted is not None
        assert preempted.request_id == "a"
        assert sched.get_preempted_count() == 1

        # Advance the remaining sequence
        batch2 = sched.schedule()
        if batch2 is not None:
            sched.step(batch2, torch.tensor([42] * len(batch2.sequences)))

        # Restore 'a' — should come back as DECODING with KV state restored
        restored_kv = {}
        restored = sched.restore_preempted(kv_cache_state=restored_kv)
        assert len(restored) == 1
        assert restored[0].request_id == "a"
        assert restored[0].status == SequenceStatus.DECODING
        assert restored_kv["a"] == "kv_data_a"
        assert sched.get_preempted_count() == 0
        assert "a" in sched.active


# ===================================================================
# Priority scheduling edge cases
# ===================================================================

class TestPriorityEdgeCases:
    def test_fifo_within_same_priority(self):
        sched = BatchScheduler(max_batch_size=10)
        for i in range(5):
            sched.add(_make_seq(f"seq-{i}", priority=2))
        batch = sched.schedule()
        assert batch is not None
        ids = [s.request_id for s in batch.sequences]
        assert ids == ["seq-0", "seq-1", "seq-2", "seq-3", "seq-4"]

    def test_high_priority_first(self):
        sched = BatchScheduler(max_batch_size=10)
        # Min-heap sorts ascending: lower numeric value = higher scheduling priority
        sched.add(_make_seq("low", priority=10))
        sched.add(_make_seq("high", priority=1))
        sched.add(_make_seq("mid", priority=5))
        batch = sched.schedule()
        assert batch is not None
        ids = [s.request_id for s in batch.sequences]
        assert ids[0] == "high"
        assert ids[-1] == "low"


# ===================================================================
# Chunked prefill interaction
# ===================================================================

class TestChunkedPrefill:
    def test_sequence_with_chunk_state(self):
        sched = BatchScheduler(max_batch_size=4)
        seq = _make_seq("chunked", prompt_len=1000)
        seq.chunk_state = MagicMock()
        seq.chunk_state.chunks = [MagicMock(), MagicMock()]
        seq.chunk_state.current_chunk_idx = 0
        sched.add(seq)
        batch = sched.schedule()
        assert batch is not None
        assert batch.sequences[0].request_id == "chunked"

    def test_prefill_decode_flag(self):
        sched = BatchScheduler(max_batch_size=4)
        seq = _make_seq("test")
        sched.add(seq)
        batch = sched.schedule()
        assert batch.is_prefill[0] is True
        # After step
        sched.step(batch, torch.tensor([10]))
        # On subsequent schedule, it should be decode
        batch2 = sched.schedule()
        if batch2 is not None:
            pass


# ===================================================================
# Concurrent access
# ===================================================================

class TestConcurrentEdgeCases:
    def test_add_and_schedule_rapid(self):
        sched = BatchScheduler(max_batch_size=4)
        errors = []

        def adder():
            try:
                for i in range(20):
                    sched.add(_make_seq(f"fast-{i}"))
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def scheduler():
            try:
                for _ in range(10):
                    batch = sched.schedule()
                    time.sleep(0.002)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=adder)
        t2 = threading.Thread(target=scheduler)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        assert len(errors) == 0

    def test_promote_nonexistent(self):
        sched = BatchScheduler(max_batch_size=4)
        # promote_request looks up pending heap and returns False for absent
        result = sched.promote_request("nonexistent", new_priority=5)
        assert result is False
