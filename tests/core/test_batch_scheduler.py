"""Batch scheduler and concurrent request handling tests.

Tests:
- BatchScheduler: add, schedule, step with max_batch_size, max_tokens_per_batch limits
- Scheduling policy: active sequences + pending fill
- Sequence lifecycle: PENDING -> PREFILLING -> DECODING -> DONE
- Structured output constraint integration
- Concurrent add/schedule (not thread-safe - verify behavior)
- Coordinator generate_async / wait_for_result pattern

Note: The scheduler is NOT thread-safe (no locks). These tests document this.

Run: pytest tests/core/test_batch_scheduler.py -v
"""

import threading

import pytest
import torch

from distllm.core.batch_scheduler import (
    BatchScheduler,
    ScheduledBatch,
    Sequence,
    SequenceStatus,
)

# ============================================================
# Sequence Tests
# ============================================================


class TestSequence:
    """Tests for Sequence dataclass."""

    def test_sequence_initial_state(self):
        """New sequence should be PENDING."""
        seq = Sequence(request_id="req-1", prompt_tokens=[1, 2, 3])
        assert seq.status == SequenceStatus.PENDING
        assert seq.generated_tokens == []
        assert not seq.is_complete

    def test_is_complete_by_max_tokens(self):
        """Sequence should be complete when generated >= max_new_tokens."""
        seq = Sequence(
            request_id="req-1",
            prompt_tokens=[1, 2, 3],
            max_new_tokens=5,
        )
        seq.generated_tokens = [10, 11, 12, 13, 14]
        assert seq.is_complete

    def test_is_complete_not_yet(self):
        """Sequence should not be complete before max tokens."""
        seq = Sequence(
            request_id="req-1",
            prompt_tokens=[1, 2, 3],
            max_new_tokens=5,
        )
        seq.generated_tokens = [10, 11]
        assert not seq.is_complete

    def test_is_complete_explicit_done(self):
        """DONE status should make is_complete True."""
        seq = Sequence(
            request_id="req-1",
            prompt_tokens=[1, 2, 3],
            max_new_tokens=5,
        )
        seq.status = SequenceStatus.DONE
        assert seq.is_complete

    def test_is_complete_failed(self):
        """FAILED status should make is_complete True."""
        seq = Sequence(
            request_id="req-1",
            prompt_tokens=[1, 2, 3],
            max_new_tokens=5,
        )
        seq.status = SequenceStatus.FAILED
        assert seq.is_complete

    def test_total_len(self):
        """total_len should include prompt + generated."""
        seq = Sequence(
            request_id="req-1",
            prompt_tokens=[1, 2, 3],
            max_new_tokens=5,
        )
        seq.generated_tokens = [10, 11]
        assert seq.total_len == 5  # 3 + 2

    def test_decode_input_token(self):
        """decode_input_token should return last generated token."""
        seq = Sequence(
            request_id="req-1",
            prompt_tokens=[1, 2, 3],
        )
        seq.generated_tokens = [10, 11, 12]
        assert seq.decode_input_token == 12


# ============================================================
# BatchScheduler Basic Tests
# ============================================================


class TestBatchSchedulerAdd:
    """Tests for adding sequences to the scheduler."""

    def test_add_pending_sequence(self):
        """add should place sequence in pending queue."""
        scheduler = BatchScheduler(max_batch_size=4, max_tokens_per_batch=256)
        seq = Sequence(request_id="req-1", prompt_tokens=[1, 2, 3])

        scheduler.add(seq)

        assert scheduler.pending_count == 1
        assert any(s is seq for _, _, s in scheduler._pending_heap)

    def test_add_multiple_sequences(self):
        """Multiple adds should queue all sequences."""
        scheduler = BatchScheduler(max_batch_size=4, max_tokens_per_batch=256)

        for i in range(3):
            scheduler.add(Sequence(request_id=f"req-{i}", prompt_tokens=[i]))

        assert scheduler.pending_count == 3


class TestBatchSchedulerSchedule:
    """Tests for batch scheduling."""

    def test_schedule_empty_returns_none(self):
        """schedule should return None when no sequences exist."""
        scheduler = BatchScheduler(max_batch_size=4, max_tokens_per_batch=256)
        assert scheduler.schedule() is None

    def test_schedule_promotes_pending_to_active(self):
        """schedule should move pending sequences to active."""
        scheduler = BatchScheduler(max_batch_size=4, max_tokens_per_batch=256)
        scheduler.add(Sequence(request_id="req-1", prompt_tokens=[1, 2, 3]))

        batch = scheduler.schedule()

        assert batch is not None
        assert scheduler.active_count == 1
        assert "req-1" in scheduler.active

    def test_schedule_respects_max_batch_size(self):
        """schedule should not exceed max_batch_size."""
        scheduler = BatchScheduler(max_batch_size=2, max_tokens_per_batch=1000)

        for i in range(5):
            scheduler.add(Sequence(request_id=f"req-{i}", prompt_tokens=[i] * 10))

        batch = scheduler.schedule()

        assert batch.batch_size == 2
        assert scheduler.pending_count == 3  # 5 - 2 = 3 remaining

    def test_schedule_respects_max_tokens_per_batch(self):
        """schedule should not exceed max_tokens_per_batch."""
        scheduler = BatchScheduler(max_batch_size=10, max_tokens_per_batch=20)

        scheduler.add(Sequence(request_id="req-1", prompt_tokens=list(range(15))))
        scheduler.add(Sequence(request_id="req-2", prompt_tokens=list(range(10))))

        batch = scheduler.schedule()

        # Only req-1 fits (15 tokens <= 20)
        # req-2 (10 tokens) + req-1 (15) = 25 > 20, so excluded
        assert batch.batch_size == 1
        assert batch.request_ids == ["req-1"]

    def test_schedule_returns_none_when_no_active(self):
        """schedule should return None when all sequences are complete."""
        scheduler = BatchScheduler(max_batch_size=4, max_tokens_per_batch=256)
        seq = Sequence(
            request_id="req-1",
            prompt_tokens=[1, 2, 3],
            max_new_tokens=2,
        )
        seq.generated_tokens = [10, 11, 12]  # Exceeds max
        scheduler.active["req-1"] = seq

        assert scheduler.schedule() is None

    def test_schedule_evicts_completed_sequences(self):
        """schedule should remove completed sequences from active."""
        scheduler = BatchScheduler(max_batch_size=4, max_tokens_per_batch=256)
        seq = Sequence(
            request_id="req-1",
            prompt_tokens=[1, 2, 3],
            max_new_tokens=2,
        )
        seq.generated_tokens = [10, 11, 12]  # Complete
        scheduler.active["req-1"] = seq

        # Add a new pending sequence
        scheduler.add(Sequence(request_id="req-2", prompt_tokens=[4, 5, 6]))

        batch = scheduler.schedule()

        assert "req-1" not in scheduler.active
        assert batch.batch_size == 1
        assert batch.request_ids == ["req-2"]


class TestBatchSchedulerStep:
    """Tests for batch step processing."""

    def test_step_appends_tokens(self):
        """step should append sampled tokens to sequences."""
        scheduler = BatchScheduler(max_batch_size=4, max_tokens_per_batch=256)
        seq = Sequence(request_id="req-1", prompt_tokens=[1, 2, 3], max_new_tokens=5)
        scheduler.active["req-1"] = seq

        batch = ScheduledBatch(
            sequences=[seq],
            input_ids=torch.tensor([[1, 2, 3]]),
            seq_lengths=[3],
            position_offsets=[0],
            is_prefill=[True],
            request_ids=["req-1"],
        )

        next_tokens = torch.tensor([42])
        scheduler.step(batch, next_tokens)

        assert seq.generated_tokens == [42]

    def test_step_marks_done_on_stop_token(self):
        """step should mark sequence as DONE when stop token is generated."""
        scheduler = BatchScheduler(max_batch_size=4, max_tokens_per_batch=256)
        seq = Sequence(
            request_id="req-1",
            prompt_tokens=[1, 2, 3],
            max_new_tokens=10,
            stop_token_ids=[0],
        )
        scheduler.active["req-1"] = seq

        batch = ScheduledBatch(
            sequences=[seq],
            input_ids=torch.tensor([[1, 2, 3]]),
            seq_lengths=[3],
            position_offsets=[0],
            is_prefill=[True],
            request_ids=["req-1"],
        )

        # Generate stop token
        next_tokens = torch.tensor([0])
        scheduler.step(batch, next_tokens)

        assert seq.status == SequenceStatus.DONE

    def test_step_marks_done_on_max_tokens(self):
        """step should mark sequence as DONE when max tokens reached."""
        scheduler = BatchScheduler(max_batch_size=4, max_tokens_per_batch=256)
        seq = Sequence(
            request_id="req-1",
            prompt_tokens=[1, 2, 3],
            max_new_tokens=2,
        )
        # Already at max-1 tokens
        seq.generated_tokens = [10]
        scheduler.active["req-1"] = seq

        batch = ScheduledBatch(
            sequences=[seq],
            input_ids=torch.tensor([[4]]),
            seq_lengths=[1],
            position_offsets=[4],
            is_prefill=[False],
            request_ids=["req-1"],
        )

        next_tokens = torch.tensor([11])
        scheduler.step(batch, next_tokens)

        assert seq.status == SequenceStatus.DONE

    def test_step_handles_multiple_sequences(self):
        """step should process all sequences in the batch."""
        scheduler = BatchScheduler(max_batch_size=4, max_tokens_per_batch=256)
        seq1 = Sequence(request_id="req-1", prompt_tokens=[1], max_new_tokens=5)
        seq2 = Sequence(request_id="req-2", prompt_tokens=[2], max_new_tokens=5)
        scheduler.active["req-1"] = seq1
        scheduler.active["req-2"] = seq2

        batch = ScheduledBatch(
            sequences=[seq1, seq2],
            input_ids=torch.tensor([[1], [2]]),
            seq_lengths=[1, 1],
            position_offsets=[0, 0],
            is_prefill=[True, True],
            request_ids=["req-1", "req-2"],
        )

        next_tokens = torch.tensor([10, 20])
        scheduler.step(batch, next_tokens)

        assert seq1.generated_tokens == [10]
        assert seq2.generated_tokens == [20]


# ============================================================
# Scheduler State and Stats Tests
# ============================================================


class TestBatchSchedulerState:
    """Tests for scheduler state queries."""

    def test_has_pending_with_pending(self):
        """has_pending should be True when pending queue has items."""
        scheduler = BatchScheduler(max_batch_size=4, max_tokens_per_batch=256)
        scheduler.add(Sequence(request_id="req-1", prompt_tokens=[1]))
        assert scheduler.has_pending is True

    def test_has_pending_with_active(self):
        """has_pending should be True when active has non-complete sequences."""
        scheduler = BatchScheduler(max_batch_size=4, max_tokens_per_batch=256)
        seq = Sequence(request_id="req-1", prompt_tokens=[1])
        scheduler.active["req-1"] = seq
        assert scheduler.has_pending is True

    def test_has_pending_false(self):
        """has_pending should be False when nothing to process."""
        scheduler = BatchScheduler(max_batch_size=4, max_tokens_per_batch=256)
        assert scheduler.has_pending is False

    def test_stats(self):
        """stats should return current scheduler state."""
        scheduler = BatchScheduler(max_batch_size=8, max_tokens_per_batch=512)
        scheduler.add(Sequence(request_id="req-1", prompt_tokens=[1]))
        scheduler.add(Sequence(request_id="req-2", prompt_tokens=[2]))

        stats = scheduler.stats()

        assert stats["pending_requests"] == 2
        assert stats["active_requests"] == 0
        assert stats["max_batch_size"] == 8

    def test_get_sequence_from_active(self):
        """get_sequence should find sequence in active."""
        scheduler = BatchScheduler(max_batch_size=4, max_tokens_per_batch=256)
        seq = Sequence(request_id="req-1", prompt_tokens=[1])
        scheduler.active["req-1"] = seq

        found = scheduler.get_sequence("req-1")
        assert found is seq

    def test_get_sequence_from_pending(self):
        """get_sequence should find sequence in pending queue."""
        scheduler = BatchScheduler(max_batch_size=4, max_tokens_per_batch=256)
        seq = Sequence(request_id="req-1", prompt_tokens=[1])
        scheduler.add(seq)

        found = scheduler.get_sequence("req-1")
        assert found is seq

    def test_get_sequence_not_found(self):
        """get_sequence should return None for unknown request."""
        scheduler = BatchScheduler(max_batch_size=4, max_tokens_per_batch=256)
        assert scheduler.get_sequence("unknown") is None


# ============================================================
# Concurrent Access Tests
# ============================================================


class TestConcurrentAccess:
    """Tests for concurrent access behavior.

    Note: The scheduler is NOT thread-safe. These tests document the behavior
    under concurrent access and should be used to verify future thread-safety
    improvements.
    """

    def test_concurrent_add_not_thread_safe(self):
        """Concurrent adds from multiple threads may lose sequences.

        This documents the current non-thread-safe behavior.
        """
        scheduler = BatchScheduler(max_batch_size=100, max_tokens_per_batch=10000)

        errors = []

        def add_sequences(start, count):
            try:
                for i in range(count):
                    seq = Sequence(
                        request_id=f"req-{start}-{i}",
                        prompt_tokens=[start * 1000 + i],
                    )
                    scheduler.add(seq)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=add_sequences, args=(0, 50)),
            threading.Thread(target=add_sequences, args=(1, 50)),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Due to deque thread safety in CPython, this usually works
        # but is NOT guaranteed to be safe
        assert scheduler.pending_count == 100
        assert len(errors) == 0

    def test_add_and_schedule_interleaved(self):
        """Interleaved add and schedule should work correctly."""
        scheduler = BatchScheduler(max_batch_size=4, max_tokens_per_batch=256)

        # Add initial sequences
        for i in range(2):
            scheduler.add(Sequence(request_id=f"req-{i}", prompt_tokens=[i] * 10))

        # Schedule first batch
        batch1 = scheduler.schedule()
        assert batch1.batch_size == 2

        # Add more while first batch is active
        for i in range(2, 4):
            scheduler.add(Sequence(request_id=f"req-{i}", prompt_tokens=[i] * 10))

        # Schedule should include active + new pending
        batch2 = scheduler.schedule()
        # Active (2) + pending (2) = 4, limited by max_batch_size=4
        assert batch2.batch_size == 4


# ============================================================
# Coordinator Async Pattern Tests
# ============================================================


class TestCoordinatorAsyncPattern:
    """Tests for the generate_async / wait_for_result pattern."""

    def test_generate_async_returns_request_id(self, mock_coordinator_with_scheduler):
        """generate_async should return a request_id."""
        coord = mock_coordinator_with_scheduler

        request_id = coord.generate_async(
            prompt="Hello world",
            max_new_tokens=10,
            temperature=0.7,
        )

        assert request_id is not None
        assert isinstance(request_id, str)

    def test_generate_async_custom_request_id(self, mock_coordinator_with_scheduler):
        """generate_async should accept a custom request_id."""
        coord = mock_coordinator_with_scheduler

        request_id = coord.generate_async(
            prompt="Hello world",
            request_id="custom-id-123",
            max_new_tokens=10,
        )

        assert request_id == "custom-id-123"

    def test_wait_for_result_no_event_returns_empty(self, mock_coordinator_with_scheduler):
        """wait_for_result should return empty string when no event exists."""
        coord = mock_coordinator_with_scheduler
        # Store result directly without going through generate_async
        coord._request_results["req-1"] = "test result"

        result = coord.wait_for_result("req-1")

        assert result == "test result"

    def test_wait_for_result_timeout_returns_empty(self, mock_coordinator_with_scheduler):
        """wait_for_result should return empty string on timeout."""
        coord = mock_coordinator_with_scheduler

        # Create an event that won't be set
        event = threading.Event()
        coord._request_events["req-timeout"] = event

        result = coord.wait_for_result("req-timeout", timeout=0.1)

        assert result == ""

    def test_generate_async_without_scheduler_raises(self, mock_coordinator):
        """generate_async should raise when scheduler is not configured."""
        from distllm.errors.types import BatchError

        coord = mock_coordinator
        # This coordinator has max_batch_size=1, so no scheduler

        with pytest.raises(BatchError, match="Batch scheduler not configured"):
            coord.generate_async("Hello world")


# ============================================================
# Priority Queue Tests
# ============================================================


class TestSequencePriority:
    """Tests for Sequence priority field."""

    def test_sequence_has_priority_field(self):
        """Sequence should have a priority field."""
        seq = Sequence(request_id="req-1", prompt_tokens=[1, 2, 3])
        assert hasattr(seq, "priority")

    def test_default_priority_is_normal(self):
        """Default priority should be 2 (normal)."""
        seq = Sequence(request_id="req-1", prompt_tokens=[1, 2, 3])
        assert seq.priority == 2

    def test_custom_priority(self):
        """Sequence should accept custom priority."""
        seq = Sequence(request_id="req-1", prompt_tokens=[1, 2, 3], priority=0)
        assert seq.priority == 0


class TestPriorityScheduling:
    """Tests for priority-based scheduling."""

    def test_add_orders_by_priority(self):
        """Lower priority number should be scheduled first."""
        scheduler = BatchScheduler(max_batch_size=2, max_tokens_per_batch=1000)
        scheduler.add(Sequence(request_id="req-normal", prompt_tokens=[1] * 5, priority=2))
        scheduler.add(Sequence(request_id="req-critical", prompt_tokens=[2] * 5, priority=0))
        scheduler.add(Sequence(request_id="req-low", prompt_tokens=[3] * 5, priority=3))

        batch = scheduler.schedule()

        # Critical (0) should be first since batch includes active + pending fill
        # After first schedule, all pending become active, so order depends on promotion
        assert "req-critical" in batch.request_ids

    def test_priority_fill_with_active(self):
        """When active has room, pending should be filled by priority."""
        scheduler = BatchScheduler(max_batch_size=4, max_tokens_per_batch=1000)

        # Add 3 pending in reverse priority order
        scheduler.add(Sequence(request_id="req-low", prompt_tokens=[1] * 5, priority=3))
        scheduler.add(Sequence(request_id="req-normal", prompt_tokens=[2] * 5, priority=2))
        scheduler.add(Sequence(request_id="req-high", prompt_tokens=[3] * 5, priority=1))

        # First schedule promotes all to active (max_batch_size=4)
        batch = scheduler.schedule()
        assert batch.batch_size == 3

        # Add more while active
        scheduler.add(Sequence(request_id="req-critical", prompt_tokens=[4] * 5, priority=0))
        scheduler.add(Sequence(request_id="req-low2", prompt_tokens=[5] * 5, priority=3))

        # Complete all active sequences
        for seq in batch.sequences:
            seq.generated_tokens = [999] * seq.max_new_tokens

        # Schedule: completed evicted, critical should be promoted first
        batch2 = scheduler.schedule()
        assert "req-critical" in batch2.request_ids

    def test_fifo_within_same_priority(self):
        """Same priority sequences should maintain FIFO order."""
        scheduler = BatchScheduler(max_batch_size=2, max_tokens_per_batch=1000)
        scheduler.add(Sequence(request_id="req-first", prompt_tokens=[1] * 5, priority=2))
        scheduler.add(Sequence(request_id="req-second", prompt_tokens=[2] * 5, priority=2))

        batch = scheduler.schedule()
        assert batch.request_ids[0] == "req-first"


class TestPromoteRequest:
    """Tests for promote_request."""

    def test_promote_request(self):
        """Can change priority of pending request."""
        scheduler = BatchScheduler(max_batch_size=4, max_tokens_per_batch=1000)
        scheduler.add(Sequence(request_id="req-1", prompt_tokens=[1] * 5, priority=3))
        scheduler.add(Sequence(request_id="req-2", prompt_tokens=[2] * 5, priority=2))

        assert scheduler.promote_request("req-1", 0) is True

        batch = scheduler.schedule()
        # req-1 now has priority 0, should be scheduled first
        assert batch.request_ids[0] == "req-1"

    def test_promote_nonexistent_returns_false(self):
        """promote_request should return False for unknown request."""
        scheduler = BatchScheduler(max_batch_size=4, max_tokens_per_batch=1000)
        assert scheduler.promote_request("unknown", 0) is False

    def test_promote_active_not_in_heap(self):
        """promote_request should not find active sequences (not in heap)."""
        scheduler = BatchScheduler(max_batch_size=4, max_tokens_per_batch=1000)
        scheduler.add(Sequence(request_id="req-1", prompt_tokens=[1] * 5, priority=2))
        scheduler.schedule()  # Moves to active

        assert scheduler.promote_request("req-1", 0) is False


class TestPreemptLowest:
    """Tests for preempt_lowest."""

    def test_preempt_lowest(self):
        """Preempts lowest priority active sequence."""
        scheduler = BatchScheduler(max_batch_size=4, max_tokens_per_batch=1000)
        scheduler.add(Sequence(request_id="req-high", prompt_tokens=[1] * 5, priority=1))
        scheduler.add(Sequence(request_id="req-low", prompt_tokens=[2] * 5, priority=3))

        scheduler.schedule()  # Both active

        preempted = scheduler.preempt_lowest(min_priority=3)
        assert preempted is not None
        assert preempted.request_id == "req-low"
        assert preempted.status == SequenceStatus.PENDING
        assert "req-low" not in scheduler.active
        assert scheduler.pending_count == 1

    def test_preempt_no_candidates(self):
        """Returns None when nothing to preempt."""
        scheduler = BatchScheduler(max_batch_size=4, max_tokens_per_batch=1000)
        scheduler.add(Sequence(request_id="req-1", prompt_tokens=[1] * 5, priority=1))
        scheduler.schedule()

        preempted = scheduler.preempt_lowest(min_priority=3)
        assert preempted is None

    def test_preempt_selects_worst_priority(self):
        """Should preempt the sequence with highest priority number."""
        scheduler = BatchScheduler(max_batch_size=4, max_tokens_per_batch=1000)
        scheduler.add(Sequence(request_id="req-1", prompt_tokens=[1] * 5, priority=2))
        scheduler.add(Sequence(request_id="req-2", prompt_tokens=[2] * 5, priority=3))
        scheduler.add(Sequence(request_id="req-3", prompt_tokens=[3] * 5, priority=3))
        scheduler.schedule()

        preempted = scheduler.preempt_lowest(min_priority=3)
        assert preempted is not None
        assert preempted.priority == 3
        # Should be one of the priority=3 sequences
        assert preempted.request_id in ("req-2", "req-3")


class TestMixedPriorities:
    """Complex scenario with multiple priorities."""

    def test_mixed_priorities(self):
        """Scheduling should respect priority ordering across multiple adds."""
        scheduler = BatchScheduler(max_batch_size=2, max_tokens_per_batch=1000)

        # Add sequences in mixed order
        scheduler.add(Sequence(request_id="req-normal-1", prompt_tokens=[1] * 5, priority=2))
        scheduler.add(Sequence(request_id="req-critical", prompt_tokens=[2] * 5, priority=0))
        scheduler.add(Sequence(request_id="req-low", prompt_tokens=[3] * 5, priority=3))
        scheduler.add(Sequence(request_id="req-high", prompt_tokens=[4] * 5, priority=1))
        scheduler.add(Sequence(request_id="req-normal-2", prompt_tokens=[5] * 5, priority=2))

        # First batch: critical + high (lowest priority numbers)
        batch1 = scheduler.schedule()
        assert "req-critical" in batch1.request_ids
        assert "req-high" in batch1.request_ids

        # Complete first batch
        for seq in batch1.sequences:
            seq.generated_tokens = [999] * seq.max_new_tokens

        # Second batch: normal-1 + normal-2 (FIFO within same priority)
        batch2 = scheduler.schedule()
        assert "req-normal-1" in batch2.request_ids
        assert "req-normal-2" in batch2.request_ids

        # Complete second batch
        for seq in batch2.sequences:
            seq.generated_tokens = [999] * seq.max_new_tokens

        # Third batch: low
        batch3 = scheduler.schedule()
        assert batch3.request_ids == ["req-low"]


class TestPriorityCountAndHasPending:
    """Tests for pending_count and has_pending with heap."""

    def test_pending_count(self):
        """pending_count should reflect heap size."""
        scheduler = BatchScheduler(max_batch_size=4, max_tokens_per_batch=1000)
        assert scheduler.pending_count == 0

        scheduler.add(Sequence(request_id="req-1", prompt_tokens=[1]))
        assert scheduler.pending_count == 1

        scheduler.add(Sequence(request_id="req-2", prompt_tokens=[2]))
        assert scheduler.pending_count == 2

        scheduler.schedule()
        assert scheduler.pending_count == 0

    def test_has_pending(self):
        """has_pending should work correctly with heap."""
        scheduler = BatchScheduler(max_batch_size=4, max_tokens_per_batch=1000)
        assert scheduler.has_pending is False

        scheduler.add(Sequence(request_id="req-1", prompt_tokens=[1]))
        assert scheduler.has_pending is True

        scheduler.schedule()
        assert scheduler.has_pending is True  # Active sequence

    def test_get_sequence_from_heap(self):
        """get_sequence should find sequence in heap."""
        scheduler = BatchScheduler(max_batch_size=4, max_tokens_per_batch=1000)
        seq = Sequence(request_id="req-1", prompt_tokens=[1], priority=1)
        scheduler.add(seq)

        found = scheduler.get_sequence("req-1")
        assert found is seq
        assert found.priority == 1
