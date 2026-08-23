"""Regression test for F-041: completed sequences must release their paged KV blocks.

Root cause: the batch-decode completion path in
``BatchScheduler.step()`` / ``_record_step_metrics`` pruned completed
sequences from ``self.active`` without calling ``free_paged_blocks``.
Only ``_prefetch_and_snapshot`` (run on the *next* ``schedule()``) freed
completed sequences, but sequences that finished during the decode step
were already removed from ``self.active`` and so were never seen by that
path — their blocks leaked from the pool permanently, causing unbounded
memory growth.

Hermetic: uses a fake PagedAttention manager (CPU only, no GPU).
"""

import torch

from distllm.core.batch_scheduler import BatchScheduler, ScheduledBatch
from distllm.core.scheduler.sequence import Sequence, SequenceStatus


class _FakePagedAttention:
    """Tracks per-sequence block allocation and a bounded free pool."""

    def __init__(self, block_size: int = 16):
        self.block_size = block_size
        self._free = 256
        self._allocated: dict[str, list[int]] = {}

    @property
    def free_count(self) -> int:
        return self._free

    def allocate_sequence(self, request_id: str, num_tokens: int) -> list[int]:
        blocks = list(range((num_tokens + self.block_size - 1) // self.block_size))
        self._free -= len(blocks)
        self._allocated[request_id] = blocks
        return blocks

    def free_sequence(self, request_id: str) -> None:
        blocks = self._allocated.pop(request_id, [])
        self._free += len(blocks)


def _make_batch(seq: Sequence) -> ScheduledBatch:
    return ScheduledBatch(
        sequences=[seq],
        input_ids=torch.tensor([[1, 2, 3]]),
        seq_lengths=[3],
        position_offsets=[0],
        is_prefill=[True],
        request_ids=[seq.request_id],
    )


def test_completed_sequence_releases_blocks():
    """A sequence that finishes during the decode step frees its blocks."""
    stub = _FakePagedAttention()
    scheduler = BatchScheduler(
        max_batch_size=4,
        max_tokens_per_batch=256,
        paged_attention_mgr=stub,
    )

    seq = Sequence(
        request_id="req-1",
        prompt_tokens=[1, 2, 3],
        max_new_tokens=10,
        stop_token_ids=[0],
    )
    scheduler.active["req-1"] = seq

    # Allocate paged blocks for the sequence (as a schedule/build would).
    free_before = stub.free_count
    scheduler.allocate_paged_blocks(seq)
    # Blocks were actually taken (3 prompt + 10 max_new = 13 -> one block).
    assert stub.free_count < free_before

    # Complete the sequence inside the decode step via a stop token.
    batch = _make_batch(seq)
    scheduler.step(batch, next_tokens=torch.tensor([0]))

    assert seq.status == SequenceStatus.DONE
    # F-041: the completed sequence's blocks must be returned to the pool —
    # the count returns to the PRE-allocation level.
    assert stub.free_count == free_before
    assert "req-1" not in scheduler.active


def test_multiple_completed_sequences_bounded_pool():
    """Repeatedly completing sequences does not exhaust the pool (no leak)."""
    stub = _FakePagedAttention(block_size=8)
    scheduler = BatchScheduler(
        max_batch_size=16,
        max_tokens_per_batch=1024,
        paged_attention_mgr=stub,
        enable_chunked_prefill=False,
    )

    free_before = stub.free_count
    for i in range(5):
        rid = f"req-{i}"
        seq = Sequence(
            request_id=rid,
            prompt_tokens=list(range(i + 1, i + 6)),
            max_new_tokens=10,
            stop_token_ids=[0],
        )
        scheduler.active[rid] = seq
        scheduler.allocate_paged_blocks(seq)
        scheduler.step(_make_batch(seq), next_tokens=torch.tensor([0]))
        assert stub.free_count == free_before, f"leak after completing {rid}"

    # Pool is fully restored after all sequences complete.
    assert stub.free_count == free_before
