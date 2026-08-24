"""Regression tests for C1: chunked-prefill sequences hung forever on prefix-cache hits.

``ChunkedPrefillInfo(total_prompt_tokens=...)`` used to be seeded with the
FULL prompt length even though a prefix-cache hit makes processing start at
``seq.prefix_match_len`` — so the maximum attainable ``tokens_processed``
was ``len(prompt) - prefix_match_len < total_prompt_tokens`` and
``is_complete()`` could never become true.  The sequence stayed in
PREFILLING forever, holding its slot and budget.

The fix charges the chunk state with only the *remaining* work
(``total_len - prefix_match_len``) and handles fully-cached prompts by
priming decode with the final prompt token.

Run: pytest tests/core/test_chunked_prefill_prefix_hit.py -v
"""

import torch

from distllm.core.batch_scheduler import BatchScheduler, Sequence, SequenceStatus


def _run_scheduler(scheduler: BatchScheduler, seq: Sequence, max_iters: int = 100):
    """Drive schedule()/step() until the sequence completes or max_iters."""
    prefill_widths: list[int] = []
    prefill_offsets: list[int] = []
    saw_decode = False
    for _ in range(max_iters):
        batch = scheduler.schedule()
        assert batch is not None, "scheduler returned None while sequence active"
        if batch.is_prefill[0]:
            prefill_widths.append(int(batch.seq_lengths[0]))
            prefill_offsets.append(int(batch.position_offsets[0]))
        else:
            saw_decode = True
        next_tokens = torch.full((len(batch.sequences),), 7, dtype=torch.long)
        scheduler.step(batch, next_tokens)
        if seq.is_complete:
            return prefill_widths, prefill_offsets, saw_decode
    raise AssertionError(
        f"Sequence did not complete within {max_iters} iterations "
        f"(status={seq.status}, generated={len(seq.generated_tokens)}, "
        f"chunk_state={scheduler._chunked_prefill.get(seq.request_id)})"
    )


class TestChunkedPrefillPrefixHit:
    """C1: prefix hit + chunked prefill must complete and reach decoding."""

    def test_partial_prefix_hit_completes_and_decodes(self):
        scheduler = BatchScheduler(
            max_batch_size=4, max_tokens_per_batch=4096,
            enable_chunked_prefill=True, max_prefill_tokens=64,
        )
        # 200-token prompt with a 50-token cache hit -> 150 tokens to prefill.
        seq = Sequence(
            request_id="c1", prompt_tokens=list(range(200)),
            max_new_tokens=5, prefix_match_len=50,
        )
        scheduler.add(seq)

        widths, offsets, saw_decode = _run_scheduler(scheduler, seq)

        assert seq.is_complete
        assert len(seq.generated_tokens) == 5
        # Exactly the unmatched suffix was prefetched: 150 tokens total.
        assert sum(widths) == 150
        assert widths == [64, 64, 22]
        # Chunk cursor starts after the matched prefix.
        assert offsets == [50, 114, 178]
        # The final prefill chunk transitions into real decode steps.
        assert saw_decode
        assert seq.request_id not in scheduler._chunked_prefill
        assert seq.request_id not in scheduler.active

    def test_chunk_state_accounts_for_matched_prefix(self):
        """White-box: ChunkedPrefillInfo must charge only remaining work."""
        scheduler = BatchScheduler(
            max_batch_size=4, max_tokens_per_batch=4096,
            enable_chunked_prefill=True, max_prefill_tokens=64,
        )
        seq = Sequence(
            request_id="c1", prompt_tokens=list(range(200)),
            max_new_tokens=2, prefix_match_len=50,
        )
        scheduler.add(seq)
        batch = scheduler.schedule()
        assert batch is not None
        cinfo = scheduler._chunked_prefill["c1"]
        # Before the fix this was 200 (full total_len) => never complete.
        # _build_seq_tokens consumed the first 64-token chunk during
        # schedule() itself.
        assert cinfo.total_prompt_tokens == 150
        assert cinfo.tokens_processed == 64
        assert cinfo.remaining == 86
        assert cinfo.chunks_remaining == 2
        next_tokens = torch.full((len(batch.sequences),), 7, dtype=torch.long)
        scheduler.step(batch, next_tokens)  # sampling does not move the cursor

        batch = scheduler.schedule()
        assert batch is not None
        # Second chunk consumed: 128 of 150 processed, 22 left.
        assert scheduler._chunked_prefill["c1"].remaining == 22
        next_tokens = torch.full((len(batch.sequences),), 7, dtype=torch.long)
        scheduler.step(batch, next_tokens)

    def test_full_prefix_hit_primes_decode_without_crash(self):
        """A fully-matched long prompt creates no chunk state and decodes.

        Pre-fix this either wedged (chunk state that could never complete)
        or crashed on ``decode_input_token`` of an empty generated list.
        """
        scheduler = BatchScheduler(
            max_batch_size=4, max_tokens_per_batch=4096,
            enable_chunked_prefill=True, max_prefill_tokens=64,
        )
        seq = Sequence(
            request_id="c2", prompt_tokens=list(range(200)),
            max_new_tokens=3, prefix_match_len=200,
        )
        scheduler.add(seq)

        widths, offsets, saw_decode = _run_scheduler(scheduler, seq)

        assert seq.is_complete
        assert len(seq.generated_tokens) == 3
        # Nothing to prefetch; the first iteration primes with ONLY the
        # final prompt token so the model can produce first-token logits.
        assert widths == [1]
        assert offsets == [199]
        assert saw_decode
        assert "c2" not in scheduler._chunked_prefill


class TestFreshSequencePrefixHitNoChunking:
    """Partial hits on short prompts (no chunking) must prefill the suffix."""

    def test_partial_hit_short_prompt_prefills_suffix(self):
        scheduler = BatchScheduler(max_batch_size=4, max_tokens_per_batch=4096)
        seq = Sequence(
            request_id="c3", prompt_tokens=list(range(20)),
            max_new_tokens=2, prefix_match_len=10,
        )
        scheduler.add(seq)

        batch = scheduler.schedule()
        assert batch is not None
        assert batch.is_prefill[0]
        # Only the unmatched suffix is emitted (was an IndexError before:
        # fresh seqs with prefix_match_len > 0 fell through to the decode
        # branch and hit decode_input_token on an empty generated list).
        assert int(batch.seq_lengths[0]) == 10
        assert int(batch.position_offsets[0]) == 10
        assert seq.status == SequenceStatus.PREFILLING

        next_tokens = torch.full((1,), 7, dtype=torch.long)
        scheduler.step(batch, next_tokens)
        assert len(seq.generated_tokens) == 1

        # Second iteration decodes normally.
        batch = scheduler.schedule()
        assert batch is not None
        assert not batch.is_prefill[0]

    def test_no_prefix_hit_still_prefills_full_prompt(self):
        """Guard: the restructured prefill branch keeps the plain path intact."""
        scheduler = BatchScheduler(max_batch_size=4, max_tokens_per_batch=4096)
        seq = Sequence(request_id="c4", prompt_tokens=list(range(20)), max_new_tokens=2)
        scheduler.add(seq)

        batch = scheduler.schedule()
        assert batch is not None
        assert batch.is_prefill[0]
        assert int(batch.seq_lengths[0]) == 20
        assert int(batch.position_offsets[0]) == 0
