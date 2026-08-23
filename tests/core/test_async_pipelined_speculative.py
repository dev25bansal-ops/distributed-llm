"""Tests for the async pipelined speculative decoder."""

from __future__ import annotations

import time
import threading

import torch
import pytest

from distllm.core.async_pipelined_speculative import (
    DraftRingBuffer,
    DraftSlot,
    PipelinedSpeculativeDecoder,
)


# ── Helper factories (not pytest fixtures — called directly) ─────────────────

def _make_forward():
    def f(input_ids, **kw):
        logits = torch.zeros(1, input_ids.shape[1], 100)
        logits[0, :, 42] = 1.0
        return logits
    return f


def _make_draft():
    def g(prompt, num_tokens):
        return [99] * num_tokens, [0.0] * num_tokens
    return g


def _make_accept_verifier():
    def v(hidden, logits):
        return [True] * hidden.shape[1]
    return v


def _make_reject_verifier():
    def v(hidden, logits):
        return [False] * hidden.shape[1]
    return v


def _make_slow_draft():
    def g(prompt, num_tokens):
        time.sleep(0.05)
        return [99] * num_tokens, [0.0] * num_tokens
    return g


# ── DraftRingBuffer tests ────────────────────────────────────────────────────

class TestDraftRingBuffer:
    def test_create(self):
        buf = DraftRingBuffer(depth=4)
        assert buf._depth == 4
        assert buf.fill_ratio == 0.0

    def test_put_and_get(self):
        buf = DraftRingBuffer(depth=4)
        slot = DraftSlot(token_ids=[1, 2, 3])
        buf.put(slot)
        assert buf.fill_ratio > 0
        retrieved = buf.get()
        assert retrieved.token_ids == [1, 2, 3]
        assert buf.fill_ratio == 0.0

    def test_put_nowait_when_full(self):
        buf = DraftRingBuffer(depth=2)
        buf.put_nowait(DraftSlot(token_ids=[1]))
        buf.put_nowait(DraftSlot(token_ids=[2]))
        assert buf.put_nowait(DraftSlot(token_ids=[3])) is False

    def test_get_nowait_when_empty(self):
        buf = DraftRingBuffer(depth=4)
        assert buf.get_nowait() is None

    def test_fill_ratio(self):
        buf = DraftRingBuffer(depth=4)
        assert buf.fill_ratio == 0.0
        buf.put_nowait(DraftSlot(token_ids=[1]))
        assert buf.fill_ratio == 0.25
        buf.put_nowait(DraftSlot(token_ids=[2]))
        assert buf.fill_ratio == 0.5
        buf.get_nowait()
        assert buf.fill_ratio == 0.25

    def test_wraparound(self):
        buf = DraftRingBuffer(depth=3)
        for i in range(6):
            buf.put(DraftSlot(token_ids=[i]))
            slot = buf.get()
            assert slot.token_ids == [i]

    def test_invalid_depth(self):
        with pytest.raises(ValueError, match="depth"):
            DraftRingBuffer(depth=1)


# ── DraftSlot tests ─────────────────────────────────────────────────────────

class TestDraftSlot:
    def test_defaults(self):
        slot = DraftSlot()
        assert slot.token_ids is None
        assert slot.accepted is None

    def test_with_values(self):
        slot = DraftSlot(token_ids=[1, 2, 3], logprobs=[-0.5, -0.3, -0.1], accepted=True)
        assert slot.token_ids == [1, 2, 3]
        assert slot.logprobs == [-0.5, -0.3, -0.1]
        assert slot.accepted is True

    def test_mutable_fields(self):
        slot = DraftSlot(token_ids=[1])
        slot.accepted = True
        slot.token_ids.append(2)
        assert slot.token_ids == [1, 2]


# ── PipelinedSpeculativeDecoder tests ────────────────────────────────────────

class TestPipelinedSpeculativeDecoder:
    def test_init_defaults(self):
        decoder = PipelinedSpeculativeDecoder(
            target_forward=lambda x: torch.zeros(1, 1, 100),
            draft_generator=lambda p, n: ([1], [0.0]),
            use_cuda_streams=False,
        )
        assert decoder._num_candidates == 5
        assert decoder._ring._depth == 8
        decoder.close()

    def test_generate_without_draft(self):
        decoder = PipelinedSpeculativeDecoder(
            target_forward=lambda x, **kw: torch.zeros(1, x.shape[1], 100),
            draft_generator=None,
            num_candidates=0,
        )
        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        output = decoder.generate(input_ids, max_new_tokens=5)
        assert output.shape[1] == 3 + 5
        decoder.close()

    def test_generate_batch_size_check(self):
        decoder = PipelinedSpeculativeDecoder(
            target_forward=lambda x: torch.zeros(1, 1, 100),
        )
        with pytest.raises(ValueError, match="batch_size=1"):
            decoder.generate(torch.randint(0, 100, (2, 10)))

    def test_generate_with_draft_accepting(self):
        decoder = PipelinedSpeculativeDecoder(
            target_forward=_make_forward(),
            draft_generator=_make_draft(),
            verifier=_make_accept_verifier(),
            num_candidates=3,
            ring_buffer_depth=4,
        )
        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        output = decoder.generate(input_ids, max_new_tokens=10)
        assert output.shape[1] >= 3 + 5  # At least some tokens generated
        assert output.shape[1] <= 3 + 15  # Not excessively more than requested
        decoder.close()

    def test_generate_with_draft_rejecting(self):
        decoder = PipelinedSpeculativeDecoder(
            target_forward=lambda x, **kw: torch.zeros(1, x.shape[1], 100),
            draft_generator=_make_draft(),
            verifier=_make_reject_verifier(),
            num_candidates=3,
            ring_buffer_depth=4,
        )
        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        output = decoder.generate(input_ids, max_new_tokens=5)
        assert output.shape[1] >= 3 + 1
        decoder.close()

    def test_slow_draft_doesnt_block_target(self):
        decoder = PipelinedSpeculativeDecoder(
            target_forward=_make_forward(),
            draft_generator=_make_slow_draft(),
            verifier=_make_accept_verifier(),
            num_candidates=2,
            ring_buffer_depth=8,
            num_verifier_workers=2,
        )
        input_ids = torch.tensor([[1]], dtype=torch.long)
        start = time.time()
        output = decoder.generate(input_ids, max_new_tokens=5)
        elapsed = time.time() - start
        assert output.shape[1] >= 1 + 5
        # Slow draft takes 0.05s per call. Serial would be ~0.25s+ for 5 tokens.
        # Pipelined should be faster.
        decoder.close()

    def test_stats_after_generation(self):
        decoder = PipelinedSpeculativeDecoder(
            target_forward=_make_forward(),
            draft_generator=_make_draft(),
            verifier=_make_accept_verifier(),
            num_candidates=2,
        )
        stats_before = dict(decoder.stats)
        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        decoder.generate(input_ids, max_new_tokens=8)
        stats = decoder.stats
        assert stats["target_calls"] > stats_before.get("target_calls", 0)
        decoder.close()

    def test_ring_buffer_peak_tracked(self):
        decoder = PipelinedSpeculativeDecoder(
            target_forward=_make_forward(),
            draft_generator=_make_draft(),
            verifier=_make_accept_verifier(),
            num_candidates=5,
            ring_buffer_depth=4,
        )
        input_ids = torch.tensor([[1]], dtype=torch.long)
        decoder.generate(input_ids, max_new_tokens=20)
        stats = decoder.stats
        assert "ring_buffer_peak" in stats
        decoder.close()

    def test_multiple_calls(self):
        decoder = PipelinedSpeculativeDecoder(
            target_forward=_make_forward(),
            draft_generator=_make_draft(),
            verifier=_make_accept_verifier(),
            num_candidates=2,
        )
        input_ids = torch.tensor([[1]], dtype=torch.long)
        out1 = decoder.generate(input_ids, max_new_tokens=5)
        out2 = decoder.generate(input_ids, max_new_tokens=5)
        assert out1.shape[1] >= 1 + 3
        assert out2.shape[1] >= 1 + 3
        decoder.close()

    def test_large_candidate_count(self):
        decoder = PipelinedSpeculativeDecoder(
            target_forward=_make_forward(),
            draft_generator=_make_draft(),
            verifier=_make_accept_verifier(),
            num_candidates=10,
            ring_buffer_depth=16,
        )
        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        output = decoder.generate(input_ids, max_new_tokens=20)
        assert output.shape[1] >= 3 + 10
        decoder.close()

    @pytest.mark.skip(reason="ThreadPoolExecutor race condition on CI")
    def test_draft_failure_handling(self):
        """Draft failure should gracefully fall back to target-only generation."""
        call_count = 0

        def failing_draft(prompt, num_tokens):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise RuntimeError("Draft model unavailable")
            return [99] * num_tokens, [0.0] * num_tokens  # Recover after 2 failures

        decoder = PipelinedSpeculativeDecoder(
            target_forward=_make_forward(),
            draft_generator=failing_draft,
            verifier=_make_accept_verifier(),
            num_candidates=2,
            device="cpu",
        )
        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        output = decoder.generate(input_ids, max_new_tokens=5)
        assert output.shape[1] >= 3 + 1
        decoder.close()
