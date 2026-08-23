"""Regression tests for the C4 release-blocker.

``PipelinedSpeculativeDecoder`` accepted every draft token unverified: slots
were enqueued with ``accepted=None`` and ``_collect_verifications`` treated
anything ``not False`` as accepted, so the verifier never ran (``verify_calls``
stayed 0).  The verifier is now wired: slots are verified before acceptance, a
configured verifier is fail-safe (a slot it cannot verify is rejected), and
``generate()`` falls back to a target-only token so a rejected draft cannot
stall the decoder.
"""

import pytest
import torch

from distllm.core.async_pipelined_speculative import (
    DraftSlot,
    PipelinedSpeculativeDecoder,
)


def _make_decoder(verifier, device="cpu"):
    return PipelinedSpeculativeDecoder(
        target_forward=_target_forward,
        draft_generator=_draft_generator,
        verifier=verifier,
        ring_buffer_depth=4,
        num_verifier_workers=1,
        num_candidates=2,
        device=device,
        use_cuda_streams=False,
    )


def _target_forward(generated, **kwargs):
    """Always predict token 5 at the last position (vocab=8)."""
    seq = generated.shape[1]
    logits = torch.zeros(1, seq, 8)
    logits[0, :, 5] = 10.0
    return logits


def _draft_generator(prompt, n):
    # Draft proposes token 7, which the target never predicts.
    return ([7] * n, [0.5] * n)


def test_drafts_are_not_accepted_unverified():
    """A configured verifier must reject unverifiable drafts (no token 7 out)."""
    decoder = _make_decoder(verifier=lambda hs, cl: [False])
    try:
        out = decoder.generate(torch.tensor([[0]]), max_new_tokens=3)
        # Progress guaranteed: input(1) + 3 new tokens.
        assert out.shape[1] == 4
        # Draft token 7 must never be emitted when verification cannot accept.
        assert 7 not in out[0].tolist()
        # Only target-predicted token 5 appears.
        assert out[0, 1:].tolist() == [5, 5, 5]
    finally:
        decoder.close()


def test_verify_worker_accepts_when_no_verifier():
    """No verifier configured -> built-in greedy check gates acceptance
    (F-040: blind accept removed).  A matching draft passes; a mismatching
    or input-less draft is rejected."""
    decoder = _make_decoder(verifier=None)
    try:
        # No verification inputs captured -> fail-safe rejection.
        slot = DraftSlot(token_ids=[7, 7], logprobs=[0.5, 0.5])
        slot = decoder._verify_worker(slot)
        assert slot.accepted is False

        # Captured target logits whose argmax matches -> accepted.
        logits = torch.full((1, 2, 8), -10.0)
        logits[0, :, 7] = 10.0
        good = DraftSlot(token_ids=[7, 7], logprobs=[0.5, 0.5],
                         hidden_states=logits.clone(),
                         compressed_logits=logits.clone())
        good = decoder._verify_worker(good)
        assert good.accepted is True

        # Captured logits whose argmax disagrees -> rejected.
        logits[:, :, 7] = -10.0
        logits[:, :, 5] = 10.0
        bad = DraftSlot(token_ids=[7, 7], logprobs=[0.5, 0.5],
                        hidden_states=logits,
                        compressed_logits=logits)
        bad = decoder._verify_worker(bad)
        assert bad.accepted is False
    finally:
        decoder.close()


def test_verify_worker_rejects_unverifiable_slot():
    """A configured verifier must not accept a slot it cannot verify."""
    decoder = _make_decoder(verifier=lambda hs, cl: [True, True])
    try:
        slot = DraftSlot(token_ids=[7, 7], logprobs=[0.5, 0.5], accepted=None)
        slot = decoder._verify_worker(slot)
        assert slot.accepted is False
        assert decoder.stats["verify_calls"] > 0
    finally:
        decoder.close()


def test_verify_worker_honors_verifier():
    """With verification inputs present, the verifier's decisions gate."""
    decoder = _make_decoder(verifier=lambda hs, cl: [True, False])
    try:
        slot = DraftSlot(
            token_ids=[7, 7],
            logprobs=[0.5, 0.5],
            hidden_states=torch.zeros(1, 4),
            compressed_logits=torch.zeros(1, 4),
            accepted=None,
        )
        slot = decoder._verify_worker(slot)
        # all() must pass; one False => reject.
        assert slot.accepted is False
        assert decoder.stats["verify_calls"] >= 1
    finally:
        decoder.close()


def test_verify_worker_accepts_when_all_pass():
    decoder = _make_decoder(verifier=lambda hs, cl: [True, True])
    try:
        slot = DraftSlot(
            token_ids=[7, 7],
            logprobs=[0.5, 0.5],
            hidden_states=torch.zeros(1, 4),
            compressed_logits=torch.zeros(1, 4),
            accepted=None,
        )
        slot = decoder._verify_worker(slot)
        assert slot.accepted is True
    finally:
        decoder.close()