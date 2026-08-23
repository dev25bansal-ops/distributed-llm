"""Regression tests for audit finding F-040.

``PipelinedSpeculativeDecoder`` never populated ``DraftSlot.hidden_states`` /
``compressed_logits``: ``_draft_worker`` built slots from the draft generator's
``(token_ids, logprobs)`` only.  With a configured verifier every draft was
therefore rejected (zero speculative speedup), and with ``verifier=None``
(the constructor default) every draft was accepted *unverified* — pure draft
output, not speculative decoding.

Fix under test:

1. ``_draft_worker`` feeds the draft tokens through the target model and
   stores the resulting logits (and hidden states, when available) on the
   slot.
2. With no external verifier, a built-in greedy check accepts a draft only
   when its tokens match the target argmax at each draft position.
3. A slot that carries no verification inputs is rejected fail-safe in both
   configurations — unverified draft tokens are never emitted.
"""

import time

import torch

from distllm.core.async_pipelined_speculative import (
    DraftSlot,
    PipelinedSpeculativeDecoder,
    _unpack_target_output,
)

VOCAB = 8


def _make_decoder(verifier=None, draft=None, target=None):
    return PipelinedSpeculativeDecoder(
        target_forward=target or _target_forward,
        draft_generator=draft or _draft_generator,
        verifier=verifier,
        ring_buffer_depth=4,
        num_verifier_workers=1,
        num_candidates=3,
        device="cpu",
        use_cuda_streams=False,
    )


def _target_forward(generated, **kwargs):
    """Target always predicts token 5 at every position (vocab=8)."""
    seq = generated.shape[1]
    logits = torch.zeros(1, seq, VOCAB)
    logits[0, :, 5] = 10.0
    return logits


def _draft_generator(prompt, n):
    # Draft proposes token 7, which the target never predicts -> must be
    # rejected by any real verification.
    return ([7] * n, [0.5] * n)


def _drain_ring(decoder):
    """Verify any slots left unconsumed in the ring buffer."""
    while True:
        slot = decoder._ring.get_nowait()
        if slot is None:
            return
        if slot.accepted is None:
            decoder._verify_worker(slot)


# ── _draft_worker populates verification inputs ──────────────────────────────


class TestDraftWorkerPopulatesVerifierInputs:
    def test_pipeline_slots_carry_logits(self):
        """Slots produced by the real pipeline must carry captured logits."""
        decoder = _make_decoder(verifier=lambda hs, cl: [True])
        try:
            decoder._launch_draft(torch.tensor([[1, 2, 3]]), 2)
            slot = decoder._ring.get()  # blocks until the worker wrote it
            assert slot.token_ids == [7, 7]
            assert slot.compressed_logits is not None
            # Logits for the positions of the 2 draft tokens, full vocab.
            assert tuple(slot.compressed_logits.shape) == (1, 2, VOCAB)
            assert slot.accepted is None  # not yet verified at enqueue time
        finally:
            decoder.close()

    def test_pipeline_slots_carry_hidden_states_when_target_returns_them(self):
        def target_with_hs(generated, **kwargs):
            logits = _target_forward(generated)
            hidden = torch.zeros(1, generated.shape[1], 4)
            return logits, hidden

        decoder = _make_decoder(
            verifier=lambda hs, cl: [True], target=target_with_hs,
        )
        try:
            decoder._launch_draft(torch.tensor([[1, 2, 3]]), 2)
            slot = decoder._ring.get()
            assert slot.hidden_states is not None
            assert slot.compressed_logits is not None
        finally:
            decoder.close()

    def test_generate_end_to_end_verifier_sees_real_inputs(self):
        """A configured verifier receives non-None inputs; its rejections
        mean draft tokens never leak into the output."""
        seen = []

        def verifier(hs, cl):
            seen.append((hs is not None, cl is not None))
            return [False]

        decoder = _make_decoder(verifier=verifier)
        try:
            out = decoder.generate(torch.tensor([[1]]), max_new_tokens=2)
            _drain_ring(decoder)  # verify any slots the loop never consumed
            assert out.shape[1] == 3
            # The verifier was actually invoked with captured inputs.
            assert seen and all(cl for _, cl in seen)
            # Draft token 7 never emitted; target token 5 fills the output.
            assert out[0, 1:].tolist() == [5, 5]
        finally:
            decoder.close()


# ── No-verifier mode is now verified (built-in greedy check) ────────────────


class TestNoVerifierModeIsVerified:
    def test_mismatched_draft_rejected_without_verifier(self):
        """verifier=None must NOT blindly accept: 7 != target argmax 5."""
        decoder = _make_decoder(verifier=None)
        try:
            out = decoder.generate(torch.tensor([[1]]), max_new_tokens=3)
            _drain_ring(decoder)
            assert out.shape[1] == 4
            assert out[0, 1:].tolist() == [5, 5, 5]
            assert 7 not in out[0].tolist()
            assert decoder.stats["verify_calls"] > 0
        finally:
            decoder.close()

    def test_matching_draft_accepted_without_verifier(self):
        """Draft agreeing with the target argmax passes the greedy check."""
        decoder = _make_decoder(
            verifier=None,
            draft=lambda prompt, n: ([5] * n, [0.9] * n),
        )
        try:
            decoder._running = True
            decoder._launch_draft(torch.tensor([[1]]), 2)
            accepted: list[int] = []
            deadline = time.time() + 5.0
            while not accepted and time.time() < deadline:
                decoder._collect_verifications(accepted)
                if not accepted:
                    time.sleep(0.005)
            assert accepted == [5, 5]
        finally:
            decoder.close()

    def test_verify_worker_greedy_check_direct(self):
        decoder = _make_decoder(verifier=None)
        try:
            logits = torch.zeros(1, 2, VOCAB)
            logits[0, :, 5] = 10.0

            bad = DraftSlot(token_ids=[7, 7], compressed_logits=logits.clone())
            bad = decoder._verify_worker(bad)
            assert bad.accepted is False

            good = DraftSlot(token_ids=[5, 5], compressed_logits=logits.clone())
            good = decoder._verify_worker(good)
            assert good.accepted is True
        finally:
            decoder.close()


# ── Fail-safe rejection when inputs are missing ──────────────────────────────


class TestFailSafeRejection:
    def test_verifier_configured_no_inputs_rejected(self):
        decoder = _make_decoder(verifier=lambda hs, cl: [True])
        try:
            slot = DraftSlot(token_ids=[7], logprobs=[0.5])
            slot = decoder._verify_worker(slot)
            assert slot.accepted is False
        finally:
            decoder.close()

    def test_no_verifier_no_inputs_rejected(self):
        """Neither verifier nor captured logits -> reject, never accept blind."""
        decoder = _make_decoder(verifier=None)
        try:
            slot = DraftSlot(token_ids=[7], logprobs=[0.5])
            slot = decoder._verify_worker(slot)
            assert slot.accepted is False
        finally:
            decoder.close()


# ── Helper ───────────────────────────────────────────────────────────────────


class TestUnpackTargetOutput:
    def test_bare_logits(self):
        logits = torch.zeros(1, 2, VOCAB)
        out_logits, hidden = _unpack_target_output(logits)
        assert out_logits is logits
        assert hidden is None

    def test_tuple(self):
        logits = torch.zeros(1, 2, VOCAB)
        hidden = torch.zeros(1, 2, 3)
        out_logits, out_hidden = _unpack_target_output((logits, hidden))
        assert out_logits is logits
        assert out_hidden is hidden

    def test_hf_style_object(self):
        class FakeOut:
            logits = torch.zeros(1, 2, VOCAB)
            hidden_states = None

        out_logits, hidden = _unpack_target_output(FakeOut())
        assert out_logits is FakeOut.logits
        assert hidden is None
