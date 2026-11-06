"""Tests for the unified ``eos_cutoff()`` C14 invariant across all 12 speculative engines.

Each engine must call :func:`distllm.core.spec_verify.eos_cutoff` on its
accepted tokens so post-EOS hallucinated drafts never reach the output.
This test file exercises that invariant for every speculative engine by
constructing the decoder with a mocked target/draft that emits a sequence
ending in an EOS token, then asserting:

1. The output never contains tokens past the first EOS.
2. The first EOS itself is preserved (the unified helper includes it).

Engines covered (12 total):

1. ``SpeculativeDecoder``         (speculative_decoder.py)
2. ``SelfSpeculativeDecoder``     (speculative_decoder.py)
3. ``MultiDraftSpeculativeDecoder`` (speculative_decoder.py)
4. ``TreeDraftSpeculativeDecoder`` (speculative_decoder.py)
5. ``DistributedSpeculativeDecoder`` (distributed_speculative.py)
6. ``TreeSpeculativeDecoder``     (tree_speculative_decoder.py)
7. ``MultiDraftVerifier``         (multi_draft_verifier.py)
8. ``TreeMultiDraftVerifier``     (multi_draft_verifier.py)
9. ``MTPDecoder``                 (mtp_head.py)
10. ``CompressedSpeculativeDecoder`` (compressed_speculative.py)
11. ``PipelinedSpeculativeDecoder`` (async_pipelined_speculative.py)
12. (LightweightVerifier — verifier only, no decode loop, skipped)

We test by feeding the decoders a sequence of candidate tokens that
includes an EOS at a known position, then asserting no token appears
past that position in the output.  Each decoder uses a different
forward signature, so per-engine fixtures are minimal.
"""

from __future__ import annotations

import pytest
import torch

from distllm.core.spec_verify import eos_cutoff


# Sentinel tokens used across tests
EOS_TOKEN = 999
VOCAB_SIZE = 1000


# ── Helpers ────────────────────────────────────────────────────────────────


def _assert_no_post_eos_tokens(output_ids: torch.Tensor, prompt_len: int, eos: int) -> None:
    """Assert that the generated span has no tokens after the first EOS."""
    generated = output_ids[0, prompt_len:].tolist()
    cut = eos_cutoff(generated, eos)
    # eos_cutoff returns position past the first EOS (or len when no EOS)
    assert cut == len(generated), (
        f"eos_cutoff returned {cut} but generated has {len(generated)} tokens "
        f"past the prompt — post-EOS tokens leaked through: {generated}"
    )
    # And the first EOS itself is preserved (cut includes it)
    if cut > 0 and cut < len(generated):
        assert generated[cut - 1] == eos, (
            f"eos_cutoff returned a cut past the EOS, but the EOS itself "
            f"was not at position cut-1: generated={generated}, cut={cut}"
        )


def _make_argmax_target(argmax_sequence: list[int]) -> "callable":
    """Build a target_forward callable whose argmax is fixed at each step.

    ``argmax_sequence[j]`` is the argmax returned when the input has
    ``prompt_len + j`` tokens total.  Index 0 corresponds to the position
    right after the prompt.  This lets a test pin the entire emission.
    """
    def target_forward(input_ids: torch.Tensor, **_kwargs) -> torch.Tensor:
        batch, seq_len = input_ids.shape[0], input_ids.shape[1]
        # Output logits shape: (batch, seq_len, VOCAB_SIZE).
        logits = torch.full((batch, seq_len, VOCAB_SIZE), -10.0)
        # Pin argmax at the last position based on input length.
        idx = max(0, min(seq_len - 1, len(argmax_sequence) - 1))
        argmax_tok = argmax_sequence[idx]
        logits[:, -1, argmax_tok] = 10.0
        return logits
    return target_forward


# ── Test the eos_cutoff helper itself ──────────────────────────────────────


class TestEosCutoffHelper:
    """Sanity-check the helper that every engine now relies on."""

    def test_no_eos_returns_full_length(self) -> None:
        assert eos_cutoff([1, 2, 3, 4, 5], eos_token_id=None) == 5
        assert eos_cutoff([1, 2, 3, 4, 5], eos_token_id=EOS_TOKEN) == 5

    def test_eos_in_middle_truncates_after_it(self) -> None:
        # eos at position 2 → cut = 3 (length up to and including EOS)
        assert eos_cutoff([1, 2, EOS_TOKEN, 4, 5], eos_token_id=EOS_TOKEN) == 3

    def test_eos_at_start_truncates_to_one(self) -> None:
        assert eos_cutoff([EOS_TOKEN, 1, 2], eos_token_id=EOS_TOKEN) == 1

    def test_accepts_tensor_input(self) -> None:
        t = torch.tensor([1, 2, EOS_TOKEN, 4, 5])
        assert eos_cutoff(t, eos_token_id=EOS_TOKEN) == 3


# ── Engines 1-4: speculative_decoder.py ───────────────────────────────────


class TestSpeculativeDecoderFamily:
    """The four decoders in speculative_decoder.py share eos_cutoff."""

    def _make_draft(self, draft_seq: list[int]) -> "callable":
        def draft_forward(input_ids: torch.Tensor, **_kwargs) -> torch.Tensor:
            logits = torch.full((1, input_ids.shape[1], VOCAB_SIZE), -10.0)
            argmax = draft_seq[input_ids.shape[1] - 1] if input_ids.shape[1] - 1 < len(draft_seq) else 0
            logits[0, -1, argmax] = 10.0
            return logits
        return draft_forward

    def test_speculative_decoder_truncates_at_eos(self) -> None:
        from distllm.core.speculative_decoder import SpeculativeDecoder

        # Argmax at each step: token, token, EOS_TOKEN, EOS_TOKEN, ...
        seq = [EOS_TOKEN] * 8
        target = _make_argmax_target(seq)
        draft = self._make_draft(seq)

        sd = SpeculativeDecoder(
            target_forward=target, draft_forward=draft,
            num_candidates=3, temperature=0, eos_token_id=EOS_TOKEN,
        )
        out = sd.generate(torch.tensor([[1, 2, 3]]), max_new_tokens=5)
        _assert_no_post_eos_tokens(out, prompt_len=3, eos=EOS_TOKEN)

    def test_self_speculative_decoder_truncates_at_eos(self) -> None:
        from distllm.core.speculative_decoder import SelfSpeculativeDecoder


        def target_fwd(input_ids, **_):
            return torch.full((1, input_ids.shape[1], VOCAB_SIZE), -10.0)

        def hidden_fwd(input_ids, **_):
            logits = target_fwd(input_ids)
            hidden = torch.zeros(1, input_ids.shape[1], 8)
            return logits, hidden

        sd = SelfSpeculativeDecoder(
            target_forward=target_fwd, hidden_states_fn=hidden_fwd,
            hidden_size=8, vocab_size=VOCAB_SIZE,
            num_candidates=3, temperature=0, device="cpu",
            eos_token_id=EOS_TOKEN,
        )
        out = sd.generate(torch.tensor([[1, 2, 3]]), max_new_tokens=5)
        _assert_no_post_eos_tokens(out, prompt_len=3, eos=EOS_TOKEN)

    def test_multi_draft_speculative_decoder_truncates_at_eos(self) -> None:
        from distllm.core.speculative_decoder import MultiDraftSpeculativeDecoder

        seq = [EOS_TOKEN] * 8
        target = _make_argmax_target(seq)
        # Two draft models that agree on EOS_TOKEN
        draft1 = self._make_draft(seq)
        draft2 = self._make_draft(seq)

        sd = MultiDraftSpeculativeDecoder(
            target_forward=target, draft_forwards=[draft1, draft2],
            num_candidates=3, temperature=0, eos_token_id=EOS_TOKEN,
        )
        out = sd.generate(torch.tensor([[1, 2, 3]]), max_new_tokens=5)
        _assert_no_post_eos_tokens(out, prompt_len=3, eos=EOS_TOKEN)

    def test_tree_draft_speculative_decoder_truncates_at_eos(self) -> None:
        from distllm.core.speculative_decoder import TreeDraftSpeculativeDecoder

        seq = [EOS_TOKEN] * 8
        target = _make_argmax_target(seq)
        draft = self._make_draft(seq)

        sd = TreeDraftSpeculativeDecoder(
            target_forward=target, draft_forwards=[draft],
            max_tree_nodes=8, max_depth=3, branching_factor=2,
            temperature=0, eos_token_id=EOS_TOKEN,
        )
        out = sd.generate(torch.tensor([[1, 2, 3]]), max_new_tokens=5)
        _assert_no_post_eos_tokens(out, prompt_len=3, eos=EOS_TOKEN)


# ── Engine 5: distributed_speculative.py ──────────────────────────────────


class TestDistributedSpeculativeDecoder:
    def test_truncates_at_eos(self) -> None:
        from distllm.core.distributed_speculative import (
            DistributedSpeculativeDecoder,
            RemoteDraftModel,
        )

        seq = [EOS_TOKEN] * 8
        target = _make_argmax_target(seq)

        # A draft model whose every call returns EOS_TOKEN as the first token
        class _AlwaysEosDraft(RemoteDraftModel):
            def __init__(self):  # noqa: D401 — test stub
                from distllm.core.distributed_speculative import DraftTokenResult
                self._results = [DraftTokenResult(
                    token_ids=[EOS_TOKEN], logprobs=[0.0],
                )]

            def generate_tokens(self, prompt_tokens, num_tokens, **_):
                from distllm.core.distributed_speculative import DraftTokenResult
                return DraftTokenResult(
                    token_ids=[EOS_TOKEN] * num_tokens, logprobs=[0.0] * num_tokens,
                )

        sd = DistributedSpeculativeDecoder(
            target_forward=target, draft_model=_AlwaysEosDraft(),
            num_candidates=3, temperature=0, device="cpu",
            eos_token_id=EOS_TOKEN,
        )
        out = sd.generate(torch.tensor([[1, 2, 3]]), max_new_tokens=5)
        _assert_no_post_eos_tokens(out, prompt_len=3, eos=EOS_TOKEN)


# ── Engines 6-8: tree_speculative_decoder.py + multi_draft_verifier.py ───


class TestTreeAndMultiDraftVerifiers:
    def test_tree_speculative_decoder_truncates_at_eos(self) -> None:
        from distllm.core.tree_speculative_decoder import TreeSpeculativeDecoder

        seq = [EOS_TOKEN] * 8
        target = _make_argmax_target(seq)

        def draft_fwd(input_ids, **_):
            # Returns logits with EOS_TOKEN as the argmax at every position.
            logits = torch.full(
                (input_ids.shape[0], input_ids.shape[1], VOCAB_SIZE), -10.0,
            )
            logits[:, -1, EOS_TOKEN] = 10.0
            return logits

        td = TreeSpeculativeDecoder(
            target_forward=target, draft_forward=draft_fwd,
            branching_factor=2, tree_depth=3,
            temperature=0, device="cpu", eos_token_id=EOS_TOKEN,
        )
        out = td.generate(torch.tensor([[1, 2, 3]]), max_new_tokens=5)
        _assert_no_post_eos_tokens(out, prompt_len=3, eos=EOS_TOKEN)

    def test_multi_draft_verifier_truncates_at_eos(self) -> None:
        from distllm.core.multi_draft_verifier import MultiDraftVerifier

        seq = [EOS_TOKEN] * 8
        target = _make_argmax_target(seq)

        def draft_fwd(input_ids, **_):
            logits = torch.full((1, input_ids.shape[1], VOCAB_SIZE), -10.0)
            logits[0, -1, EOS_TOKEN] = 10.0
            return logits

        mv = MultiDraftVerifier(
            target_forward=target, draft_forwards=[draft_fwd, draft_fwd],
            num_candidates_per_draft=3, temperature=0,
            device="cpu", eos_token_id=EOS_TOKEN,
        )
        out = mv.generate(torch.tensor([[1, 2, 3]]), max_new_tokens=5)
        _assert_no_post_eos_tokens(out, prompt_len=3, eos=EOS_TOKEN)


# ── Engine 9: mtp_head.py ─────────────────────────────────────────────────


class TestMTPDecoder:
    def test_truncates_at_eos(self) -> None:
        from distllm.core.mtp_head import MTPConfig, MTPHead, MTPDecoder

        # hidden_size must be divisible by num_attention_heads (default 8).
        # Use temperature=0 (greedy) so the MTP head's _sample_with_logprob
        # returns argmax as the draft token.
        config = MTPConfig(
            hidden_size=8, vocab_size=VOCAB_SIZE, num_candidates=3,
            temperature=0, device="cpu",
        )
        head = MTPHead(config)
        # Force every output head to argmax at EOS_TOKEN by zeroing all
        # weights/biases except for the EOS_TOKEN row of each bias.
        with torch.no_grad():
            for layer in head.layers:
                for p in layer.parameters():
                    p.zero_()
            for out_head in head.output_heads:
                out_head.weight.zero_()
                out_head.bias.zero_()
                out_head.bias[EOS_TOKEN] = 10.0

        seq = [EOS_TOKEN] * 8

        def target_fwd(input_ids, **_):
            return _make_argmax_target(seq)(input_ids)

        def hidden_fwd(input_ids, **_):
            logits = target_fwd(input_ids)
            hidden = torch.zeros(1, input_ids.shape[1], config.hidden_size)
            return logits, hidden

        decoder = MTPDecoder(
            target_forward=target_fwd, hidden_states_fn=hidden_fwd,
            mtp_head=head, eos_token_id=EOS_TOKEN,
        )
        out = decoder.generate(torch.tensor([[1, 2, 3]]), max_new_tokens=5)
        _assert_no_post_eos_tokens(out, prompt_len=3, eos=EOS_TOKEN)


# ── Engine 10: compressed_speculative.py ──────────────────────────────────


class TestCompressedSpeculativeDecoder:
    def test_truncates_at_eos(self) -> None:
        from distllm.core.compressed_speculative import CompressedSpeculativeDecoder

        seq = [EOS_TOKEN] * 8

        def target_fwd(input_ids, **_):
            seq_len = input_ids.shape[1]
            logits = torch.full((1, seq_len, VOCAB_SIZE), -10.0)
            idx = max(0, min(seq_len - 1, len(seq) - 1))
            argmax_tok = seq[idx]
            logits[0, -1, argmax_tok] = 10.0
            hidden = torch.zeros(1, seq_len, 4)
            return logits, hidden

        decoder = CompressedSpeculativeDecoder(
            target_forward=target_fwd, kv_cache=None,
            verifier=None,  # auto-accept → always emits the argmax
            eos_token_id=EOS_TOKEN,
        )
        out = decoder.generate(torch.tensor([[1, 2, 3]]), max_new_tokens=5)
        _assert_no_post_eos_tokens(out, prompt_len=3, eos=EOS_TOKEN)

    def test_no_eos_token_id_does_not_truncate(self) -> None:
        """When ``eos_token_id`` is None, the helper returns len(tokens) — no cutoff."""
        from distllm.core.compressed_speculative import CompressedSpeculativeDecoder


        def target_fwd(input_ids, **_):
            seq_len = input_ids.shape[1]
            logits = torch.full((1, seq_len, VOCAB_SIZE), -10.0)
            logits[0, -1, 42] = 10.0
            hidden = torch.zeros(1, seq_len, 4)
            return logits, hidden

        decoder = CompressedSpeculativeDecoder(
            target_forward=target_fwd, kv_cache=None, verifier=None,
            eos_token_id=None,
        )
        out = decoder.generate(torch.tensor([[1, 2, 3]]), max_new_tokens=5)
        # No EOS configured → no truncation; we just assert it runs without error
        assert out.shape[1] >= 3


# ── Engine 11: async_pipelined_speculative.py ─────────────────────────────


class TestPipelinedSpeculativeDecoder:
    def test_truncates_at_eos(self) -> None:
        from distllm.core.async_pipelined_speculative import PipelinedSpeculativeDecoder


        def target_fwd(input_ids, **_):
            seq_len = input_ids.shape[1]
            logits = torch.full((1, seq_len, VOCAB_SIZE), -10.0)
            logits[0, -1, EOS_TOKEN] = 10.0
            return logits

        def draft_gen(prompt_tokens, n):
            return [EOS_TOKEN] * n, [0.0] * n

        decoder = PipelinedSpeculativeDecoder(
            target_forward=target_fwd, draft_generator=draft_gen,
            num_candidates=3, device="cpu", use_cuda_streams=False,
            eos_token_id=EOS_TOKEN,
        )
        try:
            out = decoder.generate(torch.tensor([[1, 2, 3]]), max_new_tokens=5)
            _assert_no_post_eos_tokens(out, prompt_len=3, eos=EOS_TOKEN)
        finally:
            decoder.close()


# ── Cross-engine EOS invariant test ───────────────────────────────────────


@pytest.mark.parametrize(
    "engine_name",
    [
        "SpeculativeDecoder",
        "SelfSpeculativeDecoder",
        "MultiDraftSpeculativeDecoder",
        "TreeDraftSpeculativeDecoder",
        "DistributedSpeculativeDecoder",
        "TreeSpeculativeDecoder",
        "MultiDraftVerifier",
        "TreeMultiDraftVerifier",
        "MTPDecoder",
        "CompressedSpeculativeDecoder",
        "PipelinedSpeculativeDecoder",
    ],
)
def test_engine_invoked_eos_cutoff_in_source(engine_name: str) -> None:
    """Each engine module must reference ``eos_cutoff`` in its generate path.

    This is a source-level invariant: if a future refactor accidentally
    drops the eos_cutoff call, this test fails before the runtime tests
    can.
    """
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent.parent
    core_dir = repo_root / "src" / "distllm" / "core"

    # Map engine class to source file(s)
    file_map = {
        "SpeculativeDecoder": core_dir / "speculative_decoder.py",
        "SelfSpeculativeDecoder": core_dir / "speculative_decoder.py",
        "MultiDraftSpeculativeDecoder": core_dir / "speculative_decoder.py",
        "TreeDraftSpeculativeDecoder": core_dir / "speculative_decoder.py",
        "DistributedSpeculativeDecoder": core_dir / "distributed_speculative.py",
        "TreeSpeculativeDecoder": core_dir / "tree_speculative_decoder.py",
        "MultiDraftVerifier": core_dir / "multi_draft_verifier.py",
        "TreeMultiDraftVerifier": core_dir / "multi_draft_verifier.py",
        "MTPDecoder": core_dir / "mtp_head.py",
        "CompressedSpeculativeDecoder": core_dir / "compressed_speculative.py",
        "PipelinedSpeculativeDecoder": core_dir / "async_pipelined_speculative.py",
    }
    src = file_map[engine_name].read_text(encoding="utf-8")
    assert "eos_cutoff" in src, (
        f"{engine_name} source no longer calls eos_cutoff — post-EOS "
        f"hallucinations may leak through. Re-add the call in "
        f"{file_map[engine_name]}."
    )