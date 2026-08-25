"""C14 regression tests — every speculative decoder must stop at EOS.

Audit C14: all speculative-decoding ``generate`` loops ran to
``max_new_tokens`` ignoring end-of-text.  Post-EOS tokens survive
``skip_special_tokens`` decoding, so users were billed hallucinated
continuations.  Each test drives a stub model whose greedy next-token
follows a fixed chain ``7 -> EOS(0) -> 8 -> 9`` and asserts that
generation stops at (and includes) the EOS token when one is configured,
and runs to ``max_new_tokens`` when it is not.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
import torch

from distllm.core.distributed_speculative import (
    DistributedSpeculativeDecoder,
    DraftTokenResult,
    RemoteDraftModel,
)
from distllm.core.multi_draft_verifier import (
    MultiDraftVerifier,
    TreeMultiDraftVerifier,
)
from distllm.core.speculative_decoder import (
    MultiDraftSpeculativeDecoder,
    SelfSpeculativeDecoder,
    SpeculativeDecoder,
    TreeDraftSpeculativeDecoder,
)
from distllm.core.tree_speculative_decoder import TreeSpeculativeDecoder

VOCAB = 100
EOS_ID = 0
PROMPT_TOKEN = 5
PROMPT = [[PROMPT_TOKEN, PROMPT_TOKEN, PROMPT_TOKEN]]
SCRIPT = [7, EOS_ID, 8, 9]

# Greedy next-token chain: prompt token -> 7 -> EOS -> 8 -> 9 (self-loop).
_SUCC = {
    PROMPT_TOKEN: SCRIPT[0],
    SCRIPT[0]: SCRIPT[1],
    SCRIPT[1]: SCRIPT[2],
    SCRIPT[2]: SCRIPT[3],
    SCRIPT[3]: SCRIPT[3],
}


def _scripted_logits(input_ids, **kwargs):
    """Deterministic causal-LM logits: row ``i`` favors ``succ(input[i])``.

    Content-addressed (not position-addressed) so every decoder's indexing
    convention agrees.  Non-favored entries get strictly decreasing scores
    so ``topk`` tie-breaks are deterministic for the tree builders.
    """
    batch, seq = input_ids.shape
    base = -100.0 - torch.arange(VOCAB, dtype=torch.float32) * 1e-3
    logits = base.view(1, 1, VOCAB).expand(batch, seq, VOCAB).clone()
    for b in range(batch):
        for c in range(seq):
            nxt = _SUCC.get(int(input_ids[b, c]), SCRIPT[3])
            logits[b, c, nxt] = 10.0
    return logits


def _new_tokens(output):
    return [int(t) for t in output[0, len(PROMPT[0]):]]


# ── Expected EOS behaviour ───────────────────────────────────────────────


class TestSpeculativeDecoderEos:
    def test_generate_stops_at_eos(self):
        d = SpeculativeDecoder(
            target_forward=_scripted_logits,
            draft_forward=_scripted_logits,
            num_candidates=3,
            temperature=0,
            device="cpu",
            eos_token_id=EOS_ID,
        )
        out = d.generate(torch.tensor(PROMPT), max_new_tokens=10)
        assert _new_tokens(out) == [7, EOS_ID]

    def test_generate_kwarg_overrides(self):
        d = SpeculativeDecoder(
            target_forward=_scripted_logits,
            draft_forward=_scripted_logits,
            num_candidates=3,
            temperature=0,
            device="cpu",
        )
        out = d.generate(torch.tensor(PROMPT), max_new_tokens=10, eos_token_id=EOS_ID)
        assert _new_tokens(out) == [7, EOS_ID]

    def test_no_eos_runs_to_max_new_tokens(self):
        d = SpeculativeDecoder(
            target_forward=_scripted_logits,
            draft_forward=_scripted_logits,
            num_candidates=3,
            temperature=0,
            device="cpu",
        )
        out = d.generate(torch.tensor(PROMPT), max_new_tokens=10)
        assert out.shape[1] == len(PROMPT[0]) + 10


class TestSelfSpeculativeDecoderEos:
    def test_generate_stops_at_eos(self):
        d = SelfSpeculativeDecoder(
            target_forward=_scripted_logits,
            hidden_states_fn=lambda ids, **kw: (
                _scripted_logits(ids),
                (torch.zeros(1, ids.shape[1], 8), torch.zeros(1, ids.shape[1], 8)),
            ),
            hidden_size=8,
            vocab_size=VOCAB,
            num_candidates=2,
            temperature=0,
            device="cpu",
            eos_token_id=EOS_ID,
        )
        # Zero-init the draft head so proposals are deterministic.
        torch.nn.init.zeros_(d._draft_head.weight)
        torch.nn.init.zeros_(d._draft_head.bias)
        out = d.generate(torch.tensor(PROMPT), max_new_tokens=10)
        assert _new_tokens(out) == [7, EOS_ID]

    def test_no_eos_runs_to_max_new_tokens(self):
        d = SelfSpeculativeDecoder(
            target_forward=_scripted_logits,
            hidden_states_fn=lambda ids, **kw: (
                _scripted_logits(ids),
                (torch.zeros(1, ids.shape[1], 8), torch.zeros(1, ids.shape[1], 8)),
            ),
            hidden_size=8,
            vocab_size=VOCAB,
            num_candidates=2,
            temperature=0,
            device="cpu",
        )
        torch.nn.init.zeros_(d._draft_head.weight)
        torch.nn.init.zeros_(d._draft_head.bias)
        out = d.generate(torch.tensor(PROMPT), max_new_tokens=10)
        assert out.shape[1] == len(PROMPT[0]) + 10


class TestMultiDraftSpeculativeDecoderEos:
    def test_generate_stops_at_eos(self):
        d = MultiDraftSpeculativeDecoder(
            target_forward=_scripted_logits,
            draft_forwards=[_scripted_logits, _scripted_logits],
            num_candidates=3,
            temperature=0,
            device="cpu",
            eos_token_id=EOS_ID,
        )
        out = d.generate(torch.tensor(PROMPT), max_new_tokens=10)
        assert _new_tokens(out) == [7, EOS_ID]

    def test_no_eos_runs_to_max_new_tokens(self):
        d = MultiDraftSpeculativeDecoder(
            target_forward=_scripted_logits,
            draft_forwards=[_scripted_logits, _scripted_logits],
            num_candidates=3,
            temperature=0,
            device="cpu",
        )
        out = d.generate(torch.tensor(PROMPT), max_new_tokens=10)
        assert out.shape[1] == len(PROMPT[0]) + 10


class TestTreeDraftSpeculativeDecoderEos:
    def test_generate_stops_at_eos(self):
        d = TreeDraftSpeculativeDecoder(
            target_forward=_scripted_logits,
            draft_forwards=[_scripted_logits],
            max_tree_nodes=16,
            max_depth=2,
            branching_factor=2,
            temperature=0,
            device="cpu",
            eos_token_id=EOS_ID,
        )
        out = d.generate(torch.tensor(PROMPT), max_new_tokens=10)
        new = _new_tokens(out)
        assert new.index(EOS_ID) == new.index(EOS_ID)  # EOS present exactly once
        assert new.count(EOS_ID) == 1
        assert new[-1] == EOS_ID
        assert len(new) < 10

    def test_no_eos_runs_to_max_new_tokens(self):
        d = TreeDraftSpeculativeDecoder(
            target_forward=_scripted_logits,
            draft_forwards=[_scripted_logits],
            max_tree_nodes=16,
            max_depth=2,
            branching_factor=2,
            temperature=0,
            device="cpu",
        )
        out = d.generate(torch.tensor(PROMPT), max_new_tokens=10)
        assert out.shape[1] == len(PROMPT[0]) + 10


class TestTreeSpeculativeDecoderEos:
    def test_generate_stops_at_eos(self):
        d = TreeSpeculativeDecoder(
            target_forward=_scripted_logits,
            draft_forward=_scripted_logits,
            branching_factor=2,
            tree_depth=2,
            temperature=0,
            device="cpu",
            eos_token_id=EOS_ID,
        )
        out = d.generate(torch.tensor(PROMPT), max_new_tokens=10)
        new = _new_tokens(out)
        assert new.count(EOS_ID) == 1
        assert new[-1] == EOS_ID
        assert len(new) < 10

    def test_no_eos_runs_to_max_new_tokens(self):
        d = TreeSpeculativeDecoder(
            target_forward=_scripted_logits,
            draft_forward=_scripted_logits,
            branching_factor=2,
            tree_depth=2,
            temperature=0,
            device="cpu",
        )
        out = d.generate(torch.tensor(PROMPT), max_new_tokens=10)
        assert out.shape[1] == len(PROMPT[0]) + 10


class TestMultiDraftVerifierEos:
    def test_generate_stops_at_eos(self):
        v = MultiDraftVerifier(
            target_forward=_scripted_logits,
            draft_forwards=[_scripted_logits, _scripted_logits],
            num_candidates_per_draft=3,
            temperature=0,
            device="cpu",
            eos_token_id=EOS_ID,
        )
        out = v.generate(torch.tensor(PROMPT), max_new_tokens=10)
        assert _new_tokens(out) == [7, EOS_ID]

    def test_no_eos_runs_to_max_new_tokens(self):
        v = MultiDraftVerifier(
            target_forward=_scripted_logits,
            draft_forwards=[_scripted_logits, _scripted_logits],
            num_candidates_per_draft=3,
            temperature=0,
            device="cpu",
        )
        out = v.generate(torch.tensor(PROMPT), max_new_tokens=10)
        assert out.shape[1] == len(PROMPT[0]) + 10


class TestTreeMultiDraftVerifierEos:
    def test_generate_stops_at_eos(self):
        v = TreeMultiDraftVerifier(
            target_forward=_scripted_logits,
            draft_forwards=[_scripted_logits, _scripted_logits],
            branching_factor=2,
            depth=2,
            temperature=0,
            device="cpu",
            eos_token_id=EOS_ID,
        )
        out = v.generate(torch.tensor(PROMPT), max_new_tokens=10)
        new = _new_tokens(out)
        assert new.count(EOS_ID) == 1
        assert new[-1] == EOS_ID
        assert len(new) < 10

    def test_no_eos_runs_to_max_new_tokens(self):
        v = TreeMultiDraftVerifier(
            target_forward=_scripted_logits,
            draft_forwards=[_scripted_logits, _scripted_logits],
            branching_factor=2,
            depth=2,
            temperature=0,
            device="cpu",
        )
        out = v.generate(torch.tensor(PROMPT), max_new_tokens=10)
        assert out.shape[1] == len(PROMPT[0]) + 10


# ── DistributedSpeculativeDecoder ────────────────────────────────────────


def _mock_remote_draft(chains):
    """RemoteDraftModel mock serving successive fixed token chains."""
    model = MagicMock(spec=RemoteDraftModel)
    calls = {"n": 0}

    def generate_tokens(**kwargs):
        idx = min(calls["n"], len(chains) - 1)
        calls["n"] += 1
        return DraftTokenResult(token_ids=list(chains[idx]), logprobs=[-0.01] * len(chains[idx]))

    model.generate_tokens.side_effect = generate_tokens
    model.agenerate_tokens.side_effect = None
    model.agenerate_tokens = AsyncMock(side_effect=lambda **kw: generate_tokens())
    return model


class TestDistributedSpeculativeDecoderEos:
    def test_generate_stops_at_eos(self):
        draft = _mock_remote_draft([[7, EOS_ID, 8]])
        sd = DistributedSpeculativeDecoder(
            target_forward=_scripted_logits,
            draft_model=draft,
            num_candidates=3,
            temperature=0,
            device="cpu",
            eos_token_id=EOS_ID,
        )
        out = sd.generate(torch.tensor(PROMPT), max_new_tokens=10)
        assert _new_tokens(out) == [7, EOS_ID]

    def test_fallback_path_stops_at_eos(self):
        """Draft failure -> target-only fallback must still honour EOS."""
        draft = MagicMock(spec=RemoteDraftModel)
        draft.generate_tokens.return_value = DraftTokenResult(
            token_ids=[], logprobs=[], error="boom",
        )
        sd = DistributedSpeculativeDecoder(
            target_forward=_scripted_logits,
            draft_model=draft,
            num_candidates=3,
            temperature=0,
            device="cpu",
            fallback_batch=2,
            eos_token_id=EOS_ID,
        )
        out = sd.generate(torch.tensor(PROMPT), max_new_tokens=10)
        assert _new_tokens(out) == [7, EOS_ID]

    @pytest.mark.asyncio
    async def test_agenerate_stops_at_eos(self):
        draft = _mock_remote_draft([[7, EOS_ID, 8]])
        sd = DistributedSpeculativeDecoder(
            target_forward=_scripted_logits,
            draft_model=draft,
            num_candidates=3,
            temperature=0,
            device="cpu",
            eos_token_id=EOS_ID,
        )
        out = await sd.agenerate(torch.tensor(PROMPT), max_new_tokens=10)
        assert _new_tokens(out) == [7, EOS_ID]

    def test_no_eos_runs_to_max_new_tokens(self):
        draft = _mock_remote_draft([[7, EOS_ID, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9]])
        sd = DistributedSpeculativeDecoder(
            target_forward=_scripted_logits,
            draft_model=draft,
            num_candidates=3,
            temperature=0,
            device="cpu",
        )
        out = sd.generate(torch.tensor(PROMPT), max_new_tokens=10)
        assert out.shape[1] == len(PROMPT[0]) + 10
