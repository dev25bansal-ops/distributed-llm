"""Determinism test — same inputs produce same outputs."""

import torch
from unittest.mock import MagicMock

from distllm.core.distributed_speculative import (
    DistributedSpeculativeDecoder,
    DraftTokenResult,
    RemoteDraftModel,
)


def _deterministic_target(input_ids, **kwargs):
    batch, seq = input_ids.shape
    logits = torch.full((batch, seq, 100), -10.0)
    logits[:, :, 10] = 10.0
    return logits


def _make_draft():
    draft = MagicMock(spec=RemoteDraftModel)
    draft.generate_tokens.return_value = DraftTokenResult(
        token_ids=[10, 10, 10], logprobs=[-0.1, -0.1, -0.1],
    )
    draft.stats = {
        "total_calls": 0, "total_tokens": 0,
        "avg_latency_ms": 0, "tokens_per_second": 0, "errors": 0,
    }
    return draft


class TestDeterminism:
    def test_same_input_same_output(self):
        """Same input_ids and config produce identical output."""
        sd1 = DistributedSpeculativeDecoder(
            target_forward=_deterministic_target,
            draft_model=_make_draft(),
            num_candidates=3,
            temperature=0,
            device="cpu",
        )
        out1 = sd1.generate(torch.tensor([[1, 2, 3]]), max_new_tokens=5)

        sd2 = DistributedSpeculativeDecoder(
            target_forward=_deterministic_target,
            draft_model=_make_draft(),
            num_candidates=3,
            temperature=0,
            device="cpu",
        )
        out2 = sd2.generate(torch.tensor([[1, 2, 3]]), max_new_tokens=5)

        assert torch.equal(out1, out2)

    def test_different_prompt_different_output(self):
        """Different prompts produce different outputs."""
        sd = DistributedSpeculativeDecoder(
            target_forward=_deterministic_target,
            draft_model=_make_draft(),
            num_candidates=3,
            temperature=0,
            device="cpu",
        )
        out_a = sd.generate(torch.tensor([[1, 2]]), max_new_tokens=3)
        out_b = sd.generate(torch.tensor([[5, 6]]), max_new_tokens=3)

        # Prompts differ, so outputs should differ
        assert not torch.equal(out_a, out_b)

    def test_prompt_preserved_in_output(self):
        """Output contains the original prompt prefix."""
        sd = DistributedSpeculativeDecoder(
            target_forward=_deterministic_target,
            draft_model=_make_draft(),
            num_candidates=3,
            temperature=0,
            device="cpu",
        )
        prompt = torch.tensor([[42, 99, 7]])
        output = sd.generate(prompt, max_new_tokens=3)

        assert output[0, 0].item() == 42
        assert output[0, 1].item() == 99
        assert output[0, 2].item() == 7
