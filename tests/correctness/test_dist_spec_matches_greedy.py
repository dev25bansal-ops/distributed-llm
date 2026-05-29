"""Correctness test — distributed speculative matches target-only greedy output.

Verifies that speculative decoding with a perfect draft model produces
the same output as target-only generation (greedy, temperature=0).
"""

import torch
from unittest.mock import MagicMock

from distllm.core.distributed_speculative import (
    DistributedSpeculativeDecoder,
    DraftTokenResult,
    RemoteDraftModel,
)


def _greedy_target(input_ids, **kwargs):
    """Deterministic target: always picks token 10."""
    batch, seq = input_ids.shape
    logits = torch.full((batch, seq, 100), -10.0)
    logits[:, :, 10] = 10.0
    return logits


def _identity_target(input_ids, **kwargs):
    """Target that predicts the next token in the input sequence."""
    batch, seq_len = input_ids.shape
    logits = torch.full((batch, seq_len, 100), -10.0)
    for i in range(seq_len - 1):
        t = input_ids[0, i + 1].item()
        if t < 100:
            logits[0, i, t] = 10.0
    last_t = input_ids[0, -1].item()
    if last_t < 100:
        logits[0, -1, last_t] = 10.0
    return logits


def _perfect_draft(target_token):
    """Draft that always predicts the correct target token."""
    draft = MagicMock(spec=RemoteDraftModel)
    call_count = [0]

    def generate(prompt_tokens, num_tokens, **kwargs):
        call_count[0] += 1
        return DraftTokenResult(
            token_ids=[target_token] * num_tokens,
            logprobs=[-0.01] * num_tokens,
        )

    draft.generate_tokens.side_effect = generate
    draft.stats = {
        "total_calls": 0, "total_tokens": 0,
        "avg_latency_ms": 0, "tokens_per_second": 0, "errors": 0,
    }
    return draft


class TestGreedyMatch:
    def test_perfect_draft_matches_target_only(self):
        """With a perfect draft and greedy decoding, output should
        match target-only generation."""
        # Target-only
        input_ids = torch.tensor([[1, 2, 3]])
        target_only_output = _greedy_target(input_ids)
        target_only_token = target_only_output[0, -1, :].argmax().item()
        assert target_only_token == 10

        # Speculative with perfect draft
        draft = _perfect_draft(10)
        sd = DistributedSpeculativeDecoder(
            target_forward=_greedy_target,
            draft_model=draft,
            num_candidates=3,
            temperature=0,
            device="cpu",
        )
        output = sd.generate(input_ids.clone(), max_new_tokens=5)
        assert output.shape[1] == 8
        assert all(output[0, 3 + i].item() == 10 for i in range(5))

    def test_deterministic_greedy_same_seed(self):
        """Same seed produces same output (greedy, temp=0)."""
        draft = _perfect_draft(10)

        torch.manual_seed(42)
        sd1 = DistributedSpeculativeDecoder(
            target_forward=_greedy_target,
            draft_model=draft,
            num_candidates=3,
            temperature=0,
            device="cpu",
        )
        out1 = sd1.generate(torch.tensor([[1, 2]]), max_new_tokens=4)

        draft2 = _perfect_draft(10)
        torch.manual_seed(42)
        sd2 = DistributedSpeculativeDecoder(
            target_forward=_greedy_target,
            draft_model=draft2,
            num_candidates=3,
            temperature=0,
            device="cpu",
        )
        out2 = sd2.generate(torch.tensor([[1, 2]]), max_new_tokens=4)

        assert torch.equal(out1, out2)


class TestRejectionSamplingCorrectness:
    def test_bad_draft_rejected(self):
        """When draft predicts wrong token, it should be rejected."""
        # Draft always predicts 99, target always predicts 10
        draft = MagicMock(spec=RemoteDraftModel)
        draft.generate_tokens.return_value = DraftTokenResult(
            token_ids=[99, 99], logprobs=[-0.1, -0.2],
        )
        draft.stats = {
            "total_calls": 0, "total_tokens": 0,
            "avg_latency_ms": 0, "tokens_per_second": 0, "errors": 0,
        }

        sd = DistributedSpeculativeDecoder(
            target_forward=_greedy_target,
            draft_model=draft,
            num_candidates=2,
            temperature=0,
            device="cpu",
        )
        output = sd.generate(torch.tensor([[1]]), max_new_tokens=3)
        # All generated tokens should be 10 (target's choice), not 99
        assert all(output[0, 1 + i].item() == 10 for i in range(3))

    def test_acceptance_rate_reflects_draft_quality(self):
        """Good draft has high acceptance, bad draft has low."""
        # Perfect draft
        good_draft = _perfect_draft(10)
        sd_good = DistributedSpeculativeDecoder(
            target_forward=_greedy_target,
            draft_model=good_draft,
            num_candidates=5,
            temperature=0,
            device="cpu",
        )
        sd_good.generate(torch.tensor([[1]]), max_new_tokens=10)
        good_rate = sd_good.stats.get("acceptance_rate", 0)

        # Bad draft
        bad_draft = MagicMock(spec=RemoteDraftModel)
        bad_draft.generate_tokens.return_value = DraftTokenResult(
            token_ids=[99] * 5, logprobs=[-0.1] * 5,
        )
        bad_draft.stats = {
            "total_calls": 0, "total_tokens": 0,
            "avg_latency_ms": 0, "tokens_per_second": 0, "errors": 0,
        }
        sd_bad = DistributedSpeculativeDecoder(
            target_forward=_greedy_target,
            draft_model=bad_draft,
            num_candidates=5,
            temperature=0,
            device="cpu",
        )
        sd_bad.generate(torch.tensor([[1]]), max_new_tokens=10)
        bad_rate = sd_bad.stats.get("acceptance_rate", 0)

        assert good_rate > bad_rate
