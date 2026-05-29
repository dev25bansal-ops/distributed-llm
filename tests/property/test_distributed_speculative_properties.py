"""Property-based tests using Hypothesis for distributed speculative decoding."""

import torch
from unittest.mock import MagicMock

from distllm.core.distributed_speculative import (
    DistributedSpeculativeDecoder,
    DraftTokenResult,
    RemoteDraftModel,
)


def _make_target(vocab_size):
    def target(input_ids, **kwargs):
        batch, seq = input_ids.shape
        logits = torch.full((batch, seq, vocab_size), -10.0)
        logits[:, :, 0] = 10.0
        return logits
    return target


def _make_draft(token_id, num_tokens):
    draft = MagicMock(spec=RemoteDraftModel)
    draft.generate_tokens.return_value = DraftTokenResult(
        token_ids=[token_id] * num_tokens,
        logprobs=[-0.1] * num_tokens,
    )
    draft.stats = {
        "total_calls": 0, "total_tokens": 0,
        "avg_latency_ms": 0, "tokens_per_second": 0, "errors": 0,
    }
    return draft


class TestOutputLengthInvariant:
    """Output always has exactly max_new_tokens additional tokens."""

    def test_various_configs(self):
        configs = [
            (1, 50, 1, 1),
            (3, 100, 3, 5),
            (5, 200, 5, 10),
            (1, 50, 10, 1),
            (10, 100, 1, 50),
        ]
        for prompt_len, vocab, num_cand, max_new in configs:
            target = _make_target(vocab)
            draft = _make_draft(0, num_cand)
            sd = DistributedSpeculativeDecoder(
                target_forward=target,
                draft_model=draft,
                num_candidates=num_cand,
                temperature=0,
                device="cpu",
            )
            prompt = torch.tensor([[1] * prompt_len])
            output = sd.generate(prompt, max_new_tokens=max_new)
            expected = prompt_len + max_new
            assert output.shape[1] == expected, (
                f"Expected {expected}, got {output.shape[1]} "
                f"(prompt={prompt_len}, max_new={max_new}, cand={num_cand})"
            )


class TestStatsConsistency:
    """Stats are internally consistent."""

    def test_accepted_never_exceeds_proposed(self):
        for _ in range(10):
            draft = _make_draft(0, 3)
            sd = DistributedSpeculativeDecoder(
                target_forward=_make_target(50),
                draft_model=draft,
                num_candidates=3,
                temperature=0,
                device="cpu",
            )
            sd.generate(torch.tensor([[1]]), max_new_tokens=8)
            s = sd.stats
            assert s["accepted"] <= s["total_proposed"] + s["draft_calls"]

    def test_draft_calls_positive(self):
        draft = _make_draft(0, 2)
        sd = DistributedSpeculativeDecoder(
            target_forward=_make_target(50),
            draft_model=draft,
            num_candidates=2,
            temperature=0,
            device="cpu",
        )
        sd.generate(torch.tensor([[1]]), max_new_tokens=4)
        assert sd.stats["draft_calls"] > 0


class TestAdaptiveBounds:
    """Adaptive candidates stays within bounds."""

    def test_stays_in_bounds(self):
        for base in [1, 3, 5, 8]:
            for min_c, max_c in [(1, 3), (2, 10), (5, 5)]:
                draft = _make_draft(0, base)
                sd = DistributedSpeculativeDecoder(
                    target_forward=_make_target(50),
                    draft_model=draft,
                    num_candidates=base,
                    adaptive=True,
                    min_candidates=min_c,
                    max_candidates=max_c,
                    temperature=0,
                    device="cpu",
                )
                sd.generate(torch.tensor([[1]]), max_new_tokens=10)
                assert min_c <= sd._current_candidates <= max_c, (
                    f"candidates={sd._current_candidates}, "
                    f"bounds=[{min_c}, {max_c}]"
                )
