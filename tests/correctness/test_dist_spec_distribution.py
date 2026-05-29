"""Distribution test — rejection sampling preserves target distribution.

Statistical test: over many samples, the distribution of accepted
tokens should match the target model's distribution.
"""

import torch
from collections import Counter
from unittest.mock import MagicMock

from distllm.core.distributed_speculative import (
    DistributedSpeculativeDecoder,
    DraftTokenResult,
    RemoteDraftModel,
)


def _biased_target(input_ids, **kwargs):
    """Target that strongly favors token 10 (80%) over token 20 (20%)."""
    batch, seq = input_ids.shape
    logits = torch.full((batch, seq, 100), -10.0)
    logits[:, :, 10] = 2.0   # ~88% probability
    logits[:, :, 20] = 0.0   # ~12% probability
    return logits


def _uniform_draft():
    """Draft that returns tokens uniformly at random."""
    import random
    draft = MagicMock(spec=RemoteDraftModel)

    def generate(prompt_tokens, num_tokens, **kwargs):
        tokens = [random.choice([10, 20]) for _ in range(num_tokens)]
        return DraftTokenResult(
            token_ids=tokens,
            logprobs=[-0.693] * num_tokens,  # log(0.5)
        )

    draft.generate_tokens.side_effect = generate
    draft.stats = {
        "total_calls": 0, "total_tokens": 0,
        "avg_latency_ms": 0, "tokens_per_second": 0, "errors": 0,
    }
    return draft


class TestDistributionPreservation:
    def test_rejection_sampling_biases_toward_target(self):
        """With a uniform draft, rejection sampling should produce
        tokens that follow the target distribution (mostly token 10)."""
        draft = _uniform_draft()
        sd = DistributedSpeculativeDecoder(
            target_forward=_biased_target,
            draft_model=draft,
            num_candidates=3,
            temperature=1.0,
            device="cpu",
        )

        counts = Counter()
        for _ in range(20):
            output = sd.generate(torch.tensor([[1]]), max_new_tokens=5)
            for i in range(1, output.shape[1]):
                counts[output[0, i].item()] += 1

        total = sum(counts.values())
        ratio_10 = counts.get(10, 0) / max(total, 1)

        # Token 10 should dominate (target assigns ~88% probability)
        assert ratio_10 > 0.5, f"Expected token 10 ratio > 0.5, got {ratio_10:.2f}"

    def test_greedy_deterministic_distribution(self):
        """With temperature=0, output should be fully deterministic."""
        draft = MagicMock(spec=RemoteDraftModel)
        draft.generate_tokens.return_value = DraftTokenResult(
            token_ids=[10, 10], logprobs=[-0.1, -0.2],
        )
        draft.stats = {
            "total_calls": 0, "total_tokens": 0,
            "avg_latency_ms": 0, "tokens_per_second": 0, "errors": 0,
        }

        sd = DistributedSpeculativeDecoder(
            target_forward=_biased_target,
            draft_model=draft,
            num_candidates=2,
            temperature=0,
            device="cpu",
        )

        outputs = []
        for _ in range(5):
            output = sd.generate(torch.tensor([[1]]), max_new_tokens=3)
            outputs.append(output[0].tolist())

        # All outputs should be identical
        for o in outputs[1:]:
            assert o == outputs[0]
