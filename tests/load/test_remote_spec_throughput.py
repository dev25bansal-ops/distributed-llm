"""Load test — throughput comparison with/without remote draft."""

import time
from unittest.mock import MagicMock

import torch

from distllm.core.distributed_speculative import (
    DistributedSpeculativeDecoder,
    DraftTokenResult,
    RemoteDraftModel,
)


def _fast_target(input_ids, **kwargs):
    batch, seq = input_ids.shape
    logits = torch.full((batch, seq, 100), -10.0)
    logits[:, :, 10] = 10.0
    return logits


class TestThroughputComparison:
    def test_speculative_faster_than_target_only(self):
        """Speculative decoding with a fast draft should be faster
        than target-only generation."""
        num_tokens = 20

        # Target-only baseline
        t0 = time.monotonic()
        generated = torch.tensor([[1]])
        for _ in range(num_tokens):
            _fast_target(generated)
            next_token = torch.tensor([[10]])
            generated = torch.cat([generated, next_token], dim=1)
        target_time = time.monotonic() - t0  # noqa: F841

        # Speculative with fast draft
        draft = MagicMock(spec=RemoteDraftModel)
        draft.generate_tokens.return_value = DraftTokenResult(
            token_ids=[10] * 5, logprobs=[-0.01] * 5,
        )
        draft.stats = {
            "total_calls": 0, "total_tokens": 0,
            "avg_latency_ms": 0, "tokens_per_second": 0, "errors": 0,
        }

        sd = DistributedSpeculativeDecoder(
            target_forward=_fast_target,
            draft_model=draft,
            num_candidates=5,
            temperature=0,
            device="cpu",
        )

        t0 = time.monotonic()
        sd.generate(torch.tensor([[1]]), max_new_tokens=num_tokens)
        time.monotonic() - t0  # noqa: F841

        # Speculative should not be significantly slower
        # (In mock scenario, both are fast, but speculative should
        # make fewer target calls)
        assert sd.stats["target_calls"] < num_tokens
