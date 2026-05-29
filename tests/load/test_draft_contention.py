"""Load test — multiple concurrent speculative requests."""

import threading
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


class TestDraftContention:
    def test_multiple_decoders_sharing_draft(self):
        """Multiple decoders using the same draft model should work."""
        draft = MagicMock(spec=RemoteDraftModel)
        draft.generate_tokens.return_value = DraftTokenResult(
            token_ids=[10, 10], logprobs=[-0.1, -0.2],
        )
        draft.stats = {
            "total_calls": 0, "total_tokens": 0,
            "avg_latency_ms": 0, "tokens_per_second": 0, "errors": 0,
        }

        decoders = [
            DistributedSpeculativeDecoder(
                target_forward=_fast_target,
                draft_model=draft,
                num_candidates=2,
                temperature=0,
                device="cpu",
            )
            for _ in range(3)
        ]

        results = []
        errors = []

        def worker(idx):
            try:
                output = decoders[idx].generate(
                    torch.tensor([[idx]]), max_new_tokens=4,
                )
                results.append(output.shape[1])
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0, f"Errors: {errors}"
        assert len(results) == 3

    def test_draft_error_rate_under_load(self):
        """Draft model with intermittent errors should degrade gracefully."""
        call_count = [0]

        def flaky_generate(prompt_tokens, num_tokens, **kwargs):
            call_count[0] += 1
            if call_count[0] % 3 == 0:
                return DraftTokenResult(
                    token_ids=[], logprobs=[], error="503",
                )
            return DraftTokenResult(
                token_ids=[10] * num_tokens,
                logprobs=[-0.1] * num_tokens,
            )

        draft = MagicMock(spec=RemoteDraftModel)
        draft.generate_tokens.side_effect = flaky_generate
        draft.stats = {
            "total_calls": 0, "total_tokens": 0,
            "avg_latency_ms": 0, "tokens_per_second": 0, "errors": 0,
        }

        sd = DistributedSpeculativeDecoder(
            target_forward=_fast_target,
            draft_model=draft,
            num_candidates=3,
            temperature=0,
            device="cpu",
            fallback_batch=2,
        )

        # Should complete without crashing despite errors
        output = sd.generate(torch.tensor([[1]]), max_new_tokens=10)
        assert output.shape[1] == 11
