"""Load test — remote draft model latency under concurrent requests."""

import time
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


class TestDraftLatencyUnderLoad:
    def test_conquential_latency_stable(self):
        """Latency stays stable across sequential requests."""
        draft = MagicMock(spec=RemoteDraftModel)
        draft.generate_tokens.return_value = DraftTokenResult(
            token_ids=[10, 10], logprobs=[-0.1, -0.2],
        )
        draft.stats = {
            "total_calls": 0, "total_tokens": 0,
            "avg_latency_ms": 0, "tokens_per_second": 0, "errors": 0,
        }

        sd = DistributedSpeculativeDecoder(
            target_forward=_fast_target,
            draft_model=draft,
            num_candidates=2,
            temperature=0,
            device="cpu",
        )

        latencies = []
        for _ in range(20):
            t0 = time.monotonic()
            sd.generate(torch.tensor([[1]]), max_new_tokens=2)
            latencies.append(time.monotonic() - t0)

        p50 = sorted(latencies)[len(latencies) // 2]
        assert p50 < 1.0  # Should be well under 1 second

    def test_concurrent_requests_dont_crash(self):
        """Multiple concurrent requests should not crash."""
        draft = MagicMock(spec=RemoteDraftModel)
        draft.generate_tokens.return_value = DraftTokenResult(
            token_ids=[10], logprobs=[-0.1],
        )
        draft.stats = {
            "total_calls": 0, "total_tokens": 0,
            "avg_latency_ms": 0, "tokens_per_second": 0, "errors": 0,
        }

        sd = DistributedSpeculativeDecoder(
            target_forward=_fast_target,
            draft_model=draft,
            num_candidates=1,
            temperature=0,
            device="cpu",
        )

        results = []
        errors = []

        def worker():
            try:
                output = sd.generate(torch.tensor([[1]]), max_new_tokens=2)
                results.append(output.shape[1])
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0, f"Errors: {errors}"
        assert len(results) == 5
