"""Integration tests — end-to-end distributed speculative decoding.

Uses a mock HTTP server to simulate a remote draft model and verifies
the full generate() loop works correctly.
"""

from unittest.mock import MagicMock

import torch

from distllm.core.distributed_speculative import (
    DistributedSpeculativeDecoder,
    DraftTokenResult,
    RemoteDraftModel,
)

_MOCK_STATS = {
    "total_calls": 0, "total_tokens": 0,
    "avg_latency_ms": 0, "tokens_per_second": 0, "errors": 0,
}


def _always_logits(token_id: int, vocab: int = 100):
    def fn(input_ids, **kwargs):
        batch, seq = input_ids.shape
        logits = torch.full((batch, seq, vocab), -10.0)
        logits[:, :, token_id] = 10.0
        return logits
    return fn


def _identity_target(input_ids, **kwargs):
    batch, seq_len = input_ids.shape
    vocab = 100
    logits = torch.full((batch, seq_len, vocab), -10.0)
    for i in range(seq_len - 1):
        t = input_ids[0, i + 1].item()
        if t < vocab:
            logits[0, i, t] = 10.0
    last_t = input_ids[0, -1].item()
    if last_t < vocab:
        logits[0, -1, last_t] = 10.0
    return logits


# ── Full loop — mock draft model ─────────────────────────────────────────


class TestFullLoop:
    def test_generate_multiple_steps(self):
        """Full generation across multiple speculative steps."""
        call_count = [0]
        draft_sequences = [
            [10, 10, 10],  # Step 1: all accepted
            [10, 10],      # Step 2: all accepted
            [10],           # Step 3: accepted
        ]

        def mock_generate(prompt_tokens, num_tokens, **kwargs):
            idx = min(call_count[0], len(draft_sequences) - 1)
            tokens = draft_sequences[idx][:num_tokens]
            call_count[0] += 1
            return DraftTokenResult(
                token_ids=tokens, logprobs=[-0.1] * len(tokens),
            )

        draft = MagicMock(spec=RemoteDraftModel)
        draft.generate_tokens.side_effect = mock_generate
        draft.stats = dict(_MOCK_STATS)

        sd = DistributedSpeculativeDecoder(
            target_forward=_always_logits(10),
            draft_model=draft,
            num_candidates=3,
            temperature=0,
            device="cpu",
        )
        output = sd.generate(torch.tensor([[1, 2]]), max_new_tokens=6)
        assert output.shape[1] == 8  # 2 prompt + 6 generated
        assert all(output[0, i].item() == 10 for i in range(2, 8))

    def test_mixed_accept_reject_loop(self):
        """Some tokens accepted, some rejected across steps."""
        call_count = [0]
        draft_sequences = [
            [10, 99, 10],  # Step 1: first accepted, second rejected
            [10, 10],       # Step 2: both accepted
        ]

        def mock_generate(prompt_tokens, num_tokens, **kwargs):
            idx = min(call_count[0], len(draft_sequences) - 1)
            tokens = draft_sequences[idx][:num_tokens]
            call_count[0] += 1
            return DraftTokenResult(
                token_ids=tokens, logprobs=[-0.1] * len(tokens),
            )

        draft = MagicMock(spec=RemoteDraftModel)
        draft.generate_tokens.side_effect = mock_generate
        draft.stats = dict(_MOCK_STATS)

        sd = DistributedSpeculativeDecoder(
            target_forward=_always_logits(10),
            draft_model=draft,
            num_candidates=3,
            temperature=0,
            device="cpu",
        )
        output = sd.generate(torch.tensor([[1]]), max_new_tokens=4)
        assert output.shape[1] == 5


# ── Latency stats ────────────────────────────────────────────────────────


class TestLatencyStats:
    def test_stats_accumulate(self):
        draft = MagicMock(spec=RemoteDraftModel)
        draft.generate_tokens.return_value = DraftTokenResult(
            token_ids=[10, 10], logprobs=[-0.1, -0.2],
        )
        draft.stats = {
            "total_calls": 2, "total_tokens": 4,
            "avg_latency_ms": 15.0, "tokens_per_second": 66.7, "errors": 0,
        }

        sd = DistributedSpeculativeDecoder(
            target_forward=_always_logits(10),
            draft_model=draft,
            num_candidates=2,
            temperature=0,
            device="cpu",
        )
        sd.generate(torch.tensor([[1]]), max_new_tokens=4)

        s = sd.stats
        assert s["draft_calls"] >= 1
        assert s["draft_model_stats"]["total_calls"] == 2


# ── String URL auto-config ───────────────────────────────────────────────


class TestStringURLConfig:
    def test_string_url_creates_model(self):
        sd = DistributedSpeculativeDecoder(
            target_forward=_always_logits(10),
            draft_model="http://localhost:9000/v1/completions",
            num_candidates=1,
            temperature=0,
            device="cpu",
        )
        assert isinstance(sd._draft, RemoteDraftModel)
        assert sd._draft._config.endpoint_url == "http://localhost:9000/v1/completions"
        sd.close()
