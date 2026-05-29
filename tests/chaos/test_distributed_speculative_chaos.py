"""Chaos tests — network failures, timeouts, malformed responses."""

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


# ── Network partition mid-generation ─────────────────────────────────────


class TestNetworkPartition:
    def test_draft_fails_after_initial_success(self):
        """Draft model works first, then fails — should fallback gracefully."""
        call_count = [0]

        def mock_generate(prompt_tokens, num_tokens, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 1:
                return DraftTokenResult(
                    token_ids=[10, 10], logprobs=[-0.1, -0.2],
                )
            return DraftTokenResult(
                token_ids=[], logprobs=[], error="connection refused",
            )

        draft = MagicMock(spec=RemoteDraftModel)
        draft.generate_tokens.side_effect = mock_generate
        draft.stats = dict(_MOCK_STATS)

        sd = DistributedSpeculativeDecoder(
            target_forward=_always_logits(10),
            draft_model=draft,
            num_candidates=2,
            temperature=0,
            device="cpu",
            fallback_batch=2,
        )
        output = sd.generate(torch.tensor([[1, 2]]), max_new_tokens=6)
        # Should still produce output despite failures
        assert output.shape[1] == 8

    def test_draft_always_fails(self):
        """Draft model never succeeds — pure target-only fallback."""
        draft = MagicMock(spec=RemoteDraftModel)
        draft.generate_tokens.return_value = DraftTokenResult(
            token_ids=[], logprobs=[], error="unreachable",
        )
        draft.stats = dict(_MOCK_STATS)

        sd = DistributedSpeculativeDecoder(
            target_forward=_always_logits(10),
            draft_model=draft,
            num_candidates=3,
            temperature=0,
            device="cpu",
            fallback_batch=2,
        )
        output = sd.generate(torch.tensor([[1]]), max_new_tokens=4)
        assert output.shape[1] == 5


# ── Timeout simulation ───────────────────────────────────────────────────


class TestTimeout:
    def test_draft_timeout_fallback(self):
        """Simulate draft model timeout — should fallback to target."""
        import time

        def slow_generate(prompt_tokens, num_tokens, **kwargs):
            time.sleep(0.01)  # Simulate slow response
            return DraftTokenResult(
                token_ids=[], logprobs=[], error="timeout",
            )

        draft = MagicMock(spec=RemoteDraftModel)
        draft.generate_tokens.side_effect = slow_generate
        draft.stats = dict(_MOCK_STATS)

        sd = DistributedSpeculativeDecoder(
            target_forward=_always_logits(10),
            draft_model=draft,
            num_candidates=2,
            temperature=0,
            device="cpu",
        )
        output = sd.generate(torch.tensor([[1]]), max_new_tokens=2)
        assert output.shape[1] == 3


# ── Malformed responses ─────────────────────────────────────────────────


class TestMalformedResponses:
    def test_draft_returns_garbage_tokens(self):
        """Draft returns tokens outside vocab — should handle gracefully."""
        draft = MagicMock(spec=RemoteDraftModel)
        draft.generate_tokens.return_value = DraftTokenResult(
            token_ids=[99999, -1], logprobs=[-0.1, -0.2],
        )
        draft.stats = dict(_MOCK_STATS)

        sd = DistributedSpeculativeDecoder(
            target_forward=_always_logits(10),
            draft_model=draft,
            num_candidates=2,
            temperature=0,
            device="cpu",
        )
        # Should not crash — rejection sampling handles mismatched tokens
        output = sd.generate(torch.tensor([[1]]), max_new_tokens=2)
        assert output.shape[1] == 3

    def test_draft_returns_single_token(self):
        """Draft returns only 1 token when 5 requested."""
        draft = MagicMock(spec=RemoteDraftModel)
        draft.generate_tokens.return_value = DraftTokenResult(
            token_ids=[10], logprobs=[-0.1],
        )
        draft.stats = dict(_MOCK_STATS)

        sd = DistributedSpeculativeDecoder(
            target_forward=_always_logits(10),
            draft_model=draft,
            num_candidates=5,
            temperature=0,
            device="cpu",
        )
        output = sd.generate(torch.tensor([[1]]), max_new_tokens=3)
        assert output.shape[1] == 4
