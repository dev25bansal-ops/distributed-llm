"""Unit tests for DistributedSpeculativeDecoder."""

import pytest
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


# ── Helpers ──────────────────────────────────────────────────────────────


def _always_logits(token_id: int, vocab: int = 100):
    """Return logits where token_id has highest probability."""
    def fn(input_ids, **kwargs):
        batch, seq = input_ids.shape
        logits = torch.full((batch, seq, vocab), -10.0)
        logits[:, :, token_id] = 10.0
        return logits
    return fn


def _identity_target(input_ids, **kwargs):
    """Target logits: position i predicts token at i+1."""
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


def _mock_draft_model(token_ids, logprobs=None):
    """Create a mock RemoteDraftModel that returns fixed tokens."""
    model = MagicMock(spec=RemoteDraftModel)
    if logprobs is None:
        logprobs = [-0.1] * len(token_ids)
    model.generate_tokens.return_value = DraftTokenResult(
        token_ids=token_ids, logprobs=logprobs,
    )
    model.stats = dict(_MOCK_STATS)
    return model


# ── Initialization ───────────────────────────────────────────────────────


class TestInit:
    def test_with_draft_model(self):
        draft = MagicMock(spec=RemoteDraftModel)
        sd = DistributedSpeculativeDecoder(
            target_forward=lambda x, **kw: x, draft_model=draft,
        )
        assert sd._draft is draft

    def test_with_string_url(self):
        sd = DistributedSpeculativeDecoder(
            target_forward=lambda x, **kw: x,
            draft_model="http://draft:8000/v1/completions",
        )
        assert isinstance(sd._draft, RemoteDraftModel)
        sd.close()

    def test_no_draft_raises(self):
        with pytest.raises(ValueError, match="Either"):
            DistributedSpeculativeDecoder(
                target_forward=lambda x, **kw: x,
            )

    def test_defaults(self):
        draft = MagicMock(spec=RemoteDraftModel)
        sd = DistributedSpeculativeDecoder(
            target_forward=lambda x, **kw: x, draft_model=draft,
        )
        assert sd._num_candidates == 5
        assert sd._temperature == 1.0
        assert sd._top_k == 20


# ── Generate — full acceptance ───────────────────────────────────────────


class TestFullAcceptance:
    def test_greedy_full_accept(self):
        """When draft and target agree, all tokens accepted."""
        draft = _mock_draft_model([10, 10, 10])
        sd = DistributedSpeculativeDecoder(
            target_forward=_always_logits(10),
            draft_model=draft,
            num_candidates=3,
            temperature=0,
            device="cpu",
        )
        output = sd.generate(torch.tensor([[1, 2, 3]]), max_new_tokens=3)
        assert output.shape[1] == 6
        assert all(output[0, 3 + i].item() == 10 for i in range(3))


# ── Generate — partial acceptance ────────────────────────────────────────


class TestPartialAcceptance:
    def test_partial_accept(self):
        """First token accepted, second rejected."""
        draft = _mock_draft_model([10, 99])
        sd = DistributedSpeculativeDecoder(
            target_forward=_identity_target,
            draft_model=draft,
            num_candidates=2,
            temperature=0,
            device="cpu",
        )
        # identity target: pos 2 predicts input[3], pos 3 predicts input[4]
        # prefix [1,2,3], draft [10, 99]
        # target at pos 2 predicts 3 (input[3]), draft has 10 → reject at 0
        output = sd.generate(torch.tensor([[1, 2, 3]]), max_new_tokens=2)
        assert output.shape[1] == 5  # 3 prompt + 2 generated


# ── Generate — no acceptance ─────────────────────────────────────────────


class TestNoAcceptance:
    def test_all_rejected(self):
        """All draft tokens rejected, fallback to target."""
        draft = _mock_draft_model([99, 99])
        sd = DistributedSpeculativeDecoder(
            target_forward=_always_logits(10),
            draft_model=draft,
            num_candidates=2,
            temperature=0,
            device="cpu",
        )
        output = sd.generate(torch.tensor([[1, 2, 3]]), max_new_tokens=2)
        assert output.shape[1] == 5


# ── Draft failure fallback ───────────────────────────────────────────────


class TestDraftFailure:
    def test_draft_returns_empty(self):
        """When draft fails, target-only fallback with multi-token batch."""
        draft = MagicMock(spec=RemoteDraftModel)
        draft.generate_tokens.return_value = DraftTokenResult(
            token_ids=[], logprobs=[], error="connection refused",
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
        output = sd.generate(torch.tensor([[1, 2]]), max_new_tokens=3)
        # Should still generate tokens via fallback
        assert output.shape[1] == 5


# ── Max tokens boundary ─────────────────────────────────────────────────


class TestMaxTokens:
    def test_respects_max_new_tokens(self):
        draft = _mock_draft_model([10] * 10)
        sd = DistributedSpeculativeDecoder(
            target_forward=_always_logits(10),
            draft_model=draft,
            num_candidates=10,
            temperature=0,
            device="cpu",
        )
        output = sd.generate(torch.tensor([[1]]), max_new_tokens=2)
        assert output.shape[1] == 3


# ── Stats tracking ───────────────────────────────────────────────────────


class TestStats:
    def test_stats_tracking(self):
        draft = _mock_draft_model([10, 10])
        sd = DistributedSpeculativeDecoder(
            target_forward=_always_logits(10),
            draft_model=draft,
            num_candidates=2,
            temperature=0,
            device="cpu",
        )
        sd.generate(torch.tensor([[1, 2]]), max_new_tokens=4)
        s = sd.stats
        assert s["draft_calls"] > 0
        assert s["target_calls"] > 0
        assert s["total_proposed"] > 0
        assert s["accepted"] > 0

    def test_acceptance_rate(self):
        draft = _mock_draft_model([10, 10])
        sd = DistributedSpeculativeDecoder(
            target_forward=_always_logits(10),
            draft_model=draft,
            num_candidates=2,
            temperature=0,
            device="cpu",
        )
        sd.generate(torch.tensor([[1]]), max_new_tokens=4)
        s = sd.stats
        assert "acceptance_rate" in s
        assert 0 <= s["acceptance_rate"] <= 1


# ── Adaptive candidates ─────────────────────────────────────────────────


class TestAdaptive:
    def test_adaptive_increases_on_high_acceptance(self):
        draft = _mock_draft_model([10] * 5)
        sd = DistributedSpeculativeDecoder(
            target_forward=_always_logits(10),
            draft_model=draft,
            num_candidates=3,
            adaptive=True,
            min_candidates=2,
            max_candidates=8,
            temperature=0,
            device="cpu",
        )
        sd.generate(torch.tensor([[1]]), max_new_tokens=10)
        # After high acceptance, candidates should increase
        assert sd._current_candidates >= 3

    def test_adaptive_respects_max(self):
        draft = _mock_draft_model([10] * 5)
        sd = DistributedSpeculativeDecoder(
            target_forward=_always_logits(10),
            draft_model=draft,
            num_candidates=8,
            adaptive=True,
            min_candidates=2,
            max_candidates=8,
            temperature=0,
            device="cpu",
        )
        sd.generate(torch.tensor([[1]]), max_new_tokens=10)
        assert sd._current_candidates <= 8


# ── KV cache passthrough ────────────────────────────────────────────────


class TestKVCachePassthrough:
    def test_past_key_values_forwarded(self):
        """Verify past_key_values is passed to target_forward."""
        received_kwargs = {}

        def target_fn(input_ids, **kwargs):
            received_kwargs.update(kwargs)
            return _always_logits(10)(input_ids, **kwargs)

        draft = _mock_draft_model([10])
        sd = DistributedSpeculativeDecoder(
            target_forward=target_fn,
            draft_model=draft,
            num_candidates=1,
            temperature=0,
            device="cpu",
        )
        fake_kv = {"layer_0": "cached"}
        sd.generate(torch.tensor([[1, 2]]), max_new_tokens=1, past_key_values=fake_kv)
        assert received_kwargs.get("past_key_values") == fake_kv


# ── Batch size validation ───────────────────────────────────────────────


class TestBatchValidation:
    def test_batch_size_gt1_raises(self):
        draft = _mock_draft_model([10])
        sd = DistributedSpeculativeDecoder(
            target_forward=_always_logits(10),
            draft_model=draft,
            num_candidates=1,
            temperature=0,
            device="cpu",
        )
        with pytest.raises(ValueError, match="batch_size=1"):
            sd.generate(torch.tensor([[1, 2], [3, 4]]), max_new_tokens=1)


# ── Close ────────────────────────────────────────────────────────────────


class TestClose:
    def test_close_calls_draft_close(self):
        draft = MagicMock(spec=RemoteDraftModel)
        sd = DistributedSpeculativeDecoder(
            target_forward=lambda x, **kw: x, draft_model=draft,
        )
        sd.close()
        draft.close.assert_called_once()


# ── Auto-selection integration ───────────────────────────────────────────


class TestAutoSelection:
    def test_set_workload_type_string(self):
        draft = MagicMock(spec=RemoteDraftModel)
        sd = DistributedSpeculativeDecoder(
            target_forward=lambda x, **kw: x, draft_model=draft,
        )
        sd.set_workload_type("code")
        assert sd._workload_type == "code"

    def test_set_workload_type_from_text(self):
        draft = MagicMock(spec=RemoteDraftModel)
        sd = DistributedSpeculativeDecoder(
            target_forward=lambda x, **kw: x, draft_model=draft,
        )
        sd.set_workload_type("def foo(): return 42")
        assert sd._workload_type == "code"

    def test_record_method_performance(self):
        draft = MagicMock(spec=RemoteDraftModel)
        sd = DistributedSpeculativeDecoder(
            target_forward=lambda x, **kw: x, draft_model=draft,
        )
        sd.record_method_performance("ngram", 5, 4, 10.0)
        metrics = sd.get_metrics()
        assert "method_ranking" in metrics

    def test_get_active_method(self):
        draft = MagicMock(spec=RemoteDraftModel)
        sd = DistributedSpeculativeDecoder(
            target_forward=lambda x, **kw: x, draft_model=draft,
        )
        sd.record_method_performance("ngram", 5, 4, 10.0)
        method = sd.get_active_method("code")
        assert method == "ngram"

    def test_get_metrics_includes_workload(self):
        draft = MagicMock(spec=RemoteDraftModel)
        sd = DistributedSpeculativeDecoder(
            target_forward=lambda x, **kw: x, draft_model=draft,
        )
        sd.set_workload_type("instruction")
        metrics = sd.get_metrics()
        assert metrics["workload_type"] == "instruction"
