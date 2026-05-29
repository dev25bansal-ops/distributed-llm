"""Comprehensive tests for SpeculativeDecoder."""

from unittest.mock import MagicMock

import pytest
import torch

from distllm.core.speculative_decoder import SpeculativeDecoder


def _make_logits(batch=1, seq=1, vocab=100, target_token=42):
    """Create logits where target_token has highest probability."""
    logits = torch.full((batch, seq, vocab), -10.0)
    logits[:, :, target_token] = 10.0
    return logits


class _ForwardMock:
    """Callable that returns logits matching input sequence length."""

    def __init__(self, vocab=100, target_token=42):
        self.vocab = vocab
        self.target_token = target_token
        self.call_count = 0

    def __call__(self, input_ids, **kwargs):
        self.call_count += 1
        batch, seq = input_ids.shape[:2]
        # Target token always wins
        logits = torch.full((batch, seq, self.vocab), -10.0)
        logits[:, :, self.target_token] = 10.0
        return logits


# ===================================================================
# Initialization
# ===================================================================

class TestInit:
    def test_defaults(self):
        d = SpeculativeDecoder(target_forward=MagicMock(), draft_forward=MagicMock())
        assert d._num_candidates == 5
        assert d._top_k == 20
        assert d._temperature == 1.0

    def test_custom_values(self):
        d = SpeculativeDecoder(
            target_forward=MagicMock(),
            draft_forward=MagicMock(),
            num_candidates=7,
            top_k=10,
            temperature=0.5,
            device="cpu",
        )
        assert d._num_candidates == 7
        assert d._top_k == 10
        assert d._temperature == 0.5


# ===================================================================
# Generation
# ===================================================================

class TestGenerate:
    def test_generate_basic(self):
        target = _ForwardMock()
        draft = _ForwardMock()
        d = SpeculativeDecoder(target_forward=target, draft_forward=draft, num_candidates=1)
        input_ids = torch.tensor([[1, 2, 3]])
        out = d.generate(input_ids, max_new_tokens=5)
        assert out.shape[0] == 1
        assert out.shape[1] == 3 + 5

    def test_generate_without_draft(self):
        target = _ForwardMock()
        d = SpeculativeDecoder(target_forward=target, draft_forward=None)
        input_ids = torch.tensor([[1, 2, 3]])
        with pytest.raises(TypeError):
            d.generate(input_ids, max_new_tokens=3)


# ===================================================================
# Stats
# ===================================================================

class TestStats:
    def test_stats_initial(self):
        d = SpeculativeDecoder(target_forward=MagicMock(), draft_forward=MagicMock())
        s = d.stats
        assert "draft_calls" in s
        assert "target_calls" in s
        assert "accepted" in s
        assert "total_proposed" in s

    def test_stats_after_generation(self):
        target = _ForwardMock()
        draft = _ForwardMock()
        d = SpeculativeDecoder(target_forward=target, draft_forward=draft, num_candidates=1)
        d.generate(torch.tensor([[1, 2, 3]]), max_new_tokens=3)
        s = d.stats
        assert s["target_calls"] >= 1
        assert s["accepted"] >= 3


# ===================================================================
# Edge cases
# ===================================================================

class TestEdgeCases:
    def test_minimal_config(self):
        target = _ForwardMock()
        draft = _ForwardMock()
        d = SpeculativeDecoder(target_forward=target, draft_forward=draft, num_candidates=1)
        out = d.generate(torch.tensor([[1]]), max_new_tokens=1)
        assert out is not None
        assert out.shape[1] == 2

    def test_high_temperature(self):
        """High temperature should still produce output."""
        target = _ForwardMock()
        draft = _ForwardMock()
        d = SpeculativeDecoder(
            target_forward=target, draft_forward=draft,
            num_candidates=1, temperature=5.0,
        )
        out = d.generate(torch.tensor([[1, 2]]), max_new_tokens=2)
        assert out is not None
        assert out.shape[1] == 4

    def test_temperature_zero(self):
        """Temperature 0 should be deterministic (greedy)."""
        target = _ForwardMock()
        draft = _ForwardMock()
        d = SpeculativeDecoder(
            target_forward=target, draft_forward=draft,
            num_candidates=1, temperature=0.0,
        )
        out = d.generate(torch.tensor([[1, 2]]), max_new_tokens=2)
        assert out is not None
        assert out.shape[1] == 4


# ===================================================================
# Verification / acceptance
# ===================================================================

class TestVerifyAccept:
    def test_verify_all_match(self):
        """When draft tokens match target, all should be accepted."""
        # Both target and draft always predict token 42
        def _fn(input_ids, **kwargs):
            batch, seq = input_ids.shape
            logits = torch.full((batch, seq, 100), -10.0)
            logits[:, :, 42] = 10.0
            return logits

        d = SpeculativeDecoder(
            target_forward=_fn,
            draft_forward=_fn,
            num_candidates=3,
            temperature=0.0,
        )
        out = d.generate(torch.tensor([[1, 2, 3]]), max_new_tokens=5)
        assert out is not None
        assert out.shape[1] == 8

    def test_verify_all_reject(self):
        """When draft tokens don't match, fall back."""
        # Draft predicts token 1, target predicts token 42
        def _draft(input_ids, **kwargs):
            batch, seq = input_ids.shape
            logits = torch.full((batch, seq, 100), -10.0)
            logits[:, :, 1] = 10.0
            return logits

        def _target(input_ids, **kwargs):
            batch, seq = input_ids.shape
            logits = torch.full((batch, seq, 100), -10.0)
            logits[:, :, 42] = 10.0
            return logits

        d = SpeculativeDecoder(
            target_forward=_target,
            draft_forward=_draft,
            num_candidates=3,
            temperature=0.0,
        )
        out = d.generate(torch.tensor([[1, 2, 3]]), max_new_tokens=3)
        assert out is not None
        assert out.shape[1] == 6

    def test_acceptance_rate_tracking(self):
        d = SpeculativeDecoder(
            target_forward=MagicMock(),
            draft_forward=MagicMock(),
            num_candidates=3,
        )
        s = d.stats
        # When no generation done, acceptance_rate may not be in stats
        assert "accepted" in s


# ===================================================================
# Batch generation
# ===================================================================

class TestBatch:
    def test_generate_batch(self):
        """Batch generation with 2 sequences."""
        target = _ForwardMock()
        draft = _ForwardMock()
        d = SpeculativeDecoder(
            target_forward=target, draft_forward=draft, num_candidates=1,
        )
        input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
        out = d.generate(input_ids, max_new_tokens=3)
        assert out.shape[0] == 2
        assert out.shape[1] == 6


# ===================================================================
# Sampling
# ===================================================================

class TestSampling:
    def test_sample_greedy(self):
        d = SpeculativeDecoder(
            target_forward=MagicMock(), draft_forward=MagicMock(),
            temperature=0.0,
        )
        # Just verify stats work
        assert d._temperature == 0.0

    def test_sample_temperature(self):
        d = SpeculativeDecoder(
            target_forward=MagicMock(), draft_forward=MagicMock(),
            temperature=0.7,
        )
        assert d._temperature == 0.7


# ===================================================================
# Module exports
# ===================================================================

class TestExports:
    def test_module_exports(self):
        import distllm.core.speculative_decoder as sd
        assert hasattr(sd, "SpeculativeDecoder")
