"""Tests for multi-draft speculative decoding with consensus."""

import importlib.util
import os

import torch


def _get_module():
    """Load speculative_decoder.py bypassing distllm circular import."""
    path = os.path.join("src", "distllm", "core", "speculative_decoder.py")
    spec = importlib.util.spec_from_file_location("speculative_decoder", path)
    mod = importlib.util.module_from_spec(spec)
    import types
    mod.torch = __import__("torch")
    mod.F = __import__("torch.nn.functional", fromlist=["nn"])
    mod.Any = __import__("typing").Any
    mod.Callable = __import__("typing").Callable
    logger = types.ModuleType("logger")
    logger.info = lambda *a, **kw: None
    logger.warning = lambda *a, **kw: None
    mod.logger = logger
    spec.loader.exec_module(mod)
    return mod


def _always_logit(token_id: int):
    """Return logits (1, 1, 100) with *token_id* having high value."""
    return torch.full((1, 1, 100), -10.0).scatter_(-1, torch.tensor([[[token_id]]]), 10.0)


def _identity_target(input_ids, **kwargs):
    """Target logits: position i predicts token at i+1."""
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


class TestMultiDraftSpeculativeDecoder:
    @classmethod
    def setup_class(cls):
        cls.mod = _get_module()
        cls.MD = cls.mod.MultiDraftSpeculativeDecoder

    def test_init_requires_at_least_two_models(self):
        import pytest
        with pytest.raises(ValueError, match=">=2 draft models"):
            self.MD(target_forward=lambda x: x, draft_forwards=[lambda x: x])

    def test_init_accepts_two_or_more_models(self):
        d = self.MD(target_forward=lambda x: torch.randn(1, 1, 100),
                     draft_forwards=[lambda x: torch.randn(1, 1, 100),
                                     lambda x: torch.randn(1, 1, 100)])
        assert len(d._draft_forwards) == 2

    def test_consensus_draft_all_agree(self):
        d = self.MD(target_forward=lambda x: torch.randn(1, 1, 100),
                     draft_forwards=[lambda x: _always_logit(10),
                                     lambda x: _always_logit(10)],
                     num_candidates=4, temperature=0)
        tokens, length = d._generate_consensus_draft(torch.tensor([[1, 2, 3]]), 3)
        assert length == 3
        assert tokens.shape == (1, 3)
        assert tokens[0, 0].item() == 10

    def test_consensus_draft_disagree(self):
        d = self.MD(target_forward=lambda x: torch.randn(1, 1, 100),
                     draft_forwards=[lambda x: _always_logit(5),
                                     lambda x: _always_logit(7)],
                     num_candidates=4, temperature=0)
        tokens, length = d._generate_consensus_draft(torch.tensor([[1, 2, 3]]), 3)
        assert length == 0
        assert tokens.shape[1] == 0

    def test_consensus_draft_partial_agreement(self):
        from unittest.mock import MagicMock
        a = MagicMock(side_effect=[_always_logit(42), _always_logit(99)])
        b = MagicMock(side_effect=[_always_logit(42), _always_logit(88)])
        d = self.MD(target_forward=lambda x: torch.randn(1, 1, 100),
                     draft_forwards=[a, b], num_candidates=4, temperature=0)
        tokens, length = d._generate_consensus_draft(torch.tensor([[1, 2, 3]]), 3)
        assert length == 1
        assert tokens[0, 0].item() == 42

    def test_verify_accepted(self):
        prefix = torch.tensor([[1, 2, 3]])
        consensus = torch.tensor([[10, 20]])
        full_input = torch.cat([prefix, consensus], dim=1)
        target_logits = _identity_target(full_input)
        d = self.MD(target_forward=_identity_target,
                     draft_forwards=[lambda x: _always_logit(10),
                                     lambda x: _always_logit(10)],
                     temperature=0)
        accepted = d._verify_tokens(prefix, full_input, consensus, target_logits)
        assert accepted == 2

    def test_verify_none_accepted(self):
        def target_fn(input_ids, **kw):
            logits = torch.full((1, input_ids.shape[1], 100), -10.0)
            logits[0, :, 99] = 10.0
            return logits
        prefix = torch.tensor([[1, 2, 3]])
        consensus = torch.tensor([[5]])
        full_input = torch.cat([prefix, consensus], dim=1)
        d = self.MD(target_forward=target_fn,
                     draft_forwards=[lambda x: _always_logit(5),
                                     lambda x: _always_logit(5)],
                     temperature=0)
        accepted = d._verify_tokens(prefix, full_input, consensus, target_fn(full_input))
        assert accepted == 0

    def test_generate_full_consensus(self):
        """All models predict 10, target accepts it — all generated tokens are 10."""
        d = self.MD(target_forward=_identity_target,
                     draft_forwards=[lambda x: _always_logit(10),
                                     lambda x: _always_logit(10),
                                     lambda x: _always_logit(10)],
                     num_candidates=5, temperature=0)
        output = d.generate(torch.tensor([[1, 2, 3]]), max_new_tokens=5)
        assert output.shape[1] == 8
        # All generated tokens should be 10
        assert all(output[0, 3 + i].item() == 10 for i in range(5))

    def test_generate_partial_consensus(self):
        """Draft models agree on first token (10), disagree on second — first is accepted."""
        class _DivergeAfterFirst:
            def __init__(self, second_token):
                self._call = 0
                self._second = second_token
            def __call__(self, input_ids, **kwargs):
                self._call += 1
                tgt = 10 if self._call <= 2 else self._second
                return _always_logit(tgt)

        d = self.MD(target_forward=_identity_target,
                     draft_forwards=[_DivergeAfterFirst(50),
                                     _DivergeAfterFirst(50),
                                     _DivergeAfterFirst(99)],
                     num_candidates=5, temperature=0)
        output = d.generate(torch.tensor([[1, 2, 3]]), max_new_tokens=3)
        assert output.shape[1] == 6
        assert output[0, 3].item() == 10

    def test_generate_no_consensus(self):
        """Models disagree at position 0 — fallback to target-only generation."""
        d = self.MD(target_forward=_identity_target,
                     draft_forwards=[lambda x: _always_logit(5),
                                     lambda x: _always_logit(7)],
                     num_candidates=3, temperature=0)
        output = d.generate(torch.tensor([[1, 2, 3]]), max_new_tokens=3)
        assert output.shape[1] == 6

    def test_generate_no_consensus_fallback_produces_tokens(self):
        """After no-consensus fallback, generation continues normally."""
        d = self.MD(target_forward=_identity_target,
                     draft_forwards=[lambda x: _always_logit(5),
                                     lambda x: _always_logit(7)],
                     num_candidates=3, temperature=0)
        output = d.generate(torch.tensor([[1, 2, 3]]), max_new_tokens=5)
        assert output.shape[1] == 8

    def test_consensus_length_stats(self):
        d = self.MD(target_forward=_identity_target,
                     draft_forwards=[lambda x: _always_logit(10),
                                     lambda x: _always_logit(10)],
                     num_candidates=5, temperature=0)
        d.generate(torch.tensor([[1, 2, 3]]), max_new_tokens=5)
        stats = d.stats
        assert "consensus_lengths" in stats
        assert len(stats["consensus_lengths"]) > 0
        assert max(stats["consensus_lengths"]) <= 5

    def test_stats_property(self):
        d = self.MD(target_forward=lambda x: torch.randn(1, 1, 100),
                     draft_forwards=[lambda x: torch.randn(1, 1, 100),
                                     lambda x: torch.randn(1, 1, 100)])
        stats = d.stats
        assert "draft_calls" in stats
        assert "target_calls" in stats
        assert "consensus_lengths" in stats

    def test_sample_greedy(self):
        d = self.MD(target_forward=lambda x: torch.randn(1, 1, 100),
                     draft_forwards=[lambda x: torch.randn(1, 1, 100),
                                     lambda x: torch.randn(1, 1, 100)])
        logits = torch.randn(1, 100)
        token = d._sample(logits)
        assert token.shape == (1, 1)

    def test_sample_temperature_zero(self):
        d = self.MD(target_forward=lambda x: torch.randn(1, 1, 100),
                     draft_forwards=[lambda x: torch.randn(1, 1, 100),
                                     lambda x: torch.randn(1, 1, 100)],
                     temperature=0)
        logits = torch.full((1, 100), -10.0)
        logits[0, 42] = 10.0
        token = d._sample(logits)
        assert token.item() == 42

    def test_generate_extra_kwargs(self):
        def target_fn(input_ids, **kwargs):
            assert "extra" in kwargs
            return _identity_target(input_ids)
        calls = [0]
        def draft_fn(input_ids, **kwargs):
            calls[0] += 1
            assert "extra" in kwargs
            return _always_logit(10)
        d = self.MD(target_forward=target_fn,
                     draft_forwards=[draft_fn, draft_fn],
                     num_candidates=2, temperature=0)
        output = d.generate(torch.tensor([[1, 2, 3]]), max_new_tokens=4, extra="value")
        assert output.shape[1] == 7

    def test_different_num_candidates(self):
        for nc in [1, 3, 10]:
            d = self.MD(target_forward=lambda x: torch.randn(1, 1, 100),
                         draft_forwards=[lambda x: _always_logit(10),
                                         lambda x: _always_logit(10)],
                         num_candidates=nc, temperature=0)
            tokens, length = d._generate_consensus_draft(torch.tensor([[1, 2, 3]]), nc)
            assert length == nc
            assert tokens.shape[1] == nc

    def test_generate_respects_max_new_tokens(self):
        d = self.MD(target_forward=_identity_target,
                     draft_forwards=[lambda x: _always_logit(10),
                                     lambda x: _always_logit(10)],
                     num_candidates=10, temperature=0)
        output = d.generate(torch.tensor([[1]]), max_new_tokens=2)
        assert output.shape[1] == 3  # 1 prompt + 2 generated

    def test_stats_after_generate(self):
        d = self.MD(target_forward=_identity_target,
                     draft_forwards=[lambda x: _always_logit(10),
                                     lambda x: _always_logit(10)],
                     num_candidates=3, temperature=0)
        d.generate(torch.tensor([[1, 2, 3]]), max_new_tokens=5)
        stats = d.stats
        assert stats["draft_calls"] > 0
        assert stats["target_calls"] > 0
        assert stats["accepted"] > 0
        assert stats["total_proposed"] > 0
        assert stats["acceptance_rate"] > 0


def test_module_has_multi_draft():
    mod = _get_module()
    assert hasattr(mod, "MultiDraftSpeculativeDecoder")
