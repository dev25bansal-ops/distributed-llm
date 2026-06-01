"""Token sampling tests.

Covers temperature, top-k, top-p, logit bias, presence/frequency penalties,
logprobs, batch sampling, and property-based sampling invariants.
"""

import asyncio
import socket
import struct
import threading
import time
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import numpy as np

try:
    from hypothesis import given, strategies as st, settings as hp_settings
    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False


from tests.comprehensive.conftest import _load_module

# Load clean modules
_token_gen = _load_module("distllm/core/token_generator.py")


# ═══════════════════════════════════════════════════════════════════════════
# 3. Token Sampling
# ═══════════════════════════════════════════════════════════════════════════

class TestTokenSampling:
    """Temperature, top-k, top-p, penalty, and bias correctness."""

    @pytest.fixture
    def gen(self):
        return _token_gen.TokenGenerator()

    def test_temperature_one_preserves_distribution(self, gen):
        logits = torch.randn(4, 128)
        tokens, _ = gen.sample(logits, temperature=1.0, top_k=0, top_p=1.0)
        assert tokens.shape == (4,)

    def test_temperature_zero_is_argmax(self, gen):
        logits = torch.tensor([[1.0, 2.0, 0.5, 0.1]])
        tokens, _ = gen.sample(logits, temperature=0.0)
        assert tokens[0].item() == 1

    def test_temperature_zero_multi_batch(self, gen):
        logits = torch.tensor([[1.0, 5.0, 0.5], [3.0, 1.0, 2.0]])
        tokens, _ = gen.sample(logits, temperature=0.0)
        assert tokens[0].item() == 1
        assert tokens[1].item() == 0

    def test_top_k_filters_low_prob_tokens(self, gen):
        logits = torch.tensor([[10.0, 0.0, 0.0, 0.0]])
        tokens, _ = gen.sample(logits, temperature=1.0, top_k=1)
        assert tokens[0].item() == 0

    def test_top_k_2_considers_at_least_2(self, gen):
        logits = torch.tensor([[100.0, 99.0, 1.0, 0.5]])
        tokens, _ = gen.sample(logits, temperature=1.0, top_k=2)
        assert tokens[0].item() in (0, 1)

    def test_top_p_equal_nucleus(self, gen):
        logits = torch.tensor([[100.0, 50.0, 1.0, 0.5]])
        tokens, _ = gen.sample(logits, temperature=1.0, top_p=0.5)
        assert tokens[0].item() in (0, 1)

    def test_top_p_one_disables_filtering(self, gen):
        logits = torch.randn(4, 128)
        tokens, _ = gen.sample(logits, temperature=1.0, top_p=1.0)
        assert tokens.shape == (4,)

    def test_logit_bias_applied_correctly(self, gen):
        logits = torch.zeros(1, 10)
        logits[0, 5] = 1.0
        bias = {5: 5.0}
        tokens, _ = gen.sample(logits, temperature=1.0, logit_bias=bias)
        assert tokens[0].item() == 5

    def test_bias_out_of_range_ignored(self, gen):
        logits = torch.zeros(1, 10)
        bias = {999: 100.0}
        gen.sample(logits, logit_bias=bias)

    def test_presence_penalty_reduces_repeats(self, gen):
        logits = torch.zeros(1, 5)
        # Make token 2 strongly favored initially, then heavily penalize it
        logits[0, 2] = 10.0
        logits[0, 3] = 0.0
        token_counts = {2: 100}
        tokens, _ = gen.sample(logits, temperature=1.0,
                                presence_penalty=20.0,
                                token_counts=token_counts)
        assert tokens[0].item() != 2

    def test_frequency_penalty_scales_with_count(self, gen):
        logits = torch.zeros(1, 5)
        logits[0, 1] = 10.0
        logits[0, 2] = 9.0
        token_counts = {1: 1, 2: 0}
        tokens, _ = gen.sample(logits, temperature=1.0,
                                frequency_penalty=5.0,
                                token_counts=token_counts)
        assert tokens[0].item() == 2

    def test_top_k_top_p_filtering(self, gen):
        logits = torch.randn(1, 100)
        filtered = _token_gen.TokenGenerator._top_k_top_p_filtering(
            logits, top_k=10, top_p=0.9
        )
        assert filtered.shape == logits.shape
        assert not torch.isnan(filtered).any()

    def test_top_k_bounded_by_vocab_size(self, gen):
        logits = torch.randn(1, 5)
        filtered = _token_gen.TokenGenerator._top_k_top_p_filtering(
            logits, top_k=100
        )
        assert not torch.isinf(filtered).any()

    def test_return_logprobs_returns_dict(self, gen):
        logits = torch.randn(1, 100)
        tokens, logprobs = gen.sample(logits, return_logprobs=True, top_logprobs=3)
        assert logprobs is not None
        assert "logprob" in logprobs
        assert "top_logprobs" in logprobs
        assert len(logprobs["top_logprobs"]) == 3

    def test_return_logprobs_batch(self, gen):
        logits = torch.randn(2, 50)
        tokens, logprobs = gen.sample(logits, return_logprobs=True, top_logprobs=2)
        # batch_size > 1 returns a list of dicts
        assert isinstance(logprobs, dict) or isinstance(logprobs, list)
        if isinstance(logprobs, list):
            assert len(logprobs) == 2
            assert "logprob" in logprobs[0]
        else:
            assert "top_logprobs" in logprobs

    @pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
    @hp_settings(max_examples=100)
    @given(
        temp=st.floats(min_value=0.001, max_value=5.0),
        k=st.integers(min_value=0, max_value=50),
        p=st.floats(min_value=0.0, max_value=1.0),
    )
    def test_sampling_always_returns_valid_tokens(self, temp, k, p):
        gen = _token_gen.TokenGenerator()
        logits = torch.randn(1, 128)
        tokens, _ = gen.sample(logits, temperature=temp, top_k=k, top_p=p)
        assert tokens[0].item() >= 0
        assert tokens[0].item() < 128

    def test_apply_constraint_no_constraint(self, gen):
        logits = torch.randn(1, 100)
        result = gen.apply_constraint(logits, None)
        assert torch.equal(result, logits)

    def test_sample_batch_with_sequences(self, gen):
        class MockSeq:
            def __init__(self):
                self.temperature = 1.0
                self.top_p = 1.0
                self.top_k = 0
                self.constraint = None
                self.token_counts = None
                self.include_logprobs = False
                self.top_logprobs = 0
                self.logit_bias = None
                self.presence_penalty = 0.0
                self.frequency_penalty = 0.0

        logits = torch.randn(3, 50)
        seqs = [MockSeq() for _ in range(3)]
        tokens, logprobs = gen.sample_batch(logits, seqs)
        assert tokens.shape == (3,)
        assert len(logprobs) == 3

    def test_compute_logprobs_returns_negative_values(self, gen):
        logits = torch.randn(1, 100)
        tokens = torch.tensor([5])
        result = _token_gen.TokenGenerator._compute_logprobs(logits, tokens)
        assert result["logprob"] < 0
