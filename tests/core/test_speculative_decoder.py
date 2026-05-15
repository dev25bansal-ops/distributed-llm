"""Tests for speculative decoding."""

import pytest
import torch
from unittest.mock import MagicMock

from distllm.core.speculative_decoder import SpeculativeDecoder


class MockTokenizer:
    """Mock tokenizer for testing."""
    def decode(self, tokens, **kwargs):
        return " ".join(str(t) for t in tokens)


class TestSpeculativeDecoder:
    """Tests for the SpeculativeDecoder class."""

    @pytest.fixture
    def decoder(self):
        return SpeculativeDecoder(
            num_assistant_tokens=3,
            min_acceptance_rate=0.3,
            warmup_steps=2,
        )

    def test_init_defaults(self):
        decoder = SpeculativeDecoder()
        assert decoder.num_assistant_tokens == 5
        assert decoder.min_acceptance_rate == 0.3
        assert decoder.warmup_steps == 10
        assert decoder.is_enabled is True
        assert decoder.acceptance_rate == 1.0

    def test_init_custom(self):
        decoder = SpeculativeDecoder(
            num_assistant_tokens=7,
            min_acceptance_rate=0.5,
            warmup_steps=20,
        )
        assert decoder.num_assistant_tokens == 7
        assert decoder.min_acceptance_rate == 0.5
        assert decoder.warmup_steps == 20

    def test_verify_and_accept_empty_draft(self, decoder):
        """When no draft tokens, sample directly from target."""
        logits = torch.tensor([[0.1, 0.5, 0.3, 0.1]])  # vocab size 4
        # Use greedy sampling for deterministic result
        token = decoder._sample_token(logits, temperature=0)
        assert token.item() == 1  # argmax of logits

    def test_verify_and_accept_all_match(self, decoder):
        """All draft tokens match target argmax."""
        # Target logits where argmax matches draft tokens [5, 3]
        logits = torch.zeros(1, 3, 10)  # [batch, seq_len, vocab]
        logits[0, 0, 5] = 10.0  # draft token 5 is argmax
        logits[0, 1, 3] = 10.0  # draft token 3 is argmax
        logits[0, 2, 7] = 10.0  # next token would be 7

        accepted, tokens, next_token = decoder.verify_and_accept(
            [5, 3], logits, MockTokenizer()
        )
        assert accepted == 2
        assert tokens == [5, 3]
        assert next_token == 7

    def test_verify_and_accept_partial_match(self, decoder):
        """First draft token matches, second doesn't."""
        logits = torch.zeros(1, 3, 10)
        logits[0, 0, 5] = 10.0  # matches draft[0]
        logits[0, 1, 8] = 10.0  # draft[1]=3, but target=8
        logits[0, 2, 7] = 10.0

        accepted, tokens, next_token = decoder.verify_and_accept(
            [5, 3], logits, MockTokenizer()
        )
        assert accepted == 1  # only first token accepted
        assert tokens == [5, 8]  # second position has target's token
        assert next_token == 8  # rejection point

    def test_verify_and_accept_none_match(self, decoder):
        """No draft tokens match target."""
        logits = torch.zeros(1, 3, 10)
        logits[0, 0, 9] = 10.0  # draft[0]=5, but target=9

        accepted, tokens, next_token = decoder.verify_and_accept(
            [5, 3], logits, MockTokenizer()
        )
        assert accepted == 0
        assert tokens == [9]  # target's token at position 0
        assert next_token == 9

    def test_verify_single_step(self, decoder):
        """Single-step verification (2D logits)."""
        logits = torch.tensor([[0.1, 0.8, 0.1]])  # [batch, vocab], argmax=1
        accepted, tokens, next_token = decoder.verify_and_accept(
            [1], logits, MockTokenizer()
        )
        assert accepted == 1
        assert tokens == [1]
        assert next_token == 1

    def test_verify_single_step_mismatch(self, decoder):
        """Single-step mismatch."""
        logits = torch.tensor([[0.1, 0.8, 0.1]])  # argmax=1
        accepted, tokens, next_token = decoder.verify_and_accept(
            [2], logits, MockTokenizer()
        )
        assert accepted == 0
        assert next_token == 1  # target's token

    def test_acceptance_rate_tracking(self, decoder):
        """Acceptance rate updates after warmup."""
        # First call during warmup - no EMA update
        decoder._record_acceptance(3, 3)
        assert decoder._step_count == 1
        assert decoder._acceptance_rate == 1.0  # unchanged during warmup

        # Second call - still during warmup (warmup_steps=2)
        decoder._record_acceptance(3, 3)
        assert decoder._step_count == 2

        # Third call - past warmup, EMA update
        decoder._record_acceptance(3, 0)  # 0% acceptance
        assert decoder._step_count == 3
        assert decoder._acceptance_rate < 1.0  # EMA decreased

    def test_auto_disable_low_acceptance(self, decoder):
        """Auto-disable when acceptance rate drops below threshold."""
        # Pass warmup period and exceed the warmup_steps * 2 threshold
        total_steps = decoder.warmup_steps * 2 + 5  # warmup_steps=2, so need > 4
        for _ in range(total_steps):
            decoder._record_acceptance(3, 0)  # 0% acceptance

        assert decoder.is_enabled is False

    def test_get_metrics(self, decoder):
        metrics = decoder.get_metrics()
        assert "acceptance_rate" in metrics
        assert "total_draft_tokens" in metrics
        assert "total_accepted" in metrics
        assert "step_count" in metrics
        assert "enabled" in metrics
        assert metrics["step_count"] == 0
        assert metrics["enabled"] is True

    def test_reset(self, decoder):
        decoder._total_draft_tokens = 100
        decoder._total_accepted = 50
        decoder._step_count = 20
        decoder._acceptance_rate = 0.5
        decoder._enabled = False

        decoder.reset()

        assert decoder._total_draft_tokens == 0
        assert decoder._total_accepted == 0
        assert decoder._step_count == 0
        assert decoder._acceptance_rate == 1.0
        assert decoder._enabled is True

    def test_sample_token_greedy(self, decoder):
        """Greedy sampling (temperature=0) returns argmax."""
        logits = torch.tensor([[0.1, 0.5, 0.3, 0.1]])
        token = decoder._sample_token(logits, temperature=0)
        assert token.item() == 1

    def test_sample_token_temperature(self, decoder):
        """Temperature sampling returns valid token."""
        logits = torch.tensor([[0.1, 0.5, 0.3, 0.1]])
        token = decoder._sample_token(logits, temperature=1.0)
        assert 0 <= token.item() <= 3

    def test_draft_tokens_exceeds_target_length(self, decoder):
        """Draft tokens longer than target output."""
        logits = torch.zeros(1, 2, 10)  # Only 2 positions
        logits[0, 0, 5] = 10.0
        logits[0, 1, 3] = 10.0

        accepted, tokens, next_token = decoder.verify_and_accept(
            [5, 3, 7, 9], logits, MockTokenizer()  # 4 draft tokens
        )
        assert accepted == 2
        assert tokens == [5, 3]
        assert next_token == 3  # All accepted, need more
