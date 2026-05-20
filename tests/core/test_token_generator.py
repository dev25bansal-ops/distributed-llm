"""Tests for TokenGenerator: sampling, constraints, logprobs, penalties, batch.

Tests: sample (greedy, temperature, top-k, top-p), logit_bias,
presence/frequency penalties, _compute_logprobs, sample_batch,
apply_constraint, and edge cases.

Run: pytest tests/core/test_token_generator.py -v
"""

from unittest.mock import MagicMock

import pytest
import torch

from distllm.core.token_generator import TokenGenerator


# --- Init tests ---


class TestTokenGeneratorInit:
    """Tests for TokenGenerator initialization."""

    def test_no_tokenizer(self):
        gen = TokenGenerator()
        assert gen.tokenizer is None

    def test_with_tokenizer(self):
        mock_tokenizer = MagicMock()
        gen = TokenGenerator(tokenizer=mock_tokenizer)
        assert gen.tokenizer is mock_tokenizer


# --- Logit bias tests ---


class TestApplyLogitBias:
    """Tests for _apply_logit_bias."""

    def test_positive_bias(self):
        gen = TokenGenerator()
        logits = torch.tensor([[0.0, 0.0, 0.0]])
        biased = gen._apply_logit_bias(logits, {1: 10.0})
        assert biased[0, 1] == 10.0

    def test_negative_bias(self):
        gen = TokenGenerator()
        logits = torch.tensor([[0.0, 0.0, 0.0]])
        biased = gen._apply_logit_bias(logits, {0: -5.0})
        assert biased[0, 0] == -5.0

    def test_multiple_biases(self):
        gen = TokenGenerator()
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        biased = gen._apply_logit_bias(logits, {0: 1.0, 2: -1.0})
        assert biased[0, 0] == 2.0
        assert biased[0, 1] == 2.0
        assert biased[0, 2] == 2.0

    def test_out_of_range_token_id_ignored(self):
        gen = TokenGenerator()
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        biased = gen._apply_logit_bias(logits, {100: 50.0, -1: -50.0})
        assert torch.allclose(biased, logits)

    def test_empty_bias_no_change(self):
        gen = TokenGenerator()
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        biased = gen._apply_logit_bias(logits, {})
        assert torch.allclose(biased, logits)

    def test_batch_logits(self):
        gen = TokenGenerator()
        logits = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        biased = gen._apply_logit_bias(logits, {0: 10.0})
        assert biased[0, 0] == 11.0
        assert biased[1, 0] == 13.0


# --- Penalty tests ---


class TestApplyPenalties:
    """Tests for _apply_penalties."""

    def test_no_token_counts_returns_same(self):
        gen = TokenGenerator()
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        result = gen._apply_penalties(logits, 0.5, 0.5, None)
        assert torch.allclose(result, logits)

    def test_zero_penalties_no_change(self):
        gen = TokenGenerator()
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        result = gen._apply_penalties(logits, 0.0, 0.0, {1: 3})
        assert torch.allclose(result, logits)

    def test_presence_penalty(self):
        gen = TokenGenerator()
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        result = gen._apply_penalties(logits, 1.0, 0.0, {1: 5})
        assert result[0, 1] == 1.0  # 2.0 - 1.0

    def test_frequency_penalty(self):
        gen = TokenGenerator()
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        result = gen._apply_penalties(logits, 0.0, 0.5, {1: 4})
        assert result[0, 1] == 0.0  # 2.0 - 0.5*4

    def test_combined_penalties(self):
        gen = TokenGenerator()
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        result = gen._apply_penalties(logits, 1.0, 0.5, {1: 3})
        # presence=1.0 + frequency=0.5*3 = 2.5
        assert result[0, 1] == -0.5  # 2.0 - 2.5

    def test_only_penalizes_present_tokens(self):
        gen = TokenGenerator()
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        result = gen._apply_penalties(logits, 1.0, 0.3, {0: 5})
        # Token 0: presence(1.0) + frequency(0.3*5=1.5) = 2.5 penalty => 1.0 - 2.5 = -1.5
        assert result[0, 0] == -1.5
        assert result[0, 1] == 2.0  # unchanged
        assert result[0, 2] == 3.0  # unchanged

    def test_out_of_range_token_id_ignored(self):
        gen = TokenGenerator()
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        result = gen._apply_penalties(logits, 1.0, 1.0, {100: 5})
        assert torch.allclose(result, logits)


# --- Sample tests ---


class TestSampleGreedy:
    """Tests for sample() with temperature=0 (argmax)."""

    def test_single_batch_argmax(self):
        gen = TokenGenerator()
        logits = torch.tensor([[0.1, 0.5, 0.3, 0.1]])
        tokens, logprobs = gen.sample(logits, temperature=0)
        assert tokens.item() == 1

    def test_batch_argmax(self):
        gen = TokenGenerator()
        logits = torch.tensor([
            [0.1, 0.9, 0.2],
            [0.5, 0.1, 0.3],
        ])
        tokens, _ = gen.sample(logits, temperature=0)
        assert tokens[0].item() == 1
        assert tokens[1].item() == 0

    def test_no_logprobs_by_default(self):
        gen = TokenGenerator()
        logits = torch.tensor([[0.1, 0.5, 0.3]])
        tokens, logprobs = gen.sample(logits, temperature=0)
        assert logprobs is None


class TestSampleTemperature:
    """Tests for sample() with temperature > 0."""

    def test_temperature_returns_valid_token(self):
        gen = TokenGenerator()
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        tokens, _ = gen.sample(logits, temperature=1.0)
        assert 0 <= tokens.item() <= 3

    def test_high_temperature_uniform(self):
        gen = TokenGenerator()
        logits = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
        tokens, _ = gen.sample(logits, temperature=100.0)
        assert 0 <= tokens.item() <= 3

    def test_deterministic_with_temperature_seed(self):
        """Multiple calls with same logits should return valid tokens."""
        gen = TokenGenerator()
        logits = torch.tensor([[0.0, 10.0, 0.0]])
        tokens, _ = gen.sample(logits, temperature=0.1)
        assert tokens.item() == 1  # Should heavily favor index 1


class TestSampleTopK:
    """Tests for sample() with top-k filtering."""

    def test_topk_1_is_argmax(self):
        gen = TokenGenerator()
        logits = torch.tensor([[0.1, 0.9, 0.2]])
        tokens, _ = gen.sample(logits, temperature=1.0, top_k=1)
        assert tokens.item() == 1

    def test_topk_reduces_choices(self):
        gen = TokenGenerator()
        logits = torch.tensor([[10.0, 0.0, 0.0, 0.0, 0.0]])
        tokens, _ = gen.sample(logits, temperature=0.1, top_k=2)
        assert tokens.item() == 0


class TestSampleTopP:
    """Tests for sample() with nucleus (top-p) sampling."""

    def test_top_p_1_no_filtering(self):
        gen = TokenGenerator()
        logits = torch.tensor([[0.1, 0.5, 0.3]])
        tokens, _ = gen.sample(logits, temperature=1.0, top_p=1.0)
        assert 0 <= tokens.item() <= 2

    def test_top_p_low_filters_aggressively(self):
        gen = TokenGenerator()
        logits = torch.tensor([[100.0, 0.0, 0.0, 0.0]])
        tokens, _ = gen.sample(logits, temperature=0.01, top_p=0.1)
        assert tokens.item() == 0


class TestSampleLogprobs:
    """Tests for sample() with return_logprobs=True."""

    def test_logprobs_returned(self):
        gen = TokenGenerator()
        logits = torch.tensor([[0.1, 0.5, 0.3, 0.1]])
        tokens, logprobs = gen.sample(logits, temperature=0, return_logprobs=True)
        assert logprobs is not None
        assert "logprob" in logprobs
        assert isinstance(logprobs["logprob"], float)

    def test_logprob_value_correct(self):
        gen = TokenGenerator()
        logits = torch.tensor([[0.0, 0.0, 0.0]])  # uniform
        tokens, logprobs = gen.sample(logits, temperature=0, return_logprobs=True)
        # For uniform 3-way, log(1/3) ~ -1.098
        expected = torch.log(torch.tensor(1.0 / 3.0)).item()
        assert abs(logprobs["logprob"] - expected) < 0.01

    def test_top_logprobs(self):
        gen = TokenGenerator()
        logits = torch.tensor([[1.0, 2.0, 3.0, 0.5, 0.1]])
        _, logprobs = gen.sample(logits, temperature=0, return_logprobs=True, top_logprobs=3)
        assert "top_logprobs" in logprobs
        assert len(logprobs["top_logprobs"]) == 3

    def test_logprobs_with_tokenizer(self):
        gen = TokenGenerator()
        mock_tokenizer = MagicMock()
        mock_tokenizer.decode.return_value = "hello"
        logits = torch.tensor([[0.1, 0.5, 0.3]])
        tokens, logprobs = gen.sample(
            logits, temperature=0, return_logprobs=True, top_logprobs=2,
        )
        # _compute_logprobs is called internally but tokenizer isn't passed through sample
        # The logprobs structure should still have logprob
        assert logprobs is not None
        assert "logprob" in logprobs


# --- _compute_logprobs tests ---


class TestComputeLogprobs:
    """Tests for _compute_logprobs static method."""

    def test_single_token(self):
        logits = torch.tensor([[0.1, 0.5, 0.3, 0.1]])
        tokens = torch.tensor([1])
        result = TokenGenerator._compute_logprobs(logits, tokens, temperature=1.0)
        assert isinstance(result, dict)
        assert "logprob" in result

    def test_batch_tokens(self):
        logits = torch.tensor([
            [0.1, 0.5, 0.3],
            [0.3, 0.1, 0.5],
        ])
        tokens = torch.tensor([1, 2])
        result = TokenGenerator._compute_logprobs(logits, tokens, temperature=1.0)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_top_logprobs_entries(self):
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        tokens = torch.tensor([4])
        result = TokenGenerator._compute_logprobs(logits, tokens, top_logprobs=3, temperature=1.0)
        assert len(result["top_logprobs"]) == 3

    def test_with_tokenizer(self):
        mock_tokenizer = MagicMock()
        mock_tokenizer.decode.return_value = "test"
        logits = torch.tensor([[0.1, 0.5, 0.3]])
        tokens = torch.tensor([1])
        result = TokenGenerator._compute_logprobs(
            logits, tokens, tokenizer=mock_tokenizer, temperature=1.0,
        )
        assert result["token"] == "test"
        assert result["bytes"] is not None

    def test_single_token_scalar(self):
        """Test with scalar token input."""
        logits = torch.tensor([[0.1, 0.5, 0.3]])
        tokens = torch.tensor(1)
        result = TokenGenerator._compute_logprobs(logits, tokens, temperature=1.0)
        assert isinstance(result, dict)
        assert "logprob" in result


# --- Sample batch tests ---


class TestSampleBatch:
    """Tests for sample_batch."""

    def test_basic_batch(self):
        gen = TokenGenerator()
        logits = torch.tensor([
            [0.1, 0.9, 0.2],
            [0.7, 0.1, 0.3],
        ])
        sequences = [
            MagicMock(temperature=0, top_p=1.0, top_k=0, constraint=None,
                      token_counts=None, include_logprobs=False, top_logprobs=0,
                      logit_bias=None, presence_penalty=0.0, frequency_penalty=0.0),
            MagicMock(temperature=0, top_p=1.0, top_k=0, constraint=None,
                      token_counts=None, include_logprobs=False, top_logprobs=0,
                      logit_bias=None, presence_penalty=0.0, frequency_penalty=0.0),
        ]

        tokens, logprobs = gen.sample_batch(logits, sequences)

        assert tokens.shape[0] == 2
        assert tokens[0].item() == 1
        assert tokens[1].item() == 0
        assert len(logprobs) == 2

    def test_batch_with_logprobs(self):
        gen = TokenGenerator()
        logits = torch.tensor([
            [0.1, 0.9, 0.2],
        ])
        sequences = [
            MagicMock(temperature=0, top_p=1.0, top_k=0, constraint=None,
                      token_counts=None, include_logprobs=True, top_logprobs=2,
                      logit_bias=None, presence_penalty=0.0, frequency_penalty=0.0),
        ]

        tokens, logprobs = gen.sample_batch(logits, sequences)

        assert logprobs[0] is not None


# --- Apply constraint tests ---


class TestApplyConstraint:
    """Tests for apply_constraint."""

    def test_no_constraint_returns_same(self):
        gen = TokenGenerator()
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        result = gen.apply_constraint(logits, None)
        assert torch.allclose(result, logits)

    def test_with_constraint(self):
        gen = TokenGenerator()
        mock_constraint = MagicMock()
        mock_constraint.get_logits_mask.return_value = torch.tensor([True, False, True])
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        result = gen.apply_constraint(logits, mock_constraint)
        assert result[0, 1] == float('-inf')
        assert result[0, 0] == 1.0
        assert result[0, 2] == 3.0

    def test_with_tokenizer_override(self):
        gen = TokenGenerator()
        mock_constraint = MagicMock()
        mock_constraint.get_logits_mask.return_value = torch.tensor([True, True])
        mock_tokenizer = MagicMock()
        logits = torch.tensor([[1.0, 2.0]])
        result = gen.apply_constraint(logits, mock_constraint, tokenizer=mock_tokenizer)
        mock_constraint.get_logits_mask.assert_called_once_with(2, mock_tokenizer)


# --- Edge cases ---


class TestTokenGeneratorEdgeCases:
    """Tests for edge cases."""

    def test_very_small_temperature(self):
        gen = TokenGenerator()
        logits = torch.tensor([[0.0, 10.0, 0.0]])
        tokens, _ = gen.sample(logits, temperature=1e-8)
        assert tokens.item() == 1

    def test_large_vocab(self):
        gen = TokenGenerator()
        logits = torch.randn(1, 50000)
        tokens, _ = gen.sample(logits, temperature=1.0)
        assert 0 <= tokens.item() < 50000

    def test_batch_with_penalties_and_counts(self):
        gen = TokenGenerator()
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        sequences = [
            MagicMock(temperature=0, top_p=1.0, top_k=0, constraint=None,
                      token_counts={0: 5}, include_logprobs=False, top_logprobs=0,
                      logit_bias=None, presence_penalty=0.5, frequency_penalty=0.3),
        ]
        tokens, logprobs = gen.sample_batch(logits, sequences)
        # Token 0 should be penalized but with temperature=0, argmax should still be 2
        assert tokens.item() == 2

    def test_logit_bias_integration(self):
        gen = TokenGenerator()
        logits = torch.tensor([[0.0, 0.0, 0.0]])
        # Bias token 2 heavily
        tokens, _ = gen.sample(logits, temperature=0, logit_bias={2: 100.0})
        assert tokens.item() == 2
