"""Tests for speculative batch decoding."""

import torch
from unittest.mock import MagicMock
import pytest

from distllm.core.speculative_decoder import SpeculativeDecoder
from distllm.core.batch_scheduler import ScheduledBatch, Sequence


class TestSpeculativeBatch:
    """Test batch-level speculative decoding."""

    def test_scheduled_batch_has_speculative_enabled(self):
        batch = ScheduledBatch(
            sequences=[],
            input_ids=torch.tensor([]),
            seq_lengths=[],
            position_offsets=[],
            is_prefill=[],
            request_ids=[],
            speculative_enabled=True,
        )
        assert batch.speculative_enabled is True

    def test_scheduled_batch_speculative_default_false(self):
        batch = ScheduledBatch(
            sequences=[],
            input_ids=torch.tensor([]),
            seq_lengths=[],
            position_offsets=[],
            is_prefill=[],
            request_ids=[],
        )
        assert batch.speculative_enabled is False


class TestSpeculativeDecoderBatch:
    """Test batched draft generation and verification."""

    def test_generate_batch_draft_tokens(self):
        decoder = SpeculativeDecoder(num_assistant_tokens=3)

        # Mock draft model
        mock_model = MagicMock()
        mock_model.return_value.logits = torch.randn(1, 1, 100)
        mock_model.return_value.past_key_values = [
            (torch.randn(1, 2, 5, 64), torch.randn(1, 2, 5, 64))
        ]

        input_ids_list = [
            torch.tensor([[1, 2, 3]]),
            torch.tensor([[4, 5, 6]]),
        ]

        draft_tokens, kv_caches = decoder.generate_batch_draft_tokens(
            mock_model, input_ids_list
        )

        assert len(draft_tokens) == 2
        assert len(draft_tokens[0]) == 3
        assert len(draft_tokens[1]) == 3
        assert len(kv_caches) == 2

    def test_generate_batch_with_past_key_values(self):
        decoder = SpeculativeDecoder(num_assistant_tokens=2)

        mock_model = MagicMock()
        mock_model.return_value.logits = torch.randn(1, 1, 100)
        mock_model.return_value.past_key_values = [
            (torch.randn(1, 2, 5, 64), torch.randn(1, 2, 5, 64))
        ]

        input_ids_list = [torch.tensor([[1, 2]])]
        past_kv_list = [
            [(torch.randn(1, 2, 3, 64), torch.randn(1, 2, 3, 64))]
        ]

        draft_tokens, kv_caches = decoder.generate_batch_draft_tokens(
            mock_model, input_ids_list, past_key_values_list=past_kv_list
        )

        assert len(draft_tokens) == 1
        mock_model.assert_called()

    def test_verify_batch(self):
        decoder = SpeculativeDecoder()

        mock_tokenizer = MagicMock()

        # Single-step logits [batch, vocab]
        logits_list = [
            torch.randn(1, 100),
            torch.randn(1, 100),
        ]
        draft_tokens_list = [[10], [20]]

        results = decoder.verify_batch(draft_tokens_list, logits_list, mock_tokenizer)

        assert len(results) == 2
        for accepted_count, accepted_tokens, next_token in results:
            assert isinstance(accepted_count, int)
            assert isinstance(accepted_tokens, list)
            assert isinstance(next_token, int)

    def test_verify_batch_empty(self):
        decoder = SpeculativeDecoder()
        mock_tokenizer = MagicMock()

        results = decoder.verify_batch([], [], mock_tokenizer)
        assert results == []

    def test_batch_acceptance_rate_tracking(self):
        """Verify that batch verification updates acceptance rate."""
        decoder = SpeculativeDecoder(warmup_steps=1, min_acceptance_rate=0.0)

        mock_tokenizer = MagicMock()

        # Multi-step logits [batch, seq_len, vocab] to trigger full verification path
        logits = torch.zeros(1, 2, 100)  # 2 positions
        logits[0, 0, 42] = 10.0  # Position 0: token 42
        logits[0, 1, 43] = 10.0  # Position 1: token 43

        results = decoder.verify_batch([[42, 43]], [logits], mock_tokenizer)
        accepted_count, accepted_tokens, next_token = results[0]
        assert accepted_count == 2
        assert accepted_tokens == [42, 43]

        # Check metrics were updated
        metrics = decoder.get_metrics()
        assert metrics["total_draft_tokens"] == 2
        assert metrics["total_accepted"] == 2

    def test_batch_with_mismatch(self):
        decoder = SpeculativeDecoder(warmup_steps=1, min_acceptance_rate=0.0)
        mock_tokenizer = MagicMock()

        # Multi-step logits where first position doesn't match
        logits = torch.zeros(1, 2, 100)
        logits[0, 0, 99] = 10.0  # Position 0: token 99 (mismatch with draft 42)
        logits[0, 1, 43] = 10.0  # Position 1: token 43

        results = decoder.verify_batch([[42, 43]], [logits], mock_tokenizer)
        accepted_count, accepted_tokens, next_token = results[0]
        assert accepted_count == 0
        assert accepted_tokens == [99]  # Use target's token for mismatch
        assert next_token == 99
