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

        # Mock draft model with reusable parameters() iterator
        param = torch.nn.Parameter(torch.zeros(1))
        mock_model = MagicMock()
        mock_model.parameters.side_effect = lambda: iter([param])
        mock_model.return_value.logits = torch.randn(1, 1, 100)
        mock_model.return_value.past_key_values = [
            (torch.randn(1, 2, 5, 64), torch.randn(1, 2, 5, 64))
        ]

        input_ids_list = [
            torch.tensor([[1, 2, 3]]),
            torch.tensor([[4, 5, 6]]),
        ]

        draft_tokens, kv_caches, _ = decoder.generate_batch_draft_tokens(
            mock_model, input_ids_list
        )

        assert len(draft_tokens) == 2
        assert len(draft_tokens[0]) == 3
        assert len(draft_tokens[1]) == 3
        assert len(kv_caches) == 2

    def test_generate_batch_with_past_key_values(self):
        decoder = SpeculativeDecoder(num_assistant_tokens=2)

        mock_model = MagicMock()
        mock_model.parameters.side_effect = lambda: iter([torch.nn.Parameter(torch.zeros(1))])
        mock_model.return_value.logits = torch.randn(1, 1, 100)
        mock_model.return_value.past_key_values = [
            (torch.randn(1, 2, 5, 64), torch.randn(1, 2, 5, 64))
        ]

        input_ids_list = [torch.tensor([[1, 2]])]
        past_kv_list = [
            [(torch.randn(1, 2, 3, 64), torch.randn(1, 2, 3, 64))]
        ]

        draft_tokens, kv_caches, _ = decoder.generate_batch_draft_tokens(
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


class TestBatchSpeculationMultipleRequests:
    """Multiple requests with speculation in batch."""

    def test_batch_different_draft_lengths(self):
        decoder = SpeculativeDecoder(num_assistant_tokens=4)
        mock_model = MagicMock()
        mock_model.parameters.side_effect = lambda: iter([torch.nn.Parameter(torch.zeros(1))])
        mock_model.return_value.logits = torch.randn(1, 1, 100)
        mock_model.return_value.past_key_values = [
            (torch.randn(1, 2, 5, 64), torch.randn(1, 2, 5, 64))
        ]

        input_ids_list = [
            torch.tensor([[1, 2, 3]]),
            torch.tensor([[4, 5]]),
            torch.tensor([[6]]),
        ]
        draft_tokens, kv_caches, _ = decoder.generate_batch_draft_tokens(
            mock_model, input_ids_list
        )

        assert len(draft_tokens) == 3
        assert all(len(t) == 4 for t in draft_tokens)
        assert len(kv_caches) == 3

    def test_batch_mixed_accept_reject_greedy(self):
        decoder = SpeculativeDecoder(warmup_steps=1, min_acceptance_rate=0.0)
        mock_tokenizer = MagicMock()

        logits_list = [
            torch.zeros(1, 3, 100),
            torch.zeros(1, 3, 100),
        ]
        logits_list[0][0, 0, 10] = 10.0
        logits_list[0][0, 1, 20] = 10.0
        logits_list[0][0, 2, 30] = 10.0

        logits_list[1][0, 0, 99] = 10.0
        logits_list[1][0, 1, 20] = 10.0
        logits_list[1][0, 2, 30] = 10.0

        draft_tokens_list = [[10, 20, 30], [99, 77, 88]]
        results = decoder.verify_batch(draft_tokens_list, logits_list, mock_tokenizer)

        assert len(results) == 2
        seq0_accepted, seq0_tokens, seq0_next = results[0]
        assert seq0_accepted == 3
        assert seq0_tokens == [10, 20, 30]

        seq1_accepted, seq1_tokens, seq1_next = results[1]
        assert seq1_accepted == 1
        assert seq1_tokens == [99, 20]

    def test_batch_empty_drafts_for_one_sequence(self):
        decoder = SpeculativeDecoder(warmup_steps=1, min_acceptance_rate=0.0)
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token_id = 0

        logits_list = [
            torch.zeros(1, 2, 100),
            torch.zeros(1, 2, 100),
        ]
        logits_list[0][0, 0, 42] = 10.0
        logits_list[0][0, 1, 43] = 10.0

        results = decoder.verify_batch(
            [[42, 43], []],
            logits_list,
            mock_tokenizer,
        )

        assert len(results) == 2
        seq0_accepted, _, _ = results[0]
        assert seq0_accepted == 2

        seq1_accepted, seq1_tokens, seq1_next = results[1]
        assert seq1_accepted == 0
        assert seq1_tokens == []

    def test_batch_verify_rejection_sampling(self):
        decoder = SpeculativeDecoder(warmup_steps=1, min_acceptance_rate=0.0)
        mock_tokenizer = MagicMock()

        logits_list = [
            torch.zeros(1, 2, 4),
            torch.zeros(1, 2, 4),
        ]
        logits_list[0][0, 0, 1] = 10.0
        logits_list[0][0, 1, 2] = 10.0
        logits_list[1][0, 0, 1] = 10.0
        logits_list[1][0, 1, 3] = 10.0

        draft_tokens_list = [[1, 2], [1, 2]]
        draft_logits_list = [
            torch.zeros(2, 4),
            torch.zeros(2, 4),
        ]
        draft_logits_list[0][0, 1] = 10.0
        draft_logits_list[0][1, 2] = 10.0
        draft_logits_list[1][0, 1] = 10.0
        draft_logits_list[1][1, 2] = 10.0

        results = decoder.verify_batch(
            draft_tokens_list, logits_list, mock_tokenizer,
            draft_logits_list=draft_logits_list, temperature=1.0,
        )

        assert len(results) == 2
        assert all(isinstance(r[0], int) for r in results)
        assert all(isinstance(r[1], list) for r in results)
        assert all(isinstance(r[2], int) for r in results)

    def test_batch_verify_mismatched_lengths(self):
        decoder = SpeculativeDecoder(warmup_steps=1, min_acceptance_rate=0.0)
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token_id = 0

        logits_list = [torch.zeros(1, 1, 100)]
        draft_tokens_list = [[10, 20, 30]]

        results = decoder.verify_batch(draft_tokens_list, logits_list, mock_tokenizer)
        accepted_count, accepted_tokens, next_token = results[0]
        assert accepted_count <= 1

    def test_batch_metrics_after_mixed_results(self):
        decoder = SpeculativeDecoder(warmup_steps=1, min_acceptance_rate=0.0)
        mock_tokenizer = MagicMock()

        logits_list = [
            torch.zeros(1, 2, 100),
            torch.zeros(1, 2, 100),
        ]
        logits_list[0][0, 0, 42] = 10.0
        logits_list[0][0, 1, 43] = 10.0
        logits_list[1][0, 0, 99] = 10.0
        logits_list[1][0, 1, 43] = 10.0

        decoder.verify_batch([[42, 43], [99, 77]], logits_list, mock_tokenizer)

        metrics = decoder.get_metrics()
        assert metrics["total_draft_tokens"] == 4
        assert metrics["total_accepted"] == 3
