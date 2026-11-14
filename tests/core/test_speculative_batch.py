"""Tests for speculative batch decoding."""

from unittest.mock import MagicMock

import pytest
import torch

from distllm.core.batch_scheduler import ScheduledBatch, Sequence
from distllm.core.speculative_decoder import SpeculativeDecoder


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


class TestVerifyBatchEosKeepsResultPerSequence:
    """Finding 513: an EOS hit ``break`` dropped that sequence's result.

    The returned list was then shorter than the batch, so
    ``torch.tensor([r[2] for r in results])`` produced too few next
    tokens and ``scheduler.step`` indexed ``next_tokens[i]`` for every
    sequence — IndexError, or silent token misalignment.
    """

    def test_eos_hit_still_appends_one_result_per_sequence(self):
        decoder = SpeculativeDecoder(device="cpu")
        tok = MagicMock()
        tok.eos_token_id = 3

        logits_list = [torch.zeros(1, 2, 16), torch.zeros(1, 2, 16)]
        for i in (0, 1):
            logits_list[i][0, 0, 3] = 10.0  # draft token 3 = EOS, accepted
            logits_list[i][0, 1, 4] = 10.0

        results = decoder.verify_batch([[3, 4], [3, 4]], logits_list, tok)

        assert len(results) == 2, "one result per sequence even on EOS"
        for accepted_count, accepted_tokens, next_token in results:
            # The EOS is reported as the next token (callers append r[2]),
            # not buried inside accepted_tokens where it could be
            # double-emitted.
            assert accepted_count == 0
            assert accepted_tokens == []
            assert next_token == 3

    def test_no_eos_tokenizer_keeps_verifying_past_eos_id(self):
        """A tokenizer without an int EOS must not trigger the cutoff."""
        decoder = SpeculativeDecoder(device="cpu")
        logits_list = [torch.zeros(1, 2, 16)]
        logits_list[0][0, 0, 3] = 10.0
        logits_list[0][0, 1, 4] = 10.0

        results = decoder.verify_batch([[3, 4]], logits_list, None)
        assert results[0][0] == 2
        assert results[0][1] == [3, 4]


class TestVerifyBatchRowGeometry:
    """Finding 492: draft token 0 must be verified against its OWN row.

    ``request_pipeline`` passes full per-sequence logits whose rows are
    the fed positions; row ``span_length - 1`` predicts the first draft
    token.  The pre-fix code verified draft token *i* against row *i* —
    a prompt-interior distribution — so an endorsed draft was rejected
    and the emitted next token came from the first prompt position.
    """

    def test_fallback_anchors_draft_zero_on_last_row(self):
        """No span_lengths: a full prompt followed by one draft.

        The draft is verified against the LAST row (the row that predicts
        it) when the span is longer than the draft list.
        """
        decoder = SpeculativeDecoder(device="cpu")
        L = torch.zeros(1, 4, 16)
        L[0, :3, 9] = 10.0   # interior prompt rows favour 9
        L[0, 3, 7] = 10.0    # the row predicting draft 7 endorses it

        results = decoder.verify_batch([[7]], [L])

        accepted_count, accepted_tokens, next_token = results[0]
        assert accepted_count == 1
        assert accepted_tokens == [7]
        # No successor row for the next token inside the span.
        assert next_token == -1

    def test_span_lengths_anchor_first_draft_on_span_end(self):
        """Explicit span_lengths: rows 0..L-1 are the fed span."""
        decoder = SpeculativeDecoder(device="cpu")
        L = torch.zeros(1, 5, 16)
        L[0, :4, 9] = 10.0   # rows 0..3 (prompt span of 4)
        L[0, 3, 7] = 10.0    # last span row predicts draft 7
        L[0, 4, 4] = 10.0    # successor row predicts the NEXT token

        results = decoder.verify_batch(
            [[7]], [L], span_lengths=[4],
        )

        accepted_count, accepted_tokens, next_token = results[0]
        assert accepted_count == 1
        assert accepted_tokens == [7]
        assert next_token == 4

    def test_multiple_drafts_with_span_length(self):
        """Draft chains advance one row per draft after the anchor."""
        decoder = SpeculativeDecoder(device="cpu")
        L = torch.zeros(1, 6, 16)
        L[0, 2, 10] = 10.0   # span_length=3 -> row 2 verifies draft 0
        L[0, 3, 11] = 10.0   # row 3 verifies draft 1
        L[0, 4, 12] = 10.0   # row 4 verifies draft 2 (mismatch: draft=99)

        results = decoder.verify_batch(
            [[10, 11, 99]], [L], span_lengths=[3],
        )

        accepted_count, accepted_tokens, next_token = results[0]
        assert accepted_count == 2
        # Mismatch: accepted prefix + the target's correction at that row.
        assert accepted_tokens == [10, 11, 12]
        assert next_token == 12


class TestVerifyBatchSampledMode:
    """Finding 442: verify_batch ignored temperature and draft_logits.

    With temperature > 0 the batch path must run rejection sampling from
    the supplied draft logits (q <= 0 rejects) and sample — not argmax —
    the correction token.
    """

    def test_zero_draft_probability_rejects(self):
        """q=0 for the drafted token -> reject (Leviathan et al.)."""
        decoder = SpeculativeDecoder(device="cpu")
        L = torch.zeros(1, 1, 16)
        L[0, 0, 5] = 10.0
        # Draft distribution gives token 5 exactly zero mass: the verifier
        # must reject it even though the target fully endorses it.
        draft_logits = torch.zeros(1, 16)
        draft_logits[0, 5] = -1e9  # exp(-1e9) underflows -> q(5) = 0.0
        draft_logits[0, 6] = 10.0

        results = decoder.verify_batch(
            [[5]], [L], draft_logits_list=[draft_logits], temperature=1.0,
        )
        assert results[0][0] == 0

    def test_greedy_unchanged_when_temperature_zero(self):
        decoder = SpeculativeDecoder(device="cpu")
        L = torch.zeros(1, 1, 16)
        L[0, 0, 5] = 10.0

        results = decoder.verify_batch([[5]], [L], temperature=0.0)
        assert results[0][:2] == (1, [5])

    def test_sampled_mode_draws_not_argmax_for_next_token(self):
        """The correction token is sampled from the target row, not argmax.

        Target successor row: tokens 1 and 2 split the mass 50/50 (all
        other logits -1e9).  Over many trials the sampled next token must
        take BOTH values; a greedy argmax would always return one of them.
        """
        decoder = SpeculativeDecoder(device="cpu")
        draft_logits = torch.zeros(1, 16)
        draft_logits[0, 5] = 10.0
        torch.manual_seed(0)

        seen = set()
        for _ in range(200):
            L = torch.full((1, 2, 16), -1e9)
            L[0, 0, :] = -1e9
            L[0, 0, 5] = 10.0  # row 0 = the row that predicts draft 5
            L[0, 1, :] = -1e9
            L[0, 1, 1] = 0.0   # successor row: 1 and 2 equally likely
            L[0, 1, 2] = 0.0
            results = decoder.verify_batch(
                [[5]], [L], draft_logits_list=[draft_logits],
                temperature=1.0, span_lengths=[1],
            )
            assert results[0][0] == 1
            seen.add(results[0][2])
        assert seen == {1, 2}, f"next token must be sampled, saw only {seen}"

    def test_eagle_style_single_row_span(self):
        """EAGLE passes ``logits[:, gen_pos-1:, :]``: row j verifies draft j.

        span_lengths=[1] means the row verifying draft 0 is slice row 0
        (shift 0), which must reproduce the historical row-j-verifies-
        draft-j behaviour — and next_row after full acceptance lands on
        the successor row already present in the slice.
        """
        decoder = SpeculativeDecoder(device="cpu")
        L = torch.zeros(1, 4, 16)
        L[0, 0, 10] = 10.0  # row 0 verifies draft 0
        L[0, 1, 11] = 10.0  # row 1 verifies draft 1
        L[0, 2, 12] = 10.0  # row 2 = successor after full acceptance
        L[0, 3, 13] = 10.0

        results = decoder.verify_batch(
            [[10, 11]], [L], span_lengths=[1],
        )
        accepted_count, accepted_tokens, next_token = results[0]
        assert accepted_count == 2
        assert accepted_tokens == [10, 11]
        assert next_token == 12
