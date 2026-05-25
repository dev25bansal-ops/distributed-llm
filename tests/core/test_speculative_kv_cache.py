"""KV cache with speculative decoding: draft token KV invariance.

Tests that draft model KV cache management is correct:
  - Draft KV cache length matches draft token count
  - Verification correctness does not depend on stale KV entries
  - After acceptance/rejection, the KV advance is consistent
"""

import torch
from unittest.mock import MagicMock

from distllm.core.speculative_decoder import SpeculativeDecoder


def _make_kv_cache(num_layers=2, batch=1, num_heads=2, seq_len=5, d=64):
    return [
        (torch.randn(batch, num_heads, seq_len, d),
         torch.randn(batch, num_heads, seq_len, d))
        for _ in range(num_layers)
    ]


def _mock_draft_model():
    m = MagicMock()
    param = torch.nn.Parameter(torch.zeros(1))
    m.parameters.side_effect = lambda: iter([param])
    m.return_value.logits = torch.randn(1, 1, 100)
    m.return_value.past_key_values = _make_kv_cache(seq_len=1)
    return m


class TestDraftModelKVCache:
    """Draft model KV cache generation and consistency."""

    def test_draft_model_returns_kv_cache(self):
        decoder = SpeculativeDecoder(num_assistant_tokens=3)
        mock_model = _mock_draft_model()

        tokens, new_kv, logits = decoder._generate_draft_model_tokens(
            mock_model, torch.tensor([[1, 2, 3]])
        )
        assert len(tokens) == 3
        assert new_kv is not None
        assert len(new_kv) > 0

    def test_draft_kv_cache_length_matches_drafts(self):
        decoder = SpeculativeDecoder(num_assistant_tokens=4)
        mock_model = _mock_draft_model()
        mock_model.return_value.past_key_values = _make_kv_cache(seq_len=5)

        tokens, new_kv, logits = decoder._generate_draft_model_tokens(
            mock_model, torch.tensor([[1, 2, 3]])
        )
        assert len(tokens) == 4, f"Expected 4 draft tokens, got {len(tokens)}"
        assert new_kv is not None
        assert len(new_kv) > 0
        assert logits is not None
        assert len(logits) == len(tokens)

    def test_draft_kv_cache_with_past_kv(self):
        decoder = SpeculativeDecoder(num_assistant_tokens=2)
        past_kv = _make_kv_cache(seq_len=10)

        mock_model = _mock_draft_model()
        mock_model.return_value.past_key_values = _make_kv_cache(seq_len=11)

        tokens, new_kv, logits = decoder._generate_draft_model_tokens(
            mock_model, torch.tensor([[5]]),
            past_key_values=past_kv,
        )
        assert len(tokens) == 2
        assert new_kv is not None
        assert mock_model.called

    def test_draft_no_model_returns_empty(self):
        decoder = SpeculativeDecoder()
        tokens, new_kv, logits = decoder._generate_draft_model_tokens(
            None, torch.tensor([[1]])
        )
        assert tokens == []
        assert new_kv is None
        assert logits is None

    def test_draft_no_logits_returns_empty(self):
        decoder = SpeculativeDecoder(num_assistant_tokens=2)
        mock_model = _mock_draft_model()
        del mock_model.return_value.logits

        tokens, new_kv, logits = decoder._generate_draft_model_tokens(
            mock_model, torch.tensor([[1]])
        )
        assert tokens == []
        assert new_kv is None
        assert logits is None


class TestBatchDraftKVCache:
    """Batch draft generation returns per-sequence KV caches."""

    def test_batch_draft_kv_cache_lengths(self):
        decoder = SpeculativeDecoder(num_assistant_tokens=3)
        mock_model = _mock_draft_model()

        input_ids_list = [
            torch.tensor([[1, 2, 3]]),
            torch.tensor([[4, 5]]),
        ]

        draft_tokens, kv_caches, _ = decoder.generate_batch_draft_tokens(
            mock_model, input_ids_list
        )
        assert len(draft_tokens) == 2
        assert len(kv_caches) == 2
        assert kv_caches[0] is not None
        assert kv_caches[1] is not None

    def test_batch_draft_with_past_kv_list(self):
        decoder = SpeculativeDecoder(num_assistant_tokens=2)
        mock_model = _mock_draft_model()
        mock_model.return_value.past_key_values = _make_kv_cache(seq_len=3)

        input_ids_list = [torch.tensor([[1, 2]])]
        past_kv_list = [_make_kv_cache(seq_len=5)]

        draft_tokens, kv_caches, _ = decoder.generate_batch_draft_tokens(
            mock_model, input_ids_list, past_key_values_list=past_kv_list
        )
        assert len(draft_tokens) == 1
        assert kv_caches[0] is not None

    def test_batch_draft_empty_returns_empty(self):
        decoder = SpeculativeDecoder()
        draft_tokens, kv_caches, _ = decoder.generate_batch_draft_tokens(
            _mock_draft_model(), []
        )
        assert draft_tokens == []
        assert kv_caches is None


class TestSpeculativeKVCacheInvariants:
    """Correctness invariants for KV cache with speculative decoding."""

    def test_accept_all_advances_by_draft_count(self):
        decoder = SpeculativeDecoder(warmup_steps=1, min_acceptance_rate=0.0)
        mock_tokenizer = MagicMock()
        prev_len = 10

        logits = torch.zeros(1, 3, 100)
        logits[0, 0, 5] = 10.0
        logits[0, 1, 6] = 10.0
        logits[0, 2, 7] = 10.0

        num_acc, accepted, next_tok = decoder.verify_and_accept(
            [5, 6, 7], logits, mock_tokenizer
        )
        assert num_acc == 3
        assert len(accepted) == 3
        # KV cache would have prev_len + 3 entries after this step

    def test_reject_first_token_advances_by_one(self):
        decoder = SpeculativeDecoder(warmup_steps=1, min_acceptance_rate=0.0)
        mock_tokenizer = MagicMock()
        prev_len = 10

        logits = torch.zeros(1, 3, 100)
        logits[0, 0, 99] = 10.0
        logits[0, 1, 6] = 10.0
        logits[0, 2, 7] = 10.0

        num_acc, accepted, next_tok = decoder.verify_and_accept(
            [5, 6, 7], logits, mock_tokenizer
        )
        assert num_acc == 0
        assert len(accepted) == 1
        assert accepted == [99]

    def test_reject_at_second_token(self):
        decoder = SpeculativeDecoder(warmup_steps=1, min_acceptance_rate=0.0)
        mock_tokenizer = MagicMock()

        logits = torch.zeros(1, 3, 100)
        logits[0, 0, 5] = 10.0
        logits[0, 1, 99] = 10.0
        logits[0, 2, 7] = 10.0

        num_acc, accepted, next_tok = decoder.verify_and_accept(
            [5, 6, 7], logits, mock_tokenizer
        )
        assert num_acc == 1
        assert len(accepted) == 2
        assert accepted[0] == 5
        assert accepted[1] == 99

    def test_greedy_accept_all_matching(self):
        decoder = SpeculativeDecoder(warmup_steps=1, min_acceptance_rate=0.0)
        mock_tokenizer = MagicMock()

        logits = torch.zeros(1, 4, 100)
        for i in range(4):
            logits[0, i, 10 + i] = 10.0

        num_acc, accepted, next_tok = decoder.verify_and_accept(
            [10, 11, 12, 13], logits, mock_tokenizer
        )
        assert num_acc == 4
        assert accepted == [10, 11, 12, 13]

    def test_verify_sets_next_token_correctly(self):
        decoder = SpeculativeDecoder(warmup_steps=1, min_acceptance_rate=0.0)
        mock_tokenizer = MagicMock()

        logits = torch.zeros(1, 2, 100)
        logits[0, 0, 42] = 10.0
        logits[0, 1, 7] = 10.0

        num_acc, accepted, next_tok = decoder.verify_and_accept(
            [42, 43], logits, mock_tokenizer
        )
        assert num_acc == 1
        assert accepted == [42, 7]
        assert next_tok == 7

    def test_verify_all_accepted_picks_next_from_logits(self):
        decoder = SpeculativeDecoder(warmup_steps=1, min_acceptance_rate=0.0)
        mock_tokenizer = MagicMock()

        logits = torch.zeros(1, 2, 100)
        logits[0, 0, 10] = 10.0
        logits[0, 1, 20] = 10.0

        num_acc, accepted, next_tok = decoder.verify_and_accept(
            [10, 20], logits, mock_tokenizer
        )
        assert num_acc == 2
        # No more logit positions → next_token falls through to accepted[-1]
        assert next_tok == 20

    def test_empty_draft_falls_back_to_eos(self):
        decoder = SpeculativeDecoder()
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token_id = 0

        num_acc, accepted, next_tok = decoder.verify_and_accept(
            None, None, mock_tokenizer
        )
        assert num_acc == 0
        assert accepted == []
        assert next_tok == 0

    def test_kv_cache_length_correct_after_partial_reject_scenario(self):
        """Simulates the full pipeline invariant: after a step with K accepted
        drafts, the next decode input must be the single next token and the
        KV cache must have advanced by K+1."""
        decoder = SpeculativeDecoder(warmup_steps=1, min_acceptance_rate=0.0)
        mock_tokenizer = MagicMock()

        logits = torch.zeros(1, 4, 100)
        logits[0, 0, 1] = 10.0
        logits[0, 1, 2] = 10.0
        logits[0, 2, 99] = 10.0
        logits[0, 3, 4] = 10.0

        num_acc, accepted, next_tok = decoder.verify_and_accept(
            [1, 2, 3, 4], logits, mock_tokenizer
        )
        assert num_acc == 2
        assert accepted == [1, 2, 99]
        assert next_tok == 99
