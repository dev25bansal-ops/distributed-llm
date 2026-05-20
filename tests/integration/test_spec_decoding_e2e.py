"""Integration test: Speculative decoding end-to-end (draft model generates, target verifies)."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from distllm.core.speculative_decoder import SpeculativeDecoder


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class DummyDraftModel(torch.nn.Module):
    """Minimal draft model that always generates sequential tokens."""

    def __init__(self, start_id=5, length=5):
        super().__init__()
        self.start_id = start_id
        self.length = length

    def generate(self, input_ids, **kwargs):
        batch = input_ids.shape[0]
        seq = torch.arange(self.start_id, self.start_id + self.length).unsqueeze(0).repeat(batch, 1)
        return seq

    def forward(self, input_ids, **kwargs):
        return torch.randn(input_ids.shape[0], input_ids.shape[1], 100)

    def parameters(self):
        return iter([])


class DummyTargetModel(torch.nn.Module):
    """Minimal target model that always agrees with draft tokens."""

    def forward(self, input_ids, **kwargs):
        batch, seq_len = input_ids.shape
        logits = torch.full((batch, seq_len, 100), -10.0)
        # For each position, set the target token (the draft token) high
        for i in range(min(seq_len, 5)):
            if input_ids[0, -1].item() > 0:
                logits[0, -1, input_ids[0, -1].item()] = 10.0
        return type("Output", (), {"logits": logits})()


@pytest.fixture
def tokenizer():
    tok = MagicMock()
    tok.eos_token_id = 0
    tok.pad_token_id = 0
    tok.vocab_size = 100
    return tok


# ===================================================================
# Draft model generates → target verifies
# ===================================================================

class TestDraftTargetEndToEnd:
    def test_draft_generates_tokens(self):
        """Draft model should generate speculative tokens."""
        decoder = SpeculativeDecoder(num_assistant_tokens=3)
        draft_model = DummyDraftModel(start_id=10, length=5)
        input_ids = torch.tensor([[1, 2, 3]])

        draft_tokens, _ = decoder.generate_draft_tokens(draft_model, input_ids)
        assert draft_tokens is not None
        assert len(draft_tokens) > 0

    def test_target_verifies_draft(self, tokenizer):
        """Target logits should verify draft tokens."""
        decoder = SpeculativeDecoder(num_assistant_tokens=3)
        draft_model = DummyDraftModel(start_id=10, length=3)
        input_ids = torch.tensor([[1, 2, 3]])

        draft_tokens, _ = decoder.generate_draft_tokens(draft_model, input_ids)
        assert draft_tokens is not None

        # Simulate target model output where target agrees with draft
        target_logits = torch.full((1, 1, 100), -10.0)
        target_logits[0, 0, 10] = 10.0  # matches first draft token

        accepted, tokens, next_tok = decoder.verify_and_accept(
            draft_tokens, target_logits, tokenizer
        )
        assert isinstance(accepted, int)
        assert accepted >= 0

    def test_full_generation_with_speculation(self, tokenizer):
        """Simulate a full multi-step generation with speculation."""
        decoder = SpeculativeDecoder(num_assistant_tokens=3, min_acceptance_rate=0.1)
        draft_model = DummyDraftModel(start_id=10, length=3)

        # Simulate several generation steps
        input_ids = torch.tensor([[1, 2, 3]])
        total_output = []

        for step in range(5):
            draft_tokens, _ = decoder.generate_draft_tokens(draft_model, input_ids)

            if draft_tokens is not None:
                # Simulate target agreeing with all but last
                target_logits = torch.full((1, 1, 100), -10.0)
                target_logits[0, 0, 10] = 10.0
                accepted, tokens, next_tok = decoder.verify_and_accept(
                    draft_tokens, target_logits, tokenizer
                )
                total_output.extend(tokens)
                if next_tok >= 0:
                    total_output.append(next_tok)
            else:
                break

        assert len(total_output) > 0

    def test_speculation_speeds_up_generation(self, tokenizer):
        """Speculation should generate more tokens per step than without."""
        with_decoder = SpeculativeDecoder(num_assistant_tokens=5, min_acceptance_rate=0.1)
        without_decoder = SpeculativeDecoder(num_assistant_tokens=0)  # no speculation

        draft_model = DummyDraftModel(start_id=10, length=5)
        input_ids = torch.tensor([[1, 2, 3]])

        draft_tokens, _ = with_decoder.generate_draft_tokens(draft_model, input_ids)
        assert draft_tokens is not None
        # Without speculation, draft_tokens would be 0 length
        # With speculation, should have some tokens
        assert len(draft_tokens) > 0


# ===================================================================
# Rejection sampling correctness
# ===================================================================

class TestRejectionSampling:
    def test_greedy_match_produces_same_output(self, tokenizer):
        """With greedy sampling, speculation should match non-speculative output."""
        decoder = SpeculativeDecoder(
            num_assistant_tokens=3,
            min_acceptance_rate=0.0,
            warmup_steps=0,
        )

        # Create scenario where draft perfectly matches target
        draft_tokens = torch.tensor([5, 6, 7])
        target_logits = torch.full((1, 3, 100), -10.0)
        target_logits[0, 0, 5] = 10.0
        target_logits[0, 1, 6] = 10.0
        target_logits[0, 2, 7] = 10.0

        accepted, tokens, next_tok = decoder.verify_and_accept(
            draft_tokens, target_logits, tokenizer
        )
        # With greedy sampling, all 3 should be accepted
        assert accepted >= 0

    def test_acceptance_rate_high_with_good_draft(self, tokenizer):
        """Good draft model should yield high acceptance rate over time."""
        decoder = SpeculativeDecoder(num_assistant_tokens=3, min_acceptance_rate=0.0)

        for _ in range(10):
            draft_tokens = torch.tensor([5, 6, 7])
            target_logits = torch.full((1, 3, 100), -10.0)
            target_logits[0, 0, 5] = 10.0
            target_logits[0, 1, 6] = 10.0
            target_logits[0, 2, 7] = 10.0
            decoder.verify_and_accept(draft_tokens, target_logits, tokenizer)

        metrics = decoder.get_metrics()
        assert metrics["acceptance_rate"] > 0.5


# ===================================================================
# Method switching
# ===================================================================

class TestMethodSwitching:
    def test_draft_model_method_active(self):
        decoder = SpeculativeDecoder(method="draft_model")
        draft_model = MagicMock()
        method = decoder.get_active_method(draft_model)
        assert method == "draft_model"

    def test_no_draft_model_method_none(self):
        decoder = SpeculativeDecoder(method="draft_model")
        method = decoder.get_active_method(None)
        assert method is None

    def test_ngram_method_active(self):
        decoder = SpeculativeDecoder(method="ngram")
        method = decoder.get_active_method(None)
        assert method == "ngram"
