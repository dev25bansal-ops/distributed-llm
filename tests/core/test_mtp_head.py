"""Tests for the Multi-Token Prediction head."""

from __future__ import annotations

import torch
import pytest

from distllm.core.mtp_head import MTPConfig, MTPEmbedding, MTPHead, MTPDecoder


# ── Test config (small enough for fast tests) ────────────────────────────────

def _small_config(num_candidates: int = 3) -> MTPConfig:
    return MTPConfig(
        hidden_size=64,
        vocab_size=256,
        num_candidates=num_candidates,
        num_transformer_layers=1,
        num_attention_heads=4,
        device="cpu",
    )


# ── MTPConfig tests ─────────────────────────────────────────────────────────

class TestMTPConfig:
    def test_defaults(self):
        c = MTPConfig()
        assert c.hidden_size == 4096
        assert c.vocab_size == 32000
        assert c.num_candidates == 5

    def test_custom(self):
        c = _small_config()
        assert c.hidden_size == 64
        assert c.vocab_size == 256
        assert c.num_candidates == 3
        assert c.num_attention_heads == 4

    def test_feedforward_default(self):
        c = MTPConfig()
        assert c.feedforward_size == 4 * c.hidden_size


# ── MTPEmbedding tests ──────────────────────────────────────────────────────

class TestMTPEmbedding:
    def test_forward_shape(self):
        config = _small_config()
        emb = MTPEmbedding(config)
        hidden = torch.randn(1, 1, config.hidden_size)
        out = emb(hidden)
        assert out.shape == (1, config.num_candidates, config.hidden_size)

    def test_output_differs_per_position(self):
        config = _small_config()
        emb = MTPEmbedding(config)
        hidden = torch.randn(1, 1, config.hidden_size)
        out = emb(hidden)
        # Different positions should have different embeddings (due to pos emb)
        assert not torch.allclose(out[0, 0], out[0, 1])


# ── MTPHead tests ──────────────────────────────────────────────────────────

class TestMTPHead:
    def test_forward_shapes(self):
        config = _small_config(num_candidates=3)
        head = MTPHead(config)
        hidden = torch.randn(1, 1, config.hidden_size)
        logits_list = head(hidden)
        assert len(logits_list) == config.num_candidates
        for i, logits in enumerate(logits_list):
            assert logits.shape == (1, config.vocab_size), f"Head {i}: {logits.shape}"

    def test_generate_shapes(self):
        config = _small_config(num_candidates=3)
        head = MTPHead(config)
        hidden = torch.randn(1, 1, config.hidden_size)
        tokens, logprobs = head.generate(hidden)
        assert tokens.shape == (1, config.num_candidates)
        assert len(logprobs) == config.num_candidates
        assert all(0 < lp <= 1.0 for lp in logprobs)

    def test_generate_tokens_in_vocab(self):
        config = _small_config(num_candidates=5)
        head = MTPHead(config)
        hidden = torch.randn(1, 1, config.hidden_size)
        tokens, _ = head.generate(hidden)
        assert tokens.min().item() >= 0
        assert tokens.max().item() < config.vocab_size

    def test_different_hidden_produces_different_tokens(self):
        config = _small_config(num_candidates=3)
        head = MTPHead(config)
        h1 = torch.randn(1, 1, config.hidden_size)
        h2 = torch.randn(1, 1, config.hidden_size)
        t1, _ = head.generate(h1)
        t2, _ = head.generate(h2)
        assert not torch.equal(t1, t2)

    def test_causal_mask_is_upper_triangular(self):
        config = _small_config(num_candidates=4)
        head = MTPHead(config)
        mask = head._causal_mask
        assert mask.shape == (4, 4)
        # Check: mask[i, j] = -inf when j > i (upper triangle excluding diagonal)
        assert mask[0, 1] == float("-inf")
        assert mask[0, 0] == 0.0
        assert mask[2, 3] == float("-inf")
        assert mask[3, 2] == 0.0  # Can attend to earlier positions

    def test_large_candidates(self):
        config = _small_config(num_candidates=8)
        head = MTPHead(config)
        hidden = torch.randn(1, 1, config.hidden_size)
        tokens, logprobs = head.generate(hidden)
        assert tokens.shape == (1, 8)
        assert len(logprobs) == 8

    def test_parameter_count(self):
        config = _small_config()
        head = MTPHead(config)
        params = sum(p.numel() for p in head.parameters())
        assert params > 0
        # Should be reasonable: embedding + 1 layer + 3 output heads
        # ~ (64*64 + 3*64) + (4*64*8 + 64*256*2 + 64*4) + 3*64*256
        assert params < 200_000  # For this tiny config

    def test_deterministic_with_temperature_0(self):
        config = _small_config()
        config.temperature = 0
        head = MTPHead(config)
        hidden = torch.randn(1, 1, config.hidden_size)
        t1, _ = head.generate(hidden)
        t2, _ = head.generate(hidden)
        assert torch.equal(t1, t2)


# ── MTPDecoder tests ────────────────────────────────────────────────────────

class TestMTPDecoder:
    def _make_forward(self):
        """Returns (logits, hidden_states) from a trivial model."""
        def forward(input_ids, **kw):
            logits = torch.zeros(1, input_ids.shape[1], 256)
            logits[0, :, 42] = 1.0
            return logits
        return forward

    def _make_hidden_states_fn(self, hidden_size: int = 64):
        def fn(input_ids, **kw):
            logits = torch.zeros(1, input_ids.shape[1], 256)
            logits[0, :, 42] = 1.0
            hidden = torch.randn(1, input_ids.shape[1], hidden_size)
            return logits, [hidden] * 3  # 3 layers
        return fn

    def test_init(self):
        config = _small_config()
        head = MTPHead(config)
        decoder = MTPDecoder(
            target_forward=self._make_forward(),
            hidden_states_fn=self._make_hidden_states_fn(),
            mtp_head=head,
        )
        assert decoder._num_candidates == config.num_candidates
        assert decoder._stats["mtp_calls"] == 0

    def test_generate_batch_size_check(self):
        config = _small_config()
        head = MTPHead(config)
        decoder = MTPDecoder(
            target_forward=self._make_forward(),
            hidden_states_fn=self._make_hidden_states_fn(),
            mtp_head=head,
        )
        with pytest.raises(ValueError):
            decoder.generate(torch.randint(0, 100, (2, 10)))

    def test_generate_basic(self):
        config = _small_config(num_candidates=2)
        head = MTPHead(config)
        decoder = MTPDecoder(
            target_forward=self._make_forward(),
            hidden_states_fn=self._make_hidden_states_fn(),
            mtp_head=head,
        )
        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        output = decoder.generate(input_ids, max_new_tokens=10)
        assert output.shape[1] >= 3 + 3
        assert output.shape[1] <= 3 + 15
        assert decoder._stats["target_calls"] > 0
        assert decoder._stats["mtp_calls"] > 0

    def test_stats_after_generation(self):
        config = _small_config(num_candidates=2)
        head = MTPHead(config)
        decoder = MTPDecoder(
            target_forward=self._make_forward(),
            hidden_states_fn=self._make_hidden_states_fn(),
            mtp_head=head,
        )
        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        decoder.generate(input_ids, max_new_tokens=8)
        stats = decoder.stats
        assert "mtp_calls" in stats
        assert "target_calls" in stats
        assert "acceptance_rate" in stats

    def test_generate_multiple_calls(self):
        config = _small_config(num_candidates=2)
        head = MTPHead(config)
        decoder = MTPDecoder(
            target_forward=self._make_forward(),
            hidden_states_fn=self._make_hidden_states_fn(),
            mtp_head=head,
        )
        input_ids = torch.tensor([[1]], dtype=torch.long)
        out1 = decoder.generate(input_ids, max_new_tokens=5)
        out2 = decoder.generate(input_ids, max_new_tokens=5)
        assert out1.shape[1] >= 1 + 2
        assert out2.shape[1] >= 1 + 2

    def test_generate_rejection(self):
        """Test with mismatched hidden states (forces rejection)."""
        config = _small_config(num_candidates=3)

        def alt_forward(input_ids, **kw):
            logits = torch.zeros(1, input_ids.shape[1], 256)
            logits[0, :, 99] = 1.0  # Different token than default
            return logits

        head = MTPHead(config)
        decoder = MTPDecoder(
            target_forward=alt_forward,
            hidden_states_fn=self._make_hidden_states_fn(),
            mtp_head=head,
        )
        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        output = decoder.generate(input_ids, max_new_tokens=8)
        assert output.shape[1] >= 3 + 1  # At least generated something
