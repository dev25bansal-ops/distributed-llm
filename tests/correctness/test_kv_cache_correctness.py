"""Tests: KV cache correctness — cached generation matches non-cached.

Verifies that using the KV cache produces identical token outputs as
recomputing the full sequence from scratch. Tests:

- Exact token match between cached and non-cached generation
- Incremental decoding with cache vs full recomputation
- Prefix caching: partial cache + new tokens matches full computation
- Cache continuity across multiple decode steps
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
from loguru import logger


class SimpleSelfAttention(nn.Module):
    """Simplified self-attention with manual KV cache support."""

    def __init__(self, hidden_dim: int = 64, num_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        past_k: Optional[torch.Tensor] = None,
        past_v: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seq_len, _ = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if past_k is not None and past_v is not None:
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        # Causal mask: prevent attending to future positions
        total_k_len = k.shape[2]
        causal_mask = torch.triu(
            torch.full((seq_len, total_k_len), float('-inf'), device=x.device),
            diagonal=total_k_len - seq_len + 1,
        )
        scores = scores + causal_mask
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(bsz, seq_len, self.hidden_dim)
        out = self.o_proj(out)

        return out, k, v


class CachedTransformerLayer(nn.Module):
    """Transformer layer with KV cache support."""

    def __init__(self, hidden_dim: int = 64, num_heads: int = 4):
        super().__init__()
        self.attention = SimpleSelfAttention(hidden_dim, num_heads)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.mlp_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        past_k: Optional[torch.Tensor] = None,
        past_v: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = x
        x = self.input_norm(x)
        attn_out, new_k, new_v = self.attention(x, past_k, past_v)
        x = residual + attn_out

        residual = x
        x = self.mlp_norm(x)
        x = residual + self.mlp(x)

        return x, new_k, new_v


class CachedTransformer(nn.Module):
    """Full transformer with KV cache for testing."""

    def __init__(self, hidden_dim: int = 64, num_layers: int = 4, num_heads: int = 4, vocab_size: int = 256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([
            CachedTransformerLayer(hidden_dim, num_heads) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward_with_cache(
        self,
        input_ids: torch.Tensor,
        past_kvs: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward with optional KV cache.

        Args:
            input_ids: (batch, seq_len) token IDs.
            past_kvs: List of (k, v) tuples for each layer from previous steps.

        Returns:
            (logits, new_kvs) where new_kvs is list of (k, v) for each layer.
        """
        h = self.embed(input_ids)
        new_kvs = []

        for i, layer in enumerate(self.layers):
            pk = past_kvs[i][0] if past_kvs and i < len(past_kvs) and past_kvs[i] is not None else None
            pv = past_kvs[i][1] if past_kvs and i < len(past_kvs) and past_kvs[i] is not None else None
            h, k, v = layer(h, pk, pv)
            new_kvs.append((k, v))

        h = self.norm(h)
        logits = self.lm_head(h)
        return logits, new_kvs

    def generate_greedy(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 10,
    ) -> torch.Tensor:
        """Greedy autoregressive generation."""
        all_ids = input_ids.clone()
        past_kvs = None

        for _ in range(max_new_tokens):
            if past_kvs is None:
                logits, past_kvs = self.forward_with_cache(all_ids)
                next_logits = logits[:, -1, :]
            else:
                last_token = all_ids[:, -1:]
                logits, past_kvs = self.forward_with_cache(last_token, past_kvs)
                next_logits = logits[:, -1, :]

            next_token = next_logits.argmax(dim=-1, keepdim=True)
            all_ids = torch.cat([all_ids, next_token], dim=1)

        return all_ids


class TestKVCacheCorrectness:
    """Verify cached generation matches non-cached."""

    @pytest.fixture
    def model(self):
        return CachedTransformer(hidden_dim=64, num_layers=4, num_heads=4)

    def test_cache_vs_no_cache(self, model):
        """KV cache produces same logits as full recomputation."""
        model.eval()
        input_ids = torch.randint(0, 128, (2, 16))

        # Full forward (no cache)
        logits_full, _ = model.forward_with_cache(input_ids)

        # Incremental with cache: first 8 tokens, then 8 more
        first_half = input_ids[:, :8]
        second_half = input_ids[:, 8:]

        logits_first, kvs = model.forward_with_cache(first_half)
        logits_second, _ = model.forward_with_cache(second_half, kvs)

        # Concatenate logits from both approaches
        logits_cached = torch.cat([logits_first, logits_second], dim=1)

        assert torch.allclose(logits_full, logits_cached, atol=1e-5), \
            "KV cache produces different logits than full recomputation"

    def test_single_token_cache(self, model):
        """Single token forward with cache matches corresponding slice."""
        model.eval()
        input_ids = torch.randint(0, 128, (1, 10))

        # Full forward
        logits_full, full_kvs = model.forward_with_cache(input_ids)

        # Forward first 9 tokens, cache them, then forward token 10
        logits_prefix, prefix_kvs = model.forward_with_cache(input_ids[:, :9])
        last_token = input_ids[:, 9:]
        logits_last, _ = model.forward_with_cache(last_token, prefix_kvs)

        # Compare: logits_last[0, -1, :] should match logits_full[0, 9:10, :]
        assert torch.allclose(logits_last[:, -1, :], logits_full[:, 9:10, :], atol=1e-5), \
            "Cached single token logits mismatch"

    def test_cache_continuity(self, model):
        """Cache from step t produces same state as from step t-1 + new token."""
        model.eval()
        input_ids = torch.randint(0, 128, (1, 12))

        # Approach A: compute cache from first 8 tokens
        _, kvs_step8 = model.forward_with_cache(input_ids[:, :8])

        # Approach B: compute cache from first 4, then extend with next 4
        _, kvs_step4 = model.forward_with_cache(input_ids[:, :4])
        _, kvs_step4_extended = model.forward_with_cache(input_ids[:, 4:8], kvs_step4)

        # KV caches should match
        for i, ((k_a, v_a), (k_b, v_b)) in enumerate(zip(kvs_step8, kvs_step4_extended)):
            assert torch.allclose(k_a, k_b, atol=1e-5), f"K cache mismatch at layer {i}"
            assert torch.allclose(v_a, v_b, atol=1e-5), f"V cache mismatch at layer {i}"

    def test_greedy_generation_identical(self, model):
        """Greedy generation with cache produces same tokens as non-cached."""
        model.eval()
        prompt = torch.randint(0, 128, (1, 5))

        # Generate with cache
        with torch.no_grad():
            output_cached = model.generate_greedy(prompt, max_new_tokens=10)

        # Generate without cache (full recomputation each step)
        with torch.no_grad():
            output_no_cache = prompt.clone()
            for _ in range(10):
                logits, _ = model.forward_with_cache(output_no_cache)
                next_token = logits[:, -1:, :].argmax(dim=-1)
                output_no_cache = torch.cat([output_no_cache, next_token], dim=1)

        assert torch.equal(output_cached, output_no_cache), \
            f"Cached and non-cached generation differ"

    def test_empty_cache_forward(self, model):
        """Forward with empty cache = forward without cache."""
        model.eval()
        input_ids = torch.randint(0, 128, (1, 8))

        logits_no_cache, _ = model.forward_with_cache(input_ids)
        logits_empty_cache, _ = model.forward_with_cache(input_ids, past_kvs=None)

        assert torch.allclose(logits_no_cache, logits_empty_cache, atol=1e-5), \
            "Empty cache forward differs from no-cache forward"

    def test_cache_reuse_across_batches(self, model):
        """Same prefix in different batches produces same cache state."""
        model.eval()
        prefix = torch.randint(0, 128, (1, 6))
        suffix_a = torch.randint(0, 128, (1, 4))
        suffix_b = torch.randint(0, 128, (1, 4))

        # Cache the prefix
        _, prefix_kvs = model.forward_with_cache(prefix)

        # Extend with different suffixes
        logits_a, _ = model.forward_with_cache(suffix_a, prefix_kvs)
        logits_b, _ = model.forward_with_cache(suffix_b, prefix_kvs)

        # Full computation for verification
        full_a = torch.cat([prefix, suffix_a], dim=1)
        full_b = torch.cat([prefix, suffix_b], dim=1)
        logits_full_a, _ = model.forward_with_cache(full_a)
        logits_full_b, _ = model.forward_with_cache(full_b)

        assert torch.allclose(logits_a[:, -1, :], logits_full_a[:, -1, :], atol=1e-5), \
            "Cache reuse failed for batch A"
        assert torch.allclose(logits_b[:, -1, :], logits_full_b[:, -1, :], atol=1e-5), \
            "Cache reuse failed for batch B"
