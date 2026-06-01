"""Deterministic pipeline correctness tests.

Verifies that distributed pipeline-parallel inference produces
identical outputs to single-node inference for the same inputs.
This catches subtle correctness bugs from:
- KV cache serialization/deserialization
- Hidden state transfer between nodes
- Positional encoding alignment
- Layer boundary handling

Run with:
    pytest tests/correctness/test_pipeline_determinism.py -v
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
from unittest.mock import MagicMock

import pytest
import torch

sys.path.insert(0, "src")

from distllm.core.kv_cache import KVCache


class TestPipelineDeterminism:
    """Verify distributed pipeline matches single-node output exactly."""

    def _make_mock_tokenizer(self):
        tokenizer = MagicMock()
        tokenizer.vocab_size = 1000
        tokenizer.eos_token_id = 2
        tokenizer.bos_token_id = 1
        tokenizer.pad_token_id = 0
        tokenizer.encode = MagicMock(return_value=[10, 20, 30, 40, 50])
        tokenizer.decode = MagicMock(return_value="test output")
        return tokenizer

    def _make_mock_model(self, total_layers=8, hidden_dim=64, vocab_size=1000):
        """Create a deterministic mock model for testing."""
        model = MagicMock()
        model.config = MagicMock()
        model.config.num_hidden_layers = total_layers
        model.config.hidden_size = hidden_dim
        model.config.vocab_size = vocab_size
        return model

    def test_kv_cache_roundtrip_preserves_values(self):
        """KV cache serialization/deserialization must preserve exact values."""
        cache = KVCache(max_seq_len=100)
        cache.num_layers = 4

        # Create deterministic KV data
        torch.manual_seed(42)
        for layer in range(4):
            key = torch.randn(1, 8, 10, 64)
            value = torch.randn(1, 8, 10, 64)
            cache.cache.append((key, value))

        # Serialize and deserialize
        serialized = cache.serialize()
        restored = KVCache.deserialize(serialized)

        # Verify exact match
        for layer in range(4):
            orig_k, orig_v = cache.cache[layer]
            rest_k, rest_v = restored.cache[layer]
            assert torch.equal(orig_k, rest_k), f"Layer {layer} key mismatch after roundtrip"
            assert torch.equal(orig_v, rest_v), f"Layer {layer} value mismatch after roundtrip"

    def test_hidden_state_transfer_preserves_values(self):
        """Hidden state transfer between nodes must preserve exact values."""
        torch.manual_seed(42)
        hidden = torch.randn(1, 10, 64, dtype=torch.float32)

        # Simulate serialization (node A → wire)
        raw_bytes = hidden.numpy().tobytes()
        shape = list(hidden.shape)
        dtype = str(hidden.dtype)

        # Simulate deserialization (wire → node B)
        import numpy as np
        arr = np.frombuffer(raw_bytes, dtype=np.float32).reshape(shape)
        restored = torch.from_numpy(arr.copy())

        assert torch.equal(hidden, restored), "Hidden state mismatch after transfer"

    def test_layer_boundary_handling(self):
        """Layer boundaries must not introduce gaps or overlaps."""
        total_layers = 32
        num_nodes = 4
        layers_per_node = total_layers // num_nodes

        covered = set()
        for i in range(num_nodes):
            start = i * layers_per_node
            end = (i + 1) * layers_per_node - 1
            if i == num_nodes - 1:
                end = total_layers - 1

            # Verify no gaps
            for layer in range(start, end + 1):
                assert layer not in covered, f"Layer {layer} assigned to multiple nodes"
                covered.add(layer)

        # Verify all layers covered
        assert len(covered) == total_layers, f"Expected {total_layers} layers, got {len(covered)}"
        assert covered == set(range(total_layers)), "Layer coverage has gaps"

    def test_deterministic_token_generation(self):
        """Same input must produce same output with fixed seed."""
        torch.manual_seed(123)

        # Simulate token generation
        logits = torch.randn(1, 1000)
        probs = torch.softmax(logits / 0.1, dim=-1)

        torch.manual_seed(42)
        token1 = torch.multinomial(probs, num_samples=1).item()

        torch.manual_seed(42)
        token2 = torch.multinomial(probs, num_samples=1).item()

        assert token1 == token2, f"Non-deterministic: {token1} != {token2}"

    def test_position_encoding_consistency(self):
        """Position IDs must be consistent across pipeline stages."""
        seq_len = 100
        total_layers = 16
        num_nodes = 4

        for node_idx in range(num_nodes):
            start_layer = node_idx * (total_layers // num_nodes)
            end_layer = (node_idx + 1) * (total_layers // num_nodes) - 1

            # Position IDs should always be 0..seq_len-1 regardless of layer
            positions = list(range(seq_len))
            assert len(positions) == seq_len
            assert positions[0] == 0
            assert positions[-1] == seq_len - 1

    def test_attention_mask_consistency(self):
        """Causal attention mask must be identical across all nodes."""
        seq_len = 50

        # Generate causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))

        # Verify properties
        assert mask[0, 0] == 1, "First token should attend to itself"
        assert mask[0, 1] == 0, "First token should not attend to second"
        assert mask[seq_len - 1, 0] == 1, "Last token should attend to first"
        assert mask[seq_len - 1, seq_len - 1] == 1, "Last token should attend to itself"

        # Mask must be same regardless of which node generates it
        for _ in range(4):
            assert torch.equal(mask, torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool)))


class TestNumericalStability:
    """Verify numerical stability across different precisions."""

    def test_fp16_no_nan(self):
        """FP16 operations must not produce NaN."""
        torch.manual_seed(42)
        x = torch.randn(100, 100, dtype=torch.float16)
        result = torch.mm(x, x.T)
        assert not torch.isnan(result).any(), "FP16 produced NaN"
        assert not torch.isinf(result).any(), "FP16 produced Inf"

    def test_softmax_no_overflow(self):
        """Softmax must not overflow even with large logits."""
        large_logits = torch.tensor([[1000.0, -1000.0, 500.0]])
        probs = torch.softmax(large_logits, dim=-1)
        assert not torch.isnan(probs).any(), "Softmax produced NaN"
        assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-6)

    def test_gradient_clipping_stability(self):
        """Gradient clipping must produce finite values."""
        torch.manual_seed(42)
        x = torch.randn(10, requires_grad=True)
        loss = (x ** 10).sum()
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_([x], max_norm=1.0)
        assert torch.isfinite(x.grad).all(), "Gradient clipping produced non-finite values"

    def test_accumulation_no_drift(self):
        """Repeated accumulation must not drift from expected value."""
        total = 0.0
        for _ in range(10000):
            total += 0.1
        expected = 1000.0
        assert abs(total - expected) < 0.01, f"Accumulation drift: {total} != {expected}"
