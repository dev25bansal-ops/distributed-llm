"""Property-based tests for KV cache operations."""

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from distllm.core.kv_cache import KVCache, KVCacheManager


@given(
    num_layers=st.integers(1, 32),
    batch_size=st.integers(1, 8),
    num_heads=st.integers(1, 8),
    head_dim=st.integers(1, 128),
)
@settings(max_examples=50, deadline=None)
def test_kv_cache_init(num_layers, batch_size, num_heads, head_dim):
    """KVCache should initialize with correct structure."""
    cache = KVCache()
    cache.init_cache(num_layers, batch_size, num_heads, head_dim)

    assert cache.num_layers == num_layers
    assert len(cache.cache) == num_layers
    assert cache.sequence_length == 0

    for k, v in cache.cache:
        assert k.shape == (batch_size, num_heads, 0, head_dim)
        assert v.shape == (batch_size, num_heads, 0, head_dim)


@given(
    num_layers=st.integers(1, 16),
    batch_size=st.integers(1, 4),
    num_heads=st.integers(1, 4),
    head_dim=st.integers(1, 64),
    seq_len=st.integers(1, 32),
)
@settings(max_examples=30, deadline=None)
def test_kv_cache_update(num_layers, batch_size, num_heads, head_dim, seq_len):
    """Appending to KV cache should increase sequence length correctly."""
    cache = KVCache()
    cache.init_cache(num_layers, batch_size, num_heads, head_dim)

    for layer_idx in range(num_layers):
        new_k = torch.randn(batch_size, num_heads, seq_len, head_dim)
        new_v = torch.randn(batch_size, num_heads, seq_len, head_dim)
        k, v = cache.update(layer_idx, new_k, new_v)

        assert k.shape == (batch_size, num_heads, seq_len, head_dim)
        assert v.shape == (batch_size, num_heads, seq_len, head_dim)

    assert cache.sequence_length == seq_len

    # Second append should concatenate
    new_k = torch.randn(batch_size, num_heads, seq_len, head_dim)
    new_v = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k, v = cache.update(0, new_k, new_v)
    assert k.shape == (batch_size, num_heads, seq_len * 2, head_dim)
    assert cache.sequence_length == seq_len * 2


@given(
    num_requests=st.integers(1, 20),
    num_layers=st.integers(1, 8),
    batch_size=st.integers(1, 2),
    num_heads=st.integers(1, 2),
    head_dim=st.integers(1, 16),
)
@settings(max_examples=20, deadline=None)
def test_kv_cache_manager(num_requests, num_layers, batch_size, num_heads, head_dim):
    """KVCacheManager should handle multiple concurrent request caches."""
    manager = KVCacheManager()
    request_ids = [f"req-{i}" for i in range(num_requests)]

    for rid in request_ids:
        manager.create(rid, num_layers, batch_size, num_heads, head_dim)

    assert manager.active_requests == num_requests

    for rid in request_ids:
        cache = manager.get(rid)
        assert cache is not None
        assert cache.num_layers == num_layers

    # Delete all
    for rid in request_ids:
        manager.delete(rid)

    assert manager.active_requests == 0
    assert manager.total_memory_usage() == 0
