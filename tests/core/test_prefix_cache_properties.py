"""Property-based tests for PrefixCache using Hypothesis.

Tests invariants that must hold for any valid cache implementation.
"""

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from distllm.dist.prefix_cache import PrefixCache


# Strategy for token sequences long enough for min_prefix_len=16
token_sequences = st.lists(
    st.integers(min_value=0, max_value=32000),
    min_size=16,
    max_size=512,
)

# Strategy for short token sequences (below min_prefix_len)
short_token_sequences = st.lists(
    st.integers(min_value=0, max_value=32000),
    min_size=1,
    max_size=15,
)


class TestPrefixCacheProperties:
    """Property-based tests for PrefixCache."""

    @given(tokens=token_sequences)
    @settings(max_examples=50)
    def test_store_then_lookup_returns_same_data(self, tokens):
        """Storing tokens then looking them up should return the same data."""
        cache = PrefixCache(min_prefix_len=16, max_entries=100)
        kv_data = {"test": "value"}
        cache.store(tokens, kv_data)

        match_len, result = cache.lookup(tokens)
        assert match_len == len(tokens)
        assert result == kv_data

    @given(tokens=token_sequences)
    @settings(max_examples=50)
    def test_lookup_miss_returns_zero(self, tokens):
        """Looking up tokens that were never stored should return (0, None)."""
        cache = PrefixCache(min_prefix_len=16, max_entries=100)
        # Store something different
        other_tokens = [t + 1 for t in tokens]
        cache.store(other_tokens, {"other": "data"})

        match_len, result = cache.lookup(tokens)
        assert match_len == 0

    @given(tokens=short_token_sequences)
    @settings(max_examples=50)
    def test_short_tokens_never_stored(self, tokens):
        """Tokens shorter than min_prefix_len should never be stored."""
        cache = PrefixCache(min_prefix_len=16, max_entries=100)
        cache.store(tokens, {"should": "not_store"})

        match_len, result = cache.lookup(tokens)
        assert match_len == 0

    @given(
        tokens_a=st.lists(st.integers(min_value=0, max_value=100), min_size=16, max_size=64),
        tokens_b=st.lists(st.integers(min_value=0, max_value=100), min_size=16, max_size=64),
    )
    @settings(max_examples=30)
    def test_different_tokens_different_entries(self, tokens_a, tokens_b):
        """Different token sequences should not interfere with each other."""
        assume(tokens_a != tokens_b)

        cache = PrefixCache(min_prefix_len=16, max_entries=100)
        cache.store(tokens_a, {"id": "a"})
        cache.store(tokens_b, {"id": "b"})

        match_a, data_a = cache.lookup(tokens_a)
        match_b, data_b = cache.lookup(tokens_b)

        if match_a > 0:
            assert data_a == {"id": "a"}
        if match_b > 0:
            assert data_b == {"id": "b"}

    @given(tokens=token_sequences)
    @settings(max_examples=30)
    def test_evict_removes_entry(self, tokens):
        """Evicting tokens should remove them from the cache."""
        cache = PrefixCache(min_prefix_len=16, max_entries=100)
        cache.store(tokens, {"data": "test"})

        evicted = cache.evict(tokens)
        assert evicted is True

        match_len, result = cache.lookup(tokens)
        assert match_len == 0

    @given(tokens=token_sequences)
    @settings(max_examples=30)
    def test_clear_removes_all(self, tokens):
        """Clearing the cache should remove all entries."""
        cache = PrefixCache(min_prefix_len=16, max_entries=100)
        cache.store(tokens, {"data": "test"})

        cache.clear()
        stats = cache.stats()
        assert stats["prefix_cache_entries"] == 0

    @given(tokens=token_sequences)
    @settings(max_examples=30)
    def test_hit_rate_bounded(self, tokens):
        """Hit rate should always be between 0 and 1."""
        cache = PrefixCache(min_prefix_len=16, max_entries=100)
        cache.store(tokens, {"data": "test"})
        cache.lookup(tokens)  # hit
        cache.lookup([999] * 20)  # miss

        assert 0.0 <= cache.hit_rate <= 1.0


class TestCacheIndexProperties:
    """Property-based tests for CacheIndex."""

    @given(
        tokens=st.lists(st.integers(min_value=0, max_value=32000), min_size=1, max_size=256),
    )
    @settings(max_examples=50)
    def test_index_tokens_deterministic(self, tokens):
        """Same tokens should always produce the same hash."""
        from distllm.dist.cache import CacheIndex

        idx = CacheIndex()
        hash1 = idx.index_tokens(tokens)
        hash2 = idx.index_tokens(tokens)
        assert hash1 == hash2

    @given(
        tokens=st.lists(st.integers(min_value=0, max_value=100), min_size=1, max_size=64),
        node_id=st.text(min_size=1, max_size=20),
    )
    @settings(max_examples=30)
    def test_store_and_lookup(self, tokens, node_id):
        """Storing an entry then looking it up should return the node."""
        from distllm.dist.cache import CacheIndex

        idx = CacheIndex()
        prefix_hash = idx.index_tokens(tokens)
        idx.store(prefix_hash, node_id, "ref-1")

        result = idx.lookup(prefix_hash)
        assert result == node_id
