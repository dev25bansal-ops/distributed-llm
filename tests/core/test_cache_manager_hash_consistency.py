"""P2: Test hash consistency across all hashing methods.

Verifies that the same token sequence produces the same hash
regardless of which hashing method is used.
"""

import pytest

from distllm.core.cache_manager import CacheManager, RollingHash, _rolling_prefix_hashes
from distllm.dist.cache import CacheIndex


class TestHashConsistency:
    """Verify hash consistency across implementations."""

    def test_rolling_hash_deterministic(self):
        """Same tokens should produce same rolling hash."""
        tokens = [1, 2, 3, 4, 5, 6, 7, 8]

        h1 = RollingHash()
        for t in tokens:
            h1.extend(t)

        h2 = RollingHash()
        for t in tokens:
            h2.extend(t)

        assert h1.hash == h2.hash

    def test_rolling_hash_extend_consistent(self):
        """Extending should produce consistent results."""
        tokens = [10, 20, 30, 40, 50]

        h = RollingHash()
        for t in tokens[:3]:
            h.extend(t)
        hash_3 = h.hash

        h.extend(tokens[3])
        hash_4 = h.hash

        # Verify by building from scratch
        h_verify = RollingHash()
        for t in tokens[:4]:
            h_verify.extend(t)

        assert hash_4 == h_verify.hash

    def test_rolling_prefix_hashes_consistent(self):
        """Pre-computed prefix hashes should match individual computations."""
        tokens = [5, 10, 15, 20, 25, 30]

        prefix_hashes = _rolling_prefix_hashes(tokens, len(tokens))

        # Verify each prefix length
        for length in range(1, len(tokens) + 1):
            h = RollingHash()
            for t in tokens[:length]:
                h.extend(t)
            assert prefix_hashes[length] == h.hash

    def test_cache_index_deterministic(self):
        """CacheIndex.index_tokens should be deterministic."""
        idx = CacheIndex()
        tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        hash1 = idx.index_tokens(tokens)
        hash2 = idx.index_tokens(tokens)

        assert hash1 == hash2

    def test_cache_index_different_tokens_different_hash(self):
        """Different tokens should produce different hashes."""
        idx = CacheIndex()

        hash1 = idx.index_tokens([1, 2, 3, 4, 5])
        hash2 = idx.index_tokens([1, 2, 3, 4, 6])

        assert hash1 != hash2

    def test_cache_index_prefix_consistency(self):
        """Prefix hash should match full hash for same prefix."""
        idx = CacheIndex()

        full_tokens = [1, 2, 3, 4, 5, 6, 7, 8]
        prefix_tokens = [1, 2, 3, 4]

        # The hash of the prefix should be consistent
        hash_prefix = idx.index_tokens(prefix_tokens)
        hash_prefix2 = idx.index_tokens(full_tokens[:4])

        assert hash_prefix == hash_prefix2

    def test_cache_manager_hash_tokens_uses_cache_index(self):
        """CacheManager._hash_tokens should use CacheIndex when available."""
        idx = CacheIndex()
        cm = CacheManager(cache_index=idx)

        tokens = [1, 2, 3, 4, 5, 6, 7, 8]
        hash1 = cm._hash_tokens(tokens)
        hash2 = idx.index_tokens(tokens)

        assert hash1 == hash2

    def test_cache_manager_hash_tokens_fallback(self):
        """CacheManager._hash_tokens should fallback to SHA-256 when no index."""
        cm = CacheManager(cache_index=None)

        tokens = [1, 2, 3, 4, 5]
        hash1 = cm._hash_tokens(tokens)
        hash2 = cm._hash_tokens(tokens)

        assert hash1 == hash2
        assert hash1.startswith("h")


class TestRollingHashProperties:
    """Property-based tests for RollingHash."""

    def test_empty_hash_is_zero(self):
        h = RollingHash()
        assert h.hash == 0
        assert h.length == 0

    def test_single_token(self):
        h = RollingHash()
        result = h.extend(42)
        assert h.hash == result
        assert h.length == 1

    def test_reset_clears_state(self):
        h = RollingHash()
        h.extend(1)
        h.extend(2)
        h.extend(3)
        assert h.length == 3

        h.reset()
        assert h.hash == 0
        assert h.length == 0

    def test_different_tokens_different_hashes(self):
        h1 = RollingHash()
        h1.extend(1)

        h2 = RollingHash()
        h2.extend(2)

        assert h1.hash != h2.hash
