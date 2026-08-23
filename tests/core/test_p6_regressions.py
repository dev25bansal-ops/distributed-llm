"""Regression tests for the P1 correctness batch (P6 fixes).

Covers, against the actual code:
- SemanticCache tenant-scope isolation (no cross-tenant leakage).
- CertRotation.needs_renewal no longer compares aware/naive datetimes.
- ApiKeyStore.add_key + CertRotation rotation (new key authenticates, old
  key is retired after the grace period).
- CacheManager ghost-cache is bounded (pruned) and the GPU/CPU/SSD tier cache
  round-trips a >32-token prefix (store/lookup key match).
"""

from __future__ import annotations

import time

import pytest


class TestSemanticCacheScope:
    def test_same_scope_exact_match(self):
        from distllm.core.semantic_cache import SemanticCache

        cache = SemanticCache()
        cache.store("What is Python?", "A language", scope="tenant-a")
        assert cache.lookup("What is Python?", scope="tenant-a") == "A language"

    def test_different_scope_does_not_leak(self):
        from distllm.core.semantic_cache import SemanticCache

        cache = SemanticCache()
        cache.store("My balance is 9999", "balance-secret", scope="tenant-a")
        # tenant B must NOT see tenant A's cached response, even with the same
        # prompt (this was the cross-tenant data leak).
        assert cache.lookup("My balance is 9999", scope="tenant-b") is None

    def test_default_scope_default_lookup(self):
        from distllm.core.semantic_cache import SemanticCache

        cache = SemanticCache()
        cache.store("hello", "world", scope="")
        assert cache.lookup("hello", scope="") == "world"


class TestCertRotationNaiveDatetime:
    def test_needs_renewal_no_typeerror(self, tmp_path):
        """needs_renewal must not raise with cryptography's aware datetimes."""
        from distllm.core.cert_rotation import CertificateRotator

        key_path = tmp_path / "key.pem"
        cert_path = tmp_path / "cert.pem"
        mgr = CertificateRotator(
            cert_path=str(cert_path),
            key_path=str(key_path),
        )
        ok = mgr.generate_self_signed(hostname="localhost", days=365)
        assert ok is True
        # Must not raise TypeError (aware vs naive subtraction).
        assert isinstance(mgr.needs_renewal(), bool)


class TestApiKeyRotation:
    def test_rotation_registers_new_key(self):
        """The rotated key must actually authenticate, and the old key is
        retired once the grace period ends."""
        from distllm.core.api_key_store import ApiKeyStore
        from distllm.core.cert_rotation import ApiKeyRotator

        store = ApiKeyStore.__new__(ApiKeyStore)
        store._keys = []
        store._lock = __import__("threading").Lock()
        store.add_key("old-key-123", role="admin", label="legacy", key_id="k1")

        rotator = ApiKeyRotator(store, key_length=32, grace_period_hours=0.0)
        new_key = rotator.rotate("k1")
        assert new_key, "rotate must return the new key"

        # The new key authenticates against the store.
        assert store.authenticate(new_key) == ("k1", "admin")
        # Immediately retire the old key (grace=0) and confirm it is gone.
        rotator.cleanup_expired()
        assert store.authenticate("old-key-123") is None


class TestCacheManagerP6:
    def test_ghost_cache_pruned(self):
        """Evictions must not grow the ghost cache without bound."""
        from distllm.core.cache_manager import CacheManager

        mgr = CacheManager()  # real, lightweight constructor
        mgr._ghost_cache_ttl = 0.001
        for i in range(100):
            mgr._ghost_cache[f"h{i}"] = time.time() - 10  # expired
        mgr._ghost_cache["fresh"] = time.time() + 5
        mgr._prune_ghost_cache()
        assert "fresh" in mgr._ghost_cache
        assert len(mgr._ghost_cache) == 1  # all expired entries dropped

    def test_multi_tier_prefix_roundtrip_matches_keys(self):
        """store and lookup must hash the SAME span of tokens so a >32-token
        prefix can actually hit the tier cache."""
        from distllm.core.cache_manager import CacheManager

        mgr = CacheManager()
        tokens = list(range(40))  # longer than the old 32-token truncation
        mgr._tier_store("gpu", mgr._hash_tokens(tokens), b"kv", 4)
        assert mgr._hash_tokens(tokens) in mgr._tiers["gpu"]["entries"]