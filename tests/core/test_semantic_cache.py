"""Regression tests for SemanticCache scope-aware invalidation.

Covers B16: ``SemanticCache.invalidate()`` used to pop the bare prompt
hash while entries are keyed by ``"{scope}:{prompt_hash}"``, so it could
never remove anything — stale/poisoned responses could not be purged.

Uses the import-helper pattern to load the source module directly.
"""

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_sem_cache_mod = load_module("distllm/core/semantic_cache.py")
SemanticCache = _sem_cache_mod.SemanticCache


class TestSemanticCacheInvalidate:
    """invalidate must pop the same scoped key used by store/lookup."""

    def test_invalidate_with_scope_removes_entry(self):
        cache = SemanticCache()
        cache.store("What is Python?", response="Python is a language", scope="tenantA")

        assert cache.lookup("What is Python?", scope="tenantA") == "Python is a language"

        removed = cache.invalidate("What is Python?", scope="tenantA")
        assert removed is True
        assert cache.lookup("What is Python?", scope="tenantA") is None

    def test_invalidate_without_scope_is_safe_noop_for_scoped_entry(self):
        cache = SemanticCache()
        cache.store("What is Python?", response="Python is a language", scope="tenantA")

        removed = cache.invalidate("What is Python?")
        assert removed is False
        # The scoped entry must survive an unscoped invalidate.
        assert cache.lookup("What is Python?", scope="tenantA") == "Python is a language"

    def test_invalidate_with_wrong_scope_does_not_remove_entry(self):
        cache = SemanticCache()
        cache.store("What is Python?", response="Python is a language", scope="tenantA")

        removed = cache.invalidate("What is Python?", scope="tenantB")
        assert removed is False
        assert cache.lookup("What is Python?", scope="tenantA") == "Python is a language"

    def test_invalidate_unscoped_removes_unscoped_entry(self):
        cache = SemanticCache()
        cache.store("What is Python?", response="Python is a language")

        removed = cache.invalidate("What is Python?")
        assert removed is True
        assert cache.lookup("What is Python?") is None

    def test_invalidate_missing_prompt_returns_false(self):
        cache = SemanticCache()
        cache.store("What is Python?", response="Python is a language", scope="tenantA")

        assert cache.invalidate("No such prompt", scope="tenantA") is False
        # Entry for the real prompt is untouched.
        assert cache.lookup("What is Python?", scope="tenantA") == "Python is a language"

    def test_invalidate_one_scope_leaves_other_scope_intact(self):
        cache = SemanticCache()
        cache.store("What is Python?", response="Python is a language", scope="tenantA")
        cache.store("What is Python?", response="Python is a language (B)", scope="tenantB")

        assert cache.invalidate("What is Python?", scope="tenantA") is True
        assert cache.lookup("What is Python?", scope="tenantA") is None
        assert cache.lookup("What is Python?", scope="tenantB") == "Python is a language (B)"


class TestSemanticCacheAdaptiveThreshold:
    """Regression tests for audit 2026-11-13 semantic_cache findings.

    adaptive_threshold used to be unusable: every 10th semantic hit raised
    AttributeError (no hit_rate attribute) and, once that was patched, the
    update clamped to the hard-coded [0.80, 0.99] band — silently raising
    an explicitly configured threshold (0.5 -> 0.80).
    """

    @staticmethod
    def _warm_cache(cache, hits: int = 11):
        cache.store("Q", response="R", embedding=[1.0] * 16, scope="t")
        for _ in range(hits):
            cache.lookup("other", embedding=[0.95] * 16, scope="t")

    def test_adaptive_lookup_does_not_raise_attribute_error(self):
        cache = SemanticCache(similarity_threshold=0.5, adaptive_threshold=True)
        self._warm_cache(cache)  # 10th+ semantic hit drove _update_threshold
        assert cache._threshold > 0

    def test_explicit_threshold_is_not_raised_into_default_band(self):
        """The adaptive band must include the caller's explicit threshold."""
        cache = SemanticCache(similarity_threshold=0.5, adaptive_threshold=True)
        self._warm_cache(cache)
        assert cache._threshold < 0.8, (
            f"explicit threshold 0.5 was clamped up to {cache._threshold}; "
            f"0.6-similarity prompts the caller wanted to accept now miss"
        )
        # A lookup with true cosine similarity 0.6 must still hit: the
        # query is the unit vector at 53.13 deg from the stored one.
        half = [0.5 ** 0.5] * 8 + [0.0] * 8
        cache.store("Q2", response="R2", embedding=[1.0] * 8 + [0.0] * 8, scope="s")
        assert cache.lookup("y", embedding=half, scope="s") == "R2"

    def test_hit_rate_property_exists(self):
        cache = SemanticCache()
        assert cache.hit_rate == 0.0
        cache.store("P", response="R")
        cache.lookup("P")
        assert cache.hit_rate == 1.0
