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
