"""Regression tests for PERF-05: semantic cache brute-force search must not
re-embed stored prompts on every search.

The original brute-force fallback called ``_embedder.embed(key)`` for every
cached entry on every search (api/semantic_cache.py:178), i.e. O(N) embedding
calls per lookup — and re-embedding a 384-d sentence-transformer vector for
5000 entries dominates request latency.  ``add()`` now stores the vector it
already computed, and ``search()`` reuses it.

Also guards the fallback embedder: ``hashlib`` was used without import, so
any fallback-path call raised ``NameError``.
"""

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_mod = load_module("distllm/api/semantic_cache.py")
_FaissIndex = _mod._FaissIndex
_embedder = _mod._embedder


class _CountingEmbedder:
    """Wraps the real _embedder and counts calls."""

    def __init__(self, inner):
        self._inner = inner
        self.calls = 0
        self.texts = []

    def embed(self, text):
        self.calls += 1
        self.texts.append(text)
        return self._inner.embed(text)


class TestFallbackEmbedder:
    def test_hash_fallback_embed_works(self):
        # Before the fix: NameError: name 'hashlib' is not defined
        vec = _embedder.embed("hello world")
        assert isinstance(vec, list)
        assert len(vec) == 64
        assert abs(sum(v * v for v in vec) ** 0.5 - 1.0) < 1e-9

    def test_hash_fallback_is_deterministic(self):
        assert _embedder.embed("hello world") == _embedder.embed("hello world")


class TestNoReEmbedOnSearch:
    """Brute-force search must issue zero embed calls for stored prompts."""

    def _brute_force_cache(self):
        cache = _FaissIndex()
        # Force the brute-force path regardless of whether faiss is installed,
        # so the test exercises the code path PERF-05 targets.
        cache._index = None
        cache._dim = 64
        return cache

    def test_search_after_add_issues_no_embed_calls(self, monkeypatch):
        cache = self._brute_force_cache()
        counter = _CountingEmbedder(_embedder)
        monkeypatch.setattr(_mod, "_embedder", counter)

        for i in range(5):
            cache.add(f"prompt {i} hello world", f"response {i}")
        assert counter.calls == 5  # one embed per add

        counter.calls = 0
        counter.texts.clear()
        query = counter._inner.embed("hello world")  # embed query outside counter
        result, sim = cache.search(query, threshold=0.5)
        assert result == "response 0"
        assert counter.calls == 0, (
            f"search re-embedded {counter.calls} stored prompts: "
            f"{counter.texts} — must reuse vectors stored at add() time"
        )

    def test_add_stores_vector(self):
        cache = self._brute_force_cache()
        cache.add("capital of France?", "Paris")
        assert "capital of France?" in cache._vec_store
        assert len(cache._vec_store["capital of France?"]) == 64

    def test_evict_clears_stored_vector(self):
        cache = self._brute_force_cache()
        cache.add("capital of France?", "Paris")
        cache._evict("capital of France?")
        assert "capital of France?" not in cache._vec_store
        assert "capital of France?" not in cache._response_store

    def test_search_correctness_preserved(self):
        """Cached vectors give the same result as re-embedding would."""
        cache = self._brute_force_cache()
        prompts = [
            ("what is the capital of france", "Paris"),
            ("how do I bake bread", "Mix and bake"),
            ("explain quantum mechanics", "Qubits"),
        ]
        for prompt, resp in prompts:
            cache.add(prompt, resp)

        query = _embedder.embed("what is the capital of france")
        result, sim = cache.search(query, threshold=0.99)
        assert result == "Paris"
        assert sim > 0.99

    def test_no_match_below_threshold(self):
        cache = self._brute_force_cache()
        cache.add("what is the capital of france", "Paris")
        query = _embedder.embed("completely unrelated baking question")
        result, sim = cache.search(query, threshold=0.99)
        assert result is None
        assert sim == 0.0

    def test_legacy_entry_without_vector_is_embedded_once_and_cached(self, monkeypatch):
        """Entries inserted without a vector (e.g. direct dict writes) are
        embedded once on first search, then cached — not on every search."""
        cache = self._brute_force_cache()
        cache._response_store["legacy prompt about cats"] = "A cat response"
        cache._ts_store["legacy prompt about cats"] = 1e18  # far future, not stale

        counter = _CountingEmbedder(_embedder)
        monkeypatch.setattr(_mod, "_embedder", counter)

        query = counter._inner.embed("legacy prompt about cats")
        result, sim = cache.search(query, threshold=0.5)
        assert result == "A cat response"
        assert counter.calls == 1  # embedded once, for the legacy entry
        assert "legacy prompt about cats" in cache._vec_store

        counter.calls = 0
        result2, _ = cache.search(query, threshold=0.5)
        assert result2 == "A cat response"
        assert counter.calls == 0  # now cached, no re-embed
