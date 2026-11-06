"""Regression tests for PERF-05: semantic cache brute-force search must not
re-embed stored prompts on every search.

The original brute-force fallback called ``_embedder.embed(key)`` for every
cached entry on every search (api/semantic_cache.py:178), i.e. O(N) embedding
calls per lookup — and re-embedding a 384-d sentence-transformer vector for
5000 entries dominates request latency.  ``add()`` now stores the vector it
already computed, and ``search()`` reuses it.

Also guards the fallback embedder: ``hashlib`` was used without import, so
any fallback-path call raised ``NameError``.

PERF-08-JSON: the request body must be JSON-parsed once per request.
``BodyCacheMiddleware`` parses and stashes ``request.state.parsed_body``;
``SemanticCacheMiddleware`` must reuse it instead of re-parsing.
"""

import json

import pytest

from tests._import_helper import SRC_DIR, bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_mod = load_module("distllm/api/semantic_cache.py")
_mod_body_cache = load_module("distllm/api/body_cache_middleware.py")
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


_real_json_loads = json.loads


# ---------------------------------------------------------------------------
# PERF-08-JSON: request body must be JSON-parsed once per request
# ---------------------------------------------------------------------------


class _Countingloads:
    """Pass-through replacement for json.loads that counts invocations."""

    def __init__(self):
        self.count = 0

    def __call__(self, *args, **kwargs):
        self.count += 1
        return _real_json_loads(*args, **kwargs)


def _chat_app(body_middleware: bool):
    """Build a minimal chat app; BodyCacheMiddleware outermost when enabled."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat():
        return {"choices": [{"message": {"content": "ok"}}]}

    app.add_middleware(_mod.SemanticCacheMiddleware)
    if body_middleware:
        # add_middleware prepends: registering BodyCache LAST makes it the
        # OUTERMOST middleware, so it reads/parses the body first — same order
        # as production (server.py registers BodyCacheMiddleware last).
        app.add_middleware(_mod_body_cache.BodyCacheMiddleware)
    return TestClient(app)


class TestParseOnce:
    """The same JSON body must be parsed exactly once per request."""

    def test_body_cache_and_semantic_cache_parse_once(self, monkeypatch):
        counter = _Countingloads()
        monkeypatch.setattr(_mod.json, "loads", counter)
        monkeypatch.setattr(_mod_body_cache.json, "loads", counter)

        client = _chat_app(body_middleware=True)
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
            headers={"X-Semantic-Cache": "1"},
        )
        assert resp.status_code == 200
        assert counter.count == 1, (
            f"request body parsed {counter.count}x; BodyCacheMiddleware "
            "parses once and SemanticCacheMiddleware must reuse "
            "request.state.parsed_body instead of re-parsing"
        )

    def test_semantic_cache_fallback_parses_when_body_cache_absent(self, monkeypatch):
        """Without BodyCacheMiddleware the fallback parse still works."""
        counter = _Countingloads()
        monkeypatch.setattr(_mod.json, "loads", counter)

        client = _chat_app(body_middleware=False)
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
            headers={"X-Semantic-Cache": "1"},
        )
        assert resp.status_code == 200
        assert counter.count == 1  # parsed exactly once, by the fallback path


_real_json_loads = json.loads


# ---------------------------------------------------------------------------
# E-02-SEMANTIC-FLAG: DISTLLM_SEMANTIC_CACHE env gate
# ---------------------------------------------------------------------------


class TestSemanticCacheEnvFlag:
    """The middleware mounts only behind DISTLLM_SEMANTIC_CACHE=1."""

    def test_flag_off_middleware_not_in_stack(self):
        """Default (unset/0): SemanticCacheMiddleware is NOT mounted."""
        import os

        if os.environ.get("DISTLLM_SEMANTIC_CACHE", "0") == "1":
            pytest.skip("DISTLLM_SEMANTIC_CACHE=1 at import time — off-state "
                        "not observable in this process")
        from distllm.api.server import app

        assert not _mod.semantic_cache_enabled()
        names = [mw.cls.__name__ for mw in app.user_middleware]
        assert "SemanticCacheMiddleware" not in names

    def test_flag_on_middleware_in_stack(self):
        """DISTLLM_SEMANTIC_CACHE=1: SemanticCacheMiddleware IS mounted,
        registered BEFORE AuthMiddleware (add_middleware prepends) so the
        cache lookup runs after auth populates request.state.

        The gate is evaluated at server-module import time, so the flag-on
        case runs in a fresh subprocess — an in-process reload would rebind
        the shared ``app`` object other tests already hold.
        """
        import os
        import subprocess
        import sys

        code = (
            "from distllm.api.server import app\n"
            "classes = [mw.cls for mw in app.user_middleware]\n"
            "names = [c.__name__ for c in classes]\n"
            "assert 'SemanticCacheMiddleware' in names, names\n"
            "from distllm.api.semantic_cache import SemanticCacheMiddleware\n"
            "auth_idx = next(\n"
            "    i for i, c in enumerate(classes)\n"
            "    if c.__name__ == 'AuthMiddleware'\n"
            "    and c.__module__ == 'distllm.api.middleware')\n"
            "assert classes.index(SemanticCacheMiddleware) > auth_idx, names\n"
            "print('MOUNTED')\n"
        )
        env = dict(os.environ, DISTLLM_SEMANTIC_CACHE="1")
        env["PYTHONPATH"] = os.pathsep.join(
            [str(SRC_DIR), env.get("PYTHONPATH", "")]
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, env=env,
        )
        assert proc.returncode == 0, proc.stderr
        assert "MOUNTED" in proc.stdout

    def test_flag_gate_semantics(self, monkeypatch):
        """Only the literal value "1" enables; anything else stays off."""
        monkeypatch.delenv("DISTLLM_SEMANTIC_CACHE", raising=False)
        assert _mod.semantic_cache_enabled() is False
        monkeypatch.setenv("DISTLLM_SEMANTIC_CACHE", "0")
        assert _mod.semantic_cache_enabled() is False
        monkeypatch.setenv("DISTLLM_SEMANTIC_CACHE", "yes")
        assert _mod.semantic_cache_enabled() is False
        monkeypatch.setenv("DISTLLM_SEMANTIC_CACHE", "1")
        assert _mod.semantic_cache_enabled() is True


class TestSemanticCacheHitPath:
    """Flag on + X-Semantic-Cache header: second identical request is served
    from the cache (no route execution, X-Cache: HIT header)."""

    def _fresh_cache(self):
        cache = _FaissIndex()
        cache._index = None  # brute-force path: deterministic, no faiss needed
        cache._dim = 64
        return cache

    def test_flag_on_hit_returns_cached_response(self, monkeypatch):
        cache = self._fresh_cache()
        monkeypatch.setattr(_mod, "_cache", cache)
        monkeypatch.setenv("DISTLLM_SEMANTIC_CACHE", "1")

        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        hits = {"route": 0}

        app = FastAPI()

        @app.post("/v1/chat/completions")
        async def chat():
            hits["route"] += 1
            return {"choices": [{"message": {"content": "Paris"}}]}

        app.add_middleware(_mod.SemanticCacheMiddleware)
        client = TestClient(app)

        payload = {"model": "m", "messages": [{"role": "user", "content": "capital of france"}]}
        first = client.post("/v1/chat/completions", json=payload,
                            headers={"X-Semantic-Cache": "1"})
        assert first.status_code == 200
        assert first.headers["X-Cache"] == "MISS"
        assert hits["route"] == 1

        second = client.post("/v1/chat/completions", json=payload,
                             headers={"X-Semantic-Cache": "1"})
        assert second.status_code == 200
        assert second.headers["X-Cache"] == "HIT"
        assert second.headers["X-Cache-Similarity"]  # similarity score reported
        assert hits["route"] == 1  # route did NOT run again — served from cache
        assert second.json() == first.json()

    def test_semantic_similarity_hit(self, monkeypatch):
        """A semantically equivalent (not byte-identical) prompt hits."""
        cache = self._fresh_cache()
        monkeypatch.setattr(_mod, "_cache", cache)
        monkeypatch.setenv("DISTLLM_SEMANTIC_CACHE", "1")

        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        app = FastAPI()

        @app.post("/v1/chat/completions")
        async def chat():
            return {"choices": [{"message": {"content": "Paris"}}]}

        app.add_middleware(_mod.SemanticCacheMiddleware)
        client = TestClient(app)

        client.post("/v1/chat/completions",
                    json={"model": "m", "messages": [{"role": "user", "content": "what is the capital of france"}]},
                    headers={"X-Semantic-Cache": "0.9"})
        resp = client.post("/v1/chat/completions",
                           json={"model": "m", "messages": [{"role": "user", "content": "what is the capital city of france"}]},
                           headers={"X-Semantic-Cache": "0.5"})
        assert resp.status_code == 200
        assert resp.headers["X-Cache"] == "HIT"

    def test_flag_on_bypasses_without_header(self, monkeypatch):
        """Flag on but X-Semantic-Cache absent: normal processing, no cache."""
        cache = self._fresh_cache()
        monkeypatch.setattr(_mod, "_cache", cache)
        monkeypatch.setenv("DISTLLM_SEMANTIC_CACHE", "1")

        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        app = FastAPI()

        @app.post("/v1/chat/completions")
        async def chat():
            return {"choices": [{"message": {"content": "fresh"}}]}

        app.add_middleware(_mod.SemanticCacheMiddleware)
        client = TestClient(app)

        payload = {"model": "m", "messages": [{"role": "user", "content": "hello"}]}
        r1 = client.post("/v1/chat/completions", json=payload)
        r2 = client.post("/v1/chat/completions", json=payload)
        assert r1.headers.get("X-Cache") != "HIT"
        assert r2.headers.get("X-Cache") != "HIT"
        assert r1.json() == r2.json() == {"choices": [{"message": {"content": "fresh"}}]}

    def test_streaming_requests_never_cached(self, monkeypatch):
        """stream=true bypasses the semantic cache entirely."""
        cache = self._fresh_cache()
        monkeypatch.setattr(_mod, "_cache", cache)
        monkeypatch.setenv("DISTLLM_SEMANTIC_CACHE", "1")

        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        app = FastAPI()

        @app.post("/v1/chat/completions")
        async def chat():
            return {"choices": [{"message": {"content": "streamed"}}]}

        app.add_middleware(_mod.SemanticCacheMiddleware)
        client = TestClient(app)

        payload = {"model": "m", "stream": True,
                   "messages": [{"role": "user", "content": "hi"}]}
        r1 = client.post("/v1/chat/completions", json=payload,
                         headers={"X-Semantic-Cache": "1"})
        assert r1.headers.get("X-Cache") != "HIT"
        assert len(cache._response_store) == 0  # nothing stored
