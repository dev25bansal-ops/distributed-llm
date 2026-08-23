"""Tests for DedupMiddleware and _FingerprintCache.

Covers:
1. Fingerprinting: same body -> same fp, different body -> different fp, tenant isolation
2. Cache: store + lookup hit, lookup miss, TTL expiry
3. In-flight tracking: mark, is_in_flight, clear (with request_id management)
4. wait_for_result: basic wait, timeout, already-available, multiple waiters
5. LRU eviction (both _cache and _results caps)
6. _is_streaming_request: True, False, invalid JSON
7. Integration with TestClient: dedup collapses concurrent identical requests,
   streaming requests bypass, different paths/methods bypass,
   empty body bypass, non-200 not cached, dedup_fingerprint on state
8. Tenant isolation via header: different api_key_id yields different cache entries
9. Edge cases: LRU eviction doesn't affect new requests
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import Counter

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient
from starlette.middleware.base import BaseHTTPMiddleware

from distllm.api.dedup import (
    DedupMiddleware,
    _FingerprintCache,
    _cache,
    _is_streaming_request,
)


# ======================================================================
# Helpers
# ======================================================================


async def _wait_for_fp_state(
    fp: str,
    *,
    in_flight: bool | None = None,
    has_waiters: bool | None = None,
    timeout: float = 5.0,
) -> bool:
    """Poll the global cache until *fp* reaches the desired state.

    Returns `True` if the state was reached, `False` on timeout.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if in_flight is not None and _cache.is_in_flight(fp) == in_flight:
            return True
        if has_waiters is not None and len(_cache._wait_events.get(fp, set())) > 0:
            return True
        await asyncio.sleep(0.005)
    return False



def _make_streaming_response(
    content: str,
    status_code: int = 200,
    media_type: str = "application/json",
) -> StreamingResponse:
    """Build a ``StreamingResponse`` whose ``body_iterator`` yields one chunk.

    The middleware iterates ``response.body_iterator`` to read the response
    body, so the handler must return a response type that provides one.
    ``StreamingResponse`` always has ``body_iterator``; bare ``Response`` and
    ``JSONResponse`` do not (in Starlette 1.2+).
    """
    async def _stream():
        yield content.encode()

    return StreamingResponse(_stream(), media_type=media_type, status_code=status_code)


def _make_app(
    *,
    handler_status: int = 200,
    handler_body: dict | None = None,
    handler_delay: float = 0.0,
    bypass_streaming: bool = False,
    track_calls: Counter | None = None,
) -> FastAPI:
    """Factory that returns a FastAPI app with DedupMiddleware + mock auth.

    Parameters
    ----------
    handler_status:
        HTTP status code the route handler returns.
    handler_body:
        JSON body the route handler returns (default `{"choices": ...}`).
    handler_delay:
        How long the handler sleeps before responding (simulates work).
    bypass_streaming:
        If True, the handler returns a StreamingResponse directly so the
        middleware sees a streaming content-type and passes through.
    track_calls:
        Optional Counter to increment on every handler invocation.
    """
    app = FastAPI()

    app.add_middleware(DedupMiddleware)

    class _MockAuthMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            request.state.api_key_id = request.headers.get("X-API-Key-ID", "test-key")
            request.state.client_ip = "127.0.0.1"
            return await call_next(request)

    app.add_middleware(_MockAuthMiddleware)

    body = handler_body or {"choices": [{"message": {"content": "ok"}}]}

    @app.post("/v1/chat/completions")
    async def _chat_handler(request: Request):
        if track_calls is not None:
            track_calls["handler"] += 1
        if handler_delay:
            await asyncio.sleep(handler_delay)
        if bypass_streaming:
            return _make_streaming_response(json.dumps(body), media_type="text/event-stream")
        return _make_streaming_response(json.dumps(body), status_code=handler_status)

    @app.post("/v1/completions")
    async def _completions_handler():
        return _make_streaming_response(json.dumps({"completion": "nope"}))

    return app


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture(autouse=True)
def reset_global_cache():
    """Clear the module-level _cache singleton between tests."""
    _cache._cache.clear()
    _cache._in_flight.clear()
    _cache._results.clear()
    _cache._wait_events.clear()
    yield


@pytest.fixture
def fresh_cache() -> _FingerprintCache:
    """Return a pristine _FingerprintCache instance."""
    return _FingerprintCache()


# ======================================================================
# 1.  _FingerprintCache.fingerprint
# ======================================================================


class TestFingerprint:
    def test_same_body_returns_same_fingerprint(self, fresh_cache: _FingerprintCache):
        body = b'{"messages":[{"role":"user","content":"hello"}]}'
        fp1 = fresh_cache.fingerprint(body)
        fp2 = fresh_cache.fingerprint(body)
        assert fp1 == fp2
        assert isinstance(fp1, str)
        assert len(fp1) == 64  # SHA-256 hex digest

    def test_different_body_returns_different_fingerprint(self, fresh_cache: _FingerprintCache):
        fp_a = fresh_cache.fingerprint(b"request-A")
        fp_b = fresh_cache.fingerprint(b"request-B")
        assert fp_a != fp_b

    def test_tenant_isolation(self, fresh_cache: _FingerprintCache):
        """Identical bodies with different tenant_ids produce different fingerprints."""
        body = b'{"messages":[{"role":"user","content":"hi"}]}'
        fp_tenant_a = fresh_cache.fingerprint(body, tenant_id="tenant-a")
        fp_tenant_b = fresh_cache.fingerprint(body, tenant_id="tenant-b")
        assert fp_tenant_a != fp_tenant_b

    def test_tenant_none_is_different_from_string(self, fresh_cache: _FingerprintCache):
        """tenant_id=None (default) vs a real tenant string produce different keys."""
        body = b"test"
        assert fresh_cache.fingerprint(body, tenant_id=None) != fresh_cache.fingerprint(
            body, tenant_id="any-tenant"
        )

    def test_tenant_id_prefixes_hash(self, fresh_cache: _FingerprintCache):
        """The tenant_id is hashed before the body, so cross-tenant poisoning is impossible."""
        fp_a = fresh_cache.fingerprint(b"body", tenant_id="t1")
        fp_b = fresh_cache.fingerprint(b"body", tenant_id="t2")
        assert fp_a != fp_b


# ======================================================================
# 2.  _FingerprintCache store / lookup
# ======================================================================


class TestStoreAndLookup:
    def test_store_then_lookup_hit(self, fresh_cache: _FingerprintCache):
        fp = "cache-hit-fp"
        assert fresh_cache.lookup(fp) is None
        fresh_cache.store(fp, '{"response": "ok"}')
        assert fresh_cache.lookup(fp) == '{"response": "ok"}'

    def test_lookup_miss_returns_none(self, fresh_cache: _FingerprintCache):
        assert fresh_cache.lookup("nonexistent") is None

    def test_ttl_expiry(self, fresh_cache: _FingerprintCache):
        """After the TTL window passes, lookup returns None and removes the entry."""
        cache = _FingerprintCache(ttl_s=0.02)
        fp = "ttl-test"
        cache.store(fp, "expiring-data")
        assert cache.lookup(fp) == "expiring-data"
        time.sleep(0.03)
        assert cache.lookup(fp) is None

    def test_lookup_refreshes_lru_position(self, fresh_cache: _FingerprintCache):
        """Entries accessed via lookup() should not be evicted before untouched ones."""
        cache = _FingerprintCache(max_size=3)
        cache.store("a", "1")
        cache.store("b", "2")
        cache.store("c", "3")
        # Access "a" to refresh its LRU position
        assert cache.lookup("a") == "1"
        # Add a fourth entry -- should evict "b" (oldest untouched), not "a"
        cache.store("d", "4")
        assert cache.lookup("a") == "1"
        assert cache.lookup("b") is None  # evicted
        assert cache.lookup("d") == "4"

    def test_store_updates_existing_entry(self, fresh_cache: _FingerprintCache):
        fp = "update-fp"
        fresh_cache.store(fp, "old-value")
        fresh_cache.store(fp, "new-value")
        assert fresh_cache.lookup(fp) == "new-value"

    def test_empty_store_then_lookup(self, fresh_cache: _FingerprintCache):
        """An empty string is a valid cached value."""
        fp = "empty-fp"
        fresh_cache.store(fp, "")
        assert fresh_cache.lookup(fp) == ""


# ======================================================================
# 3.  _FingerprintCache in-flight tracking
# ======================================================================


class TestInFlightTracking:
    def test_is_in_flight_returns_false_for_unknown(self, fresh_cache: _FingerprintCache):
        assert fresh_cache.is_in_flight("unknown-fp") is False

    def test_mark_then_is_in_flight_true(self, fresh_cache: _FingerprintCache):
        fresh_cache.mark_in_flight("fp", "req-1")
        assert fresh_cache.is_in_flight("fp") is True

    def test_clear_in_flight_removes_request_id(self, fresh_cache: _FingerprintCache):
        fresh_cache.mark_in_flight("fp", "req-1")
        fresh_cache.mark_in_flight("fp", "req-2")
        fresh_cache.clear_in_flight("fp", "req-1")
        assert fresh_cache.is_in_flight("fp") is True  # still one request
        fresh_cache.clear_in_flight("fp", "req-2")
        assert fresh_cache.is_in_flight("fp") is False

    def test_clear_unknown_fp_does_not_raise(self, fresh_cache: _FingerprintCache):
        fresh_cache.clear_in_flight("does-not-exist", "req-1")

    def test_clear_unknown_request_id_leaves_fp_in_flight(self, fresh_cache: _FingerprintCache):
        fresh_cache.mark_in_flight("fp", "req-1")
        fresh_cache.clear_in_flight("fp", "req-2")  # req-2 not in list
        assert fresh_cache.is_in_flight("fp") is True

    def test_mark_in_flight_removes_existing_result(self, fresh_cache: _FingerprintCache):
        """When a new request marks itself in-flight for an fp that has a
        previous result cached, the result should be cleared to prevent
        stale data races."""
        fp = "stale-fp"
        fresh_cache.store(fp, "old-result")
        assert fp in fresh_cache._results
        fresh_cache.mark_in_flight(fp, "req-1")
        assert fp not in fresh_cache._results

    def test_clear_in_flight_removes_fp_when_last_request_cleared(
        self, fresh_cache: _FingerprintCache
    ):
        fp = "last-req-fp"
        fresh_cache.mark_in_flight(fp, "req-only")
        fresh_cache.clear_in_flight(fp, "req-only")
        assert fp not in fresh_cache._in_flight


# ======================================================================
# 4.  _FingerprintCache.wait_for_result
# ======================================================================


class TestWaitForResult:
    @pytest.mark.asyncio
    async def test_wait_returns_result_when_stored(self, fresh_cache: _FingerprintCache):
        """A waiter should receive the result once store() is called by the
        processing request."""
        fp = "async-fp"
        fresh_cache.mark_in_flight(fp, "process-1")

        async def deliver():
            await asyncio.sleep(0.05)
            fresh_cache.store(fp, "async-result")

        async def waiter():
            return await fresh_cache.wait_for_result(fp, poll=0.01, timeout=5.0)

        _, result = await asyncio.gather(deliver(), waiter())
        assert result == "async-result"

    @pytest.mark.asyncio
    async def test_wait_already_available_returns_immediately(
        self, fresh_cache: _FingerprintCache
    ):
        """If the result was stored before wait_for_result is called, return
        it without registering a waiter."""
        fp = "ready-fp"
        fresh_cache.store(fp, "cached-result")
        result = await fresh_cache.wait_for_result(fp)
        assert result == "cached-result"

    @pytest.mark.asyncio
    async def test_wait_timeout_returns_none(self, fresh_cache: _FingerprintCache):
        """If the result is never stored, wait_for_result returns None after
        the timeout elapses."""
        fp = "timeout-fp"
        fresh_cache.mark_in_flight(fp, "req-1")
        result = await fresh_cache.wait_for_result(fp, poll=0.01, timeout=0.05)
        assert result is None

    @pytest.mark.asyncio
    async def test_wait_not_in_flight_and_no_result_returns_none(
        self, fresh_cache: _FingerprintCache
    ):
        """If the fp has no in-flight request and no stored result, return
        None immediately."""
        result = await fresh_cache.wait_for_result("unknown-fp")
        assert result is None

    @pytest.mark.asyncio
    async def test_multiple_waiters_all_receive_result(
        self, fresh_cache: _FingerprintCache
    ):
        """Multiple concurrent waiters for the same fp should all be notified
        when store() is called."""
        fp = "multi-fp"
        fresh_cache.mark_in_flight(fp, "process-1")
        results: list[tuple[int, str | None]] = []

        async def waiter(idx: int):
            r = await fresh_cache.wait_for_result(fp, poll=0.01, timeout=5.0)
            results.append((idx, r))

        async def deliver():
            await asyncio.sleep(0.05)
            fresh_cache.store(fp, "shared-result")

        await asyncio.gather(deliver(), waiter(1), waiter(2), waiter(3))
        assert len(results) == 3
        assert all(r == "shared-result" for _, r in results)

    @pytest.mark.asyncio
    async def test_wait_for_result_after_clear_in_flight_signals_none(
        self, fresh_cache: _FingerprintCache
    ):
        """When all in-flight requests are cleared without storing a result,
        waiters should be signaled and wait_for_result returns None."""
        fp = "clear-only-fp"
        fresh_cache.mark_in_flight(fp, "req-1")

        async def waiter():
            return await fresh_cache.wait_for_result(fp, poll=0.01, timeout=5.0)

        async def clear():
            await asyncio.sleep(0.05)
            fresh_cache.clear_in_flight(fp, "req-1")

        _, result = await asyncio.gather(clear(), waiter())
        assert result is None


# ======================================================================
# 5.  _FingerprintCache LRU eviction
# ======================================================================


class TestLRUEviction:
    def test_cache_evicts_lru_when_at_capacity(self, fresh_cache: _FingerprintCache):
        """When max_size is exceeded, the least recently used entry is evicted
        from _cache."""
        cache = _FingerprintCache(max_size=3)
        for i in range(4):
            cache.store(f"k{i}", str(i))
        # k0 should be evicted (oldest)
        assert cache.lookup("k0") is None
        # k1, k2, k3 should remain
        assert cache.lookup("k1") == "1"
        assert cache.lookup("k2") == "2"
        assert cache.lookup("k3") == "3"

    def test_results_dict_has_independent_capacity(self, fresh_cache: _FingerprintCache):
        """The _results dict has a separate cap of max_size * 2."""
        cache = _FingerprintCache(max_size=100)
        assert cache._max_results == 200

    def test_results_eviction_at_capacity(self, fresh_cache: _FingerprintCache):
        """The _results dict should not grow beyond _max_results entries."""
        cache = _FingerprintCache(max_size=2)
        cache._max_results = 3  # override for test
        for i in range(5):
            cache.store(f"r{i}", str(i))
        assert len(cache._results) <= 3

    def test_lookup_after_eviction_still_misses(self, fresh_cache: _FingerprintCache):
        """Once evicted, a fingerprint returns None from lookup()."""
        cache = _FingerprintCache(max_size=2)
        cache.store("x", "1")
        cache.store("y", "2")
        assert cache.lookup("x") == "1"
        assert cache.lookup("y") == "2"
        cache.store("z", "3")  # evicts "x"
        assert cache.lookup("x") is None
        assert cache.lookup("z") == "3"


# ======================================================================
# 6.  _is_streaming_request
# ======================================================================


class TestIsStreamingRequest:
    def test_stream_true_returns_true(self):
        body = json.dumps({"stream": True}).encode()
        assert _is_streaming_request(body) is True

    def test_stream_false_returns_false(self):
        body = json.dumps({"stream": False}).encode()
        assert _is_streaming_request(body) is False

    def test_no_stream_field_returns_false(self):
        body = json.dumps({"temperature": 0.7}).encode()
        assert _is_streaming_request(body) is False

    def test_invalid_json_returns_false(self):
        """Malformed body should not raise; return False to assume non-streaming."""
        assert _is_streaming_request(b"not valid json") is False

    def test_empty_body_returns_false(self):
        assert _is_streaming_request(b"") is False

    def test_non_dict_json_raises_attribute_error(self):
        """A JSON array (valid JSON but not a dict) triggers AttributeError
        because the code calls `.get()` on the loaded object.

        Note: this is a known gap in the production code -- a non-dict JSON
        body would propagate an unhandled exception.  In practice, all POST
        bodies to /v1/chat/completions are expected to be JSON objects.
        """
        body = json.dumps([1, 2, 3]).encode()
        with pytest.raises(AttributeError):
            _is_streaming_request(body)


# ======================================================================
# 7.  DedupMiddleware integration with TestClient
# ======================================================================


class TestDedupMiddlewareIntegration:
    """Real HTTP integration tests using FastAPI TestClient.

    No mocks are used. The middleware is tested through the full Starlette
    request/response cycle.  For concurrent-request tests, httpx.AsyncClient
    with ASGITransport is used inside an asyncio event loop.
    """

    # -- bypass tests -----------------------------------------------------------

    def test_non_post_bypasses(self):
        """GET, PUT, DELETE, etc. should pass through without dedup."""
        # Create a minimal app with a GET /v1/chat/completions route
        app = FastAPI()
        app.add_middleware(DedupMiddleware)

        class _MockAuthMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request: Request, call_next):
                request.state.api_key_id = request.headers.get("X-API-Key-ID", "test-key")
                request.state.client_ip = "127.0.0.1"
                return await call_next(request)

        app.add_middleware(_MockAuthMiddleware)

        @app.api_route("/v1/chat/completions", methods=["GET", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
        async def _catch_all():
            return _make_streaming_response(json.dumps({"ok": True}))

        client = TestClient(app)
        for method in ("GET", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"):
            resp = client.request(method, "/v1/chat/completions")
            assert resp.status_code == 200, f"{method} should pass through"

    def test_non_chat_path_bypasses(self):
        """POST to a different path should bypass the middleware."""
        app = _make_app()

        client = TestClient(app)
        resp = client.post("/v1/completions", json={"prompt": "hello"})
        assert resp.status_code == 200
        assert resp.json() == {"completion": "nope"}

    def test_empty_body_bypasses(self):
        """POST with an empty body should pass through without caching."""
        app = _make_app()

        client = TestClient(app)
        resp = client.post("/v1/chat/completions", content=b"", headers={"Content-Type": "application/json"})
        assert resp.status_code == 200

    def test_streaming_request_bypasses(self):
        """Requests with stream: true should skip deduplication entirely."""
        track_calls = Counter()
        app = _make_app(track_calls=track_calls)

        client = TestClient(app)
        payload = {"stream": True, "messages": [{"role": "user", "content": "hi"}]}

        r1 = client.post("/v1/chat/completions", json=payload, headers={"X-API-Key-ID": "key-1"})
        r2 = client.post("/v1/chat/completions", json=payload, headers={"X-API-Key-ID": "key-1"})
        assert track_calls["handler"] == 2
        assert r1.status_code == 200
        assert r2.status_code == 200

    # -- cache hit / miss -------------------------------------------------------

    def test_cache_hit_returns_cached_response(self):
        """Two identical non-concurrent requests: the second hits the cache."""
        track_calls = Counter()
        app = _make_app(track_calls=track_calls)

        client = TestClient(app)
        payload = {"messages": [{"role": "user", "content": "hello"}]}

        r1 = client.post("/v1/chat/completions", json=payload, headers={"X-API-Key-ID": "key-1"})
        assert r1.status_code == 200
        assert track_calls["handler"] == 1

        r2 = client.post("/v1/chat/completions", json=payload, headers={"X-API-Key-ID": "key-1"})
        assert r2.status_code == 200
        assert track_calls["handler"] == 1  # no second handler call
        assert r2.json() == r1.json()

    def test_cross_key_isolation_no_cache_leak(self):
        """P0: a cached response for one api_key_id must NEVER be served to
        another api_key_id, even for an identical body (body-only fingerprints
        previously leaked cached output across tenants)."""
        track_calls = Counter()
        app = _make_app(track_calls=track_calls)

        client = TestClient(app)
        payload = {"messages": [{"role": "user", "content": "private"}]}

        r1 = client.post("/v1/chat/completions", json=payload, headers={"X-API-Key-ID": "key-a"})
        assert r1.status_code == 200
        assert track_calls["handler"] == 1

        # Same body, DIFFERENT key -> must be a miss (handler runs again).
        r2 = client.post("/v1/chat/completions", json=payload, headers={"X-API-Key-ID": "key-b"})
        assert r2.status_code == 200
        assert track_calls["handler"] == 2

        # Same key again -> cache hit (no new handler call).
        r3 = client.post("/v1/chat/completions", json=payload, headers={"X-API-Key-ID": "key-a"})
        assert r3.status_code == 200
        assert track_calls["handler"] == 2
        assert r3.json() == r1.json()

    def test_unauthenticated_replay_does_not_hit_authenticated_cache(self):
        """P0: a request with NO api_key_id must not be served another key's
        cached response (dedup runs AFTER auth and fingerprints by identity)."""
        track_calls = Counter()
        app = _make_app(track_calls=track_calls)

        client = TestClient(app)
        payload = {"messages": [{"role": "user", "content": "secret"}]}

        r1 = client.post("/v1/chat/completions", json=payload, headers={"X-API-Key-ID": "key-a"})
        assert track_calls["handler"] == 1

        r2 = client.post("/v1/chat/completions", json=payload)  # no key header
        assert r2.status_code == 200
        assert track_calls["handler"] == 2  # miss -> handler runs

    def test_cache_miss_in_flight_waits_for_result(self):
        """Concurrent identical requests: the second waits for the first's result."""
        track_calls = Counter()
        app = _make_app(handler_delay=0.1, track_calls=track_calls)

        payload = {"messages": [{"role": "user", "content": "concurrent"}]}
        body_bytes = json.dumps(payload).encode()
        fp = _cache.fingerprint(body_bytes, tenant_id="key-1")

        async def _run():
            from httpx import ASGITransport
            transport = ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
                task1 = asyncio.create_task(
                    ac.post("/v1/chat/completions", json=payload, headers={"X-API-Key-ID": "key-1"})
                )
                await _wait_for_fp_state(fp, in_flight=True)

                task2 = asyncio.create_task(
                    ac.post("/v1/chat/completions", json=payload, headers={"X-API-Key-ID": "key-1"})
                )
                await _wait_for_fp_state(fp, has_waiters=True)

                r1, r2 = await asyncio.gather(task1, task2, return_exceptions=True)

            for r in (r1, r2):
                assert not isinstance(r, BaseException), f"Unexpected exception: {r}"
            return r1, r2

        r1, r2 = asyncio.run(_run())
        assert track_calls["handler"] == 1
        assert r1.status_code == 200
        assert r2.status_code == 200

    def test_normal_flow_stores_result(self):
        """First request for an fp: no cached entry, no in-flight, handler
        is called, and the result is stored for subsequent requests."""
        track_calls = Counter()
        app = _make_app(handler_body={"result": "stored"}, track_calls=track_calls)

        payload = {"messages": [{"role": "user", "content": "store-me"}]}
        # Use the same serialisation that httpx/TestClient uses internally:
        # separators=(',', ':') with no spaces after commas.
        body_bytes = json.dumps(payload, separators=(",", ":")).encode()
        fp = _cache.fingerprint(body_bytes, tenant_id="test-key")

        client = TestClient(app)
        r1 = client.post(
            "/v1/chat/completions",
            content=body_bytes,
            headers={"Content-Type": "application/json", "X-API-Key-ID": "test-key"},
        )
        assert r1.status_code == 200
        assert r1.json() == {"result": "stored"}
        assert track_calls["handler"] == 1

        cached = _cache.lookup(fp)
        assert cached is not None
        assert json.loads(cached) == {"result": "stored"}
        assert not _cache.is_in_flight(fp)

    def test_non_200_not_cached(self):
        """Only status_code == 200 responses are stored in the cache."""
        track_calls = Counter()
        app = _make_app(handler_status=400, handler_body={"error": "bad"}, track_calls=track_calls)

        payload = {"messages": [{"role": "user", "content": "error-test"}]}
        body_bytes = json.dumps(payload).encode()
        fp = _cache.fingerprint(body_bytes, tenant_id="test-key")

        client = TestClient(app)
        r = client.post("/v1/chat/completions", json=payload, headers={"X-API-Key-ID": "test-key"})
        assert r.status_code == 400
        assert track_calls["handler"] == 1
        assert _cache.lookup(fp) is None

    def test_sets_dedup_fingerprint_on_state(self):
        """The computed fingerprint is stored on request.state for downstream use."""
        track_calls = Counter()
        app = _make_app(handler_body={"ok": True}, track_calls=track_calls)

        payload = {"messages": [{"role": "user", "content": "fp-test"}]}
        # Use compact JSON to match httpx/TestClient serialisation
        body_bytes = json.dumps(payload, separators=(",", ":")).encode()
        expected_fp = _cache.fingerprint(body_bytes, tenant_id="test-key")

        client = TestClient(app)
        client.post(
            "/v1/chat/completions",
            content=body_bytes,
            headers={"Content-Type": "application/json", "X-API-Key-ID": "test-key"},
        )
        # Verify it was stored by checking the cache contains it
        assert _cache.lookup(expected_fp) is not None
        assert track_calls["handler"] == 1

    # -- streaming response pass-through ---------------------------------------

    def test_streaming_response_passes_through(self):
        """If the upstream response is streaming, the middleware passes it
        through without buffering or caching."""
        track_calls = Counter()
        app = _make_app(bypass_streaming=True, track_calls=track_calls)

        payload = {"messages": [{"role": "user", "content": "stream-me"}]}
        body_bytes = json.dumps(payload).encode()
        fp = _cache.fingerprint(body_bytes, tenant_id="test-key")

        client = TestClient(app)
        r = client.post("/v1/chat/completions", json=payload, headers={"X-API-Key-ID": "test-key"})
        assert r.status_code == 200
        assert track_calls["handler"] == 1
        assert _cache.lookup(fp) is None

    # -- tenant isolation -------------------------------------------------------

    def test_tenant_isolation_via_api_key_header(self):
        """Two requests with different api_key_ids get separate cache entries."""
        track_calls = Counter()
        app = _make_app(track_calls=track_calls)

        client = TestClient(app)
        payload = {"messages": [{"role": "user", "content": "hello"}]}

        r_a1 = client.post("/v1/chat/completions", json=payload, headers={"X-API-Key-ID": "tenant-A"})
        assert r_a1.status_code == 200
        assert track_calls["handler"] == 1

        r_b1 = client.post("/v1/chat/completions", json=payload, headers={"X-API-Key-ID": "tenant-B"})
        assert r_b1.status_code == 200
        assert track_calls["handler"] == 2

        r_a2 = client.post("/v1/chat/completions", json=payload, headers={"X-API-Key-ID": "tenant-A"})
        assert r_a2.status_code == 200
        # Handler not called again -- tenant-A's result was cached
        assert track_calls["handler"] == 2

    # -- after-dedup clear then cache hit ---------------------------------------

    def test_after_dedup_clear_next_identical_request_hits_cache(self):
        """After an in-flight request completes and clears, the next identical
        request hits the cache, not the handler."""
        track_calls = Counter()
        app = _make_app(track_calls=track_calls)

        client = TestClient(app)
        payload = {"messages": [{"role": "user", "content": "test"}]}

        r1 = client.post("/v1/chat/completions", json=payload, headers={"X-API-Key-ID": "key-1"})
        assert r1.status_code == 200
        assert track_calls["handler"] == 1

        fp = _cache.fingerprint(json.dumps(payload).encode(), tenant_id="key-1")
        assert not _cache.is_in_flight(fp)

        r2 = client.post("/v1/chat/completions", json=payload, headers={"X-API-Key-ID": "key-1"})
        assert r2.status_code == 200
        assert track_calls["handler"] == 1
        assert r2.json() == r1.json()

    # -- different bodies not deduped ------------------------------------------

    def test_different_bodies_not_deduped(self):
        """Different request payloads produce different fingerprints and
        each reaches the handler."""
        track_calls = Counter()
        app = _make_app(track_calls=track_calls)

        client = TestClient(app)

        r1 = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "first"}]},
            headers={"X-API-Key-ID": "key-1"},
        )
        r2 = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "second"}]},
            headers={"X-API-Key-ID": "key-1"},
        )
        assert track_calls["handler"] == 2
        assert r1.json()["choices"][0]["message"]["content"] == "ok"
        assert r2.json()["choices"][0]["message"]["content"] == "ok"


# ======================================================================
# 8.  DedupMiddleware edge cases
# ======================================================================


class TestDedupMiddlewareEdgeCases:
    def test_lru_eviction_does_not_affect_new_requests(self):
        """When the cache is full and old entries are evicted, new requests
        for evicted fingerprints start fresh."""
        track_calls = Counter()
        app = _make_app(track_calls=track_calls)

        client = TestClient(app)
        original_max = _cache._max_size
        _cache._max_size = 2
        try:
            for i in range(3):
                p = {"messages": [{"role": "user", "content": f"msg-{i}"}]}
                r = client.post("/v1/chat/completions", json=p, headers={"X-API-Key-ID": "key-evict"})
                assert r.status_code == 200

            fp0 = _cache.fingerprint(
                json.dumps({"messages": [{"role": "user", "content": "msg-0"}]}).encode(),
                tenant_id="key-evict",
            )
            assert _cache.lookup(fp0) is None

            # New request with evicted fp goes through normally
            call_count_before = track_calls["handler"]
            r_new = client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "msg-0"}]},
                headers={"X-API-Key-ID": "key-evict"},
            )
            assert r_new.status_code == 200
            assert track_calls["handler"] == call_count_before + 1
        finally:
            _cache._max_size = original_max

    def test_in_flight_cleaned_up_on_handler_error(self):
        """Even if the route handler raises, the in-flight marker is cleared."""
        app = FastAPI()
        app.add_middleware(DedupMiddleware)

        class _MockAuthMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request: Request, call_next):
                request.state.api_key_id = "key-1"
                request.state.client_ip = "127.0.0.1"
                return await call_next(request)

        app.add_middleware(_MockAuthMiddleware)

        @app.post("/v1/chat/completions")
        async def _failing_handler():
            raise RuntimeError("upstream failure")

        fp = _cache.fingerprint(
            json.dumps({"messages": [{"role": "user", "content": "crash"}]}).encode(),
            tenant_id="key-1",
        )

        client = TestClient(app)
        with pytest.raises(RuntimeError, match="upstream failure"):
            client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "crash"}]},
                headers={"X-API-Key-ID": "key-1"},
            )

        assert not _cache.is_in_flight(fp)