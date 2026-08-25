"""SEC-A9 regression tests: unauthenticated JWKS fan-out on invalid tokens.

Pre-fix behavior (verified empirically before the fix): every bearer token
that failed local-JWT decoding made ``OIDCHandler.validate_token`` issue a
fresh ``httpx.get(jwks_url)`` -- no cache, no negative caching, no
concurrency cap.  Ten garbage tokens produced ten outbound HTTPS fetches to
the IdP (latency DoS on the coordinator + request flood that can get the
deployment banned by the IdP).  ``SsoMiddleware.dispatch`` amplified this by
iterating *every* registered provider per request.

Post-fix guarantees asserted here:

1. Repeated bad-token validations within the cache TTL make **zero** new
   outbound fetches (positive cache).
2. A failing JWKS endpoint is negatively cached -- retries are bounded, not
   per-request.
3. Tokens carrying attacker-crafted ``jku`` headers / foreign ``iss`` claims
   can never steer the fetch target; only the operator-configured /
   discovery-derived endpoint is ever requested.
4. Concurrent validations share a single in-flight fetch (single-flight).
5. Cache expiry genuinely refetches (the cache is not sticky forever).
6. End-to-end through :class:`SsoMiddleware`: N bad-token requests produce
   exactly one JWKS fetch, not N.
"""

from __future__ import annotations

import base64
import json
import threading
import time

import pytest

pytest.importorskip("jwt")
pytest.importorskip("httpx")

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from distllm.api.sso_auth import (
    DEFAULT_JWKS_CACHE_TTL_S,
    JWKS_NEGATIVE_CACHE_TTL_S,
    OIDCHandler,
)
from distllm.api.sso_middleware import setup_sso


TRUSTED_JWKS_URL = "https://idp.example.com/.well-known/jwks"
EVIL_JWKS_URL = "https://evil.example.attacker/jwks"


# ── Helpers ────────────────────────────────────────────────────────────────────


class FetchCounter:
    """Replaces ``httpx.get`` and records every requested URL."""

    def __init__(self, status_code: int = 200, body: dict | None = None,
                 gate: threading.Event | None = None):
        self.urls: list[str] = []
        self.status_code = status_code
        self.body = body if body is not None else {"keys": []}
        self.gate = gate
        self._lock = threading.Lock()

    def install(self, monkeypatch) -> None:
        counter = self

        def _fake_get(url, **kwargs):
            with counter._lock:
                counter.urls.append(str(url))
            if counter.gate is not None:
                # Simulate a slow IdP: hold the request open until released.
                counter.gate.wait(timeout=10.0)

            class _R:
                status_code = counter.status_code

                def json(self_inner):
                    return dict(counter.body)

            return _R()

        import httpx
        monkeypatch.setattr(httpx, "get", _fake_get)

    @property
    def count(self) -> int:
        return len(self.urls)


def _make_handler(jwks_url: str = TRUSTED_JWKS_URL,
                  ttl_s: float = DEFAULT_JWKS_CACHE_TTL_S) -> OIDCHandler:
    """Build an OIDCHandler without running network discovery (__new__
    pattern, mirroring tests/api/test_sso_auth.py conventions)."""
    h = object.__new__(OIDCHandler)
    h._client_id = "cid"
    h._client_secret = "secret"
    h._authority = "https://idp.example.com"
    h._callback_url = "https://app.example.com/cb"
    h._jwks_url = jwks_url
    h._discovered_jwks_url = ""
    h._state_store = {}
    h._nonce_store = {}
    h._nonce_ttl = 600.0
    h._jwks_cache_ttl_s = ttl_s
    h._jwks_cache_lock = threading.Lock()
    h._jwks_cache = {}
    h._jwks_inflight = {}
    h._jwks_last_error = {}
    return h


def _b64u(obj: dict) -> str:
    raw = json.dumps(obj, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _crafted_token(jku: str | None = None, iss: str | None = None) -> str:
    """A well-formed-looking RS256 JWT whose header carries an attacker-chosen
    ``jku`` and whose payload carries a foreign ``iss``."""
    header = {"alg": "RS256", "typ": "JWT", "kid": "k1"}
    if jku is not None:
        header["jku"] = jku
    payload = {"sub": "attacker", "aud": "cid"}
    if iss is not None:
        payload["iss"] = iss
    return f"{_b64u(header)}.{_b64u(payload)}.c2ln"


# ── 1. Positive cache: repeated failures do not refetch ───────────────────────


def test_repeated_bad_tokens_trigger_exactly_one_fetch(monkeypatch):
    """Task 1/2a: the pre-fix behavior was N bad tokens -> N JWKS fetches.
    Post-fix, N validations within the TTL produce exactly one fetch."""
    counter = FetchCounter()
    counter.install(monkeypatch)
    h = _make_handler()

    for _ in range(10):
        assert h.validate_token("garbage.token.value") is None

    assert counter.count == 1, (
        f"expected 1 outbound JWKS fetch for 10 bad tokens, got {counter.count}"
    )


def test_second_bad_token_within_ttl_zero_new_fetches(monkeypatch):
    """Task 3: second (and later) bad-token requests within the TTL make zero
    new outbound fetches."""
    counter = FetchCounter()
    counter.install(monkeypatch)
    h = _make_handler()

    assert h.validate_token("bad.one") is None
    baseline = counter.count
    assert baseline == 1

    for i in range(20):
        assert h.validate_token(f"bad.{i}") is None
    assert counter.count - baseline == 0, "cached JWKS must serve repeat hits"


def test_successful_validation_is_cached_too(monkeypatch):
    """Even a *valid* token path caches: two different kids validated back to
    back hit the network once."""
    counter = FetchCounter()
    counter.install(monkeypatch)
    h = _make_handler()

    t1 = f"{_b64u({'alg': 'RS256', 'typ': 'JWT', 'kid': 'a'})}.{_b64u({'sub': 'u'})}.x"
    t2 = f"{_b64u({'alg': 'RS256', 'typ': 'JWT', 'kid': 'b'})}.{_b64u({'sub': 'v'})}.y"
    assert h.validate_token(t1) is None  # unknown kid 'a'
    assert h.validate_token(t2) is None  # unknown kid 'b'
    assert counter.count == 1


# ── 2. Negative cache: failing IdP is not hammered ────────────────────────────


def test_failed_jwks_fetch_negatively_cached(monkeypatch):
    """A 500 from the IdP must not turn into a per-request retry storm: the
    failure is remembered for JWKS_NEGATIVE_CACHE_TTL_S."""
    counter = FetchCounter(status_code=500)
    counter.install(monkeypatch)
    h = _make_handler()

    for _ in range(8):
        assert h.validate_token("bad.token") is None

    assert counter.count == 1, (
        f"failing IdP should be fetched once (negative cache), got {counter.count}"
    )


def test_network_error_negatively_cached(monkeypatch):
    """Connection errors (IdP down / DNS black hole) are negatively cached."""
    import httpx

    calls: list[str] = []

    def _boom(url, **kwargs):
        calls.append(str(url))
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "get", _boom)
    h = _make_handler()

    for _ in range(8):
        assert h.validate_token("bad.token") is None
    assert len(calls) == 1


def test_negative_cache_expires_and_retries(monkeypatch):
    """After the negative-cache window the handler tries again (self-healing),
    and success repopulates the positive cache."""
    counter = FetchCounter(status_code=503)
    counter.install(monkeypatch)
    h = _make_handler(ttl_s=DEFAULT_JWKS_CACHE_TTL_S)

    assert h.validate_token("bad.token") is None
    assert counter.count == 1

    # Age out the negative entry.
    with h._jwks_cache_lock:
        url, (ts, err) = next(iter(h._jwks_last_error.items()))
        h._jwks_last_error[url] = (time.time() - 1.0, err)

    counter.status_code = 200
    assert h.validate_token(f"{_b64u({'alg': 'RS256', 'kid': 'k9'})}.{_b64u({'sub': 'u'})}.x") is None
    assert counter.count == 2

    # Now positively cached again.
    assert h.validate_token("bad.token.again") is None
    assert counter.count == 2


# ── 3. Token-controlled URLs are never fetched ────────────────────────────────


def test_crafted_jku_and_foreign_iss_never_steered(monkeypatch):
    """Task 2b/3: a token suggesting ``jku`` (or any URL material) must never
    influence where JWKS is fetched from.  Only the trusted configured
    endpoint is ever contacted."""
    counter = FetchCounter()
    counter.install(monkeypatch)
    h = _make_handler()

    attacks = [
        _crafted_token(jku=EVIL_JWKS_URL, iss="https://evil.example.attacker"),
        _crafted_token(jku="/etc/passwd"),
        _crafted_token(jku="file:///etc/passwd"),
        _crafted_token(jku="http://127.0.0.1:8500/jwks", iss="http://169.254.169.254/"),
    ]
    for tok in attacks:
        assert h.validate_token(tok) is None

    assert EVIL_JWKS_URL not in counter.urls
    for url in counter.urls:
        assert url == TRUSTED_JWKS_URL, f"unexpected fetch target: {url}"
    assert counter.count <= 1  # and only the cached trusted endpoint


def test_unknown_kid_after_cache_does_not_refetch(monkeypatch):
    """Rotation probing (random kids) must not cause refresh storms."""
    counter = FetchCounter()
    counter.install(monkeypatch)
    h = _make_handler()

    for i in range(25):
        tok = f"{_b64u({'alg': 'RS256', 'kid': f'probe-{i}'})}.{_b64u({'sub': 'u'})}.x"
        assert h.validate_token(tok) is None
    assert counter.count == 1


# ── 4. Single-flight under concurrency ─────────────────────────────────────────


def test_single_flight_under_concurrency(monkeypatch):
    """Task 2c/3: concurrent validations share ONE in-flight fetch.  A slow
    IdP response must not trigger parallel duplicate requests from waiters."""
    gate = threading.Event()
    counter = FetchCounter(gate=gate)
    counter.install(monkeypatch)
    h = _make_handler()

    n_followers = 8
    results: list[str | None] = []
    results_lock = threading.Lock()

    def worker():
        res = h.validate_token("concurrent.bad.token")
        with results_lock:
            results.append(res)

    leader = threading.Thread(target=worker, daemon=True)
    leader.start()

    # Hold the leader inside the HTTP call until followers have queued up.
    deadline = time.time() + 10
    while counter.count == 0 and time.time() < deadline:
        time.sleep(0.01)
    assert counter.count == 1, "leader should be the only fetcher in flight"

    followers = [threading.Thread(target=worker, daemon=True) for _ in range(n_followers)]
    for t in followers:
        t.start()
    time.sleep(0.5)                   # give followers time to hit event.wait()
    assert counter.count == 1, "followers must NOT issue their own fetch"

    gate.set()                        # release the leader's HTTP response
    leader.join(timeout=10)
    for t in followers:
        t.join(timeout=10)

    assert counter.count == 1, (
        f"single-flight violated: {counter.count} fetches for {n_followers + 1} concurrent validations"
    )
    assert len(results) == n_followers + 1
    assert all(r is None for r in results)


def test_follower_gets_result_when_leader_succeeds(monkeypatch):
    """Followers waiting on an in-flight fetch receive the cached document
    (not an error) once the leader succeeds."""
    gate = threading.Event()
    counter = FetchCounter(gate=gate)
    counter.install(monkeypatch)
    h = _make_handler()

    follower_result: list[object] = []

    def follower():
        follower_result.append(h.validate_token("follower.token"))

    ft = threading.Thread(target=follower, daemon=True)

    leader_done = threading.Event()

    def leader():
        h.validate_token("leader.token")
        leader_done.set()

    lt = threading.Thread(target=leader, daemon=True)
    lt.start()

    deadline = time.time() + 10
    while counter.count == 0 and time.time() < deadline:
        time.sleep(0.01)
    ft.start()
    time.sleep(0.2)                   # follower parks in event.wait()
    assert counter.count == 1
    gate.set()
    lt.join(timeout=10)
    ft.join(timeout=10)
    assert leader_done.is_set()
    assert follower_result == [None]


def test_leader_failure_propagates_to_waiters_without_refetch(monkeypatch):
    """If the leader's fetch fails, parked followers fail fast from the
    negative cache instead of issuing duplicate requests."""
    import httpx

    gate = threading.Event()
    calls: list[str] = []
    state = {"failing": True}

    def _fake_get(url, **kwargs):
        calls.append(str(url))
        gate.wait(timeout=10.0)
        if state["failing"]:
            raise httpx.ConnectError("idp down")

        class _R:
            status_code = 200

            def json(self_inner):
                return {"keys": []}

        return _R()

    monkeypatch.setattr(httpx, "get", _fake_get)
    h = _make_handler()

    leader_err: list[object] = []
    follower_err: list[object] = []

    def leader():
        try:
            h._fetch_jwks_cached(TRUSTED_JWKS_URL)
            leader_err.append(None)
        except Exception as e:
            leader_err.append(e)

    lt = threading.Thread(target=leader, daemon=True)
    lt.start()
    deadline = time.time() + 10
    while not calls and time.time() < deadline:
        time.sleep(0.01)

    def follower():
        try:
            h._fetch_jwks_cached(TRUSTED_JWKS_URL)
            follower_err.append(None)
        except Exception as e:
            follower_err.append(e)

    ft = threading.Thread(target=follower, daemon=True)
    ft.start()
    time.sleep(0.2)
    gate.set()
    lt.join(timeout=10)
    ft.join(timeout=10)

    assert isinstance(leader_err[0], Exception)
    assert isinstance(follower_err[0], Exception)
    assert len(calls) == 1, "waiter must not duplicate the failed fetch"


# ── 5. TTL expiry genuinely refetches ──────────────────────────────────────────


def test_expired_cache_entry_refetches(monkeypatch):
    """Positive control: the cache is not sticky — entries older than the TTL
    trigger exactly one refresh."""
    counter = FetchCounter()
    counter.install(monkeypatch)
    h = _make_handler(ttl_s=DEFAULT_JWKS_CACHE_TTL_S)

    assert h.validate_token("warmup") is None
    assert counter.count == 1

    with h._jwks_cache_lock:
        url, (_, data) = next(iter(h._jwks_cache.items()))
        h._jwks_cache[url] = (time.time() - 1.0, data)  # force-expire

    assert h.validate_token("after.ttl") is None
    assert counter.count == 2

    # Refreshed entry serves further hits.
    assert h.validate_token("again") is None
    assert counter.count == 2


# ── 6. End-to-end through SsoMiddleware ────────────────────────────────────────


def _minimal_app_with_provider(monkeypatch) -> tuple[FastAPI, FetchCounter]:
    """App with the SSO middleware and one registered OIDC provider whose
    JWKS traffic is captured by FetchCounter."""
    counter = FetchCounter()
    counter.install(monkeypatch)
    handler = _make_handler()

    app = FastAPI()

    @app.get("/echo")
    async def echo(request: Request):
        return {"auth_method": getattr(request.state, "auth_method", None)}

    setup_sso(app)
    return app, counter


def test_middleware_bad_token_requests_single_fetch_total(monkeypatch):
    """SEC-A9 headline scenario: N unauthenticated requests carrying garbage
    bearers cause ONE outbound JWKS fetch (pre-fix: N)."""
    app, counter = _minimal_app_with_provider(monkeypatch)

    with TestClient(app) as client:
        client.get("/echo")  # force middleware construction
        mw = getattr(app.state, "_sso_middleware", None)
        assert mw is not None
        mw.register_provider("corp", _make_handler())

        for _ in range(15):
            resp = client.get(
                "/echo", headers={"Authorization": "Bearer garbage.not.a.jwt"}
            )
            assert resp.status_code == 200  # falls through to API-key path

    assert counter.count == 1, (
        f"SsoMiddleware fanned out {counter.count} JWKS fetches for 15 bad "
        "tokens; expected exactly 1 (TTL cache)"
    )


def test_middleware_multiple_providers_still_bounded(monkeypatch):
    """With P providers, the per-TTL fetch bound is P (one each), not
    N_requests x P."""
    app, counter = _minimal_app_with_provider(monkeypatch)
    p = 3

    with TestClient(app) as client:
        client.get("/echo")
        mw = getattr(app.state, "_sso_middleware", None)
        assert mw is not None
        for i in range(p):
            mw.register_provider(
                f"prov{i}", _make_handler(f"https://idp{i}.example.com/jwks")
            )

        for _ in range(10):
            resp = client.get(
                "/echo", headers={"Authorization": "Bearer junk.value.here"}
            )
            assert resp.status_code == 200

    assert counter.count == p, (
        f"expected one fetch per provider ({p}), got {counter.count}"
    )
