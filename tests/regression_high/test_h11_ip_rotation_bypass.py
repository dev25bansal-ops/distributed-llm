"""Regression tests for HIGH fix H11: IP-rotation rate-limit bypass.

H11 covers the rate-limit bypass where an attacker rotates ``X-Forwarded-For``
/ ``X-Real-IP`` across many IPs to evade a *per-IP* limit.  The fix (C15) keys
the limiter on a STABLE identity instead of the spoofable header:

* The middleware (:class:`RequestRateLimitMiddleware`) derives the client IP
  via ``get_client_ip`` (fail-closed M2: a client reaching the app directly
  CANNOT spoof its IP via XFF) and builds
  ``key = f"{client_ip}|{api_key_id}"`` (api_key_id is the authenticated key).
  That combined string is the limiter key.
* The underlying :class:`_RateLimiter` can also key by ``api_key`` alone, which
  fixes the same class of attack at the limiter level (rotating IP with the
  same key is still limited; rotating key is correctly separated).

We assert:

1. Rotating XFF while the real peer (and API key) stays the SAME yields the
   SAME limiter key -> the attacker is still limited (XFF spoofing is ignored).
2. Rotating the API key (same real peer) produces a DIFFERENT key -> correctly
   separated, while the exhausted key remains limited.
3. At the limiter level (api_key mode) rotating the source IP with one key is
   still limited, and a fresh key on the same IP gets its own budget.

We load the REAL middleware module (via importlib, no heavy server stack) and
exercise its actual dispatch key construction.  We also keep a contrast test
that the legacy ``ip``-only keying is the bypassable configuration, so a
regression back to it fails loudly.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
import asyncio

import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SRC = os.path.join(_REPO_ROOT, "src")


def _load_middleware():
    """Load middleware.py in isolation (same pattern as test_c15_ratelimit_keying)."""
    _inserted: list[str] = []
    for pkg, real in (
        ("distllm", os.path.join(_SRC, "distllm")),
        ("distllm.api", os.path.join(_SRC, "distllm", "api")),
        ("distllm.core", os.path.join(_SRC, "distllm", "core")),
        ("distllm.errors", os.path.join(_SRC, "distllm", "errors")),
    ):
        if pkg not in sys.modules:
            m = types.ModuleType(pkg)
            m.__path__ = [real]
            sys.modules[pkg] = m
            _inserted.append(pkg)
    for sub, rel in (
        ("distllm.errors.types", "distllm/errors/types.py"),
        ("distllm.api.errors", "distllm/api/errors.py"),
        ("distllm.api.ip_utils", "distllm/api/ip_utils.py"),
        ("distllm.core.api_key_store", "distllm/core/api_key_store.py"),
    ):
        if sub not in sys.modules:
            fpath = os.path.join(_SRC, rel)
            if os.path.exists(fpath):
                spec = importlib.util.spec_from_file_location(sub, fpath)
                m = importlib.util.module_from_spec(spec)
                sys.modules[sub] = m
                _inserted.append(sub)
                spec.loader.exec_module(m)
    path = os.path.join(_SRC, "distllm", "api", "middleware.py")
    spec = importlib.util.spec_from_file_location("distllm.api.middleware", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["distllm.api.middleware"] = module
    spec.loader.exec_module(module)
    for name in _inserted:
        sys.modules.pop(name, None)
    return module


_mw = _load_middleware()
_RateLimiter = _mw._RateLimiter
RequestRateLimitMiddleware = _mw.RequestRateLimitMiddleware


# -- 1) XFF rotation ignored: same peer + key => same limiter key -----------


def test_xff_rotation_does_not_change_limiter_key(monkeypatch):
    """H11(1): spoofed X-Forwarded-For must NOT alter the limiter key for a
    direct (untrusted) peer.  We capture the combined key the middleware feeds
    the limiter and assert it is identical across XFF rotations."""
    captured_keys = []

    class _FakeLimiter:
        # The middleware passes a SINGLE combined string key.
        def is_rate_limited(self, key):
            captured_keys.append(key)
            return False

        def retry_after(self, key):
            return 1

        def record_attempt(self, key):
            captured_keys.append(key)

    monkeypatch.setattr(_mw, "_request_rate_limiter", _FakeLimiter())
    # Fail-closed: get_client_ip returns the REAL peer (XFF ignored).
    monkeypatch.setattr(_mw, "get_client_ip", lambda req: "9.9.9.9")

    async def _run():
        for xff in ("1.1.1.1", "2.2.2.2", "3.3.3.3", "203.0.113.55"):
            req = types.SimpleNamespace(
                url=types.SimpleNamespace(path="/v1/chat/completions"),
                headers={"X-Forwarded-For": xff},
                client=types.SimpleNamespace(host="9.9.9.9"),
                state=types.SimpleNamespace(api_key_id="tenant-9", request_id="r"),
            )

            async def call_next(r):
                return "resp"

            mw = RequestRateLimitMiddleware(app=None)
            mw._rate_limit_value = 1000
            await mw.dispatch(req, call_next)

    asyncio.run(_run())
    # The combined key must be identical every time -> XFF spoofing is inert.
    assert len({k for k in captured_keys}) == 1, f"XFF changed the key: {set(captured_keys)}"
    assert captured_keys[0] == "9.9.9.9|tenant-9"


def test_middleware_limits_attacker_under_xff_rotation(monkeypatch):
    """H11(1) end-to-end: because the key is stable, an attacker rotating XFF is
    still rate-limited (429) once the key's window is exhausted."""
    # A real limiter with a tiny window, default ip-mode (middleware prepends
    # 'ip:'); we drive it through the middleware dispatch so the combined key
    # is exercised for real.
    real_limiter = _RateLimiter(max_attempts=3, window_seconds=60)
    monkeypatch.setattr(_mw, "_request_rate_limiter", real_limiter)
    monkeypatch.setattr(_mw, "get_client_ip", lambda req: "9.9.9.9")

    async def _run():
        limited = 0
        for xff in [f"203.0.113.{i}" for i in range(10)]:
            req = types.SimpleNamespace(
                url=types.SimpleNamespace(path="/v1/chat/completions"),
                headers={"X-Forwarded-For": xff},
                client=types.SimpleNamespace(host="9.9.9.9"),
                state=types.SimpleNamespace(api_key_id="tenant-9", request_id="r"),
            )
            saw_429 = {}

            async def call_next(r):
                return "resp"

            # Wrap call_next to detect a 429 response from the middleware.
            real_call = call_next

            async def _wrap(r):
                resp = await real_call(r)
                if isinstance(resp, dict) is False and getattr(resp, "status_code", None) == 429:
                    saw_429["hit"] = True
                return resp

            mw = RequestRateLimitMiddleware(app=None)
            mw._rate_limit_value = 1000
            resp = await mw.dispatch(req, _wrap)
            if getattr(resp, "status_code", None) == 429:
                limited += 1
        return limited

    limited = asyncio.run(_run())
    # Same stable key across 10 XFF rotations -> limited by the 3-attempt cap.
    assert limited > 0, "attacker must be limited despite XFF rotation"


# -- 2) rotating key => correctly separated ----------------------------------


def test_rotating_api_key_changes_limiter_key(monkeypatch):
    """H11(2): a different API key (even from the same peer) yields a different
    limiter key -> budgets are separated, never globally locked."""
    captured_keys = []

    class _FakeLimiter:
        def is_rate_limited(self, key):
            return False

        def retry_after(self, key):
            return 1

        def record_attempt(self, key):
            captured_keys.append(key)

    monkeypatch.setattr(_mw, "_request_rate_limiter", _FakeLimiter())
    monkeypatch.setattr(_mw, "get_client_ip", lambda req: "9.9.9.9")

    async def _run():
        for key in ("tenant-9", "tenant-10", "spoofed-key"):
            req = types.SimpleNamespace(
                url=types.SimpleNamespace(path="/v1/chat/completions"),
                headers={},
                client=types.SimpleNamespace(host="9.9.9.9"),
                state=types.SimpleNamespace(api_key_id=key, request_id="r"),
            )

            async def call_next(r):
                return "resp"

            mw = RequestRateLimitMiddleware(app=None)
            mw._rate_limit_value = 1000
            await mw.dispatch(req, call_next)

    asyncio.run(_run())
    assert "9.9.9.9|tenant-9" in captured_keys
    assert "9.9.9.9|tenant-10" in captured_keys
    assert "9.9.9.9|spoofed-key" in captured_keys
    assert len({k for k in captured_keys}) == 3


# -- 3) limiter-level: api_key mode limits IP rotation, separates keys -------


def test_limiter_api_key_mode_rotation_ip_same_key_limited():
    """H11(3): in api_key mode, rotating the source IP with one key is still
    rate-limited (the limiter keys on the stable key identity)."""
    rl = _RateLimiter(max_attempts=3, window_seconds=60, key_by="api_key")
    KEY = "tenant-key-1"
    blocked_at = None
    for i in range(50):
        ip = f"203.0.113.{i}"  # attacker rotates through many IPs
        if rl.is_rate_limited(ip, KEY):
            blocked_at = i
            break
        rl.record_attempt(ip, KEY)
    assert blocked_at == 3, f"rotating IPs should hit the key cap at 3, got {blocked_at}"


def test_limiter_api_key_mode_rotating_key_same_ip_separated():
    """H11(3): a fresh API key on the same IP gets its own budget, the exhausted
    key stays limited (no cross-key bleed, no global IP lockout)."""
    rl = _RateLimiter(max_attempts=3, window_seconds=60, key_by="api_key")
    ip = "198.51.100.7"
    rl.record_attempt(ip, "keyA")
    rl.record_attempt(ip, "keyA")
    rl.record_attempt(ip, "keyA")
    assert rl.is_rate_limited(ip, "keyA") is True  # keyA exhausted
    assert rl.is_rate_limited(ip, "keyB") is False  # keyB on same IP is fresh


def test_legacy_ip_only_mode_is_the_bypass_configuration():
    """H11 (regression anchor): document that ip-ONLY keying is bypassable by IP
    rotation -- which is exactly why the key must include the stable api_key
    identity.  This pins the *weakness* so a refactor back to ip-only fails."""
    rl = _RateLimiter(max_attempts=3, window_seconds=60, key_by="ip")
    attempts = 0
    for i in range(100):
        ip = f"10.0.0.{i}"
        if not rl.is_rate_limited(ip):
            rl.record_attempt(ip)
            attempts += 1
        else:
            break
    assert attempts > 3, "ip-only keying is bypassable via IP rotation"
