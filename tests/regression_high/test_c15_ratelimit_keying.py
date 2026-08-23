"""Regression tests for HIGH fix C15: rate-limit keyed by client_ip only.

``RequestRateLimitMiddleware`` previously keyed the sliding-window limiter
solely by source IP. That lets a single stolen/abused API key bypass limits
from many IPs, and conversely penalises every user behind one NAT IP. The
fix keys by ``ip|api_key_id`` when an API key is present, falling back to IP
alone when unauthenticated.

We test the keying policy directly via ``_RateLimiter`` (the same structure
the middleware now uses) plus the middleware's dispatch key construction.
"""

from __future__ import annotations

import importlib.util
import sys
import types
import asyncio

import pytest

# `distllm.api.middleware` is reachable only through `distllm.api.__init__`,
# which imports the heavy server stack (transformers, etc.). To verify the
# rate-limit keying fix in isolation we load just the middleware module via
# importlib, injecting empty parent packages so the package __init__ files
# (which pull the heavy stack) are NOT executed.
_REPO_SRC = __import__("pathlib").Path(__file__).resolve().parents[2] / "src"


def _load_middleware():
    _inserted = []
    for pkg in ("distllm", "distllm.api", "distllm.core", "distllm.errors"):
        if pkg not in sys.modules:
            mod = types.ModuleType(pkg)
            mod.__path__ = []
            sys.modules[pkg] = mod
            _inserted.append(pkg)
    # Preload the lightweight submodules middleware.py imports, so they resolve
    # against the stubbed packages instead of triggering the heavy server stack.
    for sub in (
        "distllm.errors.types",
        "distllm.api.errors",
        "distllm.api.ip_utils",
        "distllm.core.api_key_store",
    ):
        if sub not in sys.modules:
            rel = sub.replace(".", "/") + ".py"
            fpath = _REPO_SRC / rel
            if fpath.exists():
                spec = importlib.util.spec_from_file_location(sub, fpath)
                m = importlib.util.module_from_spec(spec)
                sys.modules[sub] = m
                _inserted.append(sub)
                spec.loader.exec_module(m)
    path = _REPO_SRC / "distllm" / "api" / "middleware.py"
    spec = importlib.util.spec_from_file_location("distllm.api.middleware", path)
    module = importlib.util.module_from_spec(spec)
    _prev_mw = sys.modules.get("distllm.api.middleware")
    sys.modules["distllm.api.middleware"] = module
    spec.loader.exec_module(module)
    # Self-clean: remove the stub packages / isolated modules we injected so we
    # do NOT shadow the REAL distllm.api package for other tests in the same
    # pytest session (e.g. test_n3_waf_opa imports distllm.api.waf, which fails
    # if a stub distllm.api with empty __path__ lingers here). The returned
    # `module` keeps its bound references, so c15's own tests still work.
    for name in _inserted:
        sys.modules.pop(name, None)
    if _prev_mw is not None:
        sys.modules["distllm.api.middleware"] = _prev_mw
    else:
        sys.modules.pop("distllm.api.middleware", None)
    return module


_mw = _load_middleware()
_RateLimiter = _mw._RateLimiter
RequestRateLimitMiddleware = _mw.RequestRateLimitMiddleware


def test_rate_limiter_dual_key_separates_identities():
    rl = _RateLimiter(max_attempts=2, window_seconds=60, key_by="ip")
    # Two different API keys from the same IP must not share a bucket.
    key_a = "1.2.3.4|keyA"
    key_b = "1.2.3.4|keyB"
    assert rl.is_rate_limited(key_a) is False
    rl.record_attempt(key_a)
    rl.record_attempt(key_a)
    assert rl.is_rate_limited(key_a) is True
    # keyB (different API key, same IP) is independent.
    assert rl.is_rate_limited(key_b) is False


def test_middleware_dispatch_keys_by_api_key(monkeypatch):
    # Build a fake request with an api_key_id and confirm the middleware would
    # build a combined key (ip|key). We monkeypatch the limiter to capture the
    # key it records.
    captured = {}

    class _FakeLimiter:
        def is_rate_limited(self, key):
            captured["checked"] = key
            return False

        def retry_after(self, key):
            return 1

        def record_attempt(self, key):
            captured["recorded"] = key

    monkeypatch.setattr(_mw, "_request_rate_limiter", _FakeLimiter())

    fake_request = types.SimpleNamespace(
        url=types.SimpleNamespace(path="/v1/chat/completions"),
        headers={},
        client=types.SimpleNamespace(host="1.2.3.4"),
        state=types.SimpleNamespace(api_key_id="tenant-9", request_id="r1"),
    )
    called = {}

    async def call_next(req):
        called["next"] = True
        return "resp"

    mw_instance = RequestRateLimitMiddleware(app=None)
    mw_instance._rate_limit_value = 1000

    asyncio.run(mw_instance.dispatch(fake_request, call_next))
    assert captured["recorded"].endswith("|tenant-9")
    assert called.get("next") is True
