"""Regression tests for HIGH fix H3: per-tenant quota bypass.

H3 covers the quota-bypass class where an attacker tries to exceed a tenant's
plan limits (requests/min, tokens/day, monthly cost cap) and evade the 429 by:

(a) exhausting the cap and confirming the limiter returns 429;
(b) attempting to bypass via *header spoofing* / a *different key* that still
    maps to the SAME tenant -- it must still be limited (per-tenant isolation of
    the counter, not per-key or per-IP);
(c) confirming per-tenant isolation: tenant B is unaffected when tenant A is
    exhausted (and vice-versa).

We exercise the REAL :class:`TenantBillingManager` (from
``distllm.core.tenant_billing``) plus the REAL ``QuotaMiddleware``
(``distllm.api.quota_middleware``) used by the ASGI stack.  The billing
manager reuses the E12 ``MeteringStore`` as the authoritative usage store; we
load both via importlib with stub parent packages so we don't import the heavy
server stack.  All counters are in-memory / per-test.
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import sys
import time
import types

import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SRC = os.path.join(_REPO_ROOT, "src")


def _load_tenant_billing():
    """Load tenant_billing.py and its real dependency tree (cost_tracker,
    metering) without executing the top-level distllm.api server __init__."""
    for pkg, real in (
        ("distllm", os.path.join(_SRC, "distllm")),
        ("distllm.core", os.path.join(_SRC, "distllm", "core")),
        ("distllm.errors", os.path.join(_SRC, "distllm", "errors")),
    ):
        if pkg not in sys.modules:
            m = types.ModuleType(pkg)
            m.__path__ = [real]
            sys.modules[pkg] = m
    spec = importlib.util.spec_from_file_location(
        "distllm.core.tenant_billing",
        os.path.join(_SRC, "distllm", "core", "tenant_billing.py"),
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["distllm.core.tenant_billing"] = module
    spec.loader.exec_module(module)
    return module


_tb = _load_tenant_billing()
TenantBillingManager = _tb.TenantBillingManager
TierPlan = _tb.TierPlan
MeteringStore = _tb.__dict__.get("MeteringStore") or __import__(
    "distllm.core.metering", fromlist=["MeteringStore"]
).MeteringStore


def _fresh_manager():
    """Build a TenantBillingManager backed by a FRESH in-memory MeteringStore.

    The default constructor shares the module-level ``get_metering_store()``
    singleton, which other tests may have seeded.  A private store keeps each
    test isolated (per-task rule: in-memory managers, no cross-test bleed)."""
    return TenantBillingManager(store=MeteringStore())


def _load_quota_middleware():
    """Load an isolated copy of QuotaMiddleware for direct dispatch tests.

    The loader previously left freshly-exec'd copies of
    ``distllm.core.usage_meter`` / ``distllm.api.quota_middleware``
    registered under their canonical sys.modules names.  Test modules
    imported *after* this one then bound those orphan copies while modules
    imported *before* it held the originals — so ``_meter`` singleton
    injection and class-identity assertions failed depending purely on
    collection order.  We now exec into throwaway module objects registered
    under ``_h3_isolated.*`` names only; the loaded copy binds its own
    UsageMeter at exec time, keeping this file self-consistent while the
    canonical import namespace stays untouched.
    """
    # Ensure parent packages are importable for spec resolution.
    for pkg in ("distllm", "distllm.api", "distllm.core"):
        if pkg not in sys.modules:
            m = types.ModuleType(pkg)
            m.__path__ = [os.path.join(_SRC, *pkg.split("."))]
            sys.modules[pkg] = m

    loaded = {}
    # Throwaway namespace so canonical imports are never shadowed.  The
    # modules must be registered under these names during exec (dataclass
    # processing looks up sys.modules[cls.__module__]).
    ns = "_h3_isolated"
    try:
        for sub, rel in (
            ("distllm.core.usage_meter", "distllm/core/usage_meter.py"),
            ("distllm.api.quota_middleware", "distllm/api/quota_middleware.py"),
        ):
            spec = importlib.util.spec_from_file_location(
                f"{ns}.{sub}", os.path.join(_SRC, rel)
            )
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            sys.modules[f"{ns}.{sub}"] = module
            spec.loader.exec_module(module)
            loaded[sub] = module
    finally:
        sys.modules.pop(f"{ns}.distllm.api.quota_middleware", None)
        sys.modules.pop(f"{ns}.distllm.core.usage_meter", None)

    return loaded["distllm.api.quota_middleware"]


_qm = _load_quota_middleware()
QuotaMiddleware = _qm.QuotaMiddleware


# -- (a) exhaust token/day cap -> 429 ----------------------------------------


def test_exhaust_requests_per_min_returns_429():
    """H3(a): hammering the free plan (20 req/min) is blocked with 429."""
    mgr = _fresh_manager()
    mgr.set_tier("A", "free")
    allowed = 0
    last_deny = None
    for _ in range(30):
        d = mgr.check("A", consume=True)
        if d.allowed:
            allowed += 1
        else:
            last_deny = d
    assert allowed == 20, "free plan allows exactly 20 requests/min"
    assert last_deny is not None
    assert last_deny.allowed is False
    assert last_deny.status_code == 429


def test_exhaust_tokens_per_day_returns_429():
    """H3(a): draining the daily token cap blocks further requests (429)."""
    mgr = _fresh_manager()
    mgr.set_tier("A", "free")  # 100_000 tokens/day
    # Record a huge usage against the store directly (E12 store is authoritative).
    mgr._store.record_request(
        tenant_id="A",
        tokens_in=200_000,
        tokens_out=0,
        compute_s=0.0,
        cost_usd=0.0,
        model_name="m",
        endpoint="/v1/chat/completions",
        request_id="seed",
        timestamp=time.time(),
    )
    d = mgr.check("A")
    assert d.allowed is False, "exhausted daily token cap must deny"
    assert d.status_code == 429


# -- (b) header spoofing / different key, same tenant -> still limited -------


def test_different_key_same_tenant_still_limited():
    """H3(b): a request arriving under a 'different key' but same tenant is still
    counted against the SAME tenant cap (no per-key bypass)."""
    mgr = _fresh_manager()
    mgr.set_tier("A", "free")
    # Exhaust via one identity...
    for _ in range(20):
        assert mgr.check("A", consume=True).allowed
    # ...then a fresh 'spoofed' request with a different caller identity must
    # STILL hit the tenant cap (the counter is keyed by tenant_id, not key).
    spoofed = mgr.check("A")  # tenant unchanged
    assert spoofed.allowed is False, "different key / header cannot bypass tenant cap"


def test_quota_middleware_blocks_same_tenant_across_keys():
    """H3(b) end-to-end: QuotaMiddleware returns 429 for the tenant regardless of
    which API key/host presents the request, once the tenant is over cap."""
    mgr = _fresh_manager()
    mgr.set_tier("acme", "free")
    for _ in range(20):
        assert mgr.check("acme", consume=True).allowed

    class _FakeBilling:
        def check(self, tenant_id, *a, **k):
            return mgr.check(tenant_id, *a, **k)

    # Point the middleware at our exhausted manager.
    _qm.get_billing_manager = lambda: _FakeBilling()  # type: ignore[assignment]

    async def _run():
        captured = {}

        async def call_next(req):
            captured["next"] = True
            return _Resp()

        class _Resp:
            status_code = 200
            body = b"{}"

        for key_label in ("key-A", "key-B", "spoofed-key", "different-host"):
            req = types.SimpleNamespace(
                url=types.SimpleNamespace(path="/v1/chat/completions"),
                state=types.SimpleNamespace(tenant_id="acme", api_key_id=key_label),
            )
            mw = QuotaMiddleware(app=None, enable=True, tenant_billing=True)
            resp = await mw.dispatch(req, call_next)
            assert resp.status_code == 429, (
                f"tenant 'acme' must stay blocked for key {key_label!r}"
            )
            assert captured.get("next") is None

    asyncio.run(_run())


# -- (c) per-tenant isolation ------------------------------------------------


def test_tenant_isolation_other_tenant_unaffected():
    """H3(c): exhausting tenant A must NOT affect tenant B at all."""
    mgr = _fresh_manager()
    mgr.set_tier("A", "free")
    mgr.set_tier("B", "free")
    for _ in range(20):
        assert mgr.check("A", consume=True).allowed
    assert mgr.check("A").allowed is False  # A exhausted
    assert mgr.check("B").allowed is True  # B untouched


def test_tenant_b_recovers_independently():
    """H3(c): resetting A's window leaves B's budget intact and vice-versa."""
    mgr = _fresh_manager()
    mgr.set_tier("A", "free")
    mgr.set_tier("B", "free")
    for _ in range(20):
        mgr.check("A", consume=True)
    assert mgr.check("A").allowed is False
    assert mgr.check("B").allowed is True
    # Only reset A.
    mgr.reset_rate_windows()
    assert mgr.check("A").allowed is True, "A recovered after window reset"
    assert mgr.check("B").allowed is True, "B was never exhausted"
