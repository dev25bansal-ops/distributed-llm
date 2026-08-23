"""Regression tests: exact-match cache key is tenant/user scoped.

P0 finding (CVSS 7.5): ``cache_plugin.py`` `_build_cache_key` hashed only
(prompt|model|temperature|top_p), so two tenants sending the identical request
shared one cache entry — tenant A's private personalized response could be
served to tenant B on a cache hit.

Fix: `_request_scope` builds the scope from the AUTHENTICATED identity the
request dispatcher actually supplies (``api_key_id`` + server-set ``tenant``),
and ``_build_cache_key`` encodes the whole request tuple as JSON so a
delimiter character in a scope or prompt can never collide with a different
request.
"""

from __future__ import annotations

import pytest

from distllm.plugins.cache_plugin import CachePlugin, _build_cache_key


@pytest.fixture
def plugin(monkeypatch) -> CachePlugin:
    monkeypatch.setenv("DISTLLM_PLUGIN_CACHE_ENABLED", "1")
    monkeypatch.setenv("DISTLLM_PLUGIN_CACHE_SEMANTIC_ENABLED", "0")
    p = CachePlugin()
    p.on_init({"config": {}})
    return p


def _req(
    prompt: str,
    tenant: str | None = None,
    user: str | None = None,
    api_key_id: str | None = None,
    model: str = "model-x",
) -> dict:
    """Build a plugin request context.

    Mirrors the keys the production dispatcher supplies (server.py req_ctx:
    ``tenant``, ``api_key_id``, ...) plus the historical ``user_id``.
    """
    ctx: dict = {"prompt": prompt, "model": model, "temperature": 0.7, "top_p": 1.0}
    if tenant:
        ctx["tenant"] = tenant
    if user:
        ctx["user_id"] = user
    if api_key_id:
        ctx["api_key_id"] = api_key_id
    return ctx


def _store(plugin, prompt: str, response: str, **ident) -> None:
    plugin.on_response(_req(prompt, **ident), {"text": response})


class TestCacheKeyScoping:
    def test_build_cache_key_includes_scope(self):
        k1 = _build_cache_key("hi", "m", 0.7, 1.0, scope="tenant-a")
        k2 = _build_cache_key("hi", "m", 0.7, 1.0, scope="tenant-b")
        k3 = _build_cache_key("hi", "m", 0.7, 1.0, scope="")
        assert k1 != k2
        assert k1 != k3

    def test_delimiter_in_scope_or_prompt_cannot_collide(self):
        # Regression for the non-injective '|' separator: a scope containing
        # '|' must not collide with a different prompt/scope combination.
        a = _build_cache_key("y|z", "m", 0.7, 1.0, scope="x")
        b = _build_cache_key("z", "m", 0.7, 1.0, scope="x|y")
        assert a != b
        assert (
            _build_cache_key("p", "m", 0.7, 1.0, scope="a|b")
            != _build_cache_key("p", "m", 0.7, 1.0, scope="a")
        )


class TestCrossTenantIsolation:
    def test_tenant_b_never_receives_tenant_a_cached_response(self, plugin):
        _store(plugin, "what is my balance", "Your balance is $12,345", tenant="acme", api_key_id="key-acme")

        # Tenant B asks the SAME prompt -> must be a miss (no cross-tenant leak).
        miss = plugin.on_request(_req("what is my balance", tenant="planet", api_key_id="key-planet"))
        assert miss is None

    def test_tenant_a_reuses_its_own_cached_response(self, plugin):
        _store(plugin, "what is my balance", "Your balance is $12,345", tenant="acme", api_key_id="key-acme")

        hit = plugin.on_request(_req("what is my balance", tenant="acme", api_key_id="key-acme"))
        assert hit is not None
        assert "Your balance is $12,345" in hit["_cached_response"]

    def test_same_tenant_different_api_key_isolated(self, plugin):
        _store(plugin, "my notes", "alice notes", tenant="acme", api_key_id="key-alice")
        assert plugin.on_request(_req("my notes", tenant="acme", api_key_id="key-bob")) is None
        hit = plugin.on_request(_req("my notes", tenant="acme", api_key_id="key-alice"))
        assert hit is not None
        assert "alice notes" in hit["_cached_response"]

    def test_tenant_only_isolated_per_tenant(self, plugin):
        # SSO-style: no API key, only a server-set tenant.
        _store(plugin, "my history", "acme history", tenant="acme")
        assert plugin.on_request(_req("my history", tenant="planet")) is None
        hit = plugin.on_request(_req("my history", tenant="acme"))
        assert hit is not None

    def test_no_identity_requests_still_share(self, plugin):
        # Requests without any identity behave as before (shared namespace).
        _store(plugin, "hello", "world")
        hit = plugin.on_request(_req("hello"))
        assert hit is not None
        assert "world" in hit["_cached_response"]
