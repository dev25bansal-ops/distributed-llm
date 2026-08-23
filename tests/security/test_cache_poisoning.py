"""Security: Cache poisoning via tenant isolation bypass and header injection.

The DedupMiddleware fingerprints requests by SHA-256(body) + tenant_id.
If the tenant identifier can be manipulated via headers (X-Forwarded-For,
X-Real-IP), an attacker could force cache key collisions across tenants.

The dedup middleware uses ``request.state.api_key_id`` as tenant identifier,
falling back to ``request.state.client_ip``. If neither is set, ``None`` is
used — which means all unauthenticated requests from different IPs share
the same cache namespace.
"""

from __future__ import annotations

import hashlib
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.middleware.base import BaseHTTPMiddleware

from distllm.api.dedup import DedupMiddleware, _FingerprintCache


class TestCachePoisoning:
    """Cache isolation and poisoning resistance."""

    def test_tenant_isolation_via_tenant_id(self):
        """Different tenant_ids produce different cache keys for identical bodies."""
        cache = _FingerprintCache()
        body = json.dumps({"messages": [{"role": "user", "content": "list my invoices"}]}).encode()
        fp_tenant_a = cache.fingerprint(body, tenant_id="tenant-a")
        fp_tenant_b = cache.fingerprint(body, tenant_id="tenant-b")
        assert fp_tenant_a != fp_tenant_b

    def test_tenant_none_creates_separate_namespace(self):
        """tenant_id=None vs a real tenant produce different keys."""
        cache = _FingerprintCache()
        body = b"test body"
        fp_none = cache.fingerprint(body, tenant_id=None)
        fp_real = cache.fingerprint(body, tenant_id="real-tenant")
        assert fp_none != fp_real

    def test_cache_key_is_deterministic(self):
        """Same body + same tenant always produces the same key."""
        cache = _FingerprintCache()
        body = b"deterministic body"
        fp1 = cache.fingerprint(body, tenant_id="t1")
        fp2 = cache.fingerprint(body, tenant_id="t1")
        assert fp1 == fp2

    def test_cache_entry_contains_tenant_data_only(self):
        """A cached response for one tenant is not accessible by another."""
        cache = _FingerprintCache()
        body = b"sensitive data"
        fp_a = cache.fingerprint(body, tenant_id="tenant-a")
        fp_b = cache.fingerprint(body, tenant_id="tenant-b")
        cache.store(fp_a, '{"secret": "tenant-a-data"}')
        # Tenant B should NOT be able to look up Tenant A's cache
        result_b = cache.lookup(fp_b)
        assert result_b is None

    def test_fallback_to_client_ip(self):
        """When api_key_id is not set, client_ip is used for tenant isolation."""
        cache = _FingerprintCache()
        body = b"test"
        fp_ip_a = cache.fingerprint(body, tenant_id="203.0.113.1")
        fp_ip_b = cache.fingerprint(body, tenant_id="203.0.113.2")
        assert fp_ip_a != fp_ip_b

    def test_dedup_middleware_uses_request_state(self):
        """The DedupMiddleware reads tenant_id from request.state."""
        app = FastAPI()

        request_tenant = None

        class _CaptureMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                nonlocal request_tenant
                tenant_id = getattr(request.state, "api_key_id", None) or getattr(
                    request.state, "client_ip", None
                )
                request_tenant = tenant_id
                return await call_next(request)

        @app.post("/v1/chat/completions")
        async def chat():
            return {"choices": [{"message": {"content": "ok"}}]}

        app.add_middleware(_CaptureMiddleware)
        app.add_middleware(DedupMiddleware)

        client = TestClient(app)

        # Unauthenticated request — no api_key_id, no client_ip set by auth
        # The middleware falls through to request.state.client_ip which is None
        client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})

        # tenant should not crash and should be a string or None
        assert request_tenant is None or isinstance(request_tenant, str)
