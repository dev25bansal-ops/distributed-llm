"""Tenant identification and isolation middleware.

Extracts tenant identity from API key (X-Tenant-API-Key header or Bearer token),
validates against the tenant store, and enforces resource quotas.
"""

import os
import time
from collections import defaultdict

from fastapi import Request
from fastapi.responses import JSONResponse
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware

from distllm.tenant.store import TenantStore


class TenantMiddleware(BaseHTTPMiddleware):
    """Middleware that identifies tenants and enforces quotas.

    Tenants are identified by:
    1. X-Tenant-ID header (for system-to-system calls)
    2. X-Tenant-API-Key header (tenant API key)
    3. Bearer token in Authorization header (falls back to tenant lookup)
    """

    def __init__(self, app, store: TenantStore, enabled: bool = True):
        super().__init__(app)
        self.store = store
        self.enabled = enabled
        self._concurrent: dict[str, int] = defaultdict(int)

    async def dispatch(self, request: Request, call_next):
        if not self.enabled or os.environ.get("DISABLE_TENANTS") == "1":
            request.state.tenant_id = "default"
            request.state.tenant = None
            request.state.tenant_tier = "free"
            return await call_next(request)

        tenant_id = self._resolve_tenant(request)

        if tenant_id is None:
            return JSONResponse(
                status_code=401,
                content={"error": "Unauthorized", "message": "Invalid API key or tenant credentials", "type": "tenant_auth_error"},
            )

        tenant = self.store.get_tenant(tenant_id)
        if not tenant or not tenant.is_active:
            return JSONResponse(
                status_code=403,
                content={"error": "Forbidden", "message": "Tenant not found or inactive", "type": "tenant_error"},
            )

        request.state.tenant_id = tenant.tenant_id
        request.state.tenant = tenant
        request.state.tenant_tier = tenant.tier.value

        # Enforce concurrent request limit
        self._concurrent[tenant_id] += 1
        if self._concurrent[tenant_id] > tenant.quota.max_concurrent_requests:
            self._concurrent[tenant_id] -= 1
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Too Many Requests",
                    "message": f"Concurrent request limit ({tenant.quota.max_concurrent_requests}) exceeded",
                    "type": "tenant_rate_limit",
                    "tenant_id": tenant_id,
                },
            )

        try:
            response = await call_next(request)
            return response
        finally:
            self._concurrent[tenant_id] = max(0, self._concurrent[tenant_id] - 1)

    def _resolve_tenant(self, request: Request) -> str | None:
        header_id = request.headers.get("X-Tenant-ID")
        if header_id:
            return header_id

        header_key = request.headers.get("X-Tenant-API-Key")
        if header_key:
            tenant = self.store.get_tenant_by_api_key(header_key)
            if tenant:
                return tenant.tenant_id

        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            key = auth[7:]
            if key.startswith("tnt_"):
                tenant = self.store.get_tenant_by_api_key(key)
                if tenant:
                    return tenant.tenant_id

        return None
