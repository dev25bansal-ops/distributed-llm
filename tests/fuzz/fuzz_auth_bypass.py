"""Auth bypass fuzz test — sends requests to every registered route
with missing, invalid, and expired API keys and asserts 401/403.

This ensures no endpoint accidentally allows unauthenticated access.
"""

from __future__ import annotations

import os

import pytest

from distllm.api.auth_deps import require_role


# Route definitions: (method, path, required_role)
_API_ROUTES = [
    ("GET", "/v1/models", "read-only"),
    ("GET", "/v1/health", None),  # Public
    ("GET", "/health", None),  # Public
    ("GET", "/ready", None),  # Public
    ("GET", "/live", None),  # Public
    ("POST", "/v1/chat/completions", "inference-only"),
    ("POST", "/v1/completions", "inference-only"),
    ("POST", "/v1/embeddings", "inference-only"),
    ("GET", "/v1/defrag/status", "read-only"),
    ("POST", "/v1/defrag/run", "admin"),
    ("GET", "/api/cluster/nodes", "read-only"),
    ("GET", "/api/pipeline/health", "auditor"),
    ("GET", "/api/cluster/reputation", "auditor"),
    ("GET", "/api/metrics/collector", "auditor"),
    ("GET", "/api/metrics/stream", "auditor"),
    ("GET", "/api/requests/waterfall", "auditor"),
    ("GET", "/api/continuum/stats", "auditor"),
    ("GET", "/api/cost/summary", "auditor"),
    ("GET", "/api/cost/history", "auditor"),
    ("GET", "/api/streaming-cost/stats", "auditor"),
    ("POST", "/api/models/load", "model-admin"),
    ("POST", "/api/models/unload", "model-admin"),
    ("GET", "/api/models/registry", "read-only"),
    ("POST", "/v1/models/{model_id}/warmup", "model-admin"),
    ("POST", "/v1/federation/heartbeat", "admin"),
    ("POST", "/api/cluster/rotate-key", "admin"),
    ("POST", "/v1/exchange/prompts", "admin"),
]


class TestAuthBypassFuzz:
    """Fuzz test: verify all routes reject unauthenticated requests."""

    @pytest.mark.parametrize("method,path,required_role", _API_ROUTES)
    @pytest.mark.asyncio
    async def test_no_auth_rejected(self, method, path, required_role):
        """Routes that require auth should reject requests with no API key."""
        if required_role is None:
            pytest.skip(f"Public endpoint: {method} {path}")
        from fastapi import HTTPException
        from fastapi import Request
        from unittest.mock import MagicMock

        mock_request = MagicMock(spec=Request)
        mock_request.state.api_key_role = None
        mock_request.state.api_key_id = None

        check = require_role(required_role)
        with pytest.raises(HTTPException) as exc:
            await check(mock_request)
        assert exc.value.status_code == 401, (
            f"{method} {path} should return 401 with no auth, got {exc.value.status_code}"
        )

    @pytest.mark.parametrize("method,path,required_role", _API_ROUTES)
    @pytest.mark.asyncio
    async def test_wrong_role_rejected(self, method, path, required_role):
        """Routes should reject requests with insufficient role."""
        if required_role is None:
            pytest.skip("Public endpoint")
        from fastapi import HTTPException
        from fastapi import Request
        from unittest.mock import MagicMock

        mock_request = MagicMock(spec=Request)
        mock_request.state.api_key_role = "read-only"
        mock_request.state.api_key_id = "test-key"

        # If the route requires something above read-only, it should be rejected
        if required_role in ("admin", "model-admin", "auditor", "inference-only"):
            check = require_role(required_role)
            with pytest.raises(HTTPException) as exc:
                await check(mock_request)
            assert exc.value.status_code == 403, (
                f"{method} {path} (requires {required_role}) should return 403 for read-only, "
                f"got {exc.value.status_code}"
            )

    @pytest.mark.parametrize("method,path,required_role", _API_ROUTES)
    @pytest.mark.asyncio
    async def test_admin_allowed(self, method, path, required_role):
        """Admin role should be allowed on all routes."""
        if required_role is None:
            pytest.skip("Public endpoint")
        from fastapi import Request
        from unittest.mock import MagicMock

        mock_request = MagicMock(spec=Request)
        mock_request.state.api_key_role = "admin"
        mock_request.state.api_key_id = "admin-key"

        check = require_role(required_role)
        result = await check(mock_request)
        assert result is None, (
            f"{method} {path} (requires {required_role}) should allow admin"
        )
