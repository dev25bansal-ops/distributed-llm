"""Tests for role-based access control on model load/unload endpoints.

Covers:
- /api/models/load requires model-admin or higher
- /api/models/unload requires model-admin or higher
- /api/models/registry is accessible to read-only
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from distllm.api.auth_deps import require_role


class TestModelLoadRBAC:
    """RBAC enforcement for model management."""

    @pytest.mark.parametrize("role,expected_status", [
        ("admin", 200),
        ("model-admin", 200),
        ("inference-only", 403),
        ("read-only", 403),
        (None, 401),
    ])
    @pytest.mark.asyncio
    async def test_load_model_role_check(self, role, expected_status):
        """Model load should be restricted to model-admin+ roles."""
        from fastapi import HTTPException
        from fastapi import Request

        mock_request = MagicMock(spec=Request)
        mock_request.state.api_key_role = role
        mock_request.state.api_key_id = "test-key" if role else None

        check = require_role("model-admin")
        if expected_status == 401:
            with pytest.raises(HTTPException) as exc:
                await check(mock_request)
            assert exc.value.status_code == 401
        elif expected_status == 403:
            with pytest.raises(HTTPException) as exc:
                await check(mock_request)
            assert exc.value.status_code == 403
        else:
            result = await check(mock_request)
            assert result is None  # Allowed

    @pytest.mark.parametrize("role,expected_status", [
        ("admin", 200),
        ("model-admin", 200),
        ("inference-only", 403),
        ("read-only", 403),
    ])
    @pytest.mark.asyncio
    async def test_unload_model_role_check(self, role, expected_status):
        """Model unload should be restricted to model-admin+ roles."""
        from fastapi import HTTPException
        from fastapi import Request

        mock_request = MagicMock(spec=Request)
        mock_request.state.api_key_role = role
        mock_request.state.api_key_id = "test-key"

        check = require_role("model-admin")
        if expected_status == 403:
            with pytest.raises(HTTPException) as exc:
                await check(mock_request)
            assert exc.value.status_code == 403
        else:
            result = await check(mock_request)
            assert result is None  # Allowed

    @pytest.mark.parametrize("role,expected_status", [
        ("admin", 200),
        ("model-admin", 200),
        ("inference-only", 200),
        ("read-only", 200),
    ])
    @pytest.mark.asyncio
    async def test_registry_endpoint_permissive(self, role, expected_status):
        """Model registry should be accessible to all authenticated roles."""
        from fastapi import HTTPException
        from fastapi import Request

        mock_request = MagicMock(spec=Request)
        mock_request.state.api_key_role = role
        mock_request.state.api_key_id = "test-key"

        check = require_role("read-only")
        result = await check(mock_request)
        assert result is None  # All authenticated roles can read registry
