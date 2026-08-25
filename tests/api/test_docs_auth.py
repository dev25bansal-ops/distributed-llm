"""DocsAuthMiddleware ordering regression tests (audit finding B1, Area 3).

DocsAuthMiddleware guards ``/docs``, ``/redoc`` and ``/openapi.json`` behind
the admin role by reading ``request.state.api_key_role`` — a field that
AuthMiddleware (or the SSO middleware) populates.  Under Starlette,
``add_middleware()`` *prepends*, so the LAST-registered middleware runs FIRST
(outermost).  Therefore DocsAuth must be REGISTERED BEFORE AuthMiddleware in
code so that it EXECUTES AFTER auth on the request path.

Historical bug: DocsAuth was registered *after* AuthMiddleware, making it
outer — it checked the role before any identity existed, so every admin
API-key request got 403 and the docs pages were unreachable for everyone.

These tests pin both the static registration invariant and the dynamic
end-to-end behaviour on the REAL app object (stub apps hid the original bug).
"""

import json
import secrets

import pytest
from fastapi.testclient import TestClient

from distllm.api.middleware import AuthMiddleware
from distllm.api.server import DocsAuthMiddleware, app


DOCS_PATHS = ("/docs", "/redoc", "/openapi.json")


# ---------------------------------------------------------------------------
# Fixtures — exercise the real app and its real middleware registration order
# ---------------------------------------------------------------------------


@pytest.fixture
def admin_client(monkeypatch):
    """TestClient on the real app with exactly one admin API key."""
    from distllm.core.api_key_store import reset_api_key_store

    key = secrets.token_urlsafe(32)
    monkeypatch.delenv("DISTLLM_DISABLE_DOCS", raising=False)
    monkeypatch.delenv("DISABLE_AUTH", raising=False)
    monkeypatch.delenv("DISTLLM_DEV_MODE", raising=False)
    monkeypatch.delenv("API_KEYS", raising=False)
    monkeypatch.setenv("API_KEY", key)
    monkeypatch.setenv("API_KEY_WAS_SET", "1")
    reset_api_key_store()

    client = TestClient(app)
    client._test_api_key = key
    yield client
    reset_api_key_store()


@pytest.fixture
def limited_role_client(monkeypatch):
    """TestClient on the real app whose only key has a NON-admin role."""
    from distllm.core.api_key_store import reset_api_key_store

    key = secrets.token_urlsafe(32)
    monkeypatch.delenv("DISTLLM_DISABLE_DOCS", raising=False)
    monkeypatch.delenv("DISABLE_AUTH", raising=False)
    monkeypatch.delenv("DISTLLM_DEV_MODE", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("API_KEY_WAS_SET", raising=False)
    # API_KEYS takes precedence over API_KEY in ApiKeyStore._load().
    monkeypatch.setenv(
        "API_KEYS",
        json.dumps({"keys": [{"key": key, "role": "read-only", "label": "ro-test"}]}),
    )
    reset_api_key_store()

    client = TestClient(app)
    client._test_api_key = key
    yield client
    reset_api_key_store()


# ---------------------------------------------------------------------------
# Static ordering conformance — kills the B1 class at collection time
# ---------------------------------------------------------------------------


class TestDocsAuthRegistrationOrder:
    def test_auth_is_outer_relative_to_docs_auth(self):
        """AuthMiddleware must execute BEFORE DocsAuthMiddleware.

        Starlette ``add_middleware()`` inserts at index 0, so a LOWER
        ``app.user_middleware`` index means OUTER = runs first on the request
        path.  If DocsAuth ever becomes outer of Auth again, api_key_role is
        always unset at check time and admins get 403 (the original bug).
        """
        classes = [mw.cls for mw in app.user_middleware]
        assert AuthMiddleware in classes, "AuthMiddleware missing from the stack"
        assert (
            classes.index(AuthMiddleware) < classes.index(DocsAuthMiddleware)
        ), (
            "DocsAuthMiddleware is outer of AuthMiddleware — it would check "
            "request.state.api_key_role before AuthMiddleware populates it "
            "(regression of audit finding B1)"
        )


# ---------------------------------------------------------------------------
# End-to-end behaviour on the real app
# ---------------------------------------------------------------------------


class TestDocsAuthAdminAccess:
    @pytest.mark.parametrize("path", DOCS_PATHS)
    def test_admin_key_gets_200(self, admin_client, path):
        resp = admin_client.get(
            path,
            headers={"Authorization": f"Bearer {admin_client._test_api_key}"},
        )
        assert resp.status_code == 200, resp.text

    @pytest.mark.parametrize("path", DOCS_PATHS)
    def test_unauthenticated_gets_401_not_403(self, admin_client, path):
        """No credentials => 401 from AuthMiddleware, NOT 403 from DocsAuth.

        A 403 here would mean DocsAuth executed before AuthMiddleware again
        (it would reject on a missing role before auth could emit 401).
        """
        resp = admin_client.get(path)
        assert resp.status_code == 401, resp.text

    @pytest.mark.parametrize("path", DOCS_PATHS)
    def test_invalid_key_gets_401(self, admin_client, path):
        resp = admin_client.get(
            path,
            headers={"Authorization": "Bearer not-a-real-key"},
        )
        assert resp.status_code == 401, resp.text


class TestDocsAuthNonAdminRejected:
    @pytest.mark.parametrize("path", DOCS_PATHS)
    def test_read_only_role_gets_403(self, limited_role_client, path):
        resp = limited_role_client.get(
            path,
            headers={"Authorization": f"Bearer {limited_role_client._test_api_key}"},
        )
        assert resp.status_code == 403, resp.text
        assert "Admin access required" in resp.json()["error"]["message"]

    def test_valid_non_admin_still_passes_auth_layer(self, limited_role_client):
        """Sanity: the read-only key itself is VALID (401 would mean the key
        failed authentication rather than authorization)."""
        resp = limited_role_client.get(
            "/v1/models",
            headers={"Authorization": f"Bearer {limited_role_client._test_api_key}"},
        )
        assert resp.status_code != 401, resp.text
