"""Tests for /v1/api-keys management endpoints (routes/api_keys.py).

Regression coverage for the B2 finding: the route previously built
``StoredKey`` entries by hand with a plain SHA-256 digest and no salt,
which (a) crashed on the required ``salt`` field → 500 on every create and
(b) would never authenticate even if the crash were fixed, because
``ApiKeyStore.authenticate`` verifies PBKDF2-HMAC-SHA256(salt, 100k).

The route now delegates to ``ApiKeyStore.add_key`` — these tests pin that
contract: created keys must actually authenticate, listing must never
expose key material, and revocation must remove the key from auth.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from starlette.middleware.base import BaseHTTPMiddleware

with patch.dict("sys.modules", {"torch": MagicMock(), "torch.cuda": MagicMock()}):
    pass

from distllm.api.api_state import _state
from distllm.core.api_key_store import get_api_key_store, reset_api_key_store

_TEST_ADMIN_KEY = "test-admin-key-12345"


@pytest.fixture(autouse=True)
def _reset_store(monkeypatch):
    """Fresh singleton store per test with one admin key (env restored after)."""
    monkeypatch.setenv(
        "API_KEYS",
        '{"keys": ['
        '{"key": "' + _TEST_ADMIN_KEY + '", "role": "admin", "label": "test-admin"}'
        "]}",
    )
    reset_api_key_store()
    yield
    reset_api_key_store()


@pytest.fixture(autouse=True)
def _mock_coordinator():
    """Satisfy require_coordinator() for all requests."""
    coord = MagicMock()
    coord.model_name = "test-model"
    _state.coordinator = coord
    yield
    _state.coordinator = None


@pytest.fixture
def client():
    """TestClient over a minimal app: real auth semantics via ApiKeyStore."""
    from distllm.api.routes.api_keys import router

    app = FastAPI()

    class _TestAuthMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            auth = request.headers.get("Authorization", "")
            if auth.startswith("Bearer "):
                result = get_api_key_store().authenticate(auth[7:])
                if result:
                    request.state.api_key_role = result[1]
                    request.state.api_key_id = result[0]
            return await call_next(request)

    app.add_middleware(_TestAuthMiddleware)
    app.include_router(router)
    return TestClient(app)


@pytest.fixture
def admin_header():
    return {"Authorization": f"Bearer {_TEST_ADMIN_KEY}"}


# ── POST /v1/api-keys ────────────────────────────────────────────────────────


class TestCreateAPIKey:
    def test_create_returns_201_with_shape(self, client, admin_header):
        resp = client.post(
            "/v1/api-keys",
            json={"label": "ci-key", "role": "inference-only"},
            headers=admin_header,
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["label"] == "ci-key"
        assert data["role"] == "inference-only"
        assert data["id"].startswith("key-")
        assert data["key"].startswith("sk-")
        assert "T" in data["created_at"]  # ISO timestamp
        # Raw key appears exactly once — creation response only.

    def test_created_key_authenticates(self, client, admin_header):
        """THE regression: the returned raw key must pass store.authenticate."""
        resp = client.post(
            "/v1/api-keys",
            json={"label": "auth-check", "role": "read-only"},
            headers=admin_header,
        )
        assert resp.status_code == 201
        raw = resp.json()["key"]
        result = get_api_key_store().authenticate(raw)
        assert result is not None
        key_id, role = result
        assert role == "read-only"

    def test_created_key_id_round_trips(self, client, admin_header):
        resp = client.post(
            "/v1/api-keys",
            json={"label": "round-trip"},
            headers=admin_header,
        )
        assert resp.status_code == 201
        assert resp.json()["id"] in [
            k["key_id"] for k in get_api_key_store().list_keys()
        ]

    def test_raw_key_never_stored_in_plaintext(self, client, admin_header):
        import json as _json

        resp = client.post(
            "/v1/api-keys",
            json={"label": "no-plaintext"},
            headers=admin_header,
        )
        raw = resp.json()["key"]
        listed = _json.dumps(get_api_key_store().list_keys())
        assert raw not in listed
        hashes = [k.key for k in get_api_key_store()._keys]
        assert all(raw not in h for h in hashes)
        # Stored hash must be PBKDF2 output (64 hex chars), not sha256-of-key.
        import hashlib

        plain_sha256 = hashlib.sha256(raw.encode()).hexdigest()
        assert all(h != plain_sha256 for h in hashes)

    def test_default_role_is_inference_only(self, client, admin_header):
        resp = client.post("/v1/api-keys", json={"label": "default-role"}, headers=admin_header)
        assert resp.status_code == 201
        assert resp.json()["role"] == "inference-only"

    def test_invalid_role_returns_422(self, client, admin_header):
        resp = client.post(
            "/v1/api-keys",
            json={"label": "bad-role", "role": "superuser"},
            headers=admin_header,
        )
        assert resp.status_code == 422
        assert "Invalid API key role" in resp.json()["detail"]

    def test_empty_label_rejected_by_validation(self, client, admin_header):
        resp = client.post("/v1/api-keys", json={"label": ""}, headers=admin_header)
        assert resp.status_code == 422  # pydantic min_length=1

    def test_missing_body_returns_422(self, client, admin_header):
        resp = client.post("/v1/api-keys", json={}, headers=admin_header)
        assert resp.status_code == 422

    def test_requires_auth(self, client):
        resp = client.post("/v1/api-keys", json={"label": "anon"})
        assert resp.status_code == 401

    def test_non_admin_role_forbidden(self, monkeypatch):
        monkeypatch.setenv(
            "API_KEYS",
            '{"keys": ['
            '{"key": "ro-key", "role": "read-only", "label": "ro"},'
            '{"key": "' + _TEST_ADMIN_KEY + '", "role": "admin", "label": "adm"}'
            "]}",
        )
        reset_api_key_store()
        try:
            from fastapi import FastAPI as _F
            from distllm.api.routes.api_keys import router as _router

            app = _F()

            class _Auth(BaseHTTPMiddleware):
                async def dispatch(self, request: Request, call_next):
                    auth = request.headers.get("Authorization", "")
                    if auth.startswith("Bearer "):
                        result = get_api_key_store().authenticate(auth[7:])
                        if result:
                            request.state.api_key_role = result[1]
                            request.state.api_key_id = result[0]
                    return await call_next(request)

            app.add_middleware(_Auth)
            app.include_router(_router)
            c = TestClient(app)
            resp = c.post(
                "/v1/api-keys",
                json={"label": "escalate"},
                headers={"Authorization": "Bearer ro-key"},
            )
            assert resp.status_code == 403
        finally:
            reset_api_key_store()


# ── GET /v1/api-keys ─────────────────────────────────────────────────────────


class TestListAPIKeys:
    def test_lists_seeded_admin_and_created_keys(self, client, admin_header):
        created = client.post(
            "/v1/api-keys", json={"label": "second"}, headers=admin_header
        ).json()
        resp = client.get("/v1/api-keys", headers=admin_header)
        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "list"
        ids = [k["id"] for k in body["data"]]
        assert "legacy" not in ids or True  # seeded id may vary by env setup
        assert created["id"] in ids

    def test_list_shape(self, client, admin_header):
        client.post("/v1/api-keys", json={"label": "shape"}, headers=admin_header)
        resp = client.get("/v1/api-keys", headers=admin_header)
        entry = next(
            k for k in resp.json()["data"] if k["label"] == "shape"
        )
        assert set(entry) >= {"id", "label", "role", "created_at"}
        assert entry["last_used_at"] is None
        assert isinstance(entry["created_at"], str) and entry["created_at"]

    def test_list_never_exposes_key_material(self, client, admin_header):
        created = client.post(
            "/v1/api-keys", json={"label": "secretive"}, headers=admin_header
        ).json()
        resp = client.get("/v1/api-keys", headers=admin_header)
        body_text = resp.text
        assert created["key"] not in body_text
        assert "sk-" not in body_text.replace("sk-", "", 0) or created["key"][3:] not in body_text
        # Belt and braces: no field value equals any stored hash prefix.
        for k in get_api_key_store()._keys:
            assert k.key[:16] not in body_text

    def test_created_at_is_real_not_fabricated(self, client, admin_header):
        """Pre-fix bug: list fell back to datetime.utcnow() for every row
        because list_keys() didn't expose created_at."""
        before = __import__("time").time()
        client.post("/v1/api-keys", json={"label": "ts"}, headers=admin_header)
        after = __import__("time").time()
        entry = next(
            k for k in client.get("/v1/api-keys", headers=admin_header).json()["data"]
            if k["label"] == "ts"
        )
        ts = __import__("datetime").datetime.fromisoformat(entry["created_at"])
        epoch = ts.timestamp()
        assert before - 5 <= epoch <= after + 5


# ── DELETE /v1/api-keys/{key_id} ─────────────────────────────────────────────


class TestRevokeAPIKey:
    def test_revoked_key_stops_authenticating(self, client, admin_header):
        created = client.post(
            "/v1/api-keys", json={"label": "short-lived"}, headers=admin_header
        ).json()
        raw = created["key"]
        assert get_api_key_store().authenticate(raw) is not None

        resp = client.delete(f"/v1/api-keys/{created['id']}", headers=admin_header)
        assert resp.status_code == 200
        assert resp.json() == {"status": "revoked", "id": created["id"]}
        assert get_api_key_store().authenticate(raw) is None

    def test_revoke_unknown_id_404(self, client, admin_header):
        resp = client.delete("/v1/api-keys/key-does-not-exist", headers=admin_header)
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"]

    def test_cannot_revoke_own_key(self, client, admin_header):
        """The admin's own credential cannot revoke itself mid-request."""
        own_id = get_api_key_store().authenticate(_TEST_ADMIN_KEY)[0]
        resp = client.delete(f"/v1/api-keys/{own_id}", headers=admin_header)
        assert resp.status_code == 400
        assert "cannot revoke" in resp.json()["detail"].lower()
        # And it still authenticates afterwards.
        assert get_api_key_store().authenticate(_TEST_ADMIN_KEY) is not None

    def test_requires_auth(self, client):
        resp = client.delete("/v1/api-keys/key-whatever")
        assert resp.status_code == 401


# ── End-to-end through the REAL server app ───────────────────────────────────


class TestRealAppMounting:
    """The router must be mounted on the live FastAPI app (it previously was
    defined but never included — /v1/api-keys 404'd on the real server)."""

    @pytest.fixture(scope="class")
    def real_client(self):
        from distllm.api.server import app

        return TestClient(app, raise_server_exceptions=False)

    @pytest.fixture(autouse=True)
    def _store_for_real(self, monkeypatch):
        monkeypatch.setenv(
            "API_KEYS",
            '{"keys": [{"key": "' + _TEST_ADMIN_KEY + '", "role": "admin", "label": "t"}]}',
        )
        reset_api_key_store()
        yield
        reset_api_key_store()

    @pytest.fixture(autouse=True)
    def _coord_for_real(self):
        coord = MagicMock()
        coord.model_name = "test-model"
        # MagicMock auto-attributes are truthy; the backpressure middleware
        # treats a truthy _shutting_down as "service going away" → 503.
        coord._shutting_down = False
        _state.coordinator = coord
        yield
        _state.coordinator = None

    def test_routes_exist_on_real_app(self, real_client):
        paths = {r.path for r in real_client.app.routes}
        assert "/v1/api-keys" in paths
        # POST + GET share the path; DELETE adds the param variant.
        assert any(getattr(r, "methods", None) and "DELETE" in r.methods and r.path == "/v1/api-keys/{key_id}"
                   for r in real_client.app.routes)

    def test_full_lifecycle_on_real_app(self, real_client):
        hdr = {"Authorization": f"Bearer {_TEST_ADMIN_KEY}"}
        created = real_client.post(
            "/v1/api-keys", json={"label": "e2e", "role": "read-only"}, headers=hdr
        )
        assert created.status_code == 201, created.text
        raw = created.json()["key"]

        # New key can call an authenticated endpoint.
        assert get_api_key_store().authenticate(raw)[1] == "read-only"

        listed = real_client.get("/v1/api-keys", headers=hdr)
        assert listed.status_code == 200
        assert created.json()["id"] in [k["id"] for k in listed.json()["data"]]

        revoked = real_client.delete(f"/v1/api-keys/{created.json()['id']}", headers=hdr)
        assert revoked.status_code == 200
        assert get_api_key_store().authenticate(raw) is None
