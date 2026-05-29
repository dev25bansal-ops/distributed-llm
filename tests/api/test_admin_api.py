"""Tests for the REST Admin API (routes/admin.py).

Uses the role-based API key auth system.
"""

import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi import Request
from fastapi.testclient import TestClient


# Test admin key used throughout
_TEST_ADMIN_KEY = "test-admin-key-12345"

# Patch torch import before any distllm imports to speed things up
with patch.dict("sys.modules", {"torch": MagicMock(), "torch.cuda": MagicMock()}):
    pass


from distllm.core.api_key_store import reset_api_key_store


@pytest.fixture(autouse=True)
def _reset_store():
    """Reset the singleton key store before each test (re-reads from env)."""
    os.environ["API_KEYS"] = (
        '{"keys": ['
        '{"key": "' + _TEST_ADMIN_KEY + '", "role": "admin", "label": "test-admin"},'
        '{"key": "test-read-key", "role": "read-only", "label": "test-read"},'
        '{"key": "test-inf-key", "role": "inference-only", "label": "test-inf"}'
        "]}"
    )
    reset_api_key_store()
    yield
    reset_api_key_store()


@pytest.fixture(autouse=True)
def _patch_state():
    """Patch g.coordinator with a mock coordinator for all tests."""
    from distllm.api.api_state import _state
    coord = MagicMock()
    coord.model_name = "test-model"
    coord.total_layers = 24
    coord.max_batch_size = 4
    coord.max_tokens_per_batch = 4096
    coord.scheduler = MagicMock()
    coord.scheduler.default_temperature = 0.7
    coord.scheduler.default_top_p = 0.9
    coord.scheduler.default_top_k = 50

    node1 = MagicMock()
    node1.node_id = "node-0"
    node1.host = "10.0.0.1"
    node1.port = 50051
    node1.healthy = True
    node1.start_layer = 0
    node1.end_layer = 11
    node1.gpu_name = "Tesla V100"
    node1.gpu_memory_total = 16384
    node1.gpu_memory_free = 8192
    node1.gpu_sm_count = 80
    node1.role = "AUTO"
    node1.cluster_id = "default"
    node1.last_health_time = 1000.0

    node2 = MagicMock()
    node2.node_id = "node-1"
    node2.host = "10.0.0.2"
    node2.port = 50051
    node2.healthy = False
    node2.start_layer = 12
    node2.end_layer = 23
    node2.gpu_name = "Tesla V100"
    node2.gpu_memory_total = 16384
    node2.gpu_memory_free = 0
    node2.gpu_sm_count = 80
    node2.role = "AUTO"
    node2.cluster_id = "default"
    node2.last_health_time = 1000.0

    coord.nodes = {"node-0": node1, "node-1": node2}
    coord.node_order = ["node-0", "node-1"]

    _state.coordinator = coord
    yield
    _state.coordinator = None


@pytest.fixture
def auth_header():
    """Valid admin Authorization header."""
    return {"Authorization": f"Bearer {_TEST_ADMIN_KEY}"}


@pytest.fixture
def client():
    """FastAPI TestClient for the admin router with auth middleware."""
    from distllm.api.routes.admin import router
    from fastapi import FastAPI
    from starlette.middleware.base import BaseHTTPMiddleware
    from distllm.core.api_key_store import get_api_key_store

    app = FastAPI()

    class _TestAuthMiddleware(BaseHTTPMiddleware):
        """Simulate AuthMiddleware: validate Bearer token, set api_key_role."""

        async def dispatch(self, request: Request, call_next):
            auth = request.headers.get("Authorization", "")
            if auth.startswith("Bearer "):
                token = auth[7:]
                store = get_api_key_store()
                result = store.authenticate(token)
                if result:
                    request.state.api_key_role = result[1]
                    request.state.api_key_id = result[0]
            return await call_next(request)

    app.add_middleware(_TestAuthMiddleware)
    app.include_router(router)
    return TestClient(app)


# ── Auth tests ──────────────────────────────────────────────────────────────


class TestAdminAuth:
    """Verify admin API key authentication."""

    def test_no_auth_returns_401(self, client):
        resp = client.get("/admin/v1/nodes")
        assert resp.status_code == 401

    def test_bad_auth_returns_401(self, client):
        resp = client.get("/admin/v1/nodes", headers={"Authorization": "Bearer wrong-key"})
        assert resp.status_code == 401

    def test_wrong_scheme_returns_401(self, client):
        resp = client.get("/admin/v1/nodes", headers={"Authorization": "Basic xyz"})
        assert resp.status_code == 401

    def test_valid_auth_succeeds(self, client, auth_header):
        resp = client.get("/admin/v1/nodes", headers=auth_header)
        assert resp.status_code == 200

    def test_read_only_key_rejected(self, client):
        resp = client.get("/admin/v1/nodes", headers={"Authorization": "Bearer test-read-key"})
        assert resp.status_code == 403

    def test_inference_key_rejected(self, client):
        resp = client.get("/admin/v1/nodes", headers={"Authorization": "Bearer test-inf-key"})
        assert resp.status_code == 403


# ── Node listing ────────────────────────────────────────────────────────────


class TestListNode:
    """GET /admin/v1/nodes"""

    def test_returns_node_list(self, client, auth_header):
        resp = client.get("/admin/v1/nodes", headers=auth_header)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_nodes"] == 2
        assert data["healthy_count"] == 1
        assert data["total_layers"] == 24
        assert len(data["nodes"]) == 2

    def test_node_fields(self, client, auth_header):
        resp = client.get("/admin/v1/nodes", headers=auth_header)
        nodes = resp.json()["nodes"]
        node0 = next(n for n in nodes if n["node_id"] == "node-0")
        assert node0["host"] == "10.0.0.1"
        assert node0["port"] == 50051
        assert node0["healthy"] is True
        assert node0["start_layer"] == 0
        assert node0["end_layer"] == 11
        assert node0["gpu_memory_total"] == 16384
        assert node0["gpu_memory_util_pct"] > 0

    def test_unhealthy_node(self, client, auth_header):
        resp = client.get("/admin/v1/nodes", headers=auth_header)
        node1 = next(n for n in resp.json()["nodes"] if n["node_id"] == "node-1")
        assert node1["healthy"] is False

    def test_no_coordinator_returns_503(self):
        from distllm.api.api_state import _state
        _state.coordinator = None
        from distllm.api.routes.admin import router
        from fastapi import FastAPI
        from starlette.middleware.base import BaseHTTPMiddleware
        from distllm.core.api_key_store import get_api_key_store
        from fastapi import Request

        app = FastAPI()

        class _TestAuthMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request: Request, call_next):
                auth = request.headers.get("Authorization", "")
                if auth.startswith("Bearer "):
                    token = auth[7:]
                    store = get_api_key_store()
                    result = store.authenticate(token)
                    if result:
                        request.state.api_key_role = result[1]
                return await call_next(request)

        app.add_middleware(_TestAuthMiddleware)
        app.include_router(router)
        c = TestClient(app)
        resp = c.get("/admin/v1/nodes", headers={"Authorization": f"Bearer {_TEST_ADMIN_KEY}"})
        assert resp.status_code == 503


# ── Drain / Undrain ─────────────────────────────────────────────────────────


class TestDrainNode:
    """POST /admin/v1/nodes/{node_id}/drain"""

    def test_drain_existing_node(self, client, auth_header):
        resp = client.post("/admin/v1/nodes/node-0/drain", headers=auth_header)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["node_id"] == "node-0"

    def test_drain_nonexistent_node(self, client, auth_header):
        resp = client.post("/admin/v1/nodes/non-existent/drain", headers=auth_header)
        assert resp.status_code == 404

    def test_undrain_node(self, client, auth_header):
        resp = client.post("/admin/v1/nodes/node-0/undrain", headers=auth_header)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["node_id"] == "node-0"


# ── Offline / Recover ──────────────────────────────────────────────────────


class TestOfflineNode:
    """POST /admin/v1/nodes/{node_id}/offline"""

    def test_offline_existing_node(self, client, auth_header):
        resp = client.post("/admin/v1/nodes/node-0/offline", headers=auth_header)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["node_id"] == "node-0"

    def test_offline_nonexistent_node(self, client, auth_header):
        resp = client.post("/admin/v1/nodes/non-existent/offline", headers=auth_header)
        assert resp.status_code == 404


class TestRecoverNode:
    """POST /admin/v1/nodes/{node_id}/recover"""

    def test_recover_existing_node(self, client, auth_header):
        resp = client.post("/admin/v1/nodes/node-1/recover", headers=auth_header)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["node_id"] == "node-1"

    def test_recover_nonexistent_node(self, client, auth_header):
        resp = client.post("/admin/v1/nodes/non-existent/recover", headers=auth_header)
        assert resp.status_code == 404


# ── Config update ──────────────────────────────────────────────────────────


class TestUpdateConfig:
    """PATCH /admin/v1/config"""

    def test_update_temperature(self, client, auth_header):
        resp = client.patch(
            "/admin/v1/config",
            json={"temperature": 0.5, "top_p": 0.95},
            headers=auth_header,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "temperature" in data["updated"]

    def test_update_batch_size(self, client, auth_header):
        resp = client.patch(
            "/admin/v1/config",
            json={"max_batch_size": 8, "max_tokens_per_batch": 2048},
            headers=auth_header,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["updated"]["max_batch_size"] == 8

    def test_empty_update(self, client, auth_header):
        resp = client.patch("/admin/v1/config", json={}, headers=auth_header)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"


# ── Logs ───────────────────────────────────────────────────────────────────


class TestViewLogs:
    """GET /admin/v1/logs"""

    def test_returns_logs(self, client, auth_header):
        resp = client.get("/admin/v1/logs", headers=auth_header)
        assert resp.status_code == 200
        data = resp.json()
        assert "logs" in data
        assert "total" in data

    def test_logs_with_search(self, client, auth_header):
        resp = client.get("/admin/v1/logs?search=test&level=INFO&limit=10", headers=auth_header)
        assert resp.status_code == 200


# ── Compress ───────────────────────────────────────────────────────────────


class TestCompressModel:
    """POST /admin/v1/compress"""

    def test_compress_default_model(self, client, auth_header):
        resp = client.post("/admin/v1/compress", json={}, headers=auth_header)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "started"
        assert data["model_name"] == "test-model"
        assert data["method"] == "int4"

    def test_compress_with_custom_params(self, client, auth_header):
        resp = client.post(
            "/admin/v1/compress",
            json={"model_name": "custom-model", "method": "int8"},
            headers=auth_header,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_name"] == "custom-model"
        assert data["method"] == "int8"


# ── Integration: server.py registration ───────────────────────────────────


class TestServerRegistration:
    """Verify the admin router is properly registered in server.py."""

    def test_admin_router_in_server(self):
        from distllm.api.server import app
        routes = [r.path for r in app.routes]
        admin_paths = [p for p in routes if p.startswith("/admin/v1")]
        assert len(admin_paths) >= 7
        assert "/admin/v1/nodes" in admin_paths
        assert "/admin/v1/config" in admin_paths
        assert "/admin/v1/logs" in admin_paths
        assert "/admin/v1/compress" in admin_paths
