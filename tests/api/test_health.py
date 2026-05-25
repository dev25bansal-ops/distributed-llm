"""Health, readiness, liveness, models, and metrics tests."""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from distllm.api.api_state import g
from distllm.api.server import app


@pytest.fixture(autouse=True)
def _disable_auth(monkeypatch):
    monkeypatch.setenv("DISABLE_AUTH", "1")
    monkeypatch.setenv("DISTLLM_DEV_MODE", "1")
    monkeypatch.delenv("API_KEY", raising=False)


@pytest.fixture
def coordinator():
    coord = MagicMock()
    coord.model_name = "test-model"
    coord.nodes = {}
    coord._shutting_down = False
    coord.list_models = MagicMock(return_value=["test-model", "gpt-3.5-turbo"])
    coord.health_check = MagicMock(return_value={})
    coord.get_metrics = MagicMock(return_value={"requests_total": 42})
    coord.metrics_exporter = None
    coord.scheduler = None
    coord.prefix_cache = None
    return coord


class TestHealth:
    @pytest.fixture(autouse=True)
    def setup(self, coordinator):
        original = g.coordinator
        g.coordinator = coordinator
        yield
        g.coordinator = original

    def test_healthy_when_loaded(self):
        resp = TestClient(app).get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["model"] == "test-model"

    def test_unhealthy_when_no_coordinator(self):
        original = g.coordinator
        g.coordinator = None
        try:
            resp = TestClient(app).get("/health")
            assert resp.status_code == 503
        finally:
            g.coordinator = original


class TestReadiness:
    @pytest.fixture(autouse=True)
    def setup(self, coordinator):
        original = g.coordinator
        g.coordinator = coordinator
        yield
        g.coordinator = original

    def test_ready_when_loaded(self):
        resp = TestClient(app).get("/ready")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ready"

    def test_not_ready_when_no_coordinator(self):
        original = g.coordinator
        g.coordinator = None
        try:
            resp = TestClient(app).get("/ready")
            assert resp.status_code == 503
        finally:
            g.coordinator = original

    def test_not_ready_when_shutting_down(self, coordinator):
        coordinator._shutting_down = True
        resp = TestClient(app).get("/ready")
        assert resp.status_code == 503

    def test_not_ready_when_no_healthy_nodes(self, coordinator):
        coordinator.nodes = {"node1": {}, "node2": {}}
        coordinator.health_check = MagicMock(return_value={
            "node1": {"healthy": False},
            "node2": {"healthy": False},
        })
        resp = TestClient(app).get("/ready")
        assert resp.status_code == 503


class TestLiveness:
    def test_alive(self):
        resp = TestClient(app).get("/live")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "alive"
        assert isinstance(data["uptime_seconds"], float)


class TestListModels:
    @pytest.fixture(autouse=True)
    def setup(self, coordinator):
        original = g.coordinator
        g.coordinator = coordinator
        yield
        g.coordinator = original

    def test_returns_model_list(self):
        resp = TestClient(app).get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 2
        assert data["data"][0]["id"] == "test-model"
        assert data["data"][1]["id"] == "gpt-3.5-turbo"
        assert data["data"][0]["object"] == "model"

    def test_empty_when_no_coordinator(self):
        original = g.coordinator
        g.coordinator = None
        try:
            resp = TestClient(app).get("/v1/models")
            assert resp.status_code == 200
            assert resp.json()["data"] == []
        finally:
            g.coordinator = original


class TestUpdateParams:
    @pytest.fixture(autouse=True)
    def setup(self, coordinator):
        original = g.coordinator
        g.coordinator = coordinator
        coordinator._param_update_channel = MagicMock()
        coordinator._param_update_channel.update = MagicMock()
        yield
        g.coordinator = original

    def test_update_params_success(self, coordinator):
        updated = MagicMock()
        updated.temperature = 0.8
        updated.top_p = 0.95
        updated.top_k = 50
        coordinator._param_update_channel.update.return_value = updated

        resp = TestClient(app).post(
            "/v1/update-params/req-123",
            json={"temperature": 0.8, "top_p": 0.95, "top_k": 50},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["request_id"] == "req-123"
        assert data["temperature"] == 0.8
        assert data["top_p"] == 0.95
        assert data["top_k"] == 50

    def test_update_params_completed_request(self, coordinator):
        coordinator._param_update_channel.update.return_value = None

        resp = TestClient(app).post(
            "/v1/update-params/req-999",
            json={"temperature": 0.5},
        )
        assert resp.status_code == 404
        assert "not found or already completed" in resp.json()["error"]["message"]

    def test_update_params_no_coordinator(self):
        original = g.coordinator
        g.coordinator = None
        try:
            resp = TestClient(app).post(
                "/v1/update-params/req-123",
                json={"temperature": 0.5},
            )
            assert resp.status_code == 503
            assert resp.json()["error"]["message"] == "Coordinator not initialized"
        finally:
            g.coordinator = original


class TestMetrics:
    @pytest.fixture(autouse=True)
    def setup(self, coordinator):
        original = g.coordinator
        g.coordinator = coordinator
        yield
        g.coordinator = original

    def test_returns_metrics_text(self):
        resp = TestClient(app).get("/metrics")
        assert resp.status_code == 200
        assert "distllm_service_up" in resp.text
        assert "distllm_coordinator_loaded" in resp.text

    def test_metrics_without_coordinator(self):
        original = g.coordinator
        g.coordinator = None
        try:
            resp = TestClient(app).get("/metrics")
            assert resp.status_code == 200
            assert "distllm_service_up 0" in resp.text
            assert "distllm_coordinator_loaded 0" in resp.text
        finally:
            g.coordinator = original
