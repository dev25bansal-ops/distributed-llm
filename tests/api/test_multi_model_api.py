"""Tests for multi-model API endpoints."""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from distllm.api.api_state import g as api_g
from distllm.api.server import app


@pytest.fixture(autouse=True)
def _disable_auth(monkeypatch):
    monkeypatch.setenv("DISABLE_AUTH", "1")
    monkeypatch.setenv("DISTLLM_DEV_MODE", "1")
    monkeypatch.delenv("API_KEY", raising=False)


class TestMultiModelAPI:
    """Tests for multi-model API support."""

    def test_list_models_single_model(self):
        """Returns single model when no multi-model registry."""
        coord = MagicMock()
        coord.model_name = "test-model"
        coord._shutting_down = False
        # No list_models method
        del coord.list_models

        api_g.coordinator = coord

        client = TestClient(app)
        response = client.get("/v1/models")

        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 1
        assert data["data"][0]["id"] == "test-model"

    def test_list_models_multiple_models(self):
        """Returns all registered models."""
        coord = MagicMock()
        coord.model_name = "primary"
        coord._shutting_down = False
        coord.list_models.return_value = ["primary", "model-a", "model-b"]

        api_g.coordinator = coord

        client = TestClient(app)
        response = client.get("/v1/models")

        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 3
        names = {m["id"] for m in data["data"]}
        assert names == {"primary", "model-a", "model-b"}

    def test_chat_validates_unknown_model(self):
        """Returns 400 for unknown model name."""
        coord = MagicMock()
        coord.model_name = "primary"
        coord._shutting_down = False
        coord.list_models.return_value = ["primary", "model-a"]
        coord.generate.return_value = "test response"
        coord.scheduler = None
        coord._vlm_pipeline = None
        coord._spec_decoder = None
        coord.tokenizer = MagicMock()
        coord.tokenizer.encode.return_value = [1, 2, 3]

        api_g.coordinator = coord

        client = TestClient(app)
        response = client.post("/v1/chat/completions", json={
            "model": "unknown-model",
            "messages": [{"role": "user", "content": "hello"}],
        })

        assert response.status_code == 400
        assert "not found" in response.json()["error"]["message"]

    def test_chat_accepts_valid_model(self):
        """Accepts a model that exists in registry."""
        coord = MagicMock()
        coord.model_name = "primary"
        coord._shutting_down = False
        coord.list_models.return_value = ["primary", "model-a"]
        coord.generate.return_value = "test response"
        coord.scheduler = None
        coord._vlm_pipeline = None
        coord._spec_decoder = None
        coord.tokenizer = MagicMock()
        coord.tokenizer.encode.return_value = [1, 2, 3]

        api_g.coordinator = coord

        client = TestClient(app)
        response = client.post("/v1/chat/completions", json={
            "model": "model-a",
            "messages": [{"role": "user", "content": "hello"}],
        })

        assert response.status_code == 200


class TestLoadModel:
    @pytest.fixture(autouse=True)
    def setup(self):
        original = api_g.coordinator
        self.coord = MagicMock()
        self.coord.model_name = "test-model"
        self.coord.nodes = {}
        self.coord._shutting_down = False
        msm = MagicMock()
        msm.register_model = MagicMock()
        self.coord._model_hotswap = msm
        api_g.coordinator = self.coord
        yield
        api_g.coordinator = original

    def test_load_model_success(self):
        self.coord._model_hotswap.load_model.return_value = True
        resp = TestClient(app).post(
            "/v1/models/gpt-4/load",
            json={"model_path": "openai/gpt-4", "total_layers": 32},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "loaded"
        assert data["model_id"] == "gpt-4"

    def test_load_model_insufficient_memory(self):
        self.coord._model_hotswap.load_model.return_value = False
        resp = TestClient(app).post(
            "/v1/models/gpt-4/load",
            json={"model_path": "openai/gpt-4", "total_layers": 32},
        )
        assert resp.status_code == 507

    def test_load_no_hotswap(self):
        self.coord._model_hotswap = None
        resp = TestClient(app).post(
            "/v1/models/gpt-4/load",
            json={"model_path": "openai/gpt-4", "total_layers": 32},
        )
        assert resp.status_code == 503

    def test_load_no_coordinator(self):
        original = api_g.coordinator
        api_g.coordinator = None
        try:
            resp = TestClient(app).post(
                "/v1/models/gpt-4/load",
                json={"model_path": "openai/gpt-4"},
            )
            assert resp.status_code == 503
        finally:
            api_g.coordinator = original

    def test_load_auto_detect_fails(self):
        resp = TestClient(app).post(
            "/v1/models/test-model/load",
            json={"model_path": "this-model-does-not-exist-12345"},
        )
        assert resp.status_code == 400
        err = resp.json()["error"]
        assert "auto-detect" in err["message"].lower() or "could not" in err["message"].lower()


class TestUnloadModel:
    @pytest.fixture(autouse=True)
    def setup(self):
        original = api_g.coordinator
        self.coord = MagicMock()
        self.coord.model_name = "test-model"
        self.coord.nodes = {}
        self.coord._shutting_down = False
        msm = MagicMock()
        self.coord._model_hotswap = msm
        api_g.coordinator = self.coord
        yield
        api_g.coordinator = original

    def test_unload_model_success(self):
        self.coord._model_hotswap.unload_model.return_value = True
        resp = TestClient(app).post("/v1/models/gpt-4/unload")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "unloaded"
        assert data["model_id"] == "gpt-4"

    def test_unload_model_not_found(self):
        self.coord._model_hotswap.unload_model.return_value = False
        resp = TestClient(app).post("/v1/models/gpt-4/unload")
        assert resp.status_code == 404

    def test_unload_no_hotswap(self):
        self.coord._model_hotswap = None
        resp = TestClient(app).post("/v1/models/gpt-4/unload")
        assert resp.status_code == 503

    def test_unload_no_coordinator(self):
        original = api_g.coordinator
        api_g.coordinator = None
        try:
            resp = TestClient(app).post("/v1/models/gpt-4/unload")
            assert resp.status_code == 503
        finally:
            api_g.coordinator = original


class TestRemoveModel:
    @pytest.fixture(autouse=True)
    def setup(self):
        original = api_g.coordinator
        self.coord = MagicMock()
        self.coord.model_name = "test-model"
        self.coord.nodes = {}
        self.coord._shutting_down = False
        msm = MagicMock()
        self.coord._model_hotswap = msm
        api_g.coordinator = self.coord
        yield
        api_g.coordinator = original

    def test_remove_model_success(self):
        self.coord._model_hotswap.remove_model.return_value = True
        resp = TestClient(app).delete("/v1/models/gpt-4")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "removed"
        assert data["model_id"] == "gpt-4"

    def test_remove_model_not_found(self):
        self.coord._model_hotswap.remove_model.return_value = False
        resp = TestClient(app).delete("/v1/models/gpt-4")
        assert resp.status_code == 404

    def test_remove_no_hotswap(self):
        self.coord._model_hotswap = None
        resp = TestClient(app).delete("/v1/models/gpt-4")
        assert resp.status_code == 503

    def test_remove_no_coordinator(self):
        original = api_g.coordinator
        api_g.coordinator = None
        try:
            resp = TestClient(app).delete("/v1/models/gpt-4")
            assert resp.status_code == 503
        finally:
            api_g.coordinator = original


class TestMemoryBudget:
    @pytest.fixture(autouse=True)
    def setup(self):
        original = api_g.coordinator
        self.coord = MagicMock()
        self.coord.model_name = "test-model"
        self.coord.nodes = {}
        self.coord._shutting_down = False
        msm = MagicMock()
        msm.memory_budget = MagicMock()
        self.coord._model_hotswap = msm
        api_g.coordinator = self.coord
        yield
        api_g.coordinator = original

    def test_set_memory_budget(self):
        resp = TestClient(app).post(
            "/v1/models/memory/budget",
            params={"model_id": "gpt-4"},
            json={"budget_gb": 8.0},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "updated"
        assert data["budget_gb"] == 8.0
        self.coord._model_hotswap.memory_budget.set_budget.assert_called_once_with("gpt-4", 8.0)

    def test_set_memory_budget_negative(self):
        resp = TestClient(app).post(
            "/v1/models/memory/budget",
            params={"model_id": "gpt-4"},
            json={"budget_gb": -1},
        )
        assert resp.status_code == 400

    def test_set_memory_budget_no_value(self):
        resp = TestClient(app).post(
            "/v1/models/memory/budget",
            params={"model_id": "gpt-4"},
            json={},
        )
        assert resp.status_code == 400

    def test_memory_budget_no_hotswap(self):
        self.coord._model_hotswap = None
        resp = TestClient(app).post(
            "/v1/models/memory/budget",
            params={"model_id": "gpt-4"},
            json={"budget_gb": 8.0},
        )
        assert resp.status_code == 503


class TestHotSwap:
    @pytest.fixture(autouse=True)
    def setup(self):
        original = api_g.coordinator
        self.coord = MagicMock()
        self.coord.model_name = "test-model"
        self.coord.nodes = {}
        self.coord._shutting_down = False
        msm = MagicMock()
        msm.register_model = MagicMock()
        msm.load_model.return_value = True
        self.coord._model_hotswap = msm
        api_g.coordinator = self.coord
        yield
        api_g.coordinator = original

    def test_hot_swap_sequential_load(self):
        resp_a = TestClient(app).post(
            "/v1/models/model-a/load",
            json={"model_path": "path/a", "total_layers": 12},
        )
        resp_b = TestClient(app).post(
            "/v1/models/model-b/load",
            json={"model_path": "path/b", "total_layers": 24},
        )
        assert resp_a.status_code == 200
        assert resp_b.status_code == 200
        assert resp_a.json()["model_id"] == "model-a"
        assert resp_b.json()["model_id"] == "model-b"

    def test_hot_swap_unload_between_loads(self):
        self.coord._model_hotswap.unload_model.return_value = True
        resp_load = TestClient(app).post(
            "/v1/models/model-a/load",
            json={"model_path": "path/a", "total_layers": 12},
        )
        assert resp_load.status_code == 200
        resp_unload = TestClient(app).post("/v1/models/model-a/unload")
        assert resp_unload.status_code == 200
        resp_load_again = TestClient(app).post(
            "/v1/models/model-a/load",
            json={"model_path": "path/a", "total_layers": 12},
        )
        assert resp_load_again.status_code == 200
