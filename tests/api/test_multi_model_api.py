"""Tests for multi-model API endpoints."""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import distllm.api.server as server_module
from distllm.api.server import app


class TestMultiModelAPI:
    """Tests for multi-model API support."""

    def test_list_models_single_model(self):
        """Returns single model when no multi-model registry."""
        coord = MagicMock()
        coord.model_name = "test-model"
        coord._shutting_down = False
        # No list_models method
        del coord.list_models

        server_module.coordinator = coord

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

        server_module.coordinator = coord

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

        server_module.coordinator = coord

        client = TestClient(app)
        response = client.post("/v1/chat/completions", json={
            "model": "unknown-model",
            "messages": [{"role": "user", "content": "hello"}],
        })

        assert response.status_code == 400
        assert "not found" in response.json()["message"]

    def test_chat_accepts_valid_model(self):
        """Accepts a model that exists in registry."""
        coord = MagicMock()
        coord.model_name = "primary"
        coord._shutting_down = False
        coord.list_models.return_value = ["primary", "model-a"]
        coord.generate.return_value = "test response"
        coord.scheduler = None

        server_module.coordinator = coord

        client = TestClient(app)
        response = client.post("/v1/chat/completions", json={
            "model": "model-a",
            "messages": [{"role": "user", "content": "hello"}],
        })

        assert response.status_code == 200
