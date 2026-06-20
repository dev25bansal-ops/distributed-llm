"""Adapter (LoRA) loading via adapter parameter tests."""

import os
from unittest.mock import MagicMock

import pytest
import torch
from fastapi.testclient import TestClient

from distllm.api.api_state import g
from distllm.api.server import app


# ---------------------------------------------------------------------------
# Shared helpers (duplicated so each file is self-contained)
# ---------------------------------------------------------------------------

def disable_auth():
    os.environ.pop("API_KEY", None)
    os.environ.pop("API_KEY_WAS_SET", None)
    os.environ["DISABLE_AUTH"] = "1"
    os.environ["DISTLLM_DEV_MODE"] = "1"


def make_mock_coordinator():
    coord = MagicMock()
    coord.model_name = "test-model"
    coord.nodes = {}
    coord.node_order = []
    coord.scheduler = None
    coord.prefix_cache = None
    coord.metrics_exporter = None

    def encode_fn(text, **kwargs):
        tokens = [1, 2, 3, 4, 5]
        if kwargs.get("return_tensors") == "pt":
            return torch.tensor([tokens])
        return tokens

    coord.tokenizer = MagicMock()
    coord.tokenizer.encode.side_effect = encode_fn
    def decode_side_effect(tokens, **kwargs):
        if isinstance(tokens, int):
            token_list = [tokens]
        elif isinstance(tokens, list):
            token_list = tokens
        else:
            token_list = tokens.tolist()
        return " ".join(f"tok-{t}" for t in token_list)
    coord.tokenizer.decode.side_effect = decode_side_effect
    coord.tokenizer.eos_token_id = 0
    coord.generate.return_value = "Hello! This is a test response."

    mock_model = MagicMock()
    mock_model.parameters.side_effect = lambda: iter([torch.randn(10, 10)])
    mock_output = MagicMock()
    mock_output.logits = torch.randn(1, 5, 1000)
    mock_output.past_key_values = MagicMock()
    mock_model.return_value = mock_output
    coord.local_partitioner = MagicMock()
    coord.local_partitioner.full_model = mock_model
    coord.list_models.return_value = ["test-model"]
    coord._vlm_pipeline = None
    coord._spec_decoder = None
    coord._shutting_down = False
    return coord


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

class TestChatAdapter:
    """Adapter (LoRA) loading via adapter parameter."""

    VALID_ADAPTER = "my-lora"

    @pytest.fixture(autouse=True)
    def _setup(self):
        disable_auth()
        coord = make_mock_coordinator()
        coord.adapter_manager = MagicMock()
        coord.adapter_manager.list_adapters.return_value = [self.VALID_ADAPTER, "other-lora"]
        original = g.coordinator
        g.coordinator = coord
        self._coord = coord
        yield
        g.coordinator = original
        os.environ.pop("DISABLE_AUTH", None)
        os.environ.pop("DISTLLM_DEV_MODE", None)

    def test_adapter_returns_200(self):
        resp = TestClient(app).post(
            "/v1/chat/completions",
            json={
                "model": "distributed-llm",
                "messages": [{"role": "user", "content": "Hello"}],
                "adapter": self.VALID_ADAPTER,
                "max_tokens": 10,
            },
        )
        assert resp.status_code == 200

    def test_invalid_adapter_returns_400(self):
        resp = TestClient(app).post(
            "/v1/chat/completions",
            json={
                "model": "distributed-llm",
                "messages": [{"role": "user", "content": "Hello"}],
                "adapter": "nonexistent-adapter",
                "max_tokens": 10,
            },
        )
        assert resp.status_code == 400

    def test_invalid_adapter_error_message(self):
        resp = TestClient(app).post(
            "/v1/chat/completions",
            json={
                "model": "distributed-llm",
                "messages": [{"role": "user", "content": "Hello"}],
                "adapter": "nonexistent-adapter",
                "max_tokens": 10,
            },
        )
        data = resp.json()
        assert "nonexistent-adapter" in data.get("detail", str(data))

    def test_adapter_without_manager_ignored(self):
        coord = make_mock_coordinator()
        coord.adapter_manager = None
        original = g.coordinator
        g.coordinator = coord
        try:
            resp = TestClient(app).post(
                "/v1/chat/completions",
                json={
                    "model": "distributed-llm",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "adapter": self.VALID_ADAPTER,
                    "max_tokens": 10,
                },
            )
            assert resp.status_code == 200
        finally:
            g.coordinator = original

    def test_adapter_no_adapter_provided_still_works(self):
        resp = TestClient(app).post(
            "/v1/chat/completions",
            json={
                "model": "distributed-llm",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 10,
            },
        )
        assert resp.status_code == 200
