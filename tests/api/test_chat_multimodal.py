"""Multi-modal (image input) tests."""

import os
import secrets
from unittest.mock import MagicMock

import pytest
import torch
from fastapi.testclient import TestClient

from distllm.api.api_state import g
from distllm.core.api_key_store import reset_api_key_store
from distllm.api.server import app


# ---------------------------------------------------------------------------
# Shared helpers (duplicated so each file is self-contained)
# ---------------------------------------------------------------------------

def _make_client():
    test_api_key = secrets.token_urlsafe(32)
    os.environ.pop("API_KEY_WAS_SET", None)
    os.environ["API_KEY"] = test_api_key
    reset_api_key_store()
    client = TestClient(app)
    client.headers["Authorization"] = f"Bearer {test_api_key}"
    return client


def _cleanup_auth():
    os.environ.pop("API_KEY", None)
    os.environ.pop("API_KEY_WAS_SET", None)
    reset_api_key_store()


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
    coord._model_router = None
    coord._shutting_down = False
    coord.tokenizer.chat_template = None
    return coord


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

class TestChatMultiModal:
    """Multi-modal (image input) tests."""

    @pytest.fixture(autouse=True)
    def setup(self):
        coord = make_mock_coordinator()
        coord.generate.return_value = "A sunny day at the beach with blue sky and ocean waves."
        vlm = MagicMock()
        vlm.is_multimodal_message.return_value = True
        vlm.parse_messages.return_value = ("What's in this image?", ["embeddings"])
        vlm.encode_images_to_embeddings.return_value = ["embed"]
        vlm.build_prompt_with_images.return_value = (
            "user: What's in this image?\nassistant: A sunny day at the beach with blue sky and ocean waves.",
            None,
        )
        coord._vlm_pipeline = vlm
        original = g.coordinator
        g.coordinator = coord
        self._coord = coord
        self.client = _make_client()
        yield
        g.coordinator = original
        _cleanup_auth()

    def test_image_input_returns_200(self):
        resp = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "distributed-llm",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What's in this image?"},
                            {"type": "image_url", "image_url": {"url": "https://example.com/beach.jpg"}},
                        ],
                    },
                ],
                "max_tokens": 50,
            },
        )
        assert resp.status_code == 200

    def test_image_input_response_content(self):
        resp = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "distributed-llm",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What's in this image?"},
                            {"type": "image_url", "image_url": {"url": "https://example.com/beach.jpg"}},
                        ],
                    },
                ],
                "max_tokens": 50,
            },
        )
        content = resp.json()["choices"][0]["message"]["content"]
        assert isinstance(content, str)
        assert len(content) > 0

    def test_image_input_triggers_vlm_pipeline(self):
        resp = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "distributed-llm",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What's in this image?"},
                            {"type": "image_url", "image_url": {"url": "https://example.com/beach.jpg"}},
                        ],
                    },
                ],
                "max_tokens": 50,
            },
        )
        assert self._coord._vlm_pipeline.parse_messages.called
        assert self._coord._vlm_pipeline.encode_images_to_embeddings.called
        assert self._coord._vlm_pipeline.build_prompt_with_images.called

    def test_image_input_without_vlm_falls_back(self):
        coord = make_mock_coordinator()
        coord._vlm_pipeline = None
        original = g.coordinator
        g.coordinator = coord
        client = _make_client()
        try:
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "distributed-llm",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "What's in this image?"},
                                {"type": "image_url", "image_url": {"url": "https://example.com/beach.jpg"}},
                            ],
                        },
                    ],
                    "max_tokens": 50,
                },
            )
            assert resp.status_code == 200
            content = resp.json()["choices"][0]["message"]["content"]
            assert isinstance(content, str)
        finally:
            g.coordinator = original
