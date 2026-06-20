"""SSRF protection tests: image_url with internal addresses rejected."""

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

class TestChatSSRF:
    """SSRF protection: image_url with internal addresses rejected."""

    IMAGE_MSG = [
        {"type": "text", "text": "What's in this image?"},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}},
    ]

    @pytest.fixture(autouse=True)
    def auth(self):
        disable_auth()
        yield
        os.environ.pop("DISABLE_AUTH", None)
        os.environ.pop("DISTLLM_DEV_MODE", None)

    @staticmethod
    def _req(url: str):
        return TestClient(app).post(
            "/v1/chat/completions",
            json={
                "model": "distributed-llm",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe"},
                            {"type": "image_url", "image_url": {"url": url}},
                        ],
                    },
                ],
                "max_tokens": 10,
            },
        )

    def test_public_url_allowed(self):
        coord = make_mock_coordinator()
        original = g.coordinator
        g.coordinator = coord
        try:
            resp = self._req("https://example.com/image.jpg")
            assert resp.status_code == 200
        finally:
            g.coordinator = original

    def test_base64_data_uri_allowed(self):
        coord = make_mock_coordinator()
        original = g.coordinator
        g.coordinator = coord
        try:
            resp = self._req("data:image/png;base64,iVBORw0KGgo=")
            assert resp.status_code == 200
        finally:
            g.coordinator = original

    def test_localhost_hostname_rejected(self):
        resp = self._req("http://localhost/image.png")
        assert resp.status_code == 422

    def test_localhost_ip_rejected(self):
        resp = self._req("http://127.0.0.1/image.png")
        assert resp.status_code == 422

    def test_localhost_ipv6_rejected(self):
        resp = self._req("http://[::1]/image.png")
        assert resp.status_code == 422

    def test_private_10_dot_rejected(self):
        resp = self._req("http://10.0.0.1/image.png")
        assert resp.status_code == 422

    def test_private_172_dot_rejected(self):
        resp = self._req("http://172.16.0.1/image.png")
        assert resp.status_code == 422

    def test_private_192_dot_rejected(self):
        resp = self._req("http://192.168.1.1/image.png")
        assert resp.status_code == 422

    def test_link_local_rejected(self):
        resp = self._req("http://169.254.1.1/image.png")
        assert resp.status_code == 422

    def test_public_ip_allowed(self):
        coord = make_mock_coordinator()
        original = g.coordinator
        g.coordinator = coord
        try:
            resp = self._req("http://8.8.8.8/image.png")
            assert resp.status_code == 200
        finally:
            g.coordinator = original
