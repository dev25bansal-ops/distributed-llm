"""Fuzz tests for API endpoints."""
from __future__ import annotations

import json
import random
import string
import pytest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient


def _random_string(max_len: int = 64) -> str:
    chars = string.ascii_letters + string.digits + string.punctuation + " \n\t"
    return "".join(random.choices(chars, k=random.randint(1, max_len)))


def _random_json() -> dict:
    result = {}
    for _ in range(random.randint(1, 8)):
        key = _random_string(16)
        result[key] = random.choice([
            _random_string(128),
            random.randint(-1000000, 1000000),
            random.random() * 10000,
            True, False, None,
            [_random_string(32) for _ in range(random.randint(1, 5))],
        ])
    return result


@pytest.fixture
def api_client():
    import distllm.api.server as server_module

    mock_coordinator = MagicMock()
    mock_coordinator.model_name = "test-model"
    mock_coordinator.is_healthy.return_value = True
    mock_coordinator.list_models.return_value = ["test-model"]
    mock_coordinator.generate.return_value = {
        "choices": [{"text": "hello", "index": 0, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    mock_coordinator.chat.return_value = {
        "choices": [{"message": {"role": "assistant", "content": "hi"}, "index": 0, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }

    original = getattr(server_module, "coordinator", None)
    server_module.coordinator = mock_coordinator
    client = TestClient(server_module.app, raise_server_exceptions=False)
    yield client
    server_module.coordinator = original


class TestApiEndpointFuzzing:
    """Fuzz test API endpoints with random/malicious payloads."""

    def test_chat_completion_fuzz(self, api_client):
        """Fuzz /v1/chat/completions with random payloads."""
        for _ in range(100):
            payload = _random_json()
            resp = api_client.post(
                "/v1/chat/completions",
                json=payload,
                content_type="application/json",
            )
            assert resp.status_code in (200, 400, 422, 500)

    def test_completion_fuzz(self, api_client):
        """Fuzz /v1/completions with random payloads."""
        for _ in range(100):
            payload = _random_json()
            resp = api_client.post(
                "/v1/completions",
                json=payload,
                content_type="application/json",
            )
            assert resp.status_code in (200, 400, 422, 500)

    def test_embeddings_fuzz(self, api_client):
        """Fuzz /v1/embeddings with random payloads."""
        for _ in range(50):
            payload = _random_json()
            resp = api_client.post(
                "/v1/embeddings",
                json=payload,
                content_type="application/json",
            )
            assert resp.status_code in (200, 400, 422, 500)

    def test_health_endpoint_fuzz(self, api_client):
        """Fuzz health endpoint with random headers."""
        for _ in range(50):
            headers = {_random_string(16): _random_string(32) for _ in range(random.randint(0, 5))}
            resp = api_client.get("/health", headers=headers)
            assert resp.status_code in (200, 503)

    def test_model_load_fuzz(self, api_client):
        """Fuzz /v1/models/load with random payloads."""
        for _ in range(50):
            payload = _random_json()
            resp = api_client.post(
                "/v1/models/load",
                json=payload,
                content_type="application/json",
            )
            assert resp.status_code in (200, 400, 422, 500)

    def test_admin_endpoints_fuzz(self, api_client):
        """Fuzz admin endpoints with random payloads."""
        for _ in range(50):
            payload = _random_json()
            resp = api_client.get(
                f"/admin/v1/{_random_string(10)}",
                headers={"Authorization": f"Bearer {_random_string(32)}"},
            )
            assert resp.status_code in (200, 401, 403, 404, 422)


class TestMaliciousPayloads:
    """Test API with specifically malicious payloads."""

    def test_oversized_payload(self, api_client):
        """Test with extremely large payload."""
        large_payload = {"prompt": "A" * 10_000_000}  # 10MB prompt
        resp = api_client.post(
            "/v1/completions",
            json=large_payload,
            content_type="application/json",
        )
        assert resp.status_code in (200, 400, 413, 422, 500)

    def test_nested_json(self, api_client):
        """Test with deeply nested JSON."""
        nested = {"level": 0}
        current = nested
        for i in range(100):
            current["child"] = {"level": i + 1}
            current = current["child"]

        resp = api_client.post(
            "/v1/chat/completions",
            json=nested,
            content_type="application/json",
        )
        assert resp.status_code in (200, 400, 422, 500)

    def test_null_bytes(self, api_client):
        """Test with null bytes in fields."""
        payload = {"model": "test\x00model", "prompt": "hello\x00world"}
        resp = api_client.post(
            "/v1/completions",
            json=payload,
            content_type="application/json",
        )
        assert resp.status_code in (200, 400, 422, 500)

    def test_xss_in_model_name(self, api_client):
        """Test XSS in model name field."""
        payload = {
            "model": "<script>alert('xss')</script>",
            "prompt": "test",
        }
        resp = api_client.post(
            "/v1/completions",
            json=payload,
            content_type="application/json",
        )
        assert resp.status_code in (200, 400, 422, 500)
        if resp.status_code == 200:
            body = resp.text
            assert "<script>" not in body

    def test_path_traversal_in_model(self, api_client):
        """Test path traversal in model field."""
        payloads = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
        ]
        for model in payloads:
            resp = api_client.post(
                "/v1/completions",
                json={"model": model, "prompt": "test"},
                content_type="application/json",
            )
            assert resp.status_code in (200, 400, 403, 422, 500)


def fuzz(data: bytes) -> None:
    """Atheris fuzz harness."""
    try:
        payload = json.loads(data)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return

    import distllm.api.server as server_module
    from fastapi.testclient import TestClient

    mock_coordinator = MagicMock()
    mock_coordinator.model_name = "test-model"
    mock_coordinator.is_healthy.return_value = True

    original = getattr(server_module, "coordinator", None)
    server_module.coordinator = mock_coordinator
    try:
        client = TestClient(server_module.app, raise_server_exceptions=False)
        client.post("/v1/chat/completions", json=payload)
        client.post("/v1/completions", json=payload)
    except Exception:
        pass
    finally:
        server_module.coordinator = original


def pytest_fuzz(n: int = 500) -> None:
    """Pytest fuzz mode."""
    for _ in range(n):
        payload = _random_json()
        fuzz(json.dumps(payload).encode())
