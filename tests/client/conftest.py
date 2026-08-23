"""Fixtures for DistLLM client tests.

Uses the ``load_module`` pattern to bypass ``distllm/__init__.py`` and its
circular import chain, then provides sample response payloads for the test suite.

No MagicMock -- real httpx transport and data payloads.
"""

from __future__ import annotations

import httpx
import pytest

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_client_mod = load_module("distllm/client/client.py")
DistLLMClient = _client_mod.DistLLMClient
SyncDistLLMClient = _client_mod.SyncDistLLMClient
CompletionResponse = _client_mod.CompletionResponse
ChatResponse = _client_mod.ChatResponse
ModelInfo = _client_mod.ModelInfo
NodeInfo = _client_mod.NodeInfo
ClusterMetrics = _client_mod.ClusterMetrics


# -- Sample response payloads -----------------------------------------------


@pytest.fixture
def completion_payload() -> dict:
    """Standard /v1/completions response body."""
    return {
        "id": "cmpl-abc123",
        "object": "text_completion",
        "model": "test-model",
        "choices": [
            {"text": "Hello world", "index": 0, "finish_reason": "length"},
            {"text": "Second choice", "index": 1, "finish_reason": "stop"},
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
    }


@pytest.fixture
def chat_payload() -> dict:
    """Standard /v1/chat/completions response body."""
    return {
        "id": "chatcmpl-xyz789",
        "object": "chat.completion",
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "I am fine."},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


@pytest.fixture
def models_payload() -> dict:
    """/v1/models response body."""
    return {
        "data": [
            {"id": "model-a", "object": "model", "owned_by": "org-1", "created": 1700000000},
            {"id": "model-b", "object": "model", "owned_by": "org-2", "created": 1700000001},
        ]
    }


@pytest.fixture
def nodes_payload() -> list[dict]:
    """/api/v1/nodes response body."""
    return [
        {
            "node_id": "node-0",
            "host": "10.0.0.1",
            "port": 50051,
            "start_layer": 0,
            "end_layer": 5,
            "healthy": True,
            "gpu_name": "Tesla T4",
            "gpu_utilization": 0.45,
            "free_memory_bytes": 6 * 1024 * 1024 * 1024,
        },
    ]


@pytest.fixture
def metrics_payload() -> dict:
    """/api/v1/metrics response body."""
    return {
        "requests_total": 1000,
        "tokens_generated": 50000,
        "active_requests": 5,
        "pending_requests": 2,
        "node_count": 3,
        "p95_latency_ms": 250.0,
        "errors_total": 10,
        "cache_hit_rate": 0.85,
    }


# -- HTTP transport helpers --------------------------------------------------


def _json_handler(payload: dict, status_code: int = 200):
    """Return an httpx.MockTransport handler that returns *payload* as JSON."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json=payload)
    return handler


def _text_handler(text: str, status_code: int = 200, content_type: str = "application/json"):
    """Return an httpx.MockTransport handler that returns plain text."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, text=text, headers={"content-type": content_type})
    return handler


def _health_ok_handler():
    """Return a transport that responds 200 to /health and 200 JSON to everything else."""
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url).endswith("/health"):
            return httpx.Response(200, json={"status": "ok"})
        # Default: 200 empty JSON
        return httpx.Response(200, json={})
    return handler


@pytest.fixture
def client_kwargs() -> dict:
    """Default keyword arguments for ``DistLLMClient``."""
    return {
        "coordinator_url": "http://10.0.0.1:8000",
        "api_key": "sk-test-key",
        "timeout": 30.0,
        "max_retries": 3,
    }
