"""Smoke tests against a live coordinator + worker cluster.

These tests assume the Docker Compose cluster is already running.
They hit the coordinator's HTTP API and validate end-to-end inference.

Usage:
    docker compose -f tests/dist/integration/docker-compose.yml up --build -d
    python -m pytest tests/dist/integration/ -v
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import httpx
import pytest

COORDINATOR_URL = os.environ.get(
    "COORDINATOR_URL", "http://localhost:8000"
)
REQUEST_TIMEOUT = float(os.environ.get("TEST_TIMEOUT_S", "30"))


def _url(path: str) -> str:
    return f"{COORDINATOR_URL.rstrip('/')}{path}"


@pytest.fixture(scope="session")
def client() -> httpx.Client:
    return httpx.Client(timeout=REQUEST_TIMEOUT)


# ── Health ────────────────────────────────────────────────────────────


class TestHealth:
    def test_health_endpoint(self, client: httpx.Client):
        resp = client.get(_url("/v1/health"))
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert data["status"] in ("ok", "degraded", "starting")

    def test_readiness(self, client: httpx.Client):
        resp = client.get(_url("/v1/health/readiness"))
        assert resp.status_code in (200, 503)
        if resp.status_code == 200:
            assert resp.json().get("ready") is True


# ── Chat completions ──────────────────────────────────────────────────


class TestChatCompletions:
    def test_simple_chat(self, client: httpx.Client):
        """Basic chat completion without streaming."""
        resp = client.post(
            _url("/v1/chat/completions"),
            json={
                "model": "roneneldan/TinyStories-1M",
                "messages": [
                    {"role": "user", "content": "Hello, who are you?"},
                ],
                "max_tokens": 20,
                "temperature": 0.0,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert "text" in data["choices"][0] or "message" in data["choices"][0]

    def test_streaming_chat(self, client: httpx.Client):
        """Chat completion with SSE streaming."""
        resp = client.post(
            _url("/v1/chat/completions"),
            json={
                "model": "roneneldan/TinyStories-1M",
                "messages": [
                    {"role": "user", "content": "Tell me a story."},
                ],
                "max_tokens": 50,
                "stream": True,
            },
        )
        assert resp.status_code == 200
        chunks = [line for line in resp.text.split("\n") if line.startswith("data: ")]
        assert len(chunks) > 0
        # Last chunk should be [DONE]
        assert "DONE" in resp.text or len(chunks) > 1

    @pytest.mark.timeout(60)
    def test_longer_generation(self, client: httpx.Client):
        """Generate enough tokens to exercise the pipeline across both workers."""
        resp = client.post(
            _url("/v1/chat/completions"),
            json={
                "model": "roneneldan/TinyStories-1M",
                "messages": [
                    {"role": "user", "content": "Write a long paragraph about a cat."},
                ],
                "max_tokens": 100,
                "temperature": 0.7,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        choices = data.get("choices", [])
        assert len(choices) > 0


# ── Completions ───────────────────────────────────────────────────────


class TestCompletions:
    def test_simple_completion(self, client: httpx.Client):
        resp = client.post(
            _url("/v1/completions"),
            json={
                "model": "roneneldan/TinyStories-1M",
                "prompt": "Once upon a time",
                "max_tokens": 20,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "choices" in data


# ── Cluster state ─────────────────────────────────────────────────────


class TestClusterState:
    def test_list_nodes(self, client: httpx.Client):
        resp = client.get(_url("/admin/v1/nodes"))
        # May be 401 if auth required, but should respond.
        assert resp.status_code in (200, 401, 403)

    def test_cluster_status(self, client: httpx.Client):
        resp = client.get(_url("/admin/v1/cluster/status"))
        assert resp.status_code in (200, 401, 403)
        if resp.status_code == 200:
            data = resp.json()
            assert "nodes" in data or "status" in data


# ── Concurrent load ───────────────────────────────────────────────────


class TestConcurrentLoad:
    @pytest.mark.timeout(120)
    def test_concurrent_requests(self, client: httpx.Client):
        """Send multiple requests concurrently and verify all complete."""
        import concurrent.futures

        def _send(prompt: str) -> tuple[int, dict[str, Any]]:
            resp = client.post(
                _url("/v1/chat/completions"),
                json={
                    "model": "roneneldan/TinyStories-1M",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 10,
                },
            )
            return resp.status_code, resp.json() if resp.status_code == 200 else {}

        prompts = [
            "Hello",
            "What is AI?",
            "Tell me a joke",
            "Write a poem",
            "What is 2+2?",
        ]

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
            futures = [pool.submit(_send, p) for p in prompts]
            results = [f.result(timeout=60) for f in futures]

        assert all(code == 200 for code, _ in results)
        assert all("choices" in data for _, data in results)


# ── Graceful degradation ──────────────────────────────────────────────


class TestDegradation:
    @pytest.mark.timeout(60)
    def test_node_failure_still_serves(self, client: httpx.Client):
        """Basic check that the API responds even if a node is slow/dead.

        Note: This is a soft test — it doesn't actually kill a node
        (that would require Docker API access).  It verifies the system
        doesn't crash under load.
        """
        for _ in range(3):
            resp = client.post(
                _url("/v1/chat/completions"),
                json={
                    "model": "roneneldan/TinyStories-1M",
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 5,
                },
            )
            assert resp.status_code == 200
