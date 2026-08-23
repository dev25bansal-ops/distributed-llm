"""Integration test: WAN pipeline with simulated latency.

Uses ``tc`` (traffic control) to inject latency between containers,
then verifies the WAN-aware pipeline adapts accumulation and
produces correct output.

Requires Docker Compose cluster with WAN pipeline enabled.

Usage:
    docker compose -f tests/dist/integration/docker-compose.yml up --build -d
    # Inject 100ms latency between coordinator and node_0:
    docker exec coordinator tc qdisc add dev eth0 root netem delay 100ms
    python -m pytest tests/dist/integration/test_wan_scenario.py -v
"""

from __future__ import annotations

import os
import time
from typing import Any

import httpx
import pytest

COORDINATOR_URL = os.environ.get("COORDINATOR_URL", "http://localhost:8000")
REQUEST_TIMEOUT = float(os.environ.get("TEST_TIMEOUT_S", "120"))


def _url(path: str) -> str:
    return f"{COORDINATOR_URL.rstrip('/')}{path}"


@pytest.fixture(scope="module")
def client() -> httpx.Client:
    """Authenticated client: live servers always require an API key."""
    api_key = os.environ.get("TEST_API_KEY", "")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    return httpx.Client(timeout=REQUEST_TIMEOUT, headers=headers)


class TestWANPipeline:
    """Tests that the pipeline handles simulated WAN latency."""

    @pytest.mark.timeout(60)
    def test_basic_throughput(self, client: httpx.Client):
        """Even under latency, basic generation should work."""
        try:
            resp = client.post(
                _url("/v1/chat/completions"),
                json={
                    "model": "roneneldan/TinyStories-1M",
                    "messages": [{"role": "user", "content": "Write a sentence."}],
                    "max_tokens": 20,
                },
            )
        except httpx.ConnectError:
            pytest.skip("Coordinator server not available")
        assert resp.status_code == 200
        assert "choices" in resp.json()

    @pytest.mark.timeout(120)
    def test_longer_generation(self, client: httpx.Client):
        """Longer generation under latency should accumulate and complete."""
        try:
            resp = client.post(
                _url("/v1/chat/completions"),
                json={
                    "model": "roneneldan/TinyStories-1M",
                    "messages": [{"role": "user", "content": "Write a short paragraph about distributed computing."}],
                    "max_tokens": 60,
                    "temperature": 0.5,
                },
            )
        except httpx.ConnectError:
            pytest.skip("Coordinator server not available")
        assert resp.status_code == 200
        data = resp.json()
        choices = data.get("choices", [])
        assert len(choices) > 0
        # Should have generated a reasonable amount of text
        message = choices[0].get("message", {})
        content = message.get("content", "")
        assert len(content) > 20, f"Response too short: {content[:50]}..."

    @pytest.mark.timeout(30)
    def test_streaming_under_latency(self, client: httpx.Client):
        """Streaming should still work under WAN latency."""
        try:
            resp = client.post(
                _url("/v1/chat/completions"),
                json={
                    "model": "roneneldan/TinyStories-1M",
                    "messages": [{"role": "user", "content": "Count from 1 to 5."}],
                    "max_tokens": 30,
                    "stream": True,
                },
            )
        except httpx.ConnectError:
            pytest.skip("Coordinator server not available")
        assert resp.status_code == 200
        # Verify we got at least one data chunk
        chunks = [line for line in resp.text.split("\n") if line.startswith("data: ")]
        assert len(chunks) > 0
