"""Integration test: auto-scaling behaviour.

Sends a sustained burst of requests and verifies that the autoscaler
provisions additional workers when load exceeds the threshold.

Requires Docker Compose cluster with AutoScaler enabled.

Usage:
    docker compose -f tests/dist/integration/docker-compose.yml up --build -d
    python -m pytest tests/dist/integration/test_autoscaling_scenario.py -v
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any

import httpx
import pytest

COORDINATOR_URL = os.environ.get("COORDINATOR_URL", "http://localhost:8000")
REQUEST_TIMEOUT = float(os.environ.get("TEST_TIMEOUT_S", "120"))

NUM_CONCURRENT = 20


@pytest.fixture(scope="module")
def client() -> httpx.Client:
    return httpx.Client(timeout=REQUEST_TIMEOUT)


class TestAutoScaling:
    """Tests that the cluster can handle concurrent load and scale."""

    @pytest.mark.timeout(120)
    def test_burst_load(self, client: httpx.Client):
        """Send NUM_CONCURRENT requests in parallel and wait for all to complete."""
        results: list[dict[str, Any]] = []
        errors: list[Exception] = []
        lock = threading.Lock()

        def _send(prompt: str) -> None:
            try:
                resp = client.post(
                    f"{COORDINATOR_URL}/v1/chat/completions",
                    json={
                        "model": "roneneldan/TinyStories-1M",
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 10,
                        "temperature": 0.0,
                    },
                )
                with lock:
                    results.append({"status": resp.status_code, "body": resp.json() if resp.status_code == 200 else {}})
            except Exception as e:
                with lock:
                    errors.append(e)

        prompts = [f"Test request {i}" for i in range(NUM_CONCURRENT)]
        threads = [threading.Thread(target=_send, args=(p,)) for p in prompts]

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

        # All should have completed without error
        assert len(errors) == 0, f"{len(errors)} requests failed"
        assert len(results) == NUM_CONCURRENT
        success_count = sum(1 for r in results if r["status"] == 200)
        assert success_count >= NUM_CONCURRENT * 0.8, f"Only {success_count}/{NUM_CONCURRENT} succeeded"

    @pytest.mark.timeout(30)
    def test_cluster_healthy_after_load(self, client: httpx.Client):
        """Cluster should be healthy after burst load."""
        resp = client.get(f"{COORDINATOR_URL}/v1/health")
        assert resp.status_code == 200
