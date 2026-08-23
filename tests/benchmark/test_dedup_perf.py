"""Benchmark: Dedup middleware latency reduction.

Measures the latency improvement when the DedupMiddleware returns a cached
response vs processing a request normally.

Target: 50% latency reduction on duplicate requests.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from distllm.api.dedup import DedupMiddleware


@pytest.fixture(scope="module")
def dedup_app():
    """App with a slow chat handler and DedupMiddleware."""
    import time

    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat():
        time.sleep(0.05)  # simulate 50ms processing
        return {"choices": [{"message": {"content": "hello"}}]}

    app.add_middleware(DedupMiddleware)
    return app


@pytest.fixture(scope="module")
def client(dedup_app):
    return TestClient(dedup_app)


class TestDedupLatency:
    """Measure latency reduction from dedup caching."""

    def test_first_request_latency(self, benchmark, client):
        """First request (cache miss) latency baseline."""

        def _first():
            resp = client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hello"}]},
            )
            assert resp.status_code == 200
            return resp

        result = benchmark(_first)
        assert result is not None

    def test_duplicate_request_latency(self, benchmark, client):
        """Duplicate request (cache hit) should be faster than first."""
        # Prime the cache
        payload = {"messages": [{"role": "user", "content": "perf-test-payload"}]}
        client.post("/v1/chat/completions", json=payload)

        def _duplicate():
            resp = client.post("/v1/chat/completions", json=payload)
            assert resp.status_code == 200
            return resp

        result = benchmark(_duplicate)
        assert result is not None
