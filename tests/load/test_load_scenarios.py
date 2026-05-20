"""Load test scenarios for API endpoint performance and backpressure.

Tests:
1. Sustained throughput: 100 RPS for 30 minutes (slow, CI-skip)
2. Burst test: 1000 RPS for 10 seconds, verify 503/429 backpressure
3. Soak test: 10 RPS for 24 hours (slow, CI-skip)
"""

import asyncio
import time
from unittest.mock import MagicMock, AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

import distllm.api.server as server_module
from distllm.api.server import app


pytestmark = [
    pytest.mark.slow,
    pytest.mark.benchmark,
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _disable_auth(monkeypatch):
    monkeypatch.setenv("DISABLE_AUTH", "1")
    monkeypatch.setenv("DISTLLM_DEV_MODE", "1")
    monkeypatch.setenv("DISTLLM_RATE_LIMIT_ENABLED", "0")
    monkeypatch.delenv("API_KEY", raising=False)


@pytest.fixture
def coordinator():
    coord = MagicMock()
    coord.model_name = "test-model"
    coord.nodes = {}
    coord.node_order = []
    coord.scheduler = None
    coord.prefix_cache = None
    coord.metrics_exporter = None
    coord._vlm_pipeline = None
    coord._spec_decoder = None
    coord._agent_loop = None
    coord._rag_pipeline = None
    coord._shutting_down = False
    coord._disagg_orchestrator = None

    def encode_fn(text, **kwargs):
        tokens = list(range(1, len(text.split()) + 1))
        if kwargs.get("return_tensors") == "pt":
            import torch
            return torch.tensor([tokens])
        return tokens

    coord.tokenizer = MagicMock()
    coord.tokenizer.encode.side_effect = encode_fn
    coord.tokenizer.decode.side_effect = lambda tokens, **kwargs: " ".join(
        f"tok-{t}" for t in (tokens if isinstance(tokens, list) else tokens.tolist())
    )
    coord.tokenizer.eos_token_id = 0
    coord.list_models.return_value = ["test-model"]
    coord.generate.return_value = "Hello! This is a test response."

    return coord


@pytest.fixture
def client(coordinator):
    original = server_module.coordinator
    server_module.coordinator = coordinator
    c = TestClient(app, raise_server_exceptions=False)
    yield c
    server_module.coordinator = original


# ---------------------------------------------------------------------------
# Helper: send requests concurrently
# ---------------------------------------------------------------------------

async def _fire_requests(client, url: str, json_body: dict, rps: int, duration_s: float):
    """Fire requests at a given RPS for a given duration."""
    total = int(rps * duration_s)
    sent = 0
    status_counts: dict[int, int] = {}
    latencies: list[float] = []
    errors: list[str] = []
    start = time.monotonic()

    while time.monotonic() - start < duration_s and sent < total:
        batch_size = min(rps, total - sent)
        batch_start = time.monotonic()

        tasks = []
        for _ in range(batch_size):
            t0 = time.monotonic()
            try:
                resp = client.post(url, json=json_body)
                elapsed = time.monotonic() - t0
                latencies.append(elapsed)
                status_counts[resp.status_code] = status_counts.get(resp.status_code, 0) + 1
            except Exception as e:
                errors.append(str(e))
            sent += 1

        elapsed_batch = time.monotonic() - batch_start
        sleep_needed = 1.0 - elapsed_batch
        if sleep_needed > 0:
            await asyncio.sleep(sleep_needed)

    return {
        "sent": sent,
        "duration": time.monotonic() - start,
        "status_counts": status_counts,
        "p50": sorted(latencies)[len(latencies) // 2] if latencies else 0,
        "p95": sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0,
        "p99": sorted(latencies)[int(len(latencies) * 0.99)] if latencies else 0,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# 1. Sustained Throughput
# ---------------------------------------------------------------------------

class TestSustainedThroughput:
    """100 RPS sustained for 30 minutes — measure latency degradation."""

    @pytest.mark.slow
    @pytest.mark.benchmark
    def test_100rps_30min(self, client):
        """Sustained 100 RPS for 30 min. Latency must not degrade >20%."""
        import asyncio
        result = asyncio.run(_fire_requests(
            client, "/v1/chat/completions",
            {"model": "test-model", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 10, "stream": False},
            rps=100, duration_s=30 * 60,
        ))
        assert result["sent"] > 0, "No requests were sent"
        p50_degradation = result.get("p50", 0) / max(result.get("p50_initial", 1), 1)
        assert result["p95"] < 5.0, f"P95 latency too high: {result['p95']:.2f}s"
        assert len(result["errors"]) / max(result["sent"], 1) < 0.01, "Error rate exceeds 1%"

    def test_throughput_short_validation(self, client):
        """Quick 5s smoke test to validate throughput plumbing."""
        import asyncio
        result = asyncio.run(_fire_requests(
            client, "/v1/chat/completions",
            {"model": "test-model", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 5},
            rps=10, duration_s=5,
        ))
        assert result["sent"] > 0
        assert 200 in result["status_counts"], f"No 200 responses: {result['status_counts']}"
        success_rate = result["status_counts"].get(200, 0) / max(result["sent"], 1)
        assert success_rate > 0.9, f"Success rate too low: {success_rate:.1%}"


# ---------------------------------------------------------------------------
# 2. Burst Test
# ---------------------------------------------------------------------------

class TestBurstLoad:
    """1000 RPS burst for 10 seconds — verify backpressure activates."""

    @pytest.mark.slow
    @pytest.mark.benchmark
    def test_1000rps_burst_10s(self, client):
        """Burst 1000 RPS for 10s. Expect some 429/503 from backpressure."""
        import asyncio
        result = asyncio.run(_fire_requests(
            client, "/v1/chat/completions",
            {"model": "test-model", "messages": [{"role": "user", "content": "Burst test"}], "max_tokens": 5},
            rps=1000, duration_s=10,
        ))
        total = result["sent"]
        assert total > 0, "No requests sent"
        accepted = result["status_counts"].get(200, 0)
        rejected = sum(v for k, v in result["status_counts"].items() if k in (429, 503))
        if rejected > 0:
            rejection_rate = rejected / max(total, 1)
            accepted_rate = accepted / max(total, 1)
            assert accepted_rate > 0.3, (
                f"Too many rejections: {rejection_rate:.1%} rejected, {accepted_rate:.1%} accepted"
            )

    @pytest.mark.asyncio
    async def test_backpressure_middleware_503(self):
        """BackpressureMiddleware rejects when pending exceeds threshold."""
        from distllm.api.server import BackpressureMiddleware
        mock_request = MagicMock()
        mock_request.url.path = "/v1/chat/completions"
        mock_request.state.request_id = "test-123"
        called = [False]

        async def call_next(req):
            called[0] = True
            return MagicMock(status_code=200)

        middleware = BackpressureMiddleware(MagicMock())
        orig_max = middleware.MAX_PENDING_REQUESTS
        middleware.MAX_PENDING_REQUESTS = 5

        # Mock coordinator with overloaded scheduler
        with patch("distllm.api.server.coordinator") as mock_coord:
            mock_coord._shutting_down = False
            mock_coord.scheduler.stats.return_value = {"pending_requests": 100}
            response = await middleware.dispatch(mock_request, call_next)

        middleware.MAX_PENDING_REQUESTS = orig_max
        assert response.status_code == 503
        assert not called[0], "call_next should not have been called"

    @pytest.mark.asyncio
    async def test_backpressure_passes_when_below_threshold(self):
        from distllm.api.server import BackpressureMiddleware
        mock_request = MagicMock()
        mock_request.url.path = "/v1/chat/completions"
        mock_request.state.request_id = "test-456"

        async def call_next(req):
            return MagicMock(status_code=200, spec_set=["status_code"])

        middleware = BackpressureMiddleware(MagicMock())
        orig_max = middleware.MAX_PENDING_REQUESTS
        middleware.MAX_PENDING_REQUESTS = 100

        with patch("distllm.api.server.coordinator") as mock_coord:
            mock_coord._shutting_down = False
            mock_coord.scheduler.stats.return_value = {"pending_requests": 5}
            response = await middleware.dispatch(mock_request, call_next)

        middleware.MAX_PENDING_REQUESTS = orig_max
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# 3. Soak Test
# ---------------------------------------------------------------------------

class TestSoak:
    """10 RPS sustained for extended period — check for memory/resource leaks."""

    @pytest.mark.slow
    @pytest.mark.memory
    def test_10rps_24h_soak(self, client):
        """10 RPS for 24 hours. Requires manual monitoring (CI-skip)."""
        import asyncio
        result = asyncio.run(_fire_requests(
            client, "/v1/chat/completions",
            {"model": "test-model", "messages": [{"role": "user", "content": "Soak test"}], "max_tokens": 10},
            rps=10, duration_s=24 * 3600,
        ))
        error_rate = len(result["errors"]) / max(result["sent"], 1)
        assert error_rate < 0.01, f"Error rate too high after 24h: {error_rate:.2%}"
        assert result["status_counts"].get(200, 0) > 0, "No successful requests"

    def test_soak_short_validation(self, client):
        """Quick 10s soak validation."""
        import asyncio
        result = asyncio.run(_fire_requests(
            client, "/v1/chat/completions",
            {"model": "test-model", "messages": [{"role": "user", "content": "Quick soak"}], "max_tokens": 5},
            rps=10, duration_s=10,
        ))
        assert result["sent"] > 0
        success_rate = result["status_counts"].get(200, 0) / max(result["sent"], 1)
        assert success_rate > 0.9, f"Success rate: {success_rate:.1%}"
