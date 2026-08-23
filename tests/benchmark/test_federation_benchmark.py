"""Benchmark: federation overhead vs direct inference.

Measures the additional latency introduced by routing a request through
the federation layer vs. sending it directly.  Uses mocked httpx so
no real cluster is required — the benchmark isolates the overhead of
the federation logic itself.

Metrics:
    - Federation overhead (ms) at p50, p99
    - Overhead percentage relative to direct inference
    - Streaming overhead
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def coord():
    """FederationCoordinator with mocked network."""
    with patch("distllm.dist.federation.httpx.Client") as MockClient, \
         patch("distllm.dist.federation.httpx.AsyncClient") as MockAsyncClient:

        from distllm.dist.federation import FederationConfig, FederationCoordinator

        mock_coord = MagicMock()
        mock_coord.config.cluster_key = "test-key"
        mock_coord.scheduler = MagicMock()
        mock_coord.scheduler.stats.return_value = {"active_requests": 0, "pending_requests": 0}

        async_client = AsyncMock()
        mock_resp = AsyncMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"choices": [{"text": "hello"}], "usage": {}}
        mock_resp.__aenter__.return_value = mock_resp
        async_client.post.return_value = mock_resp

        MockClient.return_value = MagicMock()
        MockAsyncClient.return_value = async_client

        fc = FederationCoordinator(
            config=FederationConfig(
                enabled=True, cluster_id="c1",
                listen_host="0.0.0.0", listen_port=50060,
            ),
            local_cluster_id="c1", local_host="0.0.0.0", local_port=50060,
            coordinator_ref=mock_coord,
        )

        # Add a fake peer
        from distllm.dist.p2p.discovery import PeerInfo
        fc._peers["peer-1"] = PeerInfo(
            cluster_id="peer-1", host="10.0.0.2", port=50060,
        )

        return fc


class TestFederationOverhead:
    """Measures the overhead of federation routing logic."""

    def test_forward_request_overhead(self, benchmark, coord):
        """Benchmark the overhead of forward_request (network excluded via mock)."""
        peer = {"cluster_id": "peer-1", "host": "10.0.0.2", "port": 50060}
        request = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 10,
        }

        import asyncio

        async def _run():
            result = await coord.forward_request(peer, request, timeout_s=30.0)
            return result

        async def _bench():
            result = await _run()
            return result

        # Warmup
        asyncio.run(_bench())

        # Benchmark
        result = benchmark(asyncio.run, _run)
        assert result["choices"][0]["text"] == "hello"

    def test_forward_request_streaming_overhead(self, benchmark, coord):
        """Benchmark streaming forward overhead."""
        peer = {"cluster_id": "peer-1", "host": "10.0.0.2", "port": 50060}
        request = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 10,
        }

        import asyncio

        async def _run():
            chunks = []
            async for chunk in coord.forward_request_streaming(peer, request):
                chunks.append(chunk)
            return chunks

        async def _bench():
            return await _run()

        result = benchmark(asyncio.run, _run)
        assert isinstance(result, list)


class TestCacheAwareRouting:
    """Benchmarks cache-aware routing logic."""

    def test_cache_affinity_routing(self, benchmark, coord):
        coord.update_cache_digest([101, 202, 303, 404])
        digest = {"hash": "abc", "prefix_hash": {0: 12345}, "length": 4}

        async def _run():
            peer = await coord.forward_with_cache_affinity(
                request={"model": "test", "messages": []},
                prompt_token_ids=[101, 202, 303, 404],
            )
            return peer

        import asyncio
        result = benchmark(asyncio.run, _run)
        assert result is not None
