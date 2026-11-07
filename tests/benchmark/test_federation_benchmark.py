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
        # forward_request awaits the post and then calls raise_for_status() /
        # json() synchronously (federation.py:597-599) — the response double
        # must therefore expose sync methods, unlike the client itself.
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"choices": [{"text": "hello"}], "usage": {}}
        async_client.post = AsyncMock(return_value=mock_resp)

        # forward_request_streaming uses ``async with client.stream(...)``
        # (federation.py:633) and iterates ``resp.aiter_lines()``.
        stream_resp = MagicMock()
        stream_resp.raise_for_status = MagicMock()

        async def _aiter_lines():
            yield 'data: {"choices": [{"text": "hel"}]}'
            yield "data: [DONE]"

        stream_resp.aiter_lines = MagicMock(side_effect=_aiter_lines)
        stream_cm = MagicMock()
        stream_cm.__aenter__ = AsyncMock(return_value=stream_resp)
        stream_cm.__aexit__ = AsyncMock(return_value=False)
        async_client.stream = MagicMock(return_value=stream_cm)

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

        # Warmup
        asyncio.run(_run())

        # Benchmark — the callable must create a fresh coroutine per
        # invocation; asyncio.run(_run) would pass the function object.
        result = benchmark(lambda: asyncio.run(_run()))
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

        # Warmup (mirrors the non-streaming benchmark; the previous _bench
        # helper was defined but never called)
        asyncio.run(_run())

        result = benchmark(lambda: asyncio.run(_run()))
        # Two SSE payload lines are emitted; the [DONE] sentinel stops the
        # generator without being yielded (federation.py:638-639).
        assert len(result) == 1
        assert '"hel"' in result[0]


class TestCacheAwareRouting:
    """Benchmarks cache-aware routing logic."""

    def test_cache_affinity_routing(self, benchmark, coord):
        coord.update_cache_digest([101, 202, 303, 404])

        async def _run():
            # Returns the forwarded response dict tagged with routing
            # metadata (federation.py:1028-1034), not a bare peer dict.
            response = await coord.forward_with_cache_affinity(
                request={"model": "test", "messages": []},
                prompt_token_ids=[101, 202, 303, 404],
            )
            return response

        import asyncio
        result = benchmark(lambda: asyncio.run(_run()))
        assert result["x-distllm-federated"] == "peer-1"
        assert result["x-distllm-cache-affinity"] == 0.0
        assert result["x-distllm-matched-tokens"] == 0
