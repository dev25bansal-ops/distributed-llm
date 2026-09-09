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
    """FederationCoordinator with mocked network.

    The mocks mirror real ``httpx`` semantics:
    - ``AsyncClient.post(...)`` is awaited and returns a Response whose
      ``.json()`` is a *synchronous* method (as in real httpx).
    - ``AsyncClient.stream(...)`` is a *synchronous* call returning an async
      context manager (as in real httpx).
    """
    with patch("distllm.dist.federation.httpx.Client") as MockClient, \
         patch("distllm.dist.federation.httpx.AsyncClient") as MockAsyncClient:

        from distllm.dist.federation import FederationConfig, FederationCoordinator

        mock_coord = MagicMock()
        mock_coord.config.cluster_key = "test-key"
        mock_coord.scheduler = MagicMock()
        mock_coord.scheduler.stats.return_value = {"active_requests": 0, "pending_requests": 0}

        async_client = AsyncMock()
        # httpx Response.json() is sync — use MagicMock so .json() returns
        # the dict directly instead of an unawaitable coroutine.
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"choices": [{"text": "hello"}], "usage": {}}
        async_client.post.return_value = mock_resp

        # httpx AsyncClient.stream() is sync, returning an async CM whose
        # response supports aiter_lines().
        class _FakeStreamResponse:
            def __init__(self, lines):
                self._lines = lines
                self.raise_for_status = MagicMock()

            async def aiter_lines(self):
                for line in self._lines:
                    yield line

        class _FakeStreamCM:
            def __init__(self, lines):
                self._resp = _FakeStreamResponse(lines)

            async def __aenter__(self):
                return self._resp

            async def __aexit__(self, *exc):
                return False

        async_client.stream = MagicMock(
            return_value=_FakeStreamCM(["data: hello", "data: [DONE]"])
        )

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

        def _run():
            # asyncio.run requires a coroutine — call the async function to
            # create one each benchmark iteration.
            return asyncio.run(coord.forward_request(peer, request, timeout_s=30.0))

        result = benchmark(_run)
        assert result["choices"][0]["text"] == "hello"
        assert result["x-distllm-federated"] == "peer-1"

    def test_forward_request_streaming_overhead(self, benchmark, coord):
        """Benchmark streaming forward overhead."""
        peer = {"cluster_id": "peer-1", "host": "10.0.0.2", "port": 50060}
        request = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 10,
        }

        import asyncio

        async def _collect():
            chunks = []
            async for chunk in coord.forward_request_streaming(peer, request):
                chunks.append(chunk)
            return chunks

        def _run():
            return asyncio.run(_collect())

        result = benchmark(_run)
        assert isinstance(result, list)
        assert result == ["hello"]


class TestCacheAwareRouting:
    """Benchmarks cache-aware routing logic."""

    def test_cache_affinity_routing(self, benchmark, coord):
        """Route by cache affinity: a peer with a digest matching the prompt
        must be preferred (genuine cache-aware routing, not capacity fallback)."""
        from distllm.dist.federation import KVCacheDigest

        prompt_token_ids = [101, 202, 303, 404]

        # Seed the peer's advertised cache digest from the SAME prompt, so
        # cache affinity routing has a real match to find.
        coord._cache_digests["peer-1"] = KVCacheDigest(window_size=128).compute(
            prompt_token_ids
        )
        # Also set the local digest (as update_cache_digest would in prod).
        coord.update_cache_digest(prompt_token_ids)

        import asyncio

        def _run():
            return asyncio.run(
                coord.forward_with_cache_affinity(
                    request={"model": "test", "messages": []},
                    prompt_token_ids=prompt_token_ids,
                )
            )

        result = benchmark(_run)
        assert result is not None
        # The peer was selected by cache affinity, not capacity fallback.
        assert result["x-distllm-federated"] == "peer-1"
        assert result["x-distllm-cache-affinity"] > 0.3
        assert result["x-distllm-matched-tokens"] == len(prompt_token_ids)
