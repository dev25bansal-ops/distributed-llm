"""Tests for distllm.dist.node_client module.

Zero mocks -- uses only real gRPC channel / stub objects.
No running gRPC server is required; connection-attempt tests rely on
very short timeouts against an unreachable target.
"""

from __future__ import annotations

import grpc
import pytest
import torch

from distllm.dist import node_pb2_grpc
from distllm.dist.node_client import (
    AsyncNodeClient,
    NodeClient,
    _channel_pool,
    create_async_node_client,
    create_node_client,
    forward_request,
    forward_request_async,
    request_layer_weights,
    request_layer_weights_stream,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_channel_pool() -> None:
    """Clear the module-level channel pool for test isolation."""
    _channel_pool.clear()
    yield
    _channel_pool.clear()


# ---------------------------------------------------------------------------
# Smoke test -- module exports
# ---------------------------------------------------------------------------


class TestModuleExports:
    """Sanity-check that the module exposes all expected public names."""

    def test_all_public_exports(self) -> None:
        import distllm.dist.node_client as nc

        assert nc.NodeClient is NodeClient
        assert nc.AsyncNodeClient is AsyncNodeClient
        assert nc.create_node_client is create_node_client
        assert nc.create_async_node_client is create_async_node_client
        assert nc.forward_request is forward_request
        assert nc.forward_request_async is forward_request_async
        assert nc.request_layer_weights is request_layer_weights
        assert nc.request_layer_weights_stream is request_layer_weights_stream


# ---------------------------------------------------------------------------
# NodeClient -- synchronous client dataclass
# ---------------------------------------------------------------------------


class TestNodeClient:
    """Construction, attribute access, and close() behaviour."""

    def test_create_with_cluster_key(self) -> None:
        channel = grpc.insecure_channel("localhost:50051")
        stub = node_pb2_grpc.NodeServiceStub(channel)
        client = NodeClient(channel=channel, stub=stub, cluster_key="hello")

        assert client.channel is channel
        assert client.stub is stub
        assert client.cluster_key == "hello"
        assert client._target == ""
        client.close()

    def test_create_without_cluster_key(self) -> None:
        channel = grpc.insecure_channel("localhost:50051")
        stub = node_pb2_grpc.NodeServiceStub(channel)
        client = NodeClient(channel=channel, stub=stub)

        assert client.cluster_key is None
        client.close()

    def test_close_does_not_raise(self) -> None:
        channel = grpc.insecure_channel("localhost:50051")
        stub = node_pb2_grpc.NodeServiceStub(channel)
        client = NodeClient(channel=channel, stub=stub)
        client.close()

    def test_close_twice_is_idempotent(self) -> None:
        channel = grpc.insecure_channel("localhost:50051")
        stub = node_pb2_grpc.NodeServiceStub(channel)
        client = NodeClient(channel=channel, stub=stub)
        client.close()
        client.close()

    def test_close_after_channel_closed_externally(self) -> None:
        channel = grpc.insecure_channel("localhost:50051")
        stub = node_pb2_grpc.NodeServiceStub(channel)
        client = NodeClient(channel=channel, stub=stub)
        channel.close()
        client.close()

    def test_close_decrements_pool_when_target_present(self) -> None:
        channel = grpc.insecure_channel("localhost:50051")
        stub = node_pb2_grpc.NodeServiceStub(channel)
        target = "localhost:50051"
        _channel_pool[target] = (channel, 2)
        client = NodeClient(channel=channel, stub=stub, _target=target)
        client.close()
        # Pool entry should still exist but with decremented count
        assert target in _channel_pool
        assert _channel_pool[target][1] == 1
        channel.close()


# ---------------------------------------------------------------------------
# AsyncNodeClient -- async client dataclass
# ---------------------------------------------------------------------------


class TestAsyncNodeClient:
    """Construction, attribute access, and close() behaviour."""

    async def test_create_with_cluster_key(self) -> None:
        channel = grpc.aio.insecure_channel("localhost:50051")
        stub = node_pb2_grpc.NodeServiceStub(channel)
        client = AsyncNodeClient(channel=channel, stub=stub, cluster_key="k")

        assert client.channel is channel
        assert client.stub is stub
        assert client.cluster_key == "k"
        await client.close()

    async def test_create_without_cluster_key(self) -> None:
        channel = grpc.aio.insecure_channel("localhost:50051")
        stub = node_pb2_grpc.NodeServiceStub(channel)
        client = AsyncNodeClient(channel=channel, stub=stub)

        assert client.cluster_key is None
        await client.close()

    async def test_close_does_not_raise(self) -> None:
        channel = grpc.aio.insecure_channel("localhost:50051")
        stub = node_pb2_grpc.NodeServiceStub(channel)
        client = AsyncNodeClient(channel=channel, stub=stub)
        await client.close()

    async def test_close_twice_is_idempotent(self) -> None:
        channel = grpc.aio.insecure_channel("localhost:50051")
        stub = node_pb2_grpc.NodeServiceStub(channel)
        client = AsyncNodeClient(channel=channel, stub=stub)
        await client.close()
        await client.close()


# ---------------------------------------------------------------------------
# create_node_client  (sync factory)
# ---------------------------------------------------------------------------


class TestCreateNodeClient:
    """Factory function for synchronous gRPC clients."""

    def test_raises_on_unreachable(self) -> None:
        with pytest.raises(Exception):
            create_node_client("localhost", 1, timeout_s=0.05)

    def test_raises_on_unreachable_with_cluster_key(self) -> None:
        with pytest.raises(Exception):
            create_node_client("localhost", 1, cluster_key="k", timeout_s=0.05)

    def test_raises_file_not_found_with_bad_tls_cert(self) -> None:
        with pytest.raises(FileNotFoundError):
            create_node_client(
                "localhost",
                50051,
                use_tls=True,
                ca_cert="/nonexistent/ca.pem",
                timeout_s=0.05,
            )

    def test_pool_not_modified_on_failure(self) -> None:
        assert len(_channel_pool) == 0
        try:
            create_node_client("localhost", 1, timeout_s=0.05)
        except Exception:
            pass
        assert len(_channel_pool) == 0


# ---------------------------------------------------------------------------
# create_async_node_client  (async factory)
# ---------------------------------------------------------------------------


class TestCreateAsyncNodeClient:
    """Factory function for async gRPC clients."""

    async def test_raises_on_unreachable(self) -> None:
        with pytest.raises(Exception):
            await create_async_node_client("localhost", 1, timeout_s=0.05)

    async def test_raises_on_unreachable_with_cluster_key(self) -> None:
        with pytest.raises(Exception):
            await create_async_node_client(
                "localhost", 1, cluster_key="k", timeout_s=0.05,
            )


# ---------------------------------------------------------------------------
# request_layer_weights  (sync helper)
# ---------------------------------------------------------------------------


class TestRequestLayerWeights:
    """Synchronous weight-transfer convenience function."""

    def test_returns_none_when_unreachable(self) -> None:
        result = request_layer_weights("localhost", 1, "m", 0, 2, timeout_s=0.05)
        assert result is None

    def test_zero_layers_returns_none(self) -> None:
        result = request_layer_weights("localhost", 1, "m", 0, 0, timeout_s=0.05)
        assert result is None

    def test_negative_start_returns_none(self) -> None:
        result = request_layer_weights("localhost", 1, "m", -1, 2, timeout_s=0.05)
        assert result is None

    def test_swapped_layer_order_returns_none(self) -> None:
        result = request_layer_weights("localhost", 1, "m", 5, 2, timeout_s=0.05)
        assert result is None


# ---------------------------------------------------------------------------
# request_layer_weights_stream  (streaming helper)
# ---------------------------------------------------------------------------


class TestRequestLayerWeightsStream:
    """Streaming weight-transfer convenience function."""

    def test_returns_none_when_unreachable(self) -> None:
        result = request_layer_weights_stream(
            "localhost", 1, "m", 0, 2, timeout_s=0.05,
        )
        assert result is None

    def test_large_range_returns_none(self) -> None:
        result = request_layer_weights_stream(
            "localhost", 1, "m", 0, 1000, timeout_s=0.05,
        )
        assert result is None


# ---------------------------------------------------------------------------
# forward_request  (sync helper)
# ---------------------------------------------------------------------------


class TestForwardRequest:
    """Synchronous forward-pass convenience function."""

    def test_raises_on_unreachable(self) -> None:
        t = torch.randn(1, 768)
        with pytest.raises(Exception):
            forward_request("localhost", 1, t, timeout_s=0.05)

    def test_empty_tensor_raises_on_unreachable(self) -> None:
        t = torch.empty(0)
        with pytest.raises(Exception):
            forward_request("localhost", 1, t, timeout_s=0.05)

    def test_with_kv_cache_raises_on_unreachable(self) -> None:
        t = torch.randn(1, 768)
        kv = [(torch.randn(1, 4, 8), torch.randn(1, 4, 8))]
        with pytest.raises(Exception):
            forward_request("localhost", 1, t, kv_cache=kv, timeout_s=0.05)

    def test_with_request_id_raises_on_unreachable(self) -> None:
        t = torch.randn(1, 768)
        with pytest.raises(Exception):
            forward_request("localhost", 1, t, request_id="rid", timeout_s=0.05)

    def test_with_cluster_key_raises_on_unreachable(self) -> None:
        t = torch.randn(1, 768)
        with pytest.raises(Exception):
            forward_request("localhost", 1, t, cluster_key="ck", timeout_s=0.05)


# ---------------------------------------------------------------------------
# forward_request_async  (async helper)
# ---------------------------------------------------------------------------


class TestForwardRequestAsync:
    """Asynchronous forward-pass convenience function."""

    async def test_raises_on_unreachable(self) -> None:
        t = torch.randn(1, 768)
        with pytest.raises(Exception):
            await forward_request_async("localhost", 1, t, timeout_s=0.05)

    async def test_empty_tensor_raises(self) -> None:
        t = torch.empty(0)
        with pytest.raises(Exception):
            await forward_request_async("localhost", 1, t, timeout_s=0.05)

    async def test_with_request_id(self) -> None:
        t = torch.randn(1, 768)
        with pytest.raises(Exception):
            await forward_request_async(
                "localhost", 1, t, request_id="rid", timeout_s=0.05,
            )

    async def test_with_kv_cache(self) -> None:
        t = torch.randn(1, 768)
        kv = [(torch.randn(1, 4, 8), torch.randn(1, 4, 8))]
        with pytest.raises(Exception):
            await forward_request_async(
                "localhost", 1, t, kv_cache=kv, timeout_s=0.05,
            )

    async def test_with_cluster_key(self) -> None:
        t = torch.randn(1, 768)
        with pytest.raises(Exception):
            await forward_request_async(
                "localhost", 1, t, cluster_key="ck", timeout_s=0.05,
            )


# ---------------------------------------------------------------------------
# Module-level channel pool
# ---------------------------------------------------------------------------


class TestChannelPool:
    """Module-level gRPC channel pool invariants."""

    def test_initial_state_is_empty_dict(self) -> None:
        assert isinstance(_channel_pool, dict)
        assert len(_channel_pool) == 0

    def test_close_without_pool_entry_does_not_affect_pool(self) -> None:
        channel = grpc.insecure_channel("localhost:50051")
        stub = node_pb2_grpc.NodeServiceStub(channel)
        client = NodeClient(channel=channel, stub=stub)
        client.close()
        assert len(_channel_pool) == 0
