"""Tests for distllm.dist.node_client module.

Zero mocks -- uses only real gRPC channel / stub objects.
No running gRPC server is required; connection-attempt tests rely on
very short timeouts against an unreachable target.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import grpc
import pytest
import torch

from distllm.dist import node_pb2, node_pb2_grpc
from distllm.dist.node_client import (
    DEFAULT_RPC_TIMEOUT_S,
    AsyncNodeClient,
    ChannelPool,
    NodeClient,
    channel_pool,
    create_async_node_client,
    create_node_client,
    forward_request,
    forward_request_async,
    request_layer_weights,
    request_layer_weights_stream,
    reset_channel_pool,
    resolve_rpc_timeout,
)
from distllm.dist.pipeline.serialization import from_proto_tensor, to_proto_tensor
from distllm.errors.types import GRPCTimeoutError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_channel_pool() -> None:
    """Clear the module-level channel pool for test isolation."""
    reset_channel_pool()
    yield
    reset_channel_pool()


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
        channel_pool.set(target, channel, 2)
        client = NodeClient(channel=channel, stub=stub, _target=target)
        client.close()
        # Pool entry should still exist but with decremented count
        assert target in channel_pool
        assert channel_pool.ref_count(target) == 1
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
        assert len(channel_pool) == 0
        try:
            create_node_client("localhost", 1, timeout_s=0.05)
        except Exception:
            pass
        assert len(channel_pool) == 0


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

    def test_initial_state_is_empty(self) -> None:
        assert isinstance(channel_pool, ChannelPool)
        assert len(channel_pool) == 0

    def test_close_without_pool_entry_does_not_affect_pool(self) -> None:
        channel = grpc.insecure_channel("localhost:50051")
        stub = node_pb2_grpc.NodeServiceStub(channel)
        client = NodeClient(channel=channel, stub=stub)
        client.close()
        assert len(channel_pool) == 0


# ---------------------------------------------------------------------------
# Sync forward_request RPC timeout (W2-2)
# ---------------------------------------------------------------------------


class _HungForwardServicer(node_pb2_grpc.NodeServiceServicer):
    """Servicer whose ForwardPass never replies — simulates a hung worker."""

    def __init__(self) -> None:
        # Never set by ForwardPass: each call parks its handler thread.
        # With no client-side deadline the caller would block forever;
        # with one, grpc raises DEADLINE_EXCEEDED.  release() in fixture
        # teardown unblocks the server's worker threads so pytest exits.
        self._block = threading.Event()

    def release(self) -> None:
        self._block.set()

    def ForwardPass(self, request, context):  # noqa: N802
        self._block.wait()


class _EchoForwardServicer(node_pb2_grpc.NodeServiceServicer):
    """Servicer that returns a valid ForwardPassResponse immediately."""

    def ForwardPass(self, request, context):  # noqa: N802
        output = from_proto_tensor(request.hidden_states)
        return node_pb2.ForwardPassResponse(
            request_id=request.request_id,
            output=to_proto_tensor(output * 2.0),
            success=True,
            processing_time_ms=0.0,
        )


class TestResolveRpcTimeout:
    """Precedence rules for the sync-RPC timeout resolver."""

    def test_explicit_argument_wins_over_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DISTLLM_RPC_TIMEOUT_S", "99")
        assert resolve_rpc_timeout(1.5) == 1.5

    def test_default_when_nothing_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("DISTLLM_RPC_TIMEOUT_S", raising=False)
        assert resolve_rpc_timeout(None) == DEFAULT_RPC_TIMEOUT_S
        assert DEFAULT_RPC_TIMEOUT_S == 30.0

    def test_env_override_respected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DISTLLM_RPC_TIMEOUT_S", "7")
        assert resolve_rpc_timeout(None) == 7.0

    @pytest.mark.parametrize("bad", ["abc", "0", "-5"])
    def test_invalid_env_falls_back_to_default(
        self, bad: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DISTLLM_RPC_TIMEOUT_S", bad)
        assert resolve_rpc_timeout(None) == DEFAULT_RPC_TIMEOUT_S


class TestSyncRpcTimeoutEnforcement:
    """forward_request must enforce its deadline against hung workers."""

    @pytest.fixture()
    def hung_server(self):
        servicer = _HungForwardServicer()
        server = grpc.server(ThreadPoolExecutor(max_workers=2))
        node_pb2_grpc.add_NodeServiceServicer_to_server(servicer, server)
        port = server.add_insecure_port("127.0.0.1:0")
        server.start()
        yield ("127.0.0.1", port)
        # Unblock parked handler threads BEFORE stopping the server so the
        # executor's atexit join cannot hang pytest.
        servicer.release()
        server.stop(grace=0)

    @pytest.fixture()
    def echo_server(self):
        server = grpc.server(ThreadPoolExecutor(max_workers=2))
        node_pb2_grpc.add_NodeServiceServicer_to_server(
            _EchoForwardServicer(), server
        )
        port = server.add_insecure_port("127.0.0.1:0")
        server.start()
        yield ("127.0.0.1", port)
        server.stop(grace=0)

    def test_hung_server_raises_within_timeout(self, hung_server) -> None:
        host, port = hung_server
        tensor = torch.randn(1, 4)
        start = time.monotonic()
        with pytest.raises(GRPCTimeoutError) as excinfo:
            forward_request(host, port, tensor, timeout_s=0.5)
        elapsed = time.monotonic() - start

        # Must fail well before any "forever" hang; allow generous slack
        # for slow CI machines.
        assert elapsed < 10.0
        err = excinfo.value
        assert f"{host}:{port}" in str(err)
        assert err.node_id == f"{host}:{port}"
        assert err.timeout == 0.5
        assert err.host == host and err.port == port

    def test_hung_server_uses_env_override_deadline(
        self, hung_server, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        host, port = hung_server
        # Env picks a short deadline even though no explicit timeout passed.
        monkeypatch.setenv("DISTLLM_RPC_TIMEOUT_S", "0.3")
        start = time.monotonic()
        with pytest.raises(GRPCTimeoutError):
            forward_request(host, port, torch.randn(1, 4))
        assert time.monotonic() - start < 10.0

    def test_fast_server_unaffected(self, echo_server) -> None:
        host, port = echo_server
        tensor = torch.randn(2, 8)
        out = forward_request(host, port, tensor, timeout_s=5.0)
        assert out.shape == tensor.shape
        assert torch.allclose(out, tensor * 2.0, atol=1e-6)

    def test_fast_server_with_default_timeout(self, echo_server) -> None:
        """No timeout argument at all — default resolution path still works."""
        host, port = echo_server
        out = forward_request(host, port, torch.randn(1, 4))
        assert out.shape == (1, 4)

    def test_timeout_applies_per_call_not_connection_only(
        self, hung_server
    ) -> None:
        """The connection succeeds quickly; only the RPC deadline fires.

        Guards against a regression where timeout_s is consumed solely by
        create_node_client's channel-ready wait.
        """
        import distllm.dist.node_client as nc

        host, port = hung_server
        calls: list[float] = []

        real_create = nc.create_node_client

        def spy_create(*args, **kwargs):
            result = real_create(*args, **kwargs)
            calls.append(kwargs.get("timeout_s", -1))
            return result

        nc.create_node_client = spy_create  # type: ignore[assignment]
        try:
            with pytest.raises(GRPCTimeoutError):
                forward_request(host, port, torch.randn(1, 4), timeout_s=0.4)
        finally:
            nc.create_node_client = real_create  # type: ignore[assignment]

        assert calls == [0.4]
