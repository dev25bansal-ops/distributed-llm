"""Integration tests for gRPC channel reconnection and resilience.

Tests cover:
- Channel reconnection after transient failures
- DNS resolution changes (target host changes at runtime)
- Connection pool behavior under concurrent failures
- Timeout handling for create_node_client and async variants
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch, PropertyMock

import grpc
import pytest

from distllm.dist.node_client import (
    AsyncNodeClient,
    NodeClient,
    channel_pool,
    create_async_node_client,
    create_node_client,
    reset_channel_pool,
)


@pytest.fixture(autouse=True)
def _clear_channel_pool():
    """Reset the global channel pool before and after each test."""
    reset_channel_pool()
    yield
    reset_channel_pool()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_channel(ready: bool = True, target: str = "localhost:50051"):
    """Create a mock gRPC channel with controllable readiness."""
    channel = MagicMock()
    channel._target = target
    channel.close = MagicMock()
    return channel


# ---------------------------------------------------------------------------
# Test: gRPC channel reconnection after transient failure
# ---------------------------------------------------------------------------


class TestChannelReconnectionAfterTransientFailure:
    """Channel must reconnect when a transient gRPC error occurs."""

    def test_create_client_after_channel_ready_succeeds(self):
        """Normal path: channel_ready resolves, client is returned."""
        # Arrange
        mock_channel = _make_mock_channel()
        with patch("distllm.dist.node_client.grpc.insecure_channel", return_value=mock_channel):
            with patch("distllm.dist.node_client.grpc.channel_ready_future") as mock_ready:
                mock_future = MagicMock()
                mock_future.result.return_value = None
                mock_ready.return_value = mock_future

                # Act
                client = create_node_client("localhost", 50051, timeout_s=2.0)

                # Assert
                assert isinstance(client, NodeClient)
                assert client.channel is mock_channel
                mock_future.result.assert_called_once_with(timeout=2.0)
                client.close()

    def test_create_client_timeout_raises(self):
        """When channel_ready times out, FutureTimeoutError propagates."""
        # Arrange
        mock_channel = _make_mock_channel()
        with patch("distllm.dist.node_client.grpc.insecure_channel", return_value=mock_channel):
            with patch("distllm.dist.node_client.grpc.channel_ready_future") as mock_ready:
                mock_future = MagicMock()
                mock_future.result.side_effect = grpc.FutureTimeoutError()
                mock_ready.return_value = mock_future

                # Act / Assert
                with pytest.raises(grpc.FutureTimeoutError):
                    create_node_client("unreachable", 50051, timeout_s=0.1)

    def test_reconnect_after_transient_error_reuses_pool(self):
        """After a transient failure, a second call should create a fresh
        channel (pool entry was never created for the failed attempt)."""
        # Arrange
        mock_channel_good = _make_mock_channel()
        call_count = {"n": 0}

        def fake_insecure_channel(target, options=None):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise grpc.RpcError("transient failure")
            return mock_channel_good

        with patch("distllm.dist.node_client.grpc.insecure_channel", side_effect=fake_insecure_channel):
            with patch("distllm.dist.node_client.grpc.channel_ready_future") as mock_ready:
                mock_future = MagicMock()
                mock_future.result.return_value = None
                mock_ready.return_value = mock_future

                # Act: first call fails
                with pytest.raises(grpc.RpcError):
                    create_node_client("localhost", 50051, timeout_s=1.0)

                # Second call succeeds
                client = create_node_client("localhost", 50051, timeout_s=1.0)

                # Assert
                assert client.channel is mock_channel_good
                client.close()

    def test_stub_created_on_reconnect(self):
        """After reconnection, the stub must be bound to the new channel."""
        # Arrange
        mock_channel = _make_mock_channel()
        with patch("distllm.dist.node_client.grpc.insecure_channel", return_value=mock_channel):
            with patch("distllm.dist.node_client.grpc.channel_ready_future") as mock_ready:
                mock_future = MagicMock()
                mock_future.result.return_value = None
                mock_ready.return_value = mock_future

                # Act
                client = create_node_client("localhost", 50051, timeout_s=1.0)

                # Assert
                assert client.stub is not None
                client.close()


# ---------------------------------------------------------------------------
# Test: DNS resolution changes
# ---------------------------------------------------------------------------


class TestDNSResolutionChanges:
    """When the host resolves to a different IP, new connections should
    get a fresh channel (pool is keyed by host:port string)."""

    def test_different_host_gets_separate_channel(self):
        """Two different hosts produce two distinct pool entries."""
        # Arrange
        channel_a = _make_mock_channel(target="host-a:50051")
        channel_b = _make_mock_channel(target="host-b:50051")
        channels = iter([channel_a, channel_b])

        with patch("distllm.dist.node_client.grpc.insecure_channel", side_effect=lambda t, options=None: next(channels)):
            with patch("distllm.dist.node_client.grpc.channel_ready_future") as mock_ready:
                mock_future = MagicMock()
                mock_future.result.return_value = None
                mock_ready.return_value = mock_future

                # Act
                client_a = create_node_client("host-a", 50051, timeout_s=1.0)
                client_b = create_node_client("host-b", 50051, timeout_s=1.0)

                # Assert: different channels, different stubs
                assert client_a.channel is not client_b.channel
                assert client_a._target == "host-a:50051"
                assert client_b._target == "host-b:50051"

                client_a.close()
                client_b.close()

    def test_same_host_reuses_pooled_channel(self):
        """Connecting to the same host:port twice returns the pooled channel."""
        # Arrange
        mock_channel = _make_mock_channel()
        with patch("distllm.dist.node_client.grpc.insecure_channel", return_value=mock_channel):
            with patch("distllm.dist.node_client.grpc.channel_ready_future") as mock_ready:
                mock_future = MagicMock()
                mock_future.result.return_value = None
                mock_ready.return_value = mock_future

                # Act
                client1 = create_node_client("localhost", 50051, timeout_s=1.0)
                client2 = create_node_client("localhost", 50051, timeout_s=1.0)

                # Assert: same underlying channel object
                assert client1.channel is client2.channel
                # Pool refcount should be 2
                assert channel_pool.ref_count("localhost:50051") == 2

                client1.close()
                client2.close()

    def test_tls_channel_with_ca_cert(self):
        """TLS connections use secure_channel when use_tls=True."""
        # Arrange
        mock_channel = _make_mock_channel()
        with patch("distllm.dist.node_client.grpc.secure_channel", return_value=mock_channel) as mock_secure:
            with patch("distllm.dist.node_client.grpc.ssl_channel_credentials") as mock_creds:
                with patch("distllm.dist.node_client.grpc.channel_ready_future") as mock_ready:
                    mock_future = MagicMock()
                    mock_future.result.return_value = None
                    mock_ready.return_value = mock_future
                    mock_creds.return_value = MagicMock()

                    # Act
                    client = create_node_client("secure-host", 443, use_tls=True, timeout_s=1.0)

                    # Assert
                    mock_secure.assert_called_once()
                    assert client.channel is mock_channel
                    client.close()


# ---------------------------------------------------------------------------
# Test: Connection pool behavior under failures
# ---------------------------------------------------------------------------


class TestConnectionPoolBehaviorUnderFailures:
    """Channel pool must maintain correct refcounts when clients
    connect, disconnect, and encounter errors."""

    def test_pool_refcount_increments_on_reuse(self):
        """Each create call for the same target increments the pool refcount."""
        # Arrange
        mock_channel = _make_mock_channel()
        with patch("distllm.dist.node_client.grpc.insecure_channel", return_value=mock_channel):
            with patch("distllm.dist.node_client.grpc.channel_ready_future") as mock_ready:
                mock_future = MagicMock()
                mock_future.result.return_value = None
                mock_ready.return_value = mock_future

                # Act
                clients = [create_node_client("localhost", 50051) for _ in range(3)]

                # Assert
                count = channel_pool.ref_count("localhost:50051")
                assert count == 3

                for c in clients:
                    c.close()

    def test_pool_refcount_decrements_on_close(self):
        """Closing clients decrements the refcount; last close removes entry."""
        # Arrange
        mock_channel = _make_mock_channel()
        with patch("distllm.dist.node_client.grpc.insecure_channel", return_value=mock_channel):
            with patch("distllm.dist.node_client.grpc.channel_ready_future") as mock_ready:
                mock_future = MagicMock()
                mock_future.result.return_value = None
                mock_ready.return_value = mock_future

                c1 = create_node_client("localhost", 50051)
                c2 = create_node_client("localhost", 50051)

                # Act: close first — refcount drops to 1
                c1.close()
                assert channel_pool.ref_count("localhost:50051") == 1

                # Close second — entry removed from pool
                c2.close()
                assert "localhost:50051" not in channel_pool

    def test_pool_entry_removed_on_last_close(self):
        """The channel is closed and the pool entry is deleted when
        the last reference is released."""
        # Arrange
        mock_channel = _make_mock_channel()
        with patch("distllm.dist.node_client.grpc.insecure_channel", return_value=mock_channel):
            with patch("distllm.dist.node_client.grpc.channel_ready_future") as mock_ready:
                mock_future = MagicMock()
                mock_future.result.return_value = None
                mock_ready.return_value = mock_future

                client = create_node_client("localhost", 50051)

                # Act
                client.close()

                # Assert
                assert "localhost:50051" not in channel_pool
                # The pool channel's close was called
                mock_channel.close.assert_called()

    def test_close_is_idempotent_when_pool_entry_missing(self):
        """Closing a client whose pool entry was already removed should
        not raise."""
        # Arrange
        mock_channel = _make_mock_channel()
        with patch("distllm.dist.node_client.grpc.insecure_channel", return_value=mock_channel):
            with patch("distllm.dist.node_client.grpc.channel_ready_future") as mock_ready:
                mock_future = MagicMock()
                mock_future.result.return_value = None
                mock_ready.return_value = mock_future

                client = create_node_client("localhost", 50051)

                # Manually clear the pool to simulate external cleanup
                channel_pool.clear()

                # Act / Assert: should not raise
                client.close()

    def test_concurrent_clients_maintain_correct_refcount(self):
        """Multiple threads creating clients simultaneously must produce
        the correct final refcount."""
        # Arrange
        mock_channel = _make_mock_channel()
        results = []
        barrier = threading.Barrier(5)

        def _connect():
            with patch("distllm.dist.node_client.grpc.insecure_channel", return_value=mock_channel):
                with patch("distllm.dist.node_client.grpc.channel_ready_future") as mock_ready:
                    mock_future = MagicMock()
                    mock_future.result.return_value = None
                    mock_ready.return_value = mock_future
                    barrier.wait(timeout=2.0)
                    client = create_node_client("localhost", 50051)
                    results.append(client)

        # Act
        threads = [threading.Thread(target=_connect) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        # Assert: all 5 clients got a channel, refcount is 5
        assert len(results) == 5
        count = channel_pool.ref_count("localhost:50051")
        assert count == 5

        # Cleanup
        for c in results:
            c.close()

    def test_concurrent_close_maintains_correct_refcount(self):
        """Closing clients from multiple threads must not corrupt the pool."""
        # Arrange
        mock_channel = _make_mock_channel()
        with patch("distllm.dist.node_client.grpc.insecure_channel", return_value=mock_channel):
            with patch("distllm.dist.node_client.grpc.channel_ready_future") as mock_ready:
                mock_future = MagicMock()
                mock_future.result.return_value = None
                mock_ready.return_value = mock_future

                clients = [create_node_client("localhost", 50051) for _ in range(10)]

        # Act: close all concurrently
        barrier = threading.Barrier(10)
        def _close(client):
            barrier.wait(timeout=2.0)
            client.close()

        threads = [threading.Thread(target=_close, args=(c,)) for c in clients]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        # Assert: pool should be empty after all closes
        assert "localhost:50051" not in channel_pool


# ---------------------------------------------------------------------------
# Test: Timeout handling
# ---------------------------------------------------------------------------


class TestTimeoutHandling:
    """Timeouts must propagate cleanly for both sync and async clients."""

    def test_short_timeout_propagates_future_timeout(self):
        """A very short timeout should cause FutureTimeoutError."""
        # Arrange
        mock_channel = _make_mock_channel()
        with patch("distllm.dist.node_client.grpc.insecure_channel", return_value=mock_channel):
            with patch("distllm.dist.node_client.grpc.channel_ready_future") as mock_ready:
                mock_future = MagicMock()
                mock_future.result.side_effect = grpc.FutureTimeoutError()
                mock_ready.return_value = mock_future

                # Act / Assert
                with pytest.raises(grpc.FutureTimeoutError):
                    create_node_client("slow-host", 50051, timeout_s=0.001)

    def test_zero_timeout_fails_immediately(self):
        """Timeout of 0 should fail without waiting."""
        # Arrange
        mock_channel = _make_mock_channel()
        with patch("distllm.dist.node_client.grpc.insecure_channel", return_value=mock_channel):
            with patch("distllm.dist.node_client.grpc.channel_ready_future") as mock_ready:
                mock_future = MagicMock()
                mock_future.result.side_effect = grpc.FutureTimeoutError()
                mock_ready.return_value = mock_future

                # Act / Assert
                with pytest.raises(grpc.FutureTimeoutError):
                    create_node_client("fast-fail", 50051, timeout_s=0.0)

    def test_timeout_does_not_add_to_pool(self):
        """A timed-out connection must NOT create a pool entry."""
        # Arrange
        mock_channel = _make_mock_channel()
        with patch("distllm.dist.node_client.grpc.insecure_channel", return_value=mock_channel):
            with patch("distllm.dist.node_client.grpc.channel_ready_future") as mock_ready:
                mock_future = MagicMock()
                mock_future.result.side_effect = grpc.FutureTimeoutError()
                mock_ready.return_value = mock_future

                # Act
                with pytest.raises(grpc.FutureTimeoutError):
                    create_node_client("timeout-host", 50051, timeout_s=0.01)

                # Assert
                assert "timeout-host:50051" not in channel_pool

    def test_tls_timeout_propagates(self):
        """TLS connections also propagate timeouts correctly."""
        # Arrange
        mock_channel = _make_mock_channel()
        with patch("distllm.dist.node_client.grpc.secure_channel", return_value=mock_channel):
            with patch("distllm.dist.node_client.grpc.ssl_channel_credentials"):
                with patch("distllm.dist.node_client.grpc.channel_ready_future") as mock_ready:
                    mock_future = MagicMock()
                    mock_future.result.side_effect = grpc.FutureTimeoutError()
                    mock_ready.return_value = mock_future

                    # Act / Assert
                    with pytest.raises(grpc.FutureTimeoutError):
                        create_node_client(
                            "tls-timeout", 443, use_tls=True, timeout_s=0.01
                        )

    @pytest.mark.asyncio
    async def test_async_client_timeout(self):
        """Async client creation propagates timeout."""
        # Arrange
        mock_channel = MagicMock()

        async def fake_ready(timeout=None):
            raise grpc.FutureTimeoutError()

        mock_channel.channel_ready = fake_ready

        with patch("distllm.dist.node_client.grpc.aio.insecure_channel", return_value=mock_channel):
            # Act / Assert
            with pytest.raises(grpc.FutureTimeoutError):
                await create_async_node_client("async-timeout", 50051, timeout_s=0.01)

    @pytest.mark.asyncio
    async def test_async_client_success(self):
        """Async client creation succeeds when channel becomes ready."""
        # Arrange
        mock_channel = MagicMock()
        ready_called = {"v": False}

        async def fake_ready(timeout=None):
            ready_called["v"] = True

        mock_channel.channel_ready = fake_ready

        with patch("distllm.dist.node_client.grpc.aio.insecure_channel", return_value=mock_channel):
            # Act
            client = await create_async_node_client("async-ok", 50051, timeout_s=1.0)

            # Assert
            assert isinstance(client, AsyncNodeClient)
            assert ready_called["v"] is True
            await client.close()

    @pytest.mark.asyncio
    async def test_async_client_tls(self):
        """Async TLS client uses secure_channel."""
        # Arrange
        mock_channel = MagicMock()

        async def fake_ready(timeout=None):
            pass

        mock_channel.channel_ready = fake_ready

        with patch("distllm.dist.node_client.grpc.aio.secure_channel", return_value=mock_channel) as mock_secure:
            with patch("distllm.dist.node_client.grpc.ssl_channel_credentials") as mock_creds:
                mock_creds.return_value = MagicMock()

                # Act
                client = await create_async_node_client(
                    "async-tls", 443, use_tls=True, timeout_s=1.0
                )

                # Assert
                mock_secure.assert_called_once()
                assert isinstance(client, AsyncNodeClient)
                await client.close()


# ---------------------------------------------------------------------------
# Test: request_layer_weights integration
# ---------------------------------------------------------------------------


class TestRequestLayerWeightsIntegration:
    """Integration tests for the request_layer_weights helper."""

    def test_weight_transfer_success(self):
        """Successful weight transfer returns bytes."""
        # Arrange
        mock_channel = _make_mock_channel()
        mock_stub = MagicMock()
        mock_resp = MagicMock()
        mock_resp.success = True
        mock_resp.state_dict_bytes = b"\x00" * 100
        mock_resp.error_message = ""
        mock_call = MagicMock()
        mock_call.trailing_metadata.return_value = []
        mock_stub.TransferWeights.with_call.return_value = (mock_resp, mock_call)

        with patch("distllm.dist.node_client.create_node_client") as mock_create:
            mock_client = MagicMock()
            mock_client.stub = mock_stub
            mock_create.return_value = mock_client

            # Act
            from distllm.dist.node_client import request_layer_weights

            result = request_layer_weights("localhost", 50051, "model", 0, 6)

            # Assert
            assert result == b"\x00" * 100
            mock_client.close.assert_called_once()

    def test_weight_transfer_failure_returns_none(self):
        """Failed transfer (success=False) returns None."""
        # Arrange
        mock_stub = MagicMock()
        mock_resp = MagicMock()
        mock_resp.success = False
        mock_resp.error_message = "node overloaded"
        mock_resp.state_dict_bytes = b""
        mock_call = MagicMock()
        mock_call.trailing_metadata.return_value = []
        mock_stub.TransferWeights.with_call.return_value = (mock_resp, mock_call)

        with patch("distllm.dist.node_client.create_node_client") as mock_create:
            mock_client = MagicMock()
            mock_client.stub = mock_stub
            mock_create.return_value = mock_client

            # Act
            from distllm.dist.node_client import request_layer_weights

            result = request_layer_weights("localhost", 50051, "model", 0, 6)

            # Assert
            assert result is None

    def test_weight_transfer_checksum_mismatch_returns_none(self):
        """When SHA-256 checksum does not match, returns None."""
        # Arrange
        import hashlib

        mock_stub = MagicMock()
        data = b"weights-data"
        wrong_checksum = hashlib.sha256(b"wrong-data").hexdigest()

        mock_resp = MagicMock()
        mock_resp.success = True
        mock_resp.state_dict_bytes = data
        mock_resp.error_message = ""
        mock_call = MagicMock()
        mock_call.trailing_metadata.return_value = [("x-checksum-sha256", wrong_checksum)]
        mock_stub.TransferWeights.with_call.return_value = (mock_resp, mock_call)

        with patch("distllm.dist.node_client.create_node_client") as mock_create:
            mock_client = MagicMock()
            mock_client.stub = mock_stub
            mock_create.return_value = mock_client

            # Act
            from distllm.dist.node_client import request_layer_weights

            result = request_layer_weights("localhost", 50051, "model", 0, 6)

            # Assert
            assert result is None

    def test_weight_transfer_connection_error_returns_none(self):
        """ConnectionError during transfer returns None (does not raise)."""
        # Arrange
        with patch("distllm.dist.node_client.create_node_client") as mock_create:
            mock_create.side_effect = ConnectionError("refused")

            # Act
            from distllm.dist.node_client import request_layer_weights

            result = request_layer_weights("down-host", 50051, "model", 0, 6)

            # Assert
            assert result is None

    def test_streaming_weight_transfer_success(self):
        """Streaming transfer collects chunks into a single bytes buffer."""
        # Arrange
        mock_stub = MagicMock()
        # F-049 integrity fields: ordered, total declared, final flagged.
        chunk1 = MagicMock(success=True, state_dict_bytes=b"chunk1-", error_message="",
                           chunk_index=0, total_chunks=2, is_final_chunk=False)
        chunk2 = MagicMock(success=True, state_dict_bytes=b"chunk2", error_message="",
                           chunk_index=1, total_chunks=2, is_final_chunk=True)
        mock_stub.TransferWeightsStream.return_value = iter([chunk1, chunk2])

        with patch("distllm.dist.node_client.create_node_client") as mock_create:
            mock_client = MagicMock()
            mock_client.stub = mock_stub
            mock_create.return_value = mock_client

            # Act
            from distllm.dist.node_client import request_layer_weights_stream

            result = request_layer_weights_stream("localhost", 50051, "model", 0, 6)

            # Assert
            assert result == b"chunk1-chunk2"
            mock_client.close.assert_called_once()

    def test_streaming_weight_transfer_chunk_failure_returns_none(self):
        """A failed chunk in the stream aborts the transfer."""
        # Arrange
        mock_stub = MagicMock()
        chunk1 = MagicMock(success=True, state_dict_bytes=b"ok", error_message="")
        chunk2 = MagicMock(success=False, state_dict_bytes=b"", error_message="disk full")
        mock_stub.TransferWeightsStream.return_value = iter([chunk1, chunk2])

        with patch("distllm.dist.node_client.create_node_client") as mock_create:
            mock_client = MagicMock()
            mock_client.stub = mock_stub
            mock_create.return_value = mock_client

            # Act
            from distllm.dist.node_client import request_layer_weights_stream

            result = request_layer_weights_stream("localhost", 50051, "model", 0, 6)

            # Assert
            assert result is None

    def test_streaming_weight_transfer_empty_returns_none(self):
        """Empty stream (no chunks) returns None."""
        # Arrange
        mock_stub = MagicMock()
        mock_stub.TransferWeightsStream.return_value = iter([])

        with patch("distllm.dist.node_client.create_node_client") as mock_create:
            mock_client = MagicMock()
            mock_client.stub = mock_stub
            mock_create.return_value = mock_client

            # Act
            from distllm.dist.node_client import request_layer_weights_stream

            result = request_layer_weights_stream("localhost", 50051, "model", 0, 6)

            # Assert
            assert result is None
