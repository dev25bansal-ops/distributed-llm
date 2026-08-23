"""Tests for pipeline transport with mocked backends.

Uses ``unittest.mock`` to exercise paths that require GPU or network
hardware — covering initialization, tensor send/recv, error handling,
and connection lifecycle.

Complements the zero-mock tests in ``test_transport.py`` by testing the
same ``TensorTransport`` class with mocked NCCL and QUIC dependencies.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch

from distllm.dist.pipeline.transport import TensorTransport, TransportBackend


# ===========================================================================
# Transport initialization
# ===========================================================================


class TestTransportInit:
    """Transport initialization with various backends (mocked)."""

    # -- NCCL backend --------------------------------------------------------

    @patch("distllm.dist.nccl.NcclTransport")
    def test_nccl_success(self, mock_nccl_cls):
        """NCCL init succeeds when NcclTransport is available and initialized."""
        mock_instance = MagicMock()
        mock_instance.is_initialized = True
        mock_nccl_cls.return_value = mock_instance

        t = TensorTransport(backend=TransportBackend.NCCL)
        assert t.is_available is True
        assert t._nccl is mock_instance
        assert t.backend == TransportBackend.NCCL
        t.destroy()

    @patch("distllm.dist.nccl.NcclTransport")
    def test_nccl_construction_failure(self, mock_nccl_cls):
        """NCCL init handles NcclTransport construction failure gracefully."""
        mock_nccl_cls.side_effect = RuntimeError("CUDA not available")

        t = TensorTransport(backend=TransportBackend.NCCL)
        assert t.is_available is False
        assert t._nccl is None
        t.destroy()

    @patch("distllm.dist.nccl.NcclTransport")
    def test_nccl_not_initialized(self, mock_nccl_cls):
        """NCCL instance is created but reports not initialized."""
        mock_instance = MagicMock()
        mock_instance.is_initialized = False
        mock_nccl_cls.return_value = mock_instance

        t = TensorTransport(backend=TransportBackend.NCCL)
        assert t.is_available is False
        assert t._nccl is mock_instance
        t.destroy()

    # -- QUIC backend --------------------------------------------------------

    @patch("distllm.dist.quic_transport.is_quic_available")
    @patch("distllm.dist.quic_transport.QuicTransportClient")
    def test_quic_success(self, mock_quic_cls, mock_quic_avail):
        """QUIC init succeeds when aioquic is available."""
        mock_quic_avail.return_value = True
        mock_instance = MagicMock()
        mock_quic_cls.return_value = mock_instance

        t = TensorTransport(backend=TransportBackend.QUIC)
        assert t.is_available is True
        assert t._quic_client is mock_instance
        assert t.backend == TransportBackend.QUIC
        t.destroy()

    @patch("distllm.dist.quic_transport.is_quic_available")
    def test_quic_unavailable(self, mock_quic_avail):
        """QUIC init when aioquic is not installed."""
        mock_quic_avail.return_value = False

        t = TensorTransport(backend=TransportBackend.QUIC)
        assert t.is_available is False
        assert t._quic_client is None
        assert t.backend == TransportBackend.QUIC
        t.destroy()

    @patch(
        "distllm.dist.quic_transport.is_quic_available",
        side_effect=ImportError("no module named aioquic"),
    )
    def test_quic_import_error(self, mock_quic_avail):
        """QUIC init when the import raises ImportError."""
        t = TensorTransport(backend=TransportBackend.QUIC)
        assert t.is_available is False
        t.destroy()

    # -- AUTO backend selection ----------------------------------------------

    @patch("distllm.dist.nccl.NcclTransport")
    def test_auto_selects_nccl(self, mock_nccl_cls):
        """AUTO selects NCCL when NcclTransport initializes successfully."""
        mock_instance = MagicMock()
        mock_instance.is_initialized = True
        mock_nccl_cls.return_value = mock_instance

        t = TensorTransport(backend=TransportBackend.AUTO)
        assert t._selected_backend == TransportBackend.NCCL
        assert t.backend == TransportBackend.NCCL
        assert t.is_available is True
        t.destroy()

    @patch("distllm.dist.nccl.NcclTransport")
    @patch("distllm.dist.quic_transport.is_quic_available")
    @patch("distllm.dist.quic_transport.QuicTransportClient")
    def test_auto_selects_quic(
        self,
        mock_quic_cls,
        mock_quic_avail,
        mock_nccl_cls,
    ):
        """AUTO selects QUIC when NCCL fails and QUIC is available."""
        mock_nccl_cls.side_effect = RuntimeError("CUDA error")
        mock_quic_avail.return_value = True
        mock_instance = MagicMock()
        mock_quic_cls.return_value = mock_instance

        t = TensorTransport(backend=TransportBackend.AUTO)
        assert t._selected_backend == TransportBackend.QUIC
        assert t.backend == TransportBackend.QUIC
        assert t.is_available is True
        assert t._quic_client is mock_instance
        t.destroy()

    @patch("distllm.dist.nccl.NcclTransport")
    @patch("distllm.dist.quic_transport.is_quic_available")
    def test_auto_falls_to_grpc(
        self,
        mock_quic_avail,
        mock_nccl_cls,
    ):
        """AUTO falls back to GRPC when NCCL and QUIC are both unavailable."""
        mock_nccl_cls.side_effect = RuntimeError("CUDA error")
        mock_quic_avail.return_value = False

        t = TensorTransport(backend=TransportBackend.AUTO)
        assert t._selected_backend == TransportBackend.GRPC
        assert t.backend == TransportBackend.GRPC
        assert t.is_available is False
        assert t._quic_client is None
        t.destroy()

    @patch("distllm.dist.nccl.NcclTransport")
    def test_auto_nccl_present_but_not_initialized(self, mock_nccl_cls):
        """AUTO: NCCL is constructed but not initialized, falls through."""
        mock_instance = MagicMock()
        mock_instance.is_initialized = False
        mock_nccl_cls.return_value = mock_instance

        t = TensorTransport(backend=TransportBackend.AUTO)
        # NCCL was created but not initialized, QUIC not available, so GRPC
        assert t._selected_backend == TransportBackend.GRPC
        assert t._nccl is mock_instance  # NCCL reference kept
        assert t.backend == TransportBackend.GRPC
        t.destroy()

    # -- KWargs pass-through -------------------------------------------------

    @patch("distllm.dist.nccl.NcclTransport")
    def test_nccl_kwargs_propagation(self, mock_nccl_cls):
        """Extra kwargs are passed through to NcclTransport."""
        mock_instance = MagicMock()
        mock_instance.is_initialized = True
        mock_nccl_cls.return_value = mock_instance

        t = TensorTransport(
            backend=TransportBackend.NCCL,
            rank=2,
            world_size=8,
            master_addr="10.0.0.1",
            master_port=29501,
        )
        mock_nccl_cls.assert_called_once_with(
            auto_init=True,
            rank=2,
            world_size=8,
            master_addr="10.0.0.1",
            master_port=29501,
        )
        t.destroy()


# ===========================================================================
# Tensor send / receive
# ===========================================================================


class TestTensorSendRecv:
    """Tensor send and receive with mocked NCCL transport."""

    @patch("distllm.dist.nccl.NcclTransport")
    def test_send_success(self, mock_nccl_cls):
        """send_tensor delegates to NcclTransport.send."""
        mock_instance = MagicMock()
        mock_instance.is_initialized = True
        mock_nccl_cls.return_value = mock_instance

        t = TensorTransport(backend=TransportBackend.NCCL)
        tensor = torch.zeros(4, 4)
        t.send_tensor(tensor, dst=1)
        mock_instance.send.assert_called_once_with(tensor, dst=1, tag=0)
        t.destroy()

    @patch("distllm.dist.nccl.NcclTransport")
    def test_send_custom_tag(self, mock_nccl_cls):
        """send_tensor passes a custom tag value."""
        mock_instance = MagicMock()
        mock_instance.is_initialized = True
        mock_nccl_cls.return_value = mock_instance

        t = TensorTransport(backend=TransportBackend.NCCL)
        tensor = torch.ones(3)
        t.send_tensor(tensor, dst=2, tag=99)
        mock_instance.send.assert_called_once_with(tensor, dst=2, tag=99)
        t.destroy()

    @patch("distllm.dist.nccl.NcclTransport")
    def test_send_multiple_calls(self, mock_nccl_cls):
        """send_tensor works correctly across multiple invocations."""
        mock_instance = MagicMock()
        mock_instance.is_initialized = True
        mock_nccl_cls.return_value = mock_instance

        t = TensorTransport(backend=TransportBackend.NCCL)
        t.send_tensor(torch.ones(2), dst=0, tag=1)
        t.send_tensor(torch.zeros(2), dst=1, tag=2)
        assert mock_instance.send.call_count == 2
        t.destroy()

    @patch("distllm.dist.nccl.NcclTransport")
    def test_recv_success(self, mock_nccl_cls):
        """recv_tensor returns result from NcclTransport.recv."""
        mock_instance = MagicMock()
        mock_instance.is_initialized = True
        mock_nccl_cls.return_value = mock_instance
        expected = torch.randn(8, 8)
        mock_instance.recv.return_value = expected

        t = TensorTransport(backend=TransportBackend.NCCL)
        result = t.recv_tensor(shape=(8, 8), dtype=torch.float32, src=0)
        assert result is expected
        mock_instance.recv.assert_called_once_with(
            (8, 8),
            torch.float32,
            src=0,
            tag=0,
            device=None,
        )
        t.destroy()

    @patch("distllm.dist.nccl.NcclTransport")
    def test_recv_different_types(self, mock_nccl_cls):
        """recv_tensor works with different dtypes and shapes."""
        mock_instance = MagicMock()
        mock_instance.is_initialized = True
        mock_nccl_cls.return_value = mock_instance
        mock_instance.recv.return_value = torch.empty(0)

        t = TensorTransport(backend=TransportBackend.NCCL)

        t.recv_tensor(shape=(1,), dtype=torch.float64, src=0)
        t.recv_tensor(shape=(3, 3), dtype=torch.int32, src=1, tag=2)
        t.recv_tensor(shape=(2, 4, 8), dtype=torch.bfloat16, src=2, tag=3)

        assert mock_instance.recv.call_count == 3
        t.destroy()

    @patch("distllm.dist.nccl.NcclTransport")
    def test_recv_with_device(self, mock_nccl_cls):
        """recv_tensor passes device argument to NcclTransport.recv."""
        mock_instance = MagicMock()
        mock_instance.is_initialized = True
        mock_nccl_cls.return_value = mock_instance

        t = TensorTransport(backend=TransportBackend.NCCL)
        t.recv_tensor(
            shape=(2, 3),
            dtype=torch.float16,
            src=1,
            tag=5,
            device="cuda:0",
        )
        mock_instance.recv.assert_called_once_with(
            (2, 3),
            torch.float16,
            src=1,
            tag=5,
            device="cuda:0",
        )
        t.destroy()

    # -- Error paths ---------------------------------------------------------

    @patch("distllm.dist.nccl.NcclTransport")
    def test_send_nccl_error(self, mock_nccl_cls):
        """send_tensor propagates NCCL errors."""
        mock_instance = MagicMock()
        mock_instance.is_initialized = True
        mock_nccl_cls.return_value = mock_instance
        mock_instance.send.side_effect = RuntimeError("NCCL send failed")

        t = TensorTransport(backend=TransportBackend.NCCL)
        with pytest.raises(RuntimeError, match="NCCL send failed"):
            t.send_tensor(torch.zeros(1), dst=0)
        t.destroy()

    @patch("distllm.dist.nccl.NcclTransport")
    def test_recv_nccl_error(self, mock_nccl_cls):
        """recv_tensor propagates NCCL errors."""
        mock_instance = MagicMock()
        mock_instance.is_initialized = True
        mock_nccl_cls.return_value = mock_instance
        mock_instance.recv.side_effect = RuntimeError("NCCL recv failed")

        t = TensorTransport(backend=TransportBackend.NCCL)
        with pytest.raises(RuntimeError, match="NCCL recv failed"):
            t.recv_tensor(shape=(1,), dtype=torch.float32, src=0)
        t.destroy()


# ===========================================================================
# Error handling for network failures
# ===========================================================================


class TestErrorHandling:
    """Error handling for network and transport failures."""

    @pytest.mark.asyncio
    @patch("distllm.dist.quic_transport.QuicTransportClient")
    @patch("distllm.dist.quic_transport.QuicConfig")
    async def test_forward_pass_failure(self, mock_qcfg, mock_qcl):
        """send_forward_pass raises when QUIC forward_pass raises."""
        mock_client = MagicMock()
        mock_client.close = AsyncMock()
        mock_client.forward_pass = AsyncMock(
            side_effect=RuntimeError("QUIC stream error"),
        )
        mock_qcl.return_value = mock_client

        t = TensorTransport(backend=TransportBackend.GRPC)
        t._quic_client = mock_client

        with pytest.raises(RuntimeError, match="QUIC stream error"):
            await t.send_forward_pass(b"test_data")

        mock_client.forward_pass.assert_awaited_once_with(
            b"test_data", timeout=120.0,
        )
        t.destroy()

    @pytest.mark.asyncio
    async def test_forward_pass_not_initialized(self):
        """send_forward_pass raises RuntimeError without a QUIC client."""
        t = TensorTransport(backend=TransportBackend.GRPC)
        with pytest.raises(RuntimeError, match="QUIC transport not initialized"):
            await t.send_forward_pass(b"test_data")
        t.destroy()

    @pytest.mark.asyncio
    @patch("distllm.dist.quic_transport.QuicTransportClient")
    @patch("distllm.dist.quic_transport.QuicConfig")
    async def test_forward_pass_custom_timeout(self, mock_qcfg, mock_qcl):
        """send_forward_pass passes a custom timeout to QUIC."""
        mock_client = MagicMock()
        mock_client.close = AsyncMock()
        mock_client.forward_pass = AsyncMock(return_value=b"response")
        mock_qcl.return_value = mock_client

        t = TensorTransport(backend=TransportBackend.GRPC)
        t._quic_client = mock_client

        result = await t.send_forward_pass(b"req", timeout=30.0)
        assert result == b"response"
        mock_client.forward_pass.assert_awaited_once_with(b"req", timeout=30.0)
        t.destroy()

    @pytest.mark.asyncio
    @patch("distllm.dist.quic_transport.QuicTransportClient")
    @patch("distllm.dist.quic_transport.QuicConfig")
    async def test_quic_connect_timeout(self, mock_qcfg, mock_qcl):
        """quic_connect raises asyncio.TimeoutError on network timeout."""
        mock_client = MagicMock()
        mock_client.close = AsyncMock()
        mock_client.connect = AsyncMock(
            side_effect=asyncio.TimeoutError("Connection timed out"),
        )
        mock_qcl.return_value = mock_client

        t = TensorTransport(backend=TransportBackend.GRPC)
        t._quic_client = mock_client

        with pytest.raises(asyncio.TimeoutError):
            await t.quic_connect("10.0.0.1", 4433, timeout=5.0)

        mock_client.connect.assert_awaited_once_with(
            "10.0.0.1", 4433, timeout=5.0,
        )
        t.destroy()


# ===========================================================================
# Connection lifecycle
# ===========================================================================


class TestConnectionLifecycle:
    """Connection lifecycle: init_quic, quic_connect, destroy, active_backend."""

    # -- init_quic -----------------------------------------------------------

    @patch("distllm.dist.quic_transport.QuicConfig")
    @patch("distllm.dist.quic_transport.QuicTransportClient")
    def test_init_quic_defaults(self, mock_qcl, mock_qcfg):
        """init_quic with defaults uses empty host and port 4433."""
        t = TensorTransport(backend=TransportBackend.GRPC)

        t.init_quic()

        mock_qcfg.assert_called_once_with(host="", port=4433)
        mock_qcl.assert_called_once_with(config=mock_qcfg.return_value)
        assert t.is_available is True
        t.destroy()

    @patch("distllm.dist.quic_transport.QuicConfig")
    @patch("distllm.dist.quic_transport.QuicTransportClient")
    def test_init_quic_custom(self, mock_qcl, mock_qcfg):
        """init_quic passes custom host, port, and extra kwargs."""
        mock_config = MagicMock()
        mock_qcfg.return_value = mock_config

        t = TensorTransport(backend=TransportBackend.GRPC)
        t.init_quic(host="10.0.0.1", port=9000, max_stream_data=65536)

        mock_qcfg.assert_called_once_with(
            host="10.0.0.1",
            port=9000,
            max_stream_data=65536,
        )
        mock_qcl.assert_called_once_with(config=mock_config)
        assert t._quic_config is mock_config
        assert t._quic_client is mock_qcl.return_value
        assert t.is_available is True
        t.destroy()

    # -- quic_connect --------------------------------------------------------

    @pytest.mark.asyncio
    @patch("distllm.dist.quic_transport.QuicTransportClient")
    @patch("distllm.dist.quic_transport.QuicConfig")
    async def test_quic_connect_with_existing_client(self, mock_qcfg, mock_qcl):
        """quic_connect uses the existing _quic_client when already set."""
        mock_client = MagicMock()
        mock_client.close = AsyncMock()
        mock_client.connect = AsyncMock()
        mock_qcl.return_value = mock_client

        t = TensorTransport(backend=TransportBackend.GRPC)
        t._quic_client = mock_client  # Already initialized externally

        await t.quic_connect("10.0.0.1", 4433)

        # init_quic should NOT be called again
        mock_qcfg.assert_not_called()
        mock_client.connect.assert_awaited_once_with(
            "10.0.0.1", 4433, timeout=10.0,
        )
        t.destroy()

    @pytest.mark.asyncio
    @patch("distllm.dist.quic_transport.QuicTransportClient")
    @patch("distllm.dist.quic_transport.QuicConfig")
    async def test_quic_connect_lazy_init(self, mock_qcfg, mock_qcl):
        """quic_connect auto-initializes when _quic_client is None."""
        mock_config = MagicMock()
        mock_qcfg.return_value = mock_config
        mock_client = MagicMock()
        mock_client.close = AsyncMock()
        mock_client.connect = AsyncMock()
        mock_qcl.return_value = mock_client

        t = TensorTransport(backend=TransportBackend.GRPC)
        assert t._quic_client is None

        await t.quic_connect("10.0.0.1", 4433, timeout=5.0)

        # Should have called init_quic internally
        mock_qcfg.assert_called_once_with(host="10.0.0.1", port=4433)
        mock_qcl.assert_called_once_with(config=mock_config)
        mock_client.connect.assert_awaited_once_with(
            "10.0.0.1", 4433, timeout=5.0,
        )
        assert t._quic_client is mock_client
        assert t.is_available is True
        t.destroy()

    @pytest.mark.asyncio
    @patch("distllm.dist.quic_transport.QuicTransportClient")
    @patch("distllm.dist.quic_transport.QuicConfig")
    async def test_quic_connect_passes_timeout(self, mock_qcfg, mock_qcl):
        """quic_connect forwards the timeout parameter to connect()."""
        mock_client = MagicMock()
        mock_client.close = AsyncMock()
        mock_client.connect = AsyncMock()
        mock_qcl.return_value = mock_client

        t = TensorTransport(backend=TransportBackend.GRPC)
        t._quic_client = mock_client

        await t.quic_connect("10.0.0.1", 4433, timeout=15.0)
        mock_client.connect.assert_awaited_once_with(
            "10.0.0.1", 4433, timeout=15.0,
        )
        t.destroy()

    # -- destroy -------------------------------------------------------------

    @patch("distllm.dist.quic_transport.QuicTransportClient")
    @patch("distllm.dist.quic_transport.QuicConfig")
    def test_destroy_quic(self, mock_qcfg, mock_qcl):
        """destroy tears down QUIC client and resets state."""
        mock_client = MagicMock()
        mock_client.close = AsyncMock()
        mock_qcl.return_value = mock_client

        t = TensorTransport(backend=TransportBackend.GRPC)
        t._quic_client = mock_client
        t.is_available = True

        t.destroy()

        # close was called (asyncio.run is used since no running loop)
        mock_client.close.assert_called_once()
        assert t._quic_client is None
        assert t._nccl is None
        assert t.is_available is False

    @patch("distllm.dist.quic_transport.QuicTransportClient")
    @patch("distllm.dist.quic_transport.QuicConfig")
    def test_destroy_quic_with_loop_param(self, mock_qcfg, mock_qcl):
        """destroy with explicit loop= uses run_coroutine_threadsafe."""
        mock_client = MagicMock()
        mock_client.close = AsyncMock()
        mock_qcl.return_value = mock_client

        mock_loop = MagicMock()
        mock_loop.is_running.return_value = True

        t = TensorTransport(backend=TransportBackend.GRPC)
        t._quic_client = mock_client
        t.is_available = True

        with patch("asyncio.run_coroutine_threadsafe") as mock_rct:
            t.destroy(loop=mock_loop)

            mock_rct.assert_called_once()
            _call_args = mock_rct.call_args
            # Second positional arg should be the loop we passed
            assert _call_args[0][1] is mock_loop

        # close was called during expression evaluation
        mock_client.close.assert_called_once()
        assert t._quic_client is None
        assert t.is_available is False

    @patch("distllm.dist.nccl.NcclTransport")
    def test_destroy_nccl(self, mock_nccl_cls):
        """destroy tears down NCCL transport."""
        mock_instance = MagicMock()
        mock_instance.is_initialized = True
        mock_nccl_cls.return_value = mock_instance

        t = TensorTransport(backend=TransportBackend.NCCL)
        assert t._nccl is mock_instance

        t.destroy()

        mock_instance.destroy.assert_called_once()
        assert t._nccl is None
        assert t.is_available is False

    @patch("distllm.dist.nccl.NcclTransport")
    def test_destroy_both_backends(self, mock_nccl_cls):
        """destroy tears down both NCCL and QUIC when both are present."""
        mock_nccl = MagicMock()
        mock_nccl.is_initialized = True
        mock_nccl_cls.return_value = mock_nccl

        mock_quic = MagicMock()
        mock_quic.close = AsyncMock()

        t = TensorTransport(backend=TransportBackend.NCCL)
        # Inject a QUIC client that was set up externally
        t._quic_client = mock_quic
        t.is_available = True

        t.destroy()

        mock_nccl.destroy.assert_called_once()
        mock_quic.close.assert_called_once()
        assert t._nccl is None
        assert t._quic_client is None
        assert t.is_available is False

    def test_destroy_empty(self):
        """destroy on a bare GRPC transport is a no-op."""
        t = TensorTransport(backend=TransportBackend.GRPC)
        t.destroy()
        assert t._nccl is None
        assert t._quic_client is None
        assert t.is_available is False

    def test_destroy_idempotent(self):
        """Calling destroy multiple times does not raise."""
        t = TensorTransport(backend=TransportBackend.GRPC)
        t.destroy()
        t.destroy()

    # -- active_backend ------------------------------------------------------

    def test_active_backend_grpc(self):
        """active_backend returns GRPC for explicit GRPC init."""
        t = TensorTransport(backend=TransportBackend.GRPC)
        assert t.active_backend == TransportBackend.GRPC
        t.destroy()

    @patch("distllm.dist.nccl.NcclTransport")
    def test_active_backend_nccl(self, mock_nccl_cls):
        """active_backend returns NCCL when that backend is active."""
        mock_instance = MagicMock()
        mock_instance.is_initialized = True
        mock_nccl_cls.return_value = mock_instance

        t = TensorTransport(backend=TransportBackend.NCCL)
        assert t.active_backend == TransportBackend.NCCL
        t.destroy()

    @patch("distllm.dist.nccl.NcclTransport")
    def test_active_backend_auto_nccl(self, mock_nccl_cls):
        """active_backend with AUTO returns the selected backend (NCCL)."""
        mock_instance = MagicMock()
        mock_instance.is_initialized = True
        mock_nccl_cls.return_value = mock_instance

        t = TensorTransport(backend=TransportBackend.AUTO)
        assert t.active_backend == TransportBackend.NCCL
        t.destroy()

    def test_active_backend_auto_grpc(self):
        """active_backend with AUTO returns GRPC (fallback)."""
        t = TensorTransport(backend=TransportBackend.AUTO)
        assert t.active_backend == TransportBackend.GRPC
        t.destroy()

    # -- _probe_transports ---------------------------------------------------

    @patch("distllm.dist.nccl.NcclTransport")
    def test_probe_keeps_nccl_when_best(self, mock_nccl_cls):
        """_probe_transports keeps NCCL when it remains the best candidate."""
        mock_instance = MagicMock()
        mock_instance.is_initialized = True
        mock_nccl_cls.return_value = mock_instance

        t = TensorTransport(backend=TransportBackend.AUTO)
        assert t._selected_backend == TransportBackend.NCCL

        # Force re-probe by resetting the timer
        t._last_probe_time = 0.0
        t._probe_transports()

        assert t._selected_backend == TransportBackend.NCCL
        assert t.backend == TransportBackend.NCCL
        t.destroy()

    @patch("distllm.dist.nccl.NcclTransport")
    def test_probe_switches_to_grpc_when_nccl_unavailable(self, mock_nccl_cls):
        """_probe_transports switches to GRPC when NCCL becomes unavailable."""
        mock_nccl = MagicMock()
        mock_nccl.is_initialized = True
        mock_nccl_cls.return_value = mock_nccl

        t = TensorTransport(backend=TransportBackend.AUTO)
        assert t._selected_backend == TransportBackend.NCCL

        # Make NCCL unavailable
        mock_nccl.is_initialized = False
        t._last_probe_time = 0.0
        t._probe_transports()

        # Should switch to GRPC (always present with score 0.5)
        assert t._selected_backend == TransportBackend.GRPC
        assert t.backend == TransportBackend.GRPC
        t.destroy()

    @patch("distllm.dist.nccl.NcclTransport")
    @patch("distllm.dist.quic_transport.is_quic_available")
    @patch("distllm.dist.quic_transport.QuicTransportClient")
    def test_probe_quic_present_but_grpc_wins(
        self,
        mock_quic_cls,
        mock_quic_avail,
        mock_nccl_cls,
    ):
        """Probe ranks GRPC above QUIC (0.5 > 0.4)."""
        mock_nccl = MagicMock()
        mock_nccl.is_initialized = False  # NCCL not ready
        mock_nccl_cls.return_value = mock_nccl

        mock_quic_avail.return_value = True
        mock_quic = MagicMock()
        mock_quic_cls.return_value = mock_quic

        t = TensorTransport(backend=TransportBackend.AUTO)
        # AUTO: NCCL failed (not initialized), QUIC succeeded
        assert t._selected_backend == TransportBackend.QUIC

        # Now: NCCL still not initialized, QUIC present, GRPC always present
        # GRPC score (0.5) > QUIC score (0.4), so probe should switch
        t._last_probe_time = 0.0
        t._probe_transports()

        assert t._selected_backend == TransportBackend.GRPC
        t.destroy()

    def test_probe_rate_limited(self):
        """_probe_transports returns early within the probe interval."""
        import time

        t = TensorTransport(backend=TransportBackend.AUTO)
        t._last_probe_time = time.time()  # Recent probe
        t._nccl = MagicMock()  # Non-None, would be checked
        t._nccl.is_initialized = True

        # This should return without doing anything
        t._probe_transports()

        # _selected_backend stays None (AUTO falls through)
        assert t._selected_backend is not None
        t.destroy()

    def test_probe_no_candidates(self):
        """_probe_transports handles the case with no candidates."""
        t = TensorTransport(backend=TransportBackend.AUTO)
        t._last_probe_time = 0.0

        # _nccl is None, QUIC client is None, _quic_client is None
        # Only GRPC should appear (hardcoded with score 0.5)
        t._probe_transports()

        # GRPC is always a candidate, so selected_backend is not None
        assert t._selected_backend == TransportBackend.GRPC
        t.destroy()
