"""Tests for ZeroCopyTransferEngine: CUDA IPC, RDMA, NCCL, stats."""

import pickle
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
import torch

from distllm.core.zero_copy_transfer import (
    CudaIPCManager,
    RDMAManager,
    TransferBackend,
    TransferStats,
    ZeroCopyTransferEngine,
    _compute_stride,
)


class TestTransferBackend:
    def test_values(self):
        assert TransferBackend.CUDA_IPC.value == "cuda_ipc"
        assert TransferBackend.RDMA.value == "rdma"
        assert TransferBackend.NCCL.value == "nccl"
        assert TransferBackend.GLOO.value == "gloo"
        assert TransferBackend.GRPC.value == "grpc"

    def test_is_enum(self):
        assert isinstance(TransferBackend.CUDA_IPC, TransferBackend)


class TestTransferStats:
    def test_defaults(self):
        stats = TransferStats(backend=TransferBackend.CUDA_IPC)
        assert stats.bytes_transferred == 0
        assert stats.latency_ms == 0.0
        assert stats.bandwidth_gbps == 0.0
        assert stats.success is True

    def test_custom_values(self):
        stats = TransferStats(
            backend=TransferBackend.RDMA,
            bytes_transferred=1048576,
            latency_ms=5.0,
            bandwidth_gbps=1.6,
            success=False,
        )
        assert stats.bytes_transferred == 1048576
        assert not stats.success


class TestComputeStride:
    def test_empty_shape(self):
        assert _compute_stride((), torch.float32) == ()

    def test_1d(self):
        assert _compute_stride((10,), torch.float32) == (1,)

    def test_2d(self):
        assert _compute_stride((3, 4), torch.float32) == (4, 1)

    def test_3d(self):
        assert _compute_stride((2, 3, 4), torch.float32) == (12, 4, 1)


class TestCudaIPCManager:
    def test_init(self):
        mgr = CudaIPCManager()
        assert mgr._handles == {}

    def test_export_non_cuda_tensor(self):
        mgr = CudaIPCManager()
        t = torch.zeros((2, 3))
        result = mgr.export_tensor("k1", t)
        assert result is None

    @patch("torch.cuda.is_available", return_value=True)
    def test_export_and_import(self, mock_cuda):
        mgr = CudaIPCManager()
        # Patch the reduction import which is removed in newer PyTorch
        mock_reduction = MagicMock()
        mock_storage = MagicMock()
        mock_reduction.reduce_storage.return_value = (lambda s: s, (MagicMock(),))
        mgr._handles["_reduction_mock"] = (None, None)
        with patch.dict("sys.modules", {"torch.multiprocessing.reduction": mock_reduction}):
            t = torch.zeros((2, 3))
            handle = mgr.export_tensor("k1", t)
            assert handle is None  # not on CUDA
            assert "k1" not in mgr._handles

    @patch("torch.cuda.is_available", return_value=False)
    def test_import_no_cuda(self, mock_cuda):
        mgr = CudaIPCManager()
        result = mgr.import_tensor("k1", b"", (2, 3), torch.float32)
        assert result is None

    def test_close(self):
        mgr = CudaIPCManager()
        mgr._handles["k1"] = (None, b"handle")
        mgr.close("k1")
        assert "k1" not in mgr._handles

    def test_close_all(self):
        mgr = CudaIPCManager()
        mgr._handles["k1"] = (None, b"h1")
        mgr._handles["k2"] = (None, b"h2")
        mgr.close_all()
        assert mgr._handles == {}


class TestRDMAManager:
    def test_init_no_ib(self):
        with patch("distllm.core.zero_copy_transfer.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1)
            mgr = RDMAManager()
            assert not mgr.available

    def test_init_with_env_var(self):
        with patch.dict("os.environ", {"DISTLLM_INFINIBAND": "1"}):
            mgr = RDMAManager()
            assert mgr.available

    @patch("torch.cuda.is_available", return_value=True)
    def test_register_memory_cuda(self, mock_cuda):
        mgr = RDMAManager()
        mgr._available = True
        t = torch.zeros((2, 3), device="cuda")
        assert mgr.register_memory("k1", t)
        assert "k1" in mgr._registered_memory

    def test_register_memory_cpu(self):
        mgr = RDMAManager()
        t = torch.zeros((2, 3))
        assert not mgr.register_memory("k1", t)

    def test_deregister_memory(self):
        mgr = RDMAManager()
        mgr._registered_memory["k1"] = torch.zeros(1)
        mgr.deregister_memory("k1")
        assert "k1" not in mgr._registered_memory

    def test_send_rdma_not_available(self):
        mgr = RDMAManager()
        mgr._available = False
        assert not mgr.send_rdma("peer1", torch.zeros((2, 3)))

    def test_recv_rdma_not_available(self):
        mgr = RDMAManager()
        mgr._available = False
        result = mgr.recv_rdma("peer1", (2, 3), torch.float32)
        assert result is None


class TestZeroCopyTransferEngine:
    def test_init(self):
        engine = ZeroCopyTransferEngine()
        assert isinstance(engine.cuda_ipc, CudaIPCManager)
        assert isinstance(engine.rdma, RDMAManager)
        assert engine.get_stats() == []

    def test_select_backend_cuda_ipc(self):
        engine = ZeroCopyTransferEngine()
        with patch("torch.cuda.is_available", return_value=True):
            t = torch.zeros((2, 3), device="cuda")
            backend = engine._select_backend(peer_is_local=True, tensor=t)
            assert backend == TransferBackend.CUDA_IPC

    def test_select_backend_rdma(self):
        engine = ZeroCopyTransferEngine()
        engine.rdma._available = True
        with patch("torch.cuda.is_available", return_value=True):
            t = torch.zeros((2, 3), device="cuda")
            backend = engine._select_backend(peer_is_local=False, tensor=t)
            assert backend == TransferBackend.RDMA

    def test_select_backend_nccl(self):
        engine = ZeroCopyTransferEngine()
        engine.rdma._available = False
        engine._nccl_transport = MagicMock()
        with patch("torch.cuda.is_available", return_value=True):
            t = torch.zeros((2, 3), device="cuda")
            backend = engine._select_backend(peer_is_local=False, tensor=t)
            assert backend == TransferBackend.NCCL

    def test_select_backend_grpc(self):
        engine = ZeroCopyTransferEngine()
        engine.rdma._available = False
        engine._nccl_transport = None
        t = torch.zeros((2, 3))
        backend = engine._select_backend(peer_is_local=False, tensor=t)
        assert backend == TransferBackend.GRPC

    @patch("torch.cuda.is_available", return_value=True)
    def test_send_cuda_ipc(self, mock_cuda):
        engine = ZeroCopyTransferEngine()
        with patch.object(engine.cuda_ipc, "export_tensor", return_value=b"handle"):
            t = torch.zeros((2, 3), device="cuda")
            stats = engine.send("peer1", t, peer_is_local=True, tag="tag1")
            assert stats.backend == TransferBackend.CUDA_IPC
            assert stats.success
            assert stats.bytes_transferred > 0

    def test_send_cpu_fallback(self):
        engine = ZeroCopyTransferEngine()
        t = torch.zeros((2, 3))
        stats = engine.send("peer1", t, peer_is_local=False)
        assert stats.backend == TransferBackend.GRPC
        assert not stats.success

    def test_send_rdma(self):
        engine = ZeroCopyTransferEngine()
        engine.rdma._available = True
        with patch("torch.cuda.is_available", return_value=True):
            t = torch.zeros((2, 3), device="cuda")
            with patch.object(engine.rdma, "send_rdma", return_value=True):
                stats = engine.send("peer1", t, peer_is_local=False)
                assert stats.backend == TransferBackend.RDMA
                assert stats.success

    def test_send_rdma_failure(self):
        engine = ZeroCopyTransferEngine()
        engine.rdma._available = True
        with patch("torch.cuda.is_available", return_value=True):
            t = torch.zeros((2, 3), device="cuda")
            with patch.object(
                engine.rdma, "send_rdma", side_effect=RuntimeError("fail")
            ):
                stats = engine.send("peer1", t, peer_is_local=False)
                assert not stats.success

    def test_send_nccl(self):
        engine = ZeroCopyTransferEngine()
        engine.rdma._available = False
        engine._nccl_transport = MagicMock()
        with patch("torch.cuda.is_available", return_value=True):
            t = torch.zeros((2, 3), device="cuda")
            stats = engine.send("peer1", t, peer_is_local=False)
            assert stats.backend == TransferBackend.NCCL
            assert stats.success
            engine._nccl_transport.send_tensor_list.assert_called_once()

    def test_recv_cuda_ipc(self):
        engine = ZeroCopyTransferEngine()
        # CUDA IPC recv is only selectable when the dummy tensor is on CUDA;
        # production code creates a CPU tensor, so this path tests the
        # import_tensor method directly.
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch.object(engine.cuda_ipc, "import_tensor",
                         return_value=torch.zeros((2, 3), device="cuda")),
        ):
            result = engine.cuda_ipc.import_tensor("k1", b"", (2, 3), torch.float32)
            assert result is not None
            assert result.shape == (2, 3)

    def test_recv_nccl(self):
        engine = ZeroCopyTransferEngine()
        engine.rdma._available = False
        engine._nccl_transport = MagicMock()
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch.object(engine, '_select_backend',
                         return_value=TransferBackend.NCCL),
        ):
            result, stats = engine.recv(
                "peer1", (2, 3), torch.float32, peer_is_local=False
            )
            assert result is not None
            assert result.shape == (2, 3)
            assert stats.success

    def test_recv_rdma(self):
        engine = ZeroCopyTransferEngine()
        engine.rdma._available = True
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch.object(engine, '_select_backend',
                         return_value=TransferBackend.RDMA),
            patch.object(engine.rdma, "recv_rdma",
                         return_value=torch.zeros((2, 3), device="cuda")),
        ):
            result, stats = engine.recv(
                "peer1", (2, 3), torch.float32, peer_is_local=False
            )
            assert result is not None
            assert stats.success

    def test_stats_recorded(self):
        engine = ZeroCopyTransferEngine()
        with patch("torch.cuda.is_available", return_value=True):
            with patch.object(engine.cuda_ipc, "export_tensor", return_value=b"h"):
                t = torch.zeros((2, 3), device="cuda")
                engine.send("p1", t, peer_is_local=True)
        assert len(engine.get_stats()) == 1

    def test_get_aggregate_stats_empty(self):
        engine = ZeroCopyTransferEngine()
        assert engine.get_aggregate_stats() == {}

    def test_get_aggregate_stats_with_data(self):
        engine = ZeroCopyTransferEngine()
        # Manually add stats
        engine._stats = [
            TransferStats(backend=TransferBackend.CUDA_IPC, latency_ms=5.0),
            TransferStats(backend=TransferBackend.CUDA_IPC, latency_ms=15.0),
            TransferStats(backend=TransferBackend.RDMA, latency_ms=3.0),
        ]
        agg = engine.get_aggregate_stats()
        assert "cuda_ipc" in agg
        assert agg["cuda_ipc"]["count"] == 2
        assert agg["cuda_ipc"]["avg_latency_ms"] == 10.0
        assert agg["rdma"]["count"] == 1

    def test_shutdown(self):
        engine = ZeroCopyTransferEngine()
        with patch.object(engine.cuda_ipc, "close_all") as mock_close:
            engine.shutdown()
            mock_close.assert_called_once()

    def test_shutdown_with_nccl(self):
        engine = ZeroCopyTransferEngine()
        engine._nccl_transport = MagicMock()
        with patch.object(engine.cuda_ipc, "close_all"):
            engine.shutdown()
            engine._nccl_transport.shutdown.assert_called_once()

    def test_shutdown_nccl_shutdown_error(self):
        engine = ZeroCopyTransferEngine()
        engine._nccl_transport = MagicMock()
        engine._nccl_transport.shutdown.side_effect = RuntimeError("fail")
        engine.shutdown()  # Should not raise
