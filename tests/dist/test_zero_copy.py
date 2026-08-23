"""Tests for distllm.dist.zero_copy module.

Tests the public API using only real objects (no mocks).
All tests run on CPU without requiring GPU or RDMA hardware.

Note on source bugs
-------------------
The source file ``zero_copy.py`` line 53 imports
``torch.multiprocessing.reduction`` which is not a valid module path
(correct: ``torch.multiprocessing.reductions``).  CUDA code paths that hit
this import (``CudaIPCManager.export_tensor``) are not exercised here.
"""

from __future__ import annotations

import os

import pytest
import torch

from distllm.dist.zero_copy import (
    CudaIPCManager,
    RDMAManager,
    TransferBackend,
    TransferStats,
    ZeroCopyTransferEngine,
)


# ---------------------------------------------------------------------------
# TransferBackend enum
# ---------------------------------------------------------------------------

class TestTransferBackend:
    """TransferBackend enumeration values."""

    def test_values(self) -> None:
        assert TransferBackend.CUDA_IPC.value == "cuda_ipc"
        assert TransferBackend.RDMA.value == "rdma"
        assert TransferBackend.NCCL.value == "nccl"
        assert TransferBackend.GLOO.value == "gloo"
        assert TransferBackend.GRPC.value == "grpc"

    def test_membership(self) -> None:
        expected = {"cuda_ipc", "rdma", "nccl", "gloo", "grpc"}
        assert {e.value for e in TransferBackend} == expected


# ---------------------------------------------------------------------------
# TransferStats dataclass
# ---------------------------------------------------------------------------

class TestTransferStats:
    """TransferStats fields, defaults, and computed values."""

    def test_defaults(self) -> None:
        s = TransferStats(backend=TransferBackend.NCCL)
        assert s.backend == TransferBackend.NCCL
        assert s.bytes_transferred == 0
        assert s.latency_ms == 0.0
        assert s.bandwidth_gbps == 0.0
        assert s.success is True

    def test_custom_values(self) -> None:
        s = TransferStats(
            backend=TransferBackend.CUDA_IPC,
            bytes_transferred=1024,
            latency_ms=0.5,
            bandwidth_gbps=16.0,
            success=False,
        )
        assert s.bytes_transferred == 1024
        assert s.latency_ms == 0.5
        assert s.bandwidth_gbps == 16.0
        assert s.success is False

    def test_mutable_fields(self) -> None:
        """Dataclass is not frozen -- mutation is allowed."""
        s = TransferStats(backend=TransferBackend.RDMA)
        s.bytes_transferred = 4096
        assert s.bytes_transferred == 4096


# ---------------------------------------------------------------------------
# CudaIPCManager
# ---------------------------------------------------------------------------

class TestCudaIPCManager:
    """CUDA IPC manager on CPU (no GPU available)."""

    def test_constructor(self) -> None:
        mgr = CudaIPCManager()
        assert mgr._handles == {}
        if torch.cuda.is_available():
            assert "cuda" in str(mgr._device)
        else:
            assert str(mgr._device) == "cpu"

    def test_export_tensor_cpu_returns_none(self) -> None:
        mgr = CudaIPCManager()
        result = mgr.export_tensor("k", torch.zeros(4, 4))
        assert result is None

    def test_export_tensor_string_key_passed(self) -> None:
        """None key is passed through; export returns None on CPU."""
        mgr = CudaIPCManager()
        result = mgr.export_tensor(None, torch.zeros(4, 4))  # type: ignore[arg-type]
        assert result is None

    def test_export_tensor_scalar(self) -> None:
        mgr = CudaIPCManager()
        result = mgr.export_tensor("s", torch.tensor(42.0))
        assert result is None

    def test_import_tensor_no_cuda_returns_none(self) -> None:
        mgr = CudaIPCManager()
        result = mgr.import_tensor("k", b"handle", (4,), torch.float32)
        assert result is None

    def test_import_tensor_empty_shape(self) -> None:
        mgr = CudaIPCManager()
        result = mgr.import_tensor("k", b"handle", (), torch.float32)
        assert result is None

    def test_close_nonexistent_key(self) -> None:
        mgr = CudaIPCManager()
        mgr.close("missing")  # must not raise

    def test_close_all_empty(self) -> None:
        mgr = CudaIPCManager()
        mgr.close_all()  # must not raise

    def test_import_bad_handle_returns_none(self) -> None:
        """Import with garbage bytes; returns None gracefully."""
        mgr = CudaIPCManager()
        result = mgr.import_tensor("bad", b"\x00\x01\x02", (4,), torch.float32)
        assert result is None

    def test_close_removes_key(self) -> None:
        """Manually inject a key to close -- avoids broken CUDA export path."""
        mgr = CudaIPCManager()
        mgr._handles["del"] = (torch.zeros(2, 2), b"dummy")
        mgr.close("del")
        assert "del" not in mgr._handles

    def test_close_all_clears_everything(self) -> None:
        mgr = CudaIPCManager()
        mgr._handles["a"] = (torch.zeros(2), b"x")
        mgr._handles["b"] = (torch.zeros(2), b"y")
        mgr.close_all()
        assert mgr._handles == {}


# ---------------------------------------------------------------------------
# RDMAManager
# ---------------------------------------------------------------------------

class TestRDMAManager:
    """RDMA manager without RDMA hardware."""

    def test_constructor(self) -> None:
        mgr = RDMAManager()
        assert mgr.available is False
        assert mgr._registered_memory == {}

    def test_available_property(self) -> None:
        mgr = RDMAManager()
        assert mgr.available is mgr._available

    def test_register_memory_cpu_returns_false(self) -> None:
        mgr = RDMAManager()
        assert mgr.register_memory("k", torch.zeros(4, 4)) is False

    def test_register_memory_cuda_accepted(self) -> None:
        if not torch.cuda.is_available():
            pytest.skip("requires CUDA")
        mgr = RDMAManager()
        t = torch.zeros(4, 4, device="cuda")
        assert mgr.register_memory("k", t) is True
        assert mgr._registered_memory["k"] is t

    def test_deregister_memory_removes_key(self) -> None:
        mgr = RDMAManager()
        mgr._registered_memory["k"] = torch.zeros(4, 4)
        mgr.deregister_memory("k")
        assert "k" not in mgr._registered_memory

    def test_deregister_memory_nonexistent(self) -> None:
        mgr = RDMAManager()
        mgr.deregister_memory("ghost")  # must not raise

    def test_deregister_memory_idempotent(self) -> None:
        mgr = RDMAManager()
        mgr.deregister_memory("x")
        mgr.deregister_memory("x")  # second call also no-op

    def test_send_rdma_not_available_returns_false(self) -> None:
        mgr = RDMAManager()
        assert mgr.send_rdma("peer", torch.zeros(4, 4)) is False

    def test_recv_rdma_not_available_returns_none(self) -> None:
        mgr = RDMAManager()
        assert mgr.recv_rdma("peer", (4, 4), torch.float32) is None

    # -- RDMA-enabled via env var (stub path) --

    def test_recv_rdma_with_env_stub_returns_none(self) -> None:
        """When enabled via env var, recv_rdma catches NotImplementedError
        and returns None."""
        os.environ["DISTLLM_INFINIBAND"] = "1"
        try:
            mgr = RDMAManager()
            assert mgr.available is True
            result = mgr.recv_rdma("p", (4,), torch.float32)
            assert result is None
        finally:
            os.environ.pop("DISTLLM_INFINIBAND", None)

    def test_send_rdma_with_env_stub_returns_true(self) -> None:
        os.environ["DISTLLM_INFINIBAND"] = "1"
        try:
            mgr = RDMAManager()
            assert mgr.send_rdma("p", torch.zeros(4)) is True
        finally:
            os.environ.pop("DISTLLM_INFINIBAND", None)


# ---------------------------------------------------------------------------
# ZeroCopyTransferEngine -- construction
# ---------------------------------------------------------------------------

class TestZeroCopyTransferEngineInit:
    """Engine construction and sub-manager wiring."""

    def test_constructor(self) -> None:
        engine = ZeroCopyTransferEngine()
        assert isinstance(engine.cuda_ipc, CudaIPCManager)
        assert isinstance(engine.rdma, RDMAManager)
        assert engine._nccl_transport is not None
        assert engine._stats == []

    def test_stats_initially_empty(self) -> None:
        engine = ZeroCopyTransferEngine()
        assert engine.get_stats() == []
        assert engine.get_aggregate_stats() == {}


# ---------------------------------------------------------------------------
# ZeroCopyTransferEngine -- send
# ---------------------------------------------------------------------------

class TestZeroCopyTransferEngineSend:
    """send() with CPU tensors (backends: GRPC fallback)."""

    CPU_TENSOR = torch.zeros(4, 4)

    def test_send_cpu(self) -> None:
        engine = ZeroCopyTransferEngine()
        stats = engine.send("peer:0", self.CPU_TENSOR)
        assert stats.backend == TransferBackend.GRPC
        assert stats.success is False
        assert stats.bytes_transferred > 0

    def test_send_peer_local(self) -> None:
        engine = ZeroCopyTransferEngine()
        stats = engine.send("peer:0", self.CPU_TENSOR, peer_is_local=True)
        assert stats.backend == TransferBackend.GRPC

    def test_send_with_tag(self) -> None:
        engine = ZeroCopyTransferEngine()
        tensor = torch.zeros(2, 3)
        stats = engine.send("peer:0", tensor, tag="my_tag")
        assert isinstance(stats, TransferStats)

    def test_send_scalar(self) -> None:
        engine = ZeroCopyTransferEngine()
        stats = engine.send("peer:0", torch.tensor(42.0))
        assert stats.bytes_transferred > 0

    def test_send_empty_tensor(self) -> None:
        engine = ZeroCopyTransferEngine()
        stats = engine.send("peer:0", torch.empty(0))
        assert stats.bytes_transferred == 0

    def test_send_empty_peer_string(self) -> None:
        engine = ZeroCopyTransferEngine()
        stats = engine.send("", self.CPU_TENSOR)
        assert stats.backend == TransferBackend.GRPC


# ---------------------------------------------------------------------------
# ZeroCopyTransferEngine -- recv
# ---------------------------------------------------------------------------

class TestZeroCopyTransferEngineRecv:
    """recv() with CPU fallback (GRPC -- returns None)."""

    def test_recv_cpu(self) -> None:
        engine = ZeroCopyTransferEngine()
        result, stats = engine.recv("peer:0", (4, 4), torch.float32)
        assert result is None
        assert stats.backend == TransferBackend.GRPC
        assert stats.success is False

    def test_recv_with_tag(self) -> None:
        engine = ZeroCopyTransferEngine()
        result, stats = engine.recv("p", (2,), torch.float32, tag="t")
        assert result is None

    def test_recv_empty_shape(self) -> None:
        engine = ZeroCopyTransferEngine()
        result, stats = engine.recv("p", (), torch.float32)
        assert result is None

    def test_recv_scalar_shape(self) -> None:
        engine = ZeroCopyTransferEngine()
        result, stats = engine.recv("p", (1,), torch.float32)
        assert result is None


# ---------------------------------------------------------------------------
# ZeroCopyTransferEngine -- stats
# ---------------------------------------------------------------------------

class TestZeroCopyTransferEngineStats:
    """Stats accumulation and aggregation."""

    def test_send_appends_one_stat(self) -> None:
        engine = ZeroCopyTransferEngine()
        engine.send("p", torch.zeros(4))
        assert len(engine.get_stats()) == 1

    def test_recv_appends_one_stat(self) -> None:
        engine = ZeroCopyTransferEngine()
        engine.recv("p", (4,), torch.float32)
        assert len(engine.get_stats()) == 1

    def test_mixed_ops(self) -> None:
        engine = ZeroCopyTransferEngine()
        engine.send("p", torch.zeros(4))
        engine.recv("p", (4,), torch.float32)
        assert len(engine.get_stats()) == 2

    def test_get_stats_returns_copy(self) -> None:
        engine = ZeroCopyTransferEngine()
        engine.send("p", torch.zeros(4))
        snapshot = engine.get_stats()
        engine.send("q", torch.zeros(8))
        assert len(snapshot) == 1  # original snapshot unchanged

    def test_aggregate_empty(self) -> None:
        engine = ZeroCopyTransferEngine()
        assert engine.get_aggregate_stats() == {}

    def test_aggregate_single_backend(self) -> None:
        engine = ZeroCopyTransferEngine()
        engine.send("p", torch.zeros(4))
        engine.send("q", torch.zeros(4))
        agg = engine.get_aggregate_stats()
        assert "grpc" in agg
        assert agg["grpc"]["count"] == 2
        assert "avg_latency_ms" in agg["grpc"]
        assert "p50_latency_ms" in agg["grpc"]

    def test_aggregate_p50_odd_count(self) -> None:
        engine = ZeroCopyTransferEngine()
        for _ in range(3):
            engine.send("p", torch.zeros(4))
        agg = engine.get_aggregate_stats()
        assert agg["grpc"]["count"] == 3


# ---------------------------------------------------------------------------
# ZeroCopyTransferEngine -- shutdown
# ---------------------------------------------------------------------------

class TestZeroCopyTransferEngineShutdown:
    """Lifecycle cleanup."""

    def test_shutdown_clears_cuda_handles(self) -> None:
        engine = ZeroCopyTransferEngine()
        engine.shutdown()
        assert engine.cuda_ipc._handles == {}

    def test_shutdown_idempotent(self) -> None:
        engine = ZeroCopyTransferEngine()
        engine.shutdown()
        engine.shutdown()  # must not raise

    def test_send_after_shutdown(self) -> None:
        engine = ZeroCopyTransferEngine()
        engine.shutdown()
        stats = engine.send("p", torch.zeros(4))
        assert isinstance(stats, TransferStats)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Boundary conditions across the module."""

    def test_transfer_stats_zero_bandwidth_all_zero(self) -> None:
        s = TransferStats(
            backend=TransferBackend.NCCL,
            bytes_transferred=0,
            latency_ms=0.0,
            bandwidth_gbps=0.0,
            success=True,
        )
        assert s.bandwidth_gbps == 0.0

    def test_transfer_stats_large_values(self) -> None:
        s = TransferStats(
            backend=TransferBackend.RDMA,
            bytes_transferred=10_000_000_000,
            latency_ms=1000.0,
            bandwidth_gbps=80.0,
            success=True,
        )
        assert s.bytes_transferred == 10_000_000_000

    def test_rdma_deregister_nonexistent(self) -> None:
        RDMAManager().deregister_memory("never_registered")

    def test_cuda_ipc_close_nonexistent(self) -> None:
        CudaIPCManager().close("never_exported")

    def test_engine_no_ops_aggregate(self) -> None:
        engine = ZeroCopyTransferEngine()
        engine.shutdown()
        assert engine.get_aggregate_stats() == {}
