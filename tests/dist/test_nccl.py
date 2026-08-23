"""Tests for distllm.dist.nccl module.

Tests the public API of the NCCL transport layer using only real objects
(no mocks).  Uses a single-rank GLOO process group so all tests run on CPU
without requiring real GPU hardware or NCCL.

P2P operations (send, recv, send_recv, async_send, async_recv) require
``world_size >= 2`` and are unconditionally skipped since running them with
a single rank segfaults the process.

Note on source bugs
-------------------
The source file ``nccl.py`` has two duplicate ``preempt()`` methods and
references ``self._preempted_ops`` (in *both* definitions and in
``resume()``) that is never initialised in ``__init__``.  This causes
``AttributeError`` when ``resume()`` is called or when ``preempt()``
actually removes operations.  Preemption tests here avoid those code
paths; the remaining tests cover the parts that work.
"""

from __future__ import annotations

import os
import threading

import pytest
import torch
import torch.distributed as dist

from distllm.dist.nccl import NcclTransport, NcclTransferStats, CommType
from distllm.errors import NodeUnreachableError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _init_gloo(rank: int = 0, world_size: int = 1) -> None:
    """Initialise a GLOO process group, destroying any existing group first."""
    if dist.is_initialized():
        dist.destroy_process_group()
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)


def _destroy_gloo() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="function")
def gloo_world1():
    """Single-rank GLOO process group, cleaned up after each test."""
    _init_gloo(rank=0, world_size=1)
    yield
    _destroy_gloo()


@pytest.fixture
def transport(gloo_world1):
    """NcclTransport backed by GLOO, world_size=1, auto_init=False.

    The transport is manually initialised so the test is explicit about
    the process-group lifecycle.
    """
    t = NcclTransport(
        rank=0,
        world_size=1,
        backend="gloo",
        timeout_s=30,
        auto_init=False,
    )
    t.initialize()
    return t


# ---------------------------------------------------------------------------
# CommType enum
# ---------------------------------------------------------------------------

class TestCommType:
    """CommType enumeration values and membership."""

    def test_values(self) -> None:
        assert CommType.SEND.value == "send"
        assert CommType.RECV.value == "recv"
        assert CommType.ALL_REDUCE.value == "all_reduce"
        assert CommType.BROADCAST.value == "broadcast"
        assert CommType.ALL_GATHER.value == "all_gather"
        assert CommType.REDUCE_SCATTER.value == "reduce_scatter"
        assert CommType.P2P.value == "p2p"

    def test_membership(self) -> None:
        expected = {"send", "recv", "all_reduce", "broadcast",
                    "all_gather", "reduce_scatter", "p2p"}
        assert {e.value for e in CommType} == expected


# ---------------------------------------------------------------------------
# NcclTransferStats dataclass
# ---------------------------------------------------------------------------

class TestNcclTransferStats:
    """NcclTransferStats fields, computed properties, and edge cases."""

    def test_defaults(self) -> None:
        s = NcclTransferStats(comm_type=CommType.SEND)
        assert s.comm_type == CommType.SEND
        assert s.total_bytes == 0
        assert s.total_time_ns == 0
        assert s.count == 0
        assert s.errors == 0

    def test_avg_latency_us_zero_count(self) -> None:
        s = NcclTransferStats(comm_type=CommType.SEND)
        assert s.avg_latency_us == 0.0

    def test_avg_latency_us(self) -> None:
        s = NcclTransferStats(comm_type=CommType.SEND,
                              total_time_ns=2_000_000, count=2)
        assert s.avg_latency_us == 1000.0

    def test_bandwidth_gbps_zero_time(self) -> None:
        s = NcclTransferStats(comm_type=CommType.SEND,
                              total_bytes=8000, total_time_ns=0, count=1)
        # max(1e-12, seconds) guards against /0:
        # (8000 * 8) / 1e-12 / 1e9 = 6.4e7
        assert s.bandwidth_gbps == pytest.approx(6.4e7, rel=1e-3)

    def test_bandwidth_gbps(self) -> None:
        s = NcclTransferStats(comm_type=CommType.SEND,
                              total_bytes=1_000_000_000,
                              total_time_ns=1_000_000_000,
                              count=1)
        # 1 GB in 1 s = 8 Gbps
        assert s.bandwidth_gbps == pytest.approx(8.0, rel=1e-9)

    def test_mutable_fields(self) -> None:
        """Dataclass fields are intentionally not frozen."""
        s = NcclTransferStats(comm_type=CommType.SEND)
        s.total_bytes = 42
        assert s.total_bytes == 42


# ---------------------------------------------------------------------------
# NcclTransport: construction and initialisation
# ---------------------------------------------------------------------------

class TestNcclTransportInit:
    """Construction, auto_init, and initialize()."""

    def test_default_constructor(self) -> None:
        """auto_init=True, world_size=1: sets _initialized but no PG."""
        t = NcclTransport(rank=0, world_size=1, auto_init=True)
        assert t._initialized
        assert not dist.is_initialized()
        # is_initialized property also checks dist.is_initialized()
        assert not t.is_initialized
        t.destroy()

    def test_auto_init_world1_skips_pg(self) -> None:
        """world_size <= 1 skips dist.init_process_group."""
        t = NcclTransport(rank=0, world_size=1, auto_init=False)
        assert not t._initialized
        t.initialize()
        assert t._initialized
        assert not dist.is_initialized()
        t.destroy()

    def test_explicit_initialize_with_gloo(self, gloo_world1) -> None:
        t = NcclTransport(rank=0, world_size=1, backend="gloo",
                          auto_init=False)
        t.initialize()
        assert t.is_initialized
        assert t._effective_backend == "gloo"
        t.destroy()

    def test_double_initialize_is_idempotent(self, transport) -> None:
        transport.initialize()
        assert transport.is_initialized

    def test_constructor_stores_params(self) -> None:
        t = NcclTransport(rank=2, world_size=4, backend="nccl",
                          master_addr="10.0.0.1", master_port=12345,
                          timeout_s=60, auto_init=False,
                          allow_gloo_fallback=False)
        assert t._rank == 2
        assert t._world_size == 4
        assert t._backend == "nccl"
        assert t._master_addr == "10.0.0.1"
        assert t._master_port == 12345
        assert t._timeout_s == 60
        assert t._allow_gloo_fallback is False
        t.destroy()


# ---------------------------------------------------------------------------
# NcclTransport: properties
# ---------------------------------------------------------------------------

class TestNcclTransportProperties:
    """is_initialized, _default_device, and related properties."""

    def test_is_initialized_false_initially(self) -> None:
        t = NcclTransport(rank=0, world_size=1, backend="gloo",
                          auto_init=False)
        assert not t.is_initialized
        t.destroy()

    def test_is_initialized_true_after_init(self, transport) -> None:
        assert transport.is_initialized

    def test_is_initialized_false_after_destroy(self, transport) -> None:
        transport.destroy()
        assert not transport.is_initialized

    def test_default_device_cpu_gloo(self, transport) -> None:
        assert transport._default_device == "cpu"

    def test_default_device_cuda_if_nccl(self, gloo_world1) -> None:
        t = NcclTransport(rank=0, world_size=1, backend="nccl",
                          auto_init=False)
        t.initialize()
        if torch.cuda.is_available():
            assert t._default_device == "cuda:0"
        else:
            assert t._default_device == "cpu"
        t.destroy()

    def test_default_device_cuda_rank_oob(self, gloo_world1) -> None:
        """Rank >= device_count falls back to ``cpu``."""
        t = NcclTransport(rank=999, world_size=1, backend="nccl",
                          auto_init=False)
        t.initialize()
        assert t._default_device == "cpu"
        t.destroy()

    def test_effective_backend(self, transport) -> None:
        assert transport._effective_backend == "gloo"


# ---------------------------------------------------------------------------
# Collectives (world_size=1, GLOO)  --  these work with a single rank
# ---------------------------------------------------------------------------

class TestCollectives:
    """In-place collectives on a single rank behave as identity."""

    def test_all_reduce(self, transport) -> None:
        t = torch.tensor([1.0, 2.0, 3.0])
        result = transport.all_reduce(t)
        assert result is t
        assert torch.equal(result, torch.tensor([1.0, 2.0, 3.0]))

    def test_broadcast(self, transport) -> None:
        t = torch.tensor([42.0])
        result = transport.broadcast(t, src=0)
        assert result is t
        assert result.item() == 42.0

    def test_all_gather(self, transport) -> None:
        t = torch.tensor([1.0])
        gl = [torch.empty_like(t)]
        result = transport.all_gather(t, gl)
        assert result is gl
        assert torch.equal(gl[0], torch.tensor([1.0]))

    def test_reduce_scatter(self, transport) -> None:
        out = torch.empty(2)
        result = transport.reduce_scatter(
            list(torch.chunk(torch.tensor([1.0, 2.0]), 1)), out,
        )
        assert result is out

    def test_barrier(self, transport) -> None:
        transport.barrier()

    def test_barrier_not_initialized(self) -> None:
        t = NcclTransport(rank=0, world_size=1, backend="gloo",
                          auto_init=False)
        t.barrier()  # no-op, does not raise
        t.destroy()

    def test_monitored_barrier_succeeds(self, transport) -> None:
        assert transport.monitored_barrier(timeout_s=5.0) is True

    def test_monitored_barrier_false_when_not_initialized(self) -> None:
        t = NcclTransport(rank=0, world_size=1, backend="gloo",
                          auto_init=False)
        assert t.monitored_barrier(timeout_s=1.0) is False
        t.destroy()


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

class TestStats:
    """Stats recording, retrieval, and summary string."""

    def test_stats_empty_initially(self, transport) -> None:
        assert transport.stats() == {}

    def test_stats_after_all_reduce(self, transport) -> None:
        transport.all_reduce(torch.tensor([1.0]))
        s = transport.stats()
        assert "all_reduce" in s
        assert s["all_reduce"]["count"] == 1
        assert s["all_reduce"]["errors"] == 0
        assert s["all_reduce"]["total_bytes"] > 0

    def test_stats_multiple_ops(self, transport) -> None:
        transport.all_reduce(torch.tensor([1.0, 2.0]))
        transport.all_reduce(torch.tensor([3.0]))
        s = transport.stats()
        assert s["all_reduce"]["count"] == 2

    def test_stats_separate_comm_types(self, transport) -> None:
        transport.all_reduce(torch.tensor([1.0]))
        transport.broadcast(torch.tensor([1.0]))
        s = transport.stats()
        assert "all_reduce" in s
        assert "broadcast" in s
        assert s["all_reduce"]["count"] == 1
        assert s["broadcast"]["count"] == 1

    def test_summary_empty(self, transport) -> None:
        s = transport.summary()
        assert "NcclTransport" in s
        assert "rank=0" in s
        assert "world=1" in s
        assert "(no operations recorded)" in s

    def test_summary_after_op(self, transport) -> None:
        transport.broadcast(torch.tensor([1.0]))
        s = transport.summary()
        assert "broadcast" in s
        assert "(no operations recorded)" not in s

    def test_stats_bandwidth_in_summary(self, transport) -> None:
        transport.all_reduce(torch.tensor([42.0]))
        s = transport.summary()
        assert "Gbps" in s
        assert "us" in s


# ---------------------------------------------------------------------------
# Send / recv  (multi-rank only -- unconditionally skipped)
# ---------------------------------------------------------------------------

_skip_reason = (
    "P2P operations (isend/irecv/send/recv) require world_size >= 2; "
    "running with a single rank crashes the process"
)


class TestSendRecv:
    """P2P operations -- require multi-rank environment.

    These are structurally tested (exception paths for missing state) but
    the actual data-transfer paths are unconditionally skipped because they
    segfault with a single rank.
    """

    @pytest.mark.skipif(True, reason=_skip_reason)
    def test_send_sync(self, transport) -> None:  # pragma: no cover
        transport.send(torch.tensor([1.0]), dst=0, async_op=False)

    @pytest.mark.skipif(True, reason=_skip_reason)
    def test_send_async(self, transport) -> None:  # pragma: no cover
        work = transport.send(torch.tensor([1.0]), dst=0, async_op=True)
        if work is not None:
            work.wait()

    @pytest.mark.skipif(True, reason=_skip_reason)
    def test_recv(self, transport) -> None:  # pragma: no cover
        transport.recv((3,), torch.float32, src=0)

    @pytest.mark.skipif(True, reason=_skip_reason)
    def test_recv_with_device(self, transport) -> None:  # pragma: no cover
        transport.recv((2,), torch.float32, src=0, device="cpu")

    @pytest.mark.skipif(True, reason=_skip_reason)
    def test_async_recv(self, transport) -> None:  # pragma: no cover
        tensor, op = transport.async_recv((4,), torch.float32, src=0)
        assert tensor.shape == (4,)
        if op is not None:
            op.wait()

    @pytest.mark.skipif(True, reason=_skip_reason)
    def test_async_send(self, transport) -> None:  # pragma: no cover
        op = transport.async_send(torch.tensor([1.0, 2.0]), dst=0)
        if op is not None:
            op.wait()

    @pytest.mark.skipif(True, reason=_skip_reason)
    def test_send_recv(self, transport) -> None:  # pragma: no cover
        transport.send_recv(
            torch.tensor([1.0, 2.0]),
            torch.empty(2),
            dst=0, src=0,
        )

    @pytest.mark.skipif(True, reason=_skip_reason)
    def test_send_tensor_list(self, transport) -> None:  # pragma: no cover
        transport.send_tensor_list(
            [torch.tensor([1.0]), torch.tensor([2.0, 3.0])],
            peer="0",
        )

    def test_send_tensor_list_not_initialized(self) -> None:
        t = NcclTransport(rank=0, world_size=1, backend="gloo",
                          auto_init=False)
        with pytest.raises(RuntimeError, match="NCCL not initialized"):
            t.send_tensor_list([torch.tensor([1.0])], peer="0")
        t.destroy()


# ---------------------------------------------------------------------------
# P2P groups
# ---------------------------------------------------------------------------

class TestP2PGroups:
    """P2P process-group management."""

    def test_create_p2p_group(self, transport) -> None:
        group = transport.create_p2p_group("g1", [0])
        assert isinstance(group, dist.ProcessGroup)

    def test_create_multiple_groups(self, transport) -> None:
        g1 = transport.create_p2p_group("g1", [0])
        g2 = transport.create_p2p_group("g2", [0])
        assert g1 is not g2

    def test_p2p_group_send_missing_raises(self, transport) -> None:
        with pytest.raises(ValueError, match="P2P group unknown not found"):
            transport.p2p_group_send("unknown", torch.tensor([1.0]), dst=0)

    def test_p2p_group_recv_missing_raises(self, transport) -> None:
        with pytest.raises(ValueError, match="P2P group unknown not found"):
            transport.p2p_group_recv("unknown", (1,), torch.float32, src=0)

    def test_destroy_clears_groups(self, transport) -> None:
        transport.create_p2p_group("g1", [0])
        transport.destroy()
        assert transport._p2p_groups == {}


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    """Exception types and _ensure_initialised behaviour."""

    def test_ensure_initialized_raises(self) -> None:
        t = NcclTransport(rank=0, world_size=1, backend="gloo",
                          auto_init=False)
        with pytest.raises(RuntimeError, match="NCCL not initialized"):
            t._ensure_initialized()
        t.destroy()

    def test_ensure_initialized_ok(self, transport) -> None:
        transport._ensure_initialized()  # does not raise

    def test_all_reduce_after_destroy_raises(self, transport) -> None:
        transport.destroy()
        with pytest.raises(RuntimeError, match="NCCL not initialized"):
            transport.all_reduce(torch.tensor([1.0]))

    def test_broadcast_after_destroy_raises(self, transport) -> None:
        transport.destroy()
        with pytest.raises(RuntimeError, match="NCCL not initialized"):
            transport.broadcast(torch.tensor([1.0]))

    def test_all_gather_after_destroy_raises(self, transport) -> None:
        transport.destroy()
        with pytest.raises(RuntimeError, match="NCCL not initialized"):
            transport.all_gather(torch.tensor([1.0]), [torch.empty(1)])

    def test_reduce_scatter_after_destroy_raises(self, transport) -> None:
        transport.destroy()
        with pytest.raises(RuntimeError, match="NCCL not initialized"):
            transport.reduce_scatter([torch.tensor([1.0])], torch.empty(1))

    def test_send_to_bad_rank_raises(self, transport) -> None:
        """dist.send to a rank outside the process group wraps in
        NodeUnreachableError."""
        with pytest.raises(NodeUnreachableError):
            transport.send(torch.tensor([1.0]), dst=999, async_op=False)

    def test_recv_from_bad_rank_raises(self, transport) -> None:
        with pytest.raises(RuntimeError, match="NCCL recv from 999 failed"):
            transport.recv((1,), torch.float32, src=999)


# ---------------------------------------------------------------------------
# CUDA stream priority helpers
# ---------------------------------------------------------------------------

class TestStreamPriority:
    """get_stream_for_priority returns None on CPU (no CUDA streams)."""

    def test_low_priority(self, transport) -> None:
        assert transport.get_stream_for_priority(priority=0) is None

    def test_negative_priority(self, transport) -> None:
        assert transport.get_stream_for_priority(priority=-1) is None

    def test_high_priority(self, transport) -> None:
        assert transport.get_stream_for_priority(priority=1) is None

    def test_high_priority_large(self, transport) -> None:
        assert transport.get_stream_for_priority(priority=100) is None

    def test_zero_boundary(self, transport) -> None:
        assert transport.get_stream_for_priority(priority=0) is None
        assert transport.get_stream_for_priority(priority=-999) is None


# ---------------------------------------------------------------------------
# Preemption support  (partial -- avoids source bug with _preempted_ops)
# ---------------------------------------------------------------------------

class TestPreemption:
    """Priority-based preemption (local state only, no GPU).

    Note: ``resume()`` and ``preempt()`` with actual op removal hit a bug
    in the source (``self._preempted_ops`` never initialised), so those
    paths are not tested here.
    """

    def test_preempt_returns_zero_when_no_ops(self, transport) -> None:
        assert transport.preempt(priority_threshold=0) == 0

    def test_preempt_sets_flag(self, transport) -> None:
        assert not transport.is_preempted
        transport.preempt(0)
        assert transport.is_preempted

    def test_wait_for_resume_timeout(self, transport) -> None:
        """After preempt the event is cleared; wait without resume times out."""
        transport.preempt(0)
        assert transport.wait_for_resume(timeout=0.05) is False

    def test_register_and_unregister_op(self, transport) -> None:
        transport.register_op("op1", None, priority=5)
        assert transport.active_op_count == 1
        transport.unregister_op("op1")
        assert transport.active_op_count == 0

    def test_unregister_missing_op(self, transport) -> None:
        """Unregistering a non-existent op does not raise."""
        transport.unregister_op("nonexistent")
        assert transport.active_op_count == 0

    def test_multiple_registered_ops(self, transport) -> None:
        transport.register_op("a", None, priority=1)
        transport.register_op("b", None, priority=2)
        assert transport.active_op_count == 2
        transport.unregister_op("a")
        assert transport.active_op_count == 1
        transport.unregister_op("b")
        assert transport.active_op_count == 0

    def test_preempt_skips_high_priority(self, transport) -> None:
        """Operations with priority >= threshold are not preempted."""
        transport.register_op("keep", None, priority=10)
        count = transport.preempt(priority_threshold=5)
        assert count == 0
        assert transport.active_op_count == 1
        transport.unregister_op("keep")

    def test_is_preempted_false_initially(self, transport) -> None:
        assert not transport.is_preempted

    def test_preempt_idempotent(self, transport) -> None:
        """Calling preempt twice with no ops is harmless."""
        transport.preempt(0)
        transport.preempt(0)
        assert transport.is_preempted


# ---------------------------------------------------------------------------
# Shutdown / destroy
# ---------------------------------------------------------------------------

class TestShutdown:
    """Lifecycle cleanup."""

    def test_shutdown(self, transport) -> None:
        transport.shutdown()
        assert not transport.is_initialized

    def test_destroy_idempotent(self, transport) -> None:
        transport.destroy()
        transport.destroy()

    def test_operations_after_shutdown(self, transport) -> None:
        transport.destroy()
        with pytest.raises(RuntimeError, match="NCCL not initialized"):
            transport.all_reduce(torch.tensor([1.0]))


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Boundary conditions for collective operations."""

    def test_empty_tensor_all_reduce(self, transport) -> None:
        transport.all_reduce(torch.empty(0))

    def test_empty_tensor_broadcast(self, transport) -> None:
        transport.broadcast(torch.empty(0))

    def test_scalar_tensor_all_reduce(self, transport) -> None:
        result = transport.all_reduce(torch.tensor(42.0))
        assert result.item() == 42.0

    def test_large_tensor_broadcast(self, transport) -> None:
        transport.broadcast(torch.randn(100_000))

    def test_all_gather_empty_tensors(self, transport) -> None:
        transport.all_gather(torch.empty(0), [torch.empty(0)])

    def test_reduce_scatter_single_element(self, transport) -> None:
        transport.reduce_scatter(
            list(torch.chunk(torch.tensor([42.0]), 1)),
            torch.empty(1),
        )

    def test_non_default_master_port(self, gloo_world1) -> None:
        t = NcclTransport(rank=0, world_size=1, backend="gloo",
                          master_port=29501, auto_init=False)
        t.initialize()
        assert t.is_initialized
        t.destroy()

    def test_all_stats_comm_types(self, transport) -> None:
        transport.all_reduce(torch.tensor([1.0]))
        transport.broadcast(torch.tensor([2.0]))
        transport.broadcast(torch.tensor([3.0]))
        s = transport.stats()
        assert len(s) == 2
        assert s["all_reduce"]["count"] == 1
        assert s["broadcast"]["count"] == 2


# ---------------------------------------------------------------------------
# Thread safety (basic)
# ---------------------------------------------------------------------------

class TestThreadSafety:
    """Concurrent access to collective operations and barrier."""

    def test_concurrent_allreduce(self, transport) -> None:
        errors = []

        def work():
            try:
                for _ in range(20):
                    transport.all_reduce(torch.tensor([1.0]))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=work) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        s = transport.stats()
        assert s["all_reduce"]["count"] == 80

    def test_concurrent_barrier(self, transport) -> None:
        errors = []

        def work():
            try:
                for _ in range(10):
                    transport.barrier()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=work) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
