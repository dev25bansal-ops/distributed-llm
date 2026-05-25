"""Tests for NCCL transport send/recv and allreduce operations.

Uses the gloo backend for CPU-based testing when no GPU is available.
Multi-rank tests use spawned subprocesses with explicit env setup.

Requires: torch.distributed built with gloo backend support.
"""

import multiprocessing
import os
import socket
import sys

import pytest
import torch
import torch.distributed as dist

from distllm.core.nccl_transport import NcclTransport, CommType


BACKEND = "gloo"

_HAS_GLOO = False
try:
    _HAS_GLOO = dist.is_gloo_available()
except Exception:
    pass

HAS_MULTI_RANK = _HAS_GLOO

# Track ports in use to avoid conflicts between tests.
_ports_in_use: set[int] = set()
_ports_lock = multiprocessing.Lock()


def _get_free_port() -> int:
    with _ports_lock:
        for _ in range(100):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", 0))
                port = s.getsockname()[1]
            if port not in _ports_in_use:
                _ports_in_use.add(port)
                return port
    raise RuntimeError("could not find free port")


def _set_worker_env(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)


def _worker_send_recv(rank, world_size, backend, port):
    _set_worker_env(rank, world_size, port)
    dist.init_process_group(backend=backend)
    transport = NcclTransport(rank=rank, world_size=world_size, backend=backend, auto_init=False)
    transport._initialized = True

    data = torch.tensor([rank * 10 + i for i in range(4)], dtype=torch.float32)
    if rank == 0:
        transport.send(data, dst=1)
        result = data
    else:
        result = transport.recv((4,), torch.float32, src=0)

    expected = torch.tensor([0, 1, 2, 3], dtype=torch.float32)
    assert torch.allclose(result, expected), f"Rank {rank}: got {result}"
    transport.destroy()


def _worker_allreduce(rank, world_size, backend, port):
    _set_worker_env(rank, world_size, port)
    dist.init_process_group(backend=backend)
    transport = NcclTransport(rank=rank, world_size=world_size, backend=backend, auto_init=False)
    transport._initialized = True

    data = torch.full((4,), float(rank), dtype=torch.float32)
    result = transport.all_reduce(data)
    expected = torch.full((4,), float(sum(range(world_size))), dtype=torch.float32)
    assert torch.allclose(result, expected), f"Rank {rank}: got {result}"
    transport.destroy()


def _worker_broadcast(rank, world_size, backend, port):
    _set_worker_env(rank, world_size, port)
    dist.init_process_group(backend=backend)
    transport = NcclTransport(rank=rank, world_size=world_size, backend=backend, auto_init=False)
    transport._initialized = True

    data = torch.tensor([42.0, 43.0, 44.0], dtype=torch.float32) if rank == 0 else torch.empty(3, dtype=torch.float32)
    result = transport.broadcast(data, src=0)
    expected = torch.tensor([42.0, 43.0, 44.0], dtype=torch.float32)
    assert torch.allclose(result, expected)
    transport.destroy()


def _worker_barrier(rank, world_size, backend, port):
    _set_worker_env(rank, world_size, port)
    dist.init_process_group(backend=backend)
    transport = NcclTransport(rank=rank, world_size=world_size, backend=backend, auto_init=False)
    transport._initialized = True
    transport.barrier()
    transport.destroy()


def _worker_async_send_recv(rank, world_size, backend, port):
    _set_worker_env(rank, world_size, port)
    dist.init_process_group(backend=backend)
    transport = NcclTransport(rank=rank, world_size=world_size, backend=backend, auto_init=False)
    transport._initialized = True

    if rank == 0:
        data = torch.tensor([99.0, 88.0, 77.0], dtype=torch.float32)
        work = transport.send(data, dst=1, async_op=True)
        if work:
            work.wait()
    else:
        tensor, work = transport.async_recv((3,), torch.float32, src=0)
        if work:
            work.wait()
        expected = torch.tensor([99.0, 88.0, 77.0], dtype=torch.float32)
        assert torch.allclose(tensor, expected), f"Rank {rank}: got {tensor}"

    transport.destroy()


def _worker_stats(rank, world_size, backend, port):
    _set_worker_env(rank, world_size, port)
    dist.init_process_group(backend=backend)
    transport = NcclTransport(rank=rank, world_size=world_size, backend=backend, auto_init=False)
    transport._initialized = True

    if rank == 0:
        data = torch.tensor([1.0, 2.0], dtype=torch.float32)
        transport.send(data, dst=1)
    else:
        transport.recv((2,), torch.float32, src=0)

    data = torch.tensor([3.0, 4.0], dtype=torch.float32)
    transport.all_reduce(data)

    stats = transport.stats()
    assert CommType.ALL_REDUCE.value in stats
    assert stats[CommType.ALL_REDUCE.value]["count"] >= 1
    assert stats[CommType.ALL_REDUCE.value]["total_bytes"] > 0

    summary = transport.summary()
    assert "NcclTransport" in summary
    transport.destroy()


def _worker_all_gather(rank, world_size, backend, port):
    _set_worker_env(rank, world_size, port)
    dist.init_process_group(backend=backend)
    transport = NcclTransport(rank=rank, world_size=world_size, backend=backend, auto_init=False)
    transport._initialized = True

    data = torch.full((2,), float(rank), dtype=torch.float32)
    gather_list = [torch.empty(2, dtype=torch.float32) for _ in range(world_size)]
    transport.all_gather(data, gather_list)

    for i in range(world_size):
        expected = torch.full((2,), float(i), dtype=torch.float32)
        assert torch.allclose(gather_list[i], expected), f"Rank {rank}, gather[{i}]"

    transport.destroy()


def _worker_reduce_scatter(rank, world_size, backend, port):
    _set_worker_env(rank, world_size, port)
    dist.init_process_group(backend=backend)
    transport = NcclTransport(rank=rank, world_size=world_size, backend=backend, auto_init=False)
    transport._initialized = True

    input_list = [torch.full((2,), float(r + 1), dtype=torch.float32) for r in range(world_size)]
    output = torch.empty(2, dtype=torch.float32)
    transport.reduce_scatter(input_list, output)

    reduced = [torch.zeros(2, dtype=torch.float32) for _ in range(world_size)]
    for r in range(world_size):
        for w in range(world_size):
            reduced[r] += float(r + 1)
    expected = reduced[rank]
    assert torch.allclose(output, expected), f"Rank {rank}: {output} != {expected}"
    transport.destroy()


def _worker_p2p_group(rank, world_size, backend, port):
    _set_worker_env(rank, world_size, port)
    dist.init_process_group(backend=backend)
    transport = NcclTransport(rank=rank, world_size=world_size, backend=backend, auto_init=False)
    transport._initialized = True

    group = transport.create_p2p_group("test_group", [0, 1])
    assert group is not None

    if rank == 0:
        data = torch.tensor([7.0, 8.0, 9.0], dtype=torch.float32)
        transport.p2p_group_send("test_group", data, dst=1)
    else:
        result = transport.p2p_group_recv("test_group", (3,), torch.float32, src=0)
        expected = torch.tensor([7.0, 8.0, 9.0], dtype=torch.float32)
        assert torch.allclose(result, expected)

    transport.destroy()


def _worker_send_recv_multidim(rank, world_size, backend, port):
    _set_worker_env(rank, world_size, port)
    dist.init_process_group(backend=backend)
    t = NcclTransport(rank=rank, world_size=world_size, backend=backend, auto_init=False)
    t._initialized = True
    if rank == 0:
        data = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        t.send(data, dst=1)
    else:
        result = t.recv((3, 4), torch.float32, src=0)
        assert torch.allclose(result, torch.arange(12, dtype=torch.float32).reshape(3, 4))
    t.destroy()


def _worker_allreduce_large_tensor(rank, world_size, backend, port):
    _set_worker_env(rank, world_size, port)
    dist.init_process_group(backend=backend)
    t = NcclTransport(rank=rank, world_size=world_size, backend=backend, auto_init=False)
    t._initialized = True
    data = torch.full((100_000,), float(rank), dtype=torch.float32)
    result = t.all_reduce(data)
    expected = torch.full((100_000,), float(sum(range(world_size))), dtype=torch.float32)
    assert torch.allclose(result, expected)
    t.destroy()


def _worker_allreduce_preserves_dtype(rank, world_size, backend, port):
    _set_worker_env(rank, world_size, port)
    dist.init_process_group(backend=backend)
    t = NcclTransport(rank=rank, world_size=world_size, backend=backend, auto_init=False)
    t._initialized = True
    data = torch.full((4,), float(rank), dtype=torch.float64)
    result = t.all_reduce(data)
    assert result.dtype == torch.float64
    t.destroy()


def _run_worker(fn_name, world_size=2):
    port = _get_free_port()
    ctx = multiprocessing.get_context("spawn")
    procs = []
    for rank in range(world_size):
        p = ctx.Process(target=globals()[fn_name], args=(rank, world_size, BACKEND, port))
        procs.append(p)
        p.start()
    for p in procs:
        p.join(timeout=60)
    return [p.exitcode for p in procs]


# --- Single-rank / init tests ---


class TestNcclTransportInit:
    def test_init_single_rank(self):
        t = NcclTransport(rank=0, world_size=1, auto_init=True)
        assert t.is_initialized is False
        assert t._rank == 0
        assert t._world_size == 1
        t.destroy()

    def test_init_no_auto(self):
        t = NcclTransport(rank=0, world_size=1, auto_init=False)
        assert t._initialized is False
        t.destroy()

    def test_init_properties(self):
        t = NcclTransport(rank=2, world_size=4, master_addr="10.0.0.1",
                          master_port=12345, auto_init=False)
        assert t._rank == 2
        assert t._world_size == 4
        assert t._master_addr == "10.0.0.1"
        assert t._master_port == 12345
        t.destroy()

    def test_stats_empty_initially(self):
        t = NcclTransport(rank=0, world_size=1)
        assert t.stats() == {}
        t.destroy()

    def test_p2p_group_not_found(self):
        t = NcclTransport(rank=0, world_size=1)
        with pytest.raises(ValueError, match="P2P group nonexistent not found"):
            t.p2p_group_send("nonexistent", torch.tensor([1.0]), dst=1)
        t.destroy()

    def test_send_recv_single_rank_noop(self):
        """Single-rank transport should not crash on no-op operations."""
        t = NcclTransport(rank=0, world_size=1)
        # These are essentially no-ops with world_size=1
        t.barrier()
        t.destroy()


# --- Multi-rank tests (require gloo/NCCL backend) ---

multi_rank = pytest.mark.skipif(
    not HAS_MULTI_RANK,
    reason="torch.distributed backend not available (gloo/nccl)",
)


class TestNcclSendRecv:
    @multi_rank
    @pytest.mark.timeout(60)
    def test_send_recv_roundtrip(self):
        exits = _run_worker("_worker_send_recv")
        assert all(e == 0 for e in exits), f"Exit codes: {exits}"

    @multi_rank
    @pytest.mark.timeout(60)
    def test_async_send_recv(self):
        exits = _run_worker("_worker_async_send_recv")
        assert all(e == 0 for e in exits)

    @multi_rank
    @pytest.mark.timeout(60)
    def test_send_recv_multidim(self):
        exits = _run_worker("_worker_send_recv_multidim")
        assert all(e == 0 for e in exits)


class TestNcclAllReduce:
    @multi_rank
    @pytest.mark.timeout(60)
    def test_allreduce_sum(self):
        exits = _run_worker("_worker_allreduce")
        assert all(e == 0 for e in exits), f"Exit codes: {exits}"

    @multi_rank
    @pytest.mark.timeout(60)
    def test_allreduce_with_3_ranks(self):
        exits = _run_worker("_worker_allreduce", world_size=3)
        assert all(e == 0 for e in exits)

    @multi_rank
    @pytest.mark.timeout(60)
    def test_allreduce_large_tensor(self):
        exits = _run_worker("_worker_allreduce_large_tensor")
        assert all(e == 0 for e in exits)

    @multi_rank
    @pytest.mark.timeout(60)
    def test_allreduce_preserves_dtype(self):
        exits = _run_worker("_worker_allreduce_preserves_dtype")
        assert all(e == 0 for e in exits)


class TestNcclCollectives:
    @multi_rank
    @pytest.mark.timeout(60)
    def test_broadcast(self):
        exits = _run_worker("_worker_broadcast")
        assert all(e == 0 for e in exits)

    @multi_rank
    @pytest.mark.timeout(60)
    def test_barrier(self):
        exits = _run_worker("_worker_barrier")
        assert all(e == 0 for e in exits)

    @multi_rank
    @pytest.mark.timeout(60)
    def test_all_gather(self):
        exits = _run_worker("_worker_all_gather")
        assert all(e == 0 for e in exits)

    @multi_rank
    @pytest.mark.timeout(60)
    def test_reduce_scatter(self):
        exits = _run_worker("_worker_reduce_scatter")
        assert all(e == 0 for e in exits)


class TestNcclStats:
    @multi_rank
    @pytest.mark.timeout(60)
    def test_stats_recorded(self):
        exits = _run_worker("_worker_stats")
        assert all(e == 0 for e in exits)


class TestNcclP2PGroup:
    @multi_rank
    @pytest.mark.timeout(60)
    def test_p2p_group_send_recv(self):
        exits = _run_worker("_worker_p2p_group")
        assert all(e == 0 for e in exits)
