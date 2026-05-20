"""NCCL-based tensor transfers between GPUs (intra-node and inter-node).

Leverages NVIDIA NCCL for high-bandwidth GPU-to-GPU communication:
- Intra-node: uses NVLink/NVSwitch for up to 900 GB/s
- Inter-node: uses RDMA/InfiniBand via NCCL for up to 400 Gb/s

Provides:
- Synchronous and asynchronous send/recv
- Broadcast, all-reduce, all-gather collectives
- Custom P2P communication groups
- Integration with existing torch.distributed process groups
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
import torch.distributed as dist
from loguru import logger


class CommType(Enum):
    SEND = "send"
    RECV = "recv"
    ALL_REDUCE = "all_reduce"
    BROADCAST = "broadcast"
    ALL_GATHER = "all_gather"
    REDUCE_SCATTER = "reduce_scatter"
    P2P = "p2p"


@dataclass
class NcclTransferStats:
    comm_type: CommType
    total_bytes: int = 0
    total_time_ns: int = 0
    count: int = 0
    errors: int = 0

    @property
    def avg_latency_us(self) -> float:
        return (self.total_time_ns / max(self.count, 1)) / 1000.0

    @property
    def bandwidth_gbps(self) -> float:
        seconds = self.total_time_ns / 1e9
        return (self.total_bytes * 8) / max(seconds, 1e-12) / 1e9


class NcclTransport:
    """High-level NCCL transport for tensor transfers.

    Initializes and manages a NCCL process group, provides typed
    send/recv and collective operations with timing and error handling.

    Usage:
        transport = NcclTransport(rank=0, world_size=2)
        transport.send(tensor, dst=1)
        received = transport.recv(shape, dtype, src=0)
    """

    def __init__(
        self,
        rank: int = 0,
        world_size: int = 1,
        master_addr: str = "127.0.0.1",
        master_port: int = 29500,
        backend: str = "nccl",
        timeout_s: int = 30,
        auto_init: bool = True,
    ):
        self._rank = rank
        self._world_size = world_size
        self._master_addr = master_addr
        self._master_port = master_port
        self._backend = backend
        self._timeout_s = timeout_s
        self._initialized = False
        self._lock = threading.Lock()
        self._stats: dict[CommType, NcclTransferStats] = {}
        self._p2p_groups: dict[str, dist.ProcessGroup] = {}

        if auto_init:
            self.initialize()

    def initialize(self) -> None:
        if self._initialized:
            return
        if self._world_size <= 1:
            self._initialized = True
            return

        os.environ.setdefault("MASTER_ADDR", self._master_addr)
        os.environ.setdefault("MASTER_PORT", str(self._master_port))

        if not dist.is_initialized():
            dist.init_process_group(
                backend=self._backend,
                rank=self._rank,
                world_size=self._world_size,
                timeout=torch.distributed.default_pg_timeout if hasattr(torch.distributed, 'default_pg_timeout') else None,
            )
        self._initialized = True
        logger.info(f"NCCL initialized: rank={self._rank}, world={self._world_size}, backend={self._backend}")

    def destroy(self) -> None:
        with self._lock:
            for name, group in self._p2p_groups.items():
                dist.destroy_process_group(group)
            self._p2p_groups.clear()
        if dist.is_initialized():
            dist.destroy_process_group()
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        return self._initialized and dist.is_initialized()

    # -------------------------------------------------------------------
    # P2P Send/Recv
    # -------------------------------------------------------------------

    def send(self, tensor: torch.Tensor, dst: int, tag: int = 0, async_op: bool = False) -> dist.Work | None:
        """Send a tensor to a destination rank."""
        self._ensure_initialized()
        start = time.time_ns()
        try:
            work = dist.isend(tensor, dst=dst, tag=tag) if async_op else None
            if not async_op:
                dist.send(tensor, dst=dst, tag=tag)
            self._record(CommType.SEND, tensor.numel() * tensor.element_size(), start)
            return work
        except Exception as e:
            self._record_error(CommType.SEND)
            raise RuntimeError(f"NCCL send to {dst} failed: {e}") from e

    def recv(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        src: int,
        tag: int = 0,
        async_op: bool = False,
        device: str | None = None,
    ) -> torch.Tensor:
        """Receive a tensor from a source rank."""
        self._ensure_initialized()
        tensor = torch.empty(shape, dtype=dtype, device=device or f"cuda:{self._rank}")
        start = time.time_ns()
        try:
            work = dist.irecv(tensor, src=src, tag=tag) if async_op else None
            if not async_op:
                dist.recv(tensor, src=src, tag=tag)
            self._record(CommType.RECV, tensor.numel() * tensor.element_size(), start)
            return tensor
        except Exception as e:
            self._record_error(CommType.RECV)
            raise RuntimeError(f"NCCL recv from {src} failed: {e}") from e

    def send_recv(
        self,
        send_tensor: torch.Tensor,
        recv_tensor: torch.Tensor,
        dst: int,
        src: int,
        tag: int = 0,
    ) -> None:
        """Bidirectional send + recv (non-blocking with sync)."""
        self._ensure_initialized()
        start = time.time_ns()
        try:
            send_work = dist.isend(send_tensor, dst=dst, tag=tag)
            recv_work = dist.irecv(recv_tensor, src=src, tag=tag)
            send_work.wait()
            recv_work.wait()
            total = send_tensor.numel() + recv_tensor.numel()
            self._record(CommType.P2P, total * send_tensor.element_size(), start)
        except Exception as e:
            self._record_error(CommType.P2P)
            raise RuntimeError(f"NCCL send_recv dst={dst} src={src} failed: {e}") from e

    # -------------------------------------------------------------------
    # Collectives
    # -------------------------------------------------------------------

    def all_reduce(self, tensor: torch.Tensor, op: dist.ReduceOp = dist.ReduceOp.SUM) -> torch.Tensor:
        self._ensure_initialized()
        start = time.time_ns()
        try:
            dist.all_reduce(tensor, op=op)
            self._record(CommType.ALL_REDUCE, tensor.numel() * tensor.element_size(), start)
            return tensor
        except Exception as e:
            self._record_error(CommType.ALL_REDUCE)
            raise RuntimeError(f"NCCL all_reduce failed: {e}") from e

    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        self._ensure_initialized()
        start = time.time_ns()
        try:
            dist.broadcast(tensor, src=src)
            self._record(CommType.BROADCAST, tensor.numel() * tensor.element_size(), start)
            return tensor
        except Exception as e:
            self._record_error(CommType.BROADCAST)
            raise RuntimeError(f"NCCL broadcast failed: {e}") from e

    def all_gather(self, tensor: torch.Tensor, gather_list: list[torch.Tensor]) -> list[torch.Tensor]:
        self._ensure_initialized()
        start = time.time_ns()
        try:
            dist.all_gather(gather_list, tensor)
            total = sum(t.numel() for t in gather_list) * tensor.element_size()
            self._record(CommType.ALL_GATHER, total, start)
            return gather_list
        except Exception as e:
            self._record_error(CommType.ALL_GATHER)
            raise RuntimeError(f"NCCL all_gather failed: {e}") from e

    def reduce_scatter(self, input_list: list[torch.Tensor], output: torch.Tensor, op: dist.ReduceOp = dist.ReduceOp.SUM) -> torch.Tensor:
        self._ensure_initialized()
        start = time.time_ns()
        try:
            dist.reduce_scatter(output, input_list, op=op)
            total = sum(t.numel() for t in input_list) * input_list[0].element_size()
            self._record(CommType.REDUCE_SCATTER, total, start)
            return output
        except Exception as e:
            self._record_error(CommType.REDUCE_SCATTER)
            raise RuntimeError(f"NCCL reduce_scatter failed: {e}") from e

    def barrier(self) -> None:
        if self.is_initialized:
            dist.barrier()

    # -------------------------------------------------------------------
    # P2P Groups
    # -------------------------------------------------------------------

    def create_p2p_group(self, group_name: str, ranks: list[int]) -> dist.ProcessGroup:
        """Create a sub-communicator for a specific set of ranks."""
        self._ensure_initialized()
        group = dist.new_group(ranks=ranks, backend=self._backend)
        with self._lock:
            self._p2p_groups[group_name] = group
        return group

    def p2p_group_send(self, group_name: str, tensor: torch.Tensor, dst: int, tag: int = 0) -> None:
        group = self._p2p_groups.get(group_name)
        if group is None:
            raise ValueError(f"P2P group {group_name} not found")
        dist.send(tensor, dst=dst, tag=tag, group=group)

    def p2p_group_recv(self, group_name: str, shape: tuple[int, ...], dtype: torch.dtype, src: int, tag: int = 0) -> torch.Tensor:
        group = self._p2p_groups.get(group_name)
        if group is None:
            raise ValueError(f"P2P group {group_name} not found")
        tensor = torch.empty(shape, dtype=dtype, device=f"cuda:{self._rank}")
        dist.recv(tensor, src=src, tag=tag, group=group)
        return tensor

    # -------------------------------------------------------------------
    # Async Operations (with Work handles)
    # -------------------------------------------------------------------

    class AsyncOp:
        def __init__(self, work: dist.Work):
            self._work = work

        def wait(self) -> None:
            self._work.wait()

        @property
        def is_completed(self) -> bool:
            return self._work.is_completed()

    def async_send(self, tensor: torch.Tensor, dst: int, tag: int = 0) -> AsyncOp:
        work = self.send(tensor, dst, tag, async_op=True)
        return self.AsyncOp(work) if work else None

    def async_recv(self, shape: tuple[int, ...], dtype: torch.dtype, src: int, tag: int = 0, device: str | None = None) -> tuple[torch.Tensor, AsyncOp]:
        self._ensure_initialized()
        tensor = torch.empty(shape, dtype=dtype, device=device or f"cuda:{self._rank}")
        work = dist.irecv(tensor, src=src, tag=tag)
        return tensor, self.AsyncOp(work)

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            self.initialize()
        if not dist.is_initialized():
            raise RuntimeError("NCCL not initialized")

    def _record(self, comm_type: CommType, bytes: int, start_ns: int) -> None:
        elapsed = time.time_ns() - start_ns
        with self._lock:
            if comm_type not in self._stats:
                self._stats[comm_type] = NcclTransferStats(comm_type=comm_type)
            s = self._stats[comm_type]
            s.total_bytes += bytes
            s.total_time_ns += elapsed
            s.count += 1

    def _record_error(self, comm_type: CommType) -> None:
        with self._lock:
            if comm_type not in self._stats:
                self._stats[comm_type] = NcclTransferStats(comm_type=comm_type)
            self._stats[comm_type].errors += 1

    def stats(self) -> dict[str, Any]:
        result = {}
        with self._lock:
            for comm_type, stat in self._stats.items():
                if stat.count > 0:
                    result[comm_type.value] = {
                        "count": stat.count,
                        "total_bytes": stat.total_bytes,
                        "avg_latency_us": round(stat.avg_latency_us, 2),
                        "bandwidth_gbps": round(stat.bandwidth_gbps, 2),
                        "errors": stat.errors,
                    }
        return result

    def summary(self) -> str:
        s = self.stats()
        lines = [f"NcclTransport: rank={self._rank}, world={self._world_size}"]
        for name, data in s.items():
            lines.append(f"  {name}: {data['count']} ops, {data['bandwidth_gbps']} Gbps, {data['avg_latency_us']}us")
        if not s:
            lines.append("  (no operations recorded)")
        return "\n".join(lines)
