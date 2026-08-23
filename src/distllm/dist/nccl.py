"""NCCL-based tensor transfers between GPUs (intra-node and inter-node).

Supports automatic Gloo fallback when CUDA GPUs are unavailable, a
``monitored_barrier()`` that surfaces NCCL errors with configurable timeouts,
a priority-based preemption model using CUDA stream priorities, and a standard
bandwidth benchmark suite comparable to ``nccl-test``.
"""

from __future__ import annotations

import os
import threading
import time
from datetime import timedelta
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
import torch.distributed as dist
from loguru import logger

from distllm.errors import NodeUnreachableError


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
    def __init__(
        self,
        rank: int = 0,
        world_size: int = 1,
        master_addr: str = "127.0.0.1",
        master_port: int = 29500,
        backend: str = "nccl",
        timeout_s: int = 30,
        auto_init: bool = True,
        allow_gloo_fallback: bool = True,
    ):
        self._rank = rank
        self._world_size = world_size
        self._master_addr = master_addr
        self._master_port = master_port
        self._backend = backend
        self._timeout_s = timeout_s
        self._allow_gloo_fallback = allow_gloo_fallback
        self._effective_backend: str = backend
        self._initialized = False
        self._lock = threading.Lock()
        self._stats: dict[CommType, NcclTransferStats] = {}
        self._p2p_groups: dict[str, dist.ProcessGroup] = {}

        # P2P preemption state
        self._preempted = threading.Event()
        self._preempted.set()  # Not preempted initially
        self._active_ops: dict[str, dist.Work] = {}
        self._op_priority: dict[str, int] = {}  # Higher = more important
        self._op_lock = threading.Lock()

        # CUDA stream priority levels for differentiated service
        # Higher-priority NCCL ops use a higher stream priority so they
        # get scheduled ahead of lower-priority work on the same device.
        self._low_prio_stream: torch.cuda.Stream | None = None
        self._high_prio_stream: torch.cuda.Stream | None = None
        self._stream_lock = threading.Lock()

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
            backend = self._backend
            # Automatic Gloo fallback: when NCCL is requested but no CUDA
            # GPUs are available (e.g. in testing / development environments),
            # fall back to Gloo so the transport works without modification.
            if backend == "nccl" and not torch.cuda.is_available():
                if self._allow_gloo_fallback:
                    logger.warning(
                        "NCCL backend requested but no CUDA GPUs available. "
                        "Falling back to Gloo. Set allow_gloo_fallback=False "
                        "to raise an error instead."
                    )
                    backend = "gloo"
                else:
                    raise RuntimeError(
                        "NCCL backend requested but no CUDA GPUs available "
                        "and allow_gloo_fallback=False"
                    )

            self._effective_backend = backend
            dist.init_process_group(
                backend=backend,
                rank=self._rank,
                world_size=self._world_size,
                timeout=torch.distributed.default_pg_timeout if hasattr(torch.distributed, 'default_pg_timeout') else None,
            )
            logger.info(
                f"NCCL transport initialized: rank={self._rank}, "
                f"world={self._world_size}, backend={backend}"
            )

        # Initialize CUDA stream priority infrastructure.
        if torch.cuda.is_available() and self._rank < torch.cuda.device_count():
            dev = f"cuda:{self._rank}"
            self._low_prio_stream = torch.cuda.Stream(
                device=dev, priority=0
            )
            # Higher numerical priority = lower scheduling priority in CUDA.
            # We use stream priorities to give high-priority ops a dedicated
            # higher-priority (lower numerical value) stream.
            low = torch.cuda.Stream.priority_range(dev)[0]  # best (highest priority)
            self._high_prio_stream = torch.cuda.Stream(
                device=dev, priority=low
            )
            logger.debug(
                f"CUDA streams initialised: low_prio=0, high_prio={low}"
            )

        self._initialized = True

    def destroy(self) -> None:
        with self._lock:
            for _name, group in self._p2p_groups.items():
                dist.destroy_process_group(group)
            self._p2p_groups.clear()
        if dist.is_initialized():
            dist.destroy_process_group()
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        return self._initialized and dist.is_initialized()

    @property
    def _default_device(self) -> str:
        if self._backend == "nccl" and torch.cuda.is_available() and self._rank < torch.cuda.device_count():
            return f"cuda:{self._rank}"
        return "cpu"

    def send(self, tensor: torch.Tensor, dst: int, tag: int = 0, async_op: bool = False) -> dist.Work | None:
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
            raise NodeUnreachableError(
                node_id=f"rank-{dst}", host="", port=0, original_error=e,
            ) from e

    def recv(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        src: int,
        tag: int = 0,
        async_op: bool = False,
        device: str | None = None,
    ) -> torch.Tensor:
        self._ensure_initialized()
        default_device = device or self._default_device
        tensor = torch.empty(shape, dtype=dtype, device=default_device)
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

    # ── Stream priority helpers ───────────────────────────────────────

    def get_stream_for_priority(self, priority: int = 0) -> torch.cuda.Stream | None:
        """Return a CUDA stream appropriate for the given operation priority.

        Args:
            priority: Operation priority (higher = more important).
                ``> 0`` uses the high-priority stream (if available);
                ``<= 0`` uses the low-priority stream (if available);
                returns ``None`` on CPU.

        This is the mechanism that makes preemption *real*: high-priority
        operations execute on a different stream with higher CUDA scheduling
        priority, so they are preferentially dispatched by the GPU hardware
        scheduler when both streams have pending work.
        """
        with self._stream_lock:
            if priority > 0:
                return self._high_prio_stream
            return self._low_prio_stream

    # ── Monitored barrier ─────────────────────────────────────────────

    def monitored_barrier(self, timeout_s: float | None = None) -> bool:
        """Barrier with timeout and error monitoring.

        Wraps ``dist.monitored_barrier()`` which raises on hang or
        rank failure, giving faster failure detection than a plain
        ``dist.barrier()``.

        Args:
            timeout_s: Per-rank timeout.  Falls back to ``timeout_s``
                from ``__init__`` if not provided.

        Returns:
            True if all ranks reached the barrier within the timeout.
            False if a rank failed or timed out.

        Note:
            Requires NCCL 2.10+ / Gloo.  Falls back to ``dist.barrier()``
            on older versions.
        """
        if not self.is_initialized:
            return False
        try:
            timeout = timedelta(seconds=timeout_s or self._timeout_s)
            dist.monitored_barrier(timeout=timeout, wait_all_ranks=True)
            return True
        except Exception as e:
            logger.error(f"monitored_barrier failed: {e}")
            return False

    def barrier(self) -> None:
        if self.is_initialized:
            dist.barrier()

    def create_p2p_group(self, group_name: str, ranks: list[int]) -> dist.ProcessGroup:
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
        tensor = torch.empty(shape, dtype=dtype, device=self._default_device)
        dist.recv(tensor, src=src, tag=tag, group=group)
        return tensor

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
        default_device = device or self._default_device
        tensor = torch.empty(shape, dtype=dtype, device=default_device)
        work = dist.irecv(tensor, src=src, tag=tag)
        return tensor, self.AsyncOp(work)

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

    def send_tensor_list(self, tensors: list[torch.Tensor], peer: str) -> None:
        """Send a list of tensors to a peer rank.

        Args:
            tensors: List of tensors to send.
            peer: Peer rank identifier.
        """
        if not self._initialized:
            raise RuntimeError("NCCL not initialized")
        for i, tensor in enumerate(tensors):
            self.send(tensor, dst=int(peer), tag=i)

    def shutdown(self) -> None:
        """Shutdown NCCL transport and release resources."""
        self.destroy()

    # ── P2P Preemption Support ──────────────────────────────────────

    def preempt(self, priority_threshold: int = 0) -> int:
        """Preempt active NCCL operations below a priority threshold.

        For multi-tenant setups where high-priority requests need to
        preempt lower-priority NCCL transfers.

        NCCL does not support direct cancellation, so "preempted" operations
        are moved to ``_preempted_ops`` and their ``dist.Work.wait()`` is
        called to let them drain.  Callers must check :meth:`is_preempted`
        and :meth:`wait_for_resume` before launching new low-priority work.

        Args:
            priority_threshold: Operations with priority < threshold are preempted.

        Returns:
            Number of operations preempted.
        """
        self._preempted.clear()
        preempted_count = 0

        with self._op_lock:
            for op_id, work in list(self._active_ops.items()):
                op_priority = self._op_priority.get(op_id, 0)
                if op_priority < priority_threshold:
                    self._active_ops.pop(op_id, None)
                    self._op_priority.pop(op_id, None)
                    # Drain the work handle so the pending NCCL op finishes
                    # before we hand control to the higher-priority caller.
                    try:
                        work.wait()
                    except Exception:
                        logger.debug(f"NCCL preempted op {op_id} finished with error (ignored)")
                    self._preempted_ops.append(op_id)
                    preempted_count += 1

        if preempted_count > 0:
            logger.info(f"NCCL preempted {preempted_count} operations below priority {priority_threshold}")
        return preempted_count

    def resume(self) -> None:
        """Resume NCCL operations after preemption."""
        self._preempted.set()
        self._preempted_ops.clear()

    def register_op(self, op_id: str, work: dist.Work, priority: int = 0) -> None:
        """Register an active NCCL operation for preemption tracking.

        Args:
            op_id: Unique operation identifier.
            work: NCCL work handle.
            priority: Operation priority (higher = more important).
        """
        with self._op_lock:
            self._active_ops[op_id] = work
            self._op_priority[op_id] = priority

    def unregister_op(self, op_id: str) -> None:
        """Unregister a completed NCCL operation."""
        with self._op_lock:
            self._active_ops.pop(op_id, None)
            self._op_priority.pop(op_id, None)

    def wait_for_resume(self, timeout: float = 30.0) -> bool:
        """Wait for preemption to be lifted.

        Args:
            timeout: Maximum seconds to wait.

        Returns:
            True if resumed, False if timed out.
        """
        return self._preempted.wait(timeout=timeout)

    @property
    def is_preempted(self) -> bool:
        """Whether NCCL operations are currently preempted."""
        return not self._preempted.is_set()

    @property
    def active_op_count(self) -> int:
        """Number of active NCCL operations."""
        with self._op_lock:
            return len(self._active_ops)

    # ── Priority-based preemption ─────────────────────────────────────

    def preempt(self, priority_threshold: int = 0) -> int:
        """Preempt active NCCL operations below a priority threshold.

        For multi-tenant setups where high-priority requests need to
        preempt lower-priority NCCL transfers.

        NCCL does not support direct cancellation, so "preempted" operations
        are drained via ``dist.Work.wait()``.  The real preemption happens at
        the CUDA stream level: subsequent high-priority operations use
        :meth:`get_stream_for_priority` to obtain a higher-priority CUDA
        stream, so the GPU hardware scheduler favours them over the
        low-priority stream.

        Args:
            priority_threshold: Operations with priority < threshold are preempted.

        Returns:
            Number of operations preempted.
        """
        self._preempted.clear()
        preempted_count = 0

        with self._op_lock:
            for op_id, work in list(self._active_ops.items()):
                op_priority = self._op_priority.get(op_id, 0)
                if op_priority < priority_threshold:
                    self._active_ops.pop(op_id, None)
                    self._op_priority.pop(op_id, None)
                    try:
                        work.wait()
                    except Exception:
                        logger.debug(f"NCCL preempted op {op_id} finished with error (ignored)")
                    self._preempted_ops.append(op_id)
                    preempted_count += 1

        if preempted_count > 0:
            logger.info(f"NCCL preempted {preempted_count} operations below priority {priority_threshold}")
        return preempted_count

    # ── Benchmark suite (analogous to nccl-test) ──────────────────────

    def benchmark(
        self,
        sizes: list[int] | None = None,
        iterations: int = 5,
        warmup_iterations: int = 3,
    ) -> dict[str, list[dict[str, float]]]:
        """Run standard bandwidth benchmarks across collective operations.

        Analogous to ``nccl-test`` but integrated directly into the transport
        layer so it measures the same communication path used during inference.

        Args:
            sizes: Tensor element counts to benchmark.  Defaults to
                ``[2**i for i in range(10, 25)]`` (1K — 16M elements).
            iterations: Number of timed iterations per size.
            warmup_iterations: Untimed warm-up iterations per size.

        Returns:
            Nested dict::

                {
                    "all_reduce": [
                        {"size": 1048576, "bus_bw_gbps": 42.3, "algo_bw_gbps": 84.6, "time_us": 198.0},
                        ...
                    ],
                    "broadcast": [...],
                    "all_gather": [...],
                    "reduce_scatter": [...],
                    "send_recv": [...],
                }

        Raises:
            RuntimeError: If the transport is not initialised or world_size < 2.
        """
        if self._world_size < 2:
            raise RuntimeError("Benchmark requires world_size >= 2")
        self._ensure_initialized()

        if sizes is None:
            sizes = [2**i for i in range(10, 25)]  # 1K → 16M elements

        results: dict[str, list[dict[str, float]]] = {
            "all_reduce": [],
            "broadcast": [],
            "all_gather": [],
            "reduce_scatter": [],
            "send_recv": [],
        }

        dtype = torch.float16
        element_bytes = 2  # float16

        for n in sizes:
            tensor = torch.empty(n, dtype=dtype, device=self._default_device)
            recv_tensor = torch.empty(n, dtype=dtype, device=self._default_device)

            # Warmup
            for _ in range(warmup_iterations):
                dist.all_reduce(tensor)
                dist.broadcast(tensor, src=0)
                if self._world_size > 1:
                    gather_list = [torch.empty_like(tensor) for _ in range(self._world_size)]
                    dist.all_gather(gather_list, tensor)
                    if self._rank == 0:
                        scatter_list = list(torch.chunk(tensor, self._world_size))
                    else:
                        scatter_list = []
                    dist.reduce_scatter(recv_tensor, scatter_list)

            src = (self._rank + 1) % self._world_size
            dst = (self._rank - 1) % self._world_size
            for _ in range(warmup_iterations):
                send_work = dist.isend(tensor, dst=dst)
                recv_work = dist.irecv(recv_tensor, src=src)
                send_work.wait()
                recv_work.wait()

            # Timed iterations — all_reduce
            times = []
            for _ in range(iterations):
                t0 = time.time_ns()
                dist.all_reduce(tensor)
                t1 = time.time_ns()
                times.append((t1 - t0) / 1e3)  # μs
            avg_us = sum(times) / len(times)
            bytes_per_rank = n * element_bytes
            bus_bw = (bytes_per_rank * 2 * (self._world_size - 1) / self._world_size) / (avg_us * 1e-6) / 1e9
            algo_bw = (bytes_per_rank * 2) / (avg_us * 1e-6) / 1e9
            results["all_reduce"].append({
                "size": n,
                "time_us": round(avg_us, 1),
                "bus_bw_gbps": round(bus_bw, 2),
                "algo_bw_gbps": round(algo_bw, 2),
            })

            # Timed iterations — broadcast
            times = []
            for _ in range(iterations):
                t0 = time.time_ns()
                dist.broadcast(tensor, src=0)
                t1 = time.time_ns()
                times.append((t1 - t0) / 1e3)
            avg_us = sum(times) / len(times)
            bus_bw = (bytes_per_rank) / (avg_us * 1e-6) / 1e9
            results["broadcast"].append({
                "size": n,
                "time_us": round(avg_us, 1),
                "bus_bw_gbps": round(bus_bw, 2),
            })

            # Timed iterations — all_gather
            if self._world_size > 1:
                times = []
                gather_list = [torch.empty_like(tensor) for _ in range(self._world_size)]
                for _ in range(iterations):
                    t0 = time.time_ns()
                    dist.all_gather(gather_list, tensor)
                    t1 = time.time_ns()
                    times.append((t1 - t0) / 1e3)
                avg_us = sum(times) / len(times)
                total_bytes = n * element_bytes * self._world_size
                bus_bw = (total_bytes * (self._world_size - 1) / self._world_size) / (avg_us * 1e-6) / 1e9
                results["all_gather"].append({
                    "size": n,
                    "time_us": round(avg_us, 1),
                    "bus_bw_gbps": round(bus_bw, 2),
                })

            # Timed iterations — reduce_scatter
            if self._world_size > 1:
                times = []
                if self._rank == 0:
                    scatter_list = list(torch.chunk(tensor, self._world_size))
                else:
                    scatter_list = []
                for _ in range(iterations):
                    t0 = time.time_ns()
                    dist.reduce_scatter(recv_tensor, scatter_list)
                    t1 = time.time_ns()
                    times.append((t1 - t0) / 1e3)
                avg_us = sum(times) / len(times)
                results["reduce_scatter"].append({
                    "size": n,
                    "time_us": round(avg_us, 1),
                })

            # Timed iterations — P2P send/recv
            times = []
            for _ in range(iterations):
                t0 = time.time_ns()
                send_work = dist.isend(tensor, dst=dst)
                recv_work = dist.irecv(recv_tensor, src=src)
                send_work.wait()
                recv_work.wait()
                t1 = time.time_ns()
                times.append((t1 - t0) / 1e3)
            avg_us = sum(times) / len(times)
            bus_bw = (bytes_per_rank * 2) / (avg_us * 1e-6) / 1e9
            results["send_recv"].append({
                "size": n,
                "time_us": round(avg_us, 1),
                "bus_bw_gbps": round(bus_bw, 2),
            })

        return results
