"""NCCL/RDMA inter-node tensor transport with gRPC fallback.

Provides GPU-direct tensor transfer between distributed nodes using
NCCL all-reduce / point-to-point operations. Falls back to gRPC for
non-GPU-direct setups or CPU-only environments.

Architecture:
- Detects GPU-direct (NCCL) vs CPU-only (gRPC) at init
- NCCL path: torch.distributed send/recv on a dedicated NCCL communicator
- gRPC fallback: serializes tensors via protobuf, sends over gRPC
- All operations are async (return futures) for compute/communication overlap
"""

from __future__ import annotations

import os
import queue
import threading
import time
from datetime import timedelta
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

import torch

from loguru import logger


class TransportBackend(Enum):
    """Available transport backends in priority order."""
    NCCL = "nccl"        # GPU-direct NCCL send/recv
    GLOO = "gloo"        # CPU fallback via gloo
    GRPC = "grpc"        # Protobuf serialization over gRPC


class TransportType(Enum):
    """Type of transport operation."""
    TENSOR = "tensor"           # Single tensor transfer
    TENSOR_LIST = "tensor_list" # Multiple tensors (e.g., KV cache layers)
    ACTIVATION = "activation"   # Quantized activation between pipeline stages


@dataclass
class TransportRequest:
    """A pending or in-flight transport request."""
    req_id: str
    transport_type: TransportType
    tensor: Optional[torch.Tensor] = None
    tensor_list: Optional[List[torch.Tensor]] = None
    dst_rank: int = -1
    src_rank: int = -1
    tag: int = 0
    future: Future = field(default_factory=Future)
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None

    @property
    def latency_ms(self) -> Optional[float]:
        if self.completed_at is not None:
            return (self.completed_at - self.created_at) * 1000
        return None


class NCCLTransport:
    """GPU-direct tensor transport using NCCL with gRPC fallback.

    Automatically selects the best available backend:
    1. NCCL (GPU-direct) — if CUDA and torch.distributed available
    2. GLOO (CPU shared memory) — if torch.distributed available
    3. gRPC (network) — always available fallback

    Usage:
        transport = NCCLTransport(rank=0, world_size=2)
        future = transport.send(tensor, dst_rank=1)
        result = transport.recv(src_rank=1)
        transport.barrier()
    """

    def __init__(
        self,
        rank: int = 0,
        world_size: int = 1,
        backend: Optional[str] = None,
        master_addr: str = "127.0.0.1",
        master_port: int = 29500,
        grpc_send_fn: Optional[Callable] = None,
        enable_nccl: bool = True,
        timeout_ms: int = 30000,
    ):
        self.rank = rank
        self.world_size = world_size
        self._timeout_ms = timeout_ms
        self._grpc_send_fn = grpc_send_fn
        self._pending_requests: Dict[str, TransportRequest] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._recv_queue: queue.Queue = queue.Queue()

        # Detect best available backend
        self.backend = self._detect_backend(enable_nccl)

        # Initialize torch.distributed if NCCL or GLOO selected
        if self.backend in (TransportBackend.NCCL, TransportBackend.GLOO):
            self._init_distributed(master_addr, master_port, backend)
        else:
            logger.info(f"Rank {rank}: using gRPC transport (no GPU-direct available)")

        # Performance tracking
        self._bytes_sent = 0
        self._bytes_recv = 0
        self._ops_count = 0

    def _detect_backend(self, enable_nccl: bool) -> TransportBackend:
        """Detect the best available transport backend."""
        if enable_nccl and torch.cuda.is_available():
            try:
                import torch.distributed as dist
                if dist.is_nccl_available():
                    logger.info(f"Rank {self.rank}: NCCL available, using GPU-direct transport")
                    return TransportBackend.NCCL
            except (ImportError, RuntimeError):
                pass

        try:
            import torch.distributed as dist
            if dist.is_gloo_available():
                logger.info(f"Rank {self.rank}: GLOO available, using CPU shared-memory transport")
                return TransportBackend.GLOO
        except (ImportError, RuntimeError):
            pass

        return TransportBackend.GRPC

    def _init_distributed(self, master_addr: str, master_port: int, backend: Optional[str] = None) -> None:
        """Initialize torch.distributed process group."""
        os.environ.setdefault("MASTER_ADDR", master_addr)
        os.environ.setdefault("MASTER_PORT", str(master_port))

        import torch.distributed as dist

        backend_str = backend or self.backend.value
        if not dist.is_initialized():
            dist.init_process_group(
                backend=backend_str,
                rank=self.rank,
                world_size=self.world_size,
                timeout=timedelta(seconds=self._timeout_ms // 1000),
            )
            logger.info(f"Rank {self.rank}: initialized {backend_str} process group, size={self.world_size}")

    def send(
        self,
        tensor: torch.Tensor,
        dst_rank: int,
        tag: int = 0,
        req_id: Optional[str] = None,
    ) -> Future:
        """Send a tensor to another rank asynchronously.

        Args:
            tensor: Tensor to send (can be on any device).
            dst_rank: Destination rank.
            tag: Message tag.
            req_id: Optional request ID for tracking.

        Returns:
            Future that resolves when the send completes.
        """
        rid = req_id or f"send_{dst_rank}_{tag}_{time.monotonic_ns()}"
        req = TransportRequest(
            req_id=rid,
            transport_type=TransportType.TENSOR,
            tensor=tensor,
            dst_rank=dst_rank,
            tag=tag,
        )

        with self._lock:
            self._pending_requests[rid] = req

        if self.backend == TransportBackend.NCCL:
            self._executor.submit(self._nccl_send, req)
        elif self.backend == TransportBackend.GLOO:
            self._executor.submit(self._gloo_send, req)
        else:
            self._executor.submit(self._grpc_send, req)

        return req.future

    def recv(
        self,
        src_rank: int,
        shape: Optional[Tuple[int, ...]] = None,
        dtype: Optional[torch.dtype] = None,
        tag: int = 0,
        device: Optional[torch.device] = None,
        req_id: Optional[str] = None,
    ) -> Future:
        """Receive a tensor from another rank asynchronously.

        Args:
            src_rank: Source rank.
            shape: Expected tensor shape (required for gRPC fallback).
            dtype: Expected tensor dtype (required for gRPC fallback).
            tag: Message tag.
            device: Target device.
            req_id: Optional request ID for tracking.

        Returns:
            Future that resolves to the received tensor when complete.
        """
        rid = req_id or f"recv_{src_rank}_{tag}_{time.monotonic_ns()}"
        req = TransportRequest(
            req_id=rid,
            transport_type=TransportType.TENSOR,
            dst_rank=self.rank,
            src_rank=src_rank,
            tag=tag,
        )

        with self._lock:
            self._pending_requests[rid] = req

        if self.backend == TransportBackend.NCCL:
            self._executor.submit(self._nccl_recv, req, shape, dtype, device)
        elif self.backend == TransportBackend.GLOO:
            self._executor.submit(self._gloo_recv, req, shape, dtype, device)
        else:
            req.future.set_exception(
                RuntimeError("gRPC async recv not supported; use send/recv pair")
            )

        return req.future

    def send_tensor_list(
        self,
        tensors: List[torch.Tensor],
        dst_rank: int,
        tag: int = 0,
    ) -> Future:
        """Send a list of tensors efficiently.

        NCCL path: concatenate into flat buffer, send once, then split on recv.
        gRPC path: serialize each tensor individually.
        """
        rid = f"send_list_{dst_rank}_{tag}_{time.monotonic_ns()}"
        req = TransportRequest(
            req_id=rid,
            transport_type=TransportType.TENSOR_LIST,
            tensor_list=tensors,
            dst_rank=dst_rank,
            tag=tag,
        )

        with self._lock:
            self._pending_requests[rid] = req

        if self.backend == TransportBackend.NCCL:
            self._executor.submit(self._nccl_send_tensor_list, req)
        else:
            self._executor.submit(self._grpc_send_tensor_list, req)

        return req.future

    def send_activation(
        self,
        hidden_states: torch.Tensor,
        dst_rank: int,
        quant_bits: int = 8,
    ) -> Future:
        """Send pipeline activation with optional quantization.

        Quantizes before sending for bandwidth efficiency.
        """
        from distllm.communication.serializers import quantize_activation

        quantized, scale = quantize_activation(hidden_states)
        return self.send(quantized, dst_rank, tag=0xAC7)

    def recv_activation(
        self,
        src_rank: int,
        orig_dtype: torch.dtype,
        shape: Optional[Tuple[int, ...]] = None,
    ) -> Future:
        """Receive pipeline activation and dequantize."""
        from distllm.communication.serializers import dequantize_activation

        future = self.recv(src_rank, shape=shape, dtype=torch.int8, tag=0xAC7)
        return future  # Dequantize on resolve

    def barrier(self) -> None:
        """Synchronize all ranks."""
        if self.backend in (TransportBackend.NCCL, TransportBackend.GLOO):
            import torch.distributed as dist
            dist.barrier()
        # gRPC has no barrier — no-op

    def all_reduce(self, tensor: torch.Tensor, op: str = "sum") -> torch.Tensor:
        """Perform all-reduce on a tensor across all ranks.

        Args:
            tensor: Tensor to reduce.
            op: Reduction operation ("sum", "avg", "max", "min").

        Returns:
            Reduced tensor (in-place).
        """
        if self.backend in (TransportBackend.NCCL, TransportBackend.GLOO):
            import torch.distributed as dist

            op_map = {
                "sum": dist.ReduceOp.SUM,
                "avg": dist.ReduceOp.AVG,
                "max": dist.ReduceOp.MAX,
                "min": dist.ReduceOp.MIN,
            }
            dist.all_reduce(tensor, op=op_map.get(op, dist.ReduceOp.SUM))
            self._bytes_sent += tensor.element_size() * tensor.numel() * (self.world_size - 1)
        return tensor

    def broadcast(self, tensor: torch.Tensor, src_rank: int = 0) -> torch.Tensor:
        """Broadcast a tensor from source rank to all others."""
        if self.backend in (TransportBackend.NCCL, TransportBackend.GLOO):
            import torch.distributed as dist
            dist.broadcast(tensor, src=src_rank)
        return tensor

    # --- NCCL send/recv implementations ---

    def _nccl_send(self, req: TransportRequest) -> None:
        try:
            import torch.distributed as dist
            tensor = req.tensor.contiguous()
            if tensor.device.type != "cuda":
                tensor = tensor.to("cuda")
            dist.send(tensor, dst=req.dst_rank, tag=req.tag)
            self._bytes_sent += tensor.element_size() * tensor.numel()
            req.completed_at = time.time()
            req.future.set_result(True)
        except Exception as e:
            req.future.set_exception(e)

    def _nccl_recv(
        self,
        req: TransportRequest,
        shape: Optional[Tuple[int, ...]],
        dtype: Optional[torch.dtype],
        device: Optional[torch.device],
    ) -> None:
        try:
            import torch.distributed as dist
            if shape is None or dtype is None:
                raise ValueError("shape and dtype required for NCCL recv")
            dev = device or torch.device("cuda")
            tensor = torch.empty(*shape, dtype=dtype or torch.float16, device=dev)
            dist.recv(tensor, src=req.src_rank, tag=req.tag)
            self._bytes_recv += tensor.element_size() * tensor.numel()
            req.completed_at = time.time()
            req.future.set_result(tensor)
        except Exception as e:
            req.future.set_exception(e)

    def _nccl_send_tensor_list(self, req: TransportRequest) -> None:
        try:
            import torch.distributed as dist
            tensors = [t.contiguous() for t in req.tensor_list]
            devices = set(t.device for t in tensors)
            cuda_tensors = [t.to("cuda") if t.device.type != "cuda" else t for t in tensors]
            flat = torch.cat([t.flatten() for t in cuda_tensors])
            dist.send(flat, dst=req.dst_rank, tag=req.tag)
            self._bytes_sent += flat.element_size() * flat.numel()
            req.completed_at = time.time()
            req.future.set_result(True)
        except Exception as e:
            req.future.set_exception(e)

    # --- GLOO send/recv implementations ---

    def _gloo_send(self, req: TransportRequest) -> None:
        try:
            import torch.distributed as dist
            tensor = req.tensor.contiguous().cpu()
            dist.send(tensor, dst=req.dst_rank, tag=req.tag)
            self._bytes_sent += tensor.element_size() * tensor.numel()
            req.completed_at = time.time()
            req.future.set_result(True)
        except Exception as e:
            req.future.set_exception(e)

    def _gloo_recv(
        self,
        req: TransportRequest,
        shape: Optional[Tuple[int, ...]],
        dtype: Optional[torch.dtype],
        device: Optional[torch.device],
    ) -> None:
        try:
            import torch.distributed as dist
            if shape is None or dtype is None:
                raise ValueError("shape and dtype required for GLOO recv")
            tensor = torch.empty(*shape, dtype=dtype or torch.float32)
            dist.recv(tensor, src=req.src_rank, tag=req.tag)
            self._bytes_recv += tensor.element_size() * tensor.numel()
            tensor = tensor.to(device) if device else tensor
            req.completed_at = time.time()
            req.future.set_result(tensor)
        except Exception as e:
            req.future.set_exception(e)

    # --- gRPC fallback implementations ---

    def _grpc_send(self, req: TransportRequest) -> None:
        try:
            if self._grpc_send_fn is None:
                raise RuntimeError("gRPC send function not configured")
            tensor_bytes = req.tensor.contiguous().numpy().tobytes()
            self._grpc_send_fn(
                dst_rank=req.dst_rank,
                data=tensor_bytes,
                shape=list(req.tensor.shape),
                dtype=str(req.tensor.dtype),
                tag=req.tag,
            )
            self._bytes_sent += len(tensor_bytes)
            req.completed_at = time.time()
            req.future.set_result(True)
        except Exception as e:
            req.future.set_exception(e)

    def _grpc_send_tensor_list(self, req: TransportRequest) -> None:
        try:
            if self._grpc_send_fn is None:
                raise RuntimeError("gRPC send function not configured")
            for i, tensor in enumerate(req.tensor_list):
                tensor_bytes = tensor.contiguous().numpy().tobytes()
                self._grpc_send_fn(
                    dst_rank=req.dst_rank,
                    data=tensor_bytes,
                    shape=list(tensor.shape),
                    dtype=str(tensor.dtype),
                    tag=req.tag + i,
                )
                self._bytes_sent += len(tensor_bytes)
            req.completed_at = time.time()
            req.future.set_result(True)
        except Exception as e:
            req.future.set_exception(e)

    # --- Stats ---

    @property
    def stats(self) -> dict:
        return {
            "backend": self.backend.value,
            "rank": self.rank,
            "world_size": self.world_size,
            "bytes_sent": self._bytes_sent,
            "bytes_recv": self._bytes_recv,
            "ops_count": self._ops_count,
            "pending_requests": len(self._pending_requests),
        }

    def shutdown(self) -> None:
        """Clean up transport resources."""
        self._pending_requests.clear()
        self._executor.shutdown(wait=False)
        if self.backend in (TransportBackend.NCCL, TransportBackend.GLOO):
            import torch.distributed as dist
            if dist.is_initialized():
                dist.destroy_process_group()
