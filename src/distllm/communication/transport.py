"""Transport abstraction for inter-node tensor transfers.

Provides a unified interface for sending/receiving tensors between nodes.
Supports NCCL (GPU-direct) as primary backend with gRPC fallback.

Design:
- Control plane: gRPC for metadata (request_id, config, error codes)
- Data plane: NCCL for tensor data (hidden_states, KV cache, position_ids)
"""

from enum import Enum
from typing import Any, Callable, Optional

import torch
from loguru import logger


class TransportBackend(Enum):
    """Available transport backends."""
    NCCL = "nccl"
    GLOO = "gloo"
    GRPC = "grpc"


class TensorTransport:
    """Abstraction for tensor transfers between nodes.

    Uses NCCL for GPU-direct transfers when available, falls back to
    gRPC serialization for CPU or when NCCL is unavailable.

    Usage:
        transport = TensorTransport(backend="nccl", rank=0, world_size=4)
        transport.send_tensor(hidden_states, dst_rank=1)
        tensor = transport.recv_tensor(src_rank=1, shape=hidden.shape, dtype=hidden.dtype)
    """

    def __init__(
        self,
        backend: str | TransportBackend = "nccl",
        rank: int = 0,
        world_size: int = 1,
        group: Any = None,
    ):
        self.rank = rank
        self.world_size = world_size
        self._group = group
        self._backend = self._resolve_backend(backend)
        self._available = self._check_availability()

    def _resolve_backend(self, backend: str | TransportBackend) -> TransportBackend:
        if isinstance(backend, TransportBackend):
            return backend
        try:
            return TransportBackend(backend.lower())
        except ValueError:
            logger.warning(f"Unknown backend '{backend}', defaulting to NCCL")
            return TransportBackend.NCCL

    def _check_availability(self) -> bool:
        if self._backend == TransportBackend.NCCL:
            if not torch.cuda.is_available():
                logger.debug("NCCL requested but CUDA unavailable, falling back to gRPC")
                self._backend = TransportBackend.GRPC
                return True
            try:
                import torch.distributed as dist
                if dist.is_available():
                    return True
            except ImportError:
                pass
            logger.debug("NCCL not available, falling back to gRPC")
            self._backend = TransportBackend.GRPC
            return True
        elif self._backend == TransportBackend.GLOO:
            try:
                import torch.distributed as dist
                return dist.is_available()
            except ImportError:
                return False
        return True  # gRPC always available

    @property
    def backend(self) -> TransportBackend:
        return self._backend

    @property
    def is_available(self) -> bool:
        return self._available

    def send_tensor(self, tensor: torch.Tensor, dst_rank: int, tag: int = 0) -> None:
        """Send a tensor to a specific rank.

        For NCCL: direct GPU-to-GPU transfer (no serialization).
        For gRPC: serializes to bytes and sends via callback.
        """
        if self._backend == TransportBackend.NCCL:
            import torch.distributed as dist
            dist.send(tensor.contiguous(), dst=dst_rank, group=self._group, tag=tag)
        elif self._backend == TransportBackend.GLOO:
            import torch.distributed as dist
            dist.send(tensor.contiguous().cpu(), dst=dst_rank, group=self._group, tag=tag)
        else:
            raise NotImplementedError(
                "gRPC tensor send requires a callback. Use send_tensor_with_callback instead."
            )

    def recv_tensor(
        self, src_rank: int, shape: tuple, dtype: torch.dtype, tag: int = 0,
        device: str = "cuda",
    ) -> torch.Tensor:
        """Receive a tensor from a specific rank.

        Pre-allocates the receive buffer for zero-copy with NCCL.
        """
        if self._backend == TransportBackend.NCCL:
            import torch.distributed as dist
            tensor = torch.empty(shape, dtype=dtype, device=device)
            dist.recv(tensor, src=src_rank, group=self._group, tag=tag)
            return tensor
        elif self._backend == TransportBackend.GLOO:
            import torch.distributed as dist
            tensor = torch.empty(shape, dtype=dtype, device="cpu")
            dist.recv(tensor, src=src_rank, group=self._group, tag=tag)
            return tensor.to(device)
        else:
            raise NotImplementedError(
                "gRPC tensor recv requires a callback. Use recv_tensor_with_callback instead."
            )

    def send_tensor_with_callback(
        self,
        tensor: torch.Tensor,
        send_fn: Callable[[bytes], None],
    ) -> None:
        """Send tensor via gRPC callback (serializes to bytes).

        For fallback when NCCL is not available.
        """
        # Move to CPU for serialization
        cpu_tensor = tensor.detach().to("cpu", non_blocking=True)
        if tensor.is_cuda:
            torch.cuda.current_stream().synchronize()

        # Serialize: shape + dtype + raw bytes
        data = cpu_tensor.contiguous().view(torch.uint8).numpy(force=True).tobytes()
        metadata = f"{','.join(str(d) for d in tensor.shape)}|{str(tensor.dtype)}|".encode()
        send_fn(metadata + data)

    def recv_tensor_from_bytes(
        self, data: bytes, device: str = "cuda",
    ) -> torch.Tensor:
        """Deserialize tensor from gRPC bytes.

        Expected format: "shape|dtype|raw_bytes"
        """
        # Parse metadata
        header_end = data.index(b"|", data.index(b"|") + 1)
        header = data[:header_end].decode()
        shape_str, dtype_str = header.split("|")
        shape = tuple(int(d) for d in shape_str.split(","))
        dtype = getattr(torch, dtype_str)

        # Deserialize raw bytes
        raw_data = data[header_end + 1:]
        import numpy as np
        arr = np.frombuffer(raw_data, dtype=np.uint8)
        tensor = torch.from_numpy(arr).view(dtype).reshape(shape)
        return tensor.to(device)

    def all_reduce(self, tensor: torch.Tensor, op: str = "sum") -> torch.Tensor:
        """All-reduce across all ranks."""
        if self._backend == TransportBackend.NCCL:
            import torch.distributed as dist
            from torch.distributed import ReduceOp
            op_map = {"sum": ReduceOp.SUM, "avg": ReduceOp.AVG, "max": ReduceOp.MAX}
            dist.all_reduce(tensor, op=op_map.get(op, ReduceOp.SUM), group=self._group)
            return tensor
        elif self._backend == TransportBackend.GLOO:
            import torch.distributed as dist
            from torch.distributed import ReduceOp
            op_map = {"sum": ReduceOp.SUM, "avg": ReduceOp.AVG, "max": ReduceOp.MAX}
            cpu_tensor = tensor.cpu()
            dist.all_reduce(cpu_tensor, op=op_map.get(op, ReduceOp.SUM), group=self._group)
            return cpu_tensor.to(tensor.device)
        else:
            raise NotImplementedError("All-reduce not supported for gRPC backend")

    def broadcast(self, tensor: torch.Tensor, src_rank: int = 0) -> torch.Tensor:
        """Broadcast tensor from src_rank to all ranks."""
        if self._backend == TransportBackend.NCCL:
            import torch.distributed as dist
            if self.rank != src_rank:
                tensor = torch.empty_like(tensor)
            dist.broadcast(tensor, src=src_rank, group=self._group)
            return tensor
        else:
            raise NotImplementedError("Broadcast not supported for gRPC backend")

    def barrier(self) -> None:
        """Synchronize all ranks."""
        if self._backend in (TransportBackend.NCCL, TransportBackend.GLOO):
            import torch.distributed as dist
            dist.barrier(group=self._group)

    def stats(self) -> dict:
        return {
            "backend": self._backend.value,
            "rank": self.rank,
            "world_size": self.world_size,
            "available": self._available,
        }
