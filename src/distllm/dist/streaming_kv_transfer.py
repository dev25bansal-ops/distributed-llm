"""Streaming KV cache transfer for large caches.

Handles gRPC message size limits by chunking large KV cache
transfers into smaller streaming messages. Avoids the 4MB
default gRPC message size limit.

Usage::

    sender = StreamingKVTransfer()
    async for chunk in sender.stream_send(kv_cache, chunk_size_mb=2):
        await grpc_stub.StreamKVCache(chunk)
"""

from __future__ import annotations

import io
import threading
import time
from dataclasses import dataclass
from typing import Any, Generator

import torch
from loguru import logger


@dataclass
class KVChunk:
    """A single chunk of a KV cache transfer."""
    request_id: str
    chunk_index: int
    total_chunks: int
    layer_idx: int
    data: bytes
    shape: list[int]
    dtype: str
    is_last: bool = False


class StreamingKVTransfer:
    """Handles streaming KV cache transfers for large caches.

    Splits large KV tensors into chunks that fit within gRPC
    message size limits (default 4MB, configurable).
    """

    DEFAULT_CHUNK_SIZE_MB = 2  # 2MB per chunk (well under 4MB gRPC limit)

    def __init__(self, chunk_size_mb: float = DEFAULT_CHUNK_SIZE_MB):
        self._chunk_size_bytes = int(chunk_size_mb * 1024 * 1024)
        self._stats = {
            "transfers_sent": 0,
            "transfers_received": 0,
            "chunks_sent": 0,
            "chunks_received": 0,
            "bytes_transferred": 0,
        }

    def chunk_tensor(
        self,
        tensor: torch.Tensor,
        request_id: str,
        layer_idx: int,
    ) -> Generator[KVChunk, None, None]:
        """Split a tensor into chunks for streaming transfer.

        Yields KVChunk objects that fit within the chunk size limit.

        Args:
            tensor: The tensor to chunk.
            request_id: Request ID for reassembly.
            layer_idx: Layer index for the KV cache.

        Yields:
            KVChunk objects.
        """
        # Serialize tensor to bytes
        t = tensor.detach().contiguous()
        if t.is_cuda:
            t = t.cpu()
        raw = bytes(memoryview(t.numpy()))

        total_bytes = len(raw)
        total_chunks = max(1, (total_bytes + self._chunk_size_bytes - 1) // self._chunk_size_bytes)

        for i in range(total_chunks):
            start = i * self._chunk_size_bytes
            end = min(start + self._chunk_size_bytes, total_bytes)
            chunk_data = raw[start:end]

            yield KVChunk(
                request_id=request_id,
                chunk_index=i,
                total_chunks=total_chunks,
                layer_idx=layer_idx,
                data=chunk_data,
                shape=list(tensor.shape),
                dtype=str(tensor.dtype),
                is_last=(i == total_chunks - 1),
            )

            self._stats["chunks_sent"] += 1
            self._stats["bytes_transferred"] += len(chunk_data)

    def reassemble_chunks(
        self,
        chunks: list[KVChunk],
    ) -> torch.Tensor | None:
        """Reassemble a tensor from streamed chunks.

        Args:
            chunks: List of chunks (must be from same tensor).

        Returns:
            Reassembled tensor, or None if chunks are incomplete.
        """
        if not chunks:
            return None

        # Sort by chunk index
        chunks.sort(key=lambda c: c.chunk_index)

        # Verify completeness
        expected = chunks[0].total_chunks
        if len(chunks) != expected:
            logger.warning(f"Incomplete transfer: {len(chunks)}/{expected} chunks")
            return None

        # Concatenate data
        raw = b"".join(c.data for c in chunks)

        # Deserialize
        import numpy as np
        dtype_map = {
            "torch.float32": (np.float32, torch.float32),
            "torch.float16": (np.float16, torch.float16),
            "torch.bfloat16": (np.float16, torch.float16),  # BF16 stored as float16
        }

        shape = chunks[0].shape
        dtype_str = chunks[0].dtype
        np_dtype, torch_dtype = dtype_map.get(dtype_str, (np.float32, torch.float32))

        arr = np.frombuffer(raw, dtype=np_dtype).reshape(shape)
        tensor = torch.from_numpy(arr.copy()).to(torch_dtype)

        self._stats["transfers_received"] += 1
        self._stats["chunks_received"] += len(chunks)

        return tensor

    def estimate_chunks(self, tensor: torch.Tensor) -> int:
        """Estimate number of chunks needed for a tensor."""
        num_bytes = tensor.numel() * tensor.element_size()
        return max(1, (num_bytes + self._chunk_size_bytes - 1) // self._chunk_size_bytes)

    def needs_streaming(self, tensor: torch.Tensor) -> bool:
        """Check if tensor needs streaming (exceeds chunk size)."""
        num_bytes = tensor.numel() * tensor.element_size()
        return num_bytes > self._chunk_size_bytes

    def stats(self) -> dict:
        return {
            **self._stats,
            "chunk_size_mb": self._chunk_size_bytes / (1024 * 1024),
        }
