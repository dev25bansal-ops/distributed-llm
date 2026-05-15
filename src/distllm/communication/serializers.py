"""Tensor and KV cache serialization to/from protobuf."""

import torch
import numpy as np
from typing import List, Tuple, Optional

from distllm.communication.node_pb2 import Tensor, KVCache as ProtoKVCache, KVLayerCache
from distllm.core.kv_cache import KVCache


# Shared memory buffer pool for zero-copy deserialization
_tensor_buffer_pool: List[np.ndarray] = []
_buffer_pool_max_size: int = 256  # Max buffers to retain
_buffer_pool_size: int = 0  # Current pool size


def _get_buffer(size: int) -> np.ndarray:
    """Get a pre-allocated buffer from the pool or create a new one."""
    if _tensor_buffer_pool:
        buf = _tensor_buffer_pool.pop()
        if buf.nbytes >= size:
            return buf
    return np.empty(max(size, 1024), dtype=np.uint8)


def _release_buffer(buf: np.ndarray) -> None:
    """Return a buffer to the pool for reuse."""
    global _buffer_pool_size
    if _buffer_pool_size < _buffer_pool_max_size:
        _tensor_buffer_pool.append(buf)
        _buffer_pool_size += 1


def tensor_to_proto(tensor: torch.Tensor) -> Tensor:
    """Convert PyTorch tensor to protobuf Tensor message using efficient raw bytes.

    Uses .view(torch.uint8) for dtype-agnostic serialization — no numpy
    conversion needed, avoiding the 2x bandwidth penalty for bfloat16.
    """
    if tensor is None:
        return Tensor(data=[], shape=[], dtype="none")

    t = tensor.cpu().detach()
    dtype_str = str(t.dtype)
    if not dtype_str.startswith("torch."):
        dtype_str = f"torch.{dtype_str}"

    # Flatten 0-dim tensors since .view(torch.uint8) requires at least 1-dim
    if t.dim() == 0:
        t = t.reshape(1)

    # Ensure contiguous for zero-copy numpy conversion
    raw_bytes = t.contiguous().view(torch.uint8).numpy(force=True).tobytes()

    return Tensor(
        raw_data=raw_bytes,
        shape=list(tensor.shape),
        dtype=dtype_str,
    )


def proto_to_tensor(proto: Tensor, device: str = "cpu") -> torch.Tensor:
    """Convert protobuf Tensor message to PyTorch tensor from raw bytes.

    Uses np.frombuffer for zero-copy read of protobuf data, then copies
    only when moving to target device.
    """
    if not proto.shape:
        if proto.raw_data:
            pass  # Treat as scalar that was flattened during serialization
        else:
            return torch.empty(0, device=device)

    dtype_map = {
        "torch.float32": torch.float32,
        "torch.float16": torch.float16,
        "torch.float64": torch.float64,
        "torch.bfloat16": torch.bfloat16,
        "torch.int64": torch.int64,
        "torch.int32": torch.int32,
        "torch.int16": torch.int16,
        "torch.int8": torch.int8,
        "torch.uint8": torch.uint8,
        "torch.bool": torch.bool,
        "float32": torch.float32,
        "float16": torch.float16,
        "float64": torch.float64,
        "bfloat16": torch.bfloat16,
        "int64": torch.int64,
        "int32": torch.int32,
        "bool": torch.bool,
    }
    torch_dtype = dtype_map.get(proto.dtype, torch.float32)

    if proto.raw_data:
        dtype_item_size = {
            torch.float32: 4, torch.float16: 2, torch.float64: 8,
            torch.bfloat16: 2, torch.int64: 8, torch.int32: 4,
            torch.int16: 2, torch.int8: 1, torch.uint8: 1, torch.bool: 1,
        }
        item_size = dtype_item_size.get(torch_dtype, 4)
        expected_elements = 1
        for dim in proto.shape:
            expected_elements *= dim
        expected_bytes = expected_elements * item_size

        # Guard against malicious proto declaring huge shapes
        if expected_bytes > 64 * 1024 * 1024:
            raise ValueError(
                f"Tensor size exceeds maximum allowed (64MB): {expected_bytes} bytes "
                f"for shape {proto.shape} with dtype {proto.dtype}"
            )

        if len(proto.raw_data) != expected_bytes:
            raise ValueError(
                f"Tensor data length mismatch: declared shape {proto.shape} with dtype {proto.dtype} "
                f"expects {expected_bytes} bytes, but received {len(proto.raw_data)} bytes"
            )

        # Zero-copy: frombuffer creates view into proto.raw_data without copying
        arr = np.frombuffer(proto.raw_data, dtype=np.uint8)
        shape = list(proto.shape)
        tensor = torch.from_numpy(arr).view(torch_dtype).reshape(shape).clone()
    else:
        shape = list(proto.shape)
        tensor = torch.tensor(proto.data, dtype=torch.float32).reshape(shape)

    return tensor.to(device)


def kv_cache_to_proto(cache: KVCache) -> ProtoKVCache:
    """Serialize KVCache to protobuf KVCache message."""
    layers = []
    for k, v in cache.cache:
        layers.append(KVLayerCache(
            key_states=tensor_to_proto(k),
            value_states=tensor_to_proto(v),
        ))
    return ProtoKVCache(layers=layers)


def proto_to_kv_cache(proto: ProtoKVCache, device: str = "cpu") -> KVCache:
    """Deserialize protobuf KVCache message to KVCache."""
    cache = KVCache()
    layers = []
    for layer in proto.layers:
        k = proto_to_tensor(layer.key_states, device)
        v = proto_to_tensor(layer.value_states, device)
        layers.append((k, v))
    cache.set_all(layers)
    return cache
