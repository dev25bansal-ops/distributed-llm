"""Tensor and KV cache serialization to/from protobuf."""

import torch
import numpy as np
from typing import List, Tuple, Optional

from distllm.communication.node_pb2 import Tensor, KVCache as ProtoKVCache, KVLayerCache
from distllm.core.kv_cache import KVCache
from distllm.constants import TENSOR_MAX_DIMS as MAX_TENSOR_DIMS, TENSOR_MAX_DIM_SIZE as MAX_DIM_SIZE, TENSOR_MAX_TOTAL_BYTES as MAX_TENSOR_BYTES
from distllm.errors import SerializationError


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


# Maximum allowed dimensions for tensor shape validation
# (imported from distllm.constants as MAX_TENSOR_DIMS, MAX_DIM_SIZE, MAX_TENSOR_BYTES)
MAX_KV_CACHE_LAYERS = 256


def _validate_tensor_shape(shape: list, dtype_str: str, raw_data_len: int) -> int:
    """Validate tensor shape bounds before allocation to prevent OOM.

    Args:
        shape: List of dimension sizes from protobuf.
        dtype_str: String representation of the dtype.
        raw_data_len: Length of raw_data bytes received.

    Returns:
        expected_bytes: Total expected size in bytes.

    Raises:
        ValueError: If shape exceeds any bound.
    """
    if len(shape) > MAX_TENSOR_DIMS:
        raise SerializationError(
            f"Tensor has {len(shape)} dimensions, maximum allowed is {MAX_TENSOR_DIMS}"
        )

    for i, dim in enumerate(shape):
        if dim < 0:
            raise SerializationError(f"Tensor dimension {i} has negative size {dim}")
        if dim > MAX_DIM_SIZE:
            raise SerializationError(
                f"Tensor dimension {i} has size {dim}, maximum allowed is {MAX_DIM_SIZE}"
            )

    dtype_item_size = {
        torch.float32: 4, torch.float16: 2, torch.float64: 8,
        torch.bfloat16: 2, torch.int64: 8, torch.int32: 4,
        torch.int16: 2, torch.int8: 1, torch.uint8: 1, torch.bool: 1,
    }
    dtype_map = {
        "torch.float32": torch.float32, "torch.float16": torch.float16,
        "torch.float64": torch.float64, "torch.bfloat16": torch.bfloat16,
        "torch.int64": torch.int64, "torch.int32": torch.int32,
        "torch.int16": torch.int16, "torch.int8": torch.int8,
        "torch.uint8": torch.uint8, "torch.bool": torch.bool,
        "float32": torch.float32, "float16": torch.float16,
        "float64": torch.float64, "bfloat16": torch.bfloat16,
        "int64": torch.int64, "int32": torch.int32, "bool": torch.bool,
    }
    torch_dtype = dtype_map.get(dtype_str, torch.float32)
    item_size = dtype_item_size.get(torch_dtype, 4)

    expected_elements = 1
    for dim in shape:
        expected_elements *= dim
    expected_bytes = expected_elements * item_size

    if expected_bytes > MAX_TENSOR_BYTES:
        raise SerializationError(
            f"Tensor size {expected_bytes} bytes exceeds maximum allowed "
            f"({MAX_TENSOR_BYTES} bytes) for shape {shape} with dtype {dtype_str}"
        )

    if raw_data_len != expected_bytes:
        raise SerializationError(
            f"Tensor data length mismatch: declared shape {shape} with dtype {dtype_str} "
            f"expects {expected_bytes} bytes, but received {raw_data_len} bytes"
        )

    return expected_bytes


def proto_to_tensor(proto: Tensor, device: str = "cpu") -> torch.Tensor:
    """Convert protobuf Tensor message to PyTorch tensor from raw bytes.

    Uses np.frombuffer for zero-copy read of protobuf data, then copies
    only when moving to target device. Validates shape bounds before
    allocating memory.
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
        # Validate shape bounds before any allocation
        _validate_tensor_shape(list(proto.shape), proto.dtype, len(proto.raw_data))

        # Zero-copy: frombuffer creates view into proto.raw_data without copying
        arr = np.frombuffer(proto.raw_data, dtype=np.uint8)
        shape = list(proto.shape)
        tensor = torch.from_numpy(arr).view(torch_dtype).reshape(shape).clone()
    else:
        shape = list(proto.shape)
        if len(shape) > MAX_TENSOR_DIMS:
            raise SerializationError(
                f"Tensor has {len(shape)} dimensions, maximum allowed is {MAX_TENSOR_DIMS}"
            )
        for dim in shape:
            if dim > MAX_DIM_SIZE:
                raise SerializationError(f"Tensor dimension size {dim} exceeds maximum {MAX_DIM_SIZE}")
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
    if len(proto.layers) > MAX_KV_CACHE_LAYERS:
        raise ValueError(
            f"KV cache has {len(proto.layers)} layers, maximum allowed is {MAX_KV_CACHE_LAYERS}"
        )
    cache = KVCache()
    layers = []
    for layer in proto.layers:
        k = proto_to_tensor(layer.key_states, device)
        v = proto_to_tensor(layer.value_states, device)
        layers.append((k, v))
    cache.set_all(layers)
    return cache
