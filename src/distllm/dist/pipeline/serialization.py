"""Protobuf serialization helpers for tensor transfer."""

from __future__ import annotations

import numpy as np
import torch
from loguru import logger

from distllm.dist import node_pb2
from distllm.errors.types import NodeUnreachableError

# Module-level cache for CUDA copy streams
_tensor_copy_streams: dict[str, torch.cuda.Stream] = {}


def cleanup_tensor_copy_streams() -> None:
    """Clean up all cached CUDA tensor copy streams without synchronizing.

    Synchronization is deferred to the serialization path that actually
    consumes the data, avoiding a blocking call here.
    """
    _tensor_copy_streams.clear()


def get_tensor_copy_stream(device: str = "cuda") -> torch.cuda.Stream | None:
    """Get or create a dedicated CUDA stream for tensor transfers."""
    if not torch.cuda.is_available():
        return None
    if device not in _tensor_copy_streams:
        _tensor_copy_streams[device] = torch.cuda.Stream(device=device)
    return _tensor_copy_streams[device]


def forward_request_to_proto(
    request: node_pb2.ForwardPassRequest,
    cluster_key: str | None = None,
) -> node_pb2.ForwardPassRequest:
    """Set additional metadata on a ForwardPassRequest before sending."""
    if cluster_key:
        request.cluster_key = cluster_key
    return request


def to_proto_tensor(tensor: torch.Tensor) -> node_pb2.TensorProto:
    """Convert torch tensor to protobuf TensorProto using a dedicated copy stream."""
    if tensor is None:
        return node_pb2.TensorProto(shape=[], dtype="none", raw_data=b"")
    t = tensor.detach()
    if t.is_cuda:
        copy_stream = get_tensor_copy_stream(t.device.index)
        with torch.cuda.stream(copy_stream):
            t = t.to("cpu", non_blocking=True)
    # Sync is deferred to the caller or serialization path.
    # The caller must ensure the copy stream is synchronized before
    # accessing the data (e.g. by calling copy_stream.synchronize()).
    dtype_str = str(t.dtype)
    if t.dim() == 0:
        t = t.reshape(1)
    raw = bytes(memoryview(t.contiguous().view(torch.uint8).numpy(force=True)))
    return node_pb2.TensorProto(shape=list(tensor.shape), dtype=dtype_str, raw_data=raw)


def from_proto_tensor(pb: node_pb2.TensorProto, device: str = "cpu") -> torch.Tensor:
    """Convert protobuf TensorProto back to torch tensor."""
    dtype_map = {
        "torch.float32": torch.float32,
        "torch.float16": torch.float16,
        "torch.bfloat16": torch.bfloat16,
        "torch.float8_e4m3fn": torch.float8_e4m3fn,
        "torch.int64": torch.int64,
        "torch.int32": torch.int32,
        "torch.uint8": torch.uint8,
        "torch.bool": torch.bool,
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float8_e4m3fn": torch.float8_e4m3fn,
        "int64": torch.int64,
        "int32": torch.int32,
        "bool": torch.bool,
    }
    tdtype = dtype_map.get(pb.dtype, torch.float32)
    if pb.raw_data:
        arr = np.frombuffer(pb.raw_data, dtype=np.uint8)
        # reshape([]) restores a 0-dim scalar; reshape([...]) the full shape.
        tensor = torch.from_numpy(arr).view(tdtype).reshape(list(pb.shape))
    elif pb.data:
        tensor = torch.tensor(list(pb.data), dtype=tdtype).reshape(list(pb.shape))
    else:
        # Genuinely empty tensor (no payload at all).
        tensor = torch.empty(0, dtype=tdtype)
    return tensor.to(device)


def process_forward_response_pb(
    response_pb: node_pb2.ForwardPassResponse,
    node_id: str,
    node,
    node_kv_caches: dict[str, list | None],
    resource_mgr,
) -> torch.Tensor:
    """Handle ForwardPass response from remote node."""
    if not response_pb.success:
        resource_mgr.record_failure(node_id)
        node.healthy = False
        logger.error(f"Pipeline node={node_id} error: {response_pb.error_message}")
        raise NodeUnreachableError(
            node_id=node_id,
            host=node.host,
            port=node.port,
            original_error=RuntimeError(response_pb.error_message),
        )
    resource_mgr.record_success(node_id)

    current_hidden = from_proto_tensor(response_pb.output)
    if response_pb.kv_cache and response_pb.kv_cache.layers:
        cache = []
        for layer_pb in response_pb.kv_cache.layers:
            k = from_proto_tensor(layer_pb.key_states)
            v = from_proto_tensor(layer_pb.value_states)
            cache.append((k, v))
        node_kv_caches[node_id] = cache
    return current_hidden


def set_kv_cache_proto(
    cache_pb: node_pb2.KVCacheProto,
    delta_kv: list[tuple[torch.Tensor, torch.Tensor]],
    compress: bool = False,
    compress_bits: int = 8,
) -> None:
    """Populate a protobuf KVCacheProto from a list of (key, value) tensor pairs."""
    for k, v in delta_kv:
        layer = cache_pb.layers.add()
        if compress and compress_bits == 8:
            k_scale = k.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / 127.0
            v_scale = v.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / 127.0
            k_compressed = (k / k_scale).round().clamp(-128, 127).to(torch.int8)
            v_compressed = (v / v_scale).round().clamp(-128, 127).to(torch.int8)
            layer.key_states.CopyFrom(to_proto_tensor(k_compressed))
            layer.value_states.CopyFrom(to_proto_tensor(v_compressed))
            layer.key_scale.CopyFrom(to_proto_tensor(k_scale.squeeze(-1)))
            layer.value_scale.CopyFrom(to_proto_tensor(v_scale.squeeze(-1)))
        elif compress and compress_bits == 4:
            k_scale = k.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / 7.0
            v_scale = v.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / 7.0
            k_int4 = (k / k_scale).round().clamp(-7, 7).to(torch.int8)
            v_int4 = (v / v_scale).round().clamp(-7, 7).to(torch.int8)
            layer.key_states.CopyFrom(to_proto_tensor(k_int4))
            layer.value_states.CopyFrom(to_proto_tensor(v_int4))
            layer.key_scale.CopyFrom(to_proto_tensor(k_scale.squeeze(-1)))
            layer.value_scale.CopyFrom(to_proto_tensor(v_scale.squeeze(-1)))
        else:
            layer.key_states.CopyFrom(to_proto_tensor(k))
            layer.value_states.CopyFrom(to_proto_tensor(v))


def tensor_quantize(tensor, enabled=True, bits=8, use_fp8=False):
    """Quantize a tensor for transfer."""
    if not enabled:
        return tensor, None
    if use_fp8 and tensor.dtype == torch.float16 and hasattr(torch, "float8_e4m3fn"):
        return tensor.to(torch.float8_e4m3fn), None
    if bits == 8:
        scale = tensor.abs().max().clamp(min=1e-5) / 127.0
        return (tensor / scale).round().clamp(-128, 127).to(torch.int8), scale
    return tensor, None


def tensor_dequantize(quantized, scale, orig_dtype, use_fp8=False):
    """Dequantize a tensor after transfer."""
    if use_fp8 and hasattr(quantized, "dtype") and quantized.dtype == torch.float8_e4m3fn:
        return quantized.to(orig_dtype)
    if scale is None:
        return quantized.to(orig_dtype) if quantized.dtype != orig_dtype else quantized
    return (quantized.to(orig_dtype) * scale).to(orig_dtype)
