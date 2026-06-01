"""Tensor serialization/deserialization roundtrip tests.

Covers KV cache serialization, proto converter functions, tensor quantization,
and property-based roundtrip invariants.
"""

import asyncio
import socket
import struct
import threading
import time
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import numpy as np

try:
    from hypothesis import given, strategies as st, settings as hp_settings
    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False


from tests.comprehensive.conftest import _load_module

# Load clean modules
_kv_cache = _load_module("distllm/core/kv_cache.py")


# ═══════════════════════════════════════════════════════════════════════════
# 4. Tensor Serialization / Deserialization Roundtrip
# ═══════════════════════════════════════════════════════════════════════════

class TestKVCacheSerialization:
    """Roundtrip: serialize → deserialize → verify equality."""

    def test_serialize_deserialize_roundtrip(self):
        c = _kv_cache.KVCache()
        c.init_cache(2, 1, 4, 32, "cpu")
        k1 = torch.randn(1, 4, 3, 32)
        v1 = torch.randn(1, 4, 3, 32)
        c.update(0, k1, v1)
        c.update(1, torch.randn(1, 4, 5, 32), torch.randn(1, 4, 5, 32))
        data = _kv_cache.serialize_kv_cache(c)
        c2 = _kv_cache.deserialize_kv_cache(data)
        assert c2.num_layers == 2
        assert torch.equal(c2.cache[0][0], c.cache[0][0])
        assert torch.equal(c2.cache[0][1], c.cache[0][1])

    def test_serialize_empty_cache(self):
        data = _kv_cache.serialize_kv_cache(_kv_cache.KVCache())
        assert data == {"layers": []}
        c2 = _kv_cache.deserialize_kv_cache(data)
        assert c2.num_layers == 0

    def test_tensor_to_bytes_roundtrip(self):
        t = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        data, shape, dtype = _kv_cache._tensor_to_bytes(t)
        t2 = _kv_cache._bytes_to_tensor(data, shape, dtype, "cpu")
        assert torch.equal(t, t2)

    def test_tensor_to_bytes_bf16(self):
        t = torch.randn(2, 3, dtype=torch.bfloat16)
        data, shape, dtype = _kv_cache._tensor_to_bytes(t)
        t2 = _kv_cache._bytes_to_tensor(data, shape, dtype, "cpu")
        assert t2.dtype == torch.bfloat16
        assert torch.equal(t, t2)

    def test_tensor_to_bytes_int32(self):
        t = torch.tensor([1, 2, 3], dtype=torch.int32)
        data, shape, dtype = _kv_cache._tensor_to_bytes(t)
        t2 = _kv_cache._bytes_to_tensor(data, shape, dtype, "cpu")
        assert torch.equal(t, t2)

    def test_tensor_to_bytes_bool(self):
        t = torch.tensor([True, False, True], dtype=torch.bool)
        data, shape, dtype = _kv_cache._tensor_to_bytes(t)
        t2 = _kv_cache._bytes_to_tensor(data, shape, dtype, "cpu")
        assert torch.equal(t, t2)

    def test_save_load_disk_roundtrip(self, tmp_path):
        c = _kv_cache.KVCache()
        c.init_cache(1, 1, 2, 8, "cpu")
        c.update(0, torch.randn(1, 2, 4, 8), torch.randn(1, 2, 4, 8))
        path = str(tmp_path / "kv.pt")
        _kv_cache.save_kv_cache_to_disk(c, path)
        assert Path(path).exists()
        c2 = _kv_cache.load_kv_cache_from_disk(path)
        assert c2.num_layers == 1
        assert torch.equal(c2.cache[0][0], c.cache[0][0])

    @pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
    @hp_settings(max_examples=20, suppress_health_check=tuple())
    @given(
        s=st.integers(min_value=1, max_value=4),
        d=st.integers(min_value=4, max_value=8),
    )
    def test_serialize_roundtrip_property(self, s, d):
        c = _kv_cache.KVCache()
        c.init_cache(1, 1, 2, d, "cpu")
        k = torch.randn(1, 2, s, d)
        v = torch.randn(1, 2, s, d)
        c.update(0, k, v)
        data = _kv_cache.serialize_kv_cache(c)
        c2 = _kv_cache.deserialize_kv_cache(data)
        assert c2.num_layers == 1
        assert torch.equal(c2.cache[0][0], c.cache[0][0])
        assert torch.equal(c2.cache[0][1], c.cache[0][1])


# ── Proto converter tests (replicated from pipeline.py module-level funcs) ──

def _to_tensor_proto(tensor):
    """Replicates distllm.dist.pipeline._to_tensor_proto for testing."""
    if tensor is None:
        return _ProtoTensor(data=[], shape=[], dtype="none")
    t = tensor.detach()
    if t.is_cuda:
        t = t.to('cpu')
    dtype_str = str(t.dtype)
    if t.dim() == 0:
        t = t.reshape(1)
    raw = bytes(memoryview(t.contiguous().view(torch.uint8).numpy(force=True)))
    return _ProtoTensor(raw_data=raw, shape=list(tensor.shape), dtype=dtype_str)


def _from_tensor_proto(proto, device="cpu"):
    """Replicates distllm.dist.pipeline._from_tensor_proto for testing."""
    if not proto.shape:
        return torch.empty(0, device=device)
    dtype_map = {"torch.float32": torch.float32, "torch.float16": torch.float16,
                 "torch.bfloat16": torch.bfloat16, "torch.int64": torch.int64,
                 "torch.int32": torch.int32, "torch.uint8": torch.uint8,
                 "torch.bool": torch.bool, "float32": torch.float32,
                 "float16": torch.float16, "bfloat16": torch.bfloat16,
                 "int64": torch.int64, "int32": torch.int32, "bool": torch.bool}
    tdtype = dtype_map.get(proto.dtype, torch.float32)
    if proto.raw_data:
        arr = np.frombuffer(proto.raw_data, dtype=np.uint8)
        tensor = torch.from_numpy(arr).view(tdtype).reshape(list(proto.shape)).clone()
    else:
        tensor = torch.tensor(proto.data, dtype=torch.float32).reshape(list(proto.shape))
    return tensor.to(device)


def _tensor_quantize(tensor):
    """Replicates distllm.dist.pipeline._tensor_quantize for testing."""
    scale = tensor.abs().max().clamp(min=1e-5) / 127.0
    return (tensor / scale).round().clamp(-128, 127).to(torch.int8), scale


def _tensor_dequantize(quantized, scale, orig_dtype):
    if scale is None:
        return quantized.to(orig_dtype) if quantized.dtype != orig_dtype else quantized
    return (quantized.to(orig_dtype) * scale).to(orig_dtype)


class _ProtoTensor:
    """Minimal stand-in for a protobuf TensorProto."""
    def __init__(self, raw_data=b"", shape=None, dtype="torch.float32", data=None, scale=None):
        self.raw_data = raw_data
        self.shape = shape or []
        self.dtype = dtype
        self.data = data or []
        self.scale = scale or []


class TestProtoConverterFunctions:
    """Tests for isolated proto-converter logic (matching pipeline.py)."""

    def test_to_from_proto_roundtrip(self):
        t = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        proto = _to_tensor_proto(t)
        t2 = _from_tensor_proto(proto)
        assert torch.equal(t, t2)

    def test_proto_roundtrip_bf16(self):
        t = torch.randn(2, 4, dtype=torch.bfloat16)
        proto = _to_tensor_proto(t)
        t2 = _from_tensor_proto(proto)
        assert t2.dtype == torch.bfloat16
        assert torch.equal(t, t2)

    def test_proto_roundtrip_int64(self):
        t = torch.tensor([1, 2, 3], dtype=torch.int64)
        proto = _to_tensor_proto(t)
        t2 = _from_tensor_proto(proto)
        assert torch.equal(t, t2)

    def test_proto_none_returns_empty(self):
        proto = _to_tensor_proto(None)
        assert proto.shape == []
        assert proto.dtype == "none"

    def test_proto_empty_input(self):
        t = _from_tensor_proto(_ProtoTensor(shape=[]))
        assert t.shape == (0,)

    def test_quantize_dequantize_roundtrip(self):
        t = torch.randn(2, 4, dtype=torch.float32) * 10
        q, scale = _tensor_quantize(t)
        assert q.dtype == torch.int8
        t2 = _tensor_dequantize(q, scale, torch.float32)
        assert t2.shape == t.shape
        diff = (t - t2).abs().max().item()
        assert diff < 1.0

    def test_dequantize_noop_when_scale_none(self):
        t = torch.randn(2, 4, dtype=torch.float32)
        result = _tensor_dequantize(t, None, torch.float32)
        assert torch.equal(result, t)

    def test_quantize_preserves_layout(self):
        t = torch.randn(3, 5, dtype=torch.float32)
        q, scale = _tensor_quantize(t)
        assert q.shape == t.shape

    def test_proto_dtype_mapping_float16(self):
        t = torch.randn(2, 2, dtype=torch.float16)
        proto = _to_tensor_proto(t)
        t2 = _from_tensor_proto(proto)
        assert t2.dtype == torch.float16

    def test_proto_dtype_mapping_int32(self):
        t = torch.tensor([10, 20], dtype=torch.int32)
        proto = _to_tensor_proto(t)
        t2 = _from_tensor_proto(proto)
        assert t2.dtype == torch.int32

    def test_proto_dtype_mapping_bool(self):
        t = torch.tensor([True, False], dtype=torch.bool)
        proto = _to_tensor_proto(t)
        t2 = _from_tensor_proto(proto)
        assert t2.dtype == torch.bool

    def test_proto_empty_data_fallback(self):
        proto = _ProtoTensor(shape=[2, 2], dtype="float32", data=[1.0, 2.0, 3.0, 4.0])
        t = _from_tensor_proto(proto)
        assert t.shape == (2, 2)

    def test_proto_unknown_dtype_falls_to_float32(self):
        proto = _ProtoTensor(raw_data=b"\x00\x00\x80?", shape=[1], dtype="unknown_dtype")
        t = _from_tensor_proto(proto)
        assert t.dtype == torch.float32
        assert t[0].item() == 1.0

    def test_kv_cache_proto_conversion(self):
        c = _kv_cache.KVCache()
        c.init_cache(1, 1, 2, 8, "cpu")
        k1 = torch.randn(1, 2, 3, 8)
        v1 = torch.randn(1, 2, 3, 8)
        c.update(0, k1, v1)
        proto = _ProtoKVCache()
        for k, v in c.cache:
            proto.layers.append(
                _ProtoKVLayer(key_states=_to_tensor_proto(k),
                               value_states=_to_tensor_proto(v))
            )
        assert len(proto.layers) == 1
        k_restored = _from_tensor_proto(proto.layers[0].key_states)
        assert torch.equal(k_restored, k1)

    @pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
    @hp_settings(max_examples=50)
    @given(
        b=st.integers(min_value=1, max_value=2),
        h=st.integers(min_value=1, max_value=4),
        s=st.integers(min_value=1, max_value=8),
        d=st.integers(min_value=4, max_value=16),
    )
    def test_proto_roundtrip_property(self, b, h, s, d):
        t = torch.randn(b, h, s, d, dtype=torch.float32)
        proto = _to_tensor_proto(t)
        t2 = _from_tensor_proto(proto)
        assert torch.equal(t, t2)

    @pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
    @hp_settings(max_examples=50)
    @given(
        b=st.integers(min_value=1, max_value=2),
        h=st.integers(min_value=1, max_value=4),
        s=st.integers(min_value=1, max_value=8),
        d=st.integers(min_value=4, max_value=16),
    )
    def test_quantize_dequantize_property(self, b, h, s, d):
        t = torch.randn(b, h, s, d) * 5
        q, scale = _tensor_quantize(t)
        t2 = _tensor_dequantize(q, scale, torch.float32)
        assert q.dtype == torch.int8
        assert t2.shape == t.shape
        mse = ((t - t2) ** 2).mean().item()
        assert mse < 2.0


class _ProtoKVLayer:
    def __init__(self, key_states=None, value_states=None):
        self.key_states = key_states
        self.value_states = value_states


class _ProtoKVCache:
    def __init__(self):
        self.layers = []
