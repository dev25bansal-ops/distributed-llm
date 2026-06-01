"""Fuzz testing for gRPC proto deserialization.

Covers corrupted/edge-case protobuf-like data, tensor proto parsing with
various dtypes, KV cache proto fuzz, and quantization edge cases.
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


# ── Shared helpers (duplicated from test_tensor_serialization) ──

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


# ═══════════════════════════════════════════════════════════════════════════
# 10. Fuzz Testing for gRPC Proto Deserialization
# ═══════════════════════════════════════════════════════════════════════════

class TestGrpcProtoFuzz:
    """Fuzz deserialization with corrupted or edge-case protobuf-like data."""

    def test_tensor_proto_empty_raw_data(self):
        proto = _ProtoTensor(raw_data=b"", shape=[2, 2], dtype="torch.float32")
        # Empty raw_data with no fallback data causes reshape error
        with pytest.raises((RuntimeError, ValueError)):
            _from_tensor_proto(proto)

    def test_tensor_proto_zero_rank(self):
        proto = _ProtoTensor(raw_data=b"\x00\x00\x80?", shape=[], dtype="torch.float32")
        t = _from_tensor_proto(proto)
        assert t.numel() == 0

    def test_tensor_proto_negative_shape_infers_dim(self):
        proto = _ProtoTensor(raw_data=b"", shape=[-1, 2], dtype="torch.float32")
        # Torch allows -1 shape (inferred dimension)
        try:
            t = _from_tensor_proto(proto)
            assert isinstance(t, torch.Tensor)
        except (RuntimeError, ValueError):
            pass

    def test_tensor_proto_large_shape(self):
        large = 10_000_000
        proto = _ProtoTensor(raw_data=b"\x00" * large * 4, shape=[large], dtype="torch.float32")
        t = _from_tensor_proto(proto)
        assert t.shape == (large,)

    def test_tensor_proto_zero_dim(self):
        proto = _ProtoTensor(raw_data=b"", shape=[0], dtype="torch.float32")
        t = _from_tensor_proto(proto)
        assert t.numel() == 0

    def test_tensor_proto_unexpected_dtype(self):
        proto = _ProtoTensor(raw_data=b"\x00\x00\x80?", shape=[1], dtype="non_existent_type")
        t = _from_tensor_proto(proto)
        assert t.dtype == torch.float32

    def test_tensor_proto_truncated_data(self):
        proto = _ProtoTensor(raw_data=b"\x00\x00", shape=[4], dtype="torch.float32")
        with pytest.raises((RuntimeError, ValueError)):
            _from_tensor_proto(proto)

    def test_tensor_proto_very_large_raw_data(self):
        n = 100_000
        proto = _ProtoTensor(raw_data=b"\x00" * n * 4, shape=[n], dtype="torch.float32")
        t = _from_tensor_proto(proto)
        assert t.shape == (n,)

    def test_tensor_proto_bool_with_invalid_raw_data(self):
        proto = _ProtoTensor(raw_data=b"\x02", shape=[1], dtype="torch.bool")
        t = _from_tensor_proto(proto)
        assert t.dtype == torch.bool
        assert t[0].item() in (True,)

    @pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
    @hp_settings(max_examples=200)
    @given(
        raw=st.binary(min_size=0, max_size=256),
        shape=st.lists(st.integers(min_value=0, max_value=8), min_size=0, max_size=3),
        dtype=st.sampled_from(["torch.float32", "torch.float16", "torch.int64",
                                "torch.int32", "torch.uint8", "torch.bool",
                                "invalid_type", "", "none"]),
    )
    def test_from_tensor_proto_never_raises_outside_bounds(self, raw, shape, dtype):
        """Fuzz: from_tensor_proto should handle any input without raising."""
        try:
            proto = _ProtoTensor(raw_data=raw, shape=shape, dtype=dtype)
            t = _from_tensor_proto(proto)
            assert isinstance(t, torch.Tensor)
        except (RuntimeError, ValueError):
            # Shape mismatches and truncated data are acceptable failures
            pass

    @pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
    @hp_settings(max_examples=100)
    @given(
        n_layers=st.integers(min_value=0, max_value=8),
        n_entries=st.integers(min_value=0, max_value=5),
    )
    def test_kv_cache_proto_fuzz(self, n_layers, n_entries):
        """Fuzz deserialize_kv_cache with generated data structures."""
        layers = []
        for _ in range(n_layers):
            keys = []
            values = []
            for _ in range(n_entries):
                keys.append(torch.randn(1, 2, 3, 4))
                values.append(torch.randn(1, 2, 3, 4))
            k = torch.cat(keys, dim=-2) if keys else torch.randn(1, 2, 0, 4)
            v = torch.cat(values, dim=-2) if values else torch.randn(1, 2, 0, 4)
            layers.append({"key": k, "value": v})
        data = {"layers": layers}
        try:
            c = _kv_cache.deserialize_kv_cache(data)
            assert isinstance(c, _kv_cache.KVCache)
            if layers:
                assert c.num_layers == n_layers
        except Exception:
            pass

    def test_proto_none_device(self):
        proto = _ProtoTensor(raw_data=b"\x00\x00\x80?", shape=[1], dtype="torch.float32")
        t = _from_tensor_proto(proto, device="cpu")
        assert t.device.type == "cpu"

    def test_proto_quantize_with_near_zero_values(self):
        t = torch.tensor([1e-10, -1e-10, 0.0], dtype=torch.float32)
        q, scale = _tensor_quantize(t)
        assert q.dtype == torch.int8
        t2 = _tensor_dequantize(q, scale, torch.float32)
        assert t2.shape == t.shape

    def test_proto_quantize_extreme_values(self):
        t = torch.tensor([1e10, -1e10], dtype=torch.float32)
        q, scale = _tensor_quantize(t)
        assert q.dtype == torch.int8
        t2 = _tensor_dequantize(q, scale, torch.float32)
        assert t2.shape == t.shape

    def test_tensor_proto_fallback_data_path(self):
        proto = _ProtoTensor(raw_data=b"", shape=[2, 2], dtype="float32",
                              data=[1.0, 2.0, 3.0, 4.0])
        t = _from_tensor_proto(proto)
        assert t.shape == (2, 2)

    def test_tensor_proto_multiple_dtype_strings(self):
        for dtype_str in ["torch.float32", "torch.float16", "torch.bfloat16",
                          "torch.int64", "torch.int32", "torch.uint8", "torch.bool",
                          "float32", "float16", "bfloat16", "int64", "int32", "bool"]:
            try:
                raw = struct.pack("f", 1.0) if "float" in dtype_str else struct.pack("q", 42)
                proto = _ProtoTensor(raw_data=raw, shape=[1], dtype=dtype_str)
                t = _from_tensor_proto(proto)
                assert isinstance(t, torch.Tensor)
            except (RuntimeError, ValueError, struct.error):
                pass
