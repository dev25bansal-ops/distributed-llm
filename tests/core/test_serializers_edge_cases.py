"""Proto serialization edge case tests (legacy — serializers module removed).

Run: pytest tests/core/test_serializers_edge_cases.py -v
"""

import struct

import pytest
import torch

try:
    from distllm.communication.node_pb2 import Tensor
    from distllm.communication.serializers import (
        kv_cache_to_proto,
        proto_to_kv_cache,
        proto_to_tensor,
        tensor_to_proto,
    )
except ImportError:
    pytest.skip("distllm.communication.serializers module removed", allow_module_level=True)

from distllm.core.kv_cache import KVCache

# ============================================================
# tensor_to_proto Tests
# ============================================================


class TestTensorToProto:
    """Tests for tensor -> proto conversion."""

    def test_none_tensor(self):
        """None tensor should return empty Tensor."""
        proto = tensor_to_proto(None)
        assert proto.shape == []
        assert proto.dtype == "none"

    def test_empty_tensor(self):
        """Empty tensor should serialize with shape [0]."""
        tensor = torch.empty(0)
        proto = tensor_to_proto(tensor)
        assert list(proto.shape) == [0]
        assert len(proto.raw_data) == 0

    def test_1d_tensor(self):
        """1D tensor should serialize correctly."""
        tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        proto = tensor_to_proto(tensor)
        assert list(proto.shape) == [3]
        assert proto.dtype == "torch.float32"
        assert len(proto.raw_data) == 12  # 3 * 4 bytes

    def test_2d_tensor(self):
        """2D tensor should serialize correctly."""
        tensor = torch.randn(3, 4, dtype=torch.float32)
        proto = tensor_to_proto(tensor)
        assert list(proto.shape) == [3, 4]
        assert len(proto.raw_data) == 48  # 12 * 4 bytes

    def test_3d_tensor(self):
        """3D tensor (e.g., batch x heads x seq x head_dim) should serialize."""
        tensor = torch.randn(2, 4, 8, dtype=torch.float32)
        proto = tensor_to_proto(tensor)
        assert list(proto.shape) == [2, 4, 8]
        assert len(proto.raw_data) == 256  # 64 * 4 bytes

    def test_int64_tensor(self):
        """int64 tensor should use correct dtype string."""
        tensor = torch.tensor([1, 2, 3], dtype=torch.int64)
        proto = tensor_to_proto(tensor)
        assert proto.dtype == "torch.int64"
        assert len(proto.raw_data) == 24  # 3 * 8 bytes

    def test_bfloat16_tensor(self):
        """bfloat16 tensor should serialize (raw bytes, not numpy conversion)."""
        tensor = torch.randn(4, dtype=torch.bfloat16)
        proto = tensor_to_proto(tensor)
        assert proto.dtype == "torch.bfloat16"
        assert len(proto.raw_data) == 8  # 4 * 2 bytes

    def test_bool_tensor(self):
        """bool tensor should serialize correctly."""
        tensor = torch.tensor([True, False, True], dtype=torch.bool)
        proto = tensor_to_proto(tensor)
        assert proto.dtype == "torch.bool"
        assert len(proto.raw_data) == 3  # 3 * 1 byte

    def test_large_tensor(self):
        """Large tensor should serialize within 64MB limit."""
        # 10MB tensor (2.5M float32 elements)
        tensor = torch.randn(2_500_000, dtype=torch.float32)
        proto = tensor_to_proto(tensor)
        assert len(proto.raw_data) == 10_000_000  # 2.5M * 4 bytes

    def test_detaches_tensor(self):
        """Proto should contain detached tensor data (no gradients)."""
        tensor = torch.randn(3, requires_grad=True)
        proto = tensor_to_proto(tensor)
        # Should not raise - tensor was detached
        assert list(proto.shape) == [3]

    def test_moves_to_cpu(self):
        """Tensor should be moved to CPU before serialization."""
        tensor = torch.randn(3, device="cpu")
        proto = tensor_to_proto(tensor)
        assert list(proto.shape) == [3]


# ============================================================
# proto_to_tensor Tests
# ============================================================


class TestProtoToTensor:
    """Tests for proto -> tensor conversion."""

    def test_empty_proto(self):
        """Proto with no shape should return empty tensor."""
        proto = Tensor()
        tensor = proto_to_tensor(proto)
        assert tensor.shape == torch.Size([0])

    def test_roundtrip_float32(self):
        """Float32 tensor roundtrip should preserve values."""
        original = torch.tensor([1.5, 2.5, 3.5], dtype=torch.float32)
        proto = tensor_to_proto(original)
        result = proto_to_tensor(proto)
        assert torch.allclose(original, result)

    def test_roundtrip_int64(self):
        """Int64 tensor roundtrip should preserve values."""
        original = torch.tensor([100, 200, 300], dtype=torch.int64)
        proto = tensor_to_proto(original)
        result = proto_to_tensor(proto)
        assert torch.equal(original, result)

    def test_roundtrip_2d(self):
        """2D tensor roundtrip should preserve shape and values."""
        original = torch.randn(3, 4, dtype=torch.float32)
        proto = tensor_to_proto(original)
        result = proto_to_tensor(proto)
        assert original.shape == result.shape
        assert torch.allclose(original, result)

    def test_roundtrip_bfloat16(self):
        """bfloat16 roundtrip should preserve values."""
        original = torch.randn(5, dtype=torch.bfloat16)
        proto = tensor_to_proto(original)
        result = proto_to_tensor(proto)
        assert original.shape == result.shape
        assert original.dtype == result.dtype

    def test_unknown_dtype_defaults_float32(self):
        """Unknown dtype should default to float32."""
        proto = Tensor()
        proto.shape.extend([2, 2])
        proto.dtype = "torch.unknown_dtype"
        # 4 elements * 4 bytes = 16 bytes of float32 data
        proto.raw_data = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)

        result = proto_to_tensor(proto)

        assert result.shape == (2, 2)
        assert result.dtype == torch.float32

    def test_legacy_float_list_fallback(self):
        """Should fall back to legacy float list when raw_data is empty."""
        proto = Tensor()
        proto.shape.extend([3])
        proto.dtype = "float32"
        proto.data.extend([1.0, 2.0, 3.0])
        proto.raw_data = b""  # No raw data

        result = proto_to_tensor(proto)

        assert result.shape == (3,)
        assert torch.allclose(result, torch.tensor([1.0, 2.0, 3.0]))

    def test_byte_length_mismatch_raises(self):
        """Byte length mismatch should raise SerializationError."""
        from distllm.errors.types import SerializationError
        proto = Tensor()
        proto.shape.extend([3])  # 3 elements
        proto.dtype = "torch.float32"  # 4 bytes each = 12 bytes expected
        proto.raw_data = b"\x00" * 6  # Only 6 bytes

        with pytest.raises(SerializationError, match="data length mismatch"):
            proto_to_tensor(proto)

    def test_device_parameter(self):
        """Result should be on specified device."""
        original = torch.tensor([1.0, 2.0], dtype=torch.float32)
        proto = tensor_to_proto(original)
        result = proto_to_tensor(proto, device="cpu")
        assert result.device.type == "cpu"

    def test_empty_shape_with_data(self):
        """Empty shape with raw_data should return a 0-dim (scalar) tensor."""
        proto = Tensor()
        proto.shape.extend([])
        proto.raw_data = b"\x00" * 4

        result = proto_to_tensor(proto)
        # Empty shape with data is treated as scalar
        assert result.shape == torch.Size([])
        assert result.numel() == 1


# ============================================================
# KV Cache Serialization Tests
# ============================================================


class TestKVCacheToProto:
    """Tests for KV cache -> proto conversion."""

    def test_empty_cache(self):
        """Empty KVCache should serialize to proto with no layers."""
        cache = KVCache()
        proto = kv_cache_to_proto(cache)
        assert len(proto.layers) == 0

    def test_cache_with_data(self, sample_kv_cache):
        """KVCache with data should serialize correctly."""
        sample_kv_cache.update(0, torch.randn(1, 4, 3, 8), torch.randn(1, 4, 3, 8))

        proto = kv_cache_to_proto(sample_kv_cache)

        assert len(proto.layers) == 2
        assert proto.layers[0].key_states.shape == [1, 4, 3, 8]

    def test_multi_layer_cache(self):
        """Multi-layer cache should serialize all layers."""
        cache = KVCache()
        cache.init_cache(num_layers=4, batch_size=1, num_heads=2, head_dim=4, device="cpu")

        for i in range(4):
            cache.update(i, torch.randn(1, 2, 5, 4), torch.randn(1, 2, 5, 4))

        proto = kv_cache_to_proto(cache)

        assert len(proto.layers) == 4


class TestProtoToKVCache:
    """Tests for proto -> KV cache conversion."""

    def test_roundtrip_empty_cache(self):
        """Empty cache roundtrip should preserve emptiness."""
        cache = KVCache()
        proto = kv_cache_to_proto(cache)
        new_cache = proto_to_kv_cache(proto)

        assert len(new_cache.cache) == 0

    def test_roundtrip_with_data(self):
        """Cache with data roundtrip should preserve structure."""
        cache = KVCache()
        cache.init_cache(num_layers=2, batch_size=1, num_heads=2, head_dim=4, device="cpu")
        cache.update(0, torch.randn(1, 2, 3, 4), torch.randn(1, 2, 3, 4))
        cache.update(1, torch.randn(1, 2, 3, 4), torch.randn(1, 2, 3, 4))

        proto = kv_cache_to_proto(cache)
        new_cache = proto_to_kv_cache(proto)

        assert len(new_cache.cache) == 2
        assert new_cache.cache[0][0].shape == torch.Size([1, 2, 3, 4])


# ============================================================
# Edge Cases and Boundary Tests
# ============================================================


class TestEdgeCases:
    """Edge case tests for serialization."""

    def test_single_element_tensor(self):
        """Single element tensor should serialize/deserialize."""
        tensor = torch.tensor([42.0], dtype=torch.float32)
        proto = tensor_to_proto(tensor)
        result = proto_to_tensor(proto)
        assert result.item() == pytest.approx(42.0, abs=1e-5)

    def test_zero_dim_tensor(self):
        """Scalar tensor (0-dim) should serialize."""
        tensor = torch.tensor(3.14, dtype=torch.float32)
        proto = tensor_to_proto(tensor)
        assert list(proto.shape) == []

    def test_negative_values(self):
        """Negative values should roundtrip correctly."""
        tensor = torch.tensor([-1.0, -2.0, -3.0], dtype=torch.float32)
        proto = tensor_to_proto(tensor)
        result = proto_to_tensor(proto)
        assert torch.allclose(tensor, result)

    def test_very_large_values(self):
        """Very large float values should roundtrip."""
        tensor = torch.tensor([1e30, -1e30], dtype=torch.float32)
        proto = tensor_to_proto(tensor)
        result = proto_to_tensor(proto)
        assert torch.allclose(tensor, result, rtol=1e-5)

    def test_zero_tensor(self):
        """All-zeros tensor should roundtrip."""
        tensor = torch.zeros(100, dtype=torch.float32)
        proto = tensor_to_proto(tensor)
        result = proto_to_tensor(proto)
        assert torch.equal(tensor, result)

    def test_uint8_tensor(self):
        """uint8 tensor should serialize correctly."""
        tensor = torch.tensor([0, 127, 255], dtype=torch.uint8)
        proto = tensor_to_proto(tensor)
        assert proto.dtype == "torch.uint8"
        result = proto_to_tensor(proto)
        assert torch.equal(tensor, result)

    def test_int8_tensor(self):
        """int8 tensor should serialize correctly."""
        tensor = torch.tensor([-128, 0, 127], dtype=torch.int8)
        proto = tensor_to_proto(tensor)
        assert proto.dtype == "torch.int8"
        result = proto_to_tensor(proto)
        assert torch.equal(tensor, result)


class TestOversizedPayloads:
    """Tests for oversized payload handling."""

    def test_near_64mb_tensor(self):
        """Tensor near 64MB limit should serialize."""
        # 16M float32 elements = 64MB
        tensor = torch.randn(16_000_000, dtype=torch.float32)
        proto = tensor_to_proto(tensor)
        assert len(proto.raw_data) == 64_000_000

        result = proto_to_tensor(proto)
        assert result.shape == tensor.shape

    def test_multi_dimensional_large_tensor(self):
        """Large multi-dimensional tensor should roundtrip."""
        # ~16MB
        tensor = torch.randn(2000, 2000, dtype=torch.float32)
        proto = tensor_to_proto(tensor)
        result = proto_to_tensor(proto)
        assert result.shape == tensor.shape
