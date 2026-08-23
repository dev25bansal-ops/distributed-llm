"""Tests for distllm.dist.pipeline.serialization — protobuf serialization helpers.

Zero mocks — all tests use real objects and deterministic logic.
"""

from __future__ import annotations

import pytest
import torch

from distllm.dist import node_pb2
from distllm.dist.pipeline.serialization import (
    cleanup_tensor_copy_streams,
    forward_request_to_proto,
    from_proto_tensor,
    get_tensor_copy_stream,
    process_forward_response_pb,
    set_kv_cache_proto,
    tensor_dequantize,
    tensor_quantize,
    to_proto_tensor,
)
from distllm.errors.types import NodeUnreachableError


# ---------------------------------------------------------------------------
# Helpers — real classes, not mocks
# ---------------------------------------------------------------------------


class _FakeResourceManager:
    """Minimal resource manager that records successes and failures."""

    def __init__(self) -> None:
        self.successes: list[str] = []
        self.failures: list[str] = []

    def record_success(self, node_id: str) -> None:
        self.successes.append(node_id)

    def record_failure(self, node_id: str) -> None:
        self.failures.append(node_id)


class _FakeNode:
    """Minimal node stub for process_forward_response_pb tests."""

    def __init__(
        self,
        healthy: bool = True,
        host: str = "127.0.0.1",
        port: int = 50051,
    ) -> None:
        self.healthy = healthy
        self.host = host
        self.port = port


# ---------------------------------------------------------------------------
# cleanup_tensor_copy_streams
# ---------------------------------------------------------------------------


class TestCleanupTensorCopyStreams:
    """cleanup_tensor_copy_streams is a simple cache-clear — all idempotent."""

    def test_cleanup_empty(self) -> None:
        cleanup_tensor_copy_streams()

    def test_cleanup_twice(self) -> None:
        cleanup_tensor_copy_streams()
        cleanup_tensor_copy_streams()

    def test_get_after_cleanup(self) -> None:
        cleanup_tensor_copy_streams()
        get_tensor_copy_stream(device="cuda")
        cleanup_tensor_copy_streams()


# ---------------------------------------------------------------------------
# get_tensor_copy_stream
# ---------------------------------------------------------------------------


class TestGetTensorCopyStream:
    """CUDA copy-stream cache helpers."""

    def test_same_device_returns_same_stream(self) -> None:
        cleanup_tensor_copy_streams()
        s1 = get_tensor_copy_stream(device="cuda")
        s2 = get_tensor_copy_stream(device="cuda")
        if torch.cuda.is_available():
            assert s1 is s2
        else:
            assert s1 is None and s2 is None

    def test_default_device_is_cuda(self) -> None:
        cleanup_tensor_copy_streams()
        s = get_tensor_copy_stream()
        if torch.cuda.is_available():
            assert s is not None
            assert s.device.type == "cuda"
        else:
            assert s is None


# ---------------------------------------------------------------------------
# forward_request_to_proto
# ---------------------------------------------------------------------------


class TestForwardRequestToProto:
    """ForwardPassRequest metadata helper."""

    def test_sets_cluster_key(self) -> None:
        req = node_pb2.ForwardPassRequest()
        result = forward_request_to_proto(req, cluster_key="test-cluster")
        assert result.cluster_key == "test-cluster"

    def test_none_cluster_key_is_noop(self) -> None:
        req = node_pb2.ForwardPassRequest()
        result = forward_request_to_proto(req, cluster_key=None)
        assert result.cluster_key == ""

    def test_returns_same_object(self) -> None:
        req = node_pb2.ForwardPassRequest()
        result = forward_request_to_proto(req, cluster_key="x")
        assert result is req

    def test_preserves_existing_fields(self) -> None:
        req = node_pb2.ForwardPassRequest(
            request_id="req-001",
            model_name="llama",
        )
        result = forward_request_to_proto(req, cluster_key="ck")
        assert result.request_id == "req-001"
        assert result.model_name == "llama"
        assert result.cluster_key == "ck"

    def test_overwrites_existing_cluster_key(self) -> None:
        req = node_pb2.ForwardPassRequest(cluster_key="old")
        result = forward_request_to_proto(req, cluster_key="new")
        assert result.cluster_key == "new"


# ---------------------------------------------------------------------------
# to_proto_tensor
# ---------------------------------------------------------------------------


class TestToProtoTensor:
    """Converting torch tensors to protobuf TensorProto."""

    def test_none_input(self) -> None:
        pb = to_proto_tensor(None)
        assert pb.shape == []
        assert pb.dtype == "none"
        assert pb.raw_data == b""

    def test_empty_tensor(self) -> None:
        t = torch.tensor([], dtype=torch.float32)
        pb = to_proto_tensor(t)
        assert list(pb.shape) == [0]
        assert pb.dtype == "torch.float32"

    def test_scalar_tensor(self) -> None:
        t = torch.tensor(3.14)
        pb = to_proto_tensor(t)
        # Scalar gets reshaped internally but original shape is stored
        assert list(pb.shape) == []
        assert pb.dtype == "torch.float32"
        # raw_data should contain the value (reshaped to 1 internally)
        assert len(pb.raw_data) > 0

    def test_1d_float32(self) -> None:
        t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        pb = to_proto_tensor(t)
        assert pb.dtype == "torch.float32"
        assert list(pb.shape) == [3]
        assert len(pb.raw_data) > 0

    def test_2d_float32(self) -> None:
        t = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        pb = to_proto_tensor(t)
        assert pb.dtype == "torch.float32"
        assert list(pb.shape) == [2, 2]

    def test_int64_tensor(self) -> None:
        t = torch.tensor([1, 2, 3], dtype=torch.int64)
        pb = to_proto_tensor(t)
        assert pb.dtype == "torch.int64"
        assert list(pb.shape) == [3]

    def test_int32_tensor(self) -> None:
        t = torch.tensor([1, 2, 3], dtype=torch.int32)
        pb = to_proto_tensor(t)
        assert pb.dtype == "torch.int32"

    def test_uint8_tensor(self) -> None:
        t = torch.tensor([200, 100], dtype=torch.uint8)
        pb = to_proto_tensor(t)
        assert pb.dtype == "torch.uint8"

    def test_bool_tensor(self) -> None:
        t = torch.tensor([True, False, True], dtype=torch.bool)
        pb = to_proto_tensor(t)
        assert pb.dtype == "torch.bool"
        assert list(pb.shape) == [3]

    def test_bfloat16_tensor(self) -> None:
        t = torch.tensor([1.5, 2.5], dtype=torch.bfloat16)
        pb = to_proto_tensor(t)
        assert pb.dtype == "torch.bfloat16"
        assert list(pb.shape) == [2]

    def test_half_tensor(self) -> None:
        t = torch.tensor([1.0, 2.0], dtype=torch.float16)
        pb = to_proto_tensor(t)
        assert pb.dtype == "torch.float16"

    @pytest.mark.skipif(
        not hasattr(torch, "float8_e4m3fn"),
        reason="torch.float8_e4m3fn not available",
    )
    def test_float8_tensor(self) -> None:
        t = torch.tensor([1.0, 2.0], dtype=torch.float8_e4m3fn)
        pb = to_proto_tensor(t)
        assert pb.dtype == "torch.float8_e4m3fn"

    def test_roundtrip_float32(self) -> None:
        t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        pb = to_proto_tensor(t)
        result = from_proto_tensor(pb)
        assert torch.equal(result, t)

    def test_roundtrip_int64(self) -> None:
        t = torch.tensor([10, 20, 30], dtype=torch.int64)
        pb = to_proto_tensor(t)
        result = from_proto_tensor(pb)
        assert torch.equal(result, t)

    def test_roundtrip_bool(self) -> None:
        t = torch.tensor([True, False], dtype=torch.bool)
        pb = to_proto_tensor(t)
        result = from_proto_tensor(pb)
        assert torch.equal(result, t)

    def test_roundtrip_2d(self) -> None:
        t = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        pb = to_proto_tensor(t)
        result = from_proto_tensor(pb)
        assert torch.allclose(result, t)

    def test_roundtrip_bfloat16(self) -> None:
        t = torch.tensor([1.5, 2.5], dtype=torch.bfloat16)
        pb = to_proto_tensor(t)
        result = from_proto_tensor(pb)
        assert result.dtype == torch.bfloat16
        assert torch.equal(result, t)

    def test_negative_values(self) -> None:
        t = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32)
        pb = to_proto_tensor(t)
        result = from_proto_tensor(pb)
        assert torch.allclose(result, t)

    def test_large_values(self) -> None:
        t = torch.tensor([1e10, -1e10], dtype=torch.float32)
        pb = to_proto_tensor(t)
        result = from_proto_tensor(pb)
        assert torch.allclose(result, t)


# ---------------------------------------------------------------------------
# from_proto_tensor
# ---------------------------------------------------------------------------


class TestFromProtoTensor:
    """Converting protobuf TensorProto back to torch tensors."""

    def test_empty_shape_returns_empty(self) -> None:
        pb = node_pb2.TensorProto(shape=[], dtype="float32", raw_data=b"")
        result = from_proto_tensor(pb)
        assert result.numel() == 0
        assert result.device.type == "cpu"

    def test_zero_length_tensor(self) -> None:
        pb = node_pb2.TensorProto(shape=[0], dtype="float32", raw_data=b"")
        result = from_proto_tensor(pb)
        assert result.numel() == 0
        assert list(result.shape) == [0]

    def test_unknown_dtype_defaults_to_float32(self) -> None:
        t = torch.tensor([1.0, 2.0])
        pb = to_proto_tensor(t)
        pb.dtype = "made_up_dtype"
        result = from_proto_tensor(pb)
        assert result.dtype == torch.float32

    def test_dtype_short_name(self) -> None:
        """from_proto_tensor accepts 'float32' without 'torch.' prefix."""
        t = torch.tensor([1.0, 2.0], dtype=torch.float32)
        pb = to_proto_tensor(t)
        pb.dtype = "float32"
        result = from_proto_tensor(pb)
        assert result.dtype == torch.float32
        assert torch.allclose(result, t)

    def test_dtype_short_name_int64(self) -> None:
        pb = node_pb2.TensorProto(shape=[2], dtype="int64")
        pb.data.extend([1, 2])
        result = from_proto_tensor(pb)
        assert result.dtype == torch.int64
        assert torch.equal(result, torch.tensor([1, 2], dtype=torch.int64))

    def test_dtype_short_name_bfloat16(self) -> None:
        pb = node_pb2.TensorProto(shape=[2], dtype="bfloat16")
        pb.data.extend([1.5, 2.5])
        result = from_proto_tensor(pb)
        assert result.dtype == torch.bfloat16

    def test_raw_data_fallback_to_data_field(self) -> None:
        """When raw_data is empty, from_proto_tensor reads the data field."""
        pb = node_pb2.TensorProto(shape=[3], dtype="float32", raw_data=b"")
        pb.data.extend([1.0, 2.0, 3.0])
        result = from_proto_tensor(pb)
        assert torch.allclose(result, torch.tensor([1.0, 2.0, 3.0]))

    def test_cpu_device_explicit(self) -> None:
        t = torch.tensor([1.0, 2.0])
        pb = to_proto_tensor(t)
        result = from_proto_tensor(pb, device="cpu")
        assert result.device.type == "cpu"

    def test_bool_from_proto(self) -> None:
        pb = node_pb2.TensorProto(shape=[2], dtype="bool")
        pb.data.extend([1.0, 0.0])
        result = from_proto_tensor(pb)
        assert result.dtype == torch.bool
        assert torch.equal(result, torch.tensor([True, False]))


# ---------------------------------------------------------------------------
# process_forward_response_pb
# ---------------------------------------------------------------------------


class TestProcessForwardResponsePb:
    """Handling ForwardPassResponse from remote nodes."""

    def test_successful_response(self) -> None:
        response = node_pb2.ForwardPassResponse()
        response.success = True
        response.output.CopyFrom(to_proto_tensor(torch.tensor([1.0, 2.0, 3.0])))

        node = _FakeNode()
        rm = _FakeResourceManager()
        kv_caches: dict[str, list | None] = {}

        result = process_forward_response_pb(response, "node-0", node, kv_caches, rm)

        assert isinstance(result, torch.Tensor)
        assert torch.allclose(result, torch.tensor([1.0, 2.0, 3.0]))
        assert "node-0" in rm.successes
        assert node.healthy  # unchanged

    def test_failed_response_raises_node_unreachable(self) -> None:
        response = node_pb2.ForwardPassResponse()
        response.success = False
        response.error_message = "GPU out of memory"

        node = _FakeNode(host="10.0.0.5", port=50052)
        rm = _FakeResourceManager()
        kv_caches: dict[str, list | None] = {}

        with pytest.raises(NodeUnreachableError) as exc:
            process_forward_response_pb(response, "node-1", node, kv_caches, rm)

        assert exc.value.node_id == "node-1"
        assert exc.value.host == "10.0.0.5"
        assert exc.value.port == 50052
        assert not node.healthy  # marked unhealthy
        assert "node-1" in rm.failures

    def test_with_kv_cache(self) -> None:
        response = node_pb2.ForwardPassResponse()
        response.success = True
        response.output.CopyFrom(to_proto_tensor(torch.tensor([1.0, 2.0])))

        k = torch.randn(2, 4)
        v = torch.randn(2, 4)
        set_kv_cache_proto(response.kv_cache, [(k, v)])

        node = _FakeNode()
        rm = _FakeResourceManager()
        kv_caches: dict[str, list | None] = {}

        result = process_forward_response_pb(response, "node-2", node, kv_caches, rm)

        assert isinstance(result, torch.Tensor)
        assert "node-2" in kv_caches
        assert len(kv_caches["node-2"]) == 1
        stored_k, stored_v = kv_caches["node-2"][0]
        assert torch.allclose(stored_k, k)
        assert torch.allclose(stored_v, v)

    def test_with_multi_layer_kv_cache(self) -> None:
        response = node_pb2.ForwardPassResponse()
        response.success = True
        response.output.CopyFrom(to_proto_tensor(torch.tensor([1.0])))

        k1, v1 = torch.randn(2, 4), torch.randn(2, 4)
        k2, v2 = torch.randn(2, 4), torch.randn(2, 4)
        set_kv_cache_proto(response.kv_cache, [(k1, v1), (k2, v2)])

        node = _FakeNode()
        rm = _FakeResourceManager()
        kv_caches: dict[str, list | None] = {}

        process_forward_response_pb(response, "node-3", node, kv_caches, rm)

        assert len(kv_caches["node-3"]) == 2
        assert torch.allclose(kv_caches["node-3"][0][0], k1)
        assert torch.allclose(kv_caches["node-3"][1][1], v2)

    def test_without_kv_cache(self) -> None:
        response = node_pb2.ForwardPassResponse()
        response.success = True
        response.output.CopyFrom(to_proto_tensor(torch.tensor([1.0])))

        node = _FakeNode()
        rm = _FakeResourceManager()
        kv_caches: dict[str, list | None] = {}

        process_forward_response_pb(response, "node-4", node, kv_caches, rm)

        assert "node-4" not in kv_caches

    def test_node_id_tracked_in_successes(self) -> None:
        response = node_pb2.ForwardPassResponse(success=True)
        response.output.CopyFrom(to_proto_tensor(torch.tensor([0.0])))

        rm = _FakeResourceManager()
        process_forward_response_pb(
            response, "alpha", _FakeNode(), {}, rm,
        )
        process_forward_response_pb(
            response, "beta", _FakeNode(), {}, rm,
        )

        assert rm.successes == ["alpha", "beta"]

    def test_failure_message_in_exception(self) -> None:
        response = node_pb2.ForwardPassResponse(
            success=False,
            error_message="connection reset by peer",
        )
        with pytest.raises(NodeUnreachableError) as exc:
            process_forward_response_pb(
                response,
                "node-5",
                _FakeNode(),
                {},
                _FakeResourceManager(),
            )
        # NodeUnreachableError has its own message format; the original
        # error_message is available in the original_error attribute.
        assert exc.value.node_id == "node-5"
        assert isinstance(exc.value.original_error, RuntimeError)
        assert "connection reset by peer" in str(exc.value.original_error)


# ---------------------------------------------------------------------------
# set_kv_cache_proto
# ---------------------------------------------------------------------------


class TestSetKVCacheProto:
    """Populating KVCacheProto from tensor pairs."""

    def test_no_compression(self) -> None:
        cache_pb = node_pb2.KVCacheProto()
        k = torch.randn(2, 4)
        v = torch.randn(2, 4)
        set_kv_cache_proto(cache_pb, [(k, v)], compress=False)

        assert len(cache_pb.layers) == 1
        layer = cache_pb.layers[0]
        assert list(layer.key_states.shape) == [2, 4]
        assert list(layer.value_states.shape) == [2, 4]
        # No scales when uncompressed
        assert list(layer.key_scale.shape) == []
        assert list(layer.value_scale.shape) == []

    def test_compression_8bit(self) -> None:
        cache_pb = node_pb2.KVCacheProto()
        k = torch.randn(2, 4)
        v = torch.randn(2, 4)
        set_kv_cache_proto(cache_pb, [(k, v)], compress=True, compress_bits=8)

        assert len(cache_pb.layers) == 1
        layer = cache_pb.layers[0]

        # Compressed states are stored as int8 protos
        assert layer.key_states.dtype in ("torch.int8",)
        assert layer.value_states.dtype in ("torch.int8",)
        assert list(layer.key_states.shape) == [2, 4]
        assert list(layer.value_states.shape) == [2, 4]
        assert len(layer.key_states.raw_data) > 0
        assert len(layer.value_states.raw_data) > 0

        # Scales should be present (squeezed along dim=-1)
        assert len(layer.key_scale.shape) > 0
        assert len(layer.value_scale.shape) > 0
        assert len(layer.key_scale.raw_data) > 0
        assert len(layer.value_scale.raw_data) > 0

    def test_compression_4bit(self) -> None:
        cache_pb = node_pb2.KVCacheProto()
        k = torch.randn(2, 4)
        v = torch.randn(2, 4)
        set_kv_cache_proto(cache_pb, [(k, v)], compress=True, compress_bits=4)

        assert len(cache_pb.layers) == 1
        layer = cache_pb.layers[0]

        # Compressed states stored as int8 protos (4-bit values held in int8)
        assert layer.key_states.dtype in ("torch.int8",)
        assert layer.value_states.dtype in ("torch.int8",)
        assert list(layer.key_states.shape) == [2, 4]
        assert list(layer.value_states.shape) == [2, 4]
        assert len(layer.key_states.raw_data) > 0
        assert len(layer.value_states.raw_data) > 0

        # Int4 scale dimension is also squeezed
        assert len(layer.key_scale.raw_data) > 0
        assert len(layer.value_scale.raw_data) > 0

    def test_4bit_values_clamped(self) -> None:
        cache_pb = node_pb2.KVCacheProto()
        k = torch.tensor([[-100.0, 100.0]], dtype=torch.float32)
        v = torch.tensor([[-50.0, 50.0]], dtype=torch.float32)
        set_kv_cache_proto(cache_pb, [(k, v)], compress=True, compress_bits=4)

        layer = cache_pb.layers[0]
        # dtype is int8, we just verify the proto metadata is set
        assert list(layer.key_states.shape) == [1, 2]
        assert list(layer.value_states.shape) == [1, 2]

    def test_multiple_layers(self) -> None:
        cache_pb = node_pb2.KVCacheProto()
        pairs = [(torch.randn(2, 4), torch.randn(2, 4)) for _ in range(3)]
        set_kv_cache_proto(cache_pb, pairs)

        assert len(cache_pb.layers) == 3

    def test_empty_kv_list(self) -> None:
        cache_pb = node_pb2.KVCacheProto()
        set_kv_cache_proto(cache_pb, [])
        assert len(cache_pb.layers) == 0

    def test_default_is_no_compression(self) -> None:
        cache_pb = node_pb2.KVCacheProto()
        k = torch.randn(1, 2)
        v = torch.randn(1, 2)
        set_kv_cache_proto(cache_pb, [(k, v)])  # compress defaults to False

        layer = cache_pb.layers[0]
        k_states = from_proto_tensor(layer.key_states)
        assert k_states.dtype == torch.float32
        assert torch.allclose(k_states, k)


# ---------------------------------------------------------------------------
# tensor_quantize
# ---------------------------------------------------------------------------


class TestTensorQuantize:
    """Quantizing tensors for transfer."""

    def test_disabled_returns_original(self) -> None:
        t = torch.randn(4)
        q, scale = tensor_quantize(t, enabled=False)
        assert q is t
        assert scale is None

    def test_enabled_8bit(self) -> None:
        t = torch.tensor([1.0, -2.0, 3.0], dtype=torch.float32)
        q, scale = tensor_quantize(t, enabled=True, bits=8)
        assert q.dtype == torch.int8
        assert scale is not None
        assert scale.numel() == 1

    def test_8bit_clamps_extremes(self) -> None:
        t = torch.tensor([1000.0, -1000.0], dtype=torch.float32)
        q, scale = tensor_quantize(t, enabled=True, bits=8)
        assert q.dtype == torch.int8
        assert q.max().item() <= 127
        assert q.min().item() >= -128

    def test_8bit_approximate_roundtrip(self) -> None:
        t = torch.tensor([0.5, -1.0, 2.0], dtype=torch.float32)
        q, scale = tensor_quantize(t, enabled=True, bits=8)
        assert scale is not None
        dq = tensor_dequantize(q, scale, torch.float32)
        assert dq.dtype == torch.float32
        assert torch.allclose(dq, t, atol=0.15)

    def test_unsupported_bits_returns_original(self) -> None:
        t = torch.tensor([1.0, 2.0])
        q, scale = tensor_quantize(t, enabled=True, bits=16)
        assert q is t
        assert scale is None

    def test_quantize_fp8(self) -> None:
        if not hasattr(torch, "float8_e4m3fn"):
            pytest.skip("torch.float8_e4m3fn not available")
        t = torch.tensor([1.0, 2.0], dtype=torch.float16)
        q, scale = tensor_quantize(t, enabled=True, use_fp8=True)
        assert q.dtype == torch.float8_e4m3fn
        assert scale is None

    def test_fp8_only_applies_to_float16(self) -> None:
        if not hasattr(torch, "float8_e4m3fn"):
            pytest.skip("torch.float8_e4m3fn not available")
        t = torch.tensor([1.0, 2.0], dtype=torch.float32)
        q, scale = tensor_quantize(t, enabled=True, use_fp8=True)
        # fp8 path requires float16 input; falls through to bits=8 path
        assert q.dtype == torch.int8
        assert scale is not None

    def test_zero_tensor_quantize(self) -> None:
        t = torch.zeros(4, dtype=torch.float32)
        q, scale = tensor_quantize(t, enabled=True, bits=8)
        assert scale is not None
        # clamp(min=1e-5) is applied to abs().max() THEN divided by 127
        # expected scale = 1e-5 / 127.0 = ~7.87e-8
        assert scale.item() > 0.0
        assert torch.equal(q, torch.zeros(4, dtype=torch.int8))

    def test_small_values(self) -> None:
        t = torch.tensor([1e-6, -1e-6], dtype=torch.float32)
        q, scale = tensor_quantize(t, enabled=True, bits=8)
        assert scale is not None
        assert q.dtype == torch.int8
        # Clamp(min=1e-5) means scale ~ 1e-5/127, so 1e-6 / scale ~ 12.7 -> clamped to ±13
        assert q.abs().max().item() > 0


# ---------------------------------------------------------------------------
# tensor_dequantize
# ---------------------------------------------------------------------------


class TestTensorDequantize:
    """Dequantizing tensors after transfer."""

    def test_none_scale_returns_original_if_same_dtype(self) -> None:
        t = torch.tensor([1.0, 2.0], dtype=torch.float32)
        result = tensor_dequantize(t, None, torch.float32)
        assert result is t

    def test_none_scale_converts_dtype(self) -> None:
        q = torch.tensor([1, 2, 3], dtype=torch.int32)
        result = tensor_dequantize(q, None, torch.float32)
        assert result.dtype == torch.float32
        assert torch.allclose(result, torch.tensor([1.0, 2.0, 3.0]))

    def test_with_scale(self) -> None:
        q = torch.tensor([10, 20, 30], dtype=torch.int8)
        scale = torch.tensor(0.5)
        result = tensor_dequantize(q, scale, torch.float32)
        assert result.dtype == torch.float32
        assert torch.allclose(result, torch.tensor([5.0, 10.0, 15.0]))

    def test_dequantize_fp8(self) -> None:
        if not hasattr(torch, "float8_e4m3fn"):
            pytest.skip("torch.float8_e4m3fn not available")
        q = torch.tensor([1.0, 2.0], dtype=torch.float8_e4m3fn)
        result = tensor_dequantize(q, None, torch.float32, use_fp8=True)
        assert result.dtype == torch.float32

    def test_dequantize_fp8_returns_float16(self) -> None:
        if not hasattr(torch, "float8_e4m3fn"):
            pytest.skip("torch.float8_e4m3fn not available")
        q = torch.tensor([1.0, 2.0], dtype=torch.float8_e4m3fn)
        result = tensor_dequantize(q, None, torch.float16, use_fp8=True)
        assert result.dtype == torch.float16

    def test_identity_when_no_transform_needed(self) -> None:
        q = torch.tensor([1.0, 2.0], dtype=torch.float32)
        result = tensor_dequantize(q, None, torch.float32)
        assert result is q

    def test_int8_with_scale_roundtrip(self) -> None:
        original = torch.tensor([0.5, -1.0, 2.0], dtype=torch.float32)
        q, scale = tensor_quantize(original, enabled=True, bits=8)
        dq = tensor_dequantize(q, scale, torch.float32)
        assert torch.allclose(dq, original, atol=0.15)
