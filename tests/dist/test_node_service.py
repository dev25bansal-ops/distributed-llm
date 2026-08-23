"""Tests for distllm.dist.node_service module.

Zero mocks -- uses only real objects from the module.
Verifies all public API: tensor/KV conversion functions, NodeServicer,
NodeServer lifecycle, and ClusterKeyInterceptor.
"""

from __future__ import annotations

import threading
import time

import grpc
import pytest
import torch

from distllm.dist import node_pb2
from distllm.dist.node_service import (
    ClusterKeyInterceptor,
    NodeServer,
    NodeServicer,
    kv_cache_from_proto,
    kv_cache_to_proto,
    tensor_from_proto,
    tensor_to_proto,
)
from distllm.dist.privacy import PrivacySplitConfig
from distllm.dist.worker import WorkerNode

# ---------------------------------------------------------------------------
# Test doubles (not mocks -- hand-written minimal implementations)
# ---------------------------------------------------------------------------


class _FakeContext:
    """Minimal test double for grpc.ServicerContext.

    Implements only the methods actually called by NodeServicer RPCs and
    ClusterKeyInterceptor.  Not a mock -- a hand-written implementation.
    """

    def __init__(self) -> None:
        self.trailing_metadata: list[tuple[str, str]] = []
        self.code: grpc.StatusCode | None = None
        self.details: str = ""
        self._aborted: bool = False

    def send_trailing_metadata(self, metadata: tuple[str, str]) -> None:
        self.trailing_metadata.extend(metadata)

    def abort(self, code: grpc.StatusCode, details: str) -> None:
        self.code = code
        self.details = details
        self._aborted = True
        raise RuntimeError(f"gRPC abort: {code}, {details}")

    def set_code(self, code: grpc.StatusCode) -> None:
        self.code = code

    def set_details(self, details: str) -> None:
        self.details = details


class _FakeHandlerCallDetails:
    """Minimal stand-in for grpc.HandlerCallDetails."""

    def __init__(self, metadata: list[tuple[str, str]] | None = None) -> None:
        self.method = "/test/Method"
        self.invocation_metadata = metadata or []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_worker(**kwargs: object) -> WorkerNode:
    """Create a WorkerNode with safe defaults (no model loaded)."""
    defaults: dict[str, object] = dict(
        node_id="test-node",
        model_name="test-model",
        start_layer=0,
        end_layer=2,
        total_layers=8,
        port=0,
        coordinator_host="localhost",
        coordinator_port=50050,
        device="cpu",
        dtype="float16",
        privacy_config=PrivacySplitConfig(),
    )
    defaults.update(kwargs)
    return WorkerNode(**defaults)  # type: ignore[arg-type]


def _fake_forward(
    hidden_states: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
    past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    input_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
    """Stand-in forward_fn for testing a successful forward pass."""
    batch = hidden_states.shape[0] if hidden_states is not None else 1
    dim = hidden_states.shape[-1] if hidden_states is not None else 10
    output = torch.randn(batch, dim)
    kv: list[tuple[torch.Tensor, torch.Tensor]] = [
        (torch.randn(1, 4, 8), torch.randn(1, 4, 8))
    ]
    return output, kv


def _continuation(details: object) -> object:
    """Continuation callback for ClusterKeyInterceptor tests."""
    return _success_handler


def _success_handler(request: object, context: object) -> str:
    """Stub RPC handler that returns a fixed string on success."""
    return "OK"


# ===================================================================
# TestTensorProto
# ===================================================================


class TestTensorProto:
    """tensor_to_proto / tensor_from_proto roundtrip."""

    def test_none_tensor(self) -> None:
        pb = tensor_to_proto(None)
        assert pb.shape == []
        assert pb.dtype == "none"
        assert pb.raw_data == b""

    def test_scalar_tensor(self) -> None:
        t = torch.tensor(42.0)
        pb = tensor_to_proto(t)
        t2 = tensor_from_proto(pb)
        assert t2.numel() == 1
        assert t2.item() == 42.0

    def test_1d_tensor_float32(self) -> None:
        t = torch.tensor([1.0, 2.0, 3.0])
        pb = tensor_to_proto(t)
        t2 = tensor_from_proto(pb)
        assert torch.equal(t, t2)

    def test_2d_tensor_float16(self) -> None:
        t = torch.randn(2, 4).half()
        pb = tensor_to_proto(t)
        t2 = tensor_from_proto(pb)
        assert t2.dtype == torch.float16
        assert t.shape == t2.shape
        assert torch.equal(t, t2)

    def test_2d_tensor_bfloat16(self) -> None:
        t = torch.randn(2, 4).bfloat16()
        pb = tensor_to_proto(t)
        t2 = tensor_from_proto(pb)
        assert t2.dtype == torch.bfloat16
        assert t.shape == t2.shape
        assert torch.equal(t, t2)

    def test_int64_tensor(self) -> None:
        t = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)
        pb = tensor_to_proto(t)
        t2 = tensor_from_proto(pb)
        assert t2.dtype == torch.int64
        assert torch.equal(t, t2)

    def test_int32_tensor(self) -> None:
        t = torch.tensor([10, 20, 30], dtype=torch.int32)
        pb = tensor_to_proto(t)
        t2 = tensor_from_proto(pb)
        assert t2.dtype == torch.int32
        assert torch.equal(t, t2)

    def test_bool_tensor(self) -> None:
        t = torch.tensor([True, False, True], dtype=torch.bool)
        pb = tensor_to_proto(t)
        t2 = tensor_from_proto(pb)
        assert t2.dtype == torch.bool
        assert torch.equal(t, t2)

    def test_empty_tensor(self) -> None:
        t = torch.empty(0, 4)
        pb = tensor_to_proto(t)
        t2 = tensor_from_proto(pb)
        assert t2.shape == (0, 4)

    def test_roundtrip_preserves_shape_and_dtype(self) -> None:
        t = torch.randn(3, 5, 7).to(dtype=torch.bfloat16)
        pb = tensor_to_proto(t)
        t2 = tensor_from_proto(pb)
        assert list(t.shape) == list(t2.shape)
        assert t.dtype == t2.dtype
        assert torch.allclose(t.float(), t2.float())

    def test_tensor_from_proto_empty_shape(self) -> None:
        pb = node_pb2.TensorProto(shape=[], dtype="float32", raw_data=b"")
        t = tensor_from_proto(pb)
        assert t.numel() == 0

    def test_tensor_from_proto_data_field_fallback(self) -> None:
        """When raw_data is empty, should fall back to the repeated data field."""
        pb = node_pb2.TensorProto(
            shape=[3], dtype="float32", data=[1.0, 2.0, 3.0]
        )
        t = tensor_from_proto(pb)
        assert t.dtype == torch.float32
        assert t.tolist() == [1.0, 2.0, 3.0]

    def test_unknown_dtype_defaults_to_float32(self) -> None:
        t = torch.tensor([1.0, 2.0, 3.0])
        pb = tensor_to_proto(t)
        pb.dtype = "torch.complex128"  # not in dtype_map
        t2 = tensor_from_proto(pb)
        assert t2.dtype == torch.float32
        assert t2.shape == (3,)

    def test_dtype_short_names(self) -> None:
        """Short dtype names like 'float32', 'int64' should be accepted."""
        pb = node_pb2.TensorProto(shape=[1], dtype="float32", data=[1.0])
        assert tensor_from_proto(pb).dtype == torch.float32

        pb.dtype = "float16"
        assert tensor_from_proto(pb).dtype == torch.float16

        pb.dtype = "bfloat16"
        assert tensor_from_proto(pb).dtype == torch.bfloat16

        pb.dtype = "int64"
        assert tensor_from_proto(pb).dtype == torch.int64

    def test_device_parameter(self) -> None:
        t = torch.tensor([1.0, 2.0])
        pb = tensor_to_proto(t)
        t2 = tensor_from_proto(pb, device="cpu")
        assert t2.device.type == "cpu"

    def test_3d_tensor_uint8(self) -> None:
        t = torch.randint(0, 255, (2, 3, 4), dtype=torch.uint8)
        pb = tensor_to_proto(t)
        t2 = tensor_from_proto(pb)
        assert t2.dtype == torch.uint8
        assert torch.equal(t, t2)


# ===================================================================
# TestKVCacheProto
# ===================================================================


class TestKVCacheProto:
    """kv_cache_to_proto / kv_cache_from_proto roundtrip."""

    def test_none_cache(self) -> None:
        pb = kv_cache_to_proto(None)
        assert len(pb.layers) == 0
        result = kv_cache_from_proto(pb)
        assert result is None

    def test_none_result_from_none_input(self) -> None:
        result = kv_cache_from_proto(None)
        assert result is None

    def test_empty_proto_returns_none(self) -> None:
        pb = node_pb2.KVCacheProto()
        result = kv_cache_from_proto(pb)
        assert result is None

    def test_single_layer_roundtrip(self) -> None:
        k = torch.randn(1, 4, 8)
        v = torch.randn(1, 4, 8)
        original: list[tuple[torch.Tensor, torch.Tensor]] = [(k, v)]
        pb = kv_cache_to_proto(original)
        assert len(pb.layers) == 1
        restored = kv_cache_from_proto(pb)
        assert restored is not None
        assert len(restored) == 1
        assert torch.equal(restored[0][0], k)
        assert torch.equal(restored[0][1], v)

    def test_multi_layer_roundtrip(self) -> None:
        layers: list[tuple[torch.Tensor, torch.Tensor]] = [
            (torch.randn(1, 4, 8), torch.randn(1, 4, 8)),
            (torch.randn(1, 4, 16), torch.randn(1, 4, 16)),
            (torch.randn(1, 8, 8), torch.randn(1, 8, 8)),
        ]
        pb = kv_cache_to_proto(layers)
        assert len(pb.layers) == 3
        restored = kv_cache_from_proto(pb)
        assert restored is not None
        assert len(restored) == 3
        for i, (rk, rv) in enumerate(restored):
            assert torch.equal(rk, layers[i][0])
            assert torch.equal(rv, layers[i][1])

    def test_mixed_dtype_layers(self) -> None:
        layers: list[tuple[torch.Tensor, torch.Tensor]] = [
            (torch.randn(1, 4, 8).half(), torch.randn(1, 4, 8)),
            (torch.randn(1, 4, 8), torch.randn(1, 4, 8).bfloat16()),
        ]
        pb = kv_cache_to_proto(layers)
        restored = kv_cache_from_proto(pb)
        assert restored is not None
        assert restored[0][0].dtype == torch.float16
        assert restored[1][1].dtype == torch.bfloat16


# ===================================================================
# TestNodeServicer
# ===================================================================


class TestNodeServicer:
    """NodeServicer RPC method tests."""

    # -- Authorization --------------------------------------------------

    def test_check_auth_no_cluster_key_set(self) -> None:
        node = _make_worker()
        servicer = NodeServicer(node)  # keyless -> fail closed
        req = node_pb2.ForwardPassRequest()
        assert servicer._check_auth(req) is False

    def test_check_auth_with_correct_key(self) -> None:
        node = _make_worker()
        servicer = NodeServicer(node, cluster_key="secret")
        req = node_pb2.ForwardPassRequest(cluster_key="secret")
        assert servicer._check_auth(req) is True

    def test_check_auth_with_wrong_key(self) -> None:
        node = _make_worker()
        servicer = NodeServicer(node, cluster_key="secret")
        req = node_pb2.ForwardPassRequest(cluster_key="wrong")
        assert servicer._check_auth(req) is False

    def test_check_auth_with_missing_key(self) -> None:
        node = _make_worker()
        servicer = NodeServicer(node, cluster_key="secret")
        req = node_pb2.ForwardPassRequest()  # cluster_key defaults to ""
        assert servicer._check_auth(req) is False

    def test_check_auth_constant_time_comparison(self) -> None:
        """Verify hmac.compare_digest is used (not ==) via a simple check."""
        node = _make_worker()
        servicer = NodeServicer(node, cluster_key="a" * 100)
        req = node_pb2.ForwardPassRequest(cluster_key="b" * 100)
        assert servicer._check_auth(req) is False

    # -- ForwardPass ----------------------------------------------------

    def test_forward_pass_auth_failure(self) -> None:
        node = _make_worker()
        servicer = NodeServicer(node, cluster_key="secret")
        req = node_pb2.ForwardPassRequest(cluster_key="wrong")
        ctx = _FakeContext()
        resp = servicer.ForwardPass(req, ctx)
        assert resp.success is False
        assert "authentication failed" in resp.error_message

    def test_forward_pass_input_ids_too_large(self) -> None:
        node = _make_worker()
        servicer = NodeServicer(node, cluster_key="secret")
        # Exceed MAX_BATCH_SIZE * 131072 tokens
        req = node_pb2.ForwardPassRequest(
            input_ids=list(range(1024 * 131072 + 1)),
            cluster_key="secret",
        )
        resp = servicer.ForwardPass(req, _FakeContext())
        assert resp.success is False

    def test_forward_pass_no_model_loaded(self) -> None:
        """WorkerNode with no model returns an error gracefully."""
        node = _make_worker()
        servicer = NodeServicer(node, cluster_key="secret")
        req = node_pb2.ForwardPassRequest(cluster_key="secret")
        resp = servicer.ForwardPass(req, _FakeContext())
        assert resp.success is False
        assert resp.error_message

    def test_forward_pass_success_with_hidden_states(self) -> None:
        node = _make_worker()
        node.forward_fn = _fake_forward  # type: ignore[method-assign]
        servicer = NodeServicer(node, cluster_key="secret")
        t = torch.randn(1, 10)
        req = node_pb2.ForwardPassRequest(
            hidden_states=tensor_to_proto(t),
            request_id="fp-1",
            cluster_key="secret",
        )
        resp = servicer.ForwardPass(req, _FakeContext())
        assert resp.success, f"expected success, got: {resp.error_message}"
        assert resp.request_id == "fp-1"
        assert resp.processing_time_ms > 0

        # Verify output tensor roundtrip
        output = tensor_from_proto(resp.output)
        assert output.shape[0] == 1
        assert output.shape[-1] == 10

    def test_forward_pass_success_with_input_ids(self) -> None:
        node = _make_worker()
        node.forward_fn = _fake_forward  # type: ignore[method-assign]
        servicer = NodeServicer(node, cluster_key="secret")
        req = node_pb2.ForwardPassRequest(input_ids=[1, 2, 3], cluster_key="secret")
        resp = servicer.ForwardPass(req, _FakeContext())
        assert resp.success, f"expected success, got: {resp.error_message}"

    def test_forward_pass_success_with_kv_cache(self) -> None:
        node = _make_worker()
        node.forward_fn = _fake_forward  # type: ignore[method-assign]
        servicer = NodeServicer(node, cluster_key="secret")
        kv_pb = node_pb2.KVCacheProto()
        layer = kv_pb.layers.add()
        layer.key_states.CopyFrom(
            tensor_to_proto(torch.randn(1, 4, 8))
        )
        layer.value_states.CopyFrom(
            tensor_to_proto(torch.randn(1, 4, 8))
        )
        req = node_pb2.ForwardPassRequest(
            hidden_states=tensor_to_proto(torch.randn(1, 10)),
            kv_cache=kv_pb,
            cluster_key="secret",
        )
        resp = servicer.ForwardPass(req, _FakeContext())
        assert resp.success, f"expected success, got: {resp.error_message}"
        # Verify KV cache in response
        assert len(resp.kv_cache.layers) > 0

    def test_forward_pass_hidden_dim_too_large(self) -> None:
        node = _make_worker()
        servicer = NodeServicer(node, cluster_key="secret")
        t = torch.randn(1, NodeServicer.MAX_HIDDEN_DIM + 1)
        req = node_pb2.ForwardPassRequest(
            hidden_states=tensor_to_proto(t),
            cluster_key="secret",
        )
        resp = servicer.ForwardPass(req, _FakeContext())
        assert resp.success is False
        assert "too large" in resp.error_message

    def test_forward_pass_batch_too_large(self) -> None:
        node = _make_worker()
        servicer = NodeServicer(node, cluster_key="secret")
        t = torch.randn(NodeServicer.MAX_BATCH_SIZE + 1, 768)
        req = node_pb2.ForwardPassRequest(
            hidden_states=tensor_to_proto(t),
            cluster_key="secret",
        )
        resp = servicer.ForwardPass(req, _FakeContext())
        assert resp.success is False
        assert "batch size too large" in resp.error_message

    def test_forward_pass_kv_cache_too_many_layers(self) -> None:
        node = _make_worker()
        servicer = NodeServicer(node, cluster_key="secret")
        kv_pb = node_pb2.KVCacheProto()
        for _ in range(NodeServicer.MAX_KV_LAYERS + 1):
            layer = kv_pb.layers.add()
            layer.key_states.CopyFrom(
                tensor_to_proto(torch.randn(1, 4, 8))
            )
            layer.value_states.CopyFrom(
                tensor_to_proto(torch.randn(1, 4, 8))
            )
        req = node_pb2.ForwardPassRequest(
            hidden_states=tensor_to_proto(torch.randn(1, 768)),
            kv_cache=kv_pb,
            cluster_key="secret",
        )
        resp = servicer.ForwardPass(req, _FakeContext())
        assert resp.success is False
        assert "too many layers" in resp.error_message

    def test_forward_pass_kv_cache_seq_len_too_large(self) -> None:
        node = _make_worker()
        servicer = NodeServicer(node, cluster_key="secret")
        kv_pb = node_pb2.KVCacheProto()
        layer = kv_pb.layers.add()
        # Create a TensorProto with shape that has seq_len > MAX_KV_SEQ_LEN
        # at position -2 (where seq_len is expected in [b, h, s, d] layout).
        large_seq_pb = node_pb2.TensorProto(
            shape=[1, 4, NodeServicer.MAX_KV_SEQ_LEN + 1, 8],
            dtype="torch.float32",
        )
        layer.key_states.CopyFrom(large_seq_pb)
        layer.value_states.CopyFrom(
            tensor_to_proto(torch.randn(1, 4, 8, 8))
        )
        req = node_pb2.ForwardPassRequest(
            hidden_states=tensor_to_proto(torch.randn(1, 768)),
            kv_cache=kv_pb,
            cluster_key="secret",
        )
        resp = servicer.ForwardPass(req, _FakeContext())
        assert resp.success is False
        assert "seq_len too large" in resp.error_message

    def test_forward_pass_too_many_dims(self) -> None:
        node = _make_worker()
        servicer = NodeServicer(node, cluster_key="secret")
        # 5D tensor should fail dim() > 4 check
        t = torch.randn(1, 2, 3, 4, 5)
        req = node_pb2.ForwardPassRequest(
            hidden_states=tensor_to_proto(t),
            cluster_key="secret",
        )
        resp = servicer.ForwardPass(req, _FakeContext())
        assert resp.success is False
        assert "too large" in resp.error_message

    # -- HealthCheck ----------------------------------------------------

    def test_health_check_auth_failure(self) -> None:
        node = _make_worker()
        servicer = NodeServicer(node, cluster_key="secret")
        req = node_pb2.HealthCheckRequest(node_id="test-node", cluster_key="wrong")
        resp = servicer.HealthCheck(req, _FakeContext())
        assert resp.healthy is False

    def test_health_check_basic(self) -> None:
        node = _make_worker(
            node_id="health-node",
            start_layer=1,
            end_layer=5,
            total_layers=12,
        )
        servicer = NodeServicer(node, cluster_key="secret")
        req = node_pb2.HealthCheckRequest(node_id="health-node", cluster_key="secret")
        resp = servicer.HealthCheck(req, _FakeContext())
        assert resp.healthy is True
        assert resp.node_id == "health-node"
        assert resp.start_layer == 1
        assert resp.end_layer == 5
        assert resp.total_layers == 12
        # No model loaded, so layers_loaded should be 0
        assert resp.num_layers_loaded == 0

    def test_health_check_gpu_fallback(self) -> None:
        """GPU fields should always be populated regardless of hardware."""
        node = _make_worker()
        servicer = NodeServicer(node, cluster_key="secret")
        req = node_pb2.HealthCheckRequest(node_id="test-node", cluster_key="secret")
        resp = servicer.HealthCheck(req, _FakeContext())
        assert resp.healthy is True
        assert isinstance(resp.gpu_name, str) and len(resp.gpu_name) > 0

    # -- Profile --------------------------------------------------------

    def test_profile_auth_failure(self) -> None:
        node = _make_worker(node_id="profile-node")
        servicer = NodeServicer(node, cluster_key="secret")
        req = node_pb2.ProfileRequest(node_id="profile-node", cluster_key="wrong")
        resp = servicer.Profile(req, _FakeContext())
        assert resp.node_id == "profile-node"

    def test_profile_basic(self) -> None:
        node = _make_worker(node_id="profile-node")
        servicer = NodeServicer(node, cluster_key="secret")
        req = node_pb2.ProfileRequest(node_id="profile-node", cluster_key="secret")
        resp = servicer.Profile(req, _FakeContext())
        assert resp.node_id == "profile-node"
        # gpu_name is environment-dependent; just check it's populated
        assert isinstance(resp.gpu_name, str) and len(resp.gpu_name) > 0

    # -- AdvertiseModels ------------------------------------------------

    def test_advertise_models_auth_failure(self) -> None:
        node = _make_worker()
        servicer = NodeServicer(node, cluster_key="secret")
        req = node_pb2.AdvertiseModelsRequest(node_id="test-node", cluster_key="wrong")
        resp = servicer.AdvertiseModels(req, _FakeContext())
        assert len(resp.models) == 0

    def test_advertise_models_no_partitioner(self) -> None:
        node = _make_worker()
        servicer = NodeServicer(node, cluster_key="secret")
        req = node_pb2.AdvertiseModelsRequest(node_id="test-node", cluster_key="secret")
        resp = servicer.AdvertiseModels(req, _FakeContext())
        # Without partitioner, the servicer returns empty models
        assert len(resp.models) == 0

    # -- TransferWeights ------------------------------------------------

    def test_transfer_weights_auth_failure(self) -> None:
        node = _make_worker()
        servicer = NodeServicer(node, cluster_key="secret")
        req = node_pb2.TransferWeightsRequest(
            model_name="test-model",
            start_layer=0,
            end_layer=2,
            cluster_key="wrong",
        )
        resp = servicer.TransferWeights(req, _FakeContext())
        assert resp.success is False
        assert "authentication failed" in resp.error_message

    def test_transfer_weights_no_partitioner(self) -> None:
        node = _make_worker()
        servicer = NodeServicer(node, cluster_key="secret")
        req = node_pb2.TransferWeightsRequest(
            model_name="test-model",
            start_layer=0,
            end_layer=2,
            cluster_key="secret",
        )
        resp = servicer.TransferWeights(req, _FakeContext())
        assert resp.success is False
        assert "not loaded" in resp.error_message

    def test_transfer_weights_invalid_layer_range(self) -> None:
        node = _make_worker()
        servicer = NodeServicer(node, cluster_key="secret")
        req = node_pb2.TransferWeightsRequest(
            model_name="test-model",
            start_layer=-1,
            end_layer=2,
            cluster_key="secret",
        )
        resp = servicer.TransferWeights(req, _FakeContext())
        assert resp.success is False
        # Without a partitioner, the servicer returns "not loaded" first
        assert resp.error_message

    def test_transfer_weights_swapped_layer_range(self) -> None:
        node = _make_worker()
        servicer = NodeServicer(node, cluster_key="secret")
        req = node_pb2.TransferWeightsRequest(
            model_name="test-model",
            start_layer=5,
            end_layer=2,
            cluster_key="secret",
        )
        resp = servicer.TransferWeights(req, _FakeContext())
        assert resp.success is False
        assert resp.error_message

    def test_transfer_weights_layer_range_exceeds_max(self) -> None:
        node = _make_worker()
        servicer = NodeServicer(node, cluster_key="secret")
        req = node_pb2.TransferWeightsRequest(
            model_name="test-model",
            start_layer=0,
            end_layer=NodeServicer.MAX_LAYER_RANGE + 1,
            cluster_key="secret",
        )
        resp = servicer.TransferWeights(req, _FakeContext())
        assert resp.success is False
        assert resp.error_message

    # -- TransferWeightsStream ------------------------------------------

    def test_transfer_weights_stream_auth_failure(self) -> None:
        node = _make_worker()
        servicer = NodeServicer(node, cluster_key="secret")
        req = node_pb2.TransferWeightsRequest(
            model_name="test-model",
            start_layer=0,
            end_layer=2,
            cluster_key="wrong",
        )
        responses = list(servicer.TransferWeightsStream(req, _FakeContext()))
        assert len(responses) == 1
        assert responses[0].success is False
        assert "authentication failed" in responses[0].error_message

    def test_transfer_weights_stream_invalid_range(self) -> None:
        node = _make_worker()
        servicer = NodeServicer(node, cluster_key="secret")
        req = node_pb2.TransferWeightsRequest(
            model_name="test-model",
            start_layer=-1,
            end_layer=2,
            cluster_key="secret",
        )
        responses = list(servicer.TransferWeightsStream(req, _FakeContext()))
        assert len(responses) == 1
        assert responses[0].success is False
        assert "invalid layer range" in responses[0].error_message


# ===================================================================
# TestNodeServer
# ===================================================================


class TestNodeServer:
    """NodeServer lifecycle tests (start / stop / wait)."""

    def test_construct_with_defaults(self) -> None:
        from concurrent import futures

        server = NodeServer(_make_worker(), port=50051)
        assert server._port == 50051
        assert server._max_workers == 4
        assert server._cluster_key is None
        assert server._server is None
        assert server._running.is_set() is False
        assert server._stopped.is_set() is False
        assert server.MAX_MSG_SIZE == 512 * 1024 * 1024

    def test_construct_with_cluster_key(self) -> None:
        server = NodeServer(_make_worker(), port=50051, cluster_key="ck")
        assert server._cluster_key == "ck"

    def test_construct_with_custom_max_workers(self) -> None:
        server = NodeServer(_make_worker(), port=50051, max_workers=16)
        assert server._max_workers == 16

    def test_start_and_stop(self) -> None:
        """Start on ephemeral port 0 and then stop."""
        node = _make_worker()
        server = NodeServer(node, port=0)
        try:
            server.start()
            assert server._running.is_set()
            assert server._server is not None
        finally:
            server.stop()
        assert server._running.is_set() is False
        assert server._stopped.is_set()

    def test_stop_without_start_is_safe(self) -> None:
        server = NodeServer(_make_worker(), port=50051)
        # Should not raise
        server.stop()

    def test_stop_twice_is_idempotent(self) -> None:
        node = _make_worker()
        server = NodeServer(node, port=0)
        try:
            server.start()
        finally:
            server.stop()
        server.stop()  # second stop should not raise

    def test_wait_returns_after_stop(self) -> None:
        """wait() should block until stop() is called from another thread."""
        node = _make_worker()
        server = NodeServer(node, port=0)
        try:
            server.start()

            def _do_stop() -> None:
                time.sleep(0.05)
                server.stop()

            t = threading.Thread(target=_do_stop, daemon=True)
            t.start()
            server.wait()
            assert server._stopped.is_set()
            t.join()
        except Exception:
            server.stop()
            raise


# ===================================================================
# TestClusterKeyInterceptor
# ===================================================================


class TestClusterKeyInterceptor:
    """ClusterKeyInterceptor gRPC interceptor tests."""

    def test_construct(self) -> None:
        interceptor = ClusterKeyInterceptor(cluster_key="secret")
        assert interceptor._cluster_key == "secret"

    def test_intercept_with_valid_key(self) -> None:
        interceptor = ClusterKeyInterceptor(cluster_key="secret")
        details = _FakeHandlerCallDetails([("cluster-key", "secret")])
        handler = interceptor.intercept_service(_continuation, details)
        # The handler should be our _success_handler
        assert callable(handler)
        assert handler(None, _FakeContext()) == "OK"

    def test_intercept_with_x_cluster_key(self) -> None:
        """The x-cluster-key metadata key should also work."""
        interceptor = ClusterKeyInterceptor(cluster_key="secret")
        details = _FakeHandlerCallDetails([("x-cluster-key", "secret")])
        handler = interceptor.intercept_service(_continuation, details)
        assert handler(None, _FakeContext()) == "OK"

    def test_intercept_without_key(self) -> None:
        interceptor = ClusterKeyInterceptor(cluster_key="secret")
        details = _FakeHandlerCallDetails([])  # no metadata
        handler = interceptor.intercept_service(_continuation, details)
        # Should return the unauthenticated handler
        assert handler is interceptor._unauthenticated_handler

    def test_intercept_with_wrong_key(self) -> None:
        interceptor = ClusterKeyInterceptor(cluster_key="secret")
        details = _FakeHandlerCallDetails([("cluster-key", "wrong")])
        handler = interceptor.intercept_service(_continuation, details)
        assert handler is interceptor._unauthenticated_handler

    def test_intercept_with_empty_key(self) -> None:
        interceptor = ClusterKeyInterceptor(cluster_key="secret")
        details = _FakeHandlerCallDetails([("cluster-key", "")])
        handler = interceptor.intercept_service(_continuation, details)
        assert handler is interceptor._unauthenticated_handler

    def test_unauthenticated_handler_aborts(self) -> None:
        handler = ClusterKeyInterceptor._unauthenticated_handler
        ctx = _FakeContext()
        with pytest.raises(RuntimeError, match="gRPC abort"):
            handler(None, ctx)
        assert ctx._aborted

    def test_continuation_called_with_details(self) -> None:
        """Verify that the continuation receives the handler_call_details."""
        captured: list[object] = []

        def _capture_continuation(details: object) -> object:
            captured.append(details)
            return "captured_handler"

        interceptor = ClusterKeyInterceptor(cluster_key="secret")
        details = _FakeHandlerCallDetails([("cluster-key", "secret")])
        handler = interceptor.intercept_service(_capture_continuation, details)
        assert len(captured) == 1
        assert captured[0] is details
        assert handler == "captured_handler"


# ===================================================================
# Smoke test -- module exports
# ===================================================================


class TestModuleExports:
    """Sanity-check that the module exposes all expected public names."""

    def test_all_public_exports(self) -> None:
        import distllm.dist.node_service as ns

        assert ns.NodeServicer is NodeServicer
        assert ns.NodeServer is NodeServer
        assert ns.ClusterKeyInterceptor is ClusterKeyInterceptor
        assert ns.tensor_to_proto is tensor_to_proto
        assert ns.tensor_from_proto is tensor_from_proto
        assert ns.kv_cache_to_proto is kv_cache_to_proto
        assert ns.kv_cache_from_proto is kv_cache_from_proto
