"""Tests for the gRPC client (distllm_sdk.grpc_client).

The gRPC channel/stub are fully mocked via a fake ``distllm.dist`` module so
no network or ``grpcio``/``torch`` dependency is required (the torch code path
is exercised only behind an importorskip guard).
"""

import asyncio
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import distllm_sdk.grpc_client as grpc_client
from distllm_sdk.grpc_client import (
    NodeGRPCClient,
    TensorData,
    ForwardPassResponse,
    HealthCheckResponse,
    ProfileResponse,
    TransferWeightsResponse,
    ModelAdvertisement,
)


# --------------------------------------------------------------------------- #
# Fake gRPC protobuf objects
# --------------------------------------------------------------------------- #
class _Tensor:
    """Mimics a protobuf tensor message (shape / dtype / raw_data)."""

    def __init__(self, shape=None, dtype="", raw_data=b""):
        self.shape = shape if shape is not None else []
        self.dtype = dtype
        self.raw_data = raw_data

    def extend(self, values):
        self.shape.extend(values)


class _Request:
    """Stores all supplied kwargs as attributes (protobuf request messages)."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        # Provided so mutation code paths (hidden_states.shape.extend) don't crash
        self.hidden_states = _Tensor()
        self.output = _Tensor()


class FakeNodeServiceStub:
    """Stand-in for node_pb2_grpc.NodeServiceStub.

    RPC methods are *class-level* AsyncMocks so that configuring them on the
    class affects every freshly-created stub instance (the client constructs a
    new stub on every call).
    """

    def __init__(self, channel):
        self.channel = channel

    ForwardPass = AsyncMock()
    HealthCheck = AsyncMock()
    Profile = AsyncMock()
    TransferWeights = AsyncMock()
    AdvertiseModels = AsyncMock()
    TransferWeightsStream = None  # configured per-test


async def _empty_gen():
    return
    yield


async def _gen(chunks):
    for c in chunks:
        yield c


def _fp_response(request_id="r", output=None, success=True, error_message="",
                 error_code=0, is_logits=True, processing_time_ms=0.0):
    return SimpleNamespace(
        request_id=request_id,
        output=output if output is not None else _Tensor(),
        success=success,
        error_message=error_message,
        error_code=error_code,
        is_logits=is_logits,
        processing_time_ms=processing_time_ms,
    )


def _hc_response(node_id="n1", healthy=True, **kw):
    return SimpleNamespace(
        healthy=healthy,
        node_id=node_id,
        memory_used_bytes=kw.get("memory_used_bytes", 0),
        memory_total_bytes=kw.get("memory_total_bytes", 0),
        gpu_utilization=kw.get("gpu_utilization", 0.0),
        start_layer=kw.get("start_layer", 0),
        end_layer=kw.get("end_layer", 0),
        total_layers=kw.get("total_layers", 0),
        gpu_name=kw.get("gpu_name", ""),
        gpu_memory_total=kw.get("gpu_memory_total", 0),
        num_layers_loaded=kw.get("num_layers_loaded", 0),
    )


def _profile_response(node_id="n1", **kw):
    return SimpleNamespace(
        node_id=node_id,
        gpu_name=kw.get("gpu_name", "A100"),
        total_memory_bytes=kw.get("total_memory_bytes", 0),
        free_memory_bytes=kw.get("free_memory_bytes", 0),
        compute_tflops=kw.get("compute_tflops", 0.0),
        memory_bandwidth_gbps=kw.get("memory_bandwidth_gbps", 0.0),
        sm_count=kw.get("sm_count", 0),
    )


def _tw_response(model_name="m", start_layer=0, end_layer=1, state_dict_bytes=b"",
                 success=True, error_message=""):
    return SimpleNamespace(
        model_name=model_name,
        start_layer=start_layer,
        end_layer=end_layer,
        state_dict_bytes=state_dict_bytes,
        success=success,
        error_message=error_message,
        chunk_index=0,
        total_chunks=1,
        is_final_chunk=True,
    )


def _advertise_response(models):
    return SimpleNamespace(models=models)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def reset_stub():
    # reset_mock() alone does NOT clear side_effect/return_value, so be explicit
    # to prevent leakage between tests.
    FakeNodeServiceStub.ForwardPass.reset_mock(return_value=True, side_effect=True)
    FakeNodeServiceStub.HealthCheck.reset_mock(return_value=True, side_effect=True)
    FakeNodeServiceStub.Profile.reset_mock(return_value=True, side_effect=True)
    FakeNodeServiceStub.TransferWeights.reset_mock(return_value=True, side_effect=True)
    FakeNodeServiceStub.AdvertiseModels.reset_mock(return_value=True, side_effect=True)
    FakeNodeServiceStub.TransferWeightsStream = staticmethod(lambda req: _empty_gen())
    yield


@pytest.fixture
def fake_distllm(monkeypatch):
    fake_dist = types.ModuleType("distllm")
    fake_dist_dist = types.ModuleType("distllm.dist")
    pb2 = SimpleNamespace(
        ForwardPassRequest=_Request,
        HealthCheckRequest=_Request,
        ProfileRequest=_Request,
        TransferWeightsRequest=_Request,
        AdvertiseModelsRequest=_Request,
    )
    pb2_grpc = SimpleNamespace(NodeServiceStub=FakeNodeServiceStub)

    fake_dist_dist.node_pb2 = pb2
    fake_dist_dist.node_pb2_grpc = pb2_grpc
    fake_dist.dist = fake_dist_dist

    monkeypatch.setitem(sys.modules, "distllm", fake_dist)
    monkeypatch.setitem(sys.modules, "distllm.dist", fake_dist_dist)
    monkeypatch.setitem(sys.modules, "distllm.dist.node_pb2", pb2)
    monkeypatch.setitem(sys.modules, "distllm.dist.node_pb2_grpc", pb2_grpc)
    yield


@pytest.fixture
def client(fake_distllm):
    c = NodeGRPCClient("localhost:50051", cluster_key="secret")
    c._channel = MagicMock()
    c._connected = True
    return c


# --------------------------------------------------------------------------- #
# Forward pass
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_forward_pass_success(client):
    FakeNodeServiceStub.ForwardPass.return_value = _fp_response(
        request_id="r", output=_Tensor(), success=True
    )
    resp = await client.forward_pass(input_ids=[1, 2, 3], request_id="r")
    assert isinstance(resp, ForwardPassResponse)
    assert resp.request_id == "r"
    assert resp.success is True
    assert resp.output is None  # empty raw_data -> None


@pytest.mark.asyncio
async def test_forward_pass_no_output(client):
    FakeNodeServiceStub.ForwardPass.return_value = _fp_response(output=_Tensor(raw_data=b""))
    resp = await client.forward_pass(input_ids=[1])
    assert resp.success is True
    assert resp.output is None


@pytest.mark.asyncio
async def test_forward_pass_timeout(client):
    FakeNodeServiceStub.ForwardPass.side_effect = asyncio.TimeoutError()
    resp = await client.forward_pass(input_ids=[1])
    assert resp.success is False
    assert resp.error_code == 4
    assert "timed out" in resp.error_message


@pytest.mark.asyncio
async def test_forward_pass_auth_cluster_key(client):
    FakeNodeServiceStub.ForwardPass.return_value = _fp_response(request_id="r")
    await client.forward_pass(input_ids=[1], request_id="r")
    request = FakeNodeServiceStub.ForwardPass.call_args.args[0]
    assert request.cluster_key == "secret"


@pytest.mark.asyncio
async def test_forward_pass_with_output_tensor(client):
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    # NOTE: grpc_client.py binds `torch` as a function-local via
    # ``import torch`` inside the hidden_states branch, so to reach the
    # output-tensor decoding path we must supply a real hidden_states tensor.
    hidden = torch.arange(4, dtype=torch.float32).reshape(1, 4)
    arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    raw = arr.tobytes()
    FakeNodeServiceStub.ForwardPass.return_value = _fp_response(
        request_id="r", output=_Tensor(shape=[4], dtype="float32", raw_data=raw)
    )
    resp = await client.forward_pass(input_ids=[1], request_id="r", hidden_states=hidden)
    assert isinstance(resp.output, TensorData)
    assert resp.output.shape == [4]
    assert resp.output.raw_data == raw


# --------------------------------------------------------------------------- #
# Health check
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_health_check_success(client):
    FakeNodeServiceStub.HealthCheck.return_value = _hc_response(node_id="n1", healthy=True)
    resp = await client.health_check(node_id="n1")
    assert isinstance(resp, HealthCheckResponse)
    assert resp.healthy is True
    assert resp.node_id == "n1"


@pytest.mark.asyncio
async def test_health_check_failure(client):
    FakeNodeServiceStub.HealthCheck.side_effect = Exception("boom")
    resp = await client.health_check(node_id="n1")
    assert resp.healthy is False
    assert resp.node_id == "n1"


# --------------------------------------------------------------------------- #
# Profile
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_profile_success(client):
    FakeNodeServiceStub.Profile.return_value = _profile_response(node_id="n1", gpu_name="A100")
    resp = await client.profile(node_id="n1")
    assert isinstance(resp, ProfileResponse)
    assert resp.node_id == "n1"
    assert resp.gpu_name == "A100"


@pytest.mark.asyncio
async def test_profile_failure(client):
    FakeNodeServiceStub.Profile.side_effect = Exception("boom")
    resp = await client.profile(node_id="n1")
    assert resp.node_id == "n1"
    assert resp.gpu_name == ""


# --------------------------------------------------------------------------- #
# Weight transfer
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_transfer_weights_success(client):
    FakeNodeServiceStub.TransferWeights.return_value = _tw_response(
        model_name="m", state_dict_bytes=b"weights"
    )
    resp = await client.transfer_weights("m", 0, 1)
    assert isinstance(resp, TransferWeightsResponse)
    assert resp.success is True
    assert resp.state_dict_bytes == b"weights"


@pytest.mark.asyncio
async def test_transfer_weights_failure(client):
    FakeNodeServiceStub.TransferWeights.side_effect = Exception("boom")
    resp = await client.transfer_weights("m", 0, 1)
    assert resp.success is False
    assert "boom" in resp.error_message


# --------------------------------------------------------------------------- #
# Model advertisement
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_advertise_models_success(client):
    models = [
        SimpleNamespace(model_name="m1", start_layer=0, end_layer=1, total_layers=2,
                        node_id="n1", host="h", port=1),
        SimpleNamespace(model_name="m2", start_layer=2, end_layer=3, total_layers=2,
                        node_id="n1", host="h", port=1),
    ]
    FakeNodeServiceStub.AdvertiseModels.return_value = _advertise_response(models)
    result = await client.advertise_models(node_id="n1")
    assert len(result) == 2
    assert isinstance(result[0], ModelAdvertisement)
    assert result[0].model_name == "m1"


@pytest.mark.asyncio
async def test_advertise_models_failure(client):
    FakeNodeServiceStub.AdvertiseModels.side_effect = Exception("boom")
    with pytest.raises(RuntimeError):
        await client.advertise_models(node_id="n1")


# --------------------------------------------------------------------------- #
# Weight transfer streaming
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_transfer_weights_stream(client):
    chunks = [
        SimpleNamespace(model_name="m", start_layer=0, end_layer=1, state_dict_bytes=b"c0",
                        success=True, error_message="", chunk_index=0, total_chunks=2,
                        is_final_chunk=False),
        SimpleNamespace(model_name="m", start_layer=0, end_layer=1, state_dict_bytes=b"c1",
                        success=True, error_message="", chunk_index=1, total_chunks=2,
                        is_final_chunk=True),
    ]
    FakeNodeServiceStub.TransferWeightsStream = staticmethod(lambda req: _gen(chunks))
    collected = [c async for c in client.transfer_weights_stream("m", 0, 1)]
    assert len(collected) == 2
    assert all(isinstance(c, TransferWeightsResponse) for c in collected)
    assert collected[0].is_final_chunk is False
    assert collected[-1].is_final_chunk is True


# --------------------------------------------------------------------------- #
# Connection state
# --------------------------------------------------------------------------- #
def test_is_connected_property():
    c = NodeGRPCClient("localhost:50051")
    assert c.is_connected is False
    c._connected = True
    assert c.is_connected is True
