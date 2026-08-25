"""Regression tests for WideAreaPipeline missing-attribute crashes (audit C1).

Before the fix, ``run_pipeline_async_p2p`` crashed sequentially on:

1. ``self._topology_lock``  — never defined (parent stores ``_lock``)
2. ``self.resource_mgr``    — parent stores ``_resource_mgr``, no property
3. ``self._prepare_forward_request`` — defined nowhere
4. ``self._find_fallback_node``      — defined nowhere
5. ``self._process_forward_response`` — defined nowhere
6. ``type(request).Response()``       — protobuf messages have no Response

These tests instantiate a real WideAreaPipeline and drive each
previously-crashing path end to end with in-process fakes (no network).
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest
import torch

from distllm.dist.wide_area import WideAreaPipeline
from distllm.dist.config import WideAreaConfig
from distllm.dist import node_pb2
from distllm.dist.pipeline.serialization import to_proto_tensor
from distllm.errors.types import InputValidationError, NodeUnreachableError
from distllm.core.resource_manager import CircuitBreakerConfig, ResourceManager

HID = 8


def _ok_resp(output: torch.Tensor | None = None) -> node_pb2.ForwardPassResponse:
    resp = node_pb2.ForwardPassResponse(success=True)
    out = output if output is not None else torch.ones(1, 3, HID)
    resp.output.CopyFrom(to_proto_tensor(out))
    return resp


class _EchoStub:
    """Sync stub returning a successful ForwardPassResponse."""

    def __init__(self) -> None:
        self.requests: list[node_pb2.ForwardPassRequest] = []

    def ForwardPass(self, request):
        self.requests.append(request)
        return _ok_resp()


class _EchoClient:
    def __init__(self) -> None:
        self.stub = _EchoStub()


def _grpc_pipeline(
    resource_mgr: ResourceManager | None = None,
    wan_config: WideAreaConfig | None = None,
    num_nodes: int = 2,
) -> tuple[WideAreaPipeline, list[_EchoStub]]:
    """Build a gRPC-path WAN pipeline with ``num_nodes`` echo nodes."""
    pipeline = WideAreaPipeline(resource_mgr=resource_mgr, wan_config=wan_config)
    stubs: list[_EchoStub] = []
    ids = ("n0", "n1")[:num_nodes]
    for i, nid in enumerate(ids):
        pipeline.register_node(nid, "127.0.0.1", 5000 + i, i * 2, i * 2 + 1)
        client = _EchoClient()
        stubs.append(client.stub)
        pipeline.nodes[nid].client = client
    return pipeline, stubs


# ── 1. The original crash: run_pipeline_async_p2p no longer AttributeErrors ──


class TestRunPipelineAsyncP2pNoAttributeError:
    def test_single_node_end_to_end(self) -> None:
        pipeline, (stub,) = _grpc_pipeline(num_nodes=1)
        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        out = asyncio.run(
            pipeline.run_pipeline_async_p2p(input_ids, {}, "req-1")
        )
        assert out.shape == (1, 3, HID)
        # First hop carries token ids over the wire.
        assert list(stub.requests[0].input_ids) == [1, 2, 3]
        assert stub.requests[0].is_first_pass is True
        assert stub.requests[0].is_last_pass is True

    def test_two_node_hop_chain(self) -> None:
        pipeline, (s0, s1) = _grpc_pipeline()
        input_ids = torch.tensor([[7, 8, 9]], dtype=torch.long)
        out = asyncio.run(
            pipeline.run_pipeline_async_p2p(input_ids, {}, "req-2")
        )
        assert out.shape == (1, 3, HID)
        # Hop 0 sends ids; hop 1 receives hidden states.
        assert len(s0.requests[0].input_ids) == 3
        assert s0.requests[0].is_first_pass and not s0.requests[0].is_last_pass
        assert not len(s1.requests[0].input_ids)
        assert not s1.requests[0].is_first_pass and s1.requests[0].is_last_pass
        assert s1.requests[0].hidden_states.raw_data  # payload present

    def test_no_nodes_raises_runtime_error(self) -> None:
        pipeline = WideAreaPipeline(resource_mgr=ResourceManager())
        with pytest.raises(RuntimeError, match="No nodes registered"):
            asyncio.run(
                pipeline.run_pipeline_async_p2p(torch.tensor([[1]]), {}, "r")
            )

    def test_empty_nodes_dict_entry_skipped_kv(self) -> None:
        """node_kv_caches without an entry for the node must not crash."""
        pipeline, _ = _grpc_pipeline(num_nodes=1)
        out = asyncio.run(
            pipeline.run_pipeline_async_p2p(
                torch.tensor([[1, 2, 3]], dtype=torch.long), {}, "req-kv"
            )
        )
        assert out.shape == (1, 3, HID)


class TestTopologyLockDefined:
    def test_topology_lock_exists_after_init(self) -> None:
        pipeline = WideAreaPipeline()
        assert isinstance(pipeline._topology_lock, type(threading.RLock()))  # noqa: SLF001

    def test_discovery_loop_registers_under_topology_lock(self) -> None:
        """_discovery_loop holds _topology_lock around register_node; the
        RLock makes that safe even though register_node takes parent _lock."""
        pipeline = WideAreaPipeline(resource_mgr=ResourceManager())
        calls = {"n": 0}

        def fake_discover(service_type: str, port: int):
            calls["n"] += 1
            if calls["n"] == 1:
                return [("10.0.0.9", 50051, (0, 3), "disc-0")]
            return []

        original = WideAreaPipeline._discover_nodes
        WideAreaPipeline._discover_nodes = staticmethod(fake_discover)
        orig_sleep = time.sleep
        time.sleep = lambda s: orig_sleep(0.005)
        try:
            pipeline.start_auto_discovery()
            deadline = time.time() + 5
            while time.time() < deadline and "disc-0" not in pipeline.nodes:
                orig_sleep(0.02)
        finally:
            pipeline._auto_discovery_running = False  # noqa: SLF001
            time.sleep = orig_sleep
            WideAreaPipeline._discover_nodes = original
        assert "disc-0" in pipeline.nodes
        assert pipeline.node_order == ["disc-0"]

    def test_adjust_window_uses_topology_lock(self) -> None:
        """The H-04 getattr fallback now resolves via the real attribute."""
        config = WideAreaConfig(adaptive_batching=True, accumulation_window=4)
        pipeline = WideAreaPipeline(wan_config=config)
        pipeline.register_node("a", "h", 5000, 0, 1)
        pipeline._link_latencies[("a", "a")] = [100.0]  # noqa: SLF001
        window = pipeline._adjust_accumulation_window()  # noqa: SLF001
        assert window >= 1


# ── 2. resource_mgr property ─────────────────────────────────────────────────


class TestResourceMgrProperty:
    def test_property_returns_constructor_arg(self) -> None:
        rm = ResourceManager()
        pipeline = WideAreaPipeline(resource_mgr=rm)
        assert pipeline.resource_mgr is rm
        assert pipeline.resource_mgr is pipeline._resource_mgr  # noqa: SLF001

    def test_setter_replaces_underlying_manager(self) -> None:
        rm_a, rm_b = ResourceManager(), ResourceManager()
        pipeline = WideAreaPipeline(resource_mgr=rm_a)
        pipeline.resource_mgr = rm_b
        assert pipeline._resource_mgr is rm_b  # noqa: SLF001

    def test_default_none_still_accessible(self) -> None:
        pipeline = WideAreaPipeline()
        assert pipeline.resource_mgr is None


# ── 3. _prepare_forward_request ──────────────────────────────────────────────


class TestPrepareForwardRequest:
    def _make(self):
        return WideAreaPipeline(resource_mgr=ResourceManager())

    def test_first_hop_carries_input_ids(self) -> None:
        pipeline = self._make()
        req = pipeline._prepare_forward_request(  # noqa: SLF001
            "n0", None, True, False, 3, 1, None, None, "r",
            None, torch.tensor([[1, 2, 3]], dtype=torch.long),
        )
        assert isinstance(req, node_pb2.ForwardPassRequest)
        assert list(req.input_ids) == [1, 2, 3]
        assert req.request_id == "r"
        assert req.seq_len == 3 and req.batch_size == 1

    def test_later_hop_carries_hidden_states(self) -> None:
        pipeline = self._make()
        hidden = torch.randn(1, 3, HID)
        req = pipeline._prepare_forward_request(  # noqa: SLF001
            "n1", None, False, True, 3, 1, hidden, None, "r",
            None, torch.zeros(1, 3, dtype=torch.long),
        )
        assert not len(req.input_ids)
        assert req.hidden_states.raw_data
        restored = torch.frombuffer(
            bytearray(req.hidden_states.raw_data), dtype=torch.float32
        ).reshape(list(req.hidden_states.shape))
        assert torch.allclose(restored.float(), hidden.float(), atol=1e-6)

    def test_broken_chain_raises_input_validation(self) -> None:
        pipeline = self._make()
        with pytest.raises(InputValidationError, match="chain broken"):
            pipeline._prepare_forward_request(  # noqa: SLF001
                "n1", None, False, True, 3, 1, None, None, "r",
                None, torch.zeros(1, 3, dtype=torch.long),
            )

    def test_empty_input_ids_rejected(self) -> None:
        pipeline = self._make()
        with pytest.raises(InputValidationError, match="non-empty"):
            pipeline._prepare_forward_request(  # noqa: SLF001
                "n0", None, True, False, 0, 1, None, None, "r",
                None, torch.zeros(1, 0, dtype=torch.long),
            )

    def test_draft_tokens_and_flags(self) -> None:
        pipeline = self._make()
        req = pipeline._prepare_forward_request(  # noqa: SLF001
            "n0", None, True, True, 2, 1, None,
            [(torch.zeros(1, 1, HID), torch.zeros(1, 1, HID))],
            "r", [5, 6], torch.tensor([[1, 2]], dtype=torch.long),
        )
        assert list(req.draft_tokens) == [5, 6]
        assert req.use_cache is True
        assert len(req.kv_cache.layers) == 1
        assert req.is_last_pass is True

    def test_no_kv_means_use_cache_false(self) -> None:
        pipeline = self._make()
        req = pipeline._prepare_forward_request(  # noqa: SLF001
            "n0", None, True, True, 1, 1, None, None, "r",
            None, torch.tensor([[1]], dtype=torch.long),
        )
        assert req.use_cache is False
        assert not len(req.kv_cache.layers)


# ── 4. _process_forward_response ─────────────────────────────────────────────


class TestProcessForwardResponse:
    def test_success_returns_output(self) -> None:
        rm = ResourceManager()
        pipeline = WideAreaPipeline(resource_mgr=rm)
        pipeline.register_node("n0", "h", 5000, 0, 3)
        node = pipeline.nodes["n0"]
        kv: dict[str, list | None] = {}
        out = pipeline._process_forward_response(  # noqa: SLF001
            _ok_resp(), "n0", node, kv
        )
        assert out.shape == (1, 3, HID)
        assert rm._node_failure_counts.get("n0", 0) == 0  # noqa: SLF001

    def test_error_response_raises_and_records_failure(self) -> None:
        rm = ResourceManager()
        pipeline = WideAreaPipeline(resource_mgr=rm)
        pipeline.register_node("n0", "h", 5000, 0, 3)
        node = pipeline.nodes["n0"]
        bad = node_pb2.ForwardPassResponse(success=False, error_message="boom")
        with pytest.raises(NodeUnreachableError) as exc_info:
            pipeline._process_forward_response(bad, "n0", node, {})  # noqa: SLF001
        assert "n0" in str(exc_info.value)
        # Original error text preserved on the exception object.
        assert "boom" in str(exc_info.value.original_error)  # noqa: SLF001
        assert node.is_healthy is False  # real field flipped by WAN wrapper
        assert not hasattr(node, "healthy") or node.healthy is False
        assert rm._node_failure_counts["n0"] == 1  # noqa: SLF001

    def test_kv_cache_extracted(self) -> None:
        rm = ResourceManager()
        pipeline = WideAreaPipeline(resource_mgr=rm)
        pipeline.register_node("n0", "h", 5000, 0, 3)
        node = pipeline.nodes["n0"]
        resp = node_pb2.ForwardPassResponse(success=True)
        layer = resp.kv_cache.layers.add()
        layer.key_states.CopyFrom(to_proto_tensor(torch.ones(1, 2, 4)))
        layer.value_states.CopyFrom(to_proto_tensor(torch.ones(1, 2, 4)))
        resp.output.CopyFrom(to_proto_tensor(torch.ones(1, 3, HID)))
        kv: dict[str, list | None] = {}
        pipeline._process_forward_response(resp, "n0", node, kv)  # noqa: SLF001
        cache = kv["n0"]
        assert cache is not None and len(cache) == 1
        k, v = cache[0]
        assert k.shape == (1, 2, 4) and v.shape == (1, 2, 4)


# ── 5. _find_fallback_node ───────────────────────────────────────────────────


class TestFindFallbackNode:
    def test_no_covering_candidate_returns_none(self) -> None:
        rm = ResourceManager()
        pipeline = WideAreaPipeline(resource_mgr=rm)
        pipeline.register_node("n0", "h0", 5000, 0, 3)
        fallback = pipeline._find_fallback_node("n0", pipeline.nodes["n0"])  # noqa: SLF001
        assert fallback is None

    def test_full_cover_elected(self) -> None:
        rm = ResourceManager(CircuitBreakerConfig(threshold=10))
        pipeline = WideAreaPipeline(resource_mgr=rm)
        pipeline.register_node("bad", "h0", 5000, 0, 5)
        pipeline.register_node("good", "h1", 5001, 0, 5)
        fallback = pipeline._find_fallback_node("bad", pipeline.nodes["bad"])  # noqa: SLF001
        assert fallback is not None and fallback.node_id == "good"

    def test_partial_cover_not_enough(self) -> None:
        rm = ResourceManager(CircuitBreakerConfig(threshold=10))
        pipeline = WideAreaPipeline(resource_mgr=rm)
        pipeline.register_node("bad", "h0", 5000, 0, 5)
        pipeline.register_node("partial", "h1", 5001, 0, 3)  # covers only 0-3
        fallback = pipeline._find_fallback_node("bad", pipeline.nodes["bad"])  # noqa: SLF001
        assert fallback is None

    def test_unhealthy_or_open_cb_candidates_excluded(self) -> None:
        rm = ResourceManager(CircuitBreakerConfig(threshold=1))
        pipeline = WideAreaPipeline(resource_mgr=rm)
        pipeline.register_node("bad", "h0", 5000, 0, 5)
        pipeline.register_node("good", "h1", 5001, 0, 5)
        pipeline.nodes["good"].is_healthy = False
        assert pipeline._find_fallback_node(  # noqa: SLF001
            "bad", pipeline.nodes["bad"]
        ) is None
        # Now healthy but CB open.
        pipeline.nodes["good"].is_healthy = True
        rm.record_failure("good")  # threshold=1 -> CB open
        assert pipeline._find_fallback_node(  # noqa: SLF001
            "bad", pipeline.nodes["bad"]
        ) is None

    def test_tightest_fit_wins(self) -> None:
        rm = ResourceManager(CircuitBreakerConfig(threshold=10))
        pipeline = WideAreaPipeline(resource_mgr=rm)
        pipeline.register_node("bad", "h0", 5000, 0, 5)
        pipeline.register_node("wide", "h1", 5001, 0, 9)
        pipeline.register_node("tight", "h2", 5002, 0, 5)
        fallback = pipeline._find_fallback_node("bad", pipeline.nodes["bad"])  # noqa: SLF001
        assert fallback is not None and fallback.node_id == "tight"


# ── 6. Integration: circuit breaker routing decisions inside the run loop ────


class TestCircuitBreakerRouting:
    def test_cb_open_no_fallback_no_local_raises(self) -> None:
        rm = ResourceManager(CircuitBreakerConfig(threshold=1))
        cfg = WideAreaConfig(fallback_to_local=False, transport="grpc")
        pipeline = WideAreaPipeline(resource_mgr=rm, wan_config=cfg)
        pipeline.register_node("n0", "127.0.0.1", 5000, 0, 3)
        rm.record_failure("n0")  # opens CB
        with pytest.raises(NodeUnreachableError) as exc_info:
            asyncio.run(
                pipeline.run_pipeline_async_p2p(
                    torch.tensor([[1, 2, 3]], dtype=torch.long), {}, "r"
                )
            )
        assert "n0" in str(exc_info.value)
        # Original reason ("Circuit breaker open for n0") rides along.
        assert "Circuit breaker open" in str(exc_info.value.original_error)

    def test_cb_open_with_covering_fallback_routes_around(self) -> None:
        rm = ResourceManager(CircuitBreakerConfig(threshold=1))
        pipeline = WideAreaPipeline(resource_mgr=rm)
        stubs: dict[str, _EchoStub] = {}
        for nid, host, sl, el in (
            ("bad", "h0", 0, 5), ("good", "h1", 0, 5),
        ):
            pipeline.register_node(nid, host, 5000 + sl, sl, el)
            client = _EchoClient()
            stubs[nid] = client.stub
            pipeline.nodes[nid].client = client
        rm.record_failure("bad")
        out = asyncio.run(
            pipeline.run_pipeline_async_p2p(
                torch.tensor([[1, 2, 3]], dtype=torch.long), {}, "r"
            )
        )
        assert out.shape == (1, 3, HID)
        assert not stubs["bad"].requests  # dead node never dialed
        # "good" covers bad's full range, so it serves both logical stages:
        # first hop (ids) and second hop (hidden states).
        assert len(stubs["good"].requests) == 2
        assert len(stubs["good"].requests[0].input_ids) == 3
        assert not len(stubs["good"].requests[1].input_ids)

    def test_timeout_marks_node_unhealthy_via_is_healthy(self) -> None:
        """Regression: handler used to set a dead ``healthy`` attribute."""
        rm = ResourceManager()
        cfg = WideAreaConfig(wan_timeout_seconds=0.05, transport="grpc")
        pipeline = WideAreaPipeline(resource_mgr=rm, wan_config=cfg)
        pipeline.register_node("n0", "127.0.0.1", 5000, 0, 3)

        class SlowStub:
            @staticmethod
            def ForwardPass(_request):
                time.sleep(0.25)
                return _ok_resp()

        pipeline.nodes["n0"].client = type("C", (), {"stub": SlowStub()})()
        with pytest.raises(NodeUnreachableError):
            asyncio.run(
                pipeline.run_pipeline_async_p2p(
                    torch.tensor([[1, 2, 3]], dtype=torch.long), {}, "r"
                )
            )
        assert pipeline.nodes["n0"].is_healthy is False  # real field flipped
        assert not hasattr(pipeline.nodes["n0"], "healthy")
        assert rm._node_failure_counts["n0"] == 1  # noqa: SLF001


# ── 7. QUIC branch: previously crashed on type(request).Response() ───────────


class _FakeQuicReal:
    """Mirrors QuicTransportClient's actual interface."""

    def __init__(self) -> None:
        self.is_connected = True
        self.requests: list[bytes] = []

    async def forward_pass(self, data: bytes, timeout: float = 120.0) -> bytes:
        self.requests.append(data)
        return _ok_resp().SerializeToString()


class _FakeQuicInjected:
    """Injected-double interface (send_forward_pass + is_available)."""

    is_available = True

    def __init__(self) -> None:
        self.n = 0

    async def send_forward_pass(self, data: bytes) -> bytes:
        self.n += 1
        return _ok_resp().SerializeToString()


class TestQuicBranch:
    def test_real_quic_interface_round_trip(self) -> None:
        qt = _FakeQuicReal()
        pipeline = WideAreaPipeline(resource_mgr=ResourceManager(), quic_transport=qt)
        pipeline.register_node("n0", "h0", 5000, 0, 3)
        out = asyncio.run(
            pipeline.run_pipeline_async_p2p(
                torch.tensor([[1, 2, 3]], dtype=torch.long), {}, "r"
            )
        )
        assert out.shape == (1, 3, HID)
        assert len(qt.requests) == 1
        req_back = node_pb2.ForwardPassRequest()
        req_back.ParseFromString(qt.requests[0])
        assert list(req_back.input_ids) == [1, 2, 3]

    def test_injected_double_send_forward_pass(self) -> None:
        qt = _FakeQuicInjected()
        pipeline = WideAreaPipeline(resource_mgr=ResourceManager(), quic_transport=qt)
        pipeline.register_node("n0", "h0", 5000, 0, 3)
        out = asyncio.run(
            pipeline.run_pipeline_async_p2p(
                torch.tensor([[1, 2, 3]], dtype=torch.long), {}, "r"
            )
        )
        assert out.shape == (1, 3, HID)
        assert qt.n == 1

    def test_string_sentinel_uses_grpc_path(self) -> None:
        """Old tests passed plain strings as transports — must not route QUIC."""
        pipeline = WideAreaPipeline(
            resource_mgr=ResourceManager(), quic_transport="dummy-transport"
        )
        pipeline.register_node("n0", "h0", 5000, 0, 3)
        client = _EchoClient()
        pipeline.nodes["n0"].client = client
        out = asyncio.run(
            pipeline.run_pipeline_async_p2p(
                torch.tensor([[1, 2, 3]], dtype=torch.long), {}, "r"
            )
        )
        assert out.shape == (1, 3, HID)
        assert len(client.stub.requests) == 1


# ── 8. Accumulated entry point (also crashed pre-fix via async p2p) ──────────


class TestAccumulatedEntry:
    def test_accumulated_without_draft_delegates_to_p2p(self) -> None:
        pipeline, (stub,) = _grpc_pipeline(num_nodes=1)
        out = asyncio.run(
            pipeline.run_pipeline_accumulated(
                torch.tensor([[1, 2, 3]], dtype=torch.long), {}, "req-a"
            )
        )
        assert out.shape == (1, 3, HID)
