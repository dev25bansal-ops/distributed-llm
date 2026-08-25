"""Tests for distllm.dist.pipeline.orchestrator module.

Pure unit tests using mocks for all external dependencies (gRPC calls,
resource manager, straggler detector, etc.). No real network or GPU
communication involved.

Test coverage:
  1. Pipeline initialization and configuration
  2. Node addition and removal
  3. Batch scheduling (1F1B)
  4. Error handling for pipeline failures
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch

from distllm.dist.pipeline.orchestrator import (
    PipelineError,
    PipelineNode,
    PipelineOrchestrator,
)


# ---------------------------------------------------------------------------
# Constants / helpers
# ---------------------------------------------------------------------------

NODE_0_ARGS = ("node-0", "10.0.0.1", 50051, 0, 15)
NODE_1_ARGS = ("node-1", "10.0.0.2", 50051, 16, 31)


def register_two_nodes(orch: PipelineOrchestrator) -> None:
    """Helper: register two non-overlapping nodes."""
    orch.register_node(*NODE_0_ARGS)
    orch.register_node(*NODE_1_ARGS)


def _sync_echo(
    host: str = "",
    port: int = 0,
    hidden_states: torch.Tensor | None = None,
    **kwargs: object,
) -> torch.Tensor:
    """Return a clone of the input (sync gRPC mock helper)."""
    if hidden_states is not None:
        return hidden_states.clone()
    return torch.tensor([[0]])


async def _async_echo(
    host: str = "",
    port: int = 0,
    hidden_states: torch.Tensor | None = None,
    **kwargs: object,
) -> torch.Tensor:
    """Return a clone of the input (async gRPC mock helper)."""
    if hidden_states is not None:
        return hidden_states.clone()
    return torch.tensor([[0]])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_rm() -> MagicMock:
    return MagicMock()


@pytest.fixture
def orch(mock_rm: MagicMock) -> PipelineOrchestrator:
    return PipelineOrchestrator(resource_mgr=mock_rm)


@pytest.fixture
def sample_tensor() -> torch.Tensor:
    return torch.tensor([[1, 2, 3, 4]], dtype=torch.long)


@pytest.fixture
def kv_default() -> dict[str, None]:
    return {"node-0": None, "node-1": None}


# ===================================================================
# 1. Pipeline initialization and configuration
# ===================================================================


class TestInitAndConfig:
    """Construction, property defaults, setters, shutdown."""

    def test_default_constructor(self) -> None:
        """Default values match the class docstring."""
        orch = PipelineOrchestrator()
        assert orch._resource_mgr is None
        assert orch._timeout == 30.0
        assert orch._redundancy == 1
        assert orch._default_micro_batch_size == 4
        assert orch._max_inflight == 8
        assert orch._nodes == {}
        assert orch._node_order == []
        assert orch._total_layers == 0
        assert orch._latency_tracker is None
        assert orch._straggler_detector is None
        assert orch._wan is None

    def test_custom_constructor(self) -> None:
        """All parameters can be overridden at construction."""
        rm = MagicMock()
        orch = PipelineOrchestrator(
            resource_mgr=rm,
            pipeline_timeout=120.0,
            redundancy=2,
            default_micro_batch_size=16,
            max_inflight_micro_batches=32,
        )
        assert orch._resource_mgr is rm
        assert orch._timeout == 120.0
        assert orch._redundancy == 2
        assert orch._default_micro_batch_size == 16
        assert orch._max_inflight == 32

    def test_zero_timeout_and_redundancy(self) -> None:
        """Edge-case: zero values are accepted."""
        orch = PipelineOrchestrator(
            pipeline_timeout=0.0,
            redundancy=0,
            default_micro_batch_size=1,
            max_inflight_micro_batches=1,
        )
        assert orch._timeout == 0.0
        assert orch._redundancy == 0
        assert orch._default_micro_batch_size == 1
        assert orch._max_inflight == 1

    def test_no_resource_mgr_still_runs(self, sample_tensor: torch.Tensor) -> None:
        """Pipeline works without a resource manager."""
        orch = PipelineOrchestrator()
        orch.register_node("n", "h", 1, 0, 15)
        with patch(
            "distllm.dist.node_client.forward_request",
            side_effect=_sync_echo,
        ):
            result = orch.run_pipeline(
                sample_tensor, {"n": None}, "req-no-rm"
            )
        assert result is not None

    def test_stats_initial(self, orch: PipelineOrchestrator) -> None:
        """All counters start at zero / empty."""
        s = orch.stats()
        assert s["pipeline_runs"] == 0
        assert s["total_latency_ms"] == 0.0
        assert s["errors"] == 0
        assert s["micro_batched_runs"] == 0
        assert s["avg_micro_batch_size"] == 0.0
        assert s["micro_batch_count_total"] == 0
        assert s["dynamic_batch_adjustments"] == []
        assert s["node_count"] == 0
        assert s["healthy_nodes"] == 0
        assert s["avg_latency_ms"] == 0.0
        assert s["total_layers"] == 0
        assert s["micro_batched_enabled"] is True

    def test_nodes_property_empty(self, orch: PipelineOrchestrator) -> None:
        assert orch.nodes == {}

    def test_nodes_property_snapshot(self, orch: PipelineOrchestrator) -> None:
        """Accessing .nodes returns a snapshot, not the live dict."""
        register_two_nodes(orch)
        snap = orch.nodes
        orch.register_node("n2", "h", 1, 32, 47)
        assert "n2" not in snap

    def test_nodes_property_keys(self, orch: PipelineOrchestrator) -> None:
        register_two_nodes(orch)
        entry = orch.nodes["node-0"]
        assert entry == {
            "host": "10.0.0.1",
            "port": 50051,
            "start_layer": 0,
            "end_layer": 15,
            "healthy": True,
        }

    def test_node_order_property(self, orch: PipelineOrchestrator) -> None:
        register_two_nodes(orch)
        assert orch.node_order == ["node-0", "node-1"]

    def test_node_order_snapshot(self, orch: PipelineOrchestrator) -> None:
        register_two_nodes(orch)
        copy_ = orch.node_order
        orch.register_node("n2", "h", 1, 32, 47)
        assert len(copy_) == 2

    def test_total_layers_explicit(self, orch: PipelineOrchestrator) -> None:
        orch.total_layers = 99
        assert orch.total_layers == 99

    def test_total_layers_derived(self, orch: PipelineOrchestrator) -> None:
        register_two_nodes(orch)
        # max(end_layer) + 1 = 31 + 1 = 32
        assert orch.total_layers == 32

    def test_total_layers_empty(self) -> None:
        assert PipelineOrchestrator().total_layers == 0

    def test_pipeline_timeout_live(self, orch: PipelineOrchestrator) -> None:
        orch.pipeline_timeout = 7.5
        assert orch.pipeline_timeout == 7.5

    def test_wan_default(self) -> None:
        assert PipelineOrchestrator().wan is None

    def test_wan_setter(self, orch: PipelineOrchestrator) -> None:
        w = MagicMock()
        orch.wan = w
        assert orch.wan is w

    def test_set_latency_tracker(self, orch: PipelineOrchestrator) -> None:
        lt = MagicMock()
        orch.set_latency_tracker(lt)
        assert orch._latency_tracker is lt

    def test_set_straggler_detector(self, orch: PipelineOrchestrator) -> None:
        sd = MagicMock()
        orch.set_straggler_detector(sd)
        assert orch._straggler_detector is sd

    def test_shutdown_clears(self, orch: PipelineOrchestrator) -> None:
        register_two_nodes(orch)
        orch.shutdown()
        assert orch._nodes == {}
        assert orch._node_order == []

    def test_shutdown_empty_is_safe(self) -> None:
        PipelineOrchestrator().shutdown()

    def test_shutdown_logs(self, orch: PipelineOrchestrator) -> None:
        with patch("distllm.dist.pipeline.orchestrator.logger.info") as log:
            orch.shutdown()
        log.assert_called_once_with("Pipeline orchestrator shut down")

    def test_pipeline_node_minimal(self) -> None:
        n = PipelineNode("n", "h", 1, 0, 15)
        assert n.node_id == "n"
        assert n.is_healthy is True
        assert n.latency_ms == 0.0
        assert isinstance(n.last_heartbeat, float)

    def test_pipeline_node_all_fields(self) -> None:
        n = PipelineNode(
            node_id="x",
            host="host",
            port=8080,
            start_layer=0,
            end_layer=31,
            total_layers=32,
            is_healthy=False,
            last_heartbeat=42.0,
            latency_ms=12.3,
        )
        assert n.total_layers == 32
        assert n.is_healthy is False
        assert n.latency_ms == 12.3
        assert n.last_heartbeat == 42.0

    def test_pipeline_node_default_heartbeat_recent(self) -> None:
        before = time.time()
        n = PipelineNode("n", "h", 1, 0, 1)
        after = time.time()
        assert before <= n.last_heartbeat <= after


# ===================================================================
# 2. Node addition and removal
# ===================================================================


class TestNodeLifecycle:
    """register_node, unregister_node, remove_node, get_node,
    validate_layer_assignment, health markers."""

    def test_register_node(self, orch: PipelineOrchestrator) -> None:
        orch.register_node("n", "10.0.43.1", 50052, 4, 11, total_layers=48)
        node = orch._nodes["n"]
        assert node.host == "10.0.43.1"
        assert node.port == 50052
        assert node.start_layer == 4
        assert node.end_layer == 11
        assert node.total_layers == 48
        assert orch.node_order == ["n"]

    def test_register_node_sorts(self, orch: PipelineOrchestrator) -> None:
        orch.register_node("b", "h", 1, 16, 31)
        orch.register_node("a", "h", 1, 0, 15)
        assert orch.node_order == ["a", "b"]

    def test_register_node_maintains_order_on_equal_start(
        self, orch: PipelineOrchestrator
    ) -> None:
        orch.register_node("first", "h", 1, 0, 7)
        orch.register_node("second", "h", 1, 0, 7)
        assert orch.node_order == ["first", "second"]

    def test_register_node_extra_kwargs(self, orch: PipelineOrchestrator) -> None:
        orch.register_node("n", "h", 1, 0, 15, unused="ignored")
        assert "n" in orch._nodes

    def test_register_replaces(self, orch: PipelineOrchestrator) -> None:
        orch.register_node("n", "old", 1, 0, 15)
        orch.register_node("n", "new", 2, 0, 15)
        assert orch._nodes["n"].host == "new"
        assert orch._nodes["n"].port == 2

    def test_unregister(self, orch: PipelineOrchestrator) -> None:
        register_two_nodes(orch)
        orch.unregister_node("node-0")
        assert "node-0" not in orch._nodes
        assert orch.node_order == ["node-1"]

    def test_unregister_unknown_safe(self, orch: PipelineOrchestrator) -> None:
        orch.unregister_node("ghost")
        assert orch._nodes == {}

    def test_remove_node_alias(self, orch: PipelineOrchestrator) -> None:
        register_two_nodes(orch)
        orch.remove_node("node-1")
        assert orch.node_order == ["node-0"]

    def test_get_node(self, orch: PipelineOrchestrator) -> None:
        register_two_nodes(orch)
        n = orch.get_node("node-0")
        assert n is not None
        assert n.node_id == "node-0"

    def test_get_node_missing(self, orch: PipelineOrchestrator) -> None:
        assert orch.get_node("ghost") is None

    def test_validate_no_overlap(self, orch: PipelineOrchestrator) -> None:
        register_two_nodes(orch)
        orch.validate_layer_assignment("n2", 32, 47)

    def test_validate_overlap_raises(self, orch: PipelineOrchestrator) -> None:
        register_two_nodes(orch)
        with pytest.raises(ValueError, match="Layer overlap"):
            orch.validate_layer_assignment("intruder", 10, 20)

    def test_validate_exact_overlap_raises(self, orch: PipelineOrchestrator) -> None:
        register_two_nodes(orch)
        with pytest.raises(ValueError, match="Layer overlap"):
            orch.validate_layer_assignment("intruder", 0, 15)

    def test_validate_adjacent_allowed(self, orch: PipelineOrchestrator) -> None:
        register_two_nodes(orch)
        orch.validate_layer_assignment("n2", 32, 47)

    def test_validate_skips_self(self, orch: PipelineOrchestrator) -> None:
        register_two_nodes(orch)
        orch.validate_layer_assignment("node-0", 0, 15)

    def test_validate_empty_pipeline(self, orch: PipelineOrchestrator) -> None:
        orch.validate_layer_assignment("n", 0, 15)

    def test_validate_error_message(self, orch: PipelineOrchestrator) -> None:
        register_two_nodes(orch)
        with pytest.raises(ValueError) as exc:
            orch.validate_layer_assignment("x", 12, 18)
        msg = str(exc.value)
        assert "Layer overlap" in msg
        assert "12-18" in msg

    def test_healthy_initially(self, orch: PipelineOrchestrator) -> None:
        register_two_nodes(orch)
        assert orch.get_healthy_nodes() == ["node-0", "node-1"]

    def test_mark_unhealthy(self, orch: PipelineOrchestrator) -> None:
        register_two_nodes(orch)
        orch.mark_node_unhealthy("node-0")
        assert orch.get_healthy_nodes() == ["node-1"]

    def test_mark_unhealthy_unknown_safe(self, orch: PipelineOrchestrator) -> None:
        orch.mark_node_unhealthy("ghost")

    def test_mark_healthy_after_unhealthy(self, orch: PipelineOrchestrator) -> None:
        register_two_nodes(orch)
        orch.mark_node_unhealthy("node-0")
        orch.mark_node_healthy("node-0")
        assert orch._nodes["node-0"].is_healthy is True

    def test_mark_healthy_unknown_safe(self, orch: PipelineOrchestrator) -> None:
        orch.mark_node_healthy("ghost")

    def test_stats_after_management(self, orch: PipelineOrchestrator) -> None:
        register_two_nodes(orch)
        assert orch.stats()["healthy_nodes"] == 2
        orch.mark_node_unhealthy("node-0")
        assert orch.stats()["healthy_nodes"] == 1
        orch.unregister_node("node-0")
        assert orch.stats()["node_count"] == 1

    def test_register_logs(self, orch: PipelineOrchestrator) -> None:
        with patch("distllm.dist.pipeline.orchestrator.logger.info") as log:
            orch.register_node("n", "h", 1, 0, 15)
        log.assert_called_once_with("Pipeline: registered n (layers 0-15)")


# ===================================================================
# 3. Batch scheduling (1F1B)
# ===================================================================


class TestSequentialPipeline:
    """Synchronous run_pipeline tests."""

    def test_success(
        self, orch: PipelineOrchestrator, sample_tensor: torch.Tensor, kv_default: dict
    ) -> None:
        register_two_nodes(orch)
        mid = torch.tensor([[5, 6, 7, 8]], dtype=torch.long)
        final = torch.tensor([[9, 10, 11, 12]], dtype=torch.long)

        with patch(
            "distllm.dist.node_client.forward_request",
            side_effect=[mid, final],
        ) as fwd:
            result = orch.run_pipeline(sample_tensor, kv_default, "r1")

        assert fwd.call_count == 2
        # check correct routing
        c0 = fwd.call_args_list[0][1]
        assert c0["host"] == "10.0.0.1"
        c1 = fwd.call_args_list[1][1]
        assert c1["host"] == "10.0.0.2"
        assert c1["hidden_states"] is mid
        torch.testing.assert_close(result, final)

    def test_success_resource_mgr_called(
        self, orch: PipelineOrchestrator, sample_tensor: torch.Tensor, kv_default: dict
    ) -> None:
        register_two_nodes(orch)
        with patch("distllm.dist.node_client.forward_request", side_effect=_sync_echo):
            orch.run_pipeline(sample_tensor, kv_default, "r1")
        assert orch._resource_mgr.record_success.call_count == 2
        orch._resource_mgr.record_success.assert_any_call("node-0")
        orch._resource_mgr.record_success.assert_any_call("node-1")

    def test_no_healthy_nodes(
        self, orch: PipelineOrchestrator, sample_tensor: torch.Tensor
    ) -> None:
        register_two_nodes(orch)
        orch.mark_node_unhealthy("node-0")
        orch.mark_node_unhealthy("node-1")
        with pytest.raises(RuntimeError, match="No healthy nodes"):
            orch.run_pipeline(sample_tensor, {}, "r1")

    def test_skips_unhealthy(
        self, orch: PipelineOrchestrator, sample_tensor: torch.Tensor
    ) -> None:
        register_two_nodes(orch)
        orch.mark_node_unhealthy("node-0")
        with patch(
            "distllm.dist.node_client.forward_request", side_effect=_sync_echo
        ) as fwd:
            result = orch.run_pipeline(
                sample_tensor, {"node-0": None, "node-1": None}, "r1"
            )
        assert fwd.call_count == 1
        assert fwd.call_args[1]["host"] == "10.0.0.2"
        assert result is not None

    def test_updates_latency(
        self, orch: PipelineOrchestrator, sample_tensor: torch.Tensor, kv_default: dict
    ) -> None:
        register_two_nodes(orch)
        with patch("distllm.dist.node_client.forward_request", side_effect=_sync_echo):
            orch.run_pipeline(sample_tensor, kv_default, "r1")
        for nid in ("node-0", "node-1"):
            assert orch._nodes[nid].latency_ms > 0.0

    def test_single_node(self, orch: PipelineOrchestrator) -> None:
        orch.register_node("only", "h", 1, 0, 31)
        inp = torch.tensor([[1, 2, 3]], dtype=torch.long)
        with patch(
            "distllm.dist.node_client.forward_request", return_value=inp.clone()
        ) as fwd:
            result = orch.run_pipeline(inp, {"only": None}, "r1")
        assert fwd.call_count == 1
        torch.testing.assert_close(result, inp)


class TestMicrobatchedPipeline:
    """Async 1F1B micro-batched execution."""

    @pytest.mark.asyncio
    async def test_success(
        self, orch: PipelineOrchestrator, kv_default: dict
    ) -> None:
        register_two_nodes(orch)
        inp = torch.randn(8, 128)  # 4 micro-batches of size 2

        with patch(
            "distllm.dist.node_client.forward_request_async",
            side_effect=_async_echo,
        ) as fwd:
            result = await orch.run_pipeline_microbatched(
                inp, kv_default, "r1", micro_batch_size=2,
            )

        # 2 stages, 4 batches: 2 warmup + 6 steady = 8 steps
        assert fwd.call_count == 8
        assert result.shape == (8, 128)
        assert orch.stats()["micro_batched_runs"] == 1
        assert orch.stats()["micro_batch_count_total"] == 4

    @pytest.mark.asyncio
    async def test_no_healthy_nodes(self, orch: PipelineOrchestrator) -> None:
        register_two_nodes(orch)
        orch.mark_node_unhealthy("node-0")
        orch.mark_node_unhealthy("node-1")
        with pytest.raises(RuntimeError, match="No healthy nodes"):
            await orch.run_pipeline_microbatched(
                torch.randn(4, 128), {}, "r1",
            )

    @pytest.mark.asyncio
    async def test_resource_mgr_called(
        self, orch: PipelineOrchestrator, kv_default: dict
    ) -> None:
        register_two_nodes(orch)
        with patch(
            "distllm.dist.node_client.forward_request_async",
            side_effect=_async_echo,
        ):
            await orch.run_pipeline_microbatched(
                torch.randn(4, 128), kv_default, "r1", micro_batch_size=2,
            )
        assert orch._resource_mgr.record_success.call_count >= 2

    @pytest.mark.asyncio
    async def test_default_micro_batch_size(
        self, orch: PipelineOrchestrator, kv_default: dict
    ) -> None:
        """When micro_batch_size is None, constructor default (4) is used."""
        register_two_nodes(orch)
        inp = torch.randn(8, 128)
        with patch(
            "distllm.dist.node_client.forward_request_async",
            side_effect=_async_echo,
        ) as fwd:
            await orch.run_pipeline_microbatched(inp, kv_default, "r1")
        # 8/4 = 2 batches, 2 stages = 4 steps
        assert fwd.call_count == 4

    # -- Schedule step counts (1F1B arrangement) -----------------------

    @pytest.mark.asyncio
    async def test_schedule_2stages_4batches(
        self, orch: PipelineOrchestrator, kv_default: dict
    ) -> None:
        register_two_nodes(orch)
        with patch(
            "distllm.dist.node_client.forward_request_async",
            side_effect=_async_echo,
        ) as fwd:
            await orch.run_pipeline_microbatched(
                torch.randn(8, 128), kv_default, "r1", micro_batch_size=2,
            )
        # Warmup: (0,0),(1,0) = 2  Steady: b=1,2,3 x 2 stages each = 6  Total: 8
        assert fwd.call_count == 8

    @pytest.mark.asyncio
    async def test_schedule_2stages_2batches(
        self, orch: PipelineOrchestrator, kv_default: dict
    ) -> None:
        register_two_nodes(orch)
        with patch(
            "distllm.dist.node_client.forward_request_async",
            side_effect=_async_echo,
        ) as fwd:
            await orch.run_pipeline_microbatched(
                torch.randn(4, 128), kv_default, "r1", micro_batch_size=2,
            )
        # Warmup: 2  Steady: b=1 x 2 stages = 2  Total: 4
        assert fwd.call_count == 4

    @pytest.mark.asyncio
    async def test_schedule_3stages_4batches(self, orch: PipelineOrchestrator) -> None:
        orch.register_node("s0", "h", 1, 0, 10)
        orch.register_node("s1", "h", 1, 11, 20)
        orch.register_node("s2", "h", 1, 21, 30)
        kvs = {"s0": None, "s1": None, "s2": None}
        with patch(
            "distllm.dist.node_client.forward_request_async",
            side_effect=_async_echo,
        ) as fwd:
            await orch.run_pipeline_microbatched(
                torch.randn(8, 128), kvs, "r1", micro_batch_size=2,
            )
        # Warmup: 3  Steady: 3 batches x 3 stages = 9  Total: 12
        assert fwd.call_count == 12
        assert fwd.call_args_list[0][1]["host"] == "h"

    @pytest.mark.asyncio
    async def test_schedule_1stage_4batches(self, orch: PipelineOrchestrator) -> None:
        orch.register_node("only", "h", 1, 0, 31)
        with patch(
            "distllm.dist.node_client.forward_request_async",
            side_effect=_async_echo,
        ) as fwd:
            await orch.run_pipeline_microbatched(
                torch.randn(8, 128), {"only": None}, "r1", micro_batch_size=2,
            )
        # Warmup: 1  Steady: 3 batches x 1 stage = 3  Total: 4
        assert fwd.call_count == 4

    # -- Dynamic micro-batch sizing ------------------------------------

    @pytest.mark.asyncio
    async def test_no_straggler_detector_no_adjustment(
        self, orch: PipelineOrchestrator, kv_default: dict
    ) -> None:
        register_two_nodes(orch)
        orch._straggler_detector = None
        with patch(
            "distllm.dist.node_client.forward_request_async",
            side_effect=_async_echo,
        ) as fwd:
            await orch.run_pipeline_microbatched(
                torch.randn(8, 128), kv_default, "r1", micro_batch_size=4,
            )
        assert fwd.call_count == 4
        assert orch._stats["dynamic_batch_adjustments"] == []

    @pytest.mark.asyncio
    async def test_straggler_moderate_halves(
        self, orch: PipelineOrchestrator, kv_default: dict
    ) -> None:
        register_two_nodes(orch)
        report = SimpleNamespace(severity=SimpleNamespace(value="moderate"))
        orch._straggler_detector = MagicMock(
            get_reports=MagicMock(return_value=[report]),
            stats=MagicMock(return_value={"nodes": {}}),
        )
        with patch(
            "distllm.dist.node_client.forward_request_async",
            side_effect=_async_echo,
        ) as fwd:
            await orch.run_pipeline_microbatched(
                torch.randn(8, 128), kv_default, "r1", micro_batch_size=4,
            )
        # half → 2, so 8/2=4 batches × 2 stages = 8 steps
        assert fwd.call_count == 8
        adj = orch._stats["dynamic_batch_adjustments"][0]
        assert adj["adjustment"] == "straggler_reduce_50"
        assert adj["micro_batch_size"] == 2

    @pytest.mark.asyncio
    async def test_straggler_severe_halves(
        self, orch: PipelineOrchestrator, kv_default: dict
    ) -> None:
        register_two_nodes(orch)
        report = SimpleNamespace(severity=SimpleNamespace(value="severe"))
        orch._straggler_detector = MagicMock(
            get_reports=MagicMock(return_value=[report]),
            stats=MagicMock(return_value={"nodes": {}}),
        )
        with patch(
            "distllm.dist.node_client.forward_request_async",
            side_effect=_async_echo,
        ):
            await orch.run_pipeline_microbatched(
                torch.randn(8, 128), kv_default, "r1", micro_batch_size=4,
            )
        assert orch._stats["dynamic_batch_adjustments"][0]["adjustment"] == "straggler_reduce_50"

    @pytest.mark.asyncio
    async def test_straggler_mild_no_change(
        self, orch: PipelineOrchestrator, kv_default: dict
    ) -> None:
        register_two_nodes(orch)
        report = SimpleNamespace(severity=SimpleNamespace(value="mild"))
        orch._straggler_detector = MagicMock(
            get_reports=MagicMock(return_value=[report]),
        )
        with patch(
            "distllm.dist.node_client.forward_request_async",
            side_effect=_async_echo,
        ) as fwd:
            await orch.run_pipeline_microbatched(
                torch.randn(8, 128), kv_default, "r1", micro_batch_size=4,
            )
        assert orch._stats["dynamic_batch_adjustments"] == []
        assert fwd.call_count == 4

    @pytest.mark.asyncio
    async def test_low_latency_increases_25(
        self, orch: PipelineOrchestrator, kv_default: dict
    ) -> None:
        register_two_nodes(orch)
        mock_sd = MagicMock()
        mock_sd.get_reports = MagicMock(return_value=[])
        mock_sd.stats = MagicMock(
            return_value={
                "nodes": {
                    "node-0": {"avg_latency": 30.0},
                    "node-1": {"avg_latency": 20.0},
                },
            }
        )
        orch._straggler_detector = mock_sd
        # 4 * 1.25 = 5, and with 16 tokens total we won't be clamped
        with patch(
            "distllm.dist.node_client.forward_request_async",
            side_effect=_async_echo,
        ):
            await orch.run_pipeline_microbatched(
                torch.randn(16, 128), kv_default, "r1", micro_batch_size=4,
            )
        adj = orch._stats["dynamic_batch_adjustments"][0]
        assert adj["adjustment"] == "low_latency_increase_25"
        assert adj["micro_batch_size"] == 5

    @pytest.mark.asyncio
    async def test_detector_exception_fallback(
        self, orch: PipelineOrchestrator, kv_default: dict
    ) -> None:
        register_two_nodes(orch)
        orch._straggler_detector = MagicMock(
            get_reports=MagicMock(side_effect=ValueError("boom")),
        )
        with patch(
            "distllm.dist.node_client.forward_request_async",
            side_effect=_async_echo,
        ) as fwd:
            await orch.run_pipeline_microbatched(
                torch.randn(8, 128), kv_default, "r1", micro_batch_size=4,
            )
        assert orch._stats["dynamic_batch_adjustments"] == []
        assert fwd.call_count == 4

    @pytest.mark.asyncio
    async def test_clamp_to_half_total(
        self, orch: PipelineOrchestrator, kv_default: dict
    ) -> None:
        register_two_nodes(orch)
        with patch(
            "distllm.dist.node_client.forward_request_async",
            side_effect=_async_echo,
        ) as fwd:
            await orch.run_pipeline_microbatched(
                torch.randn(6, 128), kv_default, "r1", micro_batch_size=10,
            )
        # clamped to min(10, 6//2=3) = 3 → 6/3=2 batches × 2 stages = 4 steps
        assert fwd.call_count == 4

    @pytest.mark.asyncio
    async def test_batch_size_min_one(self, orch: PipelineOrchestrator) -> None:
        orch.register_node("n", "h", 1, 0, 15)
        with patch(
            "distllm.dist.node_client.forward_request_async",
            side_effect=_async_echo,
        ) as fwd:
            await orch.run_pipeline_microbatched(
                torch.randn(2, 128), {"n": None}, "r1", micro_batch_size=1,
            )
        # 2/1=2 batches × 1 stage = 2 steps
        assert fwd.call_count == 2

    @pytest.mark.asyncio
    async def test_empty_reports_no_node_info(
        self, orch: PipelineOrchestrator, kv_default: dict
    ) -> None:
        register_two_nodes(orch)
        orch._straggler_detector = MagicMock(
            get_reports=MagicMock(return_value=[]),
            stats=MagicMock(return_value={"nodes": {}}),
        )
        with patch(
            "distllm.dist.node_client.forward_request_async",
            side_effect=_async_echo,
        ) as fwd:
            await orch.run_pipeline_microbatched(
                torch.randn(8, 128), kv_default, "r1", micro_batch_size=4,
            )
        assert orch._stats["dynamic_batch_adjustments"] == []
        assert fwd.call_count == 4


# ===================================================================
# 4. Error handling for pipeline failures
# ===================================================================


class TestErrorHandling:
    """Pipeline failures: None returns, exceptions, timeouts, partial
    microbatch failures, resource manager recording."""

    # -- Sequential pipeline errors ------------------------------------

    def test_node_returns_none(
        self, orch: PipelineOrchestrator, sample_tensor: torch.Tensor, kv_default: dict
    ) -> None:
        register_two_nodes(orch)
        with patch(
            "distllm.dist.node_client.forward_request",
            return_value=None,
        ):
            with pytest.raises(RuntimeError, match="returned None"):
                orch.run_pipeline(sample_tensor, kv_default, "r1")
        assert orch.stats()["errors"] == 1

    def test_node_returns_none_mid_pipeline(
        self, orch: PipelineOrchestrator, sample_tensor: torch.Tensor, kv_default: dict
    ) -> None:
        """First node succeeds, second returns None."""
        register_two_nodes(orch)
        valid = torch.tensor([[5, 6, 7, 8]], dtype=torch.long)
        with patch(
            "distllm.dist.node_client.forward_request",
            side_effect=[valid, None],
        ):
            with pytest.raises(RuntimeError, match="returned None"):
                orch.run_pipeline(sample_tensor, kv_default, "r1")
        assert orch.stats()["errors"] == 1

    def test_node_raises(
        self, orch: PipelineOrchestrator, sample_tensor: torch.Tensor, kv_default: dict
    ) -> None:
        register_two_nodes(orch)
        with patch(
            "distllm.dist.node_client.forward_request",
            side_effect=RuntimeError("connection refused"),
        ):
            with pytest.raises(RuntimeError, match="connection refused"):
                orch.run_pipeline(sample_tensor, kv_default, "r1")
        assert orch.stats()["errors"] == 1
        orch._resource_mgr.record_failure.assert_called_once_with("node-0")

    def test_node_raises_mid_pipeline(
        self, orch: PipelineOrchestrator, sample_tensor: torch.Tensor, kv_default: dict
    ) -> None:
        """First node succeeds, second node raises."""
        register_two_nodes(orch)
        valid = torch.tensor([[5, 6, 7, 8]], dtype=torch.long)
        with patch(
            "distllm.dist.node_client.forward_request",
            side_effect=[valid, RuntimeError("node-1 crash")],
        ):
            with pytest.raises(RuntimeError, match="node-1 crash"):
                orch.run_pipeline(sample_tensor, kv_default, "r1")
        assert orch.stats()["errors"] == 1
        orch._resource_mgr.record_success.assert_called_once_with("node-0")
        orch._resource_mgr.record_failure.assert_called_once_with("node-1")

    def test_errors_accumulate_across_runs(
        self, orch: PipelineOrchestrator, sample_tensor: torch.Tensor
    ) -> None:
        register_two_nodes(orch)
        with patch(
            "distllm.dist.node_client.forward_request",
            side_effect=RuntimeError("fail"),
        ):
            for _ in range(3):
                try:
                    orch.run_pipeline(sample_tensor, {}, "r1")
                except RuntimeError:
                    pass
        assert orch.stats()["pipeline_runs"] == 3
        assert orch.stats()["errors"] == 3

    # -- Micro-batched pipeline errors ---------------------------------

    @pytest.mark.asyncio
    async def test_all_batches_fail(
        self, orch: PipelineOrchestrator, kv_default: dict
    ) -> None:
        """All RPCs raise at stage 0 → downstream cascades (no RPC) →
        PipelineError covering every sequence.  No dependency stall."""
        fast = PipelineOrchestrator(pipeline_timeout=0.5)
        register_two_nodes(fast)
        with patch(
            "distllm.dist.node_client.forward_request_async",
            side_effect=RuntimeError("node failure"),
        ):
            with pytest.raises(
                PipelineError, match="All micro-batches failed"
            ):
                await fast.run_pipeline_microbatched(
                    torch.randn(4, 128), kv_default, "r1", micro_batch_size=2,
                )
        assert fast.stats()["errors"] > 0

    @pytest.mark.asyncio
    async def test_partial_failure_raises_naming_failed_rows(
        self, orch: PipelineOrchestrator, kv_default: dict
    ) -> None:
        """One micro-batch fails at the last stage; the run must raise
        PipelineError naming the failed input rows instead of silently
        returning the surviving batch (legacy behaviour asserted a
        shape-(2,128) result here — that hid data loss from callers)."""
        register_two_nodes(orch)

        async def flaky_fwd(**kwargs: object) -> torch.Tensor:
            req_id = kwargs.get("request_id", "")
            if req_id == "r1-s1b0":
                raise RuntimeError("last-stage failure on batch 0")
            hs = kwargs.get("hidden_states")
            return hs.clone() if hs is not None else torch.randn(2, 128)

        with patch(
            "distllm.dist.node_client.forward_request_async",
            side_effect=flaky_fwd,
        ):
            with pytest.raises(PipelineError) as exc_info:
                await orch.run_pipeline_microbatched(
                    torch.randn(4, 128), kv_default, "r1", micro_batch_size=2,
                )
        exc = exc_info.value
        assert exc.failed_micro_batches == [0]
        assert exc.failed_sequences == [0, 1]
        assert orch.stats()["last_failed_sequences"] == [0, 1]

    @pytest.mark.asyncio
    async def test_timeout(
        self, orch: PipelineOrchestrator, kv_default: dict
    ) -> None:
        """A step that never completes raises TimeoutError."""
        fast = PipelineOrchestrator(pipeline_timeout=0.01)
        register_two_nodes(fast)

        async def never(**kwargs: object) -> torch.Tensor:
            await asyncio.sleep(3600)
            return torch.randn(2, 128)  # pragma: no cover

        with patch(
            "distllm.dist.node_client.forward_request_async",
            side_effect=never,
        ):
            with pytest.raises((TimeoutError, asyncio.TimeoutError)):
                await fast.run_pipeline_microbatched(
                    torch.randn(4, 128), kv_default, "r1", micro_batch_size=2,
                )

    @pytest.mark.asyncio
    async def test_microbatch_records_failure(
        self, orch: PipelineOrchestrator, kv_default: dict
    ) -> None:
        """All RPCs crash → PipelineError, but resource_mgr.record_failure
        should have been called before the raise."""
        fast = PipelineOrchestrator(pipeline_timeout=0.5, resource_mgr=MagicMock())
        register_two_nodes(fast)
        with patch(
            "distllm.dist.node_client.forward_request_async",
            side_effect=RuntimeError("crash"),
        ):
            with pytest.raises(PipelineError):
                await fast.run_pipeline_microbatched(
                    torch.randn(4, 128), kv_default, "r1", micro_batch_size=2,
                )
        fast._resource_mgr.record_failure.assert_called()

    @pytest.mark.asyncio
    async def test_single_microbatch_with_multi_stage_succeeds(
        self, orch: PipelineOrchestrator, kv_default: dict
    ) -> None:
        """num_batches < num_stages must still run EVERY stage on batch 0.

        Use total_tokens=1 so the clamp (max(1, …, total_tokens//2=0))
        yields micro_batch_size=1 → num_batches=1.  The warmup loop
        schedules all num_stages steps for batch 0 (regression guard for
        the old bug where capping stages by num_batches skipped the
        output-producing final stage), so the pipeline succeeds.
        """
        register_two_nodes(orch)
        with patch(
            "distllm.dist.node_client.forward_request_async",
            side_effect=_async_echo,
        ) as fwd:
            result = await orch.run_pipeline_microbatched(
                torch.randn(1, 128), kv_default, "r1", micro_batch_size=2,
            )
        # Warmup: (0,0),(1,0); steady loop empty → both stages ran.
        assert fwd.call_count == 2
        assert result.shape == (1, 128)
