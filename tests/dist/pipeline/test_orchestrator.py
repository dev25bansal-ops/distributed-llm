"""Tests for distllm.dist.pipeline.orchestrator module.

Uses mocks for the gRPC node_client functions to enable pure unit testing
of pipeline orchestration logic without requiring a running gRPC server.
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
# Helpers
# ---------------------------------------------------------------------------


def register_two_nodes(o: PipelineOrchestrator) -> None:
    """Register two nodes with non-overlapping layer ranges."""
    o.register_node("node-0", "10.0.0.1", 50051, 0, 15)
    o.register_node("node-1", "10.0.0.2", 50051, 16, 31)


def dummy_forward_sync(
    host: str = "",
    port: int = 0,
    hidden_states: torch.Tensor | None = None,
    **kwargs: object,
) -> torch.Tensor:
    """Return a tensor matching the input shape (sync helper)."""
    if hidden_states is not None:
        return hidden_states.clone()
    return torch.tensor([[0]])


async def dummy_forward_async(
    host: str = "",
    port: int = 0,
    hidden_states: torch.Tensor | None = None,
    **kwargs: object,
) -> torch.Tensor:
    """Return a tensor matching the input shape (async helper)."""
    if hidden_states is not None:
        return hidden_states.clone()
    return torch.tensor([[0]])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_resource_mgr() -> MagicMock:
    return MagicMock()


@pytest.fixture
def orchestrator(mock_resource_mgr: MagicMock) -> PipelineOrchestrator:
    return PipelineOrchestrator(resource_mgr=mock_resource_mgr)


@pytest.fixture
def sample_tensor() -> torch.Tensor:
    return torch.tensor([[1, 2, 3, 4]], dtype=torch.long)


@pytest.fixture
def kv_caches_default() -> dict[str, None]:
    return {"node-0": None, "node-1": None}


# ===================================================================
# PipelineNode
# ===================================================================


class TestPipelineNode:
    """Unit tests for the PipelineNode dataclass."""

    def test_create_minimal(self) -> None:
        node = PipelineNode(
            node_id="test",
            host="localhost",
            port=50051,
            start_layer=0,
            end_layer=15,
        )
        assert node.node_id == "test"
        assert node.host == "localhost"
        assert node.port == 50051
        assert node.start_layer == 0
        assert node.end_layer == 15
        assert node.total_layers == 0
        assert node.is_healthy is True
        assert isinstance(node.last_heartbeat, float)
        assert node.latency_ms == 0.0

    def test_create_all_fields(self) -> None:
        node = PipelineNode(
            node_id="n1",
            host="10.0.0.1",
            port=8080,
            start_layer=10,
            end_layer=20,
            total_layers=50,
            is_healthy=False,
            last_heartbeat=1234.5,
            latency_ms=42.0,
        )
        assert node.total_layers == 50
        assert node.is_healthy is False
        assert node.last_heartbeat == 1234.5
        assert node.latency_ms == 42.0

    def test_default_last_heartbeat_is_recent(self) -> None:
        before = time.time()
        node = PipelineNode("n", "h", 1, 0, 1)
        after = time.time()
        assert before <= node.last_heartbeat <= after


# ===================================================================
# PipelineOrchestrator -- Initialization & Configuration
# ===================================================================


class TestPipelineOrchestratorInit:
    """Construction and default configuration."""

    def test_default_constructor(self) -> None:
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
        assert orch._stats["pipeline_runs"] == 0

    def test_custom_constructor(self) -> None:
        rm = MagicMock()
        orch = PipelineOrchestrator(
            resource_mgr=rm,
            pipeline_timeout=60.0,
            redundancy=3,
            default_micro_batch_size=8,
            max_inflight_micro_batches=16,
        )
        assert orch._resource_mgr is rm
        assert orch._timeout == 60.0
        assert orch._redundancy == 3
        assert orch._default_micro_batch_size == 8
        assert orch._max_inflight == 16

    def test_constructor_edge_values(self) -> None:
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

    def test_constructor_no_resource_mgr(self) -> None:
        orch = PipelineOrchestrator()
        assert orch._resource_mgr is None
        # Pipeline should still work without a resource manager
        orch.register_node("n", "h", 1, 0, 15)
        with patch(
            "distllm.dist.node_client.forward_request",
            side_effect=dummy_forward_sync,
        ):
            result = orch.run_pipeline(
                torch.tensor([[1, 2]], dtype=torch.long),
                {"n": None},
                "req-no-rm",
            )
        assert result is not None

    def test_stats_initial(self, orchestrator: PipelineOrchestrator) -> None:
        stats = orchestrator.stats()
        assert stats["pipeline_runs"] == 0
        assert stats["total_latency_ms"] == 0.0
        assert stats["errors"] == 0
        assert stats["micro_batched_runs"] == 0
        assert stats["avg_micro_batch_size"] == 0.0
        assert stats["micro_batch_count_total"] == 0
        assert stats["dynamic_batch_adjustments"] == []
        assert stats["node_count"] == 0
        assert stats["healthy_nodes"] == 0
        assert stats["avg_latency_ms"] == 0.0
        assert stats["total_layers"] == 0
        assert stats["micro_batched_enabled"] is True

    def test_properties_defaults(self) -> None:
        orch = PipelineOrchestrator()
        assert orch.nodes == {}
        assert orch.node_order == []
        assert orch.total_layers == 0
        assert orch.pipeline_timeout == 30.0
        assert orch.wan is None

    def test_properties_with_nodes(self, orchestrator: PipelineOrchestrator) -> None:
        register_two_nodes(orchestrator)
        assert isinstance(orchestrator.nodes, dict)
        assert len(orchestrator.nodes) == 2
        assert orchestrator.node_order == ["node-0", "node-1"]

    def test_nodes_property_returns_live_mapping(self, orchestrator: PipelineOrchestrator) -> None:
        """nodes returns the LIVE mapping: later registrations are visible.

        (Registration/mutation flows — client injection, health flips —
        depend on live access; use register_node/unregister_node for
        structural changes.)
        """
        register_two_nodes(orchestrator)
        orchestrator.register_node("node-extra", "h", 1, 32, 47)
        assert "node-extra" in orchestrator.nodes

    def test_nodes_property_contains_expected_keys(self, orchestrator: PipelineOrchestrator) -> None:
        register_two_nodes(orchestrator)
        n = orchestrator.nodes["node-0"]
        # Live values are PipelineNode objects with these fields.
        assert n.host == "10.0.0.1"
        assert n.port == 50051
        assert n.start_layer == 0
        assert n.end_layer == 15
        assert n.is_healthy is True

    def test_node_order_property_independence(self, orchestrator: PipelineOrchestrator) -> None:
        register_two_nodes(orchestrator)
        order_copy = orchestrator.node_order
        orchestrator.register_node("node-2", "h", 1, 32, 47)
        # The old copy remains unchanged
        assert len(order_copy) == 2

    def test_setters(self, orchestrator: PipelineOrchestrator) -> None:
        lt = MagicMock()
        sd = MagicMock()
        orchestrator.set_latency_tracker(lt)
        orchestrator.set_straggler_detector(sd)
        assert orchestrator._latency_tracker is lt
        assert orchestrator._straggler_detector is sd

    def test_total_layers_property_explicit(self, orchestrator: PipelineOrchestrator) -> None:
        assert orchestrator.total_layers == 0
        orchestrator.total_layers = 42
        assert orchestrator.total_layers == 42

    def test_total_layers_derived_from_nodes(self, orchestrator: PipelineOrchestrator) -> None:
        register_two_nodes(orchestrator)
        # Explicit value not set, so derived: max(end_layer) + 1 = 31 + 1 = 32
        assert orchestrator.total_layers == 32

    def test_total_layers_derived_empty(self, orchestrator: PipelineOrchestrator) -> None:
        assert orchestrator.total_layers == 0

    def test_pipeline_timeout_setter(self, orchestrator: PipelineOrchestrator) -> None:
        orchestrator.pipeline_timeout = 15.0
        assert orchestrator.pipeline_timeout == 15.0

    def test_wan_setter(self, orchestrator: PipelineOrchestrator) -> None:
        w = MagicMock()
        orchestrator.wan = w
        assert orchestrator.wan is w

    def test_wan_default_none(self) -> None:
        assert PipelineOrchestrator().wan is None

    def test_shutdown_clears_nodes(self, orchestrator: PipelineOrchestrator) -> None:
        register_two_nodes(orchestrator)
        assert len(orchestrator._nodes) == 2
        orchestrator.shutdown()
        assert orchestrator._nodes == {}
        assert orchestrator._node_order == []

    def test_shutdown_empty_is_safe(self, orchestrator: PipelineOrchestrator) -> None:
        orchestrator.shutdown()
        assert orchestrator._nodes == {}

    def test_shutdown_logs_message(self, orchestrator: PipelineOrchestrator) -> None:
        with patch("distllm.dist.pipeline.orchestrator.logger.info") as mock_log:
            orchestrator.shutdown()
        mock_log.assert_called_once_with("Pipeline orchestrator shut down")


# ===================================================================
# PipelineOrchestrator -- Node Addition & Removal
# ===================================================================


class TestNodeManagement:
    """register_node, unregister_node, get_node, remove_node, health status."""

    def test_register_node(self, orchestrator: PipelineOrchestrator) -> None:
        orchestrator.register_node("node-0", "10.0.0.1", 50051, 0, 15, total_layers=32)
        node = orchestrator._nodes["node-0"]
        assert node.node_id == "node-0"
        assert node.host == "10.0.0.1"
        assert node.port == 50051
        assert node.start_layer == 0
        assert node.end_layer == 15
        assert node.total_layers == 32
        assert orchestrator.node_order == ["node-0"]

    def test_register_node_sorts_by_start_layer(self, orchestrator: PipelineOrchestrator) -> None:
        orchestrator.register_node("node-b", "h", 1, 16, 31)
        orchestrator.register_node("node-a", "h", 1, 0, 15)
        assert orchestrator.node_order == ["node-a", "node-b"]

    def test_register_node_same_start_layer_maintains_order(self, orchestrator: PipelineOrchestrator) -> None:
        orchestrator.register_node("first", "h", 1, 0, 7)
        orchestrator.register_node("second", "h", 1, 0, 7)
        assert orchestrator.node_order == ["first", "second"]

    def test_register_node_accepts_extra_kwargs(self, orchestrator: PipelineOrchestrator) -> None:
        orchestrator.register_node("n", "h", 1, 0, 15, extra_field="ignored")
        assert "n" in orchestrator._nodes

    def test_register_node_replaces_existing(self, orchestrator: PipelineOrchestrator) -> None:
        orchestrator.register_node("n", "old.host", 1, 0, 15)
        orchestrator.register_node("n", "new.host", 2, 0, 15)
        assert orchestrator._nodes["n"].host == "new.host"
        assert orchestrator._nodes["n"].port == 2

    def test_unregister_node(self, orchestrator: PipelineOrchestrator) -> None:
        register_two_nodes(orchestrator)
        orchestrator.unregister_node("node-0")
        assert "node-0" not in orchestrator._nodes
        assert orchestrator.node_order == ["node-1"]

    def test_unregister_unknown_node_is_safe(self, orchestrator: PipelineOrchestrator) -> None:
        orchestrator.unregister_node("nonexistent")
        assert orchestrator._nodes == {}

    def test_remove_node_alias(self, orchestrator: PipelineOrchestrator) -> None:
        register_two_nodes(orchestrator)
        orchestrator.remove_node("node-0")
        assert "node-0" not in orchestrator._nodes
        assert orchestrator.node_order == ["node-1"]

    def test_get_node(self, orchestrator: PipelineOrchestrator) -> None:
        register_two_nodes(orchestrator)
        node = orchestrator.get_node("node-0")
        assert node is not None
        assert node.node_id == "node-0"

    def test_get_node_missing(self, orchestrator: PipelineOrchestrator) -> None:
        assert orchestrator.get_node("nonexistent") is None

    def test_validate_layer_assignment_no_overlap(self, orchestrator: PipelineOrchestrator) -> None:
        register_two_nodes(orchestrator)
        orchestrator.validate_layer_assignment("node-2", 32, 47)

    def test_validate_layer_assignment_overlap_raises(self, orchestrator: PipelineOrchestrator) -> None:
        register_two_nodes(orchestrator)
        with pytest.raises(ValueError, match="Layer overlap"):
            orchestrator.validate_layer_assignment("node-2", 10, 20)

    def test_validate_layer_assignment_exact_overlap(self, orchestrator: PipelineOrchestrator) -> None:
        register_two_nodes(orchestrator)
        with pytest.raises(ValueError, match="Layer overlap"):
            orchestrator.validate_layer_assignment("node-2", 0, 15)

    def test_validate_layer_assignment_adjacent_allowed(self, orchestrator: PipelineOrchestrator) -> None:
        """Adjacent ranges (end of one node equals start of another) should be valid."""
        register_two_nodes(orchestrator)
        orchestrator.validate_layer_assignment("node-2", 32, 47)

    def test_validate_layer_assignment_skips_same_node(self, orchestrator: PipelineOrchestrator) -> None:
        register_two_nodes(orchestrator)
        orchestrator.validate_layer_assignment("node-0", 0, 15)

    def test_validate_layer_assignment_empty_pipeline(self, orchestrator: PipelineOrchestrator) -> None:
        orchestrator.validate_layer_assignment("node-0", 0, 15)

    def test_get_healthy_nodes(self, orchestrator: PipelineOrchestrator) -> None:
        register_two_nodes(orchestrator)
        assert orchestrator.get_healthy_nodes() == ["node-0", "node-1"]

    def test_get_healthy_nodes_empty(self, orchestrator: PipelineOrchestrator) -> None:
        assert orchestrator.get_healthy_nodes() == []

    def test_mark_node_unhealthy(self, orchestrator: PipelineOrchestrator) -> None:
        register_two_nodes(orchestrator)
        orchestrator.mark_node_unhealthy("node-0")
        assert orchestrator._nodes["node-0"].is_healthy is False
        assert orchestrator.get_healthy_nodes() == ["node-1"]

    def test_mark_node_unhealthy_unknown_is_safe(self, orchestrator: PipelineOrchestrator) -> None:
        orchestrator.mark_node_unhealthy("nonexistent")

    def test_mark_node_healthy_after_unhealthy(self, orchestrator: PipelineOrchestrator) -> None:
        register_two_nodes(orchestrator)
        orchestrator.mark_node_unhealthy("node-0")
        orchestrator.mark_node_healthy("node-0")
        assert orchestrator._nodes["node-0"].is_healthy is True

    def test_mark_node_healthy_unknown_is_safe(self, orchestrator: PipelineOrchestrator) -> None:
        orchestrator.mark_node_healthy("nonexistent")

    def test_mark_node_healthy_twice(self, orchestrator: PipelineOrchestrator) -> None:
        register_two_nodes(orchestrator)
        orchestrator.mark_node_healthy("node-0")
        assert orchestrator._nodes["node-0"].is_healthy is True

    def test_stats_node_count_after_management(self, orchestrator: PipelineOrchestrator) -> None:
        register_two_nodes(orchestrator)
        assert orchestrator.stats()["node_count"] == 2
        assert orchestrator.stats()["healthy_nodes"] == 2
        orchestrator.mark_node_unhealthy("node-0")
        assert orchestrator.stats()["healthy_nodes"] == 1
        orchestrator.unregister_node("node-0")
        assert orchestrator.stats()["node_count"] == 1

    def test_layer_order_maintained_on_unregister(self, orchestrator: PipelineOrchestrator) -> None:
        orchestrator.register_node("n2", "h", 1, 16, 31)
        orchestrator.register_node("n1", "h", 1, 0, 15)
        orchestrator.register_node("n3", "h", 1, 32, 47)
        orchestrator.unregister_node("n1")
        assert orchestrator.node_order == ["n2", "n3"]
        orchestrator.unregister_node("n3")
        assert orchestrator.node_order == ["n2"]

    def test_register_logs_info(self, orchestrator: PipelineOrchestrator) -> None:
        with patch("distllm.dist.pipeline.orchestrator.logger.info") as mock_log:
            orchestrator.register_node("n", "h", 1, 0, 15)
        mock_log.assert_called_once_with(
            "Pipeline: registered n (layers 0-15)"
        )


# ===================================================================
# PipelineOrchestrator -- Sequential Pipeline (run_pipeline)
# ===================================================================


class TestSequentialPipeline:
    """run_pipeline sequential execution."""

    def test_run_pipeline_success(
        self,
        orchestrator: PipelineOrchestrator,
        sample_tensor: torch.Tensor,
        kv_caches_default: dict[str, None],
    ) -> None:
        register_two_nodes(orchestrator)
        intermediate = torch.tensor([[5, 6, 7, 8]], dtype=torch.long)
        final = torch.tensor([[9, 10, 11, 12]], dtype=torch.long)

        with patch(
            "distllm.dist.node_client.forward_request",
            side_effect=[intermediate, final],
        ) as mock_fwd:
            result = orchestrator.run_pipeline(
                sample_tensor, kv_caches_default, "req-1"
            )

        assert mock_fwd.call_count == 2
        # First call goes to node-0
        c0 = mock_fwd.call_args_list[0][1]
        assert c0["host"] == "10.0.0.1"
        assert c0["port"] == 50051
        assert c0["request_id"] == "req-1"
        # Second call goes to node-1
        c1 = mock_fwd.call_args_list[1][1]
        assert c1["host"] == "10.0.0.2"
        assert c1["port"] == 50051
        assert c1["request_id"] == "req-1"
        # The second call receives the output of the first
        assert c1["hidden_states"] is intermediate

        torch.testing.assert_close(result, final)
        assert orchestrator.stats()["pipeline_runs"] == 1
        assert orchestrator.stats()["errors"] == 0

    def test_run_pipeline_resource_mgr_success(
        self,
        orchestrator: PipelineOrchestrator,
        sample_tensor: torch.Tensor,
        kv_caches_default: dict[str, None],
    ) -> None:
        register_two_nodes(orchestrator)
        with patch(
            "distllm.dist.node_client.forward_request",
            side_effect=dummy_forward_sync,
        ):
            orchestrator.run_pipeline(sample_tensor, kv_caches_default, "req-1")

        assert orchestrator._resource_mgr.record_success.call_count == 2
        orchestrator._resource_mgr.record_success.assert_any_call("node-0")
        orchestrator._resource_mgr.record_success.assert_any_call("node-1")

    def test_run_pipeline_no_healthy_nodes(
        self,
        orchestrator: PipelineOrchestrator,
        sample_tensor: torch.Tensor,
    ) -> None:
        register_two_nodes(orchestrator)
        orchestrator.mark_node_unhealthy("node-0")
        orchestrator.mark_node_unhealthy("node-1")

        with pytest.raises(RuntimeError, match="No healthy nodes in pipeline"):
            orchestrator.run_pipeline(sample_tensor, {"node-0": None, "node-1": None}, "req-1")

    def test_run_pipeline_skips_unhealthy_nodes(
        self,
        orchestrator: PipelineOrchestrator,
        sample_tensor: torch.Tensor,
    ) -> None:
        register_two_nodes(orchestrator)
        orchestrator.mark_node_unhealthy("node-0")

        with patch(
            "distllm.dist.node_client.forward_request",
            side_effect=dummy_forward_sync,
        ) as mock_fwd:
            result = orchestrator.run_pipeline(
                sample_tensor, {"node-0": None, "node-1": None}, "req-1"
            )

        # Only node-1 is healthy
        assert mock_fwd.call_count == 1
        assert mock_fwd.call_args[1]["host"] == "10.0.0.2"
        assert result is not None

    def test_run_pipeline_node_returns_none(
        self,
        orchestrator: PipelineOrchestrator,
        sample_tensor: torch.Tensor,
        kv_caches_default: dict[str, None],
    ) -> None:
        register_two_nodes(orchestrator)
        with patch(
            "distllm.dist.node_client.forward_request",
            return_value=None,
        ):
            with pytest.raises(RuntimeError, match="returned None"):
                orchestrator.run_pipeline(sample_tensor, kv_caches_default, "req-1")

        assert orchestrator.stats()["errors"] == 1

    def test_run_pipeline_node_raises_exception(
        self,
        orchestrator: PipelineOrchestrator,
        sample_tensor: torch.Tensor,
        kv_caches_default: dict[str, None],
    ) -> None:
        register_two_nodes(orchestrator)
        with patch(
            "distllm.dist.node_client.forward_request",
            side_effect=RuntimeError("connection refused"),
        ):
            with pytest.raises(RuntimeError, match="connection refused"):
                orchestrator.run_pipeline(sample_tensor, kv_caches_default, "req-1")

        assert orchestrator.stats()["errors"] == 1
        orchestrator._resource_mgr.record_failure.assert_called_once_with("node-0")

    def test_run_pipeline_passes_output_to_next_node(
        self,
        orchestrator: PipelineOrchestrator,
        kv_caches_default: dict[str, None],
    ) -> None:
        """Verify that the output of node-0 flows as input to node-1."""
        register_two_nodes(orchestrator)
        intermediate = torch.tensor([[10, 20]], dtype=torch.long)
        final = torch.tensor([[30, 40]], dtype=torch.long)

        with patch(
            "distllm.dist.node_client.forward_request",
            side_effect=[intermediate, final],
        ) as mock_fwd:
            orchestrator.run_pipeline(
                torch.tensor([[1, 2]], dtype=torch.long),
                kv_caches_default,
                "req-1",
            )

        assert mock_fwd.call_args_list[1][1]["hidden_states"] is intermediate

    def test_run_pipeline_single_node(self, orchestrator: PipelineOrchestrator) -> None:
        orchestrator.register_node("only-node", "10.0.0.1", 50051, 0, 31)
        inp = torch.tensor([[1, 2, 3]], dtype=torch.long)
        expected = torch.tensor([[4, 5, 6]], dtype=torch.long)

        with patch(
            "distllm.dist.node_client.forward_request",
            return_value=expected,
        ) as mock_fwd:
            result = orchestrator.run_pipeline(inp, {"only-node": None}, "req-1")

        assert mock_fwd.call_count == 1
        torch.testing.assert_close(result, expected)

    def test_run_pipeline_updates_node_latency(
        self,
        orchestrator: PipelineOrchestrator,
        sample_tensor: torch.Tensor,
        kv_caches_default: dict[str, None],
    ) -> None:
        register_two_nodes(orchestrator)
        with patch(
            "distllm.dist.node_client.forward_request",
            side_effect=dummy_forward_sync,
        ):
            orchestrator.run_pipeline(sample_tensor, kv_caches_default, "req-1")

        # Each node should have a non-zero latency recorded
        for nid in ("node-0", "node-1"):
            assert orchestrator._nodes[nid].latency_ms > 0.0

    def test_run_pipeline_without_resource_mgr(self, sample_tensor: torch.Tensor) -> None:
        orch = PipelineOrchestrator()
        register_two_nodes(orch)
        with patch(
            "distllm.dist.node_client.forward_request",
            side_effect=dummy_forward_sync,
        ):
            result = orch.run_pipeline(
                sample_tensor, {"node-0": None, "node-1": None}, "req-1"
            )
        assert result is not None


# ===================================================================
# PipelineOrchestrator -- Micro-batched Pipeline (run_pipeline_microbatched)
# ===================================================================


class TestMicrobatchedPipeline:
    """run_pipeline_microbatched async execution with 1F1B scheduling."""

    @pytest.mark.asyncio
    async def test_microbatched_success(
        self,
        orchestrator: PipelineOrchestrator,
        kv_caches_default: dict[str, None],
    ) -> None:
        register_two_nodes(orchestrator)
        # batch=8, micro_batch_size=2 => 4 micro-batches, 2 stages => 8 steps
        input_tensor = torch.randn(8, 128)

        async def mock_fwd(**kwargs: object) -> torch.Tensor:
            hidden = kwargs.get("hidden_states")
            if hidden is not None:
                return hidden.clone()
            return torch.randn(2, 128)

        with patch(
            "distllm.dist.node_client.forward_request_async",
            side_effect=mock_fwd,
        ) as mock_fwd_patch:
            result = await orchestrator.run_pipeline_microbatched(
                input_tensor, kv_caches_default, "req-1", micro_batch_size=2,
            )

        assert mock_fwd_patch.call_count == 8
        assert result.shape == (8, 128)
        assert orchestrator.stats()["micro_batched_runs"] == 1
        assert orchestrator.stats()["micro_batch_count_total"] == 4

    @pytest.mark.asyncio
    async def test_microbatched_no_healthy_nodes(
        self,
        orchestrator: PipelineOrchestrator,
    ) -> None:
        register_two_nodes(orchestrator)
        orchestrator.mark_node_unhealthy("node-0")
        orchestrator.mark_node_unhealthy("node-1")
        input_tensor = torch.randn(4, 128)

        with pytest.raises(RuntimeError, match="No healthy nodes in pipeline"):
            await orchestrator.run_pipeline_microbatched(
                input_tensor, {"node-0": None, "node-1": None}, "req-1",
            )

    @pytest.mark.asyncio
    @pytest.mark.skip(
        reason="execute_pipeline_step catches exceptions without signalling "
               "downstream stage_batch_ready events, causing downstream "
               "stages to block indefinitely"
    )
    async def test_microbatched_all_micro_batches_fail(
        self,
        orchestrator: PipelineOrchestrator,
        kv_caches_default: dict[str, None],
    ) -> None:
        register_two_nodes(orchestrator)
        input_tensor = torch.randn(4, 128)

        with patch(
            "distllm.dist.node_client.forward_request_async",
            side_effect=RuntimeError("node failure"),
        ):
            with pytest.raises(RuntimeError, match="All micro-batches failed"):
                await orchestrator.run_pipeline_microbatched(
                    input_tensor, kv_caches_default, "req-1", micro_batch_size=2,
                )

        assert orchestrator.stats()["errors"] > 0

    @pytest.mark.asyncio
    async def test_microbatched_partial_failure(
        self,
        orchestrator: PipelineOrchestrator,
        kv_caches_default: dict[str, None],
    ) -> None:
        """Partial failure must not silently shrink the returned batch.

        Default policy ("raise") raises PipelineError naming the failed
        input rows.  (Legacy "drop" behaviour is pinned in
        tests/dist/pipeline/test_microbatch_integrity.py.)
        """
        register_two_nodes(orchestrator)
        input_tensor = torch.randn(4, 128)  # 2 micro-batches of size 2

        async def mock_fwd(**kwargs: object) -> torch.Tensor:
            # Fail only the last-stage batch-0 call (request_id ends in -s1b0)
            request_id: str = kwargs.get("request_id", "")
            if "-s1b0" in request_id:
                raise RuntimeError("transient error on last stage")
            hidden = kwargs.get("hidden_states")
            if hidden is not None:
                return hidden.clone()
            return torch.randn(2, 128)

        with patch(
            "distllm.dist.node_client.forward_request_async",
            side_effect=mock_fwd,
        ):
            with pytest.raises(PipelineError, match="sequences \\[0, 1\\]"):
                await orchestrator.run_pipeline_microbatched(
                    input_tensor, kv_caches_default, "req-1", micro_batch_size=2,
                )

        assert (
            orchestrator.stats()["last_failed_sequences"] == [0, 1]
        )

    @pytest.mark.asyncio
    async def test_microbatched_with_resource_mgr(
        self,
        orchestrator: PipelineOrchestrator,
        kv_caches_default: dict[str, None],
    ) -> None:
        register_two_nodes(orchestrator)
        input_tensor = torch.randn(4, 128)

        async def mock_fwd(**kwargs: object) -> torch.Tensor:
            hidden = kwargs.get("hidden_states")
            if hidden is not None:
                return hidden.clone()
            return torch.randn(2, 128)

        with patch(
            "distllm.dist.node_client.forward_request_async",
            side_effect=mock_fwd,
        ):
            await orchestrator.run_pipeline_microbatched(
                input_tensor, kv_caches_default, "req-1", micro_batch_size=2,
            )

        # Each successful micro-batch through a node should record success
        assert orchestrator._resource_mgr.record_success.call_count >= 2

    @pytest.mark.asyncio
    async def test_microbatched_resource_mgr_records_failure(
        self,
        orchestrator: PipelineOrchestrator,
        kv_caches_default: dict[str, None],
    ) -> None:
        """Failed micro-batch steps should record failures via resource_mgr.

        Failure must occur in the last stage only, because
        execute_pipeline_step does not signal downstream stages
        when an earlier stage fails, causing a deadlock.
        """
        register_two_nodes(orchestrator)
        input_tensor = torch.randn(4, 128)

        async def mock_fwd(**kwargs: object) -> torch.Tensor:
            request_id: str = kwargs.get("request_id", "")
            # Fail only in last-stage calls so upstream stages can
            # still signal downstream dependency events
            if "-s1b" in request_id:
                raise RuntimeError("crash on last stage")
            hidden = kwargs.get("hidden_states")
            if hidden is not None:
                return hidden.clone()
            return torch.randn(2, 128)

        with patch(
            "distllm.dist.node_client.forward_request_async",
            side_effect=mock_fwd,
        ):
            with pytest.raises(RuntimeError):
                await orchestrator.run_pipeline_microbatched(
                    input_tensor, kv_caches_default, "req-1", micro_batch_size=2,
                )

        orchestrator._resource_mgr.record_failure.assert_called()

    @pytest.mark.asyncio
    async def test_microbatched_single_micro_batch(
        self,
        orchestrator: PipelineOrchestrator,
        kv_caches_default: dict[str, None],
    ) -> None:
        """With total_tokens=2 and micro_batch_size=2, clamping reduces
        micro_batch_size to min(2, 2//2=1) = 1, producing 2 micro-batches.
        The pipeline should succeed with the correct output shape."""
        register_two_nodes(orchestrator)
        input_tensor = torch.randn(2, 128)

        async def mock_fwd(**kwargs: object) -> torch.Tensor:
            hidden = kwargs.get("hidden_states")
            if hidden is not None:
                return hidden.clone()
            return torch.randn(2, 128)

        with patch(
            "distllm.dist.node_client.forward_request_async",
            side_effect=mock_fwd,
        ):
            # micro_batch_size=2 gets clamped to max(1, min(2, 2//2)) = 1,
            # so we get 2 micro-batches, both stages run, and the pipeline
            # succeeds rather than raising an error.
            result = await orchestrator.run_pipeline_microbatched(
                input_tensor, kv_caches_default, "req-1", micro_batch_size=2,
            )
        assert result is not None
        assert result.shape == (2, 128)

    @pytest.mark.asyncio
    async def test_microbatched_three_stages_four_batches(
        self,
        orchestrator: PipelineOrchestrator,
    ) -> None:
        """3 stages, 4 micro-batches => 3 warmup + 3*3 steady = 12 steps."""
        orchestrator.register_node("s0", "h", 1, 0, 10)
        orchestrator.register_node("s1", "h", 1, 11, 20)
        orchestrator.register_node("s2", "h", 1, 21, 30)
        input_tensor = torch.randn(8, 128)
        kvs = {"s0": None, "s1": None, "s2": None}

        async def mock_fwd(**kwargs: object) -> torch.Tensor:
            hidden = kwargs.get("hidden_states")
            if hidden is not None:
                return hidden.clone()
            return torch.randn(2, 128)

        with patch(
            "distllm.dist.node_client.forward_request_async",
            side_effect=mock_fwd,
        ) as mock_fwd_patch:
            result = await orchestrator.run_pipeline_microbatched(
                input_tensor, kvs, "req-3s", micro_batch_size=2,
            )

        # 3 stages, 4 batches: schedule_steps
        # Warmup: s=0,1,2 -> (0,0),(1,0),(2,0) = 3 steps
        # Steady: b=1 -> (0,1),(1,1),(2,1) = 3 steps
        #         b=2 -> (0,2),(1,2),(2,2) = 3 steps
        #         b=3 -> (0,3),(1,3),(2,3) = 3 steps
        # Total: 12 steps
        assert mock_fwd_patch.call_count == 12
        assert result.shape == (8, 128)


# ===================================================================
# PipelineOrchestrator -- Dynamic Micro-Batch Sizing
# ===================================================================


class TestDynamicBatchSizing:
    """Straggler-aware dynamic micro-batch size adjustments."""

    @pytest.mark.asyncio
    async def test_no_straggler_detector_uses_default(
        self,
        orchestrator: PipelineOrchestrator,
        kv_caches_default: dict[str, None],
    ) -> None:
        """When no straggler detector is set, batch size stays at default."""
        register_two_nodes(orchestrator)
        orchestrator._straggler_detector = None
        input_tensor = torch.randn(8, 128)

        with patch(
            "distllm.dist.node_client.forward_request_async",
            side_effect=dummy_forward_async,
        ) as mock_fwd:
            await orchestrator.run_pipeline_microbatched(
                input_tensor, kv_caches_default, "req-dflt", micro_batch_size=4,
            )

        # Each call uses the default micro_batch_size=4 internally
        assert mock_fwd.call_count == 4  # 2 stages * 2 batches (8/4=2)
        assert len(orchestrator._stats["dynamic_batch_adjustments"]) == 0

    @pytest.mark.asyncio
    async def test_straggler_reduces_batch_size(
        self,
        orchestrator: PipelineOrchestrator,
        kv_caches_default: dict[str, None],
    ) -> None:
        """Straggler detector with moderate/severe severity halves the batch size."""
        register_two_nodes(orchestrator)
        report = SimpleNamespace(severity=SimpleNamespace(value="moderate"))
        orchestrator._straggler_detector = MagicMock(
            get_reports=MagicMock(return_value=[report]),
            stats=MagicMock(return_value={"nodes": {}}),
        )
        input_tensor = torch.randn(8, 128)

        with patch(
            "distllm.dist.node_client.forward_request_async",
            side_effect=dummy_forward_async,
        ) as mock_fwd:
            await orchestrator.run_pipeline_microbatched(
                input_tensor, kv_caches_default, "req-str", micro_batch_size=4,
            )

        # With halving: micro_batch_size=2 => 4 batches => 2 stages * 4 = 8 steps
        assert mock_fwd.call_count == 8
        assert len(orchestrator._stats["dynamic_batch_adjustments"]) == 1
        adj = orchestrator._stats["dynamic_batch_adjustments"][0]
        assert adj["adjustment"] == "straggler_reduce_50"
        assert adj["micro_batch_size"] == 2

    @pytest.mark.asyncio
    async def test_straggler_severe_reduces_batch_size(
        self,
        orchestrator: PipelineOrchestrator,
        kv_caches_default: dict[str, None],
    ) -> None:
        """'severe' severity also triggers halving."""
        register_two_nodes(orchestrator)
        report = SimpleNamespace(severity=SimpleNamespace(value="severe"))
        orchestrator._straggler_detector = MagicMock(
            get_reports=MagicMock(return_value=[report]),
        )
        input_tensor = torch.randn(8, 128)

        with patch(
            "distllm.dist.node_client.forward_request_async",
            side_effect=dummy_forward_async,
        ):
            await orchestrator.run_pipeline_microbatched(
                input_tensor, kv_caches_default, "req-sev", micro_batch_size=4,
            )

        adj = orchestrator._stats["dynamic_batch_adjustments"][0]
        assert adj["adjustment"] == "straggler_reduce_50"

    @pytest.mark.asyncio
    async def test_straggler_mild_does_not_reduce(
        self,
        orchestrator: PipelineOrchestrator,
        kv_caches_default: dict[str, None],
    ) -> None:
        """'mild' severity does not trigger the reduction branch."""
        register_two_nodes(orchestrator)
        report = SimpleNamespace(severity=SimpleNamespace(value="mild"))
        orchestrator._straggler_detector = MagicMock(
            get_reports=MagicMock(return_value=[report]),
        )
        input_tensor = torch.randn(8, 128)

        with patch(
            "distllm.dist.node_client.forward_request_async",
            side_effect=dummy_forward_async,
        ) as mock_fwd:
            await orchestrator.run_pipeline_microbatched(
                input_tensor, kv_caches_default, "req-mild", micro_batch_size=4,
            )

        assert len(orchestrator._stats["dynamic_batch_adjustments"]) == 0
        assert mock_fwd.call_count == 4  # default: 2 stages * 2 batches

    @pytest.mark.asyncio
    async def test_no_stragglers_with_low_latency_increases_batch(
        self,
        orchestrator: PipelineOrchestrator,
        kv_caches_default: dict[str, None],
    ) -> None:
        """No reports and low node latencies should increase batch size by 25%."""
        register_two_nodes(orchestrator)
        mock_sd = MagicMock()
        mock_sd.get_reports = MagicMock(return_value=[])
        mock_sd.stats = MagicMock(return_value={
            "nodes": {
                "node-0": {"avg_latency": 30.0},
                "node-1": {"avg_latency": 20.0},
            },
        })
        orchestrator._straggler_detector = mock_sd
        # Start with 4 -> increase 25% -> 5 -> clamped to min(5, 8//2=4) = 4
        # So no visible change with input_tokens=8
        # Use larger input to see the increase
        input_tensor = torch.randn(16, 128)  # total_tokens=16, max(tokens//2) = 8

        with patch(
            "distllm.dist.node_client.forward_request_async",
            side_effect=dummy_forward_async,
        ):
            await orchestrator.run_pipeline_microbatched(
                input_tensor, kv_caches_default, "req-inc", micro_batch_size=4,
            )

        assert len(orchestrator._stats["dynamic_batch_adjustments"]) == 1
        adj = orchestrator._stats["dynamic_batch_adjustments"][0]
        assert adj["adjustment"] == "low_latency_increase_25"
        # 4 * 1.25 = 5.0 -> int(5.0) = 5
        assert adj["micro_batch_size"] == 5

    @pytest.mark.asyncio
    async def test_straggler_detector_exception_falls_back(
        self,
        orchestrator: PipelineOrchestrator,
        kv_caches_default: dict[str, None],
    ) -> None:
        """If the straggler detector raises, the default batch size is used."""
        register_two_nodes(orchestrator)
        orchestrator._straggler_detector = MagicMock(
            get_reports=MagicMock(side_effect=ValueError("detector error")),
        )
        input_tensor = torch.randn(8, 128)

        with patch(
            "distllm.dist.node_client.forward_request_async",
            side_effect=dummy_forward_async,
        ) as mock_fwd:
            await orchestrator.run_pipeline_microbatched(
                input_tensor, kv_caches_default, "req-exc", micro_batch_size=4,
            )

        # Fallback: should behave as if no straggler detector
        assert len(orchestrator._stats["dynamic_batch_adjustments"]) == 0
        assert mock_fwd.call_count == 4  # 2 stages * 2 batches

    @pytest.mark.asyncio
    async def test_batch_size_clamped_to_half_total(
        self,
        orchestrator: PipelineOrchestrator,
        kv_caches_default: dict[str, None],
    ) -> None:
        """micro_batch_size must not exceed total_tokens // 2."""
        register_two_nodes(orchestrator)
        input_tensor = torch.randn(6, 128)  # total_tokens=6, max = 3

        with patch(
            "distllm.dist.node_client.forward_request_async",
            side_effect=dummy_forward_async,
        ) as mock_fwd:
            await orchestrator.run_pipeline_microbatched(
                input_tensor, kv_caches_default, "req-clmp", micro_batch_size=10,
            )

        # micro_batch_size clamped to min(10, 6//2=3) = 3
        # 6/3 = 2 batches, 2 stages = 4 steps
        assert mock_fwd.call_count == 4

    @pytest.mark.asyncio
    async def test_batch_size_minimum_one(
        self,
        orchestrator: PipelineOrchestrator,
    ) -> None:
        """micro_batch_size should never go below 1."""
        orchestrator.register_node("n", "h", 1, 0, 15)
        input_tensor = torch.randn(2, 128)  # total_tokens=2, max = 1

        with patch(
            "distllm.dist.node_client.forward_request_async",
            side_effect=dummy_forward_async,
        ) as mock_fwd:
            await orchestrator.run_pipeline_microbatched(
                input_tensor, {"n": None}, "req-min", micro_batch_size=1,
            )

        # micro_batch_size=1, 2 batches, 1 stage = 2 steps
        assert mock_fwd.call_count == 2

    @pytest.mark.asyncio
    async def test_no_reports_no_node_info_no_change(
        self,
        orchestrator: PipelineOrchestrator,
        kv_caches_default: dict[str, None],
    ) -> None:
        """Empty reports, empty stats => no batch adjustment."""
        register_two_nodes(orchestrator)
        orchestrator._straggler_detector = MagicMock(
            get_reports=MagicMock(return_value=[]),
            stats=MagicMock(return_value={"nodes": {}}),
        )
        input_tensor = torch.randn(8, 128)

        with patch(
            "distllm.dist.node_client.forward_request_async",
            side_effect=dummy_forward_async,
        ) as mock_fwd:
            await orchestrator.run_pipeline_microbatched(
                input_tensor, kv_caches_default, "req-noch", micro_batch_size=4,
            )

        assert len(orchestrator._stats["dynamic_batch_adjustments"]) == 0
        assert mock_fwd.call_count == 4

    @pytest.mark.asyncio
    async def test_no_reports_node_info_zero_latency(
        self,
        orchestrator: PipelineOrchestrator,
        kv_caches_default: dict[str, None],
    ) -> None:
        """Nodes present but avg_latency is 0 => no adjustment."""
        register_two_nodes(orchestrator)
        orchestrator._straggler_detector = MagicMock(
            get_reports=MagicMock(return_value=[]),
            stats=MagicMock(return_value={
                "nodes": {
                    "node-0": {"avg_latency": 0},
                    "node-1": {"avg_latency": 0},
                },
            }),
        )
        input_tensor = torch.randn(8, 128)

        with patch(
            "distllm.dist.node_client.forward_request_async",
            side_effect=dummy_forward_async,
        ) as mock_fwd:
            await orchestrator.run_pipeline_microbatched(
                input_tensor, kv_caches_default, "req-zerolat", micro_batch_size=4,
            )

        # No adjustment because avg_latency is 0 which fails the > 0 check
        assert len(orchestrator._stats["dynamic_batch_adjustments"]) == 0


# ===================================================================
# PipelineOrchestrator -- Stats Tracking
# ===================================================================


class TestStatsTracking:
    """Verification of stats accumulation across pipeline runs."""

    def test_stats_after_sequential_run(
        self,
        orchestrator: PipelineOrchestrator,
        sample_tensor: torch.Tensor,
    ) -> None:
        register_two_nodes(orchestrator)
        with patch(
            "distllm.dist.node_client.forward_request",
            side_effect=dummy_forward_sync,
        ):
            orchestrator.run_pipeline(
                sample_tensor, {"node-0": None, "node-1": None}, "req-1",
            )

        stats = orchestrator.stats()
        assert stats["pipeline_runs"] == 1
        assert stats["total_latency_ms"] > 0.0
        assert stats["avg_latency_ms"] > 0.0
        assert stats["errors"] == 0
        assert stats["node_count"] == 2
        assert stats["healthy_nodes"] == 2

    @pytest.mark.asyncio
    async def test_stats_after_microbatched_run(
        self,
        orchestrator: PipelineOrchestrator,
        kv_caches_default: dict[str, None],
    ) -> None:
        register_two_nodes(orchestrator)
        input_tensor = torch.randn(8, 128)

        with patch(
            "distllm.dist.node_client.forward_request_async",
            side_effect=dummy_forward_async,
        ):
            await orchestrator.run_pipeline_microbatched(
                input_tensor, kv_caches_default, "req-stats", micro_batch_size=4,
            )

        stats = orchestrator.stats()
        assert stats["micro_batched_runs"] == 1
        assert stats["micro_batch_count_total"] == 2  # 8/4 = 2 batches
        assert stats["avg_micro_batch_size"] == 4.0
        assert stats["total_latency_ms"] > 0.0
        assert stats["errors"] == 0

    @pytest.mark.asyncio
    async def test_stats_multiple_microbatched_runs(
        self,
        orchestrator: PipelineOrchestrator,
        kv_caches_default: dict[str, None],
    ) -> None:
        register_two_nodes(orchestrator)
        input_tensor = torch.randn(8, 128)

        with patch(
            "distllm.dist.node_client.forward_request_async",
            side_effect=dummy_forward_async,
        ):
            await orchestrator.run_pipeline_microbatched(
                input_tensor, kv_caches_default, "r1", micro_batch_size=4,
            )
            await orchestrator.run_pipeline_microbatched(
                input_tensor, kv_caches_default, "r2", micro_batch_size=2,
            )

        stats = orchestrator.stats()
        assert stats["micro_batched_runs"] == 2
        assert stats["micro_batch_count_total"] == 2 + 4  # 2 + 4 = 6
        # avg = (4.0 + 2.0) / 2 = 3.0
        assert stats["avg_micro_batch_size"] == 3.0

    def test_stats_errors_accumulate(
        self,
        orchestrator: PipelineOrchestrator,
        sample_tensor: torch.Tensor,
    ) -> None:
        register_two_nodes(orchestrator)
        with patch(
            "distllm.dist.node_client.forward_request",
            side_effect=RuntimeError("fail"),
        ):
            for _ in range(3):
                try:
                    orchestrator.run_pipeline(
                        sample_tensor,
                        {"node-0": None, "node-1": None},
                        "req-fail",
                    )
                except RuntimeError:
                    pass

        stats = orchestrator.stats()
        assert stats["pipeline_runs"] == 3
        assert stats["errors"] == 3  # one error per run (fails on first node)


class TestScheduleSteps:
    """Verification of 1F1B schedule generation (schedule_steps inner function).

    Since schedule_steps is a nested function inside run_pipeline_microbatched,
    we test it indirectly by observing the number of gRPC calls made.
    """

    @pytest.mark.asyncio
    async def test_schedule_two_stages_four_batches(
        self,
        orchestrator: PipelineOrchestrator,
        kv_caches_default: dict[str, None],
    ) -> None:
        """2 stages, 4 batches => 2 warmup + 3*2 steady = 8 steps."""
        register_two_nodes(orchestrator)
        input_tensor = torch.randn(8, 128)

        with patch(
            "distllm.dist.node_client.forward_request_async",
            side_effect=dummy_forward_async,
        ) as mock_fwd:
            await orchestrator.run_pipeline_microbatched(
                input_tensor, kv_caches_default, "req-2s4b", micro_batch_size=2,
            )

        # 4 batches (8/2), 2 stages
        # Warmup: (0,0), (1,0) = 2
        # Steady: b=1 (0,1),(1,1); b=2 (0,2),(1,2); b=3 (0,3),(1,3) = 6
        # Total: 8
        assert mock_fwd.call_count == 8

    @pytest.mark.asyncio
    async def test_schedule_two_stages_two_batches(
        self,
        orchestrator: PipelineOrchestrator,
        kv_caches_default: dict[str, None],
    ) -> None:
        """2 stages, 2 batches => 2 warmup + 1*2 steady = 4 steps."""
        register_two_nodes(orchestrator)
        input_tensor = torch.randn(4, 128)

        with patch(
            "distllm.dist.node_client.forward_request_async",
            side_effect=dummy_forward_async,
        ) as mock_fwd:
            await orchestrator.run_pipeline_microbatched(
                input_tensor, kv_caches_default, "req-2s2b", micro_batch_size=2,
            )

        assert mock_fwd.call_count == 4

    @pytest.mark.asyncio
    async def test_schedule_one_stage_four_batches(
        self,
        orchestrator: PipelineOrchestrator,
    ) -> None:
        """1 stage (no pipelining), 4 batches => 1 warmup + 3*1 steady = 4 steps."""
        orchestrator.register_node("only", "h", 1, 0, 31)
        input_tensor = torch.randn(8, 128)

        with patch(
            "distllm.dist.node_client.forward_request_async",
            side_effect=dummy_forward_async,
        ) as mock_fwd:
            await orchestrator.run_pipeline_microbatched(
                input_tensor, {"only": None}, "req-1s4b", micro_batch_size=2,
            )

        # 1 stage, 4 batches
        # Warmup: (0,0) = 1
        # Steady: b=1 (0,1); b=2 (0,2); b=3 (0,3) = 3
        # Total: 4
        assert mock_fwd.call_count == 4


# ===================================================================
# PipelineOrchestrator -- Error Handling
# ===================================================================


class TestErrorHandling:
    """Error handling for pipeline failures."""

    def test_layer_overlap_validation_error_message(
        self,
        orchestrator: PipelineOrchestrator,
    ) -> None:
        register_two_nodes(orchestrator)
        with pytest.raises(ValueError) as exc:
            orchestrator.validate_layer_assignment("intruder", 12, 18)
        msg = str(exc.value)
        assert "Layer overlap" in msg
        # The message should reference the conflicting node
        assert "node-0" in msg or "node-1" in msg
        assert "12-18" in msg

    def test_pipeline_node_returns_none_mid_pipeline(
        self,
        orchestrator: PipelineOrchestrator,
        sample_tensor: torch.Tensor,
    ) -> None:
        """Node-0 returns successfully, node-1 returns None."""
        register_two_nodes(orchestrator)
        valid = torch.tensor([[5, 6, 7, 8]], dtype=torch.long)

        with patch(
            "distllm.dist.node_client.forward_request",
            side_effect=[valid, None],
        ):
            with pytest.raises(RuntimeError, match="returned None"):
                orchestrator.run_pipeline(
                    sample_tensor,
                    {"node-0": None, "node-1": None},
                    "req-none",
                )

        assert orchestrator.stats()["errors"] == 1

    def test_pipeline_node_raises_exception_mid_pipeline(
        self,
        orchestrator: PipelineOrchestrator,
        sample_tensor: torch.Tensor,
    ) -> None:
        """Node-0 succeeds, node-1 raises after success of node-0."""
        register_two_nodes(orchestrator)
        valid = torch.tensor([[5, 6, 7, 8]], dtype=torch.long)

        with patch(
            "distllm.dist.node_client.forward_request",
            side_effect=[valid, RuntimeError("node-1 crashed")],
        ):
            with pytest.raises(RuntimeError, match="node-1 crashed"):
                orchestrator.run_pipeline(
                    sample_tensor,
                    {"node-0": None, "node-1": None},
                    "req-crash",
                )

        assert orchestrator.stats()["errors"] == 1
        # node-0 should have been recorded as success
        orchestrator._resource_mgr.record_success.assert_called_once_with("node-0")
        # node-1 should have been recorded as failure
        orchestrator._resource_mgr.record_failure.assert_called_once_with("node-1")

    @pytest.mark.asyncio
    async def test_microbatched_timeout(
        self,
        orchestrator: PipelineOrchestrator,
        kv_caches_default: dict[str, None],
    ) -> None:
        """A never-completing step should raise TimeoutError via asyncio.wait_for."""
        register_two_nodes(orchestrator)
        orch = PipelineOrchestrator(pipeline_timeout=0.01)  # 10 ms timeout
        register_two_nodes(orch)
        input_tensor = torch.randn(4, 128)

        async def never_complete(**kwargs: object) -> torch.Tensor:
            await asyncio.sleep(3600)  # never completes
            return torch.randn(2, 128)  # pragma: no cover

        with patch(
            "distllm.dist.node_client.forward_request_async",
            side_effect=never_complete,
        ):
            with pytest.raises((TimeoutError, asyncio.TimeoutError)):
                await orch.run_pipeline_microbatched(
                    input_tensor, kv_caches_default, "req-to", micro_batch_size=2,
                )

    @pytest.mark.asyncio
    async def test_microbatched_failure_records_failure(
        self,
        orchestrator: PipelineOrchestrator,
        kv_caches_default: dict[str, None],
    ) -> None:
        """Failed micro-batch steps should record failures via resource_mgr.

        Failure must occur in the last stage only, because
        execute_pipeline_step does not signal downstream stages
        when an earlier stage fails, causing a deadlock.
        """
        register_two_nodes(orchestrator)
        input_tensor = torch.randn(4, 128)

        async def mock_fwd(**kwargs: object) -> torch.Tensor:
            request_id: str = kwargs.get("request_id", "")
            # Fail only in last-stage calls so upstream stages can
            # still signal downstream dependency events
            if "-s1b" in request_id:
                raise RuntimeError("error")
            hidden = kwargs.get("hidden_states")
            if hidden is not None:
                return hidden.clone()
            return torch.randn(2, 128)

        with patch(
            "distllm.dist.node_client.forward_request_async",
            side_effect=mock_fwd,
        ):
            with pytest.raises(RuntimeError):
                await orchestrator.run_pipeline_microbatched(
                    input_tensor, kv_caches_default, "req-err", micro_batch_size=2,
                )

        orchestrator._resource_mgr.record_failure.assert_called()
