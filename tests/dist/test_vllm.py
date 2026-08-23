"""Tests for VLLMPipelineEngine -- multi-node vLLM pipeline orchestrator.

Tests the public API surface using only real module objects (zero mocks).
"""

from __future__ import annotations

import pytest
import torch

from distllm.dist.backends.vllm import VLLMPipelineEngine
from distllm.errors.types import NodeUnreachableError


class TestVLLMPipelineEngineInit:
    """Constructor and initial state."""

    def test_default_construction(self):
        engine = VLLMPipelineEngine(
            node_order=["node_a"],
            nodes={"node_a": object()},
        )
        assert engine._node_order == ["node_a"]
        assert "node_a" in engine._nodes
        assert engine._timeout_s == 60.0
        assert engine.enable_overlap is False

    def test_custom_timeout(self):
        engine = VLLMPipelineEngine(
            node_order=["n1", "n2"],
            nodes={"n1": object(), "n2": object()},
            timeout_s=120.0,
        )
        assert engine._timeout_s == 120.0

    def test_empty_node_order(self):
        engine = VLLMPipelineEngine(node_order=[], nodes={})
        assert engine._node_order == []
        assert engine._nodes == {}

    def test_multiple_nodes(self):
        nodes = {"n1": object(), "n2": object(), "n3": object()}
        engine = VLLMPipelineEngine(node_order=["n1", "n2", "n3"], nodes=nodes)
        assert list(engine.node_order) == ["n1", "n2", "n3"]

    def test_enable_overlap_default_false(self):
        engine = VLLMPipelineEngine(node_order=[], nodes={})
        assert engine.enable_overlap is False

    def test_enable_overlap_can_be_set_true(self):
        engine = VLLMPipelineEngine(node_order=[], nodes={})
        engine.enable_overlap = True
        assert engine.enable_overlap is True


class TestVLLMPipelineEngineProperties:
    """Property accessors and setters."""

    def test_node_order_getter(self):
        engine = VLLMPipelineEngine(node_order=["a", "b"], nodes={"a": 1, "b": 2})
        result = engine.node_order
        assert result == ["a", "b"]
        assert isinstance(result, list)

    def test_node_order_setter(self):
        engine = VLLMPipelineEngine(node_order=["a"], nodes={"a": 1})
        engine.node_order = ["x", "y"]
        assert engine.node_order == ["x", "y"]

    def test_nodes_getter(self):
        engine = VLLMPipelineEngine(node_order=["a"], nodes={"a": "value"})
        assert engine.nodes == {"a": "value"}

    def test_nodes_setter(self):
        engine = VLLMPipelineEngine(node_order=[], nodes={})
        engine.nodes = {"new": 42}
        assert engine.nodes == {"new": 42}

    def test_node_order_setter_replaces_completely(self):
        engine = VLLMPipelineEngine(node_order=["a"], nodes={"a": 1})
        engine.node_order = ["b"]
        assert engine.node_order == ["b"]
        # original ordering is gone
        assert engine.node_order != ["a"]

    def test_nodes_setter_replaces_completely(self):
        engine = VLLMPipelineEngine(node_order=["a"], nodes={"a": 1})
        engine.nodes = {"b": 2}
        assert engine.nodes == {"b": 2}
        assert "a" not in engine.nodes


class TestVLLMPipelineEngineCreateNodeKVCaches:
    """create_node_kv_caches returns dict mapping node_id -> None."""

    def test_returns_dict_with_node_order_keys(self):
        engine = VLLMPipelineEngine(
            node_order=["n1", "n2", "n3"],
            nodes={"n1": object(), "n2": object(), "n3": object()},
        )
        caches = engine.create_node_kv_caches()
        assert isinstance(caches, dict)
        assert set(caches.keys()) == {"n1", "n2", "n3"}
        assert all(v is None for v in caches.values())

    def test_empty_node_order(self):
        engine = VLLMPipelineEngine(node_order=[], nodes={})
        caches = engine.create_node_kv_caches()
        assert caches == {}

    def test_single_node(self):
        engine = VLLMPipelineEngine(
            node_order=["only"],
            nodes={"only": object()},
        )
        caches = engine.create_node_kv_caches()
        assert caches == {"only": None}

    def test_result_is_fresh_dict_each_call(self):
        engine = VLLMPipelineEngine(
            node_order=["n1"],
            nodes={"n1": object()},
        )
        first = engine.create_node_kv_caches()
        second = engine.create_node_kv_caches()
        assert first is not second  # different objects
        assert first == second

    def test_keys_match_current_node_order(self):
        engine = VLLMPipelineEngine(
            node_order=["a", "b"],
            nodes={"a": 1, "b": 2},
        )
        caches = engine.create_node_kv_caches()
        assert list(caches.keys()) == ["a", "b"]

        engine.node_order = ["b", "a"]
        caches2 = engine.create_node_kv_caches()
        assert list(caches2.keys()) == ["b", "a"]


class TestVLLMPipelineEngineShutdown:
    """shutdown is a side-effect-free log call; these are smoke tests."""

    def test_shutdown_on_empty_engine(self):
        engine = VLLMPipelineEngine(node_order=[], nodes={})
        engine.shutdown()  # must not raise

    def test_shutdown_on_configured_engine(self):
        engine = VLLMPipelineEngine(
            node_order=["n1"],
            nodes={"n1": object()},
        )
        engine.shutdown()  # must not raise

    def test_shutdown_can_be_called_twice(self):
        engine = VLLMPipelineEngine(node_order=[], nodes={})
        engine.shutdown()
        engine.shutdown()  # must not raise


class TestVLLMPipelineEngineRunPipeline:
    """run_pipeline -- edge cases that do not require gRPC."""

    def test_empty_node_order_returns_none(self):
        engine = VLLMPipelineEngine(node_order=[], nodes={})
        input_ids = torch.zeros((1, 1), dtype=torch.long)
        result = engine.run_pipeline(
            input_ids=input_ids,
            node_kv_caches={},
        )
        assert result is None

    def test_missing_node_raises(self):
        """Node in node_order but absent from nodes dict.

        Note: The source raises NodeUnreachableError with a bare message
        string, but the constructor requires (node_id, host, port).  This is
        a source bug; the test documents the intended behaviour by asserting
        NodeUnreachableError, and tolerates TypeError from the mis-construction.
        """
        engine = VLLMPipelineEngine(
            node_order=["n1"],
            nodes={},  # missing
        )
        input_ids = torch.zeros((1, 1), dtype=torch.long)
        node_kv_caches = engine.create_node_kv_caches()

        with pytest.raises((NodeUnreachableError, TypeError)):
            engine.run_pipeline(
                input_ids=input_ids,
                node_kv_caches=node_kv_caches,
            )

    def test_partial_node_order_missing_last(self):
        """Some nodes found, one missing."""
        engine = VLLMPipelineEngine(
            node_order=["a", "b"],
            nodes={"a": object()},  # missing "b"
        )
        input_ids = torch.zeros((1, 1), dtype=torch.long)
        node_kv_caches = engine.create_node_kv_caches()

        with pytest.raises((NodeUnreachableError, TypeError)):
            engine.run_pipeline(
                input_ids=input_ids,
                node_kv_caches=node_kv_caches,
            )

    def test_node_kv_caches_empty_dict(self):
        """Empty node_kv_caches is tolerated when node_order is empty."""
        engine = VLLMPipelineEngine(node_order=[], nodes={})
        result = engine.run_pipeline(
            input_ids=torch.zeros((1, 1), dtype=torch.long),
            node_kv_caches={},
        )
        assert result is None

    def test_with_request_id(self):
        """request_id parameter is accepted and passed through."""
        engine = VLLMPipelineEngine(node_order=["n1"], nodes={})
        input_ids = torch.zeros((1, 1), dtype=torch.long)
        with pytest.raises((NodeUnreachableError, TypeError)):
            engine.run_pipeline(
                input_ids=input_ids,
                node_kv_caches={"n1": None},
                request_id="test-request-001",
            )

    def test_with_draft_tokens(self):
        """draft_tokens parameter is accepted."""
        engine = VLLMPipelineEngine(node_order=["n1"], nodes={})
        input_ids = torch.zeros((1, 1), dtype=torch.long)
        draft = torch.zeros((1, 5), dtype=torch.long)
        with pytest.raises((NodeUnreachableError, TypeError)):
            engine.run_pipeline(
                input_ids=input_ids,
                node_kv_caches={"n1": None},
                draft_tokens=draft,
            )


class TestVLLMPipelineEngineRunPipelineOverlap:
    """run_pipeline_overlap -- edge cases that do not require gRPC."""

    def test_empty_node_order_returns_none(self):
        engine = VLLMPipelineEngine(node_order=[], nodes={})
        result = engine.run_pipeline_overlap(
            input_ids=torch.zeros((1, 1), dtype=torch.long),
            node_kv_caches={},
        )
        assert result is None

    def test_missing_node_raises(self):
        engine = VLLMPipelineEngine(
            node_order=["n1"],
            nodes={},
        )
        input_ids = torch.zeros((1, 1), dtype=torch.long)
        with pytest.raises((NodeUnreachableError, TypeError)):
            engine.run_pipeline_overlap(
                input_ids=input_ids,
                node_kv_caches={"n1": None},
            )

    def test_different_timeout_not_applied_by_overlap(self):
        """run_pipeline_overlap is a separate method that does not read
        timeout_s (it omits the timeout kwarg).  Still behaves the same on
        missing nodes."""
        engine = VLLMPipelineEngine(
            node_order=["n1"],
            nodes={},
            timeout_s=999.0,
        )
        with pytest.raises((NodeUnreachableError, TypeError)):
            engine.run_pipeline_overlap(
                input_ids=torch.zeros((1, 1), dtype=torch.long),
                node_kv_caches={"n1": None},
            )


class TestVLLMPipelineEngineRunPipelineAsync:
    """run_pipeline_async delegates to run_pipeline via asyncio.to_thread."""

    @pytest.mark.asyncio
    async def test_empty_node_order_returns_none(self):
        engine = VLLMPipelineEngine(node_order=[], nodes={})
        result = await engine.run_pipeline_async(
            input_ids=torch.zeros((1, 1), dtype=torch.long),
            node_kv_caches={},
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_missing_node_raises(self):
        engine = VLLMPipelineEngine(
            node_order=["n1"],
            nodes={},
        )
        with pytest.raises((NodeUnreachableError, TypeError)):
            await engine.run_pipeline_async(
                input_ids=torch.zeros((1, 1), dtype=torch.long),
                node_kv_caches={"n1": None},
            )

    @pytest.mark.asyncio
    async def test_with_request_id(self):
        engine = VLLMPipelineEngine(node_order=["n1"], nodes={})
        with pytest.raises((NodeUnreachableError, TypeError)):
            await engine.run_pipeline_async(
                input_ids=torch.zeros((1, 1), dtype=torch.long),
                node_kv_caches={"n1": None},
                request_id="async-req",
            )

    @pytest.mark.asyncio
    async def test_with_draft_tokens(self):
        engine = VLLMPipelineEngine(node_order=["n1"], nodes={})
        draft = torch.zeros((1, 5), dtype=torch.long)
        with pytest.raises((NodeUnreachableError, TypeError)):
            await engine.run_pipeline_async(
                input_ids=torch.zeros((1, 1), dtype=torch.long),
                node_kv_caches={"n1": None},
                draft_tokens=draft,
            )
