"""Tests for the pipeline context module (NodeForwardContext, NodeCheckpoint).

Tests cover direct construction, the build() classmethod, edge cases
with None/empty values, and default field behavior — all real objects,
no mocks, no GPU required.
"""

from __future__ import annotations

import pytest
import torch

from distllm.dist.pipeline.context import NodeCheckpoint, NodeForwardContext


# =========================================================================
# NodeForwardContext
# =========================================================================


class TestNodeForwardContext:
    """Direct construction and field behaviour of NodeForwardContext."""

    def test_construct_basic(self) -> None:
        """Populate every field explicitly."""
        input_ids = torch.zeros((2, 4), dtype=torch.long)
        ctx = NodeForwardContext(
            node_id="node-a",
            node_kv_caches={"layer_0": None, "layer_1": None},
            current_hidden=None,
            request_id="req-001",
            draft_tokens=[101, 102],
            input_ids=input_ids,
            seq_len=4,
            batch_size=2,
            is_first_node=True,
            is_last_node=False,
        )
        assert ctx.node_id == "node-a"
        assert ctx.node_kv_caches == {"layer_0": None, "layer_1": None}
        assert ctx.current_hidden is None
        assert ctx.request_id == "req-001"
        assert ctx.draft_tokens == [101, 102]
        assert ctx.input_ids is input_ids
        assert ctx.seq_len == 4
        assert ctx.batch_size == 2
        assert ctx.is_first_node is True
        assert ctx.is_last_node is False

    def test_construct_node_kv_caches_empty_dict(self) -> None:
        """Empty dict is a valid value for node_kv_caches."""
        input_ids = torch.ones((1, 3), dtype=torch.long)
        ctx = NodeForwardContext(
            node_id="n1",
            node_kv_caches={},
            current_hidden=None,
            request_id="r1",
            draft_tokens=None,
            input_ids=input_ids,
            seq_len=3,
            batch_size=1,
            is_first_node=False,
            is_last_node=True,
        )
        assert ctx.node_kv_caches == {}
        assert ctx.draft_tokens is None

    def test_construct_draft_tokens_none(self) -> None:
        """draft_tokens can be None (no speculative decoding)."""
        input_ids = torch.zeros((1, 1), dtype=torch.long)
        ctx = NodeForwardContext(
            node_id="n1",
            node_kv_caches={},
            current_hidden=None,
            request_id="r1",
            draft_tokens=None,
            input_ids=input_ids,
            seq_len=1,
            batch_size=1,
            is_first_node=False,
            is_last_node=False,
        )
        assert ctx.draft_tokens is None

    def test_construct_current_hidden_none(self) -> None:
        """current_hidden can be None (first node or prefill)."""
        input_ids = torch.zeros((1, 8), dtype=torch.long)
        ctx = NodeForwardContext(
            node_id="n1",
            node_kv_caches={},
            current_hidden=None,
            request_id="r1",
            draft_tokens=None,
            input_ids=input_ids,
            seq_len=8,
            batch_size=1,
            is_first_node=True,
            is_last_node=False,
        )
        assert ctx.current_hidden is None

    def test_field_types_are_correct(self) -> None:
        """Verify each field has the expected runtime type."""
        input_ids = torch.randint(0, 100, (2, 5))
        ctx = NodeForwardContext(
            node_id="x",
            node_kv_caches={"k": None},
            current_hidden=torch.randn(2, 5, 64),
            request_id="rid",
            draft_tokens=[1, 2, 3],
            input_ids=input_ids,
            seq_len=5,
            batch_size=2,
            is_first_node=False,
            is_last_node=False,
        )
        assert isinstance(ctx.node_id, str)
        assert isinstance(ctx.node_kv_caches, dict)
        assert isinstance(ctx.current_hidden, torch.Tensor)
        assert isinstance(ctx.request_id, str)
        assert isinstance(ctx.draft_tokens, list)
        assert isinstance(ctx.input_ids, torch.Tensor)
        assert isinstance(ctx.seq_len, int)
        assert isinstance(ctx.batch_size, int)
        assert isinstance(ctx.is_first_node, bool)
        assert isinstance(ctx.is_last_node, bool)


class TestNodeForwardContextBuild:
    """build() classmethod — auto-derives seq_len and batch_size."""

    def test_build_from_input_ids(self) -> None:
        """seq_len and batch_size derived from input_ids shape."""
        input_ids = torch.zeros((3, 7), dtype=torch.long)
        ctx = NodeForwardContext.build(
            node_id="n1",
            node_kv_caches={},
            current_hidden=None,
            request_id="r1",
            draft_tokens=None,
            input_ids=input_ids,
            is_first_node=True,
            is_last_node=False,
        )
        assert ctx.seq_len == 7
        assert ctx.batch_size == 3
        assert ctx.input_ids is input_ids
        assert ctx.current_hidden is None

    def test_build_from_current_hidden_when_input_ids_none(self) -> None:
        """Fallback to current_hidden shape when input_ids is None."""
        hidden = torch.randn(4, 12, 128)
        ctx = NodeForwardContext.build(
            node_id="n1",
            node_kv_caches={},
            current_hidden=hidden,
            request_id="r1",
            draft_tokens=None,
            input_ids=None,  # type: ignore[arg-type]
            is_first_node=False,
            is_last_node=True,
        )
        assert ctx.seq_len == 12
        assert ctx.batch_size == 4
        assert ctx.input_ids is None

    def test_build_from_current_hidden_input_ids_not_none(self) -> None:
        """input_ids takes priority over current_hidden when both provided."""
        input_ids = torch.zeros((2, 5), dtype=torch.long)
        hidden = torch.randn(8, 20, 64)
        ctx = NodeForwardContext.build(
            node_id="n1",
            node_kv_caches={},
            current_hidden=hidden,
            request_id="r1",
            draft_tokens=None,
            input_ids=input_ids,
            is_first_node=True,
            is_last_node=True,
        )
        # input_ids shape should win
        assert ctx.seq_len == 5
        assert ctx.batch_size == 2

    def test_build_both_none_uses_defaults(self) -> None:
        """When both input_ids and current_hidden are None, defaults to 1/1."""
        ctx = NodeForwardContext.build(
            node_id="n1",
            node_kv_caches={},
            current_hidden=None,
            request_id="r1",
            draft_tokens=None,
            input_ids=None,  # type: ignore[arg-type]
            is_first_node=False,
            is_last_node=False,
        )
        assert ctx.seq_len == 1
        assert ctx.batch_size == 1
        assert ctx.input_ids is None
        assert ctx.current_hidden is None

    def test_build_single_token(self) -> None:
        """Single token: batch_size=1, seq_len=1."""
        input_ids = torch.zeros((1, 1), dtype=torch.long)
        ctx = NodeForwardContext.build(
            node_id="n1",
            node_kv_caches={},
            current_hidden=None,
            request_id="r1",
            draft_tokens=[100],
            input_ids=input_ids,
            is_first_node=False,
            is_last_node=False,
        )
        assert ctx.seq_len == 1
        assert ctx.batch_size == 1
        assert ctx.draft_tokens == [100]

    def test_build_derived_values_with_draft_tokens(self) -> None:
        """draft_tokens are forwarded as-is through build()."""
        input_ids = torch.zeros((1, 3), dtype=torch.long)
        ctx = NodeForwardContext.build(
            node_id="n1",
            node_kv_caches={},
            current_hidden=None,
            request_id="r1",
            draft_tokens=[10, 20],
            input_ids=input_ids,
            is_first_node=True,
            is_last_node=False,
        )
        assert ctx.draft_tokens == [10, 20]
        assert ctx.seq_len == 3
        assert ctx.batch_size == 1

    def test_build_node_kv_caches_passthrough(self) -> None:
        """node_kv_caches dict is passed unchanged."""
        caches = {"layer_0": None, "layer_1": [torch.zeros(4, 4)]}
        input_ids = torch.zeros((1, 2), dtype=torch.long)
        ctx = NodeForwardContext.build(
            node_id="n1",
            node_kv_caches=caches,
            current_hidden=None,
            request_id="r1",
            draft_tokens=None,
            input_ids=input_ids,
            is_first_node=False,
            is_last_node=True,
        )
        assert ctx.node_kv_caches is caches
        assert ctx.node_kv_caches["layer_0"] is None
        assert isinstance(ctx.node_kv_caches["layer_1"][0], torch.Tensor)

    def test_build_flag_values(self) -> None:
        """is_first_node and is_last_node forwarded exactly."""
        input_ids = torch.zeros((1, 2), dtype=torch.long)
        ctx = NodeForwardContext.build(
            node_id="n1",
            node_kv_caches={},
            current_hidden=None,
            request_id="r1",
            draft_tokens=None,
            input_ids=input_ids,
            is_first_node=True,
            is_last_node=True,
        )
        assert ctx.is_first_node is True
        assert ctx.is_last_node is True

    def test_build_both_flags_false(self) -> None:
        """Middle node: both flags False."""
        input_ids = torch.zeros((1, 2), dtype=torch.long)
        ctx = NodeForwardContext.build(
            node_id="mid-node",
            node_kv_caches={},
            current_hidden=None,
            request_id="r1",
            draft_tokens=None,
            input_ids=input_ids,
            is_first_node=False,
            is_last_node=False,
        )
        assert ctx.is_first_node is False
        assert ctx.is_last_node is False


# =========================================================================
# NodeCheckpoint
# =========================================================================


class TestNodeCheckpoint:
    """Construction and default behaviour of NodeCheckpoint."""

    def test_construct_minimal(self) -> None:
        """Required fields only — optional fields get defaults."""
        cp = NodeCheckpoint(
            request_id="req-001",
            node_id="node-a",
            node_index=0,
        )
        assert cp.request_id == "req-001"
        assert cp.node_id == "node-a"
        assert cp.node_index == 0
        # defaults
        assert cp.hidden_state is None
        assert cp.kv_cache is None
        assert cp.input_ids is None
        assert cp.draft_tokens is None

    def test_construct_all_fields(self) -> None:
        """Every field populated explicitly."""
        hidden = torch.randn(2, 8, 64)
        kv = {"layer_0": [torch.zeros(2, 4)]}
        inp = torch.randint(0, 100, (2, 8))
        cp = NodeCheckpoint(
            request_id="req-001",
            node_id="node-a",
            node_index=3,
            hidden_state=hidden,
            kv_cache=kv,
            input_ids=inp,
            draft_tokens=[101, 102],
        )
        assert cp.request_id == "req-001"
        assert cp.node_id == "node-a"
        assert cp.node_index == 3
        assert cp.hidden_state is hidden
        assert cp.kv_cache is kv
        assert cp.input_ids is inp
        assert cp.draft_tokens == [101, 102]

    def test_kv_cache_as_list(self) -> None:
        """kv_cache can also be a plain list (valid union type)."""
        kv_list = [torch.ones(2, 4)]
        cp = NodeCheckpoint(
            request_id="r1",
            node_id="n1",
            node_index=1,
            kv_cache=kv_list,
        )
        assert cp.kv_cache is kv_list
        assert isinstance(cp.kv_cache, list)
        assert torch.equal(cp.kv_cache[0], torch.ones(2, 4))  # type: ignore[union-attr]

    def test_kv_cache_none_explicit(self) -> None:
        """kv_cache set to None explicitly."""
        cp = NodeCheckpoint(
            request_id="r1",
            node_id="n1",
            node_index=2,
            kv_cache=None,
        )
        assert cp.kv_cache is None

    def test_input_ids_none(self) -> None:
        """input_ids defaults to None."""
        cp = NodeCheckpoint(
            request_id="r1",
            node_id="n1",
            node_index=0,
        )
        assert cp.input_ids is None

    def test_hidden_state_none(self) -> None:
        """hidden_state defaults to None."""
        cp = NodeCheckpoint(
            request_id="r1",
            node_id="n1",
            node_index=0,
        )
        assert cp.hidden_state is None

    def test_draft_tokens_default_none(self) -> None:
        """draft_tokens defaults to None."""
        cp = NodeCheckpoint(
            request_id="r1",
            node_id="n1",
            node_index=0,
        )
        assert cp.draft_tokens is None
        cp_with_drafts = NodeCheckpoint(
            request_id="r1",
            node_id="n1",
            node_index=0,
            draft_tokens=[1, 2, 3],
        )
        assert cp_with_drafts.draft_tokens == [1, 2, 3]

    def test_field_types(self) -> None:
        """Runtime type checks for all fields."""
        hidden = torch.randn(1, 4, 32)
        kv = {"k": [hidden]}
        inp = torch.zeros((1, 4), dtype=torch.long)
        cp = NodeCheckpoint(
            request_id="req-x",
            node_id="node-x",
            node_index=5,
            hidden_state=hidden,
            kv_cache=kv,
            input_ids=inp,
            draft_tokens=[99],
        )
        assert isinstance(cp.request_id, str)
        assert isinstance(cp.node_id, str)
        assert isinstance(cp.node_index, int)
        assert isinstance(cp.hidden_state, torch.Tensor)
        assert isinstance(cp.kv_cache, dict)
        assert isinstance(cp.input_ids, torch.Tensor)
        assert isinstance(cp.draft_tokens, list)

    def test_negative_node_index(self) -> None:
        """node_index can be negative (edge case; no validation in dataclass)."""
        cp = NodeCheckpoint(
            request_id="r1",
            node_id="n1",
            node_index=-1,
        )
        assert cp.node_index == -1

    def test_empty_request_id(self) -> None:
        """Empty string is valid for request_id."""
        cp = NodeCheckpoint(
            request_id="",
            node_id="n1",
            node_index=0,
        )
        assert cp.request_id == ""

    def test_empty_node_id(self) -> None:
        """Empty string is valid for node_id."""
        cp = NodeCheckpoint(
            request_id="r1",
            node_id="",
            node_index=0,
        )
        assert cp.node_id == ""

    def test_checkpoint_immutable_fields(self) -> None:
        """NodeCheckpoint is a regular dataclass (not frozen); fields are settable."""
        cp = NodeCheckpoint(request_id="r1", node_id="n1", node_index=0)
        cp.node_index = 42  # mutable dataclass
        assert cp.node_index == 42


# =========================================================================
# Cross-component: NodeForwardContext → NodeCheckpoint round-trip
# =========================================================================


class TestContextCheckpointRoundTrip:
    """Verify the two dataclasses can cooperate in a realistic workflow."""

    def test_forward_context_to_checkpoint(self) -> None:
        """Simulate saving a checkpoint from a forward context's data."""
        input_ids = torch.randint(0, 100, (1, 6))
        ctx = NodeForwardContext.build(
            node_id="node-2",
            node_kv_caches={"layer_0": [torch.randn(1, 2, 4)]},
            current_hidden=torch.randn(1, 6, 64),
            request_id="req-roundtrip",
            draft_tokens=[50, 51],
            input_ids=input_ids,
            is_first_node=False,
            is_last_node=False,
        )
        # Simulate creating a checkpoint from the context state
        cp = NodeCheckpoint(
            request_id=ctx.request_id,
            node_id=ctx.node_id,
            node_index=1,
            hidden_state=ctx.current_hidden,
            kv_cache=dict(ctx.node_kv_caches),  # shallow copy
            input_ids=ctx.input_ids,
            draft_tokens=ctx.draft_tokens,
        )
        assert cp.request_id == "req-roundtrip"
        assert cp.node_id == "node-2"
        assert cp.node_index == 1
        assert cp.hidden_state is ctx.current_hidden
        assert isinstance(cp.kv_cache, dict)
        assert cp.input_ids is input_ids
        assert cp.draft_tokens == [50, 51]
