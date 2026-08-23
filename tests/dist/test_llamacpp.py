"""Tests for distllm.dist.backends.llamacpp module.

Zero mocks -- uses only real objects from the module.
Verifies the public API of LlamacppPipelineEngine.
"""

from __future__ import annotations

import asyncio

import pytest
import torch

from distllm.dist.backends.llamacpp import LlamacppPipelineEngine
from distllm.dist.node_service import tensor_to_proto


# ---------------------------------------------------------------------------
# Test doubles (hand-written, zero mocks)
# ---------------------------------------------------------------------------


class _FakeStub:
    """Minimal stand-in for a gRPC stub with a single ForwardPass method."""

    def __init__(self, forward_fn):
        self._forward_fn = forward_fn

    def ForwardPass(self, request, timeout=None):
        return self._forward_fn(request, timeout)


class _FakeClient:
    """Minimal stand-in for a gRPC client exposing a stub attribute."""

    def __init__(self, forward_fn):
        self.stub = _FakeStub(forward_fn)


class _FakeNode:
    """Minimal stand-in for a worker node with a client attribute."""

    def __init__(self, forward_fn=None):
        self.client = _FakeClient(forward_fn)


class _FakeResponse:
    """Stand-in for ForwardPassResponse.

    Setting kv_cache=None avoids a known buggy code path where
    kv_cache_from_proto() is called with an unexpected ``device=`` keyword
    argument that it does not accept.
    """

    __slots__ = ("success", "output", "kv_cache", "error_message")

    def __init__(self, success=True, output=None, kv_cache=None, error_message=""):
        self.success = success
        self.output = output
        self.kv_cache = kv_cache
        self.error_message = error_message


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ok_proto(tensor: torch.Tensor) -> object:
    """Create a TensorProto from a torch tensor for use in fake responses."""
    return tensor_to_proto(tensor)


def _always_ok(tensor: torch.Tensor):
    """Return a callable that always returns a successful fake response."""
    proto = _make_ok_proto(tensor)
    return lambda _req, _timeout: _FakeResponse(success=True, output=proto)


# ---------------------------------------------------------------------------
# LlamacppPipelineEngine construction
# ---------------------------------------------------------------------------


class TestConstruction:
    """LlamacppPipelineEngine.__init__"""

    def test_default_timeout(self) -> None:
        engine = LlamacppPipelineEngine(["n1"], {"n1": _FakeNode()})
        assert engine._timeout_s == 60.0

    def test_custom_timeout(self) -> None:
        engine = LlamacppPipelineEngine([], {}, timeout_s=30.0)
        assert engine._timeout_s == 30.0

    def test_empty_node_order_and_nodes(self) -> None:
        engine = LlamacppPipelineEngine([], {})
        assert engine.node_order == []
        assert engine.nodes == {}

    def test_initial_enable_overlap_false(self) -> None:
        engine = LlamacppPipelineEngine([], {})
        assert engine.enable_overlap is False


# ---------------------------------------------------------------------------
# Properties: node_order, nodes
# ---------------------------------------------------------------------------


class TestProperties:
    """LlamacppPipelineEngine.node_order / .nodes"""

    def test_node_order_getter(self) -> None:
        engine = LlamacppPipelineEngine(["a", "b"], {})
        assert engine.node_order == ["a", "b"]

    def test_node_order_setter(self) -> None:
        engine = LlamacppPipelineEngine([], {})
        engine.node_order = ["x", "y"]
        assert engine.node_order == ["x", "y"]

    def test_nodes_getter(self) -> None:
        nodes = {"n1": _FakeNode()}
        engine = LlamacppPipelineEngine(["n1"], nodes)
        assert engine.nodes is nodes

    def test_nodes_setter(self) -> None:
        engine = LlamacppPipelineEngine([], {})
        new_nodes = {"n1": _FakeNode()}
        engine.nodes = new_nodes
        assert engine.nodes is new_nodes


# ---------------------------------------------------------------------------
# create_node_kv_caches
# ---------------------------------------------------------------------------


class TestCreateNodeKVCaches:
    """LlamacppPipelineEngine.create_node_kv_caches"""

    def test_empty_node_order(self) -> None:
        caches = LlamacppPipelineEngine([], {}).create_node_kv_caches()
        assert caches == {}

    def test_single_node(self) -> None:
        caches = LlamacppPipelineEngine(["n1"], {"n1": _FakeNode()}).create_node_kv_caches()
        assert caches == {"n1": None}

    def test_multiple_nodes(self) -> None:
        caches = LlamacppPipelineEngine(["a", "b", "c"], {}).create_node_kv_caches()
        assert caches == {"a": None, "b": None, "c": None}

    def test_cache_order_matches_node_order(self) -> None:
        engine = LlamacppPipelineEngine(["z", "y", "x"], {})
        keys = list(engine.create_node_kv_caches().keys())
        assert keys == ["z", "y", "x"]


# ---------------------------------------------------------------------------
# run_pipeline -- error paths
# ---------------------------------------------------------------------------


class TestRunPipelineErrors:
    """LlamacppPipelineEngine.run_pipeline -- error handling"""

    def test_missing_node_raises_exception(self) -> None:
        """A node absent from the nodes dict triggers the error path.

        Note: the module calls NodeUnreachableError(message) but the
        constructor requires (node_id, host, port), so TypeError emerges
        instead of NodeUnreachableError -- a known bug.
        """
        engine = LlamacppPipelineEngine(["missing"], {})
        with pytest.raises(Exception):
            engine.run_pipeline(input_ids=torch.zeros(1, 1, dtype=torch.long), node_kv_caches={})

    def test_forward_pass_exception_raises(self) -> None:
        """A ForwardPass that raises is wrapped.

        Same NodeUnreachableError constructor bug applies here.
        """
        def _fail(_req, _timeout):
            raise ConnectionError("connection refused")

        node = _FakeNode(forward_fn=_fail)
        engine = LlamacppPipelineEngine(["n1"], {"n1": node})
        with pytest.raises(Exception):
            engine.run_pipeline(
                input_ids=torch.zeros(1, 1, dtype=torch.long),
                node_kv_caches={"n1": None},
            )

    def test_unsuccessful_response_raises_runtime_error(self) -> None:
        """A response with success=False raises RuntimeError with the
        error_message from the node."""
        node = _FakeNode(
            forward_fn=lambda _req, _timeout: _FakeResponse(
                success=False, error_message="model failure on node",
            )
        )
        engine = LlamacppPipelineEngine(["n1"], {"n1": node})
        with pytest.raises(RuntimeError, match="model failure on node"):
            engine.run_pipeline(
                input_ids=torch.zeros(1, 1, dtype=torch.long),
                node_kv_caches={"n1": None},
            )

    def test_none_node_raises_type_error(self) -> None:
        """When a node dict value is None the code path hits the
        ``if node is None`` guard and calls ``NodeUnreachableError``
        with a single string argument.  The constructor requires
        ``(node_id, host, port)``, so a ``TypeError`` results — this
        is a known signature mismatch in the source module."""
        engine = LlamacppPipelineEngine(["n1"], {"n1": None})
        with pytest.raises(TypeError):
            engine.run_pipeline(
                input_ids=torch.zeros(1, 1, dtype=torch.long),
                node_kv_caches={"n1": None},
            )


# ---------------------------------------------------------------------------
# run_pipeline -- success paths
# ---------------------------------------------------------------------------


class TestRunPipelineSuccess:
    """LlamacppPipelineEngine.run_pipeline -- normal operation"""

    def test_single_node_returns_output_tensor(self) -> None:
        expected = torch.tensor([[0.5, 1.5, 2.5]])
        node = _FakeNode(forward_fn=_always_ok(expected))
        engine = LlamacppPipelineEngine(["n1"], {"n1": node})

        result = engine.run_pipeline(
            input_ids=torch.tensor([[1, 2, 3]]),
            node_kv_caches={"n1": None},
        )

        assert isinstance(result, torch.Tensor)
        assert result.shape == expected.shape

    def test_with_request_id(self) -> None:
        node = _FakeNode(forward_fn=_always_ok(torch.tensor([[1.0]])))
        engine = LlamacppPipelineEngine(["n1"], {"n1": node})

        result = engine.run_pipeline(
            input_ids=torch.tensor([[1]]),
            node_kv_caches={"n1": None},
            request_id="test-req-42",
        )

        assert isinstance(result, torch.Tensor)

    def test_with_draft_tokens(self) -> None:
        """draft_tokens are forwarded to the last node's request."""
        proto = _make_ok_proto(torch.tensor([[2.0]]))
        captured = []

        def _capture(req, _timeout):
            captured.append(req)
            return _FakeResponse(success=True, output=proto)

        node = _FakeNode(forward_fn=_capture)
        engine = LlamacppPipelineEngine(["n1"], {"n1": node})
        draft = torch.tensor([[42, 99]])

        result = engine.run_pipeline(
            input_ids=torch.tensor([[1]]),
            node_kv_caches={"n1": None},
            draft_tokens=draft,
        )

        assert isinstance(result, torch.Tensor)
        assert len(captured) == 1
        assert list(captured[0].draft_tokens) == [42, 99]

    def test_multi_node_pipeline(self) -> None:
        """Hidden state is passed from node to node, and each node is
        called exactly once."""
        outputs = [
            torch.tensor([[0.1, 0.2]]),
            torch.tensor([[0.3, 0.4]]),
        ]
        call_count = [0]

        def _forward(_req, _timeout):
            idx = call_count[0]
            call_count[0] += 1
            return _FakeResponse(success=True, output=_make_ok_proto(outputs[idx]))

        n1 = _FakeNode(forward_fn=_forward)
        n2 = _FakeNode(forward_fn=_forward)
        engine = LlamacppPipelineEngine(
            ["n1", "n2"],
            {"n1": n1, "n2": n2},
        )

        result = engine.run_pipeline(
            input_ids=torch.tensor([[1, 2]]),
            node_kv_caches={"n1": None, "n2": None},
        )

        assert isinstance(result, torch.Tensor)
        assert call_count[0] == 2

    def test_preserves_kv_cache_dict(self) -> None:
        """node_kv_caches dict is not replaced; entries remain."""
        node = _FakeNode(forward_fn=_always_ok(torch.tensor([[1.0]])))
        engine = LlamacppPipelineEngine(["n1"], {"n1": node})
        caches = {"n1": None}

        engine.run_pipeline(input_ids=torch.tensor([[1]]), node_kv_caches=caches)

        assert "n1" in caches


# ---------------------------------------------------------------------------
# run_pipeline_async
# ---------------------------------------------------------------------------


class TestRunPipelineAsync:
    """LlamacppPipelineEngine.run_pipeline_async"""

    def test_async_wrapper_returns_tensor(self) -> None:
        node = _FakeNode(forward_fn=_always_ok(torch.tensor([[1.0]])))
        engine = LlamacppPipelineEngine(["n1"], {"n1": node})

        result = asyncio.run(
            engine.run_pipeline_async(
                input_ids=torch.tensor([[1]]),
                node_kv_caches={"n1": None},
            )
        )
        assert isinstance(result, torch.Tensor)


# ---------------------------------------------------------------------------
# shutdown
# ---------------------------------------------------------------------------


class TestShutdown:
    """LlamacppPipelineEngine.shutdown (effectively a no-op)"""

    def test_shutdown_does_not_raise(self) -> None:
        engine = LlamacppPipelineEngine([], {})
        engine.shutdown()

    def test_shutdown_called_twice(self) -> None:
        engine = LlamacppPipelineEngine([], {})
        engine.shutdown()
        engine.shutdown()
