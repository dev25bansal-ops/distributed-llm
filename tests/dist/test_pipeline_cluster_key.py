"""Regression tests: cluster_key propagation on pipeline->worker gRPC forwards.

Background: NodeServicer._check_auth fails closed — every ForwardPass RPC
must carry the shared cluster secret in its ``cluster_key`` protobuf field,
or the worker rejects it with "authentication failed".  The orchestrator's
dial-through forward call sites previously passed use_tls/ca_cert but never
cluster_key, so every secured deployment's data path was broken end-to-end.

These tests pin the full chain:

  1. PipelineOrchestrator holds the key (explicit arg > DISTLLM_CLUSTER_KEY
     env var) and forwards it to node_client.forward_request /
     forward_request_async on every dial-through call site (sync sequential
     path + async 1F1B execute_pipeline_step + run_stage).
  2. node_client.forward_request / forward_request_async stamp the key onto
     the ForwardPassRequest protobuf (sync and grpc.aio paths).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch

from distllm.dist.pipeline.serialization import to_proto_tensor
from distllm.dist.pipeline.orchestrator import PipelineOrchestrator

SECRET = "regression-test-cluster-key"


def register_two_nodes(orch: PipelineOrchestrator) -> None:
    """Two non-overlapping stages, matching test_pipeline_orchestrator.py."""
    orch.register_node("node-0", "10.0.0.1", 50051, 0, 15)
    orch.register_node("node-1", "10.0.0.2", 50051, 16, 31)


def _sync_echo(**kwargs: object) -> torch.Tensor:
    hs = kwargs.get("hidden_states")
    assert isinstance(hs, torch.Tensor)
    return hs.clone()


async def _async_echo(**kwargs: object) -> torch.Tensor:
    hs = kwargs.get("hidden_states")
    assert isinstance(hs, torch.Tensor)
    return hs.clone()


def _response_like(input_tensor: torch.Tensor):
    """Build a successful ForwardPassResponse echoing input shape."""
    from distllm.dist import node_pb2

    return node_pb2.ForwardPassResponse(
        request_id="r1",
        success=True,
        output=to_proto_tensor(input_tensor),
    )


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate DISTLLM_CLUSTER_KEY so machine env cannot skew assertions."""
    monkeypatch.delenv("DISTLLM_CLUSTER_KEY", raising=False)


# ===================================================================
# 1. Orchestrator holds the key
# ===================================================================


class TestOrchestratorKeyResolution:
    def test_explicit_arg(self) -> None:
        orch = PipelineOrchestrator(cluster_key=SECRET)
        assert orch.cluster_key == SECRET

    def test_env_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DISTLLM_CLUSTER_KEY", "env-key")
        orch = PipelineOrchestrator()
        assert orch.cluster_key == "env-key"

    def test_explicit_arg_overrides_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DISTLLM_CLUSTER_KEY", "env-key")
        orch = PipelineOrchestrator(cluster_key=SECRET)
        assert orch.cluster_key == SECRET

    def test_none_when_unset(self) -> None:
        orch = PipelineOrchestrator()
        assert orch.cluster_key is None

    def test_empty_env_treated_as_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DISTLLM_CLUSTER_KEY", "")
        orch = PipelineOrchestrator()
        assert orch.cluster_key is None

    def test_property_setter_for_post_construction_wiring(self) -> None:
        """ClusterManager/coordinator can push the key after construction."""
        orch = PipelineOrchestrator()
        assert orch.cluster_key is None
        orch.cluster_key = SECRET
        assert orch.cluster_key == SECRET


# ===================================================================
# 2. Sync sequential path forwards the key
# ===================================================================


class TestSyncPathPropagatesKey:
    def test_run_pipeline_passes_cluster_key(self) -> None:
        orch = PipelineOrchestrator(cluster_key=SECRET)
        register_two_nodes(orch)
        inp = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

        with patch(
            "distllm.dist.node_client.forward_request",
            side_effect=_sync_echo,
        ) as fwd:
            result = orch.run_pipeline(inp, {"node-0": None, "node-1": None}, "r1")

        assert fwd.call_count == 2
        for call in fwd.call_args_list:
            assert call.kwargs["cluster_key"] == SECRET
        torch.testing.assert_close(result, inp)

    def test_run_pipeline_without_key_sends_none(self) -> None:
        """Insecure deployments keep working: key stays None."""
        orch = PipelineOrchestrator()
        orch.register_node("n", "h", 1, 0, 15)
        inp = torch.tensor([[1]], dtype=torch.long)

        with patch(
            "distllm.dist.node_client.forward_request",
            side_effect=_sync_echo,
        ) as fwd:
            orch.run_pipeline(inp, {"n": None}, "r1")

        assert fwd.call_args.kwargs["cluster_key"] is None


# ===================================================================
# 3. Async 1F1B path forwards the key
# ===================================================================


class TestAsyncPathPropagatesKey:
    @pytest.mark.asyncio
    async def test_execute_pipeline_step_passes_key_on_every_hop(self) -> None:
        orch = PipelineOrchestrator(cluster_key=SECRET)
        register_two_nodes(orch)
        inp = torch.randn(4, 128)

        with patch(
            "distllm.dist.node_client.forward_request_async",
            side_effect=_async_echo,
        ) as fwd:
            result = await orch.run_pipeline_microbatched(
                inp, {"node-0": None, "node-1": None}, "r1", micro_batch_size=2,
            )

        # 2 stages x 2 batches = every micro-batch crosses both hops
        assert fwd.call_count == 4
        for call in fwd.call_args_list:
            assert call.kwargs["cluster_key"] == SECRET
        assert result.shape == (4, 128)

    @pytest.mark.asyncio
    async def test_env_resolved_key_reaches_forward_calls(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Key resolved from DISTLLM_CLUSTER_KEY flows through unchanged."""
        monkeypatch.setenv("DISTLLM_CLUSTER_KEY", SECRET)
        orch = PipelineOrchestrator()
        register_two_nodes(orch)

        with patch(
            "distllm.dist.node_client.forward_request_async",
            side_effect=_async_echo,
        ) as fwd:
            await orch.run_pipeline_microbatched(
                torch.randn(4, 128),
                {"node-0": None, "node-1": None},
                "r1", micro_batch_size=2,
            )

        assert fwd.call_count >= 1
        assert all(
            c.kwargs["cluster_key"] == SECRET for c in fwd.call_args_list
        )


# ===================================================================
# 4. run_stage helper (defined in run_pipeline_microbatched) — exercised
#    via direct construction of its enclosing scope is impossible, so we
#    pin the wiring contract at the node_client boundary instead.
# ===================================================================


class TestNodeClientStampsKeyOnProto:
    """The key must land in ForwardPassRequest.cluster_key (receiver-side
    NodeServicer._check_auth reads exactly this field)."""

    def test_sync_forward_request_stamps_cluster_key(self) -> None:
        from distllm.dist import node_pb2
        from distllm.dist.node_client import forward_request

        captured: list[node_pb2.ForwardPassRequest] = []

        def make_client(*args: object, **kwargs: object) -> MagicMock:
            client = MagicMock()

            def with_call(req: node_pb2.ForwardPassRequest) -> tuple:
                captured.append(req)
                resp = _response_like(torch.randn(2, 8))
                return resp, None

            client.stub.ForwardPass.with_call.side_effect = with_call
            return client

        inp = torch.randn(2, 8)
        with patch(
            "distllm.dist.node_client.create_node_client",
            side_effect=make_client,
        ):
            out = forward_request(
                host="10.0.0.1", port=50051, hidden_states=inp,
                request_id="r1", cluster_key=SECRET,
            )

        assert len(captured) == 1
        assert captured[0].cluster_key == SECRET
        assert out.shape == inp.shape

    def test_sync_forward_request_without_key_sends_empty(self) -> None:
        from distllm.dist import node_pb2
        from distllm.dist.node_client import forward_request

        captured: list[node_pb2.ForwardPassRequest] = []

        def make_client(*args: object, **kwargs: object) -> MagicMock:
            client = MagicMock()

            def with_call(req: node_pb2.ForwardPassRequest) -> tuple:
                captured.append(req)
                return _response_like(torch.randn(2, 8)), None

            client.stub.ForwardPass.with_call.side_effect = with_call
            return client

        with patch(
            "distllm.dist.node_client.create_node_client",
            side_effect=make_client,
        ):
            forward_request(
                host="10.0.0.1", port=50051,
                hidden_states=torch.randn(2, 8),
            )

        assert captured[0].cluster_key == ""

    @pytest.mark.asyncio
    async def test_async_forward_request_stamps_cluster_key(self) -> None:
        from distllm.dist.node_client import forward_request_async

        captured: list = []

        async def make_async_client(*args: object, **kwargs: object) -> MagicMock:
            client = MagicMock()

            async def forward(req, timeout: float = 30.0):  # noqa: ANN001
                captured.append(req)
                return _response_like(torch.randn(2, 8))

            client.stub.ForwardPass = forward
            client.close = AsyncMock()
            return client

        with patch(
            "distllm.dist.node_client.create_async_node_client",
            side_effect=make_async_client,
        ):
            await forward_request_async(
                host="10.0.0.2", port=50051,
                hidden_states=torch.randn(2, 8),
                request_id="r1", cluster_key=SECRET,
            )

        assert len(captured) == 1
        assert captured[0].cluster_key == SECRET


# ===================================================================
# 5. Receiver contract sanity (fail-closed auth accepts the wired chain)
# ===================================================================


class TestReceiverAcceptsWiredChain:
    @pytest.mark.asyncio
    async def test_end_to_end_field_shape_matches_check_auth(self) -> None:
        """The field the sender stamps is the field _check_auth reads."""
        import hmac

        from distllm.dist import node_pb2
        from distllm.dist.node_client import forward_request

        captured: list[node_pb2.ForwardPassRequest] = []

        def make_client(*args: object, **kwargs: object) -> MagicMock:
            client = MagicMock()

            def with_call(req: node_pb2.ForwardPassRequest) -> tuple:
                captured.append(req)
                return _response_like(torch.randn(1, 4)), None

            client.stub.ForwardPass.with_call.side_effect = with_call
            return client

        with patch(
            "distllm.dist.node_client.create_node_client",
            side_effect=make_client,
        ):
            forward_request(
                host="h", port=1, hidden_states=torch.randn(1, 4),
                cluster_key=SECRET,
            )

        req = captured[0]
        # Mirror of NodeServicer._check_auth logic against the sent proto.
        configured = SECRET
        assert bool(req.cluster_key), "keyless request would be rejected"
        assert hmac.compare_digest(req.cluster_key, configured)
