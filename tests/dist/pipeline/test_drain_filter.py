"""`admin drain` must stop NEW work on every serving path.

The audit found the serving path filtered on ``is_healthy`` only: the admin
drain verb marked a node in ``ResourceManager._draining_nodes`` and nothing
consulted it, so a drained node kept receiving requests and could never be
quiesced for removal. The first fix covered ``run_pipeline`` (sequential);
these tests pin the filter for all serving paths, including the
overlapped/micro-batched path that ``request_pipeline`` actually calls when
overlap or 1F1B scheduling is enabled, and the in-flight semantics: a
request already running completes, later requests do not reach the node.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from distllm.core.resource_manager import ResourceManager
from distllm.dist.pipeline.orchestrator import PipelineOrchestrator


class _CountingClient:
    """Injected per-node transport double that records every call."""

    def __init__(self) -> None:
        self.calls = 0

    def forward(self, hidden_states=None, kv_cache=None, request_id=None):
        self.calls += 1
        return hidden_states


def _orchestrator_with_two_nodes() -> tuple[
    PipelineOrchestrator, ResourceManager, dict[str, _CountingClient]
]:
    rm = ResourceManager()
    orch = PipelineOrchestrator(resource_mgr=rm)
    orch.register_node("node-0", "10.0.0.1", 50051, 0, 15)
    orch.register_node("node-1", "10.0.0.2", 50052, 16, 31)
    clients = {"node-0": _CountingClient(), "node-1": _CountingClient()}
    for nid, client in clients.items():
        orch.get_node(nid).client = client
    return orch, rm, clients


class TestSequentialDrainFilter:
    def test_drained_node_receives_no_new_work(self) -> None:
        orch, rm, clients = _orchestrator_with_two_nodes()
        rm.mark_node_draining("node-1")

        orch.run_pipeline(torch.ones(1, 4), {}, "req-1")

        assert clients["node-0"].calls == 1
        assert clients["node-1"].calls == 0, "drained node must not be routed to"

    def test_all_nodes_drained_refuses_new_work(self) -> None:
        orch, rm, _ = _orchestrator_with_two_nodes()
        rm.mark_node_draining("node-0")
        rm.mark_node_draining("node-1")

        with pytest.raises(RuntimeError, match="No healthy nodes"):
            orch.run_pipeline(torch.ones(1, 4), {}, "req-2")

    def test_undrain_restores_routing(self) -> None:
        orch, rm, clients = _orchestrator_with_two_nodes()
        rm.mark_node_draining("node-1")
        rm.mark_node_alive("node-1")

        orch.run_pipeline(torch.ones(1, 4), {}, "req-3")

        assert clients["node-1"].calls == 1

    def test_inflight_request_completes_while_drain_happens(self) -> None:
        """Drain takes effect for the NEXT request, not the running one.

        The node list is snapshotted when a request starts, so a drain
        landing mid-request must not abort the request already on the node
        (that is the "existing in-flight requests are allowed to complete"
        contract of the drain route).
        """
        orch, rm, clients = _orchestrator_with_two_nodes()

        original_forward = orch.get_node("node-1").client.forward

        def drain_while_forwarding(**kwargs):
            # The operator drains node-1 while this request is in flight.
            rm.mark_node_draining("node-1")
            return original_forward(**kwargs)

        orch.get_node("node-1").client.forward = drain_while_forwarding

        # In-flight request still visits the pre-drain node set…
        orch.run_pipeline(torch.ones(1, 4), {}, "req-inflight")
        assert clients["node-1"].calls == 1

        # …while the next request skips the drained node entirely.
        orch.run_pipeline(torch.ones(1, 4), {}, "req-after")
        assert clients["node-1"].calls == 1, "new work must skip the drained node"
        assert clients["node-0"].calls == 2


class TestMicrobatchedDrainFilter:
    @pytest.mark.asyncio
    async def test_overlap_path_honours_drain(self) -> None:
        """run_pipeline_microbatched (the overlap/1F1B serving path) too.

        The micro-batched path used to filter on is_healthy only, so a
        drained node kept receiving micro-batches even though the
        sequential path (and request_pipeline's enable_overlap branch,
        which calls this method) should have skipped it.
        """
        orch, rm, clients = _orchestrator_with_two_nodes()
        rm.mark_node_draining("node-1")

        await orch.run_pipeline_microbatched(
            torch.randn(2, 8), {}, "req-overlap", micro_batch_size=1,
        )

        assert clients["node-0"].calls > 0, "healthy stage must serve"
        assert clients["node-1"].calls == 0, (
            "drained stage must not run any micro-batch"
        )

    @pytest.mark.asyncio
    async def test_overlap_path_refuses_when_all_drained(self) -> None:
        orch, rm, _ = _orchestrator_with_two_nodes()
        rm.mark_node_draining("node-0")
        rm.mark_node_draining("node-1")

        with pytest.raises(RuntimeError, match="No healthy nodes"):
            await orch.run_pipeline_microbatched(
                torch.randn(2, 8), {}, "req-overlap2", micro_batch_size=1,
            )


class TestServingNodeIdsHelper:
    def test_healthy_and_not_draining(self) -> None:
        orch, rm, _ = _orchestrator_with_two_nodes()
        rm.mark_node_draining("node-1")
        assert orch.serving_node_ids() == ["node-0"]

        orch.get_node("node-0").is_healthy = False
        assert orch.serving_node_ids() == []

    def test_tolerates_missing_resource_manager(self) -> None:
        """No resource manager means no drain information — everything healthy serves."""
        orch = PipelineOrchestrator(resource_mgr=None)
        orch.register_node("node-0", "10.0.0.1", 50051, 0, 15)
        assert orch.serving_node_ids() == ["node-0"]

    def test_tolerates_a_broken_resource_manager(self) -> None:
        broken = MagicMock()
        broken.get_draining_nodes.side_effect = RuntimeError("boom")
        orch = PipelineOrchestrator(resource_mgr=broken)
        orch.register_node("node-0", "10.0.0.1", 50051, 0, 15)
        assert orch.serving_node_ids() == ["node-0"]