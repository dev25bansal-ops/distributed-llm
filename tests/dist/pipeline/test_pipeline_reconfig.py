"""Regression tests for PipelineReconfigurator.

Focus: the rollback path restores nodes onto the orchestrator correctly.
Historically it read ``node_info["host"]`` / ``node_info["port"]`` off
PipelineNode objects (TypeError: not subscriptable) — and did so AFTER
unregistering every node, so connection info was always gone by the time it
was needed. Both defects are pinned here.
"""

from __future__ import annotations

import asyncio

from distllm.dist.pipeline.orchestrator import PipelineOrchestrator
from distllm.dist.pipeline.pipeline_reconfig import (
    NodeAssignment,
    PipelineCheckpointer,
    PipelineReconfigurator,
    TopologyVersion,
)


def _make_reconf(orch: PipelineOrchestrator) -> PipelineReconfigurator:
    versions = {
        0: TopologyVersion.create(
            0,
            "two-laptop-topology",
            [
                NodeAssignment("node-0", 0, 3),
                NodeAssignment("node-1", 4, 7),
            ],
        ),
    }
    return PipelineReconfigurator(
        orchestrator=orch,
        checkpointer=PipelineCheckpointer(),
        topology_versions=versions,
    )


class TestRollbackRestoresNodes:
    """rollback() must rebuild the orchestrator from live node info."""

    def test_rollback_restores_connection_info(self) -> None:
        orch = PipelineOrchestrator()
        orch.register_node("node-0", "10.0.0.1", 50051, 0, 3)
        orch.register_node("node-1", "10.0.0.2", 50052, 4, 7)
        reconf = _make_reconf(orch)

        ok = asyncio.run(reconf.rollback(0))
        assert ok is True

        # Nodes re-registered with their ORIGINAL host/port, taken from the
        # PipelineNode objects that existed before the orchestrator was cleared.
        node0 = orch.get_node("node-0")
        assert node0 is not None
        assert (node0.host, node0.port) == ("10.0.0.1", 50051)
        assert (node0.start_layer, node0.end_layer) == (0, 3)
        node1 = orch.get_node("node-1")
        assert node1 is not None
        assert (node1.host, node1.port) == ("10.0.0.2", 50052)
        assert (node1.start_layer, node1.end_layer) == (4, 7)

    def test_rollback_restores_all_target_assignments(self) -> None:
        orch = PipelineOrchestrator()
        # Only one node currently registered; version 0 expects two.
        orch.register_node("node-0", "10.0.0.1", 50051, 0, 3)
        reconf = _make_reconf(orch)

        ok = asyncio.run(reconf.rollback(0))
        assert ok is True
        assert set(orch.node_order) == {"node-0"}
        # node-1 has no live PipelineNode to snapshot → skipped with a
        # warning, not a crash.
        assert orch.get_node("node-1") is None

    def test_rollback_unknown_version_fails_cleanly(self) -> None:
        orch = PipelineOrchestrator()
        reconf = _make_reconf(orch)
        assert asyncio.run(reconf.rollback(99)) is False
