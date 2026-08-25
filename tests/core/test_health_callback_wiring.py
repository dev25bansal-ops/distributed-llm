"""C6 regression: health callbacks must not be clobbered; drain must write
the canonical ``is_healthy`` attribute.

Root cause: ``Coordinator.__init__`` wired NodeRecoveryManager callbacks
that set ``node.is_healthy = False``, then ``HealthManager.__init__``
overwrote them with closures setting ``node.healthy = False`` — an attribute
``PipelineNode`` does not define (it defines ``is_healthy``, which every
scheduler filter reads).  Drained/dead nodes therefore kept receiving work
and status snapshots reported healthy=False for every node.

Fix: HealthManager accepts the Coordinator's callbacks as constructor
params (single source of truth) and its own defaults use ``is_healthy``.
"""

from __future__ import annotations

from distllm.core.coordinator import Coordinator
from distllm.core.coordinator_config import CoordinatorConfig
from distllm.core.health_manager import HealthManager
from distllm.dist.recovery import NodeRecoveryManager


def make_coordinator(model: str = "test") -> Coordinator:
    return Coordinator(config=CoordinatorConfig(model_name=model))


def register_two_nodes(coord: Coordinator) -> None:
    """Register n0/n1 directly on the pipeline (no gRPC clients needed)."""
    coord._pipeline.register_node("n0", host="10.0.0.1", port=50051,
                                  start_layer=0, end_layer=1)
    coord._pipeline.register_node("n1", host="10.0.0.2", port=50052,
                                  start_layer=2, end_layer=3)


class TestWiringWins:
    """HealthManager must not overwrite the Coordinator's callbacks."""

    def test_recovery_manager_keeps_coordinator_drain_callback(self):
        coord = make_coordinator()
        # Before the fix, HealthManager.__init__ replaced this closure.
        assert coord._recovery_manager._on_drain == coord._on_node_drain

    def test_all_four_callbacks_survive_health_manager_init(self):
        coord = make_coordinator()
        rm = coord._recovery_manager
        assert rm._on_drain == coord._on_node_drain
        assert rm._on_mark_dead == coord._on_node_mark_dead
        assert rm._on_redistribute == coord._on_node_redistribute
        assert rm._on_recover == coord._on_node_recover


class TestDrainUsesCanonicalAttribute:
    """Draining a node must exclude it from scheduling."""

    def test_drained_node_excluded_from_scheduling(self):
        coord = make_coordinator()
        register_two_nodes(coord)

        healthy = coord._pipeline.get_healthy_nodes()
        assert set(healthy) == {"n0", "n1"}

        # The recovery drain path (what NodeRecoveryManager invokes).
        coord._on_node_drain("n1")

        assert coord._pipeline.get_healthy_nodes() == ["n0"]
        node = coord._pipeline.get_node("n1")
        assert node.is_healthy is False
        # The old bug wrote this nonexistent attribute instead:
        assert not hasattr(node, "healthy")

    def test_snapshot_shows_healthy_only_for_live_node(self):
        coord = make_coordinator()
        register_two_nodes(coord)
        coord._on_node_drain("n1")

        status = coord._health_mgr.get_node_status()
        assert status["n0"]["healthy"] is True
        assert status["n1"]["healthy"] is False

    def test_state_snapshot_reports_is_healthy(self):
        """HA snapshot must read the canonical attribute too."""
        coord = make_coordinator()
        register_two_nodes(coord)
        coord._on_node_drain("n1")

        snap = coord.state_snapshot()
        assert snap["nodes"]["n0"]["healthy"] is True
        assert snap["nodes"]["n1"]["healthy"] is False


class TestExplicitCallbackInjection:
    """HealthManager wires constructor-provided callbacks verbatim."""

    def test_explicit_drain_callback_used(self):
        coord = make_coordinator()
        register_two_nodes(coord)

        seen: list[str] = []
        hm = HealthManager(
            pipeline=coord._pipeline,
            resource_mgr=coord._resource_mgr,
            recovery_manager=NodeRecoveryManager(),
            drain_callback=seen.append,
        )
        hm.recovery_manager._on_drain("n1")
        assert seen == ["n1"]

    def test_default_callbacks_use_is_healthy(self):
        """Without explicit callbacks, internal defaults still write
        ``is_healthy`` (never the phantom ``healthy``)."""
        coord = make_coordinator()
        register_two_nodes(coord)

        hm = HealthManager(
            pipeline=coord._pipeline,
            resource_mgr=coord._resource_mgr,
            recovery_manager=NodeRecoveryManager(),
        )
        hm.recovery_manager._on_drain("n0")

        node = coord._pipeline.get_node("n0")
        assert node.is_healthy is False
        assert not hasattr(node, "healthy")
        status = hm.get_node_status()
        assert status["n0"]["healthy"] is False
        assert status["n1"]["healthy"] is True
