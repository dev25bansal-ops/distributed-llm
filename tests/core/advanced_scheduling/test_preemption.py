"""Tests for NodePreemptionState and DistributedPreemptionCoordinator."""

from __future__ import annotations

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_preempt = load_module("distllm/core/advanced_scheduling/preemption.py")
NodePreemptionState = _preempt.NodePreemptionState
DistributedPreemptionCoordinator = _preempt.DistributedPreemptionCoordinator


class TestNodePreemptionState:
    """Test suite for NodePreemptionState dataclass."""

    def test_default_construction(self) -> None:
        state = NodePreemptionState(node_id="gpu-0")
        assert state.node_id == "gpu-0"
        assert state.is_preempted is False
        assert state.preempted_at == 0.0
        assert state.reason == ""
        assert state.estimated_resume_time == 0.0

    def test_preempted_state(self) -> None:
        state = NodePreemptionState(
            node_id="gpu-0",
            is_preempted=True,
            preempted_at=1000.0,
            reason="higher_priority_job",
            estimated_resume_time=1100.0,
        )
        assert state.is_preempted is True
        assert state.reason == "higher_priority_job"
        assert state.estimated_resume_time == 1100.0


class TestDistributedPreemptionCoordinator:
    """Test suite for DistributedPreemptionCoordinator."""

    def test_default_construction(self) -> None:
        coord = DistributedPreemptionCoordinator()
        assert coord._max_preempted_fraction == 0.3
        assert coord._states == {}

    def test_custom_max_fraction(self) -> None:
        coord = DistributedPreemptionCoordinator(max_preempted_fraction=0.5)
        assert coord._max_preempted_fraction == 0.5

    def test_update_state(self) -> None:
        coord = DistributedPreemptionCoordinator()
        state = NodePreemptionState(node_id="gpu-0")
        coord.update_state(state)
        assert coord._states["gpu-0"] is state

    def test_update_state_overwrites(self) -> None:
        coord = DistributedPreemptionCoordinator()
        coord.update_state(
            NodePreemptionState(node_id="gpu-0", is_preempted=False)
        )
        coord.update_state(
            NodePreemptionState(node_id="gpu-0", is_preempted=True)
        )
        assert coord._states["gpu-0"].is_preempted is True

    def test_should_preempt_no_states(self) -> None:
        coord = DistributedPreemptionCoordinator()
        assert coord.should_preempt("gpu-0") is False

    def test_should_preempt_below_fraction(self) -> None:
        coord = DistributedPreemptionCoordinator(max_preempted_fraction=0.5)
        coord.update_state(NodePreemptionState(node_id="gpu-0", is_preempted=False))
        coord.update_state(NodePreemptionState(node_id="gpu-1", is_preempted=False))
        coord.update_state(NodePreemptionState(node_id="gpu-2", is_preempted=True))
        # preempted/total = 1/3 = 0.33 < 0.5 => should preempt
        assert coord.should_preempt("gpu-2") is True

    def test_should_preempt_at_fraction(self) -> None:
        coord = DistributedPreemptionCoordinator(max_preempted_fraction=0.5)
        coord.update_state(NodePreemptionState(node_id="gpu-0", is_preempted=True))
        coord.update_state(NodePreemptionState(node_id="gpu-1", is_preempted=True))
        coord.update_state(NodePreemptionState(node_id="gpu-2", is_preempted=False))
        coord.update_state(NodePreemptionState(node_id="gpu-3", is_preempted=False))
        # preempted/total = 2/4 = 0.5, not strictly < 0.5
        assert coord.should_preempt("gpu-0") is False

    def test_should_preempt_above_fraction(self) -> None:
        coord = DistributedPreemptionCoordinator(max_preempted_fraction=0.3)
        coord.update_state(NodePreemptionState(node_id="gpu-0", is_preempted=True))
        coord.update_state(NodePreemptionState(node_id="gpu-1", is_preempted=True))
        # preempted/total = 2/2 = 1.0 > 0.3
        assert coord.should_preempt("gpu-0") is False

    def test_get_preempted_nodes(self) -> None:
        coord = DistributedPreemptionCoordinator()
        coord.update_state(NodePreemptionState(node_id="gpu-0", is_preempted=True))
        coord.update_state(NodePreemptionState(node_id="gpu-1", is_preempted=False))
        coord.update_state(NodePreemptionState(node_id="gpu-2", is_preempted=True))

        preempted = coord.get_preempted_nodes()
        assert set(preempted) == {"gpu-0", "gpu-2"}

    def test_get_preempted_nodes_empty(self) -> None:
        coord = DistributedPreemptionCoordinator()
        coord.update_state(NodePreemptionState(node_id="gpu-0", is_preempted=False))
        assert coord.get_preempted_nodes() == []
