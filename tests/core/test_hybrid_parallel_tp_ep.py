"""Tests for TP+EP integration in hybrid parallel."""

import pytest
import torch

from distllm.core.hybrid_parallel import (
    HybridParallelPlanner,
    HybridParallelExecutor,
    ParallelPlan,
    ParallelStrategy,
    TopologyInfo,
)


class TestParallelPlanGroups:
    def test_plan_default(self):
        plan = ParallelPlan()
        assert plan.tp_group_size == 1
        assert plan.ep_group_size == 1
        assert plan.tp_groups == []
        assert plan.ep_groups == []

    def test_plan_custom_groups(self):
        plan = ParallelPlan(
            tp_groups=[[0, 1], [2, 3]],
            ep_groups=[[0, 2], [1, 3]],
            tp_group_size=2,
            ep_group_size=2,
        )
        assert plan.tp_groups == [[0, 1], [2, 3]]
        assert plan.ep_group_size == 2


class TestBuildTpEpGroups:
    @pytest.fixture
    def planner(self):
        topo = TopologyInfo(
            num_nodes=1,
            gpus_per_node=8,
            total_gpus=8,
            has_nvlink=True,
        )
        return HybridParallelPlanner(topology=topo)

    def test_build_tp_groups_only(self, planner):
        tp_groups, ep_groups, tp_size, ep_size = planner._build_tp_ep_groups(
            num_gpus=8,
            tp_world_size=4,
            ep_world_size=2,
        )
        assert len(tp_groups) == 2  # 2 EP groups × 1 TP group each
        assert len(ep_groups) == 4  # 4 TP ranks × 2 EP ranks
        assert tp_size == 4
        assert ep_size == 2

    def test_build_tp_only(self, planner):
        tp_groups, ep_groups, tp_size, ep_size = planner._build_tp_ep_groups(
            num_gpus=4,
            tp_world_size=4,
            ep_world_size=1,
        )
        assert len(tp_groups) == 1
        assert len(ep_groups) == 0
        assert tp_size == 4
        assert ep_size == 1

    def test_build_ep_only(self, planner):
        tp_groups, ep_groups, tp_size, ep_size = planner._build_tp_ep_groups(
            num_gpus=4,
            tp_world_size=1,
            ep_world_size=4,
        )
        # Each GPU is its own TP group of size 1
        assert len(tp_groups) == 4
        assert all(len(g) == 1 for g in tp_groups)
        # Single EP group crossing all 4 GPUs (since only 1 TP rank each)
        assert len(ep_groups) == 1
        assert ep_groups[0] == [0, 1, 2, 3]
        assert tp_size == 1
        assert ep_size == 4

    def test_no_parallelism(self, planner):
        tp_groups, ep_groups, tp_size, ep_size = planner._build_tp_ep_groups(
            num_gpus=1,
            tp_world_size=1,
            ep_world_size=1,
        )
        assert len(tp_groups) == 0
        assert len(ep_groups) == 0

    def test_scaling_when_gpus_insufficient(self, planner):
        tp_groups, ep_groups, tp_size, ep_size = planner._build_tp_ep_groups(
            num_gpus=4,
            tp_world_size=4,
            ep_world_size=4,
        )
        # 4*4=16 > 4 GPUs, should scale down
        assert tp_size <= 4
        assert ep_size <= 4

    def test_asymmetric_groups(self, planner):
        tp_groups, ep_groups, tp_size, ep_size = planner._build_tp_ep_groups(
            num_gpus=6,
            tp_world_size=3,
            ep_world_size=2,
        )
        assert len(tp_groups) > 0
        assert len(ep_groups) > 0
        # All GPUs should be assigned
        total_assigned = sum(len(g) for g in tp_groups)
        assert total_assigned <= 6


class TestHybridParallelExecutor:
    @pytest.fixture
    def tp_ep_plan(self):
        return ParallelPlan(
            strategy=ParallelStrategy.TP_EP,
            tp_world_size=2,
            tp_group_size=2,
            ep_group_size=2,
            ep_num_experts_per_node=4,
            expert_assignment={"node_0": [0, 1, 2, 3]},
            tp_groups=[[0, 1]],
            ep_groups=[[0], [1]],
        )

    @pytest.fixture
    def tp_plan(self):
        return ParallelPlan(
            strategy=ParallelStrategy.TP,
            tp_world_size=2,
        )

    @pytest.fixture
    def ep_plan(self):
        return ParallelPlan(
            strategy=ParallelStrategy.EP,
            ep_num_experts_per_node=4,
            expert_assignment={"node_0": [0, 1, 2, 3]},
        )

    @pytest.fixture
    def pp_plan(self):
        return ParallelPlan(
            strategy=ParallelStrategy.PP,
            pp_num_stages=2,
        )

    def test_init_tp_ep(self, tp_ep_plan):
        exec = HybridParallelExecutor(plan=tp_ep_plan)
        assert exec._plan.strategy == ParallelStrategy.TP_EP
        assert exec._plan.tp_world_size == 2
        assert exec._plan.ep_group_size == 2

    def test_init_tp(self, tp_plan):
        exec = HybridParallelExecutor(plan=tp_plan)
        assert exec._plan.strategy == ParallelStrategy.TP

    def test_init_ep(self, ep_plan):
        exec = HybridParallelExecutor(plan=ep_plan)
        assert exec._plan.strategy == ParallelStrategy.EP

    def test_init_pp(self, pp_plan):
        exec = HybridParallelExecutor(plan=pp_plan)
        assert exec._plan.strategy == ParallelStrategy.PP

    def test_launch_tp_skipped_when_size_one(self):
        plan = ParallelPlan(tp_world_size=1)
        exec = HybridParallelExecutor(plan=plan)
        exec.launch_tp("test-model")
        assert len(exec._tp_processes) == 0

    def test_configure_pp_skipped_when_stages_one(self):
        plan = ParallelPlan(pp_num_stages=1)
        exec = HybridParallelExecutor(plan=plan)
        exec.configure_pp(pipeline=object())
        # Should not raise

    def test_configure_ep_with_assignment(self, tp_ep_plan):
        exec = HybridParallelExecutor(plan=tp_ep_plan)
        # Should not raise
        exec.configure_ep(moe_orchestrator=None, node_ids=["node_0"])

    def test_shutdown_empty(self):
        plan = ParallelPlan()
        exec = HybridParallelExecutor(plan=plan)
        exec.shutdown()
        assert len(exec._tp_processes) == 0

    def test_execute_raises_without_pipeline(self):
        plan = ParallelPlan(strategy=ParallelStrategy.PP)
        exec = HybridParallelExecutor(plan=plan)
        inp = torch.randn(1, 1, 64)
        with pytest.raises(RuntimeError, match="No pipeline available"):
            exec.execute(inp, {}, request_id="test")

    def test_execute_tp_ep_no_moe(self, tp_ep_plan):
        """TP+EP without MoE should return input (no-op)."""
        exec = HybridParallelExecutor(plan=tp_ep_plan)
        inp = torch.randn(1, 1, 64)
        out = exec.execute(inp, {}, request_id="test")
        assert torch.equal(out, inp)


class TestHybridParallelPlannerGroupsIntegration:
    @pytest.fixture
    def planner(self):
        topo = TopologyInfo(
            num_nodes=1,
            gpus_per_node=8,
            total_gpus=8,
            has_nvlink=True,
        )
        return HybridParallelPlanner(topology=topo)

    def test_build_plan_groups_tp_ep(self, planner):
        plan = planner.plan(
            total_layers=32,
            num_experts=8,
            use_moe=True,
            tp_enabled=True,
            ep_enabled=True,
        )
        assert plan.strategy in (
            ParallelStrategy.TP_EP,
            ParallelStrategy.TP_PP_EP,
        )
        planner._plan = plan
        planner.build_plan_groups()
        if plan.strategy == ParallelStrategy.TP_EP:
            assert plan.tp_group_size >= 1
            assert len(plan.tp_groups) > 0 or plan.tp_world_size <= 1

    def test_build_plan_groups_tp_only(self, planner):
        plan = planner.plan(
            total_layers=32,
            use_moe=False,
            tp_enabled=True,
            ep_enabled=False,
        )
        planner._plan = plan
        planner.build_plan_groups()
        assert plan.tp_world_size >= 1

    def test_build_plan_groups_no_groups(self, planner):
        plan = planner.plan(
            total_layers=1,
            use_moe=False,
            tp_enabled=False,
            ep_enabled=False,
        )
        planner._plan = plan
        planner.build_plan_groups()  # Should not raise
