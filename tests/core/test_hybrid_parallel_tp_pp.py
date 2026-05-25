"""Tests for HybridParallelExecutor TP+PP combined execution."""

import pytest
import torch

from distllm.core.hybrid_parallel import (
    HybridParallelExecutor,
    HybridParallelPlanner,
    ParallelPlan,
    ParallelStrategy,
    TopologyInfo,
    _compute_chunk_sizes,
)


class FakePipeline:
    """Simulates a pipeline orchestrator for testing TP+PP execution."""

    def __init__(self):
        self.enable_overlap = False
        self._call_log: list[tuple[str, torch.Tensor]] = []

    def run_pipeline(self, hidden, node_kv_caches, request_id="", draft_tokens=None):
        self._call_log.append(("run_pipeline", hidden.clone()))
        return hidden * 2.0  # double as fake "logits"

    def run_pipeline_overlap(self, hidden, node_kv_caches, request_id="", draft_tokens=None):
        self._call_log.append(("run_pipeline_overlap", hidden.clone()))
        return hidden * 2.0


class FakeMoE:
    def __init__(self):
        self.experts = []
        self.last_router_logits = None

    def all_to_all_dispatch(self, tensor):
        return tensor


class TestTpPpChunkSizes:
    def test_even_split(self):
        sizes = _compute_chunk_sizes(4096, 4)
        assert len(sizes) == 4
        for start, end in sizes:
            assert end - start == 1024

    def test_uneven_split(self):
        sizes = _compute_chunk_sizes(10, 3)
        assert len(sizes) == 3
        total = sum(end - start for start, end in sizes)
        assert total == 10

    def test_single_chunk(self):
        sizes = _compute_chunk_sizes(64, 1)
        assert sizes == [(0, 64)]


class TestTpPpPlan:
    def test_planner_selects_tp_pp(self):
        topo = TopologyInfo(
            num_nodes=2,
            gpus_per_node=4,
            has_nvlink=True,
            total_gpus=8,
        )
        planner = HybridParallelPlanner(topo)
        plan = planner.plan(total_layers=32)
        assert plan.strategy == ParallelStrategy.TP_PP
        assert plan.tp_world_size == 4
        assert plan.pp_num_stages == 2

    def test_tp_pp_layers_distributed(self):
        topo = TopologyInfo(num_nodes=2, gpus_per_node=4, has_nvlink=True, total_gpus=8)
        planner = HybridParallelPlanner(topo)
        plan = planner.plan(total_layers=32)
        assert len(plan.layers_per_stage) == 2
        layer_count = sum(end - start + 1 for start, end in plan.layers_per_stage)
        assert layer_count == 32

    def test_tp_pp_nodes_per_stage(self):
        topo = TopologyInfo(num_nodes=2, gpus_per_node=4, has_nvlink=True, total_gpus=8)
        planner = HybridParallelPlanner(topo)
        plan = planner.plan(total_layers=32)
        assert len(plan.nodes_per_stage) == 2

    def test_planner_falls_back_to_pp_without_nvlink(self):
        topo = TopologyInfo(num_nodes=2, gpus_per_node=4, has_nvlink=False, total_gpus=8)
        planner = HybridParallelPlanner(topo)
        plan = planner.plan(total_layers=32, tp_enabled=True)
        assert plan.strategy != ParallelStrategy.TP_PP

    def test_single_node_no_tp(self):
        topo = TopologyInfo(num_nodes=1, gpus_per_node=1, has_nvlink=False, total_gpus=1)
        planner = HybridParallelPlanner(topo)
        plan = planner.plan(total_layers=16)
        assert plan.strategy in (ParallelStrategy.PP,)


class MockTpHandle:
    def __init__(self, world_size, multiplier=1.0):
        self.world_size = world_size
        self.multiplier = multiplier
        self.ports = [8000 + i for i in range(world_size)]
        self.process_context = None


class TestTpPpExecutor:
    def test_execute_tp_pp_calls_tp_then_pp(self):
        plan = ParallelPlan(
            strategy=ParallelStrategy.TP_PP,
            tp_world_size=2,
            pp_num_stages=2,
        )
        pipeline = FakePipeline()
        coordinator = type("Coordinator", (), {"_pipeline": pipeline, "_moe_orchestrator": None})()

        executor = HybridParallelExecutor(plan, coordinator)
        inp = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

        # No real TP workers — only PP path is testable without GPUs
        logits = executor.execute(inp, node_kv_caches={})
        assert torch.allclose(logits, inp * 2.0)

    def test_execute_tp_pp_with_fused(self):
        plan = ParallelPlan(
            strategy=ParallelStrategy.TP_PP,
            tp_world_size=1,
            pp_num_stages=2,
        )
        pipeline = FakePipeline()
        coordinator = type("Coordinator", (), {"_pipeline": pipeline, "_moe_orchestrator": None})()

        executor = HybridParallelExecutor(plan, coordinator)
        inp = torch.tensor([[5.0, 6.0, 7.0, 8.0]])

        executor._tp_processes = []  # no real TP workers
        result = executor._tp_pp_fused_forward(inp, node_kv_caches={})
        assert torch.allclose(result, inp * 2.0)

    def test_execute_tp_pp_with_overlap_enabled(self):
        plan = ParallelPlan(
            strategy=ParallelStrategy.TP_PP,
            tp_world_size=1,
            pp_num_stages=2,
        )
        pipeline = FakePipeline()
        pipeline.enable_overlap = True
        coordinator = type("Coordinator", (), {"_pipeline": pipeline, "_moe_orchestrator": None})()

        executor = HybridParallelExecutor(plan, coordinator)
        inp = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

        executor._tp_processes = []
        logits = executor.execute(inp, node_kv_caches={})
        assert torch.allclose(logits, inp * 2.0)
        assert len(pipeline._call_log) == 1
        assert pipeline._call_log[0][0] == "run_pipeline_overlap"

    def test_execute_tp_only(self):
        plan = ParallelPlan(strategy=ParallelStrategy.TP, tp_world_size=1)
        coordinator = type("Coordinator", (), {"_pipeline": None, "_moe_orchestrator": None})()
        executor = HybridParallelExecutor(plan, coordinator)
        inp = torch.tensor([[1.0, 2.0]])
        result = executor.execute(inp, node_kv_caches={})
        assert torch.allclose(result, inp)

    def test_execute_pp_only(self):
        plan = ParallelPlan(strategy=ParallelStrategy.PP, pp_num_stages=2)
        pipeline = FakePipeline()
        coordinator = type("Coordinator", (), {"_pipeline": pipeline, "_moe_orchestrator": None})()
        executor = HybridParallelExecutor(plan, coordinator)
        inp = torch.tensor([[1.0, 2.0]])
        result = executor.execute(inp, node_kv_caches={})
        assert torch.allclose(result, inp * 2.0)

    def test_execute_tp_pp_ep(self):
        plan = ParallelPlan(
            strategy=ParallelStrategy.TP_PP_EP,
            tp_world_size=2,
            pp_num_stages=2,
        )
        pipeline = FakePipeline()
        moe = FakeMoE()
        coordinator = type("Coordinator", (), {"_pipeline": pipeline, "_moe_orchestrator": moe})()
        executor = HybridParallelExecutor(plan, coordinator)
        inp = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        executor._tp_processes = []
        result = executor.execute(inp, node_kv_caches={})
        assert torch.allclose(result, inp * 2.0)

    def test_execute_no_pipeline_raises(self):
        plan = ParallelPlan(strategy=ParallelStrategy.PP, pp_num_stages=2)
        coordinator = type("Coordinator", (), {"_pipeline": None, "_moe_orchestrator": None})()
        executor = HybridParallelExecutor(plan, coordinator)
        with pytest.raises(RuntimeError, match="No pipeline"):
            executor.execute(torch.tensor([[1.0]]), node_kv_caches={})

    def test_configure_pp_sets_stages(self):
        plan = ParallelPlan(strategy=ParallelStrategy.TP_PP, pp_num_stages=3)
        pipeline = FakePipeline()
        coordinator = type("Coordinator", (), {"_pipeline": pipeline, "_moe_orchestrator": None})()
        executor = HybridParallelExecutor(plan, coordinator)
        executor.configure_pp(pipeline)
        assert executor._plan.pp_num_stages == 3

    def test_configure_pp_with_group_nodes(self):
        plan = ParallelPlan(strategy=ParallelStrategy.TP_PP, pp_num_stages=2)
        pipeline = FakePipeline()
        pipeline.group_nodes_into_stages = lambda n: None
        executor = type("Coordinator", (), {"_pipeline": pipeline, "_moe_orchestrator": None})()
        executor = HybridParallelExecutor(plan, executor)
        executor.configure_pp(pipeline)

    def test_tp_pp_non_nvlink_fallback(self):
        topo = TopologyInfo(num_nodes=2, gpus_per_node=2, has_nvlink=False, total_gpus=4)
        planner = HybridParallelPlanner(topo)
        plan = planner.plan(total_layers=24, tp_enabled=True)
        assert plan.tp_world_size == 1

    def test_tp_pp_two_nodes_with_nvlink(self):
        topo = TopologyInfo(num_nodes=2, gpus_per_node=4, has_nvlink=True, total_gpus=8)
        planner = HybridParallelPlanner(topo)
        plan = planner.plan(total_layers=40)
        assert plan.strategy == ParallelStrategy.TP_PP
        assert plan.tp_world_size == 4
        assert plan.pp_num_stages == 2

    def test_shutdown_clears_processes(self):
        plan = ParallelPlan(strategy=ParallelStrategy.TP, tp_world_size=2)
        executor = HybridParallelExecutor(plan, coordinator=None)
        executor._tp_processes = []
        executor.shutdown()
        assert executor._tp_processes == []

    def test_execute_with_draft_tokens(self):
        plan = ParallelPlan(strategy=ParallelStrategy.PP, pp_num_stages=2)
        pipeline = FakePipeline()
        coordinator = type("Coordinator", (), {"_pipeline": pipeline, "_moe_orchestrator": None})()
        executor = HybridParallelExecutor(plan, coordinator)
        inp = torch.tensor([[1.0, 2.0]])
        result = executor.execute(inp, node_kv_caches={}, draft_tokens=[3])
        assert torch.allclose(result, inp * 2.0)

    def test_execute_ep_only(self):
        plan = ParallelPlan(strategy=ParallelStrategy.EP)
        pipeline = FakePipeline()
        moe = FakeMoE()
        coordinator = type("Coordinator", (), {"_pipeline": pipeline, "_moe_orchestrator": moe})()
        executor = HybridParallelExecutor(plan, coordinator)
        inp = torch.tensor([[1.0, 2.0]])
        result = executor.execute(inp, node_kv_caches={})
        assert torch.allclose(result, inp * 2.0)

    def test_execute_pp_ep(self):
        plan = ParallelPlan(strategy=ParallelStrategy.PP_EP, pp_num_stages=2)
        pipeline = FakePipeline()
        moe = FakeMoE()
        coordinator = type("Coordinator", (), {"_pipeline": pipeline, "_moe_orchestrator": moe})()
        executor = HybridParallelExecutor(plan, coordinator)
        inp = torch.tensor([[1.0, 2.0]])
        result = executor.execute(inp, node_kv_caches={})
        assert torch.allclose(result, inp * 2.0)
