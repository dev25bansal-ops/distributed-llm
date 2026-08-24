"""Regression tests for the MoE partition-planning ImportError (audit C9).

Root cause: ``distllm.dist.parallel`` and ``distllm.dist.parallel_planner``
both import ``replicate_experts_across_nodes`` from
``distllm.core.moe_orchestrator`` inside their planners'
``_assign_experts`` -- but that symbol never existed, so any plan for an
MoE model crashed with ModuleNotFoundError.

The fix implements the function in ``moe_orchestrator.py`` with balanced
round-robin semantics.  These tests cover the function itself plus both
real call sites (each planner module has its own ``HybridParallelPlanner``)
and an end-to-end MoE ``plan()``.
"""

from __future__ import annotations

import pytest


class TestReplicateExpertsAcrossNodes:
    """Unit coverage of the newly implemented placement helper."""

    def test_import_succeeds(self):
        from distllm.core.moe_orchestrator import (
            replicate_experts_across_nodes,
        )

        assert callable(replicate_experts_across_nodes)

    def test_round_robin_single_replica_covers_all_experts_once(self):
        from distllm.core.moe_orchestrator import (
            replicate_experts_across_nodes,
        )

        assignment = replicate_experts_across_nodes(8, 2, replication_factor=1)
        assert set(assignment) == {"node_0", "node_1"}
        placed = [e for ids in assignment.values() for e in ids]
        assert sorted(placed) == list(range(8))
        assert len(placed) == 8  # no duplicates at factor 1

    def test_single_replica_is_balanced_within_one(self):
        from distllm.core.moe_orchestrator import (
            replicate_experts_across_nodes,
        )

        assignment = replicate_experts_across_nodes(10, 3, replication_factor=1)
        counts = {k: len(v) for k, v in assignment.items()}
        assert max(counts.values()) - min(counts.values()) <= 1
        assert sum(counts.values()) == 10

    def test_round_robin_first_batch_pattern(self):
        """factor=1 on equal counts is textbook round-robin."""
        from distllm.core.moe_orchestrator import (
            replicate_experts_across_nodes,
        )

        assignment = replicate_experts_across_nodes(4, 4, replication_factor=1)
        assert assignment == {
            "node_0": [0],
            "node_1": [1],
            "node_2": [2],
            "node_3": [3],
        }

    def test_replication_factor_places_each_expert_on_n_nodes(self):
        from distllm.core.moe_orchestrator import (
            replicate_experts_across_nodes,
        )

        assignment = replicate_experts_across_nodes(8, 4, replication_factor=2)
        assert set(assignment) == {f"node_{i}" for i in range(4)}
        for expert_id in range(8):
            hosts = [n for n, ids in assignment.items() if expert_id in ids]
            assert len(hosts) == 2  # exactly two distinct replicas
        # Balanced overall: 16 placements over 4 nodes -> 4 each.
        assert all(len(ids) == 4 for ids in assignment.values())

    def test_replication_factor_clamped_to_node_count(self):
        from distllm.core.moe_orchestrator import (
            replicate_experts_across_nodes,
        )

        assignment = replicate_experts_across_nodes(4, 2, replication_factor=99)
        for expert_id in range(4):
            hosts = [n for n, ids in assignment.items() if expert_id in ids]
            assert len(hosts) == 2  # cannot exceed available nodes

    @pytest.mark.parametrize("num_experts,nodes", [(0, 3), (5, 0), (-1, 2)])
    def test_degenerate_inputs_return_empty_map(self, num_experts, nodes):
        from distllm.core.moe_orchestrator import (
            replicate_experts_across_nodes,
        )

        assert replicate_experts_across_nodes(num_experts, nodes) == {}


class TestPlannerCallSites:
    """Both planner modules must reach _assign_experts without ImportError."""

    def test_parallel_planner_assign_experts(self):
        from distllm.dist.parallel_planner import HybridParallelPlanner

        result = HybridParallelPlanner()._assign_experts(8, 3)
        assert set(result) == {"node_0", "node_1", "node_2"}
        placed = sorted(e for ids in result.values() for e in ids)
        assert placed == list(range(8))

    def test_parallel_module_assign_experts(self):
        from distllm.dist.parallel import HybridParallelPlanner

        result = HybridParallelPlanner()._assign_experts(8, 3)
        assert set(result) == {"node_0", "node_1", "node_2"}
        placed = sorted(e for ids in result.values() for e in ids)
        assert placed == list(range(8))

    @pytest.mark.parametrize(
        "module",
        ["distllm.dist.parallel", "distllm.dist.parallel_planner"],
    )
    def test_moe_partition_plan_end_to_end(self, module):
        """Full MoE plan() -- the exact repro path (used to raise)."""
        import importlib

        mod = importlib.import_module(module)
        topology = mod.TopologyInfo(
            num_nodes=2, gpus_per_node=1, total_gpus=2,
        )
        planner = mod.HybridParallelPlanner(topology)
        plan = planner.plan(total_layers=32, num_experts=8, use_moe=True)

        assert plan.strategy == mod.ParallelStrategy.PP_EP
        assert plan.expert_assignment, "expert_assignment must be populated"
        placed = sorted(e for ids in plan.expert_assignment.values() for e in ids)
        assert placed == list(range(8))
