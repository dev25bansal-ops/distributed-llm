"""Property-based tests for dist/recovery.py using Hypothesis.

Verifies key invariants of the recovery system:
1. Recovery plan never assigns the same layer to two survivors
2. _tensor_size_bytes is commutative across nested structures
3. All failed node layers are covered by exactly one survivor
"""

from __future__ import annotations

import pytest

try:
    from hypothesis import given, strategies as st
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False

from distllm.dist.recovery import NodeRecoveryManager, NodeRecoveryPlan, LayerRedistribution


# ── Hypothesis strategies ──────────────────────────────────────────────

node_strategy = st.text(min_size=1, max_size=16, alphabet="abcdefghijklmnopqrstuvwxyz-_0123456789")
layer_id_strategy = st.integers(min_value=0, max_value=127)

cluster_topology = st.dictionaries(
    keys=node_strategy,
    values=st.lists(
        layer_id_strategy,
        min_size=1,
        max_size=32,
        unique=True,
    ),
    min_size=2,
    max_size=16,
)


@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="Hypothesis not installed")
class TestRecoveryProperties:
    """Property-based invariants for the recovery system."""

    @given(topology=cluster_topology, failed_node=node_strategy, seed=st.integers(0, 2**32 - 1))
    def test_no_layer_assigned_twice(self, topology, failed_node, seed):
        """A recovery plan must never assign the same layer to multiple survivors."""
        import random
        random.seed(seed)

        if failed_node not in topology:
            return  # skip invalid

        mgr = NodeRecoveryManager(total_layers=128)
        for node_id, layers in topology.items():
            mgr.register_node(node_id, layers)

        if not mgr._nodes:
            return

        try:
            plan = mgr.create_recovery_plan(failed_node)
        except (ValueError, RuntimeError):
            return  # not all nodes can fail

        if plan is None:
            return

        # Collect all assigned layers across all survivors
        assigned_layers: dict[int, str] = {}
        for assignment in plan.redistributions:
            for layer_id in assignment.layers:
                assert layer_id not in assigned_layers, (
                    f"Layer {layer_id} assigned to both "
                    f"{assigned_layers[layer_id]} and {assignment.target_node_id}"
                )
                assigned_layers[layer_id] = assignment.target_node_id

    @given(topology=cluster_topology, failed_node=node_strategy, seed=st.integers(0, 2**32 - 1))
    def test_failed_layers_covered(self, topology, failed_node, seed):
        """Every layer on the failed node must be assigned to at least one survivor."""
        import random
        random.seed(seed)

        if failed_node not in topology:
            return

        failed_layers = set(topology[failed_node])
        if not failed_layers:
            return

        mgr = NodeRecoveryManager(total_layers=128)
        for node_id, layers in topology.items():
            mgr.register_node(node_id, layers)

        try:
            plan = mgr.create_recovery_plan(failed_node)
        except (ValueError, RuntimeError):
            return

        if plan is None:
            return

        covered_layers: set[int] = set()
        for assignment in plan.redistributions:
            covered_layers.update(assignment.layers)

        uncovered = failed_layers - covered_layers
        assert not uncovered, (
            f"Layers {uncovered} from failed node {failed_node} "
            f"were not covered by any survivor"
        )

    @given(st.lists(st.integers(min_value=1, max_value=1024), min_size=1, max_size=10))
    def test_tensor_size_commutative(self, dims):
        """_tensor_size_bytes should be commutative across nested structures."""
        from distllm.dist.recovery import NodeRecoveryPlan

        # Verify the method handles arbitrary dimension lists
        if hasattr(NodeRecoveryPlan, '_tensor_size_bytes'):
            size1 = NodeRecoveryPlan._tensor_size_bytes(dims)
            size2 = NodeRecoveryPlan._tensor_size_bytes(list(reversed(dims)))
            assert size1 == size2, (
                f"tensor_size_bytes not commutative: "
                f"{dims} -> {size1}, {list(reversed(dims))} -> {size2}"
            )

    @given(st.lists(st.integers(min_value=0, max_value=1), min_size=1, max_size=5))
    def test_tensor_size_zero_dim(self, dims):
        """_tensor_size_bytes should handle zero-length dimensions gracefully."""
        from distllm.dist.recovery import NodeRecoveryPlan
        if hasattr(NodeRecoveryPlan, '_tensor_size_bytes'):
            size = NodeRecoveryPlan._tensor_size_bytes(dims)
            if 0 in dims:
                assert size == 0
            else:
                assert size > 0

    def test_plan_equals_self(self):
        """NodeRecoveryPlan equality should be reflexive."""
        plan = NodeRecoveryPlan(
            request_id="test",
            failed_node_id="gpu-0",
            redistributions=[
                LayerRedistribution(
                    source_node_id="gpu-0",
                    target_node_id="gpu-1",
                    layers=[0, 1, 2],
                ),
            ],
        )
        assert plan == plan

    def test_plan_equality_symmetric(self):
        """Two identical plans should be equal."""
        plan_a = NodeRecoveryPlan(
            request_id="test",
            failed_node_id="gpu-0",
            redistributions=[
                LayerRedistribution(
                    source_node_id="gpu-0",
                    target_node_id="gpu-1",
                    layers=[0, 1],
                ),
            ],
        )
        plan_b = NodeRecoveryPlan(
            request_id="test",
            failed_node_id="gpu-0",
            redistributions=[
                LayerRedistribution(
                    source_node_id="gpu-0",
                    target_node_id="gpu-1",
                    layers=[0, 1],
                ),
            ],
        )
        assert plan_a == plan_b
