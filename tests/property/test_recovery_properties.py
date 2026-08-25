"""Property-based tests for dist/recovery.py redistribution planning.

Verifies key invariants of the recovery system:
1. A recovery plan never assigns the same layer to two survivors
2. Every layer of the failed node is covered by exactly one survivor
3. Final ranges stay contiguous (start <= end)

NOTE: rewritten against the REAL recovery API.  An earlier version of
this file targeted a fictional API (``register_node`` /
``create_recovery_plan`` / ``LayerRedistribution(layers=[...])``) that
never existed in ``dist/recovery.py``, so every test failed on import
shape alone.
"""

from __future__ import annotations

import random

import pytest

try:
    from hypothesis import given, settings, strategies as st
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False

from distllm.dist.recovery import (
    LayerRedistribution,
    NodeRecoveryManager,
    NodeRecoveryPlan,
)


def _random_clean_tiling(rng: random.Random) -> tuple[dict[str, tuple[int, int]], int]:
    """A contiguous tiling 0..total-1 across 2..6 nodes with random widths."""

    n_nodes = rng.randint(2, 6)
    widths = [rng.randint(1, 8) for _ in range(n_nodes)]
    total = sum(widths)
    tiling: dict[str, tuple[int, int]] = {}
    cursor = 0
    for i, w in enumerate(widths):
        tiling[f"node-{i}"] = (cursor, cursor + w - 1)
        cursor += w
    return tiling, total


def _check_disjoint_and_complete(
    final: dict[str, tuple[int, int]],
    total: int,
) -> None:
    owner: dict[int, str] = {}
    for nid, (s, e) in final.items():
        assert s <= e, f"{nid} has inverted range ({s}, {e})"
        for layer in range(s, e + 1):
            assert layer not in owner, (
                f"Layer {layer} assigned to both "
                f"{owner[layer]} and {nid}"
            )
            owner[layer] = nid
    covered = set(owner)
    assert covered == set(range(total)), (
        f"coverage mismatch: missing {sorted(set(range(total)) - covered)[:8]}, "
        f"extra {sorted(covered - set(range(total)))[:8]}"
    )


@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="Hypothesis not installed")
class TestRedistributionProperties:
    """Property-based invariants for the redistribution planners."""

    @settings(max_examples=50, deadline=None)
    @given(seed=st.integers(0, 2**32 - 1))
    def test_no_layer_assigned_twice(self, seed):
        """A recovery plan must never assign the same layer to multiple survivors."""
        rng = random.Random(seed)
        tiling, total = _random_clean_tiling(rng)
        items = list(tiling.items())
        failed_nid, (fs, fe) = items[rng.randrange(len(items))]
        survivors = {nid: r for nid, r in tiling.items() if nid != failed_nid}

        from distllm.dist.recovery import compute_redistributions
        rds = compute_redistributions(fs, fe, survivors)

        # Apply like core/coordinator.py:_on_node_redistribute does,
        # with the dead node removed first (mark-dead fires before
        # redistribute in _on_node_failure_impl).
        final = {k: v for k, v in survivors.items()}
        for rd in rds:
            final[rd.surviving_node_id] = (rd.new_start_layer, rd.new_end_layer)
        _check_disjoint_and_complete(final, total)

    @settings(max_examples=50, deadline=None)
    @given(seed=st.integers(0, 2**32 - 1))
    def test_failed_layers_covered(self, seed):
        """Every layer of the failed node must land on exactly one survivor."""

        rng = random.Random(seed)
        tiling, _total = _random_clean_tiling(rng)
        items = list(tiling.items())
        failed_nid, (fs, fe) = items[rng.randrange(len(items))]
        survivors = {nid: r for nid, r in tiling.items() if nid != failed_nid}

        from distllm.dist.recovery import compute_redistributions
        rds = compute_redistributions(fs, fe, survivors)

        covered: dict[int, str] = {}
        for rd in rds:
            for layer in range(rd.added_start_layer, rd.added_end_layer + 1):
                assert layer not in covered, (
                    f"orphan layer {layer} added to both "
                    f"{covered[layer]} and {rd.surviving_node_id}"
                )
                covered[layer] = rd.surviving_node_id
        assert set(covered) == set(range(fs, fe + 1))

    def test_capacity_aware_invariants_seeded(self):
        """Capacity-aware planner keeps disjoint+complete coverage across
        seeded randomized eligible/ineligible memory layouts."""

        try:
            from distllm.dist.recovery import compute_redistributions_capacity_aware
        except ImportError:  # pragma: no cover
            pytest.skip("capacity-aware planner unavailable")

        rng = random.Random(20260824)
        for trial in range(100):
            tiling, total = _random_clean_tiling(rng)
            items = list(tiling.items())
            rng.shuffle(items)
            failed_nid, (fs, fe) = items[rng.randrange(len(items))]
            survivors = dict(items)
            del survivors[failed_nid]
            memory = {
                nid: rng.choice([0.0, 0.4, 2.0, 8.0, 50.0])
                for nid in survivors
            }
            rds = compute_redistributions_capacity_aware(
                fs, fe, dict(survivors),
                survivor_memory_gb=memory,
                min_memory_per_layer_gb=1.0,
            )
            final = {k: v for k, v in survivors.items()}
            for rd in rds:
                final[rd.surviving_node_id] = (rd.new_start_layer, rd.new_end_layer)
            try:
                _check_disjoint_and_complete(final, total)
            except AssertionError as e:
                raise AssertionError(f"trial {trial}: {e}") from e


class TestPlanDataclassContract:
    """The plan/redistribution dataclasses keep their documented shape."""

    def test_plan_equals_self(self):
        plan = NodeRecoveryPlan(
            failed_node_id="gpu-0",
            redistributions=[
                LayerRedistribution(
                    surviving_node_id="gpu-1",
                    added_start_layer=0,
                    added_end_layer=2,
                    new_start_layer=0,
                    new_end_layer=5,
                ),
            ],
        )
        assert plan == plan

    def test_plan_equality_symmetric(self):
        def make() -> NodeRecoveryPlan:
            return NodeRecoveryPlan(
                failed_node_id="gpu-0",
                redistributions=[
                    LayerRedistribution(
                        surviving_node_id="gpu-1",
                        added_start_layer=0,
                        added_end_layer=1,
                        new_start_layer=0,
                        new_end_layer=3,
                    ),
                ],
            )

        assert make() == make()

    def test_weights_transferred_defaults_false(self):
        """Metadata-only honesty: a computed plan never claims weight transfer."""

        plan = NodeRecoveryPlan(failed_node_id="n0")
        assert plan.weights_transferred is False
        rd = LayerRedistribution("n1", 4, 7, 0, 7)
        assert rd.requires_weight_load is True
