"""Regression + invariant tests for node-failure layer redistribution (Area-2 C3).

Pre-fix, ``compute_redistributions`` / ``compute_redistributions_capacity_aware``
merged survivor ranges with ``min()``/``max()``, producing OVERLAPPING layer
ranges whenever the failed node was not interior (two nodes claiming the same
layer indices), and ``NodeRecoveryManager._redistribute_parallel`` fired the
coordinator callback once per redistribution with the FULL plan (N x N
duplicate application). Application was metadata-only with no record that
survivors must reload weights for their added layers.

These tests pin the corrected behavior:

1. Repro: 4 nodes x 8 layers, kill one -> no overlap, full coverage.
2. Analysis repro: survivors {A:(0,3), B:(4,7), C:(8,11)}, failed (12,15)
   -> A claimed 0-13 and B claimed 4-15 (overlap on 4-13) pre-fix.
3. Seeded randomized clean tilings -> disjointness + contiguity + coverage
   for both planners (basic and capacity-aware).
4. Manager level: per-survivor callback dispatch (single-redistribution
   plans), metadata-only honesty (``requires_weight_load`` flag,
   ``nodes_needing_weight_reload`` tracking), dry-run planning.
5. ``Rebalancer.compute_new_partition`` never emits inverted or overlapping
   ranges, even for degenerate inputs (more nodes than layers).
"""

from __future__ import annotations

import random
from collections import Counter

import pytest

from distllm.dist.rebalancer import PartitionRecommendation, Rebalancer
from distllm.dist.recovery import (
    LayerRedistribution,
    NodeRecoveryManager,
    NodeRecoveryPlan,
    compute_redistributions,
    compute_redistributions_capacity_aware,
)


# ── Helpers ────────────────────────────────────────────────────────────


def apply_redistributions(
    base: dict[str, tuple[int, int]],
    rds: list[LayerRedistribution],
) -> dict[str, tuple[int, int]]:
    """Simulate the coordinator's label application (coordinator.py:394-404)."""

    final = {nid: (s, e) for nid, (s, e) in base.items()}
    for rd in rds:
        assert rd.surviving_node_id in final, (
            f"redistribution targets unknown survivor {rd.surviving_node_id}"
        )
        final[rd.surviving_node_id] = (rd.new_start_layer, rd.new_end_layer)
    return final


def assert_disjoint_and_contiguous(final: dict[str, tuple[int, int]]) -> None:
    """Every final range is well-formed and no layer is claimed twice."""

    owner: dict[int, str] = {}
    for nid, (s, e) in sorted(final.items()):
        assert s <= e, f"{nid} has inverted range ({s}, {e})"
        for layer in range(s, e + 1):
            assert layer not in owner, (
                f"layer {layer} claimed by BOTH {owner[layer]} and {nid} "
                f"(final={sorted(final.items())})"
            )
            owner[layer] = nid


def assert_coverage(
    final: dict[str, tuple[int, int]],
    expected_layers: set[int],
) -> None:
    covered: set[int] = set()
    for s, e in final.values():
        covered.update(range(s, e + 1))
    missing = expected_layers - covered
    assert not missing, (
        f"layers {sorted(missing)[:10]}... not covered by any survivor "
        f"(final={sorted(final.items())})"
    )


def clean_tiling(
    rng: random.Random,
) -> tuple[dict[str, tuple[int, int]], int]:
    """Build a clean contiguous tiling 0..total-1 across 2-6 nodes."""

    n_nodes = rng.randint(2, 6)
    widths = [rng.randint(1, 6) for _ in range(n_nodes)]
    total = sum(widths)
    tiling: dict[str, tuple[int, int]] = {}
    cursor = 0
    for i, w in enumerate(widths):
        tiling[f"node-{i}"] = (cursor, cursor + w - 1)
        cursor += w
    return tiling, total


# ── 1. Direct repros (fail pre-fix) ───────────────────────────────────


class TestOverlapRepros:
    def test_analysis_repro_tail_failure(self):
        """Survivors {A:(0,3), B:(4,7), C:(8,11)}, failed (12,15).

        Pre-fix: A became (0,13) and B became (4,15) -> layers 4-13
        claimed by both A and B.
        """
        survivors = {"A": (0, 3), "B": (4, 7), "C": (8, 11)}
        rds = compute_redistributions(12, 15, dict(survivors))
        final = apply_redistributions(survivors, rds)
        assert_disjoint_and_contiguous(final)
        assert_coverage(final, set(range(0, 16)))

    def test_four_nodes_eight_layers_kill_second(self):
        """4 nodes x 8 layers; kill node holding layers 8-15.

        Pre-fix: n2 became (11,23) and n3 became (14,31) -> overlap 14-23.
        """
        survivors = {
            "n0": (0, 7),
            "n2": (16, 23),
            "n3": (24, 31),
        }
        rds = compute_redistributions(8, 15, survivors)
        assert rds, "expected redistributions for a 3-survivor cluster"
        final = apply_redistributions(survivors, rds)
        assert_disjoint_and_contiguous(final)
        assert_coverage(final, set(range(0, 32)))

    def test_interior_failure_between_two_of_three(self):
        """Failed (8,11) strictly between n0(0,7) and n2(16,23); n3 far right.

        Only the two adjacent survivors may change; n3 must be untouched.
        """
        survivors = {"n0": (0, 7), "n2": (16, 23), "n3": (24, 31)}
        rds = compute_redistributions(8, 15, survivors)
        final = apply_redistributions(survivors, rds)
        assert final["n3"] == (24, 31), "non-adjacent survivor was mutated"
        assert_disjoint_and_contiguous(final)
        assert_coverage(final, set(range(0, 32)))

    def test_head_failure_left_edge(self):
        survivors = {"n1": (8, 15), "n2": (16, 23)}
        rds = compute_redistributions(0, 7, survivors)
        final = apply_redistributions(survivors, rds)
        assert_disjoint_and_contiguous(final)
        assert_coverage(final, set(range(0, 24)))


# ── 2. Seeded randomized invariants ───────────────────────────────────


class TestRandomizedPlannerInvariants:
    @pytest.mark.parametrize("seed", range(40))
    def test_basic_planner_clean_tilings(self, seed):
        rng = random.Random(seed)
        tiling, total = clean_tiling(rng)

        # Shuffle insertion order to catch sort-key bugs.
        items = list(tiling.items())
        rng.shuffle(items)
        survivors = dict(items)

        kill_idx = rng.randrange(len(items))
        failed_nid, (fs, fe) = items[kill_idx]
        del survivors[failed_nid]

        rds = compute_redistributions(fs, fe, survivors)
        final = apply_redistributions(tiling, rds)
        del final[failed_nid]

        assert_disjoint_and_contiguous(final)
        assert_coverage(final, set(range(total)))

    @pytest.mark.parametrize("seed", range(40))
    def test_capacity_planner_clean_tilings(self, seed):
        rng = random.Random(seed)
        tiling, total = clean_tiling(rng)

        items = list(tiling.items())
        rng.shuffle(items)
        survivors = dict(items)

        kill_idx = rng.randrange(len(items))
        failed_nid, (fs, fe) = items[kill_idx]
        del survivors[failed_nid]

        # Random memory: some survivors eligible, some not.
        memory = {nid: rng.choice([0.0, 0.4, 2.0, 8.0, 50.0]) for nid in survivors}
        rds = compute_redistributions_capacity_aware(
            fs, fe, survivors, survivor_memory_gb=memory,
            min_memory_per_layer_gb=1.0,
        )
        final = apply_redistributions(tiling, rds)
        del final[failed_nid]

        assert_disjoint_and_contiguous(final)
        assert_coverage(final, set(range(total)))

    @pytest.mark.parametrize("seed", range(20))
    def test_every_orphan_layer_assigned_exactly_once(self, seed):
        """Each failed-node layer appears in exactly one added_* range."""

        rng = random.Random(1000 + seed)
        tiling, _total = clean_tiling(rng)
        items = list(tiling.items())
        failed_nid, (fs, fe) = items[rng.randrange(len(items))]
        survivors = {nid: r for nid, r in tiling.items() if nid != failed_nid}

        rds = compute_redistributions(fs, fe, survivors)
        claimed: dict[int, str] = {}
        for rd in rds:
            for layer in range(rd.added_start_layer, rd.added_end_layer + 1):
                assert layer not in claimed, (
                    f"orphan layer {layer} added to both "
                    f"{claimed[layer]} and {rd.surviving_node_id}"
                )
                claimed[layer] = rd.surviving_node_id
        assert set(claimed) == set(range(fs, fe + 1)), (
            "orphan layers were not fully reassigned"
        )

    def test_degenerate_inputs(self):
        assert compute_redistributions(5, 3, {"n": (0, 4)}) == []
        assert compute_redistributions(0, 3, {}) == []


# ── 3. Gap topology (no adjacent survivor on one side) ────────────────


class TestGapTopology:
    def test_orphan_covered_when_only_distant_survivor(self):
        """Failed (10,13) with a single survivor (0,3): survivor absorbs the
        orphan; output must stay disjoint/contiguous and cover the orphan."""
        survivors = {"A": (0, 3)}
        rds = compute_redistributions(10, 13, survivors)
        final = apply_redistributions(survivors, rds)
        assert_disjoint_and_contiguous(final)
        assert_coverage(final, set(range(10, 14)))

    def test_intersecting_input_still_produces_clean_output(self):
        """Garbage input (a survivor already claims part of the failed range)
        must still yield a disjoint, contiguous, covering partition."""
        survivors = {"A": (0, 5), "B": (6, 9)}  # A intersects failed (4, 9)
        rds = compute_redistributions(4, 9, survivors)
        final = apply_redistributions(survivors, rds)
        assert_disjoint_and_contiguous(final)
        assert_coverage(final, set(range(0, 10)))

    def test_capacity_intersecting_input_clean_output(self):
        survivors = {"A": (0, 3), "B": (4, 7), "C": (8, 11)}
        rds = compute_redistributions_capacity_aware(
            0, 15, survivors,
            survivor_memory_gb={"A": 10.0, "B": 10.0, "C": 10.0},
        )
        final = apply_redistributions(survivors, rds)
        assert_disjoint_and_contiguous(final)


# ── 4. Manager-level dispatch + weight-reload honesty ─────────────────


def _four_by_eight_mgr():
    mgr = NodeRecoveryManager(node_id="coord")
    assignments = {
        "n0": (0, 7),
        "n1": (8, 15),
        "n2": (16, 23),
        "n3": (24, 31),
    }
    mgr.set_layer_assignments(assignments)
    return mgr, assignments


class TestManagerRedistributeDispatch:
    def test_callback_fires_once_per_survivor_with_single_rd_plan(self):
        """C3: the callback must receive ONE redistribution per call, and each
        survivor must be dispatched exactly once (pre-fix: every call carried
        the FULL plan, so the coordinator applied all redistributions N times).
        """
        mgr, assignments = _four_by_eight_mgr()
        calls: list[tuple[str, NodeRecoveryPlan]] = []

        def on_redistribute(failed: str, plan: NodeRecoveryPlan) -> None:
            calls.append((failed, plan))

        mgr.set_redistribute_layers_callback(on_redistribute)
        mgr.on_node_failure("n1")

        assert calls, "redistribute callback never fired"
        for failed, plan in calls:
            assert failed == "n1"
            assert len(plan.redistributions) == 1, (
                f"callback received a plan with {len(plan.redistributions)} "
                f"redistributions; expected exactly 1 (per-survivor dispatch)"
            )

        dispatched = Counter(
            plan.redistributions[0].surviving_node_id for _, plan in calls
        )
        # Neighbor absorption: killing n1 (layers 8-15) touches only its
        # adjacent neighbors n0 and n2.  Non-adjacent n3 must NOT receive
        # a dispatch (pre-fix every survivor got the full plan N times).
        expected = Counter({"n0": 1, "n2": 1})
        assert dispatched == expected, (
            f"dispatch counts wrong: {dict(dispatched)} vs {dict(expected)}"
        )

    def test_applied_labels_are_disjoint_and_complete(self):
        mgr, assignments = _four_by_eight_mgr()

        def on_redistribute(failed: str, plan: NodeRecoveryPlan) -> None:
            # Mimic core/coordinator.py:_on_node_redistribute exactly. The
            # real flow removes the dead node via mark-dead BEFORE the
            # redistribute callback fires (recovery Step 4 < Step 5), so
            # the dead node's label is gone from the live topology.
            assignments.pop(failed, None)
            for rd in plan.redistributions:
                assignments[rd.surviving_node_id] = (
                    rd.new_start_layer, rd.new_end_layer,
                )

        mgr.set_redistribute_layers_callback(on_redistribute)
        mgr.on_node_failure("n1")

        final = {k: v for k, v in assignments.items()}
        assert_disjoint_and_contiguous(final)
        assert_coverage(final, set(range(32)))

    def test_metadata_only_honesty_flags(self):
        """Planner output must admit it is metadata-only."""
        rds = compute_redistributions(8, 15, {"n0": (0, 7), "n2": (16, 23)})
        assert rds
        for rd in rds:
            assert rd.requires_weight_load is True

        plan = NodeRecoveryPlan(failed_node_id="n1")
        assert plan.weights_transferred is False


class TestWeightReloadTracking:
    def test_reassigned_survivors_flagged_for_weight_reload(self):
        mgr, _ = _four_by_eight_mgr()
        mgr.set_redistribute_layers_callback(lambda f, p: None)
        mgr.on_node_failure("n1")

        # Only the adjacent survivors that actually absorbed layers are
        # flagged; n3 was untouched by neighbor absorption.
        pending = set(mgr.nodes_needing_weight_reload())
        assert pending == {"n0", "n2"}, (
            f"expected adjacent absorbers flagged, got {pending}"
        )
        assert mgr.get_metrics()["pending_weight_reloads"] == 2

    def test_mark_weights_loaded_clears_flag(self):
        mgr, _ = _four_by_eight_mgr()
        mgr.set_redistribute_layers_callback(lambda f, p: None)
        mgr.on_node_failure("n0")

        assert set(mgr.nodes_needing_weight_reload()) == {"n1"}
        mgr.mark_weights_loaded("n1")
        mgr.mark_weights_loaded("n1")  # idempotent
        assert mgr.nodes_needing_weight_reload() == []

    def test_dry_run_plans_but_does_not_dispatch_or_flag(self):
        mgr, _ = _four_by_eight_mgr()
        calls: list = []
        mgr.set_redistribute_layers_callback(lambda f, p: calls.append(p))

        plan = mgr.dry_run_recovery("n1")
        # Drills need real redistribution counts (recovery_drill.py SLAs).
        assert plan.redistributions, "dry-run plan must contain redistributions"
        assert calls == [], "dry-run fired the redistribute callback"
        assert mgr.nodes_needing_weight_reload() == []


# ── 5. Rebalancer.compute_new_partition sanity ────────────────────────


class _StubTracker:
    def __init__(self, avgs):
        self._avgs = avgs

    def get_all_avg(self):
        return dict(self._avgs)


class _StubSettings:
    enabled = True
    straggler_threshold = 2.0
    grace_period_steps = 1
    auto_mitigate = False
    cooldown_seconds = 0
    min_improvement_pct = 0


def _make_rebalancer(latencies):
    from distllm.dist.latency import LatencyTracker

    tracker = LatencyTracker.__new__(LatencyTracker)  # bypass init
    tracker.get_all_avg = lambda: dict(latencies)
    return Rebalancer(tracker, _StubSettings())


class TestRebalancerNewPartition:
    def test_equal_latencies_full_coverage(self):
        rb = _make_rebalancer({"a": 1.0, "b": 1.0, "c": 1.0, "d": 1.0})
        recs = rb.compute_new_partition(32, {"a": 1.0, "b": 1.0, "c": 1.0, "d": 1.0})
        assert recs
        assert_disjoint_and_contiguous(
            {r.node_id: (r.start_layer, r.end_layer) for r in recs}
        )
        assert_coverage({r.node_id: (r.start_layer, r.end_layer) for r in recs},
                        set(range(32)))

    def test_faster_node_gets_more_layers(self):
        rb = _make_rebalancer({"fast": 1.0, "slow": 10.0})
        recs = rb.compute_new_partition(16, {"fast": 1.0, "slow": 10.0})
        widths = {r.node_id: r.end_layer - r.start_layer + 1 for r in recs}
        assert widths["fast"] > widths["slow"]

    def test_more_nodes_than_layers_no_invalid_ranges(self):
        """Degenerate: 3 nodes, 2 layers -> no inverted ranges, full coverage."""
        rb = _make_rebalancer({"a": 1.0, "b": 1.0, "c": 1.0})
        recs = rb.compute_new_partition(2, {"a": 1.0, "b": 1.0, "c": 1.0})
        ranges = {r.node_id: (r.start_layer, r.end_layer) for r in recs}
        assert_disjoint_and_contiguous(ranges)
        assert_coverage(ranges, set(range(2)))
        assert len(recs) <= 3

    def test_empty_and_zero_inputs(self):
        rb = _make_rebalancer({})
        assert rb.compute_new_partition(0, {"a": 1.0}) == []
        assert rb.compute_new_partition(8, {}) == []

    def test_recommendation_dataclass_intact(self):
        rec = PartitionRecommendation("n", 0, 3)
        assert (rec.node_id, rec.start_layer, rec.end_layer) == ("n", 0, 3)
