"""E6 regression test — WAN/topology + carbon-aware placement, canary/AB, migration.

Pure-logic tests for ``distllm.core.placement`` (and its thin hook in
``BatchScheduler``).  No real network, GPU, or global state is touched.

Asserts:
  (1) Topology-aware ranking prefers the lower-latency node when bandwidth is equal.
  (2) Carbon-aware prefers the lower-carbon node when latency is within threshold.
  (3) Canary routing sends ``canary_weight`` fraction of requests to canary nodes
      deterministically (same seed → same mapping).
  (4) plan_migration returns a valid, ordered plan that reduces cost.
"""

import math

import pytest

from distllm.core.placement import (
    CanaryRouter,
    LinkInfo,
    MigrationStep,
    NodeTopology,
    PlacementPolicy,
    plan_migration,
    route_canary,
    select_placement,
)


# ── Fixtures / helpers ────────────────────────────────────────────────────

def _topo(equal_bw=True, carbon_a=0.0, carbon_b=0.0, latency_a=20.0, latency_b=20.0):
    """Two-node topology; bandwidth equal by default (controls latency test)."""
    bw = 100.0 if equal_bw else 50.0
    return NodeTopology(
        coordinator_region="us",
        links={
            "node-a": LinkInfo(
                node_id="node-a", region="us-east", latency_ms=latency_a,
                bandwidth_gbps=bw, carbon_intensity=carbon_a,
            ),
            "node-b": LinkInfo(
                node_id="node-b", region="eu-west", latency_ms=latency_b,
                bandwidth_gbps=bw, carbon_intensity=carbon_b,
            ),
        },
    )


# ── (1) Topology-aware: lower latency wins when bandwidth equal ──────────

def test_topology_prefers_lower_latency_when_bandwidth_equal():
    topo = _topo(equal_bw=True, latency_a=10.0, latency_b=80.0)
    policy = PlacementPolicy(carbon_aware=False)  # isolate network term
    ranked = select_placement(["node-a", "node-b"], topo, policy)
    assert ranked == ["node-a", "node-b"]
    assert ranked[0] == "node-a"


def test_topology_prefers_higher_bandwidth_when_latency_equal():
    topo = NodeTopology(coordinator_region="us", links={
        "slow": LinkInfo(node_id="slow", region="r1", latency_ms=20.0, bandwidth_gbps=10.0),
        "fast": LinkInfo(node_id="fast", region="r2", latency_ms=20.0, bandwidth_gbps=200.0),
    })
    policy = PlacementPolicy(carbon_aware=False)
    ranked = select_placement(["slow", "fast"], topo, policy)
    assert ranked[0] == "fast"


# ── (2) Carbon-aware: lower carbon wins within latency threshold ─────────

def test_carbon_aware_prefers_lower_carbon_within_threshold():
    # Both latency 20 ms; node-b is greener (carbon 50 vs 400). Threshold 50 ms.
    topo = _topo(latency_a=20.0, latency_b=20.0, carbon_a=400.0, carbon_b=50.0)
    policy = PlacementPolicy(carbon_aware=True, carbon_weight=0.5, latency_threshold_ms=50.0)
    ranked = select_placement(["node-a", "node-b"], topo, policy)
    assert ranked[0] == "node-b", f"expected greener node first, got {ranked}"


def test_carbon_ignored_when_latency_outside_threshold():
    # node-b is greener but much higher latency (> threshold) → topology dominates.
    topo = _topo(latency_a=10.0, latency_b=200.0, carbon_a=400.0, carbon_b=10.0)
    policy = PlacementPolicy(carbon_aware=True, carbon_weight=1.0, latency_threshold_ms=50.0)
    ranked = select_placement(["node-a", "node-b"], topo, policy)
    assert ranked[0] == "node-a", f"topology must dominate beyond threshold, got {ranked}"


def test_carbon_aware_false_ignores_carbon():
    topo = _topo(latency_a=20.0, latency_b=20.0, carbon_a=400.0, carbon_b=10.0)
    policy = PlacementPolicy(carbon_aware=False)
    ranked = select_placement(["node-a", "node-b"], topo, policy)
    # With carbon disabled and equal latency/bw, tie → sorted by node id.
    assert ranked == ["node-a", "node-b"]


# ── (3) Canary / AB routing — deterministic fraction ─────────────────────

def test_canary_routes_weight_fraction_deterministically():
    reqs = [f"req-{i}" for i in range(100)]
    stable = ["stable-1", "stable-2"]
    canary = ["canary-1"]

    a = route_canary(reqs, stable, canary, canary_weight=0.10, seed=42)
    b = route_canary(reqs, stable, canary, canary_weight=0.10, seed=42)
    assert a == b, "canary routing must be deterministic for a fixed seed"

    canary_assigned = [rid for rid, n in a.items() if n in canary]
    assert len(canary_assigned) == 10, f"expected 10 canary reqs, got {len(canary_assigned)}"
    # All canary traffic on canary nodes; rest on stable nodes.
    assert all(a[rid] in canary for rid in canary_assigned)
    assert all(a[rid] in stable for rid in reqs if rid not in canary_assigned)


def test_canary_weight_zero_and_one_extremes():
    reqs = [f"r{i}" for i in range(7)]
    stable = ["s1"]
    canary = ["c1"]

    zero = route_canary(reqs, stable, canary, canary_weight=0.0, seed=1)
    assert all(n == "s1" for n in zero.values())

    one = route_canary(reqs, stable, canary, canary_weight=1.0, seed=1)
    assert all(n == "c1" for n in one.values())


def test_canary_router_wrapper_consistent():
    router = CanaryRouter(
        stable_nodes=["s1", "s2"], canary_nodes=["c1"], canary_weight=0.25, seed=7,
    )
    reqs = [f"r{i}" for i in range(40)]
    mapping = router.route(reqs)
    assert router.canary_fraction(40) == 10
    assert sum(1 for n in mapping.values() if n == "c1") == 10


def test_canary_rejects_impossible_config():
    with pytest.raises(ValueError):
        route_canary(["r1"], ["s1"], [], canary_weight=0.5, seed=0)
    with pytest.raises(ValueError):
        route_canary(["r1"], [], ["c1"], canary_weight=0.5, seed=0)


# ── (4) Migration plan — valid, ordered, cost-reducing ──────────────────

def _three_node_topo():
    return NodeTopology(coordinator_region="us", links={
        "dirty-slow": LinkInfo(
            node_id="dirty-slow", region="coal", latency_ms=200.0,
            bandwidth_gbps=10.0, carbon_intensity=800.0,
        ),
        "clean-fast": LinkInfo(
            node_id="clean-fast", region="hydro", latency_ms=20.0,
            bandwidth_gbps=200.0, carbon_intensity=30.0,
        ),
        "mid": LinkInfo(
            node_id="mid", region="gas", latency_ms=50.0,
            bandwidth_gbps=100.0, carbon_intensity=300.0,
        ),
    })


def test_plan_migration_returns_valid_ordered_plan():
    topo = _three_node_topo()
    policy = PlacementPolicy(carbon_aware=True, carbon_weight=0.3, latency_threshold_ms=1000.0)
    plan = plan_migration(["dirty-slow", "clean-fast", "mid"], topo, policy)

    assert plan, "expected at least one migration step"
    # every step is from a higher-cost node to a lower-cost node
    for step in plan:
        assert step.cost_reduction > 0
        assert step.cost_after < step.cost_before
    # ordered by descending reduction
    reductions = [s.cost_reduction for s in plan]
    assert reductions == sorted(reductions, reverse=True)
    # source of the top move is the dirtiest node
    assert plan[0].from_node == "dirty-slow"
    assert plan[0].to_node == "clean-fast"


def test_plan_migration_empty_when_no_improvement():
    # Single optimal node → nothing to migrate.
    topo = NodeTopology(coordinator_region="us", links={
        "best": LinkInfo(node_id="best", region="r", latency_ms=10.0,
                         bandwidth_gbps=200.0, carbon_intensity=10.0),
    })
    plan = plan_migration(["best"], topo, PlacementPolicy())
    assert plan == []


def test_plan_migration_does_not_mutate_inputs():
    topo = _three_node_topo()
    nodes = ["dirty-slow", "clean-fast", "mid"]
    snapshot = list(nodes)
    plan_migration(nodes, topo, PlacementPolicy())
    assert nodes == snapshot, "plan_migration must not mutate the node list"


# ── Integration: BatchScheduler placement hook (no select() touched) ──────

def test_batch_scheduler_placement_hook_delegates():
    from distllm.core.batch_scheduler import BatchScheduler

    sched = BatchScheduler()
    # Without a policy, returns input order unchanged.
    assert sched.recommend_placement(["x", "y"]) == ["x", "y"]

    topo = _topo(latency_a=10.0, latency_b=80.0)
    sched.set_placement_policy(PlacementPolicy(carbon_aware=False), topo)
    ranked = sched.recommend_placement(["node-a", "node-b"])
    assert ranked == ["node-a", "node-b"]


def test_node_topology_from_network_adapter():
    """The lazy adapter from distllm.dist.network.Topology works (no GPU)."""
    from distllm.dist.network import create_ring_topology

    net = create_ring_topology(num_nodes=4, bandwidth=10.0)
    topo = NodeTopology.from_network(
        net,
        regions={0: "us", 1: "eu", 2: "ap", 3: "us"},
        carbon={"us": 400.0, "eu": 100.0, "ap": 600.0},
    )
    assert set(topo.node_ids()) == {"node0", "node1", "node2", "node3"}
    # node acting as coordinator (idx 0) has ~0 latency; others higher.
    ranked = select_placement(topo.node_ids(), topo, PlacementPolicy(carbon_aware=False))
    assert ranked[0] == "node0"
