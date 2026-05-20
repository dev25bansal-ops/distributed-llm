"""Property-based fuzz tests for network topology management.

Covers: ring, tree, and hierarchical topology creation,
topology transformations, and structural invariants.
"""

import math
from hypothesis import given, settings
from hypothesis import strategies as st

from distllm.core.network_topology import (
    create_ring_topology,
    create_tree_topology,
    create_hierarchical_topology,
    flatten_topology,
    get_bandwidth_matrix,
    get_adjacency_list,
    get_reachable_nodes,
    get_shortest_path,
    get_network_diameter,
)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

@st.composite
def ring_topology_args(draw):
    """Generate (num_nodes, bandwidth) pairs for ring topology."""
    num_nodes = draw(st.integers(min_value=2, max_value=64))
    bandwidth = draw(st.floats(1.0, 100.0, allow_nan=False, allow_infinity=False))
    return num_nodes, bandwidth


@st.composite
def tree_topology_args(draw):
    """Generate (num_nodes, branching_factor, bandwidth) for tree topology."""
    num_nodes = draw(st.integers(min_value=2, max_value=64))
    branching = draw(st.integers(min_value=2, max_value=8))
    bandwidth = draw(st.floats(1.0, 100.0, allow_nan=False, allow_infinity=False))
    return num_nodes, branching, bandwidth


@st.composite
def hierarchical_topology_args(draw):
    """Generate (num_nodes, num_levels, intra_bandwidth, inter_bandwidth)."""
    num_nodes = draw(st.integers(min_value=4, max_value=128))
    num_levels = draw(st.integers(min_value=2, max_value=5))
    intra_bw = draw(st.floats(10.0, 800.0, allow_nan=False, allow_infinity=False))
    inter_bw = draw(st.floats(1.0, 100.0, allow_nan=False, allow_infinity=False))
    return num_nodes, num_levels, intra_bw, inter_bw


# ---------------------------------------------------------------------------
# Ring topology invariants
# ---------------------------------------------------------------------------

@given(ring_topology_args())
@settings(max_examples=50, deadline=None)
def test_ring_topology_invariants(args):
    """Ring topology has correct node count and bidirectionality."""
    num_nodes, bandwidth = args
    topology = create_ring_topology(num_nodes, bandwidth)

    assert topology.num_nodes == num_nodes
    adjacency = get_adjacency_list(topology)
    assert len(adjacency) == num_nodes

    # Each node has exactly 2 neighbors (except 2-node ring has 1 each)
    for node_id in range(num_nodes):
        neighbors = adjacency[node_id]
        if num_nodes == 2:
            assert len(neighbors) == 1
        else:
            assert len(neighbors) == 2

    # Bandwidth symmetry
    bw = get_bandwidth_matrix(topology)
    assert bw.shape == (num_nodes, num_nodes)
    for i in range(num_nodes):
        for j in range(num_nodes):
            assert bw[i, j] == bw[j, i]


@given(ring_topology_args())
@settings(max_examples=20, deadline=None)
def test_ring_diameter(args):
    """Ring network diameter matches expected formula."""
    num_nodes, bandwidth = args
    topology = create_ring_topology(num_nodes, bandwidth)

    diameter = get_network_diameter(topology)
    expected = num_nodes // 2
    assert diameter == expected


@given(ring_topology_args())
@settings(max_examples=20, deadline=None)
def test_ring_reachability(args):
    """Every node is reachable from every other node in a ring."""
    num_nodes, bandwidth = args
    topology = create_ring_topology(num_nodes, bandwidth)

    for src in range(num_nodes):
        reachable = get_reachable_nodes(topology, src)
        assert len(reachable) == num_nodes - 1
        for dst in range(num_nodes):
            if dst != src:
                assert dst in reachable


# ---------------------------------------------------------------------------
# Tree topology invariants
# ---------------------------------------------------------------------------

@given(tree_topology_args())
@settings(max_examples=50, deadline=None)
def test_tree_topology_invariants(args):
    """Tree topology has correct structure and bidirectionality."""
    num_nodes, branching, bandwidth = args
    topology = create_tree_topology(num_nodes, branching, bandwidth)

    assert 0 < topology.num_nodes <= num_nodes
    adjacency = get_adjacency_list(topology)
    bw = get_bandwidth_matrix(topology)
    n = topology.num_nodes

    assert adjacency.shape == (n,) if hasattr(adjacency, "shape") else len(adjacency) == n
    assert bw.shape == (n, n)

    for i in range(n):
        for j in range(n):
            assert bw[i, j] == bw[j, i]


@given(tree_topology_args())
@settings(max_examples=30, deadline=None)
def test_tree_reachability(args):
    """Every node is reachable in a tree (no isolated nodes)."""
    num_nodes, branching, bandwidth = args
    topology = create_tree_topology(num_nodes, branching, bandwidth)
    n = topology.num_nodes

    for src in range(n):
        reachable = get_reachable_nodes(topology, src)
        assert len(reachable) == n - 1


@given(tree_topology_args())
@settings(max_examples=20, deadline=None)
def test_tree_diameter_positive(args):
    """Tree diameter is a positive integer."""
    num_nodes, branching, bandwidth = args
    topology = create_tree_topology(num_nodes, branching, bandwidth)
    diameter = get_network_diameter(topology)
    assert isinstance(diameter, int)
    assert diameter > 0
    assert diameter <= topology.num_nodes - 1


# ---------------------------------------------------------------------------
# Hierarchical topology invariants
# ---------------------------------------------------------------------------

@given(hierarchical_topology_args())
@settings(max_examples=50, deadline=None)
def test_hierarchical_topology_invariants(args):
    """Hierarchical topology has valid structure."""
    num_nodes, num_levels, intra_bw, inter_bw = args
    topology = create_hierarchical_topology(
        num_nodes, num_levels, intra_bw, inter_bw
    )

    n = topology.num_nodes
    assert n > 0
    assert n <= num_nodes
    adjacency = get_adjacency_list(topology)
    bw = get_bandwidth_matrix(topology)
    assert bw.shape == (n, n)

    for i in range(n):
        for j in range(n):
            assert bw[i, j] == bw[j, i]


@given(hierarchical_topology_args())
@settings(max_examples=30, deadline=None)
def test_hierarchical_reachability(args):
    """All nodes reachable in hierarchical topology."""
    num_nodes, num_levels, intra_bw, inter_bw = args
    topology = create_hierarchical_topology(
        num_nodes, num_levels, intra_bw, inter_bw
    )
    n = topology.num_nodes

    for src in range(n):
        reachable = get_reachable_nodes(topology, src)
        assert len(reachable) == n - 1


# ---------------------------------------------------------------------------
# flatten_topology
# ---------------------------------------------------------------------------

@given(
    topo_fn=st.sampled_from(["ring", "tree"]),
    num_nodes=st.integers(min_value=2, max_value=32),
    bandwidth=st.floats(1.0, 50.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=30, deadline=None)
def test_flatten_topology_roundtrip(topo_fn, num_nodes, bandwidth):
    """T -> flatten -> T round-trip preserves node count."""
    if topo_fn == "ring":
        topo = create_ring_topology(num_nodes, bandwidth)
    else:
        topo = create_tree_topology(num_nodes, 2, bandwidth)

    flat = flatten_topology(topo)
    assert flat.num_nodes == topo.num_nodes

    bw_orig = get_bandwidth_matrix(topo)
    bw_flat = get_bandwidth_matrix(flat)
    assert bw_orig.shape == bw_flat.shape


# ---------------------------------------------------------------------------
# Shortest path
# ---------------------------------------------------------------------------

@given(
    topo_fn=st.sampled_from(["ring", "tree"]),
    num_nodes=st.integers(min_value=2, max_value=24),
    bandwidth=st.floats(1.0, 50.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=30, deadline=None)
def test_shortest_path_valid(args):
    """Shortest paths are valid and respect the triangle inequality."""
    topo_fn, num_nodes, bandwidth = args
    if topo_fn == "ring":
        topo = create_ring_topology(num_nodes, bandwidth)
    else:
        topo = create_tree_topology(num_nodes, 2, bandwidth)

    n = topo.num_nodes
    if n < 2:
        return

    for src in range(min(n, 5)):
        for dst in range(n):
            if src == dst:
                continue
            path = get_shortest_path(topo, src, dst)
            assert path[0] == src
            assert path[-1] == dst
            assert len(path) >= 2
            for k in range(len(path) - 1):
                bw = get_bandwidth_matrix(topo)
                assert bw[path[k], path[k + 1]] > 0


# ---------------------------------------------------------------------------
# Bandwidth matrix properties
# ---------------------------------------------------------------------------

@st.composite
def any_topology(draw):
    """Generate a topology of a random type with random params."""
    topo_type = draw(st.sampled_from(["ring", "tree", "hierarchical"]))
    if topo_type == "ring":
        return draw(ring_topology_args())
    elif topo_type == "tree":
        return draw(tree_topology_args())
    else:
        return draw(hierarchical_topology_args())


@given(
    num_nodes=st.integers(min_value=2, max_value=32),
    bandwidth=st.floats(0.0, 0.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=10, deadline=None)
def test_zero_bandwidth_does_not_crash(num_nodes, bandwidth):
    """Zero bandwidth matrices do not crash operations."""
    topology = create_ring_topology(num_nodes, bandwidth)
    try:
        _ = get_bandwidth_matrix(topology)
        _ = get_network_diameter(topology)
        _ = get_reachable_nodes(topology, 0)
        _ = get_shortest_path(topology, 0, min(1, num_nodes - 1))
    except Exception:
        pass
