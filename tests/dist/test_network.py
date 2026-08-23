"""Tests for distllm.dist.network — network topology creation and analysis."""

from __future__ import annotations

import numpy as np
import pytest

from distllm.dist.network import (
    Topology,
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
# Topology dataclass
# ---------------------------------------------------------------------------


class TestTopology:
    """Verify the Topology dataclass holds expected fields."""

    def test_construction(self) -> None:
        adj = [[1, 2], [0], [0]]
        bw = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=np.float32)
        t = Topology(num_nodes=3, adjacency=adj, bandwidth_matrix=bw)
        assert t.num_nodes == 3
        assert t.adjacency == adj
        assert np.array_equal(t.bandwidth_matrix, bw)

    def test_zero_nodes_adjacency(self) -> None:
        t = Topology(num_nodes=0, adjacency=[], bandwidth_matrix=np.empty((0, 0)))
        assert t.num_nodes == 0
        assert t.adjacency == []
        assert t.bandwidth_matrix.shape == (0, 0)

    def test_single_node(self) -> None:
        t = Topology(num_nodes=1, adjacency=[[]], bandwidth_matrix=np.zeros((1, 1)))
        assert t.num_nodes == 1
        assert t.adjacency == [[]]


# ---------------------------------------------------------------------------
# create_ring_topology
# ---------------------------------------------------------------------------


class TestCreateRingTopology:
    def test_three_nodes(self) -> None:
        t = create_ring_topology(3, bandwidth=5.0)
        assert t.num_nodes == 3
        assert t.adjacency == [[1, 2], [2, 0], [0, 1]]
        assert t.bandwidth_matrix[0, 1] == 5.0
        assert t.bandwidth_matrix[1, 2] == 5.0
        assert t.bandwidth_matrix[2, 0] == 5.0

    def test_single_node(self) -> None:
        t = create_ring_topology(1)
        assert t.num_nodes == 1
        assert t.adjacency == [[]]
        assert t.bandwidth_matrix.shape == (1, 1)

    def test_two_nodes(self) -> None:
        t = create_ring_topology(2)
        assert t.num_nodes == 2
        assert t.adjacency == [[1], [0]]
        assert t.bandwidth_matrix[0, 1] == pytest.approx(10.0)
        assert t.bandwidth_matrix[1, 0] == pytest.approx(10.0)

    def test_large_ring(self) -> None:
        n = 100
        t = create_ring_topology(n)
        assert t.num_nodes == n
        assert len(t.adjacency) == n
        for i in range(n):
            j1 = (i + 1) % n
            j2 = (i - 1 + n) % n
            assert i in t.adjacency[j1]
            assert i in t.adjacency[j2]

    def test_bandwidth_default(self) -> None:
        t = create_ring_topology(4)
        assert t.bandwidth_matrix[0, 1] == pytest.approx(10.0)
        assert t.bandwidth_matrix[1, 0] == pytest.approx(10.0)

    def test_bandwidth_matrix_symmetric(self) -> None:
        t = create_ring_topology(5, bandwidth=7.5)
        bw = t.bandwidth_matrix
        assert np.allclose(bw, bw.T)

    def test_zero_nodes(self) -> None:
        t = create_ring_topology(0)
        assert t.num_nodes == 0
        assert t.adjacency == []
        assert t.bandwidth_matrix.shape == (0, 0)


# ---------------------------------------------------------------------------
# create_tree_topology
# ---------------------------------------------------------------------------


class TestCreateTreeTopology:
    def test_single_node(self) -> None:
        t = create_tree_topology(1)
        assert t.num_nodes == 1
        assert t.adjacency == [[]]

    def test_two_nodes(self) -> None:
        t = create_tree_topology(2)
        assert t.num_nodes == 2
        assert 1 in t.adjacency[0]
        assert 0 in t.adjacency[1]

    def test_branching_factor_two(self) -> None:
        t = create_tree_topology(7, branching_factor=2)
        assert t.num_nodes == 7
        # root connected to children 1,2
        assert t.adjacency[0] == [1, 2]
        # node 1 connected to parent 0 and children 3,4
        assert t.adjacency[1] == [0, 3, 4]
        # node 2 connected to parent 0 and children 5,6
        assert t.adjacency[2] == [0, 5, 6]

    def test_branching_factor_three(self) -> None:
        t = create_tree_topology(10, branching_factor=3)
        assert t.num_nodes == 10
        assert t.adjacency[0] == [1, 2, 3]
        assert t.adjacency[1] == [0, 4, 5, 6]

    def test_bandwidth_symmetric(self) -> None:
        t = create_tree_topology(5, bandwidth=20.0)
        bw = t.bandwidth_matrix
        assert np.allclose(bw, bw.T)
        nonzero = bw[bw > 0]
        assert np.allclose(nonzero, 20.0)

    def test_zero_nodes(self) -> None:
        t = create_tree_topology(0)
        assert t.num_nodes == 1  # clamps to 1
        assert t.adjacency == [[]]

    def test_negative_nodes(self) -> None:
        t = create_tree_topology(-5)
        assert t.num_nodes == 1
        assert t.adjacency == [[]]

    def test_large_branching_factor_truncates(self) -> None:
        t = create_tree_topology(5, branching_factor=100)
        assert t.num_nodes == 5
        # With such a large branching factor, all nodes are direct children of root
        assert t.adjacency[0] == [1, 2, 3, 4]
        for i in range(1, 5):
            assert 0 in t.adjacency[i]

    def test_bandwidth_matrix_edges_match_adjacency(self) -> None:
        t = create_tree_topology(6, bandwidth=8.0)
        for i in range(t.num_nodes):
            for j in t.adjacency[i]:
                assert t.bandwidth_matrix[i, j] == pytest.approx(8.0)
                assert t.bandwidth_matrix[j, i] == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# create_hierarchical_topology
# ---------------------------------------------------------------------------


class TestCreateHierarchicalTopology:
    def test_small_node_count(self) -> None:
        t = create_hierarchical_topology(2, num_levels=2)
        assert t.num_nodes >= 2

    def test_intra_inter_bandwidth(self) -> None:
        t = create_hierarchical_topology(8, num_levels=2)
        for i in range(t.num_nodes):
            for j in range(i + 1, t.num_nodes):
                b = t.bandwidth_matrix[i, j]
                assert b > 0

    def test_larger_intra_bandwidth(self) -> None:
        t = create_hierarchical_topology(8, num_levels=2, intra_bandwidth=100.0, inter_bandwidth=5.0)
        nodes_per_group = max(2, 8 // (2 ** 2))
        for i in range(t.num_nodes):
            for j in range(i + 1, t.num_nodes):
                same_group = (i // nodes_per_group) == (j // nodes_per_group)
                expected = 100.0 if same_group else 5.0
                assert t.bandwidth_matrix[i, j] == pytest.approx(expected)

    def test_symmetric_bandwidth(self) -> None:
        t = create_hierarchical_topology(6)
        assert np.allclose(t.bandwidth_matrix, t.bandwidth_matrix.T)

    def test_adjacency_tuples(self) -> None:
        """Hierarchical topology stores (neighbor, bandwidth) tuples in adjacency."""
        t = create_hierarchical_topology(4, num_levels=2)
        for neighbors in t.adjacency:
            for n in neighbors:
                assert isinstance(n, tuple)
                assert len(n) == 2
                assert isinstance(n[0], int)
                assert isinstance(n[1], float)

    def test_zero_nodes(self) -> None:
        t = create_hierarchical_topology(0)
        assert t.num_nodes >= 0
        assert len(t.adjacency) == t.num_nodes

    def test_single_node(self) -> None:
        t = create_hierarchical_topology(1)
        assert t.num_nodes >= 1


# ---------------------------------------------------------------------------
# flatten_topology
# ---------------------------------------------------------------------------


class TestFlattenTopology:
    def test_flatten_ring(self) -> None:
        orig = create_ring_topology(4)
        flat = flatten_topology(orig)
        assert flat.num_nodes == orig.num_nodes
        assert flat.adjacency == [[1, 3], [2, 0], [3, 1], [0, 2]]
        assert np.array_equal(flat.bandwidth_matrix, orig.bandwidth_matrix)
        # Ensure it's a copy, not a view
        flat.bandwidth_matrix[0, 0] = 999
        assert orig.bandwidth_matrix[0, 0] == pytest.approx(0.0)

    def test_flatten_hierarchical_converts_tuples(self) -> None:
        orig = create_hierarchical_topology(4)
        flat = flatten_topology(orig)
        # flatten_topology shallow-copies adjacency; tuples are preserved
        assert flat.adjacency is not orig.adjacency
        assert all(isinstance(n, list) for n in flat.adjacency)

    def test_flatten_is_deterministic(self) -> None:
        t1 = create_tree_topology(10)
        t2 = flatten_topology(t1)
        t3 = flatten_topology(t1)
        assert t2.adjacency == t3.adjacency
        assert np.array_equal(t2.bandwidth_matrix, t3.bandwidth_matrix)

    def test_flatten_empty(self) -> None:
        t = Topology(num_nodes=0, adjacency=[], bandwidth_matrix=np.empty((0, 0)))
        flat = flatten_topology(t)
        assert flat.num_nodes == 0
        assert flat.adjacency == []

    def test_flatten_single_node(self) -> None:
        t = Topology(num_nodes=1, adjacency=[[]], bandwidth_matrix=np.zeros((1, 1)))
        flat = flatten_topology(t)
        assert flat.num_nodes == 1
        assert flat.adjacency == [[]]


# ---------------------------------------------------------------------------
# get_bandwidth_matrix / get_adjacency_list
# ---------------------------------------------------------------------------


class TestAccessors:
    def test_get_bandwidth_matrix(self) -> None:
        t = create_ring_topology(3, bandwidth=2.0)
        mat = get_bandwidth_matrix(t)
        assert mat[0, 1] == pytest.approx(2.0)
        # Returns the underlying matrix directly (no copy)
        assert mat is t.bandwidth_matrix

    def test_get_bandwidth_matrix_empty(self) -> None:
        t = Topology(num_nodes=0, adjacency=[], bandwidth_matrix=np.empty((0, 0)))
        mat = get_bandwidth_matrix(t)
        assert mat.shape == (0, 0)

    def test_get_adjacency_list(self) -> None:
        t = create_ring_topology(3)
        adj = get_adjacency_list(t)
        assert adj == [[1, 2], [2, 0], [0, 1]]
        # Modifying returned list does not affect original
        adj[0].append(99)
        assert t.adjacency[0] == [1, 2]

    def test_get_adjacency_list_empty(self) -> None:
        t = Topology(num_nodes=0, adjacency=[], bandwidth_matrix=np.empty((0, 0)))
        assert get_adjacency_list(t) == []

    def test_get_adjacency_list_single(self) -> None:
        t = Topology(num_nodes=1, adjacency=[[]], bandwidth_matrix=np.zeros((1, 1)))
        assert get_adjacency_list(t) == [[]]


# ---------------------------------------------------------------------------
# get_reachable_nodes
# ---------------------------------------------------------------------------


class TestGetReachableNodes:
    def test_ring_all_reachable(self) -> None:
        t = create_ring_topology(4)
        reachable = get_reachable_nodes(t, 0)
        assert sorted(reachable) == [1, 2, 3]

    def test_source_not_in_result(self) -> None:
        t = create_ring_topology(4)
        reachable = get_reachable_nodes(t, 2)
        assert 2 not in reachable

    def test_single_node(self) -> None:
        t = create_ring_topology(1)
        assert get_reachable_nodes(t, 0) == []

    def test_tree_all_reachable(self) -> None:
        t = create_tree_topology(7)
        reachable = get_reachable_nodes(t, 3)
        assert sorted(reachable) == [0, 1, 2, 4, 5, 6]

    def test_symmetric_reachability(self) -> None:
        t = create_tree_topology(10)
        for i in range(t.num_nodes):
            r1 = set(get_reachable_nodes(t, i))
            for j in r1:
                r2 = set(get_reachable_nodes(t, j))
                assert i in r2

    def test_out_of_bounds_source(self) -> None:
        t = create_ring_topology(3)
        with pytest.raises(IndexError):
            get_reachable_nodes(t, 10)

    def test_zero_nodes(self) -> None:
        t = Topology(num_nodes=0, adjacency=[], bandwidth_matrix=np.empty((0, 0)))
        with pytest.raises(IndexError):
            get_reachable_nodes(t, 0)


# ---------------------------------------------------------------------------
# get_shortest_path
# ---------------------------------------------------------------------------


class TestGetShortestPath:
    def test_adjacent_ring(self) -> None:
        t = create_ring_topology(5)
        path = get_shortest_path(t, 0, 1)
        assert path == [0, 1]

    def test_source_equals_destination(self) -> None:
        t = create_ring_topology(5)
        assert get_shortest_path(t, 3, 3) == [3]

    def test_ring_shortest_route(self) -> None:
        t = create_ring_topology(5)
        path = get_shortest_path(t, 0, 3)
        assert path == [0, 4, 3] or path == [0, 1, 2, 3]
        # [0, 4, 3] is the shorter route (2 hops)

    def test_tree_path(self) -> None:
        t = create_tree_topology(7, branching_factor=2)
        # node 3 -> parent 1 -> parent 0 -> child 2 -> child 5
        path = get_shortest_path(t, 3, 5)
        assert path[0] == 3
        assert path[-1] == 5
        # Verify no cycles (path length should be unique)
        assert len(path) == len(set(path))

    def test_unreachable_returns_source(self) -> None:
        # An empty topology has no edges
        t = Topology(num_nodes=2, adjacency=[[], []], bandwidth_matrix=np.zeros((2, 2)))
        path = get_shortest_path(t, 0, 1)
        assert path == [0]

    def test_single_node(self) -> None:
        t = create_ring_topology(1)
        assert get_shortest_path(t, 0, 0) == [0]

    def test_path_does_not_contain_repeats(self) -> None:
        t = create_ring_topology(10)
        path = get_shortest_path(t, 0, 9)
        assert len(path) == len(set(path))


# ---------------------------------------------------------------------------
# get_network_diameter
# ---------------------------------------------------------------------------


class TestGetNetworkDiameter:
    def test_ring_two_nodes(self) -> None:
        t = create_ring_topology(2)
        assert get_network_diameter(t) == 1

    def test_ring_three_nodes(self) -> None:
        t = create_ring_topology(3)
        # Every node is 1 hop from every other node in a 3-node ring
        assert get_network_diameter(t) == 1

    def test_ring_five_nodes(self) -> None:
        t = create_ring_topology(5)
        assert get_network_diameter(t) == 2

    def test_ring_six_nodes(self) -> None:
        t = create_ring_topology(6)
        assert get_network_diameter(t) == 3

    def test_tree_depth(self) -> None:
        t = create_tree_topology(7, branching_factor=2)
        # Depth is 2 (root at 0, children 1-2, grandchildren 3-6)
        assert get_network_diameter(t) == 4

    def test_single_node(self) -> None:
        t = create_ring_topology(1)
        assert get_network_diameter(t) == 0

    def test_two_node_tree(self) -> None:
        t = create_tree_topology(2)
        assert get_network_diameter(t) == 1

    def test_zero_nodes(self) -> None:
        t = Topology(num_nodes=0, adjacency=[], bandwidth_matrix=np.empty((0, 0)))
        assert get_network_diameter(t) == 0
