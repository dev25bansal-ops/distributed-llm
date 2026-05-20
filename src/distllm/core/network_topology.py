"""Network topology creation and analysis for distributed LLM inference."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Topology:
    num_nodes: int
    adjacency: list[list[int]]
    bandwidth_matrix: np.ndarray


def create_ring_topology(num_nodes: int, bandwidth: float = 10.0) -> Topology:
    adj = [[] for _ in range(num_nodes)]
    for i in range(num_nodes):
        j = (i + 1) % num_nodes
        if j != i:
            adj[i].append(j)
        k = (i - 1 + num_nodes) % num_nodes
        if k != i and k != j:
            adj[i].append(k)
    bw = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i in range(num_nodes):
        j = (i + 1) % num_nodes
        bw[i, j] = bandwidth
        bw[j, i] = bandwidth
    return Topology(num_nodes=num_nodes, adjacency=adj, bandwidth_matrix=bw)


def create_tree_topology(num_nodes: int, branching_factor: int = 2, bandwidth: float = 10.0) -> Topology:
    actual_nodes = max(1, num_nodes)
    adj = [[] for _ in range(actual_nodes)]
    bw = np.zeros((actual_nodes, actual_nodes), dtype=np.float32)
    for i in range(1, actual_nodes):
        parent = (i - 1) // branching_factor
        adj[parent].append(i)
        adj[i].append(parent)
        bw[i, parent] = bandwidth
        bw[parent, i] = bandwidth
    return Topology(num_nodes=actual_nodes, adjacency=adj, bandwidth_matrix=bw)


def create_hierarchical_topology(
    num_nodes: int,
    num_levels: int = 2,
    intra_bandwidth: float = 100.0,
    inter_bandwidth: float = 10.0,
) -> Topology:
    nodes_per_group = max(2, num_nodes // (num_levels ** 2))
    actual_nodes = min(num_nodes, nodes_per_group * num_levels)
    adj = [[] for _ in range(actual_nodes)]
    bw = np.zeros((actual_nodes, actual_nodes), dtype=np.float32)
    for i in range(actual_nodes):
        for j in range(i + 1, actual_nodes):
            same_group = (i // nodes_per_group) == (j // nodes_per_group)
            b = intra_bandwidth if same_group else inter_bandwidth
            bw[i, j] = b
            bw[j, i] = b
            adj[i].append((j, b))
            adj[j].append((i, b))
    return Topology(num_nodes=actual_nodes, adjacency=adj, bandwidth_matrix=bw)


def flatten_topology(topology: Topology) -> Topology:
    flat_adj = [list(neighbors) for neighbors in topology.adjacency]
    return Topology(
        num_nodes=topology.num_nodes,
        adjacency=flat_adj,
        bandwidth_matrix=topology.bandwidth_matrix.copy(),
    )


def get_bandwidth_matrix(topology: Topology) -> np.ndarray:
    return topology.bandwidth_matrix


def get_adjacency_list(topology: Topology) -> list[list[int]]:
    return [list(n) for n in topology.adjacency]


def get_reachable_nodes(topology: Topology, source: int) -> list[int]:
    visited = set()
    stack = [source]
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        for neighbor in topology.adjacency[node]:
            nid = neighbor[0] if isinstance(neighbor, tuple) else neighbor
            if nid not in visited:
                stack.append(nid)
    visited.discard(source)
    return list(visited)


def get_shortest_path(topology: Topology, source: int, destination: int) -> list[int]:
    if source == destination:
        return [source]
    visited = {source}
    queue = [[source]]
    while queue:
        path = queue.pop(0)
        node = path[-1]
        for neighbor in topology.adjacency[node]:
            nid = neighbor[0] if isinstance(neighbor, tuple) else neighbor
            if nid == destination:
                return path + [nid]
            if nid not in visited:
                visited.add(nid)
                queue.append(path + [nid])
    return [source]


def get_network_diameter(topology: Topology) -> int:
    max_dist = 0
    for i in range(topology.num_nodes):
        for j in range(topology.num_nodes):
            if i != j:
                path = get_shortest_path(topology, i, j)
                if len(path) > 1:
                    max_dist = max(max_dist, len(path) - 1)
    return max_dist
