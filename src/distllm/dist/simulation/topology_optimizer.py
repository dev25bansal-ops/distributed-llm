"""Topology optimizer that finds optimal model partitioning across a hardware pool.

Uses brute-force search for small node counts and heuristic search
for larger clusters, respecting user constraints such as max latency,
min throughput, and memory budget.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any, Callable

from distllm.dist.simulation.cluster_simulator import (
    ClusterSimulator,
    ModelConfig,
    NodeSpec,
    get_model_preset,
)


@dataclass(frozen=True)
class PartitionSolution:
    """Optimal partitioning of model layers across nodes."""

    assignments: list[tuple[str, int, int]]  # (node_id, start_layer, end_layer)
    estimated_latency_ms: float = 0.0
    estimated_throughput_tok_s: float = 0.0
    memory_used_gb: dict[str, float] = field(default_factory=dict)
    node_utilization: dict[str, float] = field(default_factory=dict)
    search_method: str = "unknown"
    num_candidates_evaluated: int = 0


@dataclass
class Constraints:
    """User-defined constraints for topology optimization.

    All constraints are optional; unspecified constraints are not enforced.
    """

    max_latency_ms: float | None = None
    min_throughput_tok_s: float | None = None
    memory_budget_gb: float | None = None  # Per-node memory budget
    max_nodes: int | None = None
    min_nodes: int | None = None
    require_tensor_parallelism: int = 1


# ── Search strategies ──────────────────────────────────────────────────


def _evaluate_partition(
    simulator: ClusterSimulator,
    model: ModelConfig,
    nodes: list[NodeSpec],
    partition: list[tuple[str, int, int]],
    batch_size: int,
    seq_len: int,
) -> tuple[float, float]:
    """Evaluate latency and throughput for a given layer-to-node assignment."""
    # Compute per-stage time using the simulator's analytical model
    total_latency = 0.0
    max_stage_time = 0.0

    for idx, (node_id, start_layer, end_layer) in enumerate(partition):
        node = next((n for n in nodes if n.node_id == node_id), None)
        if node is None:
            continue

        num_layers = end_layer - start_layer
        if num_layers <= 0:
            continue

        flops_share = model.flops_per_layer * num_layers

        # Compute time for this stage
        total_flops = flops_share * batch_size * seq_len
        compute_ms = (
            (total_flops / (node.compute_tflops * 1e12)) * 1000.0
        )

        # Communication time (hidden state transfer)
        bytes_per_elem = 2
        comm_bytes = batch_size * seq_len * model.hidden_dim * bytes_per_elem
        comm_ms = (
            (comm_bytes * 8) / (node.interconnect_gbps * 1e9) * 1000.0
            if node.interconnect_gbps > 0
            else 0.0
        )

        stage_time = compute_ms + comm_ms
        max_stage_time = max(max_stage_time, stage_time)

    # Pipeline latency with bubble: stage_time * (N + M - 1) / M
    num_stages = len(partition)
    num_micro_batches = max(1, batch_size // 2)
    if num_stages > 0:
        total_latency = max_stage_time * (
            1 + (num_stages - 1) / num_micro_batches
        )

    total_tokens = batch_size * seq_len
    throughput = (
        (total_tokens / total_latency * 1000.0)
        if total_latency > 0
        else 0.0
    )

    return total_latency, throughput


def _brute_force_search(
    model: ModelConfig,
    nodes: list[NodeSpec],
    batch_size: int,
    seq_len: int,
    constraints: Constraints,
) -> PartitionSolution | None:
    """Exhaustive search: try every contiguous assignment of layers to nodes.

    Suitable for small N (<= 8 nodes).  Uses DP over stages.
    """
    num_nodes = len(nodes)
    num_layers = model.num_layers

    # DP: dp[l][k] = (min_max_latency, split_point)
    INF = float("inf")
    dp: list[list[float]] = [[INF] * num_nodes for _ in range(num_layers + 1)]
    split: list[list[int]] = [[-1] * num_nodes for _ in range(num_layers + 1)]

    # Base: first node gets layers [0, i)
    for i in range(1, num_layers + 1):
        node = nodes[0]
        flops_part = model.flops_per_layer * i
        total_flops = flops_part * batch_size * seq_len
        compute_ms = (total_flops / (node.compute_tflops * 1e12)) * 1000.0

        bytes_elem = 2
        comm_bytes = batch_size * seq_len * model.hidden_dim * bytes_elem
        comm_ms = (
            (comm_bytes * 8) / (node.interconnect_gbps * 1e9) * 1000.0
            if node.interconnect_gbps > 0
            else 0.0
        )
        cost = compute_ms + comm_ms
        dp[i][0] = cost
        split[i][0] = 0

    # DP recurrence
    for k in range(1, num_nodes):
        for i in range(1, num_layers + 1):
            best = INF
            best_j = -1

            for j in range(k, i):
                prev_cost = dp[j][k - 1]
                if math.isinf(prev_cost):
                    continue

                node = nodes[k]
                layer_count = i - j
                flops_part = model.flops_per_layer * layer_count
                total_flops = flops_part * batch_size * seq_len
                compute_ms = (
                    (total_flops / (node.compute_tflops * 1e12)) * 1000.0
                )

                bytes_elem = 2
                comm_bytes = (
                    batch_size * seq_len * model.hidden_dim * bytes_elem
                )
                comm_ms = (
                    (comm_bytes * 8) / (node.interconnect_gbps * 1e9) * 1000.0
                    if node.interconnect_gbps > 0
                    else 0.0
                )

                node_cost = compute_ms + comm_ms
                max_cost = max(prev_cost, node_cost)

                # Apply constraints
                if (
                    constraints.max_latency_ms is not None
                    and max_cost > constraints.max_latency_ms
                ):
                    continue

                if max_cost < best:
                    best = max_cost
                    best_j = j

            dp[i][k] = best
            split[i][k] = best_j

    # Backtrack for best solution
    # Find the best k (number of nodes used) for all layers
    best_cost = INF
    best_k = 0
    for k in range(num_nodes):
        if dp[num_layers][k] < best_cost:
            best_cost = dp[num_layers][k]
            best_k = k

    if math.isinf(best_cost):
        return None

    # Backtrack
    assignments: list[tuple[str, int, int]] = []
    end = num_layers
    k = best_k
    while k >= 0 and end > 0:
        start = split[end][k]
        if start < 0:
            break
        node_id = nodes[k].node_id
        assignments.append((node_id, start, end))
        end = start
        k -= 1

    assignments.reverse()

    # Compute final metrics
    simulator = ClusterSimulator()
    for node in nodes:
        simulator.add_node(node)
    latency, throughput = _evaluate_partition(
        simulator, model, nodes, assignments, batch_size, seq_len,
    )

    # Memory usage
    mem_used: dict[str, float] = {}
    node_util: dict[str, float] = {}
    for node_id, start, end in assignments:
        node = next((n for n in nodes if n.node_id == node_id), None)
        if node is None:
            continue
        layer_count = end - start
        weights_gb = (
            model.params_per_layer * layer_count * 2 / (1024**3)
        )
        kv_gb = (
            batch_size
            * seq_len
            * model.num_heads
            * model.head_dim
            * 2
            * 2
            * layer_count
            / (1024**3)
        )
        mem_used[node_id] = round(weights_gb + kv_gb, 4)
        node_util[node_id] = round(
            (weights_gb + kv_gb) / max(node.memory_gb, 0.001), 4
        )

    return PartitionSolution(
        assignments=assignments,
        estimated_latency_ms=round(latency, 2),
        estimated_throughput_tok_s=round(throughput, 2),
        memory_used_gb=mem_used,
        node_utilization=node_util,
        search_method="brute-force-dp",
        num_candidates_evaluated=(
            num_nodes * num_layers * num_layers // 2
        ),
    )


def _heuristic_search(
    model: ModelConfig,
    nodes: list[NodeSpec],
    batch_size: int,
    seq_len: int,
    constraints: Constraints,
) -> PartitionSolution:
    """Greedy / heuristic search for large clusters (> 8 nodes).

    Uses proportional capacity-weighted allocation: faster nodes get
    more layers, slower nodes get fewer.
    """
    num_nodes = len(nodes)
    num_layers = model.num_layers

    # Compute capacity score for each node
    capacities: list[float] = []
    for node in nodes:
        # Capacity = compute TFLOPS / communication cost ratio
        comm_overhead = max(node.interconnect_gbps, 1.0)
        cap = node.compute_tflops * node.gpu_count / comm_overhead
        capacities.append(cap)

    total_cap = sum(capacities)

    # Weighted layer allocation
    assignments: list[tuple[str, int, int]] = []
    start = 0
    for idx, node in enumerate(nodes):
        weight = capacities[idx] / total_cap if total_cap > 0 else 1.0 / num_nodes
        layer_count = max(1, int(round(num_layers * weight)))

        # Ensure last node gets the remainder
        if idx == num_nodes - 1:
            layer_count = num_layers - start

        layer_count = min(layer_count, num_layers - start)
        if layer_count <= 0:
            continue

        end = start + layer_count
        assignments.append((node.node_id, start, end))
        start = end

    # Verify coverage
    if start < num_layers and assignments:
        last_node = assignments[-1][0]
        assignments[-1] = (last_node, assignments[-1][1], num_layers)

    # Evaluate
    simulator = ClusterSimulator()
    for node in nodes:
        simulator.add_node(node)
    latency, throughput = _evaluate_partition(
        simulator, model, nodes, assignments, batch_size, seq_len,
    )

    # Memory usage
    mem_used: dict[str, float] = {}
    node_util: dict[str, float] = {}
    for node_id, start, end in assignments:
        node = next((n for n in nodes if n.node_id == node_id), None)
        if node is None:
            continue
        layer_count = end - start
        weights_gb = model.params_per_layer * layer_count * 2 / (1024**3)
        kv_gb = (
            batch_size
            * seq_len
            * model.num_heads
            * model.head_dim
            * 2
            * 2
            * layer_count
            / (1024**3)
        )
        mem_used[node_id] = round(weights_gb + kv_gb, 4)
        node_util[node_id] = round(
            (weights_gb + kv_gb) / max(node.memory_gb, 0.001), 4
        )

    return PartitionSolution(
        assignments=assignments,
        estimated_latency_ms=round(latency, 2),
        estimated_throughput_tok_s=round(throughput, 2),
        memory_used_gb=mem_used,
        node_utilization=node_util,
        search_method="heuristic-capacity-weighted",
        num_candidates_evaluated=num_nodes,
    )


def _optimize_with_beam_search(
    model: ModelConfig,
    nodes: list[NodeSpec],
    batch_size: int,
    seq_len: int,
    beam_width: int = 8,
) -> PartitionSolution:
    """Beam search over layer-to-node assignments.

    Explores multiple candidate partitions in parallel, keeping the
    top-K promising solutions at each step.
    """
    num_nodes = len(nodes)
    num_layers = model.num_layers

    # State: (assignments, cost)
    # assignments: list of (node_id, start_layer, end_layer)
    # cost: max stage time
    beam: list[tuple[list[tuple[str, int, int]], float]] = [
        ([], 0.0)
    ]

    # Evaluate cost of adding layers to a node
    def _stage_cost(node: NodeSpec, layer_count: int) -> float:
        flops_part = model.flops_per_layer * layer_count
        total_flops = flops_part * batch_size * seq_len
        compute_ms = (total_flops / (node.compute_tflops * 1e12)) * 1000.0
        bytes_elem = 2
        comm_bytes = batch_size * seq_len * model.hidden_dim * bytes_elem
        comm_ms = (
            (comm_bytes * 8) / (node.interconnect_gbps * 1e9) * 1000.0
            if node.interconnect_gbps > 0
            else 0.0
        )
        return compute_ms + comm_ms

    for current_layer in range(1, num_layers + 1):
        candidates: list[tuple[list[tuple[str, int, int]], float]] = []

        for assignments, current_cost in beam:
            used_nodes = len(assignments)
            if used_nodes == 0:
                # Assign first layers to first node
                node = nodes[0]
                cost = _stage_cost(node, current_layer)
                candidates.append(
                    ([(node.node_id, 0, current_layer)], cost)
                )
            else:
                # Option 1: extend last node's range
                last_node_id, last_start, last_end = assignments[-1]
                last_node = next(
                    (n for n in nodes if n.node_id == last_node_id), nodes[-1]
                )
                new_assignments = list(assignments)
                new_assignments[-1] = (
                    last_node_id, last_start, current_layer,
                )
                cost = _stage_cost(
                    last_node, current_layer - last_start,
                )
                candidates.append(
                    (new_assignments, max(current_cost, cost))
                )

                # Option 2: start new node (if available)
                if used_nodes < num_nodes:
                    next_node = nodes[used_nodes]
                    # Close previous node if it exists
                    if current_layer > last_end:
                        cost_new = _stage_cost(next_node, current_layer - current_layer + 1)
                        candidates.append(
                            (
                                list(assignments)
                                + [(next_node.node_id, current_layer - 1, current_layer)],
                                max(current_cost, cost_new),
                            )
                        )

        # Prune to keep top-K by cost
        candidates.sort(key=lambda x: x[1])
        beam = candidates[:beam_width]

    if not beam:
        return _heuristic_search(
            model, nodes, batch_size, seq_len, Constraints(),
        )

    best_assignments, best_cost = beam[0]

    # Fill any coverage gaps
    simulator = ClusterSimulator()
    for node in nodes:
        simulator.add_node(node)
    latency, throughput = _evaluate_partition(
        simulator, model, nodes, best_assignments, batch_size, seq_len,
    )

    mem_used: dict[str, float] = {}
    node_util: dict[str, float] = {}
    for node_id, start, end in best_assignments:
        node = next((n for n in nodes if n.node_id == node_id), None)
        if node is None:
            continue
        layer_count = end - start
        weights_gb = model.params_per_layer * layer_count * 2 / (1024**3)
        kv_gb = (
            batch_size * seq_len * model.num_heads * model.head_dim
            * 2 * 2 * layer_count / (1024**3)
        )
        mem_used[node_id] = round(weights_gb + kv_gb, 4)
        node_util[node_id] = round(
            (weights_gb + kv_gb) / max(node.memory_gb, 0.001), 4
        )

    return PartitionSolution(
        assignments=best_assignments,
        estimated_latency_ms=round(latency, 2),
        estimated_throughput_tok_s=round(throughput, 2),
        memory_used_gb=mem_used,
        node_utilization=node_util,
        search_method="beam-search",
        num_candidates_evaluated=beam_width * num_layers,
    )


class TopologyOptimizer:
    """Find the optimal layer-to-node partition for a given model and cluster.

    Uses brute-force DP for small clusters (<= 8 nodes) and heuristic
    or beam-search methods for larger clusters.
    """

    def __init__(self) -> None:
        self._last_solution: PartitionSolution | None = None
        self._simulator = ClusterSimulator()

    # ── Public API ───────────────────────────────────────────────────────

    def optimize(
        self,
        model: str | ModelConfig,
        nodes: list[NodeSpec],
        constraints: Constraints | None = None,
        batch_size: int = 1,
        seq_len: int = 2048,
        method: str = "auto",
    ) -> PartitionSolution:
        """Find the optimal partitioning for a model on the given nodes.

        Args:
            model: Model name (preset) or ModelConfig.
            nodes: Available node hardware specs.
            constraints: Optional latency / throughput / memory limits.
            batch_size: Micro-batch size for simulation.
            seq_len: Sequence length.
            method: Search strategy ("auto", "brute-force", "heuristic",
                    "beam-search").  "auto" picks based on node count.

        Returns:
            PartitionSolution with assignments and estimated metrics.
        """
        constraints = constraints or Constraints()
        model_cfg = (
            get_model_preset(model) if isinstance(model, str) else model
        )

        resolved_nodes = list(nodes)
        if not resolved_nodes:
            resolved_nodes = [NodeSpec(node_id="node-0")]

        # Register nodes with the internal simulator
        self._simulator.clear_nodes()
        for node in resolved_nodes:
            self._simulator.add_node(node)

        num_nodes = len(resolved_nodes)

        # Apply node count constraints
        if constraints.max_nodes is not None:
            resolved_nodes = resolved_nodes[: constraints.max_nodes]
        if constraints.min_nodes is not None and num_nodes < constraints.min_nodes:
            resolved_nodes = (
                resolved_nodes
                + [
                    NodeSpec(node_id=f"filler-{i}")
                    for i in range(constraints.min_nodes - num_nodes)
                ]
            )

        # Choose search strategy
        effective_method = method
        if effective_method == "auto":
            if num_nodes <= 8:
                effective_method = "brute-force"
            elif num_nodes <= 32:
                effective_method = "beam-search"
            else:
                effective_method = "heuristic"

        solution: PartitionSolution | None = None

        if effective_method == "brute-force":
            solution = _brute_force_search(
                model_cfg, resolved_nodes,
                batch_size, seq_len, constraints,
            )
            if solution is None:
                # Fall back to heuristic if DP found nothing feasible
                solution = _heuristic_search(
                    model_cfg, resolved_nodes,
                    batch_size, seq_len, constraints,
                )
                solution = PartitionSolution(
                    assignments=solution.assignments,
                    estimated_latency_ms=solution.estimated_latency_ms,
                    estimated_throughput_tok_s=solution.estimated_throughput_tok_s,
                    memory_used_gb=solution.memory_used_gb,
                    node_utilization=solution.node_utilization,
                    search_method="heuristic(dpfail)",
                    num_candidates_evaluated=solution.num_candidates_evaluated,
                )
        elif effective_method == "beam-search":
            solution = _optimize_with_beam_search(
                model_cfg, resolved_nodes,
                batch_size, seq_len,
            )
        else:
            solution = _heuristic_search(
                model_cfg, resolved_nodes,
                batch_size, seq_len, constraints,
            )

        # Enforce memory budget constraint
        if constraints.memory_budget_gb is not None and solution is not None:
            over_budget = False
            for node_id, used_gb in solution.memory_used_gb.items():
                if used_gb > constraints.memory_budget_gb:
                    over_budget = True
                    break
            if over_budget:
                # Re-run with fewer layers per node (heuristic fallback)
                solution = self._constrained_memory_search(
                    model_cfg, resolved_nodes,
                    batch_size, seq_len, constraints,
                )

        # Enforce min throughput constraint
        if (
            constraints.min_throughput_tok_s is not None
            and solution is not None
            and solution.estimated_throughput_tok_s < constraints.min_throughput_tok_s
        ):
            # Try adding more nodes (if available) or warn
            solution = PartitionSolution(
                assignments=solution.assignments,
                estimated_latency_ms=solution.estimated_latency_ms,
                estimated_throughput_tok_s=solution.estimated_throughput_tok_s,
                memory_used_gb=solution.memory_used_gb,
                node_utilization=solution.node_utilization,
                search_method=solution.search_method
                + "(throughput-unmet)",
                num_candidates_evaluated=solution.num_candidates_evaluated,
            )

        if solution is None:
            solution = _heuristic_search(
                model_cfg, resolved_nodes,
                batch_size, seq_len, Constraints(),
            )

        self._last_solution = solution
        return solution

    def _constrained_memory_search(
        self,
        model: ModelConfig,
        nodes: list[NodeSpec],
        batch_size: int,
        seq_len: int,
        constraints: Constraints,
    ) -> PartitionSolution:
        """Re-run optimization with memory budget as the primary constraint.

        Spreads layers across more nodes to reduce per-node memory footprint.
        """
        budget = constraints.memory_budget_gb or nodes[0].memory_gb
        num_layers = model.num_layers
        max_layers_per_node = int(
            budget * (1024**3) / (model.params_per_layer * 2)
        )
        max_layers_per_node = max(max_layers_per_node, 1)

        assignments: list[tuple[str, int, int]] = []
        start = 0
        node_idx = 0
        while start < num_layers and node_idx < len(nodes):
            end = min(start + max_layers_per_node, num_layers)
            node = nodes[node_idx]
            assignments.append((node.node_id, start, end))
            start = end
            node_idx += 1

        latency, throughput = _evaluate_partition(
            self._simulator, model, nodes, assignments, batch_size, seq_len,
        )

        mem_used: dict[str, float] = {}
        node_util: dict[str, float] = {}
        for node_id, s, e in assignments:
            layer_count = e - s
            weights_gb = model.params_per_layer * layer_count * 2 / (1024**3)
            kv_gb = (
                batch_size * seq_len * model.num_heads * model.head_dim
                * 2 * 2 * layer_count / (1024**3)
            )
            mem_used[node_id] = round(weights_gb + kv_gb, 4)
            node_util[node_id] = round(
                (weights_gb + kv_gb) / max(nodes[0].memory_gb, 0.001), 4
            )

        return PartitionSolution(
            assignments=assignments,
            estimated_latency_ms=round(latency, 2),
            estimated_throughput_tok_s=round(throughput, 2),
            memory_used_gb=mem_used,
            node_utilization=node_util,
            search_method="constrained-memory",
            num_candidates_evaluated=len(assignments),
        )

    # ── Utilities ────────────────────────────────────────────────────────

    def solution(self) -> PartitionSolution | None:
        """Return the last computed partition solution."""
        return self._last_solution

    def compare_methods(
        self,
        model: str | ModelConfig,
        nodes: list[NodeSpec],
        batch_size: int = 1,
        seq_len: int = 2048,
    ) -> dict[str, PartitionSolution]:
        """Compare all search methods side-by-side.

        Returns a dict keyed by method name with their PartitionSolution.
        """
        results: dict[str, PartitionSolution] = {}
        for method in ("brute-force", "beam-search", "heuristic"):
            try:
                sol = self.optimize(
                    model=model,
                    nodes=nodes,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    method=method,
                )
                results[method] = sol
            except Exception:
                continue
        return results

    def summary(self) -> str:
        """Human-readable summary of the last optimization."""
        if self._last_solution is None:
            return "TopologyOptimizer: no solution computed yet"
        sol = self._last_solution
        lines: list[str] = [
            f"TopologyOptimizer ({sol.search_method})",
            f"  Nodes: {len(sol.assignments)}",
            f"  Latency: {sol.estimated_latency_ms:.2f} ms",
            f"  Throughput: {sol.estimated_throughput_tok_s:.2f} tok/s",
            f"  Candidates evaluated: {sol.num_candidates_evaluated}",
            "  Assignments:",
        ]
        for node_id, start, end in sol.assignments:
            mem = sol.memory_used_gb.get(node_id, 0.0)
            util = sol.node_utilization.get(node_id, 0.0)
            lines.append(
                f"    {node_id}: layers [{start}, {end}) "
                f"({end - start} layers, "
                f"{mem:.2f} GB, {util:.1%} util)"
            )
        return "\n".join(lines)
