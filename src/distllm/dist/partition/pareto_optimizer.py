"""Multi-objective Pareto DP optimizer for pipeline partitioning.

Extends the single-objective minimax DP to track a Pareto frontier
across multiple objectives: latency, throughput, memory utilization,
quantization quality loss, and financial cost.

Each DP cell stores a list of non-dominated (Pareto-optimal) vectors
instead of a single scalar.  Dominated solutions are pruned at each
step to keep the search tractable.

Typical usage::

    pareto = ParetoPartitionOptimizer(
        cost_model=cost_model,
        node_ids=["gpu-0", "gpu-1", "gpu-2"],
        objectives=["latency", "memory", "cost"],
    )
    frontier = pareto.solve(num_layers=80)
    best = pareto.select_solution(frontier, weights={"latency": 0.6, "memory": 0.2, "cost": 0.2})
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from loguru import logger

from distllm.dist.partition.cost_model import NodeCost, PartitionCostModel
from distllm.dist.partition.optimizer import PartitionPoint, PartitionSolution


class Objective(str, Enum):
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    MEMORY = "memory"
    QUALITY = "quality"
    COST = "cost"


@dataclass
class ObjectiveVector:
    """A point in multi-objective space."""
    latency_ms: float = 0.0
    throughput_tok_s: float = 0.0
    memory_utilization: float = 0.0
    quality_loss: float = 0.0
    cost_per_hour: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "latency_ms": self.latency_ms,
            "throughput_tok_s": self.throughput_tok_s,
            "memory_utilization": self.memory_utilization,
            "quality_loss": self.quality_loss,
            "cost_per_hour": self.cost_per_hour,
        }

    def get(self, objective: str) -> float:
        mapping = {
            "latency": self.latency_ms,
            "throughput": self.throughput_tok_s,
            "memory": self.memory_utilization,
            "quality": self.quality_loss,
            "cost": self.cost_per_hour,
        }
        return mapping.get(objective, 0.0)

    def dominates(self, other: ObjectiveVector, minimize: set[str] | None = None) -> bool:
        """Check if this vector dominates another (Pareto dominance).

        A dominates B if A is at least as good in all objectives and
        strictly better in at least one.
        """
        minimize = minimize or {"latency", "memory", "quality", "cost"}
        objectives = ["latency", "throughput", "memory", "quality", "cost"]

        at_least_as_good = True
        strictly_better = False

        for obj in objectives:
            val_self = self.get(obj)
            val_other = other.get(obj)

            if obj in minimize:
                if val_self > val_other:
                    at_least_as_good = False
                    break
                if val_self < val_other:
                    strictly_better = True
            else:
                if val_self < val_other:
                    at_least_as_good = False
                    break
                if val_self > val_other:
                    strictly_better = True

        return at_least_as_good and strictly_better


@dataclass
class ParetoSolution:
    """A partition solution with its multi-objective vector."""
    points: list[PartitionPoint]
    vector: ObjectiveVector
    assignments: list[tuple[str, int, int, str]] = field(default_factory=list)

    def to_partition_solution(self) -> PartitionSolution:
        return PartitionSolution(
            points=self.points,
            max_node_time_ms=round(self.vector.latency_ms, 2),
            total_time_ms=round(sum(p.estimated_time_ms for p in self.points), 2),
            estimated_throughput_tok_s=round(self.vector.throughput_tok_s, 0),
            num_oom_nodes=0,
            explanation=f"Pareto-optimal: latency={self.vector.latency_ms:.1f}ms, "
                        f"mem={self.vector.memory_utilization:.0%}, "
                        f"cost=${self.vector.cost_per_hour:.2f}/hr",
        )


@dataclass
class ParetoFrontier:
    """Set of non-dominated partition solutions."""
    solutions: list[ParetoSolution] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.solutions)

    def best_by(self, objective: str) -> ParetoSolution | None:
        """Return the solution that is best for a single objective."""
        if not self.solutions:
            return None
        minimize = {"latency", "memory", "quality", "cost"}
        if objective in minimize:
            return min(self.solutions, key=lambda s: s.vector.get(objective))
        return max(self.solutions, key=lambda s: s.vector.get(objective))

    def weighted_select(self, weights: dict[str, float]) -> ParetoSolution:
        """Select the best solution using weighted sum scalarization."""
        if not self.solutions:
            raise ValueError("Empty Pareto frontier")

        minimize = {"latency", "memory", "quality", "cost"}
        best_score = float("inf")
        best = self.solutions[0]

        for sol in self.solutions:
            score = 0.0
            for obj, weight in weights.items():
                val = sol.vector.get(obj)
                if obj in minimize:
                    score += weight * val
                else:
                    score -= weight * val
            if score < best_score:
                best_score = score
                best = sol

        return best

    def summary(self) -> str:
        lines = [f"Pareto frontier: {self.size} non-dominated solutions"]
        for i, sol in enumerate(self.solutions):
            lines.append(
                f"  [{i}] latency={sol.vector.latency_ms:.1f}ms, "
                f"throughput={sol.vector.throughput_tok_s:.0f}tok/s, "
                f"memory={sol.vector.memory_utilization:.0%}, "
                f"cost=${sol.vector.cost_per_hour:.2f}/hr"
            )
        return "\n".join(lines)


class ParetoPartitionOptimizer:
    """Multi-objective Pareto-optimal DP partition solver.

    Extends the minimax DP to maintain a Pareto frontier at each
    dp[i][k] cell.  Each cell stores a list of non-dominated
    (ObjectiveVector, split_point) tuples.

    Args:
        cost_model: Per-node cost estimator.
        node_ids: List of node identifiers.
        batch_size: Target batch size.
        seq_len: Target sequence length.
        allow_oom: If True, allow memory-exceeding partitions.
        node_costs_per_hour: Optional per-node hourly cost ($/hr).
        max_quality_loss: Max acceptable quality loss (0.0-1.0).
        frontier_limit: Max Pareto points per cell (prevents explosion).
    """

    def __init__(
        self,
        cost_model: PartitionCostModel,
        node_ids: list[str],
        batch_size: int = 1,
        seq_len: int = 4096,
        allow_oom: bool = False,
        node_costs_per_hour: dict[str, float] | None = None,
        max_quality_loss: float = 0.05,
        frontier_limit: int = 32,
    ):
        self._cost_model = cost_model
        self._node_ids = list(node_ids)
        self._batch_size = batch_size
        self._seq_len = seq_len
        self._allow_oom = allow_oom
        self._node_costs = node_costs_per_hour or {}
        self._max_quality_loss = max_quality_loss
        self._frontier_limit = frontier_limit

        self._num_layers: int = 0
        self._num_nodes: int = 0
        self._dp: list[list[list[tuple[ObjectiveVector, int]]]] = []

    def solve(self, num_layers: int) -> ParetoFrontier:
        """Run the Pareto DP and return the Pareto frontier."""
        self._num_layers = num_layers
        self._num_nodes = len(self._node_ids)

        if self._num_nodes == 0:
            return ParetoFrontier()

        if self._num_nodes == 1:
            return self._single_node_frontier()

        self._initialize_dp()
        self._run_dp()
        return self._build_frontier()

    def select_solution(
        self, frontier: ParetoFrontier, weights: dict[str, float],
    ) -> ParetoSolution:
        """Pick the best solution from the frontier using weighted scalarization."""
        return frontier.weighted_select(weights)

    def solve_and_select(
        self, num_layers: int, weights: dict[str, float],
    ) -> PartitionSolution:
        """Solve and return the best weighted solution as a PartitionSolution."""
        frontier = self.solve(num_layers)
        if frontier.size == 0:
            return PartitionSolution(explanation="No valid partition found")
        best = self.select_solution(frontier, weights)
        return best.to_partition_solution()

    def _initialize_dp(self) -> None:
        L = self._num_layers + 1
        N = min(self._num_nodes, L)
        self._dp = [[[] for _ in range(N)] for _ in range(L)]

        for i in range(1, L):
            vec = self._evaluate_node_vec(0, 0, i)
            if vec is not None:
                self._dp[i][0].append((vec, 0))

    def _run_dp(self) -> None:
        L = self._num_layers + 1
        N = min(self._num_nodes, L)

        for k in range(1, N):
            for i in range(1, L):
                candidates: list[tuple[ObjectiveVector, int]] = []

                for j in range(1, i):
                    prev_frontier = self._dp[j][k - 1]
                    if not prev_frontier:
                        continue

                    node_vec = self._evaluate_node_vec(k, j, i)
                    if node_vec is None:
                        continue

                    for prev_vec, prev_split in prev_frontier:
                        merged = self._merge_vectors(prev_vec, node_vec)
                        candidates.append((merged, j))

                self._dp[i][k] = self._prune_frontier(candidates)

    def _evaluate_node_vec(
        self, node_idx: int, start_layer: int, end_layer: int,
    ) -> ObjectiveVector | None:
        if node_idx >= len(self._node_ids):
            return None

        node_id = self._node_ids[node_idx]
        cost = self._cost_model.evaluate(
            node_id, start_layer, end_layer,
            self._batch_size, self._seq_len,
        )

        if not self._allow_oom and not cost.fits_in_memory:
            return None

        mem_util = cost.memory_utilization
        throughput = (self._batch_size * self._seq_len) / max(cost.total_time_ms / 1000, 1e-9)
        hourly_cost = self._node_costs.get(node_id, 0.0)

        return ObjectiveVector(
            latency_ms=cost.total_time_ms,
            throughput_tok_s=throughput,
            memory_utilization=mem_util,
            quality_loss=0.0,
            cost_per_hour=hourly_cost,
        )

    def _merge_vectors(
        self, prev: ObjectiveVector, node: ObjectiveVector,
    ) -> ObjectiveVector:
        """Merge vectors across pipeline stages.

        Latency: max across stages (pipeline bottleneck).
        Memory: max across stages (worst-case utilization).
        Cost: sum across stages (total hourly cost).
        Throughput: min across stages (bottleneck).
        Quality: max across stages.
        """
        return ObjectiveVector(
            latency_ms=max(prev.latency_ms, node.latency_ms),
            throughput_tok_s=min(prev.throughput_tok_s, node.throughput_tok_s) if node.throughput_tok_s > 0 else prev.throughput_tok_s,
            memory_utilization=max(prev.memory_utilization, node.memory_utilization),
            quality_loss=max(prev.quality_loss, node.quality_loss),
            cost_per_hour=prev.cost_per_hour + node.cost_per_hour,
        )

    def _prune_frontier(
        self, candidates: list[tuple[ObjectiveVector, int]],
    ) -> list[tuple[ObjectiveVector, int]]:
        """Remove dominated solutions and limit frontier size."""
        if not candidates:
            return []

        non_dominated: list[tuple[ObjectiveVector, int]] = []

        for vec, split in candidates:
            dominated = False
            to_remove: list[int] = []

            for idx, (existing, _) in enumerate(non_dominated):
                if existing.dominates(vec):
                    dominated = True
                    break
                if vec.dominates(existing):
                    to_remove.append(idx)

            if not dominated:
                for idx in sorted(to_remove, reverse=True):
                    non_dominated.pop(idx)
                non_dominated.append((vec, split))

        if len(non_dominated) > self._frontier_limit:
            non_dominated.sort(key=lambda x: x[0].latency_ms)
            non_dominated = non_dominated[: self._frontier_limit]

        return non_dominated

    def _build_frontier(self) -> ParetoFrontier:
        L = self._num_layers
        N = min(self._num_nodes, L)

        all_solutions: list[ParetoSolution] = []

        for k in range(N):
            for vec, _split in self._dp[L][k]:
                points = self._backtrack_from(k, vec)
                if points:
                    all_solutions.append(ParetoSolution(
                        points=points,
                        vector=vec,
                        assignments=[
                            (p.node_id, p.start_layer, p.end_layer, "")
                            for p in points
                        ],
                    ))

        global_frontier = self._prune_solution_frontier(all_solutions)
        return ParetoFrontier(solutions=global_frontier)

    def _backtrack_from(self, k: int, target_vec: ObjectiveVector) -> list[PartitionPoint]:
        L = self._num_layers
        points: list[PartitionPoint] = []
        end = L
        current_k = k

        while current_k >= 0 and end > 0:
            best_split = -1
            best_prev_k = current_k - 1

            for vec, split in self._dp[end][current_k]:
                if abs(vec.latency_ms - target_vec.latency_ms) < max(target_vec.latency_ms * 0.01, 0.01):
                    best_split = split
                    break

            if best_split < 0:
                if self._dp[end][current_k]:
                    best_split = self._dp[end][current_k][0][1]
                else:
                    break

            node_id = self._node_ids[current_k] if current_k < len(self._node_ids) else f"node-{current_k}"
            cost = self._cost_model.evaluate(
                node_id, best_split, end,
                self._batch_size, self._seq_len,
            )
            points.append(PartitionPoint(
                node_id=node_id,
                start_layer=best_split,
                end_layer=end,
                estimated_time_ms=cost.total_time_ms,
            ))
            end = best_split
            current_k -= 1

        points.reverse()
        if points and points[0].start_layer != 0:
            points[0] = PartitionPoint(
                node_id=points[0].node_id,
                start_layer=0,
                end_layer=points[0].end_layer,
                estimated_time_ms=points[0].estimated_time_ms,
            )
        return points

    def _prune_solution_frontier(
        self, solutions: list[ParetoSolution],
    ) -> list[ParetoSolution]:
        non_dominated: list[ParetoSolution] = []
        for sol in solutions:
            dominated = False
            to_remove: list[int] = []
            for idx, existing in enumerate(non_dominated):
                if existing.vector.dominates(sol.vector):
                    dominated = True
                    break
                if sol.vector.dominates(existing.vector):
                    to_remove.append(idx)
            if not dominated:
                for idx in sorted(to_remove, reverse=True):
                    non_dominated.pop(idx)
                non_dominated.append(sol)
        return non_dominated

    def _single_node_frontier(self) -> ParetoFrontier:
        node_id = self._node_ids[0]
        cost = self._cost_model.evaluate(
            node_id, 0, self._num_layers,
            self._batch_size, self._seq_len,
        )
        throughput = (self._batch_size * self._seq_len) / max(cost.total_time_ms / 1000, 1e-9)
        vec = ObjectiveVector(
            latency_ms=cost.total_time_ms,
            throughput_tok_s=throughput,
            memory_utilization=cost.memory_utilization,
            quality_loss=0.0,
            cost_per_hour=self._node_costs.get(node_id, 0.0),
        )
        point = PartitionPoint(
            node_id=node_id,
            start_layer=0,
            end_layer=self._num_layers,
            estimated_time_ms=cost.total_time_ms,
        )
        return ParetoFrontier(solutions=[
            ParetoSolution(points=[point], vector=vec)
        ])
