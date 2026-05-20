from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


from distllm.core.auto_partition.cost_model import PartitionCostModel


@dataclass
class PartitionPoint:
    """A single partition boundary in the solution."""
    node_id: str
    start_layer: int
    end_layer: int
    estimated_time_ms: float = 0.0


@dataclass
class PartitionSolution:
    """Complete partition solution covering all layers across nodes."""
    points: list[PartitionPoint] = field(default_factory=list)
    max_node_time_ms: float = 0.0
    total_time_ms: float = 0.0
    estimated_throughput_tok_s: float = 0.0
    num_oom_nodes: int = 0
    explanation: str = ""

    @property
    def num_nodes(self) -> int:
        return len(self.points)

    @property
    def coverage(self) -> tuple[int, int]:
        if not self.points:
            return (0, 0)
        return (
            self.points[0].start_layer,
            self.points[-1].end_layer,
        )

    def summary(self) -> str:
        lines = [
            f"PartitionSolution: {self.num_nodes} nodes, "
            f"{self.coverage[1] - self.coverage[0]} layers",
            f"  Max node time: {self.max_node_time_ms:.1f}ms",
            f"  Est. throughput: {self.estimated_throughput_tok_s:.0f} tok/s",
            f"  OOM nodes: {self.num_oom_nodes}",
        ]
        if self.explanation:
            lines.append(f"  Strategy: {self.explanation}")
        lines.append("  Assignments:")
        for p in self.points:
            lines.append(
                f"    {p.node_id}: layers [{p.start_layer}, {p.end_layer}) "
                f"~{p.estimated_time_ms:.1f}ms"
            )
        return "\n".join(lines)


class PartitionOptimizer:
    """Finds the optimal layer partition using dynamic programming.

    Solves the minimax optimization: minimize the maximum per-node
    latency subject to memory constraints.

    State: dp[i][k] = (min_max_latency, split_point)
        i = number of layers covered
        k = number of nodes used
        split_point = where the k-th node's partition starts

    Complexity: O(L^2 * N) where L = layers, N = nodes.
    Typical: 80^2 * 32 = ~200K evaluations — very fast.

    Usage:
        optimizer = PartitionOptimizer(cost_model, node_ids)
        solution = optimizer.solve(num_layers=32)
    """

    def __init__(
        self,
        cost_model: PartitionCostModel,
        node_ids: list[str],
        batch_size: int = 1,
        seq_len: int = 4096,
        allow_oom: bool = False,
    ):
        self._cost_model = cost_model
        self._node_ids = list(node_ids)
        self._batch_size = batch_size
        self._seq_len = seq_len
        self._allow_oom = allow_oom

        # DP tables
        self._num_layers: int = 0
        self._num_nodes: int = 0
        self._dp: list[list[float]] = []  # dp[i][k]
        self._split: list[list[int]] = []  # split[i][k]

    def solve(self, num_layers: int) -> PartitionSolution:
        self._num_layers = num_layers
        self._num_nodes = len(self._node_ids)

        if self._num_nodes == 0:
            return PartitionSolution(explanation="No nodes available")

        if self._num_nodes == 1:
            return self._single_node_solution()

        self._initialize_dp()
        self._run_dp()

        solution = self._backtrack()
        solution = self._enforce_boundary_layers(solution)
        return self._finalize(solution)

    # ------------------------------------------------------------------
    # DP algorithm
    # ------------------------------------------------------------------

    def _initialize_dp(self) -> None:
        L = self._num_layers + 1
        N = min(self._num_nodes, L)

        INF = float("inf")
        self._dp = [[INF] * N for _ in range(L)]
        self._split = [[-1] * N for _ in range(L)]

        # Base: dp[i][0] = cost of putting layers [0, i) on node 0
        for i in range(1, L):
            cost = self._evaluate_node(0, 0, i)
            if cost is not None:
                self._dp[i][0] = cost
                self._split[i][0] = 0

    def _run_dp(self) -> None:
        L = self._num_layers + 1
        N = min(self._num_nodes, L)

        for k in range(1, N):
            for i in range(1, L):
                best = float("inf")
                best_j = -1

                for j in range(1, i):  # split point
                    prev_cost = self._dp[j][k - 1]
                    if prev_cost >= float("inf"):
                        continue

                    node_cost = self._evaluate_node(k, j, i)
                    if node_cost is None:
                        continue

                    max_cost = max(prev_cost, node_cost)
                    if max_cost < best:
                        best = max_cost
                        best_j = j

                self._dp[i][k] = best
                self._split[i][k] = best_j

    def _backtrack(self) -> list[PartitionPoint]:
        L = self._num_layers
        N = min(self._num_nodes, L)

        # Find best k (number of nodes used)
        best_k = 0
        best_cost = float("inf")
        for k in range(N):
            if self._dp[L][k] < best_cost:
                best_cost = self._dp[L][k]
                best_k = k

        # Backtrack
        points: list[PartitionPoint] = []
        end = L
        k = best_k

        while k >= 0 and end > 0:
            start = self._split[end][k]
            if start < 0:
                break

            node_id = self._node_ids[k]
            cost_result = self._cost_model.evaluate(
                node_id, start, end,
                self._batch_size, self._seq_len,
            )

            points.append(PartitionPoint(
                node_id=node_id,
                start_layer=start,
                end_layer=end,
                estimated_time_ms=cost_result.total_time_ms,
            ))

            end = start
            k -= 1

        points.reverse()
        return points

    def _finalize(
        self, points: list[PartitionPoint]
    ) -> PartitionSolution:
        if not points:
            return PartitionSolution(explanation="No valid partition found")

        max_time = 0.0
        total_mem = 0
        oom_count = 0

        for pt in points:
            cost = self._cost_model.evaluate(
                pt.node_id, pt.start_layer, pt.end_layer,
                self._batch_size, self._seq_len,
            )
            pt.estimated_time_ms = cost.total_time_ms
            max_time = max(max_time, cost.total_time_ms)
            total_mem += cost.memory_bytes
            if not cost.fits_in_memory:
                oom_count += 1

        throughput = self._cost_model.combined_throughput(
            [(p.node_id, p.start_layer, p.end_layer) for p in points],
            self._batch_size, self._seq_len,
        )

        return PartitionSolution(
            points=points,
            max_node_time_ms=round(max_time, 2),
            total_time_ms=round(sum(p.estimated_time_ms for p in points), 2),
            estimated_throughput_tok_s=round(throughput, 0),
            num_oom_nodes=oom_count,
            explanation=(
                f"DP minimax optimized, "
                f"{len(points)}/total nodes, "
                f"{'OOM on ' + str(oom_count) + ' nodes' if oom_count else 'no OOM'}"
            ),
        )

    def _enforce_boundary_layers(
        self, points: list[PartitionPoint]
    ) -> list[PartitionPoint]:
        """Ensure embed (layer 0) is on first node, lm_head on last.

        The DP assigns contiguous ranges in node order, so embed is
        naturally on node 0. But if the solution leaves unused nodes
        at the start or end, we merge them back.
        """
        if not points:
            return points

        # First node must start at layer 0
        if points[0].start_layer != 0:
            points[0] = PartitionPoint(
                node_id=points[0].node_id,
                start_layer=0,
                end_layer=points[0].end_layer,
                estimated_time_ms=points[0].estimated_time_ms,
            )

        # Last node must cover the final layer
        last_expected = self._num_layers
        if points[-1].end_layer != last_expected:
            points[-1] = PartitionPoint(
                node_id=points[-1].node_id,
                start_layer=points[-1].start_layer,
                end_layer=last_expected,
                estimated_time_ms=points[-1].estimated_time_ms,
            )

        # Re-evaluate costs for adjusted boundaries
        for pt in points:
            cost = self._cost_model.evaluate(
                pt.node_id, pt.start_layer, pt.end_layer,
                self._batch_size, self._seq_len,
            )
            pt.estimated_time_ms = cost.total_time_ms

        return points

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _evaluate_node(
        self, node_idx: int, start_layer: int, end_layer: int
    ) -> float | None:
        if node_idx >= len(self._node_ids):
            return None

        node_id = self._node_ids[node_idx]
        cost = self._cost_model.evaluate(
            node_id, start_layer, end_layer,
            self._batch_size, self._seq_len,
        )

        if not self._allow_oom and not cost.fits_in_memory:
            return None

        return cost.total_time_ms

    def _single_node_solution(self) -> PartitionSolution:
        node_id = self._node_ids[0]
        cost = self._cost_model.evaluate(
            node_id, 0, self._num_layers,
            self._batch_size, self._seq_len,
        )
        point = PartitionPoint(
            node_id=node_id,
            start_layer=0,
            end_layer=self._num_layers,
            estimated_time_ms=cost.total_time_ms,
        )
        return PartitionSolution(
            points=[point],
            max_node_time_ms=cost.total_time_ms,
            total_time_ms=cost.total_time_ms,
            estimated_throughput_tok_s=self._cost_model.combined_throughput(
                [(node_id, 0, self._num_layers)],
                self._batch_size, self._seq_len,
            ),
            num_oom_nodes=0 if cost.fits_in_memory else 1,
            explanation="Single node (no partitioning needed)",
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def compare_strategies(
        self, num_layers: int, batch_size: int = 1, seq_len: int = 4096,
    ) -> dict[str, Any]:
        """Compare DP solution vs equal-split and proportional-split."""
        dp_solution = self.solve(num_layers)

        equal = self._equal_split(num_layers)
        eq_costs = self._cost_model.evaluate_partition(
            equal, batch_size, seq_len,
        )
        eq_max = max(c.total_time_ms for c in eq_costs)

        prop = self._proportional_split(num_layers)
        pr_costs = self._cost_model.evaluate_partition(
            prop, batch_size, seq_len,
        )
        pr_max = max(c.total_time_ms for c in pr_costs)

        return {
            "dp_minimax": {
                "max_latency_ms": dp_solution.max_node_time_ms,
                "throughput": dp_solution.estimated_throughput_tok_s,
            },
            "equal_split": {
                "max_latency_ms": round(eq_max, 2),
                "throughput": round(self._cost_model.combined_throughput(
                    equal, batch_size, seq_len,
                ), 0),
            },
            "proportional_split": {
                "max_latency_ms": round(pr_max, 2),
                "throughput": round(self._cost_model.combined_throughput(
                    prop, batch_size, seq_len,
                ), 0),
            },
            "improvement_over_equal": (
                f"{(1 - dp_solution.max_node_time_ms / max(eq_max, 0.001)) * 100:.1f}%"
                if eq_max > 0 else "N/A"
            ),
        }

    def _equal_split(self, num_layers: int) -> list[tuple[str, int, int]]:
        nodes = len(self._node_ids)
        per = num_layers // nodes
        rem = num_layers % nodes
        result: list[tuple[str, int, int]] = []
        start = 0
        for i, node_id in enumerate(self._node_ids):
            extra = 1 if i < rem else 0
            end = start + per + extra
            result.append((node_id, start, end))
            start = end
        return result

    def _proportional_split(
        self, num_layers: int
    ) -> list[tuple[str, int, int]]:
        capacities: list[float] = []
        for node_id in self._node_ids:
            cost = self._cost_model.evaluate(node_id, 0, num_layers)
            cap = 1.0 / max(cost.total_time_ms, 0.001)
            capacities.append(cap)

        total_cap = sum(capacities)
        result: list[tuple[str, int, int]] = []
        start = 0
        for i, node_id in enumerate(self._node_ids):
            share = capacities[i] / total_cap
            count = max(1, int(round(num_layers * share)))
            if i == len(self._node_ids) - 1:
                count = num_layers - start
            else:
                count = min(count, num_layers - start)
            end = start + count
            result.append((node_id, start, end))
            start = end
        return result
