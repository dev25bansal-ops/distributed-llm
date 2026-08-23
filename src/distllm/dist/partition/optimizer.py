from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from distllm.dist.partition.cost_model import PartitionCostModel


@dataclass
class PartitionPoint:
    node_id: str
    start_layer: int
    end_layer: int
    estimated_time_ms: float = 0.0
    quant_method: str = "none"


@dataclass
class PartitionSolution:
    points: list[PartitionPoint] = field(default_factory=list)
    max_node_time_ms: float = 0.0
    total_time_ms: float = 0.0
    estimated_throughput_tok_s: float = 0.0
    pipeline_latency_ms: float = 0.0
    num_oom_nodes: int = 0
    explanation: str = ""
    quant_plan: Any = None  # Optional[QuantizationPlan]

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
            f"  Pipeline latency: {self.pipeline_latency_ms:.1f}ms",
            f"  Est. throughput: {self.estimated_throughput_tok_s:.0f} tok/s",
            f"  OOM nodes: {self.num_oom_nodes}",
        ]
        if self.explanation:
            lines.append(f"  Strategy: {self.explanation}")
        if self.quant_plan:
            lines.append(f"  Quantization: {self.quant_plan.strategy}")
        lines.append("  Assignments:")
        for p in self.points:
            q_str = f" [{p.quant_method}]" if p.quant_method != "none" else ""
            lines.append(
                f"    {p.node_id}: layers [{p.start_layer}, {p.end_layer}) "
                f"~{p.estimated_time_ms:.1f}ms{q_str}"
            )
        return "\n".join(lines)


class PartitionOptimizer:
    def __init__(
        self,
        cost_model: PartitionCostModel,
        node_ids: list[str],
        batch_size: int = 1,
        seq_len: int = 4096,
        allow_oom: bool = False,
        gpu_counts: dict[str, int] | None = None,
        min_layers_per_node: int = 1,
        quant_tuner: Any = None,
        node_infos: list[Any] | None = None,
        model_size_bytes: int = 0,
        inter_node_bandwidth_gbps: float | None = None,
    ):
        self._cost_model = cost_model
        self._node_ids = list(node_ids)
        self._batch_size = batch_size
        self._seq_len = seq_len
        self._allow_oom = allow_oom
        # M5: Multi-GPU per node support
        self._gpu_counts = gpu_counts or {nid: 1 for nid in node_ids}
        # 3.2: Constrained DP — minimum layers per node
        self._min_layers = max(1, min_layers_per_node)
        # APO integration
        self._quant_tuner = quant_tuner
        self._node_infos = node_infos or []
        self._model_size_bytes = model_size_bytes
        self._inter_node_bandwidth_gbps = inter_node_bandwidth_gbps

        self._num_layers: int = 0
        self._num_nodes: int = 0
        self._dp: list[list[float]] = []
        self._split: list[list[int]] = []
        self._quant_choices: dict[tuple[int, int, int], str] = {}

    def solve(self, num_layers: int) -> PartitionSolution:
        self._num_layers = num_layers
        self._num_nodes = len(self._node_ids)

        if self._num_nodes == 0:
            return PartitionSolution(explanation="No nodes available")

        if self._num_nodes == 1:
            return self._single_node_solution()

        # 3.2: Beam search fallback for very large models (>200 layers)
        if num_layers > 200 and self._num_nodes > 8:
            return self._beam_search_solve(num_layers)

        self._initialize_dp()
        self._run_dp()

        solution = self._backtrack()
        solution = self._enforce_boundary_layers(solution)
        solution = self._attach_quantization_plan(solution)
        return self._finalize(solution)

    def _initialize_dp(self) -> None:
        L = self._num_layers + 1
        N = min(self._num_nodes, L)

        INF = float("inf")
        self._dp = [[INF] * N for _ in range(L)]
        self._split = [[-1] * N for _ in range(L)]

        for i in range(1, L):
            cost = self._evaluate_node(0, 0, i)
            if cost is not None:
                self._dp[i][0] = cost
                self._split[i][0] = 0

    def _run_dp(self) -> None:
        L = self._num_layers + 1
        N = min(self._num_nodes, L)
        min_l = self._min_layers

        for k in range(1, N):
            for i in range(1, L):
                best = float("inf")
                best_j = -1

                # 3.2: Constrained DP — j must leave at least min_l layers
                # for previous nodes and min_l layers for current node
                min_j = max(1, k * min_l)
                max_j = i - min_l

                # 3.2: Pruned search — skip j range if infeasible
                if max_j < min_j:
                    self._dp[i][k] = best
                    self._split[i][k] = best_j
                    continue

                for j in range(min_j, max_j + 1):
                    prev_cost = self._dp[j][k - 1]
                    # M4: Use math.isinf for safe infinity checks
                    if math.isinf(prev_cost) or math.isnan(prev_cost):
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

        best_k = 0
        best_cost = float("inf")
        for k in range(N):
            if self._dp[L][k] < best_cost:
                best_cost = self._dp[L][k]
                best_k = k

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

        # M2: Pipeline latency with bubble overhead
        pipeline_lat = self._cost_model.pipeline_latency(
            [(p.node_id, p.start_layer, p.end_layer) for p in points],
            self._batch_size, self._seq_len,
        )

        return PartitionSolution(
            points=points,
            max_node_time_ms=round(max_time, 2),
            total_time_ms=round(sum(p.estimated_time_ms for p in points), 2),
            estimated_throughput_tok_s=round(throughput, 0),
            pipeline_latency_ms=round(pipeline_lat, 2),
            num_oom_nodes=oom_count,
            explanation=(
                f"DP minimax optimized, "
                f"{len(points)}/{self._num_nodes} nodes, "
                f"{oom_count} OOM-tolerant" if self._allow_oom and oom_count else
                f"DP minimax optimized, "
                f"{len(points)}/{self._num_nodes} nodes, "
                f"{'OOM on ' + str(oom_count) + ' nodes' if oom_count else 'no OOM'}"
            ),
        )

    def _enforce_boundary_layers(
        self, points: list[PartitionPoint]
    ) -> list[PartitionPoint]:
        if not points:
            return points

        if points[0].start_layer != 0:
            points[0] = PartitionPoint(
                node_id=points[0].node_id,
                start_layer=0,
                end_layer=points[0].end_layer,
                estimated_time_ms=points[0].estimated_time_ms,
            )

        last_expected = self._num_layers
        if points[-1].end_layer != last_expected:
            points[-1] = PartitionPoint(
                node_id=points[-1].node_id,
                start_layer=points[-1].start_layer,
                end_layer=last_expected,
                estimated_time_ms=points[-1].estimated_time_ms,
            )

        for pt in points:
            cost = self._cost_model.evaluate(
                pt.node_id, pt.start_layer, pt.end_layer,
                self._batch_size, self._seq_len,
            )
            pt.estimated_time_ms = cost.total_time_ms

        return points

    def _evaluate_node(
        self, node_idx: int, start_layer: int, end_layer: int
    ) -> float | None:
        if node_idx >= len(self._node_ids):
            return None

        node_id = self._node_ids[node_idx]

        # If quant tuner is available, try each candidate method and pick best
        if self._quant_tuner and self._node_infos:
            return self._evaluate_node_with_quant(node_idx, start_layer, end_layer)

        cost = self._cost_model.evaluate(
            node_id, start_layer, end_layer,
            self._batch_size, self._seq_len,
        )

        if not self._allow_oom and not cost.fits_in_memory:
            return None

        # M5: Multi-GPU speedup — divide compute time by GPU count
        gpu_count = self._gpu_counts.get(node_id, 1)
        total_ms = cost.total_time_ms
        if gpu_count > 1:
            # Communication overhead between GPUs on same node
            intra_node_comm_ms = cost.communication_time_ms * 0.1
            total_ms = (cost.compute_time_ms / gpu_count) + cost.communication_time_ms + intra_node_comm_ms

        return total_ms

    def _evaluate_node_with_quant(
        self, node_idx: int, start_layer: int, end_layer: int,
    ) -> float | None:
        """Evaluate node cost trying multiple quantization methods.

        Uses the APO tuner to generate a recommendation for this specific
        layer range, then evaluates cost with that quantization applied.
        Returns the best (lowest) total time across all viable methods.
        """
        from distllm.dist.partition.quantization_tuner import (
            NodeInfo, QuantizationAutoTuner, QuantMethod,
        )
        from distllm.dist.partition.quant_cost import QuantizationAwareCostModel

        if node_idx >= len(self._node_ids) or node_idx >= len(self._node_infos):
            return self._evaluate_node(node_idx, start_layer, end_layer)

        node_id = self._node_ids[node_idx]
        node_info = self._node_infos[node_idx]

        # Calculate layers assigned to this node
        layers_assigned = end_layer - start_layer

        # Create a node info with the correct layer assignment
        if isinstance(node_info, NodeInfo):
            ni = NodeInfo(
                node_id=node_info.node_id,
                device_type=node_info.device_type,
                total_memory_bytes=node_info.total_memory_bytes,
                compute_capability=node_info.compute_capability,
                gpu_name=node_info.gpu_name,
                bandwidth_gbps=node_info.bandwidth_gbps,
                num_layers_assigned=layers_assigned,
                is_hopper_or_newer=node_info.is_hopper_or_newer,
            )
        else:
            ni = NodeInfo.from_dict(node_info)
            ni.num_layers_assigned = layers_assigned

        # Get APO recommendation for this node + layer range
        plan = self._quant_tuner.recommend(
            [ni], self._model_size_bytes, self._num_layers,
            self._inter_node_bandwidth_gbps,
        )

        if not plan.recommendations:
            return self._evaluate_node(node_idx, start_layer, end_layer)

        rec = plan.recommendations[0]

        # Evaluate cost with this quantization
        quant_cost_model = QuantizationAwareCostModel(self._cost_model)
        qcost = quant_cost_model.evaluate_with_quant(
            node_id, start_layer, end_layer, rec,
            self._batch_size, self._seq_len,
        )

        if not self._allow_oom and not qcost.fits_in_memory:
            return None

        # Record the best quant choice for this (node, start, end)
        key = (node_idx, start_layer, end_layer)
        self._quant_choices[key] = rec.method.value

        gpu_count = self._gpu_counts.get(node_id, 1)
        total_ms = qcost.total_time_ms
        if gpu_count > 1:
            intra_node_comm_ms = qcost.communication_time_ms * 0.1
            total_ms = (qcost.compute_time_ms / gpu_count) + qcost.communication_time_ms + intra_node_comm_ms

        return total_ms

    def _attach_quantization_plan(
        self, solution: PartitionSolution,
    ) -> PartitionSolution:
        """Attach a QuantizationPlan to the solution based on DP choices."""
        if not self._quant_tuner or not self._node_infos:
            return solution

        from distllm.dist.partition.quantization_tuner import (
            NodeInfo, QuantizationAutoTuner, QuantizationPlan,
        )

        nodes_for_plan: list[NodeInfo] = []
        for pt in solution.points:
            node_idx = self._node_ids.index(pt.node_id) if pt.node_id in self._node_ids else 0
            key = (node_idx, pt.start_layer, pt.end_layer)
            quant_method = self._quant_choices.get(key, "none")
            pt.quant_method = quant_method

            # Build node info for the plan
            if node_idx < len(self._node_infos):
                ni_src = self._node_infos[node_idx]
                if isinstance(ni_src, NodeInfo):
                    ni = NodeInfo(
                        node_id=ni_src.node_id,
                        device_type=ni_src.device_type,
                        total_memory_bytes=ni_src.total_memory_bytes,
                        compute_capability=ni_src.compute_capability,
                        gpu_name=ni_src.gpu_name,
                        num_layers_assigned=pt.end_layer - pt.start_layer,
                        is_hopper_or_newer=ni_src.is_hopper_or_newer,
                    )
                else:
                    ni = NodeInfo.from_dict(ni_src)
                    ni.num_layers_assigned = pt.end_layer - pt.start_layer
                nodes_for_plan.append(ni)

        if nodes_for_plan:
            plan = self._quant_tuner.recommend(
                nodes_for_plan, self._model_size_bytes, self._num_layers,
                self._inter_node_bandwidth_gbps,
            )
            solution.quant_plan = plan

        return solution

    def _single_node_solution(self) -> PartitionSolution:
        node_id = self._node_ids[0]
        cost = self._cost_model.evaluate(
            node_id, 0, self._num_layers,
            self._batch_size, self._seq_len,
        )

        gpu_count = self._gpu_counts.get(node_id, 1)
        total_ms = cost.total_time_ms
        if gpu_count > 1:
            intra_node_comm_ms = cost.communication_time_ms * 0.1
            total_ms = (cost.compute_time_ms / gpu_count) + cost.communication_time_ms + intra_node_comm_ms

        point = PartitionPoint(
            node_id=node_id,
            start_layer=0,
            end_layer=self._num_layers,
            estimated_time_ms=total_ms,
        )

        throughput = (self._batch_size * self._seq_len) / max(total_ms / 1000, 1e-9)

        solution = PartitionSolution(
            points=[point],
            max_node_time_ms=round(total_ms, 2),
            total_time_ms=round(total_ms, 2),
            estimated_throughput_tok_s=round(throughput, 0),
            pipeline_latency_ms=round(total_ms, 2),
            num_oom_nodes=0 if cost.fits_in_memory else 1,
            explanation="Single node (no partitioning needed)",
        )

        solution = self._attach_quantization_plan(solution)
        return solution

    def compare_strategies(
        self, num_layers: int, batch_size: int = 1, seq_len: int = 4096,
    ) -> dict[str, Any]:
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
                "pipeline_latency_ms": dp_solution.pipeline_latency_ms,
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
            extra = 1 if i >= nodes - rem else 0
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

    def _beam_search_solve(
        self, num_layers: int, beam_width: int = 10,
    ) -> PartitionSolution:
        """3.2: Beam search fallback for very large models.

        O(L·N·K) instead of O(L²·N) where K is beam_width.
        Used when num_layers > 200 and num_nodes > 8.
        """
        N = self._num_nodes
        L = num_layers
        min_l = self._min_layers

        # State: (cost, node_idx, split_history)
        # split_history: list of (start_layer, end_layer) per node used so far
        initial_state = (0.0, -1, [])

        # For each (layers_assigned, nodes_used), keep top-K states
        states: dict[tuple[int, int], list[tuple[float, list[tuple[int, int]]]]] = {}
        states[(0, 0)] = [(0.0, [])]

        for layers_used in range(1, L + 1):
            for nodes_used in range(1, min(N, layers_used // min_l + 1) + 1):
                candidates: list[tuple[float, list[tuple[int, int]]]] = []

                # Try adding layer `layers_used` to the last node
                for prev_layers, prev_nodes in [
                    (layers_used - 1, nodes_used),
                ]:
                    if (prev_layers, prev_nodes) not in states:
                        continue
                    for cost, history in states[(prev_layers, prev_nodes)]:
                        if not history:
                            continue
                        # Extend the last node's range
                        new_history = history[:-1] + [(history[-1][0], layers_used)]
                        s, e = new_history[-1]
                        node_idx = len(new_history) - 1
                        node_cost = self._evaluate_node(node_idx, s, e)
                        if node_cost is not None:
                            new_cost = max(cost, node_cost)
                            candidates.append((new_cost, new_history))

                # Try starting a new node with this layer
                if nodes_used > 0:
                    for prev_layers, prev_nodes in [
                        (layers_used - 1, nodes_used - 1),
                    ]:
                        if (prev_layers, prev_nodes) not in states:
                            continue
                        for cost, history in states[(prev_layers, prev_nodes)]:
                            if len(history) >= N:
                                continue
                            new_history = history + [(layers_used - 1, layers_used)]
                            node_idx = len(new_history) - 1
                            node_cost = self._evaluate_node(node_idx, layers_used - 1, layers_used)
                            if node_cost is not None:
                                new_cost = max(cost, node_cost)
                                candidates.append((new_cost, new_history))

                # Keep top-K candidates
                candidates.sort(key=lambda x: x[0])
                states[(layers_used, nodes_used)] = candidates[:beam_width]

        # Find best final state
        best_cost = float("inf")
        best_history: list[tuple[int, int]] = []
        for (l, n), state_list in states.items():
            if l != L:
                continue
            for cost, history in state_list:
                if cost < best_cost:
                    best_cost = cost
                    best_history = history

        if not best_history:
            return PartitionSolution(explanation="Beam search found no valid partition")

        # Convert history to PartitionPoints
        points: list[PartitionPoint] = []
        for idx, (start, end) in enumerate(best_history):
            if idx < len(self._node_ids):
                node_id = self._node_ids[idx]
                cost = self._cost_model.evaluate(
                    node_id, start, end,
                    self._batch_size, self._seq_len,
                )
                points.append(PartitionPoint(
                    node_id=node_id,
                    start_layer=start,
                    end_layer=end,
                    estimated_time_ms=cost.total_time_ms,
                ))

        return self._finalize(points)
