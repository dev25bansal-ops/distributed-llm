"""Quantization-aware DP partition solver (joint optimization).

Searches over (layer_split, quantization_method) jointly in a single
DP pass, enabling optimal decisions that balance memory, speed, and
quality across heterogeneous nodes.

Key insight: a node with low VRAM but high compute should get fewer
layers at fp16, while a node with high VRAM but slow compute should
get more layers at 4-bit.  A decoupled approach can never discover this.

Typical usage::

    solver = QuantAwarePartitionSolver(
        cost_model=cost_model,
        node_ids=["gpu-0", "gpu-1"],
        device_types={"gpu-0": "cuda", "gpu-1": "cuda"},
    )
    solution = solver.solve(num_layers=80)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from distllm.dist.partition.cost_model import NodeCost, PartitionCostModel
from distllm.dist.partition.optimizer import PartitionPoint, PartitionSolution
from distllm.dist.partition.quantization_tuner import (
    QUANT_PROFILES,
    QuantMethod,
    QuantProfile,
)


@dataclass
class QuantizedNodeCost:
    """Cost of running a layer range on a node with a specific quantization."""
    node_id: str
    start_layer: int
    end_layer: int
    quant_method: QuantMethod
    compute_time_ms: float = 0.0
    communication_time_ms: float = 0.0
    total_time_ms: float = 0.0
    memory_bytes: int = 0
    memory_available_bytes: int = 0
    fits_in_memory: bool = True
    quality_loss: float = 0.0
    speed_penalty: float = 1.0
    memory_reduction: float = 1.0


@dataclass
class QuantizedPartitionPoint:
    """A partition assignment with quantization method."""
    node_id: str
    start_layer: int
    end_layer: int
    quant_method: QuantMethod
    estimated_time_ms: float = 0.0
    quality_loss: float = 0.0
    memory_bytes: int = 0


@dataclass
class QuantAwareSolution:
    """Solution from the quantization-aware DP solver."""
    points: list[QuantizedPartitionPoint] = field(default_factory=list)
    max_node_time_ms: float = 0.0
    total_time_ms: float = 0.0
    estimated_throughput_tok_s: float = 0.0
    avg_quality_loss: float = 0.0
    total_memory_bytes: int = 0
    explanation: str = ""

    @property
    def num_nodes(self) -> int:
        return len(self.points)

    @property
    def coverage(self) -> tuple[int, int]:
        if not self.points:
            return (0, 0)
        return (self.points[0].start_layer, self.points[-1].end_layer)

    def quant_methods_used(self) -> set[QuantMethod]:
        return {p.quant_method for p in self.points}

    def to_partition_solution(self) -> PartitionSolution:
        return PartitionSolution(
            points=[
                PartitionPoint(
                    node_id=p.node_id,
                    start_layer=p.start_layer,
                    end_layer=p.end_layer,
                    estimated_time_ms=p.estimated_time_ms,
                )
                for p in self.points
            ],
            max_node_time_ms=self.max_node_time_ms,
            total_time_ms=self.total_time_ms,
            estimated_throughput_tok_s=self.estimated_throughput_tok_s,
            explanation=self.explanation,
        )

    def summary(self) -> str:
        lines = [
            f"QuantAwareSolution: {self.num_nodes} nodes, "
            f"{self.coverage[1] - self.coverage[0]} layers",
            f"  Max node time: {self.max_node_time_ms:.1f}ms",
            f"  Throughput: {self.estimated_throughput_tok_s:.0f} tok/s",
            f"  Avg quality loss: {self.avg_quality_loss:.3f}",
            f"  Methods: {', '.join(m.value for m in self.quant_methods_used())}",
        ]
        for p in self.points:
            lines.append(
                f"    {p.node_id} [{p.quant_method.value}]: "
                f"layers [{p.start_layer}, {p.end_layer}) "
                f"~{p.estimated_time_ms:.1f}ms "
                f"quality_loss={p.quality_loss:.3f}"
            )
        return "\n".join(lines)


class QuantAwarePartitionSolver:
    """DP solver that jointly optimizes layer assignment and quantization.

    DP state: dp[i][k] = best cost for assigning first i layers to k
    nodes, where each cell stores the best across all quantization
    methods for that node.

    Cost function: weighted combination of latency and quality loss,
    subject to memory constraints.

    Args:
        cost_model: Analytical cost estimator.
        node_ids: List of node identifiers.
        device_types: Per-node device type (cuda, rocm, mps, etc.).
        batch_size: Target batch size.
        seq_len: Target sequence length.
        allow_oom: Allow memory-exceeding partitions.
        max_quality_loss: Max acceptable quality loss per node.
        quality_weight: Weight of quality loss in the cost function.
        require_calibration: If True, skip methods requiring calibration.
    """

    def __init__(
        self,
        cost_model: PartitionCostModel,
        node_ids: list[str],
        device_types: dict[str, str] | None = None,
        batch_size: int = 1,
        seq_len: int = 4096,
        allow_oom: bool = False,
        max_quality_loss: float = 0.05,
        quality_weight: float = 100.0,
        require_calibration: bool = False,
    ):
        self._cost_model = cost_model
        self._node_ids = list(node_ids)
        self._device_types = device_types or {nid: "cuda" for nid in node_ids}
        self._batch_size = batch_size
        self._seq_len = seq_len
        self._allow_oom = allow_oom
        self._max_quality_loss = max_quality_loss
        self._quality_weight = quality_weight
        self._require_calibration = require_calibration

        self._num_layers: int = 0
        self._num_nodes: int = 0
        self._dp: list[list[float]] = []
        self._split: list[list[int]] = []
        self._quant: list[list[QuantMethod]] = []

    def solve(self, num_layers: int) -> QuantAwareSolution:
        self._num_layers = num_layers
        self._num_nodes = len(self._node_ids)

        if self._num_nodes == 0:
            return QuantAwareSolution(explanation="No nodes available")

        if self._num_nodes == 1:
            return self._single_node_solution()

        self._initialize_dp()
        self._run_dp()
        return self._backtrack_and_finalize()

    def _get_valid_methods(self, node_id: str) -> list[QuantMethod]:
        device = self._device_types.get(node_id, "cuda")
        methods: list[QuantMethod] = []
        for method, profile in QUANT_PROFILES.items():
            if device not in profile.supported_hardware:
                continue
            if profile.quality_loss > self._max_quality_loss:
                continue
            if self._require_calibration and profile.requires_calibration:
                continue
            methods.append(method)
        return methods

    def _evaluate_with_quant(
        self, node_idx: int, start: int, end: int, method: QuantMethod,
    ) -> QuantizedNodeCost | None:
        if node_idx >= len(self._node_ids):
            return None

        node_id = self._node_ids[node_idx]
        profile = QUANT_PROFILES[method]

        base_cost = self._cost_model.evaluate(
            node_id, start, end, self._batch_size, self._seq_len,
        )

        compute_ms = base_cost.compute_time_ms * profile.speed_penalty
        comm_ms = base_cost.communication_time_ms
        total_ms = compute_ms + comm_ms

        memory_bytes = int(base_cost.memory_bytes * profile.memory_reduction)
        memory_available = base_cost.memory_available_bytes
        fits = memory_bytes <= memory_available * 0.9

        if not self._allow_oom and not fits:
            return None

        return QuantizedNodeCost(
            node_id=node_id,
            start_layer=start,
            end_layer=end,
            quant_method=method,
            compute_time_ms=round(compute_ms, 2),
            communication_time_ms=round(comm_ms, 2),
            total_time_ms=round(total_ms, 2),
            memory_bytes=memory_bytes,
            memory_available_bytes=memory_available,
            fits_in_memory=fits,
            quality_loss=profile.quality_loss,
            speed_penalty=profile.speed_penalty,
            memory_reduction=profile.memory_reduction,
        )

    def _cost_with_quality(self, node_cost: QuantizedNodeCost) -> float:
        """Weighted cost combining latency and quality loss."""
        return node_cost.total_time_ms + self._quality_weight * node_cost.quality_loss

    def _initialize_dp(self) -> None:
        L = self._num_layers + 1
        N = min(self._num_nodes, L)
        INF = float("inf")

        self._dp = [[INF] * N for _ in range(L)]
        self._split = [[-1] * N for _ in range(L)]
        self._quant = [[QuantMethod.NONE] * N for _ in range(L)]

        methods = self._get_valid_methods(self._node_ids[0])
        for i in range(1, L):
            best_cost = INF
            best_method = QuantMethod.NONE
            for method in methods:
                nc = self._evaluate_with_quant(0, 0, i, method)
                if nc is None:
                    continue
                cost = self._cost_with_quality(nc)
                if cost < best_cost:
                    best_cost = cost
                    best_method = method
            if best_cost < INF:
                self._dp[i][0] = best_cost
                self._split[i][0] = 0
                self._quant[i][0] = best_method

    def _run_dp(self) -> None:
        L = self._num_layers + 1
        N = min(self._num_nodes, L)
        INF = float("inf")

        for k in range(1, N):
            methods = self._get_valid_methods(self._node_ids[k])
            for i in range(1, L):
                best = INF
                best_j = -1
                best_q = QuantMethod.NONE

                for j in range(1, i):
                    prev_cost = self._dp[j][k - 1]
                    if prev_cost >= INF:
                        continue

                    for method in methods:
                        nc = self._evaluate_with_quant(k, j, i, method)
                        if nc is None:
                            continue

                        cost = self._cost_with_quality(nc)
                        max_cost = max(prev_cost, cost)
                        if max_cost < best:
                            best = max_cost
                            best_j = j
                            best_q = method

                self._dp[i][k] = best
                self._split[i][k] = best_j
                self._quant[i][k] = best_q

    def _backtrack_and_finalize(self) -> QuantAwareSolution:
        L = self._num_layers
        N = min(self._num_nodes, L)
        INF = float("inf")

        best_k = 0
        best_cost = INF
        for k in range(N):
            if self._dp[L][k] < best_cost:
                best_cost = self._dp[L][k]
                best_k = k

        if best_cost >= INF:
            return QuantAwareSolution(explanation="No valid partition found")

        points: list[QuantizedPartitionPoint] = []
        end = L
        k = best_k
        total_quality = 0.0
        max_time = 0.0
        total_mem = 0

        while k >= 0 and end > 0:
            start = self._split[end][k]
            if start < 0:
                break

            method = self._quant[end][k]
            nc = self._evaluate_with_quant(k, start, end, method)
            if nc is None:
                break

            points.append(QuantizedPartitionPoint(
                node_id=self._node_ids[k],
                start_layer=start,
                end_layer=end,
                quant_method=method,
                estimated_time_ms=nc.total_time_ms,
                quality_loss=nc.quality_loss,
                memory_bytes=nc.memory_bytes,
            ))
            total_quality += nc.quality_loss
            max_time = max(max_time, nc.total_time_ms)
            total_mem += nc.memory_bytes

            end = start
            k -= 1

        points.reverse()

        if points and points[0].start_layer != 0:
            points[0] = QuantizedPartitionPoint(
                node_id=points[0].node_id,
                start_layer=0,
                end_layer=points[0].end_layer,
                quant_method=points[0].quant_method,
                estimated_time_ms=points[0].estimated_time_ms,
                quality_loss=points[0].quality_loss,
                memory_bytes=points[0].memory_bytes,
            )

        throughput = (self._batch_size * self._seq_len) / max(max_time / 1000, 1e-9)

        return QuantAwareSolution(
            points=points,
            max_node_time_ms=round(max_time, 2),
            total_time_ms=round(sum(p.estimated_time_ms for p in points), 2),
            estimated_throughput_tok_s=round(throughput, 0),
            avg_quality_loss=round(total_quality / max(len(points), 1), 4),
            total_memory_bytes=total_mem,
            explanation=(
                f"Quantization-aware DP, "
                f"{len(points)} nodes, "
                f"methods: {', '.join(m.value for m in {p.quant_method for p in points})}, "
                f"avg quality loss: {total_quality / max(len(points), 1):.3f}"
            ),
        )

    def _single_node_solution(self) -> QuantAwareSolution:
        node_id = self._node_ids[0]
        methods = self._get_valid_methods(node_id)

        best_cost = float("inf")
        best_method = QuantMethod.NONE
        best_nc: QuantizedNodeCost | None = None

        for method in methods:
            nc = self._evaluate_with_quant(0, 0, self._num_layers, method)
            if nc is None:
                continue
            cost = self._cost_with_quality(nc)
            if cost < best_cost:
                best_cost = cost
                best_method = method
                best_nc = nc

        if best_nc is None:
            return QuantAwareSolution(explanation="No valid quantization found")

        point = QuantizedPartitionPoint(
            node_id=node_id,
            start_layer=0,
            end_layer=self._num_layers,
            quant_method=best_method,
            estimated_time_ms=best_nc.total_time_ms,
            quality_loss=best_nc.quality_loss,
            memory_bytes=best_nc.memory_bytes,
        )

        return QuantAwareSolution(
            points=[point],
            max_node_time_ms=best_nc.total_time_ms,
            total_time_ms=best_nc.total_time_ms,
            estimated_throughput_tok_s=round(
                (self._batch_size * self._seq_len) / max(best_nc.total_time_ms / 1000, 1e-9), 0
            ),
            avg_quality_loss=best_nc.quality_loss,
            total_memory_bytes=best_nc.memory_bytes,
            explanation=f"Single node, {best_method.value}",
        )
