"""Synthetic stress-testing harness for partition validation.

Simulates pipeline execution to validate partition quality without
running real models.  Detects bubbles, straggler amplification,
memory hotspots, and generates what-if scenarios.

Typical usage::

    validator = PartitionValidator(cost_model)
    report = validator.validate(solution, num_layers=80)
    print(report.summary())

    what_if = validator.what_if_slowdown(solution, "node-1", slowdown_pct=20)
    print(what_if.summary())
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

from distllm.dist.partition.cost_model import NodeCost, PartitionCostModel
from distllm.dist.partition.optimizer import PartitionPoint, PartitionSolution


@dataclass
class StageTiming:
    """Simulated timing for a single pipeline stage."""
    node_id: str
    start_layer: int
    end_layer: int
    compute_ms: float
    comm_in_ms: float
    comm_out_ms: float
    total_ms: float
    memory_bytes: int
    memory_available_bytes: int
    memory_utilization: float
    is_bottleneck: bool = False
    is_oom: bool = False


@dataclass
class PipelineSimulation:
    """Full pipeline simulation result."""
    stages: list[StageTiming]
    total_pipeline_time_ms: float
    bubble_time_ms: float
    bubble_pct: float
    throughput_tok_s: float
    bottleneck_node: str
    bottleneck_ms: float
    memory_hotspots: list[str]
    straggler_amplification: float
    confidence_interval: tuple[float, float]

    def summary(self) -> str:
        lines = [
            f"Pipeline Simulation: {len(self.stages)} stages",
            f"  Total pipeline time: {self.total_pipeline_time_ms:.1f}ms",
            f"  Bubble time: {self.bubble_time_ms:.1f}ms ({self.bubble_pct:.1f}%)",
            f"  Throughput: {self.throughput_tok_s:.0f} tok/s",
            f"  Bottleneck: {self.bottleneck_node} ({self.bottleneck_ms:.1f}ms)",
            f"  Straggler amplification: {self.straggler_amplification:.2f}x",
            f"  Throughput CI: [{self.confidence_interval[0]:.0f}, {self.confidence_interval[1]:.0f}] tok/s",
        ]
        if self.memory_hotspots:
            lines.append(f"  Memory hotspots: {', '.join(self.memory_hotspots)}")
        lines.append("  Stages:")
        for s in self.stages:
            marker = " *** BOTTLENECK" if s.is_bottleneck else ""
            oom_marker = " [OOM!]" if s.is_oom else ""
            lines.append(
                f"    {s.node_id}: [{s.start_layer},{s.end_layer}) "
                f"compute={s.compute_ms:.1f}ms comm_in={s.comm_in_ms:.1f}ms "
                f"total={s.total_ms:.1f}ms mem={s.memory_utilization:.0%}{marker}{oom_marker}"
            )
        return "\n".join(lines)


@dataclass
class WhatIfScenario:
    """Result of a what-if scenario."""
    scenario: str
    original_throughput: float
    new_throughput: float
    throughput_change_pct: float
    new_bottleneck: str
    impact_description: str

    def summary(self) -> str:
        return (
            f"What-If: {self.scenario}\n"
            f"  Throughput: {self.original_throughput:.0f} → {self.new_throughput:.0f} tok/s "
            f"({self.throughput_change_pct:+.1f}%)\n"
            f"  New bottleneck: {self.new_bottleneck}\n"
            f"  Impact: {self.impact_description}"
        )


@dataclass
class ValidationReport:
    """Full validation report."""
    is_valid: bool
    issues: list[str]
    warnings: list[str]
    simulation: PipelineSimulation
    what_if_scenarios: list[WhatIfScenario] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Validation: {'PASS' if self.is_valid else 'FAIL'}",
            f"  Issues: {len(self.issues)}",
            f"  Warnings: {len(self.warnings)}",
        ]
        for issue in self.issues:
            lines.append(f"    [ISSUE] {issue}")
        for warn in self.warnings:
            lines.append(f"    [WARN]  {warn}")
        lines.append("")
        lines.append(self.simulation.summary())
        if self.what_if_scenarios:
            lines.append("")
            lines.append("  What-If Scenarios:")
            for scenario in self.what_if_scenarios:
                for line in scenario.summary().split("\n"):
                    lines.append(f"    {line}")
        return "\n".join(lines)


class PartitionValidator:
    """Validates partition solutions via synthetic simulation.

    Args:
        cost_model: Cost model for latency estimation.
        num_tokens: Number of tokens to simulate (for throughput).
        num_trials: Number of Monte Carlo trials for confidence intervals.
        jitter_pct: Random jitter to simulate real-world variance.
    """

    def __init__(
        self,
        cost_model: PartitionCostModel,
        num_tokens: int = 1,
        num_trials: int = 100,
        jitter_pct: float = 0.05,
    ):
        self._cost_model = cost_model
        self._num_tokens = num_tokens
        self._num_trials = num_trials
        self._jitter = jitter_pct

    def validate(
        self,
        solution: PartitionSolution,
        num_layers: int,
        batch_size: int = 1,
        seq_len: int = 4096,
    ) -> ValidationReport:
        """Validate a partition solution.

        Args:
            solution: Partition solution to validate.
            num_layers: Total number of layers.
            batch_size: Batch size.
            seq_len: Sequence length.

        Returns:
            ValidationReport with issues, warnings, and simulation.
        """
        issues: list[str] = []
        warnings: list[str] = []

        # Check coverage
        if solution.coverage != (0, num_layers):
            issues.append(
                f"Incomplete coverage: {solution.coverage} != (0, {num_layers})"
            )

        # Check for gaps
        for i in range(len(solution.points) - 1):
            if solution.points[i].end_layer != solution.points[i + 1].start_layer:
                issues.append(
                    f"Gap between {solution.points[i].node_id} "
                    f"(ends at {solution.points[i].end_layer}) and "
                    f"{solution.points[i+1].node_id} "
                    f"(starts at {solution.points[i+1].start_layer})"
                )

        # Simulate pipeline
        simulation = self._simulate_pipeline(solution, batch_size, seq_len)

        # Check memory
        for stage in simulation.stages:
            if stage.is_oom:
                issues.append(f"OOM on {stage.node_id}: {stage.memory_utilization:.0%} utilization")
            elif stage.memory_utilization > 0.85:
                warnings.append(
                    f"High memory on {stage.node_id}: {stage.memory_utilization:.0%}"
                )

        # Check straggler
        if simulation.straggler_amplification > 1.5:
            warnings.append(
                f"Straggler amplification: {simulation.straggler_amplification:.2f}x "
                f"(bottleneck is {simulation.straggler_amplification:.1f}x slower than average)"
            )

        # Check bubbles
        if simulation.bubble_pct > 20:
            warnings.append(f"High pipeline bubble: {simulation.bubble_pct:.1f}%")

        # Generate what-if scenarios
        what_ifs = self._generate_what_ifs(solution, batch_size, seq_len)

        return ValidationReport(
            is_valid=len(issues) == 0,
            issues=issues,
            warnings=warnings,
            simulation=simulation,
            what_if_scenarios=what_ifs,
        )

    def what_if_slowdown(
        self,
        solution: PartitionSolution,
        target_node: str,
        slowdown_pct: float = 20.0,
        batch_size: int = 1,
        seq_len: int = 4096,
    ) -> WhatIfScenario:
        """Simulate what happens if a node becomes slower.

        Args:
            solution: Original partition.
            target_node: Node to slow down.
            slowdown_pct: Percentage slowdown.
            batch_size: Batch size.
            seq_len: Sequence length.

        Returns:
            WhatIfScenario with impact analysis.
        """
        original = self._simulate_pipeline(solution, batch_size, seq_len)

        # Create modified solution with inflated cost
        modified_points = []
        for pt in solution.points:
            new_pt = PartitionPoint(
                node_id=pt.node_id,
                start_layer=pt.start_layer,
                end_layer=pt.end_layer,
                estimated_time_ms=pt.estimated_time_ms,
            )
            if pt.node_id == target_node:
                new_pt.estimated_time_ms *= (1 + slowdown_pct / 100)
            modified_points.append(new_pt)

        modified_solution = PartitionSolution(
            points=modified_points,
            max_node_time_ms=max(p.estimated_time_ms for p in modified_points),
        )

        modified = self._simulate_pipeline(modified_solution, batch_size, seq_len)

        tp_change = (
            (modified.throughput_tok_s - original.throughput_tok_s)
            / max(original.throughput_tok_s, 0.001)
        ) * 100

        new_bottleneck = modified.bottleneck_node
        if new_bottleneck == target_node:
            impact = f"{target_node} remains the bottleneck after slowdown"
        else:
            impact = f"Bottleneck shifts from {original.bottleneck_node} to {new_bottleneck}"

        return WhatIfScenario(
            scenario=f"{target_node} slows by {slowdown_pct:.0f}%",
            original_throughput=original.throughput_tok_s,
            new_throughput=modified.throughput_tok_s,
            throughput_change_pct=round(tp_change, 1),
            new_bottleneck=new_bottleneck,
            impact_description=impact,
        )

    def _simulate_pipeline(
        self,
        solution: PartitionSolution,
        batch_size: int,
        seq_len: int,
    ) -> PipelineSimulation:
        """Simulate pipeline execution with Monte Carlo jitter."""
        all_throughputs: list[float] = []

        for _ in range(self._num_trials):
            stages: list[StageTiming] = []

            for i, pt in enumerate(solution.points):
                cost = self._cost_model.evaluate(
                    pt.node_id, pt.start_layer, pt.end_layer,
                    batch_size, seq_len,
                )

                # Apply jitter
                jitter = 1.0 + random.gauss(0, self._jitter)
                jitter = max(0.5, min(1.5, jitter))

                compute = cost.compute_time_ms * jitter
                comm_in = cost.communication_time_ms * jitter if i > 0 else 0.0
                comm_out = 0.0  # Handled by next stage's comm_in
                total = compute + comm_in

                stages.append(StageTiming(
                    node_id=pt.node_id,
                    start_layer=pt.start_layer,
                    end_layer=pt.end_layer,
                    compute_ms=round(compute, 2),
                    comm_in_ms=round(comm_in, 2),
                    comm_out_ms=round(comm_out, 2),
                    total_ms=round(total, 2),
                    memory_bytes=cost.memory_bytes,
                    memory_available_bytes=cost.memory_available_bytes,
                    memory_utilization=cost.memory_utilization,
                    is_oom=not cost.fits_in_memory,
                ))

            # Mark bottleneck
            if stages:
                bottleneck = max(stages, key=lambda s: s.total_ms)
                bottleneck.is_bottleneck = True

                # Pipeline timing
                stage_times = [s.total_ms for s in stages]
                bottleneck_ms = max(stage_times)
                total_compute = sum(stage_times)
                bubble_time = bottleneck_ms * (len(stages) - 1)
                total_pipeline = bottleneck_ms * len(stages)
                bubble_pct = (bubble_time / max(total_pipeline, 0.001)) * 100

                throughput = (batch_size * seq_len) / max(bottleneck_ms / 1000, 1e-9)
                all_throughputs.append(throughput)

        # Aggregate
        if not stages:
            return PipelineSimulation(
                stages=[], total_pipeline_time_ms=0, bubble_time_ms=0,
                bubble_pct=0, throughput_tok_s=0, bottleneck_node="",
                bottleneck_ms=0, memory_hotspots=[], straggler_amplification=0,
                confidence_interval=(0, 0),
            )

        bottleneck_stage = max(stages, key=lambda s: s.total_ms)
        avg_stage = sum(s.total_ms for s in stages) / len(stages)
        straggler_amp = bottleneck_stage.total_ms / max(avg_stage, 0.001)

        hotspots = [s.node_id for s in stages if s.memory_utilization > 0.85]

        # Confidence interval
        if all_throughputs:
            all_throughputs.sort()
            ci_low = all_throughputs[int(len(all_throughputs) * 0.05)]
            ci_high = all_throughputs[int(len(all_throughputs) * 0.95)]
            avg_tp = sum(all_throughputs) / len(all_throughputs)
        else:
            ci_low = ci_high = avg_tp = 0.0

        return PipelineSimulation(
            stages=stages,
            total_pipeline_time_ms=round(bottleneck_stage.total_ms * len(stages), 2),
            bubble_time_ms=round(bottleneck_stage.total_ms * (len(stages) - 1), 2),
            bubble_pct=round(
                (bottleneck_stage.total_ms * (len(stages) - 1))
                / max(bottleneck_stage.total_ms * len(stages), 0.001) * 100, 1
            ),
            throughput_tok_s=round(avg_tp, 0),
            bottleneck_node=bottleneck_stage.node_id,
            bottleneck_ms=bottleneck_stage.total_ms,
            memory_hotspots=hotspots,
            straggler_amplification=round(straggler_amp, 2),
            confidence_interval=(round(ci_low, 0), round(ci_high, 0)),
        )

    def _generate_what_ifs(
        self,
        solution: PartitionSolution,
        batch_size: int,
        seq_len: int,
    ) -> list[WhatIfScenario]:
        """Generate what-if scenarios for each node."""
        scenarios: list[WhatIfScenario] = []
        seen_nodes: set[str] = set()

        for pt in solution.points:
            if pt.node_id in seen_nodes:
                continue
            seen_nodes.add(pt.node_id)
            scenario = self.what_if_slowdown(
                solution, pt.node_id, slowdown_pct=20.0,
                batch_size=batch_size, seq_len=seq_len,
            )
            scenarios.append(scenario)

        return scenarios
