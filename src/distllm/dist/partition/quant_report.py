"""Cluster-wide quantization report generator.

Given a QuantizationPlan, produces a rich report with:
- Per-node: recommended method, expected savings, speed impact
- Per-model: total memory reduction, estimated quality loss
- Comparison vs no-tuning baseline
- Conflict warnings (e.g., "Node-3 prefers AWQ but lacks calibration data")
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from loguru import logger

from distllm.dist.partition.quantization_tuner import (
    ActivationQuantMethod,
    KVCacheBits,
    NodeInfo,
    NodeQuantRecommendation,
    QuantMethod,
    QuantizationPlan,
)


@dataclass
class NodeReport:
    """Report entry for a single node."""
    node_id: str
    method: str
    method_display: str
    memory_without_quant_gb: float
    memory_with_quant_gb: float
    memory_savings_gb: float
    memory_savings_pct: float
    speed_penalty: float
    quality_loss: float
    reason: str
    activation_quant: str
    kv_cache_bits: str
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ConflictWarning:
    """A conflict or issue detected in the plan."""
    node_id: str
    severity: str  # "info" | "warning" | "error"
    message: str


@dataclass
class QuantizationReport:
    """Full cluster-wide quantization report."""
    strategy: str
    nodes: list[NodeReport] = field(default_factory=list)
    conflicts: list[ConflictWarning] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    baseline_comparison: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "nodes": [n.to_dict() for n in self.nodes],
            "conflicts": [asdict(c) for c in self.conflicts],
            "summary": self.summary,
            "baseline_comparison": self.baseline_comparison,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_text(self) -> str:
        """Human-readable text report."""
        lines = [
            "=" * 70,
            "  Adaptive Precision Optimizer — Cluster Report",
            "=" * 70,
            "",
            f"Strategy: {self.strategy}",
            "",
        ]

        # Summary
        s = self.summary
        if s:
            lines.append("Summary:")
            lines.append(f"  Nodes analyzed:    {s.get('num_nodes', 0)}")
            lines.append(f"  Total VRAM:        {s.get('total_vram_gb', 0):.1f} GB")
            lines.append(f"  Model size (fp16): {s.get('model_size_gb', 0):.1f} GB")
            lines.append(f"  Memory saved:      {s.get('total_savings_gb', 0):.1f} GB "
                        f"({s.get('total_savings_pct', 0):.0f}%)")
            lines.append(f"  Avg quality loss:  {s.get('avg_quality_loss', 0):.3f}")
            lines.append(f"  Methods used:      {', '.join(s.get('methods_used', []))}")
            lines.append("")

        # Per-node details
        lines.append("Per-Node Recommendations:")
        lines.append("-" * 70)
        for n in self.nodes:
            lines.append(
                f"  {n.node_id}: {n.method_display} | "
                f"{n.memory_without_quant_gb:.1f}GB -> {n.memory_with_quant_gb:.1f}GB "
                f"(save {n.memory_savings_gb:.1f}GB, {n.memory_savings_pct:.0f}%) | "
                f"speed {n.speed_penalty:.2f}x | quality loss {n.quality_loss:.3f}"
            )
            if n.activation_quant != "none":
                lines.append(f"    Activation quant: {n.activation_quant}")
            if n.kv_cache_bits != "none":
                lines.append(f"    KV cache: {n.kv_cache_bits}")
            lines.append(f"    Reason: {n.reason}")
            if n.warnings:
                for w in n.warnings:
                    lines.append(f"    WARNING: {w}")
            lines.append("")

        # Conflicts
        if self.conflicts:
            lines.append("Conflicts & Warnings:")
            lines.append("-" * 70)
            for c in self.conflicts:
                icon = {"info": "ℹ", "warning": "⚠", "error": "✗"}.get(c.severity, "•")
                lines.append(f"  [{c.severity.upper()}] {c.node_id}: {c.message}")
            lines.append("")

        # Baseline comparison
        bc = self.baseline_comparison
        if bc:
            lines.append("Comparison vs No Quantization:")
            lines.append("-" * 70)
            if "without_quant" in bc:
                w = bc["without_quant"]
                lines.append(f"  Without: {w.get('oom_nodes', 0)} OOM nodes, "
                            f"{w.get('total_memory_gb', 0):.1f} GB total memory")
            if "with_quant" in bc:
                q = bc["with_quant"]
                lines.append(f"  With APO: {q.get('oom_nodes', 0)} OOM nodes, "
                            f"{q.get('total_memory_gb', 0):.1f} GB total memory "
                            f"({q.get('avg_memory_reduction', 1):.1f}x reduction)")
            lines.append("")

        lines.append("=" * 70)
        return "\n".join(lines)


class ReportGenerator:
    """Generates cluster-wide quantization reports."""

    def generate(
        self,
        plan: QuantizationPlan,
        nodes: list[NodeInfo | dict[str, Any]] | None = None,
        model_size_bytes: int = 0,
        num_layers: int = 0,
    ) -> QuantizationReport:
        """Generate a full report from a quantization plan.

        Args:
            plan: The APO quantization plan.
            nodes: Optional list of node info for VRAM data.
            model_size_bytes: Total model size for summary.
            num_layers: Total layers for summary.

        Returns:
            QuantizationReport with all details.
        """
        node_map: dict[str, NodeInfo] = {}
        if nodes:
            for n in nodes:
                ni = n if isinstance(n, NodeInfo) else NodeInfo.from_dict(n)
                node_map[ni.node_id] = ni

        node_reports: list[NodeReport] = []
        conflicts: list[ConflictWarning] = []
        methods_used: set[str] = set()
        total_savings = 0
        total_quality = 0.0

        for rec in plan.recommendations:
            ni = node_map.get(rec.node_id)
            vram_gb = ni.total_memory_bytes / (1024**3) if ni else 0.0

            method_display = self._method_display(rec.method)
            methods_used.add(rec.method.value)

            warnings = self._check_node_warnings(rec, ni)
            for w in warnings:
                conflicts.append(ConflictWarning(
                    node_id=rec.node_id,
                    severity="warning",
                    message=w,
                ))

            node_reports.append(NodeReport(
                node_id=rec.node_id,
                method=rec.method.value,
                method_display=method_display,
                memory_without_quant_gb=round(rec.memory_bytes_without_quant / (1024**3), 2),
                memory_with_quant_gb=round(rec.memory_bytes_with_quant / (1024**3), 2),
                memory_savings_gb=round(rec.memory_savings_bytes / (1024**3), 2),
                memory_savings_pct=rec.memory_savings_pct,
                speed_penalty=rec.speed_penalty,
                quality_loss=rec.quality_loss,
                reason=rec.reason,
                activation_quant=rec.activation_quant.value,
                kv_cache_bits=rec.kv_cache_bits.value,
                warnings=warnings,
            ))

            total_savings += rec.memory_savings_bytes
            total_quality += rec.quality_loss

        num_nodes = len(plan.recommendations)
        total_vram = sum(
            (ni.total_memory_bytes for ni in node_map.values()),
            start=0,
        )

        summary = {
            "num_nodes": num_nodes,
            "total_vram_gb": round(total_vram / (1024**3), 2),
            "model_size_gb": round(model_size_bytes / (1024**3), 2),
            "num_layers": num_layers,
            "total_savings_gb": round(total_savings / (1024**3), 2),
            "total_savings_pct": round(
                (total_savings / max(model_size_bytes, 1)) * 100, 1
            ),
            "avg_quality_loss": round(total_quality / max(num_nodes, 1), 4),
            "methods_used": sorted(methods_used),
        }

        return QuantizationReport(
            strategy=plan.strategy,
            nodes=node_reports,
            conflicts=conflicts,
            summary=summary,
        )

    def _method_display(self, method: QuantMethod) -> str:
        displays = {
            QuantMethod.NONE: "FP16 (no quantization)",
            QuantMethod.BNB_8BIT: "BitsAndBytes INT8",
            QuantMethod.BNB_4BIT: "BitsAndBytes NF4",
            QuantMethod.GPTQ: "GPTQ 4-bit",
            QuantMethod.AWQ: "AWQ 4-bit",
            QuantMethod.FP8_E4M3: "FP8 E4M3",
            QuantMethod.FP8_E5M2: "FP8 E5M2",
            QuantMethod.INT8: "INT8",
            QuantMethod.NF4: "NF4",
        }
        return displays.get(method, method.value)

    def _check_node_warnings(
        self,
        rec: NodeQuantRecommendation,
        node: NodeInfo | None,
    ) -> list[str]:
        """Check for conflicts/warnings for a node recommendation."""
        warnings: list[str] = []

        if node is None:
            return warnings

        # Calibration conflict
        if rec.method in (QuantMethod.GPTQ, QuantMethod.AWQ):
            # These require calibration — warn if we don't know if it's available
            warnings.append(
                f"{rec.method.value} requires calibration data. "
                "Ensure pre-quantized model or calibration dataset is available."
            )

        # FP8 on non-Hopper
        if rec.method in (QuantMethod.FP8_E4M3, QuantMethod.FP8_E5M2):
            if node.compute_capability is not None and node.compute_capability < 9.0:
                warnings.append(
                    f"{rec.method.value} requires Hopper+ GPU (cc>=9.0), "
                    f"but node has cc={node.compute_capability}"
                )

        # High quality loss
        if rec.quality_loss > 0.03:
            warnings.append(
                f"Quality loss {rec.quality_loss:.3f} may be noticeable. "
                "Consider using a less aggressive method."
            )

        # Tight VRAM
        if node.total_memory_bytes > 0:
            utilization = rec.memory_bytes_with_quant / node.total_memory_bytes
            if utilization > 0.9:
                warnings.append(
                    f"VRAM utilization at {utilization*100:.0f}% — "
                    "very little headroom for KV cache and activations."
                )

        return warnings
