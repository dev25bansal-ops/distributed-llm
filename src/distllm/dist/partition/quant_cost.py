"""Quantization-aware cost model extension.

Extends PartitionCostModel to accept a QuantMethod parameter and adjust
memory, compute, and communication costs based on active quantization.
Bridges the Adaptive Precision Optimizer (APO) and the partition optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from loguru import logger

from distllm.dist.partition.cost_model import NodeCost, PartitionCostModel
from distllm.dist.partition.profiles import GPUProfile, LayerWeights
from distllm.dist.partition.quantization_tuner import (
    ACTIVATION_PROFILES,
    KV_PROFILES,
    ActivationQuantMethod,
    NodeQuantRecommendation,
    QuantMethod,
    QuantProfile,
    QuantizationPlan,
    QUANT_PROFILES,
)
from distllm.dist.partition.topology import TopologyGraph


@dataclass
class QuantNodeCost:
    """Extended node cost with quantization breakdown."""
    node_id: str
    start_layer: int
    end_layer: int
    # Base costs (unquantized)
    base_compute_time_ms: float = 0.0
    base_memory_bytes: int = 0
    # Quantized costs
    compute_time_ms: float = 0.0
    communication_time_ms: float = 0.0
    total_time_ms: float = 0.0
    memory_bytes: int = 0
    memory_available_bytes: int = 0
    fits_in_memory: bool = True
    # Quantization details
    weight_quant_method: str = "none"
    weight_memory_reduction: float = 1.0
    activation_quant_method: str = "none"
    activation_bandwidth_reduction: float = 1.0
    kv_cache_bits: str = "none"
    kv_memory_reduction: float = 1.0
    total_memory_reduction: float = 1.0
    quality_loss_estimate: float = 0.0

    @property
    def memory_utilization(self) -> float:
        if self.memory_available_bytes == 0:
            return 0.0
        return self.memory_bytes / self.memory_available_bytes

    @property
    def memory_savings_bytes(self) -> int:
        return self.base_memory_bytes - self.memory_bytes

    def summary(self) -> str:
        return (
            f"Node {self.node_id} [{self.start_layer},{self.end_layer}): "
            f"quant={self.weight_quant_method}, "
            f"mem={self.memory_bytes/(1024**2):.0f}MB/"
            f"{self.memory_available_bytes/(1024**3):.1f}GB "
            f"({self.total_memory_reduction:.1f}x reduction), "
            f"compute={self.compute_time_ms:.1f}ms, "
            f"total={self.total_time_ms:.1f}ms"
            f"{' [OOM]' if not self.fits_in_memory else ''}"
        )


class QuantizationAwareCostModel:
    """Cost model that adjusts estimates based on quantization method.

    Wraps PartitionCostModel and applies quantization multipliers for:
    - Weight memory reduction
    - Compute speed penalty
    - KV cache memory reduction
    - Activation communication reduction
    """

    def __init__(
        self,
        base_cost_model: PartitionCostModel,
        quant_profiles: dict[QuantMethod, QuantProfile] | None = None,
    ):
        self._base = base_cost_model
        self._profiles = quant_profiles or QUANT_PROFILES

    def evaluate_with_quant(
        self,
        node_id: str,
        start_layer: int,
        end_layer: int,
        quant_recommendation: NodeQuantRecommendation | None = None,
        batch_size: int = 1,
        seq_len: int = 4096,
    ) -> QuantNodeCost:
        """Evaluate node cost with quantization applied.

        Args:
            node_id: Node identifier.
            start_layer: First layer index.
            end_layer: Last layer index (exclusive).
            quant_recommendation: APO recommendation for this node.
            batch_size: Target batch size.
            seq_len: Target sequence length.

        Returns:
            QuantNodeCost with full breakdown.
        """
        # Get base (unquantized) cost
        base = self._base.evaluate(node_id, start_layer, end_layer, batch_size, seq_len)

        if quant_recommendation is None or quant_recommendation.method == QuantMethod.NONE:
            return QuantNodeCost(
                node_id=node_id,
                start_layer=start_layer,
                end_layer=end_layer,
                base_compute_time_ms=base.compute_time_ms,
                base_memory_bytes=base.memory_bytes,
                compute_time_ms=base.compute_time_ms,
                communication_time_ms=base.communication_time_ms,
                total_time_ms=base.total_time_ms,
                memory_bytes=base.memory_bytes,
                memory_available_bytes=base.memory_available_bytes,
                fits_in_memory=base.fits_in_memory,
                weight_quant_method="none",
                total_memory_reduction=1.0,
            )

        method = quant_recommendation.method
        profile = self._profiles.get(method)
        if profile is None:
            logger.warning(f"Unknown quant method {method}, using base cost")
            return QuantNodeCost(
                node_id=node_id,
                start_layer=start_layer,
                end_layer=end_layer,
                base_compute_time_ms=base.compute_time_ms,
                base_memory_bytes=base.memory_bytes,
                compute_time_ms=base.compute_time_ms,
                total_time_ms=base.total_time_ms,
                memory_bytes=base.memory_bytes,
                memory_available_bytes=base.memory_available_bytes,
                fits_in_memory=base.fits_in_memory,
            )

        # Apply speed penalty to compute
        compute_ms = base.compute_time_ms * profile.speed_penalty

        # Apply KV cache reduction
        kv_profile = KV_PROFILES.get(quant_recommendation.kv_cache_bits, {})
        kv_reduction = kv_profile.get("memory_reduction", 1.0)

        # Memory recomposition — never subtract from the quantized weight
        # figure. ``memory_bytes_with_quant`` is the quantized *weights* only;
        # the node's real footprint is quantized weights + quantized KV cache
        # + (unquantized) activation/other memory.
        quant_weights = quant_recommendation.memory_bytes_with_quant
        # Prefer the base cost model's own weight breakdown when available;
        # otherwise fall back to the APO recommendation's weights estimate.
        weights_fn = getattr(self._base, "weights_memory_bytes", None)
        if weights_fn is not None:
            weights_base = int(weights_fn(start_layer, end_layer))
        else:
            weights_base = quant_recommendation.memory_bytes_without_quant
        if weights_base <= 0 or weights_base > base.memory_bytes:
            # Last-resort defensive split (weights ≈ 80% of the footprint).
            weights_base = int(base.memory_bytes * 0.8)

        # KV cache is estimated as 20% of the unquantized footprint, but
        # never larger than the non-weight portion.
        kv_fraction = 0.2
        kv_base = int(base.memory_bytes * kv_fraction)
        kv_base = min(kv_base, base.memory_bytes - weights_base)
        kv_quant = int(kv_base * kv_reduction)

        # Non-weight, non-KV memory (activations etc.) stays unquantized
        activations = max(0, base.memory_bytes - weights_base - kv_base)

        total_mem = quant_weights + kv_quant + activations

        # Apply activation quantization to communication
        act_profile = ACTIVATION_PROFILES.get(quant_recommendation.activation_quant, {})
        act_reduction = act_profile.get("bandwidth_reduction", 1.0)
        act_overhead = act_profile.get("overhead_ms", 0.0)
        comm_ms = base.communication_time_ms * act_reduction + act_overhead

        total_ms = compute_ms + comm_ms
        fits = total_mem <= base.memory_available_bytes * 0.9

        total_reduction = base.memory_bytes / max(total_mem, 1)

        return QuantNodeCost(
            node_id=node_id,
            start_layer=start_layer,
            end_layer=end_layer,
            base_compute_time_ms=base.compute_time_ms,
            base_memory_bytes=base.memory_bytes,
            compute_time_ms=round(compute_ms, 2),
            communication_time_ms=round(comm_ms, 2),
            total_time_ms=round(total_ms, 2),
            memory_bytes=total_mem,
            memory_available_bytes=base.memory_available_bytes,
            fits_in_memory=fits,
            weight_quant_method=method.value,
            weight_memory_reduction=profile.memory_reduction,
            activation_quant_method=quant_recommendation.activation_quant.value,
            activation_bandwidth_reduction=act_reduction,
            kv_cache_bits=quant_recommendation.kv_cache_bits.value,
            kv_memory_reduction=kv_reduction,
            total_memory_reduction=round(total_reduction, 2),
            quality_loss_estimate=quant_recommendation.quality_loss,
        )

    def evaluate_partition_with_quant(
        self,
        partition: list[tuple[str, int, int]],
        quant_plan: QuantizationPlan | None = None,
        batch_size: int = 1,
        seq_len: int = 4096,
    ) -> list[QuantNodeCost]:
        """Evaluate a full partition with quantization applied.

        Args:
            partition: List of (node_id, start_layer, end_layer).
            quant_plan: APO plan with per-node recommendations.
            batch_size: Target batch size.
            seq_len: Target sequence length.

        Returns:
            List of QuantNodeCost per partition segment.
        """
        # Build recommendation lookup
        rec_lookup: dict[str, NodeQuantRecommendation] = {}
        if quant_plan:
            for rec in quant_plan.recommendations:
                rec_lookup[rec.node_id] = rec

        costs: list[QuantNodeCost] = []
        for node_id, start, end in partition:
            rec = rec_lookup.get(node_id)
            cost = self.evaluate_with_quant(
                node_id, start, end, rec, batch_size, seq_len,
            )
            costs.append(cost)

        return costs

    def compare_with_without_quant(
        self,
        partition: list[tuple[str, int, int]],
        quant_plan: QuantizationPlan,
        batch_size: int = 1,
        seq_len: int = 4096,
    ) -> dict[str, Any]:
        """Compare partition cost with and without quantization.

        Returns dict with 'without_quant' and 'with_quant' summaries.
        """
        without = self.evaluate_partition_with_quant(
            partition, None, batch_size, seq_len,
        )
        with_q = self.evaluate_partition_with_quant(
            partition, quant_plan, batch_size, seq_len,
        )

        def _summarize(costs: list[QuantNodeCost]) -> dict[str, Any]:
            if not costs:
                return {}
            return {
                "max_latency_ms": max(c.total_time_ms for c in costs),
                "total_memory_bytes": sum(c.memory_bytes for c in costs),
                "total_memory_gb": round(sum(c.memory_bytes for c in costs) / (1024**3), 2),
                "oom_nodes": sum(1 for c in costs if not c.fits_in_memory),
                "avg_memory_reduction": round(
                    sum(c.total_memory_reduction for c in costs) / len(costs), 2
                ),
                "nodes": [c.summary() for c in costs],
            }

        return {
            "without_quant": _summarize(without),
            "with_quant": _summarize(with_q),
        }
