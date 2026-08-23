"""Automatic Mixed-Precision Strategy Search (AMP AutoTuner).

Extends the ``QuantizationAutoTuner`` to dynamically discover the optimal
mixed-precision strategy per-layer, per-batch-size at runtime.

Workflow::

    1. Offline sensitivity analysis — profile each layer's tolerance to
       precision reduction (FP8, INT8, INT4) using KL divergence.
    2. Build a per-layer precision map — assign the lowest safe precision
       to each layer given a batch size and target quality threshold.
    3. Online switching — apply precision config at runtime via monkey-patch
       or kernel selection (FP8/INT4/INT8).

Integration::

    tuner = MixedPrecisionAutoTuner(
        num_layers=32,
        hidden_dim=4096,
        sensitivity_calibration_steps=100,
    )
    precision_map = tuner.analyze_sensitivity(layer_sample_inputs, batch_sizes=[1, 4, 16])
    plan = tuner.build_precision_plan(target_quality=0.99)
    tuner.apply_plan(plan)  # sets runtime precision configs
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from loguru import logger


class PrecisionMode(str, Enum):
    """Available precision modes for individual layers."""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    FP8_E4M3 = "fp8_e4m3"
    FP8_E5M2 = "fp8_e5m2"
    INT8 = "int8"
    INT4 = "int4"


@dataclass
class LayerPrecisionConfig:
    """Precision recommendation for a single layer at a given batch size."""
    layer_idx: int
    layer_type: str  # "attention", "mlp", "embedding", "norm"
    weight_precision: PrecisionMode = PrecisionMode.FP16
    activation_precision: PrecisionMode = PrecisionMode.FP16
    kv_cache_precision: PrecisionMode = PrecisionMode.FP16
    sensitivity_score: float = 0.0  # 0.0 = least sensitive, 1.0 = most


@dataclass
class PrecisionPlan:
    """Complete mixed-precision plan across all layers and batch sizes."""
    configs: dict[int, list[LayerPrecisionConfig]]  # batch_size -> layer configs
    estimated_speedup: float = 1.0
    estimated_memory_savings_gb: float = 0.0
    target_quality: float = 0.99


class LayerProfiler:
    """Runs small-scale profiling on a single layer to measure precision sensitivity.

    Uses KL divergence between FP16 reference output and quantized output
    as the sensitivity metric.  Lower KL = more quantizable.
    """

    def __init__(self, num_calibration_tokens: int = 512):
        self._num_calibration_tokens = num_calibration_tokens

    def profile_layer(
        self,
        layer_input: Any,
        layer_type: str,
        batch_size: int,
    ) -> dict[PrecisionMode, float]:
        """Simulate sensitivity analysis for a single layer.

        Returns a dict mapping each precision mode to a sensitivity score
        (0.0 = perfect, higher = more quality loss).

        In production this would run actual forward passes; here we
        approximate using a heuristic based on layer type.
        """
        scores: dict[PrecisionMode, float] = {}

        # Heuristic sensitivity by layer type
        type_sensitivity = {
            "attention": 0.8,  # attention is sensitive
            "mlp": 0.4,        # MLP is more robust
            "embedding": 0.2,  # embeddings are very robust
            "norm": 0.9,       # normalization is very sensitive
        }
        base = type_sensitivity.get(layer_type, 0.5)

        for mode in PrecisionMode:
            if mode == PrecisionMode.FP32:
                scores[mode] = 0.0  # baseline
            elif mode == PrecisionMode.FP16:
                scores[mode] = base * 0.1 * math.log2(max(batch_size, 2))
            elif mode == PrecisionMode.BF16:
                scores[mode] = base * 0.08 * math.log2(max(batch_size, 2))
            elif mode == PrecisionMode.FP8_E4M3:
                scores[mode] = base * 0.3 * math.log2(max(batch_size, 2))
            elif mode == PrecisionMode.FP8_E5M2:
                scores[mode] = base * 0.35 * math.log2(max(batch_size, 2))
            elif mode == PrecisionMode.INT8:
                scores[mode] = base * 0.5 * math.log2(max(batch_size, 2))
            elif mode == PrecisionMode.INT4:
                scores[mode] = base * 0.8 * math.log2(max(batch_size, 2))
            scores[mode] = min(1.0, scores[mode])

        return scores


class MixedPrecisionAutoTuner:
    """Auto-tuner that discovers the optimal per-layer, per-batch-size precision.

    Usage::

        tuner = MixedPrecisionAutoTuner(num_layers=32)
        plan = tuner.search(target_quality=0.99, batch_sizes=[1, 4, 8, 16, 32])
        tuner.apply(plan)

        # Switch plan for a specific batch size at runtime:
        current_config = plan.configs[current_batch_size]
    """

    def __init__(
        self,
        num_layers: int,
        hidden_dim: int = 4096,
        layer_types: Optional[list[str]] = None,
        profiler: Optional[LayerProfiler] = None,
    ):
        self._num_layers = num_layers
        self._hidden_dim = hidden_dim
        self._layer_types = layer_types or self._default_layer_types(num_layers)
        self._profiler = profiler or LayerProfiler()
        self._cache: dict[tuple[int, int, str], dict[PrecisionMode, float]] = {}
        self._lock = threading.RLock()

    def _default_layer_types(self, num_layers: int) -> list[str]:
        types: list[str] = []
        for i in range(num_layers):
            if i == 0:
                types.append("embedding")
            elif i == num_layers - 1:
                types.append("norm")
            elif i % 3 == 0:
                types.append("attention")
            else:
                types.append("mlp")
        return types

    def analyze_sensitivity(
        self,
        batch_sizes: Optional[list[int]] = None,
    ) -> dict[int, list[LayerPrecisionConfig]]:
        """Run sensitivity analysis across all layers and batch sizes.

        Returns: batch_size -> [LayerPrecisionConfig, ...]
        """
        if batch_sizes is None:
            batch_sizes = [1, 4, 8, 16, 32]

        result: dict[int, list[LayerPrecisionConfig]] = {}

        for bs in batch_sizes:
            configs: list[LayerPrecisionConfig] = []
            for layer_idx in range(self._num_layers):
                lt = self._layer_types[layer_idx]
                scores = self._profiler.profile_layer(None, lt, bs)

                best_weight = min(scores, key=lambda m: scores[m])
                best_act = min(scores, key=lambda m: scores[m])
                best_kv = min(scores, key=lambda m: scores[m])

                configs.append(LayerPrecisionConfig(
                    layer_idx=layer_idx,
                    layer_type=lt,
                    weight_precision=best_weight,
                    activation_precision=best_act,
                    kv_cache_precision=best_kv,
                    sensitivity_score=scores[best_weight],
                ))
            result[bs] = configs

        return result

    def build_precision_plan(
        self,
        sensitivity: dict[int, list[LayerPrecisionConfig]],
        target_quality: float = 0.99,
    ) -> PrecisionPlan:
        """Build a precision plan from the sensitivity analysis.

        For each batch size, the tuner assigns the most aggressive
        precision that stays within the quality budget.
        """
        configs: dict[int, list[LayerPrecisionConfig]] = {}
        total_speedup = 0.0
        total_memory = 0.0

        for bs, layer_configs in sensitivity.items():
            bs_configs: list[LayerPrecisionConfig] = []
            for cfg in layer_configs:
                # Greedy: pick the most aggressive precision below threshold
                threshold = 1.0 - target_quality
                modes = list(PrecisionMode)
                # Profile each mode and pick the best below threshold
                scores = self._profiler.profile_layer(None, cfg.layer_type, bs)
                best = PrecisionMode.FP16
                for mode in modes:
                    if scores.get(mode, 1.0) <= threshold:
                        best = mode
                        break

                bs_configs.append(LayerPrecisionConfig(
                    layer_idx=cfg.layer_idx,
                    layer_type=cfg.layer_type,
                    weight_precision=best,
                    activation_precision=best,
                    kv_cache_precision=best,
                    sensitivity_score=scores.get(best, 0.0),
                ))

            configs[bs] = bs_configs
            total_speedup += self._estimate_speedup(bs, bs_configs)
            total_memory += self._estimate_memory_savings(bs, bs_configs)

        return PrecisionPlan(
            configs=configs,
            estimated_speedup=total_speedup / max(len(configs), 1),
            estimated_memory_savings_gb=total_memory,
            target_quality=target_quality,
        )

    def _estimate_speedup(
        self,
        batch_size: int,
        configs: list[LayerPrecisionConfig],
    ) -> float:
        """Estimate throughput speedup from precision plan."""
        avg_precision_bits = sum(
            self._precision_bits(c.weight_precision)
            for c in configs
        ) / max(len(configs), 1)
        baseline_bits = 16  # FP16
        return baseline_bits / max(avg_precision_bits, 1)

    def _estimate_memory_savings(
        self,
        batch_size: int,
        configs: list[LayerPrecisionConfig],
    ) -> float:
        """Estimate memory savings in GB."""
        total_baseline = 0.0
        total_plan = 0.0
        for c in configs:
            params_per_layer = self._hidden_dim * self._hidden_dim * 4  # Q,K,V,O ~ 4x
            baseline = params_per_layer * 2  # FP16 = 2 bytes
            plan_bits = self._precision_bits(c.weight_precision)
            plan_val = params_per_layer * (plan_bits / 8)
            total_baseline += baseline
            total_plan += plan_val
        return (total_baseline - total_plan) / (1024 ** 3)

    @staticmethod
    def _precision_bits(mode: PrecisionMode) -> float:
        return {
            PrecisionMode.FP32: 32,
            PrecisionMode.FP16: 16,
            PrecisionMode.BF16: 16,
            PrecisionMode.FP8_E4M3: 8,
            PrecisionMode.FP8_E5M2: 8,
            PrecisionMode.INT8: 8,
            PrecisionMode.INT4: 4,
        }.get(mode, 16)

    def apply(self, plan: PrecisionPlan, batch_size: int = 8) -> list[LayerPrecisionConfig]:
        """Apply a precision plan for a specific batch size.

        In production this would monkey-patch linear layers or set
        runtime kernel configs.  Here it logs and returns the configs.
        """
        configs = plan.configs.get(batch_size)
        if configs is None:
            logger.warning(f"No precision plan for batch_size={batch_size}, using FP16")
            return []

        fp8_count = sum(1 for c in configs if "fp8" in c.weight_precision.value)
        int8_count = sum(1 for c in configs if c.weight_precision == PrecisionMode.INT8)
        int4_count = sum(1 for c in configs if c.weight_precision == PrecisionMode.INT4)

        logger.info(
            f"Applied AMP plan for batch_size={batch_size}: "
            f"{fp8_count}x FP8, {int8_count}x INT8, {int4_count}x INT4, "
            f"estimated speedup={plan.estimated_speedup:.2f}x"
        )
        return configs
