"""Auto-select quantization level based on available GPU memory.

Automatically chooses the optimal quantization level (FP16, INT8, INT4)
based on available GPU memory and model size, balancing quality vs.
memory efficiency.

Usage::

    selector = QuantizationSelector()
    level = selector.select(
        model_params_b=70,
        available_memory_gb=40,
        num_gpus=2,
    )
    # level = "int8" (70B * 2 bytes / 2 GPUs = 70GB > 40GB → need INT8)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from loguru import logger


@dataclass
class QuantizationChoice:
    """A quantization recommendation."""
    level: str  # "fp16", "int8", "int4_awq", "int4_gptq"
    bytes_per_param: float
    estimated_memory_gb: float
    quality_score: float  # 1.0 = lossless, 0.5 = significant loss
    fits_in_memory: bool
    reason: str


class QuantizationSelector:
    """Selects optimal quantization level based on hardware constraints.

    Decision matrix:
    - FP16: Default, best quality, 2 bytes/param
    - INT8: Good quality, 1 byte/param, 50% memory savings
    - INT4-AWQ: Acceptable quality, 0.5 bytes/param, 75% savings
    - INT4-GPTQ: Acceptable quality, 0.5 bytes/param, 75% savings
    """

    # Quality scores (1.0 = lossless)
    QUALITY = {
        "fp16": 1.0,
        "int8": 0.95,
        "int4_awq": 0.85,
        "int4_gptq": 0.85,
        "int4": 0.80,
    }

    BYTES_PER_PARAM = {
        "fp16": 2.0,
        "int8": 1.0,
        "int4_awq": 0.5,
        "int4_gptq": 0.5,
        "int4": 0.5,
    }

    def select(
        self,
        model_params_b: float,
        available_memory_gb: float,
        num_gpus: int = 1,
        preferred_quality: float = 0.9,
        include_kv_cache: bool = True,
    ) -> QuantizationChoice:
        """Select optimal quantization level.

        Args:
            model_params_b: Model parameters in billions.
            available_memory_gb: Available GPU memory in GB (per GPU).
            num_gpus: Number of GPUs.
            preferred_quality: Minimum acceptable quality (0-1).
            include_kv_cache: Whether to reserve memory for KV cache.

        Returns:
            QuantizationChoice with recommendation.
        """
        total_memory = available_memory_gb * num_gpus

        # Reserve 20% for KV cache and activations if requested
        usable_memory = total_memory * 0.8 if include_kv_cache else total_memory

        # Try each level from highest quality to lowest
        for level in ["fp16", "int8", "int4_awq", "int4_gptq"]:
            bpp = self.BYTES_PER_PARAM[level]
            memory_gb = model_params_b * bpp
            quality = self.QUALITY[level]
            fits = memory_gb <= usable_memory

            if fits and quality >= preferred_quality:
                return QuantizationChoice(
                    level=level,
                    bytes_per_param=bpp,
                    estimated_memory_gb=round(memory_gb, 1),
                    quality_score=quality,
                    fits_in_memory=True,
                    reason=f"{model_params_b}B * {bpp} B/param = {memory_gb:.0f}GB fits in {usable_memory:.0f}GB",
                )

        # Nothing fits with preferred quality — use lowest quality that fits
        for level in ["int4_gptq", "int4_awq", "int8", "fp16"]:
            bpp = self.BYTES_PER_PARAM[level]
            memory_gb = model_params_b * bpp
            if memory_gb <= usable_memory:
                return QuantizationChoice(
                    level=level,
                    bytes_per_param=bpp,
                    estimated_memory_gb=round(memory_gb, 1),
                    quality_score=self.QUALITY[level],
                    fits_in_memory=True,
                    reason=f"Forced {level}: only option that fits in {usable_memory:.0f}GB",
                )

        # Nothing fits at all
        return QuantizationChoice(
            level="int4",
            bytes_per_param=0.5,
            estimated_memory_gb=round(model_params_b * 0.5, 1),
            quality_score=0.8,
            fits_in_memory=False,
            reason=f"Model too large: {model_params_b}B needs {model_params_b * 0.5:.0f}GB minimum, "
                   f"only {usable_memory:.0f}GB available. Consider adding more GPUs.",
        )

    def estimate_model_params(self, model_name: str) -> float:
        """Estimate model parameters in billions from model name."""
        name = model_name.lower()
        if "405b" in name:
            return 405.0
        if "70b" in name:
            return 70.0
        if "65b" in name:
            return 65.0
        if "34b" in name:
            return 34.0
        if "13b" in name:
            return 13.0
        if "7b" in name or "8b" in name:
            return 7.0
        if "3b" in name:
            return 3.0
        if "1.5b" in name:
            return 1.5
        if "1b" in name:
            return 1.0
        if "0.5b" in name or "350m" in name:
            return 0.5
        return 7.0  # Default assumption

    def get_recommendation_string(self, choice: QuantizationChoice) -> str:
        """Return a human-readable recommendation."""
        if choice.fits_in_memory:
            return (
                f"Recommended: {choice.level.upper()} "
                f"({choice.estimated_memory_gb:.0f}GB, "
                f"quality={choice.quality_score:.0%}). "
                f"{choice.reason}"
            )
        return (
            f"WARNING: {choice.level.upper()} ({choice.estimated_memory_gb:.0f}GB) "
            f"may not fit in available memory. {choice.reason}"
        )
