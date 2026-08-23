"""Quality metrics and evaluation helpers for quantization selection.

Provides per-layer precision profiling results types, the runtime profiler
:func:`profile_layer_precision`, and the rule-based
:func:`assign_mixed_precision` helper that assigns per-layer dtypes.

This module depends on ``quantization_tuner`` for the sensitivity analyzer
and plan types.  It does **not** export any enums or the main tuner class.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch
from loguru import logger

from distllm.dist.partition.quantization_tuner import (
    LayerQuantPlan,
    MixedPrecisionPlan,
    SensitivityAnalyzer,
)


# ---------------------------------------------------------------------------
# Per-layer profiling results
# ---------------------------------------------------------------------------


@dataclass
class LayerPrecisionProfile:
    """Runtime and memory profile for a single layer at a given precision.

    Attributes:
        layer_idx: Index of the layer in the transformer stack.
        layer_name: Dot-separated attribute path to the submodule.
        precision: Precision string (``"fp16"``, ``"fp8"``, or ``"int8"``).
        runtime_ms: Average forward-pass runtime in milliseconds.
        memory_bytes: GPU memory allocated after the forward pass.
        peak_memory_bytes: Peak GPU memory during the forward pass.
    """

    layer_idx: int
    layer_name: str
    precision: str  # "fp16" | "fp8" | "int8"
    runtime_ms: float
    memory_bytes: int
    peak_memory_bytes: int


@dataclass
class LayerPrecisionResult:
    """Aggregated profiling results across precisions for one layer.

    Attributes:
        layer_idx: Index of the layer in the transformer stack.
        layer_name: Dot-separated attribute path to the submodule.
        layer_type: Classified layer type (e.g. ``"attention"``, ``"mlp"``).
        profiles: Mapping of precision string to
            :class:`LayerPrecisionProfile`.
        recommended_precision: The selected best precision for this layer.
    """

    layer_idx: int
    layer_name: str
    layer_type: str
    profiles: dict[str, LayerPrecisionProfile] = field(default_factory=dict)
    recommended_precision: str = "fp16"

    @property
    def fp16(self) -> LayerPrecisionProfile | None:
        """Profile for fp16 precision, or *None* if not profiled."""
        return self.profiles.get("fp16")

    @property
    def fp8(self) -> LayerPrecisionProfile | None:
        """Profile for fp8 precision, or *None* if not profiled."""
        return self.profiles.get("fp8")

    @property
    def int8(self) -> LayerPrecisionProfile | None:
        """Profile for int8 precision, or *None* if not profiled."""
        return self.profiles.get("int8")

    def best_precision(
        self,
        memory_budget_bytes: int = 0,
        max_runtime_ms: float = float("inf"),
    ) -> str:
        """Select the best precision given constraints.

        Args:
            memory_budget_bytes: Maximum memory budget (0 = no constraint).
            max_runtime_ms: Maximum acceptable runtime per layer.

        Returns:
            Precision string (``"fp16"``, ``"fp8"``, or ``"int8"``).
        """
        candidates: list[tuple[str, LayerPrecisionProfile]] = []
        for prec, prof in self.profiles.items():
            if prof.runtime_ms > max_runtime_ms:
                continue
            if memory_budget_bytes > 0 and prof.memory_bytes > memory_budget_bytes:
                continue
            candidates.append((prec, prof))
        if not candidates:
            return "fp16"  # fallback to highest quality
        # Prefer lowest memory; tie-break by lowest runtime
        candidates.sort(key=lambda x: (x[1].memory_bytes, x[1].runtime_ms))
        return candidates[0][0]


# ---------------------------------------------------------------------------
# Per-layer profiling
# ---------------------------------------------------------------------------


def profile_layer_precision(
    model: torch.nn.Module,
    layer_idx: int,
    layer_name: str,
    sample_input: torch.Tensor,
    precisions: list[str] | None = None,
    num_warmup: int = 3,
    num_trials: int = 5,
    device: str = "cuda",
) -> LayerPrecisionResult:
    """Run one forward pass per precision for a single layer, measuring
    runtime and memory.

    Measures wall-clock time and GPU memory (via ``torch.cuda``) for
    each precision.  The layer is cast to the target dtype before each
    measurement and restored afterwards.

    Args:
        model: The top-level model (used to access the submodule).
        layer_idx: Index of the layer in the transformer stack.
        layer_name: Dot-separated attribute path to the submodule
            (e.g. ``"model.layers.0"``).
        sample_input: A dummy input tensor of the expected shape
            ``(1, seq_len, hidden_dim)``.
        precisions: List of precision strings to profile.  Defaults to
            ``["fp16", "fp8", "int8"]``.  ``"fp8"`` is only available on
            Hopper GPUs (compute capability >= 8.9).
        num_warmup: Number of warm-up iterations before measurements.
        num_trials: Number of timed trials per precision.
        device: Torch device string.

    Returns:
        A :class:`LayerPrecisionResult` containing per-precision profiles.
    """
    if precisions is None:
        precisions = ["fp16", "fp8", "int8"]

    # Resolve the submodule by attribute path
    module: torch.nn.Module = model
    for part in layer_name.split("."):
        module = getattr(module, part)

    # Classify the layer type
    analyzer = SensitivityAnalyzer()
    layer_type = analyzer.classify_layer(layer_name, module)

    profiles: dict[str, LayerPrecisionProfile] = {}

    for prec in precisions:
        target_dtype: torch.dtype
        if prec == "fp16":
            target_dtype = torch.float16
        elif prec == "fp8":
            target_dtype = torch.float8_e4m3fn  # FP8 E4M3
        elif prec == "int8":
            target_dtype = torch.int8
        else:
            raise ValueError(f"Unsupported precision: {prec}")

        # Cast module parameters to target dtype
        original_dtypes: dict[str, torch.dtype] = {}
        for name, param in module.named_parameters(recurse=False):
            original_dtypes[name] = param.dtype
            param.data = param.data.to(target_dtype)

        input_dtype = sample_input.dtype
        cast_input = sample_input.to(target_dtype) if prec != "fp16" else sample_input

        # Warm-up
        for _ in range(num_warmup):
            with torch.no_grad():
                _ = module(cast_input)

        # Timed trials
        torch.cuda.synchronize(device) if device.startswith("cuda") else None
        start_event = torch.cuda.Event(enable_timing=True) if device.startswith("cuda") else None
        end_event = torch.cuda.Event(enable_timing=True) if device.startswith("cuda") else None

        runtimes: list[float] = []
        peak_mem = 0

        for _ in range(num_trials):
            if torch.cuda.is_available() and start_event is not None and end_event is not None:
                torch.cuda.reset_peak_memory_stats(device)
                start_event.record()
            else:
                start_cpu = time.time()

            with torch.no_grad():
                _ = module(cast_input)

            if torch.cuda.is_available() and start_event is not None and end_event is not None:
                end_event.record()
                torch.cuda.synchronize(device)
                runtimes.append(start_event.elapsed_time(end_event))
                peak_mem = max(peak_mem, torch.cuda.max_memory_allocated(device))
            else:
                runtimes.append((time.time() - start_cpu) * 1000)

        avg_runtime = sum(runtimes) / len(runtimes)
        memory_used = (
            torch.cuda.memory_allocated(device) if torch.cuda.is_available() else 0
        )

        profiles[prec] = LayerPrecisionProfile(
            layer_idx=layer_idx,
            layer_name=layer_name,
            precision=prec,
            runtime_ms=avg_runtime,
            memory_bytes=memory_used,
            peak_memory_bytes=peak_mem,
        )

        # Restore original dtypes
        for name, orig_dtype in original_dtypes.items():
            param = module.get_parameter(name)
            param.data = param.data.to(orig_dtype)

    # Determine recommended precision
    # Attention layers -> fp16 (quality-sensitive)
    # MLP layers     -> lowest memory precision that fits
    if layer_type in ("embed", "lm_head", "norm", "attention"):
        recommended = "fp16"
    else:
        # Pick the precision with smallest memory footprint
        recommended = min(profiles.keys(), key=lambda p: profiles[p].memory_bytes)

    return LayerPrecisionResult(
        layer_idx=layer_idx,
        layer_name=layer_name,
        layer_type=layer_type,
        profiles=profiles,
        recommended_precision=recommended,
    )


# ---------------------------------------------------------------------------
# Rule-based mixed-precision assignment
# ---------------------------------------------------------------------------


def assign_mixed_precision(
    model: torch.nn.Module,
    num_layers: int,
    sample_input: torch.Tensor,
    *,
    profile_all: bool = False,
    precisions: list[str] | None = None,
    memory_budget_bytes: int = 0,
    max_runtime_ms: float = float("inf"),
    attention_precision: str = "fp16",
    mlp_precision: str = "int8",
    device: str = "cuda",
) -> MixedPrecisionPlan:
    """Assign per-layer precision dtypes for all transformer layers.

    By default uses a heuristic rule (attention -> *attention_precision*,
    MLP -> *mlp_precision*).  When ``profile_all=True``, it profiles
    every layer with :func:`profile_layer_precision` and picks the best
    precision per layer subject to constraints.

    Args:
        model: The transformer model.
        num_layers: Number of transformer layers.
        sample_input: Dummy input tensor ``(1, seq_len, hidden_dim)``.
        profile_all: If True, profile each layer and select per-layer
            precision based on measurements.  Otherwise use the heuristic.
        precisions: Precisions to profile when ``profile_all=True``.
        memory_budget_bytes: Per-layer memory budget.
        max_runtime_ms: Per-layer runtime budget.
        attention_precision: Precision for attention layers (heuristic).
        mlp_precision: Precision for MLP layers (heuristic).
        device: Torch device string.

    Returns:
        A :class:`MixedPrecisionPlan` with per-layer assignments.
    """
    analyzer = SensitivityAnalyzer()
    plans: list[LayerQuantPlan] = []
    total_compression = 0.0

    for layer_idx in range(num_layers):
        layer_name = f"model.layers.{layer_idx}"

        if profile_all:
            # Profile this layer at multiple precisions
            result = profile_layer_precision(
                model=model,
                layer_idx=layer_idx,
                layer_name=layer_name,
                sample_input=sample_input,
                precisions=precisions,
                device=device,
            )
            chosen_precision = result.best_precision(
                memory_budget_bytes=memory_budget_bytes,
                max_runtime_ms=max_runtime_ms,
            )
        else:
            # Heuristic: classify the layer and apply rule
            # We try to get the submodule to classify it properly
            module: torch.nn.Module = model
            for part in layer_name.split("."):
                module = getattr(module, part)
            layer_type = analyzer.classify_layer(layer_name, module)

            if layer_type in ("embed", "lm_head", "norm", "attention"):
                chosen_precision = attention_precision
            elif layer_type == "mlp":
                chosen_precision = mlp_precision
            else:
                chosen_precision = attention_precision  # safe default

        compression = {"float16": 1.0, "int8": 2.0, "nf4": 4.0, "fp8_e4m3": 2.0}.get(
            chosen_precision, 1.0
        )
        sensitivity = analyzer.LAYER_SENSITIVITY.get(
            analyzer.classify_layer(layer_name), 0.5
        )

        plans.append(LayerQuantPlan(
            layer_idx=layer_idx,
            layer_type=analyzer.classify_layer(layer_name),
            weight_dtype=chosen_precision,
            activation_dtype="float16",
            sensitivity_score=sensitivity,
            compression_ratio=compression,
        ))
        total_compression += compression

    avg_compression = total_compression / max(num_layers, 1)
    return MixedPrecisionPlan(
        plans=plans,
        overall_compression_ratio=avg_compression,
        num_layers=num_layers,
    )
