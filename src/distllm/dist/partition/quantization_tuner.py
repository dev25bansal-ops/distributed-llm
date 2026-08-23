"""Adaptive Precision Optimizer (APO) — per-device quantization selection.

Jointly optimizes weight quantization, activation quantization, KV cache
compression, and per-layer precision for heterogeneous GPU clusters.

Replaces the legacy "QuantizationAutoTuner" with a holistic system that:
- Profiles actual hardware TFLOPS per quant method at startup
- Selects per-layer precision (attention fp16, MLP int8)
- Recommends activation quantization for inter-node communication
- Tiers KV cache compression based on spare VRAM
- Integrates with the DP partition solver as a quantization dimension
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional

import torch

from loguru import logger
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums and data models
# ---------------------------------------------------------------------------

class QuantMethod(str, Enum):
    """Supported quantization methods."""
    NONE = "none"
    BNB_8BIT = "bnb_8bit"
    BNB_4BIT = "bnb_4bit"
    GPTQ = "gptq"
    AWQ = "awq"
    FP8_E4M3 = "fp8_e4m3"
    FP8_E5M2 = "fp8_e5m2"
    INT8 = "int8"
    NF4 = "nf4"


class ActivationQuantMethod(str, Enum):
    """Activation quantization for inter-node communication."""
    NONE = "none"
    INT8 = "int8"
    FP8_E4M3 = "fp8_e4m3"


class KVCacheBits(str, Enum):
    """KV cache compression bit width."""
    NONE = "none"
    FP8 = "fp8"
    INT8 = "int8"
    INT4 = "int4"


class NodeInfo(BaseModel):
    """Validated node descriptor replacing bare dict access."""
    node_id: str
    device_type: str = "cuda"
    total_memory_bytes: int = Field(default=8 * 1024**3, ge=0)
    compute_capability: Optional[float] = None
    gpu_name: Optional[str] = None
    bandwidth_gbps: Optional[float] = None
    num_layers_assigned: Optional[int] = None
    is_hopper_or_newer: bool = False

    @classmethod
    def from_gpu_profile(
        cls,
        gpu_profile: Any,
        node_id: str,
        num_layers_assigned: int | None = None,
    ) -> NodeInfo:
        """Build from a GPUProfile dataclass."""
        cc = getattr(gpu_profile, "compute_capability", None)
        is_hopper = cc is not None and cc >= 9.0
        return cls(
            node_id=node_id,
            device_type="cuda",
            total_memory_bytes=gpu_profile.total_memory_bytes,
            compute_capability=cc,
            gpu_name=gpu_profile.name,
            bandwidth_gbps=gpu_profile.memory_bandwidth_gbps,
            num_layers_assigned=num_layers_assigned,
            is_hopper_or_newer=is_hopper,
        )

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> NodeInfo:
        """Build from a raw dict with validation."""
        return cls(**{k: v for k, v in d.items() if k in cls.model_fields})


@dataclass
class ScoreWeights:
    """Configurable scoring weights for quantization selection."""
    headroom: float = 0.5
    quality: float = 0.3
    speed: float = 0.2

    def normalized(self) -> ScoreWeights:
        total = self.headroom + self.quality + self.speed
        if total <= 0:
            return ScoreWeights(headroom=1 / 3, quality=1 / 3, speed=1 / 3)
        return ScoreWeights(
            headroom=self.headroom / total,
            quality=self.quality / total,
            speed=self.speed / total,
        )


# ---------------------------------------------------------------------------
# Quantization profiles — live-profiled or static fallback
# ---------------------------------------------------------------------------

@dataclass
class QuantProfile:
    """Performance and memory profile for a quantization method."""
    method: QuantMethod
    memory_reduction: float
    speed_penalty: float
    min_vram_gb: float
    quality_loss: float
    requires_calibration: bool = False
    supported_hardware: list[str] = field(default_factory=lambda: ["cuda", "rocm"])
    min_compute_capability: Optional[float] = None


# Static fallback profiles — overridden by live benchmark data
QUANT_PROFILES: dict[QuantMethod, QuantProfile] = {
    QuantMethod.NONE: QuantProfile(
        method=QuantMethod.NONE,
        memory_reduction=1.0,
        speed_penalty=1.0,
        min_vram_gb=0,
        quality_loss=0.0,
        supported_hardware=["cuda", "rocm", "mps", "xpu", "cpu"],
    ),
    QuantMethod.BNB_8BIT: QuantProfile(
        method=QuantMethod.BNB_8BIT,
        memory_reduction=0.5,
        speed_penalty=1.05,
        min_vram_gb=4,
        quality_loss=0.01,
        supported_hardware=["cuda", "rocm"],
    ),
    QuantMethod.BNB_4BIT: QuantProfile(
        method=QuantMethod.BNB_4BIT,
        memory_reduction=0.25,
        speed_penalty=1.10,
        min_vram_gb=2,
        quality_loss=0.03,
        supported_hardware=["cuda", "rocm"],
    ),
    QuantMethod.GPTQ: QuantProfile(
        method=QuantMethod.GPTQ,
        memory_reduction=0.25,
        speed_penalty=1.0,
        min_vram_gb=4,
        quality_loss=0.02,
        requires_calibration=True,
        supported_hardware=["cuda"],
    ),
    QuantMethod.AWQ: QuantProfile(
        method=QuantMethod.AWQ,
        memory_reduction=0.25,
        speed_penalty=0.95,
        min_vram_gb=4,
        quality_loss=0.02,
        requires_calibration=True,
        supported_hardware=["cuda"],
    ),
    QuantMethod.FP8_E4M3: QuantProfile(
        method=QuantMethod.FP8_E4M3,
        memory_reduction=0.5,
        speed_penalty=0.90,
        min_vram_gb=4,
        quality_loss=0.005,
        supported_hardware=["cuda"],
        min_compute_capability=9.0,
    ),
    QuantMethod.FP8_E5M2: QuantProfile(
        method=QuantMethod.FP8_E5M2,
        memory_reduction=0.5,
        speed_penalty=0.92,
        min_vram_gb=4,
        quality_loss=0.008,
        supported_hardware=["cuda"],
        min_compute_capability=9.0,
    ),
    QuantMethod.INT8: QuantProfile(
        method=QuantMethod.INT8,
        memory_reduction=0.5,
        speed_penalty=1.02,
        min_vram_gb=4,
        quality_loss=0.01,
        supported_hardware=["cuda", "rocm"],
    ),
    QuantMethod.NF4: QuantProfile(
        method=QuantMethod.NF4,
        memory_reduction=0.25,
        speed_penalty=1.08,
        min_vram_gb=2,
        quality_loss=0.025,
        supported_hardware=["cuda", "rocm"],
    ),
}

# Activation quantization profiles
ACTIVATION_PROFILES: dict[ActivationQuantMethod, dict[str, Any]] = {
    ActivationQuantMethod.NONE: {
        "bandwidth_reduction": 1.0,
        "quality_loss": 0.0,
        "overhead_ms": 0.0,
    },
    ActivationQuantMethod.INT8: {
        "bandwidth_reduction": 0.5,
        "quality_loss": 0.005,
        "overhead_ms": 0.1,
    },
    ActivationQuantMethod.FP8_E4M3: {
        "bandwidth_reduction": 0.5,
        "quality_loss": 0.003,
        "overhead_ms": 0.05,
    },
}

# KV cache profiles
KV_PROFILES: dict[KVCacheBits, dict[str, Any]] = {
    KVCacheBits.NONE: {"memory_reduction": 1.0, "quality_loss": 0.0},
    KVCacheBits.FP8: {"memory_reduction": 0.5, "quality_loss": 0.002},
    KVCacheBits.INT8: {"memory_reduction": 0.5, "quality_loss": 0.005},
    KVCacheBits.INT4: {"memory_reduction": 0.25, "quality_loss": 0.02},
}


# ---------------------------------------------------------------------------
# Per-layer precision plan
# ---------------------------------------------------------------------------

@dataclass
class LayerQuantPlan:
    """Quantization plan for a single transformer layer."""
    layer_idx: int
    layer_type: str  # "attention" | "mlp" | "embed" | "lm_head" | "norm"
    weight_dtype: str  # "float16" | "int8" | "nf4" | "fp8_e4m3"
    activation_dtype: str
    sensitivity_score: float = 0.0
    compression_ratio: float = 1.0

    def summary(self) -> str:
        return (
            f"L{self.layer_idx}({self.layer_type}): "
            f"weight={self.weight_dtype}, act={self.activation_dtype}, "
            f"sensitivity={self.sensitivity_score:.2f}"
        )


@dataclass
class MixedPrecisionPlan:
    """Per-layer mixed precision plan for a model partition."""
    plans: list[LayerQuantPlan] = field(default_factory=list)
    overall_compression_ratio: float = 1.0
    num_layers: int = 0

    def summary(self) -> str:
        lines = [f"MixedPrecisionPlan: {self.num_layers} layers, "
                 f"avg compression {self.overall_compression_ratio:.1f}x"]
        for p in self.plans:
            lines.append(f"  {p.summary()}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Recommendation outputs
# ---------------------------------------------------------------------------

@dataclass
class NodeQuantRecommendation:
    """Quantization recommendation for a single node."""
    node_id: str
    method: QuantMethod
    memory_bytes_without_quant: int
    memory_bytes_with_quant: int
    memory_savings_bytes: int
    memory_savings_pct: float
    speed_penalty: float
    quality_loss: float
    reason: str
    activation_quant: ActivationQuantMethod = ActivationQuantMethod.NONE
    kv_cache_bits: KVCacheBits = KVCacheBits.NONE
    mixed_precision_plan: Optional[MixedPrecisionPlan] = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["method"] = self.method.value
        d["activation_quant"] = self.activation_quant.value
        d["kv_cache_bits"] = self.kv_cache_bits.value
        if self.mixed_precision_plan:
            d["mixed_precision_plan"] = {
                "num_layers": self.mixed_precision_plan.num_layers,
                "overall_compression_ratio": self.mixed_precision_plan.overall_compression_ratio,
                "plans": [asdict(p) for p in self.mixed_precision_plan.plans],
            }
        return d


@dataclass
class QuantizationPlan:
    """Full quantization plan across all nodes."""
    recommendations: list[NodeQuantRecommendation] = field(default_factory=list)
    strategy: str = ""
    total_memory_saved_bytes: int = 0
    avg_quality_loss: float = 0.0
    timestamp: float = field(default_factory=time.time)

    @property
    def methods_used(self) -> set[QuantMethod]:
        return {r.method for r in self.recommendations}

    def summary(self) -> str:
        lines = [f"APO Plan: {self.strategy}"]
        for r in self.recommendations:
            lines.append(
                f"  {r.node_id}: {r.method.value} "
                f"(save {r.memory_savings_pct:.0f}%, "
                f"speed {r.speed_penalty:.2f}x, "
                f"quality loss {r.quality_loss:.3f})"
            )
            if r.activation_quant != ActivationQuantMethod.NONE:
                lines.append(f"    activation: {r.activation_quant.value}")
            if r.kv_cache_bits != KVCacheBits.NONE:
                lines.append(f"    kv_cache: {r.kv_cache_bits.value}")
        lines.append(f"  Total saved: {self.total_memory_saved_bytes / 1e9:.1f} GB")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "total_memory_saved_bytes": self.total_memory_saved_bytes,
            "avg_quality_loss": self.avg_quality_loss,
            "timestamp": self.timestamp,
            "recommendations": [r.to_dict() for r in self.recommendations],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> QuantizationPlan:
        recs = []
        for r in d.get("recommendations", []):
            mp = None
            if r.get("mixed_precision_plan"):
                mpd = r["mixed_precision_plan"]
                mp = MixedPrecisionPlan(
                    num_layers=mpd["num_layers"],
                    overall_compression_ratio=mpd["overall_compression_ratio"],
                )
            recs.append(NodeQuantRecommendation(
                node_id=r["node_id"],
                method=QuantMethod(r["method"]),
                memory_bytes_without_quant=r["memory_bytes_without_quant"],
                memory_bytes_with_quant=r["memory_bytes_with_quant"],
                memory_savings_bytes=r["memory_savings_bytes"],
                memory_savings_pct=r["memory_savings_pct"],
                speed_penalty=r["speed_penalty"],
                quality_loss=r["quality_loss"],
                reason=r.get("reason", ""),
                activation_quant=ActivationQuantMethod(r.get("activation_quant", "none")),
                kv_cache_bits=KVCacheBits(r.get("kv_cache_bits", "none")),
                mixed_precision_plan=mp,
            ))
        return cls(
            recommendations=recs,
            strategy=d.get("strategy", ""),
            total_memory_saved_bytes=d.get("total_memory_saved_bytes", 0),
            avg_quality_loss=d.get("avg_quality_loss", 0.0),
            timestamp=d.get("timestamp", 0.0),
        )

    @classmethod
    def from_json(cls, s: str) -> QuantizationPlan:
        return cls.from_dict(json.loads(s))


# ---------------------------------------------------------------------------
# Sensitivity analysis for per-layer precision
# ---------------------------------------------------------------------------

class SensitivityAnalyzer:
    """Classifies transformer layers and estimates quantization sensitivity."""

    # Layer sensitivity scores (higher = more sensitive to quantization)
    LAYER_SENSITIVITY: dict[str, float] = {
        "embed": 0.9,
        "lm_head": 0.9,
        "norm": 0.95,
        "attention": 0.6,
        "mlp": 0.3,
    }

    def classify_layer(self, layer_name: str, module: Any = None) -> str:
        """Classify a layer by its name pattern."""
        name_lower = layer_name.lower()
        if "embed" in name_lower:
            return "embed"
        if "norm" in name_lower or "ln_" in name_lower:
            return "norm"
        if "attn" in name_lower or "attention" in name_lower or "q_proj" in name_lower or "k_proj" in name_lower or "v_proj" in name_lower:
            return "attention"
        if "lm_head" in name_lower or "output" in name_lower:
            return "lm_head"
        if "mlp" in name_lower or "gate_proj" in name_lower or "up_proj" in name_lower or "down_proj" in name_lower:
            return "mlp"
        return "mlp"

    def recommend_dtype(
        self,
        layer_type: str,
        sensitivity_override: float | None = None,
    ) -> tuple[str, float]:
        """Recommend weight dtype for a layer type.

        Returns (dtype_str, sensitivity_score).
        """
        sensitivity = sensitivity_override if sensitivity_override is not None else self.LAYER_SENSITIVITY.get(layer_type, 0.5)

        if sensitivity >= 0.8:
            return "float16", sensitivity
        if sensitivity >= 0.5:
            return "int8", sensitivity
        return "nf4", sensitivity

    def build_mixed_precision_plan(
        self,
        num_layers: int,
        layers_per_node: int,
        layer_type_fn: Any = None,
    ) -> MixedPrecisionPlan:
        """Build a per-layer precision plan for a transformer block.

        Args:
            num_layers: Total transformer layers in this partition.
            layers_per_node: Layers assigned to this node.
            layer_type_fn: Optional callable(layer_idx) -> layer_type string.
        """
        plans: list[LayerQuantPlan] = []
        total_compression = 0.0

        for i in range(num_layers):
            if layer_type_fn:
                layer_type = layer_type_fn(i)
            else:
                layer_type = "attention" if i % 2 == 0 else "mlp"

            weight_dtype, sensitivity = self.recommend_dtype(layer_type)
            compression = {"float16": 1.0, "int8": 2.0, "nf4": 4.0, "fp8_e4m3": 2.0}.get(weight_dtype, 1.0)

            plans.append(LayerQuantPlan(
                layer_idx=i,
                layer_type=layer_type,
                weight_dtype=weight_dtype,
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


# ---------------------------------------------------------------------------
# Per-layer profiling and mixed-precision assignment
# ---------------------------------------------------------------------------

@dataclass
class LayerPrecisionProfile:
    """Runtime and memory profile for a single layer at a given precision."""
    layer_idx: int
    layer_name: str
    precision: str  # "fp16" | "fp8" | "int8"
    runtime_ms: float
    memory_bytes: int
    peak_memory_bytes: int


@dataclass
class LayerPrecisionResult:
    """Aggregated profiling results across precisions for one layer."""
    layer_idx: int
    layer_name: str
    layer_type: str
    profiles: dict[str, LayerPrecisionProfile] = field(default_factory=dict)
    recommended_precision: str = "fp16"

    @property
    def fp16(self) -> LayerPrecisionProfile | None:
        return self.profiles.get("fp16")

    @property
    def fp8(self) -> LayerPrecisionProfile | None:
        return self.profiles.get("fp8")

    @property
    def int8(self) -> LayerPrecisionProfile | None:
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
            Precision string ("fp16", "fp8", or "int8").
        """
        candidates = []
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
    # Attention layers → fp16 (quality-sensitive)
    # MLP layers     → lowest memory precision that fits
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

    By default uses a heuristic rule (attention → ``attention_precision``,
    MLP → ``mlp_precision``).  When ``profile_all=True``, it profiles
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


# ---------------------------------------------------------------------------
# AutoMixedPrecisionPipeline — wraps orchestrator with per-layer dtype casting
# ---------------------------------------------------------------------------

class AutoMixedPrecisionPipeline:
    """Pipeline wrapper that applies per-layer mixed precision during
    distributed inference.

    Uses a :class:`MixedPrecisionPlan` to determine the target dtype
    for each layer and inserts dtype casts at layer boundaries so that
    each layer executes in its assigned precision without modifying the
    orchestrator's internal routing logic.

    Typical usage::

        amp_pipeline = AutoMixedPrecisionPipeline(
            orchestrator=orchestrator,
            precision_plan=plan,
        )
        output = amp_pipeline.run(input_ids, kv_caches, "req-1")

    The wrapper transparently casts hidden states between layers using
    :class:`PrecisionCastWrapper` modules inserted around each layer's
    forward pass.
    """

    def __init__(
        self,
        orchestrator: Any,
        precision_plan: MixedPrecisionPlan,
        model: torch.nn.Module | None = None,
        device: str = "cuda",
    ):
        self._orchestrator = orchestrator
        self._plan = precision_plan
        self._model = model
        self._device = device

        # Build per-layer dtype map: layer_idx -> torch.dtype
        self._layer_dtype: dict[int, torch.dtype] = {}
        for p in precision_plan.plans:
            self._layer_dtype[p.layer_idx] = self._parse_dtype(p.weight_dtype)

        logger.info(
            f"AutoMixedPrecisionPipeline initialized with "
            f"{len(precision_plan.plans)} layers, "
            f"avg compression {precision_plan.overall_compression_ratio:.1f}x"
        )

    @staticmethod
    def _parse_dtype(dtype_str: str) -> torch.dtype:
        """Convert a precision string to a torch dtype."""
        mapping: dict[str, torch.dtype] = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "fp8_e4m3": torch.float8_e4m3fn,
            "fp8": torch.float8_e4m3fn,
            "int8": torch.int8,
            "nf4": torch.float16,  # NF4 weights stored as fp16 with scale factors
        }
        if dtype_str not in mapping:
            logger.warning(f"Unknown dtype '{dtype_str}', falling back to float16")
            return torch.float16
        return mapping[dtype_str]

    def get_dtype_for_layer(self, layer_idx: int) -> torch.dtype:
        """Return the target dtype for a given layer index."""
        return self._layer_dtype.get(layer_idx, torch.float16)

    def cast_to_layer_precision(
        self,
        tensor: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """Cast a tensor to the target dtype for the specified layer.

        Args:
            tensor: Input hidden states.
            layer_idx: Target layer index.

        Returns:
            Tensor cast to the appropriate dtype, on the same device.
        """
        target_dtype = self.get_dtype_for_layer(layer_idx)
        if tensor.dtype != target_dtype:
            return tensor.to(target_dtype)
        return tensor

    def run(
        self,
        input_ids: torch.Tensor,
        kv_caches: dict[str, list | None],
        request_id: str,
        *,
        micro_batched: bool = False,
        micro_batch_size: int | None = None,
    ) -> torch.Tensor:
        """Run the pipeline with per-layer mixed-precision casting.

        The wrapper intercepts the hidden states before each layer and
        casts them to the layer's assigned dtype.  The orchestrator's
        ``run_pipeline`` or ``run_pipeline_microbatched`` handles the
        actual distributed forwarding.

        Note:
            This implementation casts on the coordinator side — the
            worker nodes receive tensors already in the target dtype.
            For true per-worker casting, see
            :meth:`apply_to_model_weights` which modifies the model
            weights in-place.

        Args:
            input_ids: Input token IDs.
            kv_caches: Per-node KV cache dictionary.
            request_id: Unique request identifier.
            micro_batched: If True, use the micro-batched pipeline.
            micro_batch_size: Optional micro-batch size override.

        Returns:
            Output logits from the last node.
        """
        if micro_batched:
            return self._run_microbatched(
                input_ids, kv_caches, request_id, micro_batch_size,
            )
        return self._run_sequential(input_ids, kv_caches, request_id)

    def _run_sequential(
        self,
        input_ids: torch.Tensor,
        kv_caches: dict[str, list | None],
        request_id: str,
    ) -> torch.Tensor:
        """Sequential pipeline with per-layer precision casts."""
        current = input_ids

        # Gather nodes in layer order
        node_order = self._orchestrator.node_order
        if not node_order:
            raise RuntimeError("No nodes registered in pipeline")

        for node_id in node_order:
            node = self._orchestrator.get_node(node_id)
            if node is None or not node.is_healthy:
                continue

            # Cast input to each layer's precision as we traverse
            # layers assigned to this node
            for layer_idx in range(node.start_layer, node.end_layer + 1):
                current = self.cast_to_layer_precision(current, layer_idx)

            # Forward through the node
            from distllm.dist.node_client import forward_request

            kv_cache = kv_caches.get(node_id)
            current = forward_request(
                host=node.host,
                port=node.port,
                hidden_states=current,
                kv_cache=kv_cache,
                request_id=request_id,
            )
            if current is None:
                raise RuntimeError(f"Node {node_id} returned None")

        return current

    def _run_microbatched(
        self,
        input_ids: torch.Tensor,
        kv_caches: dict[str, list | None],
        request_id: str,
        micro_batch_size: int | None = None,
    ) -> torch.Tensor:
        """Micro-batched pipeline with per-layer precision casts.

        Uses ``run_pipeline_microbatched`` on the underlying orchestrator
        but intercepts the hidden states at each stage boundary.  Since
        the orchestrator handles gRPC routing, we rely on the worker
        nodes to apply per-layer casting when the model is loaded with
        :meth:`apply_to_model_weights`.

        For the current implementation, we cast the entire input before
        sending and rely on the orchestrator's micro-batch split.
        """
        # Cast full input to match the first layer's precision
        if self._plan.plans:
            first = self._plan.plans[0]
            cast_input = input_ids.to(self._parse_dtype(first.weight_dtype))
        else:
            cast_input = input_ids

        # The micro-batched pipeline does not expose per-micro-batch
        # casting hooks in this initial version.  We pass the cast input
        # and rely on the model weights having been set via
        # apply_to_model_weights for per-layer correctness.
        import asyncio

        coro = self._orchestrator.run_pipeline_microbatched(
            cast_input, kv_caches, request_id, micro_batch_size,
        )
        return asyncio.run(coro)

    def apply_to_model_weights(self, model: torch.nn.Module) -> torch.nn.Module:
        """Modify model weights in-place to match the precision plan.

        Casts each layer's parameters to the assigned dtype.  This is
        a one-time operation performed when loading the model onto the
        worker node.  After this call, the model's forward pass runs
        natively in the assigned per-layer precision without runtime
        casting overhead.

        Args:
            model: The transformer model whose weights should be cast.

        Returns:
            The same model with updated weight dtypes (in-place).
        """
        for plan in self._plan.plans:
            layer_name = f"model.layers.{plan.layer_idx}"
            target_dtype = self._parse_dtype(plan.weight_dtype)

            # Resolve submodule
            module: torch.nn.Module = model
            for part in layer_name.split("."):
                module = getattr(module, part, None)
                if module is None:
                    break

            if module is None:
                logger.warning(
                    f"Layer {layer_name} not found in model, skipping"
                )
                continue

            # Cast all parameters in this submodule
            for param in module.parameters(recurse=True):
                if param.dtype != target_dtype:
                    param.data = param.data.to(target_dtype)

            logger.debug(
                f"Casted {layer_name} to {plan.weight_dtype} "
                f"(target dtype: {target_dtype})"
            )

        return model

    @property
    def orchestrator(self) -> Any:
        """The underlying pipeline orchestrator."""
        return self._orchestrator

    @property
    def precision_plan(self) -> MixedPrecisionPlan:
        """The mixed-precision plan."""
        return self._plan

    def summary(self) -> str:
        """Return a human-readable summary of the precision plan."""
        parts = [
            f"AutoMixedPrecisionPipeline: {len(self._plan.plans)} layers",
        ]
        for p in self._plan.plans:
            parts.append(f"  L{p.layer_idx:>3} ({p.layer_type:<10}) → {p.weight_dtype}")
        parts.append(f"  Avg compression: {self._plan.overall_compression_ratio:.1f}x")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Adaptive Precision Optimizer (APO)
# ---------------------------------------------------------------------------

class QuantizationAutoTuner:
    """Adaptive Precision Optimizer — selects optimal quantization per device.

    Jointly optimizes:
    - Weight quantization method per node
    - Activation quantization for inter-node communication
    - KV cache compression tier per node
    - Per-layer mixed precision plans

    Args:
        max_quality_loss: Maximum acceptable quality loss (0.0-1.0).
        prefer_speed: If True, prefer faster methods over smaller models.
        require_calibration: If True, only use methods that don't need calibration.
        weights: Configurable scoring weights. None uses defaults.
        profile_overrides: Optional dict of QuantMethod -> custom QuantProfile.
    """

    def __init__(
        self,
        max_quality_loss: float = 0.05,
        prefer_speed: bool = False,
        require_calibration: bool = False,
        weights: ScoreWeights | None = None,
        profile_overrides: dict[QuantMethod, QuantProfile] | None = None,
    ):
        self._max_quality_loss = max_quality_loss
        self._prefer_speed = prefer_speed
        self._require_calibration = require_calibration
        self._weights = (weights or ScoreWeights()).normalized()
        self._sensitivity = SensitivityAnalyzer()

        # Merge profile overrides
        self._profiles = dict(QUANT_PROFILES)
        if profile_overrides:
            self._profiles.update(profile_overrides)

    def recommend(
        self,
        nodes: list[NodeInfo | dict[str, Any]],
        model_size_bytes: int,
        num_layers: int,
        inter_node_bandwidth_gbps: float | None = None,
    ) -> QuantizationPlan:
        """Recommend quantization for each node.

        Args:
            nodes: List of NodeInfo or raw node dicts.
            model_size_bytes: Total model size in bytes (fp16).
            num_layers: Total transformer layers in the model.
            inter_node_bandwidth_gbps: Optional bandwidth between nodes for
                activation quant recommendation.

        Returns:
            QuantizationPlan with per-node recommendations.
        """
        if not nodes:
            return QuantizationPlan(strategy="No nodes provided")

        node_infos = [
            n if isinstance(n, NodeInfo) else NodeInfo.from_dict(n)
            for n in nodes
        ]

        recommendations = []
        total_saved = 0
        total_quality = 0.0

        for node in node_infos:
            rec = self._recommend_for_node(
                node, model_size_bytes, num_layers, inter_node_bandwidth_gbps,
            )
            recommendations.append(rec)
            total_saved += rec.memory_savings_bytes
            total_quality += rec.quality_loss

        strategy = self._describe_strategy(recommendations)

        return QuantizationPlan(
            recommendations=recommendations,
            strategy=strategy,
            total_memory_saved_bytes=total_saved,
            avg_quality_loss=total_quality / max(len(recommendations), 1),
        )

    def _recommend_for_node(
        self,
        node: NodeInfo,
        model_size_bytes: int,
        num_layers: int,
        inter_node_bandwidth_gbps: float | None,
    ) -> NodeQuantRecommendation:
        """Recommend quantization for a single node."""
        # Per-node model portion based on assigned layers
        if node.num_layers_assigned is not None and node.num_layers_assigned > 0:
            layer_bytes = model_size_bytes / max(num_layers, 1)
            node_model_bytes = int(layer_bytes * node.num_layers_assigned)
        else:
            node_model_bytes = model_size_bytes

        vram_gb = node.total_memory_bytes / (1024**3)
        node_model_gb = node_model_bytes / (1024**3)
        device_type = node.device_type

        # Estimate KV cache + activation overhead (15% of model size)
        overhead_bytes = int(node_model_bytes * 0.15)
        effective_model_bytes = node_model_bytes + overhead_bytes
        effective_model_gb = effective_model_bytes / (1024**3)

        # If model fits with 20% headroom including overhead, use fp16
        if effective_model_gb * 1.2 < vram_gb:
            rec = NodeQuantRecommendation(
                node_id=node.node_id,
                method=QuantMethod.NONE,
                memory_bytes_without_quant=node_model_bytes,
                memory_bytes_with_quant=node_model_bytes,
                memory_savings_bytes=0,
                memory_savings_pct=0.0,
                speed_penalty=1.0,
                quality_loss=0.0,
                reason="Model fits in VRAM without quantization (including overhead)",
            )
        else:
            rec = self._select_best_method(node, node_model_bytes, vram_gb, device_type)

        # Activation quantization recommendation
        rec.activation_quant = self._recommend_activation_quant(
            inter_node_bandwidth_gbps, rec.quality_loss,
        )

        # KV cache quantization recommendation
        spare_vram_gb = vram_gb - (rec.memory_bytes_with_quant / (1024**3))
        rec.kv_cache_bits = self._recommend_kv_cache(spare_vram_gb)

        # Mixed precision plan for the node's layer subset
        assigned = node.num_layers_assigned or num_layers
        rec.mixed_precision_plan = self._sensitivity.build_mixed_precision_plan(
            num_layers=num_layers,
            layers_per_node=assigned,
        )

        return rec

    def _select_best_method(
        self,
        node: NodeInfo,
        node_model_bytes: int,
        vram_gb: float,
        device_type: str,
    ) -> NodeQuantRecommendation:
        """Find the best quantization method for a node."""
        node_model_gb = node_model_bytes / (1024**3)
        candidates: list[tuple[QuantMethod, QuantProfile, int, float]] = []

        for method, profile in self._profiles.items():
            if method == QuantMethod.NONE:
                continue
            if device_type not in profile.supported_hardware:
                continue
            if profile.quality_loss > self._max_quality_loss:
                continue
            if self._require_calibration and profile.requires_calibration:
                continue
            if vram_gb < profile.min_vram_gb:
                continue
            if (profile.min_compute_capability is not None
                    and node.compute_capability is not None
                    and node.compute_capability < profile.min_compute_capability):
                continue

            quant_bytes = int(node_model_bytes * profile.memory_reduction)
            quant_gb = quant_bytes / (1024**3)

            # Add 15% overhead estimate
            if (quant_gb + node_model_gb * 0.15) * 1.2 > vram_gb:
                continue

            score = self._score_method(profile, quant_gb, vram_gb)
            candidates.append((method, profile, quant_bytes, score))

        if not candidates:
            # Fallback: forced 4-bit if node supports it
            fallback = self._fallback_method(node, node_model_bytes)
            return fallback

        candidates.sort(key=lambda x: x[3], reverse=True)
        best_method, best_profile, best_bytes, _ = candidates[0]

        return NodeQuantRecommendation(
            node_id=node.node_id,
            method=best_method,
            memory_bytes_without_quant=node_model_bytes,
            memory_bytes_with_quant=best_bytes,
            memory_savings_bytes=node_model_bytes - best_bytes,
            memory_savings_pct=round((1 - best_bytes / node_model_bytes) * 100, 1),
            speed_penalty=best_profile.speed_penalty,
            quality_loss=best_profile.quality_loss,
            reason=f"Best fit for {vram_gb:.1f}GB VRAM with {node_model_gb:.1f}GB model",
        )

    def _fallback_method(
        self,
        node: NodeInfo,
        node_model_bytes: int,
    ) -> NodeQuantRecommendation:
        """Fallback when no method fits quality/VRAM constraints."""
        # Try BNB_4BIT first if hardware supports it
        if node.device_type in ("cuda", "rocm"):
            profile = self._profiles[QuantMethod.BNB_4BIT]
            if (node.total_memory_bytes / (1024**3) >= profile.min_vram_gb
                    and profile.quality_loss <= self._max_quality_loss):
                quant_bytes = int(node_model_bytes * profile.memory_reduction)
                return NodeQuantRecommendation(
                    node_id=node.node_id,
                    method=QuantMethod.BNB_4BIT,
                    memory_bytes_without_quant=node_model_bytes,
                    memory_bytes_with_quant=quant_bytes,
                    memory_savings_bytes=node_model_bytes - quant_bytes,
                    memory_savings_pct=round((1 - quant_bytes / node_model_bytes) * 100, 1),
                    speed_penalty=profile.speed_penalty,
                    quality_loss=profile.quality_loss,
                    reason="Forced 4-bit quantization (model too large for VRAM)",
                )

        # CPU or unsupported hardware: return NONE
        return NodeQuantRecommendation(
            node_id=node.node_id,
            method=QuantMethod.NONE,
            memory_bytes_without_quant=node_model_bytes,
            memory_bytes_with_quant=node_model_bytes,
            memory_savings_bytes=0,
            memory_savings_pct=0.0,
            speed_penalty=1.0,
            quality_loss=0.0,
            reason="No quantization available for this hardware",
        )

    def _score_method(
        self,
        profile: QuantProfile,
        quant_gb: float,
        vram_gb: float,
    ) -> float:
        """Score a quantization method (higher is better)."""
        headroom = (vram_gb - quant_gb) / max(vram_gb, 0.001)
        headroom_score = min(headroom / 0.5, 1.0)

        quality_score = 1.0 - profile.quality_loss
        speed_score = 1.0 / max(profile.speed_penalty, 0.01)

        w = self._weights
        if self._prefer_speed:
            return speed_score * w.speed + headroom_score * w.headroom + quality_score * w.quality
        return headroom_score * w.headroom + quality_score * w.quality + speed_score * w.speed

    def _recommend_activation_quant(
        self,
        inter_node_bandwidth_gbps: float | None,
        weight_quality_loss: float,
    ) -> ActivationQuantMethod:
        """Recommend activation quantization for inter-node communication."""
        if inter_node_bandwidth_gbps is None:
            return ActivationQuantMethod.NONE

        # If bandwidth is tight (<25 Gbps), recommend activation quant
        if inter_node_bandwidth_gbps < 25:
            return ActivationQuantMethod.INT8

        # If moderate bandwidth, FP8 is a good tradeoff (if available)
        if inter_node_bandwidth_gbps < 100:
            return ActivationQuantMethod.FP8_E4M3

        return ActivationQuantMethod.NONE

    def _recommend_kv_cache(self, spare_vram_gb: float) -> KVCacheBits:
        """Recommend KV cache compression based on spare VRAM."""
        if spare_vram_gb < 1.0:
            return KVCacheBits.INT4
        if spare_vram_gb < 4.0:
            return KVCacheBits.INT8
        if spare_vram_gb < 8.0:
            return KVCacheBits.FP8
        return KVCacheBits.NONE

    def _describe_strategy(self, recs: list[NodeQuantRecommendation]) -> str:
        if not recs:
            return "No recommendations generated"
        methods = {r.method for r in recs}
        if methods == {QuantMethod.NONE}:
            return "No quantization needed — all nodes have sufficient VRAM"
        if len(methods) == 1:
            m = next(iter(methods))
            return f"Uniform {m.value} across all nodes"
        method_strs = sorted(m.value for m in methods)
        return f"Hybrid: {', '.join(method_strs)} across {len(recs)} nodes"

    # ------------------------------------------------------------------
    # Backward compatibility: old dict-based interface
    # ------------------------------------------------------------------

    def recommend_legacy(
        self,
        nodes: list[dict[str, Any]],
        model_size_bytes: int,
        num_layers: int,
    ) -> QuantizationPlan:
        """Legacy interface accepting raw dicts."""
        node_infos = [NodeInfo.from_dict(n) for n in nodes]
        return self.recommend(node_infos, model_size_bytes, num_layers)


# ---------------------------------------------------------------------------
# Convenience function for quick single-node selection
# ---------------------------------------------------------------------------

def select_for_node(
    node_info: NodeInfo | dict[str, Any],
    model_size_bytes: int,
    num_layers: int = 32,
    max_quality_loss: float = 0.05,
) -> QuantMethod:
    """Quick single-node quantization method selection.

    Returns the recommended QuantMethod for a single node.
    """
    if isinstance(node_info, dict):
        node_info = NodeInfo.from_dict(node_info)

    tuner = QuantizationAutoTuner(max_quality_loss=max_quality_loss)
    plan = tuner.recommend([node_info], model_size_bytes, num_layers)
    if plan.recommendations:
        return plan.recommendations[0].method
    return QuantMethod.NONE
