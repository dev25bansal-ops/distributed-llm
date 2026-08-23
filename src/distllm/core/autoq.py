"""Per-layer adaptive quantization strategy engine.

Provides a pipeline for analyzing per-layer quantization sensitivity,
selecting quantisation strategies, caching strategy decisions, and
orchestrating the full adaptive quantization workflow.

Classes
-------
SensitivityAnalyzer
    Analyze per-layer sensitivity using hessian-based or
    activation-statistics-based methods.
QuantizationStrategySelector
    Select quantization strategies based on sensitivity and hardware.
StrategyCache
    LRU cache for strategies per model per hardware config.
AutoQ
    Orchestrator combining all components into a full pipeline.
QuantizationPlan
    Immutable plan mapping layer names to strategies.
HardwareConfig
    Hardware capabilities descriptor.
"""

from __future__ import annotations

import collections
import math
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from loguru import logger


# ---------------------------------------------------------------------------
# Optional dependency detection
# ---------------------------------------------------------------------------

_TORCH_AVAILABLE: bool
try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class AutoQError(Exception):
    """Base exception for auto-quantization errors."""


class TorchRequiredError(AutoQError):
    """Raised when torch is required but not installed."""


class CalibrationDataError(AutoQError):
    """Raised when calibration data is invalid."""


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class QuantMethod(str, Enum):
    """Supported quantization methods."""
    INT4 = "int4"
    INT8 = "int8"
    FP8 = "fp8"
    FP16 = "fp16"


class SensitivityLevel(str, Enum):
    """Sensitivity classification levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Data Transfer Objects
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HardwareConfig:
    """Hardware capabilities descriptor.

    Attributes:
        name: Hardware name/label (e.g., ``"a100"``, ``"h100"``,
            ``"rtx4090"``).
        supports_fp8: Whether FP8 is natively supported.
        supports_int8_tensor_core: Whether INT8 Tensor Cores are available.
        supports_int4: Whether INT4 is natively supported.
        tensor_core_bits: Native precision (e.g., 16 for FP16/BF16).
        vram_gb: Available VRAM in GB.
        compute_capability: CUDA compute capability string (e.g., ``"8.0"``).
    """
    name: str = "generic"
    supports_fp8: bool = False
    supports_int8_tensor_core: bool = False
    supports_int4: bool = False
    tensor_core_bits: int = 16
    vram_gb: float = 0.0
    compute_capability: str = ""


@dataclass(frozen=True)
class LayerSensitivity:
    """Sensitivity profile for a single layer.

    Attributes:
        layer_name: Name/path of the layer
            (e.g., ``"model.layers.0.self_attn"``).
        sensitivity_score: Normalized sensitivity score [0, 1].
            1.0 = most sensitive (keep high precision).
            0.0 = least sensitive (safe for aggressive quantization).
        hessian_trace: Trace of Hessian (or Fisher) diagonal (optional).
        activation_variance: Variance of activation outputs (optional).
        activation_range: Dynamic range max/mean ratio (optional).
        recommended_method: Suggested quantization method.
    """
    layer_name: str = ""
    sensitivity_score: float = 0.0
    hessian_trace: float | None = None
    activation_variance: float | None = None
    activation_range: float | None = None
    recommended_method: QuantMethod = QuantMethod.FP16


@dataclass(frozen=True)
class QuantizationStrategy:
    """Quantization strategy for a layer or entire model.

    Attributes:
        method: Quantization method (int4, int8, fp8, fp16).
        target_bits: Effective target bit-width per parameter.
        per_channel: Whether to apply per-channel quantization.
        symmetric: Whether to use symmetric quantization.
        group_size: Group size for group-wise quantization
            (0 = per-tensor).
        reason: Human-readable explanation for this strategy.
    """
    method: QuantMethod = QuantMethod.FP16
    target_bits: int = 16
    per_channel: bool = True
    symmetric: bool = True
    group_size: int = 0
    reason: str = ""

    @classmethod
    def int4(
        cls,
        *,
        per_channel: bool = True,
        symmetric: bool = True,
        group_size: int = 128,
    ) -> QuantizationStrategy:
        """Create an INT4 strategy."""
        return cls(
            method=QuantMethod.INT4,
            target_bits=4,
            per_channel=per_channel,
            symmetric=symmetric,
            group_size=group_size,
            reason="Low sensitivity: safe for INT4 quantization",
        )

    @classmethod
    def int8(
        cls,
        *,
        per_channel: bool = True,
        symmetric: bool = True,
    ) -> QuantizationStrategy:
        """Create an INT8 strategy."""
        return cls(
            method=QuantMethod.INT8,
            target_bits=8,
            per_channel=per_channel,
            symmetric=symmetric,
            reason="Medium sensitivity: INT8 quantization with minimal quality loss",
        )

    @classmethod
    def fp8(
        cls,
        *,
        per_channel: bool = True,
    ) -> QuantizationStrategy:
        """Create an FP8 strategy."""
        return cls(
            method=QuantMethod.FP8,
            target_bits=8,
            per_channel=per_channel,
            symmetric=True,
            reason="High sensitivity: FP8 preserves dynamic range better than INT8",
        )

    @classmethod
    def fp16(cls, *, reason: str = "") -> QuantizationStrategy:
        """Create an FP16 (no quantization) strategy."""
        return cls(
            method=QuantMethod.FP16,
            target_bits=16,
            per_channel=False,
            symmetric=False,
            reason=reason or "Critical sensitivity: keep at full FP16 precision",
        )


@dataclass(frozen=True)
class QuantizationPlan:
    """Complete quantization plan mapping layers to strategies.

    Attributes:
        strategies: Mapping of layer_name -> QuantizationStrategy.
        global_compression_ratio: Estimated compression ratio across
            all layers.
        quality_score: Estimated quality score after applying
            plan [0, 1].
        total_parameters: Total number of parameters in the plan.
        quantized_parameters: Number of parameters affected by
            quantization.
    """
    strategies: dict[str, QuantizationStrategy] = field(default_factory=dict)
    global_compression_ratio: float = 1.0
    quality_score: float = 1.0
    total_parameters: int = 0
    quantized_parameters: int = 0


@dataclass(frozen=True)
class AutoQStats:
    """Statistics snapshot from AutoQ.

    Attributes:
        num_layers: Number of layers in the model.
        quantization_map: Strategy counts per method name.
        global_compression_ratio: Overall compression ratio.
        estimated_quality: Estimated quality score.
        analysis_time_ms: Time taken for sensitivity analysis.
        calibration_samples: Number of calibration samples used.
        hardware: Hardware configuration label.
    """
    num_layers: int = 0
    quantization_map: dict[str, int] = field(default_factory=dict)
    global_compression_ratio: float = 1.0
    estimated_quality: float = 1.0
    analysis_time_ms: float = 0.0
    calibration_samples: int = 0
    hardware: str = ""


# ---------------------------------------------------------------------------
# Sensitivity Analyzer
# ---------------------------------------------------------------------------

class SensitivityAnalyzer:
    """Analyze per-layer sensitivity to quantization.

    Supports two analysis strategies:

    **Activation-statistics-based** (default, no gradients needed):

        Runs a forward pass with calibration data, collects per-layer
        activation statistics (variance, dynamic range), and derives
        sensitivity scores. Layers with high activation variance or
        large dynamic range are marked more sensitive.

    **Hessian-based** (requires gradients):

        Computes the diagonal of the Hessian (or empirical Fisher)
        for each layer's parameters with respect to the loss on
        calibration data. Layers with larger Hessian trace are more
        sensitive to quantization.

    Usage::

        analyzer = SensitivityAnalyzer(method="activation")
        sensitivities = analyzer.analyze(model, calibration_data)
        ranked = analyzer.rank_layers()
        top_sensitive = ranked[:3]
    """

    # Default sensitivity thresholds
    LOW_THRESHOLD: float = 0.30
    MEDIUM_THRESHOLD: float = 0.60
    HIGH_THRESHOLD: float = 0.85
    CRITICAL_THRESHOLD: float = 0.95

    def __init__(
        self,
        method: str = "activation",
        layer_filter: Callable[[str], bool] | None = None,
        hessian_batch_size: int = 4,
    ) -> None:
        """
        Args:
            method: Analysis method --- ``"activation"`` or
                ``"hessian"``.
            layer_filter: Optional callable returning ``True`` for
                layers that should be analyzed. If ``None``, all named
                parameter groups with at least one non-scalar dimension
                are included.
            hessian_batch_size: Micro-batch size for hessian
                computation. Only used when ``method="hessian"``.

        Raises:
            TorchRequiredError: If ``method="hessian"`` without torch.
            ValueError: If method is unknown.
        """
        if method not in ("activation", "hessian"):
            raise ValueError(
                f"Unknown sensitivity method {method!r}. "
                "Expected 'activation' or 'hessian'."
            )
        if method == "hessian" and not _TORCH_AVAILABLE:
            raise TorchRequiredError(
                "Hessian-based sensitivity analysis requires PyTorch. "
                "Install it with: pip install torch"
            )

        self._method = method
        self._layer_filter = layer_filter
        self._hessian_batch_size = hessian_batch_size
        self._sensitivities: dict[str, LayerSensitivity] = {}
        self._analysis_done = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        model: Any,
        calibration_data: Any,
    ) -> list[LayerSensitivity]:
        """Analyze per-layer sensitivity.

        Args:
            model: A PyTorch ``nn.Module`` (anything with
                ``named_parameters()`` and ``forward()``).
            calibration_data: Calibration inputs. For activation
                method: a single tensor or list of tensors. For hessian
                method: tuple of ``(inputs, targets)`` or a dataloader
                yielding ``(inputs, targets)`` pairs.

        Returns:
            List of :class:`LayerSensitivity` entries, one per
            analyzed layer.

        Raises:
            TorchRequiredError: If torch is not installed.
            CalibrationDataError: If calibration data is empty or
                invalid.
        """
        if not _TORCH_AVAILABLE:
            raise TorchRequiredError(
                "PyTorch is required for sensitivity analysis. "
                "Install it with: pip install torch"
            )

        if self._method == "hessian":
            results = self._analyze_hessian(model, calibration_data)
        else:
            results = self._analyze_activation(model, calibration_data)

        self._sensitivities = {r.layer_name: r for r in results}
        self._analysis_done = True
        return results

    def rank_layers(
        self,
        top_k: int | None = None,
    ) -> list[LayerSensitivity]:
        """Return layers sorted by sensitivity score descending.

        Args:
            top_k: If set, return only the top-k most sensitive layers.

        Returns:
            List of :class:`LayerSensitivity` entries.

        Raises:
            RuntimeError: If :meth:`analyze` has not been called yet.
        """
        if not self._analysis_done:
            raise RuntimeError(
                "Call analyze() before rank_layers()."
            )
        ranked = sorted(
            self._sensitivities.values(),
            key=lambda x: x.sensitivity_score,
            reverse=True,
        )
        if top_k is not None:
            return ranked[:top_k]
        return ranked

    def get_sensitivity(self, layer_name: str) -> LayerSensitivity | None:
        """Get sensitivity profile for a specific layer.

        Args:
            layer_name: Name of the layer.

        Returns:
            :class:`LayerSensitivity` or ``None`` if not analyzed.
        """
        return self._sensitivities.get(layer_name)

    def classify(self, score: float) -> SensitivityLevel:
        """Classify a sensitivity score into a qualitative level.

        Args:
            score: Normalized sensitivity score [0, 1].

        Returns:
            :class:`SensitivityLevel`.
        """
        if score >= self.CRITICAL_THRESHOLD:
            return SensitivityLevel.CRITICAL
        if score >= self.HIGH_THRESHOLD:
            return SensitivityLevel.HIGH
        if score >= self.MEDIUM_THRESHOLD:
            return SensitivityLevel.MEDIUM
        return SensitivityLevel.LOW

    @property
    def method(self) -> str:
        """Analysis method in use."""
        return self._method

    @property
    def sensitivities(self) -> dict[str, LayerSensitivity]:
        """Read-only view of computed sensitivities."""
        return dict(self._sensitivities)

    # ------------------------------------------------------------------
    # Activation-statistics-based analysis
    # ------------------------------------------------------------------

    def _analyze_activation(
        self,
        model: torch.nn.Module,
        calibration_data: Any,
    ) -> list[LayerSensitivity]:
        """Analyze sensitivity via activation statistics.

        Registers forward hooks to capture per-layer activation
        tensors, then computes variance and dynamic range as proxies
        for sensitivity.
        """
        calibration_inputs = _normalize_calibration(calibration_data)
        if not calibration_inputs:
            raise CalibrationDataError(
                "Calibration data is empty. Provide at least one sample."
            )

        # Collect layer names whose parameters have spatial dimensions
        layer_names = self._discover_quantizable_layers(model)
        if not layer_names:
            logger.warning("No quantizable layers found in model.")
            return []

        # Register forward hooks
        captured: dict[str, list[torch.Tensor]] = {}
        hooks_handle: list[torch.utils.hooks.RemovableHandle] = []

        def _make_hook(name: str) -> Callable:
            def _hook(
                _module: torch.nn.Module,
                _input: Any,
                output: torch.Tensor,
            ) -> None:
                out = output[0] if isinstance(output, (tuple, list)) else output
                captured.setdefault(name, []).append(out.detach())
            return _hook

        for name, module in model.named_modules():
            if name in layer_names:
                handle = module.register_forward_hook(_make_hook(name))
                hooks_handle.append(handle)

        try:
            # Run calibration forward passes
            model.eval()
            with torch.no_grad():
                for inp in calibration_inputs:
                    if isinstance(inp, (list, tuple)):
                        model(*inp)
                    else:
                        model(inp)
        finally:
            for handle in hooks_handle:
                handle.remove()

        # Compute statistics per layer
        results: list[LayerSensitivity] = []
        for name in layer_names:
            tensors = captured.get(name, [])
            if not tensors:
                continue

            # Concatenate all captured activations along batch dim
            try:
                cat = torch.cat([t.flatten(start_dim=1) for t in tensors], dim=0)
            except RuntimeError:
                # Shape mismatch --- take per-sample stats
                cat = tensors[-1].flatten(start_dim=1)

            if cat.numel() == 0:
                continue

            # Activation variance: mean variance across all features
            var = cat.var(dim=0).mean().item()

            # Dynamic range: ratio of max abs to mean abs
            abs_vals = cat.abs()
            mean_abs = abs_vals.mean().item()
            max_abs = abs_vals.max().item()
            dyn_range = max_abs / max(mean_abs, 1e-12)

            # Normalized sensitivity score [0, 1]
            norm_var = min(1.0, math.log10(1.0 + var) / 4.0)
            norm_range = min(1.0, math.log10(1.0 + dyn_range) / 3.0)
            score = min(1.0, 0.5 * norm_var + 0.5 * norm_range)

            recommended = _recommend_method(score)

            results.append(LayerSensitivity(
                layer_name=name,
                sensitivity_score=round(score, 6),
                activation_variance=round(var, 6),
                activation_range=round(dyn_range, 6),
                recommended_method=recommended,
            ))

        if not results:
            logger.warning(
                "No activation statistics captured. "
                "Model may not have executed forward pass correctly."
            )

        return results

    # ------------------------------------------------------------------
    # Hessian-based analysis
    # ------------------------------------------------------------------

    def _analyze_hessian(
        self,
        model: torch.nn.Module,
        calibration_data: Any,
    ) -> list[LayerSensitivity]:
        """Analyze sensitivity via Hessian trace approximation.

        Uses the empirical Fisher information matrix diagonal as a
        proxy for the Hessian. For each layer, computes
        ``E[ (grad * loss) ^ 2 ]`` and uses the trace as the
        sensitivity metric.
        """
        pairs = _normalize_calibration_supervised(calibration_data)
        if not pairs:
            raise CalibrationDataError(
                "Hessian analysis requires supervised calibration data "
                "as (inputs, targets) pairs."
            )

        layer_names = self._discover_quantizable_layers(model)

        # Sub-select batches
        pairs = pairs[:self._hessian_batch_size]

        # Accumulate squared gradients per layer
        grad_sq_sum: dict[str, float] = {n: 0.0 for n in layer_names}
        total_samples = 0

        model.train()
        for inputs, targets in pairs:
            model.zero_grad()
            if isinstance(inputs, (list, tuple)):
                outputs = model(*inputs)
            else:
                outputs = model(inputs)

            # Determine loss function based on output shape
            if isinstance(targets, torch.Tensor) and outputs.shape[-1] > 1:
                loss = torch.nn.functional.cross_entropy(
                    outputs.view(-1, outputs.shape[-1]),
                    targets.view(-1),
                )
            else:
                # MSE-like fallback
                loss = (
                    outputs.norm()
                    if not isinstance(targets, torch.Tensor)
                    else torch.nn.functional.mse_loss(outputs, targets)
                )

            loss.backward()

            # Accumulate squared gradients
            for name, param in model.named_parameters():
                if name not in grad_sq_sum:
                    continue
                if param.grad is not None:
                    g = param.grad.detach()
                    grad_sq_sum[name] += (g ** 2).sum().item()

            total_samples += 1

        if total_samples == 0:
            raise CalibrationDataError(
                "No calibration data was processed. "
                "Ensure the model can accept the provided inputs."
            )

        # Normalize and score
        raw_scores = {n: v / total_samples for n, v in grad_sq_sum.items()}
        max_score = max(raw_scores.values()) if raw_scores else 1.0

        results: list[LayerSensitivity] = []
        for name in layer_names:
            trace = raw_scores.get(name, 0.0)
            score = min(1.0, trace / max(max_score, 1e-12))
            recommended = _recommend_method(score)

            results.append(LayerSensitivity(
                layer_name=name,
                sensitivity_score=round(score, 6),
                hessian_trace=round(trace, 6),
                recommended_method=recommended,
            ))

        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _discover_quantizable_layers(
        self,
        model: torch.nn.Module,
    ) -> list[str]:
        """Discover layers with quantizable (>=2D) parameter tensors."""
        names: list[str] = []
        for name, module in model.named_modules():
            params = list(module.parameters(recurse=False))
            has_weight = any(p.ndim >= 2 for p in params)
            if not has_weight:
                continue
            if self._layer_filter is not None and not self._layer_filter(name):
                continue
            names.append(name)
        return names


# ---------------------------------------------------------------------------
# Module-level helper functions (no torch dependency in signatures)
# ---------------------------------------------------------------------------

def _normalize_calibration(calibration_data: Any) -> list[Any]:
    """Normalize calibration data into a list of input batches."""
    if calibration_data is None:
        return []
    if isinstance(calibration_data, (list, tuple)):
        if len(calibration_data) == 0:
            return []
        return list(calibration_data)
    return [calibration_data]


def _normalize_calibration_supervised(
    calibration_data: Any,
) -> list[tuple[Any, Any]]:
    """Normalize calibration data into list of (inputs, targets) pairs."""
    if calibration_data is None:
        return []
    if isinstance(calibration_data, (list, tuple)):
        if len(calibration_data) == 0:
            return []
        first = calibration_data[0]
        if isinstance(first, (list, tuple)) and len(first) == 2:
            return list(calibration_data)
        if hasattr(calibration_data, "dataset"):
            return list(calibration_data)
        return []
    return []


def _recommend_method(score: float) -> QuantMethod:
    """Recommend quantisation method based on sensitivity score."""
    if score >= 0.85:
        return QuantMethod.FP16
    if score >= 0.60:
        return QuantMethod.FP8
    if score >= 0.30:
        return QuantMethod.INT8
    return QuantMethod.INT4


# ---------------------------------------------------------------------------
# Strategy Selector
# ---------------------------------------------------------------------------

class QuantizationStrategySelector:
    """Select quantization strategies based on layer sensitivity and hardware.

    Applies a rule-based decision engine:

    ==================== ==================================================
    Sensitivity Level    Quantization Method
    ==================== ==================================================
    Critical (>=0.85)    FP16 (no quantization)
    High    (>=0.60)     FP8 (preserves dynamic range)
    Medium  (>=0.30)     INT8 (good balance of quality vs compression)
    Low     ( < 0.30)    INT4 (aggressive compression)
    ==================== ==================================================

    Hardware capability overrides:

    * If hardware does **not** support FP8, fallback ``high`` -> INT8.
    * If hardware does **not** support INT8 Tensor Cores,
      ``medium`` uses per-tensor INT8 instead of per-channel.

    Usage::

        selector = QuantizationStrategySelector()
        strategy = selector.select(
            layer_sensitivity=0.45,
            hardware=HardwareConfig(name="a100", supports_fp8=True),
        )
    """

    # Sensitivity thresholds (aligned with SensitivityAnalyzer defaults)
    LOW_THRESHOLD: float = 0.30
    HIGH_THRESHOLD: float = 0.60
    CRITICAL_THRESHOLD: float = 0.85

    # Batch-size-dependent scale: larger batches push toward higher
    # precision to preserve quality at throughput
    BATCH_SCALE_FACTOR: float = 0.02

    def __init__(
        self,
        rules: dict[float, QuantizationStrategy] | None = None,
    ) -> None:
        """
        Args:
            rules: Optional custom mapping of ``(min_score, strategy)``.
                If ``None``, built-in thresholds are used.
        """
        self._rules = rules

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select(
        self,
        layer_sensitivity: float | LayerSensitivity,
        hardware: HardwareConfig | None = None,
        batch_size: int = 1,
    ) -> QuantizationStrategy:
        """Select quantization strategy for a given layer.

        Args:
            layer_sensitivity: Sensitivity score [0, 1] or a full
                :class:`LayerSensitivity` instance.
            hardware: Hardware capabilities. If ``None``, a
                conservative default is used (no FP8, no INT4).
            batch_size: Active batch size (larger batches may need
                higher precision).

        Returns:
            :class:`QuantizationStrategy` with method, bit-width, and
            configuration settings.
        """
        score = (
            layer_sensitivity.sensitivity_score
            if isinstance(layer_sensitivity, LayerSensitivity)
            else float(layer_sensitivity)
        )

        hw = hardware or HardwareConfig()
        effective_score = min(
            1.0,
            score + self.BATCH_SCALE_FACTOR * math.log2(max(1, batch_size)),
        )

        # Custom rules override
        if self._rules is not None:
            sorted_thresholds = sorted(self._rules.keys(), reverse=True)
            for threshold in sorted_thresholds:
                if effective_score >= threshold:
                    return self._rules[threshold]

        # Built-in decision tree
        if effective_score >= self.CRITICAL_THRESHOLD:
            return QuantizationStrategy.fp16(
                reason=f"Critical sensitivity ({effective_score:.3f}): "
                       f"keep full precision",
            )

        if effective_score >= self.HIGH_THRESHOLD:
            if hw.supports_fp8:
                return QuantizationStrategy.fp8(per_channel=True)
            return QuantizationStrategy.int8(
                per_channel=True,
                symmetric=True,
            )

        if effective_score >= self.LOW_THRESHOLD:
            if hw.supports_int8_tensor_core:
                return QuantizationStrategy(
                    method=QuantMethod.INT8,
                    target_bits=8,
                    per_channel=True,
                    symmetric=True,
                    reason=f"Medium sensitivity ({effective_score:.3f}): "
                           f"INT8 with Tensor Core acceleration",
                )
            return QuantizationStrategy.int8(
                per_channel=False,
                symmetric=True,
            )

        # Low sensitivity
        if hw.supports_int4 or hw.supports_int8_tensor_core:
            return QuantizationStrategy.int4(
                per_channel=True,
                symmetric=True,
                group_size=128,
            )
        return QuantizationStrategy(
            method=QuantMethod.INT8,
            target_bits=8,
            per_channel=False,
            symmetric=True,
            reason="INT4 not supported by hardware, falling back to INT8",
        )

    def plan_from_sensitivities(
        self,
        sensitivities: Sequence[LayerSensitivity],
        hardware: HardwareConfig | None = None,
        batch_size: int = 1,
    ) -> dict[str, QuantizationStrategy]:
        """Generate a full strategy plan from a list of sensitivities.

        Args:
            sensitivities: Output from
                :meth:`SensitivityAnalyzer.analyze`.
            hardware: Hardware capabilities.
            batch_size: Active batch size.

        Returns:
            Mapping of layer_name -> QuantizationStrategy.
        """
        return {
            s.layer_name: self.select(s, hardware, batch_size)
            for s in sensitivities
        }


# ---------------------------------------------------------------------------
# Strategy Cache
# ---------------------------------------------------------------------------

class StrategyCache:
    """LRU cache for quantization strategies per (model, hardware) pair.

    Maps ``(model_id, hardware_name)`` -> :class:`QuantizationPlan`.
    Uses an LRU eviction policy with a configurable maximum size.
    Thread-safe via a ``threading.Lock``.

    Usage::

        cache = StrategyCache(max_size=100)
        cache.put("llama-70b", "a100", plan)
        cached = cache.get("llama-70b", "a100")
    """

    def __init__(self, max_size: int = 1000) -> None:
        """
        Args:
            max_size: Maximum number of cached plans. Default 1000.
        """
        self._max_size = max_size
        self._cache: dict[str, QuantizationPlan] = collections.OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(
        self,
        model_id: str,
        hardware: str | HardwareConfig,
    ) -> QuantizationPlan | None:
        """Retrieve a cached strategy plan.

        Args:
            model_id: Model identifier (e.g., ``"llama-70b"``).
            hardware: Hardware name or :class:`HardwareConfig`.

        Returns:
            :class:`QuantizationPlan` if cached, else ``None``.
        """
        key = self._make_key(model_id, hardware)
        with self._lock:
            plan = self._cache.get(key)
            if plan is not None:
                self._cache.move_to_end(key)
                self._hits += 1
                return plan
            self._misses += 1
            return None

    def put(
        self,
        model_id: str,
        hardware: str | HardwareConfig,
        plan: QuantizationPlan,
    ) -> None:
        """Cache a strategy plan.

        Args:
            model_id: Model identifier.
            hardware: Hardware name or :class:`HardwareConfig`.
            plan: The :class:`QuantizationPlan` to cache.
        """
        key = self._make_key(model_id, hardware)
        with self._lock:
            self._cache[key] = plan
            self._cache.move_to_end(key)
            self._evict()

    def get_or_compute(
        self,
        model_id: str,
        hardware: str | HardwareConfig,
        compute_fn: Callable[[], QuantizationPlan],
    ) -> QuantizationPlan:
        """Get cached plan or compute and cache it.

        Args:
            model_id: Model identifier.
            hardware: Hardware name or :class:`HardwareConfig`.
            compute_fn: Zero-argument callable that returns a
                :class:`QuantizationPlan`.

        Returns:
            Cached or freshly computed plan.
        """
        cached = self.get(model_id, hardware)
        if cached is not None:
            return cached
        plan = compute_fn()
        self.put(model_id, hardware, plan)
        return plan

    def invalidate(
        self,
        model_id: str | None = None,
        hardware: str | HardwareConfig | None = None,
    ) -> None:
        """Invalidate cached entries.

        Args:
            model_id: If set, invalidate only entries for this model.
                If ``None``, invalidate all entries matching hardware.
            hardware: If set, invalidate only entries for this
                hardware. If ``None``, invalidate all entries matching
                model_id. If both are ``None``, clear the entire cache.
        """
        with self._lock:
            if model_id is None and hardware is None:
                self._cache.clear()
                return

            keys_to_delete: list[str] = []
            hw_name = (
                self._resolve_hardware_name(hardware)
                if hardware is not None
                else None
            )

            for key in self._cache:
                km, kh = self._split_key(key)
                if model_id is not None and km != model_id:
                    continue
                if hw_name is not None and kh != hw_name:
                    continue
                keys_to_delete.append(key)

            for key in keys_to_delete:
                del self._cache[key]

    def stats(self) -> dict[str, Any]:
        """Return cache statistics.

        Returns:
            Dict with keys: ``size``, ``max_size``, ``hits``,
            ``misses``, ``hit_rate``.
        """
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(hit_rate, 4),
            }

    def clear(self) -> None:
        """Clear the entire cache."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_key(
        self,
        model_id: str,
        hardware: str | HardwareConfig,
    ) -> str:
        hw_name = self._resolve_hardware_name(hardware)
        return f"{model_id}||{hw_name}"

    @staticmethod
    def _split_key(key: str) -> tuple[str, str]:
        parts = key.split("||", 1)
        if len(parts) == 2:
            return parts[0], parts[1]
        return key, ""

    @staticmethod
    def _resolve_hardware_name(
        hardware: str | HardwareConfig,
    ) -> str:
        if isinstance(hardware, HardwareConfig):
            return hardware.name
        return hardware

    def _evict(self) -> None:
        """Evict least recently used entries if over max_size."""
        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)


# ---------------------------------------------------------------------------
# AutoQ --- Orchestrator
# ---------------------------------------------------------------------------

class AutoQ:
    """Adaptive quantization orchestrator.

    Combines :class:`SensitivityAnalyzer`,
    :class:`QuantizationStrategySelector`, and :class:`StrategyCache`
    into a complete pipeline.

    Typical workflow::

        autoq = AutoQ()
        plan = autoq.optimize(
            model=my_model,
            data=calibration_dataloader,
            target_quality=0.95,
        )
        for name, strategy in plan.strategies.items():
            print(f"{name}: {strategy.method.value}")
        print(f"Compression ratio: {plan.global_compression_ratio:.2f}x")
    """

    def __init__(
        self,
        analyzer: SensitivityAnalyzer | None = None,
        selector: QuantizationStrategySelector | None = None,
        cache: StrategyCache | None = None,
        cache_max_size: int = 1000,
    ) -> None:
        """
        Args:
            analyzer: Sensitivity analyzer instance. Created with
                default settings if ``None``.
            selector: Strategy selector instance. Created with default
                thresholds if ``None``.
            cache: Strategy cache instance. Created with
                ``max_size=cache_max_size`` if ``None``.
            cache_max_size: Only used when ``cache`` is not provided.
        """
        self._analyzer = analyzer or SensitivityAnalyzer()
        self._selector = selector or QuantizationStrategySelector()
        self._cache = cache or StrategyCache(max_size=cache_max_size)
        self._current_plan: QuantizationPlan | None = None
        self._model_id: str = ""
        self._hardware: HardwareConfig = HardwareConfig()
        self._analysis_time_ms: float = 0.0
        self._num_calibration_samples: int = 0

    # ------------------------------------------------------------------
    # Main optimization pipeline
    # ------------------------------------------------------------------

    def optimize(
        self,
        model: Any,
        data: Any,
        target_quality: float = 0.95,
        hardware: HardwareConfig | None = None,
        model_id: str | None = None,
        batch_size: int = 1,
        use_cache: bool = True,
    ) -> QuantizationPlan:
        """Run the full adaptive quantization pipeline.

        Steps:

        1. **Cache lookup** (if ``use_cache=True`` and ``model_id``
           given).
        2. **Sensitivity analysis** --- runs
           :meth:`SensitivityAnalyzer.analyze`.
        3. **Strategy selection** --- runs
           :meth:`QuantizationStrategySelector.plan_from_sensitivities`.
        4. **Quality calibration** --- adjusts strategies to meet
           ``target_quality``.
        5. **Cache storage** (if ``use_cache=True`` and ``model_id``
           given).

        Args:
            model: PyTorch model (``nn.Module``).
            data: Calibration data for sensitivity analysis.
            target_quality: Minimum acceptable quality score [0, 1].
            hardware: Hardware capabilities. Defaults to generic if
                not provided.
            model_id: Optional model identifier for cache operations.
            batch_size: Active batch size for strategy selection.
            use_cache: Whether to check and update the strategy cache.

        Returns:
            :class:`QuantizationPlan` with layer-to-strategy mapping.

        Raises:
            TorchRequiredError: If PyTorch is not installed.
            CalibrationDataError: If calibration data is empty.
        """
        if not _TORCH_AVAILABLE:
            raise TorchRequiredError(
                "PyTorch is required for AutoQ.optimize(). "
                "Install it with: pip install torch"
            )

        hw = hardware or HardwareConfig()
        mid = (
            model_id
            or self._model_id
            or getattr(getattr(model, "__class__", model), "__name__", "unknown")
        )
        self._model_id = mid
        self._hardware = hw

        # Step 1: Cache lookup
        if use_cache and model_id is not None:
            cached = self._cache.get(mid, hw)
            if cached is not None and self._meets_quality(cached, target_quality):
                logger.info(
                    "Using cached quantization plan for {} on {} "
                    "(quality={:.3f})",
                    mid,
                    hw.name,
                    cached.quality_score,
                )
                self._current_plan = cached
                return cached
            if cached is not None:
                logger.info(
                    "Cached plan quality ({:.3f}) below target ({:.3f}), "
                    "re-optimizing",
                    cached.quality_score,
                    target_quality,
                )

        # Step 2: Sensitivity analysis
        t0 = time.perf_counter()
        sensitivities = self._analyzer.analyze(model, data)
        self._analysis_time_ms = (time.perf_counter() - t0) * 1000.0
        self._num_calibration_samples = _count_calibration_samples(data)

        if not sensitivities:
            logger.warning(
                "No sensitivities computed. Returning full FP16 plan."
            )
            plan = _make_fallback_plan(model, hw)
            self._current_plan = plan
            return plan

        # Pre-compute parameter counts once for plan arithmetic
        param_counts = _count_params_per_layer(model)

        # Step 3: Strategy selection
        strategy_map = self._selector.plan_from_sensitivities(
            sensitivities,
            hw,
            batch_size,
        )

        # Step 4: Quality calibration
        plan = self._build_plan(strategy_map, sensitivities, param_counts)
        plan = self._calibrate_quality(plan, target_quality, sensitivities, param_counts)

        # Step 5: Cache
        if use_cache and model_id is not None:
            self._cache.put(mid, hw, plan)
            logger.info(
                "Cached quantization plan for {} on {} "
                "(quality={:.3f}, compression={:.2f}x)",
                mid,
                hw.name,
                plan.quality_score,
                plan.global_compression_ratio,
            )

        self._current_plan = plan
        return plan

    def get_strategy(self, layer_name: str) -> QuantizationStrategy | None:
        """Get the current strategy for a given layer.

        Args:
            layer_name: Name of the layer.

        Returns:
            :class:`QuantizationStrategy` if a plan has been generated
            and includes this layer, otherwise ``None``.
        """
        if self._current_plan is None:
            return None
        return self._current_plan.strategies.get(layer_name)

    def stats(self) -> AutoQStats:
        """Return statistics about the current optimization run.

        Returns:
            :class:`AutoQStats` snapshot. All numeric fields default to
            zero if no optimization has been run yet.
        """
        if self._current_plan is None:
            return AutoQStats()

        qmap: dict[str, int] = {}
        for s in self._current_plan.strategies.values():
            m = s.method.value
            qmap[m] = qmap.get(m, 0) + 1

        return AutoQStats(
            num_layers=len(self._current_plan.strategies),
            quantization_map=qmap,
            global_compression_ratio=self._current_plan.global_compression_ratio,
            estimated_quality=self._current_plan.quality_score,
            analysis_time_ms=self._analysis_time_ms,
            calibration_samples=self._num_calibration_samples,
            hardware=self._hardware.name,
        )

    def reset(self) -> None:
        """Reset the optimizer state (plan, stats, analyzer)."""
        self._current_plan = None
        self._model_id = ""
        self._hardware = HardwareConfig()
        self._analysis_time_ms = 0.0
        self._num_calibration_samples = 0
        self._analyzer = SensitivityAnalyzer()

    @property
    def current_plan(self) -> QuantizationPlan | None:
        """The most recent quantization plan, or ``None``."""
        return self._current_plan

    @property
    def cache(self) -> StrategyCache:
        """Underlying strategy cache instance."""
        return self._cache

    # ------------------------------------------------------------------
    # Quality calibration
    # ------------------------------------------------------------------

    def _calibrate_quality(
        self,
        plan: QuantizationPlan,
        target_quality: float,
        sensitivities: Sequence[LayerSensitivity],
        param_counts: dict[str, int] | None = None,
    ) -> QuantizationPlan:
        """Adjust strategies to meet the target quality.

        If the estimated quality is below target, iteratively upgrades
        the most sensitive layers (starting with the most sensitive
        that is not already FP16) to the next higher precision level.
        This repeats in multiple passes until the target is met or all
        layers reach FP16.

        Args:
            plan: Initial quantization plan.
            target_quality: Minimum acceptable quality.
            sensitivities: Full sensitivity list for ranking.
            param_counts: Pre-computed parameter counts per layer.
                If ``None``, they will be ignored during plan
                construction.

        Returns:
            Adjusted :class:`QuantizationPlan`.
        """
        if plan.quality_score >= target_quality:
            return plan

        # Build sorted layer list (most sensitive first)
        sens_map = {s.layer_name: s.sensitivity_score for s in sensitivities}
        sorted_names = sorted(
            plan.strategies.keys(),
            key=lambda n: sens_map.get(n, 0.0),
            reverse=True,
        )

        strategies = dict(plan.strategies)  # mutable working copy

        # Upgrade ladder per method
        UPGRADE: dict[QuantMethod, QuantMethod] = {
            QuantMethod.INT4: QuantMethod.INT8,
            QuantMethod.INT8: QuantMethod.FP8,
            QuantMethod.FP8: QuantMethod.FP16,
        }

        # Keep doing passes until target quality is met
        any_upgraded = True
        while any_upgraded:
            any_upgraded = False

            for name in sorted_names:
                current = strategies.get(name)
                if current is None or current.method == QuantMethod.FP16:
                    continue

                next_method = UPGRADE.get(current.method)
                if next_method is None:
                    continue

                # Build upgraded strategy
                if next_method == QuantMethod.FP16:
                    strategies[name] = QuantizationStrategy.fp16(
                        reason=f"Upgraded from {current.method.value} "
                               f"to meet quality target {target_quality}",
                    )
                elif next_method == QuantMethod.FP8:
                    strategies[name] = QuantizationStrategy.fp8(per_channel=True)
                elif next_method == QuantMethod.INT8:
                    strategies[name] = QuantizationStrategy.int8(
                        per_channel=current.per_channel,
                        symmetric=current.symmetric,
                    )
                else:
                    continue

                any_upgraded = True

                # Recompute quality after each individual upgrade
                new_plan = self._build_plan(
                    strategies, sensitivities, param_counts,
                )
                if new_plan.quality_score >= target_quality:
                    return new_plan

            # If no upgrades happened in this pass, we can't improve further
            if not any_upgraded:
                break

        return self._build_plan(strategies, sensitivities, param_counts)

    # ------------------------------------------------------------------
    # Plan construction helpers
    # ------------------------------------------------------------------

    def _build_plan(
        self,
        strategy_map: dict[str, QuantizationStrategy],
        sensitivities: Sequence[LayerSensitivity],
        param_counts: dict[str, int] | None = None,
    ) -> QuantizationPlan:
        """Build a :class:`QuantizationPlan` from strategy assignments.

        Computes:

        * Compression ratio (weighted by parameter count per layer).
        * Estimated quality score (weighted by sensitivity).
        * Parameter counts (total vs quantized).

        Args:
            strategy_map: Mapping of layer_name -> strategy.
            sensitivities: Layer sensitivity results for quality
                weighting.
            param_counts: Pre-computed parameter counts per layer.
                If ``None``, ``total_parameters`` and
                ``quantized_parameters`` will be 0.
        """
        if param_counts is None:
            param_counts = {}
        total_params = sum(param_counts.values())
        quantized_params = 0
        weighted_bits = 0

        for name, strategy in strategy_map.items():
            n = param_counts.get(name, 0)
            if strategy.method != QuantMethod.FP16:
                quantized_params += n
            weighted_bits += n * strategy.target_bits

        effective_bits = weighted_bits / max(total_params, 1)
        compression_ratio = 16.0 / max(effective_bits, 1.0)

        # Quality estimate: weighted average of per-layer quality
        sens_map = {s.layer_name: s for s in sensitivities}
        quality_weight = 0.0
        quality_sum = 0.0
        for name, strategy in strategy_map.items():
            sens = sens_map.get(name, LayerSensitivity())
            layer_quality = _strategy_quality(strategy, sens.sensitivity_score)
            w = param_counts.get(name, 1)
            quality_weight += w
            quality_sum += w * layer_quality

        quality_score = quality_sum / max(quality_weight, 1.0)

        return QuantizationPlan(
            strategies=strategy_map,
            global_compression_ratio=round(compression_ratio, 4),
            quality_score=round(quality_score, 6),
            total_parameters=total_params,
            quantized_parameters=quantized_params,
        )

    @staticmethod
    def _meets_quality(plan: QuantizationPlan, target: float) -> bool:
        """Check if a plan meets the target quality."""
        return plan.quality_score >= target


# ---------------------------------------------------------------------------
# Module-level utility functions
# ---------------------------------------------------------------------------

def _strategy_quality(
    strategy: QuantizationStrategy,
    sensitivity: float,
) -> float:
    """Estimate per-layer quality [0, 1] for a given strategy.

    Uses target bit-width as a base quality proxy, adjusted by
    per-channel and symmetric settings.
    """
    # Base quality from bit-width (FP16=1.0, INT8~0.95, INT4~0.85)
    base = min(1.0, strategy.target_bits / 16.0)
    # Per-channel bonus
    bonus = 0.03 if strategy.per_channel else 0.0
    # Symmetric penalty for INT8/INT4 (slightly less range)
    penalty = 0.02 if (
        strategy.symmetric
        and strategy.method in (QuantMethod.INT8, QuantMethod.INT4)
    ) else 0.0
    # FP8 gets a range bonus
    fp8_bonus = 0.04 if strategy.method == QuantMethod.FP8 else 0.0

    quality = base + bonus - penalty + fp8_bonus
    return max(0.0, min(1.0, quality))


def _count_calibration_samples(data: Any) -> int:
    """Count the number of samples in calibration data (best-effort)."""
    try:
        if hasattr(data, "__len__"):
            return len(data)
        if hasattr(data, "dataset"):
            return len(data.dataset)
    except (TypeError, AttributeError, NotImplementedError):
        pass
    return 0


def _count_params_per_layer(model: Any) -> dict[str, int]:
    """Count the number of quantizable parameters per named module.

    Works with and without PyTorch. Without torch, returns an empty
    dict.
    """
    if not _TORCH_AVAILABLE:
        return {}
    import torch

    counts: dict[str, int] = {}
    for name, module in model.named_modules():
        n = sum(
            p.numel()
            for p in module.parameters(recurse=False)
            if p.ndim >= 2
        )
        if n > 0:
            counts[name] = n
    return counts


def _make_fallback_plan(
    model: Any,
    hardware: HardwareConfig,
) -> QuantizationPlan:
    """Create a fallback FP16 plan when analysis yields no layers."""
    strategy_map: dict[str, QuantizationStrategy] = {}
    if _TORCH_AVAILABLE and isinstance(model, torch.nn.Module):
        for name, _module in model.named_modules():
            params = list(_module.parameters(recurse=False))
            if any(p.ndim >= 2 for p in params):
                strategy_map[name] = QuantizationStrategy.fp16()

    return QuantizationPlan(
        strategies=strategy_map,
        global_compression_ratio=1.0,
        quality_score=1.0,
        total_parameters=0,
        quantized_parameters=0,
    )
