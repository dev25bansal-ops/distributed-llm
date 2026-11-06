"""Model Hub enhancements — auto-quantization, compatibility matrix, version management.

Extends the base ModelHub with:
- Auto-quantization selection based on available VRAM
- Model compatibility matrix (model → hardware requirements)
- Version management (track installed versions, suggest upgrades)

Usage::

    hub = EnhancedModelHub()
    plan = hub.auto_quantize("meta-llama/Llama-2-70b")
    matrix = hub.get_compatibility_matrix()
    versions = hub.get_version_history("meta-llama/Llama-2-70b")
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger


@dataclass
class QuantizationPlan:
    """Auto-selected quantization plan for a model."""
    model_name: str
    original_size_gb: float
    quantized_size_gb: float
    method: str           # "fp16", "int8", "int4_awq", "int4_gptq"
    quality_score: float  # 0-1
    fits_in_vram: bool
    vram_available_gb: float
    recommendation: str


@dataclass
class ModelCompatibility:
    """Compatibility info for a model on specific hardware."""
    model_name: str
    params_billions: float
    layers: int
    hidden_size: int
    min_gpu_memory_gb: float
    recommended_gpu_memory_gb: float
    supported_dtypes: list[str]
    supported_quantizations: list[str]
    min_gpus: int
    notes: str = ""


@dataclass
class ModelVersion:
    """A tracked model version."""
    model_name: str
    version: str
    revision: str
    size_bytes: int
    downloaded_at: float
    quantization: str = "fp16"
    notes: str = ""


# Known model compatibility matrix
_COMPATIBILITY_MATRIX: dict[str, ModelCompatibility] = {
    "llama-3.2-1b": ModelCompatibility(
        model_name="Llama 3.2 1B", params_billions=1.0, layers=16, hidden_size=2048,
        min_gpu_memory_gb=2, recommended_gpu_memory_gb=4,
        supported_dtypes=["fp16", "bf16"], supported_quantizations=["int8", "int4_awq"],
        min_gpus=1, notes="Lightweight model for edge devices",
    ),
    "llama-3.1-8b": ModelCompatibility(
        model_name="Llama 3.1 8B", params_billions=8.0, layers=32, hidden_size=4096,
        min_gpu_memory_gb=16, recommended_gpu_memory_gb=24,
        supported_dtypes=["fp16", "bf16"], supported_quantizations=["int8", "int4_awq", "int4_gptq"],
        min_gpus=1, notes="Best balance of quality and speed",
    ),
    "llama-3.1-70b": ModelCompatibility(
        model_name="Llama 3.1 70B", params_billions=70.0, layers=80, hidden_size=8192,
        min_gpu_memory_gb=140, recommended_gpu_memory_gb=160,
        supported_dtypes=["fp16", "bf16"], supported_quantizations=["int8", "int4_awq", "int4_gptq"],
        min_gpus=2, notes="High quality, requires multi-GPU or quantization",
    ),
    "llama-3.1-405b": ModelCompatibility(
        model_name="Llama 3.1 405B", params_billions=405.0, layers=126, hidden_size=16384,
        min_gpu_memory_gb=810, recommended_gpu_memory_gb=1000,
        supported_dtypes=["bf16"], supported_quantizations=["int8", "int4_awq"],
        min_gpus=8, notes="Largest open model, requires cluster",
    ),
    "mistral-7b": ModelCompatibility(
        model_name="Mistral 7B", params_billions=7.0, layers=32, hidden_size=4096,
        min_gpu_memory_gb=14, recommended_gpu_memory_gb=24,
        supported_dtypes=["fp16", "bf16"], supported_quantizations=["int8", "int4_awq", "int4_gptq"],
        min_gpus=1, notes="Fast inference, sliding window attention",
    ),
    "mixtral-8x7b": ModelCompatibility(
        model_name="Mixtral 8x7B", params_billions=46.7, layers=32, hidden_size=4096,
        min_gpu_memory_gb=90, recommended_gpu_memory_gb=120,
        supported_dtypes=["fp16", "bf16"], supported_quantizations=["int8", "int4_awq"],
        min_gpus=2, notes="MoE architecture, sparse activation",
    ),
    "qwen2.5-72b": ModelCompatibility(
        model_name="Qwen 2.5 72B", params_billions=72.0, layers=80, hidden_size=8192,
        min_gpu_memory_gb=140, recommended_gpu_memory_gb=160,
        supported_dtypes=["fp16", "bf16"], supported_quantizations=["int8", "int4_awq"],
        min_gpus=2, notes="Strong multilingual, code, and math",
    ),
    "codellama-34b": ModelCompatibility(
        model_name="CodeLlama 34B", params_billions=34.0, layers=48, hidden_size=8192,
        min_gpu_memory_gb=68, recommended_gpu_memory_gb=80,
        supported_dtypes=["fp16", "bf16"], supported_quantizations=["int8", "int4_awq"],
        min_gpus=1, notes="Specialized for code generation",
    ),
}


class EnhancedModelHub:
    """Enhanced model hub with auto-quantization, compatibility, and versioning.

    Wraps the base ModelHub and adds:
    - Auto-quantization selection based on VRAM
    - Model compatibility matrix
    - Version management and history
    """

    def __init__(self, base_hub: Any = None, cache_dir: str | None = None):
        self._hub = base_hub
        self._cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "distributed-llm" / "models"
        self._versions: dict[str, list[ModelVersion]] = {}  # model_name -> versions
        self._lock = threading.Lock()
        self._history_loaded = False

    def auto_quantize(self, model_name: str, gpu_memory_gb: float = 0) -> QuantizationPlan:
        """Auto-select the best quantization for a model given available VRAM.

        Args:
            model_name: HuggingFace model ID or short name.
            gpu_memory_gb: Available GPU memory. If 0, auto-detect.

        Returns:
            QuantizationPlan with the recommended quantization.
        """
        # Auto-detect VRAM
        if gpu_memory_gb <= 0:
            gpu_memory_gb = self._detect_vram()

        # Get model profile
        profile = self._get_model_profile(model_name)
        original_size = profile.min_gpu_memory_gb if profile else 7.0

        # Decision matrix
        quantizations = [
            ("fp16", original_size, 1.0),
            ("int8", original_size / 2, 0.95),
            ("int4_awq", original_size / 4, 0.85),
            ("int4_gptq", original_size / 4, 0.85),
        ]

        # Find best that fits
        for method, size, quality in quantizations:
            if size < gpu_memory_gb * 0.9:  # 90% utilization threshold
                return QuantizationPlan(
                    model_name=model_name,
                    original_size_gb=original_size,
                    quantized_size_gb=size,
                    method=method,
                    quality_score=quality,
                    fits_in_vram=True,
                    vram_available_gb=gpu_memory_gb,
                    recommendation=f"Use {method.upper()} — fits in {gpu_memory_gb:.0f}GB VRAM with {quality:.0%} quality",
                )

        # Nothing fits — recommend INT4 with warning
        int4_size = original_size / 4
        return QuantizationPlan(
            model_name=model_name,
            original_size_gb=original_size,
            quantized_size_gb=int4_size,
            method="int4_awq",
            quality_score=0.85,
            fits_in_vram=int4_size < gpu_memory_gb * 0.9,
            vram_available_gb=gpu_memory_gb,
            recommendation=f"INT4 required — model may not fit in {gpu_memory_gb:.0f}GB. Consider adding more GPUs.",
        )

    def get_compatibility_matrix(self, gpu_memory_gb: float = 0) -> list[dict]:
        """Return the full model compatibility matrix.

        Args:
            gpu_memory_gb: If provided, annotate each model with fit status.

        Returns:
            List of model compatibility dicts.
        """
        if gpu_memory_gb <= 0:
            gpu_memory_gb = self._detect_vram()

        matrix = []
        for key, compat in _COMPATIBILITY_MATRIX.items():
            fits_fp16 = compat.min_gpu_memory_gb < gpu_memory_gb * 0.9
            fits_int8 = (compat.min_gpu_memory_gb / 2) < gpu_memory_gb * 0.9
            fits_int4 = (compat.min_gpu_memory_gb / 4) < gpu_memory_gb * 0.9

            matrix.append({
                "model": key,
                "display_name": compat.model_name,
                "params_b": compat.params_billions,
                "layers": compat.layers,
                "hidden_size": compat.hidden_size,
                "min_gpu_memory_gb": compat.min_gpu_memory_gb,
                "recommended_gpu_memory_gb": compat.recommended_gpu_memory_gb,
                "min_gpus": compat.min_gpus,
                "fits_fp16": fits_fp16,
                "fits_int8": fits_int8,
                "fits_int4": fits_int4,
                "supported_quantizations": compat.supported_quantizations,
                "notes": compat.notes,
            })

        return matrix

    def _ensure_history_loaded(self) -> None:
        """Load persisted version history once, on first access."""
        if not self._history_loaded:
            self._load_version_history()
            self._history_loaded = True

    def get_version_history(self, model_name: str) -> list[dict]:
        """Return version history for a model."""
        with self._lock:
            self._ensure_history_loaded()
            versions = self._versions.get(model_name, [])
            return [
                {
                    "version": v.version,
                    "revision": v.revision,
                    "size_bytes": v.size_bytes,
                    "quantization": v.quantization,
                    "downloaded_at": v.downloaded_at,
                    "notes": v.notes,
                }
                for v in versions
            ]

    def record_version(
        self,
        model_name: str,
        revision: str,
        size_bytes: int = 0,
        quantization: str = "fp16",
        notes: str = "",
    ) -> None:
        """Record a new model version."""
        with self._lock:
            self._ensure_history_loaded()
            if model_name not in self._versions:
                self._versions[model_name] = []

            version_num = len(self._versions[model_name]) + 1
            version = ModelVersion(
                model_name=model_name,
                version=f"v{version_num}",
                revision=revision,
                size_bytes=size_bytes,
                downloaded_at=time.time(),
                quantization=quantization,
                notes=notes,
            )
            self._versions[model_name].append(version)
            self._save_version_history()

    def suggest_model(
        self,
        task: str = "general",
        gpu_memory_gb: float = 0,
        max_params_b: float = 0,
    ) -> list[dict]:
        """Suggest models based on task and hardware constraints.

        Args:
            task: Task type ("general", "code", "math", "chat", "creative").
            gpu_memory_gb: Available GPU memory.
            max_params_b: Maximum parameter count in billions.

        Returns:
            List of suggested models with compatibility info.
        """
        if gpu_memory_gb <= 0:
            gpu_memory_gb = self._detect_vram()

        suggestions = []
        for key, compat in _COMPATIBILITY_MATRIX.items():
            if max_params_b > 0 and compat.params_billions > max_params_b:
                continue

            quant = self.auto_quantize(key, gpu_memory_gb)
            # Only suggest models that fit with the recommended
            # quantization — a suggestion that cannot be loaded is noise.
            if not quant.fits_in_vram:
                continue
            fits = True

            score = 0.5  # Base score
            if fits:
                score += 0.3
            if task == "code" and "code" in key.lower():
                score += 0.2
            if task == "math" and "qwen" in key.lower():
                score += 0.1
            if compat.params_billions <= 8:
                score += 0.1  # Prefer smaller models for speed

            suggestions.append({
                "model": key,
                "display_name": compat.model_name,
                "params_b": compat.params_billions,
                "score": round(score, 2),
                "recommended_quantization": quant.method,
                "fits_in_vram": fits,
                "notes": compat.notes,
            })

        suggestions.sort(key=lambda x: -x["score"])
        return suggestions[:5]

    def _detect_vram(self) -> float:
        """Auto-detect available GPU VRAM."""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.get_device_properties(0).total_memory / 1e9
        except Exception:
            pass
        return 0.0

    def _get_model_profile(self, model_name: str) -> ModelCompatibility | None:
        """Get model profile from compatibility matrix."""
        name_lower = model_name.lower()
        for key, compat in _COMPATIBILITY_MATRIX.items():
            if key in name_lower or compat.model_name.lower() in name_lower:
                return compat
        return None

    def _load_version_history(self) -> None:
        """Load version history from disk."""
        history_path = self._cache_dir / ".version_history.json"
        if history_path.exists():
            try:
                data = json.loads(history_path.read_text())
                for model_name, versions in data.items():
                    self._versions[model_name] = [
                        ModelVersion(**v) for v in versions
                    ]
            except Exception:
                pass

    def _save_version_history(self) -> None:
        """Save version history to disk."""
        history_path = self._cache_dir / ".version_history.json"
        try:
            history_path.parent.mkdir(parents=True, exist_ok=True)
            data = {}
            for model_name, versions in self._versions.items():
                data[model_name] = [
                    {
                        "model_name": v.model_name,
                        "version": v.version,
                        "revision": v.revision,
                        "size_bytes": v.size_bytes,
                        "downloaded_at": v.downloaded_at,
                        "quantization": v.quantization,
                        "notes": v.notes,
                    }
                    for v in versions
                ]
            history_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.warning(f"Failed to save version history: {e}")
