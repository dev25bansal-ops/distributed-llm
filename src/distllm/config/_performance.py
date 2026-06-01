"""CUDA graph, torch.compile, adaptive precision, and self-optimizing
configuration classes."""

from pydantic import BaseModel

__all__ = [
    "CudaGraphSettings",
    "CompileSettings",
    "AdaptivePrecisionSettings",
    "SelfOptimizingSettings",
]


class CudaGraphSettings(BaseModel):
    """CUDA graph capture for decode acceleration."""
    enabled: bool = False
    batch_sizes: list[int] = [1, 2, 4, 8, 16, 32]


class CompileSettings(BaseModel):
    """torch.compile integration."""
    enabled: bool = False
    mode: str = "reduce-overhead"
    fullgraph: bool = False


class AdaptivePrecisionSettings(BaseModel):
    """Adaptive precision pipeline configuration."""
    enabled: bool = False
    calibration_samples: int = 64
    target_precision: str = "auto"  # "auto", "fp16", "int8"
    max_quality_loss_pct: float = 0.1


class SelfOptimizingSettings(BaseModel):
    """Auto-tuning via hill-climbing optimization (legacy)."""
    enabled: bool = False
    tune_interval_seconds: float = 60.0
    warmup_seconds: float = 30.0
    profile_dir: str | None = None

    def to_optimization_config(self):
        """Convert to the new Bayesian OptimizationConfig."""
        return {
            "enabled": self.enabled,
            "runner": {"warmup_seconds": self.warmup_seconds},
        }
