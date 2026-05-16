"""Compression configuration for model loading.

Defines compression methods and their configuration for automatic
model compression during loading.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from distllm.errors import ConfigValidationError


class CompressionMethod(str, Enum):
    """Supported compression methods.

    Attributes:
        NONE: No compression.
        PTQ_INT8: Post-training quantization to 8-bit integers.
        PTQ_INT4: Post-training quantization to 4-bit integers.
        PRUNING_STRUCTURED: Structured pruning of attention heads/FFN neurons.
        DISTILLATION: Knowledge distillation from a teacher model.
        AUTO: Automatically select method based on VRAM budget.
    """

    NONE = "none"
    PTQ_INT8 = "ptq_int8"
    PTQ_INT4 = "ptq_int4"
    PRUNING_STRUCTURED = "pruning_structured"
    DISTILLATION = "distillation"
    AUTO = "auto"


@dataclass
class CompressionConfig:
    """Configuration for model compression.

    Attributes:
        method: Compression method to apply.
        enabled: Whether compression is enabled.
        target_bits: Target bit width for quantization (4 or 8).
        pruning_ratio: Fraction of weights/heads to prune (0.0-1.0).
        distillation_teacher: HuggingFace model name or path for teacher model.
        calibration_samples: Number of samples for PTQ calibration.
        pruning_targets: List of module name patterns to prune (e.g., ["q_proj", "v_proj"]).
    """

    method: CompressionMethod = CompressionMethod.NONE
    enabled: bool = False
    target_bits: int = 8
    pruning_ratio: float = 0.0
    distillation_teacher: Optional[str] = None
    calibration_samples: int = 128
    pruning_targets: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

    def __post_init__(self):
        if self.target_bits not in (4, 8):
            raise ConfigValidationError("target_bits", f"must be 4 or 8, got {self.target_bits}")
        if not (0.0 <= self.pruning_ratio <= 1.0):
            raise ConfigValidationError("pruning_ratio", f"must be 0.0-1.0, got {self.pruning_ratio}")
