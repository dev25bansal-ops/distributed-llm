"""Compression configuration for model loading.

Defines compression methods and their configuration for automatic
model compression during loading.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from distllm.errors import ConfigValidationError


class CompressionMethod(str, Enum):
    NONE = "none"
    PTQ_INT8 = "ptq_int8"
    PTQ_INT4 = "ptq_int4"
    QUANT_AWQ = "quant_awq"
    QUANT_GPTQ = "quant_gptq"
    PRUNING_STRUCTURED = "pruning_structured"
    DISTILLATION = "distillation"
    AUTO = "auto"


@dataclass
class CompressionConfig:
    method: CompressionMethod = CompressionMethod.NONE
    enabled: bool = False
    target_bits: int = 8
    pruning_ratio: float = 0.0
    distillation_teacher: Optional[str] = None
    calibration_samples: int = 128
    pruning_targets: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    quant_method: str = "awq"

    def __post_init__(self):
        if self.target_bits not in (4, 8, 16):
            raise ConfigValidationError("target_bits", f"must be 4, 8, or 16, got {self.target_bits}")
        if not (0.0 <= self.pruning_ratio <= 1.0):
            raise ConfigValidationError("pruning_ratio", f"must be 0.0-1.0, got {self.pruning_ratio}")
        if self.quant_method not in ("awq", "gptq"):
            raise ConfigValidationError("quant_method", f"must be 'awq' or 'gptq', got {self.quant_method}")
