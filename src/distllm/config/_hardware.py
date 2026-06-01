"""Multi-architecture hardware configuration."""

from pydantic import BaseModel, field_validator

__all__ = [
    "HardwareSettings",
]


class HardwareSettings(BaseModel):
    """Multi-architecture hardware configuration.

    Controls device selection, backend preference, and architecture-
    specific settings for heterogeneous clusters.
    """
    device_type: str = "auto"  # "auto" | "cuda" | "rocm" | "mps" | "xpu" | "cpu"
    preferred_backend: str = "auto"  # "auto" | "vllm" | "pytorch" | "llamacpp"
    force_device_id: int = -1  # -1 = auto-select
    fallback_to_cpu: bool = True

    # Architecture-specific overrides
    rocm_visible_devices: str = ""
    mps_optimize_memory: bool = True
    xpu_oneapi_verbose: bool = False
    cpu_threads: int = 0  # 0 = auto-detect via psutil
    cpu_numa_aware: bool = True

    @field_validator("device_type")
    @classmethod
    def validate_device_type(cls, v: str) -> str:
        allowed = {"auto", "cuda", "rocm", "mps", "xpu", "cpu"}
        if v not in allowed:
            raise ValueError(f"device_type must be one of {allowed}, got '{v}'")
        return v

    @field_validator("preferred_backend")
    @classmethod
    def validate_backend(cls, v: str) -> str:
        allowed = {"auto", "vllm", "pytorch", "llamacpp"}
        if v not in allowed:
            raise ValueError(f"preferred_backend must be one of {allowed}, got '{v}'")
        return v
