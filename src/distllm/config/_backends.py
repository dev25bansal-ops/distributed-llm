"""vLLM and llama.cpp backend configuration classes."""

from pydantic import BaseModel, field_validator

__all__ = [
    "VLLMSettings",
    "LlamacppSettings",
]


class VLLMSettings(BaseModel):
    """vLLM backend configuration."""
    enabled: bool = False
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 8192
    dtype: str = "auto"
    seed: int = 0
    enforce_eager: bool = False
    max_model_len: int | None = None

    @field_validator("gpu_memory_utilization")
    @classmethod
    def validate_gpu_memory_utilization(cls, v: float) -> float:
        if not 0 < v <= 1:
            raise ValueError(f"gpu_memory_utilization must be in (0, 1], got {v}")
        return v

    @field_validator("tensor_parallel_size")
    @classmethod
    def validate_tp_size(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"tensor_parallel_size must be >= 1, got {v}")
        return v

    @field_validator("dtype")
    @classmethod
    def validate_dtype(cls, v: str) -> str:
        allowed = {"auto", "float16", "float32", "bfloat16", "half", "full"}
        if v not in allowed:
            raise ValueError(f"dtype must be one of {allowed}, got '{v}'")
        return v


class LlamacppSettings(BaseModel):
    """llama.cpp backend configuration.

    Lightweight alternative to vLLM for CPU/GPU inference with GGUF models.
    Supports CPU, CUDA, AMD ROCm, and Apple Metal backends.
    """
    enabled: bool = False
    model_path: str = ""
    n_gpu_layers: int = 0
    n_ctx: int = 2048
    n_threads: int | None = None
    n_batch: int = 512
    seed: int = 0
    verbose: bool = False

    @field_validator("model_path")
    @classmethod
    def validate_model_path(cls, v: str, info) -> str:
        if info.data.get("enabled", False) and not v:
            raise ValueError("model_path is required when llamacpp is enabled")
        return v

    @field_validator("n_gpu_layers")
    @classmethod
    def validate_n_gpu_layers(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"n_gpu_layers must be >= 0, got {v}")
        return v

    @field_validator("n_ctx")
    @classmethod
    def validate_n_ctx(cls, v: int) -> int:
        if v < 128:
            raise ValueError(f"n_ctx must be >= 128, got {v}")
        return v
