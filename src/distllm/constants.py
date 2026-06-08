"""Named constants and enums for distributed LLM inference.

Centralizes magic numbers, dtype strings, and device names used across
the codebase to avoid duplication and improve maintainability.
"""

from enum import Enum


class DType(str, Enum):
    """Common torch dtypes used in model configuration."""
    FLOAT16 = "float16"
    FLOAT32 = "float32"
    BFLOAT16 = "bfloat16"
    HALF = "half"
    FLOAT = "float"


class Device(str, Enum):
    """Common device targets for model execution."""
    CUDA = "cuda"
    CPU = "cpu"
    AUTO = "auto"
    ROCM = "rocm"
    MPS = "mps"
    XPU = "xpu"


class DeviceFamily(str, Enum):
    """GPU architecture families for heterogeneous scheduling."""
    NVIDIA = "nvidia"
    AMD = "amd"
    APPLE = "apple"
    INTEL = "intel"
    CPU = "cpu"
    UNKNOWN = "unknown"


DEVICE_FAMILY_MAP: dict[str, DeviceFamily] = {
    "cuda": DeviceFamily.NVIDIA,
    "rocm": DeviceFamily.AMD,
    "mps": DeviceFamily.APPLE,
    "xpu": DeviceFamily.INTEL,
    "cpu": DeviceFamily.CPU,
}


PLATFORM_BACKEND_PRIORITY: dict[str, dict[str, int]] = {
    "nvidia": {"vllm": 10, "exllama": 8, "pytorch": 5, "llamacpp": 4, "onnx": 6},
    "amd": {"llamacpp": 9, "pytorch": 7, "onnx": 6, "vllm": 5, "exllama": 0},
    "apple": {"llamacpp": 9, "pytorch": 8, "onnx": 2, "vllm": 0, "exllama": 0},
    "intel": {"onnx": 9, "pytorch": 7, "llamacpp": 6, "vllm": 0, "exllama": 0},
    "cpu": {"llamacpp": 9, "onnx": 6, "pytorch": 5, "vllm": 0, "exllama": 0},
}

DEVICE_TO_FAMILY: dict[str, DeviceFamily] = {
    "cuda": DeviceFamily.NVIDIA,
    "rocm": DeviceFamily.AMD,
    "mps": DeviceFamily.APPLE,
    "xpu": DeviceFamily.INTEL,
    "cpu": DeviceFamily.CPU,
    "vulkan": DeviceFamily.UNKNOWN,
}


class Provider(str, Enum):
    """Supported inference provider names."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    TOGETHER = "together"
    FIREWORKS = "fireworks"
    GROQ = "groq"
    DEEPINFRA = "deepinfra"
    VLLM = "vllm"
    LLAMACPP = "llamacpp"


# --- Sampling defaults ---
DEFAULT_TEMPERATURE: float = 0.7
DEFAULT_TOP_P: float = 0.9
DEFAULT_TOP_K: int = 0
DEFAULT_MAX_TOKENS: int = 256

# --- Ports ---
COORDINATOR_PORT: int = 50050
API_PORT: int = 8000
DASHBOARD_PORT: int = 8500

# --- Network timeouts ---
DEFAULT_HTTP_TIMEOUT: float = 120.0
SHORT_HTTP_TIMEOUT: float = 10.0
SPOT_DRAIN_TIMEOUT: float = 30.0

# --- Retry ---
MAX_RETRIES: int = 3
RETRY_DELAY: float = 1.0

# --- HSTS ---
HSTS_MAX_AGE: int = 31536000

# --- Rate limiting ---
DEFAULT_RPM: float = 60.0
BURST_MULTIPLIER: float = 1.5
MAX_CLIENTS: int = 10000

# --- Security ---
TENSOR_MAX_DIMS: int = 8
TENSOR_MAX_DIM_SIZE: int = 2_000_000_000
TENSOR_MAX_TOTAL_BYTES: int = 4 * 1024 * 1024 * 1024

# --- Hardware detection fallbacks ---
INTEL_XPU_BANDWIDTH_GBPS: float = 50.0
MPS_DEFAULT_MEMORY_BYTES: int = 8 * 1024 ** 3
MPS_DEFAULT_GPU_CORES: int = 8
MPS_DEFAULT_TFLOPS_FP16: float = 2.0

def get_tensor_max_bytes() -> int:
    """Return TENSOR_MAX_TOTAL_BYTES, overridable via environment variable."""
    import os
    env_val = os.environ.get("DISTLLM_MAX_TENSOR_BYTES")
    if env_val is not None:
        try:
            return int(env_val)
        except (ValueError, TypeError):
            pass
    return TENSOR_MAX_TOTAL_BYTES
