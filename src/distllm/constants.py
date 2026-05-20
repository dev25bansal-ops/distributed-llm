"""Named constants for distributed LLM inference.

Centralizes magic numbers used across the codebase to avoid duplication
and improve maintainability.
"""

# --- Sampling defaults ---
DEFAULT_TEMPERATURE: float = 0.7
DEFAULT_TOP_P: float = 0.9
DEFAULT_TOP_K: int = 0
DEFAULT_MAX_TOKENS: int = 256

# --- Ports ---
COORDINATOR_PORT: int = 50050
API_PORT: int = 8000
GRPC_PORT: int = 50051
DASHBOARD_PORT: int = 8500

# --- Network timeouts ---
GRPC_TIMEOUT: float = 30.0
GRPC_LONG_TIMEOUT: float = 60.0
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
TENSOR_MAX_DIM_SIZE: int = 2_000_000_000  # Effectively unlimited; real limit is TENSOR_MAX_TOTAL_BYTES
TENSOR_MAX_TOTAL_BYTES: int = 4 * 1024 * 1024 * 1024  # 4 GiB (configurable via env var DISTLLM_MAX_TENSOR_BYTES)

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
