"""Per-backend-type optimization profiles for distributed inference.

Each profile captures the optimal configuration parameters for a specific
backend type (vLLM, TensorRT, ONNX, llama.cpp, PyTorch).  The profile
manager auto-detects capabilities and allows run-time overrides.
"""

from __future__ import annotations

import dataclasses
import threading
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


# ---------------------------------------------------------------------------
# Profile data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BackendProfile:
    """Immutable optimization profile for a specific backend type.

    Attributes
    ----------
    backend_type:
        Short identifier (``"vllm"``, ``"tensorrt"``, ``"onnx"``,
        ``"llamacpp"``, ``"pytorch"``).
    optimal_batch_size:
        Recommended batch size for best throughput.
    preferred_dtype:
        Recommended computation dtype (``"float16"``, ``"bfloat16"``,
        ``"float32"``, ``"int8"``, ``"fp8"``).
    memory_per_token_bytes:
        Estimated GPU memory consumed per token of KV cache in bytes.
    max_seq_len:
        Maximum supported sequence length.
    num_attention_layers:
        Number of attention layers expected (used in pipeline partitioning).
    supports_prefix_caching:
        Whether the backend can reuse KV cache across prompts.
    supports_chunked_prefill:
        Whether chunked prefill is supported.
    supports_speculative_decoding:
        Whether the backend supports draft-model speculation.
    supports_flash_attention:
        Whether flash attention is supported.
    supports_paged_attention:
        Whether paged attention (vLLM-style) is supported.
    supports_fp8_kv_cache:
        Whether FP8 quantization of the KV cache is supported.
    supports_tensor_parallel:
        Whether tensor parallelism across GPUs is supported.
    supports_pipeline_parallel:
        Whether pipeline parallelism is supported.
    preferred_batch_scheduler:
        Recommended scheduler type (``"continuous"``, ``"static"``).
    extra:
        Extension dictionary for backend-specific settings.
    """

    backend_type: str
    optimal_batch_size: int = 1
    preferred_dtype: str = "float16"
    memory_per_token_bytes: int = 2 * 1024 * 1024  # 2 MiB default
    max_seq_len: int = 4096
    num_attention_layers: int = 32
    supports_prefix_caching: bool = False
    supports_chunked_prefill: bool = False
    supports_speculative_decoding: bool = False
    supports_flash_attention: bool = False
    supports_paged_attention: bool = False
    supports_fp8_kv_cache: bool = False
    supports_tensor_parallel: bool = False
    supports_pipeline_parallel: bool = False
    preferred_batch_scheduler: str = "continuous"
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Known default profiles
# ---------------------------------------------------------------------------

_DEFAULT_PROFILES: dict[str, BackendProfile] = {
    "vllm": BackendProfile(
        backend_type="vllm",
        optimal_batch_size=64,
        preferred_dtype="float16",
        memory_per_token_bytes=2 * 1024 * 1024,
        max_seq_len=32768,
        num_attention_layers=32,
        supports_prefix_caching=True,
        supports_chunked_prefill=True,
        supports_speculative_decoding=True,
        supports_flash_attention=True,
        supports_paged_attention=True,
        supports_fp8_kv_cache=True,
        supports_tensor_parallel=True,
        supports_pipeline_parallel=True,
        preferred_batch_scheduler="continuous",
        extra={"tensor_parallel_size": 1, "pipeline_parallel_size": 1},
    ),
    "tensorrt": BackendProfile(
        backend_type="tensorrt",
        optimal_batch_size=32,
        preferred_dtype="float16",
        memory_per_token_bytes=2 * 1024 * 1024,
        max_seq_len=8192,
        num_attention_layers=32,
        supports_prefix_caching=False,
        supports_chunked_prefill=False,
        supports_speculative_decoding=False,
        supports_flash_attention=True,
        supports_paged_attention=False,
        supports_fp8_kv_cache=True,
        supports_tensor_parallel=True,
        supports_pipeline_parallel=False,
        preferred_batch_scheduler="static",
        extra={"builder_optimization_level": 5, "use_fp8": True},
    ),
    "onnx": BackendProfile(
        backend_type="onnx",
        optimal_batch_size=8,
        preferred_dtype="float32",
        memory_per_token_bytes=4 * 1024 * 1024,
        max_seq_len=2048,
        num_attention_layers=32,
        supports_prefix_caching=False,
        supports_chunked_prefill=False,
        supports_speculative_decoding=False,
        supports_flash_attention=False,
        supports_paged_attention=False,
        supports_fp8_kv_cache=False,
        supports_tensor_parallel=False,
        supports_pipeline_parallel=False,
        preferred_batch_scheduler="static",
        extra={"execution_provider": "cpu", "intra_op_threads": 4},
    ),
    "llamacpp": BackendProfile(
        backend_type="llamacpp",
        optimal_batch_size=16,
        preferred_dtype="float16",
        memory_per_token_bytes=512 * 1024,
        max_seq_len=8192,
        num_attention_layers=32,
        supports_prefix_caching=True,
        supports_chunked_prefill=False,
        supports_speculative_decoding=False,
        supports_flash_attention=False,
        supports_paged_attention=False,
        supports_fp8_kv_cache=False,
        supports_tensor_parallel=False,
        supports_pipeline_parallel=False,
        preferred_batch_scheduler="continuous",
        extra={"n_gpu_layers": 0, "rope_freq_base": 0.0, "rope_freq_scale": 0.0},
    ),
    "pytorch": BackendProfile(
        backend_type="pytorch",
        optimal_batch_size=4,
        preferred_dtype="bfloat16",
        memory_per_token_bytes=2 * 1024 * 1024,
        max_seq_len=4096,
        num_attention_layers=32,
        supports_prefix_caching=False,
        supports_chunked_prefill=True,
        supports_speculative_decoding=True,
        supports_flash_attention=True,
        supports_paged_attention=False,
        supports_fp8_kv_cache=False,
        supports_tensor_parallel=True,
        supports_pipeline_parallel=True,
        preferred_batch_scheduler="continuous",
        extra={"compile": False, "torch_dtype": "bfloat16"},
    ),
}


# ---------------------------------------------------------------------------
# BackendProfileManager
# ---------------------------------------------------------------------------


class BackendProfileManager:
    """Manages per-backend-type optimization profiles with auto-detection.

    Profiles are registered per *backend type* (not per instance).  Multiple
    backends of the same type share one profile, but individual deployments
    can override specific fields via ``update_profile(backend_id, field, value)``.

    Usage::

        mgr = BackendProfileManager()
        mgr.detect_profile("vllm")           # returns auto-detected profile
        mgr.update_profile("vllm", "optimal_batch_size", 128)
        profile = mgr.get_profile("vllm")
    """

    def __init__(self, known_types: dict[str, BackendProfile] | None = None) -> None:
        self._lock = threading.Lock()
        self._profiles: dict[str, BackendProfile] = {}
        """backend_type -> profile (mutable, overridden by detect/update)."""

        self._instance_overrides: dict[str, dict[str, Any]] = {}
        """backend_id -> {field: value} overrides that take precedence."""

        if known_types:
            self._profiles.update(known_types)

    # ------------------------------------------------------------------
    # Profile management
    # ------------------------------------------------------------------

    def detect_profile(self, backend_type: str) -> BackendProfile | None:
        """Auto-detect capabilities for *backend_type* and return a profile.

        If a known default exists it is returned immediately.  Otherwise
        the method attempts heuristics:
          - Checks for installed packages (``torch``, ``tensorrt``, etc.)
          - Uses ``torch.cuda`` to probe GPU capabilities.

        Returns ``None`` when the type is unrecognised and no heuristics
        matched.
        """
        # Fast path: known default
        known = self._profiles.get(backend_type)
        if known is not None:
            return known

        # Heuristic detection for well-known types
        profile: BackendProfile | None = None
        bt = backend_type.lower()

        if bt == "vllm":
            profile = self._detect_vllm()
        elif bt in ("tensorrt", "tensorrt-llm"):
            profile = self._detect_tensorrt()
        elif bt == "onnx":
            profile = self._detect_onnx()
        elif bt in ("llamacpp", "llama.cpp"):
            profile = self._detect_llamacpp()
        elif bt == "pytorch":
            profile = self._detect_pytorch()

        if profile is not None:
            with self._lock:
                self._profiles[backend_type] = profile
            logger.info(f"Auto-detected profile for '{backend_type}': {profile}")
        else:
            logger.warning(f"No profile could be detected for '{backend_type}'")

        return profile

    def get_profile(self, backend_id: str) -> BackendProfile | None:
        """Return the resolved profile for *backend_id*.

        Resolves as follows:
          1. Start with the base type profile.
          2. Apply any per-instance overrides registered for *backend_id*.

        Returns ``None`` if the backend type cannot be determined.
        """
        actual_type = self._resolve_type(backend_id)
        base = self._profiles.get(actual_type)
        if base is None:
            return None

        overrides = self._instance_overrides.get(backend_id, {})
        if not overrides:
            return base

        return dataclasses.replace(base, **{k: v for k, v in overrides.items() if hasattr(base, k)})

    def update_profile(self, backend_id: str, field: str, value: Any) -> None:
        """Override a single profile field for *backend_id*.

        The base profile for the type is unchanged; only the per-instance
        overrides are modified.  Use this to tune a specific deployment
        (e.g., ``update_profile("gpu-0", "optimal_batch_size", 128)``).
        """
        with self._lock:
            overrides = self._instance_overrides.setdefault(backend_id, {})
            overrides[field] = value
        logger.info(f"Updated profile override '{backend_id}.{field}' = {value!r}")

    def list_profiles(self) -> dict[str, BackendProfile]:
        """Return all known base profiles (keyed by backend type)."""
        with self._lock:
            return dict(self._profiles)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_type(self, backend_id: str) -> str:
        """Map a backend ID to its backend type.

        This implementation uses the ID as-is, assuming callers use
        meaningful names (``"vllm"``, ``"tensorrt-worker-0"``, etc.).
        Subclasses may implement a smarter mapping.
        """
        return backend_id

    # -- Heuristic detectors ------------------------------------------

    @staticmethod
    def _detect_vllm() -> BackendProfile | None:
        try:
            import vllm  # noqa: F401
            return _DEFAULT_PROFILES["vllm"]
        except ImportError:
            return None

    @staticmethod
    def _detect_tensorrt() -> BackendProfile | None:
        try:
            import tensorrt  # noqa: F401
            return _DEFAULT_PROFILES["tensorrt"]
        except ImportError:
            return None

    @staticmethod
    def _detect_onnx() -> BackendProfile | None:
        try:
            import onnxruntime  # noqa: F401
            return _DEFAULT_PROFILES["onnx"]
        except ImportError:
            return None

    @staticmethod
    def _detect_llamacpp() -> BackendProfile | None:
        try:
            import llama_cpp  # noqa: F401
            return _DEFAULT_PROFILES["llamacpp"]
        except ImportError:
            return None

    @staticmethod
    def _detect_pytorch() -> BackendProfile | None:
        try:
            import torch  # noqa: F401
            gpu_count = 0
            try:
                gpu_count = torch.cuda.device_count()
            except Exception:
                pass
            profile = _DEFAULT_PROFILES["pytorch"]
            if gpu_count > 0:
                profile = dataclasses.replace(
                    profile,
                    extra={**profile.extra, "num_gpus": gpu_count},
                )
            return profile
        except ImportError:
            return None
