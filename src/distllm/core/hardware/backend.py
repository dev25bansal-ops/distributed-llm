"""Backend selection with graceful import handling per architecture.

BackendVariant enum + BackendSelector that resolves the best backend
for a given device type, handling optional imports gracefully.
"""

from __future__ import annotations

import enum
from typing import TYPE_CHECKING

from loguru import logger

from distllm.core.hardware.device import DeviceSpec, DeviceType

if TYPE_CHECKING:
    pass


class BackendVariant(str, enum.Enum):
    """Supported inference backend variants."""

    VLLM = "vllm"          # vLLM (CUDA / ROCm)
    PYTORCH = "pytorch"    # Native PyTorch
    LLAMACPP = "llamacpp"  # llama.cpp (CPU / CUDA / ROCm / Metal)


class BackendSelector:
    """Selects and validates backend availability per device type.

    Provides graceful import handling so that missing optional
    dependencies don't raise at import time.
    """

    # Mapping of device type -> list of (variant, availability_check)
    _BACKEND_PRIORITY: dict[DeviceType, list[tuple[BackendVariant, str]]] = {
        DeviceType.CUDA: [
            (BackendVariant.VLLM, "distllm.backends.vllm_backend"),
            (BackendVariant.PYTORCH, "distllm.backends.pytorch_backend"),
            (BackendVariant.LLAMACPP, "distllm.backends.llamacpp_backend"),
        ],
        DeviceType.ROCM: [
            (BackendVariant.VLLM, "distllm.backends.vllm_backend"),
            (BackendVariant.PYTORCH, "distllm.backends.pytorch_backend"),
            (BackendVariant.LLAMACPP, "distllm.backends.llamacpp_backend"),
        ],
        DeviceType.MPS: [
            (BackendVariant.PYTORCH, "distllm.backends.pytorch_backend"),
            (BackendVariant.LLAMACPP, "distllm.backends.llamacpp_backend"),
        ],
        DeviceType.XPU: [
            (BackendVariant.PYTORCH, "distllm.backends.pytorch_backend"),
        ],
        DeviceType.CPU: [
            (BackendVariant.LLAMACPP, "distllm.backends.llamacpp_backend"),
            (BackendVariant.PYTORCH, "distllm.backends.pytorch_backend"),
        ],
        DeviceType.UNKNOWN: [
            (BackendVariant.PYTORCH, "distllm.backends.pytorch_backend"),
        ],
    }

    def __init__(self) -> None:
        self._import_cache: dict[str, bool] = {}

    def preferred_backend(self, device: DeviceSpec) -> BackendVariant:
        """Return the highest-priority available backend for *device*.

        Falls back to a safe default (PyTorch) if nothing is importable.
        """
        candidates = self._BACKEND_PRIORITY.get(device.device_type, [])
        for variant, module_path in candidates:
            if self._is_importable(module_path):
                logger.debug(
                    "Selected backend {} for {} device {}",
                    variant.value,
                    device.device_type.value,
                    device.device_id,
                )
                return variant

        logger.warning(
            "No specialized backend for {} device {}; falling back to PyTorch",
            device.device_type.value,
            device.device_id,
        )
        return BackendVariant.PYTORCH

    def available_backends(self, device: DeviceSpec) -> list[BackendVariant]:
        """List all backends that can be imported for *device*."""
        candidates = self._BACKEND_PRIORITY.get(device.device_type, [])
        return [v for v, m in candidates if self._is_importable(m)]

    def _is_importable(self, module_path: str) -> bool:
        if module_path in self._import_cache:
            return self._import_cache[module_path]

        try:
            __import__(module_path)
            self._import_cache[module_path] = True
        except ImportError:
            self._import_cache[module_path] = False
        return self._import_cache[module_path]

    def get_adapter_class(
        self, variant: BackendVariant
    ) -> type | None:
        """Return the NodeAdapter class for *variant*, or None."""
        mapping: dict[BackendVariant, str] = {
            BackendVariant.VLLM: "distllm.backends.vllm_backend.VLLMNodeAdapter",
            BackendVariant.PYTORCH: "distllm.backends.pytorch_backend.PyTorchNodeAdapter",
            BackendVariant.LLAMACPP: "distllm.backends.llamacpp_backend.LlamacppNodeAdapter",
        }
        path = mapping.get(variant)
        if path is None:
            return None
        return self._import_class(path)

    def _import_class(self, dotted_path: str) -> type | None:
        module_path, _, class_name = dotted_path.rpartition(".")
        try:
            mod = __import__(module_path, fromlist=[class_name])
            return getattr(mod, class_name, None)
        except ImportError:
            logger.debug("Could not import {}", dotted_path)
            return None
