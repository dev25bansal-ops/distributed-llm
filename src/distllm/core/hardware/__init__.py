"""Multi-architecture hardware abstraction layer.

Provides unified detection, capability profiling, and backend selection
across CUDA, ROCm, MPS, XPU, and CPU architectures.
"""

from distllm.core.hardware.device import DeviceCapabilities, DeviceSpec, DeviceType
from distllm.core.hardware.detect import HardwareDetector
from distllm.core.hardware.capabilities import get_device_capabilities
from distllm.core.hardware.backend import BackendSelector, BackendVariant
from distllm.core.hardware.registry import HardwareRegistry

__all__ = [
    "DeviceType",
    "DeviceSpec",
    "DeviceCapabilities",
    "HardwareDetector",
    "get_device_capabilities",
    "BackendSelector",
    "BackendVariant",
    "HardwareRegistry",
]
