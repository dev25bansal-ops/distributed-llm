"""Global hardware registry for device management.

Provides a singleton registry that stores detected devices, caches
capability lookups, and tracks the currently selected device.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from distllm.core.hardware.capabilities import get_device_capabilities
from distllm.core.hardware.detect import HardwareDetector
from distllm.core.hardware.device import DeviceCapabilities, DeviceSpec, DeviceType


class HardwareRegistry:
    """Singleton registry for all detected hardware devices.

    Usage::

        registry = HardwareRegistry()
        registry.detect()
        primary = registry.primary_device
        caps = registry.get_capabilities(primary)
    """

    _instance: HardwareRegistry | None = None

    def __new__(cls) -> HardwareRegistry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self._detector = HardwareDetector()
        self._devices: list[DeviceSpec] = []
        self._capabilities: dict[int, DeviceCapabilities] = {}
        self._selected_device_id: int | None = None

    # -- Detection -----------------------------------------------------------

    def detect(self, force: bool = False) -> list[DeviceSpec]:
        """Run hardware detection, caching results.

        Args:
            force: If True, re-detect even if already detected.

        Returns:
            List of detected DeviceSpec.
        """
        if self._devices and not force:
            return self._devices

        self._devices = self._detector.detect_all()
        self._capabilities.clear()

        for dev in self._devices:
            caps = get_device_capabilities(dev)
            self._capabilities[dev.device_id] = caps

        if self._selected_device_id is None and self._devices:
            self._selected_device_id = self._devices[0].device_id

        logger.info("Hardware registry: {} device(s) detected", len(self._devices))
        for d in self._devices:
            logger.info("  {}", d.summary())

        return self._devices

    # -- Accessors -----------------------------------------------------------

    @property
    def devices(self) -> list[DeviceSpec]:
        return list(self._devices)

    @property
    def primary_device(self) -> DeviceSpec | None:
        """Return the first accelerator, or the first device, or None."""
        if not self._devices:
            return None
        for dev in self._devices:
            if dev.is_accelerator:
                return dev
        return self._devices[0]

    @property
    def selected_device(self) -> DeviceSpec | None:
        if self._selected_device_id is None:
            return None
        for dev in self._devices:
            if dev.device_id == self._selected_device_id:
                return dev
        return None

    def select_device(self, device_id: int) -> None:
        self._selected_device_id = device_id

    # -- Capabilities --------------------------------------------------------

    def get_capabilities(self, device: DeviceSpec | None = None) -> DeviceCapabilities | None:
        """Return cached capabilities for *device* (or primary)."""
        if device is None:
            device = self.primary_device
        if device is None:
            return None
        if device.device_id not in self._capabilities:
            caps = get_device_capabilities(device)
            self._capabilities[device.device_id] = caps
        return self._capabilities[device.device_id]

    def get_device_ids(self) -> list[int]:
        return [d.device_id for d in self._devices]

    def count_by_type(self, device_type: DeviceType) -> int:
        return sum(1 for d in self._devices if d.device_type == device_type)

    # -- Summary -------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        return {
            "total_devices": len(self._devices),
            "primary": self.primary_device.summary() if self.primary_device else None,
            "by_type": {t.value: self.count_by_type(t) for t in DeviceType},
        }

    # -- Lifecycle -----------------------------------------------------------

    def reset(self) -> None:
        self._devices.clear()
        self._capabilities.clear()
        self._selected_device_id = None
