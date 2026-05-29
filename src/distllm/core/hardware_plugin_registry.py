"""Hardware Plugin SDK — register custom devices for distributed inference.

Allows third-party hardware vendors to add support for new AI accelerators
without modifying the core DistLLM codebase.  Plugins are installed via
``pip install distllm-hardware-mydevice`` and activated via config.

Usage in config.yaml::

    plugins:
      - module: distllm_hardware_mydevice
        options:
          device_type: "myaccel"
          priority: 10

The plugin module must expose::

    def probe() -> list[dict]:  # Returns detected devices
    def create_adapter(device_info: dict) -> object:  # Returns backend adapter
"""

from __future__ import annotations

import importlib
import os
import pkgutil
from typing import Any, Callable

from loguru import logger


class HardwarePlugin:
    """Represents a single hardware plugin instance.

    Wraps a third-party module that exposes ``probe()`` and
    ``create_adapter()`` functions.
    """

    def __init__(self, module_name: str, options: dict[str, Any] | None = None):
        self._module_name = module_name
        self._options = options or {}
        self._module: Any = None
        self._loaded = False

    def load(self) -> bool:
        try:
            self._module = importlib.import_module(self._module_name)
            self._loaded = True
            logger.info(f"Hardware plugin loaded: {self._module_name}")
            return True
        except ImportError as e:
            logger.warning(f"Hardware plugin not available: {self._module_name} ({e})")
            return False

    def probe(self) -> list[dict[str, Any]]:
        if not self._loaded or not hasattr(self._module, 'probe'):
            return []
        try:
            return self._module.probe(**self._options)
        except Exception as e:
            logger.error(f"Plugin probe failed for {self._module_name}: {e}")
            return []

    def create_adapter(self, device_info: dict[str, Any]) -> Any:
        if not self._loaded or not hasattr(self._module, 'create_adapter'):
            return None
        try:
            return self._module.create_adapter(device_info, **self._options)
        except Exception as e:
            logger.error(f"Plugin create_adapter failed for {self._module_name}: {e}")
            return None


class HardwarePluginRegistry:
    """Manages all registered hardware plugins.

    Auto-discovers installed ``distllm-hardware-*`` packages via
    ``pkgutil``, plus manually registered plugins from config.
    """

    def __init__(self):
        self._plugins: list[HardwarePlugin] = []
        self._devices: list[dict[str, Any]] = []

    def discover(self) -> list[str]:
        """Auto-discover installed distllm-hardware-* packages."""
        discovered = []
        for importer, modname, ispkg in pkgutil.iter_modules():
            if modname.startswith("distllm_hardware_") or modname.startswith("distllm-hardware-"):
                plugin = HardwarePlugin(modname)
                if plugin.load():
                    self._plugins.append(plugin)
                    discovered.append(modname)
        return discovered

    def register(self, module_name: str, options: dict[str, Any] | None = None) -> bool:
        plugin = HardwarePlugin(module_name, options)
        if plugin.load():
            self._plugins.append(plugin)
            return True
        return False

    def probe_all(self) -> list[dict[str, Any]]:
        """Probe all loaded plugins and collect detected devices."""
        self._devices = []
        for plugin in self._plugins:
            devices = plugin.probe()
            self._devices.extend(devices)
            for d in devices:
                d["_plugin"] = plugin._module_name
        return self._devices

    def create_adapter(self, device_info: dict[str, Any]) -> Any | None:
        """Create a backend adapter for a specific device."""
        plugin_name = device_info.get("_plugin", "")
        for plugin in self._plugins:
            if plugin._module_name == plugin_name:
                return plugin.create_adapter(device_info)
        return None

    @property
    def plugins(self) -> list[dict[str, Any]]:
        return [{"module": p._module_name, "loaded": p._loaded} for p in self._plugins]

    @property
    def devices(self) -> list[dict[str, Any]]:
        return self._devices
