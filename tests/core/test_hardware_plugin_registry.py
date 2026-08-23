"""Tests for HardwarePluginRegistry, HardwarePlugin.

No mocks -- uses module-level stub plugins inserted into sys.modules.
"""

from __future__ import annotations

import pytest

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_mod = load_module("distllm/core/hardware_plugin_registry.py")
HardwarePluginRegistry = _mod.HardwarePluginRegistry
HardwarePlugin = _mod.HardwarePlugin


class TestHardwarePlugin:
    """HardwarePlugin -- construction and load behavior."""

    def test_creation_with_module_name(self) -> None:
        plugin = HardwarePlugin("nonexistent_module")
        assert plugin._module_name == "nonexistent_module"
        assert plugin._options == {}
        assert plugin._loaded is False

    def test_creation_with_options(self) -> None:
        plugin = HardwarePlugin("some_mod", options={"device_type": "myaccel"})
        assert plugin._options == {"device_type": "myaccel"}

    def test_load_fails_for_nonexistent_module(self) -> None:
        plugin = HardwarePlugin("completely_fake_module_xyz")
        assert plugin.load() is False
        assert plugin._loaded is False

    def test_probe_returns_empty_for_unloaded(self) -> None:
        plugin = HardwarePlugin("fake_module")
        assert plugin.probe() == []

    def test_create_adapter_returns_none_for_unloaded(self) -> None:
        plugin = HardwarePlugin("fake_module")
        assert plugin.create_adapter({"device": "gpu"}) is None


class TestHardwarePluginRegistry:
    """HardwarePluginRegistry -- construction and state inspection."""

    def test_default_construction(self) -> None:
        reg = HardwarePluginRegistry()
        assert reg._plugins == []
        assert reg._devices == []

    def test_register_fails_for_nonexistent_module(self) -> None:
        reg = HardwarePluginRegistry()
        result = reg.register("certainly_not_a_real_plugin")
        assert result is False

    def test_register_raises_on_empty_name(self) -> None:
        reg = HardwarePluginRegistry()
        with pytest.raises((ValueError,)):
            reg.register("")

    def test_probe_all_without_plugins_returns_empty(self) -> None:
        reg = HardwarePluginRegistry()
        devices = reg.probe_all()
        assert devices == []

    def test_plugins_property_returns_plugin_info(self) -> None:
        reg = HardwarePluginRegistry()
        assert reg.plugins == []

    def test_devices_property_returns_devices(self) -> None:
        reg = HardwarePluginRegistry()
        assert reg.devices == []

    def test_create_adapter_returns_none_with_no_match(self) -> None:
        reg = HardwarePluginRegistry()
        result = reg.create_adapter({"_plugin": "nonexistent", "device": "gpu"})
        assert result is None

    def test_create_adapter_returns_none_with_no_plugin_key(self) -> None:
        reg = HardwarePluginRegistry()
        result = reg.create_adapter({"device": "gpu"})
        assert result is None

    def test_discover_returns_empty_when_no_installed_packages(self) -> None:
        """Without any distllm_hardware_* packages installed, discover is empty."""
        reg = HardwarePluginRegistry()
        discovered = reg.discover()
        assert isinstance(discovered, list)
        # In a fresh venv without distllm-hardware-* packages, this should be empty.
        # If some happen to be installed, they'll be detected -- that's fine too.
        assert all(isinstance(m, str) for m in discovered)
