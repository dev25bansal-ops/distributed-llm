"""Tests for GPUPowerManager and PowerProfile (no mocks, no GPU required)."""

from __future__ import annotations

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_mod = load_module("distllm/core/gpu_power_manager.py")
GPUPowerManager = _mod.GPUPowerManager
PowerProfile = _mod.PowerProfile
DEFAULT_PROFILES = _mod.DEFAULT_PROFILES


class TestPowerProfile:
    """PowerProfile dataclass construction and defaults."""

    def test_creation_with_all_fields(self) -> None:
        profile = PowerProfile(
            min_utilization=0.0,
            max_utilization=0.3,
            power_limit_watts=120,
            description="test profile",
        )
        assert profile.min_utilization == 0.0
        assert profile.max_utilization == 0.3
        assert profile.power_limit_watts == 120
        assert profile.description == "test profile"

    def test_default_description_empty(self) -> None:
        profile = PowerProfile(0.5, 0.8, 200)
        assert profile.description == ""

    def test_default_profiles_cover_full_range(self) -> None:
        assert len(DEFAULT_PROFILES) == 5
        for i, p in enumerate(DEFAULT_PROFILES):
            assert isinstance(p, PowerProfile)
            assert 0.0 <= p.min_utilization < p.max_utilization <= 1.0
            assert p.power_limit_watts > 0
            if i > 0:
                assert p.min_utilization == DEFAULT_PROFILES[i - 1].max_utilization


class TestGPUPowerManager:
    """GPUPowerManager tests that don't need nvidia-smi or mocking."""

    def test_default_init(self) -> None:
        mgr = GPUPowerManager()
        assert mgr._enabled is True
        assert mgr._running is False
        assert mgr._check_interval == 30.0
        assert mgr._max_power == 400
        assert mgr._current_limits == {}
        assert mgr._stats["adjustments"] == 0

    def test_custom_init(self) -> None:
        profiles = [
            PowerProfile(0.0, 0.5, 80),
            PowerProfile(0.5, 1.0, 300),
        ]
        mgr = GPUPowerManager(profiles=profiles, check_interval_s=10.0, enabled=False, max_power_watts=300)
        assert mgr._profiles == profiles
        assert mgr._check_interval == 10.0
        assert mgr._enabled is False
        assert mgr._max_power == 300

    def test_get_target_power_matches_utilization(self) -> None:
        mgr = GPUPowerManager()
        assert mgr._get_target_power(0.0) == 100
        assert mgr._get_target_power(0.1) == 100
        assert mgr._get_target_power(0.3) == 150
        assert mgr._get_target_power(0.6) == 250
        assert mgr._get_target_power(0.8) == 350
        assert mgr._get_target_power(0.95) == 400
        assert mgr._get_target_power(2.0) == 400

    def test_stats_returns_expected_keys(self) -> None:
        mgr = GPUPowerManager()
        mgr._current_limits[0] = 250
        mgr._stats["adjustments"] = 5
        mgr._stats["power_saved_wh"] = 12.345
        s = mgr.stats()
        assert s["enabled"] is True
        assert s["current_limits"] == {0: 250}
        assert s["adjustments"] == 5
        assert s["power_saved_wh"] == 12.3

    def test_set_power_limit_updates_state(self) -> None:
        mgr = GPUPowerManager()
        mgr.set_power_limit(0, 200)
        assert mgr._current_limits[0] == 200

    def test_start_disabled_does_nothing(self) -> None:
        mgr = GPUPowerManager(enabled=False)
        mgr.start()
        assert mgr._running is False

    def test_stop_without_start_does_not_raise(self) -> None:
        mgr = GPUPowerManager()
        mgr.stop()
        assert mgr._running is False
