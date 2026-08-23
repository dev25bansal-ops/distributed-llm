"""Tests for EnergyProfile and EnergyAwareScheduler."""

from __future__ import annotations

from types import SimpleNamespace

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_energy = load_module("distllm/core/advanced_scheduling/energy.py")
EnergyProfile = _energy.EnergyProfile
EnergyAwareScheduler = _energy.EnergyAwareScheduler


class TestEnergyProfile:
    """Test suite for EnergyProfile dataclass."""

    def test_default_construction(self) -> None:
        profile = EnergyProfile()
        assert profile.idle_watts == 50.0
        assert profile.max_watts == 300.0
        assert profile.current_watts == 100.0
        assert profile.thermal_limit_c == 83.0
        assert profile.current_temp_c == 60.0

    def test_custom_values(self) -> None:
        profile = EnergyProfile(
            idle_watts=75.0,
            max_watts=450.0,
            current_watts=200.0,
            thermal_limit_c=90.0,
            current_temp_c=85.0,
        )
        assert profile.current_watts == 200.0
        assert profile.current_temp_c == 85.0


class TestEnergyAwareScheduler:
    """Test suite for EnergyAwareScheduler."""

    def test_default_construction(self) -> None:
        scheduler = EnergyAwareScheduler()
        assert scheduler._thermal_threshold == 80.0
        assert scheduler._profiles == {}

    def test_custom_thermal_threshold(self) -> None:
        scheduler = EnergyAwareScheduler(thermal_threshold_c=90.0)
        assert scheduler._thermal_threshold == 90.0

    def test_update_profile(self) -> None:
        scheduler = EnergyAwareScheduler()
        profile = EnergyProfile(
            idle_watts=50.0, max_watts=300.0,
            current_watts=150.0, current_temp_c=75.0,
        )
        scheduler.update_profile("node-x", profile)
        assert scheduler._profiles["node-x"] is profile

    def test_compute_budget_below_threshold_unchanged(self) -> None:
        scheduler = EnergyAwareScheduler(thermal_threshold_c=80.0)
        profile = EnergyProfile(current_temp_c=60.0)
        scheduler.update_profile("node-x", profile)

        base_budget = SimpleNamespace(max_batch_size=32)
        result = scheduler.compute_budget(base_budget)
        assert result.max_batch_size == 32

    def test_compute_budget_above_threshold_scales_down(self) -> None:
        scheduler = EnergyAwareScheduler(thermal_threshold_c=80.0)
        profile = EnergyProfile(current_temp_c=85.0)
        scheduler.update_profile("node-x", profile)

        base_budget = SimpleNamespace(max_batch_size=32)
        result = scheduler.compute_budget(base_budget)
        # scale = max(0.5, 1.0 - (85 - 80) / 10.0) = max(0.5, 0.5) = 0.5
        assert result.max_batch_size == 16

    def test_compute_budget_hottest_node_drives_scaling(self) -> None:
        scheduler = EnergyAwareScheduler(thermal_threshold_c=80.0)
        scheduler.update_profile("cool", EnergyProfile(current_temp_c=60.0))
        scheduler.update_profile("hot", EnergyProfile(current_temp_c=95.0))

        base_budget = SimpleNamespace(max_batch_size=32)
        result = scheduler.compute_budget(base_budget)
        # max_temp = 95, scale = max(0.5, 1.0 - (95-80)/10) = max(0.5, -0.5) = 0.5
        assert result.max_batch_size == 16

    def test_compute_budget_no_profiles_returns_base(self) -> None:
        scheduler = EnergyAwareScheduler()
        base_budget = SimpleNamespace(max_batch_size=32)
        result = scheduler.compute_budget(base_budget)
        assert result is base_budget

    def test_compute_budget_moderate_thermal(self) -> None:
        scheduler = EnergyAwareScheduler(thermal_threshold_c=80.0)
        profile = EnergyProfile(current_temp_c=83.0)
        scheduler.update_profile("node-x", profile)

        base_budget = SimpleNamespace(max_batch_size=100)
        result = scheduler.compute_budget(base_budget)
        # scale = max(0.5, 1.0 - (83-80)/10) = max(0.5, 0.7) = 0.7
        assert result.max_batch_size == 70

    def test_on_before_schedule_passthrough(self) -> None:
        scheduler = EnergyAwareScheduler()
        seqs = ["a", "b", "c"]
        assert scheduler.on_before_schedule(seqs) == ["a", "b", "c"]
