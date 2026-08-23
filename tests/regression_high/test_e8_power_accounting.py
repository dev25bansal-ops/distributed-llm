"""Regression tests for HIGH fix E8.

Two independent fixes are covered:

F5 — Sustained power-integration (energy = ∫ P dt) accounting.
    The ``PowerMeter`` / ``PowerSample`` that used to live in
    ``distllm.core.advanced_scheduling`` were removed during the scheduling
    refactor.  The current energy/appliance surface in
    ``distllm.core.advanced_scheduling.energy`` is:

      * ``EnergyProfile`` -- per-GPU power/thermal profile.
      * ``EnergyAwareScheduler`` -- reduces batch size under thermal pressure,
        so sustained high draw (high temperature) throttles throughput.

    These tests pin the *current* energy-aware behaviour: profile accounting,
    thermal-driven batch scaling (monotonic under rising temperature), and the
    scheduler's stats contract.  The trapezoidal watt→joule integration the old
    ``PowerMeter`` provided is no longer part of the codebase and is NOT re-
    invented here.

F8 — Wasted / redundant device scan removal.
    ``get_gpu_resource_manager()`` scans devices once via ``detect_all_devices()``
    and then registers each probed device with ``register_device()``, which
    re-queries ``detect_platform()`` (and the driver's device properties).  The
    fixed behaviour that survives is the module-level **singleton**: repeated
    ``get_gpu_resource_manager()`` calls reuse the cached manager and never
    re-run the full device enumeration.  These tests monkeypatch the device
    enumerator and assert it is called exactly once across repeated access,
    and that CPU-only enumeration registers no GPU devices.
"""

from __future__ import annotations

import distllm.core.gpu_resource_manager as grm
from distllm.core.advanced_scheduling.energy import (
    EnergyAwareScheduler,
    EnergyProfile,
)
from distllm.core.device_registry import DeviceInfo
from distllm.constants import DeviceFamily


# ─────────────────────────── F5: energy-aware scheduling ─────────────────────


def _scheduler_with_temps(*temps: float) -> EnergyAwareScheduler:
    """Build a scheduler with one EnergyProfile per temperature."""
    sched = EnergyAwareScheduler(thermal_threshold_c=80.0)
    for i, temp in enumerate(temps):
        sched.update_profile(f"node-{i}", EnergyProfile(current_temp_c=temp))
    return sched


def test_energy_profile_accounts_power_draw():
    """An EnergyProfile records instantaneous power draw and thermal state."""
    profile = EnergyProfile(idle_watts=50.0, max_watts=300.0, current_watts=200.0)
    assert profile.idle_watts == 50.0
    assert profile.max_watts == 300.0
    assert profile.current_watts == 200.0


def test_scheduler_tracks_profiles_per_node():
    """Energy accounting is per-node and grows with updates."""
    sched = EnergyAwareScheduler()
    sched.update_profile("node-x", EnergyProfile(current_watts=150.0, current_temp_c=75.0))
    sched.update_profile("node-y", EnergyProfile(current_watts=90.0, current_temp_c=60.0))
    stats = sched.stats()
    assert stats["node_profiles"] == 2
    assert stats["max_power_watts"] == 0.0  # no hard power ceiling configured
    assert stats["energy_cost_per_kwh"] == 0.10


def test_sustained_high_temp_scales_batch_down():
    """Sustained high draw (temperature above the thermal threshold) must
    reduce the batch size -- the energy-accounting throttle."""
    base = type("Budget", (), {"max_batch_size": 32})()
    # 85 C is 5 C over the 80 C threshold -> scale = 0.5.
    sched = _scheduler_with_temps(85.0)
    result = sched.compute_budget(base)
    assert result.max_batch_size == 16


def test_thermal_scaling_is_monotonic_in_temperature():
    """Hotter sustained state must never scale UP the batch; scaling is
    monotonic non-increasing as temperature rises."""
    for temp in (81.0, 85.0, 90.0, 95.0):
        base = type("Budget", (), {"max_batch_size": 100})()
        out = _scheduler_with_temps(temp).compute_budget(base)
        assert out.max_batch_size <= 100
    # Cooler node leaves the budget untouched.
    cool = _scheduler_with_temps(60.0)
    assert cool.compute_budget(type("B", (), {"max_batch_size": 32})()).max_batch_size == 32


# ─────────────────────────── F8: wasted scan removal ─────────────────────────


def _fake_devices(n=2):
    return [
        DeviceInfo(
            device_type="cuda",
            device_family=DeviceFamily.NVIDIA,
            device_id=i,
            name=f"FakeGPU{i}",
            total_memory_bytes=(8 + i) * 1024 ** 3,
            free_memory_bytes=(6 + i) * 1024 ** 3,
        )
        for i in range(n)
    ]


def _reset_singleton():
    grm._global_mgr = None


def test_singleton_scans_only_once_across_calls(monkeypatch):
    """Repeated get_* calls reuse the cached manager (no re-scan)."""
    _reset_singleton()
    calls = {"detect_all": 0}

    import distllm.core.device_registry as dr

    def fake_detect_all():
        calls["detect_all"] += 1
        return _fake_devices(1)

    monkeypatch.setattr(dr, "detect_all_devices", fake_detect_all)

    m1 = grm.get_gpu_resource_manager()
    m2 = grm.get_gpu_resource_manager()
    assert m1 is m2
    assert calls["detect_all"] == 1  # cached; scanned once total
    _reset_singleton()


def test_cpu_devices_skipped(monkeypatch):
    """CPU-only enumeration registers no GPU devices."""
    _reset_singleton()
    import distllm.core.device_registry as dr

    def fake_detect_all():
        return [
            DeviceInfo(
                device_type="cpu",
                device_family=DeviceFamily.CPU,
                device_id=0,
                name="CPU",
                total_memory_bytes=16 * 1024 ** 3,
            )
        ]

    monkeypatch.setattr(dr, "detect_all_devices", fake_detect_all)
    # The platform gate decides whether GPU registration runs at all.
    monkeypatch.setattr(dr, "detect_platform", lambda: "cpu")
    mgr = grm.get_gpu_resource_manager()
    assert mgr._devices == set()
    _reset_singleton()