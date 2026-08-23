"""Tests for GPUResourceManager, Allocation, GPUMemorySnapshot, MemoryPriority.

No GPU required -- tests focus on data classes, constructor, release, touch,
and allocation logic that degrades gracefully without CUDA.
"""

from __future__ import annotations

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_mod = load_module("distllm/core/gpu_resource_manager.py")
GPUResourceManager = _mod.GPUResourceManager
Allocation = _mod.Allocation
GPUMemorySnapshot = _mod.GPUMemorySnapshot
MemoryPriority = _mod.MemoryPriority
get_gpu_resource_manager = _mod.get_gpu_resource_manager


class TestMemoryPriority:
    """MemoryPriority enum values."""

    def test_priority_values(self) -> None:
        assert MemoryPriority.CRITICAL.value == 0
        assert MemoryPriority.HIGH.value == 1
        assert MemoryPriority.NORMAL.value == 2
        assert MemoryPriority.LOW.value == 3

    def test_ordering(self) -> None:
        assert MemoryPriority.CRITICAL < MemoryPriority.HIGH
        assert MemoryPriority.HIGH < MemoryPriority.NORMAL
        assert MemoryPriority.NORMAL < MemoryPriority.LOW


class TestAllocation:
    """Allocation dataclass construction."""

    def test_creation_with_required_fields(self) -> None:
        alloc = Allocation(key="test", size_mb=100, device=0, priority=MemoryPriority.NORMAL)
        assert alloc.key == "test"
        assert alloc.size_mb == 100
        assert alloc.device == 0
        assert alloc.priority == MemoryPriority.NORMAL

    def test_default_values(self) -> None:
        alloc = Allocation(key="test", size_mb=50, device=1, priority=MemoryPriority.HIGH)
        assert alloc.allocated_at == 0.0
        assert alloc.last_used == 0.0
        assert alloc.owner == ""

    def test_creation_with_all_fields(self) -> None:
        alloc = Allocation(
            key="model-falcon", size_mb=14000, device=0,
            priority=MemoryPriority.CRITICAL, allocated_at=100.0,
            last_used=200.0, owner="loader",
        )
        assert alloc.key == "model-falcon"
        assert alloc.owner == "loader"


class TestGPUMemorySnapshot:
    """GPUMemorySnapshot dataclass."""

    def test_creation(self) -> None:
        snap = GPUMemorySnapshot(
            device=0, total_mb=8192, used_mb=4096, free_mb=4096,
            reserved_mb=1024, allocated_mb=2048, utilization_pct=50.0,
            temperature_c=65.0,
        )
        assert snap.device == 0
        assert snap.total_mb == 8192
        assert snap.used_mb == 4096
        assert snap.free_mb == 4096

    def test_empty_allocations(self) -> None:
        snap = GPUMemorySnapshot(
            device=0, total_mb=8192, used_mb=0, free_mb=8192,
            reserved_mb=0, allocated_mb=0, utilization_pct=0.0,
            temperature_c=30.0,
        )
        assert snap.allocations == []


class TestGPUResourceManager:
    """GPUResourceManager -- construction and non-GPU operations."""

    def test_default_construction(self) -> None:
        mgr = GPUResourceManager()
        assert mgr._safety_margin == 512.0
        assert mgr._allocations == {}
        assert mgr._devices == set()
        assert mgr._total_per_device == {}
        assert mgr._eviction_callbacks == []

    def test_custom_safety_margin(self) -> None:
        mgr = GPUResourceManager(safety_margin_mb=256.0)
        assert mgr._safety_margin == 256.0

    def test_release_returns_false_for_missing(self) -> None:
        mgr = GPUResourceManager()
        assert mgr.release("nonexistent") is False

    def test_release_returns_true_for_existing(self) -> None:
        mgr = GPUResourceManager()
        mgr._allocations["test"] = Allocation(
            key="test", size_mb=100, device=0, priority=MemoryPriority.NORMAL,
        )
        assert mgr.release("test") is True
        assert "test" not in mgr._allocations

    def test_touch_does_not_raise_for_missing(self) -> None:
        mgr = GPUResourceManager()
        mgr.touch("nonexistent")  # should not raise

    def test_touch_updates_last_used(self) -> None:
        mgr = GPUResourceManager()
        mgr._allocations["test"] = Allocation(
            key="test", size_mb=100, device=0, priority=MemoryPriority.NORMAL,
        )
        old_time = mgr._allocations["test"].last_used
        mgr.touch("test")
        assert mgr._allocations["test"].last_used >= old_time

    def test_unregister_device_removes_allocation(self) -> None:
        mgr = GPUResourceManager()
        mgr._devices.add(0)
        mgr._allocations["a"] = Allocation(
            key="a", size_mb=50, device=0, priority=MemoryPriority.NORMAL,
        )
        mgr.unregister_device(0)
        assert 0 not in mgr._devices
        assert "a" not in mgr._allocations

    def test_unregister_device_does_not_raise_for_unknown(self) -> None:
        mgr = GPUResourceManager()
        mgr.unregister_device(99)  # should not raise

    def test_total_mb_returns_zero_for_unregistered(self) -> None:
        mgr = GPUResourceManager()
        assert mgr.total_mb(0) == 0.0

    def test_is_oom_risk_returns_false_with_no_devices(self) -> None:
        mgr = GPUResourceManager()
        assert mgr.is_oom_risk(0) is False

    def test_on_eviction_registers_callback(self) -> None:
        mgr = GPUResourceManager()
        calls = []

        def cb(key, alloc):
            calls.append(key)

        mgr.on_eviction(cb)
        assert len(mgr._eviction_callbacks) == 1

    def test_try_allocate_returns_tuple(self) -> None:
        """try_allocate always returns a (bool, Allocation|None) tuple."""
        mgr = GPUResourceManager()
        ok, alloc = mgr.try_allocate("test", 10)
        assert isinstance(ok, bool)
        assert alloc is None or isinstance(alloc, Allocation)

    def test_get_gpu_resource_manager_returns_singleton(self) -> None:
        mgr1 = get_gpu_resource_manager()
        mgr2 = get_gpu_resource_manager()
        assert mgr1 is mgr2
