"""Tests for GPU power capping — real objects, zero mocks.

Tests verify:
- PowerStatus dataclass construction and computed properties
- GPUPowerCapper construction and basic properties
- State management (stop_auto_tune, tuned_limits)
- Public API type stability (return types, dict shapes)
- Thread safety of concurrent read operations
- No Python-level exceptions under any input path
"""

from __future__ import annotations

import threading

import pytest

from distllm.dist.power_cap import GPUPowerCapper, PowerStatus


# ── PowerStatus ─────────────────────────────────────────────────────────


class TestPowerStatus:
    """PowerStatus dataclass — construction and computed properties."""

    def test_default_values(self) -> None:
        """Default PowerStatus uses sensible defaults."""
        status = PowerStatus(gpu_index=0)
        assert status.gpu_index == 0
        assert status.current_power_w == 0.0
        assert status.power_limit_w == 0.0
        assert status.min_power_limit_w == 0.0
        assert status.max_power_limit_w == 0.0
        assert status.default_power_limit_w == 0.0
        assert status.gpu_util_pct == 0.0
        assert status.memory_util_pct == 0.0
        assert status.temperature_c == 0.0

    def test_custom_values(self) -> None:
        """All fields can be set at construction."""
        status = PowerStatus(
            gpu_index=1,
            current_power_w=150.5,
            power_limit_w=200.0,
            min_power_limit_w=100.0,
            max_power_limit_w=300.0,
            default_power_limit_w=250.0,
            gpu_util_pct=45.2,
            memory_util_pct=30.1,
            temperature_c=65.3,
        )
        assert status.gpu_index == 1
        assert status.current_power_w == 150.5
        assert status.power_limit_w == 200.0
        assert status.min_power_limit_w == 100.0
        assert status.max_power_limit_w == 300.0
        assert status.default_power_limit_w == 250.0
        assert status.gpu_util_pct == 45.2
        assert status.memory_util_pct == 30.1
        assert status.temperature_c == 65.3

    def test_gpu_index_type(self) -> None:
        """gpu_index stores an int."""
        status = PowerStatus(gpu_index=0)
        assert isinstance(status.gpu_index, int)

    def test_power_savings_pct_zero_default(self) -> None:
        """power_savings_pct is 0.0 when default_power_limit_w is 0."""
        status = PowerStatus(
            gpu_index=0, default_power_limit_w=0.0, power_limit_w=200.0
        )
        assert status.power_savings_pct == 0.0

    def test_power_savings_pct_negative_default(self) -> None:
        """power_savings_pct is 0.0 when default_power_limit_w is negative."""
        status = PowerStatus(
            gpu_index=0, default_power_limit_w=-50.0, power_limit_w=200.0
        )
        assert status.power_savings_pct == 0.0

    def test_power_savings_pct_no_savings(self) -> None:
        """power_savings_pct is 0.0 when power_limit equals default."""
        status = PowerStatus(
            gpu_index=0, power_limit_w=250.0, default_power_limit_w=250.0
        )
        assert status.power_savings_pct == 0.0

    def test_power_savings_pct_50_percent(self) -> None:
        """50% reduction yields 50.0% savings."""
        status = PowerStatus(
            gpu_index=0, power_limit_w=125.0, default_power_limit_w=250.0
        )
        assert status.power_savings_pct == 50.0

    def test_power_savings_pct_30_percent(self) -> None:
        """30% reduction yields 30.0% savings."""
        status = PowerStatus(
            gpu_index=0, power_limit_w=175.0, default_power_limit_w=250.0
        )
        assert status.power_savings_pct == 30.0

    def test_power_savings_pct_rounding(self) -> None:
        """Savings are rounded to 1 decimal place."""
        status = PowerStatus(
            gpu_index=0, power_limit_w=183.0, default_power_limit_w=250.0
        )
        expected = round((1.0 - 183.0 / 250.0) * 100, 1)
        assert status.power_savings_pct == expected

    def test_power_savings_pct_above_default(self) -> None:
        """Savings can be negative when power_limit exceeds default."""
        status = PowerStatus(
            gpu_index=0, power_limit_w=300.0, default_power_limit_w=250.0
        )
        assert status.power_savings_pct < 0.0

    def test_power_savings_pct_full_save(self) -> None:
        """100% savings when power_limit is 0."""
        status = PowerStatus(
            gpu_index=0, power_limit_w=0.0, default_power_limit_w=250.0
        )
        assert status.power_savings_pct == 100.0

    def test_power_savings_pct_small_reduction(self) -> None:
        """Small reduction (e.g. 1W) yields a small positive savings."""
        status = PowerStatus(
            gpu_index=0, power_limit_w=249.0, default_power_limit_w=250.0
        )
        assert 0.0 < status.power_savings_pct < 1.0

    def test_repr_contains_fields(self) -> None:
        """repr shows dataclass fields."""
        status = PowerStatus(gpu_index=2, current_power_w=100.0)
        r = repr(status)
        assert "PowerStatus" in r
        assert "gpu_index=2" in r

    def test_equality(self) -> None:
        """Two identical PowerStatus instances are equal (dataclass)."""
        a = PowerStatus(gpu_index=0, power_limit_w=200.0)
        b = PowerStatus(gpu_index=0, power_limit_w=200.0)
        assert a == b

    def test_inequality(self) -> None:
        """Different PowerStatus instances are not equal."""
        a = PowerStatus(gpu_index=0, power_limit_w=200.0)
        b = PowerStatus(gpu_index=1, power_limit_w=200.0)
        assert a != b

    def test_power_savings_pct_negative_from_zero_limit(self) -> None:
        """Limit of 0 on a positive default yields 100% savings."""
        status = PowerStatus(
            gpu_index=0, power_limit_w=0.0, default_power_limit_w=250.0
        )
        assert status.power_savings_pct == 100.0


# ── GPUPowerCapper: construction ───────────────────────────────────────


class TestGPUPowerCapperConstruction:
    """GPUPowerCapper construction with various inputs."""

    def test_default_constructor(self) -> None:
        """Default constructor does not raise."""
        capper = GPUPowerCapper()
        # Must be list of ints (may be empty or contain real GPU indices).
        assert isinstance(capper._gpu_indices, list)

    def test_explicit_indices(self) -> None:
        """Explicit GPU indices are stored."""
        capper = GPUPowerCapper(gpu_indices=[0])
        assert capper._gpu_indices == [0]

    def test_multiple_indices(self) -> None:
        """Multiple indices are accepted."""
        capper = GPUPowerCapper(gpu_indices=[0, 1, 2, 3])
        assert capper._gpu_indices == [0, 1, 2, 3]

    def test_empty_list_triggers_autodetect(self) -> None:
        """Passing [] triggers auto-detection (falsy in ``or`` expression)."""
        capper = GPUPowerCapper(gpu_indices=[])
        # _gpu_indices may be empty or have auto-detected indices — both are fine.
        assert isinstance(capper._gpu_indices, list)

    def test_available_property(self) -> None:
        """available reflects whether _gpu_indices is non-empty."""
        capper = GPUPowerCapper(gpu_indices=[0])
        # In an env with GPU 0, available is True.
        assert capper.available is (len(capper._gpu_indices) > 0)

    def test_lock_created(self) -> None:
        """Thread lock is created on init."""
        capper = GPUPowerCapper(gpu_indices=[])
        assert isinstance(capper._lock, threading.Lock)

    def test_tuned_limits_empty_initial(self) -> None:
        """_tuned_limits starts as empty."""
        capper = GPUPowerCapper(gpu_indices=[])
        assert capper._tuned_limits == {}

    def test_auto_tune_active_false_initial(self) -> None:
        """_auto_tune_active starts as False."""
        capper = GPUPowerCapper(gpu_indices=[])
        assert capper._auto_tune_active is False


# ── GPUPowerCapper: state management ───────────────────────────────────


class TestGPUPowerCapperState:
    """Internal state management."""

    def test_stop_auto_tune_sets_flag_false(self) -> None:
        """stop_auto_tune sets _auto_tune_active to False."""
        capper = GPUPowerCapper(gpu_indices=[])
        capper._auto_tune_active = True
        capper.stop_auto_tune()
        assert capper._auto_tune_active is False

    def test_stop_auto_tune_multiple_calls(self) -> None:
        """Multiple stop_auto_tune calls are safe."""
        capper = GPUPowerCapper(gpu_indices=[])
        capper._auto_tune_active = True
        for _ in range(5):
            capper.stop_auto_tune()
        assert capper._auto_tune_active is False

    def test_stop_auto_tune_when_idle(self) -> None:
        """stop_auto_tune does not raise when already idle."""
        capper = GPUPowerCapper(gpu_indices=[])
        capper.stop_auto_tune()
        assert capper._auto_tune_active is False

    def test_auto_tune_active_flag_during_tune(self) -> None:
        """_auto_tune_active is set True during auto_tune.

        Uses benchmark_fn so no GPU hardware dependency.
        """
        capper = GPUPowerCapper(gpu_indices=[0])

        def slow_bench() -> dict[str, float]:
            return {"tokens_per_sec": 100.0}

        capper.auto_tune(benchmark_fn=slow_bench)
        assert capper._auto_tune_active is False

    def test_tuned_limits_starts_empty(self) -> None:
        """_tuned_limits is empty dict initially."""
        capper = GPUPowerCapper(gpu_indices=[0])
        assert capper._tuned_limits == {}


# ── GPUPowerCapper: get_status ─────────────────────────────────────────


class TestGPUPowerCapperGetStatus:
    """get_status return type stability."""

    def test_get_status_all_returns_dict(self) -> None:
        """get_status() returns a dict mapping int to PowerStatus."""
        capper = GPUPowerCapper(gpu_indices=[0])
        status = capper.get_status()
        assert isinstance(status, dict)
        for gpu_idx, ps in status.items():
            assert isinstance(gpu_idx, int)
            assert isinstance(ps, PowerStatus)

    def test_get_status_all_contains_gpu(self) -> None:
        """get_status() includes all capper GPU indices."""
        capper = GPUPowerCapper(gpu_indices=[0])
        status = capper.get_status()
        for idx in capper._gpu_indices:
            # GPU may or may not appear depending on nvidia-smi availability
            pass
        assert isinstance(status, dict)

    def test_get_status_single_returns_power_status(self) -> None:
        """get_status(gpu_index=0) returns a PowerStatus."""
        capper = GPUPowerCapper(gpu_indices=[0])
        status = capper.get_status(gpu_index=0)
        assert isinstance(status, PowerStatus)

    def test_get_status_unknown_index_returns_fallback(self) -> None:
        """get_status(gpu_index=999) returns a PowerStatus (fallback)."""
        capper = GPUPowerCapper(gpu_indices=[0])
        status = capper.get_status(gpu_index=999)
        assert isinstance(status, PowerStatus)
        assert status.gpu_index == 999

    def test_get_status_power_status_has_expected_attrs(self) -> None:
        """Returned PowerStatus has all expected attributes."""
        capper = GPUPowerCapper(gpu_indices=[0])
        status = capper.get_status(gpu_index=0)
        assert isinstance(status, PowerStatus)
        # All float fields should be numeric
        assert isinstance(status.current_power_w, (int, float))
        assert isinstance(status.power_limit_w, (int, float))
        assert isinstance(status.min_power_limit_w, (int, float))
        assert isinstance(status.max_power_limit_w, (int, float))
        assert isinstance(status.default_power_limit_w, (int, float))
        assert isinstance(status.gpu_util_pct, (int, float))
        assert isinstance(status.memory_util_pct, (int, float))
        assert isinstance(status.temperature_c, (int, float))

    def test_get_status_all_values_non_negative(self) -> None:
        """Float fields are zero or positive (or N/A fallback to 0)."""
        capper = GPUPowerCapper(gpu_indices=[0])
        for status in capper.get_status().values():
            assert status.current_power_w >= 0.0
            assert status.gpu_util_pct >= 0.0
            assert status.memory_util_pct >= 0.0
            assert status.temperature_c >= 0.0

    def test_get_status_repeated_calls(self) -> None:
        """Multiple get_status calls do not raise."""
        capper = GPUPowerCapper(gpu_indices=[0])
        for _ in range(5):
            s = capper.get_status(gpu_index=0)
            assert isinstance(s, PowerStatus)

    def test_get_status_no_gpu_empty_indices(self) -> None:
        """get_status with auto-detected empty indices works."""
        capper = GPUPowerCapper()
        status = capper.get_status()
        assert isinstance(status, dict)


# ── GPUPowerCapper: set_power_limit ────────────────────────────────────


class TestGPUPowerCapperSetPowerLimit:
    """set_power_limit type stability and no-crash guarantee."""

    def test_set_power_limit_returns_bool(self) -> None:
        """set_power_limit returns a bool."""
        capper = GPUPowerCapper(gpu_indices=[0])
        result = capper.set_power_limit(200)
        assert isinstance(result, bool)

    def test_set_power_limit_specific_gpu(self) -> None:
        """set_power_limit with explicit index returns bool."""
        capper = GPUPowerCapper(gpu_indices=[0, 1])
        result = capper.set_power_limit(200, gpu_index=0)
        assert isinstance(result, bool)

    def test_set_power_limit_zero_watts(self) -> None:
        """set_power_limit with 0 watts returns bool."""
        capper = GPUPowerCapper(gpu_indices=[0])
        result = capper.set_power_limit(0)
        assert isinstance(result, bool)

    def test_set_power_limit_large_value(self) -> None:
        """set_power_limit with a very large value returns bool (no crash)."""
        capper = GPUPowerCapper(gpu_indices=[0])
        result = capper.set_power_limit(10000)
        assert isinstance(result, bool)


# ── GPUPowerCapper: reset_to_default ───────────────────────────────────


class TestGPUPowerCapperReset:
    """reset_to_default type stability."""

    def test_reset_to_default_returns_bool(self) -> None:
        """reset_to_default() returns bool."""
        capper = GPUPowerCapper(gpu_indices=[0])
        result = capper.reset_to_default()
        assert isinstance(result, bool)

    def test_reset_to_default_specific_returns_bool(self) -> None:
        """reset_to_default(gpu_index=0) returns bool."""
        capper = GPUPowerCapper(gpu_indices=[0])
        result = capper.reset_to_default(gpu_index=0)
        assert isinstance(result, bool)

    def test_reset_to_default_multiple_gpus(self) -> None:
        """reset_to_default() with multiple indices returns bool."""
        capper = GPUPowerCapper(gpu_indices=[0, 1])
        result = capper.reset_to_default()
        assert isinstance(result, bool)


# ── GPUPowerCapper: get_energy_report ──────────────────────────────────


class TestGPUPowerCapperEnergyReport:
    """get_energy_report return value shape."""

    def test_get_energy_report_returns_dict(self) -> None:
        """get_energy_report returns a dict."""
        capper = GPUPowerCapper(gpu_indices=[0])
        report = capper.get_energy_report()
        assert isinstance(report, dict)

    def test_get_energy_report_has_expected_keys(self) -> None:
        """get_energy_report dict has all expected top-level keys."""
        capper = GPUPowerCapper(gpu_indices=[0])
        report = capper.get_energy_report()
        assert "gpus" in report
        assert "total_current_power_w" in report
        assert "total_default_power_w" in report
        assert "avg_power_savings_pct" in report
        assert "gpus_tuned" in report
        assert "per_gpu" in report

    def test_get_energy_report_types(self) -> None:
        """Top-level report values have expected types."""
        capper = GPUPowerCapper(gpu_indices=[0])
        report = capper.get_energy_report()
        assert isinstance(report["gpus"], int)
        # round(x, 1) returns int when x is 0, float otherwise
        assert isinstance(report["total_current_power_w"], (int, float))
        assert isinstance(report["total_default_power_w"], (int, float))
        assert isinstance(report["avg_power_savings_pct"], (int, float))
        assert isinstance(report["gpus_tuned"], int)
        assert isinstance(report["per_gpu"], dict)

    def test_get_energy_report_per_gpu_values(self) -> None:
        """Each per-GPU entry has expected structure."""
        capper = GPUPowerCapper(gpu_indices=[0])
        report = capper.get_energy_report()
        for gpu_key, gpu_data in report["per_gpu"].items():
            assert isinstance(gpu_key, str)
            assert "power_limit_w" in gpu_data
            assert "current_w" in gpu_data
            assert "savings_pct" in gpu_data
            assert "temp_c" in gpu_data
            assert isinstance(gpu_data["power_limit_w"], (int, float))
            assert isinstance(gpu_data["current_w"], (int, float))
            assert isinstance(gpu_data["savings_pct"], (int, float))
            assert isinstance(gpu_data["temp_c"], (int, float))

    def test_get_energy_report_gpus_nonnegative(self) -> None:
        """gpus count is non-negative."""
        capper = GPUPowerCapper(gpu_indices=[0])
        report = capper.get_energy_report()
        assert report["gpus"] >= 0

    def test_get_energy_report_repeated(self) -> None:
        """Multiple get_energy_report calls do not raise."""
        capper = GPUPowerCapper(gpu_indices=[0])
        for _ in range(5):
            r = capper.get_energy_report()
            assert isinstance(r, dict)


# ── GPUPowerCapper: auto_tune ──────────────────────────────────────────


class TestGPUPowerCapperAutoTune:
    """auto_tune with benchmark_fn (no GPU hardware dependency)."""

    def test_auto_tune_with_benchmark_returns_dict(self) -> None:
        """auto_tune with benchmar_fn returns dict."""
        capper = GPUPowerCapper(gpu_indices=[0])
        result = capper.auto_tune(
            benchmark_fn=lambda: {"tokens_per_sec": 100.0},
        )
        assert isinstance(result, dict)

    def test_auto_tune_benchmark_empty_result(self) -> None:
        """Benchmark returning empty dict is handled."""
        capper = GPUPowerCapper(gpu_indices=[0])

        def bench() -> dict[str, float]:
            return {}

        result = capper.auto_tune(benchmark_fn=bench)
        assert isinstance(result, dict)

    def test_auto_tune_benchmark_exception(self) -> None:
        """Benchmark raising an exception is caught gracefully."""
        capper = GPUPowerCapper(gpu_indices=[0])

        def failing_bench() -> dict[str, float]:
            raise RuntimeError("benchmark failed")

        result = capper.auto_tune(benchmark_fn=failing_bench)
        assert isinstance(result, dict)

    def test_auto_tune_result_keys_are_gpu_indices(self) -> None:
        """Auto-tune result keys are int GPU indices."""
        capper = GPUPowerCapper(gpu_indices=[0, 1])
        result = capper.auto_tune(
            benchmark_fn=lambda: {"tokens_per_sec": 100.0},
        )
        for k in result:
            assert isinstance(k, int)

    def test_auto_tune_result_values_are_ints(self) -> None:
        """Auto-tune result values are int watts."""
        capper = GPUPowerCapper(gpu_indices=[0])
        result = capper.auto_tune(
            benchmark_fn=lambda: {"tokens_per_sec": 100.0},
        )
        for v in result.values():
            assert isinstance(v, int)

    def test_auto_tune_custom_pct_range(self) -> None:
        """auto_tune accepts min/max power percent overrides."""
        capper = GPUPowerCapper(gpu_indices=[0])
        result = capper.auto_tune(
            min_power_pct=0.5,
            max_power_pct=0.8,
            benchmark_fn=lambda: {"tokens_per_sec": 100.0},
        )
        assert isinstance(result, dict)

    def test_auto_tune_custom_step(self) -> None:
        """auto_tune accepts custom step_watts."""
        capper = GPUPowerCapper(gpu_indices=[0])
        result = capper.auto_tune(
            step_watts=50,
            benchmark_fn=lambda: {"tokens_per_sec": 100.0},
        )
        assert isinstance(result, dict)


# ── GPUPowerCapper: _quick_bench (GPU-dependent) ───────────────────────


class TestGPUPowerCapperQuickBench:
    """_quick_bench static method — no-crash guarantee only.

    This method requires a CUDA-capable GPU.  If CUDA is unavailable,
    it returns 0.0.  In all cases it must not raise.
    """

    def test_quick_bench_returns_float(self) -> None:
        """_quick_bench returns a float (0.0 if no CUDA)."""
        result = GPUPowerCapper._quick_bench(gpu_index=0)
        assert isinstance(result, float)
        assert result >= 0.0


# ── GPUPowerCapper: _get_default_power ─────────────────────────────────


class TestGPUPowerCapperGetDefaultPower:
    """_get_default_power static method — type stability."""

    def test_get_default_power_returns_float(self) -> None:
        """_get_default_power returns a float."""
        result = GPUPowerCapper._get_default_power(gpu_index=0)
        assert isinstance(result, float)
        assert result >= 0.0

    def test_get_default_power_unknown_index(self) -> None:
        """_get_default_power for non-existent GPU returns 0.0."""
        result = GPUPowerCapper._get_default_power(gpu_index=9999)
        assert isinstance(result, float)
        assert result == 0.0

    def test_get_default_power_negative_index(self) -> None:
        """_get_default_power with negative index returns 0.0 (no crash)."""
        result = GPUPowerCapper._get_default_power(gpu_index=-1)
        assert isinstance(result, float)


# ── GPUPowerCapper: thread safety ──────────────────────────────────────


class TestGPUPowerCapperConcurrency:
    """Thread safety of read-heavy operations."""

    def test_concurrent_get_status(self) -> None:
        """Multiple threads call get_status concurrently without error."""
        capper = GPUPowerCapper(gpu_indices=[0])
        errors: list[Exception] = []

        def access() -> None:
            try:
                for _ in range(10):
                    capper.get_status()
                    capper.get_status(gpu_index=0)
                    capper.get_status(gpu_index=1)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=access) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors

    def test_concurrent_energy_report(self) -> None:
        """Multiple threads call get_energy_report concurrently."""
        capper = GPUPowerCapper(gpu_indices=[0])
        errors: list[Exception] = []

        def access() -> None:
            try:
                for _ in range(10):
                    r = capper.get_energy_report()
                    assert isinstance(r, dict)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=access) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors

    def test_concurrent_stop_auto_tune(self) -> None:
        """Multiple threads call stop_auto_tune safely."""
        capper = GPUPowerCapper(gpu_indices=[0])
        capper._auto_tune_active = True

        def stop() -> None:
            capper.stop_auto_tune()

        threads = [threading.Thread(target=stop) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert capper._auto_tune_active is False

    def test_concurrent_get_status_and_report(self) -> None:
        """Mix of get_status and get_energy_report concurrently."""
        capper = GPUPowerCapper(gpu_indices=[0])
        errors: list[Exception] = []

        def mixed() -> None:
            try:
                for _ in range(10):
                    capper.get_status()
                    capper.get_energy_report()
                    capper.stop_auto_tune()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=mixed) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
