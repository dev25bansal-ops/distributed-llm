"""Tests for draft model auto-migration across CPU/GPU hardware.

Covers:
- MigrationReason enum values and membership
- HardwareEndpoint dataclass (CPU, GPU, defaults, properties)
- MigrationEvent dataclass (minimal and full creation)
- MigrationConfig dataclass (defaults, custom values, zero values)
- HeterogeneousDraftManager
  - Initialization with and without config
  - CPU/GPU endpoint registration and first-endpoint-activation
  - evaluate_and_migrate: disabled, no active, no gpu_util, with real GPU mgr,
    CPU-to-GPU migration, GPU-to-CPU migration, cooldown, no endpoints
  - _select_best_gpu_endpoint / _select_best_cpu_endpoint (latency, inactive skip)
  - _migrate directly, history capping, on_migrate callback + exception safety
  - get_status output structure
  - start_monitor / stop_monitor thread lifecycle
"""

from __future__ import annotations

import time

from distllm.dist.draft_migration import (
    HardwareEndpoint,
    HeterogeneousDraftManager,
    MigrationConfig,
    MigrationEvent,
    MigrationReason,
)


class TestMigrationReason:
    """Enum values and string semantics."""

    def test_values(self) -> None:
        assert MigrationReason.GPU_CONTENTION_HIGH.value == "gpu_contention_high"
        assert MigrationReason.GPU_CONTENTION_LOW.value == "gpu_contention_low"
        assert MigrationReason.COST_OPTIMIZATION.value == "cost_optimization"
        assert MigrationReason.LATENCY_OPTIMIZATION.value == "latency_optimization"
        assert MigrationReason.MANUAL.value == "manual"

    def test_all_members_distinct(self) -> None:
        values = [m.value for m in MigrationReason]
        assert len(values) == len(set(values))


class TestHardwareEndpoint:
    """Dataclass fields and computed properties."""

    def test_cpu_defaults(self) -> None:
        ep = HardwareEndpoint(
            endpoint_url="http://cpu:8000",
            hardware="cpu",
            device_id=-1,
        )
        assert ep.is_cpu is True
        assert ep.is_gpu is False
        assert ep.model_name == ""
        assert ep.cost_per_hour == 0.0
        assert ep.avg_latency_ms == 0.0
        assert ep.is_active is True
        assert ep.vram_required_mb == 0.0
        assert ep.last_health_check == 0.0

    def test_gpu_defaults(self) -> None:
        ep = HardwareEndpoint(
            endpoint_url="http://gpu:8001",
            hardware="cuda:0",
            device_id=0,
        )
        assert ep.is_gpu is True
        assert ep.is_cpu is False
        assert ep.device_id == 0

    def test_cpu_with_zero_device_id_is_cpu(self) -> None:
        """device_id 0 is a GPU; -1 is the sentinel for CPU."""
        ep_cpu = HardwareEndpoint("url", "cpu", device_id=-1)
        ep_gpu = HardwareEndpoint("url", "cuda:0", device_id=0)
        assert ep_cpu.is_cpu is True
        assert ep_gpu.is_cpu is False

    def test_custom_fields(self) -> None:
        ep = HardwareEndpoint(
            endpoint_url="http://custom:9999",
            hardware="mps",
            model_name="draft-v2",
            cost_per_hour=1.25,
            avg_latency_ms=12.3,
            is_active=False,
            device_id=3,
            vram_required_mb=2048.0,
            last_health_check=1000.0,
        )
        assert ep.model_name == "draft-v2"
        assert ep.cost_per_hour == 1.25
        assert ep.avg_latency_ms == 12.3
        assert ep.is_active is False
        assert ep.vram_required_mb == 2048.0
        assert ep.last_health_check == 1000.0

    def test_mps_not_gpu_by_device_id(self) -> None:
        """mps hardware without device_id >= 0 is not considered GPU."""
        ep = HardwareEndpoint("url", "mps", device_id=-1)
        assert ep.is_gpu is False


class TestMigrationEvent:
    """Dataclass for migration records."""

    def test_minimal_creation(self) -> None:
        event = MigrationEvent(
            timestamp=100.0,
            from_url="http://cpu:8000",
            to_url="http://gpu:8001",
            from_hardware="cpu",
            to_hardware="cuda:0",
            reason=MigrationReason.GPU_CONTENTION_HIGH,
        )
        assert event.gpu_utilization_pct == 0.0
        assert event.latency_before_ms == 0.0
        assert event.latency_after_ms == 0.0

    def test_full_creation(self) -> None:
        event = MigrationEvent(
            timestamp=200.0,
            from_url="http://old:8000",
            to_url="http://new:8001",
            from_hardware="cpu",
            to_hardware="cuda:1",
            reason=MigrationReason.MANUAL,
            gpu_utilization_pct=75.5,
            latency_before_ms=50.0,
            latency_after_ms=8.0,
        )
        assert event.gpu_utilization_pct == 75.5
        assert event.latency_before_ms == 50.0
        assert event.latency_after_ms == 8.0
        assert event.reason == MigrationReason.MANUAL


class TestMigrationConfig:
    """Configuration dataclass for auto-migration."""

    def test_defaults(self) -> None:
        cfg = MigrationConfig()
        assert cfg.enabled is True
        assert cfg.check_interval_s == 10.0
        assert cfg.gpu_high_threshold_pct == 80.0
        assert cfg.gpu_low_threshold_pct == 40.0
        assert cfg.min_migration_interval_s == 60.0
        assert cfg.vram_required_mb == 500.0
        assert cfg.prefer_gpu_for_latency is True
        assert cfg.cost_weight == 0.3
        assert cfg.latency_weight == 0.7

    def test_custom_values(self) -> None:
        cfg = MigrationConfig(
            enabled=False,
            check_interval_s=5.0,
            gpu_high_threshold_pct=90.0,
            gpu_low_threshold_pct=30.0,
            min_migration_interval_s=120.0,
            vram_required_mb=1024.0,
            prefer_gpu_for_latency=False,
            cost_weight=0.5,
            latency_weight=0.5,
        )
        assert cfg.enabled is False
        assert cfg.check_interval_s == 5.0
        assert cfg.gpu_high_threshold_pct == 90.0
        assert cfg.gpu_low_threshold_pct == 30.0
        assert cfg.min_migration_interval_s == 120.0
        assert cfg.vram_required_mb == 1024.0
        assert cfg.prefer_gpu_for_latency is False
        assert cfg.cost_weight == 0.5
        assert cfg.latency_weight == 0.5

    def test_zero_values_allowed(self) -> None:
        cfg = MigrationConfig(
            enabled=False,
            check_interval_s=0.0,
            min_migration_interval_s=0.0,
            cost_weight=0.0,
            latency_weight=0.0,
        )
        assert cfg.check_interval_s == 0.0
        assert cfg.min_migration_interval_s == 0.0
        assert cfg.cost_weight == 0.0
        assert cfg.latency_weight == 0.0


class TestHeterogeneousDraftManager:
    """Core migration manager — no GPU, no mocks, only real objects."""

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def test_init_defaults(self) -> None:
        mgr = HeterogeneousDraftManager()
        assert mgr.active_endpoint is None
        assert mgr.migration_history == []

    def test_init_with_config(self) -> None:
        cfg = MigrationConfig(enabled=False)
        mgr = HeterogeneousDraftManager(config=cfg)
        assert mgr._config.enabled is False
        assert mgr.active_endpoint is None

    def test_init_with_on_migrate(self) -> None:
        collected: list[MigrationEvent] = []

        def cb(ev: MigrationEvent) -> None:
            collected.append(ev)

        mgr = HeterogeneousDraftManager(on_migrate=cb)
        assert mgr._on_migrate is cb

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def test_register_cpu_endpoint_first_becomes_active(self) -> None:
        mgr = HeterogeneousDraftManager()
        mgr.register_cpu_endpoint("http://cpu:8000")
        assert mgr.active_endpoint is not None
        assert mgr.active_endpoint.endpoint_url == "http://cpu:8000"
        assert mgr.active_endpoint.hardware == "cpu"
        assert mgr.active_endpoint.is_cpu is True

    def test_register_multiple_cpu_endpoints(self) -> None:
        mgr = HeterogeneousDraftManager()
        mgr.register_cpu_endpoint("http://cpu1:8000", avg_latency_ms=50.0)
        mgr.register_cpu_endpoint("http://cpu2:8000", avg_latency_ms=30.0)
        # First registered remains active
        assert mgr.active_endpoint is not None
        assert mgr.active_endpoint.endpoint_url == "http://cpu1:8000"

    def test_register_gpu_endpoint_does_not_set_active(self) -> None:
        mgr = HeterogeneousDraftManager()
        mgr.register_gpu_endpoint("http://gpu:8001", device=0)
        # register_gpu_endpoint never changes active_endpoint
        assert mgr.active_endpoint is None

    def test_register_gpu_endpoint_stores_internal_dict(self) -> None:
        mgr = HeterogeneousDraftManager()
        mgr.register_gpu_endpoint(
            "http://gpu:8001", device=0, vram_required_mb=2048.0,
        )
        ep = mgr._gpu_endpoints["http://gpu:8001"]
        assert ep.hardware == "cuda:0"
        assert ep.vram_required_mb == 2048.0

    def test_register_cpu_endpoint_custom_fields(self) -> None:
        mgr = HeterogeneousDraftManager()
        mgr.register_cpu_endpoint(
            "http://cpu:8000",
            model_name="tiny-draft",
            cost_per_hour=0.1,
            avg_latency_ms=40.0,
        )
        ep = mgr._cpu_endpoints["http://cpu:8000"]
        assert ep.model_name == "tiny-draft"
        assert ep.cost_per_hour == 0.1
        assert ep.avg_latency_ms == 40.0

    # ------------------------------------------------------------------
    # evaluate_and_migrate — early returns
    # ------------------------------------------------------------------

    def test_evaluate_migrate_disabled_returns_active(self) -> None:
        cfg = MigrationConfig(enabled=False)
        mgr = HeterogeneousDraftManager(config=cfg)
        mgr.register_cpu_endpoint("http://cpu:8000")
        result = mgr.evaluate_and_migrate()
        assert result is mgr.active_endpoint

    def test_evaluate_migrate_no_active_returns_none(self) -> None:
        mgr = HeterogeneousDraftManager()
        result = mgr.evaluate_and_migrate()
        assert result is None

    def test_evaluate_migrate_no_gpu_util_stays_put(self) -> None:
        """Without GPU manager or pynvml, _get_gpu_utilization returns None."""
        mgr = HeterogeneousDraftManager()
        mgr.register_cpu_endpoint("http://cpu:8000")
        result = mgr.evaluate_and_migrate()
        # No migration because gpu_util is None
        assert result is mgr.active_endpoint
        assert result is not None
        assert result.is_cpu

    def test_evaluate_migrate_active_none_with_gpu_util(self) -> None:
        mgr = HeterogeneousDraftManager()
        mgr._get_gpu_utilization = lambda: 50.0  # type: ignore[assignment]
        result = mgr.evaluate_and_migrate()
        assert result is None

    # ------------------------------------------------------------------
    # evaluate_and_migrate — with real (test-double) GPU resource manager
    # ------------------------------------------------------------------

    def test_evaluate_migrate_with_real_gpu_mgr_snapshot(self) -> None:
        """Use a real (non-mock) GPU manager object with a snapshot method."""

        class FakeSnapshot:
            utilization_pct: float = 25.0

        class FakeGpuMgr:
            def snapshot(self, device: int = 0) -> FakeSnapshot:
                return FakeSnapshot()

        mgr = HeterogeneousDraftManager(gpu_resource_mgr=FakeGpuMgr())
        mgr.register_cpu_endpoint("http://cpu:8000", avg_latency_ms=50.0)
        mgr.register_gpu_endpoint("http://gpu:8001", device=0, avg_latency_ms=8.0)

        result = mgr.evaluate_and_migrate()
        assert result is not None
        assert result.is_gpu
        assert result.endpoint_url == "http://gpu:8001"

    def test_evaluate_migrate_failing_gpu_mgr_falls_through(self) -> None:
        """If gpu_mgr.snapshot raises, _get_gpu_utilization falls through to
        pynvml, which also fails, so gpu_util is None and no migration occurs."""

        class FailingGpuMgr:
            def snapshot(self, device: int = 0) -> None:
                raise RuntimeError("GPU not available")

        mgr = HeterogeneousDraftManager(gpu_resource_mgr=FailingGpuMgr())
        mgr.register_cpu_endpoint("http://cpu:8000")
        result = mgr.evaluate_and_migrate()
        assert result is not None
        assert result.is_cpu

    # ------------------------------------------------------------------
    # evaluate_and_migrate — migration paths (monkey-patched util)
    # ------------------------------------------------------------------

    def test_migrate_cpu_to_gpu_on_low_contention(self) -> None:
        mgr = HeterogeneousDraftManager()
        mgr.register_cpu_endpoint("http://cpu:8000", avg_latency_ms=50.0)
        mgr.register_gpu_endpoint("http://gpu:8001", device=0, avg_latency_ms=8.0)
        mgr._get_gpu_utilization = lambda: 20.0  # type: ignore[assignment]

        result = mgr.evaluate_and_migrate()
        assert result is not None
        assert result.is_gpu
        assert result.endpoint_url == "http://gpu:8001"

    def test_migrate_gpu_to_cpu_on_high_contention(self) -> None:
        mgr = HeterogeneousDraftManager()
        mgr.register_gpu_endpoint("http://gpu:8001", device=0, avg_latency_ms=8.0)
        mgr.register_cpu_endpoint("http://cpu:8000", avg_latency_ms=50.0)
        # Manually set active to the GPU endpoint
        mgr._active_endpoint = mgr._gpu_endpoints["http://gpu:8001"]
        mgr._get_gpu_utilization = lambda: 90.0  # type: ignore[assignment]

        result = mgr.evaluate_and_migrate()
        assert result is not None
        assert result.is_cpu
        assert result.endpoint_url == "http://cpu:8000"

    def test_migrate_creates_history_event(self) -> None:
        mgr = HeterogeneousDraftManager()
        mgr.register_cpu_endpoint("http://cpu:8000", avg_latency_ms=50.0)
        mgr.register_gpu_endpoint("http://gpu:8001", device=0, avg_latency_ms=8.0)
        mgr._get_gpu_utilization = lambda: 20.0  # type: ignore[assignment]

        mgr.evaluate_and_migrate()
        assert len(mgr.migration_history) == 1
        ev = mgr.migration_history[0]
        assert ev.from_url == "http://cpu:8000"
        assert ev.to_url == "http://gpu:8001"
        assert ev.reason == MigrationReason.GPU_CONTENTION_LOW
        assert ev.latency_before_ms == 50.0
        assert ev.latency_after_ms == 8.0

    def test_cooldown_prevents_thrashing(self) -> None:
        mgr = HeterogeneousDraftManager()
        mgr.register_cpu_endpoint("http://cpu:8000", avg_latency_ms=50.0)
        mgr.register_gpu_endpoint("http://gpu:8001", device=0, avg_latency_ms=8.0)
        # Set last migration to right now — cooldown will block
        mgr._last_migration_time = time.time()
        mgr._get_gpu_utilization = lambda: 20.0  # type: ignore[assignment]

        result = mgr.evaluate_and_migrate()
        assert result is not None
        assert result.is_cpu  # stayed put due to cooldown

    def test_no_gpu_endpoints_stays_on_cpu(self) -> None:
        mgr = HeterogeneousDraftManager()
        mgr.register_cpu_endpoint("http://cpu:8000")
        mgr._get_gpu_utilization = lambda: 20.0  # type: ignore[assignment]
        result = mgr.evaluate_and_migrate()
        assert result is not None
        assert result.is_cpu

    def test_no_cpu_endpoints_stays_on_gpu(self) -> None:
        mgr = HeterogeneousDraftManager()
        mgr.register_gpu_endpoint("http://gpu:8001", device=0)
        mgr._active_endpoint = mgr._gpu_endpoints["http://gpu:8001"]
        mgr._get_gpu_utilization = lambda: 90.0  # type: ignore[assignment]
        result = mgr.evaluate_and_migrate()
        assert result is not None
        assert result.is_gpu

    def test_prefer_gpu_false_blocks_migration_to_gpu(self) -> None:
        cfg = MigrationConfig(prefer_gpu_for_latency=False)
        mgr = HeterogeneousDraftManager(config=cfg)
        mgr.register_cpu_endpoint("http://cpu:8000", avg_latency_ms=50.0)
        mgr.register_gpu_endpoint("http://gpu:8001", device=0, avg_latency_ms=8.0)
        mgr._get_gpu_utilization = lambda: 20.0  # type: ignore[assignment]

        result = mgr.evaluate_and_migrate()
        assert result is not None
        assert result.is_cpu  # prefer_gpu_for_latency=False prevents switch

    # ------------------------------------------------------------------
    # _select_best_gpu_endpoint / _select_best_cpu_endpoint
    # ------------------------------------------------------------------

    def test_select_best_gpu_by_latency(self) -> None:
        mgr = HeterogeneousDraftManager()
        mgr.register_gpu_endpoint("http://gpu1:8001", device=0, avg_latency_ms=15.0)
        mgr.register_gpu_endpoint("http://gpu2:8001", device=1, avg_latency_ms=5.0)
        mgr.register_gpu_endpoint("http://gpu3:8001", device=2, avg_latency_ms=10.0)

        best = mgr._select_best_gpu_endpoint()
        assert best is not None
        assert best.endpoint_url == "http://gpu2:8001"
        assert best.avg_latency_ms == 5.0

    def test_select_best_cpu_by_latency(self) -> None:
        mgr = HeterogeneousDraftManager()
        mgr.register_cpu_endpoint("http://cpu1:8000", avg_latency_ms=100.0)
        mgr.register_cpu_endpoint("http://cpu2:8000", avg_latency_ms=30.0)
        mgr.register_cpu_endpoint("http://cpu3:8000", avg_latency_ms=60.0)

        best = mgr._select_best_cpu_endpoint()
        assert best is not None
        assert best.endpoint_url == "http://cpu2:8000"
        assert best.avg_latency_ms == 30.0

    def test_select_best_gpu_none_when_empty(self) -> None:
        mgr = HeterogeneousDraftManager()
        assert mgr._select_best_gpu_endpoint() is None

    def test_select_best_cpu_none_when_empty(self) -> None:
        mgr = HeterogeneousDraftManager()
        assert mgr._select_best_cpu_endpoint() is None

    def test_select_best_gpu_skips_inactive(self) -> None:
        mgr = HeterogeneousDraftManager()
        mgr.register_gpu_endpoint("http://gpu1:8001", device=0, avg_latency_ms=5.0)
        mgr.register_gpu_endpoint("http://gpu2:8001", device=1, avg_latency_ms=10.0)
        mgr._gpu_endpoints["http://gpu1:8001"].is_active = False

        best = mgr._select_best_gpu_endpoint()
        assert best is not None
        assert best.endpoint_url == "http://gpu2:8001"

    def test_select_best_cpu_skips_inactive(self) -> None:
        mgr = HeterogeneousDraftManager()
        mgr.register_cpu_endpoint("http://cpu1:8000", avg_latency_ms=30.0)
        mgr.register_cpu_endpoint("http://cpu2:8000", avg_latency_ms=60.0)
        mgr._cpu_endpoints["http://cpu1:8000"].is_active = False

        best = mgr._select_best_cpu_endpoint()
        assert best is not None
        assert best.endpoint_url == "http://cpu2:8000"

    # ------------------------------------------------------------------
    # _migrate — direct execution
    # ------------------------------------------------------------------

    def test_migrate_directly_updates_state(self) -> None:
        mgr = HeterogeneousDraftManager()
        cpu = HardwareEndpoint(
            endpoint_url="http://cpu:8000",
            hardware="cpu",
            avg_latency_ms=50.0,
            device_id=-1,
        )
        gpu = HardwareEndpoint(
            endpoint_url="http://gpu:8001",
            hardware="cuda:0",
            avg_latency_ms=8.0,
            device_id=0,
        )
        mgr._active_endpoint = cpu
        mgr._migrate(cpu, gpu, MigrationReason.MANUAL)

        assert mgr.active_endpoint is gpu
        assert len(mgr.migration_history) == 1
        ev = mgr.migration_history[0]
        assert ev.reason == MigrationReason.MANUAL
        assert ev.from_url == "http://cpu:8000"
        assert ev.to_url == "http://gpu:8001"

    def test_migrate_history_capped_at_100(self) -> None:
        mgr = HeterogeneousDraftManager()
        cpu = HardwareEndpoint("http://cpu:8000", "cpu", device_id=-1)
        for i in range(105):
            gpu = HardwareEndpoint(
                f"http://gpu{i}:8001", "cuda:0",
                device_id=0, is_active=True,
            )
            mgr._migrate(cpu, gpu, MigrationReason.MANUAL)

        assert len(mgr.migration_history) == 100

    def test_on_migrate_callback_invoked(self) -> None:
        collected: list[MigrationEvent] = []

        def cb(ev: MigrationEvent) -> None:
            collected.append(ev)

        mgr = HeterogeneousDraftManager(on_migrate=cb)
        cpu = HardwareEndpoint("http://cpu:8000", "cpu", device_id=-1)
        gpu = HardwareEndpoint("http://gpu:8001", "cuda:0", device_id=0)
        mgr._migrate(cpu, gpu, MigrationReason.LATENCY_OPTIMIZATION)

        assert len(collected) == 1
        assert collected[0].reason == MigrationReason.LATENCY_OPTIMIZATION

    def test_on_migrate_callback_exception_swallowed(self) -> None:
        """A callback that raises should not abort the migration."""

        def cb(ev: MigrationEvent) -> None:
            raise ValueError("callback failure")

        mgr = HeterogeneousDraftManager(on_migrate=cb)
        cpu = HardwareEndpoint("http://cpu:8000", "cpu", device_id=-1)
        gpu = HardwareEndpoint("http://gpu:8001", "cuda:0", device_id=0)
        # Must not propagate
        mgr._migrate(cpu, gpu, MigrationReason.COST_OPTIMIZATION)
        assert len(mgr.migration_history) == 1

    # ------------------------------------------------------------------
    # migration_history returns a copy
    # ------------------------------------------------------------------

    def test_migration_history_is_copy(self) -> None:
        mgr = HeterogeneousDraftManager()
        mgr.register_cpu_endpoint("http://cpu:8000", avg_latency_ms=50.0)
        mgr.register_gpu_endpoint("http://gpu:8001", device=0, avg_latency_ms=8.0)
        mgr._get_gpu_utilization = lambda: 20.0  # type: ignore[assignment]
        mgr.evaluate_and_migrate()

        history = mgr.migration_history
        history.clear()
        # Internal list should be unaffected
        assert len(mgr.migration_history) == 1

    # ------------------------------------------------------------------
    # get_status
    # ------------------------------------------------------------------

    def test_get_status_empty(self) -> None:
        mgr = HeterogeneousDraftManager()
        s = mgr.get_status()
        assert s["enabled"] is True
        assert s["active_endpoint"] is None
        assert s["active_hardware"] is None
        assert s["cpu_endpoints"] == 0
        assert s["gpu_endpoints"] == 0
        assert s["total_migrations"] == 0
        assert s["last_migration"] is None
        assert s["config"]["gpu_high_threshold"] == 80.0

    def test_get_status_after_registration(self) -> None:
        mgr = HeterogeneousDraftManager()
        mgr.register_cpu_endpoint("http://cpu:8000")
        s = mgr.get_status()
        assert s["active_endpoint"] == "http://cpu:8000"
        assert s["active_hardware"] == "cpu"
        assert s["cpu_endpoints"] == 1
        assert s["gpu_endpoints"] == 0

    def test_get_status_after_migration(self) -> None:
        mgr = HeterogeneousDraftManager()
        mgr.register_cpu_endpoint("http://cpu:8000", avg_latency_ms=50.0)
        mgr.register_gpu_endpoint("http://gpu:8001", device=0, avg_latency_ms=8.0)
        mgr._get_gpu_utilization = lambda: 20.0  # type: ignore[assignment]
        mgr.evaluate_and_migrate()

        s = mgr.get_status()
        assert s["total_migrations"] == 1
        assert s["last_migration"] is not None
        assert s["active_hardware"] == "cuda:0"

    # ------------------------------------------------------------------
    # start_monitor / stop_monitor
    # ------------------------------------------------------------------

    def test_start_monitor_creates_thread(self) -> None:
        cfg = MigrationConfig(check_interval_s=0.1)
        mgr = HeterogeneousDraftManager(config=cfg)
        mgr.register_cpu_endpoint("http://cpu:8000")

        mgr.start_monitor()
        assert mgr._monitor_thread is not None
        assert mgr._monitor_thread.is_alive()

        mgr.stop_monitor()
        assert not mgr._monitor_thread.is_alive()

    def test_start_monitor_disabled_does_nothing(self) -> None:
        cfg = MigrationConfig(enabled=False)
        mgr = HeterogeneousDraftManager(config=cfg)
        mgr.start_monitor()
        assert mgr._monitor_thread is None

    def test_stop_monitor_without_start_does_not_raise(self) -> None:
        mgr = HeterogeneousDraftManager()
        mgr.stop_monitor()  # must be a no-op

    def test_stop_monitor_twice_does_not_raise(self) -> None:
        cfg = MigrationConfig(check_interval_s=0.1)
        mgr = HeterogeneousDraftManager(config=cfg)
        mgr.register_cpu_endpoint("http://cpu:8000")
        mgr.start_monitor()
        mgr.stop_monitor()
        mgr.stop_monitor()  # second call must be safe
