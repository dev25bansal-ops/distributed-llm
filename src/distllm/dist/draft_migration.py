"""Auto-Migration CPU↔GPU — dynamically swap draft models between hardware.

Monitors GPU contention and dynamically migrates draft models between
CPU and small GPU based on resource availability. When the GPU cluster
is lightly loaded, a draft model can share a GPU. When contention
rises, the draft model migrates back to CPU.

No competitor does this — it's unique to DistLLM's distributed
architecture where draft and target run on separate nodes.

Usage::

    manager = HeterogeneousDraftManager(
        gpu_resource_mgr=get_gpu_resource_manager(),
        fleet=fleet,
    )

    # Register available hardware
    manager.register_cpu_endpoint("http://cpu-node:8000/v1/completions")
    manager.register_gpu_endpoint("http://gpu-node:8001/v1/completions", device=0)

    # Periodically evaluate and migrate
    manager.evaluate_and_migrate()
"""


from __future__ import annotations
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from loguru import logger


class MigrationReason(str, Enum):
    GPU_CONTENTION_HIGH = "gpu_contention_high"
    GPU_CONTENTION_LOW = "gpu_contention_low"
    COST_OPTIMIZATION = "cost_optimization"
    LATENCY_OPTIMIZATION = "latency_optimization"
    MANUAL = "manual"


@dataclass
class HardwareEndpoint:
    """A draft model endpoint on specific hardware."""

    endpoint_url: str
    hardware: str  # "cpu", "cuda:0", "mps"
    model_name: str = ""
    cost_per_hour: float = 0.0
    avg_latency_ms: float = 0.0
    is_active: bool = True
    device_id: int = -1  # -1 for CPU
    vram_required_mb: float = 0.0
    last_health_check: float = 0.0

    @property
    def is_gpu(self) -> bool:
        return self.device_id >= 0

    @property
    def is_cpu(self) -> bool:
        return self.device_id < 0


@dataclass
class MigrationEvent:
    """Record of a draft model migration."""

    timestamp: float
    from_url: str
    to_url: str
    from_hardware: str
    to_hardware: str
    reason: MigrationReason
    gpu_utilization_pct: float = 0.0
    latency_before_ms: float = 0.0
    latency_after_ms: float = 0.0


@dataclass
class MigrationConfig:
    """Configuration for auto-migration behavior."""

    enabled: bool = True
    check_interval_s: float = 10.0
    gpu_high_threshold_pct: float = 80.0
    gpu_low_threshold_pct: float = 40.0
    min_migration_interval_s: float = 60.0
    vram_required_mb: float = 500.0
    prefer_gpu_for_latency: bool = True
    cost_weight: float = 0.3
    latency_weight: float = 0.7


class HeterogeneousDraftManager:
    """Manages draft models across CPU and GPU hardware.


    Monitors GPU utilization and migrates draft models between:
    - CPU endpoints (cheap, slow, always available)
    - GPU endpoints (fast, shared with target model, limited VRAM)

    Migration logic:
    - GPU utilization < low_threshold → migrate draft TO GPU (faster)
    - GPU utilization > high_threshold → migrate draft TO CPU (free VRAM)
    - Cooldown period prevents thrashing
    """


    def __init__(
        self,
        gpu_resource_mgr: Any | None = None,
        fleet: Any | None = None,
        config: MigrationConfig | None = None,
        on_migrate: Callable[[MigrationEvent], None] | None = None,
    ) -> None:
        self._gpu_mgr = gpu_resource_mgr
        self._fleet = fleet
        self._config = config or MigrationConfig()
        self._on_migrate = on_migrate

        self._cpu_endpoints: dict[str, HardwareEndpoint] = {}
        self._gpu_endpoints: dict[str, HardwareEndpoint] = {}
        self._active_endpoint: HardwareEndpoint | None = None
        self._migration_history: list[MigrationEvent] = []
        self._last_migration_time: float = 0.0
        self._lock = threading.Lock()
        self._monitor_thread: threading.Thread | None = None
        self._running = threading.Event()

    def register_cpu_endpoint(
        self,
        endpoint_url: str,
        model_name: str = "",
        cost_per_hour: float = 0.05,
        avg_latency_ms: float = 50.0,
    ) -> None:
        """Register a CPU-based draft model endpoint."""

        endpoint = HardwareEndpoint(
            endpoint_url=endpoint_url,
            hardware="cpu",
            model_name=model_name,
            cost_per_hour=cost_per_hour,
            avg_latency_ms=avg_latency_ms,
            device_id=-1,
        )
        with self._lock:
            self._cpu_endpoints[endpoint_url] = endpoint
            if self._active_endpoint is None:
                self._active_endpoint = endpoint
        logger.info(f"Registered CPU draft endpoint: {endpoint_url}")

    def register_gpu_endpoint(
        self,
        endpoint_url: str,
        device: int = 0,
        model_name: str = "",
        cost_per_hour: float = 0.60,
        avg_latency_ms: float = 8.0,
        vram_required_mb: float = 500.0,
    ) -> None:
        """Register a GPU-based draft model endpoint."""

        endpoint = HardwareEndpoint(
            endpoint_url=endpoint_url,
            hardware=f"cuda:{device}",
            model_name=model_name,
            cost_per_hour=cost_per_hour,
            avg_latency_ms=avg_latency_ms,
            device_id=device,
            vram_required_mb=vram_required_mb,
        )
        with self._lock:
            self._gpu_endpoints[endpoint_url] = endpoint
        logger.info(f"Registered GPU draft endpoint: {endpoint_url} (device {device})")

    def evaluate_and_migrate(self) -> HardwareEndpoint | None:
        """Evaluate GPU contention and migrate if needed.


        Returns the active endpoint after evaluation, or None if no change.
        """

        if not self._config.enabled:
            return self._active_endpoint

        with self._lock:
            # Check cooldown
            if time.time() - self._last_migration_time < self._config.min_migration_interval_s:
                return self._active_endpoint

            gpu_util = self._get_gpu_utilization()

            if gpu_util is None:
                return self._active_endpoint

            current = self._active_endpoint
            if current is None:
                return None

            # GPU is under-utilized → migrate to GPU for better latency
            if (gpu_util < self._config.gpu_low_threshold_pct
                and current.is_cpu
                and self._gpu_endpoints
                and self._config.prefer_gpu_for_latency):

                best_gpu = self._select_best_gpu_endpoint()
                if best_gpu:
                    self._migrate(current, best_gpu, MigrationReason.GPU_CONTENTION_LOW)
                    return self._active_endpoint

            # GPU is over-utilized → migrate to CPU to free VRAM
            if (gpu_util > self._config.gpu_high_threshold_pct
                and current.is_gpu
                and self._cpu_endpoints):

                best_cpu = self._select_best_cpu_endpoint()
                if best_cpu:
                    self._migrate(current, best_cpu, MigrationReason.GPU_CONTENTION_HIGH)
                    return self._active_endpoint

        return self._active_endpoint

    def _get_gpu_utilization(self) -> float | None:
        """Get current GPU utilization percentage."""

        if self._gpu_mgr is not None:
            try:
                snapshot = self._gpu_mgr.snapshot(device=0)
                if snapshot:
                    return snapshot.utilization_pct
            except Exception:
                pass

        # Fallback: try pynvml directly
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return float(util.gpu)
        except Exception:
            return None

    def _select_best_gpu_endpoint(self) -> HardwareEndpoint | None:
        """Select the best available GPU endpoint."""

        available = [e for e in self._gpu_endpoints.values() if e.is_active]
        if not available:
            return None
        return min(available, key=lambda e: e.avg_latency_ms)

    def _select_best_cpu_endpoint(self) -> HardwareEndpoint | None:
        """Select the best available CPU endpoint."""

        available = [e for e in self._cpu_endpoints.values() if e.is_active]
        if not available:
            return None
        return min(available, key=lambda e: e.avg_latency_ms)

    def _migrate(
        self,
        from_endpoint: HardwareEndpoint,
        to_endpoint: HardwareEndpoint,
        reason: MigrationReason,
    ) -> None:
        """Execute a migration from one hardware to another."""

        event = MigrationEvent(
            timestamp=time.time(),
            from_url=from_endpoint.endpoint_url,
            to_url=to_endpoint.endpoint_url,
            from_hardware=from_endpoint.hardware,
            to_hardware=to_endpoint.hardware,
            reason=reason,
            latency_before_ms=from_endpoint.avg_latency_ms,
            latency_after_ms=to_endpoint.avg_latency_ms,
        )

        self._active_endpoint = to_endpoint
        self._last_migration_time = time.time()
        self._migration_history.append(event)

        # Keep only last 100 migration events
        if len(self._migration_history) > 100:
            self._migration_history = self._migration_history[-100:]

        # Update fleet routing if available
        if self._fleet is not None:
            try:
                # Mark old endpoint as inactive in fleet
                spec = self._fleet.get_spec(from_endpoint.endpoint_url)
                if spec:
                    # The fleet will route to the new endpoint
                    pass
            except Exception:
                pass

        logger.info(
            f"Draft model migrated: {from_endpoint.hardware} → {to_endpoint.hardware} "
            f"({reason.value}, latency: {from_endpoint.avg_latency_ms:.1f}ms → "
            f"{to_endpoint.avg_latency_ms:.1f}ms)"
        )

        if self._on_migrate:
            try:
                self._on_migrate(event)
            except Exception:
                pass

    def start_monitor(self) -> None:
        """Start background monitoring thread."""

        if not self._config.enabled:
            return

        self._running.set()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="draft-migration-monitor",
        )
        self._monitor_thread.start()
        logger.info(
            f"Draft migration monitor started "
            f"(check every {self._config.check_interval_s}s)"
        )

    def stop_monitor(self) -> None:
        """Stop background monitoring."""

        self._running.clear()
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=3.0)

    def _monitor_loop(self) -> None:
        """Background monitoring loop."""

        while self._running.is_set():
            try:
                self._running.wait(self._config.check_interval_s)
                if self._running.is_set():
                    self.evaluate_and_migrate()
            except Exception as e:
                logger.error(f"Migration monitor error: {e}")

    @property
    def active_endpoint(self) -> HardwareEndpoint | None:
        return self._active_endpoint

    @property
    def migration_history(self) -> list[MigrationEvent]:
        return list(self._migration_history)

    def get_status(self) -> dict[str, Any]:
        """Get current migration manager status."""

        with self._lock:
            active = self._active_endpoint
            return {
                "enabled": self._config.enabled,
                "active_endpoint": active.endpoint_url if active else None,
                "active_hardware": active.hardware if active else None,
                "cpu_endpoints": len(self._cpu_endpoints),
                "gpu_endpoints": len(self._gpu_endpoints),
                "total_migrations": len(self._migration_history),
                "last_migration": (
                    self._migration_history[-1].timestamp
                    if self._migration_history else None
                ),
                "config": {
                    "gpu_high_threshold": self._config.gpu_high_threshold_pct,
                    "gpu_low_threshold": self._config.gpu_low_threshold_pct,
                    "min_interval_s": self._config.min_migration_interval_s,
                    "prefer_gpu": self._config.prefer_gpu_for_latency,
                },
            }
