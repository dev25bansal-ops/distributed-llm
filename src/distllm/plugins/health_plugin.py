"""Health Check / Watchdog Plugin for distributed LLM.

Provides Kubernetes-compatible liveness and readiness probes, a background
watchdog that monitors error rate via ``MetricsPlugin``, and automatic
circuit-breaking when the error rate exceeds a configurable threshold.

Configuration is entirely via environment variables — no code changes
required to integrate with existing deployments.

Environment variables
---------------------
DISTLLM_PLUGIN_HEALTH_ENABLED : str
    Set to ``"1"`` to activate (default: ``"0"``).
DISTLLM_HEALTH_ERROR_RATE_THRESHOLD : float
    Requests-per-second error rate that triggers circuit break
    (default: ``0.5``).
DISTLLM_HEALTH_ERROR_WINDOW_SECONDS : int
    Sliding window in seconds for error rate calculation (default: ``60``).
DISTLLM_HEALTH_WATCHDOG_INTERVAL : int
    Seconds between watchdog checks (default: ``10``).
DISTLLM_HEALTH_GPU_MEMORY_MAX_PERCENT : float
    GPU memory usage percent above which readiness fails (default: ``95.0``).
DISTLLM_HEALTH_SYSTEM_MEMORY_MAX_PERCENT : float
    System memory usage percent above which readiness fails (default: ``90.0``).
DISTLLM_HEALTH_CIRCUIT_OPEN_SECONDS : int
    Seconds the circuit stays open before half-open retry (default: ``30``).
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any

from loguru import logger

from distllm.core.plugin_system import PluginBase


class HealthPlugin(PluginBase):
    """Kubernetes-ready health endpoints with watchdog-based circuit breaking.

    Registers two lightweight health surfaces that integrate with the
    existing ``MetricsPlugin`` for error rate data and ``SystemMonitor``
    for resource checks.

    **Liveness** (``/healthz``) — returns 200 if the process is responsive.
    Kubernetes restarts the pod only when this fails.

    **Readiness** (``/readyz``) — returns 200 only when:
      * A model is loaded (coordinator exists).
      * The error rate is below the configured threshold.
      * GPU memory is below the configured ceiling.
      * System memory is below the configured ceiling.
      * The circuit breaker is not open.

    The watchdog thread runs in the background, sampling error counts
    from ``MetricsPlugin`` at a fixed interval.  When the sliding-window
    error rate exceeds ``DISTLLM_HEALTH_ERROR_RATE_THRESHOLD``, the
    circuit breaker opens and readiness returns 503 until the rate drops
    or the open-timer expires (half-open state).
    """

    def name(self) -> str:
        return "health"

    def version(self) -> str:
        return "1.0.0"

    # ── Lifecycle ────────────────────────────────────────────────────────

    def on_init(self, context: dict[str, Any]) -> None:
        self._enabled = os.environ.get("DISTLLM_PLUGIN_HEALTH_ENABLED", "0") == "1"
        if not self._enabled:
            return

        # Thresholds
        self._error_rate_threshold = self._float_env(
            "DISTLLM_HEALTH_ERROR_RATE_THRESHOLD", 0.5
        )
        self._error_window = self._int_env(
            "DISTLLM_HEALTH_ERROR_WINDOW_SECONDS", 60
        )
        self._watchdog_interval = self._int_env(
            "DISTLLM_HEALTH_WATCHDOG_INTERVAL", 10
        )
        self._gpu_mem_max = self._float_env(
            "DISTLLM_HEALTH_GPU_MEMORY_MAX_PERCENT", 95.0
        )
        self._sys_mem_max = self._float_env(
            "DISTLLM_HEALTH_SYSTEM_MEMORY_MAX_PERCENT", 90.0
        )
        self._circuit_open_seconds = self._int_env(
            "DISTLLM_HEALTH_CIRCUIT_OPEN_SECONDS", 30
        )

        # Internal state
        self._lock = threading.Lock()
        self._circuit_open = False
        self._circuit_opened_at: float = 0.0
        self._last_error_count: int = 0
        self._current_error_rate: float = 0.0
        self._last_check_ts: float = time.time()
        self._ready: bool = True
        self._ready_reasons: list[str] = []

        # Watchdog thread (started in on_start)
        self._stop_event = threading.Event()
        self._watchdog_thread: threading.Thread | None = None

        # References resolved during on_start
        self._metrics_plugin: Any = None
        self._system_monitor: Any = None
        self._coordinator: Any = None

        logger.info(
            f"HealthPlugin: enabled, error_rate_threshold={self._error_rate_threshold}, "
            f"window={self._error_window}s, watchdog_interval={self._watchdog_interval}s, "
            f"gpu_mem_max={self._gpu_mem_max}%, sys_mem_max={self._sys_mem_max}%, "
            f"circuit_open={self._circuit_open_seconds}s"
        )

    def on_start(self, context: dict[str, Any]) -> None:
        if not self._enabled:
            return

        self._stop_event.clear()
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop,
            daemon=True,
            name="health-watchdog",
        )
        self._watchdog_thread.start()
        logger.info("HealthPlugin: watchdog thread started")

    def on_stop(self, context: dict[str, Any]) -> None:
        if not self._enabled:
            return
        self._stop_event.set()
        if self._watchdog_thread and self._watchdog_thread.is_alive():
            self._watchdog_thread.join(timeout=5.0)
        logger.info("HealthPlugin: watchdog thread stopped")

    # ── Health endpoints (called by route handlers) ──────────────────────

    def liveness(self) -> dict[str, Any]:
        """``/healthz`` — process is alive and responsive.

        Always returns 200 unless the plugin itself is broken.
        """
        return {
            "status": "alive",
            "plugin": self.name(),
            "timestamp": time.time(),
        }

    def readiness(self) -> tuple[dict[str, Any], int]:
        """``/readyz`` — service is ready to accept traffic.

        Returns a tuple of (body, status_code).  The caller is responsible
        for returning the appropriate HTTP response.

        Checks performed:
        1. Coordinator is loaded (model available).
        2. Circuit breaker is not open.
        3. Error rate is below threshold.
        4. GPU memory is below ceiling.
        5. System memory is below ceiling.
        """
        if not self._enabled:
            return {"status": "ready", "plugin": self.name()}, 200

        reasons: list[str] = []

        with self._lock:
            if self._circuit_open:
                elapsed = time.time() - self._circuit_opened_at
                if elapsed >= self._circuit_open_seconds:
                    # Transition to half-open: allow a probe request
                    self._circuit_open = False
                    self._circuit_opened_at = 0.0
                    logger.info("HealthPlugin: circuit breaker half-open (probe allowed)")
                else:
                    reasons.append(
                        f"circuit_breaker_open (retry in "
                        f"{int(self._circuit_open_seconds - elapsed)}s)"
                    )

            error_rate = self._current_error_rate

        # Coordinator / model check
        coord = self._resolve_coordinator()
        if coord is None:
            reasons.append("no_coordinator")

        # Resource checks
        resource_reasons = self._check_resources()
        reasons.extend(resource_reasons)

        # Error rate check
        if error_rate > self._error_rate_threshold:
            reasons.append(
                f"error_rate={error_rate:.3f} > threshold={self._error_rate_threshold}"
            )

        ready = len(reasons) == 0
        with self._lock:
            self._ready = ready
            self._ready_reasons = reasons

        body: dict[str, Any] = {
            "status": "ready" if ready else "not_ready",
            "plugin": self.name(),
            "error_rate": round(error_rate, 4),
            "circuit_breaker": "open" if self._circuit_open else "closed",
            "timestamp": time.time(),
        }
        if reasons:
            body["reasons"] = reasons

        status_code = 200 if ready else 503
        return body, status_code

    def get_status(self) -> dict[str, Any]:
        """Full health status for diagnostics / dashboards."""
        with self._lock:
            return {
                "enabled": self._enabled,
                "ready": self._ready,
                "ready_reasons": list(self._ready_reasons),
                "error_rate": round(self._current_error_rate, 4),
                "error_rate_threshold": self._error_rate_threshold,
                "circuit_breaker": "open" if self._circuit_open else "closed",
                "circuit_open_seconds": self._circuit_open_seconds,
                "gpu_memory_max_percent": self._gpu_mem_max,
                "system_memory_max_percent": self._sys_mem_max,
            }

    # ── Watchdog thread ──────────────────────────────────────────────────

    def _watchdog_loop(self) -> None:
        """Background loop that samples error rate and trips the circuit."""
        consecutive_failures = 0
        max_backoff = 60.0

        while not self._stop_event.is_set():
            try:
                self._stop_event.wait(self._watchdog_interval)
                if self._stop_event.is_set():
                    break

                self._check_error_rate()
                self._check_and_trip_circuit()
                consecutive_failures = 0

            except Exception as e:
                consecutive_failures += 1
                logger.error(
                    f"HealthPlugin watchdog error ({consecutive_failures}x): {e}"
                )
                backoff = min(consecutive_failures * 2.0, max_backoff)
                if self._stop_event.wait(backoff):
                    break

    def _check_error_rate(self) -> None:
        """Sample error count from MetricsPlugin and compute rate."""
        metrics_plugin = self._resolve_metrics_plugin()
        if metrics_plugin is None:
            return

        try:
            counts = metrics_plugin.get_counts()
            current_errors = counts.get("on_error", 0) + counts.get("server_error", 0)
        except Exception:
            return

        now = time.time()
        with self._lock:
            delta_errors = max(0, current_errors - self._last_error_count)
            delta_time = max(now - self._last_check_ts, 0.001)
            self._current_error_rate = delta_errors / delta_time
            self._last_error_count = current_errors
            self._last_check_ts = now

    def _check_and_trip_circuit(self) -> None:
        """Open the circuit breaker if the error rate exceeds the threshold."""
        with self._lock:
            if self._current_error_rate > self._error_rate_threshold:
                if not self._circuit_open:
                    self._circuit_open = True
                    self._circuit_opened_at = time.time()
                    logger.warning(
                        f"HealthPlugin: circuit breaker OPENED "
                        f"(error_rate={self._current_error_rate:.3f} > "
                        f"threshold={self._error_rate_threshold})"
                    )
            elif self._circuit_open:
                # Error rate dropped below threshold — close the circuit
                self._circuit_open = False
                self._circuit_opened_at = 0.0
                logger.info(
                    f"HealthPlugin: circuit breaker CLOSED "
                    f"(error_rate={self._current_error_rate:.3f})"
                )

    # ── Resource checks ──────────────────────────────────────────────────

    def _check_resources(self) -> list[str]:
        """Check GPU and system memory against configured ceilings."""
        reasons: list[str] = []
        monitor = self._resolve_system_monitor()
        if monitor is None:
            return reasons

        try:
            metrics = monitor.collect()
        except Exception:
            return reasons

        # GPU memory
        gpu = metrics.get("gpu")
        if gpu and "memory_percent" in gpu:
            if gpu["memory_percent"] > self._gpu_mem_max:
                reasons.append(
                    f"gpu_memory={gpu['memory_percent']:.1f}% > "
                    f"max={self._gpu_mem_max}%"
                )

        # System memory
        cpu = metrics.get("cpu")
        if cpu and "memory_percent" in cpu:
            if cpu["memory_percent"] > self._sys_mem_max:
                reasons.append(
                    f"system_memory={cpu['memory_percent']:.1f}% > "
                    f"max={self._sys_mem_max}%"
                )

        return reasons

    # ── Reference resolution ─────────────────────────────────────────────

    def _resolve_metrics_plugin(self) -> Any:
        """Lazily resolve MetricsPlugin from the plugin system context."""
        if self._metrics_plugin is not None:
            return self._metrics_plugin
        # The plugin system stores registered instances; look for "metrics"
        try:
            from distllm.api.server import state as _server_state
            ps = getattr(_server_state, "plugin_system", None)
            if ps is not None:
                inst = ps.get_plugin("metrics")
                if inst and inst.instance is not None:
                    self._metrics_plugin = inst.instance
                    return self._metrics_plugin
        except (ImportError, AttributeError):
            pass
        return None

    def _resolve_system_monitor(self) -> Any:
        """Lazily resolve SystemMonitor from the application state."""
        if self._system_monitor is not None:
            return self._system_monitor
        try:
            from distllm.api.api_state import g
            mon = g.monitor
            if mon is not None:
                self._system_monitor = mon
                return self._system_monitor
        except (ImportError, AttributeError):
            pass
        return None

    def _resolve_coordinator(self) -> Any:
        """Lazily resolve the Coordinator from application state."""
        try:
            from distllm.api.api_state import g
            return g.coordinator
        except (ImportError, AttributeError):
            return None

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _int_env(key: str, default: int) -> int:
        try:
            return int(os.environ.get(key, str(default)))
        except (ValueError, TypeError):
            return default

    @staticmethod
    def _float_env(key: str, default: float) -> float:
        try:
            return float(os.environ.get(key, str(default)))
        except (ValueError, TypeError):
            return default
