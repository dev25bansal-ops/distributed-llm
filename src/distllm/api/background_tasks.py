"""Background task manager with health monitoring, auto-restart, ordered shutdown, and Prometheus-style metrics."""

from __future__ import annotations

import asyncio
import inspect
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

from loguru import logger

try:
    from prometheus_client import Counter, Gauge, CollectorRegistry

    _HAS_PROMETHEUS = True
except ImportError:
    _HAS_PROMETHEUS = False


@dataclass
class BackgroundTask:
    """A managed background task.

    Attributes:
        name: Unique task identifier used for health reporting and lifecycle
            management.  Must be non-empty within a single manager instance.
        run: Callable to execute on each interval.  May be either a sync
            function or an async coroutine function.
        interval_s: Seconds between successive runs (wall-clock start-to-start).
        auto_restart: Whether to restart the task on failure with exponential
            backoff (1 s, 2 s, 4 s, 8 s cap).  Defaults to True.
        health_check: Optional callable invoked after each successful run.
            Return True (healthy), False (unhealthy), or None (unknown).
            A return of False sets the task status to "unhealthy" in the
            health report.
    """

    name: str
    run: Callable
    interval_s: float = 60.0
    auto_restart: bool = True
    health_check: Callable[[], bool | None] | None = None


class BackgroundTaskManager:
    """Manages registration, lifecycle, health monitoring, and ordered shutdown
    of background tasks.

    Each task runs in its own daemon thread.  The manager tracks per-task
    health (last run time, success/failure, duration, consecutive failures)
    and optionally exports Prometheus-style metrics.

    Usage::

        manager = BackgroundTaskManager()
        manager.register(BackgroundTask(name="cleanup", run=my_cleanup, interval_s=300))
        manager.start()
        # ...
        manager.stop()
    """

    _METRIC_PREFIX = "distllm_background_task_"
    _MAX_BACKOFF = 8.0

    def __init__(self) -> None:
        self._tasks: dict[str, BackgroundTask] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._stop_events: dict[str, threading.Event] = {}
        self._health: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._running = False
        self._prometheus: dict[str, Any] | None = None
        self._init_prometheus()

    # ------------------------------------------------------------------
    # Prometheus metric initialisation
    # ------------------------------------------------------------------

    def _init_prometheus(self) -> None:
        if not _HAS_PROMETHEUS:
            return
        reg = CollectorRegistry()
        self._prometheus = {
            "registry": reg,
            "runs_total": Counter(
                f"{self._METRIC_PREFIX}runs_total",
                "Total background task runs",
                ["task"],
                registry=reg,
            ),
            "success_total": Counter(
                f"{self._METRIC_PREFIX}success_total",
                "Successful background task runs",
                ["task"],
                registry=reg,
            ),
            "failure_total": Counter(
                f"{self._METRIC_PREFIX}failure_total",
                "Failed background task runs",
                ["task"],
                registry=reg,
            ),
            "duration_seconds": Gauge(
                f"{self._METRIC_PREFIX}duration_seconds",
                "Last task run duration in seconds",
                ["task"],
                registry=reg,
            ),
            "last_run_timestamp": Gauge(
                f"{self._METRIC_PREFIX}last_run_timestamp_seconds",
                "Timestamp of last task run as seconds since epoch",
                ["task"],
                registry=reg,
            ),
            "up": Gauge(
                f"{self._METRIC_PREFIX}up",
                "Task is running (1) or stopped (0)",
                ["task"],
                registry=reg,
            ),
        }

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, task: BackgroundTask) -> None:
        """Register a background task.

        Raises ``KeyError`` if a task with the same name is already
        registered.
        """
        with self._lock:
            if task.name in self._tasks:
                raise KeyError(f"Background task {task.name!r} is already registered")
            self._tasks[task.name] = task
            self._health[task.name] = {
                "status": "registered",
                "last_run": 0.0,
                "last_success": 0.0,
                "consecutive_failures": 0,
                "last_error": "",
                "last_duration": 0.0,
            }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start all registered background tasks."""
        self._running = True
        for name in list(self._tasks.keys()):
            self._start_single(name)

    def _start_single(self, name: str) -> None:
        task = self._tasks.get(name)
        if task is None:
            logger.warning(f"Cannot start unknown background task {name!r}")
            return
        stop_event = threading.Event()
        self._stop_events[name] = stop_event
        thread = threading.Thread(
            target=self._run_loop,
            args=(name, task, stop_event),
            daemon=True,
        )
        self._threads[name] = thread
        thread.start()
        self._health[name]["status"] = "running"
        self._set_prom_gauge("up", name, 1)

    def stop(self) -> None:
        """Stop all background tasks in reverse registration order."""
        self._running = False
        for name in reversed(list(self._stop_events.keys())):
            event = self._stop_events[name]
            event.set()
            thread = self._threads.get(name)
            if thread and thread.is_alive():
                thread.join(timeout=5)
            self._health[name]["status"] = "stopped"
            self._set_prom_gauge("up", name, 0)

    def restart(self, name: str) -> None:
        """Restart a single background task by name."""
        with self._lock:
            if name in self._stop_events:
                self._stop_events[name].set()
                thread = self._threads.get(name)
                if thread and thread.is_alive():
                    thread.join(timeout=5)
                del self._stop_events[name]
            self._start_single(name)

    # ------------------------------------------------------------------
    # Health & metrics
    # ------------------------------------------------------------------

    def health(self) -> dict[str, dict[str, Any]]:
        """Return a snapshot of all task health states."""
        with self._lock:
            return dict(self._health)

    def metrics(self) -> str | None:
        """Return Prometheus exposition-format string for background task metrics.

        Returns ``None`` if ``prometheus_client`` is not installed.
        """
        if self._prometheus is None:
            return None
        from prometheus_client import generate_latest

        return generate_latest(self._prometheus["registry"]).decode("utf-8")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _set_prom_gauge(self, metric_name: str, task_name: str, value: float) -> None:
        if self._prometheus is not None:
            try:
                self._prometheus[metric_name].labels(task=task_name).set(value)
            except Exception:
                pass

    def _inc_prom_counter(self, metric_name: str, task_name: str, inc: float = 1) -> None:
        if self._prometheus is not None:
            try:
                self._prometheus[metric_name].labels(task=task_name).inc(inc)
            except Exception:
                pass

    def _run_loop(self, name: str, task: BackgroundTask, stop_event: threading.Event) -> None:
        backoff = 1.0
        while self._running and not stop_event.is_set():
            start = time.monotonic()
            try:
                # Support both sync and async callables
                result = task.run()
                if inspect.isawaitable(result):
                    asyncio.run(result)

                duration = time.monotonic() - start

                with self._lock:
                    self._health[name]["last_run"] = time.time()
                    self._health[name]["last_success"] = time.time()
                    self._health[name]["consecutive_failures"] = 0
                    self._health[name]["last_error"] = ""
                    self._health[name]["status"] = "ok"
                    self._health[name]["last_duration"] = duration

                self._inc_prom_counter("runs_total", name)
                self._inc_prom_counter("success_total", name)
                self._set_prom_gauge("duration_seconds", name, duration)
                self._set_prom_gauge("last_run_timestamp", name, time.time())

                # Run health check after successful execution
                if task.health_check is not None:
                    try:
                        healthy = task.health_check()
                        if healthy is False:
                            logger.warning(f"Health check failed for background task {name}")
                            with self._lock:
                                self._health[name]["status"] = "unhealthy"
                    except Exception as hc_err:
                        logger.warning(f"Health check error for {name}: {hc_err}")

                backoff = 1.0
            except Exception as e:
                duration = time.monotonic() - start

                with self._lock:
                    self._health[name]["last_run"] = time.time()
                    self._health[name]["consecutive_failures"] += 1
                    self._health[name]["last_error"] = str(e)[:200]
                    self._health[name]["status"] = "error"
                    self._health[name]["last_duration"] = duration

                self._inc_prom_counter("runs_total", name)
                self._inc_prom_counter("failure_total", name)

                logger.warning(f"Background task {name} failed: {e}")
                if not task.auto_restart:
                    break
                backoff = min(backoff * 2, self._MAX_BACKOFF)

            elapsed = time.monotonic() - start
            sleep_time = max(0, task.interval_s - elapsed)
            stop_event.wait(timeout=sleep_time)
            if stop_event.is_set():
                break
