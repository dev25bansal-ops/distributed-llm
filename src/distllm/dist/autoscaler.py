"""Auto-scaling worker manager — starts/stops workers based on load.

Monitors queue depth, latency SLO compliance, and GPU memory usage.
When load exceeds configurable thresholds, provisions additional workers
via Kubernetes HPA, Ray Serve autoscaler, or a custom provider.

Usage::

    scaler = AutoScaler(
        min_workers=1, max_workers=10,
        scale_up_threshold=10,      # pending requests
        scale_down_threshold=2,     # pending requests
        cooldown_seconds=60,
    )
    scaler.start()
    # ... coordinator runs ...
    scaler.stop()
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable

from loguru import logger


class AutoScaler:
    """Monitors load metrics and provisions/deprovisions workers.

    Uses a simple hysteresis-based policy:
      - Scale up when pending requests > *scale_up_threshold*
      - Scale down when pending requests < *scale_down_threshold*
      - Cooldown period prevents thrashing

    The actual worker provisioning is delegated to a *provision_fn* /
    *deprovision_fn* callback pair.

    Args:
        min_workers: Minimum number of workers to keep.
        max_workers: Maximum number of workers allowed.
        scale_up_threshold: Scale up when pending requests exceed this.
        scale_down_threshold: Scale down when pending requests below this.
        cooldown_seconds: Minimum time between scale events.
        poll_interval_seconds: How often to check metrics.
        provision_fn: Callable(node_id) to start a new worker.
        deprovision_fn: Callable(node_id) to stop a worker.
        pending_requests_fn: Callable returning current pending request
            count.  Wire to ``BatchScheduler.stats()["pending_requests"]``
            for real autoscaling decisions.  When *None*, falls back to 0.
        worker_load_fn: Optional callable returning ``{node_id: active_count}``
            for per-worker load.  When provided, scale-in selects the worker
            with the *fewest* active requests instead of an arbitrary one.
    """

    def __init__(
        self,
        min_workers: int = 1,
        max_workers: int = 10,
        scale_up_threshold: int = 10,
        scale_down_threshold: int = 2,
        cooldown_seconds: float = 60.0,
        poll_interval_seconds: float = 10.0,
        provision_fn: Callable[[str], bool] | None = None,
        deprovision_fn: Callable[[str], bool] | None = None,
        pending_requests_fn: Callable[[], int] | None = None,
        worker_load_fn: Callable[[], dict[str, int]] | None = None,
    ):
        self._min = min_workers
        self._max = max_workers
        self._scale_up = scale_up_threshold
        self._scale_down = scale_down_threshold
        self._cooldown = cooldown_seconds
        self._poll_interval = poll_interval_seconds
        self._provision = provision_fn
        self._deprovision = deprovision_fn
        self._pending_requests_fn = pending_requests_fn
        self._worker_load_fn = worker_load_fn

        self._current_workers: set[str] = set()
        self._last_scale_event = 0.0
        self._running = threading.Event()
        self._thread: threading.Thread | None = None

        self._stats = {"scale_ups": 0, "scale_downs": 0, "total_provisioned": 0}

    def register_existing_worker(self, node_id: str) -> None:
        self._current_workers.add(node_id)

    def current_count(self) -> int:
        return len(self._current_workers)

    def _get_pending_requests(self) -> int:
        """Return current pending request count.

        Delegates to *pending_requests_fn* if provided; otherwise returns 0.
        """
        if self._pending_requests_fn is not None:
            try:
                return self._pending_requests_fn()
            except Exception as exc:
                logger.warning(f"pending_requests_fn failed, falling back to 0: {exc}")
                return 0
        return 0

    def start(self) -> None:
        self._running.set()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="autoscaler")
        self._thread.start()
        logger.info(
            f"AutoScaler started: min={self._min}, max={self._max}, "
            f"up={self._scale_up}, down={self._scale_down}"
        )

    def stop(self) -> None:
        self._running.clear()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)

    def _loop(self) -> None:
        while self._running.is_set():
            self._running.wait(self._poll_interval)
            if not self._running.is_set():
                break

            try:
                self._evaluate()
            except Exception as e:
                logger.error(f"AutoScaler evaluation failed: {e}")

    def _evaluate(self) -> None:
        pending = self._get_pending_requests()
        now = time.time()
        in_cooldown = (now - self._last_scale_event) < self._cooldown

        if pending > self._scale_up and not in_cooldown:
            self._scale_out()
        elif pending < self._scale_down and not in_cooldown:
            self._scale_in()

    def _scale_out(self) -> bool:
        if len(self._current_workers) >= self._max:
            return False
        if self._provision is None:
            return False

        node_id = f"worker-auto-{int(time.time())}"
        success = self._provision(node_id)
        if success:
            self._current_workers.add(node_id)
            self._last_scale_event = time.time()
            self._stats["scale_ups"] += 1
            self._stats["total_provisioned"] += 1
            logger.info(f"AutoScaler: scaled out {node_id} ({len(self._current_workers)} workers)")
        return success

    def _scale_in(self) -> bool:
        if len(self._current_workers) <= self._min:
            return False
        if self._deprovision is None:
            return False

        # POP_IDLEST: select the worker with fewest active requests.
        # When worker_load_fn is not provided or fails, fall back to
        # arbitrary removal (the previous behavior).
        if self._worker_load_fn is not None:
            try:
                loads = self._worker_load_fn()
            except Exception as exc:
                logger.warning(f"worker_load_fn failed, falling back to arbitrary: {exc}")
                loads = {}
            sorted_workers = sorted(
                self._current_workers,
                key=lambda nid: loads.get(nid, 0),
            )
            node_id = sorted_workers[0]
        else:
            node_id = next(iter(self._current_workers))

        self._current_workers.discard(node_id)
        success = self._deprovision(node_id)
        if success:
            self._last_scale_event = time.time()
            self._stats["scale_downs"] += 1
            logger.info(f"AutoScaler: scaled in {node_id} ({len(self._current_workers)} workers)")
        return success

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "current_workers": len(self._current_workers),
            "min_workers": self._min,
            "max_workers": self._max,
            **self._stats,
        }
