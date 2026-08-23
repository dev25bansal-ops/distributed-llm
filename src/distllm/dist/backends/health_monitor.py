"""Gossip-integrated health monitoring for distributed inference backends.

Maintains per-backend health scores via periodic probes covering ping
latency, throughput, memory usage, and error rate.  Health changes are
propagated through the gossip protocol so peers can make informed routing
decisions without a central health store.
"""

from __future__ import annotations

import dataclasses
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class ProbeResult:
    """Result of a single health probe."""

    latency_ms: float = 0.0
    throughput_tokens_per_s: float = 0.0
    memory_used_mb: float = 0.0
    memory_total_mb: float = 0.0
    error_rate: float = 0.0
    success: bool = True
    error_message: str = ""
    timestamp: float = 0.0


@dataclass
class HealthReport:
    """Aggregated health information for a single backend."""

    backend_id: str
    healthy: bool
    score: float
    latency_ms: float
    throughput_tokens_per_s: float
    memory_usage_pct: float
    error_rate: float
    consecutive_failures: int
    state: str  # "healthy", "degraded", "unhealthy"
    last_check_time: float
    uptime_s: float


# ---------------------------------------------------------------------------
# BackendHealthMonitor
# ---------------------------------------------------------------------------


class BackendHealthMonitor:
    """Periodically probes registered inference backends and maintains
    per-backend health scores in ``[0.0, 1.0]``.

    Health changes are propagated through the gossip protocol so that
    all peers share a consistent view of backend availability.

    Parameters
    ----------
    check_interval_s:
        Seconds between health-probe rounds (default 10).
    check_timeout_s:
        Per-probe timeout in seconds (default 5).
    gossip_protocol:
        Optional ``GossipProtocol`` instance for gossip propagation.
    gossip_client:
        Optional ``GossipClient`` instance for sending gossip messages.
    backends:
        Optional initial mapping of backend_id → probe callable.
    on_failure:
        Called when a backend transitions to degraded/unhealthy:
        ``fn(backend_id, report)``.
    on_recovery:
        Called when a previously degraded backend becomes healthy:
        ``fn(backend_id, report)``.
    window_size:
        Number of recent probe results to keep for rolling averages
        (default 10).
    failure_threshold:
        Consecutive failures before marking a backend as degraded
        (default 3).
    """

    def __init__(
        self,
        check_interval_s: float = 10.0,
        check_timeout_s: float = 5.0,
        gossip_protocol: Any = None,
        gossip_client: Any = None,
        backends: dict[str, Callable[[], ProbeResult] | None] | None = None,
        on_failure: Callable[[str, HealthReport], None] | None = None,
        on_recovery: Callable[[str, HealthReport], None] | None = None,
        window_size: int = 10,
        failure_threshold: int = 3,
    ) -> None:
        self._check_interval_s = check_interval_s
        self._check_timeout_s = check_timeout_s
        self._gossip_protocol = gossip_protocol
        self._gossip_client = gossip_client
        self._on_failure = on_failure
        self._on_recovery = on_recovery
        self._window_size = window_size
        self._failure_threshold = failure_threshold

        # backend_id → callable that performs a single probe
        self._backends: dict[str, Callable[[], ProbeResult]] = {}
        if backends:
            for bid, fn in backends.items():
                if fn is not None:
                    self._backends[bid] = fn

        # Rolling probe history per backend
        self._probe_history: dict[str, list[ProbeResult]] = {}

        # Current health state
        self._health_state: dict[str, str] = {}  # "healthy" | "degraded" | "unhealthy"
        self._consecutive_failures: dict[str, int] = {}
        self._last_state_change: dict[str, float] = {}

        # Stats
        self._total_checks: int = 0
        self._total_failures: int = 0
        self._start_time: float = time.monotonic()
        self._cumulative_response_time: float = 0.0

        self._lock = threading.Lock()
        self._running = threading.Event()
        self._health_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._executor = ThreadPoolExecutor(
            max_workers=min(16, len(self._backends) * 2 or 8),
            thread_name_prefix="backend-health",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_backend(
        self, backend_id: str, probe_fn: Callable[[], ProbeResult]
    ) -> None:
        """Register a backend for health monitoring.

        Args:
            backend_id: Unique identifier for the backend.
            probe_fn: Zero-argument callable that performs a health probe
                and returns a ``ProbeResult``.
        """
        with self._lock:
            self._backends[backend_id] = probe_fn
            if backend_id not in self._probe_history:
                self._probe_history[backend_id] = []
            if backend_id not in self._health_state:
                self._health_state[backend_id] = "healthy"
            if backend_id not in self._consecutive_failures:
                self._consecutive_failures[backend_id] = 0
        logger.info(f"Registered backend '{backend_id}' for health monitoring")

    def unregister_backend(self, backend_id: str) -> None:
        """Stop monitoring a backend."""
        with self._lock:
            self._backends.pop(backend_id, None)
            self._probe_history.pop(backend_id, None)
            self._health_state.pop(backend_id, None)
            self._consecutive_failures.pop(backend_id, None)
            self._last_state_change.pop(backend_id, None)

    def get_health(self, backend_id: str) -> HealthReport | None:
        """Return the latest aggregated health report for *backend_id*."""
        with self._lock:
            probe = self._latest_probe(backend_id)
            if probe is None:
                return None
            return self._build_report(backend_id, probe)

    def get_all_health(self) -> dict[str, HealthReport]:
        """Return health reports for all registered backends."""
        with self._lock:
            reports: dict[str, HealthReport] = {}
            for bid in list(self._backends.keys()):
                probe = self._latest_probe(bid)
                if probe is not None:
                    reports[bid] = self._build_report(bid, probe)
            return reports

    def stats(self) -> dict[str, Any]:
        """Return aggregated monitor statistics."""
        elapsed = time.monotonic() - self._start_time
        with self._lock:
            return {
                "uptime_s": round(elapsed, 2),
                "total_checks": self._total_checks,
                "total_failures": self._total_failures,
                "total_backends": len(self._backends),
                "avg_response_time_ms": round(
                    self._cumulative_response_time / max(self._total_checks, 1) * 1000,
                    2,
                ),
                "check_interval_s": self._check_interval_s,
                "running": self._running.is_set(),
            }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Begin periodic health checks in a background thread."""
        if self._running.is_set():
            logger.warning("BackendHealthMonitor is already running")
            return
        self._running.set()
        self._health_event.clear()
        self._thread = threading.Thread(
            target=self._probe_loop,
            daemon=True,
            name="backend-health-probe",
        )
        self._thread.start()
        logger.info(
            f"BackendHealthMonitor started (interval={self._check_interval_s}s, "
            f"backends={len(self._backends)})"
        )

    def stop(self) -> None:
        """Stop the background health checker."""
        self._running.clear()
        self._health_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self._executor.shutdown(wait=False)
        logger.info("BackendHealthMonitor stopped")

    # ------------------------------------------------------------------
    # Probe loop
    # ------------------------------------------------------------------

    def _probe_loop(self) -> None:
        """Background loop that periodically probes all backends."""
        consecutive_loop_errors = 0

        while not self._health_event.is_set():
            try:
                self._health_event.wait(self._check_interval_s)
                if not self._running.is_set():
                    break

                self._run_probe_round()
                consecutive_loop_errors = 0

            except Exception:
                consecutive_loop_errors += 1
                backoff = min(consecutive_loop_errors * 2.0, 60.0)
                if self._health_event.wait(backoff):
                    break

    def _run_probe_round(self) -> None:
        """Probe all backends in parallel using the thread pool."""
        with self._lock:
            backend_snapshot = dict(self._backends)

        futures: dict[Any, str] = {}
        for bid, probe_fn in backend_snapshot.items():
            fut = self._executor.submit(self._probe_single, bid, probe_fn)
            futures[fut] = bid

        from concurrent.futures import as_completed

        for future in as_completed(futures):
            bid = futures[future]
            try:
                future.result(timeout=self._check_timeout_s)
            except Exception:
                logger.warning(f"Health probe failed for backend '{bid}'")
                with self._lock:
                    self._total_failures += 1

    def _probe_single(self, backend_id: str, probe_fn: Callable[[], ProbeResult]) -> None:
        """Probe one backend and update its health state.  Runs in executor."""
        t0 = time.monotonic()
        try:
            result = probe_fn()
            elapsed = time.monotonic() - t0
            result.timestamp = t0

            with self._lock:
                self._total_checks += 1
                self._cumulative_response_time += elapsed
                self._record_probe(backend_id, result)

                if result.success:
                    self._consecutive_failures[backend_id] = 0
                    self._update_state(backend_id, True)
                else:
                    self._consecutive_failures[backend_id] = (
                        self._consecutive_failures.get(backend_id, 0) + 1
                    )
                    self._total_failures += 1
                    self._update_state(backend_id, False)

            # Propagate via gossip if available
            self._gossip_propagate(backend_id)

        except Exception as exc:
            elapsed = time.monotonic() - t0
            result = ProbeResult(
                success=False,
                error_message=str(exc),
                timestamp=t0,
            )
            with self._lock:
                self._total_checks += 1
                self._cumulative_response_time += elapsed
                self._total_failures += 1
                self._record_probe(backend_id, result)
                self._consecutive_failures[backend_id] = (
                    self._consecutive_failures.get(backend_id, 0) + 1
                )
                self._update_state(backend_id, False)
            self._gossip_propagate(backend_id)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def _update_state(self, backend_id: str, probe_succeeded: bool) -> None:
        """Evaluate state transitions based on probe result."""
        old_state = self._health_state.get(backend_id, "healthy")
        consecutive = self._consecutive_failures.get(backend_id, 0)
        new_state = old_state

        if probe_succeeded:
            if old_state in ("degraded", "unhealthy"):
                # De-escalation: need 2 consecutive successes to recover
                if consecutive == 0:
                    new_state = "healthy"
                    report = self._latest_report(backend_id)
                    if report and self._on_recovery:
                        self._on_recovery(backend_id, report)
                    self._fire_gossip_health("recovery", backend_id, 1.0)
        else:
            if consecutive >= self._failure_threshold * 2:
                new_state = "unhealthy"
            elif consecutive >= self._failure_threshold:
                new_state = "degraded"

        if new_state != old_state:
            self._health_state[backend_id] = new_state
            self._last_state_change[backend_id] = time.time()
            logger.info(
                f"Backend '{backend_id}' state change: {old_state} -> {new_state} "
                f"(failures={consecutive})"
            )
            if new_state in ("degraded", "unhealthy"):
                report = self._latest_report(backend_id)
                if report and self._on_failure:
                    self._on_failure(backend_id, report)

    def _record_probe(self, backend_id: str, result: ProbeResult) -> None:
        """Append a probe result to the rolling history for *backend_id*."""
        history = self._probe_history.setdefault(backend_id, [])
        history.append(result)
        if len(history) > self._window_size:
            history.pop(0)

    def _latest_probe(self, backend_id: str) -> ProbeResult | None:
        """Return the most recent probe result, or None."""
        history = self._probe_history.get(backend_id, [])
        return history[-1] if history else None

    def _latest_report(self, backend_id: str) -> HealthReport | None:
        probe = self._latest_probe(backend_id)
        if probe is None:
            return None
        return self._build_report(backend_id, probe)

    def _build_report(self, backend_id: str, probe: ProbeResult) -> HealthReport:
        """Build a HealthReport from the latest probe and current state."""
        score = self._compute_score(backend_id)
        state = self._health_state.get(backend_id, "healthy")
        mem_pct = (
            (probe.memory_used_mb / probe.memory_total_mb * 100.0)
            if probe.memory_total_mb > 0
            else 0.0
        )
        return HealthReport(
            backend_id=backend_id,
            healthy=(score >= 0.5 and state == "healthy"),
            score=score,
            latency_ms=probe.latency_ms,
            throughput_tokens_per_s=probe.throughput_tokens_per_s,
            memory_usage_pct=mem_pct,
            error_rate=probe.error_rate,
            consecutive_failures=self._consecutive_failures.get(backend_id, 0),
            state=state,
            last_check_time=probe.timestamp,
            uptime_s=time.time() - self._start_time,
        )

    def _compute_score(self, backend_id: str) -> float:
        """Composite health score in [0.0, 1.0] based on rolling probe history.

        Weighted factors:
          - Success ratio (50%)
          - Latency score (20%): lower is better
          - Memory usage score (15%): lower is better
          - Error rate (15%): lower is better
        """
        history = self._probe_history.get(backend_id, [])
        if not history:
            return 1.0  # No data yet -- assume healthy

        # Success ratio over the window
        successes = sum(1 for r in history if r.success)
        success_ratio = successes / len(history)

        # Average latency (lower = better)
        avg_latency = sum(r.latency_ms for r in history if r.success) / max(successes, 1)
        # Latency score: 100ms = 0.9, 1s = 0.5, 5s = 0.1
        latency_score = max(0.0, 1.0 - (avg_latency / 5000.0))

        # Average memory usage
        avg_mem_usage = 0.0
        mem_count = 0
        for r in history:
            if r.memory_total_mb > 0:
                avg_mem_usage += r.memory_used_mb / r.memory_total_mb
                mem_count += 1
        mem_score = 1.0 - (avg_mem_usage / max(mem_count, 1)) if mem_count > 0 else 1.0

        # Average error rate (inverted)
        avg_error_rate = sum(r.error_rate for r in history) / len(history)
        error_rate_score = max(0.0, 1.0 - avg_error_rate)

        score = (
            0.50 * success_ratio
            + 0.20 * latency_score
            + 0.15 * mem_score
            + 0.15 * error_rate_score
        )
        return round(max(0.0, min(1.0, score)), 4)

    # ------------------------------------------------------------------
    # Gossip propagation
    # ------------------------------------------------------------------

    def _gossip_propagate(self, backend_id: str) -> None:
        """Send a health update via gossip if protocol and client are available."""
        if self._gossip_protocol is None or self._gossip_client is None:
            return

        report = self._latest_report(backend_id)
        if report is None:
            return

        try:
            msg = {
                "type": "backend_health",
                "node_id": self._gossip_protocol.state.node_id,
                "backend_id": backend_id,
                "health_score": report.score,
                "state": report.state,
                "latency_ms": report.latency_ms,
                "timestamp": time.time(),
            }
            signed = self._gossip_protocol.sign_message(msg)
            for peer in self._gossip_protocol.get_peers():
                self._gossip_client.exchange(peer, signed)
        except Exception:
            logger.opt(exception=True).debug(
                f"Gossip propagate failed for backend '{backend_id}'"
            )

    def _fire_gossip_health(
        self, event_type: str, backend_id: str, score: float
    ) -> None:
        """Send a targeted health-state-change gossip message."""
        if self._gossip_protocol is None or self._gossip_client is None:
            return
        try:
            msg = {
                "type": f"backend_{event_type}",
                "node_id": self._gossip_protocol.state.node_id,
                "backend_id": backend_id,
                "health_score": score,
                "timestamp": time.time(),
            }
            signed = self._gossip_protocol.sign_message(msg)
            for peer in self._gossip_protocol.get_peers():
                self._gossip_client.exchange(peer, signed)
        except Exception:
            pass

    def process_gossip_health_message(self, msg: dict) -> None:
        """Process an incoming gossip health message from a peer.

        Expected fields: ``backend_id``, ``health_score``, ``state``,
        ``latency_ms``, ``timestamp``, ``node_id``.
        """
        if self._gossip_protocol and not self._gossip_protocol.verify_message(msg):
            logger.warning(f"Ignoring gossip health message with invalid HMAC from {msg.get('node_id', '?')}")
            return
        # Incoming gossip health data is informational; the local monitor
        # always uses its own probes for routing decisions but makes the
        # remote data available for aggregate views.
        # TODO: merge remote health scores for backends not locally monitored
