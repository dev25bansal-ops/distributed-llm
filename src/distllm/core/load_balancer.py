"""Load Balancer — distribute requests across multiple coordinators.

Supports multiple strategies: round-robin, least-connections, latency-weighted,
and random. Tracks coordinator health and adapts distribution accordingly.
"""

from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from loguru import logger


class LBStrategy(Enum):
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LATENCY_WEIGHTED = "latency_weighted"
    RANDOM = "random"
    POWER_OF_TWO = "power_of_two"


@dataclass
class CoordinatorTarget:
    """A backend coordinator that can serve requests."""
    host: str
    port: int
    node_id: str = ""
    active_connections: int = 0
    total_requests: int = 0
    failed_requests: int = 0
    avg_latency_ms: float = 0.0
    weight: float = 1.0
    is_healthy: bool = True
    last_checked: float = 0.0
    last_error: str = ""
    max_connections: int = 100


@dataclass
class LBStats:
    """Load balancer statistics snapshot."""
    strategy: str
    targets: list[CoordinatorTarget]
    total_requests: int
    failed_requests: int
    avg_latency_ms: float
    health_check_interval_s: float


class LoadBalancer:
    """Request-level load balancer for distributing across coordinators.

    Usage:
        lb = LoadBalancer(strategy=LBStrategy.LATENCY_WEIGHTED)
        lb.add_target("10.0.0.1", 50050, node_id="coord-1")
        lb.add_target("10.0.0.2", 50050, node_id="coord-2")

        # Pick a target for each request
        target = lb.pick("request-123")
        # On completion:
        lb.record_success(target, latency_ms=42.0)
    """

    def __init__(
        self,
        strategy: LBStrategy = LBStrategy.LEAST_CONNECTIONS,
        health_check_interval: float = 5.0,
        max_consecutive_failures: int = 3,
    ) -> None:
        self._strategy = strategy
        self._health_check_interval = health_check_interval
        self._max_failures = max_consecutive_failures
        self._lock = threading.RLock()
        self._targets: list[CoordinatorTarget] = []
        self._rr_index = 0
        self._total_requests = 0
        self._total_failures = 0

    def set_strategy(self, strategy: LBStrategy) -> None:
        with self._lock:
            self._strategy = strategy
            self._rr_index = 0

    # ── Target management ───────────────────────────────────────────────

    def add_target(
        self, host: str, port: int, node_id: str = "",
        weight: float = 1.0, max_connections: int = 100,
    ) -> None:
        with self._lock:
            for t in self._targets:
                if t.host == host and t.port == port:
                    logger.debug(f"Target {host}:{port} already registered")
                    return
            self._targets.append(CoordinatorTarget(
                host=host, port=port, node_id=node_id or f"{host}:{port}",
                weight=weight, max_connections=max_connections,
                last_checked=time.time(),
            ))

    def remove_target(self, host: str, port: int) -> bool:
        with self._lock:
            before = len(self._targets)
            self._targets = [
                t for t in self._targets
                if not (t.host == host and t.port == port)
            ]
            self._rr_index = min(self._rr_index, max(0, len(self._targets) - 1))
            return len(self._targets) < before

    def healthy_targets(self) -> list[CoordinatorTarget]:
        with self._lock:
            now = time.time()
            return [
                t for t in self._targets
                if t.is_healthy or (now - t.last_checked) > self._health_check_interval
            ]

    def all_targets(self) -> list[CoordinatorTarget]:
        with self._lock:
            return list(self._targets)

    # ── Request routing ─────────────────────────────────────────────────

    def pick(self, request_id: str = "") -> CoordinatorTarget | None:
        """Select a target coordinator for *request_id*."""
        with self._lock:
            healthy = self.healthy_targets()
            if not healthy:
                logger.warning("No healthy targets available")
                if self._targets:
                    logger.warning("Falling back to all targets (may be unhealthy)")
                    healthy = list(self._targets)
            if not healthy:
                return None

            self._total_requests += 1
            strategy = self._strategy

            if strategy == LBStrategy.RANDOM:
                target = random.choice(healthy)

            elif strategy == LBStrategy.ROUND_ROBIN:
                self._rr_index = (self._rr_index + 1) % len(healthy)
                target = healthy[self._rr_index]

            elif strategy == LBStrategy.LEAST_CONNECTIONS:
                target = min(healthy, key=lambda t: t.active_connections)

            elif strategy == LBStrategy.LATENCY_WEIGHTED:
                scores = []
                for t in healthy:
                    latency = max(t.avg_latency_ms, 1.0)
                    connections = max(t.active_connections, 1)
                    score = (latency * connections) / max(t.weight, 0.1)
                    scores.append((t, score))
                target = min(scores, key=lambda x: x[1])[0]

            elif strategy == LBStrategy.POWER_OF_TWO:
                if len(healthy) >= 2:
                    a, b = random.sample(healthy, 2)
                    target = a if a.active_connections <= b.active_connections else b
                else:
                    target = healthy[0]
            else:
                target = healthy[0]

            target.active_connections += 1
            return target

    def record_success(
        self, target: CoordinatorTarget, latency_ms: float = 0.0
    ) -> None:
        """Record a successful request completion."""
        with self._lock:
            target.active_connections = max(0, target.active_connections - 1)
            target.total_requests += 1
            target.failed_requests = 0
            if latency_ms > 0:
                alpha = 0.3  # EMA smoothing factor
                target.avg_latency_ms = (
                    alpha * latency_ms + (1 - alpha) * target.avg_latency_ms
                )
            target.last_checked = time.time()
            target.is_healthy = True

    def record_failure(self, target: CoordinatorTarget, error: str = "") -> None:
        """Record a failed request."""
        with self._lock:
            target.active_connections = max(0, target.active_connections - 1)
            target.failed_requests += 1
            target.last_error = error
            self._total_failures += 1
            if target.failed_requests >= self._max_failures:
                target.is_healthy = False
                logger.warning(
                    f"Target {target.host}:{target.port} marked unhealthy "
                    f"({target.failed_requests} consecutive failures)"
                )
            target.last_checked = time.time()

    def mark_healthy(self, target: CoordinatorTarget) -> None:
        """Manually mark a target as healthy."""
        with self._lock:
            target.is_healthy = True
            target.failed_requests = 0

    def mark_unhealthy(self, target: CoordinatorTarget) -> None:
        with self._lock:
            target.is_healthy = False

    # ── Stats ───────────────────────────────────────────────────────────

    def stats(self) -> LBStats:
        with self._lock:
            total_latency = sum(t.avg_latency_ms for t in self._targets if t.avg_latency_ms > 0)
            healthy_count = len([t for t in self._targets if t.is_healthy])
            avg_lat = total_latency / max(healthy_count, 1)
            return LBStats(
                strategy=self._strategy.value,
                targets=list(self._targets),
                total_requests=self._total_requests,
                failed_requests=self._total_failures,
                avg_latency_ms=round(avg_lat, 2),
                health_check_interval_s=self._health_check_interval,
            )

    def reset_stats(self) -> None:
        with self._lock:
            for t in self._targets:
                t.total_requests = 0
                t.failed_requests = 0
                t.avg_latency_ms = 0.0
            self._total_requests = 0
            self._total_failures = 0


def create_load_balancer(
    hosts: list[str] | None = None,
    strategy: str = "least_connections",
) -> LoadBalancer:
    """Convenience factory: create a load balancer with initial targets."""
    try:
        strat = LBStrategy(strategy)
    except ValueError:
        strat = LBStrategy.LEAST_CONNECTIONS
    lb = LoadBalancer(strategy=strat)
    if hosts:
        for host_spec in hosts:
            parts = host_spec.split(":")
            host = parts[0]
            port = int(parts[1]) if len(parts) > 1 else 50050
            lb.add_target(host, port)
    return lb
