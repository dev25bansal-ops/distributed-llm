"""Coordinator failover handler for worker nodes.

When the coordinator becomes unreachable, workers need to discover and
reconnect to a new coordinator.  This module provides:

- ``CoordinatorFailoverHandler``: Monitors coordinator health and
  triggers reconnection when the coordinator is unreachable.
- Uses the HA election protocol to discover the new leader.

Usage::

    handler = CoordinatorFailoverHandler(
        coordinator_host="10.0.0.1",
        coordinator_port=50050,
        peer_hosts=[("10.0.0.2", 50050), ("10.0.0.3", 50050)],
        on_reconnect=lambda host, port: reconnect_to_coordinator(host, port),
    )
    handler.start()
    # ... handler runs in background, calls on_reconnect when coordinator changes
    handler.stop()
"""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Callable
from typing import Any

from loguru import logger


class CoordinatorFailoverHandler:
    """Monitors coordinator health and triggers failover when needed.

    Periodically checks if the current coordinator is reachable via TCP.
    When it becomes unreachable, attempts to discover a new coordinator
    from a list of known peers.

    Args:
        coordinator_host: Current coordinator hostname.
        coordinator_port: Current coordinator gRPC port.
        peer_hosts: List of (host, port) tuples for potential coordinators.
        on_reconnect: Callback ``(host, port)`` called when a new coordinator
            is discovered.  Should handle reconnection logic.
        check_interval_s: Seconds between health checks.
        failure_threshold: Consecutive failures before triggering failover.
        reconnect_timeout_s: TCP connection timeout for peer checks.
    """

    def __init__(
        self,
        coordinator_host: str,
        coordinator_port: int,
        peer_hosts: list[tuple[str, int]] | None = None,
        on_reconnect: Callable[[str, int], None] | None = None,
        check_interval_s: float = 5.0,
        failure_threshold: int = 3,
        reconnect_timeout_s: float = 3.0,
    ) -> None:
        self._coordinator_host = coordinator_host
        self._coordinator_port = coordinator_port
        self._peer_hosts = peer_hosts or []
        self._on_reconnect = on_reconnect
        self._check_interval_s = check_interval_s
        self._failure_threshold = failure_threshold
        self._reconnect_timeout_s = reconnect_timeout_s

        self._consecutive_failures = 0
        self._current_coordinator: tuple[str, int] = (coordinator_host, coordinator_port)
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._failover_count = 0

    def start(self) -> None:
        """Start the background health monitoring loop."""
        with self._lock:
            if self._running:
                return
            self._running = True
            self._thread = threading.Thread(
                target=self._monitor_loop,
                daemon=True,
                name="coordinator-failover",
            )
            self._thread.start()
            logger.info(
                f"Coordinator failover handler started "
                f"(monitoring {self._coordinator_host}:{self._coordinator_port})"
            )

    def stop(self) -> None:
        """Stop the background monitoring loop."""
        with self._lock:
            self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=self._check_interval_s * 2)

    @property
    def current_coordinator(self) -> tuple[str, int]:
        """Return the current coordinator (host, port)."""
        with self._lock:
            return self._current_coordinator

    @property
    def failover_count(self) -> int:
        """Return the number of failovers that have occurred."""
        with self._lock:
            return self._failover_count

    def update_peer_hosts(self, peers: list[tuple[str, int]]) -> None:
        """Update the list of known peer coordinators."""
        with self._lock:
            self._peer_hosts = list(peers)

    def stats(self) -> dict:
        """Return handler statistics."""
        with self._lock:
            return {
                "current_coordinator": f"{self._current_coordinator[0]}:{self._current_coordinator[1]}",
                "consecutive_failures": self._consecutive_failures,
                "failover_count": self._failover_count,
                "peer_count": len(self._peer_hosts),
                "running": self._running,
            }

    def _monitor_loop(self) -> None:
        """Background loop that checks coordinator health."""
        while self._running:
            try:
                time.sleep(self._check_interval_s)
                if not self._running:
                    break

                host, port = self._current_coordinator
                if self._check_tcp_alive(host, port):
                    with self._lock:
                        self._consecutive_failures = 0
                else:
                    with self._lock:
                        self._consecutive_failures += 1
                        failures = self._consecutive_failures

                    if failures >= self._failure_threshold:
                        logger.warning(
                            f"Coordinator {host}:{port} unreachable "
                            f"({failures} consecutive failures), triggering failover"
                        )
                        self._trigger_failover()

            except Exception as e:
                logger.warning(f"Failover monitor error: {e}")

    def _check_tcp_alive(self, host: str, port: int) -> bool:
        """Check if a host is reachable via TCP."""
        try:
            sock = socket.create_connection(
                (host, port), timeout=self._reconnect_timeout_s
            )
            sock.close()
            return True
        except (OSError, socket.timeout, ConnectionError):
            return False

    def _trigger_failover(self) -> None:
        """Discover a new coordinator from the peer list."""
        with self._lock:
            current = self._current_coordinator
            peers = list(self._peer_hosts)

        # Try each peer until we find one that's alive
        for peer_host, peer_port in peers:
            if (peer_host, peer_port) == current:
                continue

            if self._check_tcp_alive(peer_host, peer_port):
                logger.info(
                    f"Failover: new coordinator discovered at "
                    f"{peer_host}:{peer_port}"
                )
                with self._lock:
                    self._current_coordinator = (peer_host, peer_port)
                    self._consecutive_failures = 0
                    self._failover_count += 1

                if self._on_reconnect:
                    try:
                        self._on_reconnect(peer_host, peer_port)
                    except Exception as e:
                        logger.error(f"Failover reconnect callback failed: {e}")
                return

        logger.warning(
            "Failover: no reachable coordinator found among peers"
        )
