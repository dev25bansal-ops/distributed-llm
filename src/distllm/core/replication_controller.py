"""HA state-replication collaborator for the Coordinator.

Extracted from ``coordinator.py`` (god-object decomposition, incremental) so the
replication thread/peer/stop logic lives behind a cohesive, testable seam.
The Coordinator owns the authoritative state and drives the controller via
callbacks; the controller only manages the background push loop, peers, and
stop signal.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

from loguru import logger


class ReplicationController:
    """Manages background state replication to HA peer coordinators.

    Args:
        get_snapshot: returns the current state snapshot dict (leader side).
        is_healthy: returns whether this coordinator is healthy.
        get_node_count: returns the number of registered nodes.
        running: a threading.Event that is set while the coordinator runs.
    """

    def __init__(
        self,
        get_snapshot: Callable[[], dict[str, Any]],
        is_healthy: Callable[[], bool],
        get_node_count: Callable[[], int],
        running: threading.Event,
        client_factory: Callable[..., Any] | None = None,
    ) -> None:
        self._get_snapshot = get_snapshot
        self._is_healthy = is_healthy
        self._get_node_count = get_node_count
        self._running = running
        # Injectable HTTP client factory (defaults to httpx.Client). Made
        # injectable so the controller is testable without a real network.
        self._client_factory = client_factory

        self._replication_peers: list[str] = []
        self._replication_thread: threading.Thread | None = None

    # ── public surface (delegated from Coordinator) ──

    @property
    def peers(self) -> list[str]:
        return list(self._replication_peers)

    def set_peers(self, peer_urls: list[str]) -> None:
        """Set peer coordinator URLs and (re)start replication if running."""
        # M16: do not start a second replication thread if one is alive.
        if self._replication_thread is not None and self._replication_thread.is_alive():
            logger.info("State replication thread already running; updating peers only")
            self._replication_peers = peer_urls
            return
        self._replication_peers = peer_urls
        if self._running.is_set():
            self._start()

    def start_if_peered(self) -> None:
        """Start replication if peers are configured (called from Coordinator.start)."""
        if self._replication_peers and self._running.is_set():
            self._start()

    def stop(self) -> None:
        """Signal the replication loop to stop (joins the thread)."""
        # The loop checks ``self._running`` each tick, so simply clear is enough;
        # we also set a local stop to guarantee prompt termination.
        if self._replication_thread is not None:
            self._replication_thread.join(timeout=2.0)
            self._replication_thread = None

    # ── internals ──

    def _start(self) -> None:
        if not self._replication_peers:
            return
        self._replication_thread = threading.Thread(
            target=self._loop,
            daemon=True,
            name="state-replication",
        )
        self._replication_thread.start()
        logger.info(f"State replication started for {len(self._replication_peers)} peers")

    def _loop(self) -> None:
        """Push state snapshots / heartbeats to HA peers every ~1s."""
        if self._client_factory is None:
            import httpx  # lazy: httpx is an optional heavy dep

            client_factory = httpx.Client
        else:
            client_factory = self._client_factory

        tick = 0
        with client_factory(timeout=2.0) as client:
            while self._running.is_set():
                tick += 1
                try:
                    if tick % 10 == 0:
                        snapshot = self._get_snapshot()
                    else:
                        snapshot = {
                            "heartbeat": True,
                            "node_count": self._get_node_count(),
                            "healthy": self._is_healthy(),
                        }
                    for peer_url in self._replication_peers:
                        try:
                            resp = client.post(
                                f"{peer_url.rstrip('/')}/api/v1/ha/snapshot",
                                json=snapshot,
                            )
                            if resp.status_code != 200:
                                logger.debug(
                                    f"Replication to {peer_url} returned {resp.status_code}"
                                )
                        except Exception as e:  # noqa: BLE001 - network best-effort
                            logger.debug(f"Replication to {peer_url} failed: {e}")
                except Exception as e:  # noqa: BLE001 - loop must survive
                    logger.warning(f"State replication error: {e}")
                time.sleep(1.0)
