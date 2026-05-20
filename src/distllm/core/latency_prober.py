"""Latency prober for cross-cluster federation.

Asyncio-based ping loop that measures RTT between cluster pairs
and feeds results into CrossClusterLatencyMonitor.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Callable

from loguru import logger


def _ensure_current_event_loop() -> None:
    """Keep sync callers compatible with Python versions that no longer create a loop lazily."""
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())


@dataclass
class ProbeResult:
    """Result of a single latency probe."""
    source_cluster: str
    target_cluster: str
    target_node: str
    rtt_ms: float
    success: bool
    error: str = ""
    timestamp: float = 0.0


class LatencyProber:
    """Probes cross-cluster latency via async ping loop.

    Calls NodeService.Ping() RPC on target nodes and measures RTT.
    Feeds results into a callback (typically CrossClusterLatencyMonitor.record_latency).
    """

    def __init__(
        self,
        probe_interval_s: float = 5.0,
        timeout_s: float = 2.0,
        max_history: int = 100,
    ) -> None:
        _ensure_current_event_loop()
        self.probe_interval_s = probe_interval_s
        self.timeout_s = timeout_s
        self.max_history = max_history

        # Target nodes to probe: cluster_id -> [(node_id, host, port)]
        self._targets: dict[str, list[tuple[str, str, int]]] = {}
        self._history: list[ProbeResult] = []
        self._running = False
        self._task: asyncio.Task | None = None
        self._on_result: Callable[[ProbeResult], None] | None = None

        # Ping function override (for testing without gRPC)
        self._ping_fn: Callable[[str, int, float], float] | None = None

    def set_result_callback(self, callback: Callable[[ProbeResult], None]) -> None:
        """Set callback for probe results (typically CrossClusterLatencyMonitor.record_latency)."""
        self._on_result = callback

    def set_ping_function(self, ping_fn: Callable[[str, int, float], float]) -> None:
        """Set custom ping function for testing (host, port, timeout) -> rtt_ms."""
        self._ping_fn = ping_fn

    def add_target(self, cluster_id: str, node_id: str, host: str, port: int) -> None:
        """Add a node to the probe list."""
        if cluster_id not in self._targets:
            self._targets[cluster_id] = []
        # Avoid duplicates
        existing = [(n, h, p) for n, h, p in self._targets[cluster_id] if n == node_id]
        if not existing:
            self._targets[cluster_id].append((node_id, host, port))
            logger.debug(f"Added probe target: {cluster_id}/{node_id} at {host}:{port}")

    def remove_target(self, cluster_id: str, node_id: str) -> None:
        """Remove a node from the probe list."""
        if cluster_id in self._targets:
            self._targets[cluster_id] = [
                (n, h, p) for n, h, p in self._targets[cluster_id] if n != node_id
            ]

    async def start(self) -> None:
        """Start the background probing loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._probe_loop())
        logger.info(f"Latency prober started (interval={self.probe_interval_s}s)")

    async def stop(self) -> None:
        """Stop the background probing loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Latency prober stopped")

    async def probe_once(
        self, cluster_id: str, node_id: str, host: str, port: int
    ) -> ProbeResult:
        """Send a single probe and measure RTT."""
        start = time.monotonic()
        try:
            if self._ping_fn:
                # Custom ping function (for testing)
                rtt = self._ping_fn(host, port, self.timeout_s)
            else:
                # Default: simple TCP connect latency measurement
                loop = asyncio.get_event_loop()
                rtt = await loop.run_in_executor(
                    None,
                    lambda: self._tcp_ping(host, port),
                )

            elapsed_ms = (time.monotonic() - start) * 1000

            result = ProbeResult(
                source_cluster="local",
                target_cluster=cluster_id,
                target_node=node_id,
                rtt_ms=rtt + elapsed_ms,  # Total RTT includes our measurement overhead
                success=True,
                timestamp=time.time(),
            )
        except Exception as e:
            elapsed_ms = (time.monotonic() - start) * 1000
            result = ProbeResult(
                source_cluster="local",
                target_cluster=cluster_id,
                target_node=node_id,
                rtt_ms=elapsed_ms,
                success=False,
                error=str(e),
                timestamp=time.time(),
            )

        self._record_result(result)
        return result

    def get_history(
        self,
        cluster_id: str | None = None,
        limit: int = 50,
    ) -> list[ProbeResult]:
        """Get recent probe results, optionally filtered by cluster."""
        results = self._history
        if cluster_id:
            results = [r for r in results if r.target_cluster == cluster_id]
        return results[-limit:]

    def get_latest_latency(self, cluster_id: str) -> float | None:
        """Get the latest successful RTT to a cluster."""
        for result in reversed(self._history):
            if result.target_cluster == cluster_id and result.success:
                return result.rtt_ms
        return None

    def _record_result(self, result: ProbeResult) -> None:
        """Store result and trigger callback."""
        if len(self._history) >= self.max_history:
            self._history = self._history[-(self.max_history // 2):]
        self._history.append(result)

        if self._on_result and result.success:
            try:
                self._on_result(result)
            except Exception as e:
                logger.debug(f"Latency prober callback failed: {e}")

    async def _probe_loop(self) -> None:
        """Background loop that probes all targets."""
        while self._running:
            for cluster_id, nodes in list(self._targets.items()):
                for node_id, host, port in nodes:
                    if not self._running:
                        return
                    try:
                        await asyncio.wait_for(
                            self.probe_once(cluster_id, node_id, host, port),
                            timeout=self.timeout_s,
                        )
                    except asyncio.TimeoutError:
                        logger.debug(f"Probe timeout for {cluster_id}/{node_id}")
                    except Exception as e:
                        logger.debug(f"Probe failed for {cluster_id}/{node_id}: {e}")

            await asyncio.sleep(self.probe_interval_s)

    @staticmethod
    def _tcp_ping(host: str, port: int) -> float:
        """Measure TCP connect latency in milliseconds."""
        import socket
        start = time.monotonic()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2.0)
        try:
            sock.connect((host, port))
            return (time.monotonic() - start) * 1000
        finally:
            sock.close()
