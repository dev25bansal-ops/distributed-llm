"""Coordinator lifecycle and state management.

Extracted from coordinator.py to reduce the monolith.
Handles: server lifecycle, state properties, background thread management.
"""

from __future__ import annotations

import threading
import time

from loguru import logger


class CoordinatorState:
    """Manages coordinator lifecycle, state, and background threads."""

    def __init__(self):
        self._running = threading.Event()
        self._rebalancer_task: threading.Thread | None = None
        self._gossip_loop_task: threading.Thread | None = None
        self._start_time = time.time()

    # -- Running state --

    @property
    def is_running(self) -> bool:
        return self._running.is_set()

    def start(self) -> None:
        self._running.set()

    def stop(self) -> None:
        self._running.clear()
        logger.info("Background tasks signaled to stop")

    # -- Background thread management --

    def start_gossip_loop(self, target, daemon: bool = True) -> threading.Thread:
        self._gossip_loop_task = threading.Thread(target=target, daemon=daemon)
        self._gossip_loop_task.start()
        return self._gossip_loop_task

    def start_rebalancer_loop(self, target, daemon: bool = True) -> threading.Thread:
        self._rebalancer_task = threading.Thread(target=target, daemon=daemon)
        self._rebalancer_task.start()
        return self._rebalancer_task

    def wait_for_termination(self, timeout: float = 5.0) -> None:
        if self._gossip_loop_task is not None and self._gossip_loop_task.is_alive():
            self._gossip_loop_task.join(timeout=timeout)
        if self._rebalancer_task is not None and self._rebalancer_task.is_alive():
            self._rebalancer_task.join(timeout=timeout)

    # -- Uptime --

    def uptime_s(self) -> float:
        return time.time() - self._start_time
