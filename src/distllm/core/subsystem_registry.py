"""Subsystem Registry — lifecycle manager for Coordinator subsystems.

Extracted from the Coordinator God Object pattern to provide:

- Unified lifecycle (init → start → stop) for all subsystems
- Dependency injection between subsystems
- Health checks across subsystems
- Metric aggregation
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable

from loguru import logger


class SubsystemRegistry:
    """Manages lifecycle of coordinator subsystems.

    Usage:
        registry = SubsystemRegistry()
        registry.register("hot_swap", mgr, start=True)
        registry.register("defragmenter", defrag)
        registry.start_all()
        stats = registry.status()
    """

    def __init__(self):
        self._subsystems: dict[str, _SubsystemEntry] = {}
        self._lock = threading.RLock()

    def register(
        self,
        name: str,
        instance: Any,
        depends_on: list[str] | None = None,
        start_fn: Callable | None = None,
        stop_fn: Callable | None = None,
    ) -> None:
        """Register a subsystem.

        Args:
            name: Unique subsystem name.
            instance: The subsystem object.
            depends_on: Names of subsystems that must be started first.
            start_fn: Optional custom start function (default: instance.start).
            stop_fn: Optional custom stop function (default: instance.stop).
        """
        with self._lock:
            if name in self._subsystems:
                raise ValueError(f"Subsystem '{name}' already registered")
            self._subsystems[name] = _SubsystemEntry(
                name=name,
                instance=instance,
                depends_on=depends_on or [],
                start_fn=start_fn or getattr(instance, "start", None),
                stop_fn=stop_fn or getattr(instance, "stop", None),
            )

    def get(self, name: str, default=None) -> Any:
        with self._lock:
            entry = self._subsystems.get(name)
            return entry.instance if entry else default

    def start_all(self) -> int:
        """Start all registered subsystems respecting dependency order."""
        started = 0
        with self._lock:
            order = self._resolve_deps()
            for name in order:
                entry = self._subsystems[name]
                if entry.state == "started":
                    continue
                try:
                    if entry.start_fn:
                        entry.start_fn()
                    entry.state = "started"
                    entry.started_at = time.monotonic()
                    started += 1
                    logger.debug(f"Subsystem '{name}' started")
                except Exception as e:
                    entry.state = "error"
                    entry.error = str(e)
                    logger.error(f"Subsystem '{name}' failed to start: {e}")
        return started

    def stop_all(self) -> int:
        """Stop all subsystems in reverse dependency order."""
        stopped = 0
        with self._lock:
            order = list(reversed(self._resolve_deps()))
            for name in order:
                entry = self._subsystems[name]
                if entry.state != "started":
                    continue
                try:
                    if entry.stop_fn:
                        entry.stop_fn()
                    entry.state = "stopped"
                    stopped += 1
                except Exception as e:
                    entry.state = "error"
                    entry.error = str(e)
                    logger.error(f"Subsystem '{name}' failed to stop: {e}")
        return stopped

    def status(self) -> dict[str, dict]:
        """Return status of all subsystems."""
        result = {}
        with self._lock:
            for name, entry in self._subsystems.items():
                result[name] = {
                    "state": entry.state,
                    "error": entry.error,
                    "uptime_s": round(time.monotonic() - entry.started_at, 1) if entry.started_at else 0,
                    "depends_on": entry.depends_on,
                }
        return result

    def health(self) -> dict[str, bool]:
        """Check if all required subsystems are healthy."""
        result = {}
        with self._lock:
            for name, entry in self._subsystems.items():
                result[name] = entry.state == "started" and not entry.error
        return result

    @property
    def all_healthy(self) -> bool:
        return all(self.health().values())

    def _resolve_deps(self) -> list[str]:
        """Topological sort of subsystems by dependency."""
        visited = set()
        result = []

        def _visit(name: str, path: set[str]) -> None:
            if name in visited:
                return
            if name in path:
                raise ValueError(f"Circular dependency in subsystems: {name}")
            entry = self._subsystems.get(name)
            if entry is None:
                return
            path.add(name)
            for dep in entry.depends_on:
                _visit(dep, path)
            path.remove(name)
            visited.add(name)
            result.append(name)

        for name in list(self._subsystems.keys()):
            _visit(name, set())
        return result


class _SubsystemEntry:
    """Internal subsystem state tracker."""
    def __init__(self, name: str, instance: Any, depends_on: list[str],
                 start_fn: Callable | None, stop_fn: Callable | None):
        self.name = name
        self.instance = instance
        self.depends_on = depends_on
        self.start_fn = start_fn
        self.stop_fn = stop_fn
        self.state = "registered"  # registered → started → stopped / error
        self.error: str = ""
        self.started_at: float = 0.0
