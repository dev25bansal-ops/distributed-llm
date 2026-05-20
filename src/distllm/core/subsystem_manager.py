"""Subsystem lifecycle management for the Coordinator.

Provides:
- Subsystem protocol: start(), stop(), health(), stats()
- SubsystemManager: aggregates subsystems and delegates lifecycle calls
"""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Subsystem(Protocol):
    """Interface for Coordinator subsystems with managed lifecycle."""

    def start(self) -> None:
        ...

    def stop(self) -> None:
        ...

    def health(self) -> dict[str, Any]:
        ...

    def stats(self) -> dict[str, Any]:
        ...


class SubsystemManager:
    """Aggregates subsystems and delegates lifecycle calls.

    Provides unified start, stop, health, and stats across all subsystems.
    Subsystems are started in registration order and stopped in reverse order.
    """

    def __init__(self):
        self._subsystems: dict[str, Subsystem] = {}

    def register(self, name: str, subsystem: Subsystem) -> None:
        self._subsystems[name] = subsystem

    def start_all(self) -> None:
        for name, sub in self._subsystems.items():
            sub.start()

    def stop_all(self) -> None:
        for name, sub in reversed(list(self._subsystems.items())):
            sub.stop()

    def health_all(self) -> dict[str, dict[str, Any]]:
        return {name: sub.health() for name, sub in self._subsystems.items()}

    def stats_all(self) -> dict[str, dict[str, Any]]:
        return {name: sub.stats() for name, sub in self._subsystems.items()}

    def get(self, name: str) -> Subsystem | None:
        return self._subsystems.get(name)

    def remove(self, name: str) -> None:
        self._subsystems.pop(name, None)
