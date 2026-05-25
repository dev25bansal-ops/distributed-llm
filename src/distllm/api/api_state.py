"""Shared application state proxy for cross-module access.

Routes use::

    from distllm.api.api_state import g
    if g.coordinator is None: ...

All state is held in ``AppState`` (the single source of truth).
``_ServerGlobals`` is a thin proxy that delegates to ``AppState``,
allowing route modules to access ``g.coordinator`` without
importing ``server`` directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from distllm.core.coordinator import Coordinator
    from distllm.core.monitor import SystemMonitor


class _ServerGlobals:
    """Proxy delegating to AppState — the single source of truth."""

    def __init__(self) -> None:
        object.__setattr__(self, "_app_state_ref", None)

    def _get_state(self):
        ref = object.__getattribute__(self, "_app_state_ref")
        if ref is None:
            from distllm.api.server import state as app_state
            object.__setattr__(self, "_app_state_ref", app_state)
            return app_state
        return ref

    @property
    def coordinator(self) -> Coordinator | None:
        return self._get_state().coordinator

    @coordinator.setter
    def coordinator(self, value: Coordinator | None) -> None:
        self._get_state().coordinator = value

    @property
    def monitor(self) -> "SystemMonitor | None":
        return self._get_state().monitor

    @monitor.setter
    def monitor(self, value: "SystemMonitor | None") -> None:
        self._get_state().monitor = value

    @property
    def startup_time(self) -> float:
        return self._get_state().uptime_seconds

    @startup_time.setter
    def startup_time(self, value: float) -> None:
        pass


g = _ServerGlobals()
