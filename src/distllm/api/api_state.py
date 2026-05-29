"""Shared application state proxy for cross-module access.

Routes use::

    from distllm.api.api_state import g
    if g.coordinator is None: ...

All state is held in ``AppState`` (defined in ``app_state.py``).
``_ServerGlobals`` is a thin proxy that delegates to ``AppState``,
allowing route modules to access ``g.coordinator`` without
importing ``server`` directly.
"""

from distllm.api.app_state import AppState

_state = AppState()


class _ServerGlobals:
    """Proxy delegating to ``AppState`` — the single source of truth."""

    def __init__(self) -> None:
        pass

    @property
    def coordinator(self):
        return _state.coordinator

    @coordinator.setter
    def coordinator(self, value):
        _state.coordinator = value

    @property
    def monitor(self):
        return _state.monitor

    @monitor.setter
    def monitor(self, value):
        _state.monitor = value

    @property
    def startup_time(self) -> float:
        return _state.uptime_seconds

    @startup_time.setter
    def startup_time(self, value: float) -> None:
        _state.startup_time = value


g = _ServerGlobals()
