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

    @property
    def metrics_exporter(self):
        return _state.metrics_exporter

    @metrics_exporter.setter
    def metrics_exporter(self, value):
        _state.metrics_exporter = value

    # Generic dict-style access: g.get("name") / g["name"] resolve attributes on
    # the shared AppState.  Several route modules (routes/exchange.py,
    # authz/opa.py) rely on this pattern.
    def get(self, name, default=None):
        return getattr(_state, name, default)

    def __getitem__(self, name):
        return getattr(_state, name)

    def __setitem__(self, name, value):
        setattr(_state, name, value)

    def set(self, name, value) -> None:
        setattr(_state, name, value)


def reset_app_state_for_testing() -> None:
    """Reset the shared app state to a pristine instance (for tests).

    Mutates the existing AppState in place (rather than rebinding the module
    global) so imports that captured the old ``_state`` object — and the ``g``
    proxy — keep pointing at the same, now-reset, instance.  Rebinding would
    strand old bindings on a stale object (e.g. a test that sets
    ``_state.prompt_exchange`` after ``_state`` was reassigned elsewhere).
    """
    _state.__init__()


g = _ServerGlobals()
