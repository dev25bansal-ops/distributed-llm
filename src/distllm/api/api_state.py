"""Shared application state proxy for cross-module access.

Tests monkey-patch ``distllm.api.server.coordinator`` directly. Because
Python copies values at import time for ``from X import name``, route
modules cannot use that pattern.  Instead this module uses ``sys.modules``
to always read the current value from the server module.

Routes use::

    from distllm.api.api_state import g
    if g.coordinator is None: ...
"""

import sys
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from distllm.core.coordinator import Coordinator
    from distllm.core.monitor import SystemMonitor


class _ServerGlobals:
    """Proxy that always reads/writes the live values in server.py.

    When tests do ``server_module.coordinator = mock``, accessing
    ``g.coordinator`` sees the new value because it reads via
    ``sys.modules["distllm.api.server"]`` each time.
    """

    @property
    def coordinator(self) -> "Coordinator | None":
        return sys.modules["distllm.api.server"].coordinator

    @coordinator.setter
    def coordinator(self, value: "Coordinator | None") -> None:
        sys.modules["distllm.api.server"].coordinator = value

    @property
    def monitor(self) -> "SystemMonitor | None":
        return sys.modules["distllm.api.server"].monitor

    @monitor.setter
    def monitor(self, value: "SystemMonitor | None") -> None:
        sys.modules["distllm.api.server"].monitor = value

    @property
    def _startup_time(self) -> float:
        return sys.modules["distllm.api.server"]._startup_time

    @_startup_time.setter
    def _startup_time(self, value: float) -> None:
        sys.modules["distllm.api.server"]._startup_time = value

    def __getattr__(self, name: str) -> Any:
        return getattr(sys.modules["distllm.api.server"], name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ("coordinator", "monitor", "_startup_time"):
            # Handled by the property setters above
            object.__setattr__(self, name, value)
        else:
            setattr(sys.modules["distllm.api.server"], name, value)


# Singleton used by route modules
g = _ServerGlobals()
