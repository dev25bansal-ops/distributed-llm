"""Single source of truth for application state.

Extracted from ``server.py`` to avoid circular imports — ``api_state``
references this module instead of importing ``server``.
"""

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from distllm.core.coordinator import Coordinator
    from distllm.core.monitor import SystemMonitor
    from distllm.observability.exporter import DistLLMPrometheusExporter


class AppState:
    """Manages shared application state — single source of truth."""

    def __init__(self):
        self.coordinator: "Coordinator | None" = None
        self.monitor: "SystemMonitor | None" = None
        self.startup_time: float = time.time()
        self.metrics_exporter: "DistLLMPrometheusExporter | None" = None
        self.ws_broadcast_task = None

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self.startup_time
