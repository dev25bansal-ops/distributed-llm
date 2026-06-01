"""WAN-optimized scheduling for internet-scale inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class WANConfig:
    """Configuration for WAN-optimized scheduling."""
    enabled: bool = False
    p2p_forwarding: bool = False
    tokens_before_forward: int = 4
    wan_timeout_seconds: float = 30.0
    max_retries: int = 3
    backoff_base_seconds: float = 1.0
    accumulation_window: int = 4


class WANSchedulingPolicy:
    """Scheduling policy optimized for wide-area network inference.

    Adjusts budgets to account for WAN latency by increasing
    accumulation windows and reducing per-round-trip token counts.
    """

    def __init__(self, config: WANConfig | None = None):
        self._config = config or WANConfig()

    def should_disable_pressure_adaptation(self) -> bool:
        """WAN latency variance dominates pressure signal."""
        return self._config.enabled

    def compute_budget(self, base_budget: Any) -> Any:
        if not self._config.enabled:
            return base_budget
        # Reduce batch size for WAN to avoid timeout
        base_budget.max_batch_size = min(base_budget.max_batch_size, 8)
        return base_budget

    def on_before_schedule(self, sequences: list) -> list:
        return sequences
