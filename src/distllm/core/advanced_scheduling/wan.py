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
    # batch_scheduler.set_wan_mode contract
    chunk_multiplier: float = 2.0
    batch_multiplier: float = 1.5
    rtt_threshold_ms: float = 10.0
    prefetch_kv: bool = True
    # When True (default) WAN latency variance disables Sarathi-Serve
    # pressure adaptation (see should_disable_pressure_adaptation).
    disable_sarathi_pressure: bool = True


class WANSchedulingPolicy:
    """Scheduling policy optimized for wide-area network inference.

    Adjusts budgets to account for WAN latency by increasing
    accumulation windows and reducing per-round-trip token counts.
    """

    def __init__(self, config: WANConfig | None = None):
        self._config = config or WANConfig()
        self._max_seen_rtt_ms: float = 0.0

    @property
    def is_wan_active(self) -> bool:
        """True while WAN mode is active (budget_computer contract)."""
        return self._config.enabled

    def adjust_budget_for_wan(
        self,
        base_prefill_tokens: int,
        base_batch_size: int,
        base_total_tokens: int,
    ) -> tuple[int, int, int]:
        """Scale budgets for high-latency links (budget_computer contract).

        Prefill chunks are multiplied by ``chunk_multiplier`` (amortize
        RTT across a bigger chunk) and batch size by ``batch_multiplier``
        (amortize per-request round trips); total token budget scales by
        both factors so the enlarged chunks/batches actually fit.

        Returns ``(prefill, batch_size, total_tokens)``, unchanged when
        WAN mode is inactive.
        """
        if not self._config.enabled:
            return base_prefill_tokens, base_batch_size, base_total_tokens
        adj_prefill = int(base_prefill_tokens * self._config.chunk_multiplier)
        adj_batch = int(base_batch_size * self._config.batch_multiplier)
        adj_total = int(
            base_total_tokens
            * self._config.chunk_multiplier
            * self._config.batch_multiplier
        )
        return adj_prefill, adj_batch, adj_total

    def detect_wan_mode(self, nodes: dict[str, Any]) -> bool:
        """Auto-enable WAN mode when any node looks like a WAN link.

        A node is WAN-like when it exposes a measured latency above the RTT
        threshold or a low bandwidth (<= 5 Gbps) indicating a wide-area hop.
        Explicitly-enabled WAN mode (``config.enabled=True``) is preserved;
        detection can only add to it.

        Returns:
            The resulting WAN-active flag.
        """
        wan = False
        for node in nodes.values():
            latency = getattr(node, "measured_latency_ms", None)
            if latency is not None:
                self._max_seen_rtt_ms = max(self._max_seen_rtt_ms, float(latency))
            if latency is not None and latency > self._config.rtt_threshold_ms:
                wan = True
                break
            bandwidth = getattr(
                node,
                "bandwidth_gbps",
                getattr(node, "memory_bandwidth_gbps", None),
            )
            if bandwidth is not None and 0 < bandwidth <= 5.0:
                wan = True
                break
        self._config.enabled = self._config.enabled or wan
        return self._config.enabled

    def stats(self) -> dict[str, Any]:
        """Return WAN-scheduling statistics (batch_scheduler contract)."""
        return {
            "enabled": self._config.enabled,
            "wan_active": self._config.enabled,
            "rtt_threshold_ms": self._config.rtt_threshold_ms,
            "chunk_multiplier": self._config.chunk_multiplier,
            "measured_max_rtt_ms": self._max_seen_rtt_ms,
        }

    def should_disable_pressure_adaptation(self) -> bool:
        """WAN latency variance dominates pressure signal."""
        return self._config.enabled and self._config.disable_sarathi_pressure

    def compute_budget(self, base_budget: Any) -> Any:
        if not self._config.enabled:
            return base_budget
        # Reduce batch size for WAN to avoid timeout
        base_budget.max_batch_size = min(base_budget.max_batch_size, 8)
        return base_budget

    def on_before_schedule(self, sequences: list) -> list:
        return sequences
