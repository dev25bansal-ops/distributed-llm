"""Energy-aware scheduling — trade off batch size vs GPU power draw."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class EnergyProfile:
    """Energy profile for a GPU."""
    idle_watts: float = 50.0
    max_watts: float = 300.0
    current_watts: float = 100.0
    thermal_limit_c: float = 83.0
    current_temp_c: float = 60.0
    # Extended power-budget fields (all optional; appended so existing
    # positional constructions keep working).
    node_id: str = ""
    gpu_name: str = ""
    tdp_watts: float = 0.0
    power_budget_watts: float = 0.0


class EnergyAwareScheduler:
    """Scheduling policy that reduces batch size under thermal pressure."""

    def __init__(
        self,
        thermal_threshold_c: float = 80.0,
        max_power_watts: float = 0.0,
        energy_cost_per_kwh: float = 0.10,
    ):
        self._thermal_threshold = thermal_threshold_c
        self._max_power_watts = max_power_watts
        self._energy_cost_per_kwh = energy_cost_per_kwh
        self._profiles: dict[str, EnergyProfile] = {}
        self._total_energy_wh: float = 0.0

    def update_profile(self, node_id: str, profile: EnergyProfile) -> None:
        self._profiles[node_id] = profile

    def set_node_profile(self, profile: EnergyProfile) -> None:
        """Register/update a node profile keyed by its ``node_id``.

        Profiles without an explicit node_id are auto-keyed.
        """
        key = profile.node_id or f"node-{len(self._profiles) + 1}"
        self._profiles[key] = profile

    def update_power_draw(self, node_id: str, watts: float) -> None:
        """Record live power draw for a node (batch_scheduler contract).

        Called from NVML monitoring via ``BatchScheduler.update_node_power``.
        Creates a profile on first sight so telemetry works even when no
        explicit profile was registered.
        """
        profile = self._profiles.get(node_id)
        if profile is None:
            profile = EnergyProfile(node_id=node_id)
            self._profiles[node_id] = profile
        profile.current_watts = watts

    def get_total_power_draw(self) -> float:
        """Sum of current watts across all known nodes."""
        return sum(p.current_watts for p in self._profiles.values())

    def get_power_utilization(self) -> float:
        """Total draw / configured power budget (0.0 without a budget)."""
        if self._max_power_watts <= 0:
            return 0.0
        return self.get_total_power_draw() / self._max_power_watts

    def adjust_for_energy(
        self,
        base_batch_size: int,
        base_prefill_tokens: int,
    ) -> tuple[int, int]:
        """Power-budget-aware budget adjustment (budget_computer contract).

        Returns ``(adjusted_batch_size, adjusted_prefill_tokens)``:

        - No power budget configured or no telemetry yet: passthrough.
        - Draw above budget: scale BOTH budgets down by
          ``max(0.5, budget/draw)`` to bring power under the cap.
        - Draw well under budget (<50%): widen the batch by 25% (at most
          2x) to amortize per-request overhead; prefill unchanged.

        Inputs are never mutated.
        """
        if self._max_power_watts <= 0 or not self._profiles:
            return base_batch_size, base_prefill_tokens

        draw = self.get_total_power_draw()
        if draw > self._max_power_watts:
            scale = max(0.5, min(1.0, self._max_power_watts / draw))
            return (
                max(1, int(base_batch_size * scale)),
                max(1, int(base_prefill_tokens * scale)),
            )
        utilization = draw / self._max_power_watts
        if utilization < 0.5:
            widened = min(base_batch_size * 2, max(base_batch_size + 1, int(base_batch_size * 1.25)))
            return widened, base_prefill_tokens
        return base_batch_size, base_prefill_tokens

    def record_energy_usage(self, duration_seconds: float = 0.0) -> None:
        """Accumulate energy consumed over an iteration (batch_scheduler contract).

        Uses the current total power draw as the average draw for the
        interval and accumulates watt-hours plus dollar cost at
        ``energy_cost_per_kwh``.
        """
        if duration_seconds <= 0:
            return
        draw = self.get_total_power_draw()
        self._total_energy_wh += draw * (duration_seconds / 3600.0)

    @property
    def total_energy_wh(self) -> float:
        return self._total_energy_wh

    @property
    def total_energy_cost_usd(self) -> float:
        return (self._total_energy_wh / 1000.0) * self._energy_cost_per_kwh

    def stats(self) -> dict[str, Any]:
        """Return energy-scheduling statistics (batch_scheduler contract)."""
        return {
            "max_power_watts": self._max_power_watts,
            "energy_cost_per_kwh": self._energy_cost_per_kwh,
            "thermal_threshold_c": self._thermal_threshold,
            "node_profiles": len(self._profiles),
            "total_power_watts": self.get_total_power_draw(),
            "power_utilization_pct": self.get_power_utilization() * 100.0,
            "total_energy_wh": self._total_energy_wh,
            "total_energy_cost_usd": self.total_energy_cost_usd,
        }

    def compute_budget(self, base_budget: Any) -> Any:
        # Find hottest node
        max_temp = max(
            (p.current_temp_c for p in self._profiles.values()),
            default=0.0,
        )
        if max_temp > self._thermal_threshold:
            # Reduce batch size to lower power draw
            scale = max(0.5, 1.0 - (max_temp - self._thermal_threshold) / 10.0)
            base_budget.max_batch_size = int(base_budget.max_batch_size * scale)
        return base_budget

    def on_before_schedule(self, sequences: list) -> list:
        return sequences
