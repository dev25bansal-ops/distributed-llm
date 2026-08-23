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

    def update_profile(self, node_id: str, profile: EnergyProfile) -> None:
        self._profiles[node_id] = profile

    def stats(self) -> dict[str, Any]:
        """Return energy-scheduling statistics (batch_scheduler contract)."""
        return {
            "max_power_watts": self._max_power_watts,
            "energy_cost_per_kwh": self._energy_cost_per_kwh,
            "thermal_threshold_c": self._thermal_threshold,
            "node_profiles": len(self._profiles),
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
