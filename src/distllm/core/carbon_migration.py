"""Carbon-Aware Load Migration.

Monitors carbon intensity in active regions and triggers migration
to cleaner regions when carbon spikes above a threshold. Uses KV cache
shipping from cross_cluster.py for live migration.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from loguru import logger


@dataclass
class MigrationEvent:
    """A carbon-triggered migration event."""
    from_region: str
    to_region: str
    from_carbon: float
    to_carbon: float
    carbon_saved: float
    request_ids: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    success: bool = False
    duration_ms: float = 0.0


class CarbonMigrationEngine:
    """Monitors carbon intensity and triggers live migration to cleaner regions.

    Usage::

        engine = CarbonMigrationEngine(carbon_provider=provider, threshold=400)
        engine.set_active_region("us-east-1", ["req-1", "req-2"])
        engine.start(migration_callback=my_migrate_fn)
    """

    def __init__(
        self,
        carbon_provider: Any = None,
        threshold: float = 400.0,
        check_interval_s: float = 300.0,
        migration_cooldown_s: float = 900.0,
    ):
        self._carbon_provider = carbon_provider
        self._threshold = threshold
        self._check_interval = check_interval_s
        self._cooldown = migration_cooldown_s
        self._active_region: str = ""
        self._active_requests: list[str] = []
        self._migration_callback: Callable[[str, str, list[str]], bool] | None = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._events: list[MigrationEvent] = []
        self._last_migration: float = 0.0
        self._lock = threading.Lock()

    def set_active_region(self, region: str, request_ids: list[str] | None = None) -> None:
        """Set the currently active region and tracked requests."""
        with self._lock:
            self._active_region = region
            if request_ids is not None:
                self._active_requests = request_ids

    def set_migration_callback(self, callback: Callable[[str, str, list[str]], bool]) -> None:
        """Set the migration function.

        The callback receives (from_region, to_region, request_ids) and
        returns True if migration succeeded.
        """
        self._migration_callback = callback

    def start(self) -> None:
        """Start the carbon monitoring loop."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="carbon-migration",
        )
        self._thread.start()
        logger.info(f"Carbon migration engine started (threshold={self._threshold} gCO2/kWh)")

    def stop(self) -> None:
        """Stop the monitoring loop."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    def _monitor_loop(self) -> None:
        while self._running:
            time.sleep(self._check_interval)
            try:
                self._check_and_migrate()
            except Exception as e:
                logger.debug(f"Carbon migration check failed: {e}")

    def _check_and_migrate(self) -> None:
        """Check if current region exceeds carbon threshold and migrate if needed."""
        if not self._carbon_provider or not self._active_region:
            return
        # Check cooldown
        if time.time() - self._last_migration < self._cooldown:
            return
        # Get current carbon intensity
        current = self._carbon_provider.get_intensity(self._active_region)
        if current.gco2_per_kwh <= self._threshold:
            return
        # Find a cleaner region
        from distllm.core.cross_cloud_router import _REGIONAL_CARBON_INTENSITY
        best_region = None
        best_carbon = current.gco2_per_kwh
        for region, carbon in _REGIONAL_CARBON_INTENSITY.items():
            if carbon < best_carbon * 0.7:  # At least 30% cleaner
                best_region = region
                best_carbon = carbon
        if not best_region:
            return
        # Trigger migration
        logger.warning(
            f"Carbon spike in {self._active_region}: {current.gco2_per_kwh:.0f} gCO2/kWh "
            f"(threshold: {self._threshold}). Migrating to {best_region} ({best_carbon:.0f})"
        )
        event = MigrationEvent(
            from_region=self._active_region,
            to_region=best_region,
            from_carbon=current.gco2_per_kwh,
            to_carbon=best_carbon,
            carbon_saved=current.gco2_per_kwh - best_carbon,
            request_ids=list(self._active_requests),
        )
        if self._migration_callback:
            try:
                success = self._migration_callback(
                    self._active_region, best_region, self._active_requests
                )
                event.success = success
                if success:
                    self._active_region = best_region
                    self._last_migration = time.time()
            except Exception as e:
                logger.error(f"Migration callback failed: {e}")
                event.success = False
        with self._lock:
            self._events.append(event)

    def get_events(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get recent migration events."""
        with self._lock:
            return [
                {
                    "from": e.from_region,
                    "to": e.to_region,
                    "from_carbon": e.from_carbon,
                    "to_carbon": e.to_carbon,
                    "carbon_saved": e.carbon_saved,
                    "success": e.success,
                    "timestamp": e.timestamp,
                }
                for e in self._events[-limit:]
            ]


class SLATier:
    """An SLA tier with routing constraints."""

    def __init__(
        self,
        name: str,
        prefer_spot: bool = True,
        allow_on_demand: bool = True,
        max_latency_ms: float = 200.0,
        max_carbon_intensity: float = float("inf"),
        max_price_per_hour: float = float("inf"),
        carbon_weight: float = 0.3,
    ):
        self.name = name
        self.prefer_spot = prefer_spot
        self.allow_on_demand = allow_on_demand
        self.max_latency_ms = max_latency_ms
        self.max_carbon_intensity = max_carbon_intensity
        self.max_price_per_hour = max_price_per_hour
        self.carbon_weight = carbon_weight

    def to_routing_kwargs(self) -> dict[str, Any]:
        """Convert to kwargs for CrossCloudRouter.select_provider_carbon_aware()."""
        return {
            "prefer_spot": self.prefer_spot,
            "max_latency_ms": self.max_latency_ms,
            "max_price": self.max_price_per_hour,
            "max_carbon_intensity": self.max_carbon_intensity,
            "carbon_weight": self.carbon_weight,
        }


# Predefined SLA tiers
SLA_TIERS: dict[str, SLATier] = {
    "critical": SLATier(
        name="Critical",
        prefer_spot=False,
        allow_on_demand=True,
        max_latency_ms=50.0,
        max_carbon_intensity=float("inf"),
        max_price_per_hour=float("inf"),
        carbon_weight=0.0,
    ),
    "standard": SLATier(
        name="Standard",
        prefer_spot=True,
        allow_on_demand=True,
        max_latency_ms=200.0,
        max_carbon_intensity=500.0,
        max_price_per_hour=50.0,
        carbon_weight=0.15,
    ),
    "batch": SLATier(
        name="Batch",
        prefer_spot=True,
        allow_on_demand=False,
        max_latency_ms=5000.0,
        max_carbon_intensity=300.0,
        max_price_per_hour=10.0,
        carbon_weight=0.5,
    ),
    "green": SLATier(
        name="Green",
        prefer_spot=True,
        allow_on_demand=False,
        max_latency_ms=5000.0,
        max_carbon_intensity=200.0,
        max_price_per_hour=float("inf"),
        carbon_weight=1.0,
    ),
}


def get_sla_tier(name: str) -> SLATier:
    """Get a predefined SLA tier by name."""
    return SLA_TIERS.get(name.lower(), SLA_TIERS["standard"])


def list_sla_tiers() -> list[dict[str, Any]]:
    """List all predefined SLA tiers."""
    return [
        {
            "name": tier.name,
            "prefer_spot": tier.prefer_spot,
            "max_latency_ms": tier.max_latency_ms,
            "max_carbon_intensity": tier.max_carbon_intensity,
            "carbon_weight": tier.carbon_weight,
        }
        for tier in SLA_TIERS.values()
    ]
