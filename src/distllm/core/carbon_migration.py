"""Carbon-Aware Load Migration.

Monitors carbon intensity in active regions and triggers migration
to cleaner regions when carbon spikes above a threshold. Uses KV cache
shipping from cross_cluster.py for live migration.
"""

from __future__ import annotations

import os
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


class CarbonIntensityClient:
    """Client for real-time carbon intensity data via electricityMap API.

    Uses the https://api.electricitymap.org endpoint.  Set
    ``DISTLLM_CARBON_API_KEY`` to your electricityMap API token.
    """

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key or os.environ.get("DISTLLM_CARBON_API_KEY", "")
        self._base_url = "https://api.electricitymap.org/v3"
        self._cache: dict[str, tuple[float, float]] = {}  # zone -> (gco2/kWh, timestamp)
        self._cache_ttl: float = 600.0  # 10 min
        self._lock = threading.Lock()

    def get_intensity(self, zone: str) -> float:
        """Get current carbon intensity for a zone (gCO2/kWh).

        Uses cached data when available (< 10 min old).
        Returns 0.0 on failure.
        """
        now = time.time()
        with self._lock:
            cached = self._cache.get(zone)
            if cached and (now - cached[1]) < self._cache_ttl:
                return cached[0]

        if not self._api_key:
            # Fallback to static regional data
            from distllm.core.cross_cloud_router import _REGIONAL_CARBON_INTENSITY
            val = _REGIONAL_CARBON_INTENSITY.get(zone, 0.0)
            with self._lock:
                self._cache[zone] = (val, now)
            return val

        try:
            import httpx
            resp = httpx.get(
                f"{self._base_url}/carbon-intensity/latest",
                params={"zone": zone},
                headers={"auth-token": self._api_key},
                timeout=10.0,
            )
            if resp.status_code == 200:
                data = resp.json()
                val = float(data.get("carbonIntensity", 0.0))
                with self._lock:
                    self._cache[zone] = (val, now)
                return val
        except Exception as e:
            logger.debug(f"Carbon API request failed for {zone}: {e}")

        return 0.0

    def get_all_intensities(self, zones: list[str]) -> dict[str, float]:
        """Fetch intensities for multiple zones."""
        return {z: self.get_intensity(z) for z in zones}


class CarbonMigrationEngine:
    """Monitors carbon intensity and triggers live migration to cleaner regions.

    Uses the electricityMap API when configured, with static fallback.

    Usage::

        engine = CarbonMigrationEngine(threshold=400)
        engine.set_active_region("us-east-1", ["req-1", "req-2"])
        engine.start(migration_callback=my_migrate_fn)
    """

    def __init__(
        self,
        carbon_provider: Any = None,
        threshold: float = 400.0,
        check_interval_s: float = 300.0,
        migration_cooldown_s: float = 900.0,
        min_savings_pct: float = 20.0,
    ):
        if carbon_provider is None:
            carbon_provider = CarbonIntensityClient()
        self._carbon_provider = carbon_provider
        self._threshold = threshold
        self._check_interval = check_interval_s
        self._cooldown = migration_cooldown_s
        self._min_savings_pct = min_savings_pct / 100.0
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
        # Get current carbon intensity via real-time API (with static fallback)
        current_val = self._carbon_provider.get_intensity(self._active_region) if hasattr(self._carbon_provider, 'get_intensity') else 0.0
        if isinstance(current_val, (int, float)):
            current_gco2 = current_val
        else:
            current_gco2 = current_val.gco2_per_kwh if hasattr(current_val, 'gco2_per_kwh') else 0.0

        if current_gco2 <= self._threshold:
            return

        # Find a cleaner region using carbon provider + fallback
        regions_to_check = getattr(self._carbon_provider, 'get_all_intensities', None)
        if regions_to_check:
            from distllm.core.cross_cloud_router import _REGIONAL_CARBON_INTENSITY
            intensities = regions_to_check(list(_REGIONAL_CARBON_INTENSITY.keys()))
        else:
            from distllm.core.cross_cloud_router import _REGIONAL_CARBON_INTENSITY
            intensities = dict(_REGIONAL_CARBON_INTENSITY)

        best_region = None
        best_carbon = current_gco2
        for region, carbon in intensities.items():
            if carbon > 0 and carbon < best_carbon * (1.0 - self._min_savings_pct):
                best_region = region
                best_carbon = carbon
        if not best_region:
            return
        # Trigger migration
        logger.warning(
            f"Carbon spike in {self._active_region}: {current_gco2:.0f} gCO2/kWh "
            f"(threshold: {self._threshold}). Migrating to {best_region} ({best_carbon:.0f})"
        )
        event = MigrationEvent(
            from_region=self._active_region,
            to_region=best_region,
            from_carbon=current_gco2,
            to_carbon=best_carbon,
            carbon_saved=current_gco2 - best_carbon,
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
