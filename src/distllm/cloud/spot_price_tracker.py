"""Spot price tracker for multi-cloud cost optimization.

Polls spot price APIs, maintains price history, and provides
preemption risk scoring for cost-aware instance selection.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from loguru import logger

from distllm.cloud.spot_provider import (
    CloudProvider,
    SpotPrice,
    SpotProvider,
)


@dataclass
class PriceRecord:
    """Cached spot price record with TTL."""
    provider: CloudProvider
    instance_type: str
    region: str
    price: float
    on_demand_price: float
    timestamp: float
    ttl_seconds: float = 300.0  # 5 minutes

    @property
    def is_stale(self) -> bool:
        return (time.time() - self.timestamp) > self.ttl_seconds


class SpotPriceTracker:
    """Tracks spot prices across cloud providers with TTL caching.

    Features:
    - Background polling loop (configurable interval)
    - Price history per instance type/region
    - Preemption risk prediction
    - Cheapest compatible instance selection
    """

    def __init__(
        self,
        poll_interval_s: float = 300.0,
        history_size: int = 288,  # 24 hours at 5-min intervals
    ) -> None:
        self.poll_interval_s = poll_interval_s
        self.history_size = history_size

        self._providers: dict[CloudProvider, SpotProvider] = {}
        self._cache: dict[str, PriceRecord] = {}  # key -> record
        self._history: dict[str, list[float]] = {}  # key -> [prices]
        self._lock = threading.Lock()
        self._running = False

    def register_provider(self, provider: SpotProvider) -> None:
        """Register a cloud provider for price tracking."""
        self._providers[provider.provider_name] = provider

    def get_current_price(
        self,
        provider: CloudProvider,
        instance_type: str,
        region: str,
    ) -> SpotPrice | None:
        """Get the current spot price (from cache or API)."""
        key = f"{provider.value}:{instance_type}:{region}"

        with self._lock:
            cached = self._cache.get(key)
            if cached and not cached.is_stale:
                return SpotPrice(
                    provider=provider,
                    instance_type=instance_type,
                    region=region,
                    price=cached.price,
                    timestamp=cached.timestamp,
                    on_demand_price=cached.on_demand_price,
                )

        # Cache miss or stale - query provider
        spot_provider = self._providers.get(provider)
        if spot_provider:
            result = spot_provider.get_current_spot_price(instance_type, region)
            if result:
                self._update_cache(key, provider, instance_type, region, result)
                return result

        return None

    def get_cheapest_compatible(
        self,
        required_vram_gb: float,
        required_compute: str = "",
        providers: list[CloudProvider] | None = None,
        region: str = "us-east-1",
    ) -> SpotPrice | None:
        """Find the cheapest spot instance that meets requirements.

        Args:
            required_vram_gb: Minimum VRAM in GB.
            required_compute: Minimum compute capability (e.g., "8.0").
            providers: List of providers to check (all if None).
            region: Target region.

        Returns:
            Cheapest compatible SpotPrice, or None.
        """
        # Instance type -> VRAM mapping (simplified)
        instance_vram = {
            "g5.xlarge": 24, "g5.2xlarge": 24, "g5.4xlarge": 24,
            "p4d.24xlarge": 80, "p5.48xlarge": 80,
            "Standard_NC24ads_A100_v4": 80,
            "a2-highgpu-1g": 40,
        }

        candidates = []
        target_providers = providers or list(self._providers.keys())

        for provider in target_providers:
            for instance_type, vram in instance_vram.items():
                if vram < required_vram_gb:
                    continue
                price = self.get_current_price(provider, instance_type, region)
                if price:
                    candidates.append(price)

        if not candidates:
            return None

        return min(candidates, key=lambda p: p.price)

    def predict_preemption_risk(
        self,
        provider: CloudProvider,
        instance_type: str,
        region: str,
    ) -> float:
        """Predict spot preemption risk (0.0-1.0).

        Risk factors:
        - Price volatility (high variance = higher risk)
        - Price trend (increasing = higher risk)
        - Current price vs on-demand ratio

        Args:
            provider: Cloud provider.
            instance_type: Instance type.
            region: Region.

        Returns:
            Risk score 0.0 (safe) to 1.0 (very risky).
        """
        key = f"{provider.value}:{instance_type}:{region}"

        with self._lock:
            history = self._history.get(key, [])

        if len(history) < 2:
            return 0.5  # Unknown risk

        # Price volatility (coefficient of variation)
        mean_price = sum(history) / len(history)
        if mean_price <= 0:
            return 0.5

        variance = sum((p - mean_price) ** 2 for p in history) / len(history)
        cv = (variance ** 0.5) / mean_price  # Coefficient of variation

        # Price trend (last 6 prices)
        recent = history[-6:]
        if len(recent) >= 2:
            trend = (recent[-1] - recent[0]) / max(recent[0], 0.001)
        else:
            trend = 0.0

        # Risk calculation
        volatility_risk = min(cv * 5, 0.5)  # Cap at 0.5
        trend_risk = max(0, min(trend * 2, 0.3))  # Cap at 0.3

        return min(volatility_risk + trend_risk + 0.2, 1.0)  # Base risk 0.2

    def _update_cache(
        self,
        key: str,
        provider: CloudProvider,
        instance_type: str,
        region: str,
        price: SpotPrice,
    ) -> None:
        """Update the price cache and history."""
        with self._lock:
            self._cache[key] = PriceRecord(
                provider=provider,
                instance_type=instance_type,
                region=region,
                price=price.price,
                on_demand_price=price.on_demand_price,
                timestamp=time.time(),
            )

            if key not in self._history:
                self._history[key] = []
            self._history[key].append(price.price)
            if len(self._history[key]) > self.history_size:
                self._history[key] = self._history[key][-self.history_size:]
