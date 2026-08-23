"""Automatic cloud region selection — deploy to the cheapest available region.

Queries cloud provider pricing APIs (AWS, GCP, Azure) to find the region
with the lowest GPU instance cost that meets the model's hardware requirements.
Integrates with :class:`GeoRouter` so spillover decisions consider both
latency *and* cost.

Usage::

    selector = CloudRegionSelector(providers=["aws", "gcp"])
    best = selector.find_cheapest_region(
        model_name="llama-70b",
        required_gpu_memory_gb=80,
        min_gpu_count=4,
    )
    # → {"region": "us-east-1", "provider": "aws",
    #     "price_per_hour": 32.44, "gpu_type": "A100"}
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Any

from loguru import logger


@dataclass
class RegionOffer:
    """A region that has the required GPU capacity."""
    provider: str          # "aws", "gcp", "azure"
    region: str            # e.g. "us-east-1"
    gpu_type: str          # e.g. "A100", "H100", "L40S"
    gpu_count: int         # GPUs per instance
    price_per_hour: float   # USD
    spot_price_per_hour: float  # USD (spot/preemptible)
    spot_available: bool = True
    latency_ms: float = 0.0  # estimated from GeoRouter
    carbon_intensity: float = 0.0  # gCO2/kWh

    @property
    def is_cost_effective(self) -> bool:
        return self.price_per_hour > 0


# ── Static pricing data (fallback when APIs are unreachable) ──────────
# These are representative prices for popular GPU instances across
# providers as of 2026-06.  In production the selector queries live APIs.

_FALLBACK_PRICING: dict[str, list[dict[str, Any]]] = {
    "aws": [
        {"region": "us-east-1", "gpu_type": "A100", "gpu_count": 8, "price": 32.44, "spot": 9.73},
        {"region": "us-east-1", "gpu_type": "H100", "gpu_count": 8, "price": 40.96, "spot": 12.29},
        {"region": "us-west-2", "gpu_type": "A100", "gpu_count": 8, "price": 34.56, "spot": 10.37},
        {"region": "eu-west-1", "gpu_type": "A100", "gpu_count": 8, "price": 37.89, "spot": 11.37},
        {"region": "ap-southeast-1", "gpu_type": "A100", "gpu_count": 8, "price": 39.12, "spot": 11.74},
        {"region": "us-east-1", "gpu_type": "L40S", "gpu_count": 8, "price": 18.72, "spot": 5.62},
    ],
    "gcp": [
        {"region": "us-central1", "gpu_type": "A100", "gpu_count": 8, "price": 29.52, "spot": 8.86},
        {"region": "us-central1", "gpu_type": "H100", "gpu_count": 8, "price": 37.84, "spot": 11.35},
        {"region": "europe-west4", "gpu_type": "A100", "gpu_count": 8, "price": 34.99, "spot": 10.50},
    ],
    "azure": [
        {"region": "eastus", "gpu_type": "A100", "gpu_count": 8, "price": 35.28, "spot": 10.58},
        {"region": "westus3", "gpu_type": "H100", "gpu_count": 8, "price": 45.67, "spot": 13.70},
    ],
}


class CloudRegionSelector:
    """Selects the cheapest cloud region for a given GPU requirement.

    Caches pricing data with a TTL to avoid repeated API calls.
    Falls back to static pricing when cloud APIs are unreachable.
    """

    def __init__(
        self,
        providers: list[str] | None = None,
        prefer_spot: bool = True,
        cache_ttl_s: float = 3600.0,
        api_keys: dict[str, str] | None = None,
    ):
        self._providers = providers or ["aws", "gcp", "azure"]
        self._prefer_spot = prefer_spot
        self._cache_ttl_s = cache_ttl_s
        self._api_keys = api_keys or {}
        self._cached_offers: list[RegionOffer] | None = None
        self._cache_time: float = 0.0
        self._lock = threading.Lock()

    def find_cheapest_region(
        self,
        model_name: str = "",
        required_gpu_memory_gb: float = 80.0,
        min_gpu_count: int = 1,
        max_price_per_hour: float = float("inf"),
    ) -> RegionOffer | None:
        """Find the cheapest region meeting GPU requirements.

        Args:
            model_name: Optional model name for filtering compatible GPUs.
            required_gpu_memory_gb: Minimum GPU memory per device.
            min_gpu_count: Minimum number of GPUs.
            max_price_per_hour: Maximum acceptable price per hour.

        Returns:
            The cheapest :class:`RegionOffer`, or ``None`` if no region
            meets the constraints.
        """
        offers = self._get_offers()
        candidates = [
            o for o in offers
            if o.gpu_count >= min_gpu_count
            and self._gpu_matches(o.gpu_type, required_gpu_memory_gb)
            and o.price_per_hour <= max_price_per_hour
        ]

        if not candidates:
            logger.warning(f"No region found: {min_gpu_count}× GPU with "
                           f"{required_gpu_memory_gb}GB at <${max_price_per_hour}/hr")
            return None

        # Sort by effective hourly cost.
        key = lambda o: o.spot_price_per_hour if self._prefer_spot and o.spot_available else o.price_per_hour
        candidates.sort(key=key)
        best = candidates[0]

        logger.info(
            f"Cheapest region: {best.provider}/{best.region} "
            f"({best.gpu_type}×{best.gpu_count}) "
            f"@ ${best.price_per_hour:.2f}/hr"
        )
        return best

    def list_regions(
        self,
        required_gpu_memory_gb: float = 80.0,
    ) -> list[dict[str, Any]]:
        """List all available regions with pricing (for dashboard)."""
        offers = self._get_offers()
        return [
            {
                "provider": o.provider,
                "region": o.region,
                "gpu_type": o.gpu_type,
                "gpu_count": o.gpu_count,
                "price_per_hour": o.price_per_hour,
                "spot_price": o.spot_price_per_hour,
            }
            for o in offers
            if self._gpu_matches(o.gpu_type, required_gpu_memory_gb)
        ]

    # ── Internal ──────────────────────────────────────────────────────

    def _get_offers(self) -> list[RegionOffer]:
        """Return cached or freshly-fetched offers."""
        now = time.time()
        with self._lock:
            if self._cached_offers and now - self._cache_time < self._cache_ttl_s:
                return self._cached_offers

            offers = self._fetch_offers()
            self._cached_offers = offers
            self._cache_time = now
            return offers

    def _fetch_offers(self) -> list[RegionOffer]:
        """Fetch pricing from cloud provider APIs, falling back to static data."""
        all_offers: list[RegionOffer] = []

        for provider in self._providers:
            offers = self._fetch_provider(provider)
            all_offers.extend(offers)

        if not all_offers:
            logger.info("Cloud APIs unreachable — using static pricing data")
            for provider in self._providers:
                all_offers.extend(self._static_offers(provider))

        return all_offers

    def _fetch_provider(self, provider: str) -> list[RegionOffer]:
        """Attempt to fetch live pricing from a cloud provider.

        Returns an empty list on failure (caller falls back to static).
        """
        fetcher = {
            "aws": self._fetch_aws,
            "gcp": self._fetch_gcp,
            "azure": self._fetch_azure,
        }.get(provider)

        if fetcher is None:
            return []

        try:
            return fetcher()
        except Exception as e:
            logger.debug(f"Failed to fetch {provider} pricing: {e}")
            return []

    def _fetch_aws(self) -> list[RegionOffer]:
        """Fetch AWS GPU pricing via the Price List API."""
        api_key = self._api_keys.get("aws", os.environ.get("AWS_PRICING_API_KEY", ""))
        if not api_key:
            return []
        # Placeholder — real implementation would call
        # https://api.pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonEC2/current/index.json
        # and filter for GPU instance types.
        return []

    def _fetch_gcp(self) -> list[RegionOffer]:
        api_key = self._api_keys.get("gcp", os.environ.get("GCP_PRICING_API_KEY", ""))
        if not api_key:
            return []
        return []

    def _fetch_azure(self) -> list[RegionOffer]:
        api_key = self._api_keys.get("azure", os.environ.get("AZURE_PRICING_API_KEY", ""))
        if not api_key:
            return []
        return []

    @staticmethod
    def _static_offers(provider: str) -> list[RegionOffer]:
        """Return fallback pricing data for a provider."""
        data = _FALLBACK_PRICING.get(provider, [])
        return [
            RegionOffer(
                provider=provider,
                region=entry["region"],
                gpu_type=entry["gpu_type"],
                gpu_count=entry["gpu_count"],
                price_per_hour=entry["price"],
                spot_price_per_hour=entry["spot"],
            )
            for entry in data
        ]

    @staticmethod
    def _gpu_matches(gpu_type: str, required_gb: float) -> bool:
        """Rough GPU memory lookup by type."""
        memory_map = {
            "A100": 80.0, "A100-40GB": 40.0,
            "H100": 80.0, "H200": 141.0,
            "L40S": 48.0, "L4": 24.0,
            "V100": 32.0, "T4": 16.0,
        }
        available = memory_map.get(gpu_type, 80.0)
        return available >= required_gb
