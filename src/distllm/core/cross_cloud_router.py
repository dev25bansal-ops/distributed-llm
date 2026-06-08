"""Cross-cloud cost optimization — route to cheapest available GPU.

Routes inference requests across cloud providers (AWS, GCP, Azure)
based on real-time pricing, latency, availability, and **carbon intensity**.

Carbon-aware scheduling routes inference to the region with the lowest
grid carbon intensity, using live APIs (electricityMap, WattTime) or
static regional averages.  This is unique in open-source LLM serving.

Integrates with the cost tracker for pricing data and the federation
system for cross-cloud node discovery.

Strategic features:
- Live cloud pricing via PricingManager (AWS/GCP/Azure APIs)
- GPU type filtering with instance → GPU mapping
- Per-region provider expansion
- Priority-tier-aware routing (PreemptibleScheduler integration)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from distllm.core.pricing_providers import PricingManager, InstancePricing


# GPU type → (instance_types, memory_gb, gpu_count) quick-lookup
_GPU_INSTANCE_INDEX: dict[str, list[tuple[str, str, int, float]]] = {
    "A100": [
        ("aws", "p4d.24xlarge", 8, 80.0),
        ("aws", "p4de.24xlarge", 8, 80.0),
        ("gcp", "a2-highgpu-1g", 1, 40.0),
        ("gcp", "a2-highgpu-2g", 2, 40.0),
        ("gcp", "a2-ultragpu-1g", 1, 80.0),
        ("azure", "Standard_NC24ads_A100_v4", 1, 80.0),
        ("azure", "Standard_NC48ads_A100_v4", 2, 80.0),
        ("azure", "Standard_NC96ads_A100_v4", 4, 80.0),
    ],
    "V100": [
        ("aws", "p3.2xlarge", 1, 16.0),
        ("aws", "p3.8xlarge", 4, 16.0),
    ],
    "A10G": [
        ("aws", "g5.xlarge", 1, 24.0),
        ("aws", "g5.2xlarge", 1, 24.0),
        ("aws", "g5.4xlarge", 1, 24.0),
    ],
    "A10": [
        ("azure", "Standard_NV6ads_A10_v5", 1, 24.0),
    ],
    "L4": [
        ("gcp", "g2-standard-4", 1, 24.0),
        ("aws", "g6.xlarge", 1, 24.0),
    ],
    "T4": [
        ("gcp", "n1-standard-8-t4", 1, 16.0),
    ],
}


# Cloud provider GPU pricing (USD per GPU-hour, spot/preemptible)
_CLOUD_GPU_PRICING: dict[str, dict[str, float]] = {
    "aws": {
        "p4d.24xlarge": 14.40,   # 8x A100 80GB
        "p3.2xlarge": 3.83,      # 1x V100 16GB
        "g5.xlarge": 1.41,       # 1x A10G 24GB
        "g5.2xlarge": 1.82,      # 1x A10G 24GB
        "g6.xlarge": 1.10,       # 1x L4 24GB
        "p4de.24xlarge": 27.20,  # 8x A100 80GB
    },
    "gcp": {
        "a2-highgpu-1g": 3.67,   # 1x A100 40GB
        "a2-highgpu-2g": 7.35,   # 2x A100 40GB
        "a2-ultragpu-1g": 5.07,  # 1x A100 80GB
        "g2-standard-4": 0.84,   # 1x L4 24GB
        "n1-standard-8-t4": 0.95, # 1x T4 16GB
    },
    "azure": {
        "Standard_NC24ads_A100_v4": 3.67,   # 1x A100 80GB
        "Standard_NC48ads_A100_v4": 7.35,   # 2x A100 80GB
        "Standard_NC96ads_A100_v4": 14.69,  # 4x A100 80GB
        "Standard_NV6ads_A10_v5": 1.10,     # 1x A10 24GB
    },
}

# Instance → GPU type mapping (for filtering)
_INSTANCE_GPU_TYPE: dict[str, str] = {
    "p4d.24xlarge": "A100", "p4de.24xlarge": "A100",
    "p3.2xlarge": "V100", "p3.8xlarge": "V100", "p3.16xlarge": "V100",
    "g5.xlarge": "A10G", "g5.2xlarge": "A10G", "g5.4xlarge": "A10G",
    "g6.xlarge": "L4",
    "a2-highgpu-1g": "A100", "a2-highgpu-2g": "A100", "a2-highgpu-4g": "A100",
    "a2-highgpu-8g": "A100", "a2-ultragpu-1g": "A100", "a2-ultragpu-2g": "A100",
    "g2-standard-4": "L4", "n1-standard-8-t4": "T4",
    "Standard_NC24ads_A100_v4": "A100", "Standard_NC48ads_A100_v4": "A100",
    "Standard_NC96ads_A100_v4": "A100", "Standard_NV6ads_A10_v5": "A10",
}

# Per-region latency baselines (ms, approximate from US-East)
# M-05: To get latency between any two regions, use compute_relative_latency().
# This allows callers from different origins to get correct estimates.
_REGION_LATENCY: dict[str, float] = {
    "us-east-1": 10, "us-east-2": 15, "us-west-1": 65, "us-west-2": 70,
    "eu-west-1": 80, "eu-west-2": 75, "eu-west-3": 85, "eu-central-1": 90,
    "eu-north-1": 100, "ap-northeast-1": 150, "ap-southeast-1": 180,
    "ap-southeast-2": 200, "sa-east-1": 130, "ca-central-1": 20,
    "us-central1": 30, "us-west1": 65, "europe-west1": 85,
    "europe-west4": 90, "europe-north1": 100, "asia-east1": 170,
    "eastus": 10, "westus2": 65, "westeurope": 90, "northeurope": 80,
    "japaneast": 150, "australiaeast": 200,
}

# Spot/preemptible discount factors
_SPOT_DISCOUNT: dict[str, float] = {
    "aws": 0.30,    # ~70% discount
    "gcp": 0.25,    # ~75% discount
    "azure": 0.35,  # ~65% discount
}

# ── Carbon intensity data (gCO2/kWh, regional averages) ──────────────────────
# Source: IEA 2023, electricityMap historical data
# Lower values = cleaner grid (hydro, nuclear, wind, solar)
_REGIONAL_CARBON_INTENSITY: dict[str, float] = {
    # AWS regions
    "us-east-1": 380,       # Virginia (mixed gas/nuclear)
    "us-east-2": 410,       # Ohio (coal-heavy)
    "us-west-1": 230,       # California (solar/gas)
    "us-west-2": 180,       # Oregon (hydro dominant)
    "eu-west-1": 80,        # Ireland (wind dominant)
    "eu-west-2": 210,       # UK (mixed offshore wind)
    "eu-west-3": 55,        # France (nuclear dominant)
    "eu-central-1": 330,    # Frankfurt (coal/gas)
    "eu-north-1": 15,       # Stockholm (hydro/wind)
    "ap-northeast-1": 470,  # Tokyo (gas/coal)
    "ap-southeast-1": 420,  # Singapore (gas)
    "ap-southeast-2": 650,  # Sydney (coal dominant)
    "sa-east-1": 50,        # Sao Paulo (hydro dominant)
    "ca-central-1": 30,     # Montreal (hydro dominant)
    # GCP regions
    "us-central1": 450,     # Iowa (wind growing, still coal)
    "us-west1": 180,        # Oregon (hydro)
    "europe-west1": 330,    # Belgium (nuclear/gas)
    "europe-west4": 380,    # Netherlands (gas/wind)
    "europe-north1": 15,    # Finland (hydro/nuclear/wind)
    "asia-east1": 520,      # Taiwan (coal/gas)
    # Azure regions
    "eastus": 380,          # Virginia
    "westus2": 180,         # Washington (hydro)
    "westeurope": 330,      # Netherlands
    "northeurope": 80,      # Ireland (wind)
    "japaneast": 470,       # Tokyo
    "australiaeast": 650,   # Sydney
}

# M-05: Relative latency offsets between major region pairs.
# Key format: "origin:target" -> milliseconds delta.
_REGION_LATENCY_DELTAS: dict[str, int] = {
    "us-east-1:ap-southeast-1": 170, "ap-southeast-1:us-east-1": -170,
    "us-east-1:eu-west-1": 70, "eu-west-1:us-east-1": -70,
    "us-east-1:ap-northeast-1": 140, "ap-northeast-1:us-east-1": -140,
    "us-east-1:sa-east-1": 120, "sa-east-1:us-east-1": -120,
    "us-east-1:eu-central-1": 80, "eu-central-1:us-east-1": -80,
    "us-east-1:us-west-2": 60, "us-west-2:us-east-1": -60,
}


def compute_relative_latency(origin_region: str, target_region: str) -> float:
    """Compute estimated latency between two regions in milliseconds."""
    if origin_region == target_region:
        return float(_REGION_LATENCY.get(origin_region, 10))
    key = f"{origin_region}:{target_region}"
    if key in _REGION_LATENCY_DELTAS:
        base = _REGION_LATENCY.get(origin_region, 50)
        return float(max(5, base + _REGION_LATENCY_DELTAS[key]))
    base = _REGION_LATENCY.get(origin_region, 50)
    target = _REGION_LATENCY.get(target_region, 50)
    return float(max(5, (base + target) / 2))


class CarbonProvider(Enum):
    """Carbon intensity data source."""
    STATIC = "static"           # Use built-in regional averages
    ELECTRICITYMAP = "electricitymap"  # Live API (electricityMap)
    WATTTIME = "watttime"       # Live API (WattTime)


@dataclass
class CarbonIntensity:
    """Carbon intensity reading for a region."""
    region: str
    gco2_per_kwh: float
    source: str = "static"
    timestamp: float = field(default_factory=time.time)
    renewable_pct: float = 0.0  # Percentage of renewable energy


class CarbonIntensityProvider:
    """Fetches live carbon intensity data from external APIs.

    Supports:
    - electricityMap API (https://api.electricitymap.org)
    - WattTime API (https://api.watttime.org)
    - Static fallback (built-in regional averages)

    Set ELECTRICITYMAP_AUTH_TOKEN or WATTTIME_USERNAME/WATTTIME_PASSWORD
    environment variables to enable live data.
    """

    def __init__(
        self,
        provider: CarbonProvider = CarbonProvider.STATIC,
        cache_ttl: int = 300,  # 5 minutes
    ):
        self._provider = provider
        self._cache_ttl = cache_ttl
        self._cache: dict[str, CarbonIntensity] = {}
        self._last_fetch: dict[str, float] = {}
        # WattTime token cache
        self._watttime_token: str = ""
        self._watttime_token_expiry: float = 0.0
        # Shared HTTP client with connection pooling
        self._http_client: Any = None

    def _get_http_client(self):
        """Get or create a shared httpx.Client with connection pooling."""
        if self._http_client is None:
            try:
                import httpx
                self._http_client = httpx.Client(
                    timeout=10.0,
                    limits=httpx.Limits(
                        max_connections=20,
                        max_keepalive_connections=10,
                        keepalive_expiry=60.0,
                    ),
                )
            except ImportError:
                return None
        return self._http_client

    def close(self) -> None:
        """Close the shared HTTP client."""
        if self._http_client is not None:
            try:
                self._http_client.close()
            except Exception:
                pass
            self._http_client = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self) -> None:
        self.close()

    def get_intensity(self, region: str) -> CarbonIntensity:
        """Get carbon intensity for a region (cached)."""
        now = time.time()
        cached = self._cache.get(region)
        if cached and (now - cached.timestamp) < self._cache_ttl:
            return cached

        if self._provider == CarbonProvider.ELECTRICITYMAP:
            intensity = self._fetch_electricitymap(region)
        elif self._provider == CarbonProvider.WATTTIME:
            intensity = self._fetch_watttime(region)
        else:
            intensity = self._get_static(region)

        self._cache[region] = intensity
        return intensity

    def _get_static(self, region: str) -> CarbonIntensity:
        """Get static carbon intensity from built-in data."""
        gco2 = _REGIONAL_CARBON_INTENSITY.get(region, 400)
        renewable = max(0, min(100, (1 - gco2 / 800) * 100))
        return CarbonIntensity(
            region=region,
            gco2_per_kwh=gco2,
            source="static",
            renewable_pct=renewable,
        )

    def _fetch_electricitymap(self, region: str) -> CarbonIntensity:
        """Fetch live data from electricityMap API."""
        import os
        token = os.environ.get("ELECTRICITYMAP_AUTH_TOKEN", "")
        if not token:
            logger.warning("ELECTRICITYMAP_AUTH_TOKEN not set, falling back to static")
            return self._get_static(region)

        try:
            client = self._get_http_client()
            if client is None:
                return self._get_static(region)
            zone = self._region_to_zone(region)
            if zone is None:
                return self._get_static(region)
            resp = client.get(
                f"https://api.electricitymap.org/v3/carbon-intensity/latest",
                params={"zone": zone},
                headers={"auth-token": token},
            )
            resp.raise_for_status()
            data = resp.json()
            return CarbonIntensity(
                region=region,
                gco2_per_kwh=data.get("carbonIntensity", 400),
                source="electricitymap",
                renewable_pct=data.get("renewablePercentage", 0),
            )
        except Exception as e:
            logger.debug(f"electricityMap fetch failed for {region}: {e}")
            return self._get_static(region)

    def _fetch_watttime(self, region: str) -> CarbonIntensity:
        """Fetch live data from WattTime API.

        Caches the login token for 55 minutes (WattTime tokens expire after 1 hour).
        """
        import os
        username = os.environ.get("WATTTIME_USERNAME", "")
        password = os.environ.get("WATTTIME_PASSWORD", "")
        if not username or not password:
            logger.warning("WATTTIME credentials not set, falling back to static")
            return self._get_static(region)

        try:
            client = self._get_http_client()
            if client is None:
                return self._get_static(region)
            now = time.time()
            # Refresh token if expired or about to expire (within 5 min)
            if not self._watttime_token or now >= self._watttime_token_expiry:
                token_resp = client.get(
                    "https://api.watttime.org/v2/login",
                    auth=(username, password),
                )
                token_resp.raise_for_status()
                self._watttime_token = token_resp.json().get("token", "")
                # WattTime tokens expire after 1 hour; refresh at 55 min
                self._watttime_token_expiry = now + 3300
                logger.debug("WattTime token refreshed")

            zone = self._region_to_zone(region)
            if zone is None:
                return self._get_static(region)
            index_resp = client.get(
                "https://api.watttime.org/v2/index",
                params={"region": zone},
                headers={"Authorization": f"Bearer {self._watttime_token}"},
            )
            index_resp.raise_for_status()
            data = index_resp.json()
            # WattTime returns moer (Marginal Operating Emission Rate) in lbs/MWh
            moer = data.get("moer", 400)
            gco2 = moer * 0.453592  # Convert lbs to kg, then to g
            return CarbonIntensity(
                region=region,
                gco2_per_kwh=gco2,
                source="watttime",
                renewable_pct=0,
            )
        except Exception as e:
            logger.debug(f"WattTime fetch failed for {region}: {e}")
            # Clear token on auth failure to force re-login next time
            if "401" in str(e) or "403" in str(e) or "unauthorized" in str(e).lower():
                self._watttime_token = ""
                self._watttime_token_expiry = 0.0
            return self._get_static(region)

    @staticmethod
    def _region_to_zone(region: str) -> str | None:
        """Map cloud region to electricityMap/WattTime zone code.

        Returns None for unmapped regions so callers can fall back to static data.
        """
        mapping = {
            "us-east-1": "US-VIRG", "us-east-2": "US-MIDA",
            "us-west-1": "US-CAL-CISO", "us-west-2": "US-NW-PACW",
            "eu-west-1": "IE", "eu-west-2": "GB",
            "eu-west-3": "FR", "eu-central-1": "DE",
            "eu-north-1": "SE", "ap-northeast-1": "JP-TK",
            "ap-southeast-1": "SG", "ap-southeast-2": "AU-NSW",
            "sa-east-1": "BR-CS", "ca-central-1": "CA-QC",
            "us-central1": "US-MIDW-MISO", "us-west1": "US-NW-PACW",
            "europe-west1": "BE", "europe-west4": "NL",
            "europe-north1": "FI", "asia-east1": "TW",
            "eastus": "US-VIRG", "westus2": "US-NW-PACW",
            "westeurope": "NL", "northeurope": "IE",
            "japaneast": "JP-TK", "australiaeast": "AU-NSW",
        }
        zone = mapping.get(region)
        if zone is None:
            logger.debug(f"No carbon zone mapping for region '{region}', will use static fallback")
        return zone


@dataclass
class CloudProvider:
    """A cloud provider with its current pricing, latency, and carbon footprint."""
    name: str  # "aws", "gcp", "azure"
    region: str = "us-east-1"
    instance_type: str = ""
    gpu_type: str = ""
    gpu_count: int = 1
    gpu_memory_gb: float = 0.0
    price_per_hour: float = 0.0
    spot_price: float = 0.0
    latency_ms: float = 0.0
    available: bool = True
    spot_available: bool = True


@dataclass
class RouteDecision:
    """A cross-cloud routing decision."""
    provider: str
    instance_type: str
    price_per_hour: float
    estimated_cost: float
    latency_ms: float
    reason: str
    region: str = ""
    gpu_type: str = ""
    gpu_count: int = 1
    gpu_memory_gb: float = 0.0
    carbon_intensity: float = 0.0  # gCO2/kWh
    carbon_cost_factor: float = 1.0  # Multiplier applied due to carbon
    is_spot: bool = False
    alternatives_considered: int = 0


class CrossCloudRouter:
    """Routes requests to the cheapest available GPU across cloud providers.

    Considers:
    - On-demand vs spot pricing
    - Latency to each provider
    - GPU type requirements
    - Availability and reliability
    - **Carbon intensity** (carbon-aware scheduling)

    Usage::

        router = CrossCloudRouter()
        decision = router.select_provider(
            gpu_type="A100",
            max_latency_ms=100,
            prefer_spot=True,
        )
        # decision.provider == "gcp", decision.price_per_hour == 5.07

        # Carbon-aware routing (routes to cleanest grid)
        decision = router.select_provider_carbon_aware(
            gpu_type="A100",
            max_latency_ms=100,
            carbon_weight=0.3,  # 30% carbon, 70% cost
        )
    """

    def __init__(
        self,
        carbon_provider: CarbonProvider = CarbonProvider.STATIC,
        carbon_api_cache_ttl: int = 300,
        pricing_manager: "PricingManager | None" = None,
        expand_regions: bool = True,
        latency_tracker: Any = None,
    ):
        self._providers: list[CloudProvider] = []
        self._latency_cache: dict[str, float] = {}
        self._availability_cache: dict[str, bool] = {}
        self._stats = {"routes": 0, "savings_usd": 0.0, "carbon_saved_kg": 0.0}
        self._carbon_provider = CarbonIntensityProvider(
            provider=carbon_provider,
            cache_ttl=carbon_api_cache_ttl,
        )
        self._pricing_manager = pricing_manager
        self._expand_regions = expand_regions
        self._latency_tracker = latency_tracker  # LatencyTracker instance

        # Pricing freshness tracking
        self._pricing_last_updated: float = 0.0
        self._pricing_stale_warning_issued: bool = False

        # Initialize default providers
        self._init_default_providers()

    def set_latency_tracker(self, tracker: Any) -> None:
        """Attach a LatencyTracker for real latency measurements."""
        self._latency_tracker = tracker

    def sync_latency_from_tracker(self) -> int:
        """Pull measured latencies from LatencyTracker into the router cache.

        Returns the number of latency entries synced.
        """
        if self._latency_tracker is None:
            return 0
        measured = self._latency_tracker.get_all_avg()
        synced = 0
        for node_id, latency_ms in measured.items():
            self._latency_cache[node_id] = latency_ms
            synced += 1
        return synced

    def _init_default_providers(self) -> None:
        """Initialize default cloud provider configurations.

        When expand_regions=True, creates one CloudProvider per region
        per instance type, so carbon-aware routing sees all regions.
        """
        for cloud, instances in _CLOUD_GPU_PRICING.items():
            for instance, price in instances.items():
                spot_discount = _SPOT_DISCOUNT.get(cloud, 0.3)
                gpu_type = _INSTANCE_GPU_TYPE.get(instance, "")
                if self._expand_regions:
                    regions = self._get_cloud_regions(cloud)
                    for region in regions:
                        latency = _REGION_LATENCY.get(region, 50.0)
                        self._providers.append(CloudProvider(
                            name=cloud,
                            region=region,
                            instance_type=instance,
                            gpu_type=gpu_type,
                            price_per_hour=price,
                            spot_price=price * spot_discount,
                            latency_ms=latency,
                        ))
                else:
                    self._providers.append(CloudProvider(
                        name=cloud,
                        instance_type=instance,
                        gpu_type=gpu_type,
                        price_per_hour=price,
                        spot_price=price * spot_discount,
                    ))

    @staticmethod
    def _get_cloud_regions(cloud: str) -> list[str]:
        """Get known regions for a cloud provider."""
        regions = {
            "aws": ["us-east-1", "us-east-2", "us-west-1", "us-west-2",
                     "eu-west-1", "eu-west-2", "eu-central-1", "eu-north-1",
                     "ap-northeast-1", "ap-southeast-1", "ap-southeast-2",
                     "sa-east-1", "ca-central-1"],
            "gcp": ["us-central1", "us-west1", "us-east1",
                     "europe-west1", "europe-west4", "europe-north1",
                     "asia-east1", "asia-northeast1"],
            "azure": ["eastus", "westus2", "westeurope", "northeurope",
                       "japaneast", "australiaeast", "southeastasia"],
        }
        return regions.get(cloud, ["us-east-1"])

    def sync_live_pricing(self) -> int:
        """Sync pricing from PricingManager (live cloud APIs).

        Returns the number of providers updated.
        """
        if not self._pricing_manager:
            return 0
        try:
            live_prices = self._pricing_manager.get_all_pricing()
        except Exception as e:
            logger.warning(f"Failed to sync live pricing: {e}")
            return 0
        if not live_prices:
            return 0
        updated = 0
        for p in live_prices:
            for provider in self._providers:
                if (provider.name == p.provider
                        and provider.instance_type == p.instance_type
                        and provider.region == p.region):
                    provider.price_per_hour = p.on_demand_price
                    provider.spot_price = p.spot_price if p.spot_price > 0 else p.on_demand_price * _SPOT_DISCOUNT.get(p.provider, 0.3)
                    provider.gpu_type = p.gpu_type or provider.gpu_type
                    provider.gpu_count = p.gpu_count or provider.gpu_count
                    provider.gpu_memory_gb = p.gpu_memory_gb or provider.gpu_memory_gb
                    updated += 1
        if updated > 0:
            self._pricing_last_updated = time.time()
            self._pricing_stale_warning_issued = False
        logger.info(f"Synced {updated} providers with live pricing data")
        return updated

    @property
    def pricing_age_hours(self) -> float:
        """Hours since pricing was last updated from live APIs."""
        if self._pricing_last_updated <= 0:
            return float("inf")
        return (time.time() - self._pricing_last_updated) / 3600

    def _check_pricing_staleness(self) -> None:
        """Log a warning if pricing data is stale (>24 hours old)."""
        age_h = self.pricing_age_hours
        if age_h > 24 and not self._pricing_stale_warning_issued:
            logger.warning(
                f"Pricing data is {age_h:.0f}h old (last update: "
                f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(self._pricing_last_updated))}). "
                f"Call sync_live_pricing() or set up PricingManager with auto-refresh."
            )
            self._pricing_stale_warning_issued = True

    def add_provider(self, provider: CloudProvider) -> None:
        """Add a custom cloud provider configuration."""
        self._providers.append(provider)

    def update_latency(self, provider: str, latency_ms: float) -> None:
        """Update latency measurement for a provider."""
        self._latency_cache[provider] = latency_ms

    def update_availability(self, provider: str, instance_type: str, available: bool, region: str = "") -> None:
        """Update availability for a provider instance type."""
        if region:
            self._availability_cache[f"{provider}:{instance_type}:{region}"] = available
        else:
            # Update all regions for this instance
            for key in list(self._availability_cache.keys()):
                if key.startswith(f"{provider}:{instance_type}:"):
                    self._availability_cache[key] = available
            self._availability_cache[f"{provider}:{instance_type}"] = available

    def update_carbon_intensity(self, region: str, gco2_per_kwh: float) -> None:
        """Manually override carbon intensity for a region."""
        from dataclasses import replace
        self._carbon_provider._cache[region] = CarbonIntensity(
            region=region,
            gco2_per_kwh=gco2_per_kwh,
            source="manual",
        )

    def select_provider(
        self,
        gpu_type: str = "",
        max_latency_ms: float = 200.0,
        max_price: float = float("inf"),
        prefer_spot: bool = True,
        min_gpu_memory_gb: float = 0.0,
    ) -> RouteDecision | None:
        """Select the cheapest provider that meets requirements.

        Args:
            gpu_type: Required GPU type (e.g., "A100", "V100"). Empty = any.
            max_latency_ms: Maximum acceptable latency.
            max_price: Maximum price per hour.
            prefer_spot: Prefer spot instances when available.
            min_gpu_memory_gb: Minimum GPU memory requirement.

        Returns:
            RouteDecision with the best provider, or None if none available.
        """
        self._stats["routes"] += 1
        self._check_pricing_staleness()
        self.sync_latency_from_tracker()
        candidates = []

        for provider in self._providers:
            # Check GPU type filter
            if gpu_type and not self._matches_gpu_type(provider, gpu_type):
                continue

            # Check GPU memory filter
            if min_gpu_memory_gb > 0 and provider.gpu_memory_gb > 0:
                if provider.gpu_memory_gb < min_gpu_memory_gb:
                    continue

            # Check availability
            avail_key = f"{provider.name}:{provider.instance_type}:{provider.region}"
            if not self._availability_cache.get(avail_key, provider.available):
                continue

            # Check latency (use per-region latency, then provider cache, then default)
            latency = self._latency_cache.get(
                f"{provider.name}:{provider.region}",
                self._latency_cache.get(provider.name, provider.latency_ms or 50.0),
            )
            if latency > max_latency_ms:
                continue

            # Check price
            price = provider.spot_price if (prefer_spot and provider.spot_available) else provider.price_per_hour
            if price > max_price:
                continue

            candidates.append((provider, price, latency))

        if not candidates:
            return None

        # Sort by price, then latency
        candidates.sort(key=lambda x: (x[1], x[2]))
        best_provider, best_price, best_latency = candidates[0]

        # Calculate savings vs most expensive option
        worst_price = max(c[1] for c in candidates)
        savings = worst_price - best_price

        return RouteDecision(
            provider=best_provider.name,
            instance_type=best_provider.instance_type,
            price_per_hour=best_price,
            estimated_cost=best_price,  # Per hour
            latency_ms=best_latency,
            region=best_provider.region,
            gpu_type=best_provider.gpu_type or gpu_type,
            gpu_count=best_provider.gpu_count,
            gpu_memory_gb=best_provider.gpu_memory_gb,
            is_spot=prefer_spot and best_provider.spot_available,
            alternatives_considered=len(candidates),
            reason=f"Cheapest {best_provider.name} {best_provider.instance_type} "
                   f"in {best_provider.region} (${best_price:.2f}/hr, {best_latency:.0f}ms)",
        )

    @staticmethod
    def _matches_gpu_type(provider: CloudProvider, gpu_type: str) -> bool:
        """Check if a provider matches a requested GPU type."""
        if provider.gpu_type:
            return provider.gpu_type.upper() == gpu_type.upper()
        return gpu_type.upper() in provider.instance_type.upper()

    def select_provider_carbon_aware(
        self,
        gpu_type: str = "",
        max_latency_ms: float = 200.0,
        max_price: float = float("inf"),
        prefer_spot: bool = True,
        carbon_weight: float = 0.3,
        max_carbon_intensity: float = float("inf"),
        min_gpu_memory_gb: float = 0.0,
    ) -> RouteDecision | None:
        """Select provider balancing cost, latency, and carbon intensity.

        This is unique in open-source LLM serving — no other framework
        routes inference based on grid carbon intensity.

        Args:
            gpu_type: Required GPU type.
            max_latency_ms: Maximum acceptable latency.
            max_price: Maximum price per hour.
            prefer_spot: Prefer spot instances.
            carbon_weight: Weight for carbon in scoring (0-1).
                          0 = pure cost, 1 = pure carbon.
            max_carbon_intensity: Maximum gCO2/kWh allowed.
            min_gpu_memory_gb: Minimum GPU memory requirement.

        Returns:
            RouteDecision with carbon-aware scoring, or None.
        """
        self._stats["routes"] += 1
        self._check_pricing_staleness()
        self.sync_latency_from_tracker()
        candidates = []

        for provider in self._providers:
            # Check GPU type filter
            if gpu_type and not self._matches_gpu_type(provider, gpu_type):
                continue

            # Check GPU memory filter
            if min_gpu_memory_gb > 0 and provider.gpu_memory_gb > 0:
                if provider.gpu_memory_gb < min_gpu_memory_gb:
                    continue

            avail_key = f"{provider.name}:{provider.instance_type}:{provider.region}"
            if not self._availability_cache.get(avail_key, provider.available):
                continue

            latency = self._latency_cache.get(
                f"{provider.name}:{provider.region}",
                self._latency_cache.get(provider.name, provider.latency_ms or 50.0),
            )
            if latency > max_latency_ms:
                continue

            price = provider.spot_price if (prefer_spot and provider.spot_available) else provider.price_per_hour
            if price > max_price:
                continue

            region = provider.region
            carbon = self._carbon_provider.get_intensity(region)
            if carbon.gco2_per_kwh > max_carbon_intensity:
                continue

            candidates.append((provider, price, latency, carbon))

        if not candidates:
            return None

        # Normalize cost and carbon for scoring
        prices = [c[1] for c in candidates]
        carbons = [c[3].gco2_per_kwh for c in candidates]
        min_price, max_price_val = min(prices), max(prices)
        min_carbon, max_carbon = min(carbons), max(carbons)
        price_range = max_price_val - min_price if max_price_val > min_price else 1.0
        carbon_range = max_carbon - min_carbon if max_carbon > min_carbon else 1.0

        def score(candidate):
            _, price, latency, carbon = candidate
            norm_price = (price - min_price) / price_range
            norm_carbon = (carbon.gco2_per_kwh - min_carbon) / carbon_range
            norm_latency = latency / max_latency_ms if max_latency_ms > 0 else 0
            return (1 - carbon_weight) * (norm_price * 0.7 + norm_latency * 0.3) + carbon_weight * norm_carbon

        candidates.sort(key=score)
        best_provider, best_price, best_latency, best_carbon = candidates[0]

        # Track carbon savings
        worst_carbon = max(c[3].gco2_per_kwh for c in candidates)
        if worst_carbon > best_carbon.gco2_per_kwh:
            self._stats["carbon_saved_kg"] += (worst_carbon - best_carbon.gco2_per_kwh) * 0.001

        # Derive carbon cost factor from the actual range of carbon intensities
        # rather than a hardcoded divisor. This scales the factor to [1.0, 1+weight].
        max_carbon_val = max(c[3].gco2_per_kwh for c in candidates)
        carbon_divisor = max(max_carbon_val, 1.0)

        return RouteDecision(
            provider=best_provider.name,
            instance_type=best_provider.instance_type,
            price_per_hour=best_price,
            estimated_cost=best_price,
            latency_ms=best_latency,
            region=best_provider.region,
            gpu_type=best_provider.gpu_type or gpu_type,
            gpu_count=best_provider.gpu_count,
            gpu_memory_gb=best_provider.gpu_memory_gb,
            carbon_intensity=best_carbon.gco2_per_kwh,
            carbon_cost_factor=1.0 + carbon_weight * (best_carbon.gco2_per_kwh / carbon_divisor),
            is_spot=prefer_spot and best_provider.spot_available,
            alternatives_considered=len(candidates),
            reason=(
                f"Carbon-aware: {best_provider.name} {best_provider.instance_type} "
                f"in {best_carbon.region} "
                f"({best_carbon.gco2_per_kwh:.0f} gCO2/kWh, {best_carbon.source})"
            ),
        )

    def get_carbon_report(self) -> list[dict]:
        """Get carbon intensity report for all provider+region combinations."""
        seen: set[tuple[str, str]] = set()
        report = []
        for provider in self._providers:
            key = (provider.name, provider.region)
            if key in seen:
                continue
            seen.add(key)
            carbon = self._carbon_provider.get_intensity(provider.region)
            report.append({
                "provider": provider.name,
                "region": provider.region,
                "instance_type": provider.instance_type,
                "gpu_type": provider.gpu_type,
                "gco2_per_kwh": carbon.gco2_per_kwh,
                "renewable_pct": carbon.renewable_pct,
                "source": carbon.source,
            })
        return sorted(report, key=lambda x: x["gco2_per_kwh"])

    def get_all_prices(self, gpu_type: str = "") -> list[dict]:
        """Get pricing for all available providers."""
        results = []
        for provider in self._providers:
            if gpu_type and not self._matches_gpu_type(provider, gpu_type):
                continue
            latency = self._latency_cache.get(
                f"{provider.name}:{provider.region}",
                self._latency_cache.get(provider.name, provider.latency_ms or 50.0),
            )
            avail_key = f"{provider.name}:{provider.instance_type}:{provider.region}"
            available = self._availability_cache.get(avail_key, provider.available)
            carbon = self._carbon_provider.get_intensity(provider.region)
            results.append({
                "provider": provider.name,
                "instance_type": provider.instance_type,
                "region": provider.region,
                "gpu_type": provider.gpu_type,
                "gpu_count": provider.gpu_count,
                "gpu_memory_gb": provider.gpu_memory_gb,
                "price_per_hour": provider.price_per_hour,
                "spot_price": provider.spot_price,
                "latency_ms": latency,
                "available": available,
                "carbon_gco2_kwh": carbon.gco2_per_kwh,
                "renewable_pct": carbon.renewable_pct,
            })
        return sorted(results, key=lambda x: x["spot_price"])

    def estimate_cost(self, provider: str, duration_hours: float, use_spot: bool = True, region: str = "") -> float:
        """Estimate cost for a given duration.

        Returns 0.0 if no matching provider is found (with a warning logged).
        """
        for p in self._providers:
            if p.name == provider:
                if region and p.region != region:
                    continue
                price = p.spot_price if use_spot else p.price_per_hour
                return price * duration_hours
        logger.warning(
            f"No provider found for estimate_cost(provider={provider!r}, region={region!r}). "
            f"Known providers: {sorted(set(p.name for p in self._providers))}"
        )
        return 0.0

    @property
    def stats(self) -> dict:
        return dict(self._stats)
