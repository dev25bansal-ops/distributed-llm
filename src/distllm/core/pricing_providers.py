"""Live Cloud Pricing API Providers.

Fetches real-time GPU instance pricing from AWS, GCP, and Azure APIs.
Falls back to static data when APIs are unavailable.

Providers:
- AWS Price List API (https://pricing.us-east-1.amazonaws.com)
- GCP Cloud Billing Catalog API (https://cloudbilling.googleapis.com)
- Azure Retail Prices REST API (https://prices.azure.com)

Environment variables:
  AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY  — for AWS pricing
  GOOGLE_APPLICATION_CREDENTIALS               — for GCP pricing
  AZURE_SUBSCRIPTION_ID                        — for Azure pricing
"""

from __future__ import annotations

import json
import os
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class InstancePricing:
    """Pricing for a single GPU instance type."""
    provider: str
    instance_type: str
    region: str
    gpu_type: str = ""
    gpu_count: int = 1
    gpu_memory_gb: float = 0.0
    on_demand_price: float = 0.0
    spot_price: float = 0.0
    currency: str = "USD"
    source: str = "static"
    last_updated: float = field(default_factory=time.time)

    @property
    def spot_discount_pct(self) -> float:
        if self.on_demand_price <= 0:
            return 0.0
        return (1 - self.spot_price / self.on_demand_price) * 100


# Instance type → GPU type/memory mapping
_INSTANCE_GPU_MAP: dict[str, tuple[str, int, float]] = {
    # AWS
    "p4d.24xlarge": ("A100", 8, 80.0),
    "p4de.24xlarge": ("A100", 8, 80.0),
    "p3.2xlarge": ("V100", 1, 16.0),
    "p3.8xlarge": ("V100", 4, 16.0),
    "p3.16xlarge": ("V100", 8, 16.0),
    "g5.xlarge": ("A10G", 1, 24.0),
    "g5.2xlarge": ("A10G", 1, 24.0),
    "g5.4xlarge": ("A10G", 1, 24.0),
    "g5.12xlarge": ("A10G", 4, 24.0),
    "g6.xlarge": ("L4", 1, 24.0),
    "g6.2xlarge": ("L4", 1, 24.0),
    # GCP
    "a2-highgpu-1g": ("A100", 1, 40.0),
    "a2-highgpu-2g": ("A100", 2, 40.0),
    "a2-highgpu-4g": ("A100", 4, 40.0),
    "a2-highgpu-8g": ("A100", 8, 40.0),
    "a2-ultragpu-1g": ("A100", 1, 80.0),
    "a2-ultragpu-2g": ("A100", 2, 80.0),
    "a2-ultragpu-4g": ("A100", 4, 80.0),
    "a2-ultragpu-8g": ("A100", 8, 80.0),
    "g2-standard-4": ("L4", 1, 24.0),
    "g2-standard-8": ("L4", 1, 24.0),
    "g2-standard-12": ("L4", 1, 24.0),
    "n1-standard-8-t4": ("T4", 1, 16.0),
    # Azure
    "Standard_NC24ads_A100_v4": ("A100", 1, 80.0),
    "Standard_NC48ads_A100_v4": ("A100", 2, 80.0),
    "Standard_NC96ads_A100_v4": ("A100", 4, 80.0),
    "Standard_ND96asr_v4": ("A100", 8, 40.0),
    "Standard_NV6ads_A10_v5": ("A10", 1, 24.0),
    "Standard_NV12ads_A10_v5": ("A10", 1, 24.0),
    "Standard_NV36ads_A10_v5": ("A10", 1, 24.0),
}


def lookup_gpu_info(instance_type: str) -> tuple[str, int, float]:
    """Look up GPU type, count, and memory for an instance type.

    Returns:
        (gpu_type, gpu_count, gpu_memory_gb) — ("", 0, 0.0) if unknown.
    """
    return _INSTANCE_GPU_MAP.get(instance_type, ("", 0, 0.0))


class PricingProvider(ABC):
    """Abstract base class for cloud pricing providers."""

    @abstractmethod
    def fetch_pricing(self, regions: list[str] | None = None) -> list[InstancePricing]:
        """Fetch current pricing for GPU instances.

        Args:
            regions: Optional list of regions to fetch. None = all known regions.

        Returns:
            List of InstancePricing for available instances.
        """

    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name (aws, gcp, azure)."""


class AWSPricingProvider(PricingProvider):
    """Fetches GPU pricing from the AWS Price List API.

    Uses the bulk pricing JSON endpoint for EC2 GPU instances.
    Falls back to static data on failure.
    """

    def provider_name(self) -> str:
        return "aws"

    def fetch_pricing(self, regions: list[str] | None = None) -> list[InstancePricing]:
        try:
            import httpx
            resp = httpx.get(
                "https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonEC2/current/region_index.json",
                timeout=15.0,
            )
            resp.raise_for_status()
            region_index = resp.json()
            target_regions = regions or list(region_index.get("regions", {}).keys())
            result: list[InstancePricing] = []
            for region in target_regions:
                region_info = region_index.get("regions", {}).get(region, {})
                if not region_info:
                    continue
                current_url = region_info.get("currentVersionUrl", "")
                if not current_url:
                    continue
                result.extend(self._fetch_region_pricing(region, current_url))
            if result:
                return result
        except Exception as e:
            logger.debug(f"AWS pricing API failed: {e}")
        return self._fallback()

    def _fetch_region_pricing(self, region: str, path: str) -> list[InstancePricing]:
        import httpx
        try:
            url = f"https://pricing.us-east-1.amazonaws.com{path}"
            resp = httpx.get(url, timeout=30.0)
            resp.raise_for_status()
            data = resp.json()
            result = []
            for sku, product in data.get("products", {}).items():
                attrs = product.get("attributes", {})
                inst_type = attrs.get("instanceType", "")
                if inst_type not in _INSTANCE_GPU_MAP:
                    continue
                gpu_info = _INSTANCE_GPU_MAP[inst_type]
                terms = data.get("terms", {}).get("OnDemand", {}).get(sku, {})
                on_demand = 0.0
                for term in terms.values():
                    for pd in term.get("priceDimensions", {}).values():
                        usd = pd.get("pricePerUnit", {}).get("USD", "0")
                        try:
                            on_demand = float(usd)
                        except ValueError:
                            pass
                        break
                    if on_demand > 0:
                        break
                spot_terms = data.get("terms", {}).get("Spot", {}).get(sku, {})
                spot = 0.0
                for term in spot_terms.values():
                    for pd in term.get("priceDimensions", {}).values():
                        usd = pd.get("pricePerUnit", {}).get("USD", "0")
                        try:
                            spot = float(usd)
                        except ValueError:
                            pass
                        break
                    if spot > 0:
                        break
                if on_demand > 0:
                    result.append(InstancePricing(
                        provider="aws",
                        instance_type=inst_type,
                        region=region,
                        gpu_type=gpu_info[0],
                        gpu_count=gpu_info[1],
                        gpu_memory_gb=gpu_info[2],
                        on_demand_price=on_demand,
                        spot_price=spot if spot > 0 else on_demand * 0.3,
                        source="aws_api",
                    ))
            return result
        except Exception as e:
            logger.debug(f"AWS region {region} pricing fetch failed: {e}")
            return []

    def _fallback(self) -> list[InstancePricing]:
        """Return static fallback pricing."""
        static = {
            "p4d.24xlarge": 32.77, "p3.2xlarge": 3.06, "g5.xlarge": 1.006,
            "g5.2xlarge": 1.212, "g6.xlarge": 0.8054, "p4de.24xlarge": 40.97,
        }
        spot_static = {
            "p4d.24xlarge": 14.40, "p3.2xlarge": 3.83, "g5.xlarge": 1.41,
            "g5.2xlarge": 1.82, "g6.xlarge": 1.10, "p4de.24xlarge": 27.20,
        }
        result = []
        for inst, on_demand in static.items():
            gpu_info = _INSTANCE_GPU_MAP.get(inst, ("", 1, 0.0))
            result.append(InstancePricing(
                provider="aws",
                instance_type=inst,
                region="us-east-1",
                gpu_type=gpu_info[0],
                gpu_count=gpu_info[1],
                gpu_memory_gb=gpu_info[2],
                on_demand_price=on_demand,
                spot_price=spot_static.get(inst, on_demand * 0.3),
                source="static",
            ))
        return result


class GCPPricingProvider(PricingProvider):
    """Fetches GPU pricing from the GCP Cloud Billing Catalog API.

    Requires GOOGLE_APPLICATION_CREDENTIALS or falls back to static data.
    """

    def provider_name(self) -> str:
        return "gcp"

    def fetch_pricing(self, regions: list[str] | None = None) -> list[InstancePricing]:
        cred_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
        if cred_path:
            try:
                return self._fetch_via_api(regions)
            except Exception as e:
                logger.debug(f"GCP pricing API failed: {e}")
        return self._fallback()

    def _fetch_via_api(self, regions: list[str] | None = None) -> list[InstancePricing]:
        import httpx
        token = self._get_access_token()
        if not token:
            return self._fallback()
        resp = httpx.get(
            "https://cloudbilling.googleapis.com/v1/services/6F81-5844-456A/skus",
            headers={"Authorization": f"Bearer {token}"},
            params={"pageSize": 500},
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()
        result: list[InstancePricing] = []
        for sku in data.get("skus", []):
            desc = sku.get("description", "").lower()
            if not any(gpu in desc for gpu in ("gpu", "a100", "v100", "t4", "l4", "a10")):
                continue
            inst_type = self._extract_instance_type(desc)
            if not inst_type or inst_type not in _INSTANCE_GPU_MAP:
                continue
            gpu_info = _INSTANCE_GPU_MAP[inst_type]
            pricing = sku.get("pricingInfo", [{}])
            if not pricing:
                continue
            tiers = pricing[0].get("pricingExpression", {}).get("tieredRates", [])
            if not tiers:
                continue
            unit_price = tiers[0].get("unitPrice", {})
            nanos = unit_price.get("nanos", 0)
            units = int(unit_price.get("units", 0))
            price_per_hour = units + nanos / 1e9
            for region_info in sku.get("serviceRegions", []):
                if regions and region_info not in regions:
                    continue
                result.append(InstancePricing(
                    provider="gcp",
                    instance_type=inst_type,
                    region=region_info,
                    gpu_type=gpu_info[0],
                    gpu_count=gpu_info[1],
                    gpu_memory_gb=gpu_info[2],
                    on_demand_price=price_per_hour,
                    spot_price=price_per_hour * 0.25,
                    source="gcp_api",
                ))
        return result if result else self._fallback()

    def _get_access_token(self) -> str:
        try:
            import httpx
            cred_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
            if not cred_path:
                return ""
            with open(cred_path) as f:
                creds = json.load(f)
            resp = httpx.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                    "assertion": self._make_jwt(creds),
                },
                timeout=10.0,
            )
            resp.raise_for_status()
            return resp.json().get("access_token", "")
        except Exception:
            return ""

    def _make_jwt(self, creds: dict) -> str:
        import base64
        import hashlib
        import hmac
        header = base64.urlsafe_b64encode(json.dumps({"alg": "RS256", "typ": "JWT"}).encode()).rstrip(b"=")
        now = int(time.time())
        claims = {
            "iss": creds.get("client_email", ""),
            "scope": "https://www.googleapis.com/auth/cloud-billing.readonly",
            "aud": creds.get("token_uri", ""),
            "iat": now,
            "exp": now + 3600,
        }
        payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).rstrip(b"=")
        message = header + b"." + payload
        private_key = creds.get("private_key", "")
        if not private_key:
            return ""
        try:
            from cryptography.hazmat.primitives import hashes, serialization
            from cryptography.hazmat.primitives.asymmetric import padding
            key = serialization.load_pem_private_key(private_key.encode(), password=None)
            sig = key.sign(message, padding.PKCS1v15(), hashes.SHA256())
            signature = base64.urlsafe_b64encode(sig).rstrip(b"=")
            return (message + b"." + signature).decode()
        except Exception:
            return ""

    def _extract_instance_type(self, description: str) -> str:
        for inst_type in _INSTANCE_GPU_MAP:
            if inst_type.lower() in description.lower():
                return inst_type
        return ""

    def _fallback(self) -> list[InstancePricing]:
        static = {
            "a2-highgpu-1g": (3.67, 0.92), "a2-highgpu-2g": (7.35, 1.84),
            "a2-ultragpu-1g": (5.07, 1.27), "g2-standard-4": (0.84, 0.21),
            "n1-standard-8-t4": (0.95, 0.24),
        }
        result = []
        for inst, (on_demand, spot) in static.items():
            gpu_info = _INSTANCE_GPU_MAP.get(inst, ("", 1, 0.0))
            result.append(InstancePricing(
                provider="gcp",
                instance_type=inst,
                region="us-central1",
                gpu_type=gpu_info[0],
                gpu_count=gpu_info[1],
                gpu_memory_gb=gpu_info[2],
                on_demand_price=on_demand,
                spot_price=spot,
                source="static",
            ))
        return result


class AzurePricingProvider(PricingProvider):
    """Fetches GPU pricing from the Azure Retail Prices REST API.

    Uses the public prices.azure.com endpoint (no auth required).
    """

    def provider_name(self) -> str:
        return "azure"

    def fetch_pricing(self, regions: list[str] | None = None) -> list[InstancePricing]:
        target_regions = regions or ["eastus", "westus2", "westeurope", "northeurope", "japaneast", "australiaeast"]
        result: list[InstancePricing] = []
        for region in target_regions:
            try:
                result.extend(self._fetch_region(region))
            except Exception as e:
                logger.debug(f"Azure pricing fetch failed for {region}: {e}")
        if not result:
            return self._fallback()
        return result

    def _fetch_region(self, region: str) -> list[InstancePricing]:
        import httpx
        filters = (
            f"serviceName eq 'Virtual Machines' "
            f"and armRegionName eq '{region}' "
            f"and contains(armSkuName, 'NC') "
            f"and priceType eq 'Consumption'"
        )
        resp = httpx.get(
            "https://prices.azure.com/api/retail/prices",
            params={"$filter": filters, "$top": 100},
            timeout=15.0,
        )
        resp.raise_for_status()
        data = resp.json()
        result = []
        for item in data.get("Items", []):
            sku = item.get("armSkuName", "")
            if sku not in _INSTANCE_GPU_MAP:
                continue
            gpu_info = _INSTANCE_GPU_MAP[sku]
            is_spot = item.get("type", "").endswith("Spot")
            price = item.get("retailPrice", 0.0)
            if is_spot:
                existing = [r for r in result if r.instance_type == sku and r.region == region]
                if existing:
                    existing[0].spot_price = price
                    continue
            result.append(InstancePricing(
                provider="azure",
                instance_type=sku,
                region=region,
                gpu_type=gpu_info[0],
                gpu_count=gpu_info[1],
                gpu_memory_gb=gpu_info[2],
                on_demand_price=price if not is_spot else 0.0,
                spot_price=price if is_spot else 0.0,
                source="azure_api",
            ))
        return result

    def _fallback(self) -> list[InstancePricing]:
        static = {
            "Standard_NC24ads_A100_v4": (3.67, 1.28), "Standard_NC48ads_A100_v4": (7.35, 2.57),
            "Standard_NC96ads_A100_v4": (14.69, 5.14), "Standard_NV6ads_A10_v5": (1.10, 0.39),
        }
        result = []
        for inst, (on_demand, spot) in static.items():
            gpu_info = _INSTANCE_GPU_MAP.get(inst, ("", 1, 0.0))
            result.append(InstancePricing(
                provider="azure",
                instance_type=inst,
                region="eastus",
                gpu_type=gpu_info[0],
                gpu_count=gpu_info[1],
                gpu_memory_gb=gpu_info[2],
                on_demand_price=on_demand,
                spot_price=spot,
                source="static",
            ))
        return result


class PricingManager:
    """Manages pricing data from all providers with caching and refresh.

    Usage::

        manager = PricingManager(cache_ttl=3600)
        prices = manager.get_all_pricing()  # fetches from APIs or cache
        manager.refresh()  # force refresh
    """

    def __init__(self, cache_ttl: int = 3600):
        self._providers: list[PricingProvider] = [
            AWSPricingProvider(),
            GCPPricingProvider(),
            AzurePricingProvider(),
        ]
        self._cache_ttl = cache_ttl
        self._cache: list[InstancePricing] = []
        self._last_refresh: float = 0.0
        self._lock = threading.Lock()
        self._refresh_thread: threading.Thread | None = None

    def add_provider(self, provider: PricingProvider) -> None:
        """Add a custom pricing provider."""
        self._providers.append(provider)

    def get_all_pricing(self, force_refresh: bool = False) -> list[InstancePricing]:
        """Get pricing from all providers, using cache if fresh."""
        now = time.time()
        if force_refresh or not self._cache or (now - self._last_refresh) > self._cache_ttl:
            self.refresh()
        return list(self._cache)

    def get_provider_pricing(self, provider_name: str) -> list[InstancePricing]:
        """Get pricing for a specific provider."""
        return [p for p in self.get_all_pricing() if p.provider == provider_name]

    def get_cheapest(
        self,
        gpu_type: str = "",
        min_gpu_memory_gb: float = 0.0,
        prefer_spot: bool = True,
    ) -> InstancePricing | None:
        """Get the cheapest instance matching requirements."""
        candidates = []
        for p in self.get_all_pricing():
            if gpu_type and p.gpu_type != gpu_type:
                continue
            if min_gpu_memory_gb > 0 and p.gpu_memory_gb < min_gpu_memory_gb:
                continue
            price = p.spot_price if prefer_spot and p.spot_price > 0 else p.on_demand_price
            if price <= 0:
                continue
            candidates.append((p, price))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[1])
        return candidates[0][0]

    def refresh(self) -> None:
        """Force refresh pricing from all providers."""
        with self._lock:
            all_prices: list[InstancePricing] = []
            for provider in self._providers:
                try:
                    prices = provider.fetch_pricing()
                    all_prices.extend(prices)
                    logger.debug(f"Fetched {len(prices)} prices from {provider.provider_name()}")
                except Exception as e:
                    logger.warning(f"Failed to fetch pricing from {provider.provider_name()}: {e}")
            if all_prices:
                self._cache = all_prices
                self._last_refresh = time.time()
            logger.info(f"Pricing refreshed: {len(self._cache)} instances across {len(self._providers)} providers")

    def start_background_refresh(self, interval_s: float = 3600) -> None:
        """Start background pricing refresh thread."""
        if self._refresh_thread and self._refresh_thread.is_alive():
            return
        self._refresh_thread = threading.Thread(
            target=self._background_refresh_loop,
            args=(interval_s,),
            daemon=True,
            name="pricing-refresh",
        )
        self._refresh_thread.start()
        logger.info(f"Background pricing refresh started (every {interval_s}s)")

    def _background_refresh_loop(self, interval_s: float) -> None:
        while True:
            time.sleep(interval_s)
            try:
                self.refresh()
            except Exception as e:
                logger.error(f"Background pricing refresh failed: {e}")

    @property
    def last_refresh(self) -> float:
        return self._last_refresh

    @property
    def is_stale(self) -> bool:
        return (time.time() - self._last_refresh) > self._cache_ttl
