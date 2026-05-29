"""Azure Cloud Provider SDK Wrapper.

Virtual Machine SKU info, spot pricing, and reserved instance rates.
Uses the public Azure Retail Prices REST API (no auth required).
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from distllm.cloud.common import (
    AvailabilityChecker,
    AvailabilityInfo,
    GPUSpec,
    InstanceSpec,
    PriceQuote,
    PricingFetcher,
)

_AZURE_GPU_INSTANCES: dict[str, tuple[str, int, float, int]] = {
    "Standard_NC24ads_A100_v4": ("A100", 1, 80.0, 24),
    "Standard_NC48ads_A100_v4": ("A100", 2, 80.0, 48),
    "Standard_NC96ads_A100_v4": ("A100", 4, 80.0, 96),
    "Standard_ND96asr_v4": ("A100", 8, 40.0, 96),
    "Standard_ND96amsr_A100_v4": ("A100", 8, 80.0, 96),
    "Standard_NV6ads_A10_v5": ("A10", 1, 24.0, 6),
    "Standard_NV12ads_A10_v5": ("A10", 1, 24.0, 12),
    "Standard_NV36ads_A10_v5": ("A10", 1, 24.0, 36),
    "Standard_NV72ads_A10_v5": ("A10", 2, 24.0, 72),
}

_AZURE_REGIONS = [
    "eastus", "eastus2", "westus", "westus2", "westus3",
    "centralus", "southcentralus", "northcentralus",
    "westeurope", "northeurope", "uksouth", "ukwest",
    "japaneast", "japanwest", "australiaeast", "australiasoutheast",
    "southeastasia", "eastasia",
]


class AzurePricingFetcher(PricingFetcher):
    """Fetches GPU pricing from Azure Retail Prices REST API."""

    def provider_name(self) -> str:
        return "azure"

    def fetch_gpu_pricing(self, regions: list[str] | None = None) -> list[PriceQuote]:
        target_regions = regions or _AZURE_REGIONS
        result: list[PriceQuote] = []
        for region in target_regions:
            try:
                result.extend(self._fetch_region(region))
            except Exception as e:
                logger.debug(f"Azure pricing fetch failed for {region}: {e}")
        return result if result else self._fallback(target_regions)

    def _fetch_region(self, region: str) -> list[PriceQuote]:
        import httpx
        filters = (
            f"serviceName eq 'Virtual Machines' "
            f"and armRegionName eq '{region}' "
            f"and priceType eq 'Consumption'"
        )
        resp = httpx.get(
            "https://prices.azure.com/api/retail/prices",
            params={"$filter": filters, "$top": 200},
            timeout=15.0,
        )
        resp.raise_for_status()
        result: list[PriceQuote] = []
        spot_prices: dict[str, float] = {}
        od_prices: dict[str, float] = {}

        for item in resp.json().get("Items", []):
            sku = item.get("armSkuName", "")
            if sku not in _AZURE_GPU_INSTANCES:
                continue
            price = item.get("retailPrice", 0.0)
            is_spot = item.get("type", "").endswith("Spot")
            if is_spot:
                spot_prices[sku] = price
            else:
                od_prices[sku] = price

        for inst in set(list(spot_prices.keys()) + list(od_prices.keys())):
            result.append(PriceQuote(
                provider="azure",
                instance_type=inst,
                region=region,
                on_demand_hourly=od_prices.get(inst, 0.0),
                spot_hourly=spot_prices.get(inst, 0.0),
            ))
        return result

    def _fallback(self, regions: list[str]) -> list[PriceQuote]:
        static = {
            "Standard_NC24ads_A100_v4": (3.67, 1.28),
            "Standard_NC48ads_A100_v4": (7.35, 2.57),
            "Standard_NC96ads_A100_v4": (14.69, 5.14),
            "Standard_NV6ads_A10_v5": (1.10, 0.39),
        }
        result = []
        for inst, (od, spot) in static.items():
            for region in regions[:3]:
                result.append(PriceQuote(
                    provider="azure", instance_type=inst, region=region,
                    on_demand_hourly=od, spot_hourly=spot,
                ))
        return result


class AzureAvailabilityChecker(AvailabilityChecker):
    """Checks Azure GPU instance availability."""

    def check_availability(self, instance_type: str, region: str) -> AvailabilityInfo:
        try:
            import httpx
            resp = httpx.get(
                "https://management.azure.com/subscriptions/{subscriptionId}/providers/Microsoft.Compute/locations/{region}/vmSizes",
                params={"api-version": "2023-07-01"},
                timeout=10.0,
            )
            resp.raise_for_status()
            available = any(
                vm.get("name") == instance_type
                for vm in resp.json().get("value", [])
            )
            return AvailabilityInfo(
                provider="azure", instance_type=instance_type, region=region,
                available=available,
            )
        except Exception:
            return AvailabilityInfo(
                provider="azure", instance_type=instance_type, region=region,
                available=instance_type in _AZURE_GPU_INSTANCES,
            )

    def list_gpu_instances(self, region: str | None = None) -> list[InstanceSpec]:
        specs = []
        for inst, (gpu, count, mem, vcpus) in _AZURE_GPU_INSTANCES.items():
            specs.append(InstanceSpec(
                provider="azure", instance_type=inst, region=region or "eastus",
                gpu=GPUSpec(name=gpu, memory_gb=mem), gpu_count=count, vcpus=vcpus,
            ))
        return specs
