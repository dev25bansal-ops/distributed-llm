"""AWS Cloud Provider SDK Wrapper.

EC2 DescribeInstanceTypes, Spot Price history, and Pricing API client.
Uses boto3 when available, falls back to httpx-based REST calls.
"""

from __future__ import annotations

import os
import time
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

# GPU instance type → GPU spec mapping
_AWS_GPU_INSTANCES: dict[str, tuple[str, int, float, int]] = {
    # instance_type: (gpu_name, gpu_count, memory_gb, vcpus)
    "p4d.24xlarge": ("A100", 8, 80.0, 96),
    "p4de.24xlarge": ("A100", 8, 80.0, 96),
    "p3.2xlarge": ("V100", 1, 16.0, 8),
    "p3.8xlarge": ("V100", 4, 16.0, 32),
    "p3.16xlarge": ("V100", 8, 16.0, 64),
    "g5.xlarge": ("A10G", 1, 24.0, 4),
    "g5.2xlarge": ("A10G", 1, 24.0, 8),
    "g5.4xlarge": ("A10G", 1, 24.0, 16),
    "g5.12xlarge": ("A10G", 4, 24.0, 48),
    "g5.48xlarge": ("A10G", 8, 24.0, 192),
    "g6.xlarge": ("L4", 1, 24.0, 4),
    "g6.2xlarge": ("L4", 1, 24.0, 8),
    "g6.4xlarge": ("L4", 1, 24.0, 16),
    "p5.48xlarge": ("H100", 8, 80.0, 192),
}


class AWSPricingFetcher(PricingFetcher):
    """Fetches GPU pricing from AWS Price List API or boto3."""

    def provider_name(self) -> str:
        return "aws"

    def fetch_gpu_pricing(self, regions: list[str] | None = None) -> list[PriceQuote]:
        try:
            return self._fetch_via_boto3(regions)
        except Exception:
            pass
        try:
            return self._fetch_via_http(regions)
        except Exception as e:
            logger.debug(f"AWS pricing fetch failed: {e}")
        return self._fallback()

    def _fetch_via_boto3(self, regions: list[str] | None = None) -> list[PriceQuote]:
        import boto3
        ec2 = boto3.client("ec2", region_name="us-east-1")
        target_regions = regions or [r["RegionName"] for r in ec2.describe_regions()["Regions"]]
        result = []
        for region in target_regions:
            try:
                regional = boto3.client("ec2", region_name=region)
                resp = regional.describe_spot_price_history(
                    InstanceTypes=list(_AWS_GPU_INSTANCES.keys()),
                    ProductDescriptions=["Linux/UNIX"],
                    MaxResults=100,
                )
                for entry in resp.get("SpotPriceHistory", []):
                    inst = entry["InstanceType"]
                    if inst not in _AWS_GPU_INSTANCES:
                        continue
                    spot = float(entry.get("SpotPrice", 0))
                    od = self._get_on_demand_price(inst, region)
                    result.append(PriceQuote(
                        provider="aws",
                        instance_type=inst,
                        region=region,
                        on_demand_hourly=od,
                        spot_hourly=spot,
                    ))
            except Exception as e:
                logger.debug(f"AWS spot pricing failed for {region}: {e}")
        return result

    def _fetch_via_http(self, regions: list[str] | None = None) -> list[PriceQuote]:
        import httpx
        resp = httpx.get(
            "https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonEC2/current/region_index.json",
            timeout=15.0,
        )
        resp.raise_for_status()
        region_index = resp.json()
        target_regions = regions or list(region_index.get("regions", {}).keys())
        result = []
        for region in target_regions[:3]:  # Limit to avoid rate limits
            region_info = region_index.get("regions", {}).get(region, {})
            url = region_info.get("currentVersionUrl", "")
            if not url:
                continue
            try:
                pricing_resp = httpx.get(
                    f"https://pricing.us-east-1.amazonaws.com{url}",
                    timeout=30.0,
                )
                pricing_resp.raise_for_status()
                data = pricing_resp.json()
                for sku, product in data.get("products", {}).items():
                    attrs = product.get("attributes", {})
                    inst = attrs.get("instanceType", "")
                    if inst not in _AWS_GPU_INSTANCES:
                        continue
                    terms = data.get("terms", {}).get("OnDemand", {}).get(sku, {})
                    od = 0.0
                    for term in terms.values():
                        for pd in term.get("priceDimensions", {}).values():
                            try:
                                od = float(pd.get("pricePerUnit", {}).get("USD", "0"))
                            except ValueError:
                                pass
                            break
                        if od > 0:
                            break
                    if od > 0:
                        result.append(PriceQuote(
                            provider="aws",
                            instance_type=inst,
                            region=region,
                            on_demand_hourly=od,
                            spot_hourly=od * 0.3,
                        ))
            except Exception:
                continue
        return result if result else self._fallback()

    def _fallback(self) -> list[PriceQuote]:
        static = {
            "p4d.24xlarge": (32.77, 14.40), "p3.2xlarge": (3.06, 3.83),
            "g5.xlarge": (1.006, 1.41), "g5.2xlarge": (1.212, 1.82),
            "g6.xlarge": (0.8054, 1.10), "p4de.24xlarge": (40.97, 27.20),
        }
        return [
            PriceQuote(provider="aws", instance_type=inst, region="us-east-1",
                       on_demand_hourly=od, spot_hourly=spot)
            for inst, (od, spot) in static.items()
        ]

    @staticmethod
    def _get_on_demand_price(instance_type: str, region: str) -> float:
        """Estimate on-demand price from spot price (rough heuristic)."""
        static = {
            "p4d.24xlarge": 32.77, "p3.2xlarge": 3.06, "g5.xlarge": 1.006,
            "g5.2xlarge": 1.212, "g6.xlarge": 0.8054, "p4de.24xlarge": 40.97,
        }
        return static.get(instance_type, 5.0)


class AWSAvailabilityChecker(AvailabilityChecker):
    """Checks AWS GPU instance availability."""

    def check_availability(self, instance_type: str, region: str) -> AvailabilityInfo:
        try:
            import boto3
            ec2 = boto3.client("ec2", region_name=region)
            resp = ec2.describe_instance_type_offerings(
                LocationType="region",
                Filters=[{"Name": "instance-type", "Values": [instance_type]}],
            )
            available = len(resp.get("InstanceTypeOfferings", [])) > 0
            return AvailabilityInfo(
                provider="aws",
                instance_type=instance_type,
                region=region,
                available=available,
            )
        except Exception as e:
            logger.debug(f"AWS availability check failed: {e}")
            return AvailabilityInfo(
                provider="aws", instance_type=instance_type, region=region,
                available=instance_type in _AWS_GPU_INSTANCES,
            )

    def list_gpu_instances(self, region: str | None = None) -> list[InstanceSpec]:
        specs = []
        for inst, (gpu, count, mem, vcpus) in _AWS_GPU_INSTANCES.items():
            specs.append(InstanceSpec(
                provider="aws",
                instance_type=inst,
                region=region or "us-east-1",
                gpu=GPUSpec(name=gpu, memory_gb=mem),
                gpu_count=count,
                vcpus=vcpus,
            ))
        return specs
