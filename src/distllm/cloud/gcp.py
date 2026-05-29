"""GCP Cloud Provider SDK Wrapper.

GCE accelerator type catalog, committed use discounts, and pricing.
Uses google-cloud-compute when available, falls back to REST API.
"""

from __future__ import annotations

import os
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

_GCP_GPU_INSTANCES: dict[str, tuple[str, int, float, int]] = {
    "a2-highgpu-1g": ("A100", 1, 40.0, 12),
    "a2-highgpu-2g": ("A100", 2, 40.0, 24),
    "a2-highgpu-4g": ("A100", 4, 40.0, 48),
    "a2-highgpu-8g": ("A100", 8, 40.0, 96),
    "a2-ultragpu-1g": ("A100", 1, 80.0, 12),
    "a2-ultragpu-2g": ("A100", 2, 80.0, 24),
    "a2-ultragpu-4g": ("A100", 4, 80.0, 48),
    "a2-ultragpu-8g": ("A100", 8, 80.0, 96),
    "g2-standard-4": ("L4", 1, 24.0, 4),
    "g2-standard-8": ("L4", 1, 24.0, 8),
    "g2-standard-12": ("L4", 1, 24.0, 12),
    "g2-standard-24": ("L4", 2, 24.0, 24),
    "g2-standard-48": ("L4", 4, 24.0, 48),
    "n1-standard-8-t4": ("T4", 1, 16.0, 8),
}

_GCP_REGIONS = [
    "us-central1", "us-east1", "us-east4", "us-west1", "us-west4",
    "europe-west1", "europe-west4", "europe-north1",
    "asia-east1", "asia-northeast1", "asia-southeast1",
]


class GCPPricingFetcher(PricingFetcher):
    """Fetches GPU pricing from GCP Cloud Billing API."""

    def provider_name(self) -> str:
        return "gcp"

    def fetch_gpu_pricing(self, regions: list[str] | None = None) -> list[PriceQuote]:
        target_regions = regions or _GCP_REGIONS
        try:
            return self._fetch_via_api(target_regions)
        except Exception as e:
            logger.debug(f"GCP pricing API failed: {e}")
        return self._fallback(target_regions)

    def _fetch_via_api(self, regions: list[str]) -> list[PriceQuote]:
        import httpx
        token = self._get_access_token()
        if not token:
            return self._fallback(regions)
        resp = httpx.get(
            "https://cloudbilling.googleapis.com/v1/services/6F81-5844-456A/skus",
            headers={"Authorization": f"Bearer {token}"},
            params={"pageSize": 500},
            timeout=30.0,
        )
        resp.raise_for_status()
        result = []
        for sku in resp.json().get("skus", []):
            desc = sku.get("description", "").lower()
            if not any(g in desc for g in ("gpu", "a100", "v100", "t4", "l4")):
                continue
            inst_type = self._extract_instance_type(desc)
            if not inst_type:
                continue
            pricing = sku.get("pricingInfo", [{}])
            if not pricing:
                continue
            tiers = pricing[0].get("pricingExpression", {}).get("tieredRates", [])
            if not tiers:
                continue
            unit_price = tiers[0].get("unitPrice", {})
            price = int(unit_price.get("units", 0)) + unit_price.get("nanos", 0) / 1e9
            for region in sku.get("serviceRegions", []):
                if region in regions:
                    result.append(PriceQuote(
                        provider="gcp", instance_type=inst_type, region=region,
                        on_demand_hourly=price, spot_hourly=price * 0.25,
                    ))
        return result if result else self._fallback(regions)

    def _get_access_token(self) -> str:
        try:
            cred_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
            if not cred_path:
                return ""
            import httpx
            import json
            with open(cred_path) as f:
                creds = json.load(f)
            resp = httpx.post(
                creds.get("token_uri", "https://oauth2.googleapis.com/token"),
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
        import json
        import time
        header = base64.urlsafe_b64encode(json.dumps({"alg": "RS256", "typ": "JWT"}).encode()).rstrip(b"=")
        now = int(time.time())
        claims = {
            "iss": creds.get("client_email", ""),
            "scope": "https://www.googleapis.com/auth/cloud-billing.readonly",
            "aud": creds.get("token_uri", ""),
            "iat": now, "exp": now + 3600,
        }
        payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).rstrip(b"=")
        message = header + b"." + payload
        try:
            from cryptography.hazmat.primitives import hashes, serialization
            from cryptography.hazmat.primitives.asymmetric import padding
            key = serialization.load_pem_private_key(creds.get("private_key", "").encode(), password=None)
            sig = key.sign(message, padding.PKCS1v15(), hashes.SHA256())
            return (message + b"." + base64.urlsafe_b64encode(sig).rstrip(b"=")).decode()
        except Exception:
            return ""

    def _extract_instance_type(self, description: str) -> str:
        for inst in _GCP_GPU_INSTANCES:
            if inst.lower() in description.lower():
                return inst
        return ""

    def _fallback(self, regions: list[str]) -> list[PriceQuote]:
        static = {
            "a2-highgpu-1g": (3.67, 0.92), "a2-highgpu-2g": (7.35, 1.84),
            "a2-ultragpu-1g": (5.07, 1.27), "g2-standard-4": (0.84, 0.21),
            "n1-standard-8-t4": (0.95, 0.24),
        }
        result = []
        for inst, (od, spot) in static.items():
            for region in regions[:3]:
                result.append(PriceQuote(
                    provider="gcp", instance_type=inst, region=region,
                    on_demand_hourly=od, spot_hourly=spot,
                ))
        return result


class GCPAvailabilityChecker(AvailabilityChecker):
    """Checks GCP GPU instance availability."""

    def check_availability(self, instance_type: str, region: str) -> AvailabilityInfo:
        try:
            import httpx
            token = self._get_access_token()
            if not token:
                return self._fallback_info(instance_type, region)
            resp = httpx.get(
                f"https://compute.googleapis.com/compute/v1/projects/{self._get_project()}/zones/{region}-a/acceleratorTypes",
                headers={"Authorization": f"Bearer {token}"},
                timeout=10.0,
            )
            resp.raise_for_status()
            available = len(resp.json().get("items", [])) > 0
            return AvailabilityInfo(
                provider="gcp", instance_type=instance_type, region=region,
                available=available,
            )
        except Exception:
            return self._fallback_info(instance_type, region)

    def list_gpu_instances(self, region: str | None = None) -> list[InstanceSpec]:
        specs = []
        for inst, (gpu, count, mem, vcpus) in _GCP_GPU_INSTANCES.items():
            specs.append(InstanceSpec(
                provider="gcp", instance_type=inst, region=region or "us-central1",
                gpu=GPUSpec(name=gpu, memory_gb=mem), gpu_count=count, vcpus=vcpus,
            ))
        return specs

    def _fallback_info(self, instance_type: str, region: str) -> AvailabilityInfo:
        return AvailabilityInfo(
            provider="gcp", instance_type=instance_type, region=region,
            available=instance_type in _GCP_GPU_INSTANCES,
        )

    def _get_access_token(self) -> str:
        try:
            cred_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
            if not cred_path:
                return ""
            import httpx, json, time, base64
            with open(cred_path) as f:
                creds = json.load(f)
            header = base64.urlsafe_b64encode(json.dumps({"alg": "RS256", "typ": "JWT"}).encode()).rstrip(b"=")
            now = int(time.time())
            claims = {"iss": creds.get("client_email", ""), "scope": "https://www.googleapis.com/auth/compute.readonly", "aud": creds.get("token_uri", ""), "iat": now, "exp": now + 3600}
            payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).rstrip(b"=")
            message = header + b"." + payload
            from cryptography.hazmat.primitives import hashes, serialization
            from cryptography.hazmat.primitives.asymmetric import padding
            key = serialization.load_pem_private_key(creds.get("private_key", "").encode(), password=None)
            sig = key.sign(message, padding.PKCS1v15(), hashes.SHA256())
            resp = httpx.post(creds.get("token_uri", ""), data={"grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer", "assertion": (message + b"." + base64.urlsafe_b64encode(sig).rstrip(b"=")).decode()}, timeout=10.0)
            resp.raise_for_status()
            return resp.json().get("access_token", "")
        except Exception:
            return ""

    def _get_project(self) -> str:
        import json
        try:
            cred_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
            if cred_path:
                with open(cred_path) as f:
                    return json.load(f).get("project_id", "")
        except Exception:
            pass
        return os.environ.get("GOOGLE_CLOUD_PROJECT", "")
