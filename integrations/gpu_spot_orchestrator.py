"""GPU Spot Orchestrator -- Multi-Provider Spot Instance Manager.

Integrates with RunPod, Vast.ai, and Salad GPU marketplaces to find,
bid on, and manage spot / rentable GPU instances through a unified
interface.

Provider SDKs are entirely optional -- the module falls back to direct
HTTP calls via ``httpx`` when a provider's SDK package is not installed.

Typical usage::

    from integrations.gpu_spot_orchestrator import GPUSpotMarket, SpotOrchestrator

    # 1. Query available GPU instances across one provider
    market = GPUSpotMarket()
    gpus = market.list_instances(
        provider="runpod",
        region="us-east",
        gpu_type="NVIDIA GeForce RTX 4090",
    )

    # 2. Find the cheapest across all providers
    orch = SpotOrchestrator(market=market)
    best = orch.find_cheapest(gpu_type="RTX 4090", max_price=0.50, region="US")

    # 3. Launch a cluster
    cluster = orch.launch_cluster(best)

    # 4. Monitor costs
    report = orch.monitor_costs()

    # 5. Migrate between providers
    orch.swap_providers(current="runpod", target="vast")
"""

from __future__ import annotations

import os
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from loguru import logger

# ---------------------------------------------------------------------------
# Optional provider SDKs
# ---------------------------------------------------------------------------

_HAS_RUNPOD: bool
_HAS_SALAD: bool

try:
    import runpod  # noqa: F401  # type: ignore[import-untyped]

    _HAS_RUNPOD = True
except ImportError:
    _HAS_RUNPOD = False

try:
    import salad_cloud_sdk  # noqa: F401  # type: ignore[import-untyped]

    _HAS_SALAD = True
except ImportError:
    _HAS_SALAD = False

# Vast.ai does not provide an official SDK -- always direct HTTP.

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Provider(str, Enum):
    """Supported GPU marketplace providers."""

    RUNPOD = "runpod"
    VAST = "vast"
    SALAD = "salad"


class InstanceStatus(str, Enum):
    """Lifecycle status of a GPU instance."""

    AVAILABLE = "available"
    RUNNING = "running"
    STOPPED = "stopped"
    TERMINATED = "terminated"
    PENDING = "pending"
    ERROR = "error"


class BidStatus(str, Enum):
    """Outcome of a spot bid / rental request."""

    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    ACTIVE = "active"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class GPUInstance:
    """A GPU instance listing from a marketplace provider.

    Attributes
    ----------
    provider:
        Provider identifier (e.g. "runpod", "vast", "salad").
    instance_id:
        Provider-specific instance identifier.
    gpu_type:
        GPU model name (e.g. "NVIDIA GeForce RTX 4090").
    gpu_count:
        Number of GPUs in this instance.
    vram_gb:
        VRAM per GPU in GB.
    price_per_hour:
        Current hourly price in USD (spot / rental).
    region:
        Geographic region or data-center identifier.
    status:
        Current lifecycle status.
    spot:
        Whether this is a spot / preemptible / rental instance.
    on_demand_price:
        Reference on-demand price per hour if known.
    availability_score:
        Historical availability / uptime score in [0.0, 1.0].
    cpu_cores:
        Number of virtual CPU cores.
    ram_gb:
        System RAM in GB.
    storage_gb:
        Local ephemeral storage in GB.
    """

    provider: str
    instance_id: str
    gpu_type: str
    gpu_count: int
    vram_gb: float
    price_per_hour: float
    region: str
    status: InstanceStatus = InstanceStatus.AVAILABLE
    spot: bool = True
    on_demand_price: float = 0.0
    availability_score: float = 1.0
    cpu_cores: int = 0
    ram_gb: float = 0.0
    storage_gb: float = 0.0


@dataclass
class BidResult:
    """Result of placing a bid / rental request.

    Attributes
    ----------
    provider:
        Provider identifier.
    instance_id:
        Assigned instance identifier (empty if rejected or pending).
    bid_price:
        The price used for the bid in USD per hour.
    status:
        Outcome of the bid.
    estimated_wait_minutes:
        Estimated time until the instance transitions to active.
    error_message:
        Human-readable error description if the bid failed.
    """

    provider: str
    instance_id: str
    bid_price: float
    status: BidStatus = BidStatus.PENDING
    estimated_wait_minutes: float = 0.0
    error_message: str = ""


@dataclass
class CostReport:
    """Accumulated cost for a running instance.

    Attributes
    ----------
    provider:
        Provider identifier.
    instance_id:
        Instance identifier.
    gpu_type:
        GPU model name.
    hours_running:
        Elapsed runtime in hours.
    total_cost:
        Accumulated cost in USD.
    spot_savings:
        Estimated savings vs. on-demand pricing in USD.
    """

    provider: str
    instance_id: str
    gpu_type: str
    hours_running: float
    total_cost: float
    spot_savings: float


# ---------------------------------------------------------------------------
# Reference on-demand prices (USD / GPU-hour)
#
# Used as a baseline for dynamic bidding.  Values represent typical
# cloud on-demand rates for each GPU class.  Update periodically.
# ---------------------------------------------------------------------------

GPU_ON_DEMAND_REFERENCE: dict[str, float] = {
    "NVIDIA H100": 3.50,
    "NVIDIA H100 NVL": 4.00,
    "NVIDIA A100 80GB": 3.00,
    "NVIDIA A100": 2.50,
    "NVIDIA A40": 1.50,
    "NVIDIA A6000": 1.20,
    "NVIDIA A10G": 1.00,
    "NVIDIA L40S": 2.00,
    "NVIDIA L4": 0.60,
    "NVIDIA T4": 0.40,
    "NVIDIA RTX 6000 Ada": 1.50,
    "NVIDIA RTX 5000 Ada": 1.00,
    "NVIDIA RTX 4090": 0.80,
    "NVIDIA RTX 4080": 0.55,
    "NVIDIA RTX 4070": 0.40,
    "NVIDIA RTX 3090": 0.50,
    "NVIDIA RTX 3080": 0.35,
    "NVIDIA RTX 3070": 0.25,
    "NVIDIA RTX 3060": 0.20,
    "NVIDIA GeForce RTX 4090": 0.80,
    "NVIDIA GeForce RTX 4080": 0.55,
    "NVIDIA GeForce RTX 4070": 0.40,
    "NVIDIA GeForce RTX 3090": 0.50,
    "NVIDIA GeForce RTX 3080": 0.35,
    "NVIDIA L40": 2.50,
    "AMD MI250": 2.00,
    "AMD MI210": 1.50,
}

# ---------------------------------------------------------------------------
# Price tracker  (dynamic bidding engine)
# ---------------------------------------------------------------------------


class _PriceTracker:
    """Sliding-window spot-price tracker for dynamic bid computation.

    Maintains a moving history of observed spot prices per GPU type and
    suggests bids at a configurable fraction of the on-demand reference
    price, optionally adjusted by market conditions.
    """

    def __init__(self, window_size: int = 50) -> None:
        self._window_size = window_size
        self._prices: dict[str, list[float]] = {}

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_price(self, gpu_type: str, price: float) -> None:
        """Push *price* into the sliding window for *gpu_type*."""
        if gpu_type not in self._prices:
            self._prices[gpu_type] = []
        window = self._prices[gpu_type]
        window.append(price)
        if len(window) > self._window_size:
            window.pop(0)

    def record_prices(self, instances: list[GPUInstance]) -> None:
        """Record prices from a list of ``GPUInstance`` objects."""
        for inst in instances:
            self.record_price(inst.gpu_type, inst.price_per_hour)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def recent_prices(self, gpu_type: str) -> list[float]:
        """Return the recorded price window for *gpu_type* (may be empty)."""
        return list(self._prices.get(gpu_type, []))

    def moving_average(self, gpu_type: str) -> float | None:
        prices = self._prices.get(gpu_type)
        if not prices:
            return None
        return sum(prices) / len(prices)

    def p75(self, gpu_type: str) -> float | None:
        """75th-percentile of observed spot prices (robust to spikes)."""
        prices = self._prices.get(gpu_type)
        if not prices:
            return None
        return float(statistics.quantiles(prices, n=4)[-1])  # upper quartile

    # ------------------------------------------------------------------
    # Bid suggestion
    # ------------------------------------------------------------------

    def suggest_bid(
        self,
        gpu_type: str,
        on_demand_price: float | None = None,
        min_fraction: float = 0.60,
        max_fraction: float = 0.80,
    ) -> float | None:
        """Compute a recommended bid price for *gpu_type*.

        Algorithm
        ---------
        1. Use the higher of ``min(on_demand_price, P75_historical)`` when
           historical data is available.
        2. Otherwise fall back to ``on_demand_price * 0.65``.
        3. Clamp the result to ``[on_demand_price * min_fraction,
           on_demand_price * max_fraction]``.
        4. Round to 3 decimal places (tenths of a cent).

        Returns ``None`` when no on-demand reference is available and no
        historical data exists.
        """
        on_demand = on_demand_price or GPU_ON_DEMAND_REFERENCE.get(gpu_type)
        if on_demand is None or on_demand <= 0:
            # Try the historical average as a fallback reference
            avg = self.moving_average(gpu_type)
            if avg:
                return round(avg, 3)
            return None

        # Candidate from historical P75 (more robust than mean)
        candidate = self.p75(gpu_type)
        if candidate is not None:
            candidate = min(candidate, on_demand * 0.70)
        else:
            candidate = on_demand * 0.65  # default mid-range

        # Clamp
        lower = on_demand * min_fraction
        upper = on_demand * max_fraction
        bid = max(lower, min(candidate, upper))
        return round(bid, 3)

    def describe(self, gpu_type: str) -> dict[str, Any]:
        """Return a diagnostic summary for a GPU type."""
        prices = self._prices.get(gpu_type, [])
        on_demand = GPU_ON_DEMAND_REFERENCE.get(gpu_type)
        return {
            "gpu_type": gpu_type,
            "observations": len(prices),
            "moving_avg": self.moving_average(gpu_type),
            "p75": self.p75(gpu_type),
            "on_demand_reference": on_demand,
            "recommended_bid": self.suggest_bid(gpu_type, on_demand),
        }


# ---------------------------------------------------------------------------
# Base HTTP client helper
# ---------------------------------------------------------------------------

_HTTP_TIMEOUT: float = 30.0


def _http_get(url: str, headers: dict[str, str] | None = None, params: dict[str, Any] | None = None) -> dict[str, Any] | list[Any]:
    """Wrapper over ``httpx.get`` that returns parsed JSON."""
    import httpx

    resp = httpx.get(url, headers=headers or {}, params=params, timeout=_HTTP_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def _http_post(url: str, json_body: dict[str, Any], headers: dict[str, str] | None = None) -> dict[str, Any] | list[Any]:
    """Wrapper over ``httpx.post`` that returns parsed JSON."""
    import httpx

    resp = httpx.post(url, json=json_body, headers=headers or {}, timeout=_HTTP_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def _http_delete(url: str, headers: dict[str, str] | None = None) -> dict[str, Any] | list[Any]:
    """Wrapper over ``httpx.delete`` that returns parsed JSON."""
    import httpx

    resp = httpx.delete(url, headers=headers or {}, timeout=_HTTP_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------


class _RunPodProvider:
    """RunPod GPU marketplace provider.

    Uses the `runpod` SDK if installed, otherwise falls back to the
    RunPod REST API (v2).

    Environment variables
    ---------------------
    RUNPOD_API_KEY : str
        API key for RunPod authentication.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.environ.get("RUNPOD_API_KEY", "")
        self._http_headers: dict[str, str] = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        self._use_sdk = _HAS_RUNPOD and bool(self._api_key)

    # -- listing -------------------------------------------------------

    def list_instances(
        self,
        region: str | None = None,
        gpu_type: str | None = None,
    ) -> list[GPUInstance]:
        """Query the RunPod marketplace for available GPU instances."""
        if self._use_sdk:
            return self._list_via_sdk(region, gpu_type)
        return self._list_via_http(region, gpu_type)

    def _list_via_sdk(self, region: str | None, gpu_type: str | None) -> list[GPUInstance]:
        import runpod  # type: ignore[import-untyped]

        try:
            prices = runpod.get_gpu_prices()  # dict[str, dict]
        except Exception as exc:
            logger.warning("RunPod SDK get_gpu_prices failed: {}", exc)
            return []

        results: list[GPUInstance] = []
        for gpu_name, info in prices.items():
            if gpu_type and gpu_type.lower() not in gpu_name.lower():
                continue
            price = float(info.get("minimumBidPrice", info.get("costPerHour", 0)))
            on_demand = float(info.get("costPerHour", 0))
            results.append(
                GPUInstance(
                    provider=Provider.RUNPOD.value,
                    instance_id=f"runpod-mkp-{gpu_name}",
                    gpu_type=gpu_name,
                    gpu_count=int(info.get("gpuCount", 1)),
                    vram_gb=float(info.get("gpuMemoryInGb", 0)),
                    price_per_hour=price,
                    region=region or "global",
                    status=InstanceStatus.AVAILABLE,
                    spot=True,
                    on_demand_price=on_demand,
                    availability_score=0.95,
                )
            )
        return results

    def _list_via_http(self, region: str | None, gpu_type: str | None) -> list[GPUInstance]:
        try:
            data = _http_get(
                "https://api.runpod.io/v2/gpu/prices",
                headers=self._http_headers,
            )
        except Exception as exc:
            logger.warning("RunPod HTTP list_instances failed: {}", exc)
            return []

        # Expected response: {"gpuPrices": {"gpuType": {...}, ...}}
        prices = data if isinstance(data, dict) else {}
        gpu_prices = prices.get("gpuPrices", prices) if isinstance(prices, dict) else {}

        results: list[GPUInstance] = []
        for gpu_name, info in gpu_prices.items():
            if gpu_type and gpu_type.lower() not in gpu_name.lower():
                continue
            if not isinstance(info, dict):
                continue
            price = float(info.get("minimumBidPrice", info.get("costPerHour", 0)))
            on_demand = float(info.get("costPerHour", 0))
            results.append(
                GPUInstance(
                    provider=Provider.RUNPOD.value,
                    instance_id=f"runpod-mkp-{gpu_name}",
                    gpu_type=gpu_name,
                    gpu_count=int(info.get("gpuCount", 1)),
                    vram_gb=float(info.get("gpuMemoryInGb", 0)),
                    price_per_hour=price,
                    region=region or "global",
                    status=InstanceStatus.AVAILABLE,
                    spot=True,
                    on_demand_price=on_demand,
                    availability_score=0.95,
                )
            )
        return results

    # -- bidding -------------------------------------------------------

    def bid(self, config: dict[str, Any]) -> BidResult:
        """Place a spot instance request on RunPod.

        Required config keys
        --------------------
        gpu_type : str
            GPU model identifier.
        max_price : float
            Maximum bid price per hour.
        """
        if self._use_sdk:
            return self._bid_via_sdk(config)
        return self._bid_via_http(config)

    def _bid_via_sdk(self, config: dict[str, Any]) -> BidResult:
        import runpod  # type: ignore[import-untyped]

        gpu_type = config.get("gpu_type", "")
        max_price = config.get("max_price", 0.0)
        try:
            result = runpod.create_instance(
                gpu_type=gpu_type,
                bid_price=max_price,
                image=config.get("image", ""),
                disk_size_gb=config.get("disk_size_gb", 10),
            )
            instance_id = str(result.get("id", ""))
            return BidResult(
                provider=Provider.RUNPOD.value,
                instance_id=instance_id,
                bid_price=max_price,
                status=BidStatus.ACTIVE if instance_id else BidStatus.REJECTED,
            )
        except Exception as exc:
            logger.warning("RunPod SDK bid failed: {}", exc)
            return BidResult(
                provider=Provider.RUNPOD.value,
                instance_id="",
                bid_price=max_price,
                status=BidStatus.REJECTED,
                error_message=str(exc),
            )

    def _bid_via_http(self, config: dict[str, Any]) -> BidResult:
        gpu_type = config.get("gpu_type", "")
        max_price = config.get("max_price", 0.0)

        payload: dict[str, Any] = {
            "gpuType": gpu_type,
            "bidPrice": max_price,
            "containerDiskSizeGb": config.get("disk_size_gb", 10),
        }
        if config.get("image"):
            payload["imageName"] = config["image"]
        if config.get("region"):
            payload["datacenter"] = config["region"]
        if config.get("count", 1) > 1:
            payload["quantity"] = config["count"]

        try:
            data = _http_post(
                "https://api.runpod.io/v2/instances",
                json_body=payload,
                headers=self._http_headers,
            )
        except Exception as exc:
            logger.warning("RunPod HTTP bid failed: {}", exc)
            return BidResult(
                provider=Provider.RUNPOD.value,
                instance_id="",
                bid_price=max_price,
                status=BidStatus.REJECTED,
                error_message=str(exc),
            )

        resp_data = data if isinstance(data, dict) else {}
        instance_id = str(resp_data.get("id", ""))
        return BidResult(
            provider=Provider.RUNPOD.value,
            instance_id=instance_id,
            bid_price=float(resp_data.get("bidPrice", max_price)),
            status=BidStatus.ACTIVE if instance_id else BidStatus.PENDING,
            estimated_wait_minutes=float(resp_data.get("estimatedWaitMinutes", 2)),
        )

    # -- cancellation --------------------------------------------------

    def cancel_bid(self, instance_id: str) -> bool:
        """Terminate a RunPod instance by *instance_id*."""
        if self._use_sdk:
            return self._cancel_via_sdk(instance_id)
        return self._cancel_via_http(instance_id)

    def _cancel_via_sdk(self, instance_id: str) -> bool:
        import runpod  # type: ignore[import-untyped]

        try:
            runpod.stop_instance(instance_id)
            return True
        except Exception as exc:
            logger.warning("RunPod SDK cancel failed: {}", exc)
            return False

    def _cancel_via_http(self, instance_id: str) -> bool:
        try:
            _http_delete(
                f"https://api.runpod.io/v2/instances/{instance_id}",
                headers=self._http_headers,
            )
            return True
        except Exception as exc:
            logger.warning("RunPod HTTP cancel failed: {}", exc)
            return False

    def close(self) -> None:
        """Release any held HTTP resources."""
        pass


class _VastProvider:
    """Vast.ai GPU marketplace provider.

    Vast.ai does not offer an official Python SDK -- always uses REST
    API (v0).

    Environment variables
    ---------------------
    VAST_API_KEY : str
        API key for Vast.ai authentication.
    """

    _BASE = "https://console.vast.ai/api/v0"

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.environ.get("VAST_API_KEY", "")
        self._headers: dict[str, str] = {
            "Accept": "application/json",
        }

    # -- listing -------------------------------------------------------

    def list_instances(
        self,
        region: str | None = None,
        gpu_type: str | None = None,
    ) -> list[GPUInstance]:
        """Query Vast.ai marketplace for available GPU offers."""
        try:
            data = _http_get(f"{self._BASE}/bulk_info/", params={"api_key": self._api_key})
        except Exception as exc:
            logger.warning("Vast.ai list_instances failed: {}", exc)
            return []

        offers = data if isinstance(data, list) else data.get("offers", []) if isinstance(data, dict) else []

        results: list[GPUInstance] = []
        for offer in offers:
            if not isinstance(offer, dict):
                continue
            gpu_name = str(offer.get("gpu_name", ""))
            if gpu_type and gpu_type.lower() not in gpu_name.lower():
                continue
            if region:
                offer_region = str(offer.get("geographic_location", offer.get("location", "")))
                if region.lower() not in offer_region.lower():
                    continue

            price = float(offer.get("dph_total", offer.get("billed_hourly_cost", 0)))
            on_demand = float(offer.get("listed_hourly_cost", price))
            results.append(
                GPUInstance(
                    provider=Provider.VAST.value,
                    instance_id=str(offer.get("id", "")),
                    gpu_type=gpu_name,
                    gpu_count=int(offer.get("num_gpus", 1)),
                    vram_gb=float(offer.get("gpu_ram", 0)),
                    price_per_hour=price,
                    region=str(offer.get("geographic_location", offer.get("location", "unknown"))),
                    status=InstanceStatus.AVAILABLE,
                    spot=True,
                    on_demand_price=on_demand,
                    availability_score=float(offer.get("reliability", 0.95)),
                    cpu_cores=int(offer.get("cpu_cores", 0)),
                    ram_gb=float(offer.get("cpu_ram", 0)),
                    storage_gb=float(offer.get("storage_cost", 0)),
                )
            )
        return results

    # -- bidding -------------------------------------------------------

    def bid(self, config: dict[str, Any]) -> BidResult:
        """Rent an instance on Vast.ai.

        Required config keys
        --------------------
        instance_id : str
            The offer id to rent (from ``list_instances``).
        max_price : float
            Maximum bid price per hour.
        """
        offer_id = config.get("instance_id", "")
        max_price = config.get("max_price", 0.0)
        if not offer_id:
            return BidResult(
                provider=Provider.VAST.value,
                instance_id="",
                bid_price=max_price,
                status=BidStatus.REJECTED,
                error_message="instance_id is required for Vast.ai bid",
            )

        payload: dict[str, Any] = {
            "client_id": "me",
            "image": config.get("image", "nvidia/cuda:12.2.0-base-ubuntu22.04"),
            "disk": config.get("disk_size_gb", 10),
            "offer_id": offer_id,
            "price": max_price,
        }
        if config.get("duration_hours"):
            payload["duration"] = config["duration_hours"]
        if config.get("label"):
            payload["label"] = config["label"]

        try:
            data = _http_post(
                f"{self._BASE}/create_instances/",
                json_body=payload,
                headers=self._headers,
            )
        except Exception as exc:
            logger.warning("Vast.ai bid failed: {}", exc)
            return BidResult(
                provider=Provider.VAST.value,
                instance_id="",
                bid_price=max_price,
                status=BidStatus.REJECTED,
                error_message=str(exc),
            )

        resp_data = data if isinstance(data, dict) else {}
        inst_id = str(resp_data.get("new_contract", resp_data.get("id", "")))
        return BidResult(
            provider=Provider.VAST.value,
            instance_id=inst_id,
            bid_price=float(resp_data.get("price", max_price)),
            status=BidStatus.ACTIVE if inst_id else BidStatus.PENDING,
            estimated_wait_minutes=float(resp_data.get("estimated_wait", 1)),
        )

    # -- cancellation --------------------------------------------------

    def cancel_bid(self, instance_id: str) -> bool:
        """Terminate a Vast.ai rental by *instance_id*."""
        try:
            _http_delete(
                f"{self._BASE}/instances/{instance_id}/",
                headers=self._headers,
            )
            return True
        except Exception as exc:
            logger.warning("Vast.ai cancel failed: {}", exc)
            return False

    def close(self) -> None:
        pass


class _SaladProvider:
    """Salad GPU marketplace provider.

    Uses the ``salad-cloud-sdk`` if installed, otherwise falls back to
    the Salad REST API.

    Environment variables
    ---------------------
    SALAD_API_KEY : str
        API key for Salad authentication.
    SALAD_ORGANIZATION : str
        Salad organization name.
    SALAD_PROJECT : str
        Salad project name.
    """

    _BASE = "https://api.salad.com/api/public"

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.environ.get("SALAD_API_KEY", "")
        self._org = os.environ.get("SALAD_ORGANIZATION", "")
        self._project = os.environ.get("SALAD_PROJECT", "")
        self._headers: dict[str, str] = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        self._use_sdk = _HAS_SALAD and bool(self._api_key)

    # -- listing -------------------------------------------------------

    def list_instances(
        self,
        region: str | None = None,
        gpu_type: str | None = None,
    ) -> list[GPUInstance]:
        """Query Salad marketplace for available GPU containers."""
        # Salad does not expose a public marketplace browse endpoint;
        # list the user's own containers and known GPU node types.
        try:
            data = _http_get(
                f"{self._BASE}/organizations/{self._org}/projects/{self._project}/containers",
                headers=self._headers,
            )
        except Exception as exc:
            logger.warning("Salad list_instances failed: {}", exc)
            return self._list_fallback(region, gpu_type)

        containers = data if isinstance(data, list) else data.get("items", []) if isinstance(data, dict) else []

        results: list[GPUInstance] = []
        for container in containers:
            if not isinstance(container, dict):
                continue
            gpu_name = str(container.get("gpuClass", container.get("gpu_type", "Unknown")))
            if gpu_type and gpu_type.lower() not in gpu_name.lower():
                continue
            container_region = str(container.get("region", ""))
            if region and region.lower() not in container_region.lower():
                continue

            results.append(
                GPUInstance(
                    provider=Provider.SALAD.value,
                    instance_id=str(container.get("id", "")),
                    gpu_type=gpu_name,
                    gpu_count=int(container.get("gpuCount", 1)),
                    vram_gb=float(container.get("gpuVramGb", 0)),
                    price_per_hour=float(container.get("pricePerHour", 0)),
                    region=container_region or "global",
                    status=InstanceStatus.AVAILABLE,
                    spot=True,
                    on_demand_price=float(container.get("onDemandPrice", 0)),
                    availability_score=0.90,
                )
            )
        return results

    def _list_fallback(
        self,
        region: str | None = None,
        gpu_type: str | None = None,
    ) -> list[GPUInstance]:
        """Return known Salad GPU node types when the API is unavailable."""
        known_gpus: list[dict[str, Any]] = [
            {"gpu_type": "NVIDIA RTX 4090", "gpu_count": 1, "vram": 24.0, "price": 0.45},
            {"gpu_type": "NVIDIA RTX 4080", "gpu_count": 1, "vram": 16.0, "price": 0.30},
            {"gpu_type": "NVIDIA RTX 4070", "gpu_count": 1, "vram": 12.0, "price": 0.22},
            {"gpu_type": "NVIDIA RTX 3090", "gpu_count": 1, "vram": 24.0, "price": 0.35},
            {"gpu_type": "NVIDIA A100 80GB", "gpu_count": 1, "vram": 80.0, "price": 2.00},
            {"gpu_type": "NVIDIA A10G", "gpu_count": 1, "vram": 24.0, "price": 0.70},
        ]
        results: list[GPUInstance] = []
        for entry in known_gpus:
            if gpu_type and gpu_type.lower() not in entry["gpu_type"].lower():
                continue
            results.append(
                GPUInstance(
                    provider=Provider.SALAD.value,
                    instance_id=f"salad-mkp-{entry['gpu_type'].replace(' ', '-')}",
                    gpu_type=str(entry["gpu_type"]),
                    gpu_count=int(entry["gpu_count"]),
                    vram_gb=float(entry["vram"]),
                    price_per_hour=float(entry["price"]),
                    region=region or "global",
                    status=InstanceStatus.AVAILABLE,
                    spot=True,
                    on_demand_price=float(entry["price"]) * 1.4,
                )
            )
        return results

    # -- bidding -------------------------------------------------------

    def bid(self, config: dict[str, Any]) -> BidResult:
        """Create a container instance on Salad.

        Required config keys
        --------------------
        gpu_type : str
            GPU class identifier.
        max_price : float
            Maximum bid / price per hour.
        image : str
            Docker image to run (required by Salad).
        """
        if self._use_sdk:
            return self._bid_via_sdk(config)
        return self._bid_via_http(config)

    def _bid_via_sdk(self, config: dict[str, Any]) -> BidResult:
        import salad_cloud_sdk  # type: ignore[import-untyped]

        gpu_type = config.get("gpu_type", "")
        max_price = config.get("max_price", 0.0)

        try:
            client = salad_cloud_sdk.SaladCloudSdk(api_key=self._api_key)
            result = client.create_container(
                organization_name=self._org,
                project_name=self._project,
                container_request={
                    "gpu_class": gpu_type,
                    "max_price_per_hour": max_price,
                    "image": config.get("image", "ubuntu:22.04"),
                    "disk_size_gb": config.get("disk_size_gb", 10),
                    "replicas": config.get("count", 1),
                },
            )
            instance_id = str(getattr(result, "id", ""))
            return BidResult(
                provider=Provider.SALAD.value,
                instance_id=instance_id,
                bid_price=float(getattr(result, "price_per_hour", max_price)),
                status=BidStatus.PENDING if instance_id else BidStatus.REJECTED,
            )
        except Exception as exc:
            logger.warning("Salad SDK bid failed: {}", exc)
            return BidResult(
                provider=Provider.SALAD.value,
                instance_id="",
                bid_price=max_price,
                status=BidStatus.REJECTED,
                error_message=str(exc),
            )

    def _bid_via_http(self, config: dict[str, Any]) -> BidResult:
        gpu_type = config.get("gpu_type", "")
        max_price = config.get("max_price", 0.0)

        payload: dict[str, Any] = {
            "gpuClass": gpu_type,
            "maxPricePerHour": max_price,
            "image": config.get("image", "ubuntu:22.04"),
            "diskSizeGb": config.get("disk_size_gb", 10),
            "replicas": config.get("count", 1),
        }
        if config.get("region"):
            payload["region"] = config["region"]

        try:
            data = _http_post(
                f"{self._BASE}/organizations/{self._org}/projects/{self._project}/containers",
                json_body=payload,
                headers=self._headers,
            )
        except Exception as exc:
            logger.warning("Salad HTTP bid failed: {}", exc)
            return BidResult(
                provider=Provider.SALAD.value,
                instance_id="",
                bid_price=max_price,
                status=BidStatus.REJECTED,
                error_message=str(exc),
            )

        resp_data = data if isinstance(data, dict) else {}
        instance_id = str(resp_data.get("id", ""))
        return BidResult(
            provider=Provider.SALAD.value,
            instance_id=instance_id,
            bid_price=float(resp_data.get("maxPricePerHour", max_price)),
            status=BidStatus.PENDING if instance_id else BidStatus.REJECTED,
            estimated_wait_minutes=float(resp_data.get("estimatedWaitMinutes", 3)),
        )

    # -- cancellation --------------------------------------------------

    def cancel_bid(self, instance_id: str) -> bool:
        """Delete a Salad container instance by *instance_id*."""
        if self._use_sdk:
            return self._cancel_via_sdk(instance_id)
        return self._cancel_via_http(instance_id)

    def _cancel_via_sdk(self, instance_id: str) -> bool:
        import salad_cloud_sdk  # type: ignore[import-untyped]

        try:
            client = salad_cloud_sdk.SaladCloudSdk(api_key=self._api_key)
            client.delete_container(
                organization_name=self._org,
                project_name=self._project,
                container_id=instance_id,
            )
            return True
        except Exception as exc:
            logger.warning("Salad SDK cancel failed: {}", exc)
            return False

    def _cancel_via_http(self, instance_id: str) -> bool:
        try:
            _http_delete(
                f"{self._BASE}/organizations/{self._org}/projects/{self._project}/containers/{instance_id}",
                headers=self._headers,
            )
            return True
        except Exception as exc:
            logger.warning("Salad HTTP cancel failed: {}", exc)
            return False

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Provider registry  (maps Provider enum -> implementation class)
# ---------------------------------------------------------------------------

_PROVIDER_MAP: dict[str, type] = {
    Provider.RUNPOD.value: _RunPodProvider,
    Provider.VAST.value: _VastProvider,
    Provider.SALAD.value: _SaladProvider,
}

# ---------------------------------------------------------------------------
# GPUSpotMarket  -- unified provider facade
# ---------------------------------------------------------------------------


class GPUSpotMarket:
    """Unified interface to GPU marketplace providers.

    Lazily initializes provider clients on first use.  Each provider
    SDK is optional -- if not installed, the class falls back to direct
    HTTP API calls.

    Parameters
    ----------
    price_tracker:
        Shared price tracker instance.  A new one is created when
        ``None``.
    """

    def __init__(self, price_tracker: _PriceTracker | None = None) -> None:
        self._price_tracker = price_tracker or _PriceTracker()
        self._providers: dict[str, Any] = {}
        self._closed = False

    # ------------------------------------------------------------------
    # Provider access
    # ------------------------------------------------------------------

    def _get_provider(self, provider: str) -> Any:
        if self._closed:
            raise RuntimeError("GPUSpotMarket has been closed")
        provider_key = provider.lower()
        if provider_key not in self._providers:
            cls = _PROVIDER_MAP.get(provider_key)
            if cls is None:
                valid = list(_PROVIDER_MAP)
                raise ValueError(f"Unknown provider {provider!r}. Valid: {valid}")
            logger.debug("Initialising provider: {}", provider_key)
            self._providers[provider_key] = cls()
        return self._providers[provider_key]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_instances(
        self,
        provider: str,
        region: str | None = None,
        gpu_type: str | None = None,
    ) -> list[GPUInstance]:
        """List available GPU instances from *provider*.

        Parameters
        ----------
        provider:
            Provider name ("runpod", "vast", "salad").
        region:
            Optional region filter (e.g. "US", "us-east", "EU").
        gpu_type:
            Optional GPU model filter (e.g. "RTX 4090", "A100").

        Returns
        -------
        list[GPUInstance]
            Matching instances sorted by price ascending.
        """
        p = self._get_provider(provider)
        instances = p.list_instances(region=region, gpu_type=gpu_type)
        # Feed prices into the tracker for dynamic bidding
        self._price_tracker.record_prices(instances)
        instances.sort(key=lambda i: i.price_per_hour)
        return instances

    def list_all(
        self,
        region: str | None = None,
        gpu_type: str | None = None,
        providers: list[str] | None = None,
    ) -> dict[str, list[GPUInstance]]:
        """List instances from multiple providers.

        Parameters
        ----------
        region:
            Optional region filter.
        gpu_type:
            Optional GPU type filter.
        providers:
            Subset of providers to query (default: all supported).

        Returns
        -------
        dict[str, list[GPUInstance]]
            Provider -> matching instances (sorted by price ascending).
        """
        targets = providers or list(_PROVIDER_MAP)
        result: dict[str, list[GPUInstance]] = {}
        for prov in targets:
            try:
                result[prov] = self.list_instances(provider=prov, region=region, gpu_type=gpu_type)
            except Exception as exc:
                logger.warning("Failed to list instances from {}: {}", prov, exc)
                result[prov] = []
        return result

    def bid(self, provider: str, config: dict[str, Any]) -> BidResult:
        """Place a bid / rental request on *provider*.

        Parameters
        ----------
        provider:
            Provider name.
        config:
            Provider-specific configuration dictionary.  Common keys:

            - ``gpu_type`` (str) -- GPU model.
            - ``max_price`` (float) -- Maximum $/hr.
            - ``region`` (str, optional) -- Preferred region.
            - ``count`` (int, optional) -- Number of instances.
            - ``image`` (str, optional) -- Docker image.
            - ``disk_size_gb`` (int, optional) -- Local storage.
        """
        p = self._get_provider(provider)

        # Apply dynamic pricing if ``max_price`` is not explicitly set
        if "max_price" not in config or not config.get("max_price"):
            gpu_type = config.get("gpu_type", "")
            on_demand = GPU_ON_DEMAND_REFERENCE.get(gpu_type)
            suggested = self._price_tracker.suggest_bid(gpu_type, on_demand)
            if suggested is not None:
                config = dict(config)
                config["max_price"] = suggested
                logger.info("Dynamic bid for {}: ${:.3f}/hr", provider, suggested)

        return p.bid(config)

    def cancel_bid(self, provider: str, instance_id: str) -> bool:
        """Cancel / terminate an instance on *provider*."""
        p = self._get_provider(provider)
        return p.cancel_bid(instance_id)

    def suggest_bid_price(
        self,
        gpu_type: str,
        on_demand_price: float | None = None,
    ) -> float | None:
        """Return a recommended bid price for *gpu_type* without placing it."""
        return self._price_tracker.suggest_bid(gpu_type, on_demand_price)

    def price_summary(self, gpu_type: str) -> dict[str, Any]:
        """Return diagnostic price information for *gpu_type*."""
        return self._price_tracker.describe(gpu_type)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release all provider HTTP resources."""
        for prov_name, p in self._providers.items():
            try:
                p.close()
            except Exception as exc:
                logger.debug("Error closing provider {}: {}", prov_name, exc)
        self._providers.clear()
        self._closed = True

    def __enter__(self) -> GPUSpotMarket:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# SpotOrchestrator  -- high-level orchestration
# ---------------------------------------------------------------------------


class SpotOrchestrator:
    """High-level spot GPU orchestration.

    Finds the cheapest GPU instances across providers, launches
    clusters, monitors costs, and can migrate workloads between
    providers.

    Parameters
    ----------
    market:
        ``GPUSpotMarket`` instance to use.  A new one is created when
        ``None``.
    """

    def __init__(self, market: GPUSpotMarket | None = None) -> None:
        self._market = market or GPUSpotMarket()
        # Track launched instances for cost monitoring
        self._running: dict[str, GPUInstance] = {}  # instance_id -> GPUInstance
        self._start_times: dict[str, float] = {}  # instance_id -> timestamp

    # ------------------------------------------------------------------
    # Finding instances
    # ------------------------------------------------------------------

    def find_cheapest(
        self,
        gpu_type: str,
        max_price: float,
        region: str | None = None,
        providers: list[str] | None = None,
        min_gpu_count: int = 1,
    ) -> list[GPUInstance]:
        """Find the cheapest GPU instances matching criteria across providers.

        Parameters
        ----------
        gpu_type:
            GPU model substring to match (e.g. "RTX 4090", "A100").
        max_price:
            Maximum acceptable price per hour in USD.
        region:
            Optional region filter.
        providers:
            Subset of providers to search (default: all supported).
        min_gpu_count:
            Minimum number of GPUs per instance.

        Returns
        -------
        list[GPUInstance]
            Matching instances sorted by price ascending.
        """
        all_instances = self._market.list_all(region=region, gpu_type=gpu_type, providers=providers)

        candidates: list[GPUInstance] = []
        for prov_list in all_instances.values():
            for inst in prov_list:
                if inst.price_per_hour > max_price:
                    continue
                if inst.gpu_count < min_gpu_count:
                    continue
                if region and region.lower() not in inst.region.lower():
                    continue
                candidates.append(inst)

        candidates.sort(key=lambda i: i.price_per_hour)
        logger.info(
            "find_cheapest(gpu_type={!r}, max_price={}): {} candidates",
            gpu_type,
            max_price,
            len(candidates),
        )
        return candidates

    # ------------------------------------------------------------------
    # Launching clusters
    # ------------------------------------------------------------------

    def launch_cluster(self, instances: list[GPUInstance]) -> list[BidResult]:
        """Launch (bid on) a set of instances to form a cluster.

        Parameters
        ----------
        instances:
            GPU instances to bid on (from ``find_cheapest``).

        Returns
        -------
        list[BidResult]
            Results of each bid, in the same order as *instances*.
        """
        results: list[BidResult] = []
        for inst in instances:
            config: dict[str, Any] = {
                "gpu_type": inst.gpu_type,
                "max_price": inst.price_per_hour,
                "region": inst.region,
            }
            result = self._market.bid(inst.provider, config)
            results.append(result)

            if result.status in (BidStatus.ACTIVE, BidStatus.PENDING) and result.instance_id:
                self._running[result.instance_id] = inst
                self._start_times[result.instance_id] = time.time()

        accepted = sum(1 for r in results if r.status in (BidStatus.ACTIVE, BidStatus.PENDING))
        logger.info("launch_cluster: {}/{} accepted", accepted, len(instances))
        return results

    # ------------------------------------------------------------------
    # Cost monitoring
    # ------------------------------------------------------------------

    def monitor_costs(self, refresh_instances: bool = False) -> list[CostReport]:
        """Return a cost report for all tracked running instances.

        Parameters
        ----------
        refresh_instances:
            If ``True``, query each provider for current instance status
            to compute accurate elapsed time.  Otherwise uses local
            start-time tracking.

        Returns
        -------
        list[CostReport]
            One report per active instance.
        """
        reports: list[CostReport] = []
        now = time.time()

        for inst_id, inst in list(self._running.items()):
            start = self._start_times.get(inst_id, now)
            hours = (now - start) / 3600.0
            total_cost = hours * inst.price_per_hour

            if inst.on_demand_price > 0:
                on_demand_cost = hours * inst.on_demand_price
                savings = max(0.0, on_demand_cost - total_cost)
            else:
                savings = 0.0

            reports.append(
                CostReport(
                    provider=inst.provider,
                    instance_id=inst_id,
                    gpu_type=inst.gpu_type,
                    hours_running=round(hours, 3),
                    total_cost=round(total_cost, 4),
                    spot_savings=round(savings, 4),
                )
            )

        # Optionally refresh from providers
        if refresh_instances:
            self._refresh_running()

        return reports

    def _refresh_running(self) -> None:
        """Query providers for running instances and update local state."""
        for prov in list(_PROVIDER_MAP):
            try:
                instances = self._market.list_instances(provider=prov)
            except Exception as exc:
                logger.debug("Refresh failed for {}: {}", prov, exc)
                continue

            live_ids = {i.instance_id for i in instances if i.status == InstanceStatus.RUNNING}
            # Remove terminated instances
            for inst_id in list(self._running):
                if inst_id not in live_ids:
                    logger.info("Instance {} no longer running, removing from tracker", inst_id)
                    self._running.pop(inst_id, None)
                    self._start_times.pop(inst_id, None)

    # ------------------------------------------------------------------
    # Provider swap
    # ------------------------------------------------------------------

    def swap_providers(
        self,
        current: str,
        target: str,
        dry_run: bool = False,
    ) -> list[BidResult]:
        """Migrate all running instances from *current* to *target* provider.

        Strategy
        --------
        1. Identify all running instances on the *current* provider.
        2. Find comparable instances on the *target* provider.
        3. Bid on *target* instances (if ``dry_run=False``).
        4. Track the new instances internally.
        5. Cancel / terminate the old instances.

        Parameters
        ----------
        current:
            Provider to migrate from (e.g. "runpod").
        target:
            Provider to migrate to (e.g. "vast").
        dry_run:
            If ``True``, only search for replacements without placing
            bids or cancelling anything.

        Returns
        -------
        list[BidResult]
            Results of migration bids on the *target* provider.
        """
        # Find instances to migrate
        migrating = [
            inst for inst_id, inst in self._running.items()
            if inst.provider == current
        ]

        if not migrating:
            logger.info("No running instances on {} to migrate", current)
            return []

        logger.info("swap_providers: migrating {} instance(s) from {} to {}", len(migrating), current, target)

        # For each migrating instance, find a comparable replacement on target
        target_instances: list[GPUInstance] = []
        for inst in migrating:
            replacements = self.find_cheapest(
                gpu_type=inst.gpu_type,
                max_price=inst.price_per_hour * 1.2,  # allow 20% premium
                providers=[target],
                min_gpu_count=inst.gpu_count,
            )
            if replacements:
                target_instances.append(replacements[0])
            else:
                logger.warning(
                    "No replacement found for {} ({}) on {}",
                    inst.instance_id,
                    inst.gpu_type,
                    target,
                )

        if not target_instances:
            logger.warning("No suitable replacements found on {}", target)
            return []

        if dry_run:
            logger.info("Dry-run: would bid on {} instance(s) on {}", len(target_instances), target)
            return []

        # Bid on target instances
        bid_results = self.launch_cluster(target_instances)

        # Cancel old instances (best-effort)
        for inst in migrating:
            success = self._market.cancel_bid(current, inst.instance_id)
            if success:
                self._running.pop(inst.instance_id, None)
                self._start_times.pop(inst.instance_id, None)
                logger.info("Cancelled instance {} on {}", inst.instance_id, current)
            else:
                logger.warning("Failed to cancel instance {} on {}", inst.instance_id, current)

        return bid_results

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def running_instances(self) -> list[GPUInstance]:
        """Return a list of currently tracked running instances."""
        return list(self._running.values())

    def summary(self) -> dict[str, Any]:
        """Return a high-level summary of the orchestrator state."""
        reports = self.monitor_costs()
        total_cost = sum(r.total_cost for r in reports)
        total_savings = sum(r.spot_savings for r in reports)
        return {
            "running_count": len(self._running),
            "total_cost_usd": round(total_cost, 4),
            "total_spot_savings_usd": round(total_savings, 4),
            "providers_in_use": list({r.provider for r in reports}),
            "reports": reports,
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release resources held by the underlying market."""
        self._market.close()

    def __enter__(self) -> SpotOrchestrator:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__all__ = [
    # Enums
    "Provider",
    "InstanceStatus",
    "BidStatus",
    # Data types
    "GPUInstance",
    "BidResult",
    "CostReport",
    # Reference
    "GPU_ON_DEMAND_REFERENCE",
    # Classes
    "GPUSpotMarket",
    "SpotOrchestrator",
]
