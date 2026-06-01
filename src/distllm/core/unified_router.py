"""Unified Router — converges CrossCloudRouter and Marketplace into one.

Merges commercial cloud instances (AWS/GCP/Azure) with peer-to-peer
GPU marketplace listings into a single routing decision. Users get one
sorted list of all available compute options ranked by their preferences
(price, latency, carbon, reputation).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from loguru import logger


class ComputeSource(Enum):
    """Where the compute comes from."""
    CLOUD = "cloud"       # Commercial cloud provider
    PEER = "peer"         # Peer-to-peer marketplace listing
    FEDERATED = "federated"  # Federated cluster node


@dataclass
class ComputeOption:
    """A unified compute option — either cloud or peer."""
    source: ComputeSource
    provider_name: str
    instance_type: str
    region: str = ""

    # Hardware
    gpu_type: str = ""
    gpu_count: int = 1
    gpu_memory_gb: float = 0.0

    # Pricing
    price_per_hour: float = 0.0
    spot_price: float = 0.0
    price_per_million_tokens: float = 0.0

    # Performance
    latency_ms: float = 0.0
    bandwidth_mbps: float = 0.0

    # Trust
    reputation_score: float = 0.5
    uptime_pct: float = 100.0

    # Carbon
    carbon_intensity: float = 0.0  # gCO2/kWh

    # Availability
    available: bool = True
    max_concurrent: int = 1
    current_load: int = 0

    # Metadata
    listing_id: str = ""
    tags: list[str] = field(default_factory=list)

    @property
    def effective_spot_price(self) -> float:
        return self.spot_price if self.spot_price > 0 else self.price_per_hour

    @property
    def is_available(self) -> bool:
        return self.available and self.current_load < self.max_concurrent

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source.value,
            "provider": self.provider_name,
            "instance_type": self.instance_type,
            "region": self.region,
            "gpu_type": self.gpu_type,
            "gpu_count": self.gpu_count,
            "gpu_memory_gb": self.gpu_memory_gb,
            "price_per_hour": self.price_per_hour,
            "spot_price": self.spot_price,
            "latency_ms": self.latency_ms,
            "reputation": self.reputation_score,
            "carbon_gco2_kwh": self.carbon_intensity,
            "available": self.is_available,
            "listing_id": self.listing_id,
        }


@dataclass
class UnifiedRouteDecision:
    """A routing decision from the unified router."""
    selected: ComputeOption
    all_candidates: list[ComputeOption]
    scoring_method: str = "price"
    score: float = 0.0
    reason: str = ""


class UnifiedRouter:
    """Routes to the cheapest compute across cloud providers AND peer marketplace.

    Combines CrossCloudRouter options with Marketplace listings into a
    single sorted decision surface.

    Usage::

        router = UnifiedRouter()
        router.set_cloud_options(pricing_manager.get_all_pricing())
        router.set_peer_options(marketplace.list_listings())
        decision = router.route(gpu_type="A100", max_price=10.0)
    """

    def __init__(self):
        self._cloud_options: list[ComputeOption] = []
        self._peer_options: list[ComputeOption] = []
        self._latency_cache: dict[str, float] = {}
        self._stats = {"routes": 0, "cloud_selected": 0, "peer_selected": 0}

    def set_cloud_options(self, cloud_prices: list[Any]) -> None:
        """Set cloud compute options from pricing data.

        Accepts InstancePricing objects or dicts with compatible fields.
        """
        options = []
        for p in cloud_prices:
            if hasattr(p, "to_dict"):
                d = p.to_dict() if hasattr(p, "to_dict") else p.__dict__
            elif isinstance(p, dict):
                d = p
            else:
                d = p.__dict__
            options.append(ComputeOption(
                source=ComputeSource.CLOUD,
                provider_name=d.get("provider", d.get("name", "")),
                instance_type=d.get("instance_type", ""),
                region=d.get("region", ""),
                gpu_type=d.get("gpu_type", ""),
                gpu_count=d.get("gpu_count", 1),
                gpu_memory_gb=d.get("gpu_memory_gb", 0.0),
                price_per_hour=d.get("on_demand_price", d.get("price_per_hour", 0.0)),
                spot_price=d.get("spot_price", 0.0),
                latency_ms=self._latency_cache.get(d.get("provider", ""), 50.0),
                carbon_intensity=d.get("carbon_intensity", 0.0),
                available=True,
            ))
        self._cloud_options = options

    def set_peer_options(self, listings: list[Any]) -> None:
        """Set peer compute options from marketplace listings.

        Accepts GPUListing objects or dicts with compatible fields.
        """
        options = []
        for l in listings:
            if hasattr(l, "listing_id"):
                options.append(ComputeOption(
                    source=ComputeSource.PEER,
                    provider_name=getattr(l, "provider_name", getattr(l, "provider_id", "")),
                    instance_type=getattr(l, "gpu_name", ""),
                    region=getattr(l, "region", ""),
                    gpu_type=getattr(l, "gpu_name", ""),
                    gpu_count=getattr(l, "gpu_count", 1),
                    gpu_memory_gb=getattr(l, "gpu_memory_bytes", 0) / (1024**3),
                    price_per_hour=getattr(l, "price_per_hour", 0.0),
                    spot_price=0.0,
                    latency_ms=getattr(l, "latency_ms", 0.0),
                    reputation_score=getattr(l, "reputation_score", 0.5),
                    uptime_pct=getattr(l, "uptime_pct", 100.0),
                    available=getattr(l, "is_available", True),
                    max_concurrent=getattr(l, "max_concurrent_jobs", 1),
                    current_load=getattr(l, "current_jobs", 0),
                    listing_id=getattr(l, "listing_id", ""),
                ))
            elif isinstance(l, dict):
                options.append(ComputeOption(
                    source=ComputeSource.PEER,
                    provider_name=l.get("provider_name", l.get("provider_id", "")),
                    instance_type=l.get("gpu_name", ""),
                    region=l.get("region", ""),
                    gpu_type=l.get("gpu_name", ""),
                    gpu_count=l.get("gpu_count", 1),
                    gpu_memory_gb=l.get("gpu_memory_bytes", 0) / (1024**3),
                    price_per_hour=l.get("price_per_hour", 0.0),
                    spot_price=0.0,
                    latency_ms=l.get("latency_ms", 0.0),
                    reputation_score=l.get("reputation_score", 0.5),
                    uptime_pct=l.get("uptime_pct", 100.0),
                    available=l.get("is_available", True),
                    listing_id=l.get("listing_id", ""),
                ))
        self._peer_options = options

    def set_latency(self, provider: str, latency_ms: float) -> None:
        """Update latency for a provider."""
        self._latency_cache[provider] = latency_ms

    def route(
        self,
        gpu_type: str = "",
        min_gpu_memory_gb: float = 0.0,
        max_price: float = float("inf"),
        max_latency_ms: float = 5000.0,
        min_reputation: float = 0.0,
        prefer_spot: bool = True,
        source_filter: ComputeSource | None = None,
        scoring: str = "price",
        carbon_weight: float = 0.0,
    ) -> UnifiedRouteDecision | None:
        """Route to the best compute option across all sources.

        Args:
            gpu_type: Required GPU type (e.g., "A100"). Empty = any.
            min_gpu_memory_gb: Minimum GPU memory in GB.
            max_price: Maximum price per hour.
            max_latency_ms: Maximum acceptable latency.
            min_reputation: Minimum reputation score (peer only).
            prefer_spot: Prefer spot pricing when available.
            source_filter: Only consider one source (cloud/peer).
            scoring: Scoring method: "price", "carbon", "balanced".
            carbon_weight: Weight for carbon in balanced scoring (0-1).

        Returns:
            UnifiedRouteDecision or None if no candidates.
        """
        self._stats["routes"] += 1
        candidates = []

        options = self._get_all_options(source_filter)

        for opt in options:
            if not opt.is_available:
                continue
            if gpu_type and gpu_type.lower() not in opt.gpu_type.lower():
                continue
            if min_gpu_memory_gb > 0 and opt.gpu_memory_gb < min_gpu_memory_gb:
                continue
            price = opt.effective_spot_price if prefer_spot else opt.price_per_hour
            if price > max_price or price <= 0:
                continue
            latency = opt.latency_ms or self._latency_cache.get(opt.provider_name, 50.0)
            if latency > max_latency_ms:
                continue
            if min_reputation > 0 and opt.source == ComputeSource.PEER:
                if opt.reputation_score < min_reputation:
                    continue
            candidates.append(opt)

        if not candidates:
            return None

        candidates = self._score_candidates(candidates, scoring, carbon_weight, prefer_spot, max_latency_ms)
        best = candidates[0]

        if best.source == ComputeSource.CLOUD:
            self._stats["cloud_selected"] += 1
        else:
            self._stats["peer_selected"] += 1

        price = best.effective_spot_price if prefer_spot else best.price_per_hour
        reason = f"{best.source.value}: {best.provider_name}/{best.instance_type} at ${price:.2f}/hr"

        return UnifiedRouteDecision(
            selected=best,
            all_candidates=candidates,
            scoring_method=scoring,
            reason=reason,
        )

    def _get_all_options(self, source_filter: ComputeSource | None = None) -> list[ComputeOption]:
        if source_filter == ComputeSource.CLOUD:
            return list(self._cloud_options)
        if source_filter == ComputeSource.PEER:
            return list(self._peer_options)
        return list(self._cloud_options) + list(self._peer_options)

    def _score_candidates(
        self,
        candidates: list[ComputeOption],
        scoring: str,
        carbon_weight: float,
        prefer_spot: bool,
        max_latency_ms: float,
    ) -> list[ComputeOption]:
        if scoring == "price":
            candidates.sort(key=lambda c: c.effective_spot_price if prefer_spot else c.price_per_hour)
        elif scoring == "carbon":
            candidates.sort(key=lambda c: c.carbon_intensity if c.carbon_intensity > 0 else 9999)
        elif scoring == "balanced":
            prices = [c.effective_spot_price if prefer_spot else c.price_per_hour for c in candidates]
            carbons = [c.carbon_intensity for c in candidates]
            min_p, max_p = min(prices), max(prices)
            min_c, max_c = min(carbons) if min(carbons) > 0 else 0, max(carbons) if max(carbons) > 0 else 1
            p_range = max_p - min_p if max_p > min_p else 1.0
            c_range = max_c - min_c if max_c > min_c else 1.0

            def score(c: ComputeOption) -> float:
                p = c.effective_spot_price if prefer_spot else c.price_per_hour
                norm_p = (p - min_p) / p_range
                norm_c = (c.carbon_intensity - min_c) / c_range if c.carbon_intensity > 0 else 0.5
                norm_l = (c.latency_ms or 50.0) / max_latency_ms if max_latency_ms > 0 else 0
                return (1 - carbon_weight) * (norm_p * 0.7 + norm_l * 0.3) + carbon_weight * norm_c

            candidates.sort(key=score)
        else:
            candidates.sort(key=lambda c: c.effective_spot_price if prefer_spot else c.price_per_hour)
        return candidates

    def get_all_options(self) -> list[dict[str, Any]]:
        """Get all available compute options as dicts."""
        options = self._get_all_options()
        return [o.to_dict() for o in options if o.is_available]

    @property
    def stats(self) -> dict[str, int]:
        return dict(self._stats)


class RequestPhase(str, Enum):
    """Phase of a request lifecycle."""
    PREFILL = "prefill"   # Compute-bound: process prompt tokens
    DECODE = "decode"     # Memory-bound: generate tokens one-by-one


class DisaggregatedRouter:
    """Routes requests to separate prefill and decode worker pools.

    In disaggregated P&D architecture:
    - Prefill nodes: optimized for compute-bound prompt processing (high FLOPS)
    - Decode nodes: optimized for memory-bound autoregressive generation (high bandwidth)

    This router tracks which nodes are prefill-optimized and which are
    decode-optimized, and routes requests accordingly.

    Usage::

        router = DisaggregatedRouter()
        router.set_prefill_nodes(["gpu-0", "gpu-1"])   # A100s for prefill
        router.set_decode_nodes(["gpu-2", "gpu-3"])     # H100s for decode

        prefill_node = router.route(RequestPhase.PREFILL)
        decode_node = router.route(RequestPhase.DECODE)
    """

    def __init__(self) -> None:
        self._prefill_nodes: list[str] = []
        self._decode_nodes: list[str] = []
        self._prefill_load: dict[str, int] = {}
        self._decode_load: dict[str, int] = {}
        self._lock = threading.Lock()
        self._stats = {"prefill_routes": 0, "decode_routes": 0, "fallback_routes": 0}

    def set_prefill_nodes(self, node_ids: list[str]) -> None:
        """Set nodes optimized for prefill (compute-bound)."""
        with self._lock:
            self._prefill_nodes = list(node_ids)
            self._prefill_load = {nid: 0 for nid in node_ids}

    def set_decode_nodes(self, node_ids: list[str]) -> None:
        """Set nodes optimized for decode (memory-bound)."""
        with self._lock:
            self._decode_nodes = list(node_ids)
            self._decode_load = {nid: 0 for nid in node_ids}

    def route(self, phase: RequestPhase) -> str | None:
        """Route to the least-loaded node for the given phase.

        Returns:
            Node ID, or None if no nodes available.
        """
        with self._lock:
            if phase == RequestPhase.PREFILL:
                self._stats["prefill_routes"] += 1
                nodes = self._prefill_nodes
                load = self._prefill_load
                fallback_load = self._decode_load
                fallback_nodes = self._decode_nodes
            else:
                self._stats["decode_routes"] += 1
                nodes = self._decode_nodes
                load = self._decode_load
                fallback_load = self._prefill_load
                fallback_nodes = self._prefill_nodes

            # Try preferred pool first
            if nodes:
                best = min(nodes, key=lambda n: load.get(n, 0))
                load[best] = load.get(best, 0) + 1
                return best

            # Fallback to other pool
            if fallback_nodes:
                self._stats["fallback_routes"] += 1
                best = min(fallback_nodes, key=lambda n: fallback_load.get(n, 0))
                fallback_load[best] = fallback_load.get(best, 0) + 1
                return best

            return None

    def release(self, node_id: str, phase: RequestPhase) -> None:
        """Release a slot on a node after request completion."""
        with self._lock:
            if phase == RequestPhase.PREFILL:
                self._prefill_load[node_id] = max(0, self._prefill_load.get(node_id, 1) - 1)
            else:
                self._decode_load[node_id] = max(0, self._decode_load.get(node_id, 1) - 1)

    def get_pool_sizes(self) -> dict[str, int]:
        """Return the size of each pool."""
        with self._lock:
            return {
                "prefill_nodes": len(self._prefill_nodes),
                "decode_nodes": len(self._decode_nodes),
                "total_nodes": len(self._prefill_nodes) + len(self._decode_nodes),
            }

    @property
    def stats(self) -> dict[str, int]:
        with self._lock:
            return dict(self._stats)


# Need threading for DisaggregatedRouter
import threading
