"""Cost surface modelling and arbitrage orchestration.

Provides a high-resolution view of GPU compute costs across providers,
regions, and instance types, plus an orchestrator that selects the optimal
deployment target given a workload profile and SLA budget.

Integrates with:
- :mod:`distllm.dist.cloud_selector` for region-level pricing fallbacks.
- :mod:`distllm.dist.geo` for latency estimates.
- :mod:`distllm.dist.marketplace` for listing-based pricing.
- :mod:`distllm.core.pricing_providers` for live cloud API pricing.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from loguru import logger

# ──────────────────────────────────────────────────────────────────────
#  CostSurfacePoint
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CostSurfacePoint:
    """A single point on the cost surface — one (region, provider, instance) snapshot.

    Attributes:
        region: Cloud region identifier (e.g. ``"us-east-1"``).
        provider: Cloud provider name (e.g. ``"aws"``, ``"gcp"``, ``"azure"``).
        instance_type: GPU instance type (e.g. ``"p4d.24xlarge"``).
        spot_price: Current spot/preemptible hourly price in USD.
        carbon_intensity: Grid carbon intensity in gCO\\ :sub:`2`\\ /kWh.
        queue_depth: Number of pending GPU allocations at this point.
        avg_latency_ms: Estimated network round-trip latency in milliseconds.
        reliability_pct: Historical up-time percentage (0.0‑100.0).
        hour: Hour of the day this snapshot applies to (0‑23, UTC).
    """

    region: str
    provider: str
    instance_type: str
    spot_price: float = 0.0
    carbon_intensity: float = 0.0
    queue_depth: int = 0
    avg_latency_ms: float = 0.0
    reliability_pct: float = 99.9
    hour: int = 0


# ──────────────────────────────────────────────────────────────────────
#  CostSurface
# ──────────────────────────────────────────────────────────────────────


class CostSurface:
    """A multi-dimensional cost surface for GPU compute arbitrage.

    Maintains a list of :class:`CostSurfacePoint` snapshots and provides
    filtering, sorting, and live-update capabilities.

    Usage::

        surface = CostSurface()
        surface.update_from_providers()
        cheap = surface.sort_by_cost()
        green = surface.sort_by_carbon()
        fast = surface.sort_by_latency()

        candidates = surface.query(min_gpus=4, max_price=12.0)
        print(f"Found {len(candidates)} suitable points")
    """

    def __init__(
        self,
        points: list[CostSurfacePoint] | None = None,
        pricing_providers: list[Any] | None = None,
    ) -> None:
        """Initialise the cost surface.

        Args:
            points: Optional initial point list; defaults to empty.
            pricing_providers: Optional list of
                :class:`distllm.core.pricing_providers.PricingProvider`
                instances used by :meth:`update_from_providers`.
        """
        self._points: list[CostSurfacePoint] = list(points) if points else []
        self._pricing_providers: list[Any] = list(pricing_providers) if pricing_providers else []
        self._lock = threading.Lock()
        self._last_update: float = 0.0

    # ── Public query API ──────────────────────────────────────────────

    def query(
        self,
        min_gpus: int = 1,
        max_price: float | None = None,
        max_carbon: float | None = None,
        max_latency_ms: float | None = None,
        min_reliability_pct: float | None = None,
        providers: list[str] | None = None,
        regions: list[str] | None = None,
        hours: list[int] | None = None,
    ) -> list[CostSurfacePoint]:
        """Return filtered points matching all specified criteria.

        Args:
            min_gpus: Minimum number of GPUs (checked via instance-type
                heuristic).  Defaults to 1.
            max_price: Maximum spot price in USD.  ``None`` = no limit.
            max_carbon: Maximum carbon intensity in gCO\\ :sub:`2`\\ /kWh.
                ``None`` = no limit.
            max_latency_ms: Maximum average latency in milliseconds.
                ``None`` = no limit.
            min_reliability_pct: Minimum reliability percentage.
                ``None`` = no limit.
            providers: Only include these providers.  ``None`` = all.
            regions: Only include these regions.  ``None`` = all.
            hours: Only include these hours.  ``None`` = all.

        Returns:
            Filtered list of :class:`CostSurfacePoint`.
        """
        with self._lock:
            candidates = list(self._points)

        if max_price is not None:
            candidates = [p for p in candidates if p.spot_price <= max_price]

        if max_carbon is not None:
            candidates = [p for p in candidates if p.carbon_intensity <= max_carbon]

        if max_latency_ms is not None:
            candidates = [p for p in candidates if p.avg_latency_ms <= max_latency_ms]

        if min_reliability_pct is not None:
            candidates = [p for p in candidates if p.reliability_pct >= min_reliability_pct]

        if providers is not None:
            provider_set = set(providers)
            candidates = [p for p in candidates if p.provider in provider_set]

        if regions is not None:
            region_set = set(regions)
            candidates = [p for p in candidates if p.region in region_set]

        if hours is not None:
            hour_set = set(hours)
            candidates = [p for p in candidates if p.hour in hour_set]

        # min_gpus: heuristic via instance-type GPU count lookup
        if min_gpus > 1:
            candidates = [
                p for p in candidates
                if _gpu_count_for_instance(p.instance_type) >= min_gpus
            ]

        return candidates

    # ── Live updates ──────────────────────────────────────────────────

    def update_from_providers(self) -> int:
        """Fetch latest pricing from configured pricing providers.

        Calls :meth:`fetch_pricing` on each registered
        :class:`~distllm.core.pricing_providers.PricingProvider`, converts
        results into points, and replaces the internal point list.

        If no providers are configured or all fail, the point list is
        left unchanged.

        Returns:
            Number of points after the update (0 if update failed).
        """
        if not self._pricing_providers:
            logger.warning("No pricing providers configured — cannot update")
            return len(self._points)

        all_points: list[CostSurfacePoint] = []
        for provider in self._pricing_providers:
            try:
                pricings = provider.fetch_pricing()
                for p in pricings:
                    all_points.append(
                        CostSurfacePoint(
                            region=p.region,
                            provider=p.provider,
                            instance_type=p.instance_type,
                            spot_price=p.spot_price,
                            carbon_intensity=0.0,     # filled by enrichment
                            queue_depth=0,             # filled by enrichment
                            avg_latency_ms=0.0,        # filled by enrichment
                            reliability_pct=99.9,
                            hour=_utc_hour(),
                        )
                    )
            except Exception as exc:
                logger.debug(f"Provider {provider} pricing fetch failed: {exc}")

        if not all_points:
            logger.warning("All pricing providers returned empty — surface unchanged")
            return len(self._points)

        with self._lock:
            self._points = all_points
            self._last_update = time.time()

        logger.info(f"Cost surface updated: {len(all_points)} points from "
                    f"{len(self._pricing_providers)} providers")
        return len(all_points)

    def enrich_point(
        self,
        enricher: Callable[[CostSurfacePoint], CostSurfacePoint],
    ) -> int:
        """Apply a callable enricher to every point in-place.

        Useful for filling in carbon intensity, latency, or queue-depth
        fields from external data sources after a pricing fetch.

        Args:
            enricher: A callable that receives a :class:`CostSurfacePoint`
                and returns a (possibly new) point with enriched fields.

        Returns:
            Number of points enriched.
        """
        with self._lock:
            updated = [enricher(p) for p in self._points]
            self._points = updated
        return len(updated)

    # ── Sort helpers ──────────────────────────────────────────────────

    def sort_by_cost(self, ascending: bool = True) -> list[CostSurfacePoint]:
        """Return points sorted by spot price.

        Args:
            ascending: Sort lowest price first when ``True``.

        Returns:
            Sorted (shallow-copied) list of points.
        """
        with self._lock:
            return sorted(
                self._points,
                key=lambda p: p.spot_price,
                reverse=not ascending,
            )

    def sort_by_carbon(self, ascending: bool = True) -> list[CostSurfacePoint]:
        """Return points sorted by carbon intensity.

        Args:
            ascending: Sort lowest carbon first when ``True``.

        Returns:
            Sorted (shallow-copied) list of points.
        """
        with self._lock:
            return sorted(
                self._points,
                key=lambda p: p.carbon_intensity,
                reverse=not ascending,
            )

    def sort_by_latency(self, ascending: bool = True) -> list[CostSurfacePoint]:
        """Return points sorted by average latency.

        Args:
            ascending: Sort lowest latency first when ``True``.

        Returns:
            Sorted (shallow-copied) list of points.
        """
        with self._lock:
            return sorted(
                self._points,
                key=lambda p: p.avg_latency_ms,
                reverse=not ascending,
            )

    # ── Properties ────────────────────────────────────────────────────

    @property
    def points(self) -> list[CostSurfacePoint]:
        """Return a shallow copy of all points."""
        with self._lock:
            return list(self._points)

    @property
    def last_update(self) -> float:
        """Unix timestamp of the last successful ``update_from_providers()``."""
        return self._last_update

    def stats(self) -> dict[str, Any]:
        """Return summary statistics for the current surface."""
        with self._lock:
            if not self._points:
                return {"count": 0}

            prices = [p.spot_price for p in self._points]
            carbons = [p.carbon_intensity for p in self._points]
            latencies = [p.avg_latency_ms for p in self._points]
            regions = len({p.region for p in self._points})
            providers = len({p.provider for p in self._points})

            return {
                "count": len(self._points),
                "regions": regions,
                "providers": providers,
                "price_min": min(prices),
                "price_max": max(prices),
                "price_avg": sum(prices) / len(prices),
                "carbon_min": min(carbons),
                "carbon_max": max(carbons),
                "latency_min": min(latencies),
                "latency_max": max(latencies),
            }


# ──────────────────────────────────────────────────────────────────────
#  ArbitrageOrchestrator
# ──────────────────────────────────────────────────────────────────────


class ArbitrageOrchestrator:
    """Selects the optimal (region, provider, instance) for a workload.

    Combines a :class:`CostSurface` with a workload profile and SLA budget
    to produce a deployment recommendation.  Tracks cost savings, carbon
    reduction, and decision latency as runtime metrics.

    Usage::

        orch = ArbitrageOrchestrator(cost_surface=cost_surface)
        best = orch.get_best_cluster(
            request_profile={"priority": "cost", "min_reliability": 95.0},
            sla_budget=15.0,
        )
        print(orch.metrics)
    """

    def __init__(
        self,
        cost_surface: CostSurface | None = None,
        reference_price: float = 0.0,
        reference_carbon: float = 0.0,
    ):
        """Initialise the orchestrator.

        Args:
            cost_surface: A :class:`CostSurface` instance.  Created empty
                if not provided.
            reference_price: On-demand hourly price used to compute
                ``cost_savings_pct``.  Defaults to 0.0 (savings disabled).
            reference_carbon: Baseline carbon intensity (gCO\\ :sub:`2`\\ /kWh)
                used to compute ``carbon_reduction_kg``.  Defaults to 0.0.
        """
        self._surface = cost_surface if cost_surface is not None else CostSurface()
        self._reference_price = reference_price
        self._reference_carbon = reference_carbon

        # Internal counters for metrics
        self._total_decisions: int = 0
        self._total_reference_cost: float = 0.0
        self._total_actual_cost: float = 0.0
        self._total_reference_carbon: float = 0.0
        self._total_actual_carbon: float = 0.0
        self._cumulative_decision_latency_ms: float = 0.0
        self._lock = threading.Lock()

    # ── Core selection ────────────────────────────────────────────────

    def get_best_cluster(
        self,
        request_profile: dict[str, Any] | None = None,
        sla_budget: float = float("inf"),
    ) -> tuple[str, str, str] | None:
        """Return the best (region, provider, instance) for a workload.

        Selection strategy is driven by ``request_profile``:

        * ``"cost"`` — minimise spot price.
        * ``"carbon"`` — minimise carbon intensity.
        * ``"latency"`` — minimise network latency.
        * ``"balanced"`` — weighted composite of all three (default).

        Args:
            request_profile: A dictionary that may contain:

                * ``priority`` (``"cost"`` | ``"carbon"`` | ``"latency"`` |
                  ``"balanced"``) — selection criterion.
                * ``max_price`` (float) — maximum acceptable spot price.
                * ``max_carbon`` (float) — maximum acceptable carbon
                  intensity.
                * ``max_latency_ms`` (float) — maximum acceptable latency.
                * ``min_reliability`` (float) — minimum reliability
                  percentage (0‑100).
                * ``min_gpus`` (int) — minimum number of GPUs.

            sla_budget: Maximum allowable spot price per hour in USD.
                Overrides ``max_price`` in the profile when lower.

        Returns:
            ``(region, provider, instance_type)`` or ``None`` if no
            feasible point exists.
        """
        profile = request_profile or {}
        priority = profile.get("priority", "balanced")

        # Build filter kwargs from profile + SLA budget.
        max_price = profile.get("max_price")
        if sla_budget < float("inf"):
            max_price = min(max_price or sla_budget, sla_budget)  # type: ignore[type-var]

        candidates = self._surface.query(
            min_gpus=profile.get("min_gpus", 1),
            max_price=max_price,
            max_carbon=profile.get("max_carbon"),
            max_latency_ms=profile.get("max_latency_ms"),
            min_reliability_pct=profile.get("min_reliability"),
            providers=profile.get("providers"),
            regions=profile.get("regions"),
            hours=profile.get("hours"),
        )

        if not candidates:
            logger.warning("No candidate points satisfy the request profile")
            return None

        start = time.perf_counter()

        if priority == "cost":
            candidates.sort(key=lambda p: p.spot_price)
        elif priority == "carbon":
            candidates.sort(key=lambda p: p.carbon_intensity)
        elif priority == "latency":
            candidates.sort(key=lambda p: p.avg_latency_ms)
        else:
            # Balanced: normalise each dimension and weight equally.
            candidates.sort(
                key=lambda p: _balanced_score(p, candidates),
            )

        best = candidates[0]
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        # Update metrics.
        with self._lock:
            self._total_decisions += 1
            self._total_reference_cost += self._reference_price
            self._total_actual_cost += best.spot_price
            self._total_reference_carbon += self._reference_carbon
            self._total_actual_carbon += best.carbon_intensity
            self._cumulative_decision_latency_ms += elapsed_ms

        logger.info(
            f"Best cluster: {best.region}/{best.provider}/{best.instance_type} "
            f"@ ${best.spot_price:.4f}/hr "
            f"({best.carbon_intensity:.1f} gCO2/kWh, {best.avg_latency_ms:.0f}ms) "
            f"[priority={priority}, budget=${sla_budget:.2f}]"
        )
        return (best.region, best.provider, best.instance_type)

    # ── Metrics ───────────────────────────────────────────────────────

    @property
    def metrics(self) -> dict[str, float]:
        """Cumulative orchestrator metrics.

        Returns:
            A dictionary with keys:

            * ``total_decisions`` — number of :meth:`get_best_cluster` calls.
            * ``cost_savings_pct`` — percentage saved relative to
              *reference_price* (0.0 if no decisions or reference is 0).
            * ``carbon_reduction_kg`` — total gCO\\ :sub:`2`\\ saved
              converted to kg (0.0 if no decisions).
            * ``decision_latency_ms`` — average decision latency in
              milliseconds.
        """
        with self._lock:
            if self._total_decisions == 0:
                return {
                    "total_decisions": 0.0,
                    "cost_savings_pct": 0.0,
                    "carbon_reduction_kg": 0.0,
                    "decision_latency_ms": 0.0,
                }

            cost_savings = 0.0
            if self._reference_price > 0 and self._total_reference_cost > 0:
                cost_savings = (
                    (self._total_reference_cost - self._total_actual_cost)
                    / self._total_reference_cost
                ) * 100.0

            carbon_kg = (
                max(self._total_reference_carbon - self._total_actual_carbon, 0.0)
                / 1000.0
            )

            return {
                "total_decisions": float(self._total_decisions),
                "cost_savings_pct": round(cost_savings, 2),
                "carbon_reduction_kg": round(carbon_kg, 4),
                "decision_latency_ms": round(
                    self._cumulative_decision_latency_ms / self._total_decisions, 2
                ),
            }

    @property
    def surface(self) -> CostSurface:
        """The underlying :class:`CostSurface` instance."""
        return self._surface


# ── Internal helpers ────────────────────────────────────────────────


def _gpu_count_for_instance(instance_type: str) -> int:
    """Heuristic GPU count lookup based on instance type name patterns."""
    # Known high-GPU-count instance patterns
    high_count_patterns: dict[str, int] = {
        "p5.": 8, "p4d.": 8, "p4de.": 8,
        "p3.": 8,  # 16xlarge has 8, smaller p3 variants are fewer
        "a3-": 8, "a2-highgpu-8g": 8, "a2-highgpu-4g": 4,
        "a2-highgpu-2g": 2, "a2-highgpu-1g": 1,
        "a2-ultragpu-8g": 8, "a2-ultragpu-4g": 4,
        "ND H100": 8, "ND96": 8,
        "NC24": 1, "NC48": 2, "NC96": 4,
        "g5.12": 4, "g5.4": 1, "g5.x": 1, "g5.2": 1,
        "g6.x": 1, "g6.2": 1,
        "g2-standard": 1,
    }
    for pattern, count in high_count_patterns.items():
        if instance_type.startswith(pattern):
            return count
    # Default: assume a single GPU.
    return 1


def _balanced_score(point: CostSurfacePoint, pool: list[CostSurfacePoint]) -> float:
    """Compute a normalised balanced score (lower is better).

    Each dimension is min-max normalised across *pool* so that
    cost, carbon, and latency contribute equally.
    """
    prices = [p.spot_price for p in pool]
    carbons = [p.carbon_intensity for p in pool]
    latencies = [p.avg_latency_ms for p in pool]

    def _norm(value: float, values: list[float]) -> float:
        lo, hi = min(values), max(values)
        if hi <= lo:
            return 0.0
        return (value - lo) / (hi - lo)

    return (
        _norm(point.spot_price, prices)
        + _norm(point.carbon_intensity, carbons)
        + _norm(point.avg_latency_ms, latencies)
    ) / 3.0


def _utc_hour() -> int:
    """Current UTC hour (0‑23)."""
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).hour
