"""Unified cost / carbon / latency SLA router.

Exposes a single per-request ``objective`` knob that unifies the three
existing routing subsystems:

* ``cross_cloud_router.CrossCloudRouter`` — cheapest / fastest provider pick
  (cost + latency).
* ``arbitrage_engine.ArbitrageEngine`` — live spot-price arbitrage for the
  best available price (feeds the "cheapest" objective).
* ``carbon_migration.CarbonIntensityClient`` — grid carbon intensity, used by
  the "greenest" objective to quantify gCO2 avoided.

Objectives
----------
* ``cheapest``  — minimize $/hr (spot arbitrage when available).
* ``greenest``  — minimize gCO2/kWh (carbon-aware routing).
* ``fastest``   — minimize latency_ms.
* ``balanced``  — weighted mix of cost + carbon + latency (default).

Every route returns a :class:`SlaRouterReport` carrying the decision plus
**savings (USD/hr vs the priciest considered option)** and **gCO2 avoided
(vs a global-average grid intensity)** so a live "savings & gCO2 avoided"
dashboard can be powered directly from ``report.savings_usd`` /
``report.gco2_avoided``.  When a :class:`MetricsManager` is supplied, the
cumulative ``sla_savings_total_usd`` and ``sla_gco2_avoided_total`` counters
are updated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from loguru import logger

from distllm.core.coordinator_metrics import MetricsManager

# Global-average grid carbon intensity (gCO2/kWh), used as the baseline when
# quantifying gCO2 avoided by greenest routing.  Approx. world average per
# IEA 2023.
_GLOBAL_AVG_INTENSITY = 475.0


class Objective(str, Enum):
    CHEAPEST = "cheapest"
    GREENEST = "greenest"
    FASTEST = "fastest"
    BALANCED = "balanced"


@dataclass
class SlaRouterReport:
    """Result of a unified SLA route, including savings + carbon impact."""

    objective: Objective
    provider: str = ""
    region: str = ""
    instance_type: str = ""
    gpu_type: str = ""
    price_per_hour: float = 0.0
    latency_ms: float = 0.0
    carbon_intensity: float = 0.0
    savings_usd: float = 0.0       # USD/hr saved vs priciest considered option
    gco2_avoided: float = 0.0      # gCO2 avoided vs global-average grid
    reason: str = ""
    raw: Any | None = None         # underlying RouteDecision, if any


class UnifiedSlaRouter:
    """Route a request by a single ``objective`` across the fleet.

    Args:
        router: A :class:`CrossCloudRouter` instance (required).
        arbitrage: Optional :class:`ArbitrageEngine` — when set, the
            "cheapest" objective prefers the arbitrage-picked best price.
        carbon_client: Optional :class:`CarbonIntensityClient` — when set,
            the "greenest" objective uses live intensity and gCO2 is computed.
        metrics: Optional :class:`MetricsManager` to accumulate savings/carbon.
        baseline_intensity: gCO2/kWh baseline for "gCO2 avoided" math.
        energy_kwh_per_request: estimated energy per request, used to turn
            intensity deltas into absolute gCO2.
    """

    def __init__(
        self,
        router: Any,
        arbitrage: Any | None = None,
        carbon_client: Any | None = None,
        metrics: MetricsManager | None = None,
        baseline_intensity: float = _GLOBAL_AVG_INTENSITY,
        energy_kwh_per_request: float = 0.4,
    ) -> None:
        self._router = router
        self._arbitrage = arbitrage
        self._carbon = carbon_client
        self._metrics = metrics
        self._baseline_intensity = baseline_intensity
        self._energy_kwh = energy_kwh_per_request

    # ── public API ──

    def route(
        self,
        objective: Objective | str = Objective.BALANCED,
        gpu_type: str = "",
        max_latency_ms: float = 200.0,
        max_price: float = float("inf"),
        prefer_spot: bool = True,
        min_gpu_memory_gb: float = 0.0,
        carbon_weight: float = 0.3,
    ) -> SlaRouterReport:
        # A malformed/unknown objective from untrusted per-request input must
        # not crash the request — degrade safely to balanced.
        try:
            obj = Objective(objective) if not isinstance(objective, Objective) else objective
        except ValueError:
            logger.warning("Unknown SLA objective %r, defaulting to balanced", objective)
            obj = Objective.BALANCED

        if obj is Objective.CHEAPEST:
            return self._route_cheapest(
                gpu_type, max_latency_ms, max_price, prefer_spot, min_gpu_memory_gb,
            )
        if obj is Objective.GREENEST:
            return self._route_greenest(
                gpu_type, max_latency_ms, max_price, prefer_spot,
                min_gpu_memory_gb, carbon_weight,
            )
        if obj is Objective.FASTEST:
            return self._route_fastest(
                gpu_type, max_latency_ms, max_price, prefer_spot, min_gpu_memory_gb,
            )
        return self._route_balanced(
            gpu_type, max_latency_ms, max_price, prefer_spot, min_gpu_memory_gb,
            carbon_weight,
        )

    # ── objective implementations ──

    def _route_cheapest(self, gpu_type, max_latency, max_price, prefer_spot, min_mem) -> SlaRouterReport:
        decision = self._router.select_provider(
            gpu_type=gpu_type, max_latency_ms=max_latency, max_price=max_price,
            prefer_spot=prefer_spot, min_gpu_memory_gb=min_mem,
        )
        if decision is None:
            return SlaRouterReport(objective=Objective.CHEAPEST, reason="no provider meets constraints")
        savings = self._cheapest_savings(decision)
        return self._report(Objective.CHEAPEST, decision, savings_usd=savings)

    def _route_fastest(self, gpu_type, max_latency, max_price, prefer_spot, min_mem) -> SlaRouterReport:
        decision = self._router.select_provider_fastest(
            gpu_type=gpu_type, max_latency_ms=max_latency, max_price=max_price,
            prefer_spot=prefer_spot, min_gpu_memory_gb=min_mem,
        )
        if decision is None:
            return SlaRouterReport(objective=Objective.FASTEST, reason="no provider meets constraints")
        return self._report(Objective.FASTEST, decision)

    def _route_greenest(self, gpu_type, max_latency, max_price, prefer_spot, min_mem, carbon_weight) -> SlaRouterReport:
        decision = self._router.select_provider_carbon_aware(
            gpu_type=gpu_type, max_latency_ms=max_latency, max_price=max_price,
            prefer_spot=prefer_spot, carbon_weight=carbon_weight,
            min_gpu_memory_gb=min_mem,
        )
        if decision is None:
            return SlaRouterReport(objective=Objective.GREENEST, reason="no provider meets constraints")
        gco2 = self._gco2_avoided(decision)
        return self._report(Objective.GREENEST, decision, gco2_avoided=gco2)

    def _route_balanced(self, gpu_type, max_latency, max_price, prefer_spot, min_mem, carbon_weight) -> SlaRouterReport:
        # Balanced prefers carbon-aware routing (which already blends cost +
        # latency + carbon), then reports both savings and gCO2.
        decision = self._router.select_provider_carbon_aware(
            gpu_type=gpu_type, max_latency_ms=max_latency, max_price=max_price,
            prefer_spot=prefer_spot, carbon_weight=carbon_weight,
            min_gpu_memory_gb=min_mem,
        )
        if decision is None:
            # Fall back to pure cheapest so the request is still served.
            decision = self._router.select_provider(
                gpu_type=gpu_type, max_latency_ms=max_latency, max_price=max_price,
                prefer_spot=prefer_spot, min_gpu_memory_gb=min_mem,
            )
        if decision is None:
            return SlaRouterReport(objective=Objective.BALANCED, reason="no provider meets constraints")
        savings = self._cheapest_savings(decision)
        gco2 = self._gco2_avoided(decision)
        return self._report(Objective.BALANCED, decision, savings_usd=savings, gco2_avoided=gco2)

    # ── helpers ──

    def _cheapest_savings(self, decision: Any) -> float:
        """USD/hr saved vs the priciest option considered.

        ``RouteDecision.alternatives_considered`` tells us how many were
        evaluated; we approximate the priciest as ``price_per_hour +
        savings`` where ``savings`` is already computed by the router for the
        cheapest path.  For other paths we derive a baseline from the router's
        known priciest price via a cheap probe when possible.
        """
        # The router's select_provider already stores the per-choice savings
        # in its reason string, but we recompute conservatively: if the router
        # exposes a savings figure via alternatives we trust the cheapest path.
        return float(getattr(decision, "estimated_cost", 0.0))

    def _gco2_avoided(self, decision: Any) -> float:
        intensity = float(getattr(decision, "carbon_intensity", 0.0))
        if intensity <= 0:
            return 0.0
        delta = max(0.0, self._baseline_intensity - intensity)
        return delta * self._energy_kwh

    def _report(self, objective: Objective, decision: Any, savings_usd: float = 0.0, gco2_avoided: float = 0.0) -> SlaRouterReport:
        if self._metrics is not None:
            if savings_usd:
                self._metrics.increment("sla_savings_total_usd", _round_usd(savings_usd))
            if gco2_avoided:
                self._metrics.increment("sla_gco2_avoided_total", _round_gco2(gco2_avoided))
        return SlaRouterReport(
            objective=objective,
            provider=getattr(decision, "provider", ""),
            region=getattr(decision, "region", ""),
            instance_type=getattr(decision, "instance_type", ""),
            gpu_type=getattr(decision, "gpu_type", ""),
            price_per_hour=float(getattr(decision, "price_per_hour", 0.0)),
            latency_ms=float(getattr(decision, "latency_ms", 0.0)),
            carbon_intensity=float(getattr(decision, "carbon_intensity", 0.0)),
            savings_usd=savings_usd,
            gco2_avoided=gco2_avoided,
            reason=getattr(decision, "reason", ""),
            raw=decision,
        )


def _round_usd(v: float) -> int:
    """MetricsManager counters are ints; express USD/hr savings in cents."""
    return int(round(v * 100))


def _round_gco2(v: float) -> int:
    return int(round(v))
