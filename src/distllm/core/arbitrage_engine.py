"""GPU Arbitrage Engine — spot pricing monitoring and load migration.

Monitors spot pricing across all configured cloud providers and regions,
detects arbitrage opportunities (price drops, cheaper alternatives), and
generates migration recommendations for in-flight workloads.

Integrates with:
- PricingManager for live pricing data
- CrossCloudRouter for routing decisions
- CostTracker for historical cost data
- HealthManager for node readiness checks
"""

from __future__ import annotations

import statistics
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from loguru import logger


class OpportunityType(Enum):
    """Type of arbitrage opportunity."""
    PRICE_DROP = "price_drop"           # Spot price dropped significantly
    CHEAPER_REGION = "cheaper_region"   # Another region is now cheaper
    CHEAPER_PROVIDER = "cheaper_provider"  # Another provider is now cheaper
    CARBON_SWITCH = "carbon_switch"     # Cleaner region available at similar cost


class MigrationRisk(Enum):
    """Risk level for a migration recommendation."""
    LOW = "low"         # Background migration, no disruption
    MEDIUM = "medium"   # Requires checkpoint/restore
    HIGH = "high"       # Interrupts active inference


@dataclass
class PricePoint:
    """A point-in-time price observation."""
    provider: str
    instance_type: str
    region: str
    price: float
    timestamp: float = field(default_factory=time.time)
    is_spot: bool = True


@dataclass
class PriceHistory:
    """Historical price data for one instance in one region."""
    provider: str
    instance_type: str
    region: str
    observations: list[PricePoint] = field(default_factory=list)
    window_size: int = 100

    def add(self, price: float, is_spot: bool = True) -> None:
        self.observations.append(PricePoint(
            provider=self.provider,
            instance_type=self.instance_type,
            region=self.region,
            price=price,
            is_spot=is_spot,
        ))
        if len(self.observations) > self.window_size:
            self.observations = self.observations[-self.window_size:]

    @property
    def current(self) -> float:
        return self.observations[-1].price if self.observations else 0.0

    @property
    def mean(self) -> float:
        if not self.observations:
            return 0.0
        return statistics.mean(o.price for o in self.observations)

    @property
    def stddev(self) -> float:
        prices = [o.price for o in self.observations]
        return statistics.stdev(prices) if len(prices) > 1 else 0.0

    @property
    def min_price(self) -> float:
        return min((o.price for o in self.observations), default=0.0)

    @property
    def trend_pct(self) -> float:
        """Price change over the window as a percentage."""
        if len(self.observations) < 2:
            return 0.0
        first = self.observations[0].price
        last = self.observations[-1].price
        if first <= 0:
            return 0.0
        return ((last - first) / first) * 100


@dataclass
class ArbitrageOpportunity:
    """A detected arbitrage opportunity."""
    opportunity_type: OpportunityType
    current_provider: str
    current_instance: str
    current_region: str
    current_price: float
    recommended_provider: str
    recommended_instance: str
    recommended_region: str
    recommended_price: float
    savings_per_hour: float
    savings_pct: float
    migration_risk: MigrationRisk
    carbon_savings_gco2: float = 0.0
    detected_at: float = field(default_factory=time.time)
    confidence: float = 0.0  # 0.0-1.0
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.opportunity_type.value,
            "current": f"{self.current_provider}/{self.current_instance}/{self.current_region}",
            "current_price": self.current_price,
            "recommended": f"{self.recommended_provider}/{self.recommended_instance}/{self.recommended_region}",
            "recommended_price": self.recommended_price,
            "savings_per_hour": self.savings_per_hour,
            "savings_pct": self.savings_pct,
            "migration_risk": self.migration_risk.value,
            "carbon_savings_gco2": self.carbon_savings_gco2,
            "confidence": self.confidence,
            "reason": self.reason,
        }


@dataclass
class MigrationRecommendation:
    """A recommendation to migrate workload from one provider to another."""
    from_provider: str
    from_region: str
    to_provider: str
    to_region: str
    from_price: float
    to_price: float
    estimated_savings_hourly: float
    risk: MigrationRisk
    action: str = ""
    preconditions: list[str] = field(default_factory=list)


class ArbitrageEngine:
    """Monitors spot pricing and detects arbitrage opportunities.

    Usage::

        engine = ArbitrageEngine()
        engine.update_pricing("aws", "p4d.24xlarge", "us-east-1", 14.40)
        opportunities = engine.detect_opportunities()
    """

    def __init__(
        self,
        price_drop_threshold_pct: float = 15.0,
        region_savings_threshold_pct: float = 20.0,
        provider_savings_threshold_pct: float = 25.0,
        history_window: int = 100,
        on_opportunity: Callable[[ArbitrageOpportunity], None] | None = None,
    ):
        self._price_drop_threshold = price_drop_threshold_pct
        self._region_savings_threshold = region_savings_threshold_pct
        self._provider_savings_threshold = provider_savings_threshold_pct
        self._history_window = history_window
        self._on_opportunity = on_opportunity

        self._histories: dict[str, PriceHistory] = {}
        self._opportunities: list[ArbitrageOpportunity] = []
        self._active_provider: str = ""
        self._active_instance: str = ""
        self._active_region: str = ""
        self._lock = threading.Lock()

        # Carbon intensity data (optional, from router)
        self._carbon_data: dict[str, float] = {}

    def set_active_location(self, provider: str, instance: str, region: str) -> None:
        """Set the currently active provider/instance/region for comparison."""
        self._active_provider = provider
        self._active_instance = instance
        self._active_region = region

    def set_carbon_data(self, carbon_data: dict[str, float]) -> None:
        """Set carbon intensity data (region -> gCO2/kWh)."""
        self._carbon_data = carbon_data

    def update_pricing(
        self,
        provider: str,
        instance_type: str,
        region: str,
        price: float,
        is_spot: bool = True,
    ) -> None:
        """Record a price observation."""
        key = f"{provider}:{instance_type}:{region}"
        with self._lock:
            if key not in self._histories:
                self._histories[key] = PriceHistory(
                    provider=provider,
                    instance_type=instance_type,
                    region=region,
                    window_size=self._history_window,
                )
            self._histories[key].add(price, is_spot)

    def update_pricing_batch(self, prices: list[dict[str, Any]]) -> None:
        """Record multiple price observations at once."""
        for p in prices:
            self.update_pricing(
                provider=p["provider"],
                instance_type=p["instance_type"],
                region=p["region"],
                price=p["price"],
                is_spot=p.get("is_spot", True),
            )

    def detect_opportunities(self) -> list[ArbitrageOpportunity]:
        """Detect all current arbitrage opportunities."""
        with self._lock:
            self._opportunities = []
            self._detect_price_drops()
            self._detect_cheaper_regions()
            self._detect_cheaper_providers()
            self._detect_carbon_opportunities()

            if self._opportunities and self._on_opportunity:
                for opp in self._opportunities:
                    try:
                        self._on_opportunity(opp)
                    except Exception as e:
                        logger.debug(f"Opportunity callback failed: {e}")

            return list(self._opportunities)

    def _detect_price_drops(self) -> None:
        """Detect significant spot price drops."""
        for key, history in self._histories.items():
            if len(history.observations) < 3:
                continue
            if history.current <= 0 or history.mean <= 0:
                continue
            drop_pct = ((history.mean - history.current) / history.mean) * 100
            if drop_pct >= self._price_drop_threshold:
                self._opportunities.append(ArbitrageOpportunity(
                    opportunity_type=OpportunityType.PRICE_DROP,
                    current_provider=history.provider,
                    current_instance=history.instance_type,
                    current_region=history.region,
                    current_price=history.current,
                    recommended_provider=history.provider,
                    recommended_instance=history.instance_type,
                    recommended_region=history.region,
                    recommended_price=history.current,
                    savings_per_hour=history.mean - history.current,
                    savings_pct=drop_pct,
                    migration_risk=MigrationRisk.LOW,
                    confidence=min(drop_pct / 50.0, 1.0),
                    reason=f"Spot price dropped {drop_pct:.1f}% from mean ${history.mean:.2f} to ${history.current:.2f}",
                ))

    def _detect_cheaper_regions(self) -> None:
        """Detect when another region is significantly cheaper for the same instance."""
        if not self._active_provider or not self._active_instance:
            return
        active_key = f"{self._active_provider}:{self._active_instance}:{self._active_region}"
        active_history = self._histories.get(active_key)
        if not active_history or active_history.current <= 0:
            return
        current_price = active_history.current
        for key, history in self._histories.items():
            if history.provider != self._active_provider:
                continue
            if history.instance_type != self._active_instance:
                continue
            if history.region == self._active_region:
                continue
            if history.current <= 0:
                continue
            savings_pct = ((current_price - history.current) / current_price) * 100
            if savings_pct >= self._region_savings_threshold:
                self._opportunities.append(ArbitrageOpportunity(
                    opportunity_type=OpportunityType.CHEAPER_REGION,
                    current_provider=self._active_provider,
                    current_instance=self._active_instance,
                    current_region=self._active_region,
                    current_price=current_price,
                    recommended_provider=history.provider,
                    recommended_instance=history.instance_type,
                    recommended_region=history.region,
                    recommended_price=history.current,
                    savings_per_hour=current_price - history.current,
                    savings_pct=savings_pct,
                    migration_risk=MigrationRisk.MEDIUM,
                    confidence=min(savings_pct / 40.0, 1.0),
                    reason=f"Region {history.region} is {savings_pct:.1f}% cheaper (${history.current:.2f}/hr vs ${current_price:.2f}/hr)",
                ))

    def _detect_cheaper_providers(self) -> None:
        """Detect when another provider is significantly cheaper."""
        if not self._active_provider or not self._active_instance:
            return
        active_key = f"{self._active_provider}:{self._active_instance}:{self._active_region}"
        active_history = self._histories.get(active_key)
        if not active_history or active_history.current <= 0:
            return
        current_price = active_history.current
        for key, history in self._histories.items():
            if history.provider == self._active_provider:
                continue
            if history.current <= 0:
                continue
            savings_pct = ((current_price - history.current) / current_price) * 100
            if savings_pct >= self._provider_savings_threshold:
                risk = MigrationRisk.HIGH if history.provider != self._active_provider else MigrationRisk.MEDIUM
                self._opportunities.append(ArbitrageOpportunity(
                    opportunity_type=OpportunityType.CHEAPER_PROVIDER,
                    current_provider=self._active_provider,
                    current_instance=self._active_instance,
                    current_region=self._active_region,
                    current_price=current_price,
                    recommended_provider=history.provider,
                    recommended_instance=history.instance_type,
                    recommended_region=history.region,
                    recommended_price=history.current,
                    savings_per_hour=current_price - history.current,
                    savings_pct=savings_pct,
                    migration_risk=risk,
                    confidence=min(savings_pct / 50.0, 1.0),
                    reason=f"{history.provider} {history.instance_type} in {history.region} is {savings_pct:.1f}% cheaper",
                ))

    def _detect_carbon_opportunities(self) -> None:
        """Detect when a cleaner region is available at similar cost."""
        if not self._active_region or not self._carbon_data:
            return
        active_carbon = self._carbon_data.get(self._active_region, 0)
        if active_carbon <= 0:
            return
        active_key = f"{self._active_provider}:{self._active_instance}:{self._active_region}"
        active_history = self._histories.get(active_key)
        if not active_history or active_history.current <= 0:
            return
        current_price = active_history.current
        for key, history in self._histories.items():
            if history.provider != self._active_provider:
                continue
            if history.instance_type != self._active_instance:
                continue
            if history.region == self._active_region:
                continue
            if history.current <= 0:
                continue
            other_carbon = self._carbon_data.get(history.region, 0)
            if other_carbon <= 0 or other_carbon >= active_carbon:
                continue
            price_diff_pct = abs(history.current - current_price) / current_price * 100
            if price_diff_pct <= 10:
                carbon_saved = active_carbon - other_carbon
                self._opportunities.append(ArbitrageOpportunity(
                    opportunity_type=OpportunityType.CARBON_SWITCH,
                    current_provider=self._active_provider,
                    current_instance=self._active_instance,
                    current_region=self._active_region,
                    current_price=current_price,
                    recommended_provider=history.provider,
                    recommended_instance=history.instance_type,
                    recommended_region=history.region,
                    recommended_price=history.current,
                    savings_per_hour=current_price - history.current,
                    savings_pct=0.0,
                    migration_risk=MigrationRisk.MEDIUM,
                    carbon_savings_gco2=carbon_saved,
                    confidence=0.8,
                    reason=f"Region {history.region} has {carbon_saved:.0f} gCO2/kWh less carbon at similar cost",
                ))

    def generate_migration_recommendations(self) -> list[MigrationRecommendation]:
        """Generate actionable migration recommendations from detected opportunities."""
        recommendations = []
        for opp in self._opportunities:
            action = "migrate"
            preconditions = []
            if opp.migration_risk == MigrationRisk.HIGH:
                action = "plan_migration"
                preconditions.append("Verify destination provider connectivity")
                preconditions.append("Pre-warm destination instances")
            elif opp.migration_risk == MigrationRisk.MEDIUM:
                action = "checkpoint_and_migrate"
                preconditions.append("Checkpoint active sequences")
            recommendations.append(MigrationRecommendation(
                from_provider=opp.current_provider,
                from_region=opp.current_region,
                to_provider=opp.recommended_provider,
                to_region=opp.recommended_region,
                from_price=opp.current_price,
                to_price=opp.recommended_price,
                estimated_savings_hourly=opp.savings_per_hour,
                risk=opp.migration_risk,
                action=action,
                preconditions=preconditions,
            ))
        return recommendations

    def get_price_trends(self) -> list[dict[str, Any]]:
        """Get price trend data for all tracked instances."""
        with self._lock:
            trends = []
            for key, history in self._histories.items():
                trends.append({
                    "key": key,
                    "provider": history.provider,
                    "instance_type": history.instance_type,
                    "region": history.region,
                    "current_price": history.current,
                    "mean_price": history.mean,
                    "min_price": history.min_price,
                    "stddev": history.stddev,
                    "trend_pct": history.trend_pct,
                    "observations": len(history.observations),
                })
            return sorted(trends, key=lambda t: t["current_price"])

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of the arbitrage engine state."""
        with self._lock:
            return {
                "tracked_instances": len(self._histories),
                "active_location": f"{self._active_provider}:{self._active_instance}:{self._active_region}",
                "opportunities_detected": len(self._opportunities),
                "total_savings_potential": sum(o.savings_per_hour for o in self._opportunities),
                "opportunities": [o.to_dict() for o in self._opportunities[:10]],
            }
