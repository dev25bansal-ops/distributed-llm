"""Cost & Carbon Comparison Report Generator.

Generates formatted comparison tables for GPU instances across
providers, regions, and pricing models. CLI-friendly output.

Usage::

    from distllm.core.cost_comparison import CostComparison
    comparison = CostComparison(router)
    report = comparison.compare(gpu_type="A100", hours=24)
    print(report.to_table())
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class ComparisonRow:
    """A single row in the comparison table."""
    provider: str
    instance_type: str
    region: str
    gpu_type: str
    gpu_count: int
    gpu_memory_gb: float
    on_demand_hourly: float
    spot_hourly: float
    total_on_demand: float  # price * hours
    total_spot: float
    latency_ms: float
    carbon_gco2_kwh: float
    renewable_pct: float
    available: bool
    score: float = 0.0  # Lower is better

    @property
    def best_price(self) -> float:
        return min(self.total_on_demand, self.total_spot) if self.total_spot > 0 else self.total_on_demand


@dataclass
class ComparisonReport:
    """A cost and carbon comparison report."""
    gpu_type: str
    hours: float
    regions: list[str]
    rows: list[ComparisonRow]
    generated_at: float = field(default_factory=time.time)

    def to_table(self, max_rows: int = 30) -> str:
        """Format as a CLI-friendly table."""
        if not self.rows:
            return "No matching instances found."

        # Sort by total spot price
        sorted_rows = sorted(self.rows, key=lambda r: r.best_price)[:max_rows]

        lines = [
            f"\n{'='*100}",
            f"Cost & Carbon Comparison: {self.gpu_type} GPUs ({self.hours}h)",
            f"Regions: {', '.join(self.regions) if self.regions else 'all'}",
            f"Generated: {time.strftime('%Y-%m-%d %H:%M', time.localtime(self.generated_at))}",
            f"{'='*100}\n",
            f"{'Provider':<10} {'Instance':<28} {'Region':<16} {'GPUs':>4} {'Spot/hr':>10} "
            f"{'Total':>10} {'CO2':>8} {'Renew':>6} {'Lat':>6} {'Score':>6}",
            "-" * 100,
        ]
        for r in sorted_rows:
            spot_str = f"${r.spot_hourly:.2f}" if r.spot_hourly > 0 else "N/A"
            total_str = f"${r.total_spot:.2f}" if r.total_spot > 0 else f"${r.total_on_demand:.2f}"
            carbon_str = f"{r.carbon_gco2_kwh:.0f}" if r.carbon_gco2_kwh > 0 else "N/A"
            renew_str = f"{r.renewable_pct:.0f}%" if r.renewable_pct > 0 else "N/A"
            lat_str = f"{r.latency_ms:.0f}ms" if r.latency_ms > 0 else "N/A"
            lines.append(
                f"{r.provider:<10} {r.instance_type:<28} {r.region:<16} {r.gpu_count:>4} "
                f"{spot_str:>10} {total_str:>10} {carbon_str:>8} {renew_str:>6} {lat_str:>6} "
                f"{r.score:>6.2f}"
            )
        lines.extend([
            "-" * 100,
            f"\nBest option: {sorted_rows[0].provider}/{sorted_rows[0].instance_type} "
            f"in {sorted_rows[0].region} at ${sorted_rows[0].best_price:.2f}",
        ])

        # Savings summary
        if len(sorted_rows) > 1:
            worst = sorted_rows[-1].best_price
            best = sorted_rows[0].best_price
            if worst > best:
                lines.append(f"Savings vs worst: ${worst - best:.2f} ({(worst - best) / worst * 100:.1f}%)")

        # Carbon summary
        carbon_rows = [r for r in sorted_rows if r.carbon_gco2_kwh > 0]
        if carbon_rows:
            cleanest = min(carbon_rows, key=lambda r: r.carbon_gco2_kwh)
            dirtiest = max(carbon_rows, key=lambda r: r.carbon_gco2_kwh)
            if dirtiest.carbon_gco2_kwh > cleanest.carbon_gco2_kwh:
                saved = dirtiest.carbon_gco2_kwh - cleanest.carbon_gco2_kwh
                lines.append(
                    f"Cleanest: {cleanest.region} ({cleanest.carbon_gco2_kwh:.0f} gCO2/kWh) — "
                    f"{saved:.0f} gCO2/kWh less than dirtiest ({dirtiest.region})"
                )

        lines.append(f"\n{len(sorted_rows)} of {len(self.rows)} results shown")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "gpu_type": self.gpu_type,
            "hours": self.hours,
            "regions": self.regions,
            "rows": [
                {
                    "provider": r.provider,
                    "instance": r.instance_type,
                    "region": r.region,
                    "gpu_count": r.gpu_count,
                    "spot_hourly": r.spot_hourly,
                    "total_cost": r.best_price,
                    "carbon_gco2_kwh": r.carbon_gco2_kwh,
                    "latency_ms": r.latency_ms,
                    "score": r.score,
                }
                for r in sorted(self.rows, key=lambda r: r.best_price)[:20]
            ],
            "generated_at": self.generated_at,
        }


class CostComparison:
    """Generates cost and carbon comparison reports.

    Usage::

        comparison = CostComparison(router)
        report = comparison.compare(gpu_type="A100", hours=24, regions=["us-east-1", "eu-west-1"])
        print(report.to_table())
    """

    def __init__(self, router: Any = None, pricing_manager: Any = None):
        self._router = router
        self._pricing_manager = pricing_manager

    def compare(
        self,
        gpu_type: str = "A100",
        hours: float = 24.0,
        regions: list[str] | None = None,
        include_on_demand: bool = True,
        include_spot: bool = True,
    ) -> ComparisonReport:
        """Generate a comparison report.

        Args:
            gpu_type: GPU type to compare (e.g., "A100", "V100").
            hours: Duration in hours for total cost calculation.
            regions: Optional list of regions to include.
            include_on_demand: Include on-demand pricing.
            include_spot: Include spot pricing.

        Returns:
            ComparisonReport with formatted table.
        """
        rows: list[ComparisonRow] = []

        if self._router:
            prices = self._router.get_all_prices(gpu_type=gpu_type)
            for p in prices:
                region = p.get("region", "")
                if regions and region not in regions:
                    continue
                od = p.get("price_per_hour", 0.0)
                spot = p.get("spot_price", 0.0)
                latency = p.get("latency_ms", 0.0)
                carbon = p.get("carbon_gco2_kwh", 0.0)
                renewable = p.get("renewable_pct", 0.0)
                gpu_mem = p.get("gpu_memory_gb", 0.0)
                gpu_count = p.get("gpu_count", 1)
                score = self._compute_score(od, spot, latency, carbon)
                rows.append(ComparisonRow(
                    provider=p.get("provider", ""),
                    instance_type=p.get("instance_type", ""),
                    region=region,
                    gpu_type=gpu_type,
                    gpu_count=gpu_count,
                    gpu_memory_gb=gpu_mem,
                    on_demand_hourly=od,
                    spot_hourly=spot,
                    total_on_demand=od * hours,
                    total_spot=spot * hours,
                    latency_ms=latency,
                    carbon_gco2_kwh=carbon,
                    renewable_pct=renewable,
                    available=p.get("available", True),
                    score=score,
                ))
        elif self._pricing_manager:
            all_prices = self._pricing_manager.get_all_pricing()
            for p in all_prices:
                if gpu_type.upper() not in (p.gpu_type or "").upper():
                    continue
                if regions and p.region not in regions:
                    continue
                od = p.on_demand_price
                spot = p.spot_price
                score = self._compute_score(od, spot, 0, 0)
                rows.append(ComparisonRow(
                    provider=p.provider,
                    instance_type=p.instance_type,
                    region=p.region,
                    gpu_type=p.gpu_type or gpu_type,
                    gpu_count=p.gpu_count,
                    gpu_memory_gb=p.gpu_memory_gb,
                    on_demand_hourly=od,
                    spot_hourly=spot,
                    total_on_demand=od * hours,
                    total_spot=spot * hours,
                    latency_ms=0,
                    carbon_gco2_kwh=0,
                    renewable_pct=0,
                    available=True,
                    score=score,
                ))

        return ComparisonReport(
            gpu_type=gpu_type,
            hours=hours,
            regions=regions or [],
            rows=rows,
        )

    @staticmethod
    def _compute_score(
        on_demand: float, spot: float, latency: float, carbon: float
    ) -> float:
        """Compute a composite score (lower is better)."""
        price = spot if spot > 0 else on_demand
        # Normalize: price is dominant, latency and carbon are secondary
        score = price * 0.6
        if latency > 0:
            score += (latency / 200.0) * 0.2  # 200ms = 0.2 weight
        if carbon > 0:
            score += (carbon / 800.0) * 0.2  # 800 gCO2 = 0.2 weight
        return round(score, 4)
