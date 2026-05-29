"""Predictive Spot Pricing Model.

Lightweight forecasting for spot instance pricing using Holt-Winters
exponential smoothing. Predicts optimal launch windows and estimates
interruption risk based on historical price patterns.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class PriceForecast:
    """A price forecast for a specific instance."""
    provider: str
    instance_type: str
    region: str
    current_price: float
    predicted_price: float
    predicted_min: float  # Lower bound
    predicted_max: float  # Upper bound
    confidence: float  # 0.0-1.0
    interruption_risk: float  # 0.0-1.0
    recommended_window_hours: float  # How long to hold before price spike
    horizon_hours: int = 24

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "instance_type": self.instance_type,
            "region": self.region,
            "current_price": self.current_price,
            "predicted_price": self.predicted_price,
            "predicted_range": [self.predicted_min, self.predicted_max],
            "confidence": self.confidence,
            "interruption_risk": self.interruption_risk,
            "recommended_window_hours": self.recommended_window_hours,
        }


class SpotPriceForecaster:
    """Forecasts spot instance prices using Holt-Winters exponential smoothing.

    Usage::

        forecaster = SpotPriceForecaster()
        forecaster.record("aws", "p4d.24xlarge", "us-east-1", 14.40)
        forecaster.record("aws", "p4d.24xlarge", "us-east-1", 13.80)
        # ... more observations
        forecast = forecaster.forecast("aws", "p4d.24xlarge", "us-east-1")
    """

    def __init__(
        self,
        alpha: float = 0.3,  # Level smoothing
        beta: float = 0.1,   # Trend smoothing
        gamma: float = 0.2,  # Seasonal smoothing
        season_length: int = 24,  # Hours per season (daily cycle)
        history_size: int = 168,  # 1 week of hourly data
    ):
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._season_length = season_length
        self._history_size = history_size
        self._histories: dict[str, deque[tuple[float, float]]] = {}  # key -> (timestamp, price)
        self._models: dict[str, dict[str, Any]] = {}  # key -> fitted model params
        self._lock = threading.Lock()

    def _key(self, provider: str, instance_type: str, region: str) -> str:
        return f"{provider}:{instance_type}:{region}"

    def record(
        self,
        provider: str,
        instance_type: str,
        region: str,
        price: float,
        timestamp: float | None = None,
    ) -> None:
        """Record a price observation."""
        key = self._key(provider, instance_type, region)
        ts = timestamp or time.time()
        with self._lock:
            if key not in self._histories:
                self._histories[key] = deque(maxlen=self._history_size)
            self._histories[key].append((ts, price))
            # Invalidate fitted model
            self._models.pop(key, None)

    def forecast(
        self,
        provider: str,
        instance_type: str,
        region: str,
        horizon_hours: int = 24,
    ) -> PriceForecast | None:
        """Forecast future price for an instance.

        Returns None if insufficient history (< 6 observations).
        """
        key = self._key(provider, instance_type, region)
        with self._lock:
            history = list(self._histories.get(key, []))

        if len(history) < 6:
            return None

        prices = [p for _, p in history]
        current = prices[-1]

        # Fit Holt-Winters
        try:
            forecast_val, lower, upper, confidence = self._holt_winters_forecast(
                prices, horizon_hours
            )
        except Exception:
            # Fallback: simple moving average
            window = min(24, len(prices))
            forecast_val = sum(prices[-window:]) / window
            std = self._stddev(prices[-window:])
            lower = forecast_val - 2 * std
            upper = forecast_val + 2 * std
            confidence = 0.3

        # Estimate interruption risk from price volatility
        risk = self._estimate_interruption_risk(prices)

        # Recommended window: how long until price likely rises
        window_h = self._recommended_window(prices, current)

        return PriceForecast(
            provider=provider,
            instance_type=instance_type,
            region=region,
            current_price=current,
            predicted_price=forecast_val,
            predicted_min=max(0, lower),
            predicted_max=upper,
            confidence=confidence,
            interruption_risk=risk,
            recommended_window_hours=window_h,
            horizon_hours=horizon_hours,
        )

    def _holt_winters_forecast(
        self, prices: list[float], horizon: int
    ) -> tuple[float, float, float, float]:
        """Triple exponential smoothing forecast.

        Returns: (forecast, lower_bound, upper_bound, confidence)
        """
        n = len(prices)
        sl = self._season_length

        # Initialize
        level = sum(prices[:sl]) / sl if n >= sl else sum(prices) / n
        trend = (sum(prices[sl:2*sl]) - sum(prices[:sl])) / (sl * sl) if n >= 2 * sl else 0
        seasonal = [1.0] * sl
        if n >= sl:
            avg = level if level != 0 else 1.0
            for i in range(min(sl, n)):
                seasonal[i] = prices[i] / avg

        # Fit
        for t in range(sl, n):
            s_idx = t % sl
            new_level = self._alpha * (prices[t] / seasonal[s_idx]) + (1 - self._alpha) * (level + trend)
            new_trend = self._beta * (new_level - level) + (1 - self._beta) * trend
            seasonal[s_idx] = self._gamma * (prices[t] / new_level) + (1 - self._gamma) * seasonal[s_idx]
            level = new_level
            trend = new_trend

        # Forecast
        forecast_val = (level + trend * horizon) * seasonal[(n + horizon) % sl]

        # Confidence interval from residuals
        residuals = []
        for t in range(sl, n):
            s_idx = t % sl
            fitted = (level + trend * (t - n)) * seasonal[s_idx]
            residuals.append(prices[t] - fitted)
        std = self._stddev(residuals) if residuals else abs(forecast_val * 0.1)

        confidence = max(0.1, min(1.0, 1.0 - (std / max(abs(forecast_val), 0.01))))
        lower = forecast_val - 1.96 * std
        upper = forecast_val + 1.96 * std

        return forecast_val, lower, upper, confidence

    def _estimate_interruption_risk(self, prices: list[float]) -> float:
        """Estimate interruption risk from price volatility.

        High volatility = higher risk of sudden price spikes (interruption).
        """
        if len(prices) < 2:
            return 0.5
        std = self._stddev(prices)
        mean = sum(prices) / len(prices)
        cv = std / max(mean, 0.01)  # Coefficient of variation
        # Map CV to risk: CV=0 -> 0.0, CV=0.5 -> 0.5, CV=1.0 -> 0.9
        risk = min(0.95, cv * 1.8)
        return round(risk, 3)

    def _recommended_window(self, prices: list[float], current: float) -> float:
        """Estimate hours until price likely rises above current."""
        if len(prices) < 6:
            return 1.0
        mean = sum(prices) / len(prices)
        std = self._stddev(prices)
        if std <= 0:
            return 24.0
        # How many std devs below mean is current price?
        z = (mean - current) / std
        if z > 1.0:
            return 24.0  # Very cheap, hold for a while
        if z > 0:
            return 12.0
        if z > -1:
            return 4.0
        return 1.0  # Above mean, price likely to drop

    @staticmethod
    def _stddev(values: list[float]) -> float:
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return math.sqrt(variance)


@dataclass
class CarbonBudget:
    """Monthly carbon budget for a tenant or cluster."""
    tenant_id: str
    monthly_budget_kg: float = 100.0
    current_month_kg: float = 0.0
    month_start: float = field(default_factory=time.time)

    @property
    def remaining_kg(self) -> float:
        return max(0, self.monthly_budget_kg - self.current_month_kg)

    @property
    def utilization_pct(self) -> float:
        if self.monthly_budget_kg <= 0:
            return 0.0
        return (self.current_month_kg / self.monthly_budget_kg) * 100

    @property
    def is_exceeded(self) -> bool:
        return self.current_month_kg >= self.monthly_budget_kg

    def record_emissions(self, kg: float) -> None:
        """Record carbon emissions."""
        self.current_month_kg += kg

    def suggest_carbon_weight(self) -> float:
        """Suggest carbon_weight for routing based on budget utilization.

        Returns a value between 0.0 (ignore carbon) and 1.0 (pure carbon).
        When under budget, weight is low. As budget approaches, weight increases.
        """
        pct = self.utilization_pct
        if pct < 50:
            return 0.1  # Low priority
        if pct < 70:
            return 0.3
        if pct < 85:
            return 0.5
        if pct < 95:
            return 0.8
        return 1.0  # Emergency: minimize carbon at any cost


class CarbonBudgetEnforcer:
    """Enforces carbon budgets across tenants and auto-adjusts routing.

    Usage::

        enforcer = CarbonBudgetEnforcer()
        enforcer.set_budget("team-alpha", CarbonBudget("team-alpha", 50.0))
        weight = enforcer.get_carbon_weight("team-alpha")  # 0.1 if under budget
        enforcer.record("team-alpha", 0.5)  # Record 0.5 kg CO2
    """

    def __init__(self):
        self._budgets: dict[str, CarbonBudget] = {}
        self._lock = threading.Lock()

    def set_budget(self, tenant_id: str, budget: CarbonBudget) -> None:
        """Set a carbon budget for a tenant."""
        with self._lock:
            self._budgets[tenant_id] = budget

    def record(self, tenant_id: str, kg_co2: float) -> None:
        """Record carbon emissions for a tenant."""
        with self._lock:
            budget = self._budgets.get(tenant_id)
            if budget:
                budget.record_emissions(kg_co2)

    def get_carbon_weight(self, tenant_id: str) -> float:
        """Get the recommended carbon weight for routing."""
        with self._lock:
            budget = self._budgets.get(tenant_id)
            if not budget:
                return 0.3  # Default
            # Reset if new month
            if time.time() - budget.month_start > 2592000:
                budget.current_month_kg = 0.0
                budget.month_start = time.time()
            return budget.suggest_carbon_weight()

    def get_status(self, tenant_id: str) -> dict[str, Any] | None:
        """Get budget status for a tenant."""
        with self._lock:
            budget = self._budgets.get(tenant_id)
            if not budget:
                return None
            return {
                "tenant_id": tenant_id,
                "monthly_budget_kg": budget.monthly_budget_kg,
                "current_month_kg": budget.current_month_kg,
                "remaining_kg": budget.remaining_kg,
                "utilization_pct": budget.utilization_pct,
                "is_exceeded": budget.is_exceeded,
                "suggested_carbon_weight": budget.suggest_carbon_weight(),
            }


@dataclass
class CostForecastResult:
    """Result of a cost forecast calculation."""
    model_name: str
    total_tokens: int
    forecasts: list[dict[str, Any]]  # Per-provider forecast
    best_provider: str
    best_cost: float
    worst_cost: float
    avg_cost: float
    savings_vs_worst: float

    def to_table(self) -> str:
        """Format as a CLI-friendly table."""
        lines = [
            f"Cost Forecast: {self.model_name} ({self.total_tokens:,} tokens)",
            f"{'Provider':<20} {'Instance':<25} {'Region':<15} {'Cost':>10} {'Carbon':>10}",
            "-" * 80,
        ]
        for f in self.forecasts:
            lines.append(
                f"{f['provider']:<20} {f['instance']:<25} {f['region']:<15} "
                f"${f['cost']:>9.4f} {f['carbon']:>8.0f}g"
            )
        lines.extend([
            "-" * 80,
            f"Best:  {self.best_provider} at ${self.best_cost:.4f}",
            f"Worst: ${self.worst_cost:.4f} (savings: ${self.savings_vs_worst:.4f})",
        ])
        return "\n".join(lines)


class CostForecaster:
    """Forecasts inference costs across all providers.

    Usage::

        forecaster = CostForecaster(router)
        result = forecaster.forecast("Llama-70B", total_tokens=1_000_000)
        print(result.to_table())
    """

    def __init__(self, router: Any = None, pricing_manager: Any = None):
        self._router = router
        self._pricing_manager = pricing_manager

    def forecast(
        self,
        model_name: str,
        total_tokens: int,
        prefer_spot: bool = True,
    ) -> CostForecastResult:
        """Forecast costs across all available providers.

        Args:
            model_name: Model name (for throughput estimation).
            total_tokens: Total tokens to generate.
            prefer_spot: Use spot pricing.

        Returns:
            CostForecastResult with per-provider breakdown.
        """
        # Estimate GPU-seconds for this workload
        from distllm.core.cost_tracker import _estimate_throughput
        tps = _estimate_throughput("A100-80GB", model_name)
        gpu_seconds = total_tokens / max(tps, 1)

        forecasts = []
        if self._router:
            prices = self._router.get_all_prices()
            for p in prices:
                price = p.get("spot_price" if prefer_spot else "price_per_hour", 0.0)
                if price <= 0:
                    continue
                cost = (gpu_seconds / 3600) * price
                forecasts.append({
                    "provider": p.get("provider", ""),
                    "instance": p.get("instance_type", ""),
                    "region": p.get("region", ""),
                    "cost": cost,
                    "price_per_hour": price,
                    "carbon": p.get("carbon_gco2_kwh", 0),
                })
        elif self._pricing_manager:
            all_prices = self._pricing_manager.get_all_pricing()
            for p in all_prices:
                price = p.spot_price if prefer_spot and p.spot_price > 0 else p.on_demand_price
                if price <= 0:
                    continue
                cost = (gpu_seconds / 3600) * price
                forecasts.append({
                    "provider": p.provider,
                    "instance": p.instance_type,
                    "region": p.region,
                    "cost": cost,
                    "price_per_hour": price,
                    "carbon": 0,
                })

        forecasts.sort(key=lambda f: f["cost"])
        costs = [f["cost"] for f in forecasts]

        if not forecasts:
            return CostForecastResult(
                model_name=model_name,
                total_tokens=total_tokens,
                forecasts=[],
                best_provider="",
                best_cost=0,
                worst_cost=0,
                avg_cost=0,
                savings_vs_worst=0,
            )

        return CostForecastResult(
            model_name=model_name,
            total_tokens=total_tokens,
            forecasts=forecasts,
            best_provider=forecasts[0]["provider"],
            best_cost=costs[0],
            worst_cost=costs[-1],
            avg_cost=sum(costs) / len(costs),
            savings_vs_worst=costs[-1] - costs[0],
        )
