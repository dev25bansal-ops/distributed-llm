"""Predictive auto-scaler with traffic forecasting and carbon awareness.

Combines traffic forecasting, carbon-aware scaling, custom HPA metrics,
and a top-level orchestrator into a unified auto-scaling system.

Usage::

    aria = Aria(
        min_nodes=2,
        max_nodes=20,
        region="us-east-1",
    )
    aria.start()
    # ...
    aria.stop()
    print(aria.stats())
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from loguru import logger

try:
    import numpy as np

    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class MetricPoint:
    """A single metric observation at a point in time."""
    timestamp: float
    value: float


@dataclass
class ForecastPoint:
    """A single forecasted value at a point in time."""
    timestamp: float
    predicted_value: float
    lower_bound: float | None = None
    upper_bound: float | None = None


@dataclass
class TrafficForecast:
    """A complete traffic forecast result."""
    metric_key: str
    horizon: str
    points: list[ForecastPoint]
    confidence: float  # 0.0-1.0
    trend: str  # "up", "down", "stable"


@dataclass
class CarbonIntensity:
    """Carbon intensity for a region at a specific time."""
    region: str
    gco2_per_kwh: float  # gCO2eq/kWh
    timestamp: float
    source: str = "default"  # "api", "static", "default"


@dataclass
class ScaleDecision:
    """A scale-up/down decision with carbon reasoning."""
    action: str  # "scale_up", "scale_down", "noop"
    nodes_delta: int  # Positive = add, negative = remove
    reason: str
    carbon_intensity_at_action: float = 0.0
    forecasted_load: float = 0.0
    current_load: float = 0.0
    confidence: float = 0.0


@dataclass
class ScalingAction:
    """A scheduled scaling action within a scaling plan."""
    time_offset_seconds: float
    action: str  # "scale_up", "scale_down"
    node_delta: int
    reason: str


@dataclass
class ScalingPlan:
    """A complete scaling plan from PredictiveScaler."""
    horizon: str
    actions: list[ScalingAction] = field(default_factory=list)
    total_cost_impact: float = 0.0
    carbon_saved_estimate_kg: float = 0.0
    confidence: float = 0.0


@dataclass
class ScalingRecord:
    """A record of a completed scaling action."""
    timestamp: float
    action: str
    node_delta: int
    nodes_before: int
    nodes_after: int
    carbon_intensity: float
    reason: str


# ---------------------------------------------------------------------------
# Static regional carbon intensity defaults
# ---------------------------------------------------------------------------

_REGIONAL_CARBON_DEFAULTS: dict[str, float] = {
    # North America
    "us-east-1": 405.0,
    "us-east-2": 405.0,
    "us-west-1": 230.0,
    "us-west-2": 230.0,
    "ca-central-1": 150.0,
    # Europe
    "eu-west-1": 280.0,
    "eu-west-2": 250.0,
    "eu-west-3": 200.0,
    "eu-central-1": 340.0,
    "eu-north-1": 60.0,
    "eu-south-1": 300.0,
    # Asia-Pacific
    "ap-northeast-1": 450.0,
    "ap-northeast-2": 430.0,
    "ap-southeast-1": 410.0,
    "ap-southeast-2": 380.0,
    "ap-south-1": 620.0,
    # Other
    "sa-east-1": 120.0,
    "me-south-1": 520.0,
    "af-south-1": 650.0,
}

# ---------------------------------------------------------------------------
# TrafficForecaster
# ---------------------------------------------------------------------------


class TrafficForecaster:
    """Predicts future request volume using time-series forecasting.

    Uses linear regression (via NumPy) when available, with a fallback
    to Holt-Winters-like exponential smoothing.  If torch is available
    a simple 1-layer linear net can be trained for richer patterns.

    Usage::

        forecaster = TrafficForecaster()
        forecaster.update("requests_per_sec", 120.0)
        forecaster.update("requests_per_sec", 135.0)
        # ...
        result = forecaster.forecast("requests_per_sec", window="1h")
    """

    def __init__(
        self,
        history_size: int = 10080,  # 7 days at 1-minute resolution
        min_samples: int = 10,
        season_length: int = 1440,  # 1 day in minutes
        enable_torch: bool = True,
    ):
        self._history_size = history_size
        self._min_samples = min_samples
        self._season_length = season_length
        self._enable_torch = enable_torch and _TORCH_AVAILABLE

        self._histories: dict[str, deque[MetricPoint]] = {}
        self._lock = threading.Lock()

        # Cache last forecast result per key
        self._last_forecast: dict[str, TrafficForecast] = {}

    # -- public api ---------------------------------------------------------

    def update(self, metric_key: str, value: float, timestamp: float | None = None) -> None:
        """Record a metric observation and refit internal model."""
        ts = timestamp or time.time()
        with self._lock:
            if metric_key not in self._histories:
                self._histories[metric_key] = deque(maxlen=self._history_size)
            self._histories[metric_key].append(MetricPoint(ts, value))
            # Invalidate cached forecast
            self._last_forecast.pop(metric_key, None)

    def forecast(self, metric_key: str, window: str = "1h") -> TrafficForecast | None:
        """Forecast future request volume for the given window.

        Args:
            metric_key: The metric name to forecast.
            window: Forecast horizon (e.g. ``"5m"``, ``"30m"``, ``"1h"``, ``"6h"``).

        Returns:
            A TrafficForecast with predicted points, or None if insufficient data.
        """
        with self._lock:
            history = list(self._histories.get(metric_key, []))

        if len(history) < self._min_samples:
            return None

        horizon_minutes = _parse_window(window)
        timestamps = [p.timestamp for p in history]
        values = [p.value for p in history]

        # Forecast via the best available method
        forecast_values, lower_bounds, upper_bounds, confidence = self._forecast_internal(
            timestamps, values, horizon_minutes
        )

        # Determine trend
        trend = _compute_trend(history)

        # Build output points at regular intervals
        now = time.time()
        points: list[ForecastPoint] = []
        interval_seconds = max(60, horizon_minutes * 60 / max(len(forecast_values), 1))
        for i, pred in enumerate(forecast_values):
            ts = now + i * interval_seconds
            lb = lower_bounds[i] if lower_bounds else None
            ub = upper_bounds[i] if upper_bounds else None
            points.append(ForecastPoint(timestamp=ts, predicted_value=pred, lower_bound=lb, upper_bound=ub))

        result = TrafficForecast(
            metric_key=metric_key,
            horizon=window,
            points=points,
            confidence=confidence,
            trend=trend,
        )

        with self._lock:
            self._last_forecast[metric_key] = result

        return result

    def get_history(self, metric_key: str) -> list[MetricPoint]:
        """Return raw history for a metric key."""
        with self._lock:
            return list(self._histories.get(metric_key, []))

    def clear(self, metric_key: str | None = None) -> None:
        """Clear history for a key, or all keys if None."""
        with self._lock:
            if metric_key:
                self._histories.pop(metric_key, None)
                self._last_forecast.pop(metric_key, None)
            else:
                self._histories.clear()
                self._last_forecast.clear()

    # -- internal forecast engine -------------------------------------------

    def _forecast_internal(
        self,
        timestamps: list[float],
        values: list[float],
        horizon_minutes: int,
    ) -> tuple[list[float], list[float | None], list[float | None], float]:
        """Run the best available forecast method.

        Returns (predictions, lower_bounds, upper_bounds, confidence).
        """
        # Torch-based linear net (small, fast)
        if self._enable_torch:
            try:
                return self._torch_linear_forecast(timestamps, values, horizon_minutes)
            except Exception:
                logger.debug("Torch forecast failed, falling back to numpy")

        # NumPy linear regression
        if _NUMPY_AVAILABLE:
            try:
                return self._numpy_lr_forecast(timestamps, values, horizon_minutes)
            except Exception:
                logger.debug("NumPy forecast failed, falling back to smoothing")

        # Holt-Winters exponential smoothing fallback
        return self._exponential_smoothing_forecast(values, horizon_minutes)

    def _numpy_lr_forecast(
        self,
        timestamps: list[float],
        values: list[float],
        horizon_minutes: int,
    ) -> tuple[list[float], list[float | None], list[float | None], float]:
        """Linear regression via NumPy."""
        x = np.array(timestamps, dtype=np.float64)
        y = np.array(values, dtype=np.float64)

        # Normalize time to avoid numerical issues
        t0 = x[0]
        x_norm = (x - t0) / 60.0  # in minutes

        # Fit: y = slope * x_norm + intercept
        A = np.vstack([x_norm, np.ones_like(x_norm)]).T
        slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]

        # Residuals for confidence
        residuals = y - (slope * x_norm + intercept)
        std_residual = float(np.std(residuals)) if len(residuals) > 1 else abs(np.mean(values) * 0.1)

        # Predict
        future_minutes = np.arange(1, horizon_minutes + 1, dtype=np.float64)
        last_minute = float(x_norm[-1])
        pred_x = last_minute + future_minutes
        pred_y = slope * pred_x + intercept

        # Bounds
        lower = [float(v - 1.96 * std_residual) for v in pred_y]
        upper = [float(v + 1.96 * std_residual) for v in pred_y]

        # Confidence based on goodness of fit
        ss_res = float(np.sum(residuals ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r_squared = 1.0 - (ss_res / max(ss_tot, 1e-10))
        confidence = max(0.1, min(0.99, r_squared * 0.9))

        return [float(v) for v in pred_y], lower, upper, confidence

    def _torch_linear_forecast(
        self,
        timestamps: list[float],
        values: list[float],
        horizon_minutes: int,
    ) -> tuple[list[float], list[float | None], list[float | None], float]:
        """Simple 1-layer linear model via PyTorch."""
        import torch.nn as nn
        import torch.optim as optim

        x = torch.tensor(
            [(ts - timestamps[0]) / 60.0 for ts in timestamps],
            dtype=torch.float32,
        ).unsqueeze(1)
        y = torch.tensor(values, dtype=torch.float32).unsqueeze(1)

        model = nn.Linear(1, 1)
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        loss_fn = nn.MSELoss()

        # Fast training (small model, few epochs)
        model.train()
        for _epoch in range(50):
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            last_x = float(x[-1].item())
            future_x = torch.tensor(
                [[last_x + i] for i in range(1, horizon_minutes + 1)],
                dtype=torch.float32,
            )
            pred_y = model(future_x).squeeze().tolist()
            if isinstance(pred_y, float):
                pred_y = [pred_y]
            pred_y_list: list[float] = pred_y

            # Residual-based confidence
            train_pred = model(x).squeeze().tolist()
            if isinstance(train_pred, float):
                train_pred = [train_pred]
            residuals = [float(v) - float(p) for v, p in zip(values, train_pred)]
            std_r = _std(residuals) if len(residuals) > 1 else abs(sum(values) / max(len(values), 1)) * 0.1

        lower = [v - 1.96 * std_r for v in pred_y_list]
        upper = [v + 1.96 * std_r for v in pred_y_list]
        confidence = max(0.1, min(0.99, 1.0 - (std_r / max(abs(float(np.mean(pred_y_list))), 1e-6))))

        return pred_y_list, lower, upper, confidence

    def _exponential_smoothing_forecast(
        self,
        values: list[float],
        horizon_minutes: int,
    ) -> tuple[list[float], list[float | None], list[float | None], float]:
        """Holt-Winters-like exponential smoothing fallback."""
        if len(values) < 4:
            # Simple last-value repeat
            last = values[-1]
            pred = [last] * horizon_minutes
            return pred, None, None, 0.2

        alpha = 0.3
        beta = 0.1
        gamma = 0.2
        sl = min(self._season_length, len(values))

        level = sum(values[:sl]) / sl
        trend_val = 0.0
        seasonal = [1.0] * sl
        avg = level if level != 0 else 1.0
        for i in range(min(sl, len(values))):
            seasonal[i] = values[i] / avg

        for t in range(sl, len(values)):
            s_idx = t % sl
            new_level = alpha * (values[t] / max(seasonal[s_idx], 1e-10)) + (1 - alpha) * (level + trend_val)
            new_trend = beta * (new_level - level) + (1 - beta) * trend_val
            seasonal[s_idx] = gamma * (values[t] / max(new_level, 1e-10)) + (1 - gamma) * seasonal[s_idx]
            level = new_level
            trend_val = new_trend

        pred = []
        for i in range(horizon_minutes):
            pred.append((level + trend_val * i) * seasonal[(len(values) + i) % sl])

        # Residual-based confidence
        residuals = []
        for t in range(sl, len(values)):
            s_idx = t % sl
            fitted = (level + trend_val * (t - len(values))) * seasonal[s_idx]
            residuals.append(values[t] - fitted)

        std_r = _std(residuals) if residuals else abs(sum(values[-10:]) / 10) * 0.1 if len(values) >= 10 else 1.0
        confidence = max(0.1, min(0.95, 1.0 - (std_r / max(abs(pred[-1]), 1e-6))))

        lower: list[float | None] = [v - 1.96 * std_r for v in pred]
        upper: list[float | None] = [v + 1.96 * std_r for v in pred]

        return pred, lower, upper, confidence


# ---------------------------------------------------------------------------
# CarbonAwareScaler
# ---------------------------------------------------------------------------


class CarbonAwareScaler:
    """Carbon-aware scaling decisions based on regional carbon intensity.

    Tracks real-time (or static) carbon intensity per region and produces
    scaling decisions that prefer low-carbon time windows.  Integrates with
    the TrafficForecaster to delay or accelerate scaling based on grid
    cleanliness.

    Usage::

        scaler = CarbonAwareScaler(region="us-east-1")
        intensity = scaler.get_carbon_intensity("us-east-1")
        decision = scaler.should_scale(
            forecast=traffic_forecast,
            current_load=0.75,
            carbon_threshold=400.0,
        )
    """

    def __init__(
        self,
        region: str = "us-east-1",
        carbon_threshold_gco2: float = 400.0,
        idle_carbon_intensity: float = 100.0,
        cooldown_seconds: float = 120.0,
    ):
        self._region = region
        self._carbon_threshold = carbon_threshold_gco2
        self._idle_intensity = idle_carbon_intensity
        self._cooldown = cooldown_seconds
        self._last_intensity: dict[str, CarbonIntensity] = {}
        self._lock = threading.Lock()
        self._last_decision_time = 0.0

    # -- public api ---------------------------------------------------------

    def get_carbon_intensity(self, region: str | None = None, at_time: float | None = None) -> float:
        """Return carbon intensity for a region (gCO2eq/kWh).

        Uses a time-of-day heuristic when no real-time data is available:
        lower at night / early morning (renewables dominate), higher during
        peak hours.  Falls back to the static regional default.

        Args:
            region: AWS/GCP/Azure region code.  Defaults to ``self._region``.
            at_time: Unix timestamp.  Defaults to now.

        Returns:
            gCO2eq per kWh.
        """
        target_region = region or self._region
        ts = at_time or time.time()

        # Check cached API result first
        with self._lock:
            cached = self._last_intensity.get(target_region)
            if cached and (ts - cached.timestamp) < 600.0:
                return cached.gco2_per_kwh

        intensity = self._compute_intensity(target_region, ts)

        with self._lock:
            self._last_intensity[target_region] = CarbonIntensity(
                region=target_region,
                gco2_per_kwh=intensity,
                timestamp=ts,
                source="static",
            )

        return intensity

    def should_scale(
        self,
        forecast: TrafficForecast | None,
        current_load: float,
        carbon_threshold: float | None = None,
        current_nodes: int = 1,
        min_nodes: int = 1,
        max_nodes: int = 20,
    ) -> ScaleDecision:
        """Produce a carbon-aware scaling decision.

        Args:
            forecast: TrafficForecast from TrafficForecaster (optional).
            current_load: Current GPU/utilization load (0.0-1.0).
            carbon_threshold: Max acceptable gCO2eq/kWh for scaling.  Defaults to
                ``self._carbon_threshold``.
            current_nodes: Current node count.
            min_nodes: Floor for node count.
            max_nodes: Ceiling for node count.

        Returns:
            A ScaleDecision with action, reason, and carbon context.
        """
        # Cooldown check
        now = time.time()
        if now - self._last_decision_time < self._cooldown:
            return ScaleDecision(
                action="noop",
                nodes_delta=0,
                reason="cooldown",
                current_load=current_load,
                carbon_intensity_at_action=0.0,
            )

        threshold = carbon_threshold if carbon_threshold is not None else self._carbon_threshold
        intensity = self.get_carbon_intensity(self._region, now)

        # Determine forecasted load direction
        predicted_load = current_load
        if forecast and forecast.points:
            predicted_load = forecast.points[-1].predicted_value
            # Normalize to 0-1 if it looks like raw request count
            if predicted_load > 1.0:
                # Assume max capacity normalized
                predicted_load = min(1.0, predicted_load / 1000.0)

        # High load scenarios (must scale regardless of carbon)
        if current_load > 0.85 or predicted_load > 0.80:
            delta = 1 if current_nodes < max_nodes else 0
            reason = f"high_load_{'forecasted_' if predicted_load > 0.80 and current_load <= 0.85 else ''}needs_capacity"
            self._last_decision_time = now
            return ScaleDecision(
                action="scale_up" if delta > 0 else "noop",
                nodes_delta=delta,
                reason=reason,
                carbon_intensity_at_action=intensity,
                forecasted_load=predicted_load,
                current_load=current_load,
                confidence=0.8,
            )

        # Low-carbon scaling window
        if intensity <= threshold and current_load > 0.50:
            # Good time to scale up proactively
            delta = 1 if current_nodes < max_nodes else 0
            self._last_decision_time = now
            return ScaleDecision(
                action="scale_up" if delta > 0 else "noop",
                nodes_delta=delta,
                reason="low_carbon_window_proactive_scale_up",
                carbon_intensity_at_action=intensity,
                forecasted_load=predicted_load,
                current_load=current_load,
                confidence=0.7,
            )

        # Scale down when load is low AND carbon is high or neutral
        if current_load < 0.30 and current_nodes > min_nodes:
            # Check if we can delay scale-down for a cleaner window
            next_clean_window = self._next_low_carbon_window(self._region, now)
            if next_clean_window and next_clean_window < 1800:  # within 30 min
                # Wait for cleaner window to scale down
                pass
            delta = -1
            self._last_decision_time = now
            return ScaleDecision(
                action="scale_down",
                nodes_delta=delta,
                reason="low_load_scale_down",
                carbon_intensity_at_action=intensity,
                current_load=current_load,
                confidence=0.8,
            )

        return ScaleDecision(
            action="noop",
            nodes_delta=0,
            reason="stable",
            current_load=current_load,
            carbon_intensity_at_action=intensity,
        )

    def _compute_intensity(self, region: str, ts: float) -> float:
        """Estimate carbon intensity using time-of-day heuristics."""
        base = _REGIONAL_CARBON_DEFAULTS.get(region, 450.0)

        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        hour = dt.hour

        # Night hours (22:00-06:00 UTC): lower carbon (less peak demand)
        if hour < 6 or hour >= 22:
            return base * 0.65
        # Early morning (6:00-9:00): moderate
        if hour < 9:
            return base * 0.85
        # Peak (9:00-17:00): highest
        if hour < 17:
            return base * 1.0
        # Evening (17:00-22:00): still high
        return base * 0.9

    def _next_low_carbon_window(self, region: str, ts: float) -> float | None:
        """Return seconds until next low-carbon window, or None."""
        base = _REGIONAL_CARBON_DEFAULTS.get(region, 450.0)
        now_dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        current_hour = now_dt.hour

        # Low-carbon windows: 22:00-06:00 (base * 0.65)
        if current_hour < 6:
            # Already in low-carbon window
            return 0.0
        if current_hour >= 22:
            # In evening low-carbon window
            return 0.0

        # Next window starts at 22:00
        next_window = now_dt.replace(hour=22, minute=0, second=0, microsecond=0)
        return (next_window - now_dt).total_seconds()

    def set_carbon_intensity(self, region: str, gco2_per_kwh: float) -> None:
        """Override carbon intensity for a region (e.g. from API)."""
        with self._lock:
            self._last_intensity[region] = CarbonIntensity(
                region=region,
                gco2_per_kwh=gco2_per_kwh,
                timestamp=time.time(),
                source="api",
            )


# ---------------------------------------------------------------------------
# PredictiveScaler
# ---------------------------------------------------------------------------


class PredictiveScaler:
    """Combines traffic forecast + carbon awareness for scaling plans.

    Generates multi-step scaling plans over a configurable horizon,
    then executes them via a pluggable callback (e.g. K8s API, cloud SDK).

    Usage::

        scaler = PredictiveScaler(
            forecaster=traffic_forecaster,
            carbon_scaler=carbon_aware_scaler,
            execute_callback=my_k8s_scaler,
        )
        plan = scaler.plan(horizon="30m")
        scaler.execute(plan)
    """

    def __init__(
        self,
        forecaster: TrafficForecaster,
        carbon_scaler: CarbonAwareScaler,
        execute_callback: Callable[[str, int], bool] | None = None,
        min_nodes: int = 1,
        max_nodes: int = 20,
        scale_up_threshold: float = 0.75,
        scale_down_threshold: float = 0.30,
        cooldown_seconds: float = 60.0,
    ):
        self._forecaster = forecaster
        self._carbon_scaler = carbon_scaler
        self._execute_callback = execute_callback
        self._min_nodes = min_nodes
        self._max_nodes = max_nodes
        self._scale_up_threshold = scale_up_threshold
        self._scale_down_threshold = scale_down_threshold
        self._cooldown = cooldown_seconds

        self._lock = threading.Lock()
        self._last_execution_time = 0.0
        self._executed_actions: list[ScalingAction] = []
        self._current_nodes = min_nodes

    # -- public api ---------------------------------------------------------

    def plan(self, horizon: str = "30m", metric_key: str = "requests_per_sec") -> ScalingPlan:
        """Generate a scaling plan for the given horizon.

        Evaluates traffic forecast and carbon intensity to produce a
        sequence of scaling actions (when to add/remove nodes, how many).

        Args:
            horizon: Forecast horizon (``"5m"``, ``"30m"``, ``"1h"``, etc.).
            metric_key: Which metric to base the plan on.

        Returns:
            A ScalingPlan with ordered actions, cost and carbon impact.
        """
        forecast = self._forecaster.forecast(metric_key, window=horizon)
        actions: list[ScalingAction] = []
        total_cost_impact = 0.0
        carbon_saved = 0.0

        simulated_nodes = self._current_nodes

        if forecast is None:
            # No forecast data - fall back to reactive scaling
            return ScalingPlan(
                horizon=horizon,
                actions=[],
                confidence=0.3,
            )

        now = time.time()

        # Sample forecast at plan points and evaluate scaling needs
        for point in forecast.points:
            offset = point.timestamp - now
            if offset < 0:
                continue

            predicted_load = point.predicted_value
            # Normalize if raw count
            load_ratio = min(1.0, predicted_load / 1000.0) if predicted_load > 1.0 else predicted_load

            intensity = self._carbon_scaler.get_carbon_intensity(at_time=point.timestamp)
            under_carbon_threshold = intensity <= self._carbon_scaler._carbon_threshold

            # Determine if scaling needed
            if load_ratio >= self._scale_up_threshold and simulated_nodes < self._max_nodes:
                # How many nodes to add (proportional to excess load)
                excess = load_ratio - self._scale_up_threshold
                add_count = max(1, int(excess * 4))  # up to 1 node per 25% excess
                add_count = min(add_count, self._max_nodes - simulated_nodes)

                # Prefer scaling during low-carbon windows
                if under_carbon_threshold or load_ratio > 0.85:
                    actions.append(ScalingAction(
                        time_offset_seconds=offset,
                        action="scale_up",
                        node_delta=add_count,
                        reason=f"forecasted_load_{load_ratio:.0%}_carbon_{intensity:.0f}",
                    ))
                    simulated_nodes += add_count
                    total_cost_impact += add_count * 0.50  # placeholder $/hr
                else:
                    # Delay: add a note in the plan, but defer to action on carbon
                    # For now, still add if load is critical
                    if load_ratio > 0.90:
                        actions.append(ScalingAction(
                            time_offset_seconds=offset,
                            action="scale_up",
                            node_delta=add_count,
                            reason=f"critical_forecasted_load_{load_ratio:.0%}",
                        ))
                        simulated_nodes += add_count
                        total_cost_impact += add_count * 0.50

                    # Estimate carbon saved by not scaling during dirty window
                    if intensity > 400:
                        carbon_saved += add_count * 0.1  # placeholder kg/h

            elif load_ratio < self._scale_down_threshold and simulated_nodes > self._min_nodes:
                remove_count = max(1, min(
                    (simulated_nodes - self._min_nodes),
                    int((self._scale_down_threshold - load_ratio) * 4),
                ))
                actions.append(ScalingAction(
                    time_offset_seconds=offset,
                    action="scale_down",
                    node_delta=-remove_count,
                    reason=f"low_forecasted_load_{load_ratio:.0%}",
                ))
                simulated_nodes -= remove_count
                total_cost_impact -= remove_count * 0.50

        confidence = forecast.confidence

        return ScalingPlan(
            horizon=horizon,
            actions=actions,
            total_cost_impact=total_cost_impact,
            carbon_saved_estimate_kg=carbon_saved,
            confidence=confidence,
        )

    def execute(self, plan: ScalingPlan) -> int:
        """Execute a scaling plan via the configured callback.

        Args:
            plan: The ScalingPlan to execute.

        Returns:
            Number of actions successfully executed.
        """
        if not self._execute_callback:
            logger.warning("No execute_callback configured; plan not executed.")
            return 0

        now = time.time()
        if now - self._last_execution_time < self._cooldown:
            logger.info("Cooldown active, skipping plan execution.")
            return 0

        count = 0
        for action in plan.actions:
            try:
                # Determine target action for callback
                k8s_action = "scale_up" if action.node_delta > 0 else "scale_down"
                ok = self._execute_callback(k8s_action, abs(action.node_delta))
                if ok:
                    self._current_nodes += action.node_delta
                    self._last_execution_time = now
                    self._executed_actions.append(action)
                    count += 1
                    logger.info(
                        "Executed {} (delta={}) at offset {}: {}",
                        k8s_action, action.node_delta,
                        action.time_offset_seconds, action.reason,
                    )
                else:
                    logger.warning("Callback returned False for {}", k8s_action)
            except Exception as exc:
                logger.error("Failed to execute action {}: {}", action.action, exc)

        return count

    def set_current_nodes(self, n: int) -> None:
        """Update the tracked node count (e.g. after external scale)."""
        with self._lock:
            self._current_nodes = n

    @property
    def current_nodes(self) -> int:
        return self._current_nodes

    @property
    def executed_actions(self) -> list[ScalingAction]:
        return list(self._executed_actions)


# ---------------------------------------------------------------------------
# HPAMetrics
# ---------------------------------------------------------------------------


class HPAMetrics:
    """Custom HPA metrics for Kubernetes-based autoscaling.

    Produces a metrics endpoint in the OpenMetrics / Prometheus exposition
    format that can be scraped by the Kubernetes metrics infrastructure
    (custom.metrics.k8s.io adapter).

    Usage::

        metrics = HPAMetrics()
        metrics.record_request()
        metrics.record_request()
        # ...
        openmetrics_output = metrics.generate_metrics()
        # Serve at /metrics for Prometheus or HPA scraping
    """

    def __init__(
        self,
        window_seconds: int = 60,
        gpu_count: int = 8,
    ):
        self._window_seconds = window_seconds
        self._gpu_count = gpu_count

        self._request_timestamps: deque[float] = deque(maxlen=100000)
        self._queue_depth: int = 0
        self._gpu_utils: deque[float] = deque(maxlen=self._gpu_count * 10)
        self._lock = threading.Lock()

    # -- public api ---------------------------------------------------------

    def record_request(self) -> None:
        """Record an incoming request."""
        with self._lock:
            self._request_timestamps.append(time.time())

    def record_gpu_util(self, util: float) -> None:
        """Record a GPU utilization sample (0.0-100.0)."""
        with self._lock:
            self._gpu_utils.append(util)

    def set_queue_depth(self, depth: int) -> None:
        """Set the current pending queue depth."""
        with self._lock:
            self._queue_depth = depth

    def get_request_rate(self) -> float:
        """Return requests per second over the current window."""
        with self._lock:
            return self._compute_request_rate()

    def get_queue_depth(self) -> int:
        """Return current pending queue depth."""
        with self._lock:
            return self._queue_depth

    def get_gpu_utilization(self) -> float:
        """Return average GPU utilization (0.0-100.0)."""
        with self._lock:
            if not self._gpu_utils:
                return 0.0
            return sum(self._gpu_utils) / len(self._gpu_utils)

    def generate_metrics(self) -> str:
        """Generate an OpenMetrics / Prometheus exposition format string.

        Returns a multi-line string suitable for serving on a ``/metrics``
        endpoint.  Uses the ``# TYPE`` and ``# HELP`` OpenMetadata conventions
        and includes the ``# EOF`` sentinel for OpenMetrics compliance.

        The output includes:

        - ``distllm_request_rate`` — requests/sec (gauge)
        - ``distllm_queue_depth`` — pending queue depth (gauge)
        - ``distllm_gpu_utilization_pct`` — average GPU util (gauge)
        - ``distllm_scaled_nodes`` — current node count
        """
        with self._lock:
            rate = self._compute_request_rate()
            qd = self._queue_depth
            gpu = sum(self._gpu_utils) / len(self._gpu_utils) if self._gpu_utils else 0.0

        now_ms = int(time.time() * 1000)

        lines = [
            "# HELP distllm_request_rate Request rate (requests/sec).",
            "# TYPE distllm_request_rate gauge",
            f"distllm_request_rate {rate:.2f} {now_ms}",
            "",
            "# HELP distllm_queue_depth Pending queue depth.",
            "# TYPE distllm_queue_depth gauge",
            f"distllm_queue_depth {qd} {now_ms}",
            "",
            "# HELP distllm_gpu_utilization_pct Average GPU utilization in percent.",
            "# TYPE distllm_gpu_utilization_pct gauge",
            f"distllm_gpu_utilization_pct {gpu:.1f} {now_ms}",
            "",
            "# EOF",
        ]
        return "\n".join(lines)

    def snapshot(self) -> dict[str, float]:
        """Return a dict snapshot of current metrics."""
        with self._lock:
            return {
                "request_rate": self._compute_request_rate(),
                "queue_depth": float(self._queue_depth),
                "gpu_utilization_pct": (
                    sum(self._gpu_utils) / len(self._gpu_utils) if self._gpu_utils else 0.0
                ),
            }

    def clear(self) -> None:
        """Reset all metrics counters."""
        with self._lock:
            self._request_timestamps.clear()
            self._queue_depth = 0
            self._gpu_utils.clear()

    # -- internal -----------------------------------------------------------

    def _compute_request_rate(self) -> float:
        """Compute requests/sec from the sliding timestamp window."""
        if not self._request_timestamps:
            return 0.0
        cutoff = time.time() - self._window_seconds
        # Count requests within the window (deque is ordered)
        valid = [t for t in self._request_timestamps if t >= cutoff]
        if not valid:
            return 0.0
        return len(valid) / self._window_seconds


# ---------------------------------------------------------------------------
# Aria
# ---------------------------------------------------------------------------


class Aria:
    """Top-level predictive auto-scaler combining all components.

    Aria orchestrates the TrafficForecaster, CarbonAwareScaler,
    PredictiveScaler, and HPAMetrics into a unified monitoring and
    auto-scaling loop.  Spawns a background thread that periodically
    collects metrics, runs forecasts, and applies scaling decisions.

    Usage::

        aria = Aria(
            min_nodes=2,
            max_nodes=20,
            region="us-east-1",
            execute_callback=my_k8s_scaler,
        )
        aria.start()
        # ... let it run ...
        aria.stop()
        report = aria.stats()
    """

    def __init__(
        self,
        min_nodes: int = 1,
        max_nodes: int = 20,
        region: str = "us-east-1",
        carbon_threshold_gco2: float = 400.0,
        forecast_metric_key: str = "requests_per_sec",
        check_interval_seconds: float = 60.0,
        execute_callback: Callable[[str, int], bool] | None = None,
        enable_torch: bool = True,
    ):
        self._min_nodes = min_nodes
        self._max_nodes = max_nodes
        self._region = region
        self._forecast_metric_key = forecast_metric_key
        self._check_interval = check_interval_seconds
        self._enable_torch = enable_torch

        # Sub-components
        self._forecaster = TrafficForecaster(enable_torch=enable_torch and _TORCH_AVAILABLE)
        self._carbon_scaler = CarbonAwareScaler(
            region=region,
            carbon_threshold_gco2=carbon_threshold_gco2,
        )
        self._predictive_scaler = PredictiveScaler(
            forecaster=self._forecaster,
            carbon_scaler=self._carbon_scaler,
            execute_callback=execute_callback,
            min_nodes=min_nodes,
            max_nodes=max_nodes,
        )
        self._hpa_metrics = HPAMetrics()

        # Threading
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

        # Stats tracking
        self._scaling_records: list[ScalingRecord] = []
        self._total_carbon_saved_kg: float = 0.0
        self._total_cost_impact: float = 0.0
        self._loop_count: int = 0

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        """Start the monitoring and forecasting loop in a background thread."""
        if self._running:
            logger.info("Aria is already running.")
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._loop,
            daemon=True,
            name="aria-autoscaler",
        )
        self._thread.start()
        logger.info(
            "Aria autoscaler started (min={}, max={}, region={}, interval={}s)",
            self._min_nodes,
            self._max_nodes,
            self._region,
            self._check_interval,
        )

    def stop(self) -> None:
        """Stop the monitoring loop."""
        self._running = False
        logger.info("Aria autoscaler stopped.")

    # -- public accessors ---------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Return aggregated statistics.

        Returns:
            Dict with scaling actions, carbon saved, cost impact, and more.
        """
        with self._lock:
            total_ups = sum(1 for r in self._scaling_records if r.action == "scale_up")
            total_downs = sum(1 for r in self._scaling_records if r.action == "scale_down")

            # Latest metrics snapshot
            metrics = self._hpa_metrics.snapshot()

            # Latest forecast
            forecast = self._forecaster.forecast(self._forecast_metric_key, window="1h")
            forecast_summary: dict[str, Any] = {}
            if forecast:
                forecast_summary = {
                    "trend": forecast.trend,
                    "points": len(forecast.points),
                    "confidence": forecast.confidence,
                }

            return {
                "status": "running" if self._running else "stopped",
                "uptime_seconds": self._uptime() if self._running else 0.0,
                "region": self._region,
                "current_nodes": self._predictive_scaler.current_nodes,
                "loop_count": self._loop_count,
                "scaling_actions": {
                    "total": len(self._scaling_records),
                    "scale_ups": total_ups,
                    "scale_downs": total_downs,
                },
                "carbon_saved_kg": self._total_carbon_saved_kg,
                "cost_impact_usd": self._total_cost_impact,
                "current_metrics": metrics,
                "forecast": forecast_summary,
            }

    @property
    def forecaster(self) -> TrafficForecaster:
        """Underlying TrafficForecaster (for direct metric updates)."""
        return self._forecaster

    @property
    def carbon_scaler(self) -> CarbonAwareScaler:
        """Underlying CarbonAwareScaler."""
        return self._carbon_scaler

    @property
    def predictive_scaler(self) -> PredictiveScaler:
        """Underlying PredictiveScaler."""
        return self._predictive_scaler

    @property
    def hpa_metrics(self) -> HPAMetrics:
        """Underlying HPAMetrics collector."""
        return self._hpa_metrics

    # -- internal loop ------------------------------------------------------

    def _loop(self) -> None:
        """Main monitoring loop — runs in background thread."""
        self._start_time = time.time()

        while self._running:
            try:
                self._tick()
            except Exception as exc:
                logger.error("Aria loop error: {}", exc)

            # Sleep with early exit on stop
            for _ in range(max(1, int(self._check_interval / 0.5))):
                if not self._running:
                    return
                time.sleep(0.5)

    def _tick(self) -> None:
        """Single iteration of the monitoring loop."""
        self._loop_count += 1

        # 1. Record current metrics
        metrics = self._hpa_metrics.snapshot()
        self._forecaster.update(
            self._forecast_metric_key,
            metrics["request_rate"],
            time.time(),
        )
        self._forecaster.update(
            "gpu_utilization",
            metrics["gpu_utilization_pct"],
            time.time(),
        )

        # 2. Get carbon intensity
        intensity = self._carbon_scaler.get_carbon_intensity(self._region)

        # 3. Get current load ratio from GPU utilization
        load_ratio = metrics["gpu_utilization_pct"] / 100.0

        # 4. Generate scaling plan
        plan = self._predictive_scaler.plan(
            horizon="30m",
            metric_key=self._forecast_metric_key,
        )

        # 5. Execute actionable steps now
        immediate_actions = [a for a in plan.actions if a.time_offset_seconds <= self._check_interval]
        if immediate_actions:
            # Simplified: execute combined decision
            forecast = self._forecaster.forecast(self._forecast_metric_key, window="30m")
            decision = self._carbon_scaler.should_scale(
                forecast=forecast,
                current_load=load_ratio,
                current_nodes=self._predictive_scaler.current_nodes,
                min_nodes=self._min_nodes,
                max_nodes=self._max_nodes,
            )

            if decision.action != "noop" and decision.nodes_delta != 0:
                # Record the scaling action internally
                before = self._predictive_scaler.current_nodes
                if decision.action == "scale_up":
                    self._predictive_scaler._current_nodes += decision.nodes_delta
                elif decision.action == "scale_down":
                    self._predictive_scaler._current_nodes += decision.nodes_delta

                after = self._predictive_scaler.current_nodes

                record = ScalingRecord(
                    timestamp=time.time(),
                    action=decision.action,
                    node_delta=decision.nodes_delta,
                    nodes_before=before,
                    nodes_after=after,
                    carbon_intensity=intensity,
                    reason=decision.reason,
                )

                with self._lock:
                    self._scaling_records.append(record)
                    # Track carbon savings: if we scaled down during high carbon,
                    # or delayed a scale-up, estimate savings
                    if decision.action == "scale_down" and intensity > 400:
                        self._total_carbon_saved_kg += 0.05  # placeholder

                # Also execute via external callback if configured
                if self._predictive_scaler._execute_callback:
                    try:
                        k8s_action = "scale_up" if decision.nodes_delta > 0 else "scale_down"
                        self._predictive_scaler._execute_callback(k8s_action, abs(decision.nodes_delta))
                    except Exception as exc:
                        logger.error("Aria execute callback failed: {}", exc)

        # 6. Update cost impact
        with self._lock:
            self._total_cost_impact += plan.total_cost_impact

    def _uptime(self) -> float:
        """Return seconds since start."""
        return time.time() - getattr(self, "_start_time", time.time())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_window(window: str) -> int:
    """Convert a window string like ``"1h"``, ``"30m"``, ``"6h"`` to minutes."""
    if not window:
        return 60
    window = window.strip().lower()
    if window.endswith("h"):
        try:
            return int(window[:-1]) * 60
        except ValueError:
            return 60
    if window.endswith("m"):
        try:
            return int(window[:-1])
        except ValueError:
            return 30
    if window.endswith("d"):
        try:
            return int(window[:-1]) * 1440
        except ValueError:
            return 1440
    try:
        return int(window)
    except ValueError:
        return 60


def _compute_trend(history: list[MetricPoint]) -> str:
    """Determine the trend direction from recent history."""
    if len(history) < 5:
        return "stable"

    mid = len(history) // 2
    first_half = sum(p.value for p in history[:mid]) / max(mid, 1)
    second_half = sum(p.value for p in history[mid:]) / max(len(history) - mid, 1)

    ratio = second_half / max(first_half, 1e-10)
    if ratio > 1.10:
        return "up"
    if ratio < 0.90:
        return "down"
    return "stable"


def _std(values: list[float]) -> float:
    """Compute sample standard deviation."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance)
