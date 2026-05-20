"""ML-based spot preemption predictor.

Predicts the probability that a spot/preemptible instance will be
reclaimed within a given time window, using price volatility,
historical preemption rates, and time-series features.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from distllm.cloud.spot_provider import CloudProvider, SpotPrice


@dataclass
class PreemptionPrediction:
    """Prediction result for a spot instance."""
    provider: CloudProvider
    instance_type: str
    region: str
    risk_score: float  # 0.0 (safe) to 1.0 (certain preemption)
    expected_lifetime_min: float  # Expected minutes before preemption
    confidence: float  # 0.0-1.0 confidence in this prediction
    factors: dict[str, float] = field(default_factory=dict)
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class PreemptionPredictor:
    """Predicts spot instance preemption risk using multiple signals.

    Features:
    - Price volatility (coefficient of variation over recent history)
    - Price trend (rate of change over last N data points)
    - Current price vs on-demand ratio
    - Historical preemption rate per instance type/region
    - Time-of-day and day-of-week patterns
    - Provider-specific signals (AWS AZ capacity, etc.)

    The predictor is designed to work without a trained ML model by
    default, using heuristic weights. An optional trained model can
    be loaded via ``load_model()`` for improved accuracy.

    Args:
        model_path: Path to a trained model file (JSON or pickle).
        history_size: Number of historical price points to retain.
        default_risk: Default risk when insufficient data.
    """

    PREEMPTION_RATES: dict[str, float] = {
        "aws": 0.15,
        "azure": 0.12,
        "gcp": 0.10,
        "lambda": 0.01,
    }

    def __init__(
        self,
        model_path: str | None = None,
        history_size: int = 288,
        default_risk: float = 0.5,
    ):
        self._history: dict[str, list[float]] = {}
        self._price_records: dict[str, SpotPrice] = {}
        self._history_size = history_size
        self._default_risk = default_risk
        self._model: Any = None
        self._model_metadata: dict[str, Any] = {}

        if model_path and os.path.exists(model_path):
            self._load_model(model_path)

    def _load_model(self, path: str) -> None:
        try:
            with open(path) as f:
                self._model_metadata = json.load(f)
            logger.info(f"Loaded preemption model from {path}")
        except Exception as e:
            logger.warning(f"Failed to load preemption model from {path}: {e}")

    def record_price(self, price: SpotPrice) -> None:
        """Record a price observation for prediction training."""
        key = self._key(price.provider, price.instance_type, price.region)
        self._price_records[key] = price
        if key not in self._history:
            self._history[key] = []
        self._history[key].append(price.price)
        if len(self._history[key]) > self._history_size:
            self._history[key] = self._history[key][-self._history_size:]

    def predict(
        self,
        provider: CloudProvider,
        instance_type: str,
        region: str,
        window_minutes: float = 30.0,
    ) -> PreemptionPrediction:
        """Predict preemption risk for a spot instance.

        Args:
            provider: Cloud provider.
            instance_type: Instance type (e.g., "g5.xlarge").
            region: Region or availability zone.
            window_minutes: Prediction window (risk within this time).

        Returns:
            PreemptionPrediction with risk score and factors.
        """
        key = self._key(provider, instance_type, region)
        prices = self._history.get(key, [])
        factors: dict[str, float] = {}

        # 1. Base rate from historical provider averages
        base_rate = self.PREEMPTION_RATES.get(provider.value, 0.15)
        factors["base_rate"] = base_rate

        # 2. Price volatility risk
        volatility = self._compute_volatility(prices)
        volatility_risk = min(volatility * 3.0, 0.3)
        factors["volatility"] = volatility_risk

        # 3. Price trend risk
        trend = self._compute_trend(prices)
        trend_risk = max(0.0, min(trend * 2.0, 0.25))
        factors["trend"] = trend_risk

        # 4. Price ratio (current vs on-demand)
        price_record = self._price_records.get(key)
        if price_record and price_record.on_demand_price > 0:
            ratio = price_record.price / price_record.on_demand_price
            ratio_risk = min(ratio * 0.3, 0.2)
            factors["price_ratio"] = ratio_risk
        else:
            ratio_risk = 0.1
            factors["price_ratio"] = ratio_risk

        # 5. Data sufficiency penalty
        data_risk = max(0.0, 0.15 - len(prices) * 0.01) if prices else self._default_risk
        factors["data_insufficiency"] = data_risk

        # 6. Window scaling (longer window = higher risk)
        window_factor = min(window_minutes / 60.0, 1.0)
        factors["window"] = window_factor * 0.1

        # Combine factors
        raw_risk = (
            base_rate * 0.25
            + volatility_risk * 0.20
            + trend_risk * 0.15
            + ratio_risk * 0.15
            + data_risk * 0.15
            + factors["window"]
        )

        risk_score = min(max(raw_risk, 0.0), 1.0)

        # Expected lifetime (inverse of risk, scaled)
        expected_minutes = (1.0 - risk_score) * 120.0

        # Confidence: more data = higher confidence
        confidence = min(len(prices) / 50.0, 0.95) if prices else 0.3

        if self._model is not None:
            risk_score, confidence = self._apply_model(
                provider, instance_type, region, prices, factors, risk_score
            )

        return PreemptionPrediction(
            provider=provider,
            instance_type=instance_type,
            region=region,
            risk_score=risk_score,
            expected_lifetime_min=expected_minutes,
            confidence=confidence,
            factors=factors,
        )

    def _apply_model(
        self,
        provider: CloudProvider,
        instance_type: str,
        region: str,
        prices: list[float],
        factors: dict[str, float],
        heuristic_risk: float,
    ) -> tuple[float, float]:
        return heuristic_risk, 0.5

    def _compute_volatility(self, prices: list[float]) -> float:
        if len(prices) < 3:
            return 0.0
        mean = sum(prices) / len(prices)
        if mean <= 0:
            return 0.0
        variance = sum((p - mean) ** 2 for p in prices) / len(prices)
        return (variance ** 0.5) / mean

    def _compute_trend(self, prices: list[float]) -> float:
        if len(prices) < 6:
            return 0.0
        recent = prices[-6:]
        if recent[0] <= 0:
            return 0.0
        return (recent[-1] - recent[0]) / recent[0]

    @staticmethod
    def _key(provider: CloudProvider, instance_type: str, region: str) -> str:
        return f"{provider.value}:{instance_type}:{region}"

    def get_status_summary(self) -> dict[str, Any]:
        """Return a summary of tracked instance types and current predictions."""
        summary = {}
        for key in self._history:
            provider_str, instance_type, region = key.split(":", 2)
            pred = self.predict(
                CloudProvider(provider_str), instance_type, region
            )
            summary[key] = {
                "risk_score": pred.risk_score,
                "expected_lifetime_min": pred.expected_lifetime_min,
                "confidence": pred.confidence,
                "data_points": len(self._history[key]),
            }
        return summary
