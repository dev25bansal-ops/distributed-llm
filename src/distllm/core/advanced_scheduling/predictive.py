"""Predictive batch scheduling using historical data."""

from __future__ import annotations

import time
from collections import deque
from typing import Any


class PredictiveBatchScheduler:
    """Predicts optimal batch size from historical latency data."""

    def __init__(self, history_size: int = 100):
        self._latency_history: deque = deque(maxlen=history_size)
        self._batch_history: deque = deque(maxlen=history_size)

    def record(self, batch_size: int, latency_ms: float) -> None:
        self._batch_history.append(batch_size)
        self._latency_history.append(latency_ms)

    def predict_optimal_batch_size(self, target_latency_ms: float = 100.0) -> int:
        """Predict the batch size that achieves target latency."""
        if len(self._latency_history) < 10:
            return 8  # Default

        # Simple linear regression: latency = a * batch_size + b
        n = len(self._latency_history)
        x = list(self._batch_history)
        y = list(self._latency_history)

        x_mean = sum(x) / n
        y_mean = sum(y) / n

        numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
        denominator = sum((xi - x_mean) ** 2 for xi in x)

        if denominator == 0:
            return 8

        a = numerator / denominator
        b = y_mean - a * x_mean

        if a <= 0:
            return 32  # Latency doesn't increase with batch size

        optimal = int((target_latency_ms - b) / a)
        return max(1, min(optimal, 64))
