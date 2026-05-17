"""Statistical anomaly detection for operational metrics.

Tracks rolling baselines using exponential moving average (EMA) and
standard deviation. Fires callbacks when metrics deviate beyond a
configurable number of standard deviations.
"""

import math
import time
from collections import deque
from dataclasses import dataclass, field
from collections.abc import Callable


@dataclass
class MetricBaseline:
    """Rolling baseline for a single metric."""

    window_size: int = 60
    sigma_threshold: float = 3.0
    _samples: deque[float] = field(default_factory=deque)
    _ema: float = 0.0
    _ema_var: float = 0.0
    _alpha: float = 0.1

    def update(self, value: float) -> bool:
        """Update baseline. Returns True if the value is anomalous."""
        self._samples.append(value)
        if len(self._samples) > self.window_size:
            self._samples.popleft()

        # Update EMA and variance
        if len(self._samples) == 1:
            self._ema = value
            self._ema_var = 0.0
        else:
            delta = value - self._ema
            self._ema += self._alpha * delta
            self._ema_var = (1 - self._alpha) * (self._ema_var + self._alpha * delta * delta)

        if len(self._samples) < 10:
            return False  # Warm-up period

        sigma = math.sqrt(self._ema_var) if self._ema_var > 0 else 0.001
        return abs(value - self._ema) > self.sigma_threshold * sigma

    @property
    def mean(self) -> float:
        return sum(self._samples) / len(self._samples) if self._samples else 0.0

    @property
    def std(self) -> float:
        if len(self._samples) < 2:
            return 0.0
        m = self.mean
        return math.sqrt(sum((x - m) ** 2 for x in self._samples) / (len(self._samples) - 1))


@dataclass
class AnomalyEvent:
    """Details of a detected anomaly."""
    metric: str
    value: float
    mean: float
    std: float
    deviation_sigma: float
    timestamp: float


class AnomalyDetector:
    """Detects anomalies in operational metrics via statistical deviation."""

    def __init__(self, sigma_threshold: float = 3.0):
        self._baselines: dict[str, MetricBaseline] = {}
        self._sigma_threshold = sigma_threshold
        self._callbacks: list[Callable[[AnomalyEvent], None]] = []

    def register_metric(
        self,
        name: str,
        window_size: int = 60,
        sigma_threshold: float | None = None,
    ):
        """Register a metric for anomaly detection."""
        self._baselines[name] = MetricBaseline(
            window_size=window_size,
            sigma_threshold=sigma_threshold or self._sigma_threshold,
        )

    def record(self, name: str, value: float) -> AnomalyEvent | None:
        """Record a metric value. Returns an AnomalyEvent if detected."""
        baseline = self._baselines.get(name)
        if baseline is None:
            self.register_metric(name)
            baseline = self._baselines[name]

        if baseline.update(value):
            sigma = max(baseline.std, 0.001)
            event = AnomalyEvent(
                metric=name,
                value=value,
                mean=baseline.mean,
                std=baseline.std,
                deviation_sigma=abs(value - baseline.mean) / sigma,
                timestamp=time.time(),
            )
            for cb in self._callbacks:
                cb(event)
            return event
        return None

    def on_anomaly(self, callback: Callable[[AnomalyEvent], None]):
        """Register a callback for anomaly events."""
        self._callbacks.append(callback)
