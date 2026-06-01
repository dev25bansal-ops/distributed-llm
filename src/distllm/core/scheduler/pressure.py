"""Decode pressure tracker for adaptive prefill/decode splitting."""

from __future__ import annotations


class DecodePressureTracker:
    """Tracks decode queue pressure to dynamically adapt prefill/decode split.

    Uses exponential moving average (EMA) for smooth pressure signal.
    Higher pressure → more decode slots reserved, prefill tokens throttled.
    """

    def __init__(self, alpha: float = 0.1, target_ms_per_token: float = 8.0):
        self._ema: float = 0.0
        self._alpha = alpha
        self._target_ms = target_ms_per_token
        self._sample_count: int = 0

    def record_decode_step(self, batch_decode_count: int, elapsed_ms: float) -> None:
        per_token = elapsed_ms / max(batch_decode_count, 1)
        if self._sample_count == 0:
            self._ema = per_token
        else:
            self._ema = self._alpha * per_token + (1 - self._alpha) * self._ema
        self._sample_count += 1

    @property
    def pressure(self) -> float:
        if self._sample_count == 0:
            return 0.0
        return min(1.0, self._ema / max(self._target_ms, 0.1))

    @property
    def avg_ms_per_token(self) -> float:
        if self._sample_count == 0:
            return 0.0
        return self._ema
