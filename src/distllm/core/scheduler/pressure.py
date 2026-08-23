"""Decode pressure tracker for adaptive prefill/decode splitting."""

from __future__ import annotations

from collections import deque


class DecodePressureTracker:
    """Tracks decode queue pressure to dynamically adapt prefill/decode split.

    Maintains a rolling window of recent per-token decode latencies and an
    exponential moving average (EMA) for a smooth pressure signal.
    Higher pressure → more decode slots reserved, prefill tokens throttled.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        target_ms_per_token: float = 8.0,
        window_size: int = 32,
    ):
        self._ema: float = 0.0
        self._alpha = alpha
        self._target_ms = target_ms_per_token
        self._sample_count: int = 0
        self._window_size = max(1, window_size)
        self._decode_latencies: deque[float] = deque(maxlen=self._window_size)

    def record_decode_step(self, batch_decode_count: int, elapsed_ms: float) -> None:
        per_token = elapsed_ms / max(batch_decode_count, 1)
        self._decode_latencies.append(per_token)
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
