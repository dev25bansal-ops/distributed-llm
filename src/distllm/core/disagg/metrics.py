from __future__ import annotations

import time
from collections import deque
from typing import Any



class RollingWindow:
    """Fixed-size rolling window for tracking recent values."""

    def __init__(self, maxlen: int = 100):
        self._values: deque[tuple[float, float]] = deque(maxlen=maxlen)

    def add(self, value: float) -> None:
        self._values.append((time.time(), value))

    def avg(self) -> float:
        if not self._values:
            return 0.0
        return sum(v for _, v in self._values) / len(self._values)

    def p50(self) -> float:
        if not self._values:
            return 0.0
        sorted_vals = sorted(v for _, v in self._values)
        return sorted_vals[len(sorted_vals) // 2]

    def p99(self) -> float:
        if not self._values:
            return 0.0
        sorted_vals = sorted(v for _, v in self._values)
        idx = int(len(sorted_vals) * 0.99)
        return sorted_vals[min(idx, len(sorted_vals) - 1)]

    def latest(self) -> float | None:
        if not self._values:
            return None
        return self._values[-1][1]

    def count(self) -> int:
        return len(self._values)

    def reset(self) -> None:
        self._values.clear()


class PoolMetricsCollector:
    """Collects and exposes metrics for a single pool (prefill or decode).

    Tracks latency, throughput, error rates, and queue depths.
    """

    def __init__(self, pool_name: str, window_size: int = 1000):
        self.pool_name = pool_name
        self.latency = RollingWindow(maxlen=window_size)
        self.batch_latency = RollingWindow(maxlen=window_size)
        self.request_count = 0
        self.error_count = 0
        self.total_tokens = 0
        self._start_time = time.time()

    def record_request(self, latency_ms: float, tokens: int = 0, success: bool = True) -> None:
        self.request_count += 1
        self.latency.add(latency_ms)
        self.total_tokens += tokens
        if not success:
            self.error_count += 1

    def record_batch(self, batch_size: int, latency_ms: float) -> None:
        self.batch_latency.add(latency_ms)

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self._start_time

    @property
    def error_rate(self) -> float:
        if self.request_count == 0:
            return 0.0
        return self.error_count / self.request_count

    @property
    def throughput_rps(self) -> float:
        uptime = self.uptime_seconds
        if uptime < 1.0:
            return 0.0
        return self.request_count / uptime

    @property
    def throughput_tps(self) -> float:
        uptime = self.uptime_seconds
        if uptime < 1.0:
            return 0.0
        return self.total_tokens / uptime

    def snapshot(self) -> dict[str, Any]:
        return {
            "pool": self.pool_name,
            "uptime_seconds": self.uptime_seconds,
            "request_count": self.request_count,
            "error_count": self.error_count,
            "error_rate": round(self.error_rate, 4),
            "throughput_rps": round(self.throughput_rps, 2),
            "throughput_tps": round(self.throughput_tps, 2),
            "latency": {
                "avg_ms": round(self.latency.avg(), 2),
                "p50_ms": round(self.latency.p50(), 2),
                "p99_ms": round(self.latency.p99(), 2),
                "latest_ms": round(self.latency.latest() or 0.0, 2),
                "samples": self.latency.count(),
            },
            "batch_latency": {
                "avg_ms": round(self.batch_latency.avg(), 2),
                "p50_ms": round(self.batch_latency.p50(), 2),
                "samples": self.batch_latency.count(),
            },
        }

    def reset(self) -> None:
        self.latency.reset()
        self.batch_latency.reset()
        self.request_count = 0
        self.error_count = 0
        self.total_tokens = 0
        self._start_time = time.time()


class DisaggMetrics:
    """Aggregated metrics across prefill and decode pools."""

    def __init__(self, window_size: int = 1000):
        self.prefill = PoolMetricsCollector("prefill", window_size=window_size)
        self.decode = PoolMetricsCollector("decode", window_size=window_size)

    def snapshot(self) -> dict[str, Any]:
        return {
            "prefill": self.prefill.snapshot(),
            "decode": self.decode.snapshot(),
        }

    def summary(self) -> str:
        s = self.snapshot()
        pre = s["prefill"]
        dec = s["decode"]
        return (
            f"Prefill: {pre['throughput_rps']} rps, "
            f"{pre['latency']['avg_ms']}ms avg, "
            f"{pre['error_rate']*100:.1f}% err | "
            f"Decode: {dec['throughput_rps']} rps, "
            f"{dec['latency']['avg_ms']}ms avg, "
            f"{dec['error_rate']*100:.1f}% err"
        )

    def reset(self) -> None:
        self.prefill.reset()
        self.decode.reset()
