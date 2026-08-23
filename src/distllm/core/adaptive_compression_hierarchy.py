"""Adaptive multi-level KV cache compression hierarchy.

Dynamically selects the optimal compression level (FP16/FP8/INT4/2-bit)
per-request based on quality monitoring with automatic regression detection.

Architecture::

    Request arrives
         │
         ▼
    Session Quality Monitor ─── tracks PSNR trends across requests
         │
         ├── Quality improving → try more aggressive compression
         ├── Quality stable    → stay at current level
         └── Quality regressing → fall back to safer level
                │
                ▼
         Auto-disable if regression persists

    Per-Request Quality Monitor
         │
         ├── Records per-token PSNR between compressed/uncompressed
         ├── Tracks acceptance rate from verifier
         └── Reports to session monitor

Usage::

    hierarchy = AdaptiveCompressionHierarchy(kv_cache=kv_cache)
    level = hierarchy.select_level(request_id="req-1")
    kv_cache.compress(level)
    # ... after request completes ...
    hierarchy.report_quality(request_id="req-1", psnr=38.5, accept_rate=0.85)
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


# ── Compression levels ───────────────────────────────────────────────────────

@dataclass
class CompressionLevel:
    """A single compression level in the hierarchy.

    Attributes:
        name: Human-readable name.
        method: Method string passed to KVCache.compress().
        bits: Target bit width (16, 8, 4, 2).
        ratio: Expected compression ratio (uncompressed / compressed).
        quality_baseline: Expected PSNR baseline for quality monitoring.
    """
    name: str
    method: str
    bits: int
    ratio: float
    quality_baseline: float  # Expected PSNR

    def __post_init__(self) -> None:
        # Validate ordering by bits (lower = more compression)
        pass


# Hierarchy ordered from safest to most aggressive
_COMPRESSION_HIERARCHY: list[CompressionLevel] = [
    CompressionLevel("FP16", "fp16", 16, 1.0, 50.0),     # Baseline (no compression)
    CompressionLevel("FP8", "fp8", 8, 2.0, 45.0),        # Minimal quality loss
    CompressionLevel("INT4", "int4", 4, 4.0, 38.0),      # Moderate compression
    CompressionLevel("2-bit", "2bit", 2, 8.0, 30.0),     # Aggressive compression
]


# ── Per-Request Quality Monitor ──────────────────────────────────────────────

@dataclass
class RequestQualityReport:
    """Quality report for a single request."""
    request_id: str
    compression_level: str  # e.g. "int4"
    psnr: float              # Peak signal-to-noise ratio vs uncompressed
    accept_rate: float       # Verifier acceptance rate (0.0-1.0)
    tokens_generated: int = 0
    re_runs: int = 0
    latency_s: float = 0.0
    timestamp: float = field(default_factory=time.time)


# ── Session Quality Monitor ──────────────────────────────────────────────────

class SessionQualityMonitor:
    """Tracks compression quality across requests and detects regression.

    Maintains a sliding window of recent quality reports per compression
    level and decides when to promote (more aggressive) or demote (safer).

    Auto-disable logic::
        If quality drops below threshold for N consecutive requests,
        disable the current level and fall back.
    """

    def __init__(
        self,
        window_size: int = 20,
        psnr_threshold: float = 3.0,   # dB drop before flagging regression
        regression_limit: int = 5,      # consecutive bad requests → disable
    ):
        self._window_size = window_size
        self._psnr_threshold = psnr_threshold
        self._regression_limit = regression_limit

        # Per-level quality history: level_name -> deque of PSNR values
        self._history: dict[str, deque[float]] = {}
        # Per-level baseline (stable average, updated only on non-regression reports)
        self._baseline: dict[str, float] = {}
        self._lock = threading.RLock()

        # Disabled levels
        self._disabled: set[str] = set()

        # Current regression counters per level
        self._regression_count: dict[str, int] = {}

    def report(self, report: RequestQualityReport) -> None:
        """Record a quality report and check for regression."""
        with self._lock:
            level = report.compression_level
            if level not in self._history:
                self._history[level] = deque(maxlen=self._window_size)
            self._history[level].append(report.psnr)

            # Establish baseline from first N reports (stable reference)
            if level not in self._baseline and len(self._history[level]) >= 3:
                baseline_samples = list(self._history[level])[:3]
                # Only set baseline if the first reports show stable quality
                if max(baseline_samples) - min(baseline_samples) < self._psnr_threshold:
                    self._baseline[level] = sum(baseline_samples) / len(baseline_samples)
                    logger.debug(f"Quality baseline for {level}: {self._baseline[level]:.1f} dB")

            # Check regression against the stable baseline
            if level in self._baseline:
                if report.psnr < self._baseline[level] - self._psnr_threshold:
                    self._regression_count[level] = self._regression_count.get(level, 0) + 1
                    logger.warning(
                        f"Quality regression detected for {level}: "
                        f"PSNR={report.psnr:.1f} vs baseline={self._baseline[level]:.1f} "
                        f"({self._regression_count[level]}/{self._regression_limit})"
                    )

                    if self._regression_count[level] >= self._regression_limit:
                        self._disabled.add(level)
                        logger.warning(
                            f"Auto-disabled compression level {level} "
                            f"after {self._regression_limit} consecutive regressions"
                        )
                else:
                    # Reset regression counter on good report
                    self._regression_count[level] = 0

    def best_level(self, current_level: str) -> str:
        """Select the best compression level given quality history.

        Args:
            current_level: The currently active compression level.

        Returns:
            Recommended level name (may be the same, more aggressive,
            or less aggressive if regression detected).
        """
        with self._lock:
            hierarchy_names = [l.name for l in _COMPRESSION_HIERARCHY]
            current_idx = hierarchy_names.index(current_level) if current_level in hierarchy_names else 0

            # Check if current level is disabled
            if current_level in self._disabled:
                # Fall back to the next safe level
                for idx in range(current_idx - 1, -1, -1):
                    if hierarchy_names[idx] not in self._disabled:
                        logger.info(f"Quality monitor: falling back from {current_level} to {hierarchy_names[idx]}")
                        return hierarchy_names[idx]
                return hierarchy_names[0]  # Ultimate fallback

            # Try promoting to a more aggressive level if quality exceeds baseline
            if current_idx < len(hierarchy_names) - 1:
                next_level = hierarchy_names[current_idx + 1]
                if next_level not in self._disabled:
                    bl = self._baseline.get(current_level, 0.0)
                    level_config = _COMPRESSION_HIERARCHY[current_idx]
                    if bl > level_config.quality_baseline and len(self._history.get(current_level, [])) >= 5:
                        logger.info(f"Quality monitor: promoting from {current_level} to {next_level}")
                        return next_level

            return current_level

    @property
    def disabled_levels(self) -> set[str]:
        """Levels that have been auto-disabled due to quality regression."""
        with self._lock:
            return set(self._disabled)

    @property
    def stats(self) -> dict:
        with self._lock:
            return {
                "disabled_levels": sorted(self._disabled),
                "regression_counts": dict(self._regression_count),
                "per_level_samples": {k: len(v) for k, v in self._history.items()},
            }


# ── Adaptive Compression Hierarchy ──────────────────────────────────────────

class AdaptiveCompressionHierarchy:
    """Adaptive multi-level KV cache compression with quality monitoring.

    Manages the compression level hierarchy, selects the best level per
    request based on session-level quality trends, and collects per-request
    quality reports.

    Usage::

        ach = AdaptiveCompressionHierarchy(kv_cache=kv_cache)
        level = ach.select_level(request_id="req-1")
        # kv_cache is compressed externally by the caller
        # ... run inference with compressed KV cache ...
        ach.report_quality(
            request_id="req-1",
            level="int4",
            psnr=38.5,
            accept_rate=0.85,
            tokens_generated=128,
            re_runs=2,
            latency_s=1.2,
        )
    """

    def __init__(
        self,
        kv_cache: Any = None,
        initial_level: str = "INT4",
        window_size: int = 20,
        psnr_threshold: float = 3.0,
        regression_limit: int = 5,
    ):
        self._kv_cache = kv_cache
        self._current_level = initial_level
        self._session_monitor = SessionQualityMonitor(
            window_size=window_size,
            psnr_threshold=psnr_threshold,
            regression_limit=regression_limit,
        )

        # Per-request tracking
        self._request_levels: dict[str, str] = {}
        self._lock = threading.RLock()

    def select_level(self, request_id: str) -> str:
        """Select the best compression level for a request.

        Args:
            request_id: Unique request identifier.

        Returns:
            Compression level name (e.g., "INT4", "FP8").
        """
        with self._lock:
            level = self._session_monitor.best_level(self._current_level)
            self._request_levels[request_id] = level

            # Apply compression to KV cache
            if self._kv_cache is not None:
                level_config = self._get_level(level)
                if level_config:
                    try:
                        self._kv_cache.compress(level_config.method)
                        logger.debug(f"Compressed KV cache at {level}")
                    except Exception as e:
                        logger.warning(f"Compression failed for {level}: {e}")
                        # Fall back to FP16
                        level = "FP16"
                        self._request_levels[request_id] = level

            self._current_level = level
            return level

    def report_quality(
        self,
        request_id: str,
        level: str,
        psnr: float,
        accept_rate: float,
        tokens_generated: int = 0,
        re_runs: int = 0,
        latency_s: float = 0.0,
    ) -> None:
        """Report quality metrics after a request completes.

        This feeds into the session-level quality monitor to detect
        regression and adjust future compression level selections.
        """
        report = RequestQualityReport(
            request_id=request_id,
            compression_level=level,
            psnr=psnr,
            accept_rate=accept_rate,
            tokens_generated=tokens_generated,
            re_runs=re_runs,
            latency_s=latency_s,
        )
        self._session_monitor.report(report)

        with self._lock:
            self._request_levels.pop(request_id, None)

    def get_level_for_request(self, request_id: str) -> str | None:
        """Get the compression level selected for a request."""
        with self._lock:
            return self._request_levels.get(request_id)

    @property
    def current_level(self) -> str:
        return self._current_level

    @property
    def disabled_levels(self) -> set[str]:
        return self._session_monitor.disabled_levels

    @staticmethod
    def _get_level(name: str) -> CompressionLevel | None:
        for level in _COMPRESSION_HIERARCHY:
            if level.name == name:
                return level
        return None

    @property
    def stats(self) -> dict:
        base = self._session_monitor.stats
        base["current_level"] = self._current_level
        base["active_requests"] = len(self._request_levels)
        return base


# ── Integration helper for CompressedSpeculativeDecoder ─────────────────────

def compress_with_adaptive_hierarchy(
    decoder: Any,
    input_ids: torch.Tensor,
    max_new_tokens: int = 256,
    hierarchy: AdaptiveCompressionHierarchy | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Generate tokens with adaptive compression hierarchy.

    Wraps CompressedSpeculativeDecoder.generate() with adaptive
    compression level selection and per-request quality reporting.

    Args:
        decoder: CompressedSpeculativeDecoder instance.
        input_ids: Prompt token IDs.
        max_new_tokens: Maximum tokens to generate.
        hierarchy: AdaptiveCompressionHierarchy instance.
        **kwargs: Forwarded to decoder.generate().

    Returns:
        Generated token IDs.
    """
    if hierarchy is None:
        return decoder.generate(input_ids, max_new_tokens=max_new_tokens, **kwargs)

    request_id = f"req-{time.time_ns()}"
    level = hierarchy.select_level(request_id)

    try:
        # Run generation with compressed KV cache
        output = decoder.generate(input_ids, max_new_tokens=max_new_tokens, **kwargs)

        # Estimate quality from decoder stats
        stats = decoder.stats
        re_runs = stats.get("re_runs", 0)
        compressed_calls = stats.get("compressed_calls", 0)
        accept_rate = 1.0 - (re_runs / max(compressed_calls, 1))
        # Rough PSNR estimate from acceptance rate
        psnr = 30.0 + accept_rate * 20.0  # 30-50 dB range

        hierarchy.report_quality(
            request_id=request_id,
            level=level,
            psnr=psnr,
            accept_rate=accept_rate,
            tokens_generated=output.shape[1] - input_ids.shape[1],
            re_runs=re_runs,
        )
        return output
    except Exception as e:
        logger.error(f"Adaptive compression generation failed: {e}")
        # Fall back to direct generation
        return decoder.generate(input_ids, max_new_tokens=max_new_tokens, **kwargs)
