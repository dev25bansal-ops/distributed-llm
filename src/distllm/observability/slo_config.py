"""SLO (Service Level Objective) configuration for distributed-llm.

Provides five components for defining, measuring, alerting on, and
visualising SLOs:

* **SLOConfig** — target definitions (availability, latency, TTFT) with
  built-in error-budget computation.
* **AvailabilitySLI** — sliding-window availability measurement that
  excludes health checks and maintenance-window traffic.
* **BurnRateAlerts** — fast/slow dual-window burn-rate alerting with
  Prometheus rule generation.
* **SLO** — composite object that ties a name, target, SLI, and alert
  configuration together; provides error-budget, exhaustion-estimate,
  and summary helpers.
* **SLODashboard** — generates a complete Grafana dashboard JSON model
  with SLO status, burn-rate, error-budget, and latency panels.
"""

from __future__ import annotations

import dataclasses
import json
import math
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Optional yaml — no-op fallback when not installed
# ---------------------------------------------------------------------------

try:
    from yaml import dump as yaml_dump

    YAML_AVAILABLE = True
except ImportError:  # pragma: no cover
    YAML_AVAILABLE = False

    def yaml_dump(*args: Any, **kwargs: Any) -> str:  # type: ignore[misc]
        """Fallback no-op for pyyaml.dump."""
        return "# pyyaml not available\n"


# ===================================================================
# SLOConfig
# ===================================================================


@dataclasses.dataclass(frozen=True)
class SLOConfig:
    """SLO target configuration for a distributed-llm service.

    Attributes:
        availability_target: Target availability as a fraction of
            successful requests (default 0.999 = 99.9 %).
        latency_p50_target_ms: P50 per-request latency target in
            milliseconds (default 500.0).
        latency_p99_target_ms: P99 per-request latency target in
            milliseconds (default 2000.0).
        ttft_p99_target_ms: P99 time-to-first-token target in
            milliseconds (default 1000.0).
    """

    availability_target: float = 0.999
    latency_p50_target_ms: float = 500.0
    latency_p99_target_ms: float = 2000.0
    ttft_p99_target_ms: float = 1000.0

    def __post_init__(self) -> None:
        """Validate target values after construction."""
        if not 0.0 < self.availability_target < 1.0:
            raise ValueError(
                f"availability_target must be in (0, 1), "
                f"got {self.availability_target}"
            )
        if self.latency_p50_target_ms <= 0:
            raise ValueError(
                f"latency_p50_target_ms must be positive, "
                f"got {self.latency_p50_target_ms}"
            )
        if self.latency_p99_target_ms <= 0:
            raise ValueError(
                f"latency_p99_target_ms must be positive, "
                f"got {self.latency_p99_target_ms}"
            )
        if self.ttft_p99_target_ms <= 0:
            raise ValueError(
                f"ttft_p99_target_ms must be positive, "
                f"got {self.ttft_p99_target_ms}"
            )
        if self.latency_p50_target_ms > self.latency_p99_target_ms:
            raise ValueError(
                f"latency_p50_target_ms ({self.latency_p50_target_ms}) must "
                f"be <= latency_p99_target_ms ({self.latency_p99_target_ms})"
            )

    @property
    def error_budget(self) -> float:
        """Maximum allowable error rate (1 - availability_target).

        Returns:
            Float in (0, 1).  For a 99.9 % availability target this is
            0.001 (0.1 %).
        """
        return 1.0 - self.availability_target


# ===================================================================
# AvailabilitySLI
# ===================================================================


@dataclasses.dataclass
class _MinuteBucket:
    """Aggregated request counts for a single minute.

    Attributes:
        minute_epoch: Minutes since 1970-01-01 UTC identifying this
            bucket.
        total: Total requests observed in this minute.
        success: Successful requests in this minute.
        excluded_total: Requests excluded from SLO calculations (health
            checks + maintenance) within *total*.
        excluded_success: Successful-but-excluded requests within
            *success*.
    """

    minute_epoch: int
    total: int = 0
    success: int = 0
    excluded_total: int = 0
    excluded_success: int = 0


class AvailabilitySLI:
    """Sliding-window availability SLI that excludes health checks and
    maintenance-window traffic.

    Maintains per-minute time-series buckets with automatic pruning.
    Thread-safe for concurrent recording and measurement.

    Typical usage::

        sli = AvailabilitySLI()
        sli.record_request(success=True)
        sli.record_request(success=False, is_health_check=True)
        success, total = sli.measure(timedelta(minutes=5))
        ratio = success / total if total else 1.0
    """

    # Standard window labels exposed by ``measure_all``.
    STANDARD_WINDOWS: Dict[str, timedelta] = {
        "5m": timedelta(minutes=5),
        "1h": timedelta(hours=1),
        "1d": timedelta(days=1),
        "30d": timedelta(days=30),
    }

    def __init__(self, max_retention: timedelta = timedelta(days=31)) -> None:
        """Initialise the SLI.

        Args:
            max_retention: How far back to keep bucket data.  Data older
                than this is pruned during ``record_request``.  Defaults
                to 31 days (~ 44 640 buckets).
        """
        self._lock = threading.Lock()
        self._max_retention = max_retention
        self._buckets: List[_MinuteBucket] = []

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_request(
        self,
        success: bool,
        *,
        is_health_check: bool = False,
        is_maintenance: bool = False,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Record a single request outcome.

        Args:
            success: Whether the request completed successfully.
            is_health_check: Mark this request as a health check, which
                is excluded from SLO calculations by default.
            is_maintenance: Mark this request as occurring during a
                maintenance window, which is excluded from SLO
                calculations by default.
            timestamp: When the request occurred (defaults to now).
        """
        now = timestamp if timestamp is not None else datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        minute_epoch = int(now.timestamp() // 60)

        with self._lock:
            self._ensure_bucket(minute_epoch)
            bucket = self._buckets[-1]
            bucket.total += 1
            if success:
                bucket.success += 1
            if is_health_check or is_maintenance:
                bucket.excluded_total += 1
                if success:
                    bucket.excluded_success += 1

    def _ensure_bucket(self, minute_epoch: int) -> None:
        """Ensure a bucket exists for *minute_epoch*, pruning old data.

        Caller must hold ``_lock``.
        """
        if self._buckets and self._buckets[-1].minute_epoch == minute_epoch:
            return  # current bucket still matches

        # Prune buckets whose data has fallen outside the retention window.
        cutoff = minute_epoch - int(self._max_retention.total_seconds() // 60)
        self._buckets = [b for b in self._buckets if b.minute_epoch >= cutoff]

        self._buckets.append(_MinuteBucket(minute_epoch=minute_epoch))

    # ------------------------------------------------------------------
    # Measurement
    # ------------------------------------------------------------------

    def measure(
        self,
        window: timedelta,
        *,
        exclude_health_checks: bool = True,
        exclude_maintenance: bool = True,
    ) -> Tuple[int, int]:
        """Count successful and total requests over a sliding window.

        Args:
            window: Duration of the measurement window (e.g. 5 minutes,
                1 hour, 1 day, 30 days).
            exclude_health_checks: If ``True`` (default), health-check
                requests are excluded from the counts.
            exclude_maintenance: If ``True`` (default), requests that
                occurred during maintenance windows are excluded.

        Returns:
            ``(success, total)`` pair for the window.  ``total`` is 0
            when no data exists for the window.
        """
        now_dt = datetime.now(timezone.utc)
        cutoff = now_dt - window
        cutoff_epoch = int(cutoff.timestamp() // 60)

        total = 0
        success = 0

        with self._lock:
            for bucket in self._buckets:
                if bucket.minute_epoch < cutoff_epoch:
                    continue
                if exclude_health_checks or exclude_maintenance:
                    total += bucket.total - bucket.excluded_total
                    success += bucket.success - bucket.excluded_success
                else:
                    total += bucket.total
                    success += bucket.success

        return success, total

    def measure_all(self) -> Dict[str, Tuple[int, int]]:
        """Measure availability across all standard windows.

        Returns:
            Dictionary mapping window labels (``"5m"``, ``"1h"``,
            ``"1d"``, ``"30d"``) to ``(success, total)`` tuples.
        """
        return {
            label: self.measure(window)
            for label, window in self.STANDARD_WINDOWS.items()
        }

    def availability(self, window: timedelta) -> float:
        """Return the availability ratio over *window*.

        Returns 1.0 if no data exists for the window.
        """
        success, total = self.measure(window)
        return success / total if total else 1.0

    # ------------------------------------------------------------------
    # Inspection / helpers
    # ------------------------------------------------------------------

    @property
    def bucket_count(self) -> int:
        """Number of stored minute buckets (for diagnostics)."""
        with self._lock:
            return len(self._buckets)

    def clear(self) -> None:
        """Clear all recorded data (useful in tests)."""
        with self._lock:
            self._buckets.clear()


# ===================================================================
# BurnRateAlerts
# ===================================================================


class BurnRateAlerts:
    """Multi-window burn-rate alert configuration.

    Implements the Google SRE recommended approach: two burn-rate
    windows (fast and slow) are evaluated independently against
    configured burn-rate multipliers.  The generated Prometheus alerting
    rules fire when **both** windows exceed their respective thresholds,
    which reduces false positives from transient blips.

    Burn rate is defined as::

        burn_rate = observed_error_rate / (1.0 - slo_target)

    A burn rate of 1.0 means the error budget is being consumed at
    exactly the planned pace.  Values > 1.0 indicate over-consumption.

    Defaults (Google SRE standard for 99.9 % SLO):

        * Fast burn: 1-hour window, evaluated every 1 minute, rate > 10x
        * Slow burn: 6-hour window, evaluated every 5 minutes, rate > 3x
    """

    def __init__(
        self,
        fast_burn_window: timedelta = timedelta(hours=1),
        fast_burn_eval: timedelta = timedelta(minutes=1),
        fast_burn_rate: float = 10.0,
        slow_burn_window: timedelta = timedelta(hours=6),
        slow_burn_eval: timedelta = timedelta(minutes=5),
        slow_burn_rate: float = 3.0,
        slo_name: str = "distllm",
        severity: str = "critical",
    ) -> None:
        """Initialise burn-rate alert configuration.

        Args:
            fast_burn_window: Duration of the fast-burn evaluation
                window (default 1 hour).
            fast_burn_eval: Prometheus ``for`` duration for the
                fast-burn rule (default 1 minute).
            fast_burn_rate: Burn-rate multiplier threshold for the
                fast window (default 10.0).
            slow_burn_window: Duration of the slow-burn evaluation
                window (default 6 hours).
            slow_burn_eval: Prometheus ``for`` duration for the
                slow-burn rule (default 5 minutes).
            slow_burn_rate: Burn-rate multiplier threshold for the
                slow window (default 3.0).
            slo_name: Name prefix used in generated Prometheus rules
                and labels (default ``"distllm"``).
            severity: Alert severity label for fast-burn alerts
                (default ``"critical"``).  Slow-burn alerts always use
                ``"warning"``.

        Raises:
            ValueError: If any time window or burn rate is zero or
                negative.
        """
        if fast_burn_window <= timedelta(0):
            raise ValueError("fast_burn_window must be positive")
        if slow_burn_window <= timedelta(0):
            raise ValueError("slow_burn_window must be positive")
        if fast_burn_rate <= 0:
            raise ValueError("fast_burn_rate must be positive")
        if slow_burn_rate <= 0:
            raise ValueError("slow_burn_rate must be positive")

        self._fast_burn_window = fast_burn_window
        self._fast_burn_eval = fast_burn_eval
        self._fast_burn_rate = fast_burn_rate
        self._slow_burn_window = slow_burn_window
        self._slow_burn_eval = slow_burn_eval
        self._slow_burn_rate = slow_burn_rate
        self._slo_name = slo_name
        self._severity = severity

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def fast_burn_window(self) -> timedelta:
        """Fast-burn evaluation window duration."""
        return self._fast_burn_window

    @property
    def fast_burn_eval(self) -> timedelta:
        """Prometheus ``for`` duration for fast-burn alert."""
        return self._fast_burn_eval

    @property
    def fast_burn_rate(self) -> float:
        """Burn-rate multiplier threshold for the fast window."""
        return self._fast_burn_rate

    @property
    def slow_burn_window(self) -> timedelta:
        """Slow-burn evaluation window duration."""
        return self._slow_burn_window

    @property
    def slow_burn_eval(self) -> timedelta:
        """Prometheus ``for`` duration for slow-burn alert."""
        return self._slow_burn_eval

    @property
    def slow_burn_rate(self) -> float:
        """Burn-rate multiplier threshold for the slow window."""
        return self._slow_burn_rate

    @property
    def slo_name(self) -> str:
        """Name prefix used in generated Prometheus rules."""
        return self._slo_name

    # ------------------------------------------------------------------
    # Prometheus rules generation
    # ------------------------------------------------------------------

    def generate_prometheus_rules(self) -> str:
        """Generate Prometheus recording and alerting rules as YAML.

        Produces recording rules for burn-rate calculation and alerting
        rules for fast and slow burn-rate breaches.  The output is
        suitable for use with PrometheusOperator ``PrometheusRule`` CRDs
        or file-based rule loading.

        Generated rules:

        * ``<slo_name>:error_ratio:rate<fast>m``  —  recording rule
          for the observed error ratio over the fast window.
        * ``<slo_name>:burn_rate:fast``  —  recording rule for the
          fast-window burn rate.
        * ``<slo_name>:burn_rate:slow``  —  recording rule for the
          slow-window burn rate.
        * ``<SLO_NAME>SLOFastBurnRate``  —  alerting rule (severity:
          configured severity).
        * ``<SLO_NAME>SLOSlowBurnRate``  —  alerting rule (severity:
          warning).

        Returns:
            YAML string.  Returns a fallback comment string if PyYAML
            is not installed.
        """
        fast_minutes = int(self._fast_burn_window.total_seconds() // 60)
        slow_minutes = int(self._slow_burn_window.total_seconds() // 60)
        fast_eval = max(1, int(self._fast_burn_eval.total_seconds() // 60))
        slow_eval = max(1, int(self._slow_burn_eval.total_seconds() // 60))

        error_ratio_recording = (
            f"{self._slo_name}:error_ratio:rate{fast_minutes}m"
        )
        burn_rate_fast_recording = f"{self._slo_name}:burn_rate:fast"
        burn_rate_slow_recording = f"{self._slo_name}:burn_rate:slow"

        rules: List[Dict[str, Any]] = [
            # -- Recording rule: error ratio over fast window --------------
            {
                "record": error_ratio_recording,
                "expr": (
                    f"sum(rate(errors_total{{slo=\"{self._slo_name}\"}}"
                    f"[{fast_minutes}m])) "
                    f"/ sum(rate(requests_total{{slo=\"{self._slo_name}\"}}"
                    f"[{fast_minutes}m]))"
                ),
                "labels": {"slo": self._slo_name},
            },
            # -- Recording rule: fast burn rate ----------------------------
            {
                "record": burn_rate_fast_recording,
                "expr": (
                    f"({error_ratio_recording} "
                    f"/ (1 - slo_target{{slo=\"{self._slo_name}\"}}))"
                ),
                "labels": {"slo": self._slo_name, "window": "fast"},
            },
            # -- Recording rule: slow burn rate ----------------------------
            {
                "record": burn_rate_slow_recording,
                "expr": (
                    f"({self._slo_name}:error_ratio:rate{slow_minutes}m "
                    f"/ (1 - slo_target{{slo=\"{self._slo_name}\"}}))"
                ),
                "labels": {"slo": self._slo_name, "window": "slow"},
            },
            # -- Alerting rule: fast burn rate -----------------------------
            {
                "alert": f"{self._slo_name.upper()}SLOFastBurnRate",
                "expr": (
                    f"{burn_rate_fast_recording} > {self._fast_burn_rate}"
                ),
                "for": f"{fast_eval}m",
                "labels": {
                    "severity": self._severity,
                    "slo": self._slo_name,
                    "burn_rate": "fast",
                },
                "annotations": {
                    "summary": "SLO fast burn-rate exceeded",
                    "description": (
                        f"Burn rate over {fast_minutes}m window "
                        f"({self._fast_burn_rate}x) exceeded for SLO "
                        f"\"{self._slo_name}\".  Error budget is being "
                        f"consumed dangerously fast."
                    ),
                },
            },
            # -- Alerting rule: slow burn rate -----------------------------
            {
                "alert": f"{self._slo_name.upper()}SLOSlowBurnRate",
                "expr": (
                    f"{burn_rate_slow_recording} > {self._slow_burn_rate}"
                ),
                "for": f"{slow_eval}m",
                "labels": {
                    "severity": "warning",
                    "slo": self._slo_name,
                    "burn_rate": "slow",
                },
                "annotations": {
                    "summary": "SLO slow burn-rate exceeded",
                    "description": (
                        f"Burn rate over {slow_minutes}m window "
                        f"({self._slow_burn_rate}x) exceeded for SLO "
                        f"\"{self._slo_name}\".  Error budget is being "
                        f"consumed steadily."
                    ),
                },
            },
        ]

        payload: Dict[str, Any] = {
            "groups": [
                {
                    "name": f"{self._slo_name}_slo_alerts",
                    "rules": rules,
                }
            ]
        }

        if YAML_AVAILABLE:
            return yaml_dump(payload, default_flow_style=False, sort_keys=False)
        return yaml_dump(payload)

    # ------------------------------------------------------------------
    # Local burn-rate evaluation (in-process, for summary / dashboard)
    # ------------------------------------------------------------------

    def evaluate(
        self,
        slo_target: float,
        error_budget: float,
        fast_burn_errors: int = 0,
        fast_burn_total: int = 1,
        slow_burn_errors: int = 0,
        slow_burn_total: int = 1,
    ) -> str:
        """Evaluate burn-rate status for current in-process measurements.

        Calculates the burn rate for each window and returns ``"fire"``
        when **both** windows exceed their respective thresholds (with a
        small epsilon tolerance to avoid floating-point edge cases).

        Args:
            slo_target: Target SLO as a fraction (e.g. 0.999).
            error_budget: Maximum allowable error rate, typically
                ``1.0 - slo_target``.
            fast_burn_errors: Observed errors in the fast-burn window.
            fast_burn_total: Total requests in the fast-burn window.
            slow_burn_errors: Observed errors in the slow-burn window.
            slow_burn_total: Total requests in the slow-burn window.

        Returns:
            ``"fire"`` when **both** windows exceed their respective
            burn-rate thresholds; ``"ok"`` otherwise.

        Raises:
            ValueError: If denominators are zero or *slo_target* is
                out of range.
        """
        if not 0.0 < slo_target < 1.0:
            raise ValueError(
                f"slo_target must be in (0, 1), got {slo_target}"
            )
        if error_budget <= 0.0:
            raise ValueError(
                f"error_budget must be positive, got {error_budget}"
            )
        if fast_burn_total <= 0 or slow_burn_total <= 0:
            raise ValueError("burn total denominators must be positive")

        fast_error_rate = fast_burn_errors / fast_burn_total
        slow_error_rate = slow_burn_errors / slow_burn_total

        fast_rate = fast_error_rate / error_budget
        slow_rate = slow_error_rate / error_budget

        epsilon = 1e-12
        if (
            fast_rate + epsilon >= self._fast_burn_rate
            and slow_rate + epsilon >= self._slow_burn_rate
        ):
            return "fire"
        return "ok"


# ===================================================================
# SLO
# ===================================================================


class SLO:
    """A complete Service Level Objective.

    ``SLO`` ties a named target (``SLOConfig``) with an availability
    measurement source (``AvailabilitySLI``) and burn-rate alert
    configuration (``BurnRateAlerts``).  It provides:

    * ``error_budget_remaining()`` — fraction of budget left over a
      given window.
    * ``exhaustion_estimate()`` — projected datetime of budget
      exhaustion at the current error rate.
    * ``summary()`` — snapshot dictionary with all status fields.

    Typical usage::

        config = SLOConfig()
        sli = AvailabilitySLI()
        alerts = BurnRateAlerts(slo_name="api_availability")
        slo = SLO(name="api_availability", config=config, sli=sli,
                  burn_rate_alerts=alerts)

        slo.record_request(success=True)
        slo.record_request(success=False)
        report = slo.summary()
    """

    def __init__(
        self,
        name: str,
        config: SLOConfig,
        sli: AvailabilitySLI,
        burn_rate_alerts: Optional[BurnRateAlerts] = None,
    ) -> None:
        """Initialise an SLO.

        Args:
            name: Human-readable SLO name (e.g. ``"api_availability"``).
            config: SLO target configuration.
            sli: Availability SLI measurement source.
            burn_rate_alerts: Burn-rate alert configuration.  When
                ``None``, a default ``BurnRateAlerts`` is created using
                *name* as the ``slo_name``.
        """
        if not name:
            raise ValueError("name must not be empty")

        self._name = name
        self._config = config
        self._sli = sli
        self._burn_rate_alerts = burn_rate_alerts or BurnRateAlerts(
            slo_name=name
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """SLO name."""
        return self._name

    @property
    def config(self) -> SLOConfig:
        """SLO target configuration."""
        return self._config

    @property
    def sli(self) -> AvailabilitySLI:
        """Underlying availability SLI."""
        return self._sli

    @property
    def burn_rate_alerts(self) -> BurnRateAlerts:
        """Burn-rate alert configuration."""
        return self._burn_rate_alerts

    # ------------------------------------------------------------------
    # Delegated recording
    # ------------------------------------------------------------------

    def record_request(
        self,
        success: bool,
        *,
        is_health_check: bool = False,
        is_maintenance: bool = False,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Record a request outcome via the underlying SLI.

        Args:
            success: Whether the request succeeded.
            is_health_check: Mark as a health check (excluded from SLO).
            is_maintenance: Mark as maintenance traffic (excluded from
                SLO).
            timestamp: When the request occurred (defaults to now).
        """
        self._sli.record_request(
            success=success,
            is_health_check=is_health_check,
            is_maintenance=is_maintenance,
            timestamp=timestamp,
        )

    # ------------------------------------------------------------------
    # SLO calculations
    # ------------------------------------------------------------------

    def error_budget_remaining(
        self, window: timedelta = timedelta(days=30)
    ) -> float:
        """Fraction of error budget remaining over *window*.

        Calculated as::

            remaining = 1.0 - (observed_error_rate / max_error_rate)

        where ``max_error_rate = 1.0 - availability_target``.

        Args:
            window: Measurement window (default 30 days).

        Returns:
            Float in [0.0, 1.0].  1.0 means the full budget is intact;
            0.0 means the budget is exhausted.  Returns 1.0 when no
            data is available for the window.
        """
        success, total = self._sli.measure(window)
        if total == 0:
            return 1.0

        observed_error_rate = 1.0 - (success / total)
        max_error_rate = self._config.error_budget
        if max_error_rate <= 0.0:
            return 1.0

        budget_consumed = observed_error_rate / max_error_rate
        return max(0.0, 1.0 - budget_consumed)

    def exhaustion_estimate(
        self,
        window: timedelta = timedelta(hours=1),
    ) -> Optional[datetime]:
        """Estimate when the error budget will be exhausted.

        Projects the current error rate (measured over *window*) forward
        to determine when the remaining error budget reaches zero.

        Args:
            window: Time window for measuring the current error rate
                (default 1 hour).

        Returns:
            ``datetime`` of estimated exhaustion in UTC, or ``None`` if
            the error rate is zero (budget will never exhaust at the
            current rate) or no data is available.
        """
        success, total = self._sli.measure(window)
        if total == 0:
            return None

        observed_error_rate = 1.0 - (success / total)
        max_error_rate = self._config.error_budget

        if observed_error_rate <= 0.0 or max_error_rate <= 0.0:
            return None

        window_seconds = window.total_seconds()
        consumption_per_second = observed_error_rate / window_seconds

        if consumption_per_second <= 0.0:
            return None

        # Compute the remaining absolute error budget
        remaining = self.error_budget_remaining(timedelta(days=30))
        remaining_absolute = max_error_rate * remaining

        seconds_until_exhaustion = remaining_absolute / consumption_per_second
        return datetime.now(timezone.utc) + timedelta(
            seconds=seconds_until_exhaustion
        )

    def summary(self) -> Dict[str, Any]:
        """Return a comprehensive snapshot of current SLO status.

        The returned dictionary includes:

        * ``name`` — SLO name.
        * ``availability_target``, ``error_budget_max`` — target values.
        * ``latency_p50_target_ms``, ``latency_p99_target_ms``,
          ``ttft_p99_target_ms`` — latency targets.
        * ``availability`` — dict of window label -> ratio.
        * ``error_budget_remaining`` — 30d budget remaining (fraction).
        * ``error_budget_remaining_pct`` — same as a percentage.
        * ``exhaustion_estimate`` — ISO-8601 datetime or ``None``.
        * ``burn_rate_status`` — ``"fire"`` or ``"ok"``.
        * ``timestamp`` — ISO-8601 datetime of the snapshot.

        Returns:
            Dictionary suitable for serialisation (JSON, logging, etc.).
        """
        avail_windows = self._sli.measure_all()

        avail_values: Dict[str, float] = {}
        for label, (success, total) in avail_windows.items():
            avail_values[label] = success / total if total else 1.0

        budget_30d = self.error_budget_remaining(timedelta(days=30))

        # Burn-rate evaluation for supporting windows.
        # Default to "ok" when no data exists (avoid division by zero).
        fast_s, fast_t = self._sli.measure(
            self._burn_rate_alerts.fast_burn_window
        )
        slow_s, slow_t = self._sli.measure(
            self._burn_rate_alerts.slow_burn_window
        )

        if fast_t > 0 and slow_t > 0:
            burn_rate_status = self._burn_rate_alerts.evaluate(
                slo_target=self._config.availability_target,
                error_budget=self._config.error_budget,
                fast_burn_errors=fast_t - fast_s,
                fast_burn_total=fast_t,
                slow_burn_errors=slow_t - slow_s,
                slow_burn_total=slow_t,
            )
        else:
            burn_rate_status = "ok"

        return {
            "name": self._name,
            "availability_target": self._config.availability_target,
            "error_budget_max": self._config.error_budget,
            "latency_p50_target_ms": self._config.latency_p50_target_ms,
            "latency_p99_target_ms": self._config.latency_p99_target_ms,
            "ttft_p99_target_ms": self._config.ttft_p99_target_ms,
            "availability": avail_values,
            "error_budget_remaining": round(budget_30d, 6),
            "error_budget_remaining_pct": round(budget_30d * 100, 4),
            "exhaustion_estimate": (
                self.exhaustion_estimate().isoformat()
                if self.exhaustion_estimate()
                else None
            ),
            "burn_rate_status": burn_rate_status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


# ===================================================================
# SLODashboard  —  Grafana dashboard generator
# ===================================================================

# Grafana dashboard constants
_GRAFANA_PANEL_H = 8  # default panel height
_GRAFANA_PANEL_W = 8  # default panel width (out of 24 columns)
_GRAFANA_ROW_GAP = 1  # vertical gap between rows
_GRAFANA_TOTAL_W = 24  # total grid width


class SLODashboard:
    """Generates Grafana dashboards for SLO monitoring.

    Produces a complete, importable Grafana dashboard JSON model with
    panels for:

    * SLO status — current availability stat with colour thresholds.
    * Error budget — gauge showing remaining budget.
    * Burn rate — stat panel showing ``"fire"`` / ``"ok"``.
    * Availability graph — time-series over the configured window.
    * Latency P50, P99, and TTFT P99 graphs — time-series with
      threshold reference lines.

    Each ``SLO`` in the dashboard produces one block of three rows.
    Multiple SLOs are stacked vertically.

    Typical usage::

        dashboard = SLODashboard(slos=[slo1, slo2])
        grafana_json = dashboard.generate_grafana_json()
        # Import via Grafana API or "Import dashboard" UI.
    """

    def __init__(
        self,
        slos: List[SLO],
        datasource: str = "Prometheus",
        dashboard_title: str = "DistLLM SLO Dashboard",
        refresh_interval: str = "30s",
    ) -> None:
        """Initialise the dashboard.

        Args:
            slos: List of SLO instances to include in the dashboard
                panels.
            datasource: Grafana datasource name (default
                ``"Prometheus"``).
            dashboard_title: Dashboard title.
            refresh_interval: Auto-refresh interval (e.g. ``"30s"``).
        """
        self._slos = list(slos)
        self._datasource = datasource
        self._title = dashboard_title
        self._refresh = refresh_interval

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def slos(self) -> List[SLO]:
        """Immutable snapshot of configured SLOs."""
        return list(self._slos)

    @property
    def datasource(self) -> str:
        """Grafana datasource name."""
        return self._datasource

    @property
    def title(self) -> str:
        """Dashboard title."""
        return self._title

    # ------------------------------------------------------------------
    # Dashboard generation
    # ------------------------------------------------------------------

    def generate_grafana_json(self) -> str:
        """Generate a complete, importable Grafana dashboard JSON model.

        The dashboard uses a 24-column grid.  Each SLO produces:

        * Row 1 (y = base): 3 stat/gauge panels (w=8 each).
        * Row 2 (y = base + 9): 2 time-series panels (w=12 each).
        * Row 3 (y = base + 18): 2 time-series panels (w=12 each).

        Returns:
            Pretty-printed JSON string of the Grafana dashboard
            definition.
        """
        panels: List[Dict[str, Any]] = []
        panel_id = [1]  # mutable counter for unique IDs

        for slo in self._slos:
            self._add_slo_panels(slo, panels, panel_id)

        dashboard: Dict[str, Any] = {
            "title": self._title,
            "description": (
                "SLO dashboard generated by DistLLM SLODashboard.  "
                "Monitors availability, burn rate, error budget, and "
                "latency across configured SLOs."
            ),
            "refresh": self._refresh,
            "tags": ["slo", "distllm", "observability"],
            "schemaVersion": 39,
            "version": 1,
            "time": {
                "from": "now-6h",
                "to": "now",
            },
            "timepicker": {
                "refresh_intervals": [
                    "5s",
                    "10s",
                    "30s",
                    "1m",
                    "5m",
                    "15m",
                    "30m",
                    "1h",
                    "2h",
                    "1d",
                ],
            },
            "panels": panels,
            "templating": {
                "list": [
                    {
                        "name": "datasource",
                        "type": "datasource",
                        "query": "prometheus",
                        "current": {
                            "value": self._datasource,
                            "text": self._datasource,
                        },
                        "hide": 0,
                    }
                ]
            },
        }

        return json.dumps(dashboard, indent=2, default=str)

    # ------------------------------------------------------------------
    # Row builder
    # ------------------------------------------------------------------

    def _add_slo_panels(
        self,
        slo: SLO,
        panels: List[Dict[str, Any]],
        panel_id: List[int],
    ) -> None:
        """Append all panels for a single SLO to *panels*.

        Args:
            slo: The SLO instance.
            panels: Mutable list being accumulated.
            panel_id: Mutable single-element list with the next ID.
        """
        if not panels:
            current_y = 0
        else:
            # Find the maximum y + h among existing panels, then round
            # up to a multiple of the block height.
            max_bottom = max(
                (p["gridPos"]["y"] + p["gridPos"]["h"]) for p in panels
            )
            # Round up to next block boundary
            block_height = 3 * (_GRAFANA_PANEL_H + _GRAFANA_ROW_GAP)
            current_y = math.ceil(max_bottom / block_height) * block_height

        w8 = _GRAFANA_PANEL_W  # 8
        h8 = _GRAFANA_PANEL_H  # 8

        # ---- Row 1: stat / gauge panels (y = current_y) -------------
        row1_y = current_y

        # Panel 1: SLO Status
        panels.append(
            self._build_status_panel(slo, panel_id[0], row1_y, 0, w8, h8)
        )
        panel_id[0] += 1

        # Panel 2: Error Budget
        panels.append(
            self._build_error_budget_panel(
                slo, panel_id[0], row1_y, w8, w8, h8
            )
        )
        panel_id[0] += 1

        # Panel 3: Burn Rate
        panels.append(
            self._build_burn_rate_panel(
                slo, panel_id[0], row1_y, w8 * 2, w8, h8
            )
        )
        panel_id[0] += 1

        # ---- Row 2: time-series (y = current_y + 9) -----------------
        row2_y = current_y + _GRAFANA_PANEL_H + _GRAFANA_ROW_GAP

        # Panel 4: Availability graph (w=12)
        panels.append(
            self._build_availability_graph(
                slo, panel_id[0], row2_y, 0, 12, h8
            )
        )
        panel_id[0] += 1

        # Panel 5: Latency P50 graph (w=12)
        panels.append(
            self._build_latency_timeseries(
                slo=slo,
                pid=panel_id[0],
                y=row2_y,
                x=12,
                w=12,
                h=h8,
                title=f"{slo.name} — P50 Latency",
                query=(
                    "histogram_quantile(0.50, "
                    "sum(rate(distllm_node_latency_seconds_bucket"
                    "{slo=\"%s\"}[5m])) by (le))"
                )
                % slo.name,
                threshold_seconds=slo.config.latency_p50_target_ms / 1000.0,
                unit="s",
            )
        )
        panel_id[0] += 1

        # ---- Row 3: time-series (y = current_y + 18) ----------------
        row3_y = current_y + 2 * (_GRAFANA_PANEL_H + _GRAFANA_ROW_GAP)

        # Panel 6: Latency P99 graph (w=12)
        panels.append(
            self._build_latency_timeseries(
                slo=slo,
                pid=panel_id[0],
                y=row3_y,
                x=0,
                w=12,
                h=h8,
                title=f"{slo.name} — P99 Latency",
                query=(
                    "histogram_quantile(0.99, "
                    "sum(rate(distllm_node_latency_seconds_bucket"
                    "{slo=\"%s\"}[5m])) by (le))"
                )
                % slo.name,
                threshold_seconds=slo.config.latency_p99_target_ms / 1000.0,
                unit="s",
            )
        )
        panel_id[0] += 1

        # Panel 7: TTFT P99 graph (w=12)
        panels.append(
            self._build_latency_timeseries(
                slo=slo,
                pid=panel_id[0],
                y=row3_y,
                x=12,
                w=12,
                h=h8,
                title=f"{slo.name} — TTFT P99",
                query=(
                    "histogram_quantile(0.99, "
                    "sum(rate(distllm_ttft_seconds_bucket"
                    "{slo=\"%s\"}[5m])) by (le))"
                )
                % slo.name,
                threshold_seconds=slo.config.ttft_p99_target_ms / 1000.0,
                unit="s",
            )
        )
        panel_id[0] += 1

    # ==============================================================
    # Individual panel builders
    # ==============================================================

    @staticmethod
    def _build_status_panel(
        slo: SLO, pid: int, y: int, x: int, w: int, h: int
    ) -> Dict[str, Any]:
        """Stat panel showing current 30d availability with colour thresholds.

        Green when >= target, yellow when close, red when below.
        """
        return {
            "id": pid,
            "title": f"{slo.name} — Availability (30d)",
            "type": "stat",
            "gridPos": {"h": h, "w": w, "x": x, "y": y},
            "datasource": {"type": "prometheus", "uid": "$datasource"},
            "fieldConfig": {
                "defaults": {
                    "unit": "percentunit",
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {
                                "color": "semi-dark-red",
                                "value": None,
                            },
                            {
                                "color": "semi-dark-red",
                                "value": (
                                    slo.config.availability_target * 0.98
                                ),
                            },
                            {
                                "color": "semi-dark-yellow",
                                "value": slo.config.availability_target,
                            },
                            {
                                "color": "semi-dark-green",
                                "value": 1.0,
                            },
                        ],
                    },
                    "color": {"mode": "thresholds"},
                    "mappings": [],
                    "min": 0.0,
                    "max": 1.0,
                },
                "overrides": [],
            },
            "targets": [
                {
                    "expr": (
                        f"sum(rate("
                        f"  requests_total{{slo=\"{slo.name}\"}}[30d]))"
                    ),
                    "legendFormat": "Availability",
                    "refId": "A",
                },
            ],
            "options": {
                "reduceOptions": {
                    "calcs": ["lastNotNull"],
                    "fields": "",
                    "values": False,
                },
                "orientation": "auto",
                "textMode": "auto",
                "colorMode": "value",
                "graphMode": "area",
                "justifyMode": "auto",
            },
        }

    @staticmethod
    def _build_error_budget_panel(
        slo: SLO, pid: int, y: int, x: int, w: int, h: int
    ) -> Dict[str, Any]:
        """Gauge panel showing error budget remaining.

        Green when > 50 %, yellow when > 20 %, red when <= 20 %.
        """
        return {
            "id": pid,
            "title": f"{slo.name} — Error Budget",
            "type": "gauge",
            "gridPos": {"h": h, "w": w, "x": x, "y": y},
            "datasource": {"type": "prometheus", "uid": "$datasource"},
            "fieldConfig": {
                "defaults": {
                    "unit": "percentunit",
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {"color": "semi-dark-red", "value": None},
                            {"color": "semi-dark-red", "value": 0.0},
                            {"color": "semi-dark-yellow", "value": 0.2},
                            {"color": "semi-dark-green", "value": 0.5},
                            {"color": "semi-dark-green", "value": 1.0},
                        ],
                    },
                    "color": {"mode": "thresholds"},
                    "min": 0.0,
                    "max": 1.0,
                },
                "overrides": [],
            },
            "targets": [
                {
                    "expr": (
                        f"1 - ("
                        f"  sum(rate("
                        f"    errors_total{{slo=\"{slo.name}\"}}"
                        f"    [30d]))"
                        f"  / (sum(rate("
                        f"    requests_total{{slo=\"{slo.name}\"}}"
                        f"    [30d])) * {slo.config.error_budget})"
                        f")"
                    ),
                    "legendFormat": "Error Budget Remaining",
                    "refId": "A",
                },
            ],
            "options": {
                "reduceOptions": {
                    "calcs": ["lastNotNull"],
                    "fields": "",
                    "values": False,
                },
                "showThresholdLabels": False,
                "showThresholdMarkers": True,
                "orientation": "auto",
            },
        }

    @staticmethod
    def _build_burn_rate_panel(
        slo: SLO, pid: int, y: int, x: int, w: int, h: int
    ) -> Dict[str, Any]:
        """Stat panel showing current burn-rate status (fire / ok).

        Red background when firing, green when OK.
        """
        return {
            "id": pid,
            "title": f"{slo.name} — Burn Rate",
            "type": "stat",
            "gridPos": {"h": h, "w": w, "x": x, "y": y},
            "datasource": {"type": "prometheus", "uid": "$datasource"},
            "fieldConfig": {
                "defaults": {
                    "unit": "none",
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {"color": "semi-dark-green", "value": None},
                            {"color": "semi-dark-green", "value": 0},
                            {"color": "semi-dark-red", "value": 1},
                        ],
                    },
                    "color": {"mode": "thresholds"},
                    "mappings": [
                        {
                            "type": "value",
                            "options": {
                                "0": {"text": "OK", "color": "green"},
                                "1": {"text": "FIRE", "color": "red"},
                            },
                        },
                    ],
                    "min": 0,
                    "max": 1,
                },
                "overrides": [],
            },
            "targets": [
                {
                    "expr": (
                        f"max("
                        f"  {slo.name}:burn_rate:fast "
                        f"  > {slo.burn_rate_alerts.fast_burn_rate}"
                        f"  and "
                        f"  {slo.name}:burn_rate:slow "
                        f"  > {slo.burn_rate_alerts.slow_burn_rate}"
                        f")"
                    ),
                    "legendFormat": "Burn Rate",
                    "refId": "A",
                },
            ],
            "options": {
                "reduceOptions": {
                    "calcs": ["lastNotNull"],
                    "fields": "",
                    "values": False,
                },
                "orientation": "auto",
                "textMode": "auto",
                "colorMode": "value",
                "graphMode": "none",
                "justifyMode": "auto",
            },
        }

    @staticmethod
    def _build_availability_graph(
        slo: SLO, pid: int, y: int, x: int, w: int, h: int
    ) -> Dict[str, Any]:
        """Time-series panel showing availability over the last 6 hours.

        Includes a threshold line at the SLO target.
        """
        target_pct = slo.config.availability_target * 100
        return {
            "id": pid,
            "title": f"{slo.name} — Availability (6h)",
            "type": "timeseries",
            "gridPos": {"h": h, "w": w, "x": x, "y": y},
            "datasource": {"type": "prometheus", "uid": "$datasource"},
            "fieldConfig": {
                "defaults": {
                    "unit": "percentunit",
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {"color": "semi-dark-red", "value": None},
                            {
                                "color": "semi-dark-red",
                                "value": (
                                    slo.config.availability_target * 0.98
                                ),
                            },
                            {
                                "color": "semi-dark-yellow",
                                "value": (
                                    slo.config.availability_target
                                ),
                            },
                            {
                                "color": "semi-dark-green",
                                "value": 1.0,
                            },
                        ],
                    },
                    "color": {"mode": "palette-classic"},
                    "custom": {
                        "lineInterpolation": "smooth",
                        "spanNulls": False,
                        "showPoints": "never",
                        "gradientMode": "opacity",
                    },
                },
                "overrides": [],
            },
            "targets": [
                {
                    "expr": (
                        f"sum(rate("
                        f"  requests_total{{slo=\"{slo.name}\"}}[5m]))"
                    ),
                    "legendFormat": "Availability",
                    "refId": "A",
                },
            ],
            "options": {
                "legend": {
                    "displayMode": "list",
                    "placement": "bottom",
                    "showLegend": True,
                },
                "tooltip": {"mode": "multi", "sort": "desc"},
            },
        }

    @staticmethod
    def _build_latency_timeseries(
        slo: SLO,
        pid: int,
        y: int,
        x: int,
        w: int,
        h: int,
        title: str,
        query: str,
        threshold_seconds: float,
        unit: str = "s",
    ) -> Dict[str, Any]:
        """Time-series panel showing a latency quantile over time.

        Includes a threshold reference line dashed in red.
        """
        return {
            "id": pid,
            "title": title,
            "type": "timeseries",
            "gridPos": {"h": h, "w": w, "x": x, "y": y},
            "datasource": {"type": "prometheus", "uid": "$datasource"},
            "fieldConfig": {
                "defaults": {
                    "unit": unit,
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {"color": "semi-dark-green", "value": None},
                            {
                                "color": "semi-dark-red",
                                "value": threshold_seconds,
                            },
                        ],
                    },
                    "color": {"mode": "palette-classic"},
                    "custom": {
                        "lineInterpolation": "smooth",
                        "spanNulls": False,
                        "showPoints": "never",
                        "gradientMode": "opacity",
                    },
                },
                "overrides": [
                    {
                        "matcher": {"id": "byName", "options": "SLO Target"},
                        "properties": [
                            {
                                "id": "custom.lineStyle",
                                "value": {"fill": "dash", "dash": [10, 10]},
                            },
                            {
                                "id": "color",
                                "value": {"mode": "fixed"},
                            },
                            {"id": "custom.lineWidth", "value": 1},
                        ],
                    },
                ],
            },
            "targets": [
                {
                    "expr": query,
                    "legendFormat": "Latency",
                    "refId": "A",
                },
                {
                    "expr": f"{threshold_seconds}",
                    "legendFormat": "SLO Target",
                    "refId": "B",
                },
            ],
            "options": {
                "legend": {
                    "displayMode": "list",
                    "placement": "bottom",
                    "showLegend": True,
                },
                "tooltip": {"mode": "multi", "sort": "desc"},
            },
        }


__all__ = [
    "AvailabilitySLI",
    "BurnRateAlerts",
    "SLO",
    "SLOConfig",
    "SLODashboard",
]
