"""Alerting configuration for distributed-llm.

Provides runbook registration, severity-based routing,
SLO burn-rate evaluation, maintenance windows, and
Prometheus / Alertmanager YAML generation.

pyyaml is imported optionally; when unavailable all YAML generation
becomes a silent no-op so the application never crashes.
"""

from __future__ import annotations

import dataclasses
import threading
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

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


# ---------------------------------------------------------------------------
# Severity levels
# ---------------------------------------------------------------------------

class AlertSeverity(str, Enum):
    """Standard alert severity levels."""

    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


# ---------------------------------------------------------------------------
# Burn-rate evaluation results
# ---------------------------------------------------------------------------

class BurnRateResult(str, Enum):
    """Outcome of a burn-rate evaluation."""

    FIRE = "fire"
    OK = "ok"


# ---------------------------------------------------------------------------
# RunbookRegistry
# ---------------------------------------------------------------------------

class RunbookRegistry:
    """Maps alert names to runbook URLs.

    Thread-safe registry that associates each known alert name with a
    link to its operational runbook (playbook).

    Typical usage::

        registry = RunbookRegistry()
        registry.register("HighErrorRate", "https://runbooks/internal/high-error-rate")
        url = registry.get("HighErrorRate")  # returns the URL or None
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._runbooks: Dict[str, str] = {}

    def register(self, alert_name: str, url: str) -> None:
        """Register a runbook URL for *alert_name*.

        Args:
            alert_name: Canonical alert name (e.g. ``"HighErrorRate"``).
            url: Full URL or path to the runbook.

        Raises:
            ValueError: If *alert_name* is empty.
        """
        if not alert_name:
            raise ValueError("alert_name must not be empty")
        with self._lock:
            self._runbooks[alert_name] = url

    def get(self, alert_name: str) -> Optional[str]:
        """Return the runbook URL for *alert_name*, or ``None``.

        Args:
            alert_name: Canonical alert name.
        """
        with self._lock:
            return self._runbooks.get(alert_name)

    def __contains__(self, alert_name: str) -> bool:
        """Check whether *alert_name* has a registered runbook."""
        with self._lock:
            return alert_name in self._runbooks

    def __len__(self) -> int:
        """Return number of registered runbooks."""
        with self._lock:
            return len(self._runbooks)


# ---------------------------------------------------------------------------
# Receiver types for AlertRouter
# ---------------------------------------------------------------------------

class ReceiverType(str, Enum):
    """Supported notification channel types."""

    PAGERDUTY = "pagerduty"
    SLACK = "slack"
    LOG = "log"


# ---------------------------------------------------------------------------
# AlertRule — a single routing rule
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class AlertRule:
    """Describes a single routing rule for an alert.

    Attributes:
        severities: Severities this rule matches (empty = all).
        teams: Team labels this rule matches (empty = all).
        alert_names: Specific alert names this rule matches (empty = all).
        receivers: Notification channel types to notify when matched.
    """

    severities: List[AlertSeverity] = dataclasses.field(default_factory=list)
    teams: List[str] = dataclasses.field(default_factory=list)
    alert_names: List[str] = dataclasses.field(default_factory=list)
    receivers: List[ReceiverType] = dataclasses.field(default_factory=list)

    def matches(self, severity: AlertSeverity, team: str, alert_name: str) -> bool:
        """Check if this rule applies to the given alert attributes.

        Empty filter-lists are treated as wildcards (match everything).
        """
        if self.severities and severity not in self.severities:
            return False
        if self.teams and team not in self.teams:
            return False
        if self.alert_names and alert_name not in self.alert_names:
            return False
        return True


# ---------------------------------------------------------------------------
# AlertRouter
# ---------------------------------------------------------------------------

class AlertRouter:
    """Routes alerts to notification channels based on severity, team, and name.

    Supports multiple receivers per alert via configurable routing rules.
    Default rules route by severity:

    * ``critical`` -> PagerDuty
    * ``warning`` -> Slack
    * ``info`` -> log

    Callers may add custom rules with :meth:`add_rule`.  Rules are evaluated
    in order; the **first** matching rule determines the receivers.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # Default severity-based rules
        self._rules: List[AlertRule] = [
            AlertRule(
                severities=[AlertSeverity.CRITICAL],
                receivers=[ReceiverType.PAGERDUTY],
            ),
            AlertRule(
                severities=[AlertSeverity.WARNING],
                receivers=[ReceiverType.SLACK],
            ),
            AlertRule(
                severities=[AlertSeverity.INFO],
                receivers=[ReceiverType.LOG],
            ),
        ]

    def add_rule(self, rule: AlertRule) -> None:
        """Prepend a custom routing rule (evaluated before defaults).

        Args:
            rule: The :class:`AlertRule` to insert.
        """
        with self._lock:
            self._rules.insert(0, rule)

    def route(
        self,
        severity: AlertSeverity,
        alert_name: str,
        team: str = "platform",
    ) -> List[ReceiverType]:
        """Return the receiver list for an alert with the given attributes.

        Args:
            severity: Alert severity.
            alert_name: Canonical alert name.
            team: Team label for the alert.

        Returns:
            List of :class:`ReceiverType` values to notify.  Empty if no
            rule matched (should not happen with the default rules).
        """
        with self._lock:
            for rule in self._rules:
                if rule.matches(severity, team, alert_name):
                    return list(rule.receivers)
        # Fallback: log only
        return [ReceiverType.LOG]

    @property
    def rules(self) -> List[AlertRule]:
        """Return a snapshot of current routing rules."""
        with self._lock:
            return list(self._rules)


# ---------------------------------------------------------------------------
# SLOBurnRateAlert
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class BurnRateWindows:
    """Time-window configuration for burn-rate evaluation.

    Attributes:
        fast_burn_window: Duration of the fast-burn evaluation window
            (default 1 hour).
        fast_burn_measurement: Measurement interval inside the fast-burn
            window (default 5 minutes).
        slow_burn_window: Duration of the slow-burn evaluation window
            (default 6 hours).
        slow_burn_measurement: Measurement interval inside the slow-burn
            window (default 30 minutes).
    """

    fast_burn_window: timedelta = timedelta(hours=1)
    fast_burn_measurement: timedelta = timedelta(minutes=5)
    slow_burn_window: timedelta = timedelta(hours=6)
    slow_burn_measurement: timedelta = timedelta(minutes=30)


class SLOBurnRateAlert:
    """Multi-window, multi-burn-rate SLO alert evaluator.

    Implements the Google SRE workbook approach: two burn-rate windows
    (fast and slow) are evaluated independently against configured burn
    rate multipliers.  When **both** windows exceed their threshold the
    alert fires, reducing false positives from transient blips.

    Burn rate is defined as::

        burn_rate = observed_error_rate / (1.0 - slo_target)

    A burn rate of 1.0 means the error budget is being consumed at
    exactly the planned pace.  Values > 1.0 indicate over-consumption.

    Window & threshold defaults (Google SRE standard):

    * Fast burn: 1-hour window, burn rate >= 14.0
    * Slow burn: 6-hour window, burn rate >= 2.0
    """

    def __init__(
        self,
        windows: Optional[BurnRateWindows] = None,
        fast_burn_threshold: float = 14.0,
        slow_burn_threshold: float = 2.0,
    ) -> None:
        """Initialize the evaluator.

        Args:
            windows: Custom burn-rate windows, or ``None`` for defaults.
            fast_burn_threshold: Burn-rate multiplier for the fast window
                (default 14.0).
            slow_burn_threshold: Burn-rate multiplier for the slow window
                (default 2.0).
        """
        self._windows = windows or BurnRateWindows()
        self._fast_burn_threshold = fast_burn_threshold
        self._slow_burn_threshold = slow_burn_threshold

    @property
    def windows(self) -> BurnRateWindows:
        """Current burn-rate window configuration."""
        return self._windows

    @property
    def fast_burn_threshold(self) -> float:
        """Burn-rate multiplier for the fast window."""
        return self._fast_burn_threshold

    @property
    def slow_burn_threshold(self) -> float:
        """Burn-rate multiplier for the slow window."""
        return self._slow_burn_threshold

    def evaluate(
        self,
        slo_target: float,
        error_budget: float,
        fast_burn_errors: int = 0,
        fast_burn_total: int = 1,
        slow_burn_errors: int = 0,
        slow_burn_total: int = 1,
    ) -> BurnRateResult:
        """Evaluate burn-rate against SLO targets.

        Calculates the burn rate for each window and fires when **both**
        exceed their respective thresholds (with a small epsilon tolerance
        to avoid floating-point edge cases).

        Args:
            slo_target: Target SLO as a fraction (e.g. ``0.99`` for 99%).
            error_budget: Maximum allowable error rate as a fraction
                (typically ``1.0 - slo_target``).
            fast_burn_errors: Observed errors in the fast-burn window.
            fast_burn_total: Total requests in the fast-burn window.
            slow_burn_errors: Observed errors in the slow-burn window.
            slow_burn_total: Total requests in the slow-burn window.

        Returns:
            ``BurnRateResult.FIRE`` when **both** windows exceed their
            respective burn-rate thresholds; ``BurnRateResult.OK``
            otherwise.

        Raises:
            ValueError: If denominators are zero or *slo_target* is out
                of range.
        """
        if not 0.0 < slo_target < 1.0:
            raise ValueError(f"slo_target must be in (0, 1), got {slo_target}")
        if error_budget <= 0.0:
            raise ValueError(f"error_budget must be positive, got {error_budget}")
        if fast_burn_total <= 0 or slow_burn_total <= 0:
            raise ValueError("burn total denominators must be positive")

        # Observed error rates
        fast_error_rate = fast_burn_errors / fast_burn_total
        slow_error_rate = slow_burn_errors / slow_burn_total

        # Burn-rate multipliers using the caller-supplied error budget
        # as the denominator (avoids 1.0 - slo_target floating-point drift)
        fast_burn_rate = fast_error_rate / error_budget
        slow_burn_rate = slow_error_rate / error_budget

        # Epsilon tolerance for floating-point threshold comparisons
        epsilon = 1e-12

        # Fire when both windows exceed their respective thresholds
        if (
            fast_burn_rate + epsilon >= self._fast_burn_threshold
            and slow_burn_rate + epsilon >= self._slow_burn_threshold
        ):
            return BurnRateResult.FIRE

        return BurnRateResult.OK


# ---------------------------------------------------------------------------
# MaintenanceWindow
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class MaintenanceWindow:
    """Defines a time window during which alerts are suppressed.

    During maintenance, targeted alerts should be muted to avoid
    false-positive pages for expected downtime.

    Attributes:
        start_time: UTC start of the maintenance window.
        end_time: UTC end of the maintenance window.
        affected_services: List of service names affected
            (e.g. ``["api", "worker"]``).  Empty means all services.
        reason: Human-readable reason for the maintenance.

    Raises:
        ValueError: If *end_time* is not after *start_time*.
    """

    start_time: datetime
    end_time: datetime
    affected_services: List[str] = dataclasses.field(default_factory=list)
    reason: str = ""

    def __post_init__(self) -> None:
        """Validate window bounds after construction."""
        # Use aware comparisons by normalising to UTC
        start = self.start_time if self.start_time.tzinfo else self.start_time.replace(tzinfo=timezone.utc)
        end = self.end_time if self.end_time.tzinfo else self.end_time.replace(tzinfo=timezone.utc)
        if end <= start:
            raise ValueError(
                f"MaintenanceWindow end_time ({end}) must be after "
                f"start_time ({start})"
            )

    def in_maintenance(
        self,
        at: Optional[datetime] = None,
        service: str = "",
    ) -> bool:
        """Check if the given time falls within this maintenance window.

        Args:
            at: Timestamp to check (defaults to now).
            service: Optional service name.  When provided, only returns
                ``True`` if the service is in the affected list (or the
                affected list is empty, meaning all services).

        Returns:
            ``True`` if *at* is within the window and *service* is
            affected.
        """
        now = at if at is not None else datetime.now(timezone.utc)

        # Normalise timezone awareness for comparisons
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        start = self.start_time if self.start_time.tzinfo else self.start_time.replace(tzinfo=timezone.utc)
        end = self.end_time if self.end_time.tzinfo else self.end_time.replace(tzinfo=timezone.utc)

        if not (start <= now <= end):
            return False
        if service and self.affected_services and service not in self.affected_services:
            return False
        return True

    def remaining(self, at: Optional[datetime] = None) -> timedelta:
        """Return the time remaining until the window closes.

        Args:
            at: Reference timestamp (defaults to now).

        Returns:
            ``timedelta(0)`` if the window has already ended or not yet
            started.
        """
        now = at if at is not None else datetime.now(timezone.utc)
        end = self.end_time if self.end_time.tzinfo else self.end_time.replace(tzinfo=timezone.utc)
        remaining = end - now
        return remaining if remaining > timedelta(0) else timedelta(0)


# ---------------------------------------------------------------------------
# MaintenanceManager — collection of maintenance windows
# ---------------------------------------------------------------------------

class MaintenanceManager:
    """Manages multiple :class:`MaintenanceWindow` instances.

    Provides a unified ``in_maintenance()`` check across all registered
    windows.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._windows: List[MaintenanceWindow] = []

    def add(self, window: MaintenanceWindow) -> None:
        """Register a maintenance window.

        Args:
            window: The :class:`MaintenanceWindow` to add.
        """
        with self._lock:
            self._windows.append(window)

    def remove(self, window: MaintenanceWindow) -> None:
        """Remove a previously registered window.

        Args:
            window: The window to remove.  Uses identity comparison.
        """
        with self._lock:
            self._windows[:] = [w for w in self._windows if w is not window]

    def in_maintenance(
        self,
        at: Optional[datetime] = None,
        service: str = "",
    ) -> bool:
        """Check if any active maintenance window currently applies.

        Args:
            at: Timestamp to check (defaults to now).
            service: Optional service name.

        Returns:
            ``True`` if at least one active window covers the time and
            (optionally) service.
        """
        with self._lock:
            for window in self._windows:
                if window.in_maintenance(at=at, service=service):
                    return True
        return False

    @property
    def windows(self) -> List[MaintenanceWindow]:
        """Return a snapshot of all registered windows."""
        with self._lock:
            return list(self._windows)


# ---------------------------------------------------------------------------
# PrometheusRule — structured representation of a Prometheus alerting rule
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class PrometheusRule:
    """A single Prometheus alerting rule.

    Attributes:
        alert: Alert name.
        expr: PromQL expression.
        duration: How long the condition must hold before firing.
        labels: Additional labels to attach.
        annotations: Annotation key-value pairs (summary, description, …).
    """

    alert: str
    expr: str
    duration: str = "0m"
    labels: Dict[str, str] = dataclasses.field(default_factory=dict)
    annotations: Dict[str, str] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class PrometheusRuleGroup:
    """A named group of Prometheus alerting rules.

    Attributes:
        name: Group name.
        rules: Alerting rules belonging to this group.
    """

    name: str = "distllm_alerts"
    rules: List[PrometheusRule] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(frozen=True)
class AlertmanagerReceiver:
    """An Alertmanager receiver configuration.

    Attributes:
        name: Receiver name.
        pagerduty_routing_key: PagerDuty integration key (optional).
        slack_channel: Slack channel (optional).
        slack_token: Slack bot token (optional).
    """

    name: str
    pagerduty_routing_key: str = ""
    slack_channel: str = ""
    slack_token: str = ""


@dataclasses.dataclass(frozen=True)
class AlertmanagerRoute:
    """A route within the Alertmanager routing tree.

    Attributes:
        receiver: Target receiver name.
        matchers: List of ``"severity = critical"``-style matchers.
        continue_: Whether to continue matching subsequent routes.
    """

    receiver: str
    matchers: List[str] = dataclasses.field(default_factory=list)
    continue_: bool = False


# ---------------------------------------------------------------------------
# AlertingConfigurator
# ---------------------------------------------------------------------------

class AlertingConfigurator:
    """Combines runbook registry, routing, burn-rate alerts, and maintenance
    windows into cohesive Prometheus and Alertmanager configurations.

    Usage::

        configurator = AlertingConfigurator(
            runbook_registry=registry,
            alert_router=router,
            burn_rate_alert=burn_alert,
            maintenance_manager=mgr,
        )

        prom_rules = configurator.generate_prometheus_rules()
        am_config = configurator.generate_alertmanager_config()
    """

    def __init__(
        self,
        runbook_registry: Optional[RunbookRegistry] = None,
        alert_router: Optional[AlertRouter] = None,
        burn_rate_alert: Optional[SLOBurnRateAlert] = None,
        maintenance_manager: Optional[MaintenanceManager] = None,
    ) -> None:
        """Initialise the configurator.

        Each component defaults to a fresh instance when not provided.
        """
        self.runbook_registry = runbook_registry or RunbookRegistry()
        self.alert_router = alert_router or AlertRouter()
        self.burn_rate_alert = burn_rate_alert or SLOBurnRateAlert()
        self.maintenance_manager = maintenance_manager or MaintenanceManager()

    # ------------------------------------------------------------------
    # Prometheus rules YAML
    # ------------------------------------------------------------------

    def generate_prometheus_rules(
        self,
        extra_rules: Optional[List[PrometheusRule]] = None,
    ) -> str:
        """Generate a Prometheus ``rules.yaml`` with SLO burn-rate alerting.

        Produces a YAML document suitable for use with
        ``prometheus-operator`` ``PrometheusRule`` CRDs or direct
        Prometheus file-based rule loading.

        Args:
            extra_rules: Additional :class:`PrometheusRule` instances to
                include alongside the built-in burn-rate rules.

        Returns:
            YAML string.
        """
        rules = list(extra_rules) if extra_rules else []

        # Build burn-rate alert from the configured windows
        w = self.burn_rate_alert.windows
        fast_minutes = int(w.fast_burn_window.total_seconds() // 60)
        slow_minutes = int(w.slow_burn_window.total_seconds() // 60)

        # Fast burn-rate rule
        rules.append(
            PrometheusRule(
                alert="SLOFastBurnRate",
                expr=(
                    f"(1 - rate(errors_total[{fast_minutes}m]) "
                    f"/ rate(requests_total[{fast_minutes}m])) "
                    f"< 0.99"
                ),
                duration=f"{int(w.fast_burn_measurement.total_seconds() // 60)}m",
                labels={"severity": "critical", "team": "platform"},
                annotations={
                    "summary": "SLO fast burn-rate exceeded",
                    "description": (
                        f"Error rate over {fast_minutes}m window exceeds SLO."
                    ),
                },
            )
        )

        # Slow burn-rate rule
        rules.append(
            PrometheusRule(
                alert="SLOSlowBurnRate",
                expr=(
                    f"(1 - rate(errors_total[{slow_minutes}m]) "
                    f"/ rate(requests_total[{slow_minutes}m])) "
                    f"< 0.99"
                ),
                duration=f"{int(w.slow_burn_measurement.total_seconds() // 60)}m",
                labels={"severity": "warning", "team": "platform"},
                annotations={
                    "summary": "SLO slow burn-rate exceeded",
                    "description": (
                        f"Error rate over {slow_minutes}m window exceeds SLO."
                    ),
                },
            )
        )

        group = PrometheusRuleGroup(name="distllm_slo_alerts", rules=rules)
        return self._format_rule_group(group)

    @staticmethod
    def _format_rule_group(group: PrometheusRuleGroup) -> str:
        """Convert a :class:`PrometheusRuleGroup` to a YAML string."""
        raw: Dict[str, Any] = {
            "groups": [
                {
                    "name": group.name,
                    "rules": [
                        {
                            "alert": r.alert,
                            "expr": r.expr,
                            "for": r.duration,
                            "labels": dict(r.labels),
                            "annotations": dict(r.annotations),
                        }
                        for r in group.rules
                    ],
                }
            ]
        }
        return yaml_dump(raw, default_flow_style=False, sort_keys=False)

    # ------------------------------------------------------------------
    # Alertmanager config YAML
    # ------------------------------------------------------------------

    def generate_alertmanager_config(
        self,
        extra_receivers: Optional[List[AlertmanagerReceiver]] = None,
        extra_routes: Optional[List[AlertmanagerRoute]] = None,
    ) -> str:
        """Generate an Alertmanager ``config.yaml``.

        Routes are built from the current :class:`AlertRouter` rules and
        the :class:`RunbookRegistry`.  Maintenance windows are included
        as mute time intervals.

        Args:
            extra_receivers: Additional :class:`AlertmanagerReceiver`
                configurations.
            extra_routes: Additional :class:`AlertmanagerRoute` entries
                inserted before the default catch-all.

        Returns:
            YAML string.
        """
        # --- Receivers --------------------------------------------------
        receivers_raw: List[Dict[str, Any]] = [
            {
                "name": "pagerduty-critical",
                "pagerduty_configs": [{"routing_key": "${PD_ROUTING_KEY}"}],
            },
            {
                "name": "slack-warning",
                "slack_configs": [
                    {
                        "api_url": "${SLACK_WEBHOOK_URL}",
                        "channel": "#alerts-warning",
                    }
                ],
            },
            {
                "name": "log-info",
                "webhook_configs": [
                    {"url": "http://localhost:9093/log-info"}
                ],
            },
        ]

        if extra_receivers:
            for rec in extra_receivers:
                entry: Dict[str, Any] = {"name": rec.name}
                if rec.pagerduty_routing_key:
                    entry.setdefault("pagerduty_configs", []).append(
                        {"routing_key": rec.pagerduty_routing_key}
                    )
                if rec.slack_channel:
                    entry.setdefault("slack_configs", []).append(
                        {
                            "api_url": "${SLACK_WEBHOOK_URL}",
                            "channel": rec.slack_channel,
                        }
                    )
                receivers_raw.append(entry)

        # --- Routes -----------------------------------------------------
        routes_raw: List[Dict[str, Any]] = []

        # Insert extra routes first (highest priority)
        if extra_routes:
            for r in extra_routes:
                route_entry: Dict[str, Any] = {
                    "receiver": r.receiver,
                    "continue": r.continue_,
                }
                if r.matchers:
                    route_entry["matchers"] = list(r.matchers)
                routes_raw.append(route_entry)

        # Build routes from the AlertRouter rules
        for rule in self.alert_router.rules:
            matchers: List[str] = []
            if rule.severities:
                sevs = "|".join(s.value for s in rule.severities)
                matchers.append(f"severity =~ \"{sevs}\"")
            if rule.teams:
                teams = "|".join(rule.teams)
                matchers.append(f"team =~ \"{teams}\"")

            receiver_name = self._receiver_name_for_rule(rule)
            routes_raw.append(
                {
                    "receiver": receiver_name,
                    "matchers": matchers,
                    "continue": True,
                }
            )

        # Default catch-all route
        routes_raw.append(
            {
                "receiver": "log-info",
                "matchers": [],
                "continue": False,
            }
        )

        # --- Mute time intervals from maintenance windows ----------------
        mute_times: List[Dict[str, Any]] = []
        for window in self.maintenance_manager.windows:
            mute_entry: Dict[str, Any] = {
                "name": window.reason or "maintenance",
                "match": {},
            }
            if window.affected_services:
                mute_entry["match"] = {
                    "service": f"({'|'.join(window.affected_services)})"
                }
            mute_times.append(mute_entry)

        # --- Assemble final config --------------------------------------
        config: Dict[str, Any] = {
            "route": {
                "receiver": "log-info",
                "routes": routes_raw,
            },
            "receivers": receivers_raw,
        }

        if mute_times:
            config["mute_time_intervals"] = mute_times

        return yaml_dump(config, default_flow_style=False, sort_keys=False)

    @staticmethod
    def _receiver_name_for_rule(rule: AlertRule) -> str:
        """Derive a receiver name from an :class:`AlertRule`."""
        if ReceiverType.PAGERDUTY in rule.receivers:
            return "pagerduty-critical"
        if ReceiverType.SLACK in rule.receivers:
            return "slack-warning"
        return "log-info"


__all__ = [
    "AlertRule",
    "AlertRouter",
    "AlertSeverity",
    "AlertingConfigurator",
    "AlertmanagerReceiver",
    "AlertmanagerRoute",
    "BurnRateResult",
    "BurnRateWindows",
    "MaintenanceManager",
    "MaintenanceWindow",
    "PrometheusRule",
    "PrometheusRuleGroup",
    "ReceiverType",
    "RunbookRegistry",
    "SLOBurnRateAlert",
]
