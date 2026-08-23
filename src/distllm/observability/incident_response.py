"""Incident response, on-call, and postmortem management for distributed LLM.

Provides six components that work together to manage the full incident
lifecycle — from detection through acknowledgment, mitigation, resolution,
and post-incident review:

* **Incident** — Core incident model with status transitions and MTTA/MTTR.
* **IncidentManager** — CRUD, auto-creation from critical alerts, SLA breach
  detection, and aggregate MTTA/MTTR tracking.
* **Postmortem** — Post-incident review document with timeline, root-cause
  analysis, and action items. Generates Markdown output.
* **Runbook** — Predefined response procedures (node_failure, coordinator_
  failure, network_partition, oom, certificate_expiry) with step-by-step
  commands and verification.
* **OnCallSchedule** — Schedule-based on-call rotation with escalation,
  paging, and current-on-call lookup.
* **IncidentResponseConfigurator** — Top-level orchestrator that ties all
  subsystems together with a unified configuration.
"""

from __future__ import annotations

import dataclasses
import threading
import time
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class IncidentSeverity(str, Enum):
    """Severity levels for incidents."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class IncidentStatus(str, Enum):
    """Lifecycle status for an incident."""
    FIRING = "firing"
    ACKNOWLEDGED = "acknowledged"
    MITIGATED = "mitigated"
    RESOLVED = "resolved"


class RunbookScenario(str, Enum):
    """Predefined runbook scenarios."""
    NODE_FAILURE = "node_failure"
    COORDINATOR_FAILURE = "coordinator_failure"
    NETWORK_PARTITION = "network_partition"
    OOM = "oom"
    CERTIFICATE_EXPIRY = "certificate_expiry"


class NotificationChannel(str, Enum):
    """Supported notification channels for on-call paging."""
    PAGERDUTY = "pagerduty"
    SLACK = "slack"
    EMAIL = "email"
    SMS = "sms"
    LOG = "log"


# ---------------------------------------------------------------------------
# RunbookStep / Runbook
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class RunbookStep:
    """A single step within a runbook.

    Attributes:
        title: Short description of the step.
        command: Shell command or action to execute.
        expected_result: What a successful execution looks like.
        verification: How to verify the step had the intended effect.
    """
    title: str
    command: str
    expected_result: str
    verification: str


class Runbook:
    """Predefined response procedures for common failure scenarios.

    Supports five built-in scenarios (node_failure, coordinator_failure,
    network_partition, oom, certificate_expiry) plus custom additions.

    Usage::

        steps = Runbook().get(RunbookScenario.NODE_FAILURE)
    """

    _SCENARIOS: Dict[RunbookScenario, List[RunbookStep]] = {
        RunbookScenario.NODE_FAILURE: [
            RunbookStep(title="Verify node connectivity", command="ping -c 5 <node_ip>",
                        expected_result="0% packet loss", verification="ping returns successfully"),
            RunbookStep(title="Check SSH access", command="ssh <user>@<node_ip> 'uptime'",
                        expected_result="SSH connection established", verification="SSH returns uptime value"),
            RunbookStep(title="Inspect GPU health", command="nvidia-smi --query-gpu=index,name,temperature.gpu,memory.used,memory.total --format=csv,noheader",
                        expected_result="GPU temp < 85C, mem < 95%", verification="nvidia-smi completes without error"),
            RunbookStep(title="Check agent logs", command="journalctl -u distllm-agent -n 100 --no-pager",
                        expected_result="No fatal errors in recent logs", verification="Agent is running and heartbeating"),
            RunbookStep(title="Drain node", command="distllm-admin drain-node --node-id <node_id> --timeout 300",
                        expected_result="Node drain initiated", verification="Coordinator shows DRAINING state"),
            RunbookStep(title="Restart agent", command="ssh <user>@<node_ip> 'systemctl restart distllm-agent'",
                        expected_result="Agent restarts and re-registers", verification="Coordinator shows HEALTHY within 60s"),
            RunbookStep(title="Verify rejoin", command="distllm-admin list-nodes --status healthy",
                        expected_result="Node appears in healthy list", verification="Deep health probe returns success"),
        ],
        RunbookScenario.COORDINATOR_FAILURE: [
            RunbookStep(title="Determine leader", command="distllm-admin get-coordinator --cluster-status",
                        expected_result="Leader ID displayed", verification="Shows one leader and N followers"),
            RunbookStep(title="Check coordinator process", command="ssh <coordinator_host> 'systemctl status distllm-coordinator'",
                        expected_result="Process active or inactive", verification="systemctl returns clear status"),
            RunbookStep(title="Inspect coordinator logs", command="journalctl -u distllm-coordinator -n 200 --no-pager",
                        expected_result="Recent logs show panic/OOM/last heartbeat", verification="Root cause identifiable"),
            RunbookStep(title="Trigger leader election", command="distllm-admin trigger-election",
                        expected_result="New leader elected within 30s", verification="get-coordinator shows new leader"),
            RunbookStep(title="Restart coordinator", command="ssh <coordinator_host> 'systemctl restart distllm-coordinator'",
                        expected_result="Coordinator restarts, nodes re-register", verification="All nodes CONNECTED within 120s"),
            RunbookStep(title="Verify cluster rebalance", command="distllm-admin cluster-status",
                        expected_result="All nodes healthy, load balanced", verification="No DEGRADED/UNKNOWN nodes"),
        ],
        RunbookScenario.NETWORK_PARTITION: [
            RunbookStep(title="Identify partitioned nodes", command="distllm-admin list-nodes --status unknown --timeout 10",
                        expected_result="List of unreachable nodes", verification="Node list matches expected membership"),
            RunbookStep(title="Check inter-node latency", command="distllm-admin ping-nodes --source <node> --targets <node_list>",
                        expected_result="Latency < 50ms same-region, < 200ms cross-region", verification="High latency indicates partition"),
            RunbookStep(title="Inspect network interfaces", command='ssh <node_ip> \'ip link show | grep -E "(state|mtu)"\'',
                        expected_result="Interfaces UP with expected MTU", verification="No interfaces in DOWN/UNKNOWN state"),
            RunbookStep(title="Check firewall rules", command="ssh <node_ip> 'iptables -L -n 2>/dev/null || nft list ruleset'",
                        expected_result="Required ports (50051, 50052) ACCEPT", verification="No DROP/REJECT rules blocking traffic"),
            RunbookStep(title="Restart gossip protocol", command="distllm-admin restart-gossip --nodes <node_list>",
                        expected_result="Gossip converges within 2 cycles", verification="All nodes CONNECTED"),
            RunbookStep(title="Verify full convergence", command="distllm-admin cluster-status --wait 60",
                        expected_result="Cluster HEALTHY with all nodes", verification="No partition-related alerts lingering"),
        ],
        RunbookScenario.OOM: [
            RunbookStep(title="Identify OOM victim", command='dmesg | grep -i "oom\\|out of memory" | tail -20',
                        expected_result="Kernel OOM messages with PID/process name", verification="OOM victim identified"),
            RunbookStep(title="Check GPU memory", command="nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader",
                        expected_result="Memory utilization per GPU visible", verification="At least one GPU > 90% used"),
            RunbookStep(title="Check process memory", command="ps aux --sort=-%mem | head -10",
                        expected_result="Top memory consumers listed", verification="Python/distllm processes show high RSS"),
            RunbookStep(title="Reduce batch size", command="distllm-admin set-config --node <node_id> --key max_batch_size --value <reduced>",
                        expected_result="Config updated on node", verification="Node acknowledges new config"),
            RunbookStep(title="Restart service", command="ssh <node_ip> 'systemctl restart distllm-agent'",
                        expected_result="Service restarts and reconnects", verification="GPU memory freed, service HEALTHY"),
            RunbookStep(title="Verify memory baseline", command="nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader",
                        expected_result="Memory usage below 80%", verification="No immediate OOM recurrence"),
        ],
        RunbookScenario.CERTIFICATE_EXPIRY: [
            RunbookStep(title="Check expiry dates", command="openssl x509 -in /etc/distllm/tls.crt -noout -enddate && openssl x509 -in /etc/distllm/tls.crt -noout -checkend 86400",
                        expected_result="End date printed; checkend 0=valid 1=expiring", verification="Certificate status visible"),
            RunbookStep(title="List cluster certificates", command="distllm-admin list-certificates --expiry-window 7d",
                        expected_result="All certs expiring within 7 days", verification="Complete inventory of soon-to-expire certs"),
            RunbookStep(title="Renew certificate", command="certbot renew --cert-name distllm --deploy-hook 'systemctl reload distllm-coordinator'",
                        expected_result="Certificate renewed successfully", verification="openssl checkend returns 0"),
            RunbookStep(title="Distribute certificate", command="distllm-admin distribute-cert --source <coordinator_host>",
                        expected_result="New cert distributed to all nodes", verification="All nodes report update success"),
            RunbookStep(title="Verify TLS handshake", command="openssl s_client -connect <node_ip>:50051 -CAfile /etc/distllm/ca.crt 2>&1 | grep -i 'verify return'",
                        expected_result="Verify return code: 0 (ok)", verification="All inter-node TLS connections succeed"),
        ],
    }

    def __init__(self, extra_scenarios: Optional[Dict[str, List[RunbookStep]]] = None) -> None:
        self._scenarios: Dict[str, List[RunbookStep]] = {k.value: list(v) for k, v in self._SCENARIOS.items()}
        if extra_scenarios:
            self._scenarios.update(extra_scenarios)

    def get(self, scenario: str) -> List[RunbookStep]:
        """Return the runbook steps for *scenario*, or empty list."""
        return list(self._scenarios.get(scenario, []))

    @property
    def scenarios(self) -> List[str]:
        """Return all registered scenario names."""
        return list(self._scenarios.keys())

    def add_scenario(self, name: str, steps: List[RunbookStep]) -> None:
        """Register a custom scenario."""
        self._scenarios[name] = list(steps)

    def to_markdown(self, scenario: str) -> str:
        """Render a scenario as a Markdown runbook document."""
        steps = self._scenarios.get(scenario, [])
        lines: List[str] = [f"# Runbook: {scenario.replace('_', ' ').title()}", "", "## Steps", ""]
        for i, step in enumerate(steps, start=1):
            lines.extend([f"### {i}. {step.title}", "", "**Command:**", "", "```bash", step.command, "```",
                          "", f"**Expected Result:** {step.expected_result}", "",
                          f"**Verification:** {step.verification}", ""])
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# OnCallEntry / OnCallSchedule
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class OnCallEntry:
    """A single on-call schedule entry.

    Attributes:
        name: Engineer name.
        start: Shift start time (UTC-aware).
        end: Shift end time (UTC-aware).
    """
    name: str
    start: datetime
    end: datetime


class OnCallSchedule:
    """Manages an on-call rotation schedule with escalation and paging.

    Usage::

        sched = OnCallSchedule()
        sched.add(OnCallEntry(name="alice", start=..., end=...))
        sched.get_current_oncall()  # "alice"
        sched.notify(incident)
        sched.escalate(incident)
    """

    def __init__(self, notification_callback: Optional[callable] = None,
                 escalation_callback: Optional[callable] = None) -> None:
        self._lock = threading.Lock()
        self._entries: List[OnCallEntry] = []
        self._notification_callback = notification_callback or self._default_notify
        self._escalation_callback = escalation_callback or self._default_escalate

    @staticmethod
    def _default_notify(name: str, message: str) -> None:
        print(f"[ONCALL] Notifying {name}: {message}")  # noqa: T201

    @staticmethod
    def _default_escalate(name: str, incident_id: str, level: int) -> None:
        print(f"[ONCALL] Escalating incident {incident_id} to {name} (level {level})")  # noqa: T201

    def add(self, entry: OnCallEntry) -> None:
        """Register an on-call schedule entry.

        Raises:
            ValueError: If end is not after start.
        """
        if entry.end <= entry.start:
            raise ValueError(f"OnCallEntry end ({entry.end}) must be after start ({entry.start})")
        with self._lock:
            self._entries.append(entry)
            self._entries.sort(key=lambda e: e.start)

    def remove(self, entry: OnCallEntry) -> None:
        """Remove a previously registered entry (identity comparison)."""
        with self._lock:
            self._entries[:] = [e for e in self._entries if e is not entry]

    def get_current_oncall(self, at: Optional[datetime] = None) -> Optional[str]:
        """Return the name of the currently on-call engineer, or ``None``."""
        now = _ensure_tz(at) if at is not None else datetime.now(timezone.utc)
        with self._lock:
            for entry in self._entries:
                if entry.start <= now <= entry.end:
                    return entry.name
        return None

    def get_oncall_at(self, dt: datetime) -> Optional[str]:
        """Return the engineer on call at a specific time."""
        return self.get_current_oncall(at=dt)

    def notify(self, incident: "Incident", message: str = "") -> None:
        """Page the currently on-call engineer about *incident*."""
        name = self.get_current_oncall() or "unattached"
        text = message or f"Incident {incident.id}: {incident.title} [{incident.severity.value}]"
        self._notification_callback(name, text)

    def escalate(self, incident: "Incident", minutes_without_ack: int = 15,
                 at: Optional[datetime] = None) -> bool:
        """Escalate if *incident* has not been acknowledged within *minutes_without_ack*.

        Returns ``True`` if escalation was triggered.
        """
        now = _ensure_tz(at) if at is not None else datetime.now(timezone.utc)
        created = _ensure_tz(incident.created_at)
        if incident.status != IncidentStatus.FIRING:
            return False
        if (now - created).total_seconds() < minutes_without_ack * 60:
            return False
        name = self.get_current_oncall(at=now) or "unattached"
        self._escalation_callback(name, incident.id, level=1)
        return True

    @property
    def entries(self) -> List[OnCallEntry]:
        """Return a snapshot of all schedule entries."""
        with self._lock:
            return list(self._entries)

    def is_covered(self, at: Optional[datetime] = None) -> bool:
        """Check whether any engineer is on call at *at*."""
        return self.get_current_oncall(at=at) is not None


# ---------------------------------------------------------------------------
# Incident
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Incident:
    """Represents a single operational incident.

    Tracks lifecycle timestamps and provides computed MTTA and MTTR
    properties as well as transition methods.

    Attributes:
        id: Unique incident identifier.
        title: Human-readable title.
        severity: Severity level.
        status: Current lifecycle status.
        created_at: UTC timestamp of creation.
        acknowledged_at: UTC timestamp of acknowledgment (``None`` until ack).
        mitigated_at: UTC timestamp of mitigation (``None`` until mitigated).
        resolved_at: UTC timestamp of resolution (``None`` until resolved).
        source_alert: Name of the triggering alert (optional).
        metadata: Arbitrary key-value store.
    """
    id: str
    title: str
    severity: IncidentSeverity
    status: IncidentStatus = IncidentStatus.FIRING
    created_at: datetime = dataclasses.field(default_factory=lambda: datetime.now(timezone.utc))
    acknowledged_at: Optional[datetime] = None
    mitigated_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    source_alert: Optional[str] = None
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)

    # ------------------------------------------------------------------
    # Status transitions
    # ------------------------------------------------------------------

    def acknowledge(self, at: Optional[datetime] = None) -> None:
        """Mark as acknowledged.

        Raises:
            ValueError: If already acknowledged, mitigated, or resolved.
        """
        if self.status != IncidentStatus.FIRING:
            raise ValueError(f"Cannot acknowledge incident in status {self.status.value!r}")
        self.status = IncidentStatus.ACKNOWLEDGED
        self.acknowledged_at = _ensure_tz(at) if at is not None else datetime.now(timezone.utc)

    def mitigate(self, at: Optional[datetime] = None) -> None:
        """Mark as mitigated (auto-acknowledges if still firing).

        Raises:
            ValueError: If already mitigated or resolved.
        """
        if self.status in (IncidentStatus.MITIGATED, IncidentStatus.RESOLVED):
            raise ValueError(f"Cannot mitigate incident in status {self.status.value!r}")
        if self.status == IncidentStatus.FIRING:
            self.acknowledged_at = self.acknowledged_at or (datetime.now(timezone.utc) if at is None else _ensure_tz(at))
            self.status = IncidentStatus.ACKNOWLEDGED
        self.status = IncidentStatus.MITIGATED
        self.mitigated_at = _ensure_tz(at) if at is not None else datetime.now(timezone.utc)

    def resolve(self, at: Optional[datetime] = None) -> None:
        """Mark as resolved (auto-acknowledges and auto-mitigates if earlier steps skipped).

        Raises:
            ValueError: If already resolved.
        """
        if self.status == IncidentStatus.RESOLVED:
            raise ValueError(f"Cannot resolve incident in status {self.status.value!r}")
        if self.status == IncidentStatus.FIRING:
            now = datetime.now(timezone.utc) if at is None else _ensure_tz(at)
            self.acknowledged_at = self.acknowledged_at or now
            self.status = IncidentStatus.ACKNOWLEDGED
            self.mitigated_at = self.mitigated_at or self.acknowledged_at
        if self.status == IncidentStatus.ACKNOWLEDGED:
            self.mitigated_at = self.mitigated_at or (datetime.now(timezone.utc) if at is None else _ensure_tz(at))
            self.status = IncidentStatus.MITIGATED
        self.status = IncidentStatus.RESOLVED
        self.resolved_at = _ensure_tz(at) if at is not None else datetime.now(timezone.utc)

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------

    @property
    def mtta(self) -> Optional[timedelta]:
        """Mean Time To Acknowledge (created_at -> acknowledged_at)."""
        if self.acknowledged_at is None:
            return None
        return _ensure_tz(self.acknowledged_at) - _ensure_tz(self.created_at)

    @property
    def mttr(self) -> Optional[timedelta]:
        """Mean Time To Resolve (created_at -> resolved_at)."""
        if self.resolved_at is None:
            return None
        return _ensure_tz(self.resolved_at) - _ensure_tz(self.created_at)

    @property
    def time_to_mitigate(self) -> Optional[timedelta]:
        """Time from creation to mitigation."""
        if self.mitigated_at is None:
            return None
        return _ensure_tz(self.mitigated_at) - _ensure_tz(self.created_at)

    @property
    def duration(self) -> timedelta:
        """Total duration so far (creation to now or resolution)."""
        end = _ensure_tz(self.resolved_at) if self.resolved_at is not None else datetime.now(timezone.utc)
        return end - _ensure_tz(self.created_at)


# ---------------------------------------------------------------------------
# IncidentManager
# ---------------------------------------------------------------------------


class IncidentManager:
    """Manages incident lifecycle — create, read, list, SLA enforcement.

    Supports auto-creation from critical alerts and tracks aggregate
    MTTA/MTTR across all resolved incidents.

    Usage::

        manager = IncidentManager(oncall_schedule=schedule)
        inc = manager.create(title="High latency", severity=IncidentSeverity.CRITICAL)
        inc.acknowledge()
        manager.check_sla(inc)  # timedelta if breached, None otherwise
    """

    def __init__(self, oncall_schedule: Optional[OnCallSchedule] = None,
                 sla_mtta_critical_minutes: float = 15.0,
                 sla_mtta_high_minutes: float = 30.0) -> None:
        self._lock = threading.Lock()
        self._incidents: Dict[str, Incident] = {}
        self._oncall_schedule = oncall_schedule
        self._sla_mtta_critical_minutes = sla_mtta_critical_minutes
        self._sla_mtta_high_minutes = sla_mtta_high_minutes

    def create(self, title: str, severity: IncidentSeverity,
               source_alert: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None,
               incident_id: Optional[str] = None) -> Incident:
        """Create a new incident.

        Auto-notifies the on-call engineer for critical severity.
        """
        inc = Incident(id=incident_id or self._next_id(), title=title, severity=severity,
                        source_alert=source_alert, metadata=metadata or {})
        with self._lock:
            self._incidents[inc.id] = inc
        if severity == IncidentSeverity.CRITICAL and self._oncall_schedule is not None:
            self._oncall_schedule.notify(inc)
        return inc

    def get(self, incident_id: str) -> Optional[Incident]:
        """Retrieve an incident by ID, or ``None``."""
        with self._lock:
            return self._incidents.get(incident_id)

    def list(self, severity: Optional[IncidentSeverity] = None,
             status: Optional[IncidentStatus] = None,
             source_alert: Optional[str] = None,
             since: Optional[datetime] = None,
             until: Optional[datetime] = None) -> List[Incident]:
        """List incidents with optional filters, newest first."""
        with self._lock:
            results = list(self._incidents.values())
        if severity is not None:
            results = [i for i in results if i.severity == severity]
        if status is not None:
            results = [i for i in results if i.status == status]
        if source_alert is not None:
            results = [i for i in results if i.source_alert == source_alert]
        if since is not None:
            s = _ensure_tz(since)
            results = [i for i in results if _ensure_tz(i.created_at) >= s]
        if until is not None:
            u = _ensure_tz(until)
            results = [i for i in results if _ensure_tz(i.created_at) < u]
        results.sort(key=lambda i: i.created_at, reverse=True)
        return results

    def update(self, incident_id: str, title: Optional[str] = None,
               severity: Optional[IncidentSeverity] = None,
               metadata: Optional[Dict[str, Any]] = None) -> Optional[Incident]:
        """Update mutable fields on an existing incident, or ``None``."""
        with self._lock:
            inc = self._incidents.get(incident_id)
            if inc is None:
                return None
            if title is not None:
                inc.title = title
            if severity is not None:
                inc.severity = severity
            if metadata is not None:
                inc.metadata.update(metadata)
            return inc

    def check_sla(self, incident: Incident) -> Optional[timedelta]:
        """Check if *incident* breached its MTTA SLA.

        Critical SLA: *sla_mtta_critical_minutes* (default 15 min).
        High SLA: *sla_mtta_high_minutes* (default 30 min).

        Returns the breach duration as a positive ``timedelta``, or
        ``None`` if within SLA or not yet acknowledged.
        """
        mtta = incident.mtta
        if mtta is None:
            return None
        if incident.severity == IncidentSeverity.CRITICAL:
            threshold = timedelta(minutes=self._sla_mtta_critical_minutes)
        elif incident.severity == IncidentSeverity.HIGH:
            threshold = timedelta(minutes=self._sla_mtta_high_minutes)
        else:
            return None
        return None if mtta <= threshold else mtta - threshold

    @property
    def average_mtta(self) -> Optional[timedelta]:
        """Average MTTA across all acknowledged/resolved incidents."""
        with self._lock:
            values = [i.mtta for i in self._incidents.values() if i.mtta is not None]
        if not values:
            return None
        return timedelta(seconds=sum(v.total_seconds() for v in values) / len(values))

    @property
    def average_mttr(self) -> Optional[timedelta]:
        """Average MTTR across all resolved incidents."""
        with self._lock:
            values = [i.mttr for i in self._incidents.values() if i.mttr is not None]
        if not values:
            return None
        return timedelta(seconds=sum(v.total_seconds() for v in values) / len(values))

    @property
    def incident_count(self) -> int:
        """Total number of tracked incidents."""
        with self._lock:
            return len(self._incidents)

    def _next_id(self) -> str:
        ts = int(time.time() * 1000)
        with self._lock:
            return f"INC-{ts}-{len(self._incidents)}"


# ---------------------------------------------------------------------------
# Postmortem
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ActionItem:
    """A single action item from a postmortem.

    Attributes:
        description: What needs to be done.
        owner: Person or team responsible.
        tracked_by: Issue tracker reference (e.g. ``JIRA-123``).
        completed: Whether the action item is done.
    """
    description: str
    owner: str = ""
    tracked_by: str = ""
    completed: bool = False


@dataclasses.dataclass
class TimelineEntry:
    """A single entry in the incident timeline.

    Attributes:
        time: UTC timestamp of the event.
        description: What happened at this time.
    """
    time: datetime
    description: str


class Postmortem:
    """A post-incident review document that renders to Markdown.

    Usage::

        pm = Postmortem.generate_template(incident)
        pm.summary = "What happened."
        pm.root_cause = "Root cause analysis."
        pm.action_items.append(ActionItem(description="Add alerts"))
        print(pm.to_markdown())
    """

    @classmethod
    def generate_template(cls, incident: Incident) -> Postmortem:
        """Create a pre-filled postmortem from an incident's lifecycle."""
        timeline: List[TimelineEntry] = [
            TimelineEntry(time=incident.created_at,
                          description=f"Incident created: {incident.title} (severity={incident.severity.value})"),
        ]
        if incident.acknowledged_at is not None:
            timeline.append(TimelineEntry(time=incident.acknowledged_at, description="Incident acknowledged"))
        if incident.mitigated_at is not None:
            timeline.append(TimelineEntry(time=incident.mitigated_at, description="Incident mitigated"))
        if incident.resolved_at is not None:
            timeline.append(TimelineEntry(time=incident.resolved_at, description="Incident resolved"))
        return cls(title=f"Postmortem: {incident.title}", incident_id=incident.id, timeline=timeline)

    def __init__(self, title: str, incident_id: str, summary: str = "",
                 timeline: Optional[List[TimelineEntry]] = None, root_cause: str = "",
                 action_items: Optional[List[ActionItem]] = None, lessons_learned: str = "") -> None:
        self.title = title
        self.incident_id = incident_id
        self.summary = summary
        self.timeline = timeline or []
        self.root_cause = root_cause
        self.action_items = action_items or []
        self.lessons_learned = lessons_learned

    def to_markdown(self) -> str:
        """Render the postmortem as a structured Markdown document.

        Sections: Summary, Timeline, Root Cause, Action Items, Lessons Learned.
        """
        lines: List[str] = [
            f"# {self.title}", "",
            f"**Incident ID:** {self.incident_id}",
            f"**Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}", "",
            "## Summary", "",
            self.summary or "*No summary provided.*", "",
            "## Timeline", "",
        ]
        if self.timeline:
            lines.append("| Time (UTC) | Event |")
            lines.append("|------------|-------|")
            for entry in self.timeline:
                lines.append(f"| {entry.time.strftime('%Y-%m-%d %H:%M:%S')} | {entry.description} |")
        else:
            lines.append("*No timeline entries.*")
        lines.extend(["", "## Root Cause", "", self.root_cause or "*No root cause identified.*", "",
                       "## Action Items", ""])
        if self.action_items:
            lines.append("| # | Description | Owner | Tracked By | Done |")
            lines.append("|---|-------------|-------|------------|------|")
            for i, item in enumerate(self.action_items, start=1):
                lines.append(f"| {i} | {item.description} | {item.owner} | {item.tracked_by} | {'Yes' if item.completed else 'No'} |")
        else:
            lines.append("*No action items recorded.*")
        lines.extend(["", "## Lessons Learned", "", self.lessons_learned or "*No lessons learned captured.*", ""])
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# IncidentResponseConfig / IncidentResponseConfigurator
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class IncidentResponseConfig:
    """Configuration for the incident response system.

    Attributes:
        escalation_minutes: Minutes before unacknowledged critical incident
            triggers escalation (default 15).
        auto_create_rules: Severity-to-boolean mapping for auto-creation.
        notification_channels: Notification channels for alerts.
        sla_mtta_critical_minutes: MTTA SLA for critical incidents (default 15).
        sla_mtta_high_minutes: MTTA SLA for high incidents (default 30).
    """
    escalation_minutes: int = 15
    auto_create_rules: Dict[str, bool] = dataclasses.field(
        default_factory=lambda: {"critical": True, "high": True, "medium": False, "low": False})
    notification_channels: List[NotificationChannel] = dataclasses.field(
        default_factory=lambda: [NotificationChannel.PAGERDUTY, NotificationChannel.SLACK, NotificationChannel.LOG])
    sla_mtta_critical_minutes: float = 15.0
    sla_mtta_high_minutes: float = 30.0


class IncidentResponseConfigurator:
    """Top-level orchestrator for incident response.

    Combines :class:`IncidentManager`, :class:`Runbook`,
    :class:`OnCallSchedule`, and postmortem generation.

    Usage::

        c = IncidentResponseConfigurator(escalation_minutes=10)
        c.manager      # IncidentManager
        c.runbook      # Runbook
        c.oncall       # OnCallSchedule
        c.config       # IncidentResponseConfig
    """

    def __init__(self, escalation_minutes: int = 15,
                 auto_create_rules: Optional[Dict[str, bool]] = None,
                 notification_channels: Optional[List[NotificationChannel]] = None,
                 sla_mtta_critical_minutes: float = 15.0,
                 sla_mtta_high_minutes: float = 30.0,
                 runbook: Optional[Runbook] = None,
                 oncall_schedule: Optional[OnCallSchedule] = None) -> None:
        self.config = IncidentResponseConfig(
            escalation_minutes=escalation_minutes,
            auto_create_rules=auto_create_rules or {"critical": True, "high": True, "medium": False, "low": False},
            notification_channels=notification_channels or [NotificationChannel.PAGERDUTY, NotificationChannel.SLACK, NotificationChannel.LOG],
            sla_mtta_critical_minutes=sla_mtta_critical_minutes,
            sla_mtta_high_minutes=sla_mtta_high_minutes,
        )
        self.runbook = runbook or Runbook()
        self.oncall = oncall_schedule or OnCallSchedule()
        self.manager = IncidentManager(oncall_schedule=self.oncall,
                                       sla_mtta_critical_minutes=sla_mtta_critical_minutes,
                                       sla_mtta_high_minutes=sla_mtta_high_minutes)

    def should_auto_create(self, severity: str) -> bool:
        """Check whether incidents should be auto-created for *severity*."""
        return self.config.auto_create_rules.get(severity, False)

    def create_postmortem(self, incident: Incident, summary: str = "",
                          timeline: Optional[List[TimelineEntry]] = None,
                          root_cause: str = "",
                          action_items: Optional[List[str]] = None,
                          lessons_learned: str = "") -> Postmortem:
        """Convenience method to generate a postmortem from an incident."""
        pm = Postmortem.generate_template(incident)
        pm.summary = summary
        if timeline:
            pm.timeline.extend(timeline)
        pm.root_cause = root_cause
        pm.action_items = [ActionItem(description=desc) for desc in (action_items or [])]
        pm.lessons_learned = lessons_learned
        return pm

    def escalate_if_needed(self, incident: Incident, at: Optional[datetime] = None) -> bool:
        """Escalate an unacknowledged incident if escalation grace period has passed."""
        return self.oncall.escalate(incident, minutes_without_ack=self.config.escalation_minutes, at=at)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _ensure_tz(dt: datetime) -> datetime:
    """Ensure a datetime is timezone-aware, assuming UTC if naive."""
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


__all__ = [
    "ActionItem",
    "Incident",
    "IncidentManager",
    "IncidentResponseConfig",
    "IncidentResponseConfigurator",
    "IncidentSeverity",
    "IncidentStatus",
    "NotificationChannel",
    "OnCallEntry",
    "OnCallSchedule",
    "Postmortem",
    "Runbook",
    "RunbookScenario",
    "RunbookStep",
    "TimelineEntry",
]
