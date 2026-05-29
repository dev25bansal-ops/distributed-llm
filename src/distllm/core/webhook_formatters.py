"""Platform-specific webhook payload formatters for Slack, Discord, and PagerDuty.

Each formatter is a callable with signature::

    (event: str, payload: dict) -> (body: dict, headers: dict)

The returned ``body`` is JSON-serialized and POSTed to the webhook URL.
The returned ``headers`` are merged into the HTTP request headers.

Usage::

    from distllm.core.webhook_manager import WebhookManager
    from distllm.core.webhook_formatters import slack_formatter

    mgr = WebhookManager()
    mgr.register("https://hooks.slack.com/...", formatter=slack_formatter)
    mgr.dispatch(WebhookEvent.NODE_FAILED, {"node_id": "node-0", ...})
"""

from __future__ import annotations

import time
from typing import Any

from .webhook_manager import WebhookEvent, WebhookFormatter


# ── Colour palette ──────────────────────────────────────────────────────────

_COLORS = {
    "critical": 15158332,   # red
    "error": 15158332,
    "warning": 16497928,    # yellow/amber
    "info": 3447003,        # blue/green
    "success": 3066993,     # green
    "healthy": 3066993,
}

_SLACK_COLORS = {
    "critical": "#ff0000",
    "error": "#ff0000",
    "warning": "#ffaa00",
    "info": "#0066cc",
    "success": "#00aa00",
    "healthy": "#00aa00",
}

# ── Helpers ─────────────────────────────────────────────────────────────────


def _severity(event: str) -> str:
    """Map an event name to a severity level."""
    low = {"node.joined", "node.recovered", "model.loaded", "backup.created",
           "circuit_breaker.closed", "recovery.completed", "leader.elected"}
    medium = {"node.left", "node.draining", "threshold.breached", "high_latency",
              "straggler.detected", "recovery.started", "certificate.expiring",
              "cluster.health_changed", "leader.lost", "coordinator.failover"}
    if event in low:
        return "info"
    if event in medium:
        return "warning"
    return "error"


def _summary(event: str, payload: dict[str, Any]) -> str:
    """Generate a human-readable summary from an event + payload."""
    node = payload.get("node_id") or payload.get("node", "")
    model = payload.get("model_name") or payload.get("model", "")

    summaries = {
        "node.joined": f"Node {node} joined the cluster",
        "node.left": f"Node {node} left the cluster",
        "node.failed": f"Node {node} failed — check logs immediately",
        "node.draining": f"Node {node} is now draining (maintenance)",
        "node.recovered": f"Node {node} recovered and returned to service",
        "circuit_breaker.opened": f"Circuit breaker OPENED for node {node} — requests paused",
        "circuit_breaker.closed": f"Circuit breaker CLOSED for node {node} — requests resumed",
        "high_latency": f"High latency detected on node {node}: {payload.get('latency_ms', '?')}ms",
        "cluster.health_changed": f"Cluster health changed: {payload.get('status', payload.get('state', 'unknown'))}",
        "straggler.detected": f"Straggler detected: node {node} — pipeline slowed",
        "recovery.started": f"Recovery started for node {node}",
        "recovery.completed": f"Recovery completed for node {node}",
        "threshold.breached": f"Threshold breached: {payload.get('metric', '?')} = {payload.get('value', '?')}",
        "model.loaded": f"Model {model or '?'} loaded",
        "model.unloaded": f"Model {model or '?'} unloaded",
        "model.error": f"Model error on {model or '?'}: {payload.get('error', '?')}",
        "coordinator.failover": f"Coordinator failover to {payload.get('new_leader', '?')}",
        "certificate.expiring": f"Certificate expiring for {payload.get('subject', '?')} ({payload.get('days_left', '?')} days)",
        "leader.elected": f"New leader elected: {payload.get('leader_id', '?')}",
        "leader.lost": f"Leader lost: {payload.get('leader_id', '?')}",
    }
    return summaries.get(event, f"Event: {event} on {node or 'cluster'}")


def _timestamp() -> str:
    """ISO-8601 timestamp (UTC) for Discord / PagerDuty."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ── Slack Block Kit formatter ───────────────────────────────────────────────


def slack_formatter(event: str, payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    """Format an event as a Slack Block Kit message.

    Returns a dict with ``blocks`` and optional ``text`` fields,
    plus an empty header dict (Slack webhooks don't need extra headers).
    """
    severity = _severity(event)
    summary = _summary(event, payload)
    color = _SLACK_COLORS.get(severity, "#cccccc")
    node = payload.get("node_id") or payload.get("node", "")
    details = {k: v for k, v in payload.items()
               if k not in ("event", "timestamp", "node_id", "node", "model_name", "model") and v is not None}

    blocks: list[dict[str, Any]] = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": summary, "emoji": True},
        },
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*Severity:* {severity}\n*Event:* `{event}`"},
        },
    ]

    if node:
        blocks.append({
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Node:* `{node}`"},
                {"type": "mrkdwn", "text": f"*Time:* {_timestamp()}"},
            ],
        })

    if details:
        fields = [{"type": "mrkdwn", "text": f"*{k}:* `{v}`"} for k, v in list(details.items())[:6]]
        if fields:
            blocks.append({"type": "section", "fields": fields})

    if severity in ("error", "critical"):
        blocks.append({
            "type": "context",
            "elements": [{"type": "mrkdwn", "text": ":warning: Requires immediate attention"}],
        })

    return {"text": summary, "blocks": blocks, "attachments": [{"color": color}]}, {}


# ── Discord embed formatter ─────────────────────────────────────────────────


def discord_formatter(event: str, payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    """Format an event as a Discord webhook message with embeds.

    Returns a dict with ``embeds`` and ``content`` fields,
    plus an empty header dict (Discord webhooks don't need extra headers).
    """
    severity = _severity(event)
    summary = _summary(event, payload)
    color = _COLORS.get(severity, 9807270)
    node = payload.get("node_id") or payload.get("node", "")
    details = {k: v for k, v in payload.items()
               if k not in ("event", "timestamp", "node_id", "node", "model_name", "model") and v is not None}

    embed: dict[str, Any] = {
        "title": summary,
        "color": color,
        "timestamp": _timestamp(),
        "footer": {"text": f"distllm | {severity}"},
        "fields": [
            {"name": "Event", "value": f"`{event}`", "inline": True},
            {"name": "Severity", "value": severity, "inline": True},
        ],
    }

    if node:
        embed["fields"].append({"name": "Node", "value": f"`{node}`", "inline": True})

    for k, v in details.items():
        if len(embed["fields"]) < 20:
            embed["fields"].append({"name": k, "value": str(v)[:200], "inline": True})

    body: dict[str, Any] = {
        "content": f"**{severity.upper()}**: {summary}" if severity in ("error", "critical") else summary,
        "embeds": [embed],
    }
    return body, {}


# ── PagerDuty Events API v2 formatter ──────────────────────────────────────


def pagerduty_formatter(
    event: str, payload: dict[str, Any],
    routing_key: str = "",
) -> tuple[dict[str, Any], dict[str, str]]:
    """Format an event as a PagerDuty Events API v2 payload.

    Requires ``routing_key`` (PagerDuty integration key). Pass it via
    ``register(…, secret=routing_key)`` — the manager sends it as
    ``X-Webhook-Signature`` by default, so for PagerDuty we read it
    from the payload and re-inject it into the body.

    Alternately, use ``functools.partial(pagerduty_formatter, routing_key=…).``
    """
    severity = _severity(event)
    summary = _summary(event, payload)
    node = payload.get("node_id") or payload.get("node", "")
    source = node or "cluster"

    details = {k: str(v) for k, v in payload.items() if v is not None}

    body = {
        "routing_key": routing_key,
        "event_action": "trigger",
        "payload": {
            "summary": summary[:120],
            "severity": severity if severity in ("critical", "error", "warning", "info") else "info",
            "source": source,
            "component": "distllm",
            "group": "node" if node else "cluster",
            "class": "infrastructure",
            "custom_details": details,
        },
        "dedup_key": f"distllm:{event}:{source}",
    }
    return body, {}


# ── Formatter registry ─────────────────────────────────────────────────────


_FORMATTERS: dict[str, WebhookFormatter] = {
    "slack": slack_formatter,
    "discord": discord_formatter,
    "pagerduty": pagerduty_formatter,
}


def get_formatter(name: str) -> WebhookFormatter | None:
    """Look up a built-in formatter by name (``slack``, ``discord``, ``pagerduty``)."""
    return _FORMATTERS.get(name)
