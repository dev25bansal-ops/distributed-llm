"""Tests for webhook platform formatters (Slack, Discord, PagerDuty)."""

import json
from unittest.mock import MagicMock, patch

import pytest

from distllm.core.webhook_manager import WebhookEvent, WebhookManager
from distllm.core.webhook_formatters import (
    slack_formatter,
    discord_formatter,
    pagerduty_formatter,
    get_formatter,
    _summary,
    _severity,
)


# ── Sample payloads ─────────────────────────────────────────────────────────

NODE_FAILED_PAYLOAD = {
    "node_id": "node-0",
    "host": "10.0.0.1",
    "error": "Connection refused",
    "attempts": 3,
}

NODE_DRAINING_PAYLOAD = {
    "node_id": "node-2",
    "reason": "maintenance",
}

HIGH_LATENCY_PAYLOAD = {
    "node_id": "node-1",
    "latency_ms": 4520,
    "threshold": 2000,
}

CIRCUIT_BREAKER_PAYLOAD = {
    "node_id": "node-0",
    "failure_count": 5,
}

CLUSTER_HEALTH_PAYLOAD = {
    "status": "degraded",
    "unhealthy_nodes": 2,
    "total_nodes": 4,
}


# ── _severity / _summary ────────────────────────────────────────────────────


class TestHelpers:
    def test_severity_error_for_failure(self):
        assert _severity("node.failed") == "error"

    def test_severity_warning_for_draining(self):
        assert _severity("node.draining") == "warning"

    def test_severity_info_for_join(self):
        assert _severity("node.joined") == "info"

    def test_severity_info_for_recovered(self):
        assert _severity("node.recovered") == "info"

    def test_severity_error_for_unknown(self):
        assert _severity("some.random.event") == "error"

    def test_summary_node_failed(self):
        s = _summary("node.failed", {"node_id": "node-0"})
        assert "node-0" in s
        assert "failed" in s

    def test_summary_circuit_breaker(self):
        s = _summary("circuit_breaker.opened", {"node_id": "node-0"})
        assert "OPENED" in s

    def test_summary_high_latency(self):
        s = _summary("high_latency", {"node_id": "node-1", "latency_ms": 4520})
        assert "4520ms" in s


# ── Slack Formatter ─────────────────────────────────────────────────────────


class TestSlackFormatter:
    def test_returns_blocks_and_text(self):
        body, headers = slack_formatter("node.failed", NODE_FAILED_PAYLOAD)
        assert "blocks" in body
        assert "text" in body
        assert "attachments" in body
        assert headers == {}

    def test_contains_header_block(self):
        body, _ = slack_formatter("node.failed", NODE_FAILED_PAYLOAD)
        blocks = body["blocks"]
        header_block = next(b for b in blocks if b["type"] == "header")
        assert "Node" in header_block["text"]["text"]
        assert "failed" in header_block["text"]["text"]

    def test_contains_node_info(self):
        body, _ = slack_formatter("node.failed", NODE_FAILED_PAYLOAD)
        sections = [b for b in body["blocks"] if b["type"] == "section"]
        texts = []
        for s in sections:
            if "fields" in s:
                for f in s["fields"]:
                    texts.append(f["text"])
            if "text" in s:
                texts.append(s["text"]["text"])
        combined = " ".join(texts)
        assert "node-0" in combined

    def test_attachment_color_red_for_error(self):
        body, _ = slack_formatter("node.failed", NODE_FAILED_PAYLOAD)
        assert body["attachments"][0]["color"] == "#ff0000"

    def test_attachment_color_blue_for_info(self):
        body, _ = slack_formatter("node.recovered", NODE_FAILED_PAYLOAD)
        assert body["attachments"][0]["color"] == "#0066cc"

    def test_node_draining_format(self):
        body, _ = slack_formatter("node.draining", NODE_DRAINING_PAYLOAD)
        header = next(b for b in body["blocks"] if b["type"] == "header")
        assert "draining" in header["text"]["text"].lower()


# ── Discord Formatter ───────────────────────────────────────────────────────


class TestDiscordFormatter:
    def test_returns_embeds(self):
        body, headers = discord_formatter("node.failed", NODE_FAILED_PAYLOAD)
        assert "embeds" in body
        assert "content" in body
        assert headers == {}

    def test_embed_has_title(self):
        body, _ = discord_formatter("node.failed", NODE_FAILED_PAYLOAD)
        embed = body["embeds"][0]
        assert "Node" in embed["title"]
        assert "failed" in embed["title"]

    def test_embed_has_color_red(self):
        body, _ = discord_formatter("node.failed", NODE_FAILED_PAYLOAD)
        assert body["embeds"][0]["color"] == 15158332

    def test_embed_has_color_blue_for_info(self):
        body, _ = discord_formatter("node.recovered", NODE_FAILED_PAYLOAD)
        assert body["embeds"][0]["color"] == 3447003

    def test_embed_has_fields(self):
        body, _ = discord_formatter("node.failed", NODE_FAILED_PAYLOAD)
        fields = body["embeds"][0]["fields"]
        field_names = [f["name"] for f in fields]
        assert "Event" in field_names
        assert "Severity" in field_names
        assert "Node" in field_names

    def test_embed_timestamp(self):
        body, _ = discord_formatter("node.failed", NODE_FAILED_PAYLOAD)
        assert "timestamp" in body["embeds"][0]

    def test_content_for_critical(self):
        body, _ = discord_formatter("node.failed", NODE_FAILED_PAYLOAD)
        assert "ERROR" in body["content"]


# ── PagerDuty Formatter ─────────────────────────────────────────────────────


class TestPagerDutyFormatter:
    def test_returns_pd_v2_payload(self):
        body, headers = pagerduty_formatter("node.failed", NODE_FAILED_PAYLOAD, routing_key="abc123")
        assert body["routing_key"] == "abc123"
        assert body["event_action"] == "trigger"
        assert headers == {}

    def test_payload_contains_summary(self):
        body, _ = pagerduty_formatter("node.failed", NODE_FAILED_PAYLOAD, routing_key="abc123")
        assert "Node" in body["payload"]["summary"]

    def test_payload_severity_mapping(self):
        body, _ = pagerduty_formatter("node.failed", NODE_FAILED_PAYLOAD, routing_key="abc123")
        assert body["payload"]["severity"] == "error"

    def test_payload_severity_info(self):
        body, _ = pagerduty_formatter("node.joined", {"node_id": "node-0"}, routing_key="abc123")
        assert body["payload"]["severity"] == "info"

    def test_payload_source(self):
        body, _ = pagerduty_formatter("node.failed", NODE_FAILED_PAYLOAD, routing_key="abc123")
        assert body["payload"]["source"] == "node-0"

    def test_payload_custom_details(self):
        body, _ = pagerduty_formatter("node.failed", NODE_FAILED_PAYLOAD, routing_key="abc123")
        assert "error" in body["payload"]["custom_details"]
        assert body["payload"]["custom_details"]["error"] == "Connection refused"

    def test_dedup_key(self):
        body, _ = pagerduty_formatter("node.failed", NODE_FAILED_PAYLOAD, routing_key="abc123")
        assert body["dedup_key"] == "distllm:node.failed:node-0"

    def test_summary_truncated(self):
        long_payload = {"node_id": "x", "error": "x" * 200}
        body, _ = pagerduty_formatter("node.failed", long_payload, routing_key="abc123")
        assert len(body["payload"]["summary"]) <= 120


# ── Formatter registry ──────────────────────────────────────────────────────


class TestFormatterRegistry:
    def test_get_slack_formatter(self):
        assert get_formatter("slack") is slack_formatter

    def test_get_discord_formatter(self):
        assert get_formatter("discord") is discord_formatter

    def test_get_pagerduty_formatter(self):
        assert get_formatter("pagerduty") is pagerduty_formatter

    def test_get_unknown_returns_none(self):
        assert get_formatter("teams") is None


# ── Integration: WebhookManager + formatter ─────────────────────────────────


class TestManagerWithFormatter:
    def _dispatch_and_wait(self, mgr, event, payload):
        """Dispatch and wait for worker thread to deliver."""
        mgr.start()
        mgr.dispatch(event, payload)
        import time
        time.sleep(0.5)
        mgr.stop()
        # Give worker time to finish current iteration
        if mgr._worker_thread and mgr._worker_thread.is_alive():
            mgr._worker_thread.join(timeout=2)

    def test_formatter_applied_during_delivery(self):
        mgr = WebhookManager()
        captured = {}

        def fake_post(url, content, headers, timeout):
            captured["body"] = json.loads(content)
            captured["headers"] = headers
            resp = MagicMock()
            resp.status_code = 200
            return resp

        mgr.register(
            "http://localhost:9999/hook",
            events=["node.failed"],
            formatter=slack_formatter,
            allow_private=True,
        )
        with patch("httpx.post", side_effect=fake_post):
            self._dispatch_and_wait(mgr, WebhookEvent.NODE_FAILED, {"node_id": "node-0"})

        assert "blocks" in captured["body"]
        assert captured["headers"]["Content-Type"] == "application/json"

    def test_plain_delivery_still_works(self):
        mgr = WebhookManager()
        captured = {}

        def fake_post(url, content, headers, timeout):
            captured["body"] = json.loads(content)
            return MagicMock(status_code=200)

        mgr.register("http://localhost:9999/hook", events=["node.joined"], allow_private=True)
        with patch("httpx.post", side_effect=fake_post):
            self._dispatch_and_wait(mgr, WebhookEvent.NODE_JOINED, {"node_id": "node-0"})

        assert captured["body"]["event"] == "node.joined"

    def test_new_events_are_valid(self):
        """Verify the new incident event types work with dispatch."""
        mgr = WebhookManager()
        mgr.register("http://localhost:9999/hook", events=["*"], allow_private=True)

        for event in [
            WebhookEvent.NODE_DRAINING,
            WebhookEvent.CIRCUIT_BREAKER_OPENED,
            WebhookEvent.CIRCUIT_BREAKER_CLOSED,
            WebhookEvent.HIGH_LATENCY,
            WebhookEvent.CLUSTER_HEALTH_CHANGED,
            WebhookEvent.STRAAGLER_DETECTED,
            WebhookEvent.RECOVERY_STARTED,
            WebhookEvent.RECOVERY_COMPLETED,
            WebhookEvent.NODE_RECOVERED,
        ]:
            count = mgr.dispatch(event, {"node_id": "test"})
            assert count == 1, f"{event.value} should dispatch to 1 target"

    def test_all_formatters_produce_valid_json(self):
        """Verify each formatter produces JSON-serializable output."""
        payloads = [
            ("node.failed", NODE_FAILED_PAYLOAD),
            ("node.draining", NODE_DRAINING_PAYLOAD),
            ("high_latency", HIGH_LATENCY_PAYLOAD),
            ("circuit_breaker.opened", CIRCUIT_BREAKER_PAYLOAD),
            ("cluster.health_changed", CLUSTER_HEALTH_PAYLOAD),
            ("node.recovered", {"node_id": "node-0"}),
        ]
        for event, payload in payloads:
            for fmt_name in ("slack", "discord"):
                fmt = get_formatter(fmt_name)
                body, _ = fmt(event, payload)
                json.dumps(body)
            pd_body, _ = pagerduty_formatter(event, payload, routing_key="test")
            json.dumps(pd_body)
