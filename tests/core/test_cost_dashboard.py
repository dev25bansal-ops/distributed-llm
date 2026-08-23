"""Tests for CostDashboard and SLAManager."""

from __future__ import annotations

import time

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_cd_mod = load_module("distllm/core/cost_dashboard.py")
CostDashboard = _cd_mod.CostDashboard
SLAManager = _cd_mod.SLAManager
CostRecord = _cd_mod.CostRecord
BudgetAlert = _cd_mod.BudgetAlert


class TestCostRecord:
    """CostRecord dataclass -- construction."""

    def test_default_construction(self):
        record = CostRecord(user_id="user-1", model="llama-3", tokens=100, cost_usd=0.05)
        assert record.user_id == "user-1"
        assert record.model == "llama-3"
        assert record.tokens == 100
        assert record.cost_usd == 0.05
        assert record.timestamp > 0
        assert record.request_id == ""

    def test_custom_construction(self):
        record = CostRecord(
            user_id="user-2", model="gpt-4", tokens=500,
            cost_usd=0.25, timestamp=1000.0, request_id="req-123",
        )
        assert record.user_id == "user-2"
        assert record.request_id == "req-123"
        assert record.timestamp == 1000.0


class TestBudgetAlert:
    """BudgetAlert dataclass -- construction."""

    def test_default_construction(self):
        alert = BudgetAlert(
            user_id="user-1", threshold_pct=0.5,
            current_spend=50.0, budget_usd=100.0, message="test",
        )
        assert alert.user_id == "user-1"
        assert alert.threshold_pct == 0.5
        assert alert.current_spend == 50.0
        assert alert.budget_usd == 100.0
        assert alert.message == "test"
        assert alert.timestamp > 0


class TestCostDashboard:
    """CostDashboard -- recording, reports, projections, alerts."""

    def test_default_construction(self):
        dashboard = CostDashboard()
        assert dashboard._default_budget == 100.0
        assert dashboard._alert_thresholds == [0.5, 0.75, 0.9, 1.0]
        assert dashboard._records == []
        assert dashboard._budgets == {}

    def test_set_budget(self):
        dashboard = CostDashboard()
        dashboard.set_budget("user-1", 500.0)
        assert dashboard._budgets["user-1"] == 500.0

    def test_record_cost(self):
        dashboard = CostDashboard()
        result = dashboard.record_cost(
            user_id="user-1", model="llama-3", tokens=100, cost_usd=0.05,
        )
        assert result is None  # no alert threshold crossed
        assert len(dashboard._records) == 1
        assert dashboard._records[0].cost_usd == 0.05

    def test_record_cost_triggers_alert(self):
        dashboard = CostDashboard(default_budget_usd=1.0)
        dashboard.set_budget("user-1", 1.0)
        # Record enough to trip 50% threshold
        result = dashboard.record_cost(
            user_id="user-1", model="llama-3", tokens=100, cost_usd=1.0,
        )
        assert result is not None
        assert result.threshold_pct == 0.5  # first threshold crossed

    def test_record_cost_triggers_multiple_thresholds(self):
        dashboard = CostDashboard(default_budget_usd=1.0)
        dashboard.set_budget("user-1", 1.0)
        # Spend 60% — should trigger 50% only
        dashboard.record_cost(user_id="user-1", model="m", tokens=100, cost_usd=0.60)
        # Spend another 30% (now 90%) — should trigger 75% and 90%
        result = dashboard.record_cost(user_id="user-1", model="m", tokens=100, cost_usd=0.30)
        assert result is not None
        # Threshold should be 75% or 90% (the newly crossed ones)
        assert result.threshold_pct in (0.75, 0.9)

    def test_alert_fires_only_once_per_threshold(self):
        dashboard = CostDashboard(default_budget_usd=1.0)
        dashboard.set_budget("user-1", 1.0)
        dashboard.record_cost(user_id="user-1", model="m", tokens=100, cost_usd=0.60)
        # Second call at same spend level should not fire a new alert
        result = dashboard.record_cost(user_id="user-1", model="m", tokens=100, cost_usd=0.0)
        assert result is None  # no new threshold crossed

    def test_get_report_for_user(self):
        dashboard = CostDashboard()
        dashboard.record_cost(user_id="user-1", model="llama-3", tokens=100, cost_usd=0.05)
        dashboard.record_cost(user_id="user-1", model="gpt-4", tokens=200, cost_usd=0.20)
        dashboard.record_cost(user_id="user-2", model="llama-3", tokens=50, cost_usd=0.02)

        report = dashboard.get_report(user_id="user-1")
        assert report["total_cost_usd"] == 0.25
        assert report["total_tokens"] == 300
        assert report["record_count"] == 2
        assert "llama-3" in report["by_model"]
        assert "gpt-4" in report["by_model"]

    def test_get_report_aggregate(self):
        dashboard = CostDashboard()
        dashboard.record_cost(user_id="user-1", model="llama-3", tokens=100, cost_usd=0.05)
        dashboard.record_cost(user_id="user-2", model="gpt-4", tokens=200, cost_usd=0.20)

        report = dashboard.get_report()
        assert report["total_cost_usd"] == 0.25
        assert report["total_tokens"] == 300
        assert report["record_count"] == 2

    def test_get_report_empty(self):
        dashboard = CostDashboard()
        report = dashboard.get_report(user_id="user-1")
        assert report["total_cost_usd"] == 0.0
        assert report["total_tokens"] == 0
        assert report["record_count"] == 0

    def test_get_projection_no_records(self):
        dashboard = CostDashboard()
        proj = dashboard.get_projection("user-1")
        assert proj["projected_usd"] == 0
        assert proj["days_remaining"] == 30

    def test_get_forecast_no_records(self):
        dashboard = CostDashboard()
        forecast = dashboard.get_forecast("user-1")
        assert forecast["projected_monthly_usd"] == 0.0
        assert forecast["trend"] == "stable"
        assert forecast["confidence"] == "low"

    def test_get_forecast_trend_tracking(self):
        dashboard = CostDashboard()
        now = time.time()
        # Simulate records spread across days (use old timestamps)
        day_seconds = 86400
        for i in range(5):
            record = CostRecord(
                user_id="user-1", model="llama-3",
                tokens=100, cost_usd=1.0,
                timestamp=now - (10 - i) * day_seconds,
            )
            dashboard._records.append(record)

        forecast = dashboard.get_forecast("user-1")
        assert forecast["current_spend_usd"] == 5.0
        assert forecast["days_with_data"] > 0
        assert forecast["projected_monthly_usd"] > 0

    def test_reset_alerts(self):
        dashboard = CostDashboard()
        dashboard._alerted["user-1"] = {0.5, 0.75}
        dashboard.reset_alerts("user-1")
        assert "user-1" not in dashboard._alerted


class TestSLAManager:
    """SLAManager -- defining SLOs and evaluating violations."""

    def test_define_slo(self):
        slo_mgr = SLAManager()
        slo_mgr.define_slo("latency", "latency_p99", threshold=100.0)
        assert "latency" in slo_mgr._slos
        assert slo_mgr._slos["latency"]["metric"] == "latency_p99"
        assert slo_mgr._slos["latency"]["threshold"] == 100.0
        assert slo_mgr._slos["latency"]["violations"] == 0

    def test_evaluate_no_violation(self):
        slo_mgr = SLAManager()
        slo_mgr.define_slo("latency", "latency_p99", threshold=100.0, violation_threshold=3)
        # Single breach below threshold
        result = slo_mgr.evaluate("latency", 50.0)
        assert result is None

    def test_evaluate_violation_threshold_reached(self):
        slo_mgr = SLAManager()
        slo_mgr.define_slo("latency", "latency_p99", threshold=100.0, violation_threshold=2)
        slo_mgr.evaluate("latency", 150.0)  # breach 1
        result = slo_mgr.evaluate("latency", 200.0)  # breach 2 -> triggers
        assert result is not None
        assert result["slo_name"] == "latency"
        assert result["actual"] == 200.0
        assert result["remediation"] == "alert"

    def test_evaluate_resets_on_ok(self):
        slo_mgr = SLAManager()
        slo_mgr.define_slo("latency", "latency_p99", threshold=100.0, violation_threshold=3)
        slo_mgr.evaluate("latency", 150.0)  # breach 1
        slo_mgr.evaluate("latency", 50.0)  # OK -> resets counter
        result = slo_mgr.evaluate("latency", 200.0)  # breach 1 again
        assert result is None  # still only 1

    def test_evaluate_unknown_slo(self):
        slo_mgr = SLAManager()
        result = slo_mgr.evaluate("nonexistent", 100.0)
        assert result is None

    def test_get_status(self):
        slo_mgr = SLAManager()
        slo_mgr.define_slo("latency", "latency_p99", threshold=100.0)
        slo_mgr.define_slo("throughput", "tokens_per_sec", threshold=50.0)
        status = slo_mgr.get_status()
        assert "latency" in status
        assert "throughput" in status
        assert status["latency"]["threshold"] == 100.0

    def test_get_violations(self):
        slo_mgr = SLAManager()
        slo_mgr.define_slo("latency", "latency_p99", threshold=100.0, violation_threshold=1)
        slo_mgr.evaluate("latency", 200.0)
        violations = slo_mgr.get_violations()
        assert len(violations) == 1
        assert violations[0]["slo_name"] == "latency"
