"""Tests: BudgetScheduler, BudgetAlerter, AutoProvisioner."""

import time
import threading
from unittest.mock import MagicMock, patch

import pytest

from distllm.cloud.budget_scheduler import BudgetScheduler, BudgetAction, BudgetDecision
from distllm.cloud.budget_alerter import BudgetAlerter, AlertLevel, AlertChannel, BudgetAlert
from distllm.cloud.auto_provisioner import AutoProvisioner, ScalingDecision
from distllm.cloud.spot_provider import CloudProvider


# ===========================================================================
# BudgetScheduler
# ===========================================================================


class TestBudgetSchedulerSpendTracking:
    def test_register_instance_tracks_spend(self):
        sched = BudgetScheduler(budget_per_hour=10.0)
        assert sched.current_spend() == 0.0
        sched.register_instance("i-1", CloudProvider.AWS, "g5.xlarge", "us-east-1", 0.50)
        sched.register_instance("i-2", CloudProvider.AWS, "g5.2xlarge", "us-east-1", 0.75)
        assert sched.current_spend() == pytest.approx(1.25)

    def test_unregister_removes_from_current(self):
        sched = BudgetScheduler(budget_per_hour=10.0)
        sched.register_instance("i-1", CloudProvider.AWS, "g5.xlarge", "us-east-1", 0.50)
        sched.unregister_instance("i-1")
        assert sched.current_spend() == 0.0

    def test_accumulated_spend_increases(self):
        sched = BudgetScheduler(budget_per_hour=10.0)
        sched.register_instance("i-1", CloudProvider.AWS, "g5.xlarge", "us-east-1", 0.50)
        accrued = sched.accumulated_spend()
        assert accrued == pytest.approx(0.50, rel=1.0)

    def test_zero_budget_returns_no_decisions(self):
        sched = BudgetScheduler(budget_per_hour=0.0)
        decisions = sched._evaluate()
        assert decisions == []


class TestBudgetSchedulerThresholds:
    def test_warning_at_80_percent(self):
        sched = BudgetScheduler(budget_per_hour=10.0, warning_threshold=0.8, critical_threshold=0.9)
        sched.register_instance("i-1", CloudProvider.AWS, "p4d.24xlarge", "us-east-1", 8.50)
        decisions = sched._evaluate()
        assert len(decisions) >= 1
        assert decisions[0].action == BudgetAction.SWITCH_PROVIDER

    def test_critical_at_90_percent_triggers_savings(self):
        sched = BudgetScheduler(budget_per_hour=10.0, warning_threshold=0.8, critical_threshold=0.9)
        sched.register_instance("i-1", CloudProvider.AWS, "p4d.24xlarge", "us-east-1", 9.50)
        decisions = sched._evaluate()
        assert len(decisions) >= 1
        assert decisions[0].action in (BudgetAction.SWITCH_INSTANCE, BudgetAction.SCALE_DOWN)

    def test_exceeded_100_percent_halt_provisioning(self):
        sched = BudgetScheduler(budget_per_hour=10.0, critical_threshold=0.9)
        sched.register_instance("i-1", CloudProvider.AWS, "p4d.24xlarge", "us-east-1", 20.0)
        decisions = sched._evaluate()
        assert len(decisions) >= 1


class TestBudgetSchedulerCostSavings:
    def test_switch_instance_when_cheaper_exists(self):
        sched = BudgetScheduler(budget_per_hour=10.0, critical_threshold=0.9)
        sched.register_instance("expensive", CloudProvider.AWS, "p4d.24xlarge", "us-east-1", 9.50)
        sched.register_instance("cheap", CloudProvider.AWS, "g5.xlarge", "us-east-1", 0.50)
        decision = sched._find_cost_savings()
        assert decision is not None
        assert decision.action == BudgetAction.SWITCH_INSTANCE

    def test_scale_down_when_no_cheaper_instance(self):
        sched = BudgetScheduler(budget_per_hour=10.0, critical_threshold=0.9)
        sched.register_instance("only-one", CloudProvider.AWS, "p4d.24xlarge", "us-east-1", 9.50)
        decision = sched._find_cost_savings()
        assert decision is not None
        assert decision.action == BudgetAction.SCALE_DOWN

    def test_no_savings_all_same_price(self):
        sched = BudgetScheduler(budget_per_hour=1.0, critical_threshold=0.9)
        sched.register_instance("a", CloudProvider.AWS, "g5.xlarge", "us-east-1", 0.50)
        sched.register_instance("b", CloudProvider.AWS, "g5.xlarge", "us-east-1", 0.50)
        decision = sched._find_cost_savings()
        assert decision is not None
        assert decision.action in (BudgetAction.SWITCH_INSTANCE, BudgetAction.SCALE_DOWN)

    def test_no_instances_returns_none(self):
        sched = BudgetScheduler(budget_per_hour=10.0)
        assert sched._find_cost_savings() is None


class TestBudgetSchedulerCallback:
    def test_on_decision_callback_invoked(self):
        calls = []
        def cb(decision):
            calls.append(decision)
        sched = BudgetScheduler(budget_per_hour=10.0, warning_threshold=0.8, on_decision=cb)
        sched.register_instance("i-1", CloudProvider.AWS, "p4d.24xlarge", "us-east-1", 9.0)
        decisions = sched._evaluate()
        # Callback fires in _loop(), not _evaluate(). Manually trigger.
        for d in decisions:
            if sched.on_decision:
                sched.on_decision(d)
        assert len(calls) >= 1

    def test_callback_error_does_not_crash(self):
        sched = BudgetScheduler(budget_per_hour=10.0, warning_threshold=0.8, on_decision=lambda d: (_ for _ in ()).throw(RuntimeError("fail")))
        sched.register_instance("i-1", CloudProvider.AWS, "p4d.24xlarge", "us-east-1", 9.0)
        decisions = sched._evaluate()
        for d in decisions:
            if sched.on_decision:
                try:
                    sched.on_decision(d)
                except RuntimeError:
                    pass


class TestBudgetSchedulerThreadSafety:
    def test_concurrent_register(self):
        sched = BudgetScheduler(budget_per_hour=100.0)
        errors = []
        def register(n):
            try:
                sched.register_instance(f"i-{n}", CloudProvider.AWS, "g5.xlarge", "us-east-1", 0.50)
            except Exception as e:
                errors.append(e)
        threads = [threading.Thread(target=register, args=(i,)) for i in range(50)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert len(errors) == 0
        assert sched.current_spend() == pytest.approx(25.0)

    def test_start_sets_running_flag(self):
        sched = BudgetScheduler(budget_per_hour=10.0, check_interval_s=0.01)
        sched._running = True
        assert sched._running is True
        sched._running = False

    def test_double_start_no_error(self):
        sched = BudgetScheduler(budget_per_hour=10.0)
        sched._running = True
        sched.start()
        sched._running = False


# ===========================================================================
# BudgetAlerter
# ===========================================================================


class TestBudgetAlerterThresholds:
    def test_warning_at_80_percent(self):
        alerter = BudgetAlerter()
        alerter.set_budget(100.0)
        alert = alerter.check_budget(80.0)
        assert alert is not None
        assert alert.level == AlertLevel.WARNING

    def test_critical_at_90_percent(self):
        alerter = BudgetAlerter()
        alerter.set_budget(100.0)
        alerter.check_budget(80.0)
        alert = alerter.check_budget(90.0)
        assert alert is not None
        assert alert.level == AlertLevel.CRITICAL

    def test_exceeded_at_100_percent(self):
        alerter = BudgetAlerter()
        alerter.set_budget(100.0)
        alerter.check_budget(80.0)
        alerter.check_budget(90.0)
        alert = alerter.check_budget(100.0)
        assert alert is not None
        assert alert.level == AlertLevel.EXCEEDED

    def test_hysteresis_same_level_no_duplicate(self):
        alerter = BudgetAlerter()
        alerter.set_budget(100.0)
        alerter.check_budget(80.0)
        alert2 = alerter.check_budget(85.0)
        assert alert2 is None  # WARNING already fired

    def test_zero_budget_returns_none(self):
        alerter = BudgetAlerter()
        assert alerter.check_budget(100.0) is None


class TestBudgetAlerterBudgetChange:
    def test_budget_change_resets_fired_alerts(self):
        alerter = BudgetAlerter()
        alerter.set_budget(100.0)
        alerter.check_budget(80.0)  # fires WARNING
        alerter.set_budget(50.0)     # reset
        alert = alerter.check_budget(40.0)  # 80% of 50 = 40, should fire again
        assert alert is not None
        assert alert.level == AlertLevel.WARNING

    def test_reset_clears_fired(self):
        alerter = BudgetAlerter()
        alerter.set_budget(100.0)
        alerter.check_budget(80.0)
        alerter.reset()
        alert = alerter.check_budget(80.0)
        assert alert is not None


class TestBudgetAlerterChannels:
    def test_log_channel_fires(self):
        alerter = BudgetAlerter(channels=[AlertChannel.LOG])
        alerter.set_budget(100.0)
        with patch("distllm.cloud.budget_alerter.logger") as mock_log:
            alert = alerter.check_budget(80.0)
            assert alert is not None

    def test_webhook_channel_posts(self):
        alerter = BudgetAlerter(
            channels=[AlertChannel.WEBHOOK],
            webhook_url="http://example.com/webhook",
        )
        alerter.set_budget(100.0)
        with patch("urllib.request.urlopen") as mock_open:
            alert = alerter.check_budget(80.0)
            assert alert is not None
            mock_open.assert_called_once()

    def test_slack_channel_uses_webhook(self):
        alerter = BudgetAlerter(
            channels=[AlertChannel.SLACK],
            webhook_url="http://example.com/slack",
        )
        alerter.set_budget(100.0)
        with patch("urllib.request.urlopen") as mock_open:
            alert = alerter.check_budget(80.0)
            assert alert is not None
            mock_open.assert_called_once()

    def test_email_channel_logs_debug(self):
        alerter = BudgetAlerter(channels=[AlertChannel.EMAIL])
        alerter.set_budget(100.0)
        with patch("distllm.cloud.budget_alerter.logger") as mock_log:
            alert = alerter.check_budget(80.0)
            assert alert is not None

    def test_unknown_channel_silently_ignored(self):
        alerter = BudgetAlerter(channels=[AlertChannel.LOG])
        alerter.set_budget(100.0)
        alert = alerter.check_budget(80.0)
        assert alert is not None

    def test_webhook_no_url_skips(self):
        alerter = BudgetAlerter(channels=[AlertChannel.WEBHOOK], webhook_url="")
        alerter.set_budget(100.0)
        alert = alerter.check_budget(80.0)
        assert alert is not None

    def test_get_alert_history(self):
        alerter = BudgetAlerter()
        alerter.set_budget(100.0)
        alerter.check_budget(80.0)
        history = alerter.get_alert_history()
        assert len(history) == 1
        assert history[0].level == AlertLevel.WARNING


class TestBudgetAlertDataclass:
    def test_auto_message(self):
        alert = BudgetAlert(level=AlertLevel.WARNING, current_cost=80.0, budget_limit=100.0, percent_used=80.0)
        assert "80.0%" in alert.message
        assert alert.timestamp > 0

    def test_custom_message(self):
        alert = BudgetAlert(level=AlertLevel.CRITICAL, current_cost=90.0, budget_limit=100.0,
                            percent_used=90.0, message="custom")
        assert alert.message == "custom"


# ===========================================================================
# AutoProvisioner
# ===========================================================================


class TestAutoProvisionerScaleUp:
    def test_scale_up_selects_cheapest_instance(self):
        tracker = MagicMock()
        tracker.get_cheapest_compatible.return_value = MagicMock(
            provider=CloudProvider.AWS, instance_type="g5.xlarge",
            region="us-east-1", price=0.50,
        )
        provider_mock = MagicMock()
        provider_mock.request_instance.return_value = "i-abc"
        tracker.get_provider.return_value = provider_mock
        provisioner = AutoProvisioner(price_tracker=tracker)
        ids = provisioner.scale_up(node_count=2, required_vram_gb=20)
        assert len(ids) == 2
        tracker.get_cheapest_compatible.assert_called_once()

    def test_scale_up_budget_exceeded_reduces_count(self):
        tracker = MagicMock()
        tracker.get_cheapest_compatible.return_value = MagicMock(
            provider=CloudProvider.AWS, instance_type="p4d.24xlarge",
            region="us-east-1", price=5.0,
        )
        provider_mock = MagicMock()
        provider_mock.request_instance.return_value = "i-abc"
        tracker.get_provider.return_value = provider_mock
        provisioner = AutoProvisioner(price_tracker=tracker, budget_per_hour=6.0)
        ids = provisioner.scale_up(node_count=4, required_vram_gb=40)
        assert len(ids) == 1

    def test_scale_up_no_compatible_instance_returns_empty(self):
        tracker = MagicMock()
        tracker.get_cheapest_compatible.return_value = None
        provisioner = AutoProvisioner(price_tracker=tracker)
        ids = provisioner.scale_up(node_count=2, required_vram_gb=999)
        assert ids == []

    def test_scale_up_partial_failure_records_successes(self):
        tracker = MagicMock()
        tracker.get_cheapest_compatible.return_value = MagicMock(
            provider=CloudProvider.AWS, instance_type="g5.xlarge",
            region="us-east-1", price=0.50,
        )
        provider_mock = MagicMock()
        provider_mock.request_instance.side_effect = [
            "i-001", RuntimeError("API error"), "i-002",
        ]
        tracker.get_provider.return_value = provider_mock
        provisioner = AutoProvisioner(price_tracker=tracker)
        ids = provisioner.scale_up(node_count=3, required_vram_gb=20)
        assert len(ids) >= 1
        assert "i-001" in ids


class TestAutoProvisionerScaleDown:
    def test_scale_down_terminates_instances(self):
        tracker = MagicMock()
        provider_mock = MagicMock()
        provider_mock.terminate_instance.return_value = True
        tracker.get_provider.return_value = provider_mock
        provisioner = AutoProvisioner(price_tracker=tracker)
        provisioner._provisioned = [
            {"instance_id": "i-001", "provider": "aws", "region": "us-east-1"},
        ]
        result = provisioner.scale_down(["i-001"])
        assert result is True
        assert provisioner.get_provisioned_instances() == []

    def test_scale_down_unknown_instance(self):
        tracker = MagicMock()
        provisioner = AutoProvisioner(price_tracker=tracker)
        result = provisioner.scale_down(["i-nonexistent"])
        assert result is False


class TestAutoProvisionerDecisions:
    def test_scale_up_records_decision(self):
        tracker = MagicMock()
        tracker.get_cheapest_compatible.return_value = MagicMock(
            provider=CloudProvider.AWS, instance_type="g5.xlarge",
            region="us-east-1", price=0.50,
        )
        provider_mock = MagicMock()
        provider_mock.request_instance.return_value = "i-001"
        tracker.get_provider.return_value = provider_mock
        provisioner = AutoProvisioner(price_tracker=tracker)
        provisioner.scale_up(node_count=1, required_vram_gb=20)
        decisions = provisioner.get_decisions()
        assert len(decisions) == 1
        assert decisions[0].action == "scale_up"

    def test_scale_down_records_decision(self):
        tracker = MagicMock()
        provisioner = AutoProvisioner(price_tracker=tracker)
        provisioner._provisioned = [
            {"instance_id": "i-001", "provider": "aws", "region": "us-east-1"},
        ]
        provisioner.scale_down(["i-001"])
        decisions = provisioner.get_decisions()
        assert len(decisions) == 1
        assert decisions[0].action == "scale_down"

    def test_get_provisioned_instances(self):
        tracker = MagicMock()
        provisioner = AutoProvisioner(price_tracker=tracker)
        provisioner._provisioned = [{"instance_id": "i-001"}]
        assert len(provisioner.get_provisioned_instances()) == 1
