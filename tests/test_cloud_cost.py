"""Tests for cost-optimized cloud inference components."""

import time

from distllm.cloud.spot_provider import (
    CloudProvider,
    AWSSpotProvider,
    AzureSpotProvider,
    GCPSpotProvider,
    LambdaSpotProvider,
    SpotPrice,
)
from distllm.cloud.spot_price_tracker import SpotPriceTracker, PriceRecord
from distllm.cloud.workload_migrator import WorkloadMigrator
from distllm.cloud.budget_alerter import BudgetAlerter, BudgetAlert, AlertLevel, AlertChannel
from distllm.cloud.auto_provisioner import AutoProvisioner


class TestSpotProvider:
    def test_aws_provider_name(self):
        provider = AWSSpotProvider()
        assert provider.provider_name == CloudProvider.AWS

    def test_azure_provider_name(self):
        provider = AzureSpotProvider()
        assert provider.provider_name == CloudProvider.AZURE

    def test_gcp_provider_name(self):
        provider = GCPSpotProvider()
        assert provider.provider_name == CloudProvider.GCP

    def test_lambda_provider_name(self):
        provider = LambdaSpotProvider()
        assert provider.provider_name == CloudProvider.LAMBDA

    def test_spot_price_savings(self):
        price = SpotPrice(
            provider=CloudProvider.AWS,
            instance_type="g5.xlarge",
            region="us-east-1",
            price=0.50,
            on_demand_price=1.00,
        )
        assert price.savings_percent == 50.0


class TestSpotPriceTracker:
    def test_register_provider(self):
        tracker = SpotPriceTracker()
        tracker.register_provider(AWSSpotProvider())
        assert CloudProvider.AWS in tracker._providers

    def test_cache_staleness(self):
        record = PriceRecord(
            provider=CloudProvider.AWS,
            instance_type="g5.xlarge",
            region="us-east-1",
            price=0.50,
            on_demand_price=1.00,
            timestamp=time.time() - 600,  # 10 minutes ago
            ttl_seconds=300,
        )
        assert record.is_stale

    def test_cache_fresh(self):
        record = PriceRecord(
            provider=CloudProvider.AWS,
            instance_type="g5.xlarge",
            region="us-east-1",
            price=0.50,
            on_demand_price=1.00,
            timestamp=time.time(),
            ttl_seconds=300,
        )
        assert not record.is_stale

    def test_preemption_risk_insufficient_data(self):
        tracker = SpotPriceTracker()
        risk = tracker.predict_preemption_risk(CloudProvider.AWS, "g5.xlarge", "us-east-1")
        assert risk == 0.5  # Unknown risk with no history

    def test_preemption_risk_with_history(self):
        tracker = SpotPriceTracker()
        # Manually populate history
        key = "aws:g5.xlarge:us-east-1"
        tracker._history[key] = [0.40, 0.42, 0.45, 0.50, 0.55, 0.60, 0.65]
        risk = tracker.predict_preemption_risk(CloudProvider.AWS, "g5.xlarge", "us-east-1")
        assert 0.0 <= risk <= 1.0


class TestBudgetAlerter:
    def test_warning_at_80_percent(self):
        alerter = BudgetAlerter(channels=[AlertChannel.LOG])
        alerter.set_budget(100.0)
        alert = alerter.check_budget(80.0)
        assert alert is not None
        assert alert.level == AlertLevel.WARNING
        assert alert.percent_used == 80.0

    def test_critical_at_90_percent(self):
        alerter = BudgetAlerter(channels=[AlertChannel.LOG])
        alerter.set_budget(100.0)
        # First warning at 80%
        alerter.check_budget(80.0)
        # Then critical at 90%
        alert = alerter.check_budget(90.0)
        assert alert is not None
        assert alert.level == AlertLevel.CRITICAL

    def test_exceeded_at_100_percent(self):
        alerter = BudgetAlerter(channels=[AlertChannel.LOG])
        alerter.set_budget(100.0)
        alerter.check_budget(80.0)
        alerter.check_budget(90.0)
        alert = alerter.check_budget(100.0)
        assert alert is not None
        assert alert.level == AlertLevel.EXCEEDED

    def test_no_duplicate_alerts(self):
        alerter = BudgetAlerter(channels=[AlertChannel.LOG])
        alerter.set_budget(100.0)
        alert1 = alerter.check_budget(80.0)
        alert2 = alerter.check_budget(85.0)  # Still in warning zone
        assert alert1 is not None
        assert alert2 is None  # Already fired warning

    def test_reset(self):
        alerter = BudgetAlerter(channels=[AlertChannel.LOG])
        alerter.set_budget(100.0)
        alerter.check_budget(80.0)
        alerter.reset()
        alert = alerter.check_budget(80.0)
        assert alert is not None  # Should fire again after reset


class TestWorkloadMigrator:
    def test_migrate_with_state(self):
        migrator = WorkloadMigrator()
        kv_state = {"layer_0": "data", "layer_1": "data"}
        result = migrator.migrate_node_workload("node-a", "node-b", kv_state)
        assert result is True

    def test_migrate_without_state(self):
        migrator = WorkloadMigrator()
        result = migrator.migrate_node_workload("node-a", "node-b")
        assert result is True  # Completes with no-state status

    def test_migration_tracking(self):
        migrator = WorkloadMigrator()
        migrator.migrate_node_workload("node-a", "node-b", {"k": "v"})
        status = migrator.get_migration_status("node-a->node-b")
        assert status is not None
        assert status["status"] == "completed"


class TestAutoProvisioner:
    def test_scale_up_no_provider(self):
        provisioner = AutoProvisioner()
        ids = provisioner.scale_up(2, required_vram_gb=24.0)
        assert ids == []  # No providers registered

    def test_decisions_tracking(self):
        provisioner = AutoProvisioner()
        decisions = provisioner.get_decisions()
        assert decisions == []
