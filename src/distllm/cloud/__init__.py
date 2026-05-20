"""DistLLM cloud optimization package.

Multi-cloud spot instance orchestration with auto-scaling,
preemption handling, and budget management.
"""

from distllm.cloud.spot_provider import (
    CloudProvider,
    SpotProvider,
    AWSSpotProvider,
    AzureSpotProvider,
    GCPSpotProvider,
    LambdaSpotProvider,
    SpotPrice,
    SpotInstance,
)
from distllm.cloud.spot_price_tracker import SpotPriceTracker, PriceRecord
from distllm.cloud.workload_migrator import WorkloadMigrator
from distllm.cloud.budget_alerter import BudgetAlerter, BudgetAlert, AlertLevel, AlertChannel
from distllm.cloud.auto_provisioner import AutoProvisioner, ScalingDecision

__all__ = [
    "CloudProvider",
    "SpotProvider",
    "AWSSpotProvider",
    "AzureSpotProvider",
    "GCPSpotProvider",
    "LambdaSpotProvider",
    "SpotPrice",
    "SpotInstance",
    "SpotPriceTracker",
    "PriceRecord",
    "WorkloadMigrator",
    "BudgetAlerter",
    "BudgetAlert",
    "AlertLevel",
    "AlertChannel",
    "AutoProvisioner",
    "ScalingDecision",
]
