"""Multi-cloud spot orchestrator for cost-optimized inference.

Provides a complete stack for running distributed LLM inference on
spot/preemptible instances across AWS, Azure, and GCP:

- **Providers**: Cloud-specific spot instance APIs (request, terminate, check)
- **Price tracking**: Multi-provider spot price history, caching, cheapest selection
- **Preemption prediction**: ML-based risk scoring (price volatility, trend, ratio)
- **Auto-provisioning**: Scale up/down based on cost and SLA requirements
- **Budget scheduling**: Per-hour budget enforcement with automatic cost optimization
- **Migration planning**: Graceful KV cache migration on preemption
"""

from distllm.cloud.spot_provider import (
    CloudProvider,
    SpotProvider,
    SpotPrice,
    SpotInstance,
)
from distllm.cloud.providers import (
    AWSSpotProvider,
    GCPSpotProvider,
    AzureSpotProvider,
)
from distllm.cloud.spot_price_tracker import SpotPriceTracker, PriceRecord
from distllm.cloud.preemption_predictor import (
    PreemptionPredictor,
    PreemptionPrediction,
)
from distllm.cloud.budget_scheduler import BudgetScheduler, BudgetDecision, BudgetAction
from distllm.cloud.migration_planner import (
    MigrationPlanner,
    MigrationPlan,
    MigrationStep,
    MigrationStrategy,
)
from distllm.cloud.auto_provisioner import AutoProvisioner, ScalingDecision
from distllm.cloud.budget_alerter import BudgetAlerter, BudgetAlert, AlertLevel, AlertChannel
from distllm.cloud.workload_migrator import WorkloadMigrator

__all__ = [
    "CloudProvider",
    "SpotProvider",
    "SpotPrice",
    "SpotInstance",
    "AWSSpotProvider",
    "AzureSpotProvider",
    "GCPSpotProvider",
    "SpotPriceTracker",
    "PriceRecord",
    "PreemptionPredictor",
    "PreemptionPrediction",
    "BudgetScheduler",
    "BudgetDecision",
    "BudgetAction",
    "MigrationPlanner",
    "MigrationPlan",
    "MigrationStep",
    "MigrationStrategy",
    "AutoProvisioner",
    "ScalingDecision",
    "BudgetAlerter",
    "BudgetAlert",
    "AlertLevel",
    "AlertChannel",
    "WorkloadMigrator",
]
