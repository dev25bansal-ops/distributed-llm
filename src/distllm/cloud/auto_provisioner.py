"""Auto-provisioner for cost-optimized cloud inference.

Integrates the scheduler with cloud providers to automatically
scale nodes based on cost and SLA requirements.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from distllm.cloud.spot_provider import CloudProvider, SpotPrice, SpotProvider
from distllm.cloud.spot_price_tracker import SpotPriceTracker


@dataclass
class ScalingDecision:
    """A scaling decision with rationale."""
    action: str  # "scale_up" or "scale_down"
    instance_type: str
    provider: CloudProvider
    region: str
    count: int
    reason: str
    estimated_cost_per_hour: float = 0.0
    timestamp: float = 0.0


class AutoProvisioner:
    """Automatically provisions cloud instances based on cost and SLA.

    Integrates with:
    - SpotPriceTracker for cheapest instance selection
    - BudgetAlerter for cost constraint enforcement
    - Karpenter manifests for K8s deployments
    """

    def __init__(
        self,
        price_tracker: SpotPriceTracker | None = None,
        budget_per_hour: float = 0.0,
        spot_preference: float = 0.8,
        default_region: str = "us-east-1",
    ) -> None:
        self.price_tracker = price_tracker or SpotPriceTracker()
        self.budget_per_hour = budget_per_hour
        self.spot_preference = spot_preference
        self.default_region = default_region

        self._provisioned: list[dict[str, Any]] = []
        self._decisions: list[ScalingDecision] = []

    def scale_up(
        self,
        node_count: int,
        required_vram_gb: float = 24.0,
        providers: list[CloudProvider] | None = None,
        region: str | None = None,
    ) -> list[str]:
        """Provision new nodes using cheapest compatible spot instances.

        Args:
            node_count: Number of nodes to provision.
            required_vram_gb: Minimum VRAM per node.
            providers: Preferred cloud providers.
            region: Target region (uses default if None).

        Returns:
            List of provisioned instance IDs.
        """
        target_region = region or self.default_region
        instance_ids = []

        # Find cheapest compatible instance
        best_price = self.price_tracker.get_cheapest_compatible(
            required_vram_gb=required_vram_gb,
            providers=providers,
            region=target_region,
        )

        if best_price is None:
            logger.warning("No compatible spot instance found for scale-up")
            return []

        # Check budget
        if self.budget_per_hour > 0:
            total_cost = best_price.price * node_count
            if total_cost > self.budget_per_hour:
                logger.warning(
                    f"Scale-up would exceed budget: ${total_cost:.2f}/hr > ${self.budget_per_hour:.2f}/hr"
                )
                # Scale down node count to fit budget
                node_count = max(1, int(self.budget_per_hour / best_price.price))
                logger.info(f"Reduced scale-up to {node_count} nodes to fit budget")

        # Provision instances
        provider = self.price_tracker._providers.get(best_price.provider)
        if provider:
            for _ in range(node_count):
                try:
                    instance_id = provider.request_instance(
                        instance_type=best_price.instance_type,
                        region=target_region,
                    )
                    instance_ids.append(instance_id)
                    self._provisioned.append({
                        "instance_id": instance_id,
                        "provider": best_price.provider.value,
                        "instance_type": best_price.instance_type,
                        "region": target_region,
                        "price": best_price.price,
                        "launched_at": time.time(),
                    })
                except Exception as e:
                    logger.error(f"Failed to provision instance: {e}")
        else:
            logger.warning(f"Provider {best_price.provider} not found in price tracker")

        decision = ScalingDecision(
            action="scale_up",
            instance_type=best_price.instance_type,
            provider=best_price.provider,
            region=target_region,
            count=len(instance_ids),
            reason=f"Scale-up: {node_count} nodes, ${best_price.price:.4f}/hr each",
            estimated_cost_per_hour=best_price.price * len(instance_ids),
        )
        self._decisions.append(decision)

        logger.info(
            f"Provisioned {len(instance_ids)} spot instances: "
            f"{best_price.instance_type} in {target_region}"
        )
        return instance_ids

    def scale_down(self, instance_ids: list[str]) -> bool:
        """Terminate specified instances.

        Args:
            instance_ids: List of instance IDs to terminate.

        Returns:
            True if all instances were terminated.
        """
        all_success = True
        for instance_id in instance_ids:
            # Find provider from provisioned list
            entry = next(
                (e for e in self._provisioned if e["instance_id"] == instance_id),
                None,
            )
            if entry:
                provider = self.price_tracker._providers.get(
                    CloudProvider(entry["provider"])
                )
                if provider:
                    success = provider.terminate_instance(
                        instance_id=instance_id,
                        region=entry["region"],
                    )
                    if success:
                        self._provisioned = [
                            e for e in self._provisioned
                            if e["instance_id"] != instance_id
                        ]
                    else:
                        all_success = False
            else:
                logger.warning(f"Instance {instance_id} not found in provisioned list")
                all_success = False

        decision = ScalingDecision(
            action="scale_down",
            instance_type="",
            provider=CloudProvider.AWS,
            region="",
            count=len(instance_ids),
            reason=f"Scale-down: {len(instance_ids)} instances",
        )
        self._decisions.append(decision)

        logger.info(f"Terminated {len(instance_ids)} instances")
        return all_success

    def get_provisioned_instances(self) -> list[dict[str, Any]]:
        """Return list of currently provisioned instances."""
        return list(self._provisioned)

    def get_decisions(self, limit: int = 20) -> list[ScalingDecision]:
        """Return recent scaling decisions."""
        return self._decisions[-limit:]
