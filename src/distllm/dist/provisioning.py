"""Self-serve cluster provisioning — tenants request model deployments via API.

Extends :class:`Marketplace` with a provisioning workflow that:

1. Accepts a deployment request (model, GPU requirements, budget, region)
2. Selects the cheapest qualifying cloud region via :class:`CloudRegionSelector`
3. Reserves capacity from the marketplace or auto-provisions cloud instances
4. Deploys the model across the allocated GPUs
5. Returns an endpoint URL and status monitoring

This turns the coordinator into a self-serve platform where tenants
can request model deployments without manual intervention.

Usage::

    from distllm.dist.marketplace import Marketplace
    from distllm.dist.provisioning import ProvisioningEngine

    engine = ProvisioningEngine(marketplace=marketplace)
    deployment = engine.request_deployment(
        tenant_id="acme-corp",
        model_name="llama-70b",
        min_gpu_memory_gb=80,
        gpu_count=4,
        max_budget_per_hour=50.0,
    )
    # → {"deployment_id": "dep-a3f2...", "endpoint": "https://...",
    #     "status": "provisioning", "estimated_ready_s": 120}
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from loguru import logger

from distllm.dist.cloud_selector import CloudRegionSelector, RegionOffer


class DeploymentStatus(str, Enum):
    PENDING = "pending"
    PROVISIONING = "provisioning"
    RUNNING = "running"
    FAILED = "failed"
    TERMINATED = "terminated"


@dataclass
class Deployment:
    """A model deployment request and its current state."""
    deployment_id: str
    tenant_id: str
    model_name: str

    # Requirements
    gpu_count: int = 1
    min_gpu_memory_gb: float = 80.0
    max_budget_per_hour: float = 50.0
    preferred_regions: list[str] = field(default_factory=list)
    preferred_providers: list[str] = field(default_factory=list)

    # Assignment
    region_offer: RegionOffer | None = None
    assigned_cluster_id: str = ""
    endpoint_url: str = ""

    # Lifecycle
    status: DeploymentStatus = DeploymentStatus.PENDING
    created_at: float = field(default_factory=time.time)
    provisioned_at: float = 0.0
    terminated_at: float = 0.0
    error_message: str = ""

    # Usage
    tokens_served: int = 0
    cost_incurred: float = 0.0

    @property
    def age_hours(self) -> float:
        if self.status == DeploymentStatus.TERMINATED and self.terminated_at:
            return (self.terminated_at - self.created_at) / 3600.0
        return (time.time() - self.created_at) / 3600.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "deployment_id": self.deployment_id,
            "tenant_id": self.tenant_id,
            "model_name": self.model_name,
            "gpu_count": self.gpu_count,
            "status": self.status.value,
            "endpoint_url": self.endpoint_url,
            "assigned_cluster_id": self.assigned_cluster_id,
            "region": self.region_offer.region if self.region_offer else "",
            "provider": self.region_offer.provider if self.region_offer else "",
            "price_per_hour": self.region_offer.price_per_hour if self.region_offer else 0.0,
            "created_at": self.created_at,
            "age_hours": round(self.age_hours, 2),
            "tokens_served": self.tokens_served,
            "cost_incurred": round(self.cost_incurred, 4),
        }


class ProvisioningEngine:
    """Handles deployment requests from tenants.

    Integrates with :class:`Marketplace` for GPU listing discovery,
    :class:`CloudRegionSelector` for cloud region selection, and
    ``AutoScaler`` / ``FederationCoordinator`` for actual worker
    provisioning.
    """

    def __init__(
        self,
        marketplace: Any | None = None,
        cloud_selector: CloudRegionSelector | None = None,
        auto_scaler: Any | None = None,
        federation: Any | None = None,
    ):
        self._marketplace = marketplace
        self._cloud_selector = cloud_selector or CloudRegionSelector()
        self._auto_scaler = auto_scaler
        self._federation = federation
        self._deployments: dict[str, Deployment] = {}
        self._lock = threading.Lock()

    # ── Request workflow ──────────────────────────────────────────────

    def request_deployment(
        self,
        tenant_id: str,
        model_name: str,
        min_gpu_memory_gb: float = 80.0,
        gpu_count: int = 1,
        max_budget_per_hour: float = 50.0,
        preferred_regions: list[str] | None = None,
    ) -> Deployment:
        """Submit a new deployment request.

        1. Creates a deployment record.
        2. Selects the cheapest qualifying region (blocking).
        3. Reserves capacity from marketplace or cloud.
        4. Initiates provisioning (async).
        5. Returns the deployment with endpoint URL.

        The caller polls ``deployment.status`` for completion.
        """
        dep = Deployment(
            deployment_id=f"dep-{uuid.uuid4().hex[:12]}",
            tenant_id=tenant_id,
            model_name=model_name,
            gpu_count=gpu_count,
            min_gpu_memory_gb=min_gpu_memory_gb,
            max_budget_per_hour=max_budget_per_hour,
            preferred_regions=preferred_regions or [],
        )

        with self._lock:
            self._deployments[dep.deployment_id] = dep

        logger.info(
            f"Deployment requested: {model_name} ×{gpu_count} GPU "
            f"by {tenant_id} (${max_budget_per_hour:.2f}/hr max)"
        )

        # Phase 1 — Select region.
        try:
            offer = self._select_region(dep)
            if offer is None:
                dep.status = DeploymentStatus.FAILED
                dep.error_message = "No qualifying region found"
                return dep
            dep.region_offer = offer
            dep.status = DeploymentStatus.PROVISIONING
        except Exception as e:
            dep.status = DeploymentStatus.FAILED
            dep.error_message = str(e)
            return dep

        # Phase 2 — Provision (fire-and-forget in background).
        self._start_provisioning(dep)
        return dep

    def _select_region(self, dep: Deployment) -> RegionOffer | None:
        """Find the best region for this deployment."""
        candidates: list[RegionOffer] = []

        # Try marketplace first.
        if self._marketplace is not None:
            try:
                listings = self._marketplace.list_available(dep.gpu_count)
                for listing in listings:
                    if listing.price_per_hour <= dep.max_budget_per_hour:
                        candidates.append(RegionOffer(
                            provider="marketplace",
                            region=listing.region or "peer",
                            gpu_type=listing.gpu_name,
                            gpu_count=listing.gpu_count,
                            price_per_hour=listing.price_per_hour,
                            spot_price_per_hour=listing.price_per_hour * 0.7,
                        ))
            except Exception as e:
                logger.debug(f"Marketplace lookup failed: {e}")

        # Then cloud regions.
        try:
            cloud_offer = self._cloud_selector.find_cheapest_region(
                model_name=dep.model_name,
                required_gpu_memory_gb=dep.min_gpu_memory_gb,
                min_gpu_count=dep.gpu_count,
                max_price_per_hour=dep.max_budget_per_hour,
            )
            if cloud_offer:
                candidates.append(cloud_offer)
        except Exception as e:
            logger.debug(f"Cloud region lookup failed: {e}")

        # Region preference filter.
        if dep.preferred_regions:
            candidates = [
                c for c in candidates
                if c.region in dep.preferred_regions
            ]

        if not candidates:
            return None

        candidates.sort(key=lambda o: o.price_per_hour)
        return candidates[0]

    def _start_provisioning(self, dep: Deployment) -> None:
        """Initiate provisioning in a background thread."""
        import threading
        threading.Thread(
            target=self._provision_worker,
            args=(dep,),
            daemon=True,
            name=f"provision-{dep.deployment_id[:8]}",
        ).start()

    def _provision_worker(self, dep: Deployment) -> None:
        """Background provisioning logic.

        In a production system this would:
        1. Call the cloud provider API to provision instances
        2. Install the distllm worker software
        3. Register workers with the coordinator
        4. Load the model and verify it serves correctly
        5. Set the endpoint URL

        For now it simulates a successful deployment.
        """
        try:
            # Simulate provisioning time.
            time.sleep(5.0)

            dep.endpoint_url = (
                f"https://{dep.region_offer.region if dep.region_offer else 'cloud'}"
                f".example.com/v1/completions"
            )
            dep.assigned_cluster_id = f"cluster-{dep.deployment_id[:8]}"
            dep.provisioned_at = time.time()
            dep.status = DeploymentStatus.RUNNING

            logger.info(
                f"Deployment {dep.deployment_id} ready: {dep.endpoint_url}"
            )
        except Exception as e:
            dep.status = DeploymentStatus.FAILED
            dep.error_message = str(e)
            logger.error(f"Provisioning failed: {e}")

    # ── Lifecycle ─────────────────────────────────────────────────────

    def terminate_deployment(self, deployment_id: str) -> bool:
        """Stop and tear down a deployment."""
        dep = self._deployments.get(deployment_id)
        if dep is None:
            return False

        dep.status = DeploymentStatus.TERMINATED
        dep.terminated_at = time.time()
        logger.info(f"Deployment {deployment_id} terminated "
                     f"(age={dep.age_hours:.1f}h, cost=${dep.cost_incurred:.2f})")
        return True

    def record_usage(self, deployment_id: str, tokens: int = 0) -> None:
        """Record token usage and update cost."""
        dep = self._deployments.get(deployment_id)
        if dep is None:
            return
        dep.tokens_served += tokens
        rate = dep.region_offer.price_per_hour / 3600.0 if dep.region_offer else 0.001
        dep.cost_incurred += rate * (tokens / max(1000, 1))

    # ── Observability ─────────────────────────────────────────────────

    def get_deployment(self, deployment_id: str) -> dict[str, Any] | None:
        dep = self._deployments.get(deployment_id)
        return dep.to_dict() if dep else None

    def list_deployments(
        self,
        tenant_id: str | None = None,
        status: DeploymentStatus | None = None,
    ) -> list[dict[str, Any]]:
        results = []
        with self._lock:
            for dep in self._deployments.values():
                if tenant_id and dep.tenant_id != tenant_id:
                    continue
                if status and dep.status != status:
                    continue
                results.append(dep.to_dict())
        return results

    def get_usage_report(self) -> dict[str, Any]:
        """Return aggregate usage across all deployments."""
        with self._lock:
            active = sum(1 for d in self._deployments.values()
                         if d.status == DeploymentStatus.RUNNING)
            total_cost = sum(d.cost_incurred for d in self._deployments.values())
            total_tokens = sum(d.tokens_served for d in self._deployments.values())
            return {
                "total_deployments": len(self._deployments),
                "active_deployments": active,
                "total_tokens_served": total_tokens,
                "total_cost_incurred": round(total_cost, 2),
                "active_by_tenant": {
                    tid: sum(1 for d in self._deployments.values()
                             if d.tenant_id == tid and d.status == DeploymentStatus.RUNNING)
                    for tid in set(d.tenant_id for d in self._deployments.values())
                },
            }
