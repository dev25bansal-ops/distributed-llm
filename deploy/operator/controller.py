"""Kuberentes operator controller for DistributedLLMCluster.

Watches ``DistributedLLMCluster`` CRs and reconciles the desired state
(coordinator + worker StatefulSets, Services, HPAs) to match the spec.

This controller is designed to be deployed as a Deployment in the
same cluster it manages.  It uses the Kubernetes API via ``lightkube``
or ``pykube-ng`` — no ``kopf`` or ``kubernetes`` SDK dependencies
required (lightkube preferred for minimal size).

Usage:
    # Run locally (dev):
    python deploy/operator/controller.py --dry-run

    # Deployed in-cluster:
    python deploy/operator/controller.py --sync-period 5m
"""

from __future__ import annotations

import argparse
import os
import time
import logging
from typing import Any

logger = logging.getLogger("distllm-operator")


class DistLLMOperator:
    """Reconciles DistributedLLMCluster resources to create/update/delete
    coordinator and worker StatefulSets, Services, and HPAs.

    In a production implementation this would use ``lightkube`` to watch
    CR events.  The current version is a polling-based controller that
    demonstrates the reconciliation loop structure.
    """

    def __init__(
        self,
        sync_period: int = 300,
        max_concurrent: int = 3,
        dry_run: bool = False,
        default_coordinator_image: str = "ghcr.io/distributed-llm/coordinator:0.4.0",
        default_worker_image: str = "ghcr.io/distributed-llm/worker:0.4.0",
    ):
        self.sync_period = sync_period
        self.max_concurrent = max_concurrent
        self.dry_run = dry_run
        self.default_coordinator_image = default_coordinator_image
        self.default_worker_image = default_worker_image
        self._running = False

    def run(self) -> None:
        """Main reconciliation loop."""
        self._running = True
        logger.info(
            "Operator started: sync_period=%ds, max_concurrent=%d, dry_run=%s",
            self.sync_period, self.max_concurrent, self.dry_run,
        )

        while self._running:
            try:
                self._reconcile_all()
            except Exception as e:
                logger.error("Reconciliation failed: %s", e, exc_info=True)
            time.sleep(self.sync_period)

    def stop(self) -> None:
        self._running = False

    def _reconcile_all(self) -> None:
        """Discover all DistributedLLMCluster CRs and reconcile each."""
        clusters = self._list_crs()
        if not clusters:
            logger.debug("No DistributedLLMCluster resources found")
            return

        for cluster in clusters:
            try:
                self._reconcile_one(cluster)
            except Exception as e:
                logger.error(
                    "Failed to reconcile cluster %s: %s",
                    cluster.get("metadata", {}).get("name", "unknown"), e,
                )

    def _list_crs(self) -> list[dict[str, Any]]:
        """List all DistributedLLMCluster CRs.

        Uses the Kubernetes API via ``lightkube`` when deployed in-cluster.
        Falls back to reading from a file for development.
        """
        # Production: use lightkube Client to list CRs.
        #   client = lightkube.Client()
        #   crs = client.list(DistributedLLMCluster)
        #
        # Dev fallback: read from a sample file.
        sample = os.environ.get("DISTLLM_CR_SAMPLE")
        if sample:
            import json
            try:
                return [json.loads(sample)]
            except json.JSONDecodeError:
                pass
        return []

    def _reconcile_one(self, cluster: dict[str, Any]) -> None:
        """Reconcile a single DistributedLLMCluster CR.

        1. Validate the spec.
        2. Ensure the coordinator StatefulSet + Service exists.
        3. Ensure the worker StatefulSet + Service exists per node_pool.
        4. Optionally create/update HPA.
        5. Update the CR status.
        """
        meta = cluster.get("metadata", {})
        name = meta.get("name", "unknown")
        spec = cluster.get("spec", {})
        model = spec.get("model", {})
        coord_spec = spec.get("coordinator", {})
        node_pools = spec.get("node_pools", [])
        hpa_spec = spec.get("hpa", {})

        logger.info(
            "Reconciling cluster %s: model=%s, layers=%d, pools=%d",
            name, model.get("name"), model.get("layers", 0), len(node_pools),
        )

        if self.dry_run:
            logger.info("[DRY-RUN] Would reconcile cluster %s", name)
            return

        # Step 1: Coordinator.
        coord_image = coord_spec.get("image") or self.default_coordinator_image
        self._ensure_coordinator(name, coord_image, coord_spec, hpa_spec)

        # Step 2: Worker pools.
        for pool in node_pools:
            pool_name = pool.get("name", f"pool-{pool['start_layer']}-{pool['end_layer']}")
            pool_image = pool.get("image") or self.default_worker_image
            self._ensure_worker_pool(name, pool_name, pool, pool_image)

        # Step 3: Update CR status.
        self._update_status(name, {
            "ready_nodes": sum(p.get("replicas", 1) for p in node_pools),
            "conditions": [{
                "type": "Ready",
                "status": "True",
                "lastTransitionTime": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }],
        })

    def _ensure_coordinator(
        self, cluster_name: str, image: str,
        coord_spec: dict[str, Any], hpa_spec: dict[str, Any],
    ) -> None:
        """Create or update the coordinator StatefulSet and Service.

        In production this calls the Kubernetes API to create/update
        the resources.  Currently logs the intent.
        """
        logger.info(
            "Ensuring coordinator for %s: image=%s, ports=%s",
            cluster_name, image, coord_spec.get("port", 8000),
        )

    def _ensure_worker_pool(
        self, cluster_name: str, pool_name: str,
        pool: dict[str, Any], image: str,
    ) -> None:
        """Create or update a worker pool StatefulSet and headless Service."""
        logger.info(
            "Ensuring worker pool %s/%s: layers=%d-%d, replicas=%d, image=%s",
            cluster_name, pool_name,
            pool["start_layer"], pool["end_layer"],
            pool.get("replicas", 1), image,
        )

    def _update_status(self, cluster_name: str, status: dict[str, Any]) -> None:
        """Update the CR's status subresource.

        In production this uses the Kubernetes API to PATCH the status.
        """
        logger.info("Updating status for %s: %s", cluster_name, status)


def main() -> None:
    parser = argparse.ArgumentParser(description="DistLLM Kubernetes Operator")
    parser.add_argument("--sync-period", type=int, default=300, help="Reconciliation interval")
    parser.add_argument("--dry-run", action="store_true", help="Log actions without applying")
    parser.add_argument("--default-coordinator-image", default="ghcr.io/distributed-llm/coordinator:0.4.0")
    parser.add_argument("--default-worker-image", default="ghcr.io/distributed-llm/worker:0.4.0")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

    op = DistLLMOperator(
        sync_period=args.sync_period,
        dry_run=args.dry_run,
        default_coordinator_image=args.default_coordinator_image,
        default_worker_image=args.default_worker_image,
    )

    try:
        op.run()
    except KeyboardInterrupt:
        op.stop()
        logger.info("Operator stopped")


if __name__ == "__main__":
    main()
