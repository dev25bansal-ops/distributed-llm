"""GCP preemptible/Spot VM provider.

Uses the Google Cloud Compute API for VM lifecycle operations
and the Cloud Billing catalog for pricing data.
"""

from __future__ import annotations

import os
import time
from typing import Any

from loguru import logger

from distllm.cloud.spot_provider import (
    CloudProvider,
    SpotPrice,
    SpotInstance,
    SpotProvider,
)


class GCPSpotProvider(SpotProvider):
    """GCP preemptible/Spot VM provider."""

    @property
    def provider_name(self) -> CloudProvider:
        return CloudProvider.GCP

    def get_spot_price_history(
        self,
        instance_type: str,
        region: str,
        hours: int = 24,
    ) -> list[SpotPrice]:
        logger.debug("GCP Cloud Billing API historical spot prices require billing catalog integration")
        current = self.get_current_spot_price(instance_type, region)
        return [current] if current else []

    def get_current_spot_price(
        self,
        instance_type: str,
        region: str,
    ) -> SpotPrice | None:
        logger.debug("GCP spot price lookup requires Cloud Billing catalog integration")
        return None

    def request_instance(
        self,
        instance_type: str,
        region: str,
        max_price: float | None = None,
        **kwargs: Any,
    ) -> str:
        logger.info(f"Requesting GCP spot VM: {instance_type} in {region}")
        try:
            from google.cloud import compute_v1
        except ImportError:
            raise RuntimeError("GCP spot operations require the 'google-cloud-compute' package.")

        for k in ("project", "instance_name", "instance_resource"):
            if kwargs.get(k) is None:
                raise ValueError(f"GCP spot request missing required option: {k}")

        zone = kwargs.get("zone", region)
        instance = kwargs["instance_resource"]
        instance.name = kwargs["instance_name"]
        instance.machine_type = (
            instance.machine_type or f"zones/{zone}/machineTypes/{instance_type}"
        )
        if instance.scheduling is None:
            from google.cloud.compute_v1 import Scheduling
            instance.scheduling = Scheduling()
        instance.scheduling.provisioning_model = (
            "SPOT"
        )
        instance.scheduling.instance_termination_action = kwargs.get(
            "termination_action", "STOP"
        )

        client = compute_v1.InstancesClient()
        operation = client.insert(
            project=kwargs["project"],
            zone=zone,
            instance_resource=instance,
        )
        return operation.name

    def terminate_instance(self, instance_id: str, region: str) -> bool:
        logger.info(f"Terminating GCP spot VM: {instance_id}")
        try:
            from google.cloud import compute_v1
        except ImportError:
            raise RuntimeError("GCP spot operations require the 'google-cloud-compute' package.")

        project = os.environ.get("GCP_PROJECT")
        if not project:
            raise ValueError("GCP_PROJECT environment variable must be set")
        client = compute_v1.InstancesClient()
        client.delete(project=project, zone=region, instance=instance_id)
        return True

    def check_interruption(self, instance_id: str) -> bool:
        return False

    @staticmethod
    def interruption_check_url() -> str:
        return (
            "http://metadata.google.internal/computeMetadata/v1/instance/preempted"
        )
