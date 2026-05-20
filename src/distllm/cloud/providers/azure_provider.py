"""Azure Spot VM provider.

Uses the Azure Retail Prices API for cost data and the Azure SDK
for VM lifecycle operations.
"""

from __future__ import annotations

import os
import time
from typing import Any

import httpx
from loguru import logger

from distllm.cloud.spot_provider import (
    CloudProvider,
    SpotPrice,
    SpotInstance,
    SpotProvider,
)


class AzureSpotProvider(SpotProvider):
    """Azure Spot VM provider."""

    @property
    def provider_name(self) -> CloudProvider:
        return CloudProvider.AZURE

    def get_spot_price_history(
        self,
        instance_type: str,
        region: str,
        hours: int = 24,
    ) -> list[SpotPrice]:
        logger.debug("Azure Retail Prices API does not expose historical spot prices directly")
        current = self.get_current_spot_price(instance_type, region)
        return [current] if current else []

    def get_current_spot_price(
        self,
        instance_type: str,
        region: str,
    ) -> SpotPrice | None:
        url = "https://prices.azure.com/api/retail/prices"
        query = (
            f"serviceName eq 'Virtual Machines' and armRegionName eq '{region}' "
            f"and armSkuName eq '{instance_type}' and priceType eq 'Consumption'"
        )
        response = httpx.get(url, params={"$filter": query}, timeout=10.0)
        response.raise_for_status()
        items = response.json().get("Items", [])
        spot = [i for i in items if "spot" in i.get("meterName", "").lower()]
        if not spot:
            return None
        cheapest = min(spot, key=lambda i: float(i.get("unitPrice", 0.0)))
        return SpotPrice(
            provider=CloudProvider.AZURE,
            instance_type=instance_type,
            region=region,
            price=float(cheapest.get("unitPrice", 0.0)),
            timestamp=time.time(),
        )

    def request_instance(
        self,
        instance_type: str,
        region: str,
        max_price: float | None = None,
        **kwargs: Any,
    ) -> str:
        logger.info(f"Requesting Azure Spot VM: {instance_type} in {region}")
        try:
            from azure.identity import DefaultAzureCredential
            from azure.mgmt.compute import ComputeManagementClient
        except ImportError:
            raise RuntimeError(
                "Azure spot operations require 'azure-identity' and 'azure-mgmt-compute'"
            )

        missing = [k for k in ("subscription_id", "resource_group", "vm_name", "vm_parameters") if kwargs.get(k) is None]
        if missing:
            raise ValueError(f"Azure spot request missing: {', '.join(missing)}")

        credential = kwargs.get("credential") or DefaultAzureCredential()
        client = ComputeManagementClient(credential, kwargs["subscription_id"])
        vm_params = dict(kwargs["vm_parameters"])
        vm_params["priority"] = "Spot"
        vm_params["eviction_policy"] = kwargs.get("eviction_policy", "Deallocate")
        if max_price is not None:
            vm_params["billing_profile"] = {"max_price": max_price}
        poller = client.virtual_machines.begin_create_or_update(
            kwargs["resource_group"],
            kwargs["vm_name"],
            vm_params,
        )
        vm = poller.result()
        return vm.id or kwargs["vm_name"]

    def terminate_instance(self, instance_id: str, region: str) -> bool:
        logger.info(f"Terminating Azure Spot VM: {instance_id}")
        try:
            from azure.identity import DefaultAzureCredential
            from azure.mgmt.compute import ComputeManagementClient
        except ImportError:
            raise RuntimeError(
                "Azure spot operations require 'azure-identity' and 'azure-mgmt-compute'"
            )

        if "/" not in instance_id:
            raise ValueError("Azure terminate_instance requires a full VM resource ID")
        parts = instance_id.strip("/").split("/")
        resource_group = parts[parts.index("resourceGroups") + 1]
        vm_name = parts[parts.index("virtualMachines") + 1]
        subscription_id = parts[parts.index("subscriptions") + 1]
        client = ComputeManagementClient(
            DefaultAzureCredential(),
            subscription_id,
        )
        client.virtual_machines.begin_delete(resource_group, vm_name).result()
        return True

    def check_interruption(self, instance_id: str) -> bool:
        return False

    @staticmethod
    def interruption_check_url() -> str:
        return "http://169.254.169.254/metadata/scheduledevents"
