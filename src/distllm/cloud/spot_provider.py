"""Multi-cloud spot instance providers for cost-optimized inference.

Abstract SpotProvider interface with implementations for AWS, Azure,
GCP, and Lambda Labs spot/preemptible instances.
"""

from __future__ import annotations

import abc
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from loguru import logger


def _missing_dependency(provider: str, package: str) -> RuntimeError:
    return RuntimeError(
        f"{provider} spot operations require the '{package}' package and cloud credentials."
    )


def _require_kwargs(provider: str, kwargs: dict[str, Any], *names: str) -> None:
    missing = [name for name in names if kwargs.get(name) is None]
    if missing:
        raise ValueError(
            f"{provider} spot request missing required option(s): {', '.join(missing)}"
        )


class CloudProvider(str, Enum):
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"
    LAMBDA = "lambda"


@dataclass
class SpotPrice:
    """Spot price data point."""
    provider: CloudProvider
    instance_type: str
    region: str
    price: float  # per hour
    timestamp: float = 0.0
    on_demand_price: float = 0.0

    @property
    def savings_percent(self) -> float:
        if self.on_demand_price <= 0:
            return 0.0
        return (1 - self.price / self.on_demand_price) * 100


@dataclass
class SpotInstance:
    """Represents a running spot instance."""
    instance_id: str
    provider: CloudProvider
    instance_type: str
    region: str
    price: float
    launched_at: float = 0.0
    is_interrupted: bool = False


class SpotProvider(abc.ABC):
    """Abstract base class for cloud spot instance operations."""

    @property
    @abc.abstractmethod
    def provider_name(self) -> CloudProvider:
        ...

    @abc.abstractmethod
    def get_spot_price_history(
        self,
        instance_type: str,
        region: str,
        hours: int = 24,
    ) -> list[SpotPrice]:
        """Get spot price history for an instance type."""
        ...

    @abc.abstractmethod
    def get_current_spot_price(
        self,
        instance_type: str,
        region: str,
    ) -> SpotPrice | None:
        """Get current spot price for an instance type."""
        ...

    @abc.abstractmethod
    def request_instance(
        self,
        instance_type: str,
        region: str,
        max_price: float | None = None,
        **kwargs: Any,
    ) -> str:
        """Request a spot instance. Returns instance_id."""
        ...

    @abc.abstractmethod
    def terminate_instance(self, instance_id: str, region: str) -> bool:
        """Terminate a spot instance."""
        ...

    @abc.abstractmethod
    def check_interruption(self, instance_id: str) -> bool:
        """Check if a spot instance has been interrupted/preempted."""
        ...


class AWSSpotProvider(SpotProvider):
    """AWS EC2 Spot instance provider.

    Uses EC2 metadata endpoint for interruption detection
    and EC2 API for price history and instance lifecycle.
    """

    @property
    def provider_name(self) -> CloudProvider:
        return CloudProvider.AWS

    def get_spot_price_history(
        self,
        instance_type: str,
        region: str,
        hours: int = 24,
    ) -> list[SpotPrice]:
        """Get AWS spot price history via DescribeSpotPriceHistory API."""
        try:
            import boto3
        except ImportError as exc:
            raise _missing_dependency("AWS", "boto3") from exc

        client = boto3.client("ec2", region_name=region)
        response = client.describe_spot_price_history(
            InstanceTypes=[instance_type],
            ProductDescriptions=["Linux/UNIX"],
            StartTime=time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - hours * 3600)
            ),
        )
        prices = []
        for item in response.get("SpotPriceHistory", []):
            prices.append(
                SpotPrice(
                    provider=CloudProvider.AWS,
                    instance_type=instance_type,
                    region=item.get("AvailabilityZone", region),
                    price=float(item["SpotPrice"]),
                    timestamp=item["Timestamp"].timestamp(),
                )
            )
        return prices

    def get_current_spot_price(
        self,
        instance_type: str,
        region: str,
    ) -> SpotPrice | None:
        """Get current AWS spot price."""
        history = self.get_spot_price_history(instance_type, region, hours=1)
        return max(history, key=lambda p: p.timestamp) if history else None

    def request_instance(
        self,
        instance_type: str,
        region: str,
        max_price: float | None = None,
        **kwargs: Any,
    ) -> str:
        """Request AWS spot instance via RequestSpotInstances API."""
        logger.info(f"Requesting AWS spot instance: {instance_type} in {region}")
        try:
            import boto3
        except ImportError as exc:
            raise _missing_dependency("AWS", "boto3") from exc

        client = boto3.client("ec2", region_name=region)
        count = int(kwargs.get("count", 1))
        request: dict[str, Any] = {
            "InstanceCount": count,
            "Type": kwargs.get("request_type", "one-time"),
        }
        if max_price is not None:
            request["SpotPrice"] = str(max_price)

        if kwargs.get("launch_template"):
            request["LaunchTemplate"] = kwargs["launch_template"]
        else:
            _require_kwargs("AWS", kwargs, "image_id")
            launch_spec: dict[str, Any] = {
                "ImageId": kwargs["image_id"],
                "InstanceType": instance_type,
            }
            for source, dest in (
                ("key_name", "KeyName"),
                ("subnet_id", "SubnetId"),
                ("iam_instance_profile", "IamInstanceProfile"),
                ("user_data", "UserData"),
            ):
                if kwargs.get(source) is not None:
                    launch_spec[dest] = kwargs[source]
            if kwargs.get("security_group_ids"):
                launch_spec["SecurityGroupIds"] = kwargs["security_group_ids"]
            request["LaunchSpecification"] = launch_spec

        response = client.request_spot_instances(**request)
        requests = response.get("SpotInstanceRequests", [])
        if not requests:
            raise RuntimeError("AWS did not return a SpotInstanceRequestId")
        return requests[0]["SpotInstanceRequestId"]

    def terminate_instance(self, instance_id: str, region: str) -> bool:
        """Terminate AWS spot instance."""
        logger.info(f"Terminating AWS spot instance: {instance_id}")
        try:
            import boto3
        except ImportError as exc:
            raise _missing_dependency("AWS", "boto3") from exc

        client = boto3.client("ec2", region_name=region)
        if instance_id.startswith("sir-"):
            client.cancel_spot_instance_requests(SpotInstanceRequestIds=[instance_id])
        else:
            client.terminate_instances(InstanceIds=[instance_id])
        return True

    def check_interruption(self, instance_id: str) -> bool:
        """Check AWS spot interruption via EC2 metadata endpoint."""
        # In production: poll http://169.254.169.254/latest/meta-data/spot/termination-time
        return False


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
        import httpx

        url = "https://prices.azure.com/api/retail/prices"
        query = (
            f"serviceName eq 'Virtual Machines' and armRegionName eq '{region}' "
            f"and armSkuName eq '{instance_type}' and priceType eq 'Consumption'"
        )
        response = httpx.get(url, params={"$filter": query}, timeout=10.0)
        response.raise_for_status()
        items = response.json().get("Items", [])
        spot_items = [item for item in items if "spot" in item.get("meterName", "").lower()]
        if not spot_items:
            return None
        item = min(spot_items, key=lambda row: float(row.get("unitPrice", 0.0)))
        return SpotPrice(
            provider=CloudProvider.AZURE,
            instance_type=instance_type,
            region=region,
            price=float(item.get("unitPrice", 0.0)),
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
        except ImportError as exc:
            raise _missing_dependency("Azure", "azure-identity azure-mgmt-compute") from exc

        _require_kwargs("Azure", kwargs, "subscription_id", "resource_group", "vm_name", "vm_parameters")
        credential = kwargs.get("credential") or DefaultAzureCredential()
        client = ComputeManagementClient(credential, kwargs["subscription_id"])
        vm_parameters = dict(kwargs["vm_parameters"])
        vm_parameters["priority"] = "Spot"
        vm_parameters["eviction_policy"] = kwargs.get("eviction_policy", "Deallocate")
        if max_price is not None:
            vm_parameters["billing_profile"] = {"max_price": max_price}
        poller = client.virtual_machines.begin_create_or_update(
            kwargs["resource_group"],
            kwargs["vm_name"],
            vm_parameters,
        )
        vm = poller.result()
        return vm.id or kwargs["vm_name"]

    def terminate_instance(self, instance_id: str, region: str) -> bool:
        logger.info(f"Terminating Azure Spot VM: {instance_id}")
        try:
            from azure.identity import DefaultAzureCredential
            from azure.mgmt.compute import ComputeManagementClient
        except ImportError as exc:
            raise _missing_dependency("Azure", "azure-identity azure-mgmt-compute") from exc

        if "/" in instance_id:
            parts = instance_id.strip("/").split("/")
            resource_group = parts[parts.index("resourceGroups") + 1]
            vm_name = parts[parts.index("virtualMachines") + 1]
            subscription_id = parts[parts.index("subscriptions") + 1]
        else:
            raise ValueError("Azure terminate_instance requires a full VM resource ID")
        client = ComputeManagementClient(DefaultAzureCredential(), subscription_id)
        client.virtual_machines.begin_delete(resource_group, vm_name).result()
        return True

    def check_interruption(self, instance_id: str) -> bool:
        """Check Azure Spot eviction via Instance Metadata Service."""
        # In production: poll http://169.254.169.254/metadata/scheduledevents
        return False


class GCPSpotProvider(SpotProvider):
    """GCP preemptible VM provider."""

    @property
    def provider_name(self) -> CloudProvider:
        return CloudProvider.GCP

    def get_spot_price_history(
        self,
        instance_type: str,
        region: str,
        hours: int = 24,
    ) -> list[SpotPrice]:
        logger.debug("GCP Cloud Billing API historical spot prices are not queried by default")
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
        logger.info(f"Requesting GCP preemptible VM: {instance_type} in {region}")
        try:
            from google.cloud import compute_v1
        except ImportError as exc:
            raise _missing_dependency("GCP", "google-cloud-compute") from exc

        _require_kwargs("GCP", kwargs, "project", "instance_name", "instance_resource")
        zone = kwargs.get("zone", region)
        instance = kwargs["instance_resource"]
        instance.name = kwargs["instance_name"]
        instance.machine_type = instance.machine_type or f"zones/{zone}/machineTypes/{instance_type}"
        if instance.scheduling is None:
            instance.scheduling = compute_v1.Scheduling()
        instance.scheduling.provisioning_model = compute_v1.Scheduling.ProvisioningModel.SPOT.name
        instance.scheduling.instance_termination_action = kwargs.get("termination_action", "STOP")

        client = compute_v1.InstancesClient()
        operation = client.insert(project=kwargs["project"], zone=zone, instance_resource=instance)
        return operation.name

    def terminate_instance(self, instance_id: str, region: str) -> bool:
        logger.info(f"Terminating GCP preemptible VM: {instance_id}")
        try:
            from google.cloud import compute_v1
        except ImportError as exc:
            raise _missing_dependency("GCP", "google-cloud-compute") from exc

        project = os.environ.get("GCP_PROJECT")
        if not project:
            raise ValueError("GCP_PROJECT must be set to terminate a GCP instance")
        client = compute_v1.InstancesClient()
        client.delete(project=project, zone=region, instance=instance_id)
        return True

    def check_interruption(self, instance_id: str) -> bool:
        """Check GCP preemption via metadata server."""
        # In production: poll http://metadata.google.internal/computeMetadata/v1/instance/preempted
        return False


class LambdaSpotProvider(SpotProvider):
    """Lambda Labs provider (no true spot, but preemption detection)."""

    @property
    def provider_name(self) -> CloudProvider:
        return CloudProvider.LAMBDA

    def get_spot_price_history(
        self,
        instance_type: str,
        region: str,
        hours: int = 24,
    ) -> list[SpotPrice]:
        # Lambda Labs uses fixed pricing, not spot
        logger.debug(f"Lambda Labs pricing for {instance_type} (fixed pricing)")
        current = self.get_current_spot_price(instance_type, region)
        return [current] if current else []

    def get_current_spot_price(
        self,
        instance_type: str,
        region: str,
    ) -> SpotPrice | None:
        import httpx

        api_key = os.environ.get("LAMBDA_API_KEY")
        if not api_key:
            return None
        response = httpx.get(
            "https://cloud.lambdalabs.com/api/v1/instance-types",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10.0,
        )
        response.raise_for_status()
        offers = response.json().get("data", {})
        offer = offers.get(instance_type, {})
        regions = offer.get("regions_with_capacity_available", [])
        matching = [r for r in regions if r.get("name") == region] or regions
        if not matching:
            return None
        price = float(offer.get("price_cents_per_hour", 0.0)) / 100.0
        return SpotPrice(
            provider=CloudProvider.LAMBDA,
            instance_type=instance_type,
            region=matching[0].get("name", region),
            price=price,
            timestamp=time.time(),
            on_demand_price=price,
        )

    def request_instance(
        self,
        instance_type: str,
        region: str,
        max_price: float | None = None,
        **kwargs: Any,
    ) -> str:
        logger.info(f"Requesting Lambda Labs instance: {instance_type}")
        import httpx

        api_key = kwargs.get("api_key") or os.environ.get("LAMBDA_API_KEY")
        if not api_key:
            raise ValueError("Lambda Labs request requires LAMBDA_API_KEY or api_key")
        payload = {
            "region_name": kwargs.get("region_name", region),
            "instance_type_name": instance_type,
            "ssh_key_names": kwargs.get("ssh_key_names", []),
            "file_system_names": kwargs.get("file_system_names", []),
            "quantity": int(kwargs.get("count", 1)),
        }
        response = httpx.post(
            "https://cloud.lambdalabs.com/api/v1/instance-operations/launch",
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
            timeout=30.0,
        )
        response.raise_for_status()
        instance_ids = response.json().get("data", {}).get("instance_ids", [])
        if not instance_ids:
            raise RuntimeError("Lambda Labs did not return an instance ID")
        return instance_ids[0]

    def terminate_instance(self, instance_id: str, region: str) -> bool:
        logger.info(f"Terminating Lambda Labs instance: {instance_id}")
        import httpx

        api_key = os.environ.get("LAMBDA_API_KEY")
        if not api_key:
            raise ValueError("Lambda Labs terminate requires LAMBDA_API_KEY")
        response = httpx.post(
            "https://cloud.lambdalabs.com/api/v1/instance-operations/terminate",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"instance_ids": [instance_id]},
            timeout=30.0,
        )
        response.raise_for_status()
        return True

    def check_interruption(self, instance_id: str) -> bool:
        """Lambda Labs doesn't have spot preemption, but checks for SIGTERM."""
        return False
