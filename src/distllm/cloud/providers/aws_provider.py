"""AWS EC2 Spot instance provider.

Uses boto3 for EC2 Spot API operations and the instance metadata
endpoint for interruption detection.
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


class AWSSpotProvider(SpotProvider):
    """AWS EC2 Spot instance provider."""

    @property
    def provider_name(self) -> CloudProvider:
        return CloudProvider.AWS

    def get_spot_price_history(
        self,
        instance_type: str,
        region: str,
        hours: int = 24,
    ) -> list[SpotPrice]:
        try:
            import boto3
        except ImportError:
            raise RuntimeError("AWS spot operations require the 'boto3' package.")

        client = boto3.client("ec2", region_name=region)
        since = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - hours * 3600))
        response = client.describe_spot_price_history(
            InstanceTypes=[instance_type],
            ProductDescriptions=["Linux/UNIX"],
            StartTime=since,
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
        history = self.get_spot_price_history(instance_type, region, hours=1)
        return max(history, key=lambda p: p.timestamp) if history else None

    def request_instance(
        self,
        instance_type: str,
        region: str,
        max_price: float | None = None,
        **kwargs: Any,
    ) -> str:
        logger.info(f"Requesting AWS spot instance: {instance_type} in {region}")
        try:
            import boto3
        except ImportError:
            raise RuntimeError("AWS spot operations require the 'boto3' package.")

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
            missing = [k for k in ("image_id",) if kwargs.get(k) is None]
            if missing:
                raise ValueError(f"AWS spot request missing: {', '.join(missing)}")
            spec: dict[str, Any] = {
                "ImageId": kwargs["image_id"],
                "InstanceType": instance_type,
            }
            for src, dst in (
                ("key_name", "KeyName"), ("subnet_id", "SubnetId"),
                ("iam_instance_profile", "IamInstanceProfile"),
                ("user_data", "UserData"),
            ):
                if kwargs.get(src) is not None:
                    spec[dst] = kwargs[src]
            if kwargs.get("security_group_ids"):
                spec["SecurityGroupIds"] = kwargs["security_group_ids"]
            request["LaunchSpecification"] = spec

        response = client.request_spot_instances(**request)
        requests = response.get("SpotInstanceRequests", [])
        if not requests:
            raise RuntimeError("AWS did not return a SpotInstanceRequestId")
        return requests[0]["SpotInstanceRequestId"]

    def terminate_instance(self, instance_id: str, region: str) -> bool:
        logger.info(f"Terminating AWS spot instance: {instance_id}")
        try:
            import boto3
        except ImportError:
            raise RuntimeError("AWS spot operations require the 'boto3' package.")

        client = boto3.client("ec2", region_name=region)
        if instance_id.startswith("sir-"):
            client.cancel_spot_instance_requests(SpotInstanceRequestIds=[instance_id])
        else:
            client.terminate_instances(InstanceIds=[instance_id])
        return True

    def check_interruption(self, instance_id: str) -> bool:
        return False

    @staticmethod
    def interruption_check_url() -> str:
        return "http://169.254.169.254/latest/meta-data/spot/termination-time"
