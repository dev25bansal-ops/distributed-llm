"""Tests: Cloud spot providers — AWS, Azure, GCP, Lambda.

Covers get_spot_price_history, request_instance, terminate, check_interruption,
error handling for missing credentials/params, and missing dependencies.

Run: pytest tests/test_cloud_providers.py -v
"""

import sys
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from distllm.cloud.spot_provider import (
    AWSSpotProvider,
    AzureSpotProvider,
    GCPSpotProvider,
    LambdaSpotProvider,
    CloudProvider,
    SpotPrice,
    SpotInstance,
)


def _mock_import(name: str, mock_obj=None):
    """Patch sys.modules so function-level import statements find the mock."""
    m = mock_obj or MagicMock()
    return patch.dict("sys.modules", {name: m, **{name: m}})


# ===========================================================================
# AWS
# ===========================================================================


class TestAWSSpotProvider:
    def test_provider_name(self):
        assert AWSSpotProvider().provider_name == CloudProvider.AWS

    def test_get_spot_price_history(self):
        mock_boto3 = MagicMock()
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_client.describe_spot_price_history.return_value = {
            "SpotPriceHistory": [
                {"InstanceType": "p3.2xlarge", "AvailabilityZone": "us-east-1a",
                 "SpotPrice": "0.50", "Timestamp": MagicMock(timestamp=lambda: 1000)},
                {"InstanceType": "p3.2xlarge", "AvailabilityZone": "us-east-1b",
                 "SpotPrice": "0.55", "Timestamp": MagicMock(timestamp=lambda: 2000)},
            ]
        }
        with _mock_import("boto3", mock_boto3):
            provider = AWSSpotProvider()
            prices = provider.get_spot_price_history("p3.2xlarge", "us-east-1", hours=24)
        assert len(prices) == 2
        assert prices[0].price == 0.50
        assert prices[0].provider == CloudProvider.AWS

    def test_get_current_spot_price(self):
        mock_boto3 = MagicMock()
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_client.describe_spot_price_history.return_value = {
            "SpotPriceHistory": [
                {"InstanceType": "p3.2xlarge", "AvailabilityZone": "us-east-1a",
                 "SpotPrice": "0.50", "Timestamp": MagicMock(timestamp=lambda: 3000)},
            ]
        }
        with _mock_import("boto3", mock_boto3):
            provider = AWSSpotProvider()
            price = provider.get_current_spot_price("p3.2xlarge", "us-east-1")
        assert price is not None
        assert price.price == 0.50

    def test_request_instance_with_launch_template(self):
        mock_boto3 = MagicMock()
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_client.request_spot_instances.return_value = {
            "SpotInstanceRequests": [{"SpotInstanceRequestId": "sir-abc123"}]
        }
        with _mock_import("boto3", mock_boto3):
            provider = AWSSpotProvider()
            instance_id = provider.request_instance(
                "p3.2xlarge", "us-east-1", launch_template={"LaunchTemplateName": "my-template"}
            )
        assert instance_id == "sir-abc123"

    def test_request_instance_with_full_spec(self):
        mock_boto3 = MagicMock()
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_client.request_spot_instances.return_value = {
            "SpotInstanceRequests": [{"SpotInstanceRequestId": "sir-def456"}]
        }
        with _mock_import("boto3", mock_boto3):
            provider = AWSSpotProvider()
            instance_id = provider.request_instance(
                "p3.2xlarge", "us-east-1", image_id="ami-12345",
                key_name="my-key", subnet_id="subnet-abc"
            )
        assert instance_id == "sir-def456"

    def test_request_instance_missing_image_id_raises(self):
        mock_boto3 = MagicMock()
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        with _mock_import("boto3", mock_boto3):
            provider = AWSSpotProvider()
            with pytest.raises(ValueError):
                provider.request_instance("p3.2xlarge", "us-east-1")

    def test_terminate_spot_request_sir_prefix(self):
        mock_boto3 = MagicMock()
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        with _mock_import("boto3", mock_boto3):
            provider = AWSSpotProvider()
            result = provider.terminate_instance("sir-xyz789", "us-east-1")
        assert result is True
        mock_client.cancel_spot_instance_requests.assert_called_once()

    def test_terminate_instance_regular_id(self):
        mock_boto3 = MagicMock()
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        with _mock_import("boto3", mock_boto3):
            provider = AWSSpotProvider()
            result = provider.terminate_instance("i-0abcd1234", "us-east-1")
        assert result is True
        mock_client.terminate_instances.assert_called_once()

    def test_check_interruption_returns_false(self):
        provider = AWSSpotProvider()
        assert provider.check_interruption("any-id") is False

    def test_missing_image_id_raises_value_error(self):
        mock_boto3 = MagicMock()
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        import types
        # Skip the import check by making the function think boto3 is available
        mock_sys = {k: v for k, v in sys.modules.items()}
        mock_sys["boto3"] = mock_boto3
        with _mock_import("boto3", mock_boto3):
            provider = AWSSpotProvider()
            with pytest.raises(ValueError):
                provider.request_instance("p3.2xlarge", "us-east-1")


# ===========================================================================
# Azure
# ===========================================================================


class TestAzureSpotProvider:
    def test_provider_name(self):
        assert AzureSpotProvider().provider_name == CloudProvider.AZURE

    def test_get_current_spot_price(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "Items": [
                {"armSkuName": "Standard_D2s_v3", "armRegionName": "eastus",
                 "meterName": "D2s v3 Spot", "unitPrice": 0.05},
            ]
        }
        mock_httpx = MagicMock()
        mock_httpx.get.return_value = mock_response
        with _mock_import("httpx", mock_httpx):
            provider = AzureSpotProvider()
            price = provider.get_current_spot_price("Standard_D2s_v3", "eastus")
        assert price is not None
        assert price.price == 0.05
        assert price.provider == CloudProvider.AZURE

    def test_get_current_spot_price_returns_none_on_empty(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"Items": []}
        mock_httpx = MagicMock()
        mock_httpx.get.return_value = mock_response
        with _mock_import("httpx", mock_httpx):
            provider = AzureSpotProvider()
            price = provider.get_current_spot_price("nonexistent", "eastus")
        assert price is None

    def test_request_instance(self):
        mock_vm = MagicMock()
        mock_vm.id = "/subscriptions/sub/resourceGroups/rg/providers/Microsoft.Compute/virtualMachines/myvm"
        mock_client = MagicMock()
        mock_client.virtual_machines.begin_create_or_update.return_value.result.return_value = mock_vm
        mock_cred = MagicMock()
        mock_compute_cls = MagicMock(return_value=mock_client)
        import types
        azure_mod = types.ModuleType("azure.mgmt.compute")
        azure_mod.ComputeManagementClient = mock_compute_cls
        azure_id = types.ModuleType("azure.identity")
        azure_id.DefaultAzureCredential = lambda: mock_cred
        with _mock_import("azure.identity", azure_id), \
             _mock_import("azure.mgmt.compute", azure_mod):
            provider = AzureSpotProvider()
            instance_id = provider.request_instance(
                "Standard_D2s_v3", "eastus",
                subscription_id="sub-1", resource_group="rg-1",
                vm_name="myvm", vm_parameters={}
            )
        assert instance_id is not None

    def test_request_instance_missing_kwargs_raises(self):
        import types
        azure_id = types.ModuleType("azure.identity")
        azure_id.DefaultAzureCredential = MagicMock()
        azure_mod = types.ModuleType("azure.mgmt.compute")
        azure_mod.ComputeManagementClient = MagicMock()
        with _mock_import("azure.identity", azure_id), \
             _mock_import("azure.mgmt.compute", azure_mod):
            provider = AzureSpotProvider()
            with pytest.raises(ValueError):
                provider.request_instance("Standard_D2s_v3", "eastus")

    def test_terminate_with_resource_id(self):
        mock_client = MagicMock()
        mock_cred = MagicMock()
        mock_compute_cls = MagicMock(return_value=mock_client)
        import types
        azure_mod = types.ModuleType("azure.mgmt.compute")
        azure_mod.ComputeManagementClient = mock_compute_cls
        azure_id = types.ModuleType("azure.identity")
        azure_id.DefaultAzureCredential = lambda: mock_cred
        with _mock_import("azure.identity", azure_id), \
             _mock_import("azure.mgmt.compute", azure_mod):
            provider = AzureSpotProvider()
            resource_id = "/subscriptions/sub/resourceGroups/rg/providers/Microsoft.Compute/virtualMachines/myvm"
            result = provider.terminate_instance(resource_id, "eastus")
        assert result is True

    def test_terminate_bad_id_raises(self):
        mock_client = MagicMock()
        mock_cred = MagicMock()
        mock_compute_cls = MagicMock(return_value=mock_client)
        import types
        azure_mod = types.ModuleType("azure.mgmt.compute")
        azure_mod.ComputeManagementClient = mock_compute_cls
        azure_id = types.ModuleType("azure.identity")
        azure_id.DefaultAzureCredential = lambda: mock_cred
        with _mock_import("azure.identity", azure_id), \
             _mock_import("azure.mgmt.compute", azure_mod):
            provider = AzureSpotProvider()
            with pytest.raises(ValueError):
                provider.terminate_instance("bad-id-no-slash", "eastus")

    def test_check_interruption_returns_false(self):
        provider = AzureSpotProvider()
        assert provider.check_interruption("any-id") is False


# ===========================================================================
# GCP
# ===========================================================================


class TestGCPSpotProvider:
    def test_provider_name(self):
        assert GCPSpotProvider().provider_name == CloudProvider.GCP

    def test_get_current_spot_price_returns_none(self):
        provider = GCPSpotProvider()
        price = provider.get_current_spot_price("n1-standard-4", "us-central1")
        assert price is None

    def test_get_spot_price_history_returns_fallback(self):
        provider = GCPSpotProvider()
        prices = provider.get_spot_price_history("n1-standard-4", "us-central1")
        assert isinstance(prices, list)

    def test_request_instance(self):
        mock_client = MagicMock()
        mock_client.insert.return_value = MagicMock()
        import types
        compute_mod = types.ModuleType("google.cloud.compute_v1")
        compute_mod.InstancesClient = MagicMock(return_value=mock_client)
        # Create mock for Scheduling.ProvisioningModel.SPOT
        mock_scheduling = MagicMock()
        mock_provisioning_model = MagicMock()
        mock_provisioning_model.SPOT = MagicMock()
        mock_provisioning_model.SPOT.name = "SPOT"
        mock_scheduling.ProvisioningModel = mock_provisioning_model
        compute_mod.Scheduling = mock_scheduling
        with _mock_import("google.cloud.compute_v1", compute_mod):
            provider = GCPSpotProvider()
            instance_id = provider.request_instance(
                "n1-standard-4", "us-central1",
                project="my-project", instance_name="my-instance",
                instance_resource=MagicMock()
            )
        assert instance_id is not None

    def test_request_instance_missing_kwargs_raises(self):
        import types
        compute_mod = types.ModuleType("google.cloud.compute_v1")
        compute_mod.InstancesClient = MagicMock()
        with _mock_import("google.cloud.compute_v1", compute_mod):
            provider = GCPSpotProvider()
            with pytest.raises(ValueError):
                provider.request_instance("n1-standard-4", "us-central1")

    def test_terminate(self):
        mock_client = MagicMock()
        import types
        compute_mod = types.ModuleType("google.cloud.compute_v1")
        compute_mod.InstancesClient = MagicMock(return_value=mock_client)
        with _mock_import("google.cloud.compute_v1", compute_mod), \
             patch.dict("os.environ", {"GCP_PROJECT": "my-project"}):
            provider = GCPSpotProvider()
            result = provider.terminate_instance("my-instance", "us-central1")
        assert result is True

    def test_terminate_no_project_raises(self):
        import types
        compute_mod = types.ModuleType("google.cloud.compute_v1")
        compute_mod.InstancesClient = MagicMock()
        with _mock_import("google.cloud.compute_v1", compute_mod), \
             patch.dict("os.environ", {}, clear=True):
            provider = GCPSpotProvider()
            with pytest.raises(ValueError):
                provider.terminate_instance("my-instance", "us-central1")

    def test_inst_to_resource_group(self):
        result = GCPSpotProvider._inst_to_resource_group("n1-standard-4")
        assert result == "N1Standard"

    def test_check_interruption_returns_false(self):
        provider = GCPSpotProvider()
        assert provider.check_interruption("any-id") is False

    def test_request_instance_missing_dependency(self):
        provider = GCPSpotProvider()
        with pytest.raises(RuntimeError):
            provider.request_instance("n1-standard-4", "us-central1",
                                      project="p", instance_name="n", instance_resource=MagicMock())


# ===========================================================================
# Lambda
# ===========================================================================


class TestLambdaSpotProvider:
    def test_provider_name(self):
        assert LambdaSpotProvider().provider_name == CloudProvider.LAMBDA

    def test_get_current_spot_price_with_key(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": {
                "gpu_1x_a100": {
                    "price_cents_per_hour": 110,
                    "regions_with_capacity_available": [{"name": "us-east-1"}]
                }
            }
        }
        mock_httpx = MagicMock()
        mock_httpx.get.return_value = mock_response
        with _mock_import("httpx", mock_httpx), \
             patch.dict("os.environ", {"LAMBDA_API_KEY": "test-key"}):
            provider = LambdaSpotProvider()
            price = provider.get_current_spot_price("gpu_1x_a100", "us-east-1")
        assert price is not None
        assert price.price == 1.10
        assert price.provider == CloudProvider.LAMBDA

    def test_get_current_spot_price_no_key_returns_none(self):
        with patch.dict("os.environ", {}, clear=True):
            provider = LambdaSpotProvider()
            price = provider.get_current_spot_price("gpu_1x_a100", "us-east-1")
        assert price is None

    def test_request_instance(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        # response.json().get("data", {}).get("instance_ids", [])
        mock_response.json.return_value = {"data": {"instance_ids": ["lambda-inst-001"]}}
        mock_httpx = MagicMock()
        mock_httpx.post.return_value = mock_response
        with _mock_import("httpx", mock_httpx), \
             patch.dict("os.environ", {"LAMBDA_API_KEY": "test-key"}):
            provider = LambdaSpotProvider()
            instance_id = provider.request_instance("gpu_1x_a100", "us-east-1", api_key="test-key")
        assert instance_id == "lambda-inst-001"

    def test_request_instance_missing_api_key_raises(self):
        provider = LambdaSpotProvider()
        with pytest.raises(ValueError):
            provider.request_instance("gpu_1x_a100", "us-east-1")

    def test_terminate(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_httpx = MagicMock()
        mock_httpx.post.return_value = mock_response
        with _mock_import("httpx", mock_httpx), \
             patch.dict("os.environ", {"LAMBDA_API_KEY": "test-key"}):
            provider = LambdaSpotProvider()
            result = provider.terminate_instance("lambda-inst-001", "us-east-1")
        assert result is True

    def test_check_interruption_returns_false(self):
        provider = LambdaSpotProvider()
        assert provider.check_interruption("any-id") is False


# ===========================================================================
# SpotPrice and SpotInstance dataclasses
# ===========================================================================


class TestSpotPrice:
    def test_savings_percent(self):
        sp = SpotPrice(provider=CloudProvider.AWS, instance_type="p3.2xlarge",
                       region="us-east-1", price=0.40, on_demand_price=1.00)
        assert sp.savings_percent == 60.0

    def test_zero_on_demand_price(self):
        sp = SpotPrice(provider=CloudProvider.AWS, instance_type="t2.micro",
                       region="us-east-1", price=0.01)
        assert sp.savings_percent == 0.0


class TestSpotInstance:
    def test_defaults(self):
        si = SpotInstance(instance_id="i-123", provider=CloudProvider.AWS,
                          instance_type="t2.micro", region="us-east-1", price=0.01)
        assert si.is_interrupted is False
        assert si.launched_at == 0.0
