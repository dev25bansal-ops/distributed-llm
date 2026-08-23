"""Fixtures for cloud module tests.

Uses the ``load_module`` pattern to bypass ``distllm/__init__.py`` and its
circular import chain.
"""

from __future__ import annotations

import pytest

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

# Load the cloud common module cleanly
_common_mod = load_module("distllm/cloud/common.py")
GPUSpec = _common_mod.GPUSpec
InstanceSpec = _common_mod.InstanceSpec
PriceQuote = _common_mod.PriceQuote
AvailabilityInfo = _common_mod.AvailabilityInfo
PricingFetcher = _common_mod.PricingFetcher
AvailabilityChecker = _common_mod.AvailabilityChecker
ProviderSession = _common_mod.ProviderSession


@pytest.fixture
def sample_gpu_spec() -> GPUSpec:
    return GPUSpec(name="A100", memory_gb=80.0, generation="Ampere", cuda_cores=6912, tensor_cores=432, tdp_watts=400)


@pytest.fixture
def sample_instance_spec(sample_gpu_spec: GPUSpec) -> InstanceSpec:
    return InstanceSpec(
        provider="aws",
        instance_type="p4d.24xlarge",
        region="us-east-1",
        gpu=sample_gpu_spec,
        gpu_count=8,
        vcpus=96,
        ram_gb=1152.0,
        storage_gb=8000.0,
        network_gbps=400.0,
    )


@pytest.fixture
def sample_price_quote() -> PriceQuote:
    return PriceQuote(
        provider="aws",
        instance_type="p4d.24xlarge",
        region="us-east-1",
        on_demand_hourly=32.77,
        spot_hourly=9.83,
        currency="USD",
    )
