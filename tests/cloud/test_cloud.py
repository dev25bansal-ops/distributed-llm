"""Tests for ``distllm.cloud.common`` — dataclasses, ABCs, and ProviderSession.

All tests use the ``load_module`` pattern from ``tests._import_helper`` to
bypass ``distllm/__init__.py`` and its circular-import chain.
"""

from __future__ import annotations

import pytest

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_common_mod = load_module("distllm/cloud/common.py")
GPUSpec = _common_mod.GPUSpec
InstanceSpec = _common_mod.InstanceSpec
PriceQuote = _common_mod.PriceQuote
AvailabilityInfo = _common_mod.AvailabilityInfo
PricingFetcher = _common_mod.PricingFetcher
AvailabilityChecker = _common_mod.AvailabilityChecker
ProviderSession = _common_mod.ProviderSession


class TestGPUSpec:
    """GPUSpec dataclass construction and defaults."""

    def test_minimal_construction(self):
        spec = GPUSpec(name="T4", memory_gb=16.0)
        assert spec.name == "T4"
        assert spec.memory_gb == 16.0
        assert spec.generation == ""
        assert spec.cuda_cores == 0
        assert spec.tensor_cores == 0
        assert spec.tdp_watts == 0

    def test_full_construction(self, sample_gpu_spec: GPUSpec):
        assert sample_gpu_spec.name == "A100"
        assert sample_gpu_spec.memory_gb == 80.0
        assert sample_gpu_spec.generation == "Ampere"
        assert sample_gpu_spec.cuda_cores == 6912
        assert sample_gpu_spec.tensor_cores == 432
        assert sample_gpu_spec.tdp_watts == 400


class TestInstanceSpec:
    """InstanceSpec dataclass construction and computed properties."""

    def test_minimal_construction(self):
        spec = InstanceSpec(provider="gcp", instance_type="a2-highgpu-8g")
        assert spec.provider == "gcp"
        assert spec.instance_type == "a2-highgpu-8g"
        assert spec.region == ""
        assert spec.gpu is None
        assert spec.gpu_count == 1
        assert spec.vcpus == 0
        assert spec.ram_gb == 0.0
        assert spec.storage_gb == 0.0
        assert spec.network_gbps == 0.0

    def test_total_gpu_memory_with_gpu(self, sample_instance_spec: InstanceSpec):
        assert sample_instance_spec.total_gpu_memory_gb == 80.0 * 8  # 640 GB

    def test_total_gpu_memory_without_gpu(self):
        spec = InstanceSpec(provider="aws", instance_type="t3.medium")
        assert spec.total_gpu_memory_gb == 0.0


class TestPriceQuote:
    """PriceQuote dataclass and computed discount."""

    def test_minimal_construction(self):
        quote = PriceQuote(provider="azure", instance_type="Standard_ND96amsr_A100_v4", region="eastus")
        assert quote.on_demand_hourly == 0.0
        assert quote.spot_hourly == 0.0
        assert quote.currency == "USD"
        assert quote.timestamp > 0

    def test_spot_discount_percentage(self, sample_price_quote: PriceQuote):
        expected_discount = (1 - 9.83 / 32.77) * 100
        assert sample_price_quote.spot_discount_pct == pytest.approx(expected_discount)

    def test_spot_discount_zero_on_demand(self):
        quote = PriceQuote(provider="aws", instance_type="x", region="us-east-1", on_demand_hourly=0.0, spot_hourly=5.0)
        assert quote.spot_discount_pct == 0.0

    def test_spot_discount_zero_spot(self):
        quote = PriceQuote(provider="aws", instance_type="x", region="us-east-1", on_demand_hourly=10.0, spot_hourly=0.0)
        assert quote.spot_discount_pct == 100.0


class TestAvailabilityInfo:
    """AvailabilityInfo dataclass defaults."""

    def test_defaults(self):
        info = AvailabilityInfo(provider="gcp", instance_type="a2-highgpu-8g", region="us-central1")
        assert info.available is True
        assert info.spot_available is True
        assert info.capacity_status == "available"
        assert info.spot_interrupt_rate == 0.0


class TestPricingFetcher:
    """PricingFetcher ABC cannot be instantiated directly."""

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            PricingFetcher()  # type: ignore


class TestAvailabilityChecker:
    """AvailabilityChecker ABC cannot be instantiated directly."""

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            AvailabilityChecker()  # type: ignore


class TestProviderSession:
    """ProviderSession context manager and provider routing."""

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider: nonexistent"):
            ProviderSession("nonexistent")._init_provider()

    def test_context_manager(self):
        with ProviderSession("aws", region="us-east-1") as session:
            assert session.provider == "aws"
            assert session.region == "us-east-1"
            assert session._closed is False
        assert session._closed is True
