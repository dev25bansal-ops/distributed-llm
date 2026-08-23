"""Tests: GPU spot market orchestrator (``distllm.cloud.spot_orchestrator``).

Covers dataclasses, enums, GPU name normalisation, price tracker, HTTP
helpers (with retry), provider implementations (via mocked httpx),
GPUSpotMarket, and SpotOrchestrator.

Run: pytest tests/cloud/test_spot_orchestrator.py -v
"""

from __future__ import annotations

import json
import time
from typing import Any
from unittest.mock import ANY, MagicMock, call, patch

import pytest

from distllm.cloud.spot_orchestrator import (
    BidResult,
    BidStatus,
    CostReport,
    GPUInstance,
    GPU_ON_DEMAND_REFERENCE,
    GPUSpotMarket,
    InstanceStatus,
    Provider,
    ProviderProtocol,
    SpotOrchestrator,
    _PriceTracker,
    _http_delete,
    _http_get,
    _http_post,
    normalize_gpu_name,
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def sample_gpu() -> GPUInstance:
    return GPUInstance(
        provider="runpod",
        instance_id="inst-001",
        gpu_type="NVIDIA RTX 4090",
        gpu_count=1,
        vram_gb=24.0,
        price_per_hour=0.45,
        region="us-east",
    )


@pytest.fixture
def sample_bid_accepted() -> BidResult:
    return BidResult(
        provider="vast",
        instance_id="cnt-001",
        bid_price=0.50,
        status=BidStatus.ACTIVE,
    )


# ===========================================================================
# Dataclass tests
# ===========================================================================


class TestGPUInstance:
    def test_defaults(self) -> None:
        inst = GPUInstance(
            provider="runpod",
            instance_id="i-1",
            gpu_type="RTX 4090",
            gpu_count=1,
            vram_gb=24.0,
            price_per_hour=0.50,
            region="us-east",
        )
        assert inst.status == InstanceStatus.AVAILABLE
        assert inst.spot is True
        assert inst.on_demand_price == 0.0
        assert inst.availability_score == 1.0
        assert inst.cpu_cores == 0
        assert inst.ram_gb == 0.0
        assert inst.storage_gb == 0.0

    def test_equality(self) -> None:
        a = GPUInstance(provider="r", instance_id="i1", gpu_type="A100", gpu_count=1, vram_gb=80.0, price_per_hour=2.0, region="us")
        b = GPUInstance(provider="r", instance_id="i1", gpu_type="A100", gpu_count=1, vram_gb=80.0, price_per_hour=2.0, region="us")
        assert a == b


class TestBidResult:
    def test_defaults(self) -> None:
        r = BidResult(provider="runpod", instance_id="", bid_price=0.5)
        assert r.status == BidStatus.PENDING
        assert r.estimated_wait_minutes == 0.0
        assert r.error_message == ""


class TestCostReport:
    def test_fields(self) -> None:
        r = CostReport(provider="runpod", instance_id="i-1", gpu_type="RTX 4090", hours_running=2.5, total_cost=1.25, spot_savings=0.75)
        assert r.total_cost == 1.25
        assert r.spot_savings == 0.75


# ===========================================================================
# Enum tests
# ===========================================================================


class TestEnums:
    def test_provider_values(self) -> None:
        assert Provider.RUNPOD.value == "runpod"
        assert Provider.VAST.value == "vast"
        assert Provider.SALAD.value == "salad"

    def test_instance_status_values(self) -> None:
        assert InstanceStatus.AVAILABLE.value == "available"
        assert InstanceStatus.RUNNING.value == "running"

    def test_bid_status_values(self) -> None:
        assert BidStatus.ACTIVE.value == "active"
        assert BidStatus.REJECTED.value == "rejected"


# ===========================================================================
# GPU name normalisation
# ===========================================================================


class TestNormalizeGPUName:
    def test_strips_geforce(self) -> None:
        assert normalize_gpu_name("NVIDIA GeForce RTX 4090") == "NVIDIA RTX 4090"

    def test_keeps_canonical(self) -> None:
        assert normalize_gpu_name("NVIDIA RTX 4090") == "NVIDIA RTX 4090"

    def test_other_names_unchanged(self) -> None:
        assert normalize_gpu_name("NVIDIA A100") == "NVIDIA A100"
        assert normalize_gpu_name("AMD MI250") == "AMD MI250"


# ===========================================================================
# GPU_ON_DEMAND_REFERENCE dedup verification
# ===========================================================================


class TestGPUOnDemandReference:
    def test_no_geforce_entries(self) -> None:
        """Dedup requirement: no 'GeForce' keys remain."""
        for key in GPU_ON_DEMAND_REFERENCE:
            assert "GeForce" not in key, f"Found GeForce key: {key}"

    def test_canonical_names_present(self) -> None:
        assert "NVIDIA RTX 4090" in GPU_ON_DEMAND_REFERENCE
        assert "NVIDIA RTX 4080" in GPU_ON_DEMAND_REFERENCE
        assert "NVIDIA RTX 3090" in GPU_ON_DEMAND_REFERENCE

    def test_amd_keys_present(self) -> None:
        assert "AMD MI250" in GPU_ON_DEMAND_REFERENCE
        assert "AMD MI210" in GPU_ON_DEMAND_REFERENCE


# ===========================================================================
# _PriceTracker
# ===========================================================================


class TestPriceTracker:
    def test_record_and_moving_average(self) -> None:
        t = _PriceTracker(window_size=5)
        t.record_price("NVIDIA RTX 4090", 0.50)
        t.record_price("NVIDIA RTX 4090", 0.60)
        t.record_price("NVIDIA RTX 4090", 0.70)
        avg = t.moving_average("NVIDIA RTX 4090")
        assert avg == pytest.approx(0.60)

    def test_moving_average_normalises_key(self) -> None:
        """Keys with 'GeForce' should match canonical entries."""
        t = _PriceTracker()
        t.record_price("NVIDIA GeForce RTX 4090", 0.50)
        avg = t.moving_average("NVIDIA RTX 4090")
        assert avg == pytest.approx(0.50)

    def test_p75(self) -> None:
        t = _PriceTracker()
        for p in [0.1, 0.2, 0.3, 0.4, 0.5]:
            t.record_price("RTX 4090", p)
        p75 = t.p75("RTX 4090")
        assert p75 is not None
        assert p75 >= 0.4

    def test_p75_empty(self) -> None:
        assert _PriceTracker().p75("unknown") is None

    def test_suggest_bid_with_on_demand(self) -> None:
        t = _PriceTracker()
        bid = t.suggest_bid("NVIDIA RTX 4090", on_demand_price=0.80)
        assert bid is not None
        assert 0.48 <= bid <= 0.64  # within [0.6*0.8, 0.8*0.8]

    def test_suggest_bid_no_reference(self) -> None:
        t = _PriceTracker()
        assert t.suggest_bid("Unknown GPU") is None

    def test_suggest_bid_falls_back_to_average(self) -> None:
        t = _PriceTracker()
        t.record_price("Unknown GPU", 0.30)
        t.record_price("Unknown GPU", 0.34)
        bid = t.suggest_bid("Unknown GPU")
        assert bid == pytest.approx(0.32, abs=0.005)

    def test_record_prices_from_instances(self) -> None:
        t = _PriceTracker()
        instances = [
            GPUInstance(provider="r", instance_id="1", gpu_type="RTX 4090", gpu_count=1, vram_gb=24.0, price_per_hour=0.50, region="us"),
            GPUInstance(provider="r", instance_id="2", gpu_type="RTX 4090", gpu_count=1, vram_gb=24.0, price_per_hour=0.60, region="us"),
        ]
        t.record_prices(instances)
        assert t.moving_average("RTX 4090") == pytest.approx(0.55)

    def test_describe(self) -> None:
        t = _PriceTracker()
        t.record_price("RTX 4090", 0.50)
        desc = t.describe("RTX 4090")
        assert desc["gpu_type"] == "RTX 4090"
        assert desc["observations"] == 1
        assert desc["recommended_bid"] is not None


# ===========================================================================
# HTTP helpers with retry
# ===========================================================================


class TestHttpHelpers:
    """Test retry logic by patching ``httpx``."""

    def test_http_get_success(self) -> None:
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"status": "ok"}

        with patch("httpx.get", return_value=mock_response) as mock_get:
            result = _http_get("https://example.com/api")
            assert result == {"status": "ok"}
            mock_get.assert_called_once_with(
                "https://example.com/api",
                headers={},
                params=None,
                timeout=30.0,
            )

    def test_http_get_retry_then_success(self) -> None:
        """First two calls fail, third succeeds."""
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"status": "ok"}

        fail_response = MagicMock()
        fail_response.raise_for_status.side_effect = __import__("httpx").HTTPStatusError(
            "Server Error", request=MagicMock(), response=MagicMock()
        )

        with patch("httpx.get", side_effect=[fail_response, fail_response, mock_response]) as mock_get:
            result = _http_get("https://example.com/api")
            assert result == {"status": "ok"}
            assert mock_get.call_count == 3

    def test_http_get_all_fail(self) -> None:
        """All retries exhausted raises last exception."""
        fail_response = MagicMock()
        fail_response.raise_for_status.side_effect = __import__("httpx").HTTPStatusError(
            "Server Error", request=MagicMock(), response=MagicMock()
        )

        with patch("httpx.get", return_value=fail_response) as mock_get:
            with pytest.raises(__import__("httpx").HTTPStatusError):
                _http_get("https://example.com/api")
            assert mock_get.call_count == 3

    def test_http_post_success(self) -> None:
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"id": "new-instance"}

        with patch("httpx.post", return_value=mock_response) as mock_post:
            result = _http_post("https://api.example.com/create", json_body={"key": "val"}, headers={"Auth": "token"})
            assert result == {"id": "new-instance"}
            mock_post.assert_called_once_with(
                "https://api.example.com/create",
                json={"key": "val"},
                headers={"Auth": "token"},
                timeout=30.0,
            )

    def test_http_delete_success(self) -> None:
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"deleted": True}

        with patch("httpx.delete", return_value=mock_response) as mock_delete:
            result = _http_delete("https://api.example.com/instances/1")
            assert result == {"deleted": True}

    def test_http_get_retry_on_timeout(self) -> None:
        """Retry on httpx.TimeoutException."""
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"ok": True}

        with patch("httpx.get", side_effect=[__import__("httpx").TimeoutException("timeout"), mock_response]) as mock_get:
            result = _http_get("https://example.com/api")
            assert result == {"ok": True}
            assert mock_get.call_count == 2


# ===========================================================================
# ProviderProtocol conformance
# ===========================================================================


class TestProviderProtocol:
    """Verify that all internal providers conform to ProviderProtocol."""

    def test_runpod_conforms(self) -> None:
        from distllm.cloud.spot_orchestrator import _RunPodProvider
        assert isinstance(_RunPodProvider, type)
        # Structural subtyping: check methods exist
        p = _RunPodProvider(api_key="test-key")
        assert hasattr(p, "list_instances")
        assert hasattr(p, "bid")
        assert hasattr(p, "cancel_bid")
        assert hasattr(p, "close")

    def test_vast_conforms(self) -> None:
        from distllm.cloud.spot_orchestrator import _VastProvider
        p = _VastProvider(api_key="test-key")
        assert hasattr(p, "list_instances")
        assert hasattr(p, "bid")
        assert hasattr(p, "cancel_bid")
        assert hasattr(p, "close")

    def test_salad_conforms(self) -> None:
        from distllm.cloud.spot_orchestrator import _SaladProvider
        p = _SaladProvider(api_key="test-key")
        assert hasattr(p, "list_instances")
        assert hasattr(p, "bid")
        assert hasattr(p, "cancel_bid")
        assert hasattr(p, "close")


# ===========================================================================
# _VastProvider -- API key header fix & storage_gb fix
# ===========================================================================


class TestVastProvider:
    """Covers: API key in Authorization header (not query param) and storage_cost→disk_size fix."""

    @patch("distllm.cloud.spot_orchestrator._http_get")
    def test_api_key_in_header_not_query(self, mock_get: MagicMock) -> None:
        """Vast API key must be in Authorization header, never in query params."""
        from distllm.cloud.spot_orchestrator import _VastProvider

        mock_get.return_value = []

        provider = _VastProvider(api_key="vast-secret-123")
        provider.list_instances()

        # Verify call: headers should have Authorization, params should NOT have api_key
        call_kwargs = mock_get.call_args[1]
        assert "Authorization" in call_kwargs.get("headers", {})
        auth_header = call_kwargs["headers"]["Authorization"]
        assert auth_header == "Bearer vast-secret-123"
        # Verify api_key is not in params
        params = call_kwargs.get("params") or {}
        assert "api_key" not in params

    @patch("distllm.cloud.spot_orchestrator._http_get")
    def test_storage_gb_from_disk_size(self, mock_get: MagicMock) -> None:
        """storage_gb should be populated from 'disk_size' field, not 'storage_cost'."""
        from distllm.cloud.spot_orchestrator import _VastProvider

        mock_get.return_value = [
            {
                "id": "offer-1",
                "gpu_name": "NVIDIA RTX 4090",
                "num_gpus": 1,
                "gpu_ram": 24.0,
                "dph_total": 0.45,
                "disk_size": 200,  # real disk size in GB
                "storage_cost": 0.02,  # cost in USD -- should NOT be used
                "geographic_location": "US",
                "cpu_cores": 8,
                "cpu_ram": 32.0,
            }
        ]

        provider = _VastProvider(api_key="test-key")
        instances = provider.list_instances()

        assert len(instances) == 1
        assert instances[0].storage_gb == 200.0  # from disk_size, NOT storage_cost
        assert instances[0].storage_gb != 0.02  # not the cost value

    @patch("distllm.cloud.spot_orchestrator._http_get")
    def test_list_instances_empty_on_error(self, mock_get: MagicMock) -> None:
        from distllm.cloud.spot_orchestrator import _VastProvider

        mock_get.side_effect = RuntimeError("API unreachable")

        provider = _VastProvider(api_key="test-key")
        instances = provider.list_instances()
        assert instances == []

    @patch("distllm.cloud.spot_orchestrator._http_get")
    def test_list_instances_filters_by_gpu_type(self, mock_get: MagicMock) -> None:
        from distllm.cloud.spot_orchestrator import _VastProvider

        mock_get.return_value = [
            {"id": "1", "gpu_name": "NVIDIA RTX 4090", "num_gpus": 1, "gpu_ram": 24.0, "dph_total": 0.45, "geographic_location": "US"},
            {"id": "2", "gpu_name": "NVIDIA A100", "num_gpus": 1, "gpu_ram": 80.0, "dph_total": 2.50, "geographic_location": "US"},
        ]

        provider = _VastProvider(api_key="test-key")
        instances = provider.list_instances(gpu_type="4090")
        assert len(instances) == 1
        assert instances[0].instance_id == "1"


# ===========================================================================
# _RunPodProvider
# ===========================================================================


class TestRunPodProvider:
    @patch("distllm.cloud.spot_orchestrator._http_get")
    def test_list_instances_http(self, mock_get: MagicMock) -> None:
        from distllm.cloud.spot_orchestrator import _RunPodProvider

        mock_get.return_value = {
            "gpuPrices": {
                "NVIDIA RTX 4090": {
                    "minimumBidPrice": 0.45,
                    "costPerHour": 0.80,
                    "gpuCount": 1,
                    "gpuMemoryInGb": 24.0,
                }
            }
        }

        provider = _RunPodProvider(api_key="rp-key")
        instances = provider.list_instances()
        assert len(instances) == 1
        assert instances[0].gpu_type == "NVIDIA RTX 4090"
        assert instances[0].price_per_hour == 0.45
        assert instances[0].on_demand_price == 0.80

    @patch("distllm.cloud.spot_orchestrator._http_get")
    def test_list_instances_handles_missing_prices_key(self, mock_get: MagicMock) -> None:
        from distllm.cloud.spot_orchestrator import _RunPodProvider

        # gpuPrices key is present but empty -> no instances
        mock_get.return_value = {"gpuPrices": {}}

        provider = _RunPodProvider(api_key="rp-key")
        instances = provider.list_instances()
        assert instances == []

    @patch("distllm.cloud.spot_orchestrator._http_post")
    def test_bid_via_http(self, mock_post: MagicMock) -> None:
        from distllm.cloud.spot_orchestrator import _RunPodProvider

        mock_post.return_value = {"id": "inst-abc", "bidPrice": 0.50}

        provider = _RunPodProvider(api_key="rp-key")
        result = provider.bid({"gpu_type": "RTX 4090", "max_price": 0.50})
        assert result.instance_id == "inst-abc"
        assert result.status == BidStatus.ACTIVE

    @patch("distllm.cloud.spot_orchestrator._http_post")
    def test_bid_fails_gracefully(self, mock_post: MagicMock) -> None:
        from distllm.cloud.spot_orchestrator import _RunPodProvider

        mock_post.side_effect = RuntimeError("API error")

        provider = _RunPodProvider(api_key="rp-key")
        result = provider.bid({"gpu_type": "RTX 4090", "max_price": 0.50})
        assert result.status == BidStatus.REJECTED
        assert "API error" in result.error_message

    @patch("distllm.cloud.spot_orchestrator._http_delete")
    def test_cancel_bid(self, mock_delete: MagicMock) -> None:
        from distllm.cloud.spot_orchestrator import _RunPodProvider

        mock_delete.return_value = {}
        provider = _RunPodProvider(api_key="rp-key")
        assert provider.cancel_bid("inst-abc") is True


# ===========================================================================
# _SaladProvider
# ===========================================================================


class TestSaladProvider:
    @patch.multiple(
        "distllm.cloud.spot_orchestrator",
        _HAS_SALAD=False,
    )
    @patch("distllm.cloud.spot_orchestrator._http_get")
    def test_list_instances_http(self, mock_get: MagicMock) -> None:
        from distllm.cloud.spot_orchestrator import _SaladProvider

        mock_get.return_value = {
            "items": [
                {
                    "id": "ct-1",
                    "gpuClass": "NVIDIA RTX 4090",
                    "gpuCount": 1,
                    "gpuVramGb": 24.0,
                    "pricePerHour": 0.45,
                    "region": "us-east",
                }
            ]
        }

        provider = _SaladProvider(api_key="salad-key")
        # Set org and project for Salad
        with patch.dict("os.environ", {"SALAD_ORGANIZATION": "myorg", "SALAD_PROJECT": "myproj"}):
            # Re-init to pick up env vars
            provider = _SaladProvider(api_key="salad-key")
            instances = provider.list_instances()
            assert len(instances) == 1
            assert instances[0].gpu_type == "NVIDIA RTX 4090"
            assert instances[0].price_per_hour == 0.45

    @patch("distllm.cloud.spot_orchestrator._http_get")
    def test_list_instances_fallback(self, mock_get: MagicMock) -> None:
        from distllm.cloud.spot_orchestrator import _SaladProvider

        mock_get.side_effect = RuntimeError("API unreachable")

        provider = _SaladProvider(api_key="salad-key")
        with patch.dict("os.environ", {"SALAD_ORGANIZATION": "myorg", "SALAD_PROJECT": "myproj"}):
            provider = _SaladProvider(api_key="salad-key")
            instances = provider.list_instances()
            # Should return the fallback list
            assert len(instances) > 0
            assert instances[0].provider == "salad"

    @patch("distllm.cloud.spot_orchestrator._http_post")
    def test_bid_via_http(self, mock_post: MagicMock) -> None:
        from distllm.cloud.spot_orchestrator import _SaladProvider

        mock_post.return_value = {"id": "ct-new"}

        provider = _SaladProvider(api_key="salad-key")
        with patch.dict("os.environ", {"SALAD_ORGANIZATION": "myorg", "SALAD_PROJECT": "myproj"}):
            provider = _SaladProvider(api_key="salad-key")
            result = provider.bid({"gpu_type": "RTX 4090", "max_price": 0.50, "image": "ubuntu:22.04"})
            assert result.instance_id == "ct-new"
            assert result.status == BidStatus.PENDING


# ===========================================================================
# GPUSpotMarket
# ===========================================================================


class TestGPUSpotMarket:
    def test_unknown_provider_raises(self) -> None:
        market = GPUSpotMarket()
        with pytest.raises(ValueError, match="Unknown provider"):
            market._get_provider("nonexistent")

    def test_closed_market_raises(self) -> None:
        market = GPUSpotMarket()
        market.close()
        with pytest.raises(RuntimeError, match="closed"):
            market.list_instances(provider="runpod")

    @patch("distllm.cloud.spot_orchestrator._VastProvider.list_instances")
    def test_list_instances_returns_sorted(self, mock_list: MagicMock) -> None:
        mock_list.return_value = [
            GPUInstance(provider="vast", instance_id="2", gpu_type="RTX 4090", gpu_count=1, vram_gb=24.0, price_per_hour=0.60, region="us"),
            GPUInstance(provider="vast", instance_id="1", gpu_type="RTX 4090", gpu_count=1, vram_gb=24.0, price_per_hour=0.45, region="us"),
        ]

        market = GPUSpotMarket()
        instances = market.list_instances(provider="vast")
        assert len(instances) == 2
        # Should be sorted by price ascending
        assert instances[0].price_per_hour == 0.45
        assert instances[1].price_per_hour == 0.60

    @patch("distllm.cloud.spot_orchestrator._RunPodProvider.list_instances")
    @patch("distllm.cloud.spot_orchestrator._VastProvider.list_instances")
    def test_list_all(self, mock_vast: MagicMock, mock_runpod: MagicMock) -> None:
        mock_vast.return_value = [GPUInstance(provider="vast", instance_id="v1", gpu_type="RTX 4090", gpu_count=1, vram_gb=24.0, price_per_hour=0.45, region="us")]
        mock_runpod.return_value = [GPUInstance(provider="runpod", instance_id="r1", gpu_type="RTX 4090", gpu_count=1, vram_gb=24.0, price_per_hour=0.50, region="us")]

        market = GPUSpotMarket()
        result = market.list_all(providers=["vast", "runpod"])
        assert "vast" in result
        assert "runpod" in result
        assert len(result["vast"]) == 1
        assert len(result["runpod"]) == 1

    @patch("distllm.cloud.spot_orchestrator._RunPodProvider.bid")
    def test_bid_with_dynamic_pricing(self, mock_bid: MagicMock) -> None:
        """When max_price is not set, it falls back to dynamic pricing."""
        from distllm.cloud.spot_orchestrator import _PriceTracker

        mock_bid.return_value = BidResult(provider="runpod", instance_id="r1", bid_price=0.50, status=BidStatus.ACTIVE)

        tracker = _PriceTracker()
        tracker.record_price("NVIDIA RTX 4090", 0.50)

        market = GPUSpotMarket(price_tracker=tracker)
        result = market.bid("runpod", {"gpu_type": "NVIDIA RTX 4090"})

        assert result.status == BidStatus.ACTIVE
        # Verify dynamic pricing was used (no max_price set -> tracker suggests)
        mock_bid.assert_called_once()
        config = mock_bid.call_args[0][0]
        assert "max_price" in config

    @patch("distllm.cloud.spot_orchestrator._RunPodProvider.cancel_bid")
    def test_cancel_bid(self, mock_cancel: MagicMock) -> None:
        mock_cancel.return_value = True

        market = GPUSpotMarket()
        assert market.cancel_bid("runpod", "inst-1") is True

    def test_context_manager(self) -> None:
        with GPUSpotMarket() as market:
            assert not market._closed
        assert market._closed

    def test_suggest_bid_price(self) -> None:
        market = GPUSpotMarket()
        bid = market.suggest_bid_price("NVIDIA RTX 4090", on_demand_price=0.80)
        assert bid is not None
        assert 0.0 < bid < 0.80

    def test_price_summary(self) -> None:
        market = GPUSpotMarket()
        summary = market.price_summary("RTX 4090")
        assert summary["gpu_type"] == "RTX 4090"
        assert "observations" in summary


# ===========================================================================
# SpotOrchestrator
# ===========================================================================


class TestSpotOrchestrator:
    def test_find_cheapest_filters_by_price(self, sample_gpu: GPUInstance) -> None:
        market = MagicMock(spec=GPUSpotMarket)
        expensive = GPUInstance(provider="runpod", instance_id="e1", gpu_type="RTX 4090", gpu_count=1, vram_gb=24.0, price_per_hour=1.50, region="us")
        market.list_all.return_value = {"runpod": [sample_gpu, expensive]}

        orch = SpotOrchestrator(market=market)
        candidates = orch.find_cheapest(gpu_type="RTX 4090", max_price=1.00)
        assert len(candidates) == 1
        assert candidates[0].price_per_hour == 0.45

    def test_find_cheapest_filters_by_region(self, sample_gpu: GPUInstance) -> None:
        market = MagicMock(spec=GPUSpotMarket)
        other_region = GPUInstance(provider="runpod", instance_id="eu1", gpu_type="RTX 4090", gpu_count=1, vram_gb=24.0, price_per_hour=0.50, region="eu-west")
        market.list_all.return_value = {"runpod": [sample_gpu, other_region]}

        orch = SpotOrchestrator(market=market)
        candidates = orch.find_cheapest(gpu_type="RTX 4090", max_price=1.00, region="us")
        assert len(candidates) == 1
        assert candidates[0].region == "us-east"

    def test_find_cheapest_filters_by_gpu_count(self) -> None:
        market = MagicMock(spec=GPUSpotMarket)
        single = GPUInstance(provider="runpod", instance_id="s1", gpu_type="RTX 4090", gpu_count=1, vram_gb=24.0, price_per_hour=0.45, region="us")
        multi = GPUInstance(provider="runpod", instance_id="m1", gpu_type="RTX 4090", gpu_count=4, vram_gb=24.0, price_per_hour=1.50, region="us")
        market.list_all.return_value = {"runpod": [single, multi]}

        orch = SpotOrchestrator(market=market)
        candidates = orch.find_cheapest(gpu_type="RTX 4090", max_price=2.00, min_gpu_count=4)
        assert len(candidates) == 1
        assert candidates[0].gpu_count == 4

    def test_launch_cluster(self, sample_gpu: GPUInstance) -> None:
        market = MagicMock(spec=GPUSpotMarket)
        market.bid.return_value = BidResult(provider="runpod", instance_id="r1", bid_price=0.45, status=BidStatus.ACTIVE)

        orch = SpotOrchestrator(market=market)
        results = orch.launch_cluster([sample_gpu])

        assert len(results) == 1
        assert results[0].status == BidStatus.ACTIVE
        assert "r1" in orch._running

    def test_launch_cluster_no_accepted(self, sample_gpu: GPUInstance) -> None:
        market = MagicMock(spec=GPUSpotMarket)
        market.bid.return_value = BidResult(provider="runpod", instance_id="", bid_price=0.45, status=BidStatus.REJECTED)

        orch = SpotOrchestrator(market=market)
        results = orch.launch_cluster([sample_gpu])

        assert len(results) == 1
        assert results[0].status == BidStatus.REJECTED
        assert len(orch._running) == 0

    def test_monitor_costs(self, sample_gpu: GPUInstance) -> None:
        market = MagicMock(spec=GPUSpotMarket)
        orch = SpotOrchestrator(market=market)

        # Manually track an instance with an on-demand price for savings calc
        inst = sample_gpu
        inst.on_demand_price = 0.80
        orch._running["r1"] = inst
        orch._start_times["r1"] = time.time() - 3600  # 1 hour ago

        reports = orch.monitor_costs()
        assert len(reports) == 1
        assert reports[0].instance_id == "r1"
        assert reports[0].hours_running == pytest.approx(1.0, abs=0.1)
        assert reports[0].total_cost == pytest.approx(0.45, abs=0.05)
        assert reports[0].spot_savings > 0  # on_demand_price=0.8 vs 0.45

    def test_monitor_costs_refresh(self, sample_gpu: GPUInstance) -> None:
        market = MagicMock(spec=GPUSpotMarket)
        market.list_instances.return_value = []  # no running instances

        orch = SpotOrchestrator(market=market)
        orch._running["r1"] = sample_gpu
        orch._start_times["r1"] = time.time()

        orch.monitor_costs(refresh_instances=True)
        # Instance should be removed since it's not in the refreshed list
        assert "r1" not in orch._running

    def test_swap_providers_no_migrating(self) -> None:
        market = MagicMock(spec=GPUSpotMarket)
        orch = SpotOrchestrator(market=market)
        results = orch.swap_providers(current="runpod", target="vast")
        assert results == []

    def test_swap_providers_dry_run(self, sample_gpu: GPUInstance) -> None:
        market = MagicMock(spec=GPUSpotMarket)
        market.list_all.return_value = {"vast": [sample_gpu]}

        orch = SpotOrchestrator(market=market)
        orch._running["r1"] = sample_gpu
        orch._start_times["r1"] = time.time()

        results = orch.swap_providers(current="runpod", target="vast", dry_run=True)
        assert results == []  # dry_run returns empty

    @patch("distllm.cloud.spot_orchestrator.GPUSpotMarket.bid")
    def test_swap_providers_full(self, mock_bid: MagicMock, sample_gpu: GPUInstance) -> None:
        mock_bid.return_value = BidResult(provider="vast", instance_id="v1", bid_price=0.45, status=BidStatus.ACTIVE)

        market = MagicMock(spec=GPUSpotMarket)

        # find_cheapest -> market.list_all
        target = GPUInstance(provider="vast", instance_id="v1", gpu_type="NVIDIA RTX 4090", gpu_count=1, vram_gb=24.0, price_per_hour=0.45, region="us")
        market.list_all.return_value = {"vast": [target]}

        orch = SpotOrchestrator(market=market)
        orch._running["r1"] = sample_gpu
        orch._start_times["r1"] = time.time()

        results = orch.swap_providers(current="runpod", target="vast")
        # Should have bid results
        assert len(results) > 0

    def test_running_instances(self, sample_gpu: GPUInstance) -> None:
        orch = SpotOrchestrator()
        orch._running["r1"] = sample_gpu
        assert len(orch.running_instances()) == 1
        assert orch.running_instances()[0] == sample_gpu

    def test_summary(self) -> None:
        orch = SpotOrchestrator()
        summary = orch.summary()
        assert summary["running_count"] == 0
        assert summary["total_cost_usd"] == 0.0

    def test_context_manager(self) -> None:
        market = MagicMock(spec=GPUSpotMarket)
        with SpotOrchestrator(market=market) as orch:
            assert orch._market is market
        market.close.assert_called_once()

    def test_close(self) -> None:
        market = MagicMock(spec=GPUSpotMarket)
        orch = SpotOrchestrator(market=market)
        orch.close()
        market.close.assert_called_once()


# ===========================================================================
# Module __all__ exports
# ===========================================================================


class TestModuleExports:
    def test_all_defined(self) -> None:
        from distllm.cloud.spot_orchestrator import __all__ as module_all

        expected = {
            "Provider",
            "InstanceStatus",
            "BidStatus",
            "GPUInstance",
            "BidResult",
            "CostReport",
            "GPU_ON_DEMAND_REFERENCE",
            "normalize_gpu_name",
            "ProviderProtocol",
            "GPUSpotMarket",
            "SpotOrchestrator",
        }
        assert set(module_all) == expected

    def test_public_exports_via_cloud_package(self) -> None:
        """Verify all spot orchestrator symbols are re-exported from distllm.cloud."""
        import distllm.cloud

        for name in [
            "Provider",
            "InstanceStatus",
            "BidStatus",
            "GPUInstance",
            "BidResult",
            "CostReport",
            "GPU_ON_DEMAND_REFERENCE",
            "normalize_gpu_name",
            "ProviderProtocol",
            "GPUSpotMarket",
            "SpotOrchestrator",
        ]:
            assert hasattr(distllm.cloud, name), f"Missing re-export: {name}"
