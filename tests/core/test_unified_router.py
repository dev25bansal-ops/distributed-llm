"""Tests for UnifiedRouter, ComputeOption, UnifiedRouteDecision, DisaggregatedRouter.

Uses the import-helper pattern to avoid circular imports.
"""

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_unified_mod = load_module("distllm/core/unified_router.py")
UnifiedRouter = _unified_mod.UnifiedRouter
ComputeOption = _unified_mod.ComputeOption
ComputeSource = _unified_mod.ComputeSource
UnifiedRouteDecision = _unified_mod.UnifiedRouteDecision
DisaggregatedRouter = _unified_mod.DisaggregatedRouter
RequestPhase = _unified_mod.RequestPhase


# ── ComputeOption dataclass ──────────────────────────────────────────────────


class TestComputeOption:
    def test_defaults(self):
        opt = ComputeOption(source=ComputeSource.CLOUD, provider_name="aws", instance_type="p4d")
        assert opt.region == ""
        assert opt.gpu_count == 1
        assert opt.price_per_hour == 0.0
        assert opt.reputation_score == 0.5
        assert opt.uptime_pct == 100.0
        assert opt.available is True

    def test_effective_spot_price_zero_spot(self):
        opt = ComputeOption(source=ComputeSource.CLOUD, provider_name="aws", instance_type="p4d",
                            price_per_hour=10.0, spot_price=0.0)
        assert opt.effective_spot_price == 10.0

    def test_effective_spot_price_positive(self):
        opt = ComputeOption(source=ComputeSource.CLOUD, provider_name="aws", instance_type="p4d",
                            price_per_hour=10.0, spot_price=5.0)
        assert opt.effective_spot_price == 5.0

    def test_is_available_true(self):
        opt = ComputeOption(source=ComputeSource.CLOUD, provider_name="aws", instance_type="p4d",
                            available=True, max_concurrent=5, current_load=3)
        assert opt.is_available is True

    def test_is_available_false_no_capacity(self):
        opt = ComputeOption(source=ComputeSource.CLOUD, provider_name="aws", instance_type="p4d",
                            available=True, max_concurrent=5, current_load=5)
        assert opt.is_available is False

    def test_is_available_false_not_available(self):
        opt = ComputeOption(source=ComputeSource.CLOUD, provider_name="aws", instance_type="p4d",
                            available=False, max_concurrent=5, current_load=0)
        assert opt.is_available is False

    def test_to_dict(self):
        opt = ComputeOption(source=ComputeSource.CLOUD, provider_name="aws", instance_type="p4d",
                            region="us-east-1", gpu_type="A100", gpu_count=8,
                            price_per_hour=14.40, spot_price=4.32)
        d = opt.to_dict()
        assert d["source"] == "cloud"
        assert d["provider"] == "aws"
        assert d["gpu_count"] == 8


# ── UnifiedRouter construction and defaults ──────────────────────────────────


class TestUnifiedRouterInit:
    def test_defaults(self):
        router = UnifiedRouter()
        assert router._cloud_options == []
        assert router._peer_options == []
        assert router._latency_cache == {}
        assert router._stats == {"routes": 0, "cloud_selected": 0, "peer_selected": 0}

    def test_stats_property(self):
        router = UnifiedRouter()
        assert router.stats == {"routes": 0, "cloud_selected": 0, "peer_selected": 0}


# ── set_cloud_options / set_peer_options ──────────────────────────────────────


class TestUnifiedRouterSetOptions:
    def test_set_cloud_options_empty(self):
        router = UnifiedRouter()
        router.set_cloud_options([])
        assert router._cloud_options == []

    def test_set_cloud_options_with_dicts(self):
        router = UnifiedRouter()
        router.set_cloud_options([
            {"provider": "aws", "instance_type": "p4d.24xlarge", "on_demand_price": 14.40,
             "region": "us-east-1", "gpu_type": "A100", "gpu_count": 8},
        ])
        assert len(router._cloud_options) == 1
        opt = router._cloud_options[0]
        assert opt.provider_name == "aws"
        assert opt.price_per_hour == 14.40

    def test_set_cloud_options_with_objects(self):
        router = UnifiedRouter()
        instance = type("Pricing", (), {
            "provider": "gcp", "instance_type": "a2-highgpu-1g",
            "on_demand_price": 3.67, "region": "us-central1",
            "gpu_type": "A100", "gpu_count": 1, "gpu_memory_gb": 40.0,
        })()
        router.set_cloud_options([instance])
        assert len(router._cloud_options) == 1
        opt = router._cloud_options[0]
        assert opt.provider_name == "gcp"

    def test_set_peer_options_empty(self):
        router = UnifiedRouter()
        router.set_peer_options([])
        assert router._peer_options == []

    def test_set_peer_options_with_dicts(self):
        router = UnifiedRouter()
        router.set_peer_options([
            {"listing_id": "abc-123", "provider_name": "peer1", "gpu_name": "RTX4090",
             "region": "us-east-1", "gpu_count": 1, "price_per_hour": 2.0,
             "is_available": True, "reputation_score": 0.9},
        ])
        assert len(router._peer_options) == 1
        opt = router._peer_options[0]
        assert opt.source == ComputeSource.PEER
        assert opt.listing_id == "abc-123"

    def test_set_peer_options_with_objects(self):
        router = UnifiedRouter()
        obj = type("Listing", (), {
            "listing_id": "xyz", "gpu_name": "A100", "region": "us-east-1",
            "gpu_count": 1, "price_per_hour": 3.0, "is_available": True,
        })()
        router.set_peer_options([obj])
        assert len(router._peer_options) == 1

    def test_set_latency(self):
        router = UnifiedRouter()
        router.set_latency("aws", 50.0)
        assert router._latency_cache["aws"] == 50.0


# ── route() ──────────────────────────────────────────────────────────────────


class TestUnifiedRouterRoute:
    def test_no_options_returns_none(self):
        router = UnifiedRouter()
        result = router.route()
        assert result is None

    def test_route_to_cloud_option(self):
        router = UnifiedRouter()
        router.set_cloud_options([
            {"provider": "aws", "instance_type": "p4d.24xlarge", "on_demand_price": 14.40,
             "region": "us-east-1", "gpu_type": "A100", "gpu_count": 8},
        ])
        result = router.route()
        assert result is not None
        assert isinstance(result, UnifiedRouteDecision)
        assert result.selected.provider_name == "aws"
        assert result.scoring_method == "price"

    def test_route_to_peer_option(self):
        router = UnifiedRouter()
        router.set_peer_options([
            {"listing_id": "abc", "provider_name": "peer1", "gpu_name": "RTX4090",
             "region": "us-east-1", "price_per_hour": 2.0, "is_available": True},
        ])
        result = router.route()
        assert result is not None
        assert result.selected.source == ComputeSource.PEER

    def test_route_respects_max_price(self):
        router = UnifiedRouter()
        router.set_cloud_options([
            {"provider": "aws", "instance_type": "p4d.24xlarge", "on_demand_price": 14.40,
             "region": "us-east-1", "gpu_type": "A100"},
        ])
        result = router.route(max_price=1.0)
        assert result is None

    def test_route_respects_gpu_type_filter(self):
        router = UnifiedRouter()
        router.set_cloud_options([
            {"provider": "aws", "instance_type": "p4d.24xlarge", "on_demand_price": 14.40,
             "region": "us-east-1", "gpu_type": "A100"},
            {"provider": "gcp", "instance_type": "g2-standard-4", "on_demand_price": 0.84,
             "region": "us-central1", "gpu_type": "L4"},
        ])
        result = router.route(gpu_type="L4")
        assert result is not None
        assert "L4" in result.selected.gpu_type

    def test_route_respects_source_filter_cloud(self):
        router = UnifiedRouter()
        router.set_cloud_options([
            {"provider": "aws", "instance_type": "p4d.24xlarge", "on_demand_price": 14.40,
             "region": "us-east-1"},
        ])
        router.set_peer_options([
            {"listing_id": "abc", "provider_name": "peer1", "gpu_name": "RTX4090",
             "price_per_hour": 2.0, "is_available": True},
        ])
        result = router.route(source_filter=ComputeSource.CLOUD)
        assert result is not None
        assert result.selected.source == ComputeSource.CLOUD

    def test_route_respects_min_reputation(self):
        router = UnifiedRouter()
        router.set_peer_options([
            {"listing_id": "abc", "provider_name": "peer1", "gpu_name": "RTX4090",
             "price_per_hour": 2.0, "is_available": True, "reputation_score": 0.3},
        ])
        result = router.route(min_reputation=0.5)
        assert result is None

    def test_route_stats_incremented(self):
        router = UnifiedRouter()
        router.set_cloud_options([
            {"provider": "aws", "instance_type": "p4d.24xlarge", "on_demand_price": 14.40,
             "region": "us-east-1"},
        ])
        before = router.stats["routes"]
        router.route()
        assert router.stats["routes"] == before + 1

    def test_route_scoring_balanced(self):
        router = UnifiedRouter()
        router.set_cloud_options([
            {"provider": "aws", "instance_type": "p4d.24xlarge", "on_demand_price": 14.40,
             "region": "us-east-1", "gpu_type": "A100", "gpu_memory_gb": 80.0},
            {"provider": "gcp", "instance_type": "a2-highgpu-1g", "on_demand_price": 3.67,
             "region": "us-central1", "gpu_type": "A100", "gpu_memory_gb": 40.0},
        ])
        result = router.route(scoring="balanced", carbon_weight=0.5)
        assert result is not None

    def test_route_scoring_carbon(self):
        router = UnifiedRouter()
        router.set_cloud_options([
            {"provider": "aws", "instance_type": "p4d.24xlarge", "on_demand_price": 14.40,
             "region": "us-east-1", "gpu_type": "A100", "carbon_intensity": 380.0},
            {"provider": "gcp", "instance_type": "a2-highgpu-1g", "on_demand_price": 3.67,
             "region": "europe-north1", "gpu_type": "A100", "carbon_intensity": 15.0},
        ])
        result = router.route(scoring="carbon")
        assert result is not None
        # Carbon sorting picks lowest carbon first
        assert result.selected.carbon_intensity == 15.0

    def test_route_respects_min_gpu_memory(self):
        router = UnifiedRouter()
        router.set_cloud_options([
            {"provider": "aws", "instance_type": "p4d.24xlarge", "on_demand_price": 14.40,
             "region": "us-east-1", "gpu_memory_gb": 80.0},
            {"provider": "gcp", "instance_type": "a2-highgpu-1g", "on_demand_price": 3.67,
             "region": "us-central1", "gpu_memory_gb": 40.0},
        ])
        result = router.route(min_gpu_memory_gb=50.0)
        assert result is not None
        assert result.selected.gpu_memory_gb >= 50.0


# ── get_all_options ──────────────────────────────────────────────────────────


class TestUnifiedRouterGetAllOptions:
    def test_empty(self):
        router = UnifiedRouter()
        assert router.get_all_options() == []

    def test_returns_only_available(self):
        router = UnifiedRouter()
        router.set_cloud_options([
            {"provider": "aws", "instance_type": "p4d.24xlarge", "on_demand_price": 14.40,
             "region": "us-east-1", "available": False},
            {"provider": "gcp", "instance_type": "a2-highgpu-1g", "on_demand_price": 3.67,
             "region": "us-central1", "available": True},
        ])
        options = router.get_all_options()
        assert any(o["available"] for o in options)

    def test_to_dict_format(self):
        router = UnifiedRouter()
        router.set_cloud_options([
            {"provider": "aws", "instance_type": "p4d.24xlarge", "on_demand_price": 14.40,
             "region": "us-east-1"},
        ])
        options = router.get_all_options()
        assert len(options) == 1
        assert "source" in options[0]


# ── DisaggregatedRouter ──────────────────────────────────────────────────────


class TestDisaggregatedRouter:
    def test_init(self):
        router = DisaggregatedRouter()
        assert router._prefill_nodes == []
        assert router._decode_nodes == []
        assert router._stats == {"prefill_routes": 0, "decode_routes": 0, "fallback_routes": 0}

    def test_set_prefill_nodes(self):
        router = DisaggregatedRouter()
        router.set_prefill_nodes(["node-a", "node-b"])
        assert router._prefill_nodes == ["node-a", "node-b"]

    def test_set_decode_nodes(self):
        router = DisaggregatedRouter()
        router.set_decode_nodes(["node-c", "node-d"])
        assert router._decode_nodes == ["node-c", "node-d"]

    def test_route_prefill(self):
        router = DisaggregatedRouter()
        router.set_prefill_nodes(["node-a", "node-b"])
        node = router.route(RequestPhase.PREFILL)
        assert node in ("node-a", "node-b")

    def test_route_decode(self):
        router = DisaggregatedRouter()
        router.set_decode_nodes(["node-c", "node-d"])
        node = router.route(RequestPhase.DECODE)
        assert node in ("node-c", "node-d")

    def test_route_no_nodes(self):
        router = DisaggregatedRouter()
        assert router.route(RequestPhase.PREFILL) is None

    def test_route_fallback_to_decode_pool(self):
        router = DisaggregatedRouter()
        router.set_decode_nodes(["node-c"])
        # No prefill nodes -> fallback to decode pool
        node = router.route(RequestPhase.PREFILL)
        assert node == "node-c"

    def test_route_fallback_to_prefill_pool(self):
        router = DisaggregatedRouter()
        router.set_prefill_nodes(["node-a"])
        node = router.route(RequestPhase.DECODE)
        assert node == "node-a"

    def test_route_least_loaded(self):
        router = DisaggregatedRouter()
        router.set_prefill_nodes(["node-a", "node-b"])
        # Pre-load one node
        router._prefill_load["node-a"] = 5
        # Should pick node-b (less loaded)
        node = router.route(RequestPhase.PREFILL)
        assert node == "node-b"

    def test_release_prefill(self):
        router = DisaggregatedRouter()
        router.set_prefill_nodes(["node-a"])
        router._prefill_load["node-a"] = 3
        router.release("node-a", RequestPhase.PREFILL)
        assert router._prefill_load["node-a"] == 2

    def test_release_decode(self):
        router = DisaggregatedRouter()
        router.set_decode_nodes(["node-b"])
        router._decode_load["node-b"] = 3
        router.release("node-b", RequestPhase.DECODE)
        assert router._decode_load["node-b"] == 2

    def test_release_below_zero(self):
        router = DisaggregatedRouter()
        router.set_prefill_nodes(["node-a"])
        router._prefill_load["node-a"] = 0
        router.release("node-a", RequestPhase.PREFILL)
        assert router._prefill_load["node-a"] == 0  # max(0, -1) = 0

    def test_get_pool_sizes(self):
        router = DisaggregatedRouter()
        router.set_prefill_nodes(["a", "b", "c"])
        router.set_decode_nodes(["d", "e"])
        sizes = router.get_pool_sizes()
        assert sizes["prefill_nodes"] == 3
        assert sizes["decode_nodes"] == 2
        assert sizes["total_nodes"] == 5

    def test_stats(self):
        router = DisaggregatedRouter()
        router.set_prefill_nodes(["a"])
        router.set_decode_nodes(["b"])
        router.route(RequestPhase.PREFILL)
        router.route(RequestPhase.DECODE)
        assert router.stats["prefill_routes"] == 1
        assert router.stats["decode_routes"] == 1

    def test_fallback_stats(self):
        router = DisaggregatedRouter()
        router.set_prefill_nodes(["a"])
        router.route(RequestPhase.DECODE)  # falls to prefill pool
        assert router.stats["fallback_routes"] == 1
