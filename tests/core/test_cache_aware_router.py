"""Tests for CacheAwareRouter.

Uses the import-helper pattern to avoid circular imports.
"""

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_router_mod = load_module("distllm/core/cache_aware_router.py")
CacheAwareRouter = _router_mod.CacheAwareRouter


class TestCacheAwareRouterInit:
    """CacheAwareRouter construction and defaults."""

    def test_defaults(self):
        router = CacheAwareRouter()
        assert router._cache_weight == 0.7
        assert router._load_weight == 0.3
        assert router._route_stats == {}

    def test_custom_weights(self):
        router = CacheAwareRouter(cache_weight=0.5, load_weight=0.5)
        assert router._cache_weight == 0.5
        assert router._load_weight == 0.5

    def test_weight_ratio_bounds(self):
        router = CacheAwareRouter(cache_weight=0.0, load_weight=1.0)
        assert router._cache_weight == 0.0
        assert router._load_weight == 1.0

    def test_get_route_stats_empty(self):
        router = CacheAwareRouter()
        assert router.get_route_stats() == {}


class TestCacheAwareRouterGetLoadScore:
    """_get_load_score with dict and non-dict node info."""

    def test_dict_with_load(self):
        router = CacheAwareRouter()
        score = router._get_load_score({"load": 0.8})
        assert score == 0.8

    def test_dict_without_load(self):
        router = CacheAwareRouter()
        score = router._get_load_score({"other": 42})
        assert score == 0.5

    def test_dict_default(self):
        router = CacheAwareRouter()
        score = router._get_load_score({})
        assert score == 0.5

    def test_object_with_load_attr(self):
        router = CacheAwareRouter()
        node = type("Node", (), {"load": 0.3})()
        score = router._get_load_score(node)
        assert score == 0.3

    def test_object_without_load_attr(self):
        router = CacheAwareRouter()
        node = type("Node", (), {})()
        score = router._get_load_score(node)
        assert score == 0.5

    def test_none(self):
        router = CacheAwareRouter()
        score = router._get_load_score(None)
        assert score == 0.5


class TestCacheAwareRouterCheckCacheAffinity:
    """_check_cache_affinity with and without cache_manager."""

    async def test_no_cache_manager(self):
        router = CacheAwareRouter()
        score = await router._check_cache_affinity("node-1", [1, 2, 3], None)
        assert score == 0.0

    async def test_cache_manager_without_prefix_cache(self):
        router = CacheAwareRouter()
        cm = type("CM", (), {"prefix_cache": None})()
        score = await router._check_cache_affinity("node-1", [1, 2, 3], cm)
        assert score == 0.0

    async def test_cache_manager_prefix_cache_hit(self):
        router = CacheAwareRouter()
        prefix_cache = type("PC", (), {})()
        prefix_cache.lookup = lambda tokens: (3, "data")
        cm = type("CM", (), {"prefix_cache": prefix_cache})()
        score = await router._check_cache_affinity("node-1", [1, 2, 3], cm)
        assert score == 1.0  # 3/3 match -> min(1.0, 1.0)

    async def test_cache_manager_partial_hit(self):
        router = CacheAwareRouter()
        prefix_cache = type("PC", (), {})()
        prefix_cache.lookup = lambda tokens: (2, "data")
        cm = type("CM", (), {"prefix_cache": prefix_cache})()
        score = await router._check_cache_affinity("node-1", [1, 2, 3], cm)
        assert score == 2.0 / 3.0

    async def test_cache_manager_miss(self):
        router = CacheAwareRouter()
        prefix_cache = type("PC", (), {})()
        prefix_cache.lookup = lambda tokens: (0, None)
        cm = type("CM", (), {"prefix_cache": prefix_cache})()
        score = await router._check_cache_affinity("node-1", [1, 2, 3], cm)
        assert score == 0.0


class TestCacheAwareRouterRoute:
    """route() - selecting best node."""

    async def test_no_nodes_returns_none(self):
        router = CacheAwareRouter()
        result = await router.route([1, 2, 3], {})
        assert result is None

    async def test_single_node(self):
        router = CacheAwareRouter()
        result = await router.route(
            [1, 2, 3],
            {"node-1": {"load": 0.5}},
        )
        assert result == "node-1"

    async def test_two_nodes_selects_better(self):
        """Node with lower load AND no cache manager should still pick the
        one with better combined score. Since cache_score=0 for both, higher
        weight goes to lower load (load_score inverted with 1.0-load_score)."""
        router = CacheAwareRouter()
        nodes = {
            "busy-node": {"load": 0.9},
            "idle-node": {"load": 0.1},
        }
        result = await router.route([1, 2, 3], nodes)
        # idle-node has (1 - 0.1) = 0.9 inverted load vs busy (1 - 0.9) = 0.1
        # Cache_score is 0 for both, so combined = load_weight * (1-load_score)
        assert result == "idle-node"

    async def test_cache_affinity_same_cache_lighter_wins(self):
        """With identical cache affinity, lighter-loaded node wins."""
        router = CacheAwareRouter(cache_weight=0.5, load_weight=0.5)
        prefix_cache = type("PC", (), {})()
        prefix_cache.lookup = lambda tokens: (3, "data")
        cm_with_cache = type("CM", (), {"prefix_cache": prefix_cache})()

        nodes = {
            "busy-node": {"load": 0.9},
            "idle-node": {"load": 0.1},
        }
        result = await router.route(
            [1, 2, 3],
            nodes,
            cache_manager=cm_with_cache,
        )
        assert result == "idle-node"

    async def test_route_stats_tracked(self):
        router = CacheAwareRouter()
        nodes = {"node-1": {"load": 0.5}, "node-2": {"load": 0.4}}
        await router.route([1, 2, 3], nodes)
        stats = router.get_route_stats()
        # Only the selected node gets stats
        selected = "node-2"  # lower load -> higher combined score
        assert selected in stats
        assert stats[selected]["routed"] == 1

    async def test_route_stats_multiple_routes(self):
        router = CacheAwareRouter()
        nodes = {"node-1": {"load": 0.5}}
        for _ in range(3):
            await router.route([1, 2, 3], nodes)
        stats = router.get_route_stats()
        assert stats["node-1"]["routed"] == 3
