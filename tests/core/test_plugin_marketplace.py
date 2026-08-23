"""Tests for PluginMarketplace, PluginEntry, SamplingStrategy, RoutingStrategy.

No mocks -- pure state inspection and strategy registration.
"""

from __future__ import annotations

from typing import Any

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_mod = load_module("distllm/core/plugin_marketplace.py")
PluginMarketplace = _mod.PluginMarketplace
PluginEntry = _mod.PluginEntry
PluginCategory = _mod.PluginCategory
SamplingStrategy = _mod.SamplingStrategy
RoutingStrategy = _mod.RoutingStrategy


class TestPluginCategory:
    """PluginCategory enum."""

    def test_values(self) -> None:
        assert PluginCategory.BACKEND.value == "backend"
        assert PluginCategory.ROUTING.value == "routing"
        assert PluginCategory.SAMPLING.value == "sampling"
        assert PluginCategory.MIDDLEWARE.value == "middleware"
        assert PluginCategory.TRANSPORT.value == "transport"
        assert PluginCategory.CACHE.value == "cache"
        assert PluginCategory.MONITORING.value == "monitoring"
        assert PluginCategory.AUTH.value == "auth"
        assert PluginCategory.OTHER.value == "other"


class TestPluginEntry:
    """PluginEntry dataclass."""

    def test_default_construction(self) -> None:
        entry = PluginEntry(name="test", category=PluginCategory.OTHER)
        assert entry.name == "test"
        assert entry.version == "1.0.0"
        assert entry.installed is False
        assert entry.enabled is True
        assert entry.source == ""

    def test_creation_with_all_fields(self) -> None:
        entry = PluginEntry(
            name="vllm-backend", category=PluginCategory.BACKEND,
            version="0.1.0", description="vLLM integration",
            author="NVIDIA", entry_point="vllm_mod:VLLMBackend",
            dependencies=["torch"], tags=["gpu", "fast"],
            installed=True, enabled=False, source="pypi",
            config_schema={"type": "object"},
        )
        assert entry.author == "NVIDIA"
        assert entry.installed is True
        assert entry.enabled is False
        assert entry.source == "pypi"
        assert "gpu" in entry.tags


class TestSamplingStrategy:
    """SamplingStrategy dataclass."""

    def test_creation(self) -> None:
        def dummy_fn(logits, **kw):
            return 0

        strategy = SamplingStrategy(
            name="top-a", description="Top-A sampling",
            fn=dummy_fn, requires_params=["temperature"],
        )
        assert strategy.name == "top-a"
        assert strategy.fn is dummy_fn
        assert strategy.requires_params == ["temperature"]


class TestRoutingStrategy:
    """RoutingStrategy dataclass."""

    def test_creation(self) -> None:
        def dummy_fn(request, available_models, **kw):
            return available_models[0] if available_models else ""

        strategy = RoutingStrategy(
            name="fastest", description="Fastest model routing",
            fn=dummy_fn, requires_params=[],
        )
        assert strategy.name == "fastest"
        assert strategy.fn is dummy_fn


class TestPluginMarketplace:
    """PluginMarketplace -- construction and registry operations."""

    def test_default_construction(self) -> None:
        mp = PluginMarketplace(enable_pypi=False)
        assert mp._plugin_dirs == []
        assert mp._enable_pypi is False
        assert mp._registry == {}
        assert mp._sampling_strategies != {}
        assert mp._routing_strategies != {}

    def test_register_backend(self) -> None:
        mp = PluginMarketplace(enable_pypi=False)

        def factory():
            return object()

        mp.register_backend("my-backend", factory, description="My backend")
        entry = mp.get_plugin("my-backend")
        assert entry is not None
        assert entry.category == PluginCategory.BACKEND
        assert mp.get_backend_factory("my-backend") is factory

    def test_list_backends(self) -> None:
        mp = PluginMarketplace(enable_pypi=False)
        mp.register_backend("b1", lambda: None)
        mp.register_backend("b2", lambda: None)
        backends = mp.list_backends()
        assert "b1" in backends
        assert "b2" in backends

    def test_register_sampling_strategy(self) -> None:
        mp = PluginMarketplace(enable_pypi=False)

        def sample(logits, **kw):
            return 0

        mp.register_sampling_strategy("top-a", sample, "custom", ["alpha"])
        strategy = mp.get_sampling_strategy("top-a")
        assert strategy is not None
        assert strategy.name == "top-a"
        assert strategy.fn is sample

    def test_list_sampling_strategies(self) -> None:
        mp = PluginMarketplace(enable_pypi=False)
        names = mp.list_sampling_strategies()
        # Built-in strategies
        assert "greedy" in names
        assert "top-p" in names
        assert "min-p" in names

    def test_register_routing_strategy(self) -> None:
        mp = PluginMarketplace(enable_pypi=False)

        def router(request, models, **kw):
            return models[0] if models else ""

        mp.register_routing_strategy("my-router", router)
        strategy = mp.get_routing_strategy("my-router")
        assert strategy is not None
        assert strategy.name == "my-router"

    def test_list_routing_strategies(self) -> None:
        mp = PluginMarketplace(enable_pypi=False)
        names = mp.list_routing_strategies()
        assert "round-robin" in names
        assert "least-loaded" in names

    def test_enable_disable_plugin(self) -> None:
        mp = PluginMarketplace(enable_pypi=False)
        mp.register_backend("test-backend", lambda: None)
        assert mp.enable_plugin("test-backend") is True
        entry = mp.get_plugin("test-backend")
        assert entry is not None
        assert entry.enabled is True
        assert mp.disable_plugin("test-backend") is True
        assert entry.enabled is False

    def test_enable_disable_nonexistent(self) -> None:
        mp = PluginMarketplace(enable_pypi=False)
        assert mp.enable_plugin("nonexistent") is False
        assert mp.disable_plugin("nonexistent") is False

    def test_list_plugins_by_category(self) -> None:
        mp = PluginMarketplace(enable_pypi=False)
        mp.register_backend("b1", lambda: None)
        mp.register_backend("b2", lambda: None)

        backends = mp.list_plugins(category=PluginCategory.BACKEND)
        assert len(backends) == 2
        names = {p.name for p in backends}
        assert names == {"b1", "b2"}

    def test_stats(self) -> None:
        mp = PluginMarketplace(enable_pypi=False)
        mp.register_backend("b1", lambda: None)
        s = mp.stats()
        assert s["total_plugins"] >= 1
        assert s["custom_backends"] >= 1
        assert s["sampling_strategies"] == 3  # built-in: greedy, top-p, min-p
        assert s["routing_strategies"] == 2  # built-in: round-robin, least-loaded

    def test_get_plugin_nonexistent(self) -> None:
        mp = PluginMarketplace(enable_pypi=False)
        assert mp.get_plugin("nonexistent") is None

    def test_get_backend_factory_nonexistent(self) -> None:
        mp = PluginMarketplace(enable_pypi=False)
        assert mp.get_backend_factory("nonexistent") is None

    def test_strategies_are_callable(self) -> None:
        mp = PluginMarketplace(enable_pypi=False)
        greedy = mp.get_sampling_strategy("greedy")
        assert greedy is not None
        # Should be callable with dummy logits-like object
        rr = mp.get_routing_strategy("round-robin")
        assert rr is not None
