"""Tests for Feature 28: Plugin System."""

from unittest.mock import MagicMock, call

import pytest

from distllm.core.plugin import (
    PluginManager,
    HookRegistry,
    IPlugin,
    HookPoint,
    RequestLoggingPlugin,
    MetricsPlugin,
    HealthCheckPlugin,
    BUILTIN_PLUGINS,
)


# --- HookRegistry Tests ---

class TestHookRegistry:
    def test_register_and_emit(self):
        registry = HookRegistry()
        results = []

        def callback(x):
            results.append(x * 2)
            return x * 2

        registry.register("test_hook", callback)
        registry.emit("test_hook", 5)
        assert results == [10]

    def test_unregister(self):
        registry = HookRegistry()
        called = []

        def cb1():
            called.append(1)

        def cb2():
            called.append(2)

        registry.register("hook", cb1)
        registry.register("hook", cb2)
        registry.unregister("hook", cb1)
        registry.emit("hook")
        assert called == [2]

    def test_emit_multiple_callbacks(self):
        registry = HookRegistry()
        results = []

        registry.register("hook", lambda: results.append(1))
        registry.register("hook", lambda: results.append(2))
        registry.register("hook", lambda: results.append(3))
        registry.emit("hook")
        assert results == [1, 2, 3]

    def test_emit_with_exception(self):
        registry = HookRegistry()
        results = []

        registry.register("hook", lambda: 1)
        registry.register("hook", lambda: 1 / 0)
        registry.register("hook", lambda: results.append("ok"))

        returned = registry.emit("hook")
        # The exception is caught, but other callbacks still run
        assert "ok" in results

    def test_emit_nonexistent_hook(self):
        registry = HookRegistry()
        result = registry.emit("nonexistent_hook", "arg")
        assert result == []

    def test_list_hooks(self):
        registry = HookRegistry()
        registry.register("hook_a", lambda: None)
        registry.register("hook_a", lambda: None)
        registry.register("hook_b", lambda: None)

        counts = registry.list_hooks()
        assert counts["hook_a"] == 2
        assert counts["hook_b"] == 1

    def test_register_kwargs(self):
        registry = HookRegistry()
        result = []

        registry.register("hook", lambda x, y: result.append(x + y))
        registry.emit("hook", 3, y=7)
        assert result == [10]


# --- PluginManager Tests ---

class TestPluginManager:
    def test_register_plugin(self):
        manager = PluginManager()

        class TestPlugin:
            name = "test"
            version = "1.0"
            description = "Test plugin"

            def initialize(self, context):
                context["initialized"] = True

            def shutdown(self):
                pass

        manager.register_plugin(TestPlugin())
        assert manager.get_plugin("test") is not None
        assert manager._context.get("initialized") is True

    def test_unregister_plugin(self):
        manager = PluginManager()

        class TestPlugin:
            name = "test"
            version = "1.0"

            def initialize(self, context):
                pass

            def shutdown(self):
                context.clear() if isinstance(context, dict) else None

        manager.register_plugin(TestPlugin())
        manager.unregister_plugin("test")
        assert manager.get_plugin("test") is None

    def test_unregister_nonexistent(self):
        manager = PluginManager()
        manager.unregister_plugin("nonexistent")

    def test_replace_plugin(self):
        manager = PluginManager()

        class Plugin1:
            name = "test"
            version = "1.0"

            def initialize(self, context):
                context["v"] = 1

            def shutdown(self):
                pass

        class Plugin2:
            name = "test"
            version = "2.0"

            def initialize(self, context):
                context["v"] = 2

            def shutdown(self):
                pass

        manager.register_plugin(Plugin1())
        manager.register_plugin(Plugin2())
        assert manager._context.get("v") == 2

    def test_list_plugins(self):
        manager = PluginManager()

        class P1:
            name = "p1"
            version = "0.1"
            description = "Plugin 1"

            def initialize(self, ctx):
                pass

            def shutdown(self):
                pass

        class P2:
            name = "p2"
            version = "0.2"
            description = "Plugin 2"

            def initialize(self, ctx):
                pass

            def shutdown(self):
                pass

        manager.register_plugin(P1())
        manager.register_plugin(P2())
        plugins = manager.list_plugins()
        assert len(plugins) == 2
        assert plugins[0]["name"] == "p1"
        assert plugins[1]["name"] == "p2"

    def test_emit_hook(self):
        manager = PluginManager()
        manager.hooks.register("test", lambda x: x * 2)
        results = manager.emit_hook("test", 5)
        assert results == [10]

    def test_shutdown_all(self):
        manager = PluginManager()

        class TestPlugin:
            name = "test"
            version = "1.0"

            def initialize(self, ctx):
                ctx["shutdown"] = False

            def shutdown(self):
                manager._context["shutdown"] = True

        manager.register_plugin(TestPlugin())
        manager.shutdown_all()
        assert manager._context.get("shutdown") is True
        assert len(manager.list_plugins()) == 0

    def test_discover_from_config(self):
        manager = PluginManager()

        config = {
            "plugins": [
                {
                    "module": "distllm.core.plugin.RequestLoggingPlugin",
                    "config": {"log_level": "DEBUG"},
                }
            ]
        }
        plugins = manager.discover_from_config(config)
        assert len(plugins) == 1
        assert plugins[0].name == "request_logger"

    def test_discover_from_config_string_format(self):
        manager = PluginManager()

        config = {
            "plugins": ["distllm.core.plugin.MetricsPlugin"]
        }
        plugins = manager.discover_from_config(config)
        assert len(plugins) == 1
        assert plugins[0].name == "metrics_collector"

    def test_discover_from_config_invalid(self):
        manager = PluginManager()

        config = {
            "plugins": ["invalid_module.NonExistent"]
        }
        plugins = manager.discover_from_config(config)
        assert len(plugins) == 0


# --- Built-in Plugin Tests ---

class TestRequestLoggingPlugin:
    def test_initialization(self):
        hooks = HookRegistry()
        context = {"hooks": hooks}
        plugin = RequestLoggingPlugin(log_level="DEBUG")
        plugin.initialize(context)

        assert hooks.list_hooks().get(HookPoint.ON_REQUEST, 0) >= 1
        assert hooks.list_hooks().get(HookPoint.ON_RESPONSE, 0) >= 1

    def test_shutdown(self):
        plugin = RequestLoggingPlugin()
        plugin.shutdown()


class TestMetricsPlugin:
    def test_initialization(self):
        hooks = HookRegistry()
        context = {"hooks": hooks}
        plugin = MetricsPlugin()
        plugin.initialize(context)

        assert hooks.list_hooks().get(HookPoint.ON_REQUEST, 0) >= 1

    def test_count_requests(self):
        hooks = HookRegistry()
        context = {"hooks": hooks}
        plugin = MetricsPlugin()
        plugin.initialize(context)

        hooks.emit(HookPoint.ON_REQUEST, MagicMock())
        hooks.emit(HookPoint.ON_REQUEST, MagicMock())

        metrics = plugin.get_metrics()
        assert metrics.get("total_requests") == 2

    def test_shutdown_clears_metrics(self):
        plugin = MetricsPlugin()
        plugin._metrics["total_requests"] = 100
        plugin.shutdown()
        assert plugin.get_metrics() == {}


class TestHealthCheckPlugin:
    def test_initialization(self):
        hooks = HookRegistry()
        context = {"hooks": hooks}
        plugin = HealthCheckPlugin()
        plugin.initialize(context)

        assert hooks.list_hooks().get(HookPoint.ON_ERROR, 0) >= 1
        assert plugin.is_healthy() is True

    def test_error_marks_unhealthy(self):
        hooks = HookRegistry()
        context = {"hooks": hooks}
        plugin = HealthCheckPlugin()
        plugin.initialize(context)

        hooks.emit(HookPoint.ON_ERROR, ValueError("test error"))
        assert plugin.is_healthy() is False


class TestBuiltinPlugins:
    def test_all_builtin_plugins_have_required_attrs(self):
        for plugin_cls in BUILTIN_PLUGINS:
            plugin = plugin_cls()
            assert hasattr(plugin, "name")
            assert hasattr(plugin, "version")
            assert hasattr(plugin, "description")
            assert hasattr(plugin, "initialize")
            assert hasattr(plugin, "shutdown")

    def test_all_builtin_plugins_are_unique(self):
        names = [p().name for p in BUILTIN_PLUGINS]
        assert len(names) == len(set(names))


# --- Plugin Settings Tests ---

class TestPluginSettings:
    def test_default_settings(self):
        from distllm.config.settings import PluginSettings

        settings = PluginSettings()
        assert settings.enabled is False
        assert settings.plugins == []

    def test_valid_plugins_config(self):
        from distllm.config.settings import PluginSettings

        settings = PluginSettings(plugins=[
            {"module": "distllm.core.plugin.RequestLoggingPlugin", "config": {}}
        ])
        assert len(settings.plugins) == 1

    def test_invalid_plugin_module(self):
        from distllm.config.settings import PluginSettings

        with pytest.raises(ValueError, match="fully qualified"):
            PluginSettings(plugins=[{"module": "invalid_module"}])


# --- Integration Tests ---

class TestPluginIntegration:
    def test_full_lifecycle(self):
        manager = PluginManager(context={"counter": 0})

        class CounterPlugin:
            name = "counter"
            version = "1.0"
            description = "Counts hook calls"

            def initialize(self, ctx):
                manager.hooks.register(HookPoint.ON_REQUEST, self._on_request)

            def shutdown(self):
                pass

            def _on_request(self, req):
                manager._context["counter"] += 1

        manager.register_plugin(CounterPlugin())
        manager.emit_hook(HookPoint.ON_REQUEST, "test request")
        manager.emit_hook(HookPoint.ON_REQUEST, "another request")
        assert manager._context["counter"] == 2
        manager.shutdown_all()

    def test_multiple_plugins_same_hook(self):
        manager = PluginManager(context={"results": []})

        class PluginA:
            name = "a"
            version = "1.0"

            def initialize(self, ctx):
                manager.hooks.register(HookPoint.ON_RESPONSE, lambda r: ctx["results"].append("A"))

            def shutdown(self):
                pass

        class PluginB:
            name = "b"
            version = "1.0"

            def initialize(self, ctx):
                manager.hooks.register(HookPoint.ON_RESPONSE, lambda r: ctx["results"].append("B"))

            def shutdown(self):
                pass

        manager.register_plugin(PluginA())
        manager.register_plugin(PluginB())
        manager.emit_hook(HookPoint.ON_RESPONSE, "test response")
        assert set(manager._context["results"]) == {"A", "B"}
