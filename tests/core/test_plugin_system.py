"""Tests for PluginSystem, PluginBase, PluginState, PluginMetadata.

No mocks -- uses real PluginBase subclasses and direct state inspection.
"""

from __future__ import annotations

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_mod = load_module("distllm/core/plugin_system.py")
PluginSystem = _mod.PluginSystem
PluginBase = _mod.PluginBase
PluginState = _mod.PluginState
PluginMetadata = _mod.PluginMetadata
PluginInstance = _mod.PluginInstance
PluginContext = _mod.PluginContext


class TestPluginState:
    """PluginState enum values."""

    def test_state_values(self) -> None:
        assert PluginState.DISCOVERED.value == "discovered"
        assert PluginState.LOADED.value == "loaded"
        assert PluginState.INITIALIZED.value == "initialized"
        assert PluginState.STARTED.value == "started"
        assert PluginState.STOPPED.value == "stopped"
        assert PluginState.ERROR.value == "error"


class TestPluginMetadata:
    """PluginMetadata dataclass."""

    def test_default_construction(self) -> None:
        meta = PluginMetadata(name="test-plugin")
        assert meta.name == "test-plugin"
        assert meta.version == "1.0.0"
        assert meta.description == ""
        assert meta.author == ""
        assert meta.entry_point == ""
        assert meta.dependencies == []
        assert meta.tags == []

    def test_creation_with_all_fields(self) -> None:
        meta = PluginMetadata(
            name="my-plugin", version="2.1.0", description="does stuff",
            author="me", entry_point="mymod:MyPlugin",
            dependencies=["other-plugin"], tags=["auth", "middleware"],
        )
        assert meta.author == "me"
        assert "auth" in meta.tags


class TestPluginBase:
    """PluginBase default behavior."""

    def test_name_uses_class_name(self) -> None:
        class MyPlugin(PluginBase):
            pass

        inst = MyPlugin()
        assert inst.name() == "MyPlugin"

    def test_version_default(self) -> None:
        inst = PluginBase()
        assert inst.version() == "1.0.0"

    def test_metadata(self) -> None:
        inst = PluginBase()
        meta = inst.metadata()
        assert meta.name == "PluginBase"
        assert meta.version == "1.0.0"

    def test_lifecycle_hooks_return_none(self) -> None:
        inst = PluginBase()
        assert inst.on_init({}) is None
        assert inst.on_start({}) is None
        assert inst.on_stop({}) is None

    def test_event_hooks(self) -> None:
        inst = PluginBase()
        assert inst.on_request({}) is None
        assert inst.on_response({}, {}) is None
        assert inst.on_error({}, Exception("x")) is None
        assert inst.on_model_load("m", {}) is None
        assert inst.on_model_unload("m") is None
        assert inst.on_config_change("k", "old", "new") is None


class TestPluginSystemRegistration:
    """PluginSystem registration and lifecycle."""

    def test_default_construction(self) -> None:
        system = PluginSystem()
        assert system._plugins == {}
        assert system._config == {}
        assert system._started_cache is None

    def test_register_adds_plugin(self) -> None:
        system = PluginSystem()

        class SimplePlugin(PluginBase):
            def name(self) -> str:
                return "simple"

        meta = PluginMetadata(name="simple")
        system.register(SimplePlugin, meta)
        inst = system.get_plugin("simple")
        assert inst is not None
        assert inst.state == PluginState.DISCOVERED

    def test_register_returns_true(self) -> None:
        system = PluginSystem()

        class P(PluginBase):
            pass

        assert system.register(P) is True

    def test_list_plugins(self) -> None:
        system = PluginSystem()

        class P1(PluginBase):
            def name(self) -> str:
                return "p1"

        class P2(PluginBase):
            def name(self) -> str:
                return "p2"

        system.register(P1, PluginMetadata(name="p1"))
        system.register(P2, PluginMetadata(name="p2"))
        plugins = system.list_plugins()
        assert len(plugins) == 2
        names = {p.metadata.name for p in plugins}
        assert names == {"p1", "p2"}


class TestPluginSystemLifecycle:
    """PluginSystem load_all -> init_all -> start_all -> stop_all."""

    def test_load_all_instantiates_plugins(self) -> None:
        system = PluginSystem()

        class P(PluginBase):
            def name(self) -> str:
                return "p"

        system.register(P, PluginMetadata(name="p"))
        count = system.load_all()
        assert count == 1
        inst = system.get_plugin("p")
        assert inst is not None
        assert inst.instance is not None
        assert inst.state == PluginState.LOADED

    def test_load_all_skips_already_loaded(self) -> None:
        system = PluginSystem()

        class P(PluginBase):
            def name(self) -> str:
                return "p"

        system.register(P, PluginMetadata(name="p"))
        system.load_all()
        count = system.load_all()
        assert count == 0  # already loaded

    def test_init_all_calls_on_init(self) -> None:
        system = PluginSystem()
        inited = []

        class P(PluginBase):
            def name(self) -> str:
                return "p"

            def on_init(self, ctx):
                inited.append(True)

        system.register(P, PluginMetadata(name="p"))
        system.load_all()
        count = system.init_all()
        assert count == 1
        assert inited == [True]

    def test_start_all_calls_on_start(self) -> None:
        system = PluginSystem()
        started = []

        class P(PluginBase):
            def name(self) -> str:
                return "p"

            def on_start(self, ctx):
                started.append(True)

        system.register(P, PluginMetadata(name="p"))
        system.load_all()
        system.init_all()
        count = system.start_all()
        assert count == 1
        assert started == [True]

    def test_is_loaded_after_full_lifecycle(self) -> None:
        system = PluginSystem()

        class P(PluginBase):
            def name(self) -> str:
                return "p"

        system.register(P, PluginMetadata(name="p"))
        system.load_all()
        system.init_all()
        system.start_all()
        assert system.is_loaded("p") is True

    def test_stop_all_calls_on_stop(self) -> None:
        system = PluginSystem()
        stopped = []

        class P(PluginBase):
            def name(self) -> str:
                return "p"

            def on_stop(self, ctx):
                stopped.append(True)

        system.register(P, PluginMetadata(name="p"))
        system.load_all()
        system.init_all()
        system.start_all()
        count = system.stop_all()
        assert count == 1
        assert stopped == [True]
        assert system.is_loaded("p") is False


class TestPluginSystemDispatch:
    """Hook dispatch."""

    def test_dispatch_calls_on_request(self) -> None:
        system = PluginSystem()

        class ModPlugin(PluginBase):
            def name(self) -> str:
                return "mod"

            def on_request(self, ctx):
                modified = dict(ctx)
                modified["extra"] = True
                return modified

        system.register(ModPlugin, PluginMetadata(name="mod"))
        system.load_all()
        system.init_all()
        system.start_all()

        result = system.dispatch_on_request({"key": "val"})
        assert result["key"] == "val"
        assert result["extra"] is True

    def test_dispatch_returns_non_none_results(self) -> None:
        system = PluginSystem()

        class RPlugin(PluginBase):
            def name(self) -> str:
                return "r"

            def on_response(self, request, response):
                return "handled"

        system.register(RPlugin, PluginMetadata(name="r"))
        system.load_all()
        system.init_all()
        system.start_all()

        results = system.dispatch("on_response", {"req": 1}, {"resp": 2})
        assert results == ["handled"]

    def test_dispatch_on_nonexistent_hook_returns_empty(self) -> None:
        system = PluginSystem()

        class P(PluginBase):
            def name(self) -> str:
                return "p"

        system.register(P, PluginMetadata(name="p"))
        system.load_all()
        system.init_all()
        system.start_all()

        results = system.dispatch("no_such_hook")
        assert results == []


class TestPluginContext:
    """PluginContext construction."""

    def test_creation(self) -> None:
        system = PluginSystem()
        ctx = PluginContext(system)
        assert ctx.system is system
        assert ctx.data == {}
        assert ctx.config == {}

    def test_config_synced_from_system(self) -> None:
        system = PluginSystem(config={"debug": True})
        assert system._context.config["debug"] is True
