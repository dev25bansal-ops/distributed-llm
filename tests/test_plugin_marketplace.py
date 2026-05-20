"""Tests for the DistLLM plugin marketplace infrastructure."""

import json
import pytest

from distllm.plugins.metadata import PluginMetadata, PluginManifest, validate_metadata
from distllm.plugins.config_schema import PluginConfigValidator
from distllm.plugins.compatibility import CompatibilityChecker, CompatibilityResult
from distllm.plugins.installer import PluginInstaller, PluginInstallResult
from distllm.plugins.sandbox import PluginSandbox, SandboxContext, SandboxStats
from distllm.plugins.telemetry import PluginTelemetry, PluginStats


class TestPluginMetadata:
    def test_valid_metadata(self):
        meta = PluginMetadata(
            name="test-plugin",
            version="1.0.0",
            description="A test plugin",
            author="Test Author",
            license="MIT",
            entry_point="distllm_plugins.test.TestPlugin",
            categories=["observability"],
        )
        errors = meta.validate()
        assert errors == []

    def test_invalid_name(self):
        meta = PluginMetadata(name="INVALID NAME", entry_point="mod.Class")
        errors = meta.validate()
        assert any("name" in e for e in errors)

    def test_invalid_version(self):
        meta = PluginMetadata(name="test", version="not-a-version", entry_point="mod.Class")
        errors = meta.validate()
        assert any("version" in e for e in errors)

    def test_missing_entry_point(self):
        meta = PluginMetadata(name="test", version="1.0.0")
        errors = meta.validate()
        assert any("entry_point" in e for e in errors)

    def test_invalid_category(self):
        meta = PluginMetadata(
            name="test",
            version="1.0.0",
            entry_point="mod.Class",
            categories=["nonexistent-category"],
        )
        errors = validate_metadata(meta)
        assert any("category" in e for e in errors)

    def test_to_dict_roundtrip(self):
        meta = PluginMetadata(
            name="test",
            version="1.0.0",
            entry_point="mod.Class",
            description="Test desc",
        )
        d = meta.to_dict()
        restored = PluginMetadata.from_dict(d)
        assert restored.name == meta.name
        assert restored.version == meta.version
        assert restored.entry_point == meta.entry_point


class TestPluginConfigValidator:
    def test_valid_config(self):
        validator = PluginConfigValidator()
        validator.register_schema("test", {
            "type": "object",
            "properties": {
                "log_level": {"type": "string", "enum": ["DEBUG", "INFO", "WARNING"]},
                "max_retries": {"type": "integer", "minimum": 0, "maximum": 10},
            },
            "required": ["log_level"],
        })
        errors = validator.validate_config("test", {"log_level": "INFO", "max_retries": 3})
        assert errors == []

    def test_missing_required(self):
        validator = PluginConfigValidator()
        validator.register_schema("test", {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        })
        errors = validator.validate_config("test", {})
        assert any("required" in e for e in errors)

    def test_type_mismatch(self):
        validator = PluginConfigValidator()
        validator.register_schema("test", {
            "type": "object",
            "properties": {"count": {"type": "integer"}},
        })
        errors = validator.validate_config("test", {"count": "not-an-int"})
        assert any("expected type" in e for e in errors)

    def test_range_violation(self):
        validator = PluginConfigValidator()
        validator.register_schema("test", {
            "type": "object",
            "properties": {"threshold": {"type": "number", "minimum": 0, "maximum": 1}},
        })
        errors = validator.validate_config("test", {"threshold": 1.5})
        assert any("above maximum" in e for e in errors)

    def test_default_config(self):
        validator = PluginConfigValidator()
        validator.register_schema("test", {
            "type": "object",
            "properties": {
                "log_level": {"type": "string", "default": "INFO"},
                "timeout": {"type": "integer", "default": 30},
            },
        })
        defaults = validator.get_default_config("test")
        assert defaults == {"log_level": "INFO", "timeout": 30}

    def test_no_schema_registered(self):
        validator = PluginConfigValidator()
        errors = validator.validate_config("unknown", {})
        assert any("No config schema" in e for e in errors)


class TestCompatibilityChecker:
    def test_compatible(self):
        checker = CompatibilityChecker(host_version="0.5.0")
        result = checker.check_compatibility(
            min_host_version="0.3.0",
            max_host_version="0.6.0",
        )
        assert result.compatible
        assert result.errors == []

    def test_host_version_too_old(self):
        checker = CompatibilityChecker(host_version="0.2.0")
        result = checker.check_compatibility(min_host_version="0.3.0")
        assert not result.compatible
        assert any("below minimum" in e for e in result.errors)

    def test_host_version_too_new(self):
        checker = CompatibilityChecker(host_version="0.7.0")
        result = checker.check_compatibility(max_host_version="0.6.0")
        assert not result.compatible
        assert any("above maximum" in e for e in result.errors)

    def test_missing_dependency(self):
        checker = CompatibilityChecker(host_version="0.5.0")
        result = checker.check_compatibility(
            dependencies=["this-package-definitely-does-not-exist-xyz123"],
        )
        assert not result.compatible
        assert any("Missing dependency" in e for e in result.errors)


class TestPluginSandbox:
    def test_successful_sync(self):
        sandbox = PluginSandbox()
        def success_fn():
            pass
        stats = sandbox.run_hook_sync("test", "on_request", success_fn)
        assert stats.success
        assert stats.error == ""

    def test_failed_sync(self):
        sandbox = PluginSandbox()
        def fail_fn():
            raise ValueError("test error")
        stats = sandbox.run_hook_sync("test", "on_request", fail_fn)
        assert not stats.success
        assert "test error" in stats.error

    def test_stats_tracking(self):
        sandbox = PluginSandbox()
        def fn():
            pass
        sandbox.run_hook_sync("plugin-a", "on_request", fn)
        sandbox.run_hook_sync("plugin-a", "on_response", fn)
        sandbox.run_hook_sync("plugin-b", "on_request", fn)

        stats_a = sandbox.get_stats("plugin-a")
        assert len(stats_a) == 2

        all_stats = sandbox.get_stats()
        assert len(all_stats) == 3

    def test_clear_stats(self):
        sandbox = PluginSandbox()
        def fn():
            pass
        sandbox.run_hook_sync("test", "on_request", fn)
        sandbox.clear_stats()
        assert sandbox.get_stats() == []


class TestPluginTelemetry:
    def test_record_and_stats(self):
        telemetry = PluginTelemetry()
        telemetry.record_usage("plugin-a", "on_request", 15.5, True)
        telemetry.record_usage("plugin-a", "on_request", 25.0, False, error="timeout")
        telemetry.record_usage("plugin-b", "on_response", 10.0, True)

        stats_a = telemetry.get_plugin_stats("plugin-a")
        assert stats_a.total_executions == 2
        assert stats_a.failed_executions == 1
        assert stats_a.error_rate == 0.5

    def test_error_rates(self):
        telemetry = PluginTelemetry()
        telemetry.record_usage("a", "hook", 1.0, True)
        telemetry.record_usage("a", "hook", 1.0, False)
        telemetry.record_usage("b", "hook", 1.0, True)

        rates = telemetry.get_error_rates()
        assert rates["a"] == 0.5
        assert rates["b"] == 0.0

    def test_recent_errors(self):
        telemetry = PluginTelemetry()
        telemetry.record_usage("a", "hook", 1.0, False, error="err1")
        telemetry.record_usage("a", "hook", 1.0, False, error="err2")
        telemetry.record_usage("a", "hook", 1.0, True)

        errors = telemetry.get_recent_errors()
        assert len(errors) == 2
        assert all(not e.success for e in errors)

    def test_export_json(self):
        telemetry = PluginTelemetry()
        telemetry.record_usage("test", "hook", 10.0, True)
        data = json.loads(telemetry.export_json())
        assert "test" in data
        assert data["test"]["total_executions"] == 1

    def test_reset(self):
        telemetry = PluginTelemetry()
        telemetry.record_usage("a", "hook", 1.0, True)
        telemetry.record_usage("b", "hook", 1.0, True)
        telemetry.reset("a")
        assert telemetry.get_plugin_stats("a").total_executions == 0
        assert telemetry.get_plugin_stats("b").total_executions == 1


class TestPluginManagerMarketplace:
    def test_context_backward_compat(self):
        from distllm.core.plugin import PluginManager, PluginContext
        mgr = PluginManager(context=PluginContext(config={"key": "value"}))
        assert mgr._context.get("key") == "value"

    def test_dict_context_backward_compat(self):
        from distllm.core.plugin import PluginManager
        mgr = PluginManager(context={"key": "value"})
        assert mgr._context == {"key": "value"}

    def test_metadata_registry(self):
        from distllm.core.plugin import PluginManager, PluginContext
        from distllm.plugins.metadata import PluginMetadata

        class PluginWithMeta:
            name = "meta-plugin"
            version = "1.0.0"
            description = "Has metadata"

            @property
            def metadata(self):
                return PluginMetadata(name="meta-plugin", version="1.0.0", entry_point="mod.Class")

            def initialize(self, context):
                pass

            def shutdown(self):
                pass

        mgr = PluginManager()
        mgr.register_plugin(PluginWithMeta())
        meta = mgr.get_plugin_metadata("meta-plugin")
        assert meta is not None
        assert meta.name == "meta-plugin"

    def test_config_validation_on_register(self):
        from distllm.core.plugin import PluginManager, PluginContext
        from distllm.plugins.config_schema import PluginConfigValidator

        class SimplePlugin:
            name = "config-plugin"
            version = "1.0.0"
            description = ""

            def initialize(self, context):
                pass

            def shutdown(self):
                pass

        mgr = PluginManager()
        validator = PluginConfigValidator()
        validator.register_schema("config-plugin", {
            "type": "object",
            "properties": {"required_field": {"type": "string"}},
            "required": ["required_field"],
        })
        mgr.set_marketplace_subsystems(config_validator=validator)

        # Should fail without required field
        with pytest.raises(ValueError, match="Invalid plugin config"):
            mgr.register_plugin(SimplePlugin(), config={})

        # Should succeed with valid config
        mgr.register_plugin(SimplePlugin(), config={"required_field": "ok"})
        assert mgr.get_plugin("config-plugin") is not None

    def test_sandboxed_hook_emit(self):
        import asyncio
        from distllm.core.plugin import PluginManager, PluginContext, HookPoint
        from distllm.plugins.sandbox import PluginSandbox
        from distllm.plugins.telemetry import PluginTelemetry

        class HookPlugin:
            name = "hook-plugin"
            version = "1.0.0"
            description = ""

            def initialize(self, context):
                if hasattr(context, "hooks") and context.hooks:
                    context.hooks.register(HookPoint.ON_REQUEST, self._handler)

            def _handler(self, request):
                pass

            def shutdown(self):
                pass

        mgr = PluginManager()
        sandbox = PluginSandbox()
        telemetry = PluginTelemetry()
        mgr.set_marketplace_subsystems(sandbox=sandbox, telemetry=telemetry)
        mgr.register_plugin(HookPlugin())

        # Test sync emit still works
        mgr.emit_hook(HookPoint.ON_REQUEST, "test-request")

        # Test sandboxed emit
        async def run():
            await mgr.emit_hook_sandboxed(HookPoint.ON_REQUEST, "test-request")

        asyncio.get_event_loop().run_until_complete(run())
        assert telemetry.get_plugin_stats("hook-plugin").total_executions >= 1
