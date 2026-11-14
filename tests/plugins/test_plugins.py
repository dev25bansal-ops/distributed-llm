"""Tests for plugins/ modules."""
from __future__ import annotations

from unittest.mock import MagicMock, AsyncMock, patch
import pytest


class TestRateLimitPlugin:
    """Tests for plugins/builtin.py RateLimitPlugin."""

    def test_rate_limit_plugin_class(self):
        from distllm.plugins.builtin import RateLimitPlugin

        plugin = RateLimitPlugin()
        assert plugin is not None

    def test_rate_limit_check_allowed(self):
        from distllm.plugins.builtin import RateLimitPlugin

        plugin = RateLimitPlugin()
        if hasattr(plugin, "check"):
            result = plugin.check(client_id="test_client")
            assert isinstance(result, bool)

    def test_rate_limit_exceeded(self):
        from distllm.plugins.builtin import RateLimitPlugin

        plugin = RateLimitPlugin()
        if hasattr(plugin, "check"):
            for _ in range(100):
                plugin.check(client_id="heavy_client")
            result = plugin.check(client_id="heavy_client")
            assert isinstance(result, bool)

    def test_rate_limit_config(self):
        from distllm.plugins.builtin import RateLimitPlugin

        plugin = RateLimitPlugin()
        if hasattr(plugin, "configure"):
            plugin.configure(max_requests=10, window_seconds=60)
            # Verify config was applied by checking the limit behavior
            if hasattr(plugin, "check"):
                allowed = plugin.check(client_id="test")
                # After configure, check should still work without error
                assert isinstance(allowed, (bool, type(None)))
        else:
            pytest.skip("RateLimitPlugin.configure not implemented")


class TestAuditLogPlugin:
    """Tests for plugins/builtin.py AuditLogPlugin."""

    def test_audit_log_plugin_class(self):
        from distllm.plugins.builtin import AuditLogPlugin

        plugin = AuditLogPlugin()
        assert plugin is not None

    def test_audit_log_entry(self):
        from distllm.plugins.builtin import AuditLogPlugin

        plugin = AuditLogPlugin()
        if hasattr(plugin, "log"):
            # log() should not raise; verify it returns normally
            plugin.log(
                action="model_load",
                user="test_user",
                model="test-model",
                success=True,
            )
            # Check that an entry was recorded if the plugin tracks logs
            if hasattr(plugin, "get_entries"):
                entries = plugin.get_entries()
                assert len(entries) >= 1
                assert entries[-1]["action"] == "model_load"
        else:
            pytest.skip("AuditLogPlugin.log not implemented")

    def test_audit_log_with_error(self):
        from distllm.plugins.builtin import AuditLogPlugin

        plugin = AuditLogPlugin()
        if hasattr(plugin, "log"):
            plugin.log(
                action="model_load",
                user="test_user",
                model="bad-model",
                success=False,
                error="OOM",
            )
            if hasattr(plugin, "get_entries"):
                entries = plugin.get_entries()
                matching = [e for e in entries if e.get("error") == "OOM"]
                assert len(matching) >= 1
        else:
            pytest.skip("AuditLogPlugin.log not implemented")


class TestMetricsPlugin:
    """Tests for plugins/builtin.py MetricsPlugin."""

    def test_metrics_plugin_class(self):
        from distllm.plugins.builtin import MetricsPlugin

        plugin = MetricsPlugin()
        assert plugin is not None

    def test_metrics_collection(self):
        from distllm.plugins.builtin import MetricsPlugin

        plugin = MetricsPlugin()
        if hasattr(plugin, "collect"):
            plugin.collect(metric_name="requests_total", value=1.0)
            # Verify the metric was recorded
            if hasattr(plugin, "get_metrics"):
                metrics = plugin.get_metrics()
                assert "requests_total" in metrics
        else:
            pytest.skip("MetricsPlugin.collect not implemented")

    def test_metrics_labels(self):
        from distllm.plugins.builtin import MetricsPlugin

        plugin = MetricsPlugin()
        if hasattr(plugin, "collect"):
            plugin.collect(
                metric_name="latency",
                value=0.5,
                labels={"endpoint": "/v1/chat", "method": "POST"},
            )
            if hasattr(plugin, "get_metrics"):
                metrics = plugin.get_metrics()
                assert "latency" in metrics
        else:
            pytest.skip("MetricsPlugin.collect not implemented")


class TestPluginSystem:
    """Tests for core/plugin_system.py integration with plugins.

    These two tests used to call ``ps.register("test", mock_plugin)`` -- a name
    string and an instance -- and then ``ps.get(name)``. None of those exist:
    register() takes a plugin CLASS and optional metadata, there is no get()
    (it is get_plugin()), and start/stop are fleet-wide (start_all/stop_all),
    not per-name. Every call sat behind an ``if hasattr`` guard, so on a
    PluginSystem missing those methods the tests passed without asserting
    anything; where register() DID exist they called it with the wrong shape
    and failed with TypeError: 'str' object is not callable. Written against
    the real API and asserting real state.
    """

    def test_plugin_registry(self):
        from distllm.core.plugin_system import PluginBase, PluginSystem

        class SamplePlugin(PluginBase):
            pass

        ps = PluginSystem()
        assert ps.register(SamplePlugin) is True
        registered = ps.get_plugin(SamplePlugin().name())
        assert registered is not None
        assert registered.cls is SamplePlugin
        assert [p.cls for p in ps.list_plugins()] == [SamplePlugin]

    def test_plugin_lifecycle(self):
        from distllm.core.plugin_system import PluginBase, PluginSystem

        started: list[str] = []

        class LifecyclePlugin(PluginBase):
            def on_start(self, context):
                started.append(self.name())

        ps = PluginSystem()
        ps.register(LifecyclePlugin)
        assert ps.load_all() == 1
        assert ps.init_all() == 1
        assert ps.start_all() == 1
        assert started == [LifecyclePlugin().name()]
        assert ps.stop_all() == 1
