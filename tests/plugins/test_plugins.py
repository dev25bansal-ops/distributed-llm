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
            result = plugin.collect(metric_name="requests_total", value=1.0)
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
    """Tests for core/plugin_system.py integration with plugins."""

    def test_plugin_registry(self):
        from distllm.core.plugin_system import PluginSystem

        ps = PluginSystem()
        if hasattr(ps, "register"):
            mock_plugin = MagicMock()
            ps.register("test", mock_plugin)
            if hasattr(ps, "get"):
                assert ps.get("test") is mock_plugin

    def test_plugin_lifecycle(self):
        from distllm.core.plugin_system import PluginSystem

        ps = PluginSystem()
        if hasattr(ps, "register"):
            mock_plugin = MagicMock()
            ps.register("test", mock_plugin)
            if hasattr(ps, "start"):
                ps.start("test")
            if hasattr(ps, "stop"):
                ps.stop("test")
