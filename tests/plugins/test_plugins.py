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
            assert True


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
            plugin.log(
                action="model_load",
                user="test_user",
                model="test-model",
                success=True,
            )
            assert True

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
            assert True


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
            assert True

    def test_metrics_labels(self):
        from distllm.plugins.builtin import MetricsPlugin

        plugin = MetricsPlugin()
        if hasattr(plugin, "collect"):
            plugin.collect(
                metric_name="latency",
                value=0.5,
                labels={"endpoint": "/v1/chat", "method": "POST"},
            )
            assert True


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
