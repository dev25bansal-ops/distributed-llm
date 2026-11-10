"""Tests for MLflow plugin."""
from unittest.mock import patch

from distllm.plugins.mlflow_plugin import MLflowPlugin, MLflowConfig


class TestMLflowPlugin:
    def test_create_config(self):
        config = MLflowConfig()
        assert config.tracking_uri == "http://localhost:5000"

    def test_log_run_start_without_mlflow_returns_false(self):
        """When the mlflow package is absent the plugin reports unavailable.

        The previous version of this test called log_run_start() for real and
        asserted False, on the assumption that mlflow was not installed. On a
        machine where it IS installed, that call reaches
        mlflow.start_run() against the default localhost:5000 tracking URI and
        blocks for minutes in urllib3 retry backoff before it raises -- the
        plugin's except clause never gets a chance, and the test hangs instead
        of failing. The unavailability path is what this test is about, so it
        is now exercised directly by forcing the module-level flag.
        """
        with patch("distllm.plugins.mlflow_plugin._MLFLOW_AVAILABLE", False):
            plugin = MLflowPlugin()
            assert plugin.log_run_start({"model": "test"}) is False

    def test_log_metrics_without_run_returns_false(self):
        """No active run means nothing to log, regardless of mlflow presence."""
        plugin = MLflowPlugin()
        assert plugin.log_metrics({"loss": 0.1}) is False

    def test_log_run_end_without_run_returns_false(self):
        plugin = MLflowPlugin()
        assert plugin.log_run_end() is False