"""Tests for MLflow plugin."""
from distllm.plugins.mlflow_plugin import MLflowPlugin, MLflowConfig


class TestMLflowPlugin:
    def test_create_config(self):
        config = MLflowConfig()
        assert config.tracking_uri == "http://localhost:5000"

    def test_log_run_start_no_mlflow(self):
        plugin = MLflowPlugin()
        result = plugin.log_run_start({"model": "test"})
        assert result is False  # mlflow not installed in test env
