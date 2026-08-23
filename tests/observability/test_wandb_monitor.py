"""Tests for W&B monitor."""
from distllm.observability.wandb_monitor import WandBMonitor, WandBConfig


class TestWandBMonitor:
    def test_create_config(self):
        config = WandBConfig()
        assert config.project == "distllm"

    def test_start_no_wandb(self):
        monitor = WandBMonitor()
        result = monitor.start()
        assert result is False  # wandb not installed in test env
