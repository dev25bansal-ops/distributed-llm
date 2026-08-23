"""Tests for WANConfig and WANSchedulingPolicy."""

from __future__ import annotations

from types import SimpleNamespace

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_wan = load_module("distllm/core/advanced_scheduling/wan.py")
WANConfig = _wan.WANConfig
WANSchedulingPolicy = _wan.WANSchedulingPolicy


class TestWANConfig:
    """Test suite for WANConfig dataclass."""

    def test_default_construction(self) -> None:
        config = WANConfig()
        assert config.enabled is False
        assert config.p2p_forwarding is False
        assert config.tokens_before_forward == 4
        assert config.wan_timeout_seconds == 30.0
        assert config.max_retries == 3
        assert config.backoff_base_seconds == 1.0
        assert config.accumulation_window == 4

    def test_custom_values(self) -> None:
        config = WANConfig(
            enabled=True,
            p2p_forwarding=True,
            tokens_before_forward=8,
            wan_timeout_seconds=60.0,
            max_retries=5,
            backoff_base_seconds=2.0,
            accumulation_window=8,
        )
        assert config.enabled is True
        assert config.wan_timeout_seconds == 60.0
        assert config.accumulation_window == 8


class TestWANSchedulingPolicy:
    """Test suite for WANSchedulingPolicy."""

    def test_default_construction(self) -> None:
        policy = WANSchedulingPolicy()
        assert policy._config.enabled is False

    def test_construction_with_config(self) -> None:
        config = WANConfig(enabled=True, max_retries=5)
        policy = WANSchedulingPolicy(config=config)
        assert policy._config is config

    def test_construction_with_custom_config(self) -> None:
        policy = WANSchedulingPolicy(config=WANConfig(enabled=True))
        assert policy._config.enabled is True

    def test_should_disable_pressure_adaptation_disabled(self) -> None:
        policy = WANSchedulingPolicy()
        assert policy.should_disable_pressure_adaptation() is False

    def test_should_disable_pressure_adaptation_enabled(self) -> None:
        policy = WANSchedulingPolicy(config=WANConfig(enabled=True))
        assert policy.should_disable_pressure_adaptation() is True

    def test_compute_budget_disabled_passthrough(self) -> None:
        policy = WANSchedulingPolicy()
        budget = SimpleNamespace(max_batch_size=64)
        result = policy.compute_budget(budget)
        assert result.max_batch_size == 64

    def test_compute_budget_enabled_reduces_batch_size(self) -> None:
        policy = WANSchedulingPolicy(config=WANConfig(enabled=True))
        budget = SimpleNamespace(max_batch_size=64)
        result = policy.compute_budget(budget)
        assert result.max_batch_size == min(64, 8) == 8

    def test_compute_budget_enabled_small_batch_unchanged(self) -> None:
        policy = WANSchedulingPolicy(config=WANConfig(enabled=True))
        budget = SimpleNamespace(max_batch_size=4)
        result = policy.compute_budget(budget)
        assert result.max_batch_size == 4  # min(4, 8) == 4

    def test_on_before_schedule_passthrough(self) -> None:
        policy = WANSchedulingPolicy()
        seqs = ["a", "b"]
        assert policy.on_before_schedule(seqs) is seqs
