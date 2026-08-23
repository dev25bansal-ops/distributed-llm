"""Comprehensive tests for distllm.dist.config.

Covers all fields, edge cases, frozen behaviour, environment variable
loading, and validators for ``WideAreaConfig``.
"""

from __future__ import annotations

import os
from typing import Any

import pytest
from pydantic import ValidationError


class TestWideAreaConfigDefaults:
    """Default construction of WideAreaConfig."""

    def test_defaults(self):
        from distllm.dist.config import WideAreaConfig

        c = WideAreaConfig()
        assert c.enabled is False
        assert c.p2p_forwarding is True
        assert c.token_accumulation is True
        assert c.accumulation_window == 3
        assert c.wan_timeout_seconds == 120.0
        assert c.max_accumulation_retries == 3
        assert c.adaptive_batching is True
        assert c.latency_sample_interval == 10.0
        assert c.fallback_to_local is True
        assert c.compression_level == 2
        assert c.heartbeat_interval_seconds == 5.0
        assert c.transport == "auto"


class TestWideAreaConfigBools:
    """Boolean fields: enabled, p2p_forwarding, token_accumulation,
    adaptive_batching, fallback_to_local."""

    @pytest.mark.parametrize("field", ["enabled", "p2p_forwarding",
                                        "token_accumulation", "adaptive_batching",
                                        "fallback_to_local"])
    def test_bool_true(self, field: str) -> None:
        from distllm.dist.config import WideAreaConfig

        c = WideAreaConfig(**{field: True})
        assert getattr(c, field) is True

    @pytest.mark.parametrize("field", ["enabled", "p2p_forwarding",
                                        "token_accumulation", "adaptive_batching",
                                        "fallback_to_local"])
    def test_bool_false(self, field: str) -> None:
        from distllm.dist.config import WideAreaConfig

        c = WideAreaConfig(**{field: False})
        assert getattr(c, field) is False

    @pytest.mark.parametrize("field", ["enabled", "p2p_forwarding",
                                        "token_accumulation", "adaptive_batching",
                                        "fallback_to_local"])
    def test_bool_none_raises(self, field: str) -> None:
        from distllm.dist.config import WideAreaConfig

        with pytest.raises(ValidationError):
            WideAreaConfig(**{field: None})


class TestWideAreaConfigAccumulationWindow:
    """accumulation_window: int, ge=1."""

    def test_min_valid(self) -> None:
        from distllm.dist.config import WideAreaConfig

        WideAreaConfig(accumulation_window=1)

    def test_large_value(self) -> None:
        from distllm.dist.config import WideAreaConfig

        WideAreaConfig(accumulation_window=10_000)

    @pytest.mark.parametrize("val", [0, -1, -100])
    def test_less_than_one_raises(self, val: int) -> None:
        from distllm.dist.config import WideAreaConfig

        with pytest.raises(ValidationError):
            WideAreaConfig(accumulation_window=val)


class TestWideAreaConfigWanTimeout:
    """wan_timeout_seconds: float, gt=0."""

    def test_positive(self) -> None:
        from distllm.dist.config import WideAreaConfig

        WideAreaConfig(wan_timeout_seconds=0.1)

    def test_large(self) -> None:
        from distllm.dist.config import WideAreaConfig

        WideAreaConfig(wan_timeout_seconds=86_400.0)  # 24h

    @pytest.mark.parametrize("val", [0.0, -1.0, -1e-6])
    def test_non_positive_raises(self, val: float) -> None:
        from distllm.dist.config import WideAreaConfig

        with pytest.raises(ValidationError):
            WideAreaConfig(wan_timeout_seconds=val)


class TestWideAreaConfigMaxAccumulationRetries:
    """max_accumulation_retries: int, ge=0."""

    def test_zero(self) -> None:
        from distllm.dist.config import WideAreaConfig

        WideAreaConfig(max_accumulation_retries=0)

    def test_positive(self) -> None:
        from distllm.dist.config import WideAreaConfig

        WideAreaConfig(max_accumulation_retries=10)

    @pytest.mark.parametrize("val", [-1, -100])
    def test_negative_raises(self, val: int) -> None:
        from distllm.dist.config import WideAreaConfig

        with pytest.raises(ValidationError):
            WideAreaConfig(max_accumulation_retries=val)


class TestWideAreaConfigLatencySampleInterval:
    """latency_sample_interval: float, gt=0."""

    def test_tiny_positive(self) -> None:
        from distllm.dist.config import WideAreaConfig

        WideAreaConfig(latency_sample_interval=1e-6)

    def test_large(self) -> None:
        from distllm.dist.config import WideAreaConfig

        WideAreaConfig(latency_sample_interval=3600.0)

    @pytest.mark.parametrize("val", [0.0, -0.1])
    def test_non_positive_raises(self, val: float) -> None:
        from distllm.dist.config import WideAreaConfig

        with pytest.raises(ValidationError):
            WideAreaConfig(latency_sample_interval=val)


class TestWideAreaConfigHeartbeatInterval:
    """heartbeat_interval_seconds: float, gt=0."""

    def test_positive(self) -> None:
        from distllm.dist.config import WideAreaConfig

        WideAreaConfig(heartbeat_interval_seconds=0.5)

    def test_large(self) -> None:
        from distllm.dist.config import WideAreaConfig

        WideAreaConfig(heartbeat_interval_seconds=300.0)

    @pytest.mark.parametrize("val", [0.0, -1.0])
    def test_non_positive_raises(self, val: float) -> None:
        from distllm.dist.config import WideAreaConfig

        with pytest.raises(ValidationError):
            WideAreaConfig(heartbeat_interval_seconds=val)


class TestWideAreaConfigCompressionLevel:
    """compression_level: int with custom validator (0, 1, 2 only)."""

    @pytest.mark.parametrize("val", [0, 1, 2])
    def test_valid_levels(self, val: int) -> None:
        from distllm.dist.config import WideAreaConfig

        c = WideAreaConfig(compression_level=val)
        assert c.compression_level == val

    @pytest.mark.parametrize("val", [-1, 3, 10, 999])
    def test_invalid_levels_raises(self, val: int) -> None:
        from distllm.dist.config import WideAreaConfig

        with pytest.raises(ValidationError, match="compression_level"):
            WideAreaConfig(compression_level=val)


class TestWideAreaConfigTransport:
    """transport: Literal['auto', 'quic', 'grpc']."""

    @pytest.mark.parametrize("val", ["auto", "quic", "grpc"])
    def test_valid_transport(self, val: str) -> None:
        from distllm.dist.config import WideAreaConfig

        c = WideAreaConfig(transport=val)
        assert c.transport == val

    @pytest.mark.parametrize("val", ["tcp", "udp", "http", "", "   "])
    def test_invalid_transport_raises(self, val: str) -> None:
        from distllm.dist.config import WideAreaConfig

        with pytest.raises(ValidationError, match="transport"):
            WideAreaConfig(transport=val)


class TestWideAreaConfigFrozen:
    """Config is frozen — attribute writes must be rejected."""

    def test_cannot_set_attribute(self) -> None:
        from distllm.dist.config import WideAreaConfig

        c = WideAreaConfig()
        with pytest.raises(ValidationError, match="frozen_instance"):
            c.enabled = True

    def test_cannot_set_through_init(self) -> None:
        from distllm.dist.config import WideAreaConfig

        c = WideAreaConfig(enabled=True)
        with pytest.raises(ValidationError, match="frozen_instance"):
            c.enabled = False


class TestWideAreaConfigExtraIgnore:
    """Model is configured with extra='ignore'.

    Passing unknown fields should not raise.
    """

    def test_unknown_field_is_ignored(self) -> None:
        from distllm.dist.config import WideAreaConfig

        c = WideAreaConfig(unknown_param=42, another_unknown="x")
        assert c.enabled is False  # default still holds

    def test_unknown_field_does_not_appear_on_model(self) -> None:
        from distllm.dist.config import WideAreaConfig

        c = WideAreaConfig(unknown_param=42)
        assert not hasattr(c, "unknown_param")


class TestWideAreaConfigEnvOverride:
    """Environment variables with WIDE_AREA_ prefix override defaults."""

    ENV_VARS = {
        "WIDE_AREA_ENABLED": "true",
        "WIDE_AREA_TRANSPORT": "quic",
        "WIDE_AREA_COMPRESSION_LEVEL": "0",
        "WIDE_AREA_WAN_TIMEOUT_SECONDS": "300",
        "WIDE_AREA_ACCUMULATION_WINDOW": "16",
    }

    def test_env_overrides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for k, v in self.ENV_VARS.items():
            monkeypatch.setenv(k, v)
        from distllm.dist.config import WideAreaConfig

        c = WideAreaConfig()
        assert c.enabled is True
        assert c.transport == "quic"
        assert c.compression_level == 0
        assert c.wan_timeout_seconds == 300.0
        assert c.accumulation_window == 16

    def test_env_prefix_is_wire_area(self) -> None:
        """Verify env var prefix is exactly WIDE_AREA_."""
        from distllm.dist.config import WideAreaConfig

        assert WideAreaConfig.model_config.get("env_prefix") == "WIDE_AREA_"

    def test_constructor_takes_precedence_over_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("WIDE_AREA_ENABLED", "false")
        from distllm.dist.config import WideAreaConfig

        c = WideAreaConfig(enabled=True)
        assert c.enabled is True


class TestWideAreaConfigEdgeCases:
    """Miscellaneous edge cases for WideAreaConfig."""

    def test_all_fields_explicit(self) -> None:
        """Construct with every field set explicitly."""
        from distllm.dist.config import WideAreaConfig

        c = WideAreaConfig(
            enabled=True,
            p2p_forwarding=False,
            token_accumulation=False,
            accumulation_window=7,
            wan_timeout_seconds=60.0,
            max_accumulation_retries=5,
            adaptive_batching=False,
            latency_sample_interval=2.5,
            fallback_to_local=False,
            compression_level=1,
            heartbeat_interval_seconds=3.0,
            transport="grpc",
        )
        assert c.enabled is True
        assert c.p2p_forwarding is False
        assert c.token_accumulation is False
        assert c.accumulation_window == 7
        assert c.wan_timeout_seconds == 60.0
        assert c.max_accumulation_retries == 5
        assert c.adaptive_batching is False
        assert c.latency_sample_interval == 2.5
        assert c.fallback_to_local is False
        assert c.compression_level == 1
        assert c.heartbeat_interval_seconds == 3.0
        assert c.transport == "grpc"

    def test_float_ints_accepted(self) -> None:
        """Integer values are accepted for float fields."""
        from distllm.dist.config import WideAreaConfig

        c = WideAreaConfig(wan_timeout_seconds=30, heartbeat_interval_seconds=5)
        assert isinstance(c.wan_timeout_seconds, float)
        assert c.wan_timeout_seconds == 30.0
        assert c.heartbeat_interval_seconds == 5.0

    def test_custom_values_equality(self) -> None:
        """Two configs with same values compare equal (frozen dataclass-like)."""
        from distllm.dist.config import WideAreaConfig

        a = WideAreaConfig(enabled=True, transport="quic")
        b = WideAreaConfig(enabled=True, transport="quic")
        # pydantic BaseModel __eq__ compares by field values
        assert a == b

    def test_different_values_not_equal(self) -> None:
        from distllm.dist.config import WideAreaConfig

        a = WideAreaConfig(enabled=False)
        b = WideAreaConfig(enabled=True)
        assert a != b

    def test_wrong_type_raises(self) -> None:
        """Passing a string for an int field should raise."""
        from distllm.dist.config import WideAreaConfig

        with pytest.raises(ValidationError):
            WideAreaConfig(accumulation_window="not-an-int")
