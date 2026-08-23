"""Configuration for wide-area distributed inference.

Uses pydantic.BaseSettings so every field can be overridden via environment
variable with the ``WIDE_AREA_`` prefix — e.g. ``WIDE_AREA_TIMEOUT=60``
overrides :attr:`WideAreaConfig.wan_timeout_seconds`.
"""

from __future__ import annotations
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class WideAreaConfig(BaseSettings):
    """Configuration for wide-area (high-latency) inference.

    Controls how the pipeline handles nodes separated by WAN links
    (high latency, limited bandwidth, potential packet loss).

    .. rubric:: 12-factor overrides

    Every field can be set via environment variable with the ``WIDE_AREA_``
    prefix.  Examples::

        export WIDE_AREA_ENABLED=true
        export WIDE_AREA_TIMEOUT=60
        export WIDE_AREA_TRANSPORT=quic

    Attributes:
        enabled: Enable WAN-aware pipeline execution.
        p2p_forwarding: Nodes forward hidden_states directly (bypass coordinator).
        token_accumulation: Buffer N decode steps before sending across WAN.
        accumulation_window: Max tokens to accumulate when token_accumulation=True.
        wan_timeout_seconds: Per-hop gRPC timeout for WAN links (much larger than LAN).
        max_accumulation_retries: Max retries per accumulated batch.
        adaptive_batching: Dynamically adjust accumulation window based on measured RTT.
        latency_sample_interval: Seconds between latency measurements.
        fallback_to_local: Fall back to local-only inference if WAN link fails.
        compression_level: gRPC compression level (0=off, 1=fast, 2=gzip).
        heartbeat_interval_seconds: Seconds between WAN link health checks.
        transport: Transport backend for WAN links. ``"auto"`` selects QUIC when
            available, falling back to gRPC. ``"quic"`` forces QUIC (fails if
            aioquic is not installed). ``"grpc"`` forces gRPC.
    """

    model_config = SettingsConfigDict(
        env_prefix="WIDE_AREA_",
        extra="ignore",
        frozen=True,
    )

    enabled: bool = False
    p2p_forwarding: bool = True
    token_accumulation: bool = True
    accumulation_window: int = Field(default=3, ge=1)
    wan_timeout_seconds: float = Field(default=120.0, gt=0)
    max_accumulation_retries: int = Field(default=3, ge=0)
    adaptive_batching: bool = True
    latency_sample_interval: float = Field(default=10.0, gt=0)
    fallback_to_local: bool = True
    compression_level: int = 2
    heartbeat_interval_seconds: float = Field(default=5.0, gt=0)
    transport: Literal["auto", "quic", "grpc"] = "auto"

    @field_validator("compression_level")
    @classmethod
    def _validate_compression(cls, v: int) -> int:
        if v not in (0, 1, 2):
            raise ValueError(f"compression_level must be 0, 1, or 2, got {v}")
        return v
