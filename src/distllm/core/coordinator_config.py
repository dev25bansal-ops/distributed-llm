"""Configuration for the distributed coordinator.

Uses Pydantic BaseModel for validation, consistent with the rest
of the config system in ``distllm.config.settings``.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from distllm.config.settings import DistLLMSettings


class CoordinatorConfig(BaseModel):
    """Configuration for the distributed coordinator.

    Use :meth:`from_settings` to create from a :class:`DistLLMSettings`
    instance, which extracts all values from the Pydantic settings model.

    Example::

        config = CoordinatorConfig(model_name="meta-llama/Llama-2-7b")
        config = CoordinatorConfig.from_settings(settings)
    """

    model_name: str = Field(
        default="",
        description="HuggingFace model name or local path.",
    )
    port: int = Field(
        default=50050,
        ge=1,
        le=65535,
        description="gRPC port for the coordinator.",
    )
    dtype: str = Field(
        default="float16",
        description="Model dtype: float16, float32, or bfloat16.",
    )
    trust_remote_code: bool | None = Field(
        default=None,
        description="Trust remote code when loading models from HuggingFace.",
    )
    max_batch_size: int = Field(
        default=4,
        ge=1,
        description="Maximum number of sequences per batch.",
    )
    max_tokens_per_batch: int = Field(
        default=1024,
        ge=1,
        description="Maximum total tokens per batch iteration.",
    )
    pipeline_timeout: float = Field(
        default=30.0,
        gt=0,
        description="Timeout in seconds for a single pipeline forward pass.",
    )
    cluster_key: str | None = Field(
        default=None,
        description="Shared secret for cluster authentication.",
    )
    model_cache_dir: str | None = Field(
        default=None,
        description="Directory for caching downloaded models.",
    )
    metrics_exporter: Any = Field(
        default=None,
        description="Optional metrics exporter instance.",
    )
    discovery_mode: str | None = Field(
        default=None,
        description="Service discovery mode (e.g. 'mdns').",
    )
    wide_area_config: Any = Field(
        default=None,
        description="WideAreaConfig for WAN inference.",
    )
    redundancy: int = Field(
        default=1,
        ge=1,
        description="Number of redundant copies for fault-tolerant execution.",
    )
    ha_enabled: bool = Field(
        default=False,
        description="Enable HA leader election + state replication across "
                    "multiple coordinators.",
    )
    ha_peer_coordinators: list[tuple[str, str, int]] | None = Field(
        default=None,
        description="HA peers as (coordinator_id, host, port) tuples.",
    )
    ha_replication_peers: list[str] | None = Field(
        default=None,
        description="Peer API base URLs for HA state replication "
                    "(e.g. ['http://10.0.0.2:8000']).",
    )
    ha_heartbeat_interval_s: float = Field(default=2.0, gt=0,
        description="Seconds between coordinator election heartbeats.")
    ha_election_timeout_s: float = Field(default=10.0, gt=0,
        description="Seconds without a heartbeat before a peer is dead.")
    federation_config: Any = Field(
        default=None,
        description="FederationConfig for cross-cluster federation.",
    )
    plugin_system: Any = Field(
        default=None,
        description="PluginSystem instance for hook dispatch.",
    )
    min_reputation: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum reputation score for node participation.",
    )
    prefix_cache_enabled: bool = Field(
        default=False,
        description="Enable prefix caching for common prompt prefixes.",
    )
    prefix_cache_max_entries: int = Field(
        default=256,
        ge=1,
        description="Maximum number of prefix cache entries.",
    )
    prefix_cache_min_prefix_len: int = Field(
        default=4,
        ge=1,
        description="Minimum prefix length to cache.",
    )
    radix_tree_cache_enabled: bool = Field(
        default=False,
        description="Enable radix tree cache for prefix sharing.",
    )
    chunked_prefill_enabled: bool = Field(
        default=False,
        description="Enable chunked prefill for long prompts.",
    )
    chunked_prefill_chunk_size: int = Field(
        default=512,
        ge=1,
        description="Chunk size for chunked prefill.",
    )
    enable_pipeline_overlap: bool = Field(
        default=False,
        description="Enable overlapping communication with compute.",
    )

    @field_validator("dtype")
    @classmethod
    def validate_dtype(cls, v: str) -> str:
        allowed = {"float16", "float32", "bfloat16"}
        if v not in allowed:
            raise ValueError(f"dtype must be one of {allowed}, got '{v}'")
        return v

    model_config = {"arbitrary_types_allowed": True}

    @classmethod
    def from_settings(
        cls, settings: DistLLMSettings, **overrides: Any
    ) -> CoordinatorConfig:
        """Create a CoordinatorConfig from a DistLLMSettings instance.

        Args:
            settings: The application settings.
            **overrides: Additional keyword arguments to override settings values.

        Returns:
            A validated CoordinatorConfig instance.
        """
        wa = settings.wide_area
        wide_area_config = None
        if wa.enabled:
            from distllm.dist.config import WideAreaConfig

            wide_area_config = WideAreaConfig(
                enabled=wa.enabled,
                p2p_forwarding=wa.p2p_forwarding,
                tokens_before_forward=wa.tokens_before_forward,
                wan_timeout_seconds=wa.wan_timeout_seconds,
                max_retries=wa.max_retries,
                backoff_base_seconds=wa.backoff_base_seconds,
            )

        config = cls(
            model_name=settings.model.name,
            port=settings.coordinator.port,
            dtype=settings.model.dtype,
            trust_remote_code=settings.model.trust_remote_code or None,
            max_batch_size=settings.batching.max_batch_size,
            max_tokens_per_batch=settings.batching.max_tokens_per_batch,
            pipeline_timeout=settings.network.grpc_timeout,
            model_cache_dir=settings.model_hub.cache_dir,
            wide_area_config=wide_area_config,
        )

        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)

        return config
