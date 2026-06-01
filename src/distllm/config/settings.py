"""Pydantic-settings based configuration for distributed LLM inference.

All config classes use pydantic BaseModel for validation and BaseSettings
for environment variable support with env_prefix="DISTLLM_" and nested
delimiter "__" (e.g., DISTLLM__MODEL__NAME).
"""

import os
from typing import Any, TYPE_CHECKING

from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

# Domain module imports
from distllm.config._model import (
    AdaptiveCompressionSettings,
    CompressionSettings,
    EmbeddingSettings,
    LoRASettings,
    ModelHubSettings,
    ModelSettings,
    MoESettings,
    MultiModelSettings,
    PromptTemplateSettings,
    QuantizationSettings,
    SloRaSettings,
    SpeculativeSettings,
)
from distllm.config._network import (
    ChatRouterSettings,
    CoordinatorSettings,
    NetworkSettings,
    RateLimitSettings,
    RouteRuleSettings,
    TLSSettings,
    WideAreaSettings,
)
from distllm.config._cache import (
    CachePersistenceSettings,
    CacheSettings,
    DefragmentationSettings,
    GossipSettings,
    PredictiveCacheSettings,
    PrefixCacheSettings,
)
from distllm.config._parallelism import (
    BatchingSettings,
    ChunkedPrefillSettings,
    DisaggSettings,
    HybridParallelSettings,
    NodeRole,
    NodeSettings,
    PartitioningSettings,
    PrioritySettings,
    RebalancerSettings,
    TensorParallelSettings,
    ZeroCopySettings,
)
from distllm.config._performance import (
    AdaptivePrecisionSettings,
    CompileSettings,
    CudaGraphSettings,
    SelfOptimizingSettings,
)
from distllm.config._hardware import HardwareSettings
from distllm.config._backends import LlamacppSettings, VLLMSettings
from distllm.config._generation import GenerationSettings
from distllm.config._observability import (
    AlertingSettings,
    ChaosSettings,
    MonitoringSettings,
)
from distllm.config._deployment import (
    CanarySettings,
    CostSettings,
    RolloutStageModel,
    TenantSettings,
    VersionSettings,
)
from distllm.config._application import AgentSettings, PluginSettings, RAGSettings

if TYPE_CHECKING:
    pass

__all__ = [
    "NodeRole",
    "ModelSettings",
    "CoordinatorSettings",
    "NodeSettings",
    "GenerationSettings",
    "NetworkSettings",
    "TLSSettings",
    "BatchingSettings",
    "PrefixCacheSettings",
    "ChunkedPrefillSettings",
    "MonitoringSettings",
    "QuantizationSettings",
    "SpeculativeSettings",
    "PartitioningSettings",
    "RebalancerSettings",
    "CachePersistenceSettings",
    "PrioritySettings",
    "MultiModelSettings",
    "TensorParallelSettings",
    "LoRASettings",
    "MoESettings",
    "GossipSettings",
    "CompressionSettings",
    "AdaptiveCompressionSettings",
    "AlertingSettings",
    "ChaosSettings",
    "RolloutStageModel",
    "CanarySettings",
    "CostSettings",
    "RateLimitSettings",
    "ModelHubSettings",
    "PromptTemplateSettings",
    "EmbeddingSettings",
    "VersionSettings",
    "PluginSettings",
    "HybridParallelSettings",
    "ZeroCopySettings",
    "AdaptivePrecisionSettings",
    "PredictiveCacheSettings",
    "CacheSettings",
    "SelfOptimizingSettings",
    "CudaGraphSettings",
    "CompileSettings",
    "SloRaSettings",
    "RAGSettings",
    "AgentSettings",
    "DisaggSettings",
    "WideAreaSettings",
    "VLLMSettings",
    "LlamacppSettings",
    "RouteRuleSettings",
    "ChatRouterSettings",
    "HardwareSettings",
    "DefragmentationSettings",
    "TenantSettings",
    "DistLLMSettings",
]


class DistLLMSettings(BaseSettings):
    """Root configuration for distributed LLM inference.

    Environment variables use DISTLLM__ prefix with __ delimiter.
    Example: DISTLLM__MODEL__NAME=my-model sets model.name.
    """
    model_config = SettingsConfigDict(
        env_prefix="DISTLLM_",
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
    )

    model: ModelSettings = Field(default_factory=ModelSettings)
    coordinator: CoordinatorSettings = Field(default_factory=CoordinatorSettings)
    nodes: list[NodeSettings] = Field(default_factory=list)
    generation: GenerationSettings = Field(default_factory=GenerationSettings)
    network: NetworkSettings = Field(default_factory=NetworkSettings)
    tls: TLSSettings = Field(default_factory=TLSSettings)
    batching: BatchingSettings = Field(default_factory=BatchingSettings)
    prefix_cache: PrefixCacheSettings = Field(default_factory=PrefixCacheSettings)
    chunked_prefill: ChunkedPrefillSettings = Field(default_factory=ChunkedPrefillSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    quantization: QuantizationSettings = Field(default_factory=QuantizationSettings)
    speculative: SpeculativeSettings = Field(default_factory=SpeculativeSettings)
    partitioning: PartitioningSettings = Field(default_factory=PartitioningSettings)
    auto_partition: dict = Field(default_factory=lambda: {"enabled": False, "strategy": "auto", "safety_margin": 0.1})
    rebalancer: RebalancerSettings = Field(default_factory=RebalancerSettings)
    cache_persistence: CachePersistenceSettings = Field(default_factory=CachePersistenceSettings)
    priority: PrioritySettings = Field(default_factory=PrioritySettings)
    multi_model: MultiModelSettings = Field(default_factory=MultiModelSettings)
    tensor_parallel: TensorParallelSettings = Field(default_factory=TensorParallelSettings)
    lora: LoRASettings = Field(default_factory=LoRASettings)
    moe: MoESettings = Field(default_factory=MoESettings)
    gossip: GossipSettings = Field(default_factory=GossipSettings)
    compression: CompressionSettings = Field(default_factory=CompressionSettings)
    adaptive_compression: AdaptiveCompressionSettings = Field(default_factory=AdaptiveCompressionSettings)
    alerting: AlertingSettings = Field(default_factory=AlertingSettings)
    chaos: ChaosSettings = Field(default_factory=ChaosSettings)
    canary: CanarySettings = Field(default_factory=CanarySettings)
    cost: CostSettings = Field(default_factory=CostSettings)
    tenant: TenantSettings = Field(default_factory=TenantSettings)
    rate_limit: RateLimitSettings = Field(default_factory=RateLimitSettings)
    model_hub: ModelHubSettings = Field(default_factory=ModelHubSettings)
    prompt_template: PromptTemplateSettings = Field(default_factory=PromptTemplateSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    version: VersionSettings = Field(default_factory=VersionSettings)
    plugins: PluginSettings = Field(default_factory=PluginSettings)
    hybrid_parallel: HybridParallelSettings = Field(default_factory=HybridParallelSettings)
    zero_copy: ZeroCopySettings = Field(default_factory=ZeroCopySettings)
    adaptive_precision: AdaptivePrecisionSettings = Field(default_factory=AdaptivePrecisionSettings)
    predictive_cache: PredictiveCacheSettings = Field(default_factory=PredictiveCacheSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)  # E15: Unified cache config
    predictive_migration: dict = Field(default_factory=lambda: {"enabled": False})
    structured_output: dict = Field(default_factory=lambda: {"enabled": False})
    self_optimizing: SelfOptimizingSettings = Field(default_factory=SelfOptimizingSettings)
    optimization: dict = Field(default_factory=lambda: {"enabled": False})
    cuda_graph: CudaGraphSettings = Field(default_factory=CudaGraphSettings)
    compile: CompileSettings = Field(default_factory=CompileSettings)
    slora: SloRaSettings = Field(default_factory=SloRaSettings)
    rag: RAGSettings = Field(default_factory=RAGSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    disagg: DisaggSettings = Field(default_factory=DisaggSettings)
    chat_router: ChatRouterSettings = Field(default_factory=ChatRouterSettings)
    hardware: HardwareSettings = Field(default_factory=HardwareSettings)
    vllm: VLLMSettings = Field(default_factory=VLLMSettings)
    llamacpp: LlamacppSettings = Field(default_factory=LlamacppSettings)
    wide_area: WideAreaSettings = Field(default_factory=WideAreaSettings)
    defragmentation: DefragmentationSettings = Field(default_factory=DefragmentationSettings)

    @classmethod
    def from_yaml(
        cls,
        config_path: str | None = None,
        cli_overrides: dict[str, Any] | None = None,
    ) -> "DistLLMSettings":
        """Load settings with full precedence: CLI > env vars > YAML > defaults.

        Environment variables are handled automatically by pydantic-settings
        using the ``DISTLLM__`` prefix and ``__`` nested delimiter
        (e.g. ``DISTLLM__MODEL__NAME=my-model``).

        Args:
            config_path: Path to a YAML config file. ``None`` to skip YAML.
            cli_overrides: Flat or nested dict of CLI overrides applied last.

        Returns:
            Validated DistLLMSettings instance.
        """
        import yaml

        data: dict[str, Any] = {}

        if config_path and os.path.exists(config_path):
            with open(config_path) as f:
                data = yaml.safe_load(f) or {}

        # Construct with YAML as base — pydantic-settings applies env vars on
        # top (env > YAML), so YAML cannot override env vars.
        settings = cls(**data)

        if cli_overrides:
            merged = cls._apply_cli_overrides(settings.model_dump(), cli_overrides)
            settings = cls.model_validate(merged)

        return settings

    @classmethod
    def from_profile(cls, config_path: str, profile: str | None = None) -> "DistLLMSettings":
        """Load settings from a profile-based YAML config.

        Args:
            config_path: Path to the YAML config file.
            profile: Profile name (dev, staging, production). If None, reads
                DISTLLM_PROFILE env var, defaults to "dev".

        Returns:
            Validated DistLLMSettings instance.
        """
        from distllm.config.profiles import ProfileConfig
        merged = ProfileConfig.load(config_path, profile)
        return cls.model_validate(merged)

    @staticmethod
    def _apply_cli_overrides(data: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
        """Apply flat or nested CLI overrides into a config dict."""
        result = dict(data)
        for key, value in overrides.items():
            if isinstance(value, dict):
                result.setdefault(key, {})
                if isinstance(result[key], dict):
                    result[key].update(value)
                else:
                    result[key] = value
            else:
                result[key] = value
        return result

    @classmethod
    def validate_startup(
        cls,
        config_path: str | None = None,
        cli_overrides: dict[str, Any] | None = None,
    ) -> "DistLLMSettings":
        """Load and validate configuration at startup.

        Accepts an optional YAML path and CLI overrides for the full
        precedence chain.  Simple usage without arguments validates only
        environment-variable and default-based configuration.

        Args:
            config_path: Optional path to a YAML config file.
            cli_overrides: Optional dict of CLI argument overrides.

        Returns:
            Validated DistLLMSettings instance.

        Raises:
            SystemExit: If validation fails, prints human-readable errors.
        """
        try:
            if config_path or cli_overrides:
                return cls.from_yaml(config_path=config_path, cli_overrides=cli_overrides)
            return cls()
        except ValidationError as e:
            errors = []
            for error in e.errors():
                field = ".".join(str(loc) for loc in error["loc"])
                msg = error["msg"]
                input_val = error.get("input", "")
                if input_val != "":
                    errors.append(f"  - {field}: {msg} (got {input_val!r})")
                else:
                    errors.append(f"  - {field}: {msg}")

            print("\n❌ Config validation failed:")
            for err in errors:
                print(err)
            print()
            raise SystemExit(1) from e
