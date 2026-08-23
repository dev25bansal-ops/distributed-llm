"""Pydantic-settings based configuration for distributed LLM inference.

All config classes use pydantic BaseModel for validation and BaseSettings
for environment variable support with env_prefix="DISTLLM_" and nested
delimiter "__" (e.g., DISTLLM__MODEL__NAME).
"""

import os
from typing import Any, TYPE_CHECKING

from pydantic import Field, ValidationError, model_validator
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
    auto_partition: dict[str, bool | str | float] = Field(default_factory=lambda: {"enabled": False, "strategy": "auto", "safety_margin": 0.1})
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

    # ------------------------------------------------------------------
    # Cross-field validation
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def _validate_cross_field(self) -> "DistLLMSettings":
        """Validate inter-field constraints that span multiple sections.

        Raises ValueError with a descriptive message listing all violations
        so the user can fix them in a single pass.
        """
        errors: list[str] = []

        # 0. vLLM dtype consistency with model.dtype
        if self.vllm.enabled:
            if self.vllm.dtype not in ("auto", self.model.dtype):
                errors.append(
                    f"model.dtype is '{self.model.dtype}' but vllm.dtype is "
                    f"'{self.vllm.dtype}'; these should be consistent when "
                    "vllm is enabled, or set vllm.dtype='auto'"
                )

        # 1. Chunked prefill requires a positive prefill token budget.
        if self.chunked_prefill.enabled and self.batching.max_tokens_per_batch < 1:
            errors.append(
                "chunked_prefill is enabled but batching.max_tokens_per_batch "
                f"is {self.batching.max_tokens_per_batch}; must be > 0"
            )

        # 2. Speculative decoding with draft_model method needs a model path.
        #    An empty draft_model is the default (speculative disabled).
        spec = self.speculative
        if spec.method == "draft_model" and not spec.draft_model.strip():
            # Only flag if other settings suggest speculative was intended
            # (non-default num_assistant_tokens or min_acceptance_rate).
            if spec.num_assistant_tokens != 5 or spec.min_acceptance_rate != 0.3:
                errors.append(
                    "speculative method is 'draft_model' and non-default "
                    "speculative settings were provided, but "
                    "speculative.draft_model is empty; set it to a model ID"
                )
        if spec.method == "eagle" and not spec.eagle_checkpoint.strip():
            errors.append(
                "speculative method is 'eagle' but "
                "speculative.eagle_checkpoint is empty"
            )

        # 3. TLS must point to existing certificate and key files.
        if self.tls.enabled:
            if not self.tls.cert_file:
                errors.append(
                    "tls is enabled but tls.cert_file is not set"
                )
            elif not os.path.isfile(self.tls.cert_file):
                errors.append(
                    f"tls.cert_file does not exist: {self.tls.cert_file}"
                )
            if not self.tls.key_file:
                errors.append(
                    "tls is enabled but tls.key_file is not set"
                )
            elif not os.path.isfile(self.tls.key_file):
                errors.append(
                    f"tls.key_file does not exist: {self.tls.key_file}"
                )

        # 4. Multi-GPU tensor parallelism requires network configuration.
        if self.tensor_parallel.enabled and self.tensor_parallel.num_gpus > 1:
            net = self.network
            if net.grpc_timeout < 1 or net.max_retries < 1:
                errors.append(
                    "tensor_parallel is enabled with num_gpus > 1 but network "
                    f"settings are invalid (grpc_timeout={net.grpc_timeout}, "
                    f"max_retries={net.max_retries}); both must be >= 1"
                )

        if errors:
            raise ValueError(
                "Cross-field validation failed:\n  - " + "\n  - ".join(errors)
            )
        return self

    # ------------------------------------------------------------------
    # Diff utility
    # ------------------------------------------------------------------

    def diff(self, other: "DistLLMSettings") -> dict[str, tuple[Any, Any]]:
        """Return fields that differ between *self* and *other*.

        Returns a dict mapping dotted field paths to ``(self_value, other_value)``
        tuples.  Nested models are compared recursively so only the leaf values
        that actually changed appear in the result.
        """
        a = self.model_dump()
        b = other.model_dump()
        return self._diff_dicts(a, b, prefix="")

    @classmethod
    def _diff_dicts(
        cls,
        a: dict[str, Any],
        b: dict[str, Any],
        prefix: str,
    ) -> dict[str, tuple[Any, Any]]:
        """Recursively compare two dicts, returning dotted-path differences."""
        changes: dict[str, tuple[Any, Any]] = {}
        all_keys = set(a) | set(b)
        for key in sorted(all_keys):
            path = f"{prefix}.{key}" if prefix else key
            val_a = a.get(key)
            val_b = b.get(key)
            if isinstance(val_a, dict) and isinstance(val_b, dict):
                changes.update(cls._diff_dicts(val_a, val_b, prefix=path))
            elif val_a != val_b:
                changes[path] = (val_a, val_b)
        return changes

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
