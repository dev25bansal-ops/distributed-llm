"""Pydantic-settings based configuration for distributed LLM inference.

All config classes use pydantic BaseModel for validation and BaseSettings
for environment variable support with env_prefix="DISTLLM_" and nested
delimiter "__" (e.g., DISTLLM__MODEL__NAME).
"""

from enum import Enum
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class NodeRole(str, Enum):
    """Node role for prefill-decode disaggregation."""
    AUTO = "auto"
    PREFILL = "prefill"
    DECODE = "decode"


class ModelSettings(BaseModel):
    """Model configuration."""
    name: str = "roneneldan/TinyStories-1M"
    dtype: str = "float16"
    trust_remote_code: bool = False

    @field_validator("dtype")
    @classmethod
    def validate_dtype(cls, v: str) -> str:
        allowed = {"float16", "float32", "bfloat16"}
        if v not in allowed:
            raise ValueError(f"dtype must be one of {allowed}, got '{v}'")
        return v


class CoordinatorSettings(BaseModel):
    """Coordinator configuration."""
    host: str = "localhost"
    port: int = 50050
    api_port: int = 8000
    cors_origins: str = "http://localhost:3000,http://localhost:8080"

    @field_validator("port", "api_port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        if not (1 <= v <= 65535):
            raise ValueError(f"Port must be 1-65535, got {v}")
        return v

    @field_validator("cors_origins")
    @classmethod
    def validate_origins(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("cors_origins must not be empty")
        return v


class NodeSettings(BaseModel):
    """Worker node configuration."""
    node_id: str
    host: str = "localhost"
    port: int = 50051
    start_layer: int = 0
    end_layer: int = 3
    device: str = "cuda"
    role: NodeRole = NodeRole.AUTO

    @field_validator("port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        if not (1 <= v <= 65535):
            raise ValueError(f"Port must be 1-65535, got {v}")
        return v

    @field_validator("end_layer")
    @classmethod
    def validate_end_layer(cls, v: int, info) -> int:
        values = info.data
        if "start_layer" in values and v < values["start_layer"]:
            raise ValueError(f"end_layer ({v}) must be >= start_layer ({values['start_layer']})")
        return v


class GenerationSettings(BaseModel):
    """Text generation configuration."""
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 0

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        if not (0.0 <= v <= 2.0):
            raise ValueError(f"temperature must be 0-2.0, got {v}")
        return v

    @field_validator("top_p")
    @classmethod
    def validate_top_p(cls, v: float) -> float:
        if not (0.0 < v <= 1.0):
            raise ValueError(f"top_p must be 0-1.0, got {v}")
        return v

    @field_validator("top_k")
    @classmethod
    def validate_top_k(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"top_k must be >= 0, got {v}")
        return v


class NetworkSettings(BaseModel):
    """Network configuration."""
    grpc_timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0

    @field_validator("grpc_timeout", "max_retries")
    @classmethod
    def validate_positive_int(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"Must be positive, got {v}")
        return v


class TLSSettings(BaseModel):
    """TLS/Security configuration."""
    enabled: bool = False
    cert_dir: str = "certs"
    cert_file: Optional[str] = None
    key_file: Optional[str] = None
    ca_cert_file: Optional[str] = None


class BatchingSettings(BaseModel):
    """Continuous batching configuration."""
    max_batch_size: int = 32
    max_tokens_per_batch: int = 4096

    @field_validator("max_batch_size", "max_tokens_per_batch")
    @classmethod
    def validate_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"Must be >= 1, got {v}")
        return v


class PrefixCacheSettings(BaseModel):
    """Prefix cache configuration."""
    enabled: bool = True
    max_entries: int = 1024
    min_prefix_len: int = 16
    radix_tree_enabled: bool = True  # Use RadixTree (trie) instead of hash-based LRU

    @field_validator("max_entries", "min_prefix_len")
    @classmethod
    def validate_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"Must be >= 1, got {v}")
        return v


class ChunkedPrefillSettings(BaseModel):
    """Chunked prefill configuration."""
    enabled: bool = True
    chunk_size: int = 512

    @field_validator("chunk_size")
    @classmethod
    def validate_chunk_size(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"chunk_size must be >= 1, got {v}")
        return v


class MonitoringSettings(BaseModel):
    """System monitoring configuration."""
    enabled: bool = True


class QuantizationSettings(BaseModel):
    """Quantization configuration for model loading."""
    method: str = "none"  # "none" | "bnb_4bit" | "bnb_8bit" | "gptq" | "awq" | "fp8"
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    llm_int8_threshold: float = 6.0
    # GPTQ-specific
    gptq_bits: int = 4
    gptq_group_size: int = 128
    gptq_desc_act: bool = False
    gptq_use_marlin: bool = True  # Use Marlin kernel for Hopper
    # AWQ-specific
    awq_bits: int = 4
    awq_group_size: int = 128
    # FP8-specific
    fp8_scheme: str = "e4m3"  # "e4m3" | "e5m2"
    fp8_dynamic: bool = True
    # KV cache quantization
    kv_cache_quant: bool = False
    kv_cache_bits: int = 8  # 4 or 8

    @field_validator("method")
    @classmethod
    def validate_method(cls, v: str) -> str:
        allowed = {"none", "bnb_4bit", "bnb_8bit", "gptq", "awq", "fp8"}
        if v not in allowed:
            raise ValueError(f"method must be one of {allowed}, got '{v}'")
        return v

    @field_validator("gptq_bits", "awq_bits")
    @classmethod
    def validate_bits(cls, v: int) -> int:
        if v not in (4, 8):
            raise ValueError(f"bits must be 4 or 8, got {v}")
        return v

    @field_validator("kv_cache_bits")
    @classmethod
    def validate_kv_bits(cls, v: int) -> int:
        if v not in (4, 8):
            raise ValueError(f"kv_cache_bits must be 4 or 8, got {v}")
        return v


class SpeculativeSettings(BaseModel):
    """Speculative decoding configuration."""
    draft_model: str = ""
    num_assistant_tokens: int = 5
    min_acceptance_rate: float = 0.3
    warmup_steps: int = 10
    method: str = "draft_model"  # "draft_model" | "medusa" | "ngram" | "auto"
    medusa_num_heads: int = 4
    medusa_num_tokens_per_head: int = 3
    ngram_min_match: int = 4  # Minimum n-gram match length

    @field_validator("num_assistant_tokens")
    @classmethod
    def validate_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"Must be >= 1, got {v}")
        return v

    @field_validator("method")
    @classmethod
    def validate_method(cls, v: str) -> str:
        allowed = {"draft_model", "medusa", "ngram", "auto"}
        if v not in allowed:
            raise ValueError(f"method must be one of {allowed}, got '{v}'")
        return v


class PartitioningSettings(BaseModel):
    """Layer partitioning strategy configuration."""
    strategy: str = "gpu_aware"  # "equal" | "gpu_aware"
    safety_margin: float = 0.1  # leave 10% VRAM free

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, v: str) -> str:
        allowed = {"equal", "gpu_aware"}
        if v not in allowed:
            raise ValueError(f"strategy must be one of {allowed}, got '{v}'")
        return v


class RebalancerSettings(BaseModel):
    """Dynamic pipeline rebalancing configuration."""
    enabled: bool = False
    check_interval: float = 30.0
    straggler_threshold: float = 1.5
    min_improvement_pct: float = 0.1
    cooldown_seconds: float = 300.0


class CachePersistenceSettings(BaseModel):
    """KV cache persistence to disk configuration."""
    enabled: bool = False
    storage_path: str = ".distllm_cache"
    max_disk_gb: float = 50.0
    ttl_hours: float = 24.0


class PrioritySettings(BaseModel):
    """Request priority queuing configuration."""
    enabled: bool = False
    num_levels: int = 4
    preemption_enabled: bool = False
    max_preempted: int = 10


class MultiModelSettings(BaseModel):
    """Multi-model serving configuration."""
    models: Dict[str, str] = Field(default_factory=dict)  # name -> path
    default_model: str = ""
    max_models: int = 4


class TensorParallelSettings(BaseModel):
    """Tensor parallelism configuration."""
    enabled: bool = False
    num_gpus: int = 2

    @field_validator("num_gpus")
    @classmethod
    def validate_num_gpus(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"num_gpus must be >= 1, got {v}")
        return v


class LoRASettings(BaseModel):
    """LoRA multi-adapter configuration."""
    enabled: bool = False
    adapters: Dict[str, str] = Field(default_factory=dict)


class MoESettings(BaseModel):
    """Mixture of Experts configuration."""
    enabled: bool = False
    num_experts: int = 8
    num_experts_per_tok: int = 2


class GossipSettings(BaseModel):
    """P2P KV cache gossip protocol configuration."""
    enabled: bool = False
    interval: float = 10.0
    max_peers: int = 16
    cache_ttl: float = 300.0


class CompressionSettings(BaseModel):
    """Model compression pipeline configuration."""
    enabled: bool = False
    method: str = "none"
    target_bits: int = 8
    pruning_ratio: float = 0.0
    distillation_teacher: Optional[str] = None
    calibration_samples: int = 128
    pruning_targets: List[str] = ["q_proj", "v_proj"]

    @field_validator("target_bits")
    @classmethod
    def validate_bits(cls, v: int) -> int:
        if v not in (4, 8):
            raise ValueError(f"target_bits must be 4 or 8, got {v}")
        return v

    @field_validator("pruning_ratio")
    @classmethod
    def validate_pruning_ratio(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"pruning_ratio must be 0.0-1.0, got {v}")
        return v


class AlertingSettings(BaseModel):
    """Prometheus alerting rules configuration."""
    enabled: bool = False
    prometheus_url: str = "http://localhost:9090"
    rule_file: Optional[str] = None

    @field_validator("prometheus_url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError(f"prometheus_url must start with http:// or https://, got '{v}'")
        return v


class ChaosSettings(BaseModel):
    """Chaos engineering fault injection configuration."""
    enabled: bool = False
    allowed_scenarios: List[str] = Field(default_factory=lambda: ["kill_node", "add_latency", "drop_message", "corrupt_data"])
    max_latency_ms: int = 5000

    @field_validator("max_latency_ms")
    @classmethod
    def validate_latency(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"max_latency_ms must be >= 1, got {v}")
        return v


class RolloutStageModel(BaseModel):
    """Single stage in a canary rollout."""
    weight_pct: float
    analysis_duration_s: int = 300


class CanarySettings(BaseModel):
    """Automated canary deployment configuration."""
    enabled: bool = False
    stable_version: str = "stable"
    canary_version: str = "canary"
    rollback_threshold: float = 0.05
    stages: List[RolloutStageModel] = Field(default_factory=lambda: [
        RolloutStageModel(weight_pct=5, analysis_duration_s=300),
        RolloutStageModel(weight_pct=25, analysis_duration_s=600),
        RolloutStageModel(weight_pct=50, analysis_duration_s=600),
        RolloutStageModel(weight_pct=75, analysis_duration_s=300),
        RolloutStageModel(weight_pct=100, analysis_duration_s=300),
    ])

    @field_validator("rollback_threshold")
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        if not (0.0 < v <= 1.0):
            raise ValueError(f"rollback_threshold must be 0.0-1.0, got {v}")
        return v

    @field_validator("stages")
    @classmethod
    def validate_stages(cls, v: List[RolloutStageModel]) -> List[RolloutStageModel]:
        if not v:
            raise ValueError("stages must not be empty")
        for stage in v:
            if not (0 < stage.weight_pct <= 100):
                raise ValueError(f"stage weight_pct must be 0-100, got {stage.weight_pct}")
        return v


class CostSettings(BaseModel):
    """Cost-aware scheduling configuration."""
    enabled: bool = False
    budget_per_hour: float = 0.0
    spot_preference: float = 0.8

    @field_validator("budget_per_hour")
    @classmethod
    def validate_budget(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"budget_per_hour must be >= 0, got {v}")
        return v

    @field_validator("spot_preference")
    @classmethod
    def validate_preference(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"spot_preference must be 0.0-1.0, got {v}")
        return v


class RateLimitSettings(BaseModel):
    """API rate limiting configuration."""
    enabled: bool = False
    default_rpm: float = 60.0
    endpoint_limits: Dict[str, float] = Field(default_factory=lambda: {
        "/v1/chat/completions": 30.0,
        "/v1/completions": 30.0,
        "/health": 120.0,
        "/metrics": 120.0,
    })
    burst_multiplier: float = 1.5

    @field_validator("default_rpm", "burst_multiplier")
    @classmethod
    def validate_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"Must be positive, got {v}")
        return v


class ModelHubSettings(BaseModel):
    """HuggingFace model hub integration configuration."""
    enabled: bool = True
    cache_dir: Optional[str] = None
    max_cache_size_gb: float = 50.0
    offline_mode: bool = False
    hf_token: Optional[str] = None
    download_timeout_s: int = 300

    @field_validator("max_cache_size_gb")
    @classmethod
    def validate_cache_size(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"max_cache_size_gb must be positive, got {v}")
        return v

    @field_validator("download_timeout_s")
    @classmethod
    def validate_timeout(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"download_timeout_s must be >= 1, got {v}")
        return v


class PromptTemplateSettings(BaseModel):
    """Prompt template engine configuration."""
    template: str = "auto"
    custom_template_path: Optional[str] = None

    @field_validator("template")
    @classmethod
    def validate_template(cls, v: str) -> str:
        if not v:
            raise ValueError("template must not be empty")
        return v


class EmbeddingSettings(BaseModel):
    """Embedding and reranking model configuration."""
    embedding_model: str = ""  # Dedicated embedding model (e.g., sentence-transformers)
    rerank_model: str = ""  # Cross-encoder reranking model
    normalize: bool = True  # L2-normalize embeddings
    max_length: int = 512
    batch_size: int = 32


class VersionSettings(BaseModel):
    """Model versioning and A/B testing configuration."""
    enabled: bool = False
    max_versions: int = 4
    shadow_enabled: bool = False
    shadow_pct: float = 0.0  # Percentage of traffic to shadow (0-100)
    blue_green_enabled: bool = False
    ab_testing_enabled: bool = False
    ab_test_split: float = 50.0  # Percentage for variant B (0-100)
    auto_promote_enabled: bool = False
    min_samples: int = 100  # Minimum samples before statistical test
    significance_level: float = 0.05  # p-value threshold


class PluginSettings(BaseModel):
    """Plugin system configuration."""
    enabled: bool = False
    plugins: List[Dict[str, Any]] = Field(default_factory=list)

    @field_validator("plugins")
    @classmethod
    def validate_plugins(cls, v: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for item in v:
            if isinstance(item, dict) and "module" in item:
                if "." not in item["module"]:
                    raise ValueError(f"Plugin module must be fully qualified, got {item['module']}")
        return v


class HybridParallelSettings(BaseModel):
    """Hybrid parallelism (TP + PP + EP) configuration."""
    enabled: bool = False
    auto_detect: bool = True
    tp_enabled: bool = True
    pp_overlap: bool = True
    ep_enabled: bool = True
    force_tp_world_size: int = 0
    force_pp_stages: int = 0


class ZeroCopySettings(BaseModel):
    """Zero-copy GPU tensor transfer configuration."""
    enabled: bool = False
    prefer_rdma: bool = True
    fallback_to_nccl: bool = True
    intranode_ipc: bool = True


class AdaptivePrecisionSettings(BaseModel):
    """Adaptive precision pipeline configuration."""
    enabled: bool = False
    calibration_samples: int = 64
    target_precision: str = "auto"  # "auto", "fp16", "int8"
    max_quality_loss_pct: float = 0.1


class PredictiveCacheSettings(BaseModel):
    """Predictive KV cache management configuration."""
    enabled: bool = False
    gpu_cache_mb: int = 512
    cpu_cache_mb: int = 4096
    pattern_decay_hours: float = 24.0
    min_prefix_len: int = 8
    background_compress_interval_s: int = 300


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
    nodes: List[NodeSettings] = Field(default_factory=list)
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
    rebalancer: RebalancerSettings = Field(default_factory=RebalancerSettings)
    cache_persistence: CachePersistenceSettings = Field(default_factory=CachePersistenceSettings)
    priority: PrioritySettings = Field(default_factory=PrioritySettings)
    multi_model: MultiModelSettings = Field(default_factory=MultiModelSettings)
    tensor_parallel: TensorParallelSettings = Field(default_factory=TensorParallelSettings)
    lora: LoRASettings = Field(default_factory=LoRASettings)
    moe: MoESettings = Field(default_factory=MoESettings)
    gossip: GossipSettings = Field(default_factory=GossipSettings)
    compression: CompressionSettings = Field(default_factory=CompressionSettings)
    alerting: AlertingSettings = Field(default_factory=AlertingSettings)
    chaos: ChaosSettings = Field(default_factory=ChaosSettings)
    canary: CanarySettings = Field(default_factory=CanarySettings)
    cost: CostSettings = Field(default_factory=CostSettings)
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

    @classmethod
    def validate_startup(cls) -> "DistLLMSettings":
        """Load and validate configuration at startup.

        Returns:
            Validated DistLLMSettings instance.

        Raises:
            SystemExit: If validation fails, prints human-readable errors.
        """
        from pydantic import ValidationError

        try:
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
            raise SystemExit(1)
