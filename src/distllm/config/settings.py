"""Pydantic-settings based configuration for distributed LLM inference.

All config classes use pydantic BaseModel for validation and BaseSettings
for environment variable support with env_prefix="DISTLLM_" and nested
delimiter "__" (e.g., DISTLLM__MODEL__NAME).
"""

from enum import Enum
import os
from typing import Any, TYPE_CHECKING

from pydantic import BaseModel, Field, field_validator, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

if TYPE_CHECKING:
    pass


class NodeRole(str, Enum):
    """Node role for prefill-decode disaggregation."""
    AUTO = "auto"
    PREFILL = "prefill"
    DECODE = "decode"


class ModelSettings(BaseModel):
    """Model configuration."""
    name: str = Field(default="", description="Model name or path. Must be explicitly set.")
    dtype: str = "float16"
    trust_remote_code: bool = False

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError(
                "model.name must be set — specify a HuggingFace model ID (e.g. "
                "'meta-llama/Llama-2-7b') or a local path, or set the "
                "DISTLLM__MODEL__NAME environment variable."
            )
        return v.strip()

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
        # Validate each origin is a well-formed URL
        for origin in v.split(","):
            origin = origin.strip()
            if origin == "*":
                continue  # Wildcard handled separately in _get_cors_origins
            if not (origin.startswith("http://") or origin.startswith("https://") or origin.startswith("chrome-extension://") or origin.startswith("moz-extension://")):
                raise ValueError(
                    f"CORS origin '{origin}' must be a valid URL starting with http://, https://, "
                    f"or a browser extension scheme."
                )
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
    cert_file: str | None = None
    key_file: str | None = None
    ca_cert_file: str | None = None


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
    method: str = "draft_model"  # "draft_model" | "medusa" | "eagle" | "ngram" | "auto"
    medusa_num_heads: int = 4
    medusa_num_tokens_per_head: int = 3
    eagle_checkpoint: str = ""
    eagle_variant: str = "eagle"
    eagle_hidden_size: int = 4096
    eagle_vocab_size: int = 32000
    eagle_num_layers: int = 2
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
        allowed = {"draft_model", "medusa", "eagle", "ngram", "auto"}
        if v not in allowed:
            raise ValueError(f"method must be one of {allowed}, got '{v}'")
        return v


class PartitioningSettings(BaseModel):
    """Layer partitioning strategy configuration (legacy)."""
    strategy: str = "gpu_aware"  # "equal" | "gpu_aware"
    safety_margin: float = 0.1  # leave 10% VRAM free

    def to_auto_partition_config(self):
        """Convert to dict (legacy AutoPartitionConfig)."""
        return {
            "enabled": self.strategy != "equal",
            "strategy": self.strategy if self.strategy != "gpu_aware" else "auto",
            "safety_margin": self.safety_margin,
        }

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
    grace_period_steps: int = 3
    auto_mitigate: bool = False


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
    """Multi-model serving configuration.

    The ``max_models`` limit is a safety cap — actual capacity depends on
    available GPU memory and model sizes.  A single large model may not fit
    even when ``max_models > 1``.
    """
    models: dict[str, str] = Field(default_factory=dict)  # name -> path
    default_model: str = ""
    max_models: int = Field(default=4, ge=1, description="Maximum number of models to load concurrently. Actual capacity depends on GPU memory.")


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
    adapters: dict[str, str] = Field(default_factory=dict)


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
    distillation_teacher: str | None = None
    calibration_samples: int = 128
    pruning_targets: list[str] = ["q_proj", "v_proj"]

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
    rule_file: str | None = None

    @field_validator("prometheus_url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError(f"prometheus_url must start with http:// or https://, got '{v}'")
        return v


class ChaosSettings(BaseModel):
    """Chaos engineering fault injection configuration."""
    enabled: bool = False
    allowed_scenarios: list[str] = Field(default_factory=lambda: ["kill_node", "add_latency", "drop_message", "corrupt_data"])
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
    stages: list[RolloutStageModel] = Field(default_factory=lambda: [
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
    def validate_stages(cls, v: list[RolloutStageModel]) -> list[RolloutStageModel]:
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
    enabled: bool = True
    default_rpm: float = 60.0
    endpoint_limits: dict[str, float] = Field(default_factory=lambda: {
        "/v1/chat/completions": 30.0,
        "/v1/completions": 30.0,
        "/health": 120.0,
        "/metrics": 120.0,
    })
    burst_multiplier: float = 1.5
    # Security: Separate rate limits for authenticated vs unauthenticated clients
    auth_rpm_multiplier: float = 2.0  # Authenticated clients get higher limits

    @field_validator("default_rpm", "burst_multiplier", "auth_rpm_multiplier")
    @classmethod
    def validate_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"Must be positive, got {v}")
        return v


class ModelHubSettings(BaseModel):
    """HuggingFace model hub integration configuration."""
    enabled: bool = True
    cache_dir: str | None = None
    max_cache_size_gb: float = 50.0
    offline_mode: bool = False
    hf_token: SecretStr | None = Field(default=None, description="HuggingFace token. Set via DISTLLM__MODEL_HUB__HF_TOKEN env var or .env file, NOT in YAML config.")
    download_timeout_s: int = 300

    @field_validator("hf_token")
    @classmethod
    def warn_if_plain_text(cls, v: SecretStr | None) -> SecretStr | None:
        """Log a warning if token is set (reminds users to use env vars)."""
        if v is not None:
            import os
            env_token = os.environ.get("DISTLLM__MODEL_HUB__HF_TOKEN") or os.environ.get("HF_TOKEN")
            if env_token is None:
                import warnings
                warnings.warn(
                    "hf_token is set in config rather than environment variable. "
                    "Consider using DISTLLM__MODEL_HUB__HF_TOKEN or HF_TOKEN env var to avoid "
                    "committing secrets to config files.",
                    UserWarning,
                    stacklevel=2,
                )
        return v

    @property
    def hf_token_value(self) -> str | None:
        """Get the actual token value. Prefer env var over config value."""
        import os
        env_token = os.environ.get("DISTLLM__MODEL_HUB__HF_TOKEN") or os.environ.get("HF_TOKEN")
        if env_token:
            return env_token
        return self.hf_token.get_secret_value() if self.hf_token else None

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
    custom_template_path: str | None = None

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
    plugins: list[dict[str, Any]] = Field(default_factory=list)

    @field_validator("plugins")
    @classmethod
    def validate_plugins(cls, v: list[dict[str, Any]]) -> list[dict[str, Any]]:
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


class SelfOptimizingSettings(BaseModel):
    """Auto-tuning via hill-climbing optimization (legacy)."""
    enabled: bool = False
    tune_interval_seconds: float = 60.0
    warmup_seconds: float = 30.0
    profile_dir: str | None = None

    def to_optimization_config(self):
        """Convert to the new Bayesian OptimizationConfig."""
        return {
            "enabled": self.enabled,
            "runner": {"warmup_seconds": self.warmup_seconds},
        }


class CudaGraphSettings(BaseModel):
    """CUDA graph capture for decode acceleration."""
    enabled: bool = False
    batch_sizes: list[int] = [1, 2, 4, 8, 16, 32]


class CompileSettings(BaseModel):
    """torch.compile integration."""
    enabled: bool = False
    mode: str = "reduce-overhead"
    fullgraph: bool = False


class SloRaSettings(BaseModel):
    """SLoRA multi-adapter serving."""
    enabled: bool = False
    max_adapters: int = 64


class RAGSettings(BaseModel):
    """RAG pipeline with FAISS."""
    enabled: bool = False
    dimension: int = 768
    chunk_size: int = 512
    chunk_overlap: int = 50
    index_path: str | None = None


class AgentSettings(BaseModel):
    """ReAct agent loop."""
    enabled: bool = False
    max_iterations: int = 10
    reflection_enabled: bool = True


class DisaggSettings(BaseModel):
    """Disaggregated prefill/decode serving configuration.

    Delegates to the full config model from the disagg package.
    """

    enabled: bool = False
    prefill_nodes: list[dict] = []
    decode_nodes: list[dict] = []

    def to_full_config(self):
        """Convert to the package-level DisaggFullConfig."""
        return {
            "enabled": self.enabled,
            "prefill_nodes": self.prefill_nodes,
            "decode_nodes": self.decode_nodes,
        }


class WideAreaSettings(BaseModel):
    """Wide-area network distributed inference configuration.

    Enables P2P node forwarding to reduce coordinator round trips
    across high-latency links (geographically distributed nodes).
    """
    enabled: bool = False
    p2p_forwarding: bool = True
    tokens_before_forward: int = 10
    wan_timeout_seconds: int = 60
    max_retries: int = 3
    backoff_base_seconds: float = 1.0

    @field_validator("tokens_before_forward")
    @classmethod
    def validate_tokens_before_forward(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"tokens_before_forward must be >= 1, got {v}")
        return v

    @field_validator("wan_timeout_seconds")
    @classmethod
    def validate_timeout(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"wan_timeout_seconds must be >= 1, got {v}")
        return v


class VLLMSettings(BaseModel):
    """vLLM backend configuration."""
    enabled: bool = False
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 8192
    dtype: str = "auto"
    seed: int = 0
    enforce_eager: bool = False
    max_model_len: int | None = None

    @field_validator("gpu_memory_utilization")
    @classmethod
    def validate_gpu_memory_utilization(cls, v: float) -> float:
        if not 0 < v <= 1:
            raise ValueError(f"gpu_memory_utilization must be in (0, 1], got {v}")
        return v

    @field_validator("tensor_parallel_size")
    @classmethod
    def validate_tp_size(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"tensor_parallel_size must be >= 1, got {v}")
        return v

    @field_validator("dtype")
    @classmethod
    def validate_dtype(cls, v: str) -> str:
        allowed = {"auto", "float16", "float32", "bfloat16", "half", "full"}
        if v not in allowed:
            raise ValueError(f"dtype must be one of {allowed}, got '{v}'")
        return v


class LlamacppSettings(BaseModel):
    """llama.cpp backend configuration.

    Lightweight alternative to vLLM for CPU/GPU inference with GGUF models.
    Supports CPU, CUDA, AMD ROCm, and Apple Metal backends.
    """
    enabled: bool = False
    model_path: str = ""
    n_gpu_layers: int = 0
    n_ctx: int = 2048
    n_threads: int | None = None
    n_batch: int = 512
    seed: int = 0
    verbose: bool = False

    @field_validator("model_path")
    @classmethod
    def validate_model_path(cls, v: str, info) -> str:
        if info.data.get("enabled", False) and not v:
            raise ValueError("model_path is required when llamacpp is enabled")
        return v

    @field_validator("n_gpu_layers")
    @classmethod
    def validate_n_gpu_layers(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"n_gpu_layers must be >= 0, got {v}")
        return v

    @field_validator("n_ctx")
    @classmethod
    def validate_n_ctx(cls, v: int) -> int:
        if v < 128:
            raise ValueError(f"n_ctx must be >= 128, got {v}")
        return v


class RouteRuleSettings(BaseModel):
    """A single routing rule for the multi-model chat router."""
    name: str = Field(default="", description="Rule name for identification")
    match_type: str = Field(default="keyword", description="Matching strategy: keyword, regex, or workload")
    match: str = Field(default="", description="Pattern to match against the user message")
    target_model: str = Field(default="", description="Model to route to when this rule matches")
    priority: int = Field(default=0, ge=0, description="Rule priority (higher = evaluated first)")


class ChatRouterSettings(BaseModel):
    """Multi-model chat router configuration.

    Allows defining compound/hybrid models that route queries to different
    backend models based on content matching rules.

    Example config (YAML):
    .. code-block:: yaml

        chat_router:
          enabled: true
          name: hybrid
          default_model: llama3
          routes:
            - name: code-route
              match_type: keyword
              match: "write a function"
              target_model: codellama
              priority: 10
            - name: creative-route
              match_type: keyword
              match: "write a story"
              target_model: llama3
              priority: 5
    """
    enabled: bool = False
    name: str = Field(default="hybrid", description="Model name that clients use to invoke this router (e.g., 'hybrid', 'smart-router')")
    default_model: str = Field(default="", description="Default model when no rules match")
    routes: list[RouteRuleSettings] = Field(default_factory=list, description="Ordered routing rules")


class TenantSettings(BaseModel):
    """Multi-tenant SaaS configuration."""
    enabled: bool = False
    default_tier: str = "free"
    admin_api_key: SecretStr | None = Field(default=None, description="Admin API key for tenant management. Set via DISTLLM__TENANT__ADMIN_API_KEY env var.")


class HardwareSettings(BaseModel):
    """Multi-architecture hardware configuration.

    Controls device selection, backend preference, and architecture-
    specific settings for heterogeneous clusters.
    """
    device_type: str = "auto"  # "auto" | "cuda" | "rocm" | "mps" | "xpu" | "cpu"
    preferred_backend: str = "auto"  # "auto" | "vllm" | "pytorch" | "llamacpp"
    force_device_id: int = -1  # -1 = auto-select
    fallback_to_cpu: bool = True

    # Architecture-specific overrides
    rocm_visible_devices: str = ""
    mps_optimize_memory: bool = True
    xpu_oneapi_verbose: bool = False
    cpu_threads: int = 0  # 0 = auto-detect via psutil
    cpu_numa_aware: bool = True

    @field_validator("device_type")
    @classmethod
    def validate_device_type(cls, v: str) -> str:
        allowed = {"auto", "cuda", "rocm", "mps", "xpu", "cpu"}
        if v not in allowed:
            raise ValueError(f"device_type must be one of {allowed}, got '{v}'")
        return v

    @field_validator("preferred_backend")
    @classmethod
    def validate_backend(cls, v: str) -> str:
        allowed = {"auto", "vllm", "pytorch", "llamacpp"}
        if v not in allowed:
            raise ValueError(f"preferred_backend must be one of {allowed}, got '{v}'")
        return v


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
        from pydantic import ValidationError

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
