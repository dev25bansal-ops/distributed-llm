"""Model, quantization, speculative decoding, LoRA, MoE, compression,
embedding, prompt template, and model hub configuration classes."""

from pydantic import BaseModel, Field, field_validator, SecretStr

__all__ = [
    "ModelSettings",
    "QuantizationSettings",
    "SpeculativeSettings",
    "LoRASettings",
    "SloRaSettings",
    "MoESettings",
    "MultiModelSettings",
    "CompressionSettings",
    "AdaptiveCompressionSettings",
    "ModelHubSettings",
    "EmbeddingSettings",
    "PromptTemplateSettings",
]


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


class LoRASettings(BaseModel):
    """LoRA multi-adapter configuration."""
    enabled: bool = False
    adapters: dict[str, str] = Field(default_factory=dict)


class SloRaSettings(BaseModel):
    """SLoRA multi-adapter serving."""
    enabled: bool = False
    max_adapters: int = 64


class MoESettings(BaseModel):
    """Mixture of Experts configuration."""
    enabled: bool = False
    num_experts: int = 8
    num_experts_per_tok: int = 2


class MultiModelSettings(BaseModel):
    """Multi-model serving configuration.

    The ``max_models`` limit is a safety cap — actual capacity depends on
    available GPU memory and model sizes.  A single large model may not fit
    even when ``max_models > 1``.
    """
    models: dict[str, str] = Field(default_factory=dict)  # name -> path
    default_model: str = ""
    max_models: int = Field(default=4, ge=1, description="Maximum number of models to load concurrently. Actual capacity depends on GPU memory.")


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


class AdaptiveCompressionSettings(BaseModel):
    """Adaptive compression during idle periods.

    When cluster utilization falls below ``idle_threshold_pct`` for at least
    ``idle_duration_s`` seconds, a background compression job is triggered
    on the currently loaded model. The compressed variant is registered with
    the hot-swap manager so it can be swapped in during high load.
    """
    enabled: bool = False
    idle_threshold_pct: float = 30.0
    idle_duration_s: int = 60
    check_interval_s: int = 15
    compression_method: str = "int4"
    calibration_samples: int = 128
    output_dir: str = "/tmp/distllm-compress"


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
    def reject_plain_text_token(cls, v: SecretStr | None) -> SecretStr | None:
        """Reject token set directly in config to prevent secret leakage.

        Raises ValueError if the token is set in config rather than
        environment variable. This prevents accidental commits of
        secrets to config files.
        """
        if v is not None:
            import os
            env_token = os.environ.get("DISTLLM__MODEL_HUB__HF_TOKEN") or os.environ.get("HF_TOKEN")
            if env_token is None:
                raise ValueError(
                    "hf_token cannot be set in config file. "
                    "Use DISTLLM__MODEL_HUB__HF_TOKEN or HF_TOKEN environment variable instead. "
                    "This prevents accidental commits of secrets to config files."
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


class EmbeddingSettings(BaseModel):
    """Embedding and reranking model configuration."""
    embedding_model: str = ""  # Dedicated embedding model (e.g., sentence-transformers)
    rerank_model: str = ""  # Cross-encoder reranking model
    normalize: bool = True  # L2-normalize embeddings
    max_length: int = 512
    batch_size: int = 32


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
