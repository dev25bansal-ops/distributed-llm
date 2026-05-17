"""Central registry for supported model architectures.

Provides per-architecture configuration: attribute name mappings, MoE configs,
attention types, RoPE scaling defaults, hardware requirements, and special
handling for each architecture.

This is the single source of truth for architecture-specific knowledge.
The rest of the codebase looks up architectures here rather than hardcoding
attribute names or model_type checks in multiple places.

Supported architectures:
  - DeepSeek-V3 (671B MoE): 256 experts, shared expert, MLA
  - Qwen2.5 (72B): standard transformer, GQA
  - Llama 3.1 (405B): largest dense model, 126 layers
  - Mixtral (8x22B): 8 expert MoE, top-2 routing
  - Gemma 2 (27B): soft-cap attention, alternating sliding window
  - Phi-3.5 (3.8B): small efficient transformer
  - Falcon (180B): parallel attention/MLP, different naming
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AttributeMapping:
    """Attribute name mapping for a model architecture.

    Maps logical component names to the actual attribute names
    used by this architecture in HuggingFace.
    """
    base_model: list[str] = field(default_factory=lambda: ["model", "transformer", "encoder"])
    embedding: list[str] = field(default_factory=lambda: ["embed_tokens", "wte", "word_embeddings"])
    layers: list[str] = field(default_factory=lambda: ["layers", "block", "h"])
    final_norm: list[str] = field(default_factory=lambda: ["norm", "final_layer_norm", "ln_f"])
    lm_head: list[str] = field(default_factory=lambda: ["lm_head", "embed_out"])

    # Attention projection names for FlashAttention patching
    attn_q: str = "q_proj"
    attn_k: str = "k_proj"
    attn_v: str = "v_proj"
    attn_o: str = "o_proj"


@dataclass
class MoEConfig:
    """MoE-specific configuration for the architecture."""
    num_experts: int = 0
    num_routed_experts: int = 0
    num_shared_experts: int = 0
    top_k: int = 2
    moe_layer_frequency: int = 1      # Every N layers are MoE (1=all, 4=every 4th)
    shared_expert_intermediate_size: int = 0
    norm_topk_prob: bool = False


@dataclass
class AttentionConfig:
    """Attention-specific configuration."""
    attention_type: str = "sdpa"       # sdpa, flash_attn, mla, sliding_window
    use_qkv_parallel: bool = False     # Parallel QKV (Falcon style)
    soft_cap: float | None = None   # Attention logit soft-capping (Gemma 2)
    sliding_window_size: int = 0       # Sliding window attention (Mistral, Gemma 2)
    use_gqa: bool = True               # Grouped-query attention
    kv_compression_ratio: int = 0      # MLA KV compression (DeepSeek)


@dataclass
class RoPEScalingDefaults:
    """Default RoPE scaling configuration for the architecture."""
    base: float = 10000.0
    scaling_type: str = "linear"       # linear, ntk, yarn, dynamic_ntk
    max_position_embeddings: int = 4096
    original_max_position_embeddings: int = 4096
    rope_theta: float | None = None # Override for rope_theta if different from base


@dataclass
class ArchitectureInfo:
    """Complete specification for a supported model architecture."""
    name: str
    model_type: str                      # HuggingFace config.model_type
    model_type_aliases: list[str] = field(default_factory=list)
    description: str = ""
    min_vram_gb: float = 0.0
    recommended_gpus: str = ""
    trust_remote_code: bool = False
    is_moe: bool = False
    supports_flash_attention: bool = True
    supports_pipeline_parallelism: bool = True
    supports_tensor_parallelism: bool = True
    supports_quantization: bool = True

    attrs: AttributeMapping = field(default_factory=AttributeMapping)
    moe: MoEConfig = field(default_factory=MoEConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    rope: RoPEScalingDefaults = field(default_factory=RoPEScalingDefaults)

    # Model size reference
    default_num_layers: int = 32
    default_hidden_size: int = 4096
    default_num_heads: int = 32
    default_num_kv_heads: int = 0       # 0 = same as num_heads
    default_intermediate_size: int = 11008
    default_vocab_size: int = 32000


# ---------------------------------------------------------------------------
# Architecture Registry
# ---------------------------------------------------------------------------

ARCHITECTURE_REGISTRY: dict[str, ArchitectureInfo] = {}


def register(info: ArchitectureInfo) -> None:
    """Register an architecture by its primary model_type."""
    ARCHITECTURE_REGISTRY[info.model_type] = info
    for alias in info.model_type_aliases:
        ARCHITECTURE_REGISTRY[alias] = info


def lookup(model_type: str) -> ArchitectureInfo | None:
    """Look up architecture info by model_type string."""
    return ARCHITECTURE_REGISTRY.get(model_type)


def lookup_by_model_name(model_name: str) -> ArchitectureInfo | None:
    """Look up architecture info by a HuggingFace model name/path.

    Checks the lowercase model name against allowlist-based detection.
    """
    lower = model_name.lower()
    for info in ARCHITECTURE_REGISTRY.values():
        if info.model_type in lower:
            return info
        for alias in info.model_type_aliases:
            if alias in lower:
                return info
    return None


def get_trusted_models() -> set[str]:
    """Return the set of model families that require trust_remote_code."""
    trusted: set[str] = set()
    for info in ARCHITECTURE_REGISTRY.values():
        if info.trust_remote_code:
            trusted.add(info.model_type)
            trusted.update(info.model_type_aliases)
    return trusted


# ---------------------------------------------------------------------------
# Register: DeepSeek-V3 (671B MoE)
# ---------------------------------------------------------------------------
register(ArchitectureInfo(
    name="DeepSeek-V3",
    model_type="deepseek_v3",
    model_type_aliases=["deepseek", "deepseek_v2", "deepseek-v2", "deepseek-v3"],
    description="DeepSeek-V3: 671B MoE with 256 routed experts + shared expert, MLA attention",
    min_vram_gb=720.0,
    recommended_gpus="8x H100 (NVLink)",
    trust_remote_code=True,
    is_moe=True,
    supports_tensor_parallelism=True,
    supports_pipeline_parallelism=True,
    attrs=AttributeMapping(
        base_model=["model", "transformer"],
        embedding=["embed_tokens", "wte"],
        layers=["layers", "block"],
        final_norm=["norm", "final_layer_norm"],
    ),
    moe=MoEConfig(
        num_experts=256,
        num_routed_experts=256,
        num_shared_experts=1,
        top_k=8,
        moe_layer_frequency=1,
        shared_expert_intermediate_size=2048,
        norm_topk_prob=True,
    ),
    attention=AttentionConfig(
        attention_type="mla",
        kv_compression_ratio=4,  # MLA compresses KV to 1/4 hidden dim
    ),
    rope=RoPEScalingDefaults(
        base=10000.0,
        max_position_embeddings=163840,
        original_max_position_embeddings=4096,
    ),
    default_num_layers=67,
    default_hidden_size=7168,
    default_num_heads=128,
    default_num_kv_heads=128,
    default_intermediate_size=2048,
    default_vocab_size=129280,
))

# ---------------------------------------------------------------------------
# Register: Qwen2.5 (72B)
# ---------------------------------------------------------------------------
register(ArchitectureInfo(
    name="Qwen2.5",
    model_type="qwen2",
    model_type_aliases=["qwen2.5", "qwen2_5", "qwen"],
    description="Qwen2.5: strong open model, GQA, SwiGLU, RoPE",
    min_vram_gb=145.0,
    recommended_gpus="2-4x H100",
    trust_remote_code=False,
    attrs=AttributeMapping(
        base_model=["model", "transformer"],
        embedding=["embed_tokens", "wte"],
        layers=["layers", "block"],
        final_norm=["norm", "final_layer_norm"],
    ),
    rope=RoPEScalingDefaults(
        base=1000000.0,  # Qwen uses 1M base for RoPE
        max_position_embeddings=32768,
        original_max_position_embeddings=32768,
        rope_theta=1000000.0,
    ),
    default_num_layers=80,
    default_hidden_size=8192,
    default_num_heads=64,
    default_num_kv_heads=8,
    default_intermediate_size=29568,
    default_vocab_size=152064,
))

# ---------------------------------------------------------------------------
# Register: Llama 3.1 (405B)
# ---------------------------------------------------------------------------
register(ArchitectureInfo(
    name="Llama 3.1",
    model_type="llama",
    model_type_aliases=["llama3", "llama-3", "llama-3.1"],
    description="Llama 3.1 405B: largest open dense model, GQA, RoPE, SwiGLU",
    min_vram_gb=810.0,
    recommended_gpus="8x H100 (NVLink)",
    trust_remote_code=False,
    attrs=AttributeMapping(
        base_model=["model", "transformer"],
        embedding=["embed_tokens", "wte"],
        layers=["layers", "block"],
        final_norm=["norm", "final_layer_norm"],
    ),
    rope=RoPEScalingDefaults(
        base=500000.0,   # Llama 3 uses 500K base
        max_position_embeddings=131072,
        original_max_position_embeddings=8192,
        rope_theta=500000.0,
    ),
    default_num_layers=126,
    default_hidden_size=16384,
    default_num_heads=128,
    default_num_kv_heads=8,
    default_intermediate_size=53248,
    default_vocab_size=128256,
))

# ---------------------------------------------------------------------------
# Register: Mixtral 8x22B
# ---------------------------------------------------------------------------
register(ArchitectureInfo(
    name="Mixtral 8x22B",
    model_type="mistral",
    model_type_aliases=["mixtral"],
    description="Mixtral 8x22B: 8 expert MoE, top-2 routing, sliding window attention",
    min_vram_gb=90.0,
    recommended_gpus="2x H100",
    trust_remote_code=False,
    is_moe=True,
    supports_tensor_parallelism=True,
    supports_pipeline_parallelism=True,
    attrs=AttributeMapping(
        base_model=["model", "transformer"],
        embedding=["embed_tokens", "wte"],
        layers=["layers", "block"],
        final_norm=["norm", "final_layer_norm"],
    ),
    moe=MoEConfig(
        num_experts=8,
        num_routed_experts=8,
        num_shared_experts=0,
        top_k=2,
        moe_layer_frequency=1,
        norm_topk_prob=False,
    ),
    attention=AttentionConfig(
        sliding_window_size=4096,
    ),
    rope=RoPEScalingDefaults(
        base=1000000.0,
        max_position_embeddings=32768,
        original_max_position_embeddings=32768,
        rope_theta=1000000.0,
    ),
    default_num_layers=56,
    default_hidden_size=6144,
    default_num_heads=48,
    default_num_kv_heads=8,
    default_intermediate_size=16384,
    default_vocab_size=32000,
))

# ---------------------------------------------------------------------------
# Register: Gemma 2 (27B)
# ---------------------------------------------------------------------------
register(ArchitectureInfo(
    name="Gemma 2",
    model_type="gemma2",
    model_type_aliases=["gemma"],
    description="Gemma 2 27B: Google open model, soft-cap attention, alternating sliding window, GeGLU",
    min_vram_gb=56.0,
    recommended_gpus="1-2x H100",
    trust_remote_code=False,
    supports_flash_attention=True,
    supports_quantization=True,
    attrs=AttributeMapping(
        base_model=["model", "transformer"],
        embedding=["embed_tokens", "wte"],
        layers=["layers", "block"],
        final_norm=["norm", "final_layer_norm", "ln_f"],
    ),
    attention=AttentionConfig(
        attention_type="sdpa",
        soft_cap=50.0,           # Gemma 2 uses attention logit soft-capping
        sliding_window_size=4096,
    ),
    rope=RoPEScalingDefaults(
        base=10000.0,
        max_position_embeddings=8192,
        original_max_position_embeddings=8192,
    ),
    default_num_layers=46,
    default_hidden_size=4608,
    default_num_heads=32,
    default_num_kv_heads=16,
    default_intermediate_size=36864,
    default_vocab_size=256000,
))

# ---------------------------------------------------------------------------
# Register: Phi-3.5 (3.8B)
# ---------------------------------------------------------------------------
register(ArchitectureInfo(
    name="Phi-3.5",
    model_type="phi3",
    model_type_aliases=["phi-3", "phi3.5", "phi-3.5"],
    description="Phi-3.5 3.8B: Microsoft small efficient model, ideal for edge deployments",
    min_vram_gb=8.0,
    recommended_gpus="1x consumer GPU (RTX 3090+)",
    trust_remote_code=True,     # Phi-3 requires trust_remote_code
    supports_pipeline_parallelism=False,
    supports_tensor_parallelism=False,
    supports_quantization=True,
    attrs=AttributeMapping(
        base_model=["model", "transformer"],
        embedding=["embed_tokens", "wte"],
        layers=["layers", "block"],
        final_norm=["norm", "final_layer_norm", "ln_f"],
        lm_head=["lm_head", "embed_out"],
    ),
    rope=RoPEScalingDefaults(
        base=10000.0,
        max_position_embeddings=4096,
        original_max_position_embeddings=4096,
    ),
    default_num_layers=32,
    default_hidden_size=3072,
    default_num_heads=32,
    default_num_kv_heads=0,
    default_intermediate_size=8192,
    default_vocab_size=32064,
))

# ---------------------------------------------------------------------------
# Register: Falcon 180B
# ---------------------------------------------------------------------------
register(ArchitectureInfo(
    name="Falcon",
    model_type="falcon",
    model_type_aliases=["falcon-180b"],
    description="Falcon 180B: TII's large dense model, parallel attention/MLP, multi-query attention",
    min_vram_gb=360.0,
    recommended_gpus="4-8x H100",
    trust_remote_code=False,
    supports_flash_attention=True,
    supports_tensor_parallelism=True,
    supports_pipeline_parallelism=True,
    attrs=AttributeMapping(
        base_model=["transformer", "model"],    # Falcon uses 'transformer' not 'model'
        embedding=["word_embeddings", "embed_tokens", "wte"],
        layers=["h", "layers", "block"],        # Falcon uses 'h' for layers
        final_norm=["ln_f", "norm", "final_layer_norm"],
        lm_head=["lm_head", "embed_out"],
        attn_q="query_key_value",               # Falcon uses fused QKV
        attn_o="dense",
    ),
    attention=AttentionConfig(
        attention_type="sdpa",
        use_qkv_parallel=True,                   # Falcon does QKV in parallel
    ),
    rope=RoPEScalingDefaults(
        base=10000.0,
        max_position_embeddings=2048,
        original_max_position_embeddings=2048,
    ),
    default_num_layers=80,
    default_hidden_size=14848,
    default_num_heads=148,                       # Falcon uses 148 heads (multi-query)
    default_num_kv_heads=8,                      # Multi-query: 8 KV heads
    default_intermediate_size=44928,
    default_vocab_size=65024,
))


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def get_architecture_info(model_type: str, model_name: str = "") -> ArchitectureInfo:
    """Get architecture info, falling back to model_name lookup or defaults.

    Args:
        model_type: HuggingFace config.model_type value.
        model_name: Full model name for fallback lookup.

    Returns:
        ArchitectureInfo for the model, or a sensible default.
    """
    info = lookup(model_type)
    if info is not None:
        return info

    if model_name:
        info = lookup_by_model_name(model_name)
        if info is not None:
            return info

    return ArchitectureInfo(
        name=model_type or model_name or "unknown",
        model_type=model_type or "unknown",
    )


def list_supported_architectures() -> list[dict[str, Any]]:
    """Return a human-readable list of supported architectures."""
    result = []
    for info in ARCHITECTURE_REGISTRY.values():
        result.append({
            "name": info.name,
            "model_type": info.model_type,
            "parameters": f"{info.default_num_layers} layers, {info.default_hidden_size} hidden",
            "is_moe": info.is_moe,
            "min_vram_gb": info.min_vram_gb,
            "recommended_gpus": info.recommended_gpus,
            "trust_remote_code": info.trust_remote_code,
        })
    return result


def get_model_size_category(model_type: str, model_name: str = "") -> str:
    """Categorize model size for parallelism strategy selection."""
    info = get_architecture_info(model_type, model_name)
    vr = info.min_vram_gb
    if vr >= 500:
        return "massive"     # 405B, 671B — requires full distributed
    if vr >= 100:
        return "large"       # 70B-180B — requires multi-node
    if vr >= 30:
        return "medium"      # 13B-34B — fits on 2-4 GPUs
    return "small"           # < 13B — fits on single GPU
