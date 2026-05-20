"""Property-based fuzz tests for model configuration and loading.

Covers: ModelPartitioner init, partition_model_across_nodes,
build_rope_scaling_config, architecture registry lookup,
get_model_info parameter validation.
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from distllm.models.partitioner import (
    partition_model_across_nodes,
    get_model_info,
    build_rope_scaling_config,
)
from distllm.core.architecture_registry import (
    ArchitectureInfo,
    AttributeMapping,
    MoEConfig,
    AttentionConfig,
    RoPEScalingDefaults,
    lookup,
    register,
    list_supported_architectures,
    get_model_size_category,
)


# ---------------------------------------------------------------------------
# partition_model_across_nodes
# ---------------------------------------------------------------------------

@given(
    model_name=st.text(min_size=1, max_size=64, alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/-_.:"),
    num_nodes=st.integers(1, 16),
    trust_remote=st.booleans(),
)
@settings(max_examples=50, deadline=None)
def test_partition_model_across_nodes_returns_list(model_name, num_nodes, trust_remote):
    """partition_model_across_nodes always returns a list of (start, end) tuples."""
    try:
        result = partition_model_across_nodes(model_name, num_nodes, trust_remote_code=trust_remote)
        assert isinstance(result, list)
        if result:
            for start, end in result:
                assert isinstance(start, int)
                assert isinstance(end, int)
                assert 0 <= start <= end
    except (ImportError, ValueError, RuntimeError, OSError, KeyError, TypeError, AttributeError):
        pass  # Expected when model isn't actually available


@given(num_nodes=st.integers(0, 32))
@settings(max_examples=20, deadline=None)
def test_partition_zero_or_many_nodes(num_nodes):
    """partition_model_across_nodes handles edge case node counts."""
    try:
        result = partition_model_across_nodes(
            "roneneldan/TinyStories-1M", num_nodes, trust_remote_code=False
        )
        assert isinstance(result, list)
        if num_nodes > 0 and result:
            _, last_end = result[-1]
            assert last_end >= 0
    except (ValueError, RuntimeError, ImportError):
        pass


# ---------------------------------------------------------------------------
# build_rope_scaling_config
# ---------------------------------------------------------------------------

@given(
    model_type=st.sampled_from(["llama", "mistral", "mixtral", "gemma", "phi3", "qwen2"]),
    original_max_pos=st.integers(512, 65536),
    target_max_pos=st.integers(1024, 262144),
    scaling_type=st.sampled_from(["yarn", "linear", "dynamic", "su"]),
    rope_theta=st.floats(1000.0, 1000000.0, allow_nan=False, allow_infinity=False),
    head_dim=st.integers(32, 256),
)
@settings(max_examples=50, deadline=None)
def test_build_rope_scaling_config_params(
    model_type, original_max_pos, target_max_pos, scaling_type, rope_theta, head_dim
):
    """build_rope_scaling_config always returns a valid config dict."""
    try:
        config = build_rope_scaling_config(
            model_type=model_type,
            original_max_pos=original_max_pos,
            target_max_pos=target_max_pos,
            scaling_type=scaling_type,
            rope_theta=rope_theta,
            attention_head_dim=head_dim,
        )
        # Every rope config must have a type
        assert isinstance(config, dict)
        assert "type" in config
        assert config["type"] in ("yarn", "linear", "dynamic", "su")
    except (ValueError, ImportError):
        pass


@given(
    model_type=st.text(min_size=1, max_size=32),
    target_max_pos=st.integers(512, 262144),
)
@settings(max_examples=20, deadline=None)
def test_build_rope_scaling_config_unknown_type(model_type, target_max_pos):
    """build_rope_scaling_config handles unknown model types gracefully."""
    try:
        config = build_rope_scaling_config(
            model_type=model_type,
            original_max_pos=4096,
            target_max_pos=target_max_pos,
        )
        assert isinstance(config, dict)
    except (ValueError, KeyError, ImportError):
        pass


# ---------------------------------------------------------------------------
# Architecture registry
# ---------------------------------------------------------------------------

@st.composite
def architecture_info_strategy(draw):
    """Generate a random ArchitectureInfo with valid field ranges."""
    model_type = draw(st.sampled_from([
        "llama", "mistral", "mixtral", "gemma", "gemma2",
        "phi3", "qwen2", "deepseek_v3", "falcon", "baichuan2",
    ]))
    is_moe = draw(st.booleans()) if "mixtral" in model_type or "deepseek" in model_type else st.just(False)
    if isinstance(is_moe, st._internal.strategies.BooleanStrategy):
        is_moe_val = draw(is_moe)
    else:
        is_moe_val = is_moe
    return ArchitectureInfo(
        name=draw(st.text(min_size=2, max_size=32)),
        model_type=model_type,
        model_type_aliases=[model_type],
        description=draw(st.text(min_size=1, max_size=128)),
        min_vram_gb=draw(st.floats(0.5, 800.0, allow_nan=False, allow_infinity=False)),
        recommended_gpus=draw(st.sampled_from(["1x GPU", "2x GPU", "4x GPU", "8x H100"])),
        trust_remote_code=draw(st.booleans()),
        is_moe=is_moe_val,
        supports_flash_attention=draw(st.booleans()),
        supports_pipeline_parallelism=draw(st.booleans()),
        supports_tensor_parallelism=draw(st.booleans()),
        supports_quantization=draw(st.booleans()),
        attrs=AttributeMapping(
            embed_tokens=draw(st.sampled_from(["embed_tokens", "wte", "embd"])),
            layers=draw(st.sampled_from(["model.layers", "transformer.h", "decoder.layers"])),
            norm=draw(st.sampled_from(["model.norm", "transformer.ln_f", "decoder.final_norm"])),
            lm_head=draw(st.sampled_from(["lm_head", "embed_out"])),
        ),
        moe=MoEConfig(
            num_experts=draw(st.integers(2, 256)) if is_moe_val else 1,
            top_k=draw(st.integers(1, 8)) if is_moe_val else 1,
        ),
        attention=AttentionConfig(
            causal=draw(st.booleans()),
            use_qkv_packed=draw(st.booleans()),
            use_gqa=draw(st.booleans()),
            supports_sliding_window=draw(st.booleans()),
            sliding_window_size=draw(st.integers(0, 65536)),
        ),
        rope=RoPEScalingDefaults(
            rope_theta=draw(st.floats(10000.0, 1000000.0, allow_nan=False, allow_infinity=False)),
            scaling_factor=draw(st.floats(1.0, 64.0, allow_nan=False, allow_infinity=False)),
            original_max_pos=draw(st.integers(512, 65536)),
            max_position_embeddings=draw(st.integers(1024, 262144)),
        ),
        default_num_layers=draw(st.integers(1, 256)),
        default_hidden_size=draw(st.sampled_from([4096, 5120, 8192, 10240, 14336, 24576])),
        default_num_heads=draw(st.sampled_from([8, 16, 32, 40, 64, 128])),
        default_num_kv_heads=draw(st.sampled_from([4, 8, 16, 32])),
        default_intermediate_size=draw(st.sampled_from([11008, 14336, 20480, 27392, 52224])),
        default_vocab_size=draw(st.sampled_from([32000, 128256, 152064, 32001])),
    )


@given(architecture_info_strategy())
@settings(max_examples=50, deadline=None)
def test_architecture_info_roundtrip(info):
    """ArchitectureInfo can be registered and looked up."""
    register(info)
    result = lookup(info.model_type)
    if result is not None:
        assert result.model_type == info.model_type


@given(st.lists(architecture_info_strategy(), min_size=1, max_size=5))
@settings(max_examples=20, deadline=None)
def test_list_supported_architectures_returns_all(infos):
    """list_supported_architectures returns all registered architectures."""
    for info in infos:
        register(info)
    listed = list_supported_architectures()
    registered_types = {i.model_type for i in infos}
    listed_types = {l["model_type"] for l in listed}
    for t in registered_types:
        assert t in listed_types


@given(architecture_info_strategy())
@settings(max_examples=20, deadline=None)
def test_get_model_size_category_valid(info):
    """get_model_size_category returns a valid category string."""
    register(info)
    try:
        category = get_model_size_category(info.model_type, info.name)
        assert isinstance(category, str)
        assert category in ("small", "medium", "large", "xlarge", "xxlarge", "unknown")
    except (KeyError, ValueError):
        pass


# ---------------------------------------------------------------------------
# get_model_info
# ---------------------------------------------------------------------------

@given(
    model_name=st.text(min_size=1, max_size=64),
    trust_remote=st.booleans(),
)
@settings(max_examples=30, deadline=None)
def test_get_model_info_returns_dict(model_name, trust_remote):
    """get_model_info always returns a dict (possibly empty on error)."""
    try:
        info = get_model_info(model_name, trust_remote_code=trust_remote)
        assert isinstance(info, dict)
    except (ImportError, OSError, RuntimeError, ValueError, KeyError, AttributeError):
        pass
