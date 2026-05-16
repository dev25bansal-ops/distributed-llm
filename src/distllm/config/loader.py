"""Unified configuration system for distributed LLM inference.

Configuration precedence: environment variables > CLI args > config.yaml > defaults

Note: This module is a thin compatibility layer over the pydantic-settings
based configuration in config/settings.py. New code should import from
config.settings directly.
"""

import os
import warnings
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional

import yaml

from distllm.config.settings import (
    DistLLMSettings,
    ModelSettings,
    CoordinatorSettings,
    NodeSettings,
    GenerationSettings,
    NetworkSettings,
    TLSSettings,
    BatchingSettings,
    PrefixCacheSettings,
    ChunkedPrefillSettings,
    MonitoringSettings,
    QuantizationSettings,
    SpeculativeSettings,
    TensorParallelSettings,
    LoRASettings,
    MoESettings,
    NodeRole,
)

# Backward compatibility: alias old dataclass names to new pydantic models
ModelConfig = ModelSettings
CoordinatorConfig = CoordinatorSettings
NodeConfig = NodeSettings
GenerationConfig = GenerationSettings
NetworkConfig = NetworkSettings
TLSConfig = TLSSettings
BatchingConfig = BatchingSettings
PrefixCacheConfig = PrefixCacheSettings
ChunkedPrefillConfig = ChunkedPrefillSettings
MonitoringConfig = MonitoringSettings
QuantizationConfig = QuantizationSettings
SpeculativeConfig = SpeculativeSettings
TensorParallelConfig = TensorParallelSettings
LoRAConfig = LoRASettings
MoEConfig = MoESettings
DistLLMConfig = DistLLMSettings


def load_config_file(path: str) -> dict:
    """Load configuration from a YAML file."""
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def dict_to_config(data: dict) -> DistLLMSettings:
    """Convert a nested dictionary to DistLLMSettings.

    Uses pydantic's model_validate for automatic type coercion and validation.
    """
    return DistLLMSettings.model_validate(data)


def apply_env_overrides(data: dict) -> dict:
    """Apply DISTLLM_* environment variable overrides to a config dict.

    Deprecated: pydantic-settings handles env var parsing automatically.
    """
    warnings.warn(
        "apply_env_overrides is deprecated; use DistLLMSettings with env vars directly",
        DeprecationWarning,
        stacklevel=2,
    )

    mapping = {
        "DISTLLM_MODEL_NAME": ("model", "name"),
        "DISTLLM_MODEL_DTYPE": ("model", "dtype"),
        "DISTLLM_COORDINATOR_HOST": ("coordinator", "host"),
        "DISTLLM_COORDINATOR_PORT": ("coordinator", "port"),
        "DISTLLM_COORDINATOR_API_PORT": ("coordinator", "api_port"),
        "DISTLLM_GENERATION_MAX_NEW_TOKENS": ("generation", "max_new_tokens"),
        "DISTLLM_GENERATION_TEMPERATURE": ("generation", "temperature"),
        "DISTLLM_GENERATION_TOP_P": ("generation", "top_p"),
        "DISTLLM_NETWORK_GRPC_TIMEOUT": ("network", "grpc_timeout"),
        "DISTLLM_NETWORK_MAX_RETRIES": ("network", "max_retries"),
        "DISTLLM_NETWORK_RETRY_DELAY": ("network", "retry_delay"),
        "DISTLLM_TLS_ENABLED": ("tls", "enabled"),
        "DISTLLM_TLS_CERT_DIR": ("tls", "cert_dir"),
        "DISTLLM_BATCHING_MAX_BATCH_SIZE": ("batching", "max_batch_size"),
        "DISTLLM_BATCHING_MAX_TOKENS_PER_BATCH": ("batching", "max_tokens_per_batch"),
        "DISTLLM_PREFIX_CACHE_ENABLED": ("prefix_cache", "enabled"),
        "DISTLLM_PREFIX_CACHE_MAX_ENTRIES": ("prefix_cache", "max_entries"),
        "DISTLLM_PREFIX_CACHE_MIN_PREFIX_LEN": ("prefix_cache", "min_prefix_len"),
        "DISTLLM_CHUNKED_PREFILL_ENABLED": ("chunked_prefill", "enabled"),
        "DISTLLM_CHUNKED_PREFILL_CHUNK_SIZE": ("chunked_prefill", "chunk_size"),
        "DISTLLM_MONITORING_ENABLED": ("monitoring", "enabled"),
        "DISTLLM_QUANTIZATION_METHOD": ("quantization", "method"),
        "DISTLLM_QUANTIZATION_BNB_4BIT_COMPUTE_DTYPE": ("quantization", "bnb_4bit_compute_dtype"),
        "DISTLLM_QUANTIZATION_BNB_4BIT_QUANT_TYPE": ("quantization", "bnb_4bit_quant_type"),
        "DISTLLM_QUANTIZATION_BNB_4BIT_USE_DOUBLE_QUANT": ("quantization", "bnb_4bit_use_double_quant"),
        "DISTLLM_QUANTIZATION_LLM_INT8_THRESHOLD": ("quantization", "llm_int8_threshold"),
        "DISTLLM_SPECULATIVE_DRAFT_MODEL": ("speculative", "draft_model"),
        "DISTLLM_SPECULATIVE_NUM_ASSISTANT_TOKENS": ("speculative", "num_assistant_tokens"),
        "DISTLLM_TENSOR_PARALLEL_ENABLED": ("tensor_parallel", "enabled"),
        "DISTLLM_TENSOR_PARALLEL_NUM_GPUS": ("tensor_parallel", "num_gpus"),
        "DISTLLM_DISTRIBUTED_KV_CACHE_ENABLED": ("distributed_kv_cache", "enabled"),
        "DISTLLM_LORA_ENABLED": ("lora", "enabled"),
        "DISTLLM_KV_OFFLOAD_ENABLED": ("kv_offload", "enabled"),
        "DISTLLM_KV_OFFLOAD_GPU_LIMIT_GB": ("kv_offload", "gpu_limit_gb"),
        "DISTLLM_KV_OFFLOAD_CPU_LIMIT_GB": ("kv_offload", "cpu_limit_gb"),
        "DISTLLM_RING_ATTENTION_ENABLED": ("ring_attention", "enabled"),
        "DISTLLM_RING_ATTENTION_THRESHOLD_TOKENS": ("ring_attention", "threshold_tokens"),
        "DISTLLM_MOE_ENABLED": ("moe", "enabled"),
        "DISTLLM_MOE_NUM_EXPERTS": ("moe", "num_experts"),
        "DISTLLM_MOE_NUM_EXPERTS_PER_TOK": ("moe", "num_experts_per_tok"),
        "DISTLLM_COST_ENABLED": ("cost", "enabled"),
        "DISTLLM_COST_BUDGET_PER_HOUR": ("cost", "budget_per_hour"),
        "DISTLLM_COST_SPOT_PREFERENCE": ("cost", "spot_preference"),
    }

    for env_key, (section, key) in mapping.items():
        value = os.environ.get(env_key)
        if value is not None:
            if section not in data:
                data[section] = {}
            existing = data[section].get(key)
            if isinstance(existing, bool):
                value = value.lower() in ("true", "1", "yes")
            elif isinstance(existing, int):
                value = int(value)
            elif isinstance(existing, float):
                value = float(value)
            data[section][key] = value

    return data


def apply_cli_overrides(data: dict, cli_args: dict) -> dict:
    """Apply CLI argument overrides to a config dict.

    Recognized CLI keys: model, dtype, host, port, api_port, local,
    trust_remote_code, nodes, grpc_timeout, max_retries, retry_delay,
    quantization, draft_model, num_assistant_tokens, tensor_parallel,
    num_gpus, lora, lora_adapters
    """
    warnings.warn(
        "apply_cli_overrides is deprecated; pass overrides to dict_to_config directly",
        DeprecationWarning,
        stacklevel=2,
    )

    if cli_args.get("model"):
        data.setdefault("model", {})["name"] = cli_args["model"]
    if cli_args.get("dtype"):
        data.setdefault("model", {})["dtype"] = cli_args["dtype"]
    if cli_args.get("host"):
        data.setdefault("coordinator", {})["host"] = cli_args["host"]
    if cli_args.get("port"):
        data.setdefault("coordinator", {})["port"] = cli_args["port"]
    if cli_args.get("api_port"):
        data.setdefault("coordinator", {})["api_port"] = cli_args["api_port"]
    if cli_args.get("trust_remote_code") is not None:
        data.setdefault("model", {})["trust_remote_code"] = cli_args["trust_remote_code"]
    if cli_args.get("grpc_timeout"):
        data.setdefault("network", {})["grpc_timeout"] = cli_args["grpc_timeout"]
    if cli_args.get("max_retries"):
        data.setdefault("network", {})["max_retries"] = cli_args["max_retries"]
    if cli_args.get("retry_delay"):
        data.setdefault("network", {})["retry_delay"] = cli_args["retry_delay"]
    if cli_args.get("nodes"):
        data["nodes"] = cli_args["nodes"]
    if cli_args.get("quantization"):
        data.setdefault("quantization", {})["method"] = cli_args["quantization"]
    if cli_args.get("draft_model"):
        data.setdefault("speculative", {})["draft_model"] = cli_args["draft_model"]
    if cli_args.get("num_assistant_tokens"):
        data.setdefault("speculative", {})["num_assistant_tokens"] = cli_args["num_assistant_tokens"]
    if cli_args.get("tensor_parallel"):
        data.setdefault("tensor_parallel", {})["enabled"] = cli_args["tensor_parallel"]
    if cli_args.get("num_gpus"):
        data.setdefault("tensor_parallel", {})["num_gpus"] = cli_args["num_gpus"]
    if cli_args.get("lora"):
        data.setdefault("lora", {})["enabled"] = True
    if cli_args.get("lora_adapters"):
        data.setdefault("lora", {})["adapters"] = cli_args["lora_adapters"]
    if cli_args.get("kv_offload"):
        data.setdefault("kv_offload", {})["enabled"] = cli_args["kv_offload"]
    if cli_args.get("ring_attention"):
        data.setdefault("ring_attention", {})["enabled"] = cli_args["ring_attention"]
    if cli_args.get("moe"):
        data.setdefault("moe", {})["enabled"] = True
    if cli_args.get("cost_budget"):
        data.setdefault("cost", {})["budget_per_hour"] = cli_args["cost_budget"]
    if cli_args.get("cost"):
        data.setdefault("cost", {})["enabled"] = cli_args["cost"]

    return data


def load_config(
    config_path: Optional[str] = None,
    cli_args: Optional[dict] = None,
) -> DistLLMSettings:
    """Load configuration with full precedence: env > CLI > YAML > defaults.

    Deprecated: Use DistLLMSettings directly with pydantic-settings env var
    support, or use server._load_settings() for the full precedence chain.

    Args:
        config_path: Path to YAML config file. If None, looks for config.yaml
                     in current directory, then project root.
        cli_args: Dictionary of CLI argument overrides.

    Returns:
        DistLLMSettings with all overrides applied.
    """
    warnings.warn(
        "load_config is deprecated; use DistLLMSettings directly or server._load_settings()",
        DeprecationWarning,
        stacklevel=2,
    )
    data = {}

    if config_path is None:
        for candidate in ["config.yaml", os.path.join(os.path.dirname(__file__), "..", "..", "..", "config.yaml")]:
            if os.path.exists(candidate):
                config_path = candidate
                break

    if config_path and os.path.exists(config_path):
        yaml_data = load_config_file(config_path)
        data.update(yaml_data)

    if cli_args:
        data = apply_cli_overrides(data, cli_args)

    data = apply_env_overrides(data)

    return dict_to_config(data)
