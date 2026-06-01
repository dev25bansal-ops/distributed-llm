"""Unified configuration system for distributed LLM inference.

Configuration precedence: environment variables > CLI args > config.yaml > defaults

This module is a thin compatibility layer over the pydantic-settings
based configuration in ``config/settings.py`` and the CLI resolver in
``config/resolver.py``.  New code should import from those modules directly.
"""

import os
import warnings

from loguru import logger

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
from distllm.config.resolver import _find_config, ConfigResolver

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
    """Load configuration from a YAML file.

    Raises:
        ValueError: If the YAML file contains syntax errors.
    """
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in config file '{path}': {e}") from e


def validate_config(data: dict) -> list[str]:
    """Validate a config dict without creating a full DistLLMSettings.

    Returns a list of human-readable error messages.  An empty list
    means the config is valid.

    Usage::

        errors = validate_config(load_config_file("config.yaml"))
        if errors:
            for err in errors:
                print(f"Config error: {err}")
            sys.exit(1)
    """
    from pydantic import ValidationError

    try:
        DistLLMSettings.model_validate(data)
        return []
    except ValidationError as exc:
        errors = []
        for err in exc.errors():
            loc = ".".join(str(l) for l in err["loc"])
            errors.append(f"{loc}: {err['msg']} (got {err.get('input', '?')})")
        return errors


def dict_to_config(data: dict) -> DistLLMSettings:
    """Convert a nested dictionary to DistLLMSettings.

    Uses pydantic's model_validate for automatic type coercion and validation.
    """
    return DistLLMSettings.model_validate(data)


def load_config(
    config_path: str | None = None,
    cli_args: dict | None = None,
) -> DistLLMSettings:
    """Load configuration with full precedence: env > CLI > YAML > defaults.

    Deprecated: Use ``DistLLMSettings()`` directly for env-var-based config,
    or ``DistLLMSettings.from_yaml(config_path, cli_overrides)`` for the full
    precedence chain including YAML and CLI overrides.

    Args:
        config_path: Path to YAML config file. If None, auto-discovers.
        cli_args: Dictionary of CLI argument overrides.

    Returns:
        DistLLMSettings with all overrides applied.
    """
    logger.warning(
        "load_config is deprecated; use DistLLMSettings.from_yaml() or "
        "DistLLMSettings() for env-var-only config"
    )
    warnings.warn(
        "load_config is deprecated; use DistLLMSettings.from_yaml() or "
        "DistLLMSettings() for env-var-only config",
        DeprecationWarning,
        stacklevel=2,
    )

    if config_path is None:
        config_path = _find_config(ConfigResolver.COMMON_CONFIG_CANDIDATES)

    return DistLLMSettings.from_yaml(config_path=config_path, cli_overrides=cli_args)


def apply_env_overrides(data: dict) -> dict:
    """Apply environment variable overrides to config data.

    Deprecated: DistLLMSettings handles env vars automatically via pydantic-settings.
    This function is a no-op stub for backward compatibility.
    """
    warnings.warn(
        "apply_env_overrides is deprecated; DistLLMSettings handles env vars automatically",
        DeprecationWarning,
        stacklevel=2,
    )
    return data


def apply_cli_overrides(data: dict, cli_args: dict | None = None) -> dict:
    """Apply CLI argument overrides to config data.

    Deprecated: Use DistLLMSettings.from_yaml(config_path, cli_overrides) instead.
    This function is a no-op stub for backward compatibility.
    """
    warnings.warn(
        "apply_cli_overrides is deprecated; use DistLLMSettings.from_yaml() with cli_overrides",
        DeprecationWarning,
        stacklevel=2,
    )
    if cli_args:
        data.update(cli_args)
    return data
