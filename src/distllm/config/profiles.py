"""Configuration profile loader with curated presets.

Supports dev/staging/production profiles with inheritance,
plus curated presets for common deployment scenarios.

Profile resolution order:
1. Base config from top-level YAML keys
2. Profile-specific overrides (e.g. "production:" section)
3. Preset overrides (e.g. "gpu-single", "multi-node")
4. Environment variables (DISTLLM__*) -- always take final precedence

Usage::

    from distllm.config.profiles import ProfileConfig
    config = ProfileConfig.load("config.yaml", profile="production", preset="gpu-single")
"""

from __future__ import annotations

import os
from typing import Any


# Hardened defaults applied when running in production profile.
_PRODUCTION_SECURE_DEFAULTS: dict[str, Any] = {
    "coordinator": {"host": "0.0.0.0"},
    "tls": {"enabled": True},
}

# Curated presets for common deployment scenarios
_PRESETS: dict[str, dict[str, Any]] = {
    "gpu-single": {
        "description": "Single GPU, single node — simplest setup",
        "model": {"dtype": "float16"},
        "coordinator": {"host": "127.0.0.1", "port": 50050},
        "batching": {"max_batch_size": 8, "max_tokens_per_batch": 4096},
        "prefix_cache": {"enabled": True, "max_entries": 512},
        "chunked_prefill": {"enabled": True, "chunk_size": 256},
    },
    "gpu-multi": {
        "description": "Multiple GPUs on one node",
        "model": {"dtype": "float16"},
        "coordinator": {"host": "0.0.0.0", "port": 50050},
        "tensor_parallel": {"enabled": True},
        "batching": {"max_batch_size": 16, "max_tokens_per_batch": 8192},
        "prefix_cache": {"enabled": True, "max_entries": 1024},
    },
    "multi-node": {
        "description": "Multiple nodes with GPU, pipeline parallelism",
        "model": {"dtype": "float16"},
        "coordinator": {"host": "0.0.0.0", "port": 50050},
        "batching": {"max_batch_size": 32, "max_tokens_per_batch": 16384},
        "prefix_cache": {"enabled": True, "max_entries": 2048},
        "defragmentation": {"enabled": True, "policy": "balanced"},
    },
    "cpu-only": {
        "description": "CPU-only inference (slower, but works everywhere)",
        "model": {"dtype": "float32"},
        "coordinator": {"host": "127.0.0.1", "port": 50050},
        "batching": {"max_batch_size": 4, "max_tokens_per_batch": 2048},
        "chunked_prefill": {"enabled": True, "chunk_size": 128},
    },
    "edge": {
        "description": "Edge device (Raspberry Pi, Jetson, etc.)",
        "model": {"dtype": "float16"},
        "coordinator": {"host": "127.0.0.1", "port": 50050},
        "batching": {"max_batch_size": 2, "max_tokens_per_batch": 1024},
        "prefix_cache": {"enabled": False},
        "defragmentation": {"enabled": False},
    },
    "high-throughput": {
        "description": "Optimized for maximum throughput (sacrifices latency)",
        "model": {"dtype": "float16"},
        "batching": {"max_batch_size": 64, "max_tokens_per_batch": 32768},
        "prefix_cache": {"enabled": True, "max_entries": 4096},
        "defragmentation": {"enabled": True, "policy": "aggressive"},
    },
    "low-latency": {
        "description": "Optimized for minimum latency (sacrifices throughput)",
        "model": {"dtype": "float16"},
        "batching": {"max_batch_size": 4, "max_tokens_per_batch": 2048},
        "chunked_prefill": {"enabled": True, "chunk_size": 128},
        "priority": {"enabled": True, "aging_enabled": True},
    },
}


class ProfileConfig:
    """Load and merge configuration profiles with presets."""

    SUPPORTED_PROFILES = {"dev", "staging", "production"}
    SUPPORTED_PRESETS = set(_PRESETS.keys())

    @staticmethod
    def load(
        config_path: str,
        profile: str | None = None,
        preset: str | None = None,
    ) -> dict[str, Any]:
        """Load config YAML, apply profile and preset overrides.

        Args:
            config_path: Path to the YAML config file.
            profile: Profile name (dev, staging, production). If None, reads
                DISTLLM_PROFILE env var, defaults to "dev".
            preset: Preset name (gpu-single, multi-node, etc.). If None, reads
                DISTLLM_PRESET env var.

        Returns:
            Merged configuration dict.
        """
        import yaml

        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}

        profile_name = profile or os.environ.get("DISTLLM_PROFILE", "dev")
        preset_name = preset or os.environ.get("DISTLLM_PRESET")

        if profile_name not in ProfileConfig.SUPPORTED_PROFILES:
            raise ValueError(
                f"Unknown profile: {profile_name}. Must be one of {ProfileConfig.SUPPORTED_PROFILES}"
            )

        # Separate base config from profile sections
        base = {k: v for k, v in raw.items() if k not in ProfileConfig.SUPPORTED_PROFILES}
        overrides = raw.get(profile_name, {})

        merged = ProfileConfig._deep_merge(base, overrides)

        # Apply preset if specified
        if preset_name:
            if preset_name not in _PRESETS:
                raise ValueError(
                    f"Unknown preset: {preset_name}. Must be one of {list(_PRESETS.keys())}"
                )
            preset_config = {k: v for k, v in _PRESETS[preset_name].items() if k != "description"}
            merged = ProfileConfig._deep_merge(merged, preset_config)

        # Apply secure defaults for production
        if profile_name == "production":
            merged = ProfileConfig._deep_merge(merged, _PRODUCTION_SECURE_DEFAULTS)

        return merged

    @staticmethod
    def list_presets() -> dict[str, str]:
        """Return available presets with descriptions."""
        return {
            name: config.get("description", "")
            for name, config in _PRESETS.items()
        }

    @staticmethod
    def get_preset(name: str) -> dict[str, Any] | None:
        """Get a specific preset configuration."""
        return _PRESETS.get(name)

    @staticmethod
    def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        """Recursively merge override into base."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ProfileConfig._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
