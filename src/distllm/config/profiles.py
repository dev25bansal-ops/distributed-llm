"""Configuration profile loader.

Supports dev/staging/production profiles with inheritance.
Profile resolution order:
1. Base config from top-level YAML keys
2. Profile-specific overrides (e.g. "production:" section)
3. Environment variables (DISTLLM__*) — always take final precedence
"""

from __future__ import annotations

import os
from typing import Any


class ProfileConfig:
    """Load and merge configuration profiles."""

    SUPPORTED_PROFILES = {"dev", "staging", "production"}

    @staticmethod
    def load(config_path: str, profile: str | None = None) -> dict[str, Any]:
        """Load config YAML, apply profile overrides.

        Args:
            config_path: Path to the YAML config file.
            profile: Profile name (dev, staging, production). If None, reads
                DISTLLM_PROFILE env var, defaults to "dev".

        Returns:
            Merged configuration dict.
        """
        import yaml

        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}

        profile_name = profile or os.environ.get("DISTLLM_PROFILE", "dev")
        if profile_name not in ProfileConfig.SUPPORTED_PROFILES:
            raise ValueError(
                f"Unknown profile: {profile_name}. Must be one of {ProfileConfig.SUPPORTED_PROFILES}"
            )

        # Separate base config from profile sections
        base = {k: v for k, v in raw.items() if k not in ProfileConfig.SUPPORTED_PROFILES}
        overrides = raw.get(profile_name, {})

        return ProfileConfig._deep_merge(base, overrides)

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
