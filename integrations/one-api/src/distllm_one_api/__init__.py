"""DistLLM one-api provider — register DistLLM as a provider in one-api.

Usage::

    from distllm_one_api import DistLLMProviderConfig

    config = DistLLMProviderConfig(
        base_url="http://localhost:8000/v1",
        api_key="sk-...",
    )
    config.apply()
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

__version__ = "0.1.0"

__all__ = ["__version__", "DistLLMProviderConfig", "apply_config"]


@dataclass
class DistLLMProviderConfig:
    """Configuration for the DistLLM one-api provider.

    Args:
        name: Provider name as registered in one-api.
        base_url: DistLLM API base URL.
        api_key: API key for authentication.
        models: List of model names to advertise. Auto-detects if empty.
        priority: Provider priority (lower = preferred).
        retry_count: Number of retries on failure.
        cooldown_seconds: Cooldown after failure before retrying.
    """

    name: str = "distllm"
    base_url: str = ""
    api_key: str = ""
    models: list[str] = field(default_factory=lambda: ["distributed-llm"])
    priority: int = 10
    retry_count: int = 2
    cooldown_seconds: int = 30

    def __post_init__(self) -> None:
        self.base_url = self.base_url or os.environ.get(
            "DISTLLM_ONEAPI_BASE_URL", "http://localhost:8000/v1"
        )
        self.api_key = self.api_key or os.environ.get("DISTLLM_ONEAPI_API_KEY", "")

        models_env = os.environ.get("DISTLLM_ONEAPI_MODELS", "")
        if models_env and not self.models:
            self.models = [m.strip() for m in models_env.split(",") if m.strip()]

    def to_oneapi_config(self) -> dict[str, Any]:
        """Convert to a one-api channel configuration dict.

        Returns a dict that can be applied to one-api's provider/channel config.
        """
        return {
            "name": self.name,
            "base_url": self.base_url,
            "api_key": self.api_key,
            "models": self.models,
            "priority": self.priority,
            "retry_count": self.retry_count,
            "cooldown_seconds": self.cooldown_seconds,
        }

    def apply(self) -> None:
        """Apply this config to one-api at runtime.

        Attempts to register the provider via one-api's admin API.
        Falls back to printing the config JSON for manual application.
        """
        import json

        config = self.to_oneapi_config()
        admin_url = os.environ.get("DISTLLM_ONEAPI_ADMIN_URL", "")

        if admin_url:
            try:
                import httpx

                resp = httpx.post(
                    f"{admin_url}/api/channel",
                    json=config,
                    timeout=10.0,
                )
                if resp.status_code in (200, 201):
                    return
                else:
                    print(f"one-api registration returned {resp.status_code}: {resp.text}")
                    return
            except Exception as e:
                print(f"one-api registration failed: {e}")

        # Fallback: print config for manual setup
        print("Apply this configuration to one-api manually:")
        print(json.dumps(config, indent=2))
        print(f"\nOr set DISTLLM_ONEAPI_ADMIN_URL to auto-register.")


def apply_config(**kwargs: Any) -> None:
    """Convenience: create and apply a provider config in one call.

    Usage::

        from distllm_one_api import apply_config

        apply_config(base_url="http://localhost:8000/v1", api_key="sk-...")
    """
    config = DistLLMProviderConfig(**kwargs)
    config.apply()
