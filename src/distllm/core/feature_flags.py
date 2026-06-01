"""Feature flag system for controlled feature rollouts.

Supports:
- JSON file-based flags (simple, no external dependencies)
- Environment variable overrides
- Percentage-based rollouts
- User/group targeting
- Runtime flag updates without restart

Usage::

    flags = FeatureFlags("feature_flags.json")
    if flags.is_enabled("new_scheduler", user_id="user-123"):
        use_new_scheduler()

    # Or via environment variable override
    # DISTLLM_FLAG_NEW_SCHEDULER=1
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

__all__ = [
    "FeatureFlags",
    "get_feature_flags",
]


@dataclass
class FlagConfig:
    """Configuration for a single feature flag."""
    name: str
    enabled: bool = False
    description: str = ""
    rollout_pct: float = 100.0  # 0-100 percentage rollout
    allowed_users: list[str] = field(default_factory=list)
    allowed_groups: list[str] = field(default_factory=list)
    enabled_from: float = 0.0  # Unix timestamp (0 = always)
    enabled_until: float = 0.0  # Unix timestamp (0 = no expiry)
    metadata: dict[str, Any] = field(default_factory=dict)


class FeatureFlags:
    """Feature flag manager with JSON file backend.

    Loads flags from a JSON file and supports runtime overrides
    via environment variables.
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        auto_reload: bool = False,
        reload_interval_s: float = 30.0,
    ):
        self._path = Path(config_path) if config_path else None
        self._flags: dict[str, FlagConfig] = {}
        self._lock = threading.Lock()
        self._auto_reload = auto_reload
        self._reload_interval = reload_interval_s
        self._last_load = 0.0

        if self._path and self._path.exists():
            self._load()

    def _load(self) -> None:
        """Load flags from JSON file."""
        try:
            data = json.loads(self._path.read_text())
            with self._lock:
                for name, config in data.items():
                    if isinstance(config, dict):
                        self._flags[name] = FlagConfig(name=name, **config)
                    elif isinstance(config, bool):
                        self._flags[name] = FlagConfig(name=name, enabled=config)
            self._last_load = time.time()
            logger.debug(f"Loaded {len(self._flags)} feature flags from {self._path}")
        except Exception as e:
            logger.warning(f"Failed to load feature flags: {e}")

    def _check_env_override(self, flag_name: str) -> bool | None:
        """Check for environment variable override.

        Env var format: DISTLLM_FLAG_<NAME>=1/0/true/false
        """
        env_key = f"DISTLLM_FLAG_{flag_name.upper()}"
        value = os.environ.get(env_key)
        if value is not None:
            return value.lower() in ("1", "true", "yes")
        return None

    def _check_rollout(self, flag: FlagConfig, user_id: str | None = None) -> bool:
        """Check if user is in the rollout percentage."""
        if flag.rollout_pct >= 100.0:
            return True
        if flag.rollout_pct <= 0.0:
            return False

        # Deterministic hash-based rollout
        seed = f"{flag.name}:{user_id or 'anonymous'}"
        hash_val = int(hashlib.md5(seed.encode()).hexdigest()[:8], 16)
        return (hash_val % 100) < flag.rollout_pct

    def _check_time_window(self, flag: FlagConfig) -> bool:
        """Check if current time is within the flag's time window."""
        now = time.time()
        if flag.enabled_from > 0 and now < flag.enabled_from:
            return False
        if flag.enabled_until > 0 and now > flag.enabled_until:
            return False
        return True

    def is_enabled(
        self,
        flag_name: str,
        user_id: str | None = None,
        group: str | None = None,
        default: bool = False,
    ) -> bool:
        """Check if a feature flag is enabled.

        Resolution order:
        1. Environment variable override (DISTLLM_FLAG_<NAME>)
        2. User/group targeting
        3. Rollout percentage
        4. Time window
        5. Default enabled state
        6. Fallback default

        Args:
            flag_name: Name of the feature flag.
            user_id: Optional user ID for targeting/rollout.
            group: Optional group name for targeting.
            default: Default value if flag not found.

        Returns:
            True if the flag is enabled for this context.
        """
        # Auto-reload if needed
        if self._auto_reload and self._path:
            if time.time() - self._last_load > self._reload_interval:
                self._load()

        # Check env override first
        env_override = self._check_env_override(flag_name)
        if env_override is not None:
            return env_override

        with self._lock:
            flag = self._flags.get(flag_name)
            if flag is None:
                return default

            # Base enabled check
            if not flag.enabled:
                return False

            # User targeting
            if flag.allowed_users and user_id:
                if user_id in flag.allowed_users:
                    return True

            # Group targeting
            if flag.allowed_groups and group:
                if group in flag.allowed_groups:
                    return True

            # If targeting lists exist but user not in them, deny
            if flag.allowed_users and user_id and user_id not in flag.allowed_users:
                if not flag.allowed_groups:
                    return False

            # Time window
            if not self._check_time_window(flag):
                return False

            # Rollout percentage
            return self._check_rollout(flag, user_id)

    def get_all_flags(self) -> dict[str, dict]:
        """Return all flags and their current state."""
        with self._lock:
            return {
                name: {
                    "enabled": flag.enabled,
                    "description": flag.description,
                    "rollout_pct": flag.rollout_pct,
                    "allowed_users": len(flag.allowed_users),
                    "allowed_groups": len(flag.allowed_groups),
                }
                for name, flag in self._flags.items()
            }

    def set_flag(self, flag_name: str, enabled: bool) -> None:
        """Dynamically enable/disable a flag at runtime."""
        with self._lock:
            if flag_name in self._flags:
                self._flags[flag_name].enabled = enabled
            else:
                self._flags[flag_name] = FlagConfig(name=flag_name, enabled=enabled)
        logger.info(f"Feature flag '{flag_name}' set to {enabled}")

    def save(self) -> None:
        """Save current flags to the JSON file."""
        if not self._path:
            return
        with self._lock:
            data = {
                name: {
                    "enabled": flag.enabled,
                    "description": flag.description,
                    "rollout_pct": flag.rollout_pct,
                    "allowed_users": flag.allowed_users,
                    "allowed_groups": flag.allowed_groups,
                }
                for name, flag in self._flags.items()
            }
        self._path.write_text(json.dumps(data, indent=2))
        logger.info(f"Saved {len(data)} feature flags to {self._path}")


# Global singleton
_flags: FeatureFlags | None = None
_flags_lock = threading.Lock()


def get_feature_flags(config_path: str | None = None) -> FeatureFlags:
    """Get or create the global feature flags instance."""
    global _flags
    if _flags is None:
        with _flags_lock:
            if _flags is None:
                path = config_path or os.environ.get("DISTLLM_FEATURE_FLAGS")
                _flags = FeatureFlags(config_path=path, auto_reload=True)
    return _flags
