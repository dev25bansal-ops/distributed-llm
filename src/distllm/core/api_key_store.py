"""Multi-role API key authentication store.

Supports loading API keys from:
1. ``API_KEYS`` env var — inline JSON string
2. ``API_KEYS_FILE`` env var — path to JSON or YAML file
3. ``API_KEY`` env var — single key (backward compatible, admin role)
4. Auto-generated fallback — single admin key (legacy behavior)

Each key has a ``role``: ``admin``, ``read-only``, or ``inference-only``.

Usage::

    store = ApiKeyStore()
    result = store.authenticate("sk-admin-abc123")
    if result:
        key_id, role = result  # ("my-key", "admin")
"""

from __future__ import annotations

import json
import os
import secrets
import time
from dataclasses import dataclass
from typing import Any
from hmac import compare_digest

from loguru import logger


# ── Roles ───────────────────────────────────────────────────────────────────

VALID_ROLES = frozenset({"admin", "read-only", "inference-only"})

ROLE_HIERARCHY: dict[str, int] = {
    "admin": 3,
    "inference-only": 2,
    "read-only": 1,
}


def role_satisfies(actual: str, required: str) -> bool:
    """Check if *actual* role satisfies *required* role.

    Admin satisfies everything. Read-only only satisfies read-only.
    Inference-only satisfies inference-only and read-only.
    """
    if actual == required:
        return True
    if actual == "admin":
        return True
    if actual == "inference-only" and required == "read-only":
        return True
    return False


# ── Key entry ───────────────────────────────────────────────────────────────


@dataclass
class StoredKey:
    """An API key stored in memory."""
    key: str
    role: str
    label: str
    key_id: str
    created_at: float


# ── Key store ───────────────────────────────────────────────────────────────


class ApiKeyStore:
    """In-memory store of API keys with role assignments.

    Thread-safe for reads (writes only happen during init).
    """

    def __init__(self) -> None:
        self._keys: list[StoredKey] = []
        self._load()

    # ── Public API ──────────────────────────────────────────────────────────

    def authenticate(self, token: str) -> tuple[str, str] | None:
        """Validate a bearer token.

        Returns ``(key_id, role)`` on success, ``None`` on failure.
        Uses constant-time comparison to prevent timing attacks.
        """
        for k in self._keys:
            if compare_digest(token, k.key):
                return (k.key_id, k.role)
        return None

    def get_key_count(self) -> int:
        return len(self._keys)

    def list_keys(self) -> list[dict[str, Any]]:
        """Return metadata for all stored keys (excluding the key value itself)."""
        return [
            {"key_id": k.key_id, "role": k.role, "label": k.label}
            for k in self._keys
        ]

    def has_role(self, role: str) -> bool:
        """Return True if at least one key has the given role."""
        return any(k.role == role for k in self._keys)

    # ── Loading ─────────────────────────────────────────────────────────────

    def _load(self) -> None:
        """Load keys from environment or file."""
        # 1. Try API_KEYS env var (inline JSON)
        raw = os.environ.get("API_KEYS")
        if raw:
            try:
                parsed = json.loads(raw)
                self._load_from_dict(parsed)
                if self._keys:
                    logger.info(
                        f"Loaded {len(self._keys)} API key(s) from API_KEYS env var "
                        f"(roles: {sorted(set(k.role for k in self._keys))})"
                    )
                    return
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Failed to parse API_KEYS env var: {e}")

        # 2. Try API_KEYS_FILE env var
        path = os.environ.get("API_KEYS_FILE")
        if path:
            try:
                with open(path) as f:
                    if path.endswith((".yaml", ".yml")):
                        try:
                            import yaml
                            parsed = yaml.safe_load(f)
                        except ImportError:
                            logger.warning("PyYAML not installed, falling back to JSON parsing")
                            parsed = json.load(f)
                    else:
                        parsed = json.load(f)
                self._load_from_dict(parsed)
                if self._keys:
                    logger.info(
                        f"Loaded {len(self._keys)} API key(s) from {path} "
                        f"(roles: {sorted(set(k.role for k in self._keys))})"
                    )
                    return
            except (FileNotFoundError, json.JSONDecodeError, Exception) as e:
                logger.warning(f"Failed to load API_KEYS_FILE {path}: {e}")

        # 3. Fall back to single API_KEY env var (backward compatible)
        legacy_key = os.environ.get("API_KEY")
        if legacy_key:
            self._keys.append(StoredKey(
                key=legacy_key,
                role="admin",
                label="legacy-api-key",
                key_id="legacy",
                created_at=time.time(),
            ))
            logger.info("Loaded single API_KEY (admin role, backward compatible)")
            return

        # 4. Auto-generate an admin key (legacy behavior)
        generated = secrets.token_urlsafe(48)
        self._keys.append(StoredKey(
            key=generated,
            role="admin",
            label="auto-generated",
            key_id="auto",
            created_at=time.time(),
        ))
        logger.warning(
            "No API keys configured. Generated an in-memory admin API key. "
            "Set API_KEYS, API_KEYS_FILE, or API_KEY in the environment."
        )

    def _load_from_dict(self, data: dict[str, Any]) -> None:
        """Parse a dict like ``{"keys": [{"key": "...", "role": "...", ...}]}``."""
        raw_keys = data.get("keys", data if isinstance(data, list) else [])
        if isinstance(raw_keys, dict):
            raw_keys = [raw_keys]
        for entry in raw_keys:
            key_str = entry.get("key", "")
            role = entry.get("role", "admin")
            label = entry.get("label", "")
            key_id = entry.get("key_id", entry.get("id", label or key_str[:8]))

            if not key_str:
                continue
            if role not in VALID_ROLES:
                logger.warning(f"Invalid role '{role}' for key '{key_id}', defaulting to 'admin'")
                role = "admin"

            self._keys.append(StoredKey(
                key=key_str,
                role=role,
                label=label or key_id,
                key_id=key_id,
                created_at=time.time(),
            ))


# ── Module-level singleton ──────────────────────────────────────────────────

_store: ApiKeyStore | None = None


def get_api_key_store() -> ApiKeyStore:
    """Get or create the singleton API key store."""
    global _store
    if _store is None:
        _store = ApiKeyStore()
    return _store


def reset_api_key_store() -> None:
    """Reset the singleton store (for testing)."""
    global _store
    _store = None
