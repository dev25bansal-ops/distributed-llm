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

import hashlib
import json
import os
import secrets
import threading
import time
from dataclasses import dataclass
from typing import Any
from hmac import compare_digest

from loguru import logger


# ── Roles ───────────────────────────────────────────────────────────────────

VALID_ROLES = frozenset({
    "admin",           # Full access to all operations
    "model-admin",     # Manage models (load, unload, configure) but not cluster
    "user-admin",      # Manage users/API keys but not models or cluster
    "inference-only",  # Run inference (chat, completions, embeddings) only
    "read-only",       # Read-only access (health, metrics, list models)
    "auditor",         # Read-only + access to audit logs and security events
})

ROLE_HIERARCHY: dict[str, int] = {
    "admin": 6,
    "user-admin": 5,
    "model-admin": 4,
    "auditor": 3,
    "inference-only": 2,
    "read-only": 1,
}


def role_satisfies(actual: str, required: str) -> bool:
    """Check if *actual* role satisfies *required* role.

    Role hierarchy (highest to lowest):
    - admin: satisfies everything
    - user-admin: satisfies user management + read-only
    - model-admin: satisfies model management + inference + read-only
    - auditor: satisfies read-only + audit access
    - inference-only: satisfies inference + read-only
    - read-only: satisfies read-only only
    """
    if actual == required:
        return True
    if actual == "admin":
        return True

    # user-admin can manage users and read
    if actual == "user-admin" and required in ("read-only",):
        return True

    # model-admin can do inference and read
    if actual == "model-admin" and required in ("inference-only", "read-only"):
        return True

    # auditor can read
    if actual == "auditor" and required in ("read-only",):
        return True

    # inference-only can read
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
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        for k in self._keys:
            if compare_digest(token_hash, k.key):
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
                key=hashlib.sha256(legacy_key.encode()).hexdigest(),
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
            key=hashlib.sha256(generated.encode()).hexdigest(),
            role="admin",
            label="auto-generated",
            key_id="auto",
            created_at=time.time(),
        ))
        # Store the raw key so it can be displayed to the user
        self._auto_generated_key = generated
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
                logger.error(f"Invalid role '{role}' for key '{key_id}' — valid roles: {VALID_ROLES}")
                raise ValueError(f"Invalid API key role '{role}'. Must be one of {VALID_ROLES}")

            self._keys.append(StoredKey(
                key=hashlib.sha256(key_str.encode()).hexdigest(),
                role=role,
                label=label or key_id,
                key_id=key_id,
                created_at=time.time(),
            ))


    def get_display_key(self) -> str | None:
        """Return the API key for display to the user.

        Returns the raw key if it was auto-generated, or the API_KEY
        env var value if set. Returns None if keys were loaded from
        a file or JSON config.

        .. warning::

            This method exposes the auto-generated admin key in plaintext.
            Callers must not log, persist, or transmit the returned value.
            The auto-generated key persists in process memory for the
            lifetime of the ApiKeyStore object.
        """
        # Auto-generated key — stored in plaintext in process memory.
        # For production, set API_KEY explicitly via env var or config file.
        if hasattr(self, '_auto_generated_key'):
            return self._auto_generated_key

        # Single API_KEY env var
        env_key = os.environ.get("API_KEY")
        if env_key:
            return env_key

        return None


# ── Module-level singleton ──────────────────────────────────────────────────

_store: ApiKeyStore | None = None
_store_lock = threading.Lock()


def get_api_key_store() -> ApiKeyStore:
    """Get or create the singleton API key store."""
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = ApiKeyStore()
    return _store


def reset_api_key_store() -> None:
    """Reset the singleton store (for testing)."""
    global _store
    _store = None
