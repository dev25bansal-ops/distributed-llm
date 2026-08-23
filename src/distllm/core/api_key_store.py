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


def hash_api_key(key: str, salt: str | None = None) -> str:
    """Return a salted, non-reversible PBKDF2-HMAC-SHA256 hash of *key*.

    The hash is NOT the plaintext key and includes a fresh per-call salt, so
    two hashes of the same key differ (expensive to brute-force; no rainbow
    tables).  This is the module-level counterpart of ``ApiKeyStore._hash_key``
    used for forged/standalone verifier checks.
    """
    salt = salt or secrets.token_hex(16)
    return salt + ":" + hashlib.pbkdf2_hmac(
        "sha256", key.encode("utf-8"), salt.encode("utf-8"), 100000
    ).hex()


def verify_api_key(key: str, stored: str) -> bool:
    """Verify *key* against a ``hash_api_key``-produced *stored* value.

    Safe against garbage: returns False (never raises) for malformed hashes.
    """
    if not isinstance(stored, str) or ":" not in stored:
        return False
    salt, _, digest = stored.partition(":")
    try:
        candidate = hash_api_key(key, salt)
    except Exception:
        return False
    return _constant_time_eq(candidate, stored)


def _constant_time_eq(a: str, b: str) -> bool:
    if len(a) != len(b):
        return False
    import hmac as _hmac
    return _hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


# ── Key entry ───────────────────────────────────────────────────────────────


@dataclass
class StoredKey:
    """An API key stored in memory (salted hash)."""
    key: str        # hex(PBKDF2-HMAC-SHA256(key, salt, 100k iterations))
    role: str
    label: str
    key_id: str
    salt: str       # hex-encoded 16-byte random salt
    created_at: float
    # Rotation grace deadline. When set, the key stops authenticating once
    # ``time.time()`` passes this instant. ``None`` = never expires.
    expires_at: float | None = None


# ── Key store ───────────────────────────────────────────────────────────────


class ApiKeyStore:
    """In-memory store of API keys with role assignments.

    Thread-safe for reads (writes only happen during init).
    """

    def __init__(self) -> None:
        self._keys: list[StoredKey] = []
        self._lock = threading.Lock()  # guards dynamic add/remove during rotation
        self._load()

    # ── Public API ──────────────────────────────────────────────────────────

    def _hash_key(self, key_str: str, salt: str) -> str:
        """PBKDF2-HMAC-SHA256 with 100 000 iterations, returning hex digest.

        Uses a constant number of iterations per key so the cost is
        proportional for both legitimate clients and brute-force attackers.
        """
        return hashlib.pbkdf2_hmac(
            "sha256",
            key_str.encode("utf-8"),
            salt.encode("utf-8"),
            100000,
        ).hex()

    def _generate_salt(self) -> str:
        """Return a hex-encoded 16-byte random salt."""
        return secrets.token_hex(16)

    def authenticate(self, token: str) -> tuple[str, str] | None:
        """Validate a bearer token.

        Returns ``(key_id, role)`` on success, ``None`` on failure.
        Uses constant-time comparison to prevent timing attacks.
        Hashes with per-key salt so a leaked DB does not enable
        rainbow-table attacks.
        """
        for k in self._keys:
            # A retired (rotated) key stops authenticating once its grace
            # deadline passes — this is the authoritative retirement boundary.
            if k.expires_at is not None and time.time() > k.expires_at:
                continue
            token_hash = self._hash_key(token, k.salt)
            if compare_digest(token_hash, k.key):
                return (k.key_id, k.role)
        return None

    def add_key(
        self,
        key_str: str,
        role: str = "admin",
        label: str = "",
        key_id: str | None = None,
    ) -> str:
        """Register a new API key and return its ``key_id``.

        Used by key rotation (and dynamic provisioning).  A new key added with
        an existing ``key_id`` coexists with the old entry so both authenticate
        during a rotation grace period; the old entry can be retired with
        :meth:`retire_key` (or removed with :meth:`remove_key_hash`).
        """
        if not key_str:
            raise ValueError("key cannot be empty")
        if role not in VALID_ROLES:
            raise ValueError(
                f"Invalid API key role '{role}'. Must be one of {VALID_ROLES}"
            )
        key_id = key_id or f"key-{secrets.token_hex(8)}"
        salt = self._generate_salt()
        with self._lock:
            self._keys.append(StoredKey(
                key=self._hash_key(key_str, salt),
                role=role,
                label=label or key_id,
                key_id=key_id,
                salt=salt,
                created_at=time.time(),
            ))
        return key_id

    def get_key_hash(self, key_id: str) -> str | None:
        """Return the stored hash of the first key with *key_id*, or None."""
        for k in self._keys:
            if k.key_id == key_id:
                return k.key
        return None

    def get_latest_key_hash(self, key_id: str) -> str | None:
        """Return the stored hash of the most recently added key with *key_id*.

        During rotation the current ACTIVE key is the last one added for that
        id (an earlier entry may already be retired from a prior rotation).
        """
        found: str | None = None
        for k in self._keys:
            if k.key_id == key_id:
                found = k.key
        return found

    def remove_key_hash(self, key_hash: str) -> bool:
        """Remove the key entry whose stored hash equals *key_hash*."""
        with self._lock:
            for i, k in enumerate(self._keys):
                if k.key == key_hash:
                    del self._keys[i]
                    return True
        return False

    def retire_key_hash(self, key_hash: str, expires_at: float) -> bool:
        """Mark the entry with *key_hash* as expiring at *expires_at*.

        Precise retirement for rotation: only the specific old key is retired,
        never the replacement (which has a different hash).  Returns True if a
        matching entry was marked.
        """
        with self._lock:
            for k in self._keys:
                if k.key == key_hash:
                    k.expires_at = expires_at
                    return True
        return False

    def retire_key(self, key_id: str, expires_at: float) -> int:
        """Mark every key entry with *key_id* as expiring at *expires_at*.

        Used during rotation so the retired key stops authenticating once its
        grace period passes, while the replacement (added separately) stays
        valid.  Returns the number of entries marked.
        """
        with self._lock:
            marked = 0
            for k in self._keys:
                if k.key_id == key_id:
                    k.expires_at = expires_at
                    marked += 1
            return marked

    def remove_expired(self, now: float | None = None) -> int:
        """Remove entries whose rotation grace deadline has passed.

        Returns the number of entries removed.
        """
        now = time.time() if now is None else now
        with self._lock:
            before = len(self._keys)
            self._keys = [
                k for k in self._keys
                if not (k.expires_at is not None and now > k.expires_at)
            ]
            return before - len(self._keys)

    def remove_key(self, key_id: str) -> bool:
        """Remove every key entry with *key_id*."""
        with self._lock:
            before = len(self._keys)
            self._keys = [k for k in self._keys if k.key_id != key_id]
            return len(self._keys) < before

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
            # MD-09: Enforce size limit to prevent memory exhaustion on startup
            if len(raw) > 100_000:
                logger.error(
                    f"API_KEYS env var is {len(raw)} bytes, "
                    f"exceeds 100 KB limit — refusing to parse"
                )
                raise ValueError(
                    "API_KEYS env var exceeds 100 KB size limit"
                )
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
            salt = self._generate_salt()
            self._keys.append(StoredKey(
                key=self._hash_key(legacy_key, salt),
                role="admin",
                label="legacy-api-key",
                key_id="legacy",
                salt=salt,
                created_at=time.time(),
            ))
            logger.info("Loaded single API_KEY (admin role, backward compatible)")
            return

        # 4. Auto-generate an admin key (legacy behavior)
        generated = secrets.token_urlsafe(48)
        salt = self._generate_salt()
        self._keys.append(StoredKey(
            key=self._hash_key(generated, salt),
            role="admin",
            label="auto-generated",
            key_id="auto",
            salt=salt,
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

            salt = self._generate_salt()
            self._keys.append(StoredKey(
                key=self._hash_key(key_str, salt),
                role=role,
                label=label or key_id,
                key_id=key_id,
                salt=salt,
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
