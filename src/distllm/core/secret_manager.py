"""External secret manager integration for API keys and credentials.

Supports multiple backends:
- Environment variables (default)
- HashiCorp Vault
- AWS Secrets Manager
- File-based secrets

Usage::

    manager = SecretManager(backend="vault", url="https://vault:8200")
    api_key = manager.get_secret("distllm/api-key")
    manager.rotate_secret("distllm/api-key", new_value="new-key")
"""

from __future__ import annotations

import json
import os
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger


class SecretBackend(ABC):
    """Abstract base for secret storage backends."""

    @abstractmethod
    def get(self, key: str) -> str | None:
        """Retrieve a secret by key."""

    @abstractmethod
    def put(self, key: str, value: str) -> bool:
        """Store a secret."""

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete a secret."""


class EnvSecretBackend(SecretBackend):
    """Secrets stored in environment variables."""

    def get(self, key: str) -> str | None:
        return os.environ.get(key)

    def put(self, key: str, value: str) -> bool:
        os.environ[key] = value
        return True

    def delete(self, key: str) -> bool:
        return os.environ.pop(key, None) is not None


class FileSecretBackend(SecretBackend):
    """Secrets stored in a local file (JSON)."""

    def __init__(self, path: str = ".distllm_secrets.json"):
        self._path = Path(path)
        self._lock = threading.Lock()
        self._ensure_file()

    def _ensure_file(self):
        if not self._path.exists():
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "w") as f:
                json.dump({}, f)
            # Restrict permissions
            try:
                os.chmod(self._path, 0o600)
            except OSError:
                pass

    def _read(self) -> dict:
        try:
            with open(self._path) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _write(self, data: dict):
        with open(self._path, "w") as f:
            json.dump(data, f, indent=2)

    def get(self, key: str) -> str | None:
        with self._lock:
            return self._read().get(key)

    def put(self, key: str, value: str) -> bool:
        with self._lock:
            data = self._read()
            data[key] = value
            self._write(data)
            return True

    def delete(self, key: str) -> bool:
        with self._lock:
            data = self._read()
            if key in data:
                del data[key]
                self._write(data)
                return True
            return False


class VaultSecretBackend(SecretBackend):
    """Secrets stored in HashiCorp Vault."""

    def __init__(self, url: str = "https://localhost:8200", token: str = "", mount: str = "secret"):
        self._url = url.rstrip("/")
        self._token = token
        self._mount = mount

    def _headers(self) -> dict:
        return {"X-Vault-Token": self._token, "Content-Type": "application/json"}

    def get(self, key: str) -> str | None:
        try:
            import httpx
            resp = httpx.get(
                f"{self._url}/v1/{self._mount}/data/{key}",
                headers=self._headers(),
                timeout=5.0,
            )
            if resp.status_code == 200:
                data = resp.json()
                return data.get("data", {}).get("data", {}).get("value")
            return None
        except Exception as e:
            logger.warning(f"Vault get failed for '{key}': {e}")
            return None

    def put(self, key: str, value: str) -> bool:
        try:
            import httpx
            resp = httpx.post(
                f"{self._url}/v1/{self._mount}/data/{key}",
                headers=self._headers(),
                json={"data": {"value": value}},
                timeout=5.0,
            )
            return resp.status_code in (200, 204)
        except Exception as e:
            logger.warning(f"Vault put failed for '{key}': {e}")
            return False

    def delete(self, key: str) -> bool:
        try:
            import httpx
            resp = httpx.delete(
                f"{self._url}/v1/{self._mount}/data/{key}",
                headers=self._headers(),
                timeout=5.0,
            )
            return resp.status_code in (200, 204)
        except Exception as e:
            logger.warning(f"Vault delete failed for '{key}': {e}")
            return False


class AWSSecretsBackend(SecretBackend):
    """Secrets stored in AWS Secrets Manager.

    Requires ``boto3`` to be installed. Uses the default AWS credential
    chain (env vars, ~/.aws/credentials, IAM role).

    Usage::

        backend = AWSSecretsBackend(region="us-east-1")
        backend.get("distllm/api-key")
    """

    def __init__(self, region: str = "us-east-1", prefix: str = "distllm/"):
        self._region = region
        self._prefix = prefix
        self._client = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import boto3
                self._client = boto3.client("secretsmanager", region_name=self._region)
            except ImportError:
                raise RuntimeError("boto3 required for AWS Secrets Manager: pip install boto3")
        return self._client

    def _full_key(self, key: str) -> str:
        return f"{self._prefix}{key}" if not key.startswith(self._prefix) else key

    def get(self, key: str) -> str | None:
        try:
            client = self._get_client()
            resp = client.get_secret_value(SecretId=self._full_key(key))
            return resp.get("SecretString")
        except Exception as e:
            logger.warning(f"AWS Secrets Manager get failed for '{key}': {e}")
            return None

    def put(self, key: str, value: str) -> bool:
        try:
            client = self._get_client()
            full_key = self._full_key(key)
            try:
                client.put_secret_value(SecretId=full_key, SecretString=value)
            except client.exceptions.ResourceNotFoundException:
                client.create_secret(Name=full_key, SecretString=value)
            return True
        except Exception as e:
            logger.warning(f"AWS Secrets Manager put failed for '{key}': {e}")
            return False

    def delete(self, key: str) -> bool:
        try:
            client = self._get_client()
            client.delete_secret(SecretId=self._full_key(key), ForceDeleteWithoutRecovery=True)
            return True
        except Exception as e:
            logger.warning(f"AWS Secrets Manager delete failed for '{key}': {e}")
            return False


class SecretManager:
    """Unified secret management across multiple backends.

    Args:
        backend: Backend type ("env", "file", "vault").
        **kwargs: Backend-specific configuration.
    """

    def __init__(self, backend: str = "env", **kwargs: Any):
        if backend == "env":
            self._backend = EnvSecretBackend()
        elif backend == "file":
            self._backend = FileSecretBackend(**kwargs)
        elif backend == "vault":
            self._backend = VaultSecretBackend(**kwargs)
        elif backend == "aws":
            self._backend = AWSSecretsBackend(**kwargs)
        else:
            raise ValueError(f"Unknown backend: {backend}. Supported: env, file, vault, aws")

        self._cache: dict[str, tuple[str, float]] = {}
        self._cache_ttl = kwargs.get("cache_ttl", 300)
        self._lock = threading.Lock()

    def get_secret(self, key: str, use_cache: bool = True) -> str | None:
        """Get a secret, optionally from cache.

        Args:
            key: Secret key.
            use_cache: Whether to use cached value.

        Returns:
            Secret value, or None if not found.
        """
        if use_cache:
            with self._lock:
                if key in self._cache:
                    value, cached_at = self._cache[key]
                    if time.time() - cached_at < self._cache_ttl:
                        return value

        value = self._backend.get(key)

        if value is not None:
            with self._lock:
                self._cache[key] = (value, time.time())

        return value

    def set_secret(self, key: str, value: str) -> bool:
        """Store a secret."""
        result = self._backend.put(key, value)
        if result:
            with self._lock:
                self._cache[key] = (value, time.time())
        return result

    def rotate_secret(self, key: str, new_value: str) -> bool:
        """Rotate a secret to a new value."""
        old = self.get_secret(key, use_cache=False)
        if old == new_value:
            return False
        result = self.set_secret(key, new_value)
        if result:
            logger.info(f"Secret '{key}' rotated successfully")
        return result

    def delete_secret(self, key: str) -> bool:
        """Delete a secret."""
        with self._lock:
            self._cache.pop(key, None)
        return self._backend.delete(key)

    def clear_cache(self) -> None:
        """Clear the secret cache."""
        with self._lock:
            self._cache.clear()
