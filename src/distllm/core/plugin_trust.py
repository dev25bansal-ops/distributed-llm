"""Plugin trust policy: install gating and signature verification (SEC-3).

``PluginSystem.install_plugin`` / ``PluginMarketplace.install`` execute
third-party code (``pip install distllm-plugin-<name>`` + entry-point load)
inside the coordinator process. That is RCE-by-design unless the operator
explicitly accepts it, so installation is **fail-closed**:

* ``DISTLLM_PLUGIN_INSTALL_ALLOWED=1`` must be set for any install.
* When ``DISTLLM_PLUGIN_SIGNING_PUBKEY`` points to an Ed25519 public key
  (PEM), a signed manifest must verify before install proceeds. The
  manifest (``<plugin_dir>/distllm-plugin-<name>.manifest.json``) lists the
  package name and version; its Ed25519 signature lives alongside as
  ``.manifest.json.sig``. A missing or invalid ``cryptography`` install
  then fails closed (verification is required but impossible).
* When no key is configured, the env gate alone protects the install and
  the operator inherits the PyPI trust model (logged as such).
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

INSTALL_ALLOWED_ENV = "DISTLLM_PLUGIN_INSTALL_ALLOWED"
SIGNING_PUBKEY_ENV = "DISTLLM_PLUGIN_SIGNING_PUBKEY"


class PluginInstallRefused(PermissionError):
    """Raised when a plugin install violates the trust policy."""


def install_allowed() -> bool:
    """True only when the operator explicitly enabled plugin installation."""
    return os.environ.get(INSTALL_ALLOWED_ENV, "").strip().lower() in ("1", "true", "yes", "on")


def enforce_install_allowed(package_name: str) -> None:
    """Raise :class:`PluginInstallRefused` unless installs are enabled."""
    if install_allowed():
        return
    raise PluginInstallRefused(
        f"Refusing to install '{package_name}': plugin installation executes "
        f"third-party code inside this process. Set {INSTALL_ALLOWED_ENV}=1 to "
        f"accept the trust model, and prefer a signing key "
        f"({SIGNING_PUBKEY_ENV}) for any non-local source."
    )


@dataclass
class ManifestVerification:
    """Outcome of a signed-manifest check."""

    verified: bool
    reason: str


def verify_plugin_manifest(
    plugin_name: str,
    package_name: str,
    plugin_dir: str | Path,
    pubkey_path: str | Path,
) -> ManifestVerification:
    """Verify an Ed25519-signed plugin manifest against *pubkey_path*.

    The manifest is a JSON document with at least ``{"name": ...,
    "package": ...}``; the signature is the base64 Ed25519 signature of the
    exact manifest bytes in ``<manifest>.sig``.
    """
    manifest_path = Path(plugin_dir) / f"{package_name}.manifest.json"
    sig_path = Path(str(manifest_path) + ".sig")

    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import ed25519
    except ImportError:
        return ManifestVerification(
            False,
            f"cryptography unavailable but a signing key is configured "
            f"({SIGNING_PUBKEY_ENV}) — failing closed",
        )

    if not manifest_path.exists() or not sig_path.exists():
        return ManifestVerification(
            False,
            f"missing signed manifest for '{plugin_name}' "
            f"(expected {manifest_path.name} + .sig)",
        )

    try:
        pub_bytes = Path(pubkey_path).read_bytes()
        public_key = serialization.load_pem_public_key(pub_bytes)
        if not isinstance(public_key, ed25519.Ed25519PublicKey):
            return ManifestVerification(False, "configured key is not an Ed25519 public key")

        manifest_bytes = manifest_path.read_bytes()
        signature = base64.b64decode(sig_path.read_bytes().strip())
        public_key.verify(signature, manifest_bytes)
    except (InvalidSignature, ValueError, OSError) as e:
        return ManifestVerification(False, f"signature verification failed: {e}")

    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        return ManifestVerification(False, f"manifest is not valid JSON: {e}")

    if manifest.get("package") != package_name:
        return ManifestVerification(
            False,
            f"manifest names package '{manifest.get('package')}', "
            f"expected '{package_name}'",
        )

    logger.info(f"Plugin manifest verified: {package_name}")
    return ManifestVerification(True, "ok")
