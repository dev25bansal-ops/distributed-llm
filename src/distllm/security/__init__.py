"""Security utilities and E2E encryption for tensor transport.

Provides:
  - ``safe_urlopen``, ``validate_http_url``, ``hf_revision`` — SSRF/DNS
    rebinding protection and model revision pinning
  - ``E2EEncryption`` — NaCl/libsodium-based end-to-end encryption for
    tensor bytes with per-session X25519 key exchange
  - ``SessionManager`` — manages per-node-pair encryption sessions
"""

from .utils import hf_revision, safe_urlopen, validate_http_url
from .e2e import (
    E2EEncryption,
    SessionKeys,
    encrypt_tensor_payload,
    decrypt_tensor_payload,
    HAS_NACL,
)

__all__ = [
    "hf_revision",
    "safe_urlopen",
    "validate_http_url",
    "E2EEncryption",
    "SessionKeys",
    "encrypt_tensor_payload",
    "decrypt_tensor_payload",
    "HAS_NACL",
]
