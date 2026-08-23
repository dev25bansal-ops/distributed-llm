"""Signed Correctness Certificate writer/verifier (HMAC-SHA256).

A Correctness Certificate is a JSON document::

    {
      "generated_at": "...",
      "logit_cosine_sim": 0.9995,
      "token_exact_match": 1.0,
      "determinism_hash": "sha256...",
      "pairs": [ {PairResult...}, ... ],
      "signer": "distllm-ci",       # or "distllm-dev (UNSIGNED DEV KEY)"
      "signature": "hex HMAC-SHA256"
    }

The ``signature`` is HMAC-SHA256 over the **canonical** JSON of the payload
(i.e. every key except ``signature`` and ``signer``), computed with a key
selected in priority order:

1. ``DISTLLM_CERT_KEY`` environment variable, or
2. a repo dev key file (``scripts/ci/.distllm_dev_cert.key``) if present, or
3. a built-in, clearly-marked DEV key (so the harness is runnable out of the
   box, but the cert is stamped ``UNSIGNED DEV KEY`` to warn reviewers).

``Signed`` means verifiable: :func:`verify_certificate` recomputes the HMAC and
compares in constant time, and also asserts the certificate clears the
certification thresholds (cosine >= 0.999 and token_exact_match == 1.0).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path
from typing import Any

# Built-in DEV key.  Intentionally public-looking and stamped UNSIGNED so a
# reviewer knows this is NOT a production signing key.  Real releases must set
# DISTLLM_CERT_KEY (or ship the per-repo .key file).
_DEV_CERT_KEY = b"distllm-dev-cert-key-DO-NOT-SHIP-IN-PROD"
_DEV_SIGNER_LABEL = "distllm-dev (UNSIGNED DEV KEY)"

DEFAULT_DEV_KEY_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "ci" / ".distllm_dev_cert.key"
)


def _canonical_payload(payload: dict[str, Any]) -> str:
    """Serialize the cert payload (minus signature/signer) deterministically."""
    clean = {k: v for k, v in payload.items() if k not in ("signature", "signer")}
    return json.dumps(clean, sort_keys=True, separators=(",", ":"))


def _resolve_key() -> tuple[bytes, str]:
    """Resolve the signing key + signer label in priority order.

    Returns:
        ``(key_bytes, signer_label)``.
    """
    env_key = os.environ.get("DISTLLM_CERT_KEY")
    if env_key:
        return env_key.encode("utf-8"), "distllm-ci"

    key_path = DEFAULT_DEV_KEY_PATH
    if key_path.exists():
        raw = key_path.read_bytes().strip()
        if raw:
            return raw, "distllm-dev (repo key file)"

    return _DEV_CERT_KEY, _DEV_SIGNER_LABEL


def sign_certificate(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a signed copy of ``payload`` (payload is not mutated)."""
    key, signer = _resolve_key()
    clean = _canonical_payload(payload)
    signature = hmac.new(key, clean.encode("utf-8"), hashlib.sha256).hexdigest()
    signed = dict(payload)
    signed["signer"] = signer
    signed["signature"] = signature
    return signed


def verify_certificate(
    cert: dict[str, Any],
    *,
    key: bytes | None = None,
    require_thresholds: bool = True,
    cosine_threshold: float = 0.999,
) -> tuple[bool, str]:
    """Verify a cert's signature + (optionally) its certification thresholds.

    Args:
        cert: The full signed cert dict.
        key: Override the signing key (e.g. when verifying a prod-signed cert
            with a known key).  When ``None`` the key is resolved via
            :func:`_resolve_key` (env -> repo file -> DEV key).
        require_thresholds: If ``True``, also assert
            ``logit_cosine_sim >= cosine_threshold`` and
            ``token_exact_match == 1.0``.
        cosine_threshold: Threshold used when ``require_thresholds``.

    Returns:
        ``(ok, reason)``.  ``ok`` is ``True`` only when the signature matches
        AND (if required) the thresholds are cleared.
    """
    if "signature" not in cert:
        return False, "certificate has no signature"
    if "signer" not in cert:
        return False, "certificate has no signer"

    try:
        clean = _canonical_payload(cert)
    except (TypeError, ValueError) as exc:
        return False, f"cannot canonicalize cert: {exc}"

    if key is None:
        key, _ = _resolve_key()
    expected = hmac.new(key, clean.encode("utf-8"), hashlib.sha256).hexdigest()

    if not hmac.compare_digest(expected, cert["signature"]):
        return False, "signature mismatch"

    if require_thresholds:
        cos = cert.get("logit_cosine_sim")
        match = cert.get("token_exact_match")
        if not isinstance(cos, (int, float)) or cos < cosine_threshold:
            return False, f"logit_cosine_sim {cos} below threshold {cosine_threshold}"
        if match != 1.0:
            return False, f"token_exact_match {match} != 1.0"

    return True, "ok"
