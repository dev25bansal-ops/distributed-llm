"""Security: JWT algorithm confusion attack (alg=none).

The ``validate_jwt`` function must reject tokens with ``alg: "none"``
to prevent signature bypass attacks where an attacker crafts a token
with a valid payload but no signature.

CVE-2015-9235 (JWT algorithm confusion) affects PyJWT < 1.5.0.
This test verifies the current library version and the application's
defence in depth.
"""

from __future__ import annotations

import importlib.metadata
import json
import base64

import pytest


def _b64url(data: bytes) -> str:
    """Base64url-encode *data* without padding."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _make_alg_none_token(payload: dict) -> str:
    """Craft a JWT with ``alg: "none"`` (no signature)."""
    header = _b64url(json.dumps({"alg": "none", "typ": "JWT"}).encode())
    body = _b64url(json.dumps(payload).encode())
    return f"{header}.{body}."


class TestJWTAlgorithmNone:
    """JWT ``alg: "none"`` attacks."""

    def test_library_rejects_alg_none(self):
        """PyJWT >= 2.0 rejects alg=none by default."""
        import jwt as pyjwt
        token = _make_alg_none_token({"sub": "admin", "role": "admin"})
        with pytest.raises(pyjwt.InvalidTokenError):
            pyjwt.decode(token, options={"verify_signature": True})

    def test_library_version_is_current(self):
        """PyJWT version should be >= 2.0 to block alg=none."""
        version = importlib.metadata.version("pyjwt")
        major = int(version.split(".")[0])
        assert major >= 2, f"PyJWT {version} is vulnerable to alg=none"

    def test_validate_jwt_rejects_alg_none(self):
        """The application's validate_jwt rejects alg=none."""
        from distllm.plugins.auth_plugin import validate_jwt
        token = _make_alg_none_token({"sub": "attacker"})
        result = validate_jwt(token, secret="any-secret")
        assert result is None, "alg=none token must be rejected"

    def test_validate_jwt_rejects_alg_none(self):
        """validate_jwt rejects alg=none tokens (defence in depth)."""
        from distllm.plugins.auth_plugin import validate_jwt
        token = _make_alg_none_token({"sub": "attacker", "role": "admin"})
        result = validate_jwt(token, secret="any-secret-key-here")
        assert result is None, "alg=none token must be rejected by validate_jwt"

    def test_empty_hmac_key_rejected(self):
        """PyJWT rejects HS256 tokens with empty HMAC key.

        When an empty string "" is provided as the HMAC secret, PyJWT raises
        ``InvalidKeyError`` (NOT ``InvalidTokenError``), which blocks any
        attempt to bypass signature verification via blank secrets.

        Note: InvalidKeyError inherits from PyJWTError but not InvalidTokenError.
        """
        import jwt as pyjwt
        token = pyjwt.encode({"sub": "test"}, "secret", algorithm="HS256")
        with pytest.raises(pyjwt.InvalidKeyError):
            pyjwt.decode(token, "", algorithms=["HS256"])

    def test_rs256_to_hs256_confusion_attack_blocked(self):
        """Classical algorithm confusion: forged HS256 with PEM pubkey is rejected.

        Simulates CVE-2015-9235-style confusion where an attacker:
        1. Obtains the RSA public key (PEM format) from infrastructure/config.
        2. Forges an HS256 token HMAC'd with the PEM public key as the secret.
        3. Submits this token expecting it to be accepted with the same PEM as config.

        This must fail-closed because:
        * HS256 signed with a public key is forgeable by anyone possessing it.
        * A system configured with a PEM key must ONLY accept RS256 signatures.
        """
        import jwt as pyjwt
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.hazmat.primitives import serialization
        import hmac, hashlib, json, base64

        # Generate real RSA keypair
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        pub = key.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        ).decode()

        payload = {"sub": "attacker", "role": "admin"}

        # Forge: HMAC the payload with the PEM public key (anyone can do this!)
        def b64(b: bytes) -> str:
            return base64.urlsafe_b64encode(b).rstrip(b"=").decode()

        h = b64(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
        p = b64(json.dumps(payload).encode())
        sig = b64(hmac.new(pub.encode(), f"{h}.{p}".encode(), hashlib.sha256).digest())
        forged = f"{h}.{p}.{sig}"

        # App's validate_jwt must reject this on ALL paths (PyJWT or fallback).
        # It returns None for failures — never raises or accepts forged tokens.
        from distllm.plugins.auth_plugin import validate_jwt

        # The forged token MUST NOT authenticate.
        assert validate_jwt(forged, pub) is None
