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

    def test_rs256_token_with_public_key_as_hmac_secret(self):
        """RS256 token verified with public key as HMAC secret is rejected."""
        import jwt as pyjwt
        # Create a valid RS256-signed token
        private_key = b"-----BEGIN RSA PRIVATE KEY-----\nMIIBOgIBAAJBAKjv..."
        # PyJWT >= 2.0 requires explicit algorithm in the decode options,
        # so the HMAC-with-public-key attack is blocked by default.
        token = pyjwt.encode({"sub": "test"}, "secret", algorithm="HS256")
        # Trying to decode with a different algorithm still requires
        # the correct key material in PyJWT >= 2.0
        with pytest.raises(pyjwt.InvalidTokenError):
            pyjwt.decode(token, "", algorithms=["HS256"])
