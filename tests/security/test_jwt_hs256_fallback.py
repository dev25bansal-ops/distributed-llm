"""Regression tests: hardened HS256 JWT fallback validator (auth_plugin).

Covers the P0 finding "JWT authentication bypass via algorithm confusion in
HS256 fallback validator" (src/distllm/plugins/auth_plugin.py). The pure-Python
fallback (used when PyJWT is unavailable) must be fail-closed:

* only ``alg == "HS256"`` tokens are accepted (rejects ``alg: none`` and
  asymmetric-algorithm confusion);
* a PEM key (public material) must never be usable as an HMAC secret;
* a shared secret shorter than 32 chars must never authenticate a token;
* when an audience/issuer is configured, the claim must be present AND match;
* the AuthPlugin must disable JWT (fail closed) on an insecure secret config.

Most tests force the fallback path by monkeypatching ``_HAS_PYJWT`` to False so
the pure-Python validator is exercised regardless of whether PyJWT is installed.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time

import pytest

from distllm.plugins import auth_plugin
from distllm.plugins.auth_plugin import AuthPlugin, _validate_jwt_hs256, validate_jwt

#: 40-char shared secret — passes the >=32 requirement.
STRONG_SECRET = "super-secret-key-that-is-32-chars-long!!"

#: A PEM public key is PUBLIC material — forging with it must fail.
PEM_PUBLIC_KEY = (
    "-----BEGIN PUBLIC KEY-----\n"
    "MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA8gC8t5dW7zJ6dKe8x0K9\n"
    "nT7cYp2vQGqA1rB3mD4sHfE5jWkLmNpQrStUvWxYz09aBcDeFgHiJkLmNoPqRs\n"
    "-----END PUBLIC KEY-----\n"
)


def _b64url(data: bytes) -> str:
    """Base64url-encode *data* without padding."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _make_token(
    payload: dict,
    secret: str,
    alg: str = "HS256",
    sign: bool = True,
) -> str:
    """Craft a JWT. With ``sign=False`` the signature is left empty (alg:none)."""
    header = _b64url(json.dumps({"alg": alg, "typ": "JWT"}).encode())
    body = _b64url(json.dumps(payload).encode())
    message = f"{header}.{body}".encode()
    sig = _b64url(hmac.new(secret.encode(), message, hashlib.sha256).digest()) if sign else ""
    return f"{header}.{body}.{sig}"


def _future_payload(**overrides) -> dict:
    payload = {"sub": "test", "role": "admin", "exp": time.time() + 3600}
    payload.update(overrides)
    return payload


@pytest.fixture(autouse=True)
def _force_fallback(monkeypatch):
    """Most tests target the pure-Python fallback; force it module-wide."""
    monkeypatch.setattr(auth_plugin, "_HAS_PYJWT", False)


class TestFallbackHS256Validator:
    """Direct tests of _validate_jwt_hs256 (pure-Python path)."""

    def test_valid_hs256_returns_payload(self):
        token = _make_token(_future_payload(), STRONG_SECRET)
        result = _validate_jwt_hs256(token, STRONG_SECRET)
        assert result is not None
        assert result["role"] == "admin"

    def test_rejects_alg_none(self):
        token = _make_token(_future_payload(), STRONG_SECRET, alg="none", sign=False)
        assert _validate_jwt_hs256(token, STRONG_SECRET) is None

    def test_rejects_asymmetric_alg_header(self):
        # An attacker flips the alg header to RS256 but signs with the shared
        # secret. The algorithm whitelist must reject it regardless of signature.
        token = _make_token(_future_payload(), STRONG_SECRET, alg="RS256")
        assert _validate_jwt_hs256(token, STRONG_SECRET) is None

    def test_rejects_pem_public_key_as_secret(self):
        token = _make_token(_future_payload(), PEM_PUBLIC_KEY)
        assert _validate_jwt_hs256(token, PEM_PUBLIC_KEY) is None

    def test_rejects_short_secret(self):
        token = _make_token(_future_payload(), "short-secret")
        assert _validate_jwt_hs256(token, "short-secret") is None

    def test_rejects_wrong_signature(self):
        token = _make_token(_future_payload(), "another-secret-key-that-is-long-enough!!")
        assert _validate_jwt_hs256(token, STRONG_SECRET) is None

    def test_rejects_expired(self):
        token = _make_token(_future_payload(exp=time.time() - 3600), STRONG_SECRET)
        assert _validate_jwt_hs256(token, STRONG_SECRET) is None

    def test_rejects_malformed(self):
        assert _validate_jwt_hs256("not-a-jwt", STRONG_SECRET) is None
        assert _validate_jwt_hs256("a.b", STRONG_SECRET) is None
        assert _validate_jwt_hs256("", STRONG_SECRET) is None

    def test_rejects_non_dict_payload(self):
        # A signed token whose payload decodes to a non-object (e.g. a JSON
        # array) must not crash the caller with AttributeError — fail closed.
        token = _make_token([1, 2, 3], STRONG_SECRET)
        assert _validate_jwt_hs256(token, STRONG_SECRET) is None
        assert validate_jwt(token, STRONG_SECRET) is None


class TestValidateJWTFallback:
    """validate_jwt() exercising the fallback path (PyJWT forced off)."""

    def test_valid_hs256_fallback_returns_payload(self):
        token = _make_token(_future_payload(), STRONG_SECRET)
        result = validate_jwt(token, STRONG_SECRET)
        assert result is not None
        assert result["role"] == "admin"

    def test_alg_none_confusion_rejected(self):
        # CVE-2015-9235-style: admin token with alg:none and no signature.
        token = _make_token({"sub": "attacker", "role": "admin"}, STRONG_SECRET, alg="none", sign=False)
        assert validate_jwt(token, STRONG_SECRET) is None

    def test_pem_public_key_confusion_rejected(self):
        # The headline attack: use the PUBLIC PEM key as an HMAC secret so
        # anyone (knowing the public key) could sign an admin token.
        token = _make_token({"role": "admin", "exp": time.time() + 3600}, PEM_PUBLIC_KEY)
        assert validate_jwt(token, PEM_PUBLIC_KEY) is None

    def test_short_secret_confusion_rejected(self):
        token = _make_token({"role": "admin", "exp": time.time() + 3600}, "guessme")
        assert validate_jwt(token, "guessme") is None

    def test_strict_audience_when_configured(self):
        match = _make_token(_future_payload(aud="api"), STRONG_SECRET)
        wrong = _make_token(_future_payload(aud="other"), STRONG_SECRET)
        missing = _make_token(_future_payload(), STRONG_SECRET)

        assert validate_jwt(match, STRONG_SECRET, audience="api") is not None
        assert validate_jwt(wrong, STRONG_SECRET, audience="api") is None
        assert validate_jwt(missing, STRONG_SECRET, audience="api") is None

    def test_strict_issuer_when_configured(self):
        match = _make_token(_future_payload(iss="distllm"), STRONG_SECRET)
        wrong = _make_token(_future_payload(iss="evil"), STRONG_SECRET)
        missing = _make_token(_future_payload(), STRONG_SECRET)

        assert validate_jwt(match, STRONG_SECRET, issuer="distllm") is not None
        assert validate_jwt(wrong, STRONG_SECRET, issuer="distllm") is None
        assert validate_jwt(missing, STRONG_SECRET, issuer="distllm") is None

    def test_audience_not_required_when_not_configured(self):
        token = _make_token(_future_payload(), STRONG_SECRET)
        assert validate_jwt(token, STRONG_SECRET) is not None


class TestAuthPluginConfigFailClosed:
    """AuthPlugin.on_init must disable JWT when the secret is insecure."""

    def _init(self, monkeypatch, secret: str | None):
        monkeypatch.setenv("DISTLLM_AUTH_ENABLED", "1")
        if secret is None:
            monkeypatch.delenv("DISTLLM_AUTH_SECRET", raising=False)
        else:
            monkeypatch.setenv("DISTLLM_AUTH_SECRET", secret)
        plugin = AuthPlugin()
        plugin.on_init({"config": {}})
        return plugin

    def test_strong_secret_enables_jwt(self, monkeypatch):
        plugin = self._init(monkeypatch, STRONG_SECRET)
        assert plugin._jwt_secret == STRONG_SECRET

    def test_short_secret_disables_jwt(self, monkeypatch):
        plugin = self._init(monkeypatch, "short-secret")
        assert plugin._jwt_secret == ""

    def test_pem_secret_without_pyjwt_disables_jwt(self, monkeypatch):
        # PyJWT is already forced off by the autouse fixture.
        plugin = self._init(monkeypatch, PEM_PUBLIC_KEY)
        assert plugin._jwt_secret == ""

    def test_pem_secret_with_pyjwt_enables_rs256(self, monkeypatch):
        # Restore PyJWT for this test: RS256 via PEM is legitimate when PyJWT
        # is present.
        monkeypatch.setattr(auth_plugin, "_HAS_PYJWT", True)
        plugin = self._init(monkeypatch, PEM_PUBLIC_KEY)
        assert plugin._jwt_secret == PEM_PUBLIC_KEY

    def test_no_secret_leaves_jwt_disabled(self, monkeypatch):
        plugin = self._init(monkeypatch, None)
        assert plugin._jwt_secret == ""


class TestPyJWTMainPathStillWorks:
    """The primary PyJWT path is unchanged (guards only add fail-closed rules)."""

    def test_valid_hs256_accepted_when_pyjwt_present(self, monkeypatch):
        monkeypatch.setattr(auth_plugin, "_HAS_PYJWT", True)
        token = _make_token(_future_payload(), STRONG_SECRET)
        assert validate_jwt(token, STRONG_SECRET) is not None

    def test_alg_none_rejected_when_pyjwt_present(self, monkeypatch):
        monkeypatch.setattr(auth_plugin, "_HAS_PYJWT", True)
        token = _make_token(_future_payload(), STRONG_SECRET, alg="none", sign=False)
        assert validate_jwt(token, STRONG_SECRET) is None
