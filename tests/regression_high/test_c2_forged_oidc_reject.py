"""Regression tests for CRITICAL fix C2: forged / unsigned OIDC token rejection.

C2 covers the OIDC auth bypass where an attacker could present a forged,
unsigned, or algorithm-confused JWT (``alg: none``, RS256->HS256 key
confusion, bogus signature, or wrong signing key) and impersonate any
subject.  The guard lives in two places:

* :meth:`OIDCHandler.handle_callback` -- fail-closed: if the id_token
  signature cannot be cryptographically verified (no JWKS discovered) and
  unverified tokens are not explicitly allowed, the callback returns ``None``
  (this is the path covered by ``test_oidc_c2_regression.py``).
* :meth:`OIDCHandler.validate_token` -- the REAL cryptographic verifier used
  when JWKS *is* available.  This test exercises it directly with genuinely
  forged tokens so we prove the signature check (not just the discovery
  failure fallback) rejects every attack variant.

We load ``oidc.py`` in isolation (stub parent packages, mirror the C2 loader)
and drive the actual ``validate_token`` with a stubbed JWKS endpoint.  No
network or GPU is required.

Attack variants asserted to be rejected (return ``None``):
  1. ``alg: none`` unsigned token.
  2. RS256 -> HS256 algorithm confusion (sign HS256 with the public key PEM).
  3. Token signed by an *unknown* RSA key (wrong key / forged signature).
  4. Valid RS256 token with a *tampered* signature.
  5. Valid RS256 token but with the wrong ``aud`` (audience mismatch).
  6. Malformed / truncated token.

And the positive control: a legitimately RS256-signed token with the trusted
key + correct audience MUST be accepted.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import importlib.util
import json
import os
import sys
import types

import pytest

jwt = pytest.importorskip("jwt")
pytest.importorskip("httpx")
# cryptography is required to mint the RSA keys for the confusion / wrong-key
# attacks.  It is present in the security/dev extras; skip otherwise.
pytest.importorskip("cryptography")

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _load_oidc_modules():
    """Load oidc.py (and its models) in isolation, mirroring the C2 loader."""
    repo_src = os.path.join(_REPO_ROOT, "src")
    _inserted: list[str] = []
    for pkg in ("distllm", "distllm.api", "distllm.api.auth"):
        if pkg not in sys.modules:
            mod = types.ModuleType(pkg)
            mod.__path__ = []  # mark as package -- never run the heavy __init__
            sys.modules[pkg] = mod
            _inserted.append(pkg)

    models_path = os.path.join(repo_src, "distllm", "api", "auth", "models.py")
    oidc_path = os.path.join(repo_src, "distllm", "api", "auth", "oidc.py")
    for name, path in (
        ("distllm.api.auth.models", models_path),
        ("distllm.api.auth.oidc", oidc_path),
    ):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)

    result = sys.modules["distllm.api.auth.oidc"]
    models_result = sys.modules["distllm.api.auth.models"]

    # Self-clean so we do NOT shadow the real distllm.api package for the rest
    # of the pytest session (other tests import distllm.api.*).
    for name in _inserted:
        sys.modules.pop(name, None)
    return result, models_result


_oidc_mod, _models_mod = _load_oidc_modules()
OIDCHandler = _oidc_mod.OIDCHandler
SSOUserInfo = _models_mod.SSOUserInfo


# -- Forging helpers ---------------------------------------------------------


def _b64u(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def _make_keys():
    """Generate a trusted RSA keypair + its JWK (kid set to 'k1')."""
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.primitives import serialization

    priv = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pub = priv.public_key()
    priv_pem = priv.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    pub_pem = pub.public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(pub))
    jwk["kid"] = "k1"
    return priv_pem, pub_pem, jwk


def _make_handler(jwk):
    """Construct a handler whose JWKS endpoint returns the given ``jwk``."""
    import httpx

    h = object.__new__(OIDCHandler)
    h._client_id = "cid"
    h._client_secret = "csec"
    h._authority = "https://idp.example.com"
    h._callback_url = "https://app/cb"
    h._jwks_url = "https://idp.example.com/.well-known/jwks"
    h._discovered_jwks_url = "https://idp.example.com/.well-known/jwks"
    h._allow_unverified_id_token = False
    h._state_store = {}
    h._nonce_store = {}
    h._pkce_store = {}

    def _fake_get(url, timeout=None):
        class _R:
            status_code = 200

            def json(self):
                return {"keys": [jwk]}

        return _R()

    httpx.get = _fake_get
    return h


# -- Tests -------------------------------------------------------------------


def test_legit_rs256_accepted():
    """Positive control: a properly signed token with the trusted key + aud."""
    priv_pem, _pub_pem, jwk = _make_keys()
    h = _make_handler(jwk)
    token = jwt.encode(
        {"sub": "alice", "aud": "cid"}, priv_pem, algorithm="RS256", headers={"kid": "k1"}
    )
    res = h.validate_token(token)
    assert isinstance(res, SSOUserInfo), "legit RS256 token must be accepted"
    assert res.sub == "alice"


def test_none_alg_rejected():
    """C2: an unsigned ``alg: none`` token must be rejected."""
    _priv_pem, _pub_pem, jwk = _make_keys()
    h = _make_handler(jwk)
    header = _b64u(json.dumps({"alg": "none", "typ": "JWT", "kid": "k1"}).encode())
    payload = _b64u(json.dumps({"sub": "victim-admin", "aud": "cid"}).encode())
    token = f"{header}.{payload}."  # no signature segment
    assert h.validate_token(token) is None, "'none' alg token must be rejected"


def test_rs256_to_hs256_confusion_rejected():
    """C2: RS256->HS256 confusion -- sign HS256 with the public key as the HMAC secret."""
    _priv_pem, pub_pem, jwk = _make_keys()
    h = _make_handler(jwk)
    header = _b64u(json.dumps({"alg": "HS256", "typ": "JWT", "kid": "k1"}).encode())
    payload = _b64u(json.dumps({"sub": "attacker", "aud": "cid"}).encode())
    signing = f"{header}.{payload}".encode()
    sig = hmac.new(pub_pem, signing, hashlib.sha256).digest()
    token = f"{header}.{payload}.{_b64u(sig)}"
    assert h.validate_token(token) is None, "RS256->HS256 confusion must be rejected"


def test_wrong_signing_key_rejected():
    """C2: token signed by a different (attacker-controlled) RSA key."""
    _priv_pem, _pub_pem, jwk = _make_keys()
    h = _make_handler(jwk)
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.primitives import serialization

    evil = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    evil_pem = evil.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    token = jwt.encode(
        {"sub": "attacker", "aud": "cid"}, evil_pem, algorithm="RS256", headers={"kid": "k1"}
    )
    assert h.validate_token(token) is None, "token signed by unknown key must be rejected"


def test_tampered_signature_rejected():
    """C2: valid RS256 token with its signature corrupted must be rejected."""
    priv_pem, _pub_pem, jwk = _make_keys()
    h = _make_handler(jwk)
    token = jwt.encode(
        {"sub": "alice", "aud": "cid"}, priv_pem, algorithm="RS256", headers={"kid": "k1"}
    )
    parts = token.split(".")
    parts[2] = _b64u(b"X" * 32)  # overwrite signature with garbage
    assert h.validate_token(".".join(parts)) is None, "tampered signature must be rejected"


def test_wrong_audience_rejected():
    """C2: correct signature but wrong audience must be rejected (no tenant confusion)."""
    priv_pem, _pub_pem, jwk = _make_keys()
    h = _make_handler(jwk)
    token = jwt.encode(
        {"sub": "alice", "aud": "other-audience"},
        priv_pem,
        algorithm="RS256",
        headers={"kid": "k1"},
    )
    assert h.validate_token(token) is None, "wrong-audience token must be rejected"


def test_malformed_token_rejected():
    """C2: a structurally broken token must be rejected, never crash/accept."""
    _priv_pem, _pub_pem, jwk = _make_keys()
    h = _make_handler(jwk)
    for bad in ("", "not.a.jwt", "header.payload", "%%%not-base64%%%"):
        assert h.validate_token(bad) is None, f"malformed token {bad!r} must be rejected"


def test_unknown_kid_rejected():
    """C2: token signed with a kid the IdP never published must be rejected."""
    priv_pem, _pub_pem, jwk = _make_keys()
    h = _make_handler(jwk)
    token = jwt.encode(
        {"sub": "alice", "aud": "cid"}, priv_pem, algorithm="RS256", headers={"kid": "unknown-key"}
    )
    assert h.validate_token(token) is None, "token with unknown kid must be rejected"
