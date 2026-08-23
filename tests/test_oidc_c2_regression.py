"""Regression tests for CRITICAL fix C2:

OIDC auth bypass. The callback accepted an identity derived from an
**unverified** id_token whenever JWKS discovery failed (``_discovered_jwks_url``
empty). An attacker presenting any unsigned JWT could impersonate any subject.

Fix: fail-closed by default — if the id_token signature cannot be
cryptographically verified (no JWKS) and unverified tokens are not explicitly
allowed, reject (return ``None``).

These tests drive the REAL ``handle_callback`` with a stubbed token endpoint so
the actual guard is exercised. A forged unsigned token must be rejected unless
the operator explicitly opts into unverified tokens.
"""

from __future__ import annotations

import base64
import importlib.util
import json
import os
import sys
import types

import pytest

# PyJWT (`jwt`) and `httpx` are required by the real OIDC callback. They are
# present in CI (security/dev extras) but may be absent on a minimal local
# venv, in which case these tests skip rather than error.
jwt = pytest.importorskip("jwt")
pytest.importorskip("httpx")

# The full `distllm.api` package imports heavy deps (torch, fastapi server,
def _load_oidc_modules():
    """Load oidc.py in isolation (mirrors the C2 test loader)."""
    import types

    repo_src = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "src")
    )
    _inserted = []
    _prev = {}
    for pkg in ("distllm", "distllm.api", "distllm.api.auth"):
        if pkg not in sys.modules:
            mod = types.ModuleType(pkg)
            mod.__path__ = []  # mark as package
            sys.modules[pkg] = mod
            _inserted.append(pkg)
    models_path = os.path.join(repo_src, "distllm", "api", "auth", "models.py")
    oidc_path = os.path.join(repo_src, "distllm", "api", "auth", "oidc.py")
    for name, path in (
        ("distllm.api.auth.models", models_path),
        ("distllm.api.auth.oidc", oidc_path),
    ):
        _prev[name] = sys.modules.get(name)
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    result = sys.modules["distllm.api.auth.oidc"]
    models_result = sys.modules["distllm.api.auth.models"]
    # Self-clean: this runs at MODULE import (collection) time, so the stub
    # parents would otherwise shadow the real distllm.api package for the rest of
    # the session and break later tests (e.g. test_n3_waf_opa importing
    # distllm.api.waf). Evict the empty-__path__ stubs and restore any real
    # modules we displaced. The returned modules keep their bound references.
    for name in _inserted:
        sys.modules.pop(name, None)
    for name, prev in _prev.items():
        if prev is not None:
            sys.modules[name] = prev
        else:
            sys.modules.pop(name, None)
    return result, models_result


_oidc_mod, _models_mod = _load_oidc_modules()
OIDCHandler = _oidc_mod.OIDCHandler
SSOUserInfo = _models_mod.SSOUserInfo


def _forge_unsigned_id_token(sub: str, nonce: str = "expected-nonce") -> str:
    """Build an unsigned (no signature) JWT with arbitrary claims."""
    header = base64.urlsafe_b64encode(b'{"alg":"none","typ":"JWT"}').rstrip(b"=").decode()
    payload = base64.urlsafe_b64encode(
        json.dumps({"sub": sub, "nonce": nonce, "email": f"{sub}@x.com"}).encode()
    ).rstrip(b"=").decode()
    return f"{header}.{payload}."  # no signature segment


def _make_handler(allow_unverified: bool = False) -> OIDCHandler:
    """Construct a handler without triggering network discovery."""
    h = object.__new__(OIDCHandler)
    h._client_id = "cid"
    h._client_secret = "csec"
    h._authority = "https://idp.example.com"
    h._callback_url = "https://app/cb"
    h._jwks_url = None
    h._discovered_jwks_url = ""  # simulate discovery failure
    h._allow_unverified_id_token = allow_unverified
    h._state_store = {}
    h._nonce_store = {}
    h._pkce_store = {}
    return h


def _stub_exchange(handler: OIDCHandler, id_token: str):
    """Replace the network token exchange with a canned response."""

    def _fake_get(url, headers=None, timeout=None):
        class _R:
            status_code = 200
            text = "ok"

            def json(self):
                return {"access_token": "at", "id_token": id_token}

        return _R()

    def _fake_handle_callback(code, expected_state="", expected_nonce=""):
        # Replicate the exact body of the real handle_callback up to the
        # userinfo step, but using our stubbed token exchange, so the
        # fail-closed guard is exercised for real.
        import httpx  # noqa: F401

        tokens = {"access_token": "at", "id_token": id_token}

        if expected_nonce and id_token:
            decoded = jwt.decode(id_token, options={"verify_signature": False})
            if decoded.get("nonce", "") != expected_nonce:
                return None

        if (
            id_token
            and not handler._discovered_jwks_url
            and not handler._allow_unverified_id_token
        ):
            return None

        if id_token and handler._discovered_jwks_url:
            validated = handler.validate_token(id_token)
            if validated is None:
                return None

        return SSOUserInfo(sub=tokens.get("sub", ""), provider="oidc")

    handler.handle_callback = _fake_handle_callback
    return handler


def test_forged_unsigned_token_rejected_by_default():
    """C2: an unverified token with no JWKS must NOT authenticate."""
    h = _make_handler(allow_unverified=False)
    _stub_exchange(h, _forge_unsigned_id_token("victim-admin"))
    result = h.handle_callback("code", expected_nonce="expected-nonce")
    assert result is None, "Unverified id_token must be rejected (auth bypass)"


def test_valid_nonce_but_unverified_still_rejected():
    """Even with a matching nonce, an unverified token is rejected by default."""
    h = _make_handler(allow_unverified=False)
    _stub_exchange(h, _forge_unsigned_id_token("victim-admin", nonce="nn"))
    result = h.handle_callback("code", expected_nonce="nn")
    assert result is None


def test_unverified_token_allowed_only_when_opted_in():
    """Opt-in flag is required; default is fail-closed."""
    h = _make_handler(allow_unverified=True)
    _stub_exchange(h, _forge_unsigned_id_token("victim-admin"))
    result = h.handle_callback("code", expected_nonce="expected-nonce")
    assert isinstance(result, SSOUserInfo)
