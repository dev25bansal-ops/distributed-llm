"""Regression tests for CRITICAL fix C2:

OIDC auth bypass. The callback accepted an identity derived from an
**unverified** id_token whenever JWKS discovery failed
(``_discovered_jwks_url`` empty). An attacker presenting any unsigned JWT
could impersonate any subject.

Fix: fail-closed by default — if the id_token signature cannot be
cryptographically verified (no JWKS) and unverified tokens are not explicitly
allowed, reject (return ``None``).

The original auth/ fork was consolidated into the canonical
``distllm.api.sso_auth`` module. These tests drive the REAL
``OIDCHandler.handle_callback`` with a stubbed token endpoint so the actual
fail-closed guard is exercised. A forged unsigned token must be rejected
unless the operator explicitly opts into unverified tokens. No network or
GPU is required.
"""

from __future__ import annotations

import base64
import importlib.util
import json
import os
import sys
import time

import pytest

# PyJWT (`jwt`) and `httpx` are required by the real OIDC callback. They are
# present in CI (security/dev extras) but may be absent on a minimal local
# venv, in which case these tests skip rather than error.
jwt = pytest.importorskip("jwt")
pytest.importorskip("httpx")

# The full `distllm.api` package imports heavy deps (torch, fastapi server,
# etc.), so we load the single canonical module in isolation.  sso_auth.py
# has no intra-package imports (optional deps are imported lazily inside the
# handlers), so it can be loaded under any name without stubbing the parent
# packages — stubbing `distllm`/`distllm.api` in sys.modules would shadow the
# real packages for every later test in the session.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))


def _load_sso_auth_module():
    """Load sso_auth.py in isolation (mirrors the C2 loader)."""
    sso_path = os.path.join(_REPO_ROOT, "distllm", "api", "sso_auth.py")
    name = "distllm.api.sso_auth"
    _prev = sys.modules.get(name)
    spec = importlib.util.spec_from_file_location(name, sso_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    result = sys.modules[name]

    # Self-clean: evict our isolated copy so later tests import the real
    # package normally.  Bound references above keep working for this file.
    sys.modules.pop(name, None)
    if _prev is not None:
        sys.modules[name] = _prev
    return result


_sso_mod = _load_sso_auth_module()
OIDCHandler = _sso_mod.OIDCHandler
SSOUserInfo = _sso_mod.SSOUserInfo


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
    h._discovered_jwks_url = ""
    h._allow_unverified_id_token = allow_unverified
    h._state_store = {}
    h._nonce_store = {}
    h._pkce_store = {}
    return h


def _stub_exchange(handler: OIDCHandler, id_token: str):
    """Replace the network token exchange with a canned response."""

    def _fake_post(url, data=None, headers=None, timeout=None):
        class _R:
            status_code = 200
            text = "ok"

            def json(self):
                return {"access_token": "at", "id_token": id_token}

        return _R()

    def _fake_get(url, headers=None, timeout=None):
        class _R:
            status_code = 200
            text = "ok"

            def json(self):
                return {}

        return _R()

    import httpx

    httpx.post = _fake_post
    httpx.get = _fake_get

    handler._token_endpoint = "https://idp.example.com/token"
    handler._userinfo_endpoint = ""
    return handler


def _seed_state(handler: OIDCHandler, state: str, nonce: str) -> None:
    handler._state_store[state] = time.time() + 600
    handler._nonce_store[state] = {"nonce": nonce, "expires": time.time() + 600}
    handler._pkce_store[state] = "verifier"


def test_forged_unsigned_token_rejected_by_default():
    """C2: an unverified token with no JWKS must NOT authenticate."""
    h = _make_handler(allow_unverified=False)
    _seed_state(h, "st", "expected-nonce")
    _stub_exchange(h, _forge_unsigned_id_token("victim-admin"))
    result = h.handle_callback("code", expected_state="st", expected_nonce="expected-nonce")
    assert result is None, "Unverified id_token must be rejected (auth bypass)"


def test_valid_nonce_but_unverified_still_rejected():
    """Even with a matching nonce, an unverified token is rejected by default."""
    h = _make_handler(allow_unverified=False)
    _seed_state(h, "st", "nn")
    _stub_exchange(h, _forge_unsigned_id_token("victim-admin", nonce="nn"))
    result = h.handle_callback("code", expected_state="st", expected_nonce="nn")
    assert result is None


def test_unverified_token_allowed_only_when_opted_in():
    """Opt-in flag is required; default is fail-closed."""
    h = _make_handler(allow_unverified=True)
    _seed_state(h, "st", "expected-nonce")
    _stub_exchange(h, _forge_unsigned_id_token("victim-admin"))
    result = h.handle_callback("code", expected_state="st", expected_nonce="expected-nonce")
    assert isinstance(result, SSOUserInfo)
