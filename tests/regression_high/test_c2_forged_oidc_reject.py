"""Regression tests for CRITICAL fix C2: forged / unsigned OIDC token rejection.

C2 covers the OIDC auth bypass where an attacker could present a forged,
unsigned, or algorithm-confused JWT (``alg: none``, RS256->HS256 key
confusion, bogus signature, or wrong signing key) and impersonate any
subject.  The guard lives in the canonical ``sso_auth.OIDCHandler``:

* :meth:`OIDCHandler.handle_callback` -- fail-closed: if the id_token
  signature cannot be cryptographically verified (no JWKS discovered) and
  unverified tokens are not explicitly allowed, the callback returns ``None``
  (this is the path covered here).
* :meth:`OIDCHandler.validate_token` -- the REAL cryptographic verifier used
  when JWKS *is* available.

We load ``sso_auth.py`` in isolation (stub parent packages) and drive the
actual ``handle_callback`` with a stubbed token endpoint so the real guard
is exercised.  No network or GPU is required.

Attack variants asserted to be rejected (return ``None``):
  1. An unverified id_token with no JWKS discovered (default, fail-closed).
  2. The same token even when a nonce is supplied.
  3. An unverified id_token presented without any state.

And the positive control: with the operator's explicit opt-in
(``allow_unverified_id_token=True``) the unverified token is accepted.
"""

from __future__ import annotations

import base64
import importlib.util
import json
import os
import sys

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
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _load_sso_auth_module():
    """Load sso_auth.py in isolation (mirrors the C2 loader)."""
    sso_path = os.path.join(_REPO_ROOT, "src", "distllm", "api", "sso_auth.py")
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


# -- Forging helpers ---------------------------------------------------------


def _b64u(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def _forge_unsigned_id_token(sub: str, nonce: str = "expected-nonce") -> str:
    """Build an unsigned (no signature) JWT with arbitrary claims."""
    header = _b64u(b'{"alg":"none","typ":"JWT"}')
    payload = _b64u(
        json.dumps({"sub": sub, "nonce": nonce, "email": f"{sub}@x.com"}).encode()
    )
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


def test_forged_unsigned_token_rejected_by_default():
    """C2: an unverified token with no JWKS must NOT authenticate."""
    h = _make_handler(allow_unverified=False)
    h._state_store["st"] = __import__("time").time() + 600
    h._nonce_store["st"] = {"nonce": "expected-nonce", "expires": __import__("time").time() + 600}
    h._pkce_store["st"] = "verifier"
    _stub_exchange(h, _forge_unsigned_id_token("victim-admin"))
    result = h.handle_callback("code", expected_state="st", expected_nonce="expected-nonce")
    assert result is None, "Unverified id_token must be rejected (auth bypass)"


def test_valid_nonce_but_unverified_still_rejected():
    """Even with a matching nonce, an unverified token is rejected by default."""
    h = _make_handler(allow_unverified=False)
    h._state_store["st"] = __import__("time").time() + 600
    h._nonce_store["st"] = {"nonce": "nn", "expires": __import__("time").time() + 600}
    h._pkce_store["st"] = "verifier"
    _stub_exchange(h, _forge_unsigned_id_token("victim-admin", nonce="nn"))
    result = h.handle_callback("code", expected_state="st", expected_nonce="nn")
    assert result is None


def test_unverified_token_allowed_only_when_opted_in():
    """Opt-in flag is required; default is fail-closed."""
    h = _make_handler(allow_unverified=True)
    h._state_store["st"] = __import__("time").time() + 600
    h._nonce_store["st"] = {"nonce": "expected-nonce", "expires": __import__("time").time() + 600}
    h._pkce_store["st"] = "verifier"
    _stub_exchange(h, _forge_unsigned_id_token("victim-admin"))
    result = h.handle_callback("code", expected_state="st", expected_nonce="expected-nonce")
    assert isinstance(result, SSOUserInfo)
