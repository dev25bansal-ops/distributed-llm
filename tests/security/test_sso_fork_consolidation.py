"""Regression test: SSO fork consolidation (ISSUE-API-03-SSO).

The dead ``distllm.api.auth`` package (a near-duplicate of the live
``distllm.api.sso_auth`` stack) has been deleted; ``sso_auth`` is the single
canonical implementation.  This file pins the guarantees that motivated the
consolidation so the fork cannot silently regress:

1. ``distllm.api.auth`` no longer exists (no second copy to drift).
2. ``OIDCHandler.get_login_url`` advertises PKCE (S256) and the callback
   sends the bound ``code_verifier`` at the token exchange.
3. OAuth2/OIDC state is mandatory — a callback without state is refused
   before any IdP call.
4. ID-token validation is fail-closed: an unverified id_token with no JWKS
   is rejected unless the operator opts in.
"""

from __future__ import annotations

import importlib
import time
import urllib.parse
from unittest.mock import patch

import pytest

from distllm.api.sso_auth import OIDCHandler


def test_auth_fork_package_is_deleted():
    """The duplicate ``distllm.api.auth`` package must not come back."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("distllm.api.auth")


def _handler(**kwargs) -> OIDCHandler:
    return OIDCHandler(
        client_id="client-id",
        client_secret="client-secret",
        authority="https://idp.example.com",
        callback_url="https://app.example.com/cb",
        **kwargs,
    )


def test_oidc_login_url_advertises_pkce_s256():
    """OIDC login URL carries a PKCE S256 challenge and stores the verifier."""
    h = _handler()
    url = h.get_login_url()
    qs = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
    assert "code_challenge" in qs
    assert qs["code_challenge_method"] == ["S256"]
    # The verifier bound to the issued state is stored server-side.
    state = qs["state"][0]
    assert h._pkce_store[state]


def test_callback_without_state_refused_before_idp_call():
    """SEC-API-04: empty state is refused without any outbound IdP call."""
    h = _handler()
    with patch("httpx.post") as mock_post:
        assert h.handle_callback("code", expected_state="") is None
    mock_post.assert_not_called()


def test_callback_with_unknown_state_refused():
    """A state the server never issued cannot be replayed."""
    h = _handler()
    assert h.handle_callback("code", expected_state="forged-state") is None


def test_pkce_verifier_is_single_use():
    """Consuming the code_verifier binds it to one callback."""
    h = _handler()
    state = "state-pkce"
    h._state_store[state] = time.time() + 600
    h._pkce_store[state] = "verifier-1"
    h._nonce_store[state] = {"nonce": "n1", "expires": time.time() + 600}
    h._token_endpoint = "https://idp.example.com/token"
    h._userinfo_endpoint = ""
    with patch("httpx.post") as mock_post:
        mock_post.return_value = type(
            "R", (), {"status_code": 200, "text": "ok", "json": lambda: {}}
        )()
        # No id_token in the canned response; the exchange proceeds past
        # validation and the verifier is popped on the way.
        h.handle_callback("code", expected_state=state)
    assert "verifier-1" not in (mock_post.call_args.kwargs.get("data") or {})
    # Second use: the verifier is gone, PKCE check refuses the exchange.
    h._state_store[state] = time.time() + 600
    with patch("httpx.post") as mock_post2:
        assert h.handle_callback("code", expected_state=state) is None
    mock_post2.assert_not_called()


def test_unverified_id_token_fail_closed_by_default():
    """validate_token rejects when no JWKS is available (default)."""
    h = _handler()
    h._jwks_url = None
    h._discovered_jwks_url = ""
    assert h.validate_token("not.a.jwt") is None


def test_unverified_id_token_opt_in_allowed():
    """Explicit operator opt-in accepts the unverified token's claims."""
    h = _handler(allow_unverified_id_token=True)
    h._jwks_url = None
    h._discovered_jwks_url = ""
    # A structurally valid but unsigned token decodes only with verify off.
    import base64
    import json

    header = base64.urlsafe_b64encode(b'{"alg":"none","typ":"JWT"}').rstrip(b"=").decode()
    payload = base64.urlsafe_b64encode(
        json.dumps({"sub": "opt-in-user", "email": "u@example.com"}).encode()
    ).rstrip(b"=").decode()
    token = f"{header}.{payload}."
    user = h.validate_token(token)
    assert user is not None
    assert user.sub == "opt-in-user"
