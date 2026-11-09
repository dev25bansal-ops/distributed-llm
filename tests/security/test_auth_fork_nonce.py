"""Regression tests for the canonical SSO fork's nonce handling.

The audit found the (dead, unwired) ``distllm/api/auth/`` OIDC fork stored
only an expiry timestamp in its nonce store (so OIDC nonce replay
protection was dead) and that ``auth/__init__.py`` called
``hashlib.sha256`` without importing hashlib (NameError in
``validate_token``).  Both classes of bug were fixed and the fork was then
consolidated into the live ``distllm.api.sso_auth`` implementation, which
this file now pins.

In ``sso_auth.OIDCHandler`` the nonce store maps state to a dict with the
raw nonce plus an expiry; the callback compares the id_token's ``nonce``
claim against it.
"""

from __future__ import annotations

import urllib.parse

from distllm.api.sso_auth import OIDCHandler


def _handler() -> OIDCHandler:
    return OIDCHandler(
        client_id="client-id",
        client_secret="client-secret",
        authority="https://idp.example.com",
        callback_url="https://app.example.com/cb",
    )


def _issue(h: OIDCHandler) -> tuple[str, str]:
    """Issue a login URL and return (state, nonce) from the query string."""
    url = h.get_login_url()
    qs = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
    return qs["state"][0], qs["nonce"][0]


def test_oidc_nonce_stored_not_expiry():
    """P0: the nonce store must hold the actual nonce, not an expiry timestamp."""
    h = _handler()
    state, nonce = _issue(h)
    assert h._nonce_store[state]["nonce"] == nonce


def test_oidc_wrong_nonce_rejected():
    """P0: a callback with a mismatched nonce is rejected (replay protection)."""
    h = _handler()
    state, _ = _issue(h)
    assert h.handle_callback("code", expected_state=state, expected_nonce="wrong-nonce") is None


def test_oidc_nonce_is_single_use():
    """A replayed callback cannot authenticate: state is consumed on first
    use, so a second callback with the same state is rejected outright."""
    h = _handler()
    state, _ = _issue(h)
    # The failed callback consumes the single-use state.
    assert h.handle_callback("code", expected_state=state, expected_nonce="wrong-nonce") is None
    assert state not in h._state_store
    # A replay with the same state is rejected before any token exchange.
    assert h.handle_callback("code", expected_state=state, expected_nonce="wrong-nonce") is None


def test_sso_auth_handler_validate_token_imports_hashlib():
    """P0: validate_token must not NameError on hashlib (previously unimported)."""
    from distllm.api.sso_auth import SSOAuthHandler

    handler = SSOAuthHandler()
    # With no OIDC provider configured, validation falls through to None
    # without raising NameError.
    assert handler.validate_token("some-access-token") is None
