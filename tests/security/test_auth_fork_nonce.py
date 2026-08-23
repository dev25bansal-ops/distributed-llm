"""Regression tests: the unwired `api/auth/` SSO fork's security bugs.

The audit found the (dead, unwired) `distllm/api/auth/` OIDC fork stored only an
expiry timestamp in its nonce store (so OIDC nonce replay protection was dead)
and that `auth/__init__.py` called `hashlib.sha256` without importing hashlib
(NameError in validate_token).  Both are fixed so the fork is safe if ever wired.
"""

from __future__ import annotations

import urllib.parse

from distllm.api.auth.oidc import OIDCHandler


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
    assert h._nonce_store.get(state) == nonce


def test_oidc_wrong_nonce_rejected():
    """P0: a callback with a mismatched nonce is rejected (replay protection)."""
    h = _handler()
    state, _ = _issue(h)
    assert h.handle_callback("code", expected_state=state, expected_nonce="wrong-nonce") is None


def test_oidc_nonce_is_single_use():
    """A rejected/consumed nonce cannot be replayed."""
    h = _handler()
    state, _ = _issue(h)
    # The failed callback still consumes the nonce (single-use).
    assert h.handle_callback("code", expected_state=state, expected_nonce="wrong-nonce") is None
    assert h._nonce_store.get(state) is None


def test_auth_init_validate_token_imports_hashlib():
    """P0: validate_token must not NameError on hashlib (previously unimported)."""
    from distllm.api.auth import SSOAuthHandler

    handler = SSOAuthHandler()
    # With no OIDC provider configured, validation falls through to None
    # without raising NameError.
    assert handler.validate_token("some-access-token") is None
