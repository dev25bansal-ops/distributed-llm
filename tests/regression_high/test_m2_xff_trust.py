"""Regression test for HIGH fix M2: fail-closed X-Forwarded-For IP trust.

``get_client_ip`` previously trusted ``X-Forwarded-For`` / ``X-Real-IP``
whenever proxy trust was enabled (including implicitly under pytest), without
checking whether the request actually arrived from a trusted proxy. A client
reaching the app directly could spoof ``X-Forwarded-For`` to impersonate any
IP and bypass per-IP rate limits / IP-based trust.

The fix is fail-closed: forwarded headers are honored ONLY when the immediate
peer (``request.client.host``) is in the configured trusted-proxy set
(``DISTLLM_TRUSTED_PROXIES``, default loopback). Otherwise the peer address is
returned and headers are ignored. For a trusted proxy, the rightmost
non-proxy entry of ``X-Forwarded-For`` is used.

``ip_utils.py`` is loaded via importlib to avoid importing the heavy
``distllm.api`` server stack (pattern from test_c15_ratelimit_keying.py).
"""

from __future__ import annotations

import importlib.util
import types

import pytest

_REPO_SRC = __import__("pathlib").Path(__file__).resolve().parents[2] / "src"


def _load_ip_utils():
    path = _REPO_SRC / "distllm" / "api" / "ip_utils.py"
    spec = importlib.util.spec_from_file_location("distllm.api.ip_utils", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ip_utils = _load_ip_utils()
get_client_ip = _ip_utils.get_client_ip


def _make_request(peer: str | None, xff: str | None = None, xrip: str | None = None):
    headers: dict[str, str] = {}
    if xff is not None:
        headers["X-Forwarded-For"] = xff
    if xrip is not None:
        headers["X-Real-IP"] = xrip
    # starlette Headers-like get(): case-insensitive lookup is not required here
    # because the code queries the exact header names it sets.
    return types.SimpleNamespace(
        headers=_CaseInsensitiveHeaders(headers),
        client=types.SimpleNamespace(host=peer) if peer is not None else None,
    )


class _CaseInsensitiveHeaders:
    def __init__(self, data):
        self._data = {k.lower(): v for k, v in data.items()}

    def get(self, key, default=""):
        return self._data.get(key.lower(), default)


def test_untrusted_peer_ignores_forwarded_header(monkeypatch):
    """Peer 8.8.8.8 is NOT a trusted proxy: XFF must be ignored."""
    monkeypatch.delenv("DISTLLM_TRUSTED_PROXIES", raising=False)
    req = _make_request(peer="8.8.8.8", xff="1.2.3.4")
    assert get_client_ip(req) == "8.8.8.8"


def test_untrusted_peer_ignores_real_ip(monkeypatch):
    monkeypatch.delenv("DISTLLM_TRUSTED_PROXIES", raising=False)
    req = _make_request(peer="8.8.8.8", xrip="1.2.3.4")
    assert get_client_ip(req) == "8.8.8.8"


def test_trusted_loopback_parses_rightmost_nonproxy(monkeypatch):
    """Peer 127.0.0.1 is trusted by default: take rightmost non-proxy XFF."""
    monkeypatch.delenv("DISTLLM_TRUSTED_PROXIES", raising=False)
    req = _make_request(peer="127.0.0.1", xff="9.9.9.9, 127.0.0.1")
    assert get_client_ip(req) == "9.9.9.9"


def test_trusted_proxy_single_entry(monkeypatch):
    monkeypatch.delenv("DISTLLM_TRUSTED_PROXIES", raising=False)
    req = _make_request(peer="127.0.0.1", xff="9.9.9.9")
    assert get_client_ip(req) == "9.9.9.9"


def test_env_override_adds_trusted_proxy(monkeypatch):
    """Custom proxy IP in DISTLLM_TRUSTED_PROXIES enables header parsing."""
    monkeypatch.setenv("DISTLLM_TRUSTED_PROXIES", "10.0.0.5")
    req = _make_request(peer="10.0.0.5", xff="9.9.9.9, 10.0.0.5")
    assert get_client_ip(req) == "9.9.9.9"


def test_env_override_untrusts_loopback(monkeypatch):
    """Setting the list without loopback stops trusting 127.0.0.1."""
    monkeypatch.setenv("DISTLLM_TRUSTED_PROXIES", "10.0.0.5")
    req = _make_request(peer="127.0.0.1", xff="9.9.9.9")
    assert get_client_ip(req) == "127.0.0.1"


def test_empty_env_never_trusts_headers(monkeypatch):
    """Empty DISTLLM_TRUSTED_PROXIES means never trust forwarded headers."""
    monkeypatch.setenv("DISTLLM_TRUSTED_PROXIES", "")
    req = _make_request(peer="127.0.0.1", xff="9.9.9.9")
    assert get_client_ip(req) == "127.0.0.1"


def test_explicit_trust_proxy_true(monkeypatch):
    monkeypatch.delenv("DISTLLM_TRUSTED_PROXIES", raising=False)
    req = _make_request(peer="8.8.8.8", xff="9.9.9.9, 8.8.8.8")
    # Force-trust: rightmost non-proxy (8.8.8.8 is the peer, not in default set,
    # so it is treated as non-proxy and returned as rightmost) -> 8.8.8.8.
    # Use a chain where the rightmost is a default-trusted proxy to show parsing.
    req2 = _make_request(peer="8.8.8.8", xff="9.9.9.9, 127.0.0.1")
    assert get_client_ip(req2, trust_proxy=True) == "9.9.9.9"


def test_explicit_trust_proxy_false_ignores_headers(monkeypatch):
    monkeypatch.delenv("DISTLLM_TRUSTED_PROXIES", raising=False)
    req = _make_request(peer="127.0.0.1", xff="9.9.9.9")
    assert get_client_ip(req, trust_proxy=False) == "127.0.0.1"


def test_no_client_returns_unknown(monkeypatch):
    monkeypatch.delenv("DISTLLM_TRUSTED_PROXIES", raising=False)
    req = _make_request(peer=None, xff="1.2.3.4")
    assert get_client_ip(req) == "unknown"
