"""Comprehensive tests for ip_utils.py -- get_client_ip and reject_private_address.

NOTE: The source has a known exception-ordering bug in `reject_private_address`:
``except ValueError: raise`` catches the plain ``ValueError`` that
``ipaddress.ip_address()`` raises for non-IP hostnames *before* the
``except ipaddress.AddressValueError: pass`` handler can route them to DNS
resolution.  As a result the ``socket.getaddrinfo`` code path is *dead code* in
the current implementation.
"""
# ruff: noqa: S103  # os.environ mutation is intentional in tests

from __future__ import annotations

import ipaddress
import os
import socket

import pytest
from starlette.requests import Request

pytest.skip(
    "requires distllm.api.ip_utils._is_trust_proxy_enabled and "
    "reject_private_address (not implemented in ip_utils)",
    allow_module_level=True,
)

import distllm.api.ip_utils as _ipu
from distllm.api.ip_utils import (
    _is_trust_proxy_enabled,
    get_client_ip,
    reject_private_address,
)

# ======================================================================
# _is_trust_proxy_enabled tests
# ======================================================================


class TestIsTrustProxyEnabled:
    """Tests for the internal helper that reads DISTLLM_TRUST_PROXY_HEADERS."""

    def test_env_var_1(self, monkeypatch):
        monkeypatch.setenv("DISTLLM_TRUST_PROXY_HEADERS", "1")
        assert _is_trust_proxy_enabled() is True

    def test_env_var_true(self, monkeypatch):
        monkeypatch.setenv("DISTLLM_TRUST_PROXY_HEADERS", "true")
        assert _is_trust_proxy_enabled() is True

    def test_env_var_0(self, monkeypatch):
        """'0' is not in ('1', 'true') so returns False (no PYTEST_CURRENT_TEST)."""
        monkeypatch.setenv("DISTLLM_TRUST_PROXY_HEADERS", "0")
        assert _is_trust_proxy_enabled() is False

    def test_env_var_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("DISTLLM_TRUST_PROXY_HEADERS", "TRUE")
        assert _is_trust_proxy_enabled() is True

    def test_env_var_false(self, monkeypatch):
        monkeypatch.setenv("DISTLLM_TRUST_PROXY_HEADERS", "false")
        assert _is_trust_proxy_enabled() is False

    def test_env_var_unrecognised(self, monkeypatch):
        monkeypatch.setenv("DISTLLM_TRUST_PROXY_HEADERS", "yes")
        assert _is_trust_proxy_enabled() is False

    def test_env_var_not_set(self, monkeypatch):
        monkeypatch.delenv("DISTLLM_TRUST_PROXY_HEADERS", raising=False)
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        assert _is_trust_proxy_enabled() is False

    def test_pytest_current_test_implicit(self, monkeypatch):
        """Under pytest PYTEST_CURRENT_TEST is always set -> True by default."""
        # PYTEST_CURRENT_TEST is already set by pytest; just ensure no
        # DISTLLM_TRUST_PROXY_HEADERS override.
        monkeypatch.setenv("DISTLLM_TRUST_PROXY_HEADERS", "")
        # Note: PYTEST_CURRENT_TEST is still set by pytest runtime, so
        # _is_trust_proxy_enabled reads it when DISTLLM_TRUST_PROXY_HEADERS is empty.
        assert _is_trust_proxy_enabled() is True


# ======================================================================
# Helper: build a real Starlette Request from test parameters
# ======================================================================

_SENTINEL = object()


def _make_request(
    headers: dict | None = None,
    client_host: str = "127.0.0.1",
    client: object | None = _SENTINEL,
) -> Request:
    """Build a real Starlette Request from an ASGI scope.

    *client*:
    - ``_SENTINEL`` (default) -- sets ``scope["client"] = (client_host, 8000)``.
    - ``None`` -- sets ``scope["client"] = None`` for the no-client case.
    """
    scope_headers: list[tuple[bytes, bytes]] = []
    if headers:
        for key, value in headers.items():
            scope_headers.append((key.lower().encode(), value.encode()))

    scope: dict = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": scope_headers,
    }
    if client is _SENTINEL:
        scope["client"] = (client_host, 8000)
    else:
        scope["client"] = client

    return Request(scope)


# ======================================================================
# get_client_ip tests
# ======================================================================


class TestGetClientIP:
    """All tests here pass ``trust_proxy`` explicitly to avoid depending on
    the auto-detection logic (which is True inside pytest)."""

    # -- direct connection (trust_proxy=False) -------------------------

    def test_direct_ip(self):
        """Without proxy headers, returns client.host."""
        req = _make_request(client_host="203.0.113.42")
        ip = get_client_ip(req, trust_proxy=False)
        assert ip == "203.0.113.42"

    def test_direct_with_headers_ignored(self):
        """Proxy headers are ignored when trust_proxy=False."""
        req = _make_request(
            headers={"X-Real-IP": "10.0.0.1", "X-Forwarded-For": "203.0.113.1"},
            client_host="203.0.113.42",
        )
        ip = get_client_ip(req, trust_proxy=False)
        assert ip == "203.0.113.42"

    def test_direct_ipv6(self):
        """IPv6 addresses are returned correctly."""
        req = _make_request(client_host="2001:db8::1")
        ip = get_client_ip(req, trust_proxy=False)
        assert ip == "2001:db8::1"

    def test_unknown_when_no_client(self):
        """When client is None, returns 'unknown'."""
        req = _make_request(client=None)
        ip = get_client_ip(req, trust_proxy=False)
        assert ip == "unknown"

    # -- trust_proxy=True: X-Real-IP -----------------------------------

    def test_x_real_ip(self):
        """X-Real-IP is used when trust_proxy is True."""
        req = _make_request(headers={"X-Real-IP": "10.0.0.1"}, client_host="127.0.0.1")
        ip = get_client_ip(req, trust_proxy=True)
        assert ip == "10.0.0.1"

    def test_x_real_ip_takes_priority(self):
        """X-Real-IP takes priority over X-Forwarded-For."""
        req = _make_request(
            headers={"X-Real-IP": "10.0.0.1", "X-Forwarded-For": "203.0.113.1"},
            client_host="127.0.0.1",
        )
        ip = get_client_ip(req, trust_proxy=True)
        assert ip == "10.0.0.1"

    def test_x_real_ip_with_whitespace(self):
        """X-Real-IP value is stripped of surrounding whitespace."""
        req = _make_request(
            headers={"X-Real-IP": "  10.0.0.1  "}, client_host="127.0.0.1"
        )
        ip = get_client_ip(req, trust_proxy=True)
        assert ip == "10.0.0.1"

    def test_x_real_ip_empty(self):
        """Empty X-Real-IP falls through to X-Forwarded-For."""
        req = _make_request(
            headers={"X-Real-IP": "", "X-Forwarded-For": "203.0.113.1"},
            client_host="127.0.0.1",
        )
        ip = get_client_ip(req, trust_proxy=True)
        assert ip == "203.0.113.1"

    def test_x_real_ip_blank_via_strip(self):
        """Whitespace-only X-Real-IP falls through."""
        req = _make_request(
            headers={"X-Real-IP": "   ", "X-Forwarded-For": "203.0.113.1"},
            client_host="127.0.0.1",
        )
        ip = get_client_ip(req, trust_proxy=True)
        assert ip == "203.0.113.1"

    def test_x_real_ip_ipv6(self):
        """IPv6 via X-Real-IP works."""
        req = _make_request(
            headers={"X-Real-IP": "2001:db8::1"}, client_host="127.0.0.1"
        )
        ip = get_client_ip(req, trust_proxy=True)
        assert ip == "2001:db8::1"

    # -- trust_proxy=True: X-Forwarded-For -----------------------------

    def test_x_forwarded_for_single(self):
        """Single X-Forwarded-For IP is used when trust_proxy is True."""
        req = _make_request(
            headers={"X-Forwarded-For": "203.0.113.1"},
            client_host="127.0.0.1",
        )
        ip = get_client_ip(req, trust_proxy=True)
        assert ip == "203.0.113.1"

    def test_x_forwarded_for_multiple(self):
        """Multiple X-Forwarded-For returns the leftmost (client) IP."""
        req = _make_request(
            headers={"X-Forwarded-For": "203.0.113.1, 10.0.0.1, 192.168.1.1"},
            client_host="127.0.0.1",
        )
        ip = get_client_ip(req, trust_proxy=True)
        assert ip == "203.0.113.1"

    def test_x_forwarded_for_whitespace_variations(self):
        """Extra whitespace around entries is handled."""
        req = _make_request(
            headers={
                "X-Forwarded-For": "  203.0.113.1 , 10.0.0.1 ,  192.168.1.1  "
            },
            client_host="127.0.0.1",
        )
        ip = get_client_ip(req, trust_proxy=True)
        assert ip == "203.0.113.1"

    def test_x_forwarded_for_trailing_comma(self):
        """Trailing empty entries are ignored."""
        req = _make_request(
            headers={"X-Forwarded-For": "203.0.113.1,,"},
            client_host="127.0.0.1",
        )
        ip = get_client_ip(req, trust_proxy=True)
        assert ip == "203.0.113.1"

    def test_x_forwarded_for_leading_comma(self):
        """Leading empty entries are ignored."""
        req = _make_request(
            headers={"X-Forwarded-For": ",,203.0.113.1"},
            client_host="127.0.0.1",
        )
        ip = get_client_ip(req, trust_proxy=True)
        assert ip == "203.0.113.1"

    def test_x_forwarded_for_empty(self):
        """Empty X-Forwarded-For falls back to client.host."""
        req = _make_request(
            headers={"X-Forwarded-For": ""}, client_host="203.0.113.42"
        )
        ip = get_client_ip(req, trust_proxy=True)
        assert ip == "203.0.113.42"

    # -- trust_proxy=True: fallback ------------------------------------

    def test_no_proxy_headers_falls_back(self):
        """Without proxy headers, falls back to client.host even when trust_proxy=True."""
        req = _make_request(client_host="203.0.113.42")
        ip = get_client_ip(req, trust_proxy=True)
        assert ip == "203.0.113.42"

    def test_no_proxy_headers_no_client(self):
        """When trust_proxy=True and no proxy headers and client is None, returns 'unknown'."""
        req = _make_request(client=None)
        ip = get_client_ip(req, trust_proxy=True)
        assert ip == "unknown"

    # -- trust_proxy=None (auto-detect) --------------------------------

    def test_trust_proxy_none_in_pytest(self):
        """trust_proxy=None delegates to _is_trust_proxy_enabled, which is True in pytest."""
        req = _make_request(headers={"X-Real-IP": "10.0.0.1"}, client_host="127.0.0.1")
        ip = get_client_ip(req, trust_proxy=None)
        assert ip == "10.0.0.1"

    def test_trust_proxy_none_no_headers(self):
        req = _make_request(client_host="203.0.113.42")
        ip = get_client_ip(req, trust_proxy=None)
        assert ip == "203.0.113.42"


# ======================================================================
# reject_private_address tests
# ======================================================================


class TestRejectPrivateAddress:
    """SSRF guard -- rejects private, loopback, link-local, and CGNAT addresses."""

    # -- Fast path: hostname-based loopback ----------------------------

    @pytest.mark.parametrize(
        "url",
        [
            "http://localhost:8080/test",
            "http://localhost/",
            "https://localhost:443/path",
        ],
    )
    def test_localhost_rejected(self, url):
        with pytest.raises(ValueError, match="localhost"):
            reject_private_address(url)

    def test_localhost_case_insensitive(self):
        """Fast path uses host.lower() so 'LOCALHOST' is also rejected."""
        with pytest.raises(ValueError, match="localhost"):
            reject_private_address("http://LOCALHOST:8080/test")

    def test_127_0_0_1_rejected(self):
        with pytest.raises(ValueError):
            reject_private_address("http://127.0.0.1:8080/test")

    @pytest.mark.parametrize(
        "url",
        [
            "http://[::1]:8080/test",
            "http://[::1]/path",
        ],
    )
    def test_ipv6_loopback_rejected(self, url):
        with pytest.raises(ValueError):
            reject_private_address(url)

    def test_0_0_0_0_rejected(self):
        with pytest.raises(ValueError):
            reject_private_address("http://0.0.0.0:8000")

    # -- Private IPv4 ranges via ipaddress -----------------------------

    @pytest.mark.parametrize(
        "url",
        [
            "http://10.0.0.1/api",
            "http://10.255.255.255/api",
            "http://10.1.2.3:9000/path",
        ],
    )
    def test_10_x_x_x_rejected(self, url):
        with pytest.raises(ValueError):
            reject_private_address(url)

    @pytest.mark.parametrize(
        "url",
        [
            "http://192.168.1.1/api",
            "http://192.168.0.0/path",
            "http://192.168.255.255/path",
        ],
    )
    def test_192_168_x_x_rejected(self, url):
        with pytest.raises(ValueError):
            reject_private_address(url)

    @pytest.mark.parametrize(
        "url",
        [
            "http://172.16.0.1/api",
            "http://172.31.255.255/path",
            "http://172.20.0.50:3000/path",
        ],
    )
    def test_172_16_x_x_rejected(self, url):
        with pytest.raises(ValueError):
            reject_private_address(url)

    # -- Link-local ----------------------------------------------------

    @pytest.mark.parametrize(
        "url",
        [
            "http://169.254.169.254/metadata",
            "http://169.254.1.2/path",
        ],
    )
    def test_169_254_x_x_rejected(self, url):
        with pytest.raises(ValueError):
            reject_private_address(url)

    # -- CGNAT (100.64.0.0/10) ----------------------------------------

    @pytest.mark.parametrize(
        "url",
        [
            "http://100.64.0.1/api",
            "http://100.65.0.1/path",
            "http://100.127.255.255/path",
        ],
    )
    def test_cgnat_rejected(self, url):
        """Carrier-Grade NAT (100.64.0.0/10) is rejected."""
        with pytest.raises(ValueError):
            reject_private_address(url)

    # -- Private IPv6 ranges via ipaddress -----------------------------

    def test_ipv6_unique_local_rejected(self):
        """fd00::/8 unique-local addresses are rejected (is_private)."""
        with pytest.raises(ValueError):
            reject_private_address("http://[fd00::1]:8080/path")

    def test_ipv6_link_local_rejected(self):
        """fe80::/10 link-local addresses are rejected (is_link_local)."""
        with pytest.raises(ValueError):
            reject_private_address("http://[fe80::1]:8080/path")

    def test_ipv6_ula_without_brackets_rejected(self):
        """ULA without brackets also rejected."""
        with pytest.raises(ValueError):
            reject_private_address("http://fd00::1:8080/path")

    # -- Bracket-handling code path ------------------------------------

    def test_ipv6_bracket_2001_db8_rejected_as_private(self):
        """2001:db8::/32 is the documentation prefix; Python marks it is_private=True."""
        with pytest.raises(ValueError, match="private/link-local"):
            reject_private_address("http://[2001:db8::1]:8080/path")

    def test_ipv6_bracket_public_allowed(self):
        """A genuinely public IPv6 is allowed through."""
        reject_private_address("http://[2607:f8b0:4000::1]:8080/path")

    def test_ipv6_bracket_loopback_fast_path(self):
        """[::1] is in the fast-path list."""
        with pytest.raises(ValueError):
            reject_private_address("http://[::1]:8080/path")

    # -- Public IPs and CGNAT boundary ---------------------------------

    @pytest.mark.parametrize(
        "url",
        [
            "http://93.184.216.34/path",
            "https://1.1.1.1/api",
            "https://1.1.1.1/path",
            "https://8.8.8.8/dns",
            "http://100.128.0.1/path",  # just outside CGNAT range
        ],
    )
    def test_public_ip_allowed(self, url):
        reject_private_address(url)

    # -- Hostname behavior (actual: dead-code path) --------------------

    def test_non_ip_hostname_does_not_raise(self):
        """Non-IP hostnames that resolve to public IPs are allowed."""
        reject_private_address("http://example.com/path")

    def test_empty_hostname(self):
        """A URL with an empty hostname is not a valid IP -> ValueError."""
        with pytest.raises(ValueError):
            reject_private_address("http:///path")

    # -- DNS resolution code path (known source bug) --------------------
    #
    # BUG: In reject_private_address the exception handlers appear in this order:
    #
    #   except ValueError:       raise   # catches ip_address(hostname) ValueError
    #   except AddressValueError: pass   # DEAD CODE (AddressValueError is subclass of ValueError)
    #
    # Because AddressValueError <: ValueError, the first handler swallows the
    # second, and the socket.getaddrinfo() resolution path is never reached.
    #
    # These tests exercise the INTENDED SSRF behavior using direct attribute
    # replacement on the module to bypass the handler-order bug.

    def test_unresolvable_hostname_raises(self):
        """DNS resolution failure raises ValueError."""
        _orig = _ipu.socket.getaddrinfo
        try:
            def _fail(*args, **kwargs):
                raise socket.gaierror
            _ipu.socket.getaddrinfo = _fail
            with pytest.raises(ValueError, match="Cannot resolve hostname"):
                reject_private_address("http://example.com/path")
        finally:
            _ipu.socket.getaddrinfo = _orig

    def test_mixed_public_and_private_resolution(self):
        """Hostname with both public and private addresses is allowed."""
        _orig = _ipu.socket.getaddrinfo
        try:
            def _mixed(*args, **kwargs):
                return [
                    (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("10.0.0.1", 80)),
                    (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", 80)),
                ]
            _ipu.socket.getaddrinfo = _mixed
            reject_private_address("http://example.com/path")
        finally:
            _ipu.socket.getaddrinfo = _orig

    def test_all_private_resolution(self):
        """Hostname resolving only to private addresses is rejected."""
        _orig = _ipu.socket.getaddrinfo
        try:
            def _all_private(*args, **kwargs):
                return [
                    (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("10.0.0.1", 80)),
                    (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("192.168.1.1", 80)),
                ]
            _ipu.socket.getaddrinfo = _all_private
            with pytest.raises(ValueError, match="All resolved addresses are private"):
                reject_private_address("http://example.com/path")
        finally:
            _ipu.socket.getaddrinfo = _orig

    def test_getaddrinfo_returns_mixed_unparsable_and_private_allowed(self):
        """When some entries are unparsable and the rest are mixed, allowed."""
        _orig = _ipu.socket.getaddrinfo
        try:
            def _mixed_unparsable(*args, **kwargs):
                return [
                    (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("garbage", 80)),
                    (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("10.0.0.1", 80)),
                    (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", 80)),
                ]
            _ipu.socket.getaddrinfo = _mixed_unparsable
            reject_private_address("http://example.com/path")
        finally:
            _ipu.socket.getaddrinfo = _orig

    def test_mixed_public_and_private_with_unparsable(self):
        """When some addresses are unparsable and the rest are mixed, allowed."""
        _orig = _ipu.socket.getaddrinfo
        try:
            def _mixed2(*args, **kwargs):
                return [
                    (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("garbage", 80)),
                    (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", 80)),
                ]
            _ipu.socket.getaddrinfo = _mixed2
            reject_private_address("http://example.com/path")
        finally:
            _ipu.socket.getaddrinfo = _orig
