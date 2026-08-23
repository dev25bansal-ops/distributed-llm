"""Regression tests for HIGH fix H4: SSRF to cloud metadata (169.254.169.254).

H4 protects outbound HTTP fetches from Server-Side Request Forgery that would
expose the cloud instance metadata service (``http://169.254.169.254/...`` on
AWS/GCP/Azure) or other internal/link-local addresses (``127.0.0.1``,
``localhost``, ``10.x``, Docker socket, etc.).

The real guard lives in :mod:`distllm.security.utils`:
* :func:`validate_http_url` -- rejects any URL whose hostname resolves to a
  private / loopback / link-local / multicast / reserved / unspecified address
  (unless ``allow_private_hosts=True`` is explicitly passed).
* :func:`safe_urlopen` -- the same range check *plus* DNS-rebinding protection
  (resolve once, connect to the pinned IP).

This test reuses the REAL functions directly (loaded via importlib so we don't
pull the heavy server stack).  To make the tests deterministic and offline we
monkeypatch ``socket.getaddrinfo`` to return a forced IP for the internal
hostnames; that only controls *what IP the guard sees*, the rejection decision
is made by the untouched production code.  Public hosts resolve normally and
are accepted -- proving the guard does not over-block.

NOTE: we do NOT reimplement the guard.  If ``distllm.security.utils`` ever
loses these checks, these tests fail loudly.
"""

from __future__ import annotations

import importlib.util
import ipaddress
import os
import socket
import sys
import types

import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_UTILS_PATH = os.path.join(_REPO_ROOT, "src", "distllm", "security", "utils.py")


def _load_security_utils():
    spec = importlib.util.spec_from_file_location("distllm.security.utils", _UTILS_PATH)
    module = importlib.util.module_from_spec(spec)
    # Register lightweight parent packages so relative imports (if any) resolve.
    if "distllm" not in sys.modules:
        pmod = types.ModuleType("distllm")
        pmod.__path__ = [os.path.join(_REPO_ROOT, "src", "distllm")]
        sys.modules["distllm"] = pmod
    if "distllm.security" not in sys.modules:
        smod = types.ModuleType("distllm.security")
        smod.__path__ = [os.path.join(_REPO_ROOT, "src", "distllm", "security")]
        sys.modules["distllm.security"] = smod
    spec.loader.exec_module(module)
    return module


_security_utils = _load_security_utils()
validate_http_url = _security_utils.validate_http_url


# Internal / link-local hostnames -> the IP the DNS layer would hand back.
_FORCED_RESOLUTIONS = {
    "169.254.169.254": "169.254.169.254",
    "metadata.google.internal": "169.254.169.254",
    "metadata.azure.internal": "169.254.169.254",
    "127.0.0.1": "127.0.0.1",
    "localhost": "127.0.0.1",
    "0.0.0.0": "0.0.0.0",
    "10.0.0.5": "10.0.0.5",
    "10.255.255.255": "10.255.255.255",
    "172.16.0.1": "172.16.0.1",
    "192.168.1.1": "192.168.1.1",
    "::1": "::1",
}


@pytest.fixture(autouse=True)
def _patch_dns(monkeypatch):
    """Force internal hostnames to resolve to internal IPs; leave public alone."""
    orig = socket.getaddrinfo

    def _fake_getaddrinfo(host, port, *args, **kwargs):
        if host in _FORCED_RESOLUTIONS:
            ip = _FORCED_RESOLUTIONS[host]
            if ":" in ip:
                return [(socket.AF_INET6, socket.SOCK_STREAM, 6, "", (ip, port or 80))]
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port or 80))]
        return orig(host, port, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo)
    yield


# -- Internal / link-local targets that MUST be blocked ----------------------

_INTERNAL_TARGETS = [
    "http://169.254.169.254/latest/meta-data/iam/security-credentials/",
    "http://169.254.169.254/latest/meta-data/",
    "http://metadata.google.internal/computeMetadata/v1/",
    "http://metadata.azure.internal/machine/",
    "http://127.0.0.1:2375/containers/json",
    "http://localhost:6379/",
    "http://0.0.0.0:8080/admin",
    "http://10.0.0.5/internal-api",
    "http://172.16.0.1:9000/",
    "http://192.168.1.1/router.cgi",
]


@pytest.mark.parametrize("url", _INTERNAL_TARGETS)
def test_internal_targets_blocked(url):
    """H4: every internal/link-local target must raise ValueError (rejected)."""
    with pytest.raises(ValueError):
        validate_http_url(url)


@pytest.mark.parametrize("ip", ["169.254.169.254", "127.0.0.1", "10.0.0.5", "::1"])
def test_link_local_ip_literals_blocked(ip):
    """H4: link-local / loopback / private IP literals in the URL are blocked."""
    with pytest.raises(ValueError):
        validate_http_url(f"http://{ip}/secret")


def test_metadata_endpoint_explicitly_blocked():
    """H4: the cloud metadata IP (169.254.169.254) is blocked even as a bare path."""
    with pytest.raises(ValueError):
        validate_http_url("https://169.254.169.254/latest/meta-data/iam/")


# -- Public targets that MUST still be allowed -------------------------------


def test_public_target_allowed():
    """H4: a genuine public URL must NOT be blocked (no over-blocking)."""
    # example.com resolves via the real DNS (not forced), which is a public IP.
    result = validate_http_url("https://example.com/v1/models")
    assert result.endswith("/v1/models")


def test_allow_private_hosts_opt_out_bypasses_check():
    """H4: the explicit ``allow_private_hosts=True`` operator escape hatch works."""
    out = validate_http_url("http://169.254.169.254/meta", allow_private_hosts=True)
    assert out == "http://169.254.169.254/meta"


def test_non_http_scheme_rejected():
    """H4: the guard also rejects non-HTTP(S) schemes (file://, gopher://, etc.)."""
    with pytest.raises(ValueError):
        validate_http_url("file:///etc/passwd")
    with pytest.raises(ValueError):
        validate_http_url("gopher://169.254.169.254:80/")


def test_ip_utils_resolves_internal_as_blocked():
    """H4 (defence-in-depth): confirm the address class the guard relies on."""
    for ip_str in ("169.254.169.254", "127.0.0.1", "10.0.0.5"):
        ip = ipaddress.ip_address(ip_str)
        assert (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved)
