"""Regression test for L6 SSRF guard on outbound webhook URLs."""

import os

from distllm.core.webhook_manager import _is_safe_webhook_url


def test_ssrf_blocks_internal_urls():
    # Cloud metadata endpoint (AWS/GCP) — classic SSRF target.
    assert _is_safe_webhook_url("http://169.254.169.254/latest/meta-data/") is False
    # Loopback / private.
    assert _is_safe_webhook_url("http://localhost:8080/hook") is False
    assert _is_safe_webhook_url("http://127.0.0.1/hook") is False
    assert _is_safe_webhook_url("https://10.0.0.5/hook") is False
    # Non-http(s) scheme.
    assert _is_safe_webhook_url("file:///etc/passwd") is False
    assert _is_safe_webhook_url("gopher://127.0.0.1:6379") is False


def test_ssrf_allows_public_https():
    assert _is_safe_webhook_url("https://hooks.example.com/webhook") is True
    assert _is_safe_webhook_url("https://api.my-service.io/v1/events") is True


def test_ssrf_allowlist_restricts_public():
    os.environ["DISTLLM_WEBHOOK_ALLOWLIST"] = "trusted.example.com"
    try:
        assert _is_safe_webhook_url("https://trusted.example.com/hook") is True
        assert _is_safe_webhook_url("https://other.example.com/hook") is False
    finally:
        os.environ.pop("DISTLLM_WEBHOOK_ALLOWLIST", None)
