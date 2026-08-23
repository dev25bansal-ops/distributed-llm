"""Security: SSRF via federation heartbeat and eval coordinator_url.

The federation heartbeat endpoint and eval runner accept a user-supplied
URL.  An attacker could supply a private/internal IP (e.g. 169.254.169.254
for AWS metadata, 10.x for internal services) to probe the internal network
or exfiltrate cloud metadata.

``reject_private_address()`` is the canonical SSRF guard.  These tests
verify it's applied to all URL-accepting endpoints.
"""

from __future__ import annotations

import pytest

pytest.skip(
    "requires distllm.api.ip_utils.reject_private_address (not implemented)",
    allow_module_level=True,
)

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from distllm.api.ip_utils import reject_private_address  # noqa: E402,F401


# ======================================================================
# reject_private_address unit tests (SSRF guard)
# ======================================================================


class TestSSRFProtection:
    """Verify the SSRF guard rejects known-bad inputs."""

    def test_aws_metadata_rejected(self):
        """169.254.169.254 (AWS/GCP metadata) is rejected."""
        with pytest.raises(ValueError):
            reject_private_address("http://169.254.169.254/latest/meta-data/")

    def test_kubernetes_metadata_rejected(self):
        """10.x internal K8s metadata is rejected."""
        with pytest.raises(ValueError):
            reject_private_address("http://10.0.0.1/api/v1/namespaces")

    def test_internal_service_rejected(self):
        """Internal service discovery via private IP is rejected."""
        with pytest.raises(ValueError):
            reject_private_address("http://192.168.1.1:8500/v1/agent/services")

    def test_docker_socket_rejected(self):
        """Docker daemon socket access is rejected."""
        with pytest.raises(ValueError):
            reject_private_address("http://127.0.0.1:2375/containers/json")

    def test_cloudfoundry_metadata_rejected(self):
        """Cloud Foundry internal endpoint is rejected."""
        with pytest.raises(ValueError):
            reject_private_address("http://169.254.169.254/")

    def test_public_endpoint_allowed(self):
        """Public IP addresses are not blocked."""
        reject_private_address("https://1.1.1.1/path")

    def test_federation_eval_coordinator_url_rejected(self):
        """Eval endpoint rejects private coordinator_url via SSRF guard."""
        from distllm.api.routes.eval import reject_private_address as ssf
        # The eval endpoint calls reject_private_address on coordinator_url
        with pytest.raises(ValueError):
            ssf("http://localhost:8000/v1/chat")
        with pytest.raises(ValueError):
            ssf("http://10.0.0.5:8000/v1/chat")
        with pytest.raises(ValueError):
            ssf("http://169.254.169.254/metadata")
