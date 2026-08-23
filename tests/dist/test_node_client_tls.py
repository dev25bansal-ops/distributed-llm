"""Regression tests: pipeline-parallel gRPC must be able to use TLS.

P0 finding: ``forward_request``/``forward_request_async`` created their node
clients without passing ``use_tls``, so activations/KV caches (which encode
prompt content) travelled over plaintext gRPC even when TLS was available.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from distllm.dist import node_client
from distllm.dist.pipeline.orchestrator import PipelineOrchestrator


class TestForwardRequestTLSPassthrough:
    def test_forward_request_passes_use_tls(self, monkeypatch):
        captured: dict = {}
        client = MagicMock()
        client.stub.ForwardPass.with_call.side_effect = RuntimeError("stop-here")

        def fake_create(host, port, **kwargs):
            captured.update(kwargs)
            return client

        monkeypatch.setattr(node_client, "create_node_client", fake_create)
        with pytest.raises(RuntimeError):
            node_client.forward_request(
                "host", 1, torch.zeros(2), use_tls=True, ca_cert="/tmp/ca.pem"
            )
        assert captured.get("use_tls") is True
        assert captured.get("ca_cert") == "/tmp/ca.pem"

    def test_forward_request_defaults_to_insecure_false(self, monkeypatch):
        captured: dict = {}
        client = MagicMock()
        client.stub.ForwardPass.with_call.side_effect = RuntimeError("stop-here")

        def fake_create(host, port, **kwargs):
            captured.update(kwargs)
            return client

        monkeypatch.setattr(node_client, "create_node_client", fake_create)
        with pytest.raises(RuntimeError):
            node_client.forward_request("host", 1, torch.zeros(2))
        assert captured.get("use_tls") is False


class TestOrchestratorTLSConfig:
    def test_explicit_tls_params(self):
        orch = PipelineOrchestrator(use_tls=True, ca_cert="/tmp/cluster-ca.pem")
        assert orch._use_tls is True
        assert orch._ca_cert == "/tmp/cluster-ca.pem"

    def test_env_toggle_enables_tls(self, monkeypatch):
        monkeypatch.setenv("DISTLLM_PIPELINE_TLS", "1")
        orch = PipelineOrchestrator()
        assert orch._use_tls is True

    def test_default_is_plaintext(self):
        orch = PipelineOrchestrator()
        assert orch._use_tls is False

    def test_orchestrator_threads_tls_into_forward(self, monkeypatch):
        captured: dict = {}

        def fake_forward(*args, **kwargs):
            captured.update(kwargs)
            return torch.zeros(2)

        monkeypatch.setattr(node_client, "forward_request", fake_forward)
        orch = PipelineOrchestrator(use_tls=True, ca_cert="/tmp/ca.pem")
        orch.register_node("n1", "host", 1, 0, 2, total_layers=2)
        orch._node_order = ["n1"]
        orch.run_pipeline(torch.zeros(2), {}, request_id="r1")
        assert captured.get("use_tls") is True
        assert captured.get("ca_cert") == "/tmp/ca.pem"
