"""Multi-node E2E test fixtures.

Provides:
- A ``LocalCluster`` fixture that runs the coordinator in-process with the
  full TinyStories-1M model on CPU (no subprocesses).
- A FastAPI TestClient backed by that coordinator.

This tests the full stack (tokenizer, coordinator, API, middleware) without
the overhead of subprocess gRPC worker nodes.  The coordinator runs its
internal ``_generate_local_sync`` path.
"""
import gc
import os
import time
from pathlib import Path

import pytest
import torch

# Bypass API key auth for development tests.
os.environ.pop("API_KEY", None)  # middleware generates a random key
os.environ["DISABLE_AUTH"] = "1"
os.environ["DISTLLM_DEV_MODE"] = "1"

TINYSTORIES = "roneneldan/TinyStories-1M"
MODEL_LOAD_TIMEOUT = 300  # seconds for first-time HF download + CPU materialize


class LocalCluster:
    """In-process coordinator backed by a real model loaded on CPU.

    Usage::

        cluster = LocalCluster()
        cluster.start()
        cluster.coordinator.generate("Hello", max_new_tokens=10)
        cluster.make_api_client().get("/health")
        cluster.stop()
    """

    def __init__(self, model_name: str = TINYSTORIES):
        self.model_name = model_name
        self.coordinator = None
        self._saved = None

    def start(self):
        """Load the model and create the coordinator."""
        from distllm.core.coordinator import Coordinator

        self.coordinator = Coordinator(
            model_name=self.model_name,
            dtype="float32",
            port=0,  # dummy port, won't start gRPC server
        )
        self.coordinator.load_local_model()
        # Override the tokenizer decode for consistent test output
        if self.coordinator.tokenizer is None:
            from transformers import AutoTokenizer
            self.coordinator.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=False,
            )

    def stop(self):
        if self.coordinator is not None:
            try:
                self.coordinator.stop()
            except Exception:
                pass
            self.coordinator = None
        gc.collect()

    def make_api_client(self):
        """Return a FastAPI TestClient with coordinator injected.

        ``restore_coordinator()`` should be called after the test.
        """
        import distllm.api.server as server_module
        self._saved = server_module.coordinator
        server_module.coordinator = self.coordinator
        from fastapi.testclient import TestClient
        from distllm.api.server import app
        return TestClient(app, raise_server_exceptions=False)

    def restore_coordinator(self):
        import distllm.api.server as server_module
        if self._saved is not None:
            server_module.coordinator = self._saved
            self._saved = None


# ------------------------------------------------------------------
# Pytest fixtures
# ------------------------------------------------------------------


@pytest.fixture(scope="module")
def cluster():
    """Module-scoped local cluster (one model load per module)."""
    c = LocalCluster()
    try:
        c.start()
        yield c
    finally:
        c.stop()


@pytest.fixture
def api_client(cluster):
    """Per-test FastAPI TestClient backed by the live coordinator."""
    client = cluster.make_api_client()
    yield client
    cluster.restore_coordinator()
