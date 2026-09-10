"""Multi-node E2E test fixtures.

Provides:
- A ``LocalCluster`` fixture that runs the coordinator in-process with the
  full TinyStories-1M model on CPU (no subprocesses).
- A FastAPI TestClient backed by that coordinator.

This tests the full stack (tokenizer, coordinator, API, middleware) without
the overhead of subprocess gRPC worker nodes.  The coordinator runs its
internal ``_generate_local_sync`` path.

Auth note: the DISABLE_AUTH bypass was removed in the security hardening
(auth is always required). Tests use a real per-run API key + Bearer header.
"""
import gc
import os
import secrets as _secrets
import time
from pathlib import Path

import pytest
import torch

# Real auth (the old DISABLE_AUTH bypass no longer exists): a fixed test
# key set BEFORE any distllm import so the key store singleton loads it.
TEST_API_KEY = _secrets.token_hex(32)
os.environ["API_KEY"] = TEST_API_KEY
os.environ["API_KEY_WAS_SET"] = "1"
os.environ.pop("API_KEYS", None)
os.environ.pop("API_KEYS_FILE", None)
os.environ.pop("DISABLE_AUTH", None)
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
            # CoordinatorConfig.port enforces ge=1 (0 is an invalid gRPC
            # port); the server is never .start()ed in these tests, so any
            # valid unbound port number is safe.
            port=50050,
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
        """Return an authenticated FastAPI TestClient.

        The coordinator is injected into the shared api_state (the server
        reads it from ``g``, not a module attribute) and restored by
        :meth:`restore_coordinator` after the test.
        """
        from distllm.api.api_state import g as api_state
        self._saved = api_state.coordinator
        api_state.coordinator = self.coordinator
        from fastapi.testclient import TestClient
        from distllm.api.server import app
        client = TestClient(app, raise_server_exceptions=False)
        client.headers["Authorization"] = f"Bearer {TEST_API_KEY}"
        return client

    def restore_coordinator(self):
        from distllm.api.api_state import g as api_state
        if self._saved is not None:
            api_state.coordinator = self._saved
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
