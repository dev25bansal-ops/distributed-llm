"""Batch processing tests: POST /v1/batch (sync) and /v1/batch/submit (background)."""

import asyncio
import json
import os
import secrets
import time
from unittest.mock import MagicMock, patch

import pytest
import torch
from fastapi.testclient import TestClient

from distllm.api.api_state import g
from distllm.core.api_key_store import reset_api_key_store
from distllm.api.server import app


def _make_client():
    test_api_key = secrets.token_urlsafe(32)
    os.environ["API_KEY"] = test_api_key
    reset_api_key_store()
    client = TestClient(app)
    client.headers["Authorization"] = f"Bearer {test_api_key}"
    return client


@pytest.fixture(autouse=True)
def _setup_auth(monkeypatch):
    test_api_key = secrets.token_urlsafe(32)
    monkeypatch.setenv("API_KEY", test_api_key)
    monkeypatch.delenv("API_KEY_WAS_SET", raising=False)
    reset_api_key_store()


@pytest.fixture
def coordinator():
    coord = MagicMock()
    coord.model_name = "test-model"
    coord.nodes = {}
    coord._shutting_down = False
    coord.local_partitioner = MagicMock()

    mock_model = MagicMock()
    mock_model.parameters.side_effect = lambda: iter([torch.randn(10, 10)])
    mock_output = MagicMock()
    mock_output.logits = torch.randn(1, 1, 1000)
    mock_output.past_key_values = None
    mock_model.return_value = mock_output
    coord.local_partitioner.full_model = mock_model

    coord.tokenizer = MagicMock()
    coord.tokenizer.encode.return_value = torch.tensor([[1, 2, 3]])
    coord.tokenizer.decode.return_value = "tok-1 tok-2 tok-3"
    coord.tokenizer.eos_token_id = 0
    coord.tokenizer.chat_template = None
    coord.generate.return_value = "batch response"
    coord._model_router = None
    return coord


class TestSyncBatch:
    """POST /v1/batch synchronous batch processing."""

    def test_process_chat_requests(self, coordinator):
        original = g.coordinator
        g.coordinator = coordinator
        client = _make_client()
        try:
            resp = client.post(
                "/v1/batch",
                json={
                    "requests": [
                        {"method": "chat", "body": {"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 10}},
                        {"method": "chat", "body": {"messages": [{"role": "user", "content": "World"}], "max_tokens": 10}},
                    ],
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["total_requests"] == 2
            assert data["successful"] >= 1
            assert len(data["results"]) == 2
        finally:
            g.coordinator = original

    def test_process_completion_request(self, coordinator):
        original = g.coordinator
        g.coordinator = coordinator
        client = _make_client()
        try:
            resp = client.post(
                "/v1/batch",
                json={
                    "requests": [
                        {"method": "completion", "body": {"prompt": "Hello", "max_tokens": 10}},
                    ],
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["total_requests"] == 1
            assert data["successful"] >= 1
        finally:
            g.coordinator = original

    def test_without_coordinator_returns_503(self):
        original = g.coordinator
        g.coordinator = None
        client = _make_client()
        try:
            resp = client.post(
                "/v1/batch",
                json={
                    "requests": [
                        {"method": "chat", "body": {"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 10}},
                    ],
                },
            )
            assert resp.status_code == 503
        finally:
            g.coordinator = original

    def test_unknown_method_returns_error(self, coordinator):
        original = g.coordinator
        g.coordinator = coordinator
        client = _make_client()
        try:
            resp = client.post(
                "/v1/batch",
                json={
                    "requests": [
                        {"method": "unknown", "body": {}},
                    ],
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["results"][0]["status"] == "error"
        finally:
            g.coordinator = original


class TestBackgroundBatch:
    """POST /v1/batch/submit background batch processing."""

    def test_submit_background_batch(self, coordinator):
        original = g.coordinator
        g.coordinator = coordinator
        client = _make_client()
        try:
            resp = client.post(
                "/v1/batch/submit",
                json={
                    "items": [
                        {"request_id": "r1", "prompt": "Hello", "max_tokens": 10},
                    ],
                },
            )
            assert resp.status_code == 202
            data = resp.json()
            assert "batch_id" in data
            assert data["status"] == "pending"
            assert data["total_items"] == 1
        finally:
            g.coordinator = original

    def test_submit_without_coordinator_returns_503(self):
        original = g.coordinator
        g.coordinator = None
        client = _make_client()
        try:
            resp = client.post(
                "/v1/batch/submit",
                json={
                    "items": [
                        {"request_id": "r1", "prompt": "Hello", "max_tokens": 10},
                    ],
                },
            )
            assert resp.status_code == 503
        finally:
            g.coordinator = original

    def test_submit_multiple_items(self, coordinator):
        original = g.coordinator
        g.coordinator = coordinator
        client = _make_client()
        try:
            resp = client.post(
                "/v1/batch/submit",
                json={
                    "items": [
                        {"request_id": "r1", "prompt": "First", "max_tokens": 10},
                        {"request_id": "r2", "prompt": "Second", "max_tokens": 20},
                    ],
                },
            )
            assert resp.status_code == 202
            data = resp.json()
            assert data["total_items"] == 2
        finally:
            g.coordinator = original


class TestBatchStatus:
    """GET /v1/batch/{batch_id}/status."""

    def test_status_nonexistent_batch_returns_404(self):
        client = _make_client()
        resp = client.get("/v1/batch/nonexistent-batch-id/status")
        assert resp.status_code == 404


class TestCancelBatch:
    """POST /v1/batch/{batch_id}/cancel."""

    def test_cancel_nonexistent_batch_returns_404(self):
        client = _make_client()
        resp = client.post("/v1/batch/nonexistent-batch-id/cancel")
        assert resp.status_code == 404


class TestBatchPersistence:
    """File persistence via PersistentStore."""

    @pytest.fixture(autouse=True)
    def _isolated_store(self, tmp_path):
        from distllm.api import persistent_store
        db_path = str(tmp_path / "test.db")
        store = persistent_store.PersistentStore(db_path)
        self._store = store
        yield
        # Clean up not needed with tmp_path

    def test_save_and_retrieve_batch(self):
        batch_id = "persist-test-1"
        batch = {
            "id": batch_id, "endpoint": "/v1/chat/completions",
            "input_file_id": "file-x", "completion_window": "24h",
            "status": "completed", "created_at": 100,
            "in_progress_at": None, "expires_at": None,
            "finalizing_at": None, "completed_at": None,
            "failed_at": None, "expired_at": None, "cancelled_at": None,
            "output_file_id": None, "error_file_id": None,
            "errors": None, "metadata": None,
            "request_counts": {"total": 0, "completed": 0, "failed": 0},
        }
        self._store.save_batch(batch_id, batch)
        retrieved = self._store.get_batch(batch_id)
        assert retrieved["id"] == batch_id
        assert retrieved["status"] == "completed"

    def test_get_nonexistent_batch_returns_none(self):
        assert self._store.get_batch("nonexistent") is None

    def test_list_batches_returns_all(self):
        self._store.save_batch("list-a", {
            "id": "list-a", "endpoint": "/v1/chat/completions",
            "input_file_id": "f1", "completion_window": "24h",
            "status": "completed", "created_at": 1,
            "in_progress_at": None, "expires_at": None,
            "finalizing_at": None, "completed_at": None,
            "failed_at": None, "expired_at": None, "cancelled_at": None,
            "output_file_id": None, "error_file_id": None,
            "errors": None, "metadata": None,
            "request_counts": {"total": 0, "completed": 0, "failed": 0},
        })
        self._store.save_batch("list-b", {
            "id": "list-b", "endpoint": "/v1/completions",
            "input_file_id": "f2", "completion_window": "1h",
            "status": "in_progress", "created_at": 2,
            "in_progress_at": None, "expires_at": None,
            "finalizing_at": None, "completed_at": None,
            "failed_at": None, "expired_at": None, "cancelled_at": None,
            "output_file_id": None, "error_file_id": None,
            "errors": None, "metadata": None,
            "request_counts": {"total": 0, "completed": 0, "failed": 0},
        })
        batches = self._store.list_batches()
        ids = [b["id"] for b in batches]
        assert "list-a" in ids
        assert "list-b" in ids
