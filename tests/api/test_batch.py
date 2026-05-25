"""Batch processing tests: POST /v1/batches."""

import asyncio
import json
import os
import tempfile
import time
from unittest.mock import MagicMock, patch

import pytest
import torch
from fastapi.testclient import TestClient

from distllm.api.api_state import g
from distllm.api.server import app
from distllm.api import persistent_store
from distllm.api.routes import batch as batch_module


def _create_input_file(lines: list[str]) -> str:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    for line in lines:
        f.write(line + "\n")
    f.close()
    return f.name


def _register_file(file_id: str, path: str):
    batch_module._store.save_file(file_id, {
        "id": file_id, "storage_path": path,
        "filename": os.path.basename(path),
        "bytes": os.path.getsize(path),
        "created_at": 0, "purpose": "batch_input", "status": "uploaded",
    })


@pytest.fixture(autouse=True)
def _disable_auth(monkeypatch):
    monkeypatch.setenv("DISABLE_AUTH", "1")
    monkeypatch.setenv("DISTLLM_DEV_MODE", "1")
    monkeypatch.delenv("API_KEY", raising=False)


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
    coord.generate.return_value = "batch response"
    return coord


class TestCreateBatch:
    """POST /v1/batches."""

    def test_creates_batch_with_valid_input_file(self, coordinator):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"custom_id": "req-1", "body": {"messages": [{"role": "user", "content": "hi"}]}}\n')
            input_path = f.name

        file_id = "file-abc123"
        try:
            batch_module._store.save_file(file_id, {
                "id": file_id,
                "storage_path": input_path,
                "filename": os.path.basename(input_path),
                "bytes": os.path.getsize(input_path),
                "created_at": 0,
                "purpose": "batch_input",
                "status": "uploaded",
            })

            original = g.coordinator
            g.coordinator = coordinator
            try:
                resp = TestClient(app).post(
                    "/v1/batches",
                    json={
                        "input_file_id": file_id,
                        "endpoint": "/v1/chat/completions",
                        "completion_window": "24h",
                    },
                )
                assert resp.status_code == 200
                data = resp.json()
                assert data["object"] == "batch"
                assert data["input_file_id"] == file_id
                assert data["endpoint"] == "/v1/chat/completions"
                assert data["completion_window"] == "24h"
                assert data["status"] == "validating"
            finally:
                g.coordinator = original
        finally:
            os.unlink(input_path)
            batch_module._store.delete_file(file_id)

    def test_malformed_jsonl_fails_async(self, coordinator):
        input_path = _create_input_file([
            "this is not valid json",
        ])
        file_id = "file-malformed"
        _register_file(file_id, input_path)
        original = g.coordinator
        g.coordinator = coordinator
        try:
            create_resp = TestClient(app).post("/v1/batches", json={
                "input_file_id": file_id,
                "endpoint": "/v1/chat/completions",
                "completion_window": "24h",
            })
            assert create_resp.status_code == 200
            batch_id = create_resp.json()["id"]

            result = _wait_for_batch(batch_id)
            assert result["status"] == "failed"
        finally:
            g.coordinator = original
            os.unlink(input_path)


class TestBatchWindowParsing:
    """Unit tests for _parse_window."""

    def test_one_hour(self):
        assert batch_module._parse_window("1h") == 1

    def test_eight_hours(self):
        assert batch_module._parse_window("8h") == 8

    def test_twenty_four_hours(self):
        assert batch_module._parse_window("24h") == 24

    def test_fallback_to_24(self):
        assert batch_module._parse_window("unknown") == 24

    def test_empty_string_fallback(self):
        assert batch_module._parse_window("") == 24


class TestBatchPersistence:
    """File persistence via PersistentStore."""

    @pytest.fixture(autouse=True)
    def _isolated_store(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        original_store = batch_module._store
        batch_module._store = persistent_store.PersistentStore(db_path)
        yield
        batch_module._store = original_store

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
        batch_module._store.save_batch(batch_id, batch)
        retrieved = batch_module._store.get_batch(batch_id)
        assert retrieved["id"] == batch_id
        assert retrieved["status"] == "completed"

    def test_get_nonexistent_batch_returns_none(self):
        assert batch_module._store.get_batch("nonexistent") is None

    def test_list_batches_returns_all(self):
        batch_module._store.save_batch("list-a", {
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
        batch_module._store.save_batch("list-b", {
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
        batches = batch_module._store.list_batches()
        ids = [b["id"] for b in batches]
        assert "list-a" in ids
        assert "list-b" in ids

    def test_rejects_nonexistent_file(self, coordinator):
        original = g.coordinator
        g.coordinator = coordinator
        try:
            resp = TestClient(app).post(
                "/v1/batches",
                json={
                    "input_file_id": "file-nonexistent",
                    "endpoint": "/v1/chat/completions",
                    "completion_window": "24h",
                },
            )
            assert resp.status_code == 404
        finally:
            g.coordinator = original

    def test_rejects_invalid_file_id_format(self, coordinator):
        original = g.coordinator
        g.coordinator = coordinator
        try:
            resp = TestClient(app).post(
                "/v1/batches",
                json={
                    "input_file_id": "bad-id",
                    "endpoint": "/v1/chat/completions",
                    "completion_window": "24h",
                },
            )
            assert resp.status_code == 422
        finally:
            g.coordinator = original

    def test_without_coordinator_returns_503(self):
        original = g.coordinator
        g.coordinator = None
        try:
            resp = TestClient(app).post(
                "/v1/batches",
                json={
                    "input_file_id": "file-abc123",
                    "endpoint": "/v1/chat/completions",
                    "completion_window": "24h",
                },
            )
            assert resp.status_code == 503
        finally:
            g.coordinator = original


class TestGetBatch:
    """GET /v1/batches/{batch_id}."""

    def test_get_existing_batch(self, coordinator):
        original = g.coordinator
        g.coordinator = coordinator
        try:
            batch_id = "test-batch-123"
            batch_module._store.save_batch(batch_id, {
                "id": batch_id,
                "endpoint": "/v1/chat/completions",
                "input_file_id": "file-abc",
                "completion_window": "24h",
                "status": "completed",
                "created_at": 1000,
                "in_progress_at": 1001,
                "expires_at": 10000,
                "finalizing_at": None,
                "completed_at": 2000,
                "failed_at": None,
                "expired_at": None,
                "cancelled_at": None,
                "output_file_id": None,
                "error_file_id": None,
                "errors": None,
                "metadata": None,
                "request_counts": {"total": 1, "completed": 1, "failed": 0},
            })

            resp = TestClient(app).get(f"/v1/batches/{batch_id}")
            assert resp.status_code == 200
            data = resp.json()
            assert data["id"] == batch_id
            assert data["status"] == "completed"
        finally:
            g.coordinator = original

    def test_get_nonexistent_batch(self, coordinator):
        original = g.coordinator
        g.coordinator = coordinator
        try:
            resp = TestClient(app).get("/v1/batches/nonexistent")
            assert resp.status_code == 404
        finally:
            g.coordinator = original


class TestListBatches:
    """GET /v1/batches."""

    def test_list_batches(self, coordinator):
        original = g.coordinator
        g.coordinator = coordinator
        try:
            batch_module._store.save_batch("list-test-1", {
                "id": "list-test-1", "endpoint": "/v1/chat/completions",
                "input_file_id": "file-a", "completion_window": "24h",
                "status": "completed", "created_at": 100,
                "in_progress_at": None, "expires_at": None,
                "finalizing_at": None, "completed_at": None,
                "failed_at": None, "expired_at": None, "cancelled_at": None,
                "output_file_id": None, "error_file_id": None,
                "errors": None, "metadata": None,
                "request_counts": {"total": 0, "completed": 0, "failed": 0},
            })
            batch_module._store.save_batch("list-test-2", {
                "id": "list-test-2", "endpoint": "/v1/completions",
                "input_file_id": "file-b", "completion_window": "1h",
                "status": "in_progress", "created_at": 200,
                "in_progress_at": None, "expires_at": None,
                "finalizing_at": None, "completed_at": None,
                "failed_at": None, "expired_at": None, "cancelled_at": None,
                "output_file_id": None, "error_file_id": None,
                "errors": None, "metadata": None,
                "request_counts": {"total": 0, "completed": 0, "failed": 0},
            })

            resp = TestClient(app).get("/v1/batches")
            assert resp.status_code == 200
            data = resp.json()
            assert data["object"] == "list"
            assert len(data["data"]) >= 2
        finally:
            g.coordinator = original

    def test_list_batches_with_limit(self, coordinator):
        original = g.coordinator
        g.coordinator = coordinator
        try:
            batch_module._store.save_batch("limit-test-1", {
                "id": "limit-test-1", "endpoint": "/v1/chat/completions",
                "input_file_id": "file-a", "completion_window": "24h",
                "status": "validating", "created_at": 100,
                "in_progress_at": None, "expires_at": None,
                "finalizing_at": None, "completed_at": None,
                "failed_at": None, "expired_at": None, "cancelled_at": None,
                "output_file_id": None, "error_file_id": None,
                "errors": None, "metadata": None,
                "request_counts": {"total": 0, "completed": 0, "failed": 0},
            })
            batch_module._store.save_batch("limit-test-2", {
                "id": "limit-test-2", "endpoint": "/v1/completions",
                "input_file_id": "file-b", "completion_window": "1h",
                "status": "validating", "created_at": 200,
                "in_progress_at": None, "expires_at": None,
                "finalizing_at": None, "completed_at": None,
                "failed_at": None, "expired_at": None, "cancelled_at": None,
                "output_file_id": None, "error_file_id": None,
                "errors": None, "metadata": None,
                "request_counts": {"total": 0, "completed": 0, "failed": 0},
            })

            resp = TestClient(app).get("/v1/batches?limit=1")
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["data"]) == 1
        finally:
            g.coordinator = original

    def test_list_batches_with_after_cursor(self, coordinator):
        original = g.coordinator
        g.coordinator = coordinator
        try:
            batch_module._store.save_batch("after-test-a", {
                "id": "after-test-a", "endpoint": "/v1/chat/completions",
                "input_file_id": "file-x", "completion_window": "24h",
                "status": "completed", "created_at": 50,
                "in_progress_at": None, "expires_at": None,
                "finalizing_at": None, "completed_at": None,
                "failed_at": None, "expired_at": None, "cancelled_at": None,
                "output_file_id": None, "error_file_id": None,
                "errors": None, "metadata": None,
                "request_counts": {"total": 0, "completed": 0, "failed": 0},
            })
            batch_module._store.save_batch("after-test-b", {
                "id": "after-test-b", "endpoint": "/v1/completions",
                "input_file_id": "file-y", "completion_window": "1h",
                "status": "validating", "created_at": 100,
                "in_progress_at": None, "expires_at": None,
                "finalizing_at": None, "completed_at": None,
                "failed_at": None, "expired_at": None, "cancelled_at": None,
                "output_file_id": None, "error_file_id": None,
                "errors": None, "metadata": None,
                "request_counts": {"total": 0, "completed": 0, "failed": 0},
            })

            resp = TestClient(app).get("/v1/batches?after=after-test-a")
            assert resp.status_code == 200
            data = resp.json()
            ids = [b["id"] for b in data["data"]]
            assert "after-test-a" not in ids
        finally:
            g.coordinator = original


class TestCancelBatch:
    """POST /v1/batches/{batch_id}/cancel."""

    def test_cancel_validating_batch(self, coordinator):
        original = g.coordinator
        g.coordinator = coordinator
        try:
            batch_id = "cancel-test-1"
            batch_module._store.save_batch(batch_id, {
                "id": batch_id, "endpoint": "/v1/chat/completions",
                "input_file_id": "file-x", "completion_window": "24h",
                "status": "validating", "created_at": 100,
                "in_progress_at": None, "expires_at": None,
                "finalizing_at": None, "completed_at": None,
                "failed_at": None, "expired_at": None, "cancelled_at": None,
                "output_file_id": None, "error_file_id": None,
                "errors": None, "metadata": None,
                "request_counts": {"total": 0, "completed": 0, "failed": 0},
            })

            resp = TestClient(app).post(f"/v1/batches/{batch_id}/cancel")
            assert resp.status_code == 200
            assert resp.json()["status"] == "cancelled"
        finally:
            g.coordinator = original

    def test_cancel_completed_batch_fails(self, coordinator):
        original = g.coordinator
        g.coordinator = coordinator
        try:
            batch_id = "cancel-completed"
            batch_module._store.save_batch(batch_id, {
                "id": batch_id, "endpoint": "/v1/chat/completions",
                "input_file_id": "file-y", "completion_window": "24h",
                "status": "completed", "created_at": 100,
                "in_progress_at": None, "expires_at": None,
                "finalizing_at": None, "completed_at": None,
                "failed_at": None, "expired_at": None, "cancelled_at": None,
                "output_file_id": None, "error_file_id": None,
                "errors": None, "metadata": None,
                "request_counts": {"total": 0, "completed": 0, "failed": 0},
            })

            resp = TestClient(app).post(f"/v1/batches/{batch_id}/cancel")
            assert resp.status_code == 400
        finally:
            g.coordinator = original


def _wait_for_batch(batch_id: str, timeout: float = 30.0) -> dict:
    """Poll batch status until it leaves validating state."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = TestClient(app).get(f"/v1/batches/{batch_id}")
        assert resp.status_code == 200
        data = resp.json()
        if data["status"] not in ("validating",):
            return data
        time.sleep(0.1)
    raise TimeoutError(f"Batch {batch_id} did not complete within {timeout}s")


class TestBatchProcessing:
    """Background batch processing flow."""

    def test_batch_completes_successfully(self, coordinator):
        input_path = _create_input_file([
            json.dumps({"custom_id": "req-1", "body": {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}}),
        ])
        file_id = "file-flow-1"
        _register_file(file_id, input_path)
        original = g.coordinator
        g.coordinator = coordinator
        try:
            create_resp = TestClient(app).post("/v1/batches", json={
                "input_file_id": file_id,
                "endpoint": "/v1/chat/completions",
                "completion_window": "24h",
            })
            assert create_resp.status_code == 200
            batch_id = create_resp.json()["id"]

            result = _wait_for_batch(batch_id)
            assert result["status"] == "completed"
            assert result["output_file_id"] is not None
            assert result["request_counts"]["total"] == 1
            assert result["request_counts"]["completed"] == 1
            assert result["request_counts"]["failed"] == 0
        finally:
            g.coordinator = original
            os.unlink(input_path)

    def test_batch_partial_failure_populates_error_file(self, coordinator):
        input_path = _create_input_file([
            json.dumps({"custom_id": "req-ok", "body": {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}}),
            json.dumps({"custom_id": "req-bad", "body": {"messages": []}}),
        ])
        file_id = "file-flow-2"
        _register_file(file_id, input_path)
        original = g.coordinator
        g.coordinator = coordinator
        try:
            create_resp = TestClient(app).post("/v1/batches", json={
                "input_file_id": file_id,
                "endpoint": "/v1/chat/completions",
                "completion_window": "24h",
            })
            assert create_resp.status_code == 200
            batch_id = create_resp.json()["id"]

            result = _wait_for_batch(batch_id)
            assert result["status"] == "completed"
            assert result["error_file_id"] is not None
            assert result["request_counts"]["total"] == 2
            assert result["request_counts"]["completed"] == 1
            assert result["request_counts"]["failed"] == 1
        finally:
            g.coordinator = original
            os.unlink(input_path)
