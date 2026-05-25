"""File upload, list, get, delete, and content download tests."""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from distllm.api.api_state import g
from distllm.api.server import app


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
    return coord


class TestUploadFile:
    """POST /v1/files."""

    @pytest.fixture(autouse=True)
    def setup(self, coordinator):
        original = g.coordinator
        g.coordinator = coordinator
        yield
        g.coordinator = original

    def test_upload_valid_jsonl(self):
        content = json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode()
        resp = TestClient(app).post(
            "/v1/files",
            files={"file": ("train.jsonl", content, "application/jsonl")},
            data={"purpose": "fine-tune"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "file"
        assert data["filename"] == "train.jsonl"
        assert data["purpose"] == "fine-tune"
        assert data["status"] == "uploaded"
        assert data["bytes"] == len(content)
        assert data["id"].startswith("file-")

    def test_upload_invalid_jsonl(self):
        content = b"this is not valid json"
        resp = TestClient(app).post(
            "/v1/files",
            files={"file": ("bad.jsonl", content, "application/jsonl")},
            data={"purpose": "fine-tune"},
        )
        assert resp.status_code == 400
        data = resp.json()
        assert "Invalid JSON" in data["error"]["message"]

    def test_upload_invalid_purpose(self):
        content = b"some content"
        resp = TestClient(app).post(
            "/v1/files",
            files={"file": ("test.txt", content)},
            data={"purpose": "invalid-purpose"},
        )
        assert resp.status_code == 400
        data = resp.json()
        assert "Invalid purpose" in data["error"]["message"]

    def test_upload_too_large(self):
        content = b"x" * (101 * 1024 * 1024)
        try:
            resp = TestClient(app).post(
                "/v1/files",
                files={"file": ("large.bin", content)},
                data={"purpose": "batch"},
            )
            assert resp.status_code == 413
        finally:
            del content

    def test_upload_without_coordinator_returns_503(self):
        original = g.coordinator
        g.coordinator = None
        try:
            resp = TestClient(app).post(
                "/v1/files",
                files={"file": ("test.jsonl", b"{}")},
                data={"purpose": "fine-tune"},
            )
            assert resp.status_code == 503
        finally:
            g.coordinator = original


class TestGetFile:
    """GET /v1/files/{file_id}."""

    @pytest.fixture(autouse=True)
    def setup(self, coordinator):
        original = g.coordinator
        g.coordinator = coordinator
        yield
        g.coordinator = original

    def test_get_file_metadata_by_id(self):
        resp = TestClient(app).post(
            "/v1/files",
            files={"file": ("test.jsonl", json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode())},
            data={"purpose": "fine-tune"},
        )
        assert resp.status_code == 200
        file_id = resp.json()["id"]

        resp = TestClient(app).get(f"/v1/files/{file_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == file_id
        assert data["purpose"] == "fine-tune"
        assert data["object"] == "file"
        assert data["filename"] == "test.jsonl"
        assert isinstance(data["bytes"], int)
        assert isinstance(data["created_at"], int)

    def test_get_file_not_found(self):
        resp = TestClient(app).get("/v1/files/file-nonexistent")
        assert resp.status_code == 404


class TestListFiles:
    """GET /v1/files."""

    @pytest.fixture(autouse=True)
    def setup(self, coordinator):
        original = g.coordinator
        g.coordinator = coordinator
        yield
        g.coordinator = original

    def test_list_all_files(self):
        posted = [
            TestClient(app).post(
                "/v1/files",
                files={"file": ("a.jsonl", json.dumps({"messages": [{"role": "user", "content": str(i)}]}).encode())},
                data={"purpose": "batch"},
            ).json()["id"]
            for i in range(3)
        ]
        try:
            resp = TestClient(app).get("/v1/files")
            assert resp.status_code == 200
            data = resp.json()
            assert data["object"] == "list"
            ids = [f["id"] for f in data["data"]]
            for fid in posted:
                assert fid in ids
        finally:
            _cleanup_files(posted)

    def test_list_files_filter_by_purpose(self):
        batch_id = TestClient(app).post(
            "/v1/files",
            files={"file": ("batch.jsonl", json.dumps({"messages": [{"role": "user", "content": "hello"}]}).encode())},
            data={"purpose": "batch"},
        ).json()["id"]
        fine_tune_id = TestClient(app).post(
            "/v1/files",
            files={"file": ("ft.jsonl", json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode())},
            data={"purpose": "fine-tune"},
        ).json()["id"]
        try:
            resp = TestClient(app).get("/v1/files", params={"purpose": "batch"})
            assert resp.status_code == 200
            data = resp.json()
            ids = [f["id"] for f in data["data"]]
            assert batch_id in ids
            assert fine_tune_id not in ids
        finally:
            _cleanup_files([batch_id, fine_tune_id])

    def test_list_files_limit(self):
        posted = [
            TestClient(app).post(
                "/v1/files",
                files={"file": (f"f{i}.jsonl", json.dumps({"messages": [{"role": "user", "content": str(i)}]}).encode())},
                data={"purpose": "fine-tune"},
            ).json()["id"]
            for i in range(5)
        ]
        try:
            resp = TestClient(app).get("/v1/files", params={"limit": 2})
            assert resp.status_code == 200
            assert len(resp.json()["data"]) == 2
        finally:
            _cleanup_files(posted)


class TestDeleteFile:
    """DELETE /v1/files/{file_id}."""

    @pytest.fixture(autouse=True)
    def setup(self, coordinator):
        original = g.coordinator
        g.coordinator = coordinator
        yield
        g.coordinator = original

    def test_delete_removes_from_listing(self):
        resp = TestClient(app).post(
            "/v1/files",
            files={"file": ("del.jsonl", b"test content")},
            data={"purpose": "batch"},
        )
        assert resp.status_code == 200
        file_id = resp.json()["id"]

        resp = TestClient(app).delete(f"/v1/files/{file_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == file_id
        assert data["deleted"] is True

        resp = TestClient(app).get(f"/v1/files/{file_id}")
        assert resp.status_code == 404

        resp = TestClient(app).get("/v1/files")
        ids = [f["id"] for f in resp.json()["data"]]
        assert file_id not in ids

    def test_delete_nonexistent_file(self):
        resp = TestClient(app).delete("/v1/files/file-nonexistent")
        assert resp.status_code == 404
        data = resp.json()
        assert "file-nonexistent" in data["error"]["message"]


class TestGetFileContent:
    """GET /v1/files/{file_id}/content."""

    @pytest.fixture(autouse=True)
    def setup(self, coordinator):
        original = g.coordinator
        g.coordinator = coordinator
        yield
        g.coordinator = original

    def test_download_original_content(self):
        original_content = json.dumps({"messages": [{"role": "user", "content": "hello world"}]}).encode()
        resp = TestClient(app).post(
            "/v1/files",
            files={"file": ("download.jsonl", original_content, "application/jsonl")},
            data={"purpose": "fine-tune"},
        )
        assert resp.status_code == 200
        file_id = resp.json()["id"]

        try:
            resp = TestClient(app).get(f"/v1/files/{file_id}/content")
            assert resp.status_code == 200
            assert resp.content == original_content
            assert resp.headers.get("content-type") == "application/octet-stream"
        finally:
            _cleanup_files([file_id])

    def test_download_nonexistent_file(self):
        resp = TestClient(app).get("/v1/files/file-nonexistent/content")
        assert resp.status_code == 404

    def test_db_record_but_file_missing_on_disk(self):
        from distllm.api.persistent_store import get_store

        original_content = b"content that will be deleted"
        resp = TestClient(app).post(
            "/v1/files",
            files={"file": ("gone.jsonl", original_content)},
            data={"purpose": "batch"},
        )
        assert resp.status_code == 200
        file_id = resp.json()["id"]

        try:
            file_obj = get_store().get_file(file_id)
            storage_path = Path(file_obj["storage_path"])
            storage_path.unlink()

            resp = TestClient(app).get(f"/v1/files/{file_id}/content")
            assert resp.status_code == 404
        finally:
            get_store().delete_file(file_id)


def _cleanup_files(file_ids):
    from distllm.api.persistent_store import get_store
    store = get_store()
    for fid in file_ids:
        store.delete_file(fid)
