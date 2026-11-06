"""Security regression tests for fine-tuning IDOR (SEC-API-02).

Covers the tenant-ownership guard on fine-tuning jobs, mirroring
tests/api/test_batch_security.py.  Previously list / retrieve / cancel /
events were keyed only on the global job_id with no ownership binding,
so any authenticated key could read or cancel another tenant's job.
"""

from __future__ import annotations

import json
import secrets
import time
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from distllm.api.api_state import g
from distllm.api.server import app
from distllm.api.persistent_store import PersistentStore
from distllm.core.api_key_store import reset_api_key_store


@pytest.fixture(autouse=True)
def _two_keys(monkeypatch):
    global KEY_A, KEY_B
    KEY_A = secrets.token_urlsafe(32)
    KEY_B = secrets.token_urlsafe(32)
    monkeypatch.setenv(
        "API_KEYS",
        json.dumps({
            "keys": [
                {"key": KEY_A, "role": "admin", "label": "a", "key_id": "key-a"},
                {"key": KEY_B, "role": "admin", "label": "b", "key_id": "key-b"},
            ]
        }),
    )
    monkeypatch.delenv("API_KEY", raising=False)
    reset_api_key_store()
    yield
    monkeypatch.delenv("API_KEYS", raising=False)


@pytest.fixture(autouse=True)
def coordinator():
    coord = MagicMock()
    coord.model_name = "test-model"
    coord.nodes = {}
    coord._shutting_down = False
    original = g.coordinator
    g.coordinator = coord
    yield
    g.coordinator = original


@pytest.fixture
def store(tmp_path, monkeypatch):
    db = tmp_path / "test.db"
    s = PersistentStore(db_path=str(db))
    monkeypatch.setattr(
        "distllm.api.routes.fine_tuning.get_store", lambda: s
    )
    monkeypatch.setattr(
        "distllm.api.routes.files.get_store", lambda: s
    )
    return s


def _client(key: str) -> TestClient:
    client = TestClient(app)
    client.headers["Authorization"] = f"Bearer {key}"
    return client


def _upload_file(client: TestClient) -> str:
    resp = client.post(
        "/v1/files",
        files={"file": ("train.jsonl", b'{"prompt": "hi", "completion": "yo"}\n')},
        data={"purpose": "fine-tune"},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


def _create_job(client: TestClient, file_id: str) -> dict:
    resp = client.post(
        "/v1/fine_tuning/jobs",
        json={"model": "test-model", "training_file": file_id},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


class TestFineTuningOwnership:
    def test_owner_can_retrieve_job(self, store):
        client_a = _client(KEY_A)
        file_id = _upload_file(client_a)
        job = _create_job(client_a, file_id)

        resp = client_a.get(f"/v1/fine_tuning/jobs/{job['id']}")
        assert resp.status_code == 200
        assert resp.json()["id"] == job["id"]

    def test_other_tenant_cannot_retrieve_job(self, store):
        client_a = _client(KEY_A)
        file_id = _upload_file(client_a)
        job = _create_job(client_a, file_id)

        resp = _client(KEY_B).get(f"/v1/fine_tuning/jobs/{job['id']}")
        assert resp.status_code == 404  # existence is not leaked

    def test_other_tenant_cannot_cancel_job(self, store):
        client_a = _client(KEY_A)
        file_id = _upload_file(client_a)
        job = _create_job(client_a, file_id)

        resp = _client(KEY_B).post(f"/v1/fine_tuning/jobs/{job['id']}/cancel")
        assert resp.status_code == 404

    def test_owner_can_cancel_job(self, store):
        client_a = _client(KEY_A)
        file_id = _upload_file(client_a)
        job = _create_job(client_a, file_id)

        resp = client_a.post(f"/v1/fine_tuning/jobs/{job['id']}/cancel")
        assert resp.status_code == 200
        assert resp.json()["status"] == "cancelled"

    def test_other_tenant_cannot_read_events(self, store):
        client_a = _client(KEY_A)
        file_id = _upload_file(client_a)
        job = _create_job(client_a, file_id)

        resp = _client(KEY_B).get(f"/v1/fine_tuning/jobs/{job['id']}/events")
        assert resp.status_code == 404

    def test_owner_can_read_events(self, store):
        client_a = _client(KEY_A)
        file_id = _upload_file(client_a)
        job = _create_job(client_a, file_id)

        resp = client_a.get(f"/v1/fine_tuning/jobs/{job['id']}/events")
        assert resp.status_code == 200

    def test_list_is_tenant_scoped(self, store):
        client_a = _client(KEY_A)
        file_a = _upload_file(client_a)
        job_a = _create_job(client_a, file_a)

        client_b = _client(KEY_B)
        file_b = _upload_file(client_b)
        job_b = _create_job(client_b, file_b)

        list_a = client_a.get("/v1/fine_tuning/jobs").json()
        ids_a = {j["id"] for j in list_a["data"]}
        assert job_a["id"] in ids_a
        assert job_b["id"] not in ids_a

        list_b = client_b.get("/v1/fine_tuning/jobs").json()
        ids_b = {j["id"] for j in list_b["data"]}
        assert job_b["id"] in ids_b
        assert job_a["id"] not in ids_b

    def test_owner_recorded_on_create(self, store):
        client_a = _client(KEY_A)
        file_id = _upload_file(client_a)
        job = _create_job(client_a, file_id)

        stored = store.get_fine_tuning_job(job["id"])
        assert stored["owner"] == "key-a"

    def test_legacy_unowned_job_not_visible_to_authenticated_key(self, store):
        # Simulate a job created before the owner field existed.
        store.save_fine_tuning_job(
            "ft-legacy",
            {
                "id": "ft-legacy",
                "model": "test-model",
                "created_at": time.time(),
                "status": "queued",
                "training_file": "file-x",
                "hyperparameters": {},
                "result_files": [],
            },
        )
        resp = _client(KEY_A).get("/v1/fine_tuning/jobs/ft-legacy")
        assert resp.status_code == 404
