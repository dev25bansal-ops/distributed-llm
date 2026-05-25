"""Fine-tuning job creation tests: POST /v1/fine_tuning/jobs."""

import asyncio
import json
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from distllm.api.api_state import g
from distllm.api.persistent_store import get_store
from distllm.api.server import app


async def _wait_for_job(job_id: str, timeout: float = 10) -> dict | None:
    """Poll job store until status is terminal or timeout."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        job = get_store().get_fine_tuning_job(job_id)
        if job and job["status"] not in ("validating_files", "queued", "running"):
            return job
        await asyncio.sleep(0.05)
    return get_store().get_fine_tuning_job(job_id)


@pytest.fixture(autouse=True)
def _disable_auth(monkeypatch):
    monkeypatch.setenv("DISABLE_AUTH", "1")
    monkeypatch.setenv("DISTLLM_DEV_MODE", "1")
    monkeypatch.delenv("API_KEY", raising=False)


class TestCreateFineTuningJob:
    @pytest.fixture(autouse=True)
    def setup(self):
        original = g.coordinator
        coord = MagicMock()
        coord.model_name = "test-model"
        coord.nodes = {}
        coord._shutting_down = False
        # Prevent async training from actually running
        coord.fine_tuning_backend = MagicMock()
        coord.fine_tuning_backend.train = MagicMock(return_value=0)
        g.coordinator = coord
        yield
        g.coordinator = original

    def _upload_file(self, purpose="fine-tune"):
        content = json.dumps(
            {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}
        ).encode()
        resp = TestClient(app).post(
            "/v1/files",
            files={"file": ("train.jsonl", content, "application/jsonl")},
            data={"purpose": purpose},
        )
        assert resp.status_code == 200
        return resp.json()["id"]

    def test_create_job(self):
        file_id = self._upload_file()
        try:
            resp = TestClient(app).post(
                "/v1/fine_tuning/jobs",
                json={"model": "gpt-3.5-turbo", "training_file": file_id},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["object"] == "fine_tuning.job"
            assert data["model"] == "gpt-3.5-turbo"
            assert data["training_file"] == file_id
            assert data["status"] == "validating_files"
            assert data["id"].startswith("ftjob-")
            assert data["hyperparameters"]["n_epochs"] == 3
            assert data["fine_tuned_model"].startswith("gpt-3.5-turbo:")
        finally:
            get_store().delete_file(file_id)

    def test_create_job_with_hyperparams(self):
        file_id = self._upload_file()
        try:
            resp = TestClient(app).post(
                "/v1/fine_tuning/jobs",
                json={
                    "model": "test-model",
                    "training_file": file_id,
                    "hyperparameters": {"n_epochs": 5, "batch_size": 8, "learning_rate_multiplier": 2.0},
                    "suffix": "my-model",
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["hyperparameters"]["n_epochs"] == 5
            assert data["hyperparameters"]["batch_size"] == 8
            assert data["hyperparameters"]["learning_rate_multiplier"] == 2.0
            assert "my-model" in data["fine_tuned_model"]
        finally:
            get_store().delete_file(file_id)

    def test_create_job_missing_training_file(self):
        resp = TestClient(app).post(
            "/v1/fine_tuning/jobs",
            json={"model": "gpt-3.5-turbo", "training_file": "file-nonexistent"},
        )
        assert resp.status_code == 404
        assert "Training file" in resp.json()["error"]["message"]

    def test_create_job_invalid_file_purpose(self):
        file_id = self._upload_file(purpose="fine-tune-results")
        try:
            resp = TestClient(app).post(
                "/v1/fine_tuning/jobs",
                json={"model": "gpt-3.5-turbo", "training_file": file_id},
            )
            assert resp.status_code == 400
            assert "purpose" in resp.json()["error"]["message"]
        finally:
            get_store().delete_file(file_id)

    def test_create_job_no_coordinator(self):
        original = g.coordinator
        g.coordinator = None
        try:
            resp = TestClient(app).post(
                "/v1/fine_tuning/jobs",
                json={"model": "gpt-3.5-turbo", "training_file": "file-abc"},
            )
            assert resp.status_code == 503
        finally:
            g.coordinator = original


class TestGetFineTuningJob:
    """GET /v1/fine_tuning/jobs/{job_id}."""

    @pytest.fixture(autouse=True)
    def setup(self):
        original = g.coordinator
        coord = MagicMock()
        coord.model_name = "test-model"
        coord.nodes = {}
        coord._shutting_down = False
        coord.fine_tuning_backend = MagicMock()
        coord.fine_tuning_backend.train = MagicMock(return_value=0)
        g.coordinator = coord
        yield
        g.coordinator = original

    def _create_job(self, model="gpt-3.5-turbo", hyperparams=None):
        content = json.dumps(
            {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}
        ).encode()
        resp = TestClient(app).post(
            "/v1/files",
            files={"file": ("train.jsonl", content, "application/jsonl")},
            data={"purpose": "fine-tune"},
        )
        file_id = resp.json()["id"]
        body = {"model": model, "training_file": file_id}
        if hyperparams:
            body["hyperparameters"] = hyperparams
        resp = TestClient(app).post("/v1/fine_tuning/jobs", json=body)
        job_id = resp.json()["id"]
        return file_id, job_id

    def test_get_job_status(self):
        file_id, job_id = self._create_job()
        try:
            resp = TestClient(app).get(f"/v1/fine_tuning/jobs/{job_id}")
            assert resp.status_code == 200
            data = resp.json()
            assert data["id"] == job_id
            assert data["object"] == "fine_tuning.job"
            assert data["model"] == "gpt-3.5-turbo"
            assert data["status"] == "succeeded"
            assert data["training_file"] == file_id
            assert isinstance(data["created_at"], int)
            assert data["hyperparameters"]["n_epochs"] == 3
        finally:
            get_store().delete_file(file_id)

    def test_get_job_with_custom_config(self):
        file_id, job_id = self._create_job(
            model="test-model",
            hyperparams={"n_epochs": 10, "batch_size": 16, "learning_rate_multiplier": 0.5},
        )
        try:
            resp = TestClient(app).get(f"/v1/fine_tuning/jobs/{job_id}")
            assert resp.status_code == 200
            data = resp.json()
            assert data["model"] == "test-model"
            assert data["hyperparameters"]["n_epochs"] == 10
            assert data["hyperparameters"]["batch_size"] == 16
            assert data["hyperparameters"]["learning_rate_multiplier"] == 0.5
            assert data["fine_tuned_model"].startswith("test-model:")
        finally:
            get_store().delete_file(file_id)

    def test_get_job_not_found(self):
        resp = TestClient(app).get("/v1/fine_tuning/jobs/ftjob-nonexistent")
        assert resp.status_code == 404


class TestListFineTuningJobs:
    """GET /v1/fine_tuning/jobs."""

    @pytest.fixture(autouse=True)
    def setup(self):
        original = g.coordinator
        coord = MagicMock()
        coord.model_name = "test-model"
        coord.nodes = {}
        coord._shutting_down = False
        coord.fine_tuning_backend = MagicMock()
        coord.fine_tuning_backend.train = MagicMock(return_value=0)
        g.coordinator = coord
        yield
        g.coordinator = original

    def _upload_and_create(self, model="gpt-3.5-turbo"):
        content = json.dumps(
            {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}
        ).encode()
        resp = TestClient(app).post(
            "/v1/files",
            files={"file": ("train.jsonl", content, "application/jsonl")},
            data={"purpose": "fine-tune"},
        )
        file_id = resp.json()["id"]
        resp = TestClient(app).post("/v1/fine_tuning/jobs", json={"model": model, "training_file": file_id})
        job_id = resp.json()["id"]
        return job_id

    def test_list_all_jobs(self):
        ids = [self._upload_and_create() for _ in range(3)]
        try:
            resp = TestClient(app).get("/v1/fine_tuning/jobs")
            assert resp.status_code == 200
            data = resp.json()
            assert data["object"] == "list"
            returned_ids = [j["id"] for j in data["data"]]
            for jid in ids:
                assert jid in returned_ids
        finally:
            for jid in ids:
                get_store().update_fine_tuning_job(jid, {"status": "cancelled"})

    def test_list_filter_by_model(self):
        id_a = self._upload_and_create(model="model-a")
        id_b = self._upload_and_create(model="model-b")
        try:
            resp = TestClient(app).get("/v1/fine_tuning/jobs", params={"model": "model-a"})
            assert resp.status_code == 200
            returned_models = [j["model"] for j in resp.json()["data"]]
            assert all(m == "model-a" for m in returned_models)
        finally:
            for jid in [id_a, id_b]:
                get_store().update_fine_tuning_job(jid, {"status": "cancelled"})

    def test_list_limit(self):
        ids = [self._upload_and_create() for _ in range(5)]
        try:
            resp = TestClient(app).get("/v1/fine_tuning/jobs", params={"limit": 2})
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["data"]) <= 2
        finally:
            for jid in ids:
                get_store().update_fine_tuning_job(jid, {"status": "cancelled"})


class TestCancelFineTuningJob:
    """POST /v1/fine_tuning/jobs/{job_id}/cancel."""

    def test_cancel_running_job(self):
        job_id = "ftjob-cancel-test"
        now = int(__import__("time").time())
        get_store().save_fine_tuning_job(job_id, {
            "id": job_id,
            "model": "gpt-3.5-turbo",
            "created_at": now,
            "status": "running",
            "training_file": "file-abc",
            "hyperparameters": {"n_epochs": 3, "batch_size": None, "learning_rate_multiplier": 1.0},
            "fine_tuned_model": None,
            "finished_at": None,
            "error": None,
            "estimated_finish": None,
            "integrations": None,
            "seed": None,
            "validation_file": None,
            "trained_tokens": None,
            "result_files": [],
            "organization_id": None,
            "object": "fine_tuning.job",
        })
        try:
            resp = TestClient(app).post(f"/v1/fine_tuning/jobs/{job_id}/cancel")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "cancelled"
            assert data["id"] == job_id
            assert isinstance(data["finished_at"], int)
        finally:
            get_store().update_fine_tuning_job(job_id, {"status": "cancelled"})

    def test_cancel_succeeded_job_returns_400(self):
        job_id = "ftjob-cancel-fail"
        now = int(__import__("time").time())
        get_store().save_fine_tuning_job(job_id, {
            "id": job_id,
            "model": "gpt-3.5-turbo",
            "created_at": now,
            "status": "succeeded",
            "training_file": "file-abc",
            "hyperparameters": {"n_epochs": 3, "batch_size": None, "learning_rate_multiplier": 1.0},
            "fine_tuned_model": "gpt-3.5-turbo:ftjob-cancel-fail",
            "finished_at": now + 100,
            "error": None,
            "estimated_finish": None,
            "integrations": None,
            "seed": None,
            "validation_file": None,
            "trained_tokens": 100,
            "result_files": [],
            "organization_id": None,
            "object": "fine_tuning.job",
        })
        try:
            resp = TestClient(app).post(f"/v1/fine_tuning/jobs/{job_id}/cancel")
            assert resp.status_code == 400
            assert "cannot cancel" in resp.json()["error"]["message"].lower()
        finally:
            get_store().update_fine_tuning_job(job_id, {"status": "cancelled"})

    def test_cancel_nonexistent_job(self):
        resp = TestClient(app).post("/v1/fine_tuning/jobs/ftjob-nonexistent/cancel")
        assert resp.status_code == 404


class TestTrainingFailures:
    """Background training loop failure scenarios."""

    @pytest.fixture(autouse=True)
    def setup(self):
        original = g.coordinator
        coord = MagicMock()
        coord.model_name = "test-model"
        coord.nodes = {}
        coord._shutting_down = False
        coord.fine_tuning_backend = None
        coord._fine_tuning_backend = None
        coord.local_partitioner = None
        g.coordinator = coord
        yield
        g.coordinator = original

    def _upload_file(self):
        content = json.dumps(
            {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}
        ).encode()
        resp = TestClient(app).post(
            "/v1/files",
            files={"file": ("train.jsonl", content, "application/jsonl")},
            data={"purpose": "fine-tune"},
        )
        return resp.json()["id"]

    async def test_model_loading_failure(self):
        file_id = self._upload_file()
        resp = TestClient(app).post(
            "/v1/fine_tuning/jobs",
            json={"model": "gpt-3.5-turbo", "training_file": file_id},
        )
        assert resp.status_code == 200
        job_id = resp.json()["id"]

        job = await _wait_for_job(job_id)
        assert job is not None
        assert job["status"] == "failed"
        assert "error" in job
        assert "Fine-tuning backend" in job["error"]["message"]
        assert job["error"]["code"] == "training_error"
        assert isinstance(job["finished_at"], int)
        get_store().delete_file(file_id)

    async def test_training_backend_raises_error(self):
        original = g.coordinator
        coord = MagicMock()
        coord.model_name = "test-model"
        coord.nodes = {}
        coord._shutting_down = False
        coord.fine_tuning_backend = MagicMock()
        coord.fine_tuning_backend.train = MagicMock(side_effect=RuntimeError("CUDA out of memory"))
        coord.local_partitioner = None
        g.coordinator = coord
        try:
            file_id = self._upload_file()
            resp = TestClient(app).post(
                "/v1/fine_tuning/jobs",
                json={"model": "gpt-3.5-turbo", "training_file": file_id},
            )
            assert resp.status_code == 200
            job_id = resp.json()["id"]

            job = await _wait_for_job(job_id)
            assert job is not None
            assert job["status"] == "failed"
            assert "CUDA out of memory" in job["error"]["message"]
            get_store().delete_file(file_id)
        finally:
            g.coordinator = original


class TestHyperparameterBounds:
    """Pydantic validation of hyperparameter bounds."""

    def test_n_epochs_too_low(self):
        resp = TestClient(app).post(
            "/v1/fine_tuning/jobs",
            json={"model": "gpt-3.5-turbo", "training_file": "file-abc", "hyperparameters": {"n_epochs": 0}},
        )
        assert resp.status_code == 422

    def test_n_epochs_too_high(self):
        resp = TestClient(app).post(
            "/v1/fine_tuning/jobs",
            json={"model": "gpt-3.5-turbo", "training_file": "file-abc", "hyperparameters": {"n_epochs": 51}},
        )
        assert resp.status_code == 422

    def test_batch_size_too_low(self):
        resp = TestClient(app).post(
            "/v1/fine_tuning/jobs",
            json={"model": "gpt-3.5-turbo", "training_file": "file-abc", "hyperparameters": {"batch_size": 0}},
        )
        assert resp.status_code == 422

    def test_lr_multiplier_too_low(self):
        resp = TestClient(app).post(
            "/v1/fine_tuning/jobs",
            json={"model": "gpt-3.5-turbo", "training_file": "file-abc", "hyperparameters": {"learning_rate_multiplier": 0.05}},
        )
        assert resp.status_code == 422

    def test_lr_multiplier_too_high(self):
        resp = TestClient(app).post(
            "/v1/fine_tuning/jobs",
            json={"model": "gpt-3.5-turbo", "training_file": "file-abc", "hyperparameters": {"learning_rate_multiplier": 11}},
        )
        assert resp.status_code == 422


class TestJobEvents:
    """GET /v1/fine_tuning/jobs/{job_id}/events."""

    def test_events_not_found(self):
        resp = TestClient(app).get("/v1/fine_tuning/jobs/ftjob-nonexistent/events")
        assert resp.status_code == 404

    def _save_job(self, status: str, job_id: str | None = None):
        import time
        import uuid
        jid = job_id or f"ftjob-{uuid.uuid4().hex[:10]}"
        now = int(time.time())
        get_store().save_fine_tuning_job(jid, {
            "id": jid,
            "model": "gpt-3.5-turbo",
            "created_at": now,
            "status": status,
            "training_file": "file-abc",
            "hyperparameters": {"n_epochs": 3, "batch_size": None, "learning_rate_multiplier": 1.0},
            "fine_tuned_model": "gpt-3.5-turbo:ftjob-xyz" if status == "succeeded" else None,
            "finished_at": now + 100 if status in ("succeeded", "failed", "cancelled") else None,
            "error": {"message": "bad data", "code": "training_error"} if status == "failed" else None,
            "estimated_finish": None,
            "integrations": None,
            "seed": None,
            "validation_file": None,
            "trained_tokens": 50 if status == "succeeded" else None,
            "result_files": [],
            "organization_id": None,
            "object": "fine_tuning.job",
        })
        return jid

    def test_events_for_validating_job(self):
        jid = self._save_job("validating_files")
        resp = TestClient(app).get(f"/v1/fine_tuning/jobs/{jid}/events")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert data["data"][0]["message"] == "Job created and queued for processing."

    def test_events_for_running_job(self):
        jid = self._save_job("running")
        resp = TestClient(app).get(f"/v1/fine_tuning/jobs/{jid}/events")
        assert resp.status_code == 200
        messages = [e["message"] for e in resp.json()["data"]]
        assert any("Job created and queued for processing" in m for m in messages)
        assert any("Validating training file" in m for m in messages)
        assert any("Training started" in m for m in messages)

    def test_events_for_succeeded_job(self):
        jid = self._save_job("succeeded")
        resp = TestClient(app).get(f"/v1/fine_tuning/jobs/{jid}/events")
        assert resp.status_code == 200
        messages = [e["message"] for e in resp.json()["data"]]
        assert any("completed" in m.lower() for m in messages)
        assert any("fine-tuned model" in m.lower() for m in messages)

    def test_events_for_failed_job(self):
        jid = self._save_job("failed")
        resp = TestClient(app).get(f"/v1/fine_tuning/jobs/{jid}/events")
        assert resp.status_code == 200
        messages = [e["message"] for e in resp.json()["data"]]
        assert any("failed" in m.lower() for m in messages)
        assert any("bad data" in m.lower() for m in messages)

    def test_events_ordered_timeline(self):
        jid = self._save_job("succeeded")
        resp = TestClient(app).get(f"/v1/fine_tuning/jobs/{jid}/events")
        data = resp.json()["data"]
        timestamps = [e["created_at"] for e in data]
        assert timestamps == sorted(timestamps)
