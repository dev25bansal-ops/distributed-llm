"""SEC-API-09: Upload quota + streaming in /v1/files POST.

Regression tests for the mid-stream upload guard:

1. A file larger than the per-file cap is rejected *mid-stream* — before the
   whole body is buffered (the old code read the entire file into memory and
   only then checked ``len(content) > _MAX_FILE_SIZE``).
2. A tenant whose already-stored bytes plus the incoming upload exceed the
   configurable per-tenant quota (``DISTLLM_UPLOAD_QUOTA_BYTES``) is rejected
   with 403.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from distllm.api.persistent_store import PersistentStore
from distllm.api.routes import files as files_route
from distllm.api.routes.files import router as files_router
from starlette.datastructures import UploadFile as StarletteUploadFile


def _make_client(monkeypatch, tmp_path, store: PersistentStore) -> TestClient:
    """Build a TestClient with the files router; identity injected via middleware."""
    monkeypatch.setattr(files_route, "get_store", lambda: store)
    monkeypatch.setattr(files_route, "get_data_dir", lambda: tmp_path)
    monkeypatch.setattr(
        "distllm.api.api_state.g.coordinator",
        object(),
    )
    files_router.dependencies.clear()

    app = FastAPI()

    @app.middleware("http")
    async def _identity(request, call_next):
        request.state.api_key_id = request.headers.get("x-key-id", "")
        request.state.api_key_role = request.headers.get("x-key-role", "inference-only")
        return await call_next(request)

    app.include_router(files_router)
    return TestClient(app)


@pytest.fixture
def client(monkeypatch, tmp_path):
    store = PersistentStore(db_path=str(tmp_path / "files.db"))
    yield _make_client(monkeypatch, tmp_path, store), store
    for conn in list(store._pool):
        try:
            conn.close()
        except Exception:
            pass
    store._pool.clear()


def _upload(client: TestClient, key_id: str, filename: str, body: bytes, purpose: str = "batch"):
    return client.post(
        "/v1/files",
        files={"file": (filename, body)},
        data={"purpose": purpose},
        headers={"x-key-id": key_id, "x-key-role": "inference-only"},
    )


class TestUploadStreamingCap:
    def test_oversized_file_rejected_mid_stream(self, client, monkeypatch):
        """A 2 MiB upload with the cap lowered to 1 MiB is rejected 413
        before the full body is buffered into memory."""
        c, _ = client

        # Lower the cap to 1 MiB and make the streamed chunk size 64 KiB so
        # the cap trips mid-stream (several chunks in), not on the first one.
        monkeypatch.setattr(files_route, "_MAX_FILE_SIZE", 1024 * 1024)
        monkeypatch.setattr(files_route, "_CHUNK_SIZE", 64 * 1024)

        # Count how many bytes the route actually pulls off the upload stream
        # into its incremental buffer.  A full-buffer implementation would
        # consume all 2 MiB (32 x 64 KiB) before failing; the streaming one
        # must stop as soon as the running total crosses the 1 MiB cap
        # (16 x 64 KiB = 1 MiB exactly, so the 17th chunk trips it).
        # Wrap the bound method instead of replacing the class attribute so
        # FastAPI's UploadFile.read -> super().read() chain still works.
        consumed = []
        orig_read = StarletteUploadFile.read

        async def _spy_read(self, size: int = -1):
            data = await orig_read(self, size)
            consumed.append(len(data))
            return data

        monkeypatch.setattr(StarletteUploadFile, "read", _spy_read)

        body = b"x" * (2 * 1024 * 1024)  # 2 MiB
        resp = _upload(c, "key-a", "big.bin", body)

        assert resp.status_code == 413
        # The upload was streamed in bounded chunks, not read whole.
        assert consumed
        # 2 MiB / 64 KiB = 32 chunks; the cap must trip well before that.
        # 1 MiB cap / 64 KiB chunk = 16 chunks fit, so the 17th read is the
        # last one the route performs before raising.
        assert sum(consumed) <= 1024 * 1024 + 64 * 1024
        assert len(consumed) <= 17
        # No partial file was left on disk / in the store.
        assert c.get("/v1/files").json()["data"] == []


class TestUploadTenantQuota:
    def test_tenant_over_quota_rejected(self, client, monkeypatch):
        """A tenant whose stored bytes already meet the quota is rejected 403."""
        c, store = client

        quota = 1024 * 1024  # 1 MiB
        monkeypatch.setattr(files_route, "_MAX_FILE_SIZE", 512 * 1024 * 1024)
        monkeypatch.setattr(files_route, "_CHUNK_SIZE", 256 * 1024)
        monkeypatch.setenv(files_route._UPLOAD_QUOTA_ENV, str(quota))

        # Seed the tenant's stored bytes to the quota via the store.
        store.save_file(
            "file-seed",
            {
                "id": "file-seed",
                "bytes": quota,
                "created_at": 0,
                "filename": "seed.bin",
                "purpose": "batch",
                "status": "uploaded",
                "path": "/nonexistent/seed.bin",
                "object": "file",
                "owner": "key-a",
            },
        )

        resp = _upload(c, "key-a", "over.bin", b"y" * 4096)
        assert resp.status_code == 403
        assert "quota" in resp.json()["detail"].lower()
        # Nothing was persisted for the tenant.
        assert store.list_files(owner="key-a") == [
            f for f in store.list_files(owner="key-a") if f["id"] == "file-seed"
        ]
        assert len(store.list_files(owner="key-a")) == 1

    def test_within_quota_upload_succeeds(self, client, monkeypatch):
        """A tenant comfortably under the quota still uploads fine."""
        c, store = client

        quota = 1024 * 1024
        monkeypatch.setattr(files_route, "_MAX_FILE_SIZE", 512 * 1024 * 1024)
        monkeypatch.setenv(files_route._UPLOAD_QUOTA_ENV, str(quota))

        resp = _upload(c, "key-b", "small.txt", b"hello")
        assert resp.status_code == 201
        assert resp.json()["bytes"] == 5
        assert store.get_file(resp.json()["id"])["owner"] == "key-b"

    def test_quota_disabled_when_env_zero(self, client, monkeypatch):
        """DISTLLM_UPLOAD_QUOTA_BYTES=0 disables the tenant quota entirely."""
        c, store = client

        monkeypatch.setattr(files_route, "_MAX_FILE_SIZE", 512 * 1024 * 1024)
        monkeypatch.setenv(files_route._UPLOAD_QUOTA_ENV, "0")

        # Even a huge pre-existing stored total must not block uploads when
        # the quota is disabled.
        store.save_file(
            "file-seed",
            {
                "id": "file-seed",
                "bytes": 10 * 1024 * 1024 * 1024,
                "created_at": 0,
                "filename": "seed.bin",
                "purpose": "batch",
                "status": "uploaded",
                "path": "/nonexistent/seed.bin",
                "object": "file",
                "owner": "key-a",
            },
        )

        resp = _upload(c, "key-a", "ok.txt", b"hello")
        assert resp.status_code == 201
