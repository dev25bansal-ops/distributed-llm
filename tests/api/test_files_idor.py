"""SEC-API-01: Files IDOR — owner+role enforcement on /v1/files.

Two-key PoC: key A uploads a file; key B must not be able to list,
retrieve, download, or delete it (404, existence not leaked).  Admin
role bypasses the owner check.  Files saved before the ``owner`` field
existed remain accessible (backward-compatible backfill).
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from distllm.api.persistent_store import PersistentStore
from distllm.api.routes.files import router as files_router


def _make_client(monkeypatch, tmp_path, store: PersistentStore) -> TestClient:
    """Build a TestClient with files router; require_coordinator is stubbed
    out (its import lives inside the dependency and reads api_state)."""
    monkeypatch.setattr(
        "distllm.api.routes.files.get_store",
        lambda: store,
    )
    monkeypatch.setattr(
        "distllm.api.routes.files.get_data_dir",
        lambda: tmp_path,
    )
    # Neutralize the coordinator gate: the dependency factory is called at
    # module import, and the inner checker reads api_state at request time.
    # Patch the live dependency callable's closure-free behavior by replacing
    # the api_state global coordinator with a dummy object.
    monkeypatch.setattr(
        "distllm.api.api_state.g.coordinator",
        object(),
    )
    # Neutralize the coordinator gate: clear the router-level dependency and
    # re-add nothing (auth is simulated via middleware below).
    files_router.dependencies.clear()

    app = FastAPI()

    @app.middleware("http")
    async def _identity(request, call_next):
        # Starlette request.state is plain attribute storage; assignment is
        # fine, but reading via getattr in route handlers must see these.
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


def _upload(client: TestClient, key_id: str, filename: str = "a.txt", role: str = "inference-only"):
    return client.post(
        "/v1/files",
        files={"file": (filename, b"hello")},
        data={"purpose": "batch"},
        headers={"x-key-id": key_id, "x-key-role": role},
    )


class TestFilesIDOR:
    def test_owner_stamped_on_upload(self, client):
        c, store = client
        resp = _upload(c, "key-a")
        assert resp.status_code == 201
        file_id = resp.json()["id"]
        meta = store.get_file(file_id)
        assert meta is not None and meta["owner"] == "key-a"

    def test_two_key_list_isolation(self, client):
        c, _ = client
        _upload(c, "key-a", "a.txt")
        _upload(c, "key-b", "b.txt")
        ra = c.get("/v1/files", headers={"x-key-id": "key-a"})
        rb = c.get("/v1/files", headers={"x-key-id": "key-b"})
        assert ra.status_code == 200 and rb.status_code == 200
        assert {f["filename"] for f in ra.json()["data"]} == {"a.txt"}
        assert {f["filename"] for f in rb.json()["data"]} == {"b.txt"}

    def test_cross_key_get_content_delete_404(self, client):
        c, _ = client
        file_id = _upload(c, "key-a").json()["id"]
        other = {"x-key-id": "key-b"}
        assert c.get(f"/v1/files/{file_id}", headers=other).status_code == 404
        assert c.get(f"/v1/files/{file_id}/content", headers=other).status_code == 404
        assert c.delete(f"/v1/files/{file_id}", headers=other).status_code == 404
        # Owner still has full access.
        assert c.get(f"/v1/files/{file_id}", headers={"x-key-id": "key-a"}).status_code == 200
        assert c.get(f"/v1/files/{file_id}/content", headers={"x-key-id": "key-a"}).status_code == 200

    def test_admin_bypasses_owner_check(self, client):
        c, _ = client
        file_id = _upload(c, "key-a").json()["id"]
        admin = {"x-key-id": "key-admin", "x-key-role": "admin"}
        assert c.get(f"/v1/files/{file_id}", headers=admin).status_code == 200
        assert c.get(f"/v1/files/{file_id}/content", headers=admin).status_code == 200
        # Admin list sees all files.
        listed = c.get("/v1/files", headers=admin)
        assert listed.status_code == 200
        assert any(f["id"] == file_id for f in listed.json()["data"])
        assert c.delete(f"/v1/files/{file_id}", headers=admin).status_code == 200

    def test_legacy_files_without_owner_remain_accessible(self, client):
        c, store = client
        store.save_file(
            "file-legacy",
            {
                "id": "file-legacy",
                "bytes": 5,
                "created_at": 0,
                "filename": "legacy.txt",
                "purpose": "batch",
                "status": "uploaded",
                "path": "/nonexistent",
                "object": "file",
                # no "owner" field — pre-backfill row
            },
        )
        resp = c.get("/v1/files/file-legacy", headers={"x-key-id": "key-b"})
        assert resp.status_code == 200
