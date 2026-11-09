"""Prompt version history: GET /v1/prompts/{prompt_id}/versions.

The prompts router carries its own ``prefix="/v1/prompts"``; the scratch app
below includes it without an additional prefix.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from distllm.api.routes import prompts as prompts_module
from distllm.api.routes.prompts import router


def _reset_state() -> None:
    prompts_module._prompts.clear()
    prompts_module._share_tokens.clear()
    prompts_module._version_history.clear()
    prompts_module._seeded = False


@pytest.fixture()
def client():
    _reset_state()
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as c:
        yield c
    _reset_state()


def _create(client, name="p1", prompt="hello"):
    resp = client.post(
        "/v1/prompts",
        json={"name": name, "category": "general", "prompt": prompt},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _update(client, prompt_id, name="p1", prompt="hello v2"):
    resp = client.put(
        f"/v1/prompts/{prompt_id}",
        json={"name": name, "category": "general", "prompt": prompt},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def _versions(client, prompt_id):
    resp = client.get(f"/v1/prompts/{prompt_id}/versions")
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_update_does_not_duplicate_creation_snapshot(client):
    created = _create(client, prompt="v1 content")
    updated = _update(client, created["id"], prompt="v2 content")

    # Update response still reports version 2.
    assert updated["version"] == 2
    assert updated["prompt"] == "v2 content"

    versions = _versions(client, created["id"])
    # After one update the creation snapshot v1 must appear exactly once.
    assert [(v["version"], v["prompt"]) for v in versions] == [(1, "v1 content")]

    version_numbers = [v["version"] for v in versions]
    assert len(version_numbers) == len(set(version_numbers))


def test_restore_records_each_version_once(client):
    created = _create(client, prompt="v1 content")
    _update(client, created["id"], prompt="v2 content")

    restore = client.post(f"/v1/prompts/{created['id']}/versions/1")
    assert restore.status_code == 200, restore.text
    body = restore.json()
    assert body["version"] == 1
    assert body["prompt"] == "v1 content"

    versions = _versions(client, created["id"])
    pairs = [(v["version"], v["prompt"]) for v in versions]
    # v1 exactly once, v2 exactly once (v2 was the state restored away from).
    assert pairs.count((1, "v1 content")) == 1
    assert pairs.count((2, "v2 content")) == 1
    version_numbers = [v["version"] for v in versions]
    assert len(version_numbers) == len(set(version_numbers))


def test_restore_twice_and_update_after_restore_no_duplicates(client):
    created = _create(client, prompt="v1 content")
    _update(client, created["id"], prompt="v2 content")

    first = client.post(f"/v1/prompts/{created['id']}/versions/1")
    assert first.status_code == 200, first.text
    second = client.post(f"/v1/prompts/{created['id']}/versions/1")
    assert second.status_code == 200, second.text
    assert second.json()["version"] == 1
    assert second.json()["prompt"] == "v1 content"

    # Updating after a restore must not re-append the restored v1 snapshot.
    updated = _update(client, created["id"], prompt="v3 content")
    assert updated["version"] == 2

    versions = _versions(client, created["id"])
    version_numbers = [v["version"] for v in versions]
    assert len(version_numbers) == len(set(version_numbers)), (
        f"duplicate versions in history: {version_numbers}"
    )
    pairs = [(v["version"], v["prompt"]) for v in versions]
    assert pairs.count((1, "v1 content")) == 1
    assert pairs.count((2, "v2 content")) == 1
