"""A node must be able to leave the cluster.

`cluster leave` used to call only ``POST /admin/v1/nodes/{id}/drain``. Drain
flips a flag in ``ResourceManager._draining_nodes`` and offline marks the node
unhealthy, but nothing removes the node from ``coord.nodes`` -- and nothing
removes an id from ``_draining_nodes``. A node that had gone therefore stayed
in ``GET /admin/v1/nodes`` as ``draining`` for the life of the coordinator,
which is what happened on the A10: after a full process restart the listing
still reported ``a10-worker-1`` as draining, and ``cluster leave`` could not
clear it because no delete route existed at all.

``DELETE /admin/v1/nodes/{node_id}`` is that missing verb. Auth here mirrors
``test_admin_api.py``: the admin router is role-gated at router level, so the
middleware below has to run for any of these to reach the handler.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from starlette.middleware.base import BaseHTTPMiddleware

from distllm.core.api_key_store import get_api_key_store, reset_api_key_store

_TEST_ADMIN_KEY = "test-admin-key-deregister"


@pytest.fixture(autouse=True)
def _reset_store(monkeypatch):
    monkeypatch.setenv(
        "API_KEYS",
        '{"keys": [{"key": "' + _TEST_ADMIN_KEY + '", "role": "admin", "label": "test-admin"}]}',
    )
    reset_api_key_store()
    yield
    reset_api_key_store()


@pytest.fixture
def auth_header():
    return {"Authorization": f"Bearer {_TEST_ADMIN_KEY}"}


@pytest.fixture
def make_client():
    """Build a client bound to a coordinator, the way the admin routes read it.

    ``_resolve_coordinator`` reads ``api_state._state.coordinator`` directly
    rather than through ``Depends``, so ``dependency_overrides`` does nothing
    here and the global has to be patched, as ``test_admin_api.py`` does.
    """
    from distllm.api.api_state import _state

    def _make(nodes: dict):
        from distllm.api.routes import admin as admin_routes

        coord = MagicMock()
        coord.nodes = dict(nodes)
        coord._resource_mgr = MagicMock()
        coord._pipeline = MagicMock()

        app = FastAPI()

        class _TestAuthMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request: Request, call_next):
                auth = request.headers.get("Authorization", "")
                if auth.startswith("Bearer "):
                    result = get_api_key_store().authenticate(auth[7:])
                    if result:
                        request.state.api_key_role = result[1]
                        request.state.api_key_id = result[0]
                return await call_next(request)

        app.add_middleware(_TestAuthMiddleware)
        app.include_router(admin_routes.router)

        previous = _state.coordinator
        _state.coordinator = coord
        return TestClient(app), coord, previous

    created = []

    def _tracked(nodes: dict):
        client, coord, previous = _make(nodes)
        created.append(previous)
        return client, coord

    yield _tracked
    _state.coordinator = created[-1] if created else None


class TestDeregisterNode:
    def test_deregister_removes_the_node(self, auth_header, make_client):
        client, coord = make_client({"w1": object()})

        resp = client.delete("/admin/v1/nodes/w1", headers=auth_header)

        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        assert "w1" not in coord.nodes, "node must be gone from the pool"

    def test_deregister_drains_before_removing(self, auth_header, make_client):
        """In-flight work should be allowed to finish, so drain is applied too."""
        client, coord = make_client({"w1": object()})

        client.delete("/admin/v1/nodes/w1", headers=auth_header)

        coord._resource_mgr.mark_node_draining.assert_called_once_with("w1")

    def test_deregister_tells_the_pipeline(self, auth_header, make_client):
        """A request must not be routed to a node that has left."""
        client, coord = make_client({"w1": object()})

        client.delete("/admin/v1/nodes/w1", headers=auth_header)

        coord._pipeline.remove_node.assert_called_once_with("w1")

    def test_deregister_unknown_node_is_404(self, auth_header, make_client):
        client, _ = make_client({})

        resp = client.delete("/admin/v1/nodes/ghost", headers=auth_header)

        assert resp.status_code == 404

    def test_deregistered_node_disappears_from_listing(self, auth_header, make_client):
        """The listing is what `cluster list-nodes` shows; it must agree."""
        client, _ = make_client({"w1": object(), "w2": object()})

        client.delete("/admin/v1/nodes/w1", headers=auth_header)
        listing = client.get("/admin/v1/nodes", headers=auth_header).json()

        ids = {n["node_id"] for n in listing["nodes"]}
        assert "w1" not in ids
        assert "w2" in ids

    def test_deregister_clears_the_drain_entry(self, auth_header, make_client):
        """Deregister must not leave the id draining forever.

        It used to call mark_node_draining but never mark_node_alive, so a
        node that left and rejoined with the same id reported as draining
        again and the drain gauge never returned to zero."""
        client, coord = make_client({"w1": object()})

        client.delete("/admin/v1/nodes/w1", headers=auth_header)

        coord._resource_mgr.mark_node_alive.assert_called_once_with("w1")

    def test_deregister_survives_a_broken_drain_cleanup(self, auth_header, make_client):
        client, coord = make_client({"w1": object()})
        coord._resource_mgr.mark_node_alive.side_effect = RuntimeError("boom")

        resp = client.delete("/admin/v1/nodes/w1", headers=auth_header)

        assert resp.status_code == 200
        assert "w1" not in coord.nodes


class TestOfflineAndRecoverUseTheRealAttribute:
    """offline/recover branched on attributes real nodes do not have.

    Live pipeline nodes carry ``is_healthy`` (PipelineNode); the old code
    checked ``isinstance(node, dict)`` / ``hasattr(node, 'healthy')``, so on a
    real deployment offline was a silent no-op and recover could not clear
    is_healthy — a recovered node stayed unroutable.
    """

    def test_offline_marks_pipeline_node_unhealthy(self, auth_header, make_client):
        from distllm.dist.pipeline.orchestrator import PipelineNode

        node = PipelineNode(node_id="w1", host="h", port=1, start_layer=0, end_layer=1)
        node.is_healthy = True
        client, coord = make_client({"w1": node})
        # Real pipeline method must do the work, not a MagicMock side effect.
        coord._pipeline.mark_node_unhealthy.side_effect = None
        coord._pipeline.mark_node_unhealthy = lambda nid: setattr(node, "is_healthy", False)

        resp = client.post("/admin/v1/nodes/w1/offline", headers=auth_header)

        assert resp.status_code == 200
        assert node.is_healthy is False, "offline must reach the real attribute"

    def test_recover_marks_pipeline_node_healthy(self, auth_header, make_client):
        from distllm.dist.pipeline.orchestrator import PipelineNode

        node = PipelineNode(node_id="w1", host="h", port=1, start_layer=0, end_layer=1)
        node.is_healthy = False
        client, coord = make_client({"w1": node})
        coord._pipeline.mark_node_healthy.side_effect = None
        coord._pipeline.mark_node_healthy = lambda nid: setattr(node, "is_healthy", True)

        resp = client.post("/admin/v1/nodes/w1/recover", headers=auth_header)

        assert resp.status_code == 200
        assert node.is_healthy is True, "recover must clear the real attribute"

    def test_delete_requires_auth(self, make_client):
        """Removing a node is an admin action and must not be anonymous."""
        client, coord = make_client({"w1": object()})

        resp = client.delete("/admin/v1/nodes/w1")

        assert resp.status_code in (401, 403)
        assert "w1" in coord.nodes

    @pytest.mark.parametrize("node_id", ["a10-worker-1", "node_ab12cd34"])
    def test_delete_route_is_registered(self, auth_header, make_client, node_id):
        client, _ = make_client({node_id: object()})
        assert client.delete(f"/admin/v1/nodes/{node_id}", headers=auth_header).status_code == 200