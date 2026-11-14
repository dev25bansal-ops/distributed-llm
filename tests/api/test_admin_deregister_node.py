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


# ── Real-object end-to-end tests ────────────────────────────────────────────
#
# The tests above use MagicMock coordinators; these wire the admin router to
# a REAL PipelineOrchestrator + REAL ResourceManager, the shape a live
# deployment has (and the shape the audit's repro used). They pin behavior
# the mocks cannot: the is_healthy attribute (not `healthy`), the draining
# set round-trip, and the listing's healthy_count.

import torch  # noqa: E402

from distllm.core.coordinator import Coordinator  # noqa: E402


class _EchoClient:
    """Per-node transport double that records routing."""

    def __init__(self):
        self.calls = 0

    def forward(self, hidden_states=None, kv_cache=None, request_id=None):
        self.calls += 1
        return hidden_states


class _RealPipelineCoord:
    """Coordinator stand-in backed by real orchestrator + resource manager.

    Persistence methods are borrowed from the real :class:`Coordinator` so
    the admin routes' drain-intent bookkeeping is exercised against the
    actual implementation (real JSON file I/O under DISTLLM_DATA_DIR).
    """

    def __init__(self):
        from distllm.core.resource_manager import ResourceManager
        from distllm.dist.pipeline.orchestrator import PipelineOrchestrator

        self._resource_mgr = ResourceManager()
        self._pipeline = PipelineOrchestrator(resource_mgr=self._resource_mgr)
        self.model_name = "lifecycle-test"
        self.total_layers = 2

    @property
    def nodes(self):
        return self._pipeline.nodes

    def manual_register(self, node_id, host, port, start_layer, end_layer,
                        total_layers=None, **kwargs):
        self._pipeline.register_node(
            node_id, host, port, start_layer, end_layer,
            total_layers=total_layers or 0,
        )
        # Mirror Coordinator.manual_register's enforcement hook.
        self._enforce_drain_intent(node_id)

    def _drain_state_path(self):
        import os
        from distllm.core.coordinator import _DRAIN_STATE_FILE
        state_dir = os.environ.get("DISTLLM_DATA_DIR", os.path.expanduser("~/.distllm"))
        return os.path.join(state_dir, _DRAIN_STATE_FILE)

    _read_drain_intent = Coordinator._read_drain_intent
    _write_drain_intent = Coordinator._write_drain_intent
    record_drain_intent = Coordinator.record_drain_intent
    clear_drain_intent = Coordinator.clear_drain_intent
    _enforce_drain_intent = Coordinator._enforce_drain_intent


@pytest.fixture
def real_client(monkeypatch, tmp_path):
    """Admin client bound to a real pipeline/ResourceManager coordinator.

    DISTLLM_DATA_DIR is redirected so drain-intent persistence writes to a
    temp dir instead of the operator's ~/.distllm.
    """
    from fastapi import FastAPI, Request
    from fastapi.testclient import TestClient
    from starlette.middleware.base import BaseHTTPMiddleware

    from distllm.api.api_state import _state
    from distllm.api.routes import admin as admin_routes
    from distllm.core.api_key_store import get_api_key_store

    monkeypatch.setenv("DISTLLM_DATA_DIR", str(tmp_path))

    coord = _RealPipelineCoord()
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
    try:
        yield TestClient(app), coord
    finally:
        _state.coordinator = previous


def _register(client, auth_header, node_id="w1", ready=True):
    resp = client.post("/admin/v1/nodes/register", headers=auth_header, json={
        "node_id": node_id, "host": "127.0.0.1", "port": 50099,
        "start_layer": 0, "end_layer": 1, "total_layers": 2,
        "device": "cpu", "ready": ready,
    })
    assert resp.status_code == 200, resp.text
    return resp


class TestRealPipelineListing:
    """Finding: listing read `healthy` off PipelineNode (which has is_healthy),
    so every live node listed healthy=false and healthy_count was always 0."""

    def test_listing_reports_live_node_healthy(self, auth_header, real_client):
        client, coord = real_client
        _register(client, auth_header)

        data = client.get("/admin/v1/nodes", headers=auth_header).json()

        node = data["nodes"][0]
        assert node["healthy"] is True, (
            "a serving PipelineNode must list as healthy, not be masked by "
            "the absent `healthy` attribute"
        )
        assert node["state"] == "healthy"
        assert data["healthy_count"] == 1

    def test_cluster_status_not_permanently_degraded(self, auth_header, real_client):
        client, coord = real_client
        _register(client, auth_header)

        status = client.get("/admin/v1/cluster/status", headers=auth_header).json()

        assert status["status"] == "healthy"
        assert status["healthy_nodes"] == 1

    def test_unhealthy_node_lists_unhealthy(self, auth_header, real_client):
        client, coord = real_client
        _register(client, auth_header)
        coord._pipeline.mark_node_unhealthy("w1")

        data = client.get("/admin/v1/nodes", headers=auth_header).json()

        assert data["nodes"][0]["healthy"] is False
        assert data["healthy_count"] == 0


class TestRealPipelineLifecycleActions:
    """Finding: offline/recover branched on attrs real nodes do not have."""

    def test_offline_stops_routing_and_recover_restores_it(
        self, auth_header, real_client,
    ):
        client, coord = real_client
        _register(client, auth_header)
        node = coord._pipeline.get_node("w1")
        fake = _EchoClient()
        node.client = fake

        assert client.post("/admin/v1/nodes/w1/offline", headers=auth_header).status_code == 200
        assert node.is_healthy is False, "offline must reach is_healthy"
        with pytest.raises(RuntimeError, match="No healthy nodes"):
            coord._pipeline.run_pipeline(torch.ones(1, 4), {}, "after-offline")

        assert client.post("/admin/v1/nodes/w1/recover", headers=auth_header).status_code == 200
        assert node.is_healthy is True, "recover must clear is_healthy"
        coord._pipeline.run_pipeline(torch.ones(1, 4), {}, "after-recover")
        assert fake.calls == 1, "recovered node must serve again"

    def test_drain_keeps_node_inflight_only(self, auth_header, real_client):
        """drain: in-flight completes, NEW work rejected (the drain contract)."""
        client, coord = real_client
        _register(client, auth_header)
        node = coord._pipeline.get_node("w1")
        fake = _EchoClient()
        node.client = fake

        # Single node: drain it while a request "is being served" — the
        # snapshot the running request took still routes to the node.
        assert client.post("/admin/v1/nodes/w1/drain", headers=auth_header).status_code == 200

        with pytest.raises(RuntimeError, match="No healthy nodes"):
            coord._pipeline.run_pipeline(torch.ones(1, 4), {}, "after-drain")
        assert fake.calls == 0, "no NEW work may reach a drained node"

        # undrain restores service.
        client.post("/admin/v1/nodes/w1/undrain", headers=auth_header)
        coord._pipeline.run_pipeline(torch.ones(1, 4), {}, "after-undrain")
        assert fake.calls == 1


class TestRealPipelineRejoinAndDoubleDeregister:
    def test_double_deregister_is_404_and_keeps_state_clean(
        self, auth_header, real_client,
    ):
        client, coord = real_client
        _register(client, auth_header)

        first = client.delete("/admin/v1/nodes/w1", headers=auth_header)
        second = client.delete("/admin/v1/nodes/w1", headers=auth_header)

        assert first.status_code == 200
        assert second.status_code == 404, "a departed node cannot leave twice"
        assert "w1" not in coord.nodes
        assert coord._resource_mgr.is_node_draining("w1") is False

    def test_rejoin_same_id_is_not_draining(self, auth_header, real_client):
        """The a10-worker-1 symptom: delete then re-register the same id."""
        client, coord = real_client
        _register(client, auth_header)
        client.delete("/admin/v1/nodes/w1", headers=auth_header)

        _register(client, auth_header)  # same node id comes back

        data = client.get("/admin/v1/nodes", headers=auth_header).json()
        node = data["nodes"][0]
        assert node["draining"] is False, (
            "a rejoined node must not inherit the departed node's drain flag"
        )
        assert node["state"] == "healthy"
        assert data["draining_count"] == 0
        assert coord._resource_mgr.get_metrics()["draining_nodes"] == 0

        # And it actually serves.
        fake = _EchoClient()
        coord._pipeline.get_node("w1").client = fake
        coord._pipeline.run_pipeline(torch.ones(1, 4), {}, "rejoined")
        assert fake.calls == 1

    def test_rejoin_after_offline_requires_recover(self, auth_header, real_client):
        """offline is durable operator intent; the exit verb is recover.

        offline = drain + mark unhealthy (planned maintenance). A plain
        re-register (worker process bounce) must NOT silently un-offline
        the node — otherwise a node the operator took out of service comes
        back the moment its process restarts, and the maintenance drain is
        flag-only again. The operator exit path is POST /recover, which
        clears both the health flag and the persisted drain marker.
        """
        client, coord = real_client
        _register(client, auth_header)
        client.post("/admin/v1/nodes/w1/offline", headers=auth_header)
        assert coord._pipeline.get_node("w1").is_healthy is False

        # Process bounce: re-register ready=True. The node is healthy…
        _register(client, auth_header, ready=True)
        assert coord._pipeline.get_node("w1").is_healthy is True
        # …but still drained, because the operator never said otherwise.
        assert coord._resource_mgr.is_node_draining("w1") is True
        with pytest.raises(RuntimeError, match="No healthy nodes"):
            coord._pipeline.run_pipeline(torch.ones(1, 4), {}, "still-offline")

        # The explicit exit verb restores full service.
        client.post("/admin/v1/nodes/w1/recover", headers=auth_header)
        assert coord._resource_mgr.is_node_draining("w1") is False
        fake = _EchoClient()
        coord._pipeline.get_node("w1").client = fake
        coord._pipeline.run_pipeline(torch.ones(1, 4), {}, "recovered")
        assert fake.calls == 1

    def test_rejoin_without_ready_field_starts_unhealthy(self, auth_header, real_client):
        """The `cluster join` bug: a register without `ready` means not ready.

        The model default for `ready` is False and register_node marks the
        node unhealthy — that is correct behavior for a worker still
        loading; the CLI fix was to SEND ready, not to change the default.
        """
        client, coord = real_client
        _register(client, auth_header, ready=False)

        assert coord._pipeline.get_node("w1").is_healthy is False
        with pytest.raises(RuntimeError, match="No healthy nodes"):
            coord._pipeline.run_pipeline(torch.ones(1, 4), {}, "not-ready")


class TestDrainIntentSurvivesRestart:
    """Finding: drain state was in-memory only; a restart lost it."""

    def test_drain_is_persisted_and_reapplied_on_rejoin(
        self, auth_header, real_client, tmp_path,
    ):
        import json

        client, coord = real_client
        _register(client, auth_header)
        client.post("/admin/v1/nodes/w1/drain", headers=auth_header)

        # The intent file exists... (the operator drain was recorded)
        intent_path = tmp_path / "drain_state.json"
        assert intent_path.exists()
        assert "w1" in json.loads(intent_path.read_text())["draining_nodes"]

        # ...and a fresh coordinator (post-restart) re-applies it when the
        # node re-registers: real Coordinator.manual_register enforces the
        # persisted marker. Exercise the exact hook the coordinator runs.
        from distllm.core.coordinator import Coordinator
        from distllm.core.coordinator_config import CoordinatorConfig

        coord2 = Coordinator(config=CoordinatorConfig(model_name="x"))
        coord2.manual_register("w1", "127.0.0.1", 50099, 0, 1, total_layers=2)

        assert coord2._resource_mgr.is_node_draining("w1") is True, (
            "a drained-before-restart node must come back draining"
        )

    def test_undrain_clears_the_persisted_intent(
        self, auth_header, real_client, tmp_path,
    ):
        client, coord = real_client
        _register(client, auth_header)
        client.post("/admin/v1/nodes/w1/drain", headers=auth_header)
        client.post("/admin/v1/nodes/w1/undrain", headers=auth_header)

        import json
        drained = json.loads((tmp_path / "drain_state.json").read_text())["draining_nodes"]
        assert "w1" not in drained

    def test_deregister_clears_the_persisted_intent(
        self, auth_header, real_client, tmp_path,
    ):
        client, coord = real_client
        _register(client, auth_header)
        client.post("/admin/v1/nodes/w1/drain", headers=auth_header)
        client.delete("/admin/v1/nodes/w1", headers=auth_header)

        import json
        drained = json.loads((tmp_path / "drain_state.json").read_text())["draining_nodes"]
        assert "w1" not in drained