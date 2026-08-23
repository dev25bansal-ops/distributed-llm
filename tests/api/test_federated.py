"""Tests for federated training API routes — POST/DELETE /v1/federated/*.

The federated router (prefix ``/v1/federated``) uses ``g.get("federated_merge")``
to retrieve the coordinator.  Tests inject a plain-object mock via
``monkeypatch`` on ``g.get`` so the route handlers receive a real, callable
object with the expected interface (no ``MagicMock`` anywhere).
"""

from __future__ import annotations

import secrets

import pytest
from fastapi.testclient import TestClient

from distllm.api.api_state import g, reset_app_state_for_testing
from distllm.api.circuit_breaker_middleware import (
    CircuitState,
    _breaker as _circuit_breaker,
)
from distllm.api.server import app
from distllm.core.api_key_store import reset_api_key_store


# ───────────────────────────────────────────────────────────────────────
# Mock classes — plain objects, no MagicMock / Mock / AsyncMock
# ───────────────────────────────────────────────────────────────────────


class _NodeState:
    """Return value from coordinator.register_node()."""

    def __init__(self, node_id: str, status: str, dataset_size: int) -> None:
        self.node_id = node_id
        self.status = status
        self.dataset_size = dataset_size


class _RoundInfo:
    """Return value from coordinator.start_round()."""

    def __init__(
        self,
        round_id: str,
        round_number: int,
        participating_nodes: list[str],
        status: str,
    ) -> None:
        self.round_id = round_id
        self.round_number = round_number
        self.participating_nodes = participating_nodes
        self.status = status


class _VersionInfo:
    """Version object returned by coordinator.get_versions()."""

    def __init__(
        self,
        version_id: str,
        round_number: int,
        path: str,
        metrics: dict,
        created_at: str,
    ) -> None:
        self.version_id = version_id
        self.round_number = round_number
        self.path = path
        self.metrics = metrics
        self.created_at = created_at


class FederatedCoordinatorMock:
    """Plain-object mock for the federated merge coordinator.

    Attributes
    ----------
    _nodes : dict
        Internal registry so ``register_node`` / ``unregister_node`` have
        observable side effects.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, _NodeState] = {}

    def register_node(
        self,
        node_id: str,
        dataset_size: int = 0,
        local_epochs: int = 3,
        learning_rate: float = 2e-4,
    ) -> _NodeState:
        state = _NodeState(node_id, "registered", dataset_size)
        self._nodes[node_id] = state
        return state

    def unregister_node(self, node_id: str) -> None:
        self._nodes.pop(node_id, None)

    def start_round(self) -> _RoundInfo:
        return _RoundInfo(
            round_id="round-1",
            round_number=1,
            participating_nodes=list(self._nodes.keys()),
            status="in_progress",
        )

    def submit_node_adapter(
        self,
        node_id: str,
        adapter_path: str,
        loss: float,
        dataset_size: int = 0,
    ) -> bool:
        return True

    def merge_adapters(self) -> str:
        return "/path/to/merged/adapter"

    def get_stats(self) -> dict:
        return {
            "total_rounds": 5,
            "registered_nodes": len(self._nodes),
            "active_nodes": len(self._nodes),
            "total_versions": 10,
            "merge_strategy": "fedavg",
            "current_round": 1,
            "current_round_status": "in_progress",
            "avg_loss_last_round": 0.123,
        }

    def get_versions(self) -> list[_VersionInfo]:
        return [
            _VersionInfo(
                version_id="v1",
                round_number=1,
                path="/path/v1",
                metrics={"loss": 0.5},
                created_at="2024-01-01T00:00:00Z",
            ),
        ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _err_detail(body: dict) -> str:
    """Extract the human-readable error detail from an HTTPException response.

    The global ``http_exception_handler`` wraps errors as::

        {"error": {"message": "...", "type": "http_error", ...}, "request_id": "..."}
    """
    return body["error"]["message"]


# ───────────────────────────────────────────────────────────────────────
# Fixtures
# ───────────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clean_state() -> None:
    """Reset shared state and circuit breaker before every test.

    The circuit breaker tracks 5XX responses; tests that deliberately trigger
    503/400 responses would otherwise open the breaker and cause cascade
    failures in later tests.  Resetting both ``AppState`` and the breaker
    ensures each test starts with a clean slate.
    """
    import os as _os
    _os.environ.pop("API_KEYS", None)
    reset_app_state_for_testing()
    reset_api_key_store()
    _circuit_breaker._state = CircuitState.CLOSED
    _circuit_breaker._failures = 0
    _circuit_breaker._successes = 0
    _circuit_breaker._last_failure_time = 0.0
    _circuit_breaker._half_open_calls = 0
    _circuit_breaker._recent_results.clear()


@pytest.fixture
def admin_key() -> str:
    """Generate a fresh admin API key for a single test."""
    return secrets.token_hex(32)


@pytest.fixture
def fed_coord() -> FederatedCoordinatorMock:
    """Create a fresh federated coordinator mock for each test."""
    return FederatedCoordinatorMock()


@pytest.fixture
def client(
    admin_key: str,
    fed_coord: FederatedCoordinatorMock,
    monkeypatch,
) -> TestClient:
    """Build a ``TestClient`` authenticated with an admin API key.

    Also monkeypatches ``g.get`` so that ``g.get("federated_merge")`` returns
    the ``fed_coord`` mock.  All other ``g.get`` calls fall through to the
    original implementation.
    """
    # Let the key store pick up our test key
    monkeypatch.setenv("API_KEY", admin_key)
    reset_api_key_store()

    # Inject the federated coordinator via g.get(...)
    original_get = g.get
    monkeypatch.setattr(
        g,
        "get",
        lambda key, default=None: (
            fed_coord if key == "federated_merge" else original_get(key, default)
        ),
    )

    return TestClient(app)


# ---------------------------------------------------------------------------
# Auth header helper
# ---------------------------------------------------------------------------


def _auth(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


# ===================================================================
# POST /v1/federated/nodes — register_node (admin)
# ===================================================================


class TestRegisterNode:
    def test_success(self, client: TestClient, admin_key: str) -> None:
        resp = client.post(
            "/v1/federated/nodes",
            json={"node_id": "n1", "dataset_size": 500},
            headers=_auth(admin_key),
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["node_id"] == "n1"
        assert body["status"] == "registered"
        assert body["dataset_size"] == 500

    def test_no_coordinator(
        self, client: TestClient, admin_key: str, monkeypatch
    ) -> None:
        monkeypatch.setattr(g, "get", lambda key, default=None: None)
        resp = client.post(
            "/v1/federated/nodes",
            json={"node_id": "n1"},
            headers=_auth(admin_key),
        )
        assert resp.status_code == 503
        assert _err_detail(resp.json()) == "Federated training not available"


# ===================================================================
# DELETE /v1/federated/nodes/{node_id} — unregister_node (admin)
# ===================================================================


class TestUnregisterNode:
    def test_success(self, client: TestClient, admin_key: str) -> None:
        resp = client.delete("/v1/federated/nodes/n1", headers=_auth(admin_key))
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "removed"
        assert body["node_id"] == "n1"

    def test_no_coordinator(
        self, client: TestClient, admin_key: str, monkeypatch
    ) -> None:
        monkeypatch.setattr(g, "get", lambda key, default=None: None)
        resp = client.delete("/v1/federated/nodes/n1", headers=_auth(admin_key))
        assert resp.status_code == 503
        assert _err_detail(resp.json()) == "Federated training not available"


# ===================================================================
# POST /v1/federated/rounds — start_round (admin)
# ===================================================================


class TestStartRound:
    def test_success(
        self,
        client: TestClient,
        admin_key: str,
        fed_coord: FederatedCoordinatorMock,
    ) -> None:
        # Pre-register a node so participating_nodes is non-empty
        fed_coord.register_node("n1", dataset_size=100)
        resp = client.post("/v1/federated/rounds", headers=_auth(admin_key))
        assert resp.status_code == 200
        body = resp.json()
        assert body["round_id"] == "round-1"
        assert body["round_number"] == 1
        assert body["participating_nodes"] == ["n1"]
        assert body["status"] == "in_progress"

    def test_not_enough_nodes(
        self, client: TestClient, admin_key: str, monkeypatch
    ) -> None:
        bad = FederatedCoordinatorMock()
        bad.start_round = lambda: None  # type: ignore[method-assign]
        monkeypatch.setattr(
            g,
            "get",
            lambda key, d=None: bad if key == "federated_merge" else None,
        )
        resp = client.post("/v1/federated/rounds", headers=_auth(admin_key))
        assert resp.status_code == 400
        assert _err_detail(resp.json()) == "Not enough nodes to start round"

    def test_no_coordinator(
        self, client: TestClient, admin_key: str, monkeypatch
    ) -> None:
        monkeypatch.setattr(g, "get", lambda key, default=None: None)
        resp = client.post("/v1/federated/rounds", headers=_auth(admin_key))
        assert resp.status_code == 503
        assert _err_detail(resp.json()) == "Federated training not available"


# ===================================================================
# POST /v1/federated/rounds/submit — submit_adapter (admin)
# ===================================================================


class TestSubmitAdapter:
    def test_success(self, client: TestClient, admin_key: str) -> None:
        resp = client.post(
            "/v1/federated/rounds/submit",
            json={
                "node_id": "n1",
                "adapter_path": "/tmp/adapter.bin",
                "loss": 0.42,
            },
            headers=_auth(admin_key),
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "submitted"
        assert body["node_id"] == "n1"

    def test_not_accepted(
        self, client: TestClient, admin_key: str, monkeypatch
    ) -> None:
        bad = FederatedCoordinatorMock()
        bad.submit_node_adapter = (  # type: ignore[method-assign]
            lambda node_id, adapter_path, loss, dataset_size=0: False
        )
        monkeypatch.setattr(
            g,
            "get",
            lambda key, d=None: bad if key == "federated_merge" else None,
        )
        resp = client.post(
            "/v1/federated/rounds/submit",
            json={
                "node_id": "n1",
                "adapter_path": "/tmp/adapter.bin",
                "loss": 0.42,
            },
            headers=_auth(admin_key),
        )
        assert resp.status_code == 400
        assert _err_detail(resp.json()) == "Adapter not accepted"

    def test_no_coordinator(
        self, client: TestClient, admin_key: str, monkeypatch
    ) -> None:
        monkeypatch.setattr(g, "get", lambda key, default=None: None)
        resp = client.post(
            "/v1/federated/rounds/submit",
            json={
                "node_id": "n1",
                "adapter_path": "/tmp/adapter.bin",
                "loss": 0.42,
            },
            headers=_auth(admin_key),
        )
        assert resp.status_code == 503
        assert _err_detail(resp.json()) == "Federated training not available"


# ===================================================================
# POST /v1/federated/rounds/merge — merge_adapters (admin)
# ===================================================================


class TestMergeAdapters:
    def test_success(self, client: TestClient, admin_key: str) -> None:
        resp = client.post("/v1/federated/rounds/merge", headers=_auth(admin_key))
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "merged"
        assert body["path"] == "/path/to/merged/adapter"

    def test_failed(self, client: TestClient, admin_key: str, monkeypatch) -> None:
        bad = FederatedCoordinatorMock()
        bad.merge_adapters = lambda: None  # type: ignore[method-assign]
        monkeypatch.setattr(
            g,
            "get",
            lambda key, d=None: bad if key == "federated_merge" else None,
        )
        resp = client.post("/v1/federated/rounds/merge", headers=_auth(admin_key))
        assert resp.status_code == 400
        assert _err_detail(resp.json()) == "Merge failed"

    def test_no_coordinator(
        self, client: TestClient, admin_key: str, monkeypatch
    ) -> None:
        monkeypatch.setattr(g, "get", lambda key, default=None: None)
        resp = client.post("/v1/federated/rounds/merge", headers=_auth(admin_key))
        assert resp.status_code == 503
        assert _err_detail(resp.json()) == "Federated training not available"


# ===================================================================
# GET /v1/federated/stats — get_federated_stats (no role restriction)
# ===================================================================


class TestGetFederatedStats:
    def test_success(
        self,
        client: TestClient,
        admin_key: str,
        fed_coord: FederatedCoordinatorMock,
    ) -> None:
        fed_coord.register_node("n1", dataset_size=500)
        resp = client.get("/v1/federated/stats", headers=_auth(admin_key))
        assert resp.status_code == 200
        body = resp.json()
        assert body["total_rounds"] == 5
        assert body["registered_nodes"] == 1
        assert body["active_nodes"] == 1
        assert body["total_versions"] == 10
        assert body["merge_strategy"] == "fedavg"
        assert body["current_round"] == 1
        assert body["current_round_status"] == "in_progress"
        assert body["avg_loss_last_round"] == 0.123

    def test_no_coordinator(
        self, client: TestClient, admin_key: str, monkeypatch
    ) -> None:
        monkeypatch.setattr(g, "get", lambda key, default=None: None)
        resp = client.get("/v1/federated/stats", headers=_auth(admin_key))
        assert resp.status_code == 503
        assert _err_detail(resp.json()) == "Federated training not available"


# ===================================================================
# GET /v1/federated/versions — list_versions (no role restriction)
# ===================================================================


class TestListVersions:
    def test_success(self, client: TestClient, admin_key: str) -> None:
        resp = client.get("/v1/federated/versions", headers=_auth(admin_key))
        assert resp.status_code == 200
        body = resp.json()
        assert len(body) == 1
        item = body[0]
        assert item["version_id"] == "v1"
        assert item["round_number"] == 1
        assert item["path"] == "/path/v1"
        assert item["metrics"] == {"loss": 0.5}
        assert item["created_at"] == "2024-01-01T00:00:00Z"

    def test_no_coordinator(
        self, client: TestClient, admin_key: str, monkeypatch
    ) -> None:
        monkeypatch.setattr(g, "get", lambda key, default=None: None)
        resp = client.get("/v1/federated/versions", headers=_auth(admin_key))
        assert resp.status_code == 503
        assert _err_detail(resp.json()) == "Federated training not available"
