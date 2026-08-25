"""C5 regression: HA state-replication senders must attach ``X-HA-Secret``.

The receivers (``POST /api/v1/ha/snapshot`` in api/server.py) fail closed:
they 403 any request whose ``X-HA-Secret`` header does not match
``DISTLLM_HA_SECRET``.  Both senders used to POST without the header, so
every push failed with 403 and the failure was logged at debug level only.

Covered here:
- CoordinatorElection._replication_loop attaches the header from env
- ReplicationController._loop attaches the header from env
- No header is sent when DISTLLM_HA_SECRET is unset (receiver fails closed)
- Failed pushes are logged at WARNING (previously debug-only)

HTTP is mocked (no network).
"""

from __future__ import annotations

import threading

import httpx
from loguru import logger

from distllm.core import coordinator_election as ce_mod
from distllm.core import replication_controller as rc_mod
from distllm.core.coordinator_election import CoordinatorElection
from distllm.core.replication_controller import ReplicationController


SECRET = "test-ha-secret-123"


class _WarningSink:
    """Loguru sink that captures WARNING+ records."""

    def __init__(self):
        self.messages: list[str] = []

    def __call__(self, message):
        if message.record["level"].name in ("WARNING", "ERROR", "CRITICAL"):
            self.messages.append(message.record["message"])


class _RecordingClient:
    """httpx.Client double that records POST calls instead of sending."""

    instances: list["_RecordingClient"] = []

    def __init__(self, *args, **kwargs):
        self.posts: list[dict] = []
        self.status_code = 200
        _RecordingClient.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def post(self, url, json=None, headers=None, **kwargs):
        self.posts.append({"url": url, "json": json, "headers": headers or {}})
        return _Resp(self.status_code)


class _Resp:
    def __init__(self, code):
        self.status_code = code


class _FakeCoordinator:
    """Minimal coordinator surface used by CoordinatorElection."""

    model_name = "test-model"
    port = 50050
    nodes: dict = {}
    node_order: list = []

    def __init__(self):
        self._running = threading.Event()
        self._running.set()

        class _Health:
            def is_healthy(self):
                return True

        self._health_mgr = _Health()


def _wait_for_posts(client: _RecordingClient, count: int, timeout: float = 5.0) -> bool:
    deadline = threading.Event()
    for _ in range(int(timeout * 50)):
        if len(client.posts) >= count:
            return True
        deadline.wait(0.02)
    return len(client.posts) >= count


# ── CoordinatorElection sender ────────────────────────────────────────────


def test_election_replication_attaches_ha_secret(monkeypatch):
    monkeypatch.setenv("DISTLLM_HA_SECRET", SECRET)
    monkeypatch.setattr(httpx, "Client", _RecordingClient)
    _RecordingClient.instances.clear()

    election = CoordinatorElection(_FakeCoordinator())
    election.set_replication_peers(["http://peer-a:8000"])

    try:
        assert _wait_for_posts(_RecordingClient.instances[-1], 1)
        post = _RecordingClient.instances[-1].posts[0]
        assert post["url"] == "http://peer-a:8000/api/v1/ha/snapshot"
        # THE fix: receiver demands this header; without it every push 403s.
        assert post["headers"].get("X-HA-Secret") == SECRET
    finally:
        election._coordinator._running.clear()


def test_election_replication_no_header_when_secret_unset(monkeypatch):
    monkeypatch.delenv("DISTLLM_HA_SECRET", raising=False)
    monkeypatch.setattr(httpx, "Client", _RecordingClient)
    _RecordingClient.instances.clear()

    election = CoordinatorElection(_FakeCoordinator())
    election.set_replication_peers(["http://peer-b:8000"])

    try:
        assert _wait_for_posts(_RecordingClient.instances[-1], 1)
        post = _RecordingClient.instances[-1].posts[0]
        assert "X-HA-Secret" not in post["headers"]
    finally:
        election._coordinator._running.clear()


def test_election_replication_failure_logged_at_warning(monkeypatch):
    monkeypatch.setenv("DISTLLM_HA_SECRET", SECRET)
    monkeypatch.setattr(httpx, "Client", _RecordingClient)
    _RecordingClient.instances.clear()
    sink = _WarningSink()
    handler_id = logger.add(sink, level="WARNING")

    election = CoordinatorElection(_FakeCoordinator())
    election.set_replication_peers(["http://peer-c:8000"])

    try:
        client = _RecordingClient.instances[-1]
        assert _wait_for_posts(client, 1)
        # Simulate the receiver's fail-closed rejection.
        client.status_code = 403
        assert _wait_for_posts(client, 2)
        assert any(
            "peer-c" in m for m in sink.messages
        ), (
            "Replication failures must be visible at WARNING level "
            "(were debug-only, so silent 403s went unnoticed)"
        )
    finally:
        logger.remove(handler_id)
        election._coordinator._running.clear()


# ── ReplicationController sender ──────────────────────────────────────────


def _make_controller(monkeypatch, secret):
    if secret is None:
        monkeypatch.delenv("DISTLLM_HA_SECRET", raising=False)
    else:
        monkeypatch.setenv("DISTLLM_HA_SECRET", secret)
    running = threading.Event()
    running.set()
    ctrl = ReplicationController(
        get_snapshot=lambda: {"model_name": "x", "nodes": {}, "node_order": [], "timestamp": 0},
        is_healthy=lambda: True,
        get_node_count=lambda: 0,
        running=running,
        client_factory=_RecordingClient,
    )
    ctrl.set_peers(["http://peer-d:8000"])
    return ctrl


def test_controller_replication_attaches_ha_secret(monkeypatch):
    _RecordingClient.instances.clear()
    ctrl = _make_controller(monkeypatch, SECRET)
    try:
        assert _wait_for_posts(_RecordingClient.instances[-1], 1)
        post = _RecordingClient.instances[-1].posts[0]
        assert post["headers"].get("X-HA-Secret") == SECRET
    finally:
        ctrl._running.clear()


def test_controller_replication_no_header_when_secret_unset(monkeypatch):
    _RecordingClient.instances.clear()
    ctrl = _make_controller(monkeypatch, None)
    try:
        assert _wait_for_posts(_RecordingClient.instances[-1], 1)
        post = _RecordingClient.instances[-1].posts[0]
        assert "X-HA-Secret" not in post["headers"]
    finally:
        ctrl._running.clear()


def test_ha_auth_headers_helper_matches_receiver_contract(monkeypatch):
    """Both senders read the same env var the receiver validates against."""
    monkeypatch.setenv("DISTLLM_HA_SECRET", SECRET)
    assert ce_mod._ha_auth_headers() == {"X-HA-Secret": SECRET}
    assert rc_mod._ha_auth_headers() == {"X-HA-Secret": SECRET}
    monkeypatch.delenv("DISTLLM_HA_SECRET", raising=False)
    assert ce_mod._ha_auth_headers() == {}
    assert rc_mod._ha_auth_headers() == {}
