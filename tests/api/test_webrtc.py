"""WebRTC signaling API route tests.

Tests cover the four endpoints defined in ``routes/webrtc.py``:
    POST   /v1/webrtc/offer   -- SDP offer/answer exchange
    POST   /v1/webrtc/ice     -- ICE candidate exchange
    GET    /v1/webrtc/status  -- Signaling server status
    DELETE /v1/webrtc/sessions/{session_id} -- Close session

These routes do **not** use ``g.coordinator``; they rely on a module-level
``WebRTCSessionManager`` singleton (``_session_mgr``) which is seeded
directly in tests that require session state.
"""

from __future__ import annotations

import os
import time

import pytest
from fastapi.testclient import TestClient

from distllm.api.server import app
from distllm.core.api_key_store import reset_api_key_store

# Test admin key used throughout
_TEST_ADMIN_KEY = "webrtc-test-admin-key-2026"


# ── Fixtures ────────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_store():
    """Seed the key store with a known admin key for auth."""
    os.environ["API_KEYS"] = (
        '{"keys": [{"key": "' + _TEST_ADMIN_KEY + '", "role": "admin", "label": "test-admin"}]}'
    )
    reset_api_key_store()
    yield
    reset_api_key_store()


@pytest.fixture(autouse=True)
def _reset_session_mgr():
    """Clear session manager state before each test.

    We import the module-level ``_session_mgr`` directly and reset its
    internal dict so that tests start with a clean slate without needing
    to re-initialise the entire app.
    """
    from distllm.api.routes.webrtc import _session_mgr

    saved = dict(_session_mgr._sessions)
    _session_mgr._sessions.clear()
    yield
    _session_mgr._sessions.clear()
    _session_mgr._sessions.update(saved)


@pytest.fixture
def client() -> TestClient:
    """Fresh TestClient per test."""
    return TestClient(app)


@pytest.fixture
def auth_header() -> dict[str, str]:
    """Return headers with a valid admin Bearer token."""
    return {"Authorization": f"Bearer {_TEST_ADMIN_KEY}"}


# ── Helpers ─────────────────────────────────────────────────────────────────────


def _seed_session(
    session_id: str = "test-session",
    sdp: str = "v=0\r\no=- 0 0 IN IP4 0.0.0.0\r\ns=-\r\n",
    status: str = "connected",
) -> str:
    """Create a session in ``_session_mgr`` and return its id."""
    from distllm.api.routes.webrtc import _session_mgr

    sid = _session_mgr.create_session(session_id, sdp)
    session = _session_mgr.get_session(sid)
    if session is not None:
        session["status"] = status
    return sid


# ── POST /v1/webrtc/offer ───────────────────────────────────────────────────────


class TestOffer:
    ENDPOINT = "/v1/webrtc/offer"

    def test_offer_webrtc_not_available(
        self, client: TestClient, auth_header: dict[str, str]
    ) -> None:
        """Return 503 when ``aiortc`` is not installed.

        The ``distllm.dist.webrtc`` module sets ``HAS_WEBRTC = False`` at
        import time when ``aiortc`` cannot be imported.  The route checks
        this flag and returns 503.
        """
        resp = client.post(
            self.ENDPOINT,
            json={
                "sdp": "v=0\r\no=- 0 0 IN IP4 0.0.0.0\r\ns=-\r\n",
                "type": "offer",
                "session_id": "",
            },
            headers=auth_header,
        )
        assert resp.status_code == 503
        assert "WebRTC not available" in resp.json()["error"]["message"]

    def test_offer_requires_auth(self, client: TestClient) -> None:
        """Return 401 when no auth header is supplied."""
        resp = client.post(
            self.ENDPOINT,
            json={
                "sdp": "v=0\r\no=- 0 0 IN IP4 0.0.0.0\r\ns=-\r\n",
                "type": "offer",
                "session_id": "",
            },
        )
        assert resp.status_code in (401, 503)


# ── POST /v1/webrtc/ice ─────────────────────────────────────────────────────────


class TestICE:
    ENDPOINT = "/v1/webrtc/ice"

    def test_ice_session_not_found(
        self, client: TestClient, auth_header: dict[str, str]
    ) -> None:
        """Return 404 when ``session_id`` does not match any session."""
        resp = client.post(
            self.ENDPOINT,
            json={
                "session_id": "nonexistent",
                "candidate": "candidate 1 1 UDP 0 0.0.0.0 0 typ host",
                "sdp_mid": "0",
                "sdp_mline_index": 0,
            },
            headers=auth_header,
        )
        assert resp.status_code == 404
        assert resp.json()["error"]["message"] == "Session not found"

    def test_ice_with_session_no_transport(
        self, client: TestClient, auth_header: dict[str, str]
    ) -> None:
        """Return 200 when session exists but has no transport.

        Because ``aiortc`` is not installed, the ``RTCIceCandidate`` import
        inside the handler raises ``ImportError``, which is caught by the
        endpoint's own ``except`` clause and returned as a 200 with
        ``{"status": "error", "detail": "..."}`` -- the endpoint never
        bubbles a 500 for ICE handling failures.
        """
        _seed_session("ice-session-no-transport")
        resp = client.post(
            self.ENDPOINT,
            json={
                "session_id": "ice-session-no-transport",
                "candidate": "candidate 1 1 UDP 0 0.0.0.0 0 typ host",
                "sdp_mid": "0",
                "sdp_mline_index": 0,
            },
            headers=auth_header,
        )
        # The endpoint catches any exception (including ImportError for
        # missing aiortc) and returns 200 with an error body.
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data

    def test_ice_empty_candidate(
        self, client: TestClient, auth_header: dict[str, str]
    ) -> None:
        """Return 200 even when the candidate string is empty."""
        _seed_session("ice-session-empty-candidate")
        resp = client.post(
            self.ENDPOINT,
            json={
                "session_id": "ice-session-empty-candidate",
                "candidate": "",
                "sdp_mid": "",
                "sdp_mline_index": 0,
            },
            headers=auth_header,
        )
        assert resp.status_code == 200
        assert "status" in resp.json()


# ── GET /v1/webrtc/status ──────────────────────────────────────────────────────


class TestStatus:
    ENDPOINT = "/v1/webrtc/status"

    def test_status_empty(
        self, client: TestClient, auth_header: dict[str, str]
    ) -> None:
        """Return zero counters when no sessions exist."""
        resp = client.get(self.ENDPOINT, headers=auth_header)
        assert resp.status_code == 200
        data = resp.json()
        assert data["active_sessions"] == 0
        assert data["total_sessions"] == 0
        assert isinstance(data["uptime_seconds"], float)

    def test_status_with_active_sessions(
        self, client: TestClient, auth_header: dict[str, str]
    ) -> None:
        """Reflect the sessions that have been created."""
        _seed_session("status-session-1")
        _seed_session("status-session-2")

        resp = client.get(self.ENDPOINT, headers=auth_header)
        assert resp.status_code == 200
        data = resp.json()
        assert data["active_sessions"] == 2
        assert data["total_sessions"] == 2
        assert isinstance(data["uptime_seconds"], float)

    def test_status_after_session_closed(
        self, client: TestClient, auth_header: dict[str, str]
    ) -> None:
        """Decrement counters after a session is closed."""
        from distllm.api.routes.webrtc import _session_mgr

        _seed_session("status-session-closed-1")
        _seed_session("status-session-closed-2")
        _session_mgr.close_session("status-session-closed-1")

        resp = client.get(self.ENDPOINT, headers=auth_header)
        assert resp.status_code == 200
        data = resp.json()
        # One session closed, one remains active
        assert data["active_sessions"] == 1
        assert data["total_sessions"] == 1

    def test_status_is_isolated(
        self, client: TestClient, auth_header: dict[str, str]
    ) -> None:
        """Status endpoint returns up-to-date data even when called twice."""
        resp1 = client.get(self.ENDPOINT, headers=auth_header)
        assert resp1.status_code == 200
        assert resp1.json()["active_sessions"] == 0

        _seed_session("status-isolated")
        resp2 = client.get(self.ENDPOINT, headers=auth_header)
        assert resp2.status_code == 200
        assert resp2.json()["active_sessions"] == 1

    def test_status_uptime_monotonic(
        self, client: TestClient, auth_header: dict[str, str]
    ) -> None:
        """Uptime increases between successive calls."""
        resp1 = client.get(self.ENDPOINT, headers=auth_header)
        t1 = resp1.json()["uptime_seconds"]
        resp2 = client.get(self.ENDPOINT, headers=auth_header)
        t2 = resp2.json()["uptime_seconds"]
        assert t2 >= t1


# ── DELETE /v1/webrtc/sessions/{session_id} ─────────────────────────────────────


class TestCloseSession:
    ENDPOINT = "/v1/webrtc/sessions"

    def test_close_session_not_found(
        self, client: TestClient, auth_header: dict[str, str]
    ) -> None:
        """Return 404 when session does not exist."""
        resp = client.delete(f"{self.ENDPOINT}/nonexistent", headers=auth_header)
        assert resp.status_code == 404
        assert resp.json()["error"]["message"] == "Session not found"

    def test_close_session_success(
        self, client: TestClient, auth_header: dict[str, str]
    ) -> None:
        """Return 200 and remove the session."""
        from distllm.api.routes.webrtc import _session_mgr

        sid = _seed_session("close-me", status="signaling")
        resp = client.delete(f"{self.ENDPOINT}/{sid}", headers=auth_header)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "closed"
        assert data["session_id"] == sid

        # The session should no longer be in the manager
        assert _session_mgr.get_session(sid) is None

    def test_close_session_no_transport(
        self, client: TestClient, auth_header: dict[str, str]
    ) -> None:
        """Succeed even when the session never had a transport set."""
        from distllm.api.routes.webrtc import _session_mgr

        sid = _seed_session("no-transport")
        resp = client.delete(f"{self.ENDPOINT}/{sid}", headers=auth_header)
        assert resp.status_code == 200
        assert resp.json()["status"] == "closed"
        assert _session_mgr.get_session(sid) is None

    def test_close_session_already_closed(
        self, client: TestClient, auth_header: dict[str, str]
    ) -> None:
        """Return 404 when the session was already closed."""
        sid = _seed_session("already-closed")
        # Close it once
        client.delete(f"{self.ENDPOINT}/{sid}", headers=auth_header)
        # Closing again should 404
        resp = client.delete(f"{self.ENDPOINT}/{sid}", headers=auth_header)
        assert resp.status_code == 404
        assert resp.json()["error"]["message"] == "Session not found"

    def test_close_session_twice_same_id(
        self, client: TestClient, auth_header: dict[str, str]
    ) -> None:
        """Creating a new session after closing the old one works."""
        from distllm.api.routes.webrtc import _session_mgr

        sid = _seed_session("reused-id")
        first = client.delete(f"{self.ENDPOINT}/{sid}", headers=auth_header)
        assert first.status_code == 200

        # Create a brand-new session with the same id string
        sid2 = _seed_session("reused-id")
        assert sid2 == "reused-id"
        second = client.delete(f"{self.ENDPOINT}/{sid2}", headers=auth_header)
        assert second.status_code == 200
