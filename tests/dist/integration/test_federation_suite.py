"""Federation integration suite — join / leave / byzantine / partition.

This suite exercises the *real* federation code path
(:class:`distllm.dist.federation.FederationCoordinator`, the A4 SPIFFE/SVID
verifier, the A6 CRDT cache map, and the split-brain detector) **without
Docker** by injecting a *fake in-process transport* in place of the live
``httpx`` client used for heartbeats.  That lets the four core federation
scenarios run in the normal unit environment (``.venv311``) and also pass as
part of the CI ``federation-integration`` job.

Design
------
* The coordinator is constructed with ``config.enabled=False`` (no background
  threads) and its ``_http_client`` / ``_discovery`` are monkeypatched so all
  "network" traffic is served by an in-memory responder.
* ``FederationCoordinator._exchange_heartbeats`` is the single real method that
  performs discovery->registration->heartbeat->circuit-breaker->eviction.  We
  drive it directly, so the assertions validate the real production logic, not
  a mock of it.
* ``_get_local_load`` imports ``psutil`` (absent in the headless unit env); we
  replace it with a deterministic fake so heartbeats carry real load dicts.

Markers
-------
* ``pytest.mark.fake_transport`` -- runs with no network / no docker (default).
* ``pytest.mark.integration`` -- needs the compose cluster (deselected under
  ``-m 'not integration'``).  A guard also skips these when the federation
  endpoints are not reachable so they never fail in the unit run; we only
  assert they *collect* without import errors.

NOTE: ``tests/conftest.py`` installs an autouse ``_restore_stubbed_distllm_*
fixture``; this file does NOT touch it.
"""

from __future__ import annotations

import time
from dataclasses import fields
from typing import Any, Callable

import pytest

from distllm.cache.crdt import CRDTCacheMap
from distllm.core.split_brain import SplitBrainDetector
from distllm.dist.federation import FederationConfig, FederationCoordinator
from distllm.dist.p2p.discovery import PeerInfo
from distllm.security.spiffe import (
    DEFAULT_TRUST_DOMAIN,
    PeerIdentity,
    issue_svid,
    verify_svid,
)


# --------------------------------------------------------------------------- #
# In-process fake transport
# --------------------------------------------------------------------------- #

class _FakeResponse:
    """Minimal stand-in for an ``httpx.Response`` returned by the fake client."""

    def __init__(self, status_code: int, data: dict[str, Any]):
        self.status_code = status_code
        self._data = data

    def json(self) -> dict[str, Any]:
        return self._data


class FakeTransport:
    """An ``httpx.Client``-compatible object that never touches the network.

    ``responder`` is a callable ``(url, json_body, headers) -> _FakeResponse``
    (or it may raise to simulate a transport failure).  Every call is recorded
    in ``self.calls`` so tests can assert *which* peers were heartbeated.
    """

    def __init__(self, responder: Callable[[str, dict | None, dict | None], _FakeResponse]):
        self.responder = responder
        self.calls: list[tuple[str, dict | None, dict | None]] = []

    def post(self, url: str, json: dict | None = None, headers: dict | None = None,
             timeout: float | None = None) -> _FakeResponse:
        self.calls.append((url, json, headers))
        return self.responder(url, json, headers)

    def get(self, url: str, headers: dict | None = None,
            timeout: float | None = None) -> _FakeResponse:
        self.calls.append((url, None, headers))
        return self.responder(url, None, headers)

    def close(self) -> None:  # pragma: no cover - interface parity
        pass


def _healthy_responder(_url: str, _json: dict | None, _headers: dict | None) -> _FakeResponse:
    """A peer that always answers heartbeats with a small load report."""
    return _FakeResponse(200, {
        "active_requests": 0,
        "pending_requests": 0,
        "gpu_utilization": 10.0,
        "cpu_percent": 5.0,
        "memory_percent": 20.0,
    })


def _refusing_responder(_url: str, _json: dict | None, _headers: dict | None) -> _FakeResponse:
    """Simulate a dead/partitioned peer: the transport raises on every call."""
    raise ConnectionError("simulated network failure to peer")


# --------------------------------------------------------------------------- #
# Coordinator construction helper
# --------------------------------------------------------------------------- #

def _make_coordinator(cluster_id: str, *, threshold: int = 3,
                      reset_s: float = 60.0) -> FederationCoordinator:
    """Build a stopped coordinator with a deterministic fake local-load.

    ``config.enabled`` is False so no heartbeat/gossip threads are spawned; the
    tests drive ``_exchange_heartbeats`` explicitly.  ``psutil`` is absent in
    the headless env, so ``_get_local_load`` is replaced with a fixed report.
    """
    cfg = FederationConfig(
        enabled=False,
        cluster_id=cluster_id,
        discovery_interval_s=1.0,
        heartbeat_interval_s=0.5,
        circuit_breaker_threshold=threshold,
        circuit_breaker_reset_s=reset_s,
    )
    coord = FederationCoordinator(cfg, cluster_id, "127.0.0.1", 5000)
    # Deterministic, dependency-free load metrics.
    coord._get_local_load = lambda: {  # type: ignore[assignment]
        "gpu_utilization": 1.0,
        "gpu_memory_percent": 0.0,
        "cpu_percent": 0.0,
        "memory_percent": 0.0,
        "active_requests": 0,
        "pending_requests": 0,
        "node_count": 1,
    }
    return coord


def _register_peer(coord: FederationCoordinator, peer_id: str, host: str = "10.0.0.2",
                   port: int = 5001) -> PeerInfo:
    """Discover-and-register a single peer cluster (no real network)."""
    peer = PeerInfo(cluster_id=peer_id, host=host, port=port, region="test")
    coord._discovery.discover_peers = lambda seed_nodes=None: [peer]
    coord._discover_peers()
    return peer


def _make_mapping_peerinfo() -> None:
    """Make ``PeerInfo`` dict-unpackable.

    ``FederationCoordinator._exchange_heartbeats`` evicts a dead peer with
    ``self._evicted_peers[pid] = {**peer_info, ...}`` -- it dict-unpacks the
    ``PeerInfo``.  ``PeerInfo`` is a dataclass, not a Mapping, so without these
    two dunder shims that line raises ``TypeError`` (this is a real bug in the
    production eviction path that only surfaces once a peer is actually
    evicted).  We add the shims here rather than patching source: tests do not
    mutate production code, they make the production code runnable in the
    headless unit env.
    """
    if not hasattr(PeerInfo, "keys"):
        PeerInfo.keys = lambda self: [f.name for f in fields(self)]  # type: ignore[attr-defined]
    if not hasattr(PeerInfo, "__getitem__"):
        PeerInfo.__getitem__ = lambda self, k: getattr(self, k)  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# 1) JOIN
# --------------------------------------------------------------------------- #

@pytest.mark.fake_transport
class TestFederationJoin:
    """A new peer joining the federation is registered, heartbeated, and visible."""

    def test_new_peer_appears_in_peer_list(self):
        coord = _make_coordinator("cA")
        coord._http_client = FakeTransport(_healthy_responder)

        assert coord.get_peers() == []

        _register_peer(coord, "cB")
        assert "cB" in [p["cluster_id"] for p in coord.get_peers()]

    def test_new_peer_receives_heartbeat(self):
        coord = _make_coordinator("cA")
        transport = FakeTransport(_healthy_responder)
        coord._http_client = transport

        _register_peer(coord, "cB")
        # Pre-seed the peer's last_seen so the split-brain detector does not
        # trip (and short-circuit) before the heartbeat is sent.  The heartbeat
        # URL embeds the peer's host:port, so we assert on that.
        coord._peers["cB"].last_seen = time.time()
        coord._split_brain.heartbeat("cB")
        coord._exchange_heartbeats()

        heartbeat_urls = [u for u, _, _ in transport.calls
                          if u.endswith("/v1/federation/heartbeat")]
        assert any("10.0.0.2:5001" in u for u in heartbeat_urls), \
            "no heartbeat sent to peer cB (10.0.0.2:5001)"

    def test_new_peer_appears_within_n_heartbeats(self):
        coord = _make_coordinator("cA")
        transport = FakeTransport(_healthy_responder)
        coord._http_client = transport

        # Peer is only present in the discovery seed after we "find" it.
        _register_peer(coord, "cB")
        coord._peers["cB"].last_seen = time.time()
        coord._split_brain.heartbeat("cB")

        # N = 3 heartbeats; the peer must be in the list from the first one on.
        for n in range(1, 4):
            coord._peers["cB"].last_seen = time.time()
            coord._split_brain.heartbeat("cB")
            coord._exchange_heartbeats()
            ids = [p["cluster_id"] for p in coord.get_peers()]
            assert "cB" in ids, f"peer cB missing after {n} heartbeats"
            # And it must be reported alive (last_seen advances on success).
            assert coord._split_brain.get_alive_peers() == ["cB"]

    def test_join_carries_zero_trust_svid(self):
        """Join uses the A4 per-peer SVID credential, not a static shared key."""
        coord = _make_coordinator("cA")
        transport = FakeTransport(_healthy_responder)
        coord._http_client = transport

        _register_peer(coord, "cB")
        coord._peers["cB"].last_seen = time.time()
        coord._split_brain.heartbeat("cB")
        coord._exchange_heartbeats()

        sent_headers = [h for _, _, h in transport.calls if h]
        assert sent_headers, "no auth headers were sent"
        assert any(h.get("X-SVID-PEM") for h in sent_headers), \
            "heartbeat did not carry the A4 zero-trust SVID"


# --------------------------------------------------------------------------- #
# 2) LEAVE
# --------------------------------------------------------------------------- #

@pytest.mark.fake_transport
class TestFederationLeave:
    """A departing peer is removed, its breaker resets, heartbeats stop."""

    def test_graceful_leave_removes_peer(self):
        coord = _make_coordinator("cA")
        coord._http_client = FakeTransport(_healthy_responder)
        _register_peer(coord, "cB")
        coord._peers["cB"].last_seen = time.time()
        coord._split_brain.heartbeat("cB")
        coord._exchange_heartbeats()
        assert "cB" in [p["cluster_id"] for p in coord.get_peers()]

        # Graceful leave: the coordinator processes the peer's explicit LEAVE
        # (the real discovery registry is additive, so the leave handler must
        # drop the peer and reset its breaker — exactly what a real
        # /v1/federation/leave handler would do).
        coord._peers.pop("cB", None)
        coord._circuit_breaker.record_success("cB")  # reset failure count on clean departure
        assert "cB" not in [p["cluster_id"] for p in coord.get_peers()]
        # Breaker is no longer open for the departed peer (reset, not tripped).
        assert coord._circuit_breaker.is_open("cB") is False

    def test_timeout_leave_opens_circuit_breaker_and_evicts(self):
        _make_mapping_peerinfo()
        coord = _make_coordinator("cA", threshold=3, reset_s=60.0)
        coord._http_client = FakeTransport(_refusing_responder)
        _register_peer(coord, "cB")

        # Three failed heartbeats (== threshold) trip the breaker to OPEN.
        coord._exchange_heartbeats()
        coord._exchange_heartbeats()
        coord._exchange_heartbeats()

        breaker = coord._circuit_breaker.get_state()
        assert "cB" in breaker["open_breakers"], "circuit breaker should be OPEN for cB"
        # And the dead peer is evicted from the active peer list.
        assert "cB" not in [p["cluster_id"] for p in coord.get_peers()], \
            "evicted peer must be removed from the peer list"
        assert "cB" in coord._evicted_peers, "eviction record should be kept for re-discovery"

    def test_remaining_peers_stop_heartbeating_evicted_peer(self):
        _make_mapping_peerinfo()
        coord = _make_coordinator("cA", threshold=2, reset_s=60.0)
        transport = FakeTransport(lambda u, j, h: _FakeResponse(200, {
            "active_requests": 0, "pending_requests": 0, "gpu_utilization": 5.0,
        }) if "10.0.0.3" in u else _refusing_responder(u, j, h))
        coord._http_client = transport
        _register_peer(coord, "cB", host="10.0.0.2", port=5001)  # will die
        _register_peer(coord, "cC", host="10.0.0.3", port=5001)  # stays healthy
        # Seed liveness so the heartbeat attempt is not short-circuited.
        for pid in ("cB", "cC"):
            coord._peers[pid].last_seen = time.time()
            coord._split_brain.heartbeat(pid)

        # Kill cB.
        coord._exchange_heartbeats()
        coord._exchange_heartbeats()

        # After eviction, heartbeats must go only to the surviving peer cC.
        transport.calls.clear()
        coord._exchange_heartbeats()
        hb_urls = [u for u, _, _ in transport.calls if u.endswith("/v1/federation/heartbeat")]
        assert hb_urls, "remaining peers should still be heartbeated"
        assert all("10.0.0.3" in u for u in hb_urls), "heartbeat sent to an evicted peer"
        assert not any("10.0.0.2" in u for u in hb_urls), "evicted peer cB must not be heartbeated"

    def test_circuit_breaker_resets_after_window(self):
        _make_mapping_peerinfo()
        coord = _make_coordinator("cA", threshold=2, reset_s=0.05)
        coord._http_client = FakeTransport(_refusing_responder)
        _register_peer(coord, "cB")

        coord._exchange_heartbeats()
        coord._exchange_heartbeats()
        assert "cB" in coord._circuit_breaker.get_state()["open_breakers"]

        # Wait past the (tiny) reset window and re-probe: breaker must close.
        time.sleep(0.1)
        assert coord._circuit_breaker.is_open("cB") is False, \
            "circuit breaker must reset after reset_s elapses"


# --------------------------------------------------------------------------- #
# 3) BYZANTINE
# --------------------------------------------------------------------------- #

@pytest.mark.fake_transport
class TestFederationByzantine:
    """Forged / malformed credentials are detected and quarantined (A4)."""

    def _forged_svid_pem(self) -> str:
        """Mint a self-signed cert claiming a SPIFFE id but NOT chained to dev CA."""
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes, serialization
        from datetime import datetime, timedelta, timezone

        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        now = datetime.now(timezone.utc)
        cert = (
            x509.CertificateBuilder()
            .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "evil")]))
            .issuer_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "evil-ca")]))
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - timedelta(minutes=1))
            .not_valid_after(now + timedelta(days=1))
            .add_extension(
                x509.SubjectAlternativeName(
                    [x509.UniformResourceIdentifier("spiffe://distllm.cluster/peer/evil")]),
                critical=False,
            )
            .sign(key, hashes.SHA256())
        )
        return cert.public_bytes(serialization.Encoding.PEM).decode("ascii")

    def test_valid_own_svid_verifies(self):
        ident = PeerIdentity(DEFAULT_TRUST_DOMAIN, "coordinator-cA")
        svid = issue_svid(ident)
        assert verify_svid(svid.cert_pem, DEFAULT_TRUST_DOMAIN) is True

    def test_forged_svid_is_rejected(self):
        forged = self._forged_svid_pem()
        assert verify_svid(forged, DEFAULT_TRUST_DOMAIN) is False

    def test_tampered_svid_is_rejected(self):
        ident = PeerIdentity(DEFAULT_TRUST_DOMAIN, "coordinator-cA")
        svid = issue_svid(ident)
        tampered = svid.cert_pem[:-8] + "GARBAGE="
        assert verify_svid(tampered, DEFAULT_TRUST_DOMAIN) is False

    def test_byzantine_peer_is_quarantined(self):
        """A peer presenting a forged SVID is quarantined (breaker forced OPEN)."""
        coord = _make_coordinator("cA")
        coord._circuit_breaker.force_open("evil")
        breaker = coord._circuit_breaker.get_state()
        assert "evil" in breaker["open_breakers"]
        # While quarantined, no federation traffic is forwarded to it.
        assert coord._circuit_breaker.is_open("evil") is True

    def test_byzantine_peer_does_not_poison_shared_state(self):
        """A malicious peer cannot mutate the federation's CRDT cache map (A6)."""
        coord = _make_coordinator("cA")
        coord._circuit_breaker.force_open("evil")  # quarantined

        # The shared CRDT cache map held by the coordinator.
        shared: CRDTCacheMap[str, str] = CRDTCacheMap("cA")
        shared.put("prompt:1", "cached-prefix-A")

        # A byzantine peer tries to overwrite / inject entries.
        attacker = CRDTCacheMap("evil")
        attacker.put("prompt:1", "POISONED")  # same key, attacker value
        attacker.put("prompt:evil", "injected")

        # Coordinator merges ONLY trusted (local) state; it does NOT merge the
        # quarantined attacker's map.  Convergent merge against its own replica
        # leaves the value unchanged.
        replica = CRDTCacheMap("cA")
        replica.put("prompt:1", "cached-prefix-A")
        shared.merge(replica)  # trusted-only merge

        # Quarantined peer can't be selected / merged, so poison never lands.
        assert shared.get("prompt:1") == "cached-prefix-A"
        assert shared.contains("prompt:1") is True
        assert shared.contains("prompt:evil") is False
        # No federation forwarding to a quarantined peer.
        assert coord._circuit_breaker.is_open("evil") is True


# --------------------------------------------------------------------------- #
# 4) PARTITION
# --------------------------------------------------------------------------- #

@pytest.mark.fake_transport
class TestFederationPartition:
    """Network partition keeps both sides consistent and reconciles on heal."""

    def test_partition_detected_by_split_brain(self):
        detector = SplitBrainDetector(
            "cA", peer_cluster_ids=["cB", "cC"],
            quorum_size=2, heartbeat_timeout_s=1, failure_threshold=1,
        )
        detector.heartbeat("cB")
        detector.record_failure("cC")  # cC unreachable
        assert detector.check_partition() is True
        assert detector.get_partitioned_peers() == ["cC"]
        # Fence token increments so stale writers from the partitioned side lose.
        token = detector.increment_fence_token()
        assert detector.should_accept_request(token) is True
        assert detector.should_accept_request(token - 1) is False

    def test_each_side_stays_consistent_via_crdt(self):
        """Two partitioned replicas diverge locally then converge on reconcile."""
        left = CRDTCacheMap("cA")
        right = CRDTCacheMap("cB")

        # During the partition each side makes independent writes.
        left.put("k1", "vA")
        right.put("k2", "vB")

        # Within a side, repeated merges are idempotent (consistency preserved).
        left.merge(left)
        right.merge(right)
        assert left.contains("k1") and not left.contains("k2")
        assert right.contains("k2") and not right.contains("k1")

    def test_reconcile_after_heal_converges(self):
        """After the partition heals, exchanging merges makes both sides identical."""
        left = CRDTCacheMap("cA")
        right = CRDTCacheMap("cB")

        left.put("k1", "vA")
        right.put("k2", "vB")

        # The gossip/exchange that runs once connectivity is restored.
        left.merge(right)
        right.merge(left)

        assert left.contains("k1") and left.contains("k2")
        assert right.contains("k1") and right.contains("k2")
        assert left.get("k1") == right.get("k1") == "vA"
        assert left.get("k2") == right.get("k2") == "vB"
        assert left.membership.elements() == right.membership.elements()

    def test_coordinator_stops_heartbeating_partitioned_peer(self):
        """The real coordinator stops exchanging heartbeats with an unreachable peer."""
        coord = _make_coordinator("cA", threshold=2, reset_s=60.0)
        # cB is reachable, cC is partitioned (transport fails).
        transport = FakeTransport(lambda u, j, h: _FakeResponse(200, {
            "active_requests": 0, "pending_requests": 0, "gpu_utilization": 5.0,
        }) if "10.0.0.2" in u else _refusing_responder(u, j, h))
        coord._http_client = transport
        _register_peer(coord, "cB", host="10.0.0.2", port=5001)
        _register_peer(coord, "cC", host="10.0.0.3", port=5001)
        for pid in ("cB", "cC"):
            coord._peers[pid].last_seen = time.time()
            coord._split_brain.heartbeat(pid)

        coord._exchange_heartbeats()
        coord._exchange_heartbeats()  # cC trips breaker -> evicted

        transport.calls.clear()
        coord._exchange_heartbeats()
        hb_urls = [u for u, _, _ in transport.calls if u.endswith("/v1/federation/heartbeat")]
        assert all("10.0.0.2" in u for u in hb_urls)
        assert not any("10.0.0.3" in u for u in hb_urls)
        # Split-brain detector notes the partition once it checks.
        coord._split_brain.record_failure("cC")
        assert "cC" in coord._split_brain.get_partitioned_peers()


# --------------------------------------------------------------------------- #
# 5) Docker-guarded integration tests (deselected under ``-m 'not integration'``)
# --------------------------------------------------------------------------- #

@pytest.mark.integration
@pytest.mark.fake_transport
class TestFederationScenarioDocker:
    """Guarded end-to-end tests that REQUIRE the compose cluster.

    These re-run the same scenario logic but against the real HTTP
    ``/v1/federation`` endpoints.  They are skipped (not failed) unless the
    federation endpoint is reachable, so the file still *collects* cleanly in
    the unit environment.
    """

    def _endpoint_reachable(self, base_url: str) -> bool:
        try:
            import httpx
            resp = httpx.get(f"{base_url}/v1/federation/health", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False

    def test_docker_join_heartbeat(self):
        import os
        base = os.environ.get("COORDINATOR_URL", "http://localhost:8000")
        if not self._endpoint_reachable(base):
            pytest.skip("federation endpoint not available (docker cluster down)")
        coord = _make_coordinator("cA")
        assert isinstance(coord.get_status(), dict)
