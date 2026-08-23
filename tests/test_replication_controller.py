"""Regression tests for the extracted ReplicationController (god-object
decomposition pilot) and the stable hashing refactor.
"""

import threading
import time

from distllm.core.replication_controller import ReplicationController


class _FakeClient:
    """Records POST calls instead of hitting the network."""

    def __init__(self, *a, **k):
        self.posts: list[tuple[str, dict]] = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def post(self, url, json=None, **k):
        self.posts.append((url, json))
        return _FakeResp(200)


class _FakeResp:
    def __init__(self, code):
        self.status_code = code


def _make_controller(running):
    ctrl = ReplicationController(
        get_snapshot=lambda: {"model_name": "x", "nodes": {}, "node_order": [], "timestamp": 0},
        is_healthy=lambda: True,
        get_node_count=lambda: 0,
        running=running,
        client_factory=_FakeClient,
    )
    return ctrl


def test_set_peers_starts_single_thread():
    running = threading.Event()
    running.set()
    ctrl = _make_controller(running)
    ctrl.set_peers(["http://peer1:8000", "http://peer2:8000"])
    assert ctrl._replication_thread is not None
    first = ctrl._replication_thread
    # Calling set_peers again must NOT start a second thread (M16 guard).
    ctrl.set_peers(["http://peer3:8000"])
    assert ctrl._replication_thread is first
    # peers updated
    assert ctrl.peers == ["http://peer3:8000"]
    ctrl.stop()


def test_stop_joins_thread():
    running = threading.Event()
    running.set()
    ctrl = _make_controller(running)
    ctrl.set_peers(["http://peer1:8000"])
    ctrl.stop()
    assert ctrl._replication_thread is None


def test_loop_terminates_on_running_clear():
    running = threading.Event()
    running.set()
    ctrl = _make_controller(running)
    ctrl.set_peers(["http://peer1:8000"])
    # Clear running -> loop must exit promptly.
    running.clear()
    ctrl._replication_thread.join(timeout=3.0)
    assert not ctrl._replication_thread.is_alive()
    ctrl.stop()
