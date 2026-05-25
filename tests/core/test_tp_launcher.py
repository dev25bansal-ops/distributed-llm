"""Tests for TP launcher with N-GPU readiness and worker management."""

import socket
import threading
import time

import pytest

from distllm.core.tp_launcher import TPWorkerHandle


class TestWaitUntilReady:
    def test_ready_immediately(self):
        handle = TPWorkerHandle(
            process_context=None,
            ports=[],
            world_size=0,
        )
        assert handle.wait_until_ready(timeout_s=1) is True

    def test_ready_single_port(self):
        with start_dummy_server(18081) as port:
            handle = TPWorkerHandle(
                process_context=None,
                ports=[port],
                world_size=1,
            )
            assert handle.wait_until_ready(timeout_s=5) is True

    def test_ready_multiple_ports(self):
        servers = [start_dummy_server(18082 + i) for i in range(3)]
        ports = []
        for cm in servers:
            port = cm.__enter__()
            ports.append(port)
        handle = TPWorkerHandle(
            process_context=None,
            ports=ports,
            world_size=len(ports),
        )
        assert handle.wait_until_ready(timeout_s=5) is True
        for cm in servers:
            cm.__exit__(None, None, None)

    def test_timeout_no_servers(self):
        handle = TPWorkerHandle(
            process_context=None,
            ports=[19999],
            world_size=1,
        )
        assert handle.wait_until_ready(timeout_s=1, interval_s=0.2) is False

    def test_mixed_ready_and_unready(self):
        with start_dummy_server(18085) as ready_port:
            handle = TPWorkerHandle(
                process_context=None,
                ports=[ready_port, 29999],
                world_size=2,
            )
            assert handle.wait_until_ready(timeout_s=1, interval_s=0.2) is False

    def test_becomes_ready_later(self):
        ready_event = threading.Event()

        def delayed_server():
            ready_event.wait(5)
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("127.0.0.1", 18086))
                s.listen(1)
                s.settimeout(3)
                try:
                    s.accept()
                except socket.timeout:
                    pass

        t = threading.Thread(target=delayed_server, daemon=True)
        t.start()

        handle = TPWorkerHandle(
            process_context=None,
            ports=[18086],
            world_size=1,
        )

        def signal_ready():
            time.sleep(1.5)
            ready_event.set()

        signaling = threading.Thread(target=signal_ready, daemon=True)
        signaling.start()

        assert handle.wait_until_ready(timeout_s=5, interval_s=0.5) is True


class TestTpWorkerHandle:
    def test_handle_fields(self):
        handle = TPWorkerHandle(
            process_context="ctx",
            ports=[8000, 8001],
            world_size=2,
        )
        assert handle.process_context == "ctx"
        assert handle.ports == [8000, 8001]
        assert handle.world_size == 2

    def test_empty_handle(self):
        handle = TPWorkerHandle(
            process_context=None,
            ports=[],
            world_size=0,
        )
        assert handle.wait_until_ready(timeout_s=1) is True


def start_dummy_server(default_port):
    """Context manager that starts a simple TCP server (dummy gRPC).

    Returns the actual port bound (handles port-in-use).
    """
    class _Ctx:
        def __init__(self, preferred_port):
            self._preferred = preferred_port
            self._server = None
            self.port = None

        def __enter__(self):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("127.0.0.1", 0 if self._preferred is None else self._preferred))
            s.listen(1)
            s.settimeout(5)
            self.port = s.getsockname()[1]
            self._server = s
            return self.port

        def __exit__(self, *args):
            if self._server:
                try:
                    self._server.close()
                except Exception:
                    pass

    return _Ctx(default_port)
