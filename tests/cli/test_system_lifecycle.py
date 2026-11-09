"""Behavioural tests for CLI lifecycle defects:

1. ``distllm system coordinator`` must block and keep the gRPC worker
   listener open instead of returning immediately (old code logged
   "Coordinator started" and exited 0 with no socket).
2. ``distllm system observe --metrics-only`` must keep the metrics port
   live until interrupted, and must not claim success when the bind fails
   (old code printed ✓ and exited 0 with the port dead).
"""

from __future__ import annotations

import io
import socket
import threading
import time
import types

import pytest
from typer.testing import CliRunner

from distllm.cli.coordinator_serve import serve_coordinator
from distllm.cli.observe import run_observe
from distllm.cli.main import app

runner = CliRunner()


# ── helpers ──────────────────────────────────────────────────────────────────


def _free_port() -> int:
    """Pick a port that is free right now (bind 0, read, close)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
    finally:
        s.close()


def _can_connect(port: int, timeout: float = 0.5) -> bool:
    try:
        socket.create_connection(("127.0.0.1", port), timeout).close()
        return True
    except OSError:
        return False


def _wait_until(predicate, timeout: float, interval: float = 0.1) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def _has(module: str) -> bool:
    try:
        __import__(module)
        return True
    except ImportError:
        return False


class _StubCoordinator:
    """Stand-in for ``distllm.core.coordinator.Coordinator``."""

    def __init__(self, port: int = 0, fail_start: bool = False) -> None:
        self.port = port
        self.config = types.SimpleNamespace(cluster_key=None)
        self._running = threading.Event()
        self.start_calls: list[bool] = []
        self.stop_calls = 0
        self._fail_start = fail_start

    def start(self, blocking: bool = True, **_kwargs) -> None:
        self.start_calls.append(blocking)
        if self._fail_start:
            raise RuntimeError("startup exploded")
        self._running.set()

    def stop(self) -> None:
        self.stop_calls += 1
        self._running.clear()


# ── Defect 1: serve_coordinator lifecycle ────────────────────────────────────


class TestServeCoordinator:
    def test_blocks_until_stop_event_then_stops_coordinator(self):
        """The helper must NOT return while the coordinator is alive.

        Old behaviour: the command called ``c.start()`` (whose internal
        ``self._running.wait()`` returns immediately because the event was
        just set) and the process exited 0.  On the old code this helper
        does not exist at all (ImportError), and the CLI-level exit-code
        test below catches the behavioural difference.
        """
        c = _StubCoordinator()
        stop = threading.Event()
        result: dict[str, int] = {}

        t = threading.Thread(
            target=lambda: result.setdefault(
                "rc", serve_coordinator(c, stop_event=stop, start_listener=False)
            ),
            daemon=True,
        )
        t.start()
        t.join(0.8)

        assert t.is_alive(), "serve_coordinator returned while the coordinator was alive"
        assert c.start_calls == [False], "must start the coordinator with blocking=False"
        assert c.stop_calls == 0, "must not stop before being asked to"

        stop.set()
        t.join(5)
        assert not t.is_alive()
        assert result["rc"] == 0
        assert c.stop_calls == 1, "graceful stop must call Coordinator.stop()"

    def test_startup_failure_returns_nonzero(self):
        c = _StubCoordinator(fail_start=True)
        assert serve_coordinator(c, stop_event=threading.Event(), start_listener=False) == 1
        assert c.stop_calls == 0

    def test_keyboard_interrupt_still_stops_coordinator(self, monkeypatch):
        import distllm.cli.coordinator_serve as cs

        def _boom(*_args, **_kwargs):
            raise KeyboardInterrupt

        monkeypatch.setattr(cs, "_wait_for_stop", _boom)
        c = _StubCoordinator()
        assert serve_coordinator(c, start_listener=False) == 0
        assert c.stop_calls == 1

    def test_external_shutdown_of_running_flag_returns_zero(self):
        c = _StubCoordinator()
        stop = threading.Event()
        result: dict[str, int] = {}

        def _stop_externally() -> None:
            # Simulate the coordinator stopping itself (e.g. another control path)
            while not c.start_calls:
                time.sleep(0.05)
            c._running.clear()

        t = threading.Thread(
            target=lambda: result.setdefault(
                "rc", serve_coordinator(c, stop_event=stop, poll_interval=0.05,
                                        start_listener=False)
            ),
            daemon=True,
        )
        t.start()
        threading.Thread(target=_stop_externally, daemon=True).start()
        t.join(5)
        assert not t.is_alive(), "helper must notice the cleared running flag"
        assert result["rc"] == 0
        assert c.stop_calls == 0, "already-stopped coordinator must not be stopped twice"

    def test_system_coordinator_command_propagates_serve_exit_code(self, monkeypatch):
        """CLI-level: the command must delegate to serve_coordinator and exit
        with its code.  Old code called ``c.start()`` (returns immediately)
        and exited 0 regardless."""
        import distllm.core.coordinator as core_coord
        import distllm.cli.coordinator_serve as cs

        created = _StubCoordinator()
        served: list[_StubCoordinator] = []

        monkeypatch.setattr(core_coord, "Coordinator", lambda **_kw: created)

        def _fake_serve(c, **_kwargs):
            served.append(c)
            return 3

        monkeypatch.setattr(cs, "serve_coordinator", _fake_serve)

        result = runner.invoke(app, ["system", "coordinator", "--model", "gpt2"])

        assert result.exit_code == 3, (
            f"command must propagate serve_coordinator's exit code, got {result.exit_code}"
        )
        assert served == [created], "command must run serve_coordinator on the Coordinator"
        assert created.start_calls == [], "command must not call start() itself"

    def test_start_cli_node_listener_refuses_port_occupied_by_foreign_process(self):
        """gRPC silently ignores a failed bind (NodeServer discards
        add_insecure_port's result), so a naive readiness probe would be
        answered by the foreign listener and we would claim a listener we
        do not own."""
        from distllm.cli.coordinator_serve import start_cli_node_listener

        holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            holder.bind(("0.0.0.0", 0))
            holder.listen(1)
            port = holder.getsockname()[1]
            c = _StubCoordinator(port=port)
            assert start_cli_node_listener(c) is None, (
                "must refuse to claim a listener on a port held by another process"
            )
            assert getattr(c, "_node_server", None) is None
        finally:
            holder.close()

    @pytest.mark.skipif(
        not all(_has(module) for module in ("grpc",)),
        reason="grpcio not installed",
    )
    def test_serve_opens_tcp_worker_listener_and_closes_on_stop(self, monkeypatch):
        """End-to-end helper test: a real TCP connect succeeds on
        ``coordinator.port`` while serving, and the port is closed after."""
        monkeypatch.delenv("DISTLLM_TLS_ENABLED", raising=False)
        monkeypatch.setenv("DISTLLM_CLUSTER_KEY", "test-cluster-key")
        c = _StubCoordinator(port=_free_port())
        stop = threading.Event()
        result: dict[str, int] = {}

        t = threading.Thread(
            target=lambda: result.setdefault(
                "rc", serve_coordinator(c, stop_event=stop, poll_interval=0.05)
            ),
            daemon=True,
        )
        t.start()
        try:
            assert _wait_until(lambda: _can_connect(c.port), timeout=15), (
                "gRPC worker listener never accepted a TCP connection"
            )
        finally:
            stop.set()
            t.join(30)

        assert result["rc"] == 0
        assert _wait_until(lambda: not _can_connect(c.port), timeout=10), (
            "worker listener port still open after shutdown"
        )


# ── Defect 2: observe metrics lifecycle ──────────────────────────────────────


def _get_metrics_status(port: int) -> int | None:
    import urllib.request

    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/metrics", timeout=5
        ) as resp:
            return resp.status
    except OSError:
        return None


class TestRunObserve:
    def test_metrics_only_serves_and_blocks_until_stop(self):
        """Old code returned from run_observe instantly, so the server thread
        was dead (and the port closed) before the first GET could succeed —
        the ``_wait_until(_get_metrics_status ... == 200)`` assertion fails on
        the old code."""
        port = _free_port()
        stop = threading.Event()
        result: dict[str, int] = {}

        t = threading.Thread(
            target=lambda: result.setdefault(
                "rc", run_observe(metrics_port=port, metrics_only=True, stop_event=stop)
            ),
            daemon=True,
        )
        t.start()
        try:
            assert _wait_until(lambda: _get_metrics_status(port) == 200, timeout=15), (
                "metrics endpoint never served 200 on /metrics"
            )
            assert t.is_alive(), "run_observe must keep running while metrics_only"
        finally:
            stop.set()
            t.join(10)

        assert not t.is_alive()
        assert result["rc"] == 0
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.settimeout(1)
            still_open = s.connect_ex(("127.0.0.1", port)) == 0
        finally:
            s.close()
        assert not still_open, "metrics port must be closed after shutdown"

    def test_bind_failure_returns_nonzero_and_does_not_claim_success(self):
        """Old code constructed the HTTPServer inside the thread starter and
        printed ✓ before binding — on a busy port it printed success and then
        raised OSError out of run_observe.  New code must return 1 and never
        print the success line."""
        holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            # HTTPServer binds 0.0.0.0 — occupy the same address to make the
            # bind deterministically fail on Windows too.
            holder.bind(("0.0.0.0", 0))
            holder.listen(1)
            port = holder.getsockname()[1]

            out = io.StringIO()
            from rich.console import Console

            rc = run_observe(
                metrics_port=port,
                metrics_only=True,
                console=Console(file=out, force_terminal=False),
            )
        finally:
            holder.close()

        assert rc == 1, "bind failure must exit non-zero"
        text = out.getvalue()
        assert "failed to bind" in text
        assert "✓ Metrics endpoint" not in text, "must not claim success when bind failed"

    def test_system_observe_command_propagates_exit_code(self, monkeypatch):
        """Old CLI wrapper ignored run_observe's (None) return and always
        exited 0 — asserting exit code 7 catches that."""
        import distllm.cli.observe as observe_mod

        monkeypatch.setattr(observe_mod, "run_observe", lambda **_kw: 7)

        result = runner.invoke(app, ["system", "observe", "--metrics-only"])
        assert result.exit_code == 7

    def test_system_observe_real_bind_failure_exits_nonzero_cleanly(self):
        holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            holder.bind(("0.0.0.0", 0))
            holder.listen(1)
            port = holder.getsockname()[1]
            result = runner.invoke(
                app, ["system", "observe", "--metrics-port", str(port), "--metrics-only"]
            )
        finally:
            holder.close()

        assert result.exit_code == 1
        assert result.exception is None or isinstance(result.exception, SystemExit), (
            "bind failure must exit cleanly, not leak an OSError traceback"
        )
        assert "failed to bind" in result.output
