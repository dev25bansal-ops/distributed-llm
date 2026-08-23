"""Tests for distllm.dist.tp_launcher module.

Zero mocks -- uses only real objects from the module.
Covers:
- TPWorkerHandle construction, repr, field access
- TPWorkerHandle.wait_until_ready with empty/unreachable/reachable ports
- tp_forward stub (always raises RuntimeError)
"""

from __future__ import annotations

import socket
import threading

import pytest
import torch

from distllm.dist.tp_launcher import (
    TPWorkerHandle,
    launch_tp_workers,
    tp_forward,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _listening_socket() -> socket.socket:
    """Create a TCP socket bound + listening on a random port.

    The caller is responsible for calling ``.close()`` on the returned socket.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", 0))
    s.listen(1)
    return s


def _accept_and_close(s: socket.socket) -> None:
    """Accept one connection then close the socket (runs in a thread)."""
    try:
        conn, _ = s.accept()
        conn.close()
    except OSError:
        pass
    finally:
        s.close()


# ---------------------------------------------------------------------------
# TPWorkerHandle
# ---------------------------------------------------------------------------


class TestTPWorkerHandle:
    """Construction, field access, repr, and wait_until_ready."""

    def test_construction_defaults(self) -> None:
        handle = TPWorkerHandle(
            process_context=None, ports=[], world_size=1
        )
        assert handle.process_context is None
        assert handle.ports == []
        assert handle.world_size == 1

    def test_construction_with_values(self) -> None:
        ctx = object()
        ports = [29501, 29502, 29503]
        handle = TPWorkerHandle(
            process_context=ctx, ports=ports, world_size=3
        )
        assert handle.process_context is ctx
        assert handle.ports == ports
        assert handle.world_size == 3

    def test_construction_large_world_size(self) -> None:
        handle = TPWorkerHandle(
            process_context=None, ports=list(range(29500, 29508)), world_size=8
        )
        assert handle.world_size == 8
        assert len(handle.ports) == 8

    def test_repr(self) -> None:
        handle = TPWorkerHandle(
            process_context="ctx", ports=[1234], world_size=2
        )
        r = repr(handle)
        assert "TPWorkerHandle" in r
        assert "ports=[1234]" in r
        assert "world_size=2" in r

    def test_fields_are_mutable(self) -> None:
        """Dataclass is NOT frozen -- fields can be reassigned."""
        handle = TPWorkerHandle(
            process_context=None, ports=[], world_size=1
        )
        handle.ports = [9999]
        assert handle.ports == [9999]
        handle.world_size = 4
        assert handle.world_size == 4

    # -- wait_until_ready ---------------------------------------------------

    def test_wait_ready_empty_ports_returns_immediately(self) -> None:
        """No ports to check -> trivially ready."""
        handle = TPWorkerHandle(
            process_context=None, ports=[], world_size=1
        )
        assert handle.wait_until_ready(timeout_s=0.0, interval_s=0.01) is True

    def test_wait_ready_empty_ports_negative_timeout(self) -> None:
        """Even with a negative timeout, empty ports are trivially ready."""
        handle = TPWorkerHandle(
            process_context=None, ports=[], world_size=1
        )
        assert handle.wait_until_ready(timeout_s=-1.0, interval_s=0.01) is True

    def test_wait_ready_unreachable_port_returns_false(self) -> None:
        """No service on the target port -> timeout."""
        handle = TPWorkerHandle(
            process_context=None, ports=[59999], world_size=1
        )
        assert (
            handle.wait_until_ready(timeout_s=0.1, interval_s=0.02) is False
        )

    def test_wait_ready_multiple_unreachable_ports_returns_false(self) -> None:
        """All ports unreachable -> timeout."""
        handle = TPWorkerHandle(
            process_context=None,
            ports=[59801, 59802, 59803],
            world_size=3,
        )
        assert (
            handle.wait_until_ready(timeout_s=0.15, interval_s=0.02) is False
        )

    def test_wait_ready_zero_timeout_returns_immediately(self) -> None:
        """timeout_s=0 means deadline == now, so one attempt is made."""
        handle = TPWorkerHandle(
            process_context=None, ports=[59998], world_size=1
        )
        assert (
            handle.wait_until_ready(timeout_s=0.0, interval_s=0.01) is False
        )

    def test_wait_ready_negative_timeout(self) -> None:
        """Negative timeout -> deadline in the past -> immediate return."""
        handle = TPWorkerHandle(
            process_context=None, ports=[59997], world_size=1
        )
        assert (
            handle.wait_until_ready(timeout_s=-5.0, interval_s=0.01) is False
        )

    def test_wait_ready_reachable_port(self) -> None:
        """A port with a listening socket is detected as ready."""
        srv = _listening_socket()
        port = srv.getsockname()[1]
        t = threading.Thread(
            target=_accept_and_close, args=(srv,), daemon=True
        )
        t.start()

        handle = TPWorkerHandle(
            process_context=None, ports=[port], world_size=1
        )
        assert (
            handle.wait_until_ready(timeout_s=5.0, interval_s=0.1) is True
        )
        t.join()

    def test_wait_ready_mixed_ports_all_ready(self) -> None:
        """Multiple ports, all reachable -> True."""
        sockets = [_listening_socket(), _listening_socket()]
        ports = [s.getsockname()[1] for s in sockets]
        threads = [
            threading.Thread(
                target=_accept_and_close, args=(s,), daemon=True
            )
            for s in sockets
        ]
        for t in threads:
            t.start()

        handle = TPWorkerHandle(
            process_context=None, ports=ports, world_size=2
        )
        assert (
            handle.wait_until_ready(timeout_s=5.0, interval_s=0.1) is True
        )
        for t in threads:
            t.join()

    def test_wait_ready_mixed_one_unreachable_returns_false(self) -> None:
        """If any port is unreachable, wait_until_ready times out."""
        srv = _listening_socket()
        good_port = srv.getsockname()[1]
        bad_port = 59996

        t = threading.Thread(
            target=_accept_and_close, args=(srv,), daemon=True
        )
        t.start()

        handle = TPWorkerHandle(
            process_context=None,
            ports=[good_port, bad_port],
            world_size=2,
        )
        assert (
            handle.wait_until_ready(timeout_s=0.15, interval_s=0.02) is False
        )
        t.join()


# ---------------------------------------------------------------------------
# tp_forward
# ---------------------------------------------------------------------------


class TestTPForward:
    """tp_forward is a stub that always raises RuntimeError."""

    def test_raises_runtime_error(self) -> None:
        with pytest.raises(RuntimeError) as exc:
            tp_forward(
                torch.zeros(1, 10),
                [TPWorkerHandle(process_context=None, ports=[], world_size=1)],
            )
        assert "gRPC transport" in str(exc.value)

    def test_raises_with_none_tensor(self) -> None:
        with pytest.raises(RuntimeError) as exc:
            tp_forward(None, [])
        assert "gRPC transport" in str(exc.value)

    def test_raises_with_empty_handles(self) -> None:
        with pytest.raises(RuntimeError) as exc:
            tp_forward(torch.ones(2, 5), [])
        assert "gRPC transport" in str(exc.value)

    def test_raises_with_none_handles(self) -> None:
        with pytest.raises(RuntimeError) as exc:
            tp_forward(torch.ones(2, 5), None)  # type: ignore[arg-type]
        assert "gRPC transport" in str(exc.value)

    def test_raises_with_single_handle(self) -> None:
        handle = TPWorkerHandle(
            process_context=None, ports=[], world_size=1
        )
        with pytest.raises(RuntimeError) as exc:
            tp_forward(torch.zeros(1), [handle])
        assert "gRPC transport" in str(exc.value)

    def test_raises_large_tensor(self) -> None:
        handle = TPWorkerHandle(
            process_context=None, ports=[], world_size=2
        )
        large = torch.randn(1024, 1024)
        with pytest.raises(RuntimeError) as exc:
            tp_forward(large, [handle, handle])
        assert "gRPC transport" in str(exc.value)


# ---------------------------------------------------------------------------
# launch_tp_workers  (surface-only -- requires CUDA / NCCL at runtime)
# ---------------------------------------------------------------------------


class TestLaunchTPWorkers:
    """launch_tp_workers cannot be invoked without GPUs.

    These tests verify the function is callable as a symbol and check its
    signature, but do not call it.
    """

    def test_symbol_imported(self) -> None:
        assert callable(launch_tp_workers)

    def test_signature(self) -> None:
        import inspect

        sig = inspect.signature(launch_tp_workers)
        assert "model_name" in sig.parameters
        assert "num_gpus" in sig.parameters
        assert sig.parameters["model_name"].default is inspect.Parameter.empty
        assert sig.parameters["num_gpus"].default == 2

    def test_default_port(self) -> None:
        import inspect

        sig = inspect.signature(launch_tp_workers)
        assert sig.parameters["port"].default == 29500

    def test_dtype_default(self) -> None:
        import inspect

        sig = inspect.signature(launch_tp_workers)
        assert sig.parameters["dtype"].default == "float16"
