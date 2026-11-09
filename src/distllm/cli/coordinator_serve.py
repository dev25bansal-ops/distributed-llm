"""Blocking lifecycle helpers for ``distllm system coordinator`` (CLI-owned).

The command must stay in the foreground for the lifetime of the coordinator
and expose the gRPC worker listener on ``--port``.  It deliberately does
**not** rely on ``Coordinator.start(blocking=True)``: on the current core
build that path returns immediately (the event it waits on is already set
when it starts waiting), which made ``system coordinator`` log
"Coordinator started" and exit 0 with no socket open for workers.

Instead we start the coordinator with ``blocking=False`` and block on our
own stop primitive — a ``threading.Event`` polled with a short timeout so
Ctrl+C is reliably delivered on Windows.  Shutdown always stops the gRPC
listener first, then the coordinator, so no thread keeps the process alive.
"""

from __future__ import annotations

import os
import socket
import threading
from typing import Any

from loguru import logger

DEFAULT_POLL_INTERVAL_S = 0.5


class _CliCoordinatorNode:
    """Minimal node facade for :class:`distllm.dist.node_service.NodeServer`.

    Mirrors the wrapper the API server builds (``_CoordinatorNode`` inside
    ``distllm/api/server_coordinator.py``) so workers speaking the
    NodeService gRPC protocol can reach the coordinator.
    """

    def __init__(
        self,
        coordinator: Any,
        port: int,
        # Same rationale as start_cli_node_listener: workers join over the
        # network, so bind-all is the intended default with an explicit
        # override available.
        host: str = "0.0.0.0",  # nosec B104
    ) -> None:
        self._coord = coordinator
        self.node_id = "coordinator"
        self.host = host
        self.port = port
        self.start_layer = 0
        self.end_layer = 0
        self.total_layers = 0
        self.healthy = True
        self.partitioner = None

    def forward_fn(self, **kwargs: Any) -> Any:
        return self._coord.generate(**kwargs)

    def health_check(self) -> bool:
        return True

    def _get_device(self) -> str:
        return "cpu"


def _grpc_max_workers() -> int:
    try:
        return int(os.environ.get("DISTLLM_GRPC_MAX_WORKERS", "4"))
    except ValueError:
        return 4


def _safe_server_stop(server: Any) -> None:
    try:
        server.stop()
    except Exception as exc:  # best-effort teardown of a partially started server
        logger.warning("Error stopping gRPC worker listener: {}", exc)


def start_cli_node_listener(
    coordinator: Any,
    port: int | None = None,
    # Binding all interfaces is the point: worker nodes join over the network,
    # so a loopback-only default would make `distllm cluster join` unreachable
    # from any other machine. Operators who want loopback pass host="127.0.0.1".
    # bandit: B104 is about *unintended* bind-all; this one is the documented
    # default with an explicit override, so it is suppressed deliberately.
    host: str = "0.0.0.0",  # nosec B104
) -> Any | None:
    """Start the NodeService gRPC listener for the coordinator (best effort).

    Returns the :class:`~distllm.dist.node_service.NodeServer` on success and
    ``None`` when the listener cannot be started (missing deps, TLS requested
    without cert material, port unusable, ...).  Failure is logged, never
    raised — the blocking coordinator loop is the mandatory behaviour.
    """
    port = int(port or getattr(coordinator, "port", 0) or 0)
    if port <= 0:
        logger.warning("Coordinator port unknown — gRPC worker listener not started")
        return None

    try:
        from distllm.dist.node_service import NodeServer
    except Exception as exc:
        logger.warning(
            "Could not import NodeServer ({}); gRPC worker listener not started", exc
        )
        return None

    config = getattr(coordinator, "config", None)
    cluster_key = (
        getattr(config, "cluster_key", None)
        or os.environ.get("DISTLLM_CLUSTER_KEY")
        or None
    )
    node = _CliCoordinatorNode(coordinator, port, host=host)

    # If the port already accepts connections before we start, gRPC's bind
    # will fail silently (NodeServer discards add_insecure_port's result)
    # and the readiness probe below would be answered by the foreign
    # listener — claiming a listener that is not ours.  Refuse instead.
    try:
        socket.create_connection(("127.0.0.1", port), timeout=1.0).close()
    except OSError:
        pass
    else:
        logger.warning(
            "Port {} is already in use by another process — "
            "gRPC worker listener not started", port,
        )
        return None

    server = NodeServer(
        node,
        port=port,
        max_workers=_grpc_max_workers(),
        cluster_key=cluster_key,
    )
    try:
        server.start()
    except Exception as exc:
        logger.warning(
            "Could not start gRPC worker listener on port {}: {}", port, exc
        )
        _safe_server_stop(server)
        return None

    # Readiness probe: NodeServer ignores add_insecure_port()'s 0-on-failure
    # return, so verify the port actually accepts connections before we
    # claim the listener is up.
    try:
        socket.create_connection(("127.0.0.1", port), timeout=2.0).close()
    except OSError as exc:
        logger.warning(
            "gRPC worker listener did not bind port {}: {}", port, exc
        )
        _safe_server_stop(server)
        return None

    coordinator._node_wrapper = node
    coordinator._node_server = server
    logger.info(
        "Coordinator gRPC worker listener accepting connections on port {}", port
    )
    return server


def _stop_listener(coordinator: Any) -> None:
    """Stop the gRPC listener registered on the coordinator, if any."""
    server = getattr(coordinator, "_node_server", None)
    if server is None:
        return
    _safe_server_stop(server)
    try:
        coordinator._node_server = None
    except Exception:  # pragma: no cover - exotic coordinator objects
        pass


def _graceful_stop(coordinator: Any) -> int:
    try:
        coordinator.stop()
    except Exception as exc:
        logger.error("Coordinator shutdown failed: {}", exc)
        return 1
    return 0


def _wait_for_stop(
    coordinator: Any,
    stop_event: threading.Event | None,
    poll_interval: float,
) -> bool:
    """Block until asked to stop.

    Returns ``True`` when the coordinator's own running flag was cleared
    externally (it already stopped itself), ``False`` when our stop event
    fired.  ``KeyboardInterrupt`` propagates to the caller.
    """
    ev = stop_event if stop_event is not None else threading.Event()
    while True:
        if ev.wait(poll_interval):
            return False
        running = getattr(coordinator, "_running", None)
        if isinstance(running, threading.Event) and not running.is_set():
            return True


def serve_coordinator(
    coordinator: Any,
    *,
    stop_event: threading.Event | None = None,
    poll_interval: float = DEFAULT_POLL_INTERVAL_S,
    start_listener: bool = True,
) -> int:
    """Run the coordinator in the foreground; return a process exit code.

    Composes :func:`start_coordinator` and :func:`wait_and_shutdown` so the
    start and wait phases can also be used separately (``distllm onboard``
    prints guidance between them).
    """
    rc = start_coordinator(coordinator, start_listener=start_listener)
    if rc is not None:
        return rc
    return wait_and_shutdown(
        coordinator, stop_event=stop_event, poll_interval=poll_interval
    )


def start_coordinator(coordinator: Any, *, start_listener: bool = True) -> int | None:
    """Start the coordinator non-blocking and wire the gRPC listener.

    Returns ``None`` on success, or a non-zero process exit code when
    startup fails (the coordinator is left stopped).
    """
    try:
        coordinator.start(blocking=False)
    except KeyboardInterrupt:
        logger.info("Coordinator startup interrupted — shutting down")
        return shutdown_coordinator(coordinator)
    except Exception as exc:
        logger.error("Coordinator failed to start: {}", exc)
        return 1

    if start_listener:
        try:
            start_cli_node_listener(coordinator)
        except Exception as exc:  # never let listener setup kill the daemon
            logger.warning("gRPC worker listener setup failed: {}", exc)
    return None


def shutdown_coordinator(coordinator: Any) -> int:
    """Stop the gRPC listener (if started) and the coordinator itself."""
    _stop_listener(coordinator)
    return _graceful_stop(coordinator)


def wait_and_shutdown(
    coordinator: Any,
    *,
    stop_event: threading.Event | None = None,
    poll_interval: float = DEFAULT_POLL_INTERVAL_S,
) -> int:
    """Block until asked to stop, then shut the coordinator down.

    Returns 0 for a graceful stop (stop event, Ctrl+C, or the coordinator
    already having stopped itself) and 1 when shutdown raises.
    """
    try:
        stopped_externally = _wait_for_stop(coordinator, stop_event, poll_interval)
    except KeyboardInterrupt:
        logger.info("Interrupt received — shutting down coordinator...")
        stopped_externally = False

    _stop_listener(coordinator)
    if stopped_externally:
        return 0
    return _graceful_stop(coordinator)
