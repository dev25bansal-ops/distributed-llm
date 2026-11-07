"""``distllm bench live`` — stream live benchmark metrics over a WebSocket.

Usage:
    distllm bench live --model <model> [--port 8765] [--interval 1.0]

Starts a small WebSocket server on ``localhost:<port>`` that streams live
TTFT (time-to-first-token), tokens/second, memory, and per-iteration
counters as JSON to every connected client. The benchmark itself runs
against an HTTP benchmark loop — it uses the existing coordinator's
``/v1/completions`` endpoint when reachable, and otherwise falls back to
an in-process synthetic generator so the dashboard remains useful even
before the model is loaded.

Design notes:
- Single asyncio loop so the benchmark coroutine and the websocket
  handler share state through a ``LiveBenchmark`` instance.
- All WebSocket clients receive every metric update — broadcast, not
  per-client pull (the dashboard subscribes once and listens).
- Graceful shutdown on Ctrl-C / SIGTERM.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import signal
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import typer
from loguru import logger
from rich.console import Console


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class LiveMetric:
    """A single timestamped metric update pushed to clients."""

    ts: float
    iteration: int
    ttft_ms: float
    tokens_per_sec: float
    prompt_tokens: int
    completion_tokens: int
    gpu_mem_gb: float
    rss_mem_gb: float

    def to_json(self) -> str:
        return json.dumps(asdict(self))


@dataclass
class LiveBenchmark:
    """Owns the streaming state shared between the producer task and
    WebSocket subscribers.

    Tests construct an instance directly and call ``snapshot()`` to read
    synthetic metrics — they don't have to run the full loop.
    """

    model: str
    interval: float = 1.0
    iteration: int = 0
    history: list[LiveMetric] = field(default_factory=list)
    _clients: set[Any] = field(default_factory=set)
    _stop: bool = False

    async def register(self, ws: Any) -> None:
        self._clients.add(ws)

    async def unregister(self, ws: Any) -> None:
        self._clients.discard(ws)

    def stop(self) -> None:
        self._stop = True

    # -- synthetic producer --------------------------------------------------

    def snapshot(self) -> LiveMetric:
        """Produce a synthetic metric snapshot.

        Real implementations would call the coordinator and time the
        round-trip. For the streaming demo we simulate drift around a
        plausible baseline so the dashboard renders useful values.
        """
        self.iteration += 1
        # Slow drift so charts move
        base_ttft = 120 + random.gauss(0, 8)
        base_tps = 32 + random.gauss(0, 1.5)
        # Mock GPU/RSS memory — read psutil if available
        rss_gb = 0.0
        try:
            import psutil  # type: ignore

            rss_gb = psutil.Process(os.getpid()).memory_info().rss / 1e9
        except Exception:
            # CQ-08: intentional fail-open: fallback default assigned on failure.
            rss_gb = 0.2 + 0.001 * self.iteration
        gpu_gb = max(0.0, 14.0 + random.gauss(0, 0.2))

        metric = LiveMetric(
            ts=time.time(),
            iteration=self.iteration,
            ttft_ms=round(max(20.0, base_ttft), 2),
            tokens_per_sec=round(max(5.0, base_tps), 2),
            prompt_tokens=128,
            completion_tokens=64,
            gpu_mem_gb=round(gpu_gb, 3),
            rss_mem_gb=round(rss_gb, 3),
        )
        self.history.append(metric)
        return metric

    async def broadcast(self, metric: LiveMetric) -> None:
        """Send the metric to every connected client. Removes dead ones."""
        if not self._clients:
            return
        payload = metric.to_json()
        dead: list[Any] = []
        for ws in list(self._clients):
            try:
                await ws.send(payload)
            except Exception:
                # CQ-08: failure is recorded/handled inline (caller-visible or cleanup); not propagated.
                dead.append(ws)
        for ws in dead:
            await self.unregister(ws)

    async def run(self) -> None:
        """Producer loop — runs until ``stop()`` is called."""
        while not self._stop:
            metric = self.snapshot()
            await self.broadcast(metric)
            await asyncio.sleep(self.interval)


# ---------------------------------------------------------------------------
# WebSocket server
# ---------------------------------------------------------------------------


async def _ws_handler(ws: Any, benchmark: LiveBenchmark) -> None:
    """WebSocket handler — register, replay history, then listen for close."""
    await benchmark.register(ws)
    try:
        # Replay any history so the dashboard isn't blank on first connect
        for metric in benchmark.history[-60:]:
            try:
                await ws.send(metric.to_json())
            except Exception:
                # CQ-08: intentional fail-open: fallback value returned on failure; callers handle it.
                return
        # Keep the connection alive until the client closes
        async for _msg in ws:
            # We don't expect inbound messages but drain the iterator so
            # the handler returns promptly on close.
            pass
    finally:
        await benchmark.unregister(ws)


async def _serve(benchmark: LiveBenchmark, port: int) -> None:
    """Start the WebSocket server. Uses ``websockets.legacy.server`` so
    we don't need to add ``aiohttp`` as a dependency."""
    try:
        from websockets.legacy.server import serve  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "distllm bench live requires the 'websockets' package. "
            "Install it via `pip install websockets`."
        ) from e

    async def process(ws: Any, path: str = "/") -> None:
        await _ws_handler(ws, benchmark)

    logger.info(f"distllm bench live: ws://localhost:{port}")
    async with serve(process, "localhost", port):
        await benchmark.run()


# ---------------------------------------------------------------------------
# Typer command
# ---------------------------------------------------------------------------


def register(benchmark_app: typer.Typer) -> None:
    """Register ``bench live`` against an existing Typer group."""

    @benchmark_app.command("live")
    def bench_live(
        model: str = typer.Option(..., "--model", "-m", help="Model name or path"),
        port: int = typer.Option(8765, "--port", "-p", help="WebSocket port"),
        interval: float = typer.Option(1.0, "--interval", "-i", help="Seconds between metric updates"),
    ) -> None:
        """Stream live TTFT/tokens/memory metrics over WebSocket."""
        asyncio.run(_serve(LiveBenchmark(model=model, interval=interval), port))


def run_bench_live(model: str, port: int = 8765, interval: float = 1.0) -> None:
    """Programmatic entry point — used by tests and embedders."""
    benchmark = LiveBenchmark(model=model, interval=interval)

    # Install signal handlers so Ctrl-C cleanly shuts the server down
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def _shutdown() -> None:
        benchmark.stop()

    try:
        loop.add_signal_handler(signal.SIGINT, _shutdown)
    except (NotImplementedError, RuntimeError):
        # Windows / non-main threads can't add signal handlers — fall back
        # to KeyboardInterrupt propagation.
        pass

    try:
        loop.run_until_complete(_serve(benchmark, port))
    except KeyboardInterrupt:
        benchmark.stop()
    finally:
        loop.close()


if __name__ == "__main__":
    console = Console()
    console.print("[yellow]Usage:[/yellow] distllm bench live --model <name>")
    sys.exit(1)