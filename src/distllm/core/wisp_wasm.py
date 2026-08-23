"""WebAssembly edge inference for distributed LLM.

Provides four components that together enable running inference at the edge:

- ``WasmRuntime`` — runs inference inside a WebAssembly sandbox
- ``EdgeSplitExecutor`` — partitions a model so some layers run on the
  edge device via WASM and the rest are forwarded to the cluster
- ``BrowserClient`` — browser-side inference client over WebSocket
- ``Wisp`` — top-level orchestrator that selects edge, split, or cluster mode

Usage::

    wisp = Wisp()
    output = wisp.run(model_bytes, "Hello world", mode="edge")
    print(output)
    print(wisp.stats())
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from loguru import logger

# ---------------------------------------------------------------------------
# Optional WASM backends
# ---------------------------------------------------------------------------

try:
    import wasmtime  # noqa: F401

    _HAS_WASMTIME = True
except ImportError:
    _HAS_WASMTIME = False

try:
    import wasmer  # noqa: F401

    _HAS_WASMER = True
except ImportError:
    _HAS_WASMER = False

try:
    import aiohttp  # noqa: F401

    _HAS_AIOHTTP = True
except ImportError:
    _HAS_AIOHTTP = False

try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

# ---------------------------------------------------------------------------
# Backend preference ordering
# ---------------------------------------------------------------------------

_BACKEND_PREFERRED = "wasmtime" if _HAS_WASMTIME else "wasmer" if _HAS_WASMER else None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MAX_TOKENS = 512
_DEFAULT_TEMPERATURE = 0.7
_WEBSOCKET_RECONNECT_DELAY_S = 2.0
_WEBSOCKET_MAX_RETRIES = 5

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "WasmRuntime",
    "EdgeSplitExecutor",
    "BrowserClient",
    "Wisp",
    "WasmBackend",
    "SplitPlan",
    "EdgeStats",
]


# ---------------------------------------------------------------------------
# Enums and data classes
# ---------------------------------------------------------------------------


class WasmBackend:
    """Constants identifying available WASM backends."""

    WASMTIME = "wasmtime"
    WASMER = "wasmer"
    NONE = None


@dataclass(frozen=True)
class SplitPlan:
    """Plan describing how a model is split between edge and cluster.

    Attributes:
        edge_layers: Indices of layers to run on the edge device.
        cluster_layers: Indices of layers to run on the cluster.
        estimated_edge_flops: Estimated FLOPS for the edge portion.
        estimated_cluster_flops: Estimated FLOPS for the cluster portion.
    """

    edge_layers: tuple[int, ...]
    cluster_layers: tuple[int, ...]
    estimated_edge_flops: float = 0.0
    estimated_cluster_flops: float = 0.0


@dataclass
class EdgeStats:
    """Runtime statistics for edge vs cluster distribution.

    Attributes:
        edge_count: Number of inferences run fully on edge.
        split_count: Number of inferences run in split mode.
        cluster_count: Number of inferences run fully on cluster.
        total_edge_layers: Cumulative layers processed on edge.
        total_cluster_layers: Cumulative layers processed on cluster.
        total_edge_time_s: Cumulative wall time on edge (seconds).
        total_cluster_time_s: Cumulative wall time on cluster (seconds).
    """

    edge_count: int = 0
    split_count: int = 0
    cluster_count: int = 0
    total_edge_layers: int = 0
    total_cluster_layers: int = 0
    total_edge_time_s: float = 0.0
    total_cluster_time_s: float = 0.0


# ---------------------------------------------------------------------------
# WasmRuntime
# ---------------------------------------------------------------------------


class WasmRuntime:
    """Runs inference inside a WebAssembly sandbox.

    Supports ``wasmtime`` and ``wasmer`` backends. The preferred backend is
    selected automatically based on availability, preferring ``wasmtime``.

    Usage::

        runtime = WasmRuntime()
        compiled = runtime.load_model(model_bytes)
        output = runtime.execute(compiled, {"input": [1, 2, 3]})
    """

    def __init__(self, backend: str | None = None) -> None:
        """Initialize the runtime.

        Args:
            backend: Explicit backend name (``"wasmtime"`` or ``"wasmer"``).
                Defaults to the first available backend.

        Raises:
            RuntimeError: If no WASM backend is available.
        """
        self._backend_name = backend or _BACKEND_PREFERRED

        if self._backend_name == "wasmtime" and not _HAS_WASMTIME:
            logger.warning("wasmtime requested but not installed; falling back to wasmer")
            self._backend_name = "wasmer" if _HAS_WASMER else None
        elif self._backend_name == "wasmer" and not _HAS_WASMER:
            logger.warning("wasmer requested but not installed; falling back to wasmtime")
            self._backend_name = "wasmtime" if _HAS_WASMTIME else None

        if self._backend_name is None:
            raise RuntimeError(
                "No WebAssembly backend available. Install wasmtime or wasmer:\n"
                "  pip install wasmtime\n"
                "  pip install wasmer"
            )

        self._backend: _BackendAdapter = (
            _WasmtimeAdapter() if self._backend_name == "wasmtime" else _WasmerAdapter()
        )
        logger.info("WasmRuntime using backend: {}", self._backend_name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_model(self, model_bytes: bytes) -> Any:
        """Load a WASM module from raw bytes.

        Args:
            model_bytes: Compiled WebAssembly bytecode.

        Returns:
            An opaque compiled module handle understood by :meth:`execute`.
        """
        logger.debug("Loading WASM model ({} bytes)", len(model_bytes))
        return self._backend.load(model_bytes)

    def execute(self, compiled: Any, /, **inputs: Any) -> dict[str, Any]:
        """Run inference on a compiled WASM module.

        Args:
            compiled: Module returned by :meth:`load_model`.
            **inputs: Named input tensors or values.

        Returns:
            Dictionary of named outputs.
        """
        logger.debug("Executing WASM inference")
        return self._backend.execute(compiled, inputs)

    @property
    def backend(self) -> str:
        """Name of the active WASM backend."""
        return self._backend_name  # type: ignore[return-value]

    @property
    def is_available(self) -> bool:
        """True if a WASM backend was successfully initialised."""
        return True


# ---------------------------------------------------------------------------
# Backend adapter interface (internal)
# ---------------------------------------------------------------------------


class _BackendAdapter:
    """Internal adapter wrapping a concrete WASM runtime."""

    def load(self, model_bytes: bytes) -> Any:
        """Compile a WASM module."""
        raise NotImplementedError

    def execute(self, compiled: Any, inputs: dict[str, Any]) -> dict[str, Any]:
        """Run inference."""
        raise NotImplementedError


class _WasmtimeAdapter(_BackendAdapter):
    """Adapter for the ``wasmtime`` backend."""

    def __init__(self) -> None:
        import wasmtime  # type: ignore[import-not-found]

        self._engine = wasmtime.Engine()
        self._store = wasmtime.Store(self._engine)
        self._mod: Any = None
        self._linker = wasmtime.Linker(self._engine)

    def load(self, model_bytes: bytes) -> Any:
        import wasmtime  # type: ignore[import-not-found]

        self._mod = wasmtime.Module(self._engine, model_bytes)
        return self._mod

    def execute(self, compiled: Any, inputs: dict[str, Any]) -> dict[str, Any]:
        import wasmtime  # type: ignore[import-not-found]

        store = wasmtime.Store(self._engine)
        instance = self._linker.instantiate(store, compiled)
        memory = instance.exports(store).get("memory")
        if memory is None:
            raise RuntimeError("WASM module has no 'memory' export")

        # Write inputs into WASM linear memory
        if _HAS_NUMPY and "input" in inputs:
            data = np.asarray(inputs["input"], dtype=np.float32).tobytes()
            mem_export = memory
            ptr = self._allocate(store, instance, len(data))
            mem_export.write(store, data, ptr)

        # Call the exported "infer" function
        infer_fn = instance.exports(store).get("infer")
        if infer_fn is None:
            raise RuntimeError("WASM module has no 'infer' export")

        result_ptr = infer_fn(store, ptr if "input" in inputs else 0)

        # Read output from linear memory
        if _HAS_NUMPY:
            out_bytes = mem_export.read(store, result_ptr, result_ptr + 1024)
            output = np.frombuffer(out_bytes, dtype=np.float32).tolist()
        else:
            output = list(mem_export.read(store, result_ptr, result_ptr + 1024))

        return {"output": output}

    @staticmethod
    def _allocate(store: Any, instance: Any, size: int) -> int:
        """Allocate *size* bytes in WASM linear memory via ``malloc``."""
        malloc_fn = instance.exports(store).get("malloc")
        if malloc_fn is not None:
            return malloc_fn(store, size)
        # Fallback: assume memory starts at offset 0 and caller knows layout
        return 0


class _WasmerAdapter(_BackendAdapter):
    """Adapter for the ``wasmer`` backend."""

    def __init__(self) -> None:
        import wasmer  # type: ignore[import-not-found]

        self._store = wasmer.Store()
        self._mod: Any = None

    def load(self, model_bytes: bytes) -> Any:
        import wasmer  # type: ignore[import-not-found]

        self._mod = wasmer.Module(self._store, model_bytes)
        return self._mod

    def execute(self, compiled: Any, inputs: dict[str, Any]) -> dict[str, Any]:
        import wasmer  # type: ignore[import-not-found]

        import wasmer_compiler_cranelift  # type: ignore[import-not-found]

        store = wasmer.Store()
        module = compiled
        instance = wasmer.Instance(module, {})

        memory = instance.exports.memory
        if not memory:
            raise RuntimeError("WASM module has no 'memory' export")

        if _HAS_NUMPY and "input" in inputs:
            data = np.asarray(inputs["input"], dtype=np.float32).tobytes()
            ptr = instance.exports.malloc(len(data)) if hasattr(instance.exports, "malloc") else 0
            memory.write(ptr, data)

        infer_fn = instance.exports.infer
        result_ptr = infer_fn()

        if _HAS_NUMPY:
            out_bytes = memory.read(result_ptr, result_ptr + 1024)
            output = np.frombuffer(out_bytes, dtype=np.float32).tolist()
        else:
            out_bytes = memory.read(result_ptr, result_ptr + 1024)
            output = list(out_bytes)

        return {"output": output}


# ---------------------------------------------------------------------------
# EdgeSplitExecutor
# ---------------------------------------------------------------------------


class EdgeSplitExecutor:
    """Splits a model's execution between the edge device and the cluster.

    The :meth:`analyze` method produces a :class:`SplitPlan` that describes
    which layers run locally (via WASM) and which are forwarded. The
    :meth:`execute` method carries out the plan.

    Usage::

        executor = EdgeSplitExecutor(wasm_runtime)
        plan = executor.analyze(model_info)
        output = await executor.execute(inputs, plan, cluster_send_fn)
    """

    def __init__(self, wasm_runtime: WasmRuntime) -> None:
        self._runtime = wasm_runtime

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        model: Any,
        total_layers: int = 32,
        edge_flops_capacity: float = 1e12,
        cluster_latency_ms: float = 50.0,
    ) -> SplitPlan:
        """Analyse a model and produce a split plan.

        Args:
            model: Model descriptor (any object with ``num_layers`` or the
                raw WASM module). If a dict is provided, ``num_layers`` is
                read from the ``"num_layers"`` key.
            total_layers: Total number of transformer layers in the model.
            edge_flops_capacity: Estimated FLOPS the edge device can sustain.
            cluster_latency_ms: Estimated round-trip latency to the cluster.

        Returns:
            A :class:`SplitPlan` assigning layers to edge vs cluster.
        """
        if isinstance(model, dict):
            total_layers = model.get("num_layers", total_layers)
        elif hasattr(model, "num_layers"):
            total_layers = model.num_layers  # type: ignore[union-attr]

        # Simple heuristic: keep early layers on edge when edge is fast enough
        # relative to network latency; otherwise push everything to cluster.
        edge_layer_count = self._decide_edge_layer_count(
            total_layers,
            edge_flops_capacity,
            cluster_latency_ms,
        )

        edge_layers = tuple(range(edge_layer_count))
        cluster_layers = tuple(range(edge_layer_count, total_layers))

        estimated_edge_flops = edge_layer_count * (edge_flops_capacity / total_layers)
        estimated_cluster_flops = (total_layers - edge_layer_count) * (edge_flops_capacity / total_layers)

        logger.debug(
            "SplitPlan: {} edge layers, {} cluster layers",
            len(edge_layers),
            len(cluster_layers),
        )

        return SplitPlan(
            edge_layers=edge_layers,
            cluster_layers=cluster_layers,
            estimated_edge_flops=estimated_edge_flops,
            estimated_cluster_flops=estimated_cluster_flops,
        )

    async def execute(
        self,
        inputs: dict[str, Any],
        plan: SplitPlan,
        compiled: Any,
        cluster_send_fn: Callable[[dict[str, Any]], Any] | None = None,
    ) -> dict[str, Any]:
        """Execute a split plan.

        Runs the edge layers locally via :class:`WasmRuntime`, then sends
        the intermediate activations to the cluster for the remaining layers.

        Args:
            inputs: Input tensors.
            plan: The split plan from :meth:`analyze`.
            compiled: Compiled WASM module for the full model (or the edge
                portion).
            cluster_send_fn: Async callable that sends intermediate data to
                the cluster and returns the cluster's output. If ``None``,
                only edge layers are executed.

        Returns:
            Merged output from edge and (optionally) cluster execution.
        """
        start = time.monotonic()

        # Edge pass — run layers assigned to edge
        if plan.edge_layers:
            edge_result = self._runtime.execute(compiled, **inputs)
        else:
            edge_result = inputs

        edge_time = time.monotonic() - start

        # Cluster pass — forward intermediate activations
        cluster_time = 0.0
        if plan.cluster_layers and cluster_send_fn is not None:
            cluster_start = time.monotonic()
            cluster_result = await cluster_send_fn(
                {
                    "intermediate": edge_result,
                    "cluster_layers": list(plan.cluster_layers),
                }
            )
            cluster_time = time.monotonic() - cluster_start
            result = {**edge_result, **cluster_result}
        elif plan.cluster_layers and cluster_send_fn is None:
            logger.warning("Cluster layers assigned but no cluster_send_fn provided; skipping")
            result = edge_result
        else:
            result = edge_result

        logger.debug(
            "Split execution: {:.3f}s edge + {:.3f}s cluster",
            edge_time,
            cluster_time,
        )

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _decide_edge_layer_count(
        total_layers: int,
        edge_flops_capacity: float,
        cluster_latency_ms: float,
    ) -> int:
        """Decide how many layers to run on the edge device.

        Basic heuristic:
        - If network latency is very low, push everything to the cluster.
        - If edge is reasonably capable, keep the first N layers locally.
        - Fraction is proportional to edge capacity vs a reference threshold.
        """
        if edge_flops_capacity < 1e10:  # < 10 GFLOPS → not viable
            return 0
        if cluster_latency_ms < 5.0:  # very fast cluster link → go remote
            return 0

        # Scale edge layers proportionally to capacity (reference: 1 TFLOPS per layer)
        capacity_per_layer = edge_flops_capacity / max(total_layers, 1)
        edge_fraction = min(capacity_per_layer / 1e11, 0.5)  # cap at 50 %
        return max(1, int(total_layers * edge_fraction))


# ---------------------------------------------------------------------------
# BrowserClient
# ---------------------------------------------------------------------------


class BrowserClient:
    """Browser-based inference client over WebSocket.

    Manages a WebSocket session to a browser-side WASM runtime and streams
    generated tokens asynchronously.

    Usage::

        client = BrowserClient(uri="ws://localhost:8765")
        session = await client.generate_session()
        async for token in client.stream_tokens(session, "Hello"):
            print(token)
    """

    def __init__(
        self,
        uri: str = "ws://localhost:8765",
        max_retries: int = _WEBSOCKET_MAX_RETRIES,
        reconnect_delay: float = _WEBSOCKET_RECONNECT_DELAY_S,
    ) -> None:
        if not _HAS_AIOHTTP:
            raise RuntimeError(
                "aiohttp is required for BrowserClient. Install it with: pip install aiohttp"
            )

        self._uri = uri
        self._max_retries = max_retries
        self._reconnect_delay = reconnect_delay
        self._session_id: str | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate_session(self) -> str:
        """Create a new WebSocket inference session.

        Returns:
            A session ID string.
        """
        import aiohttp  # type: ignore[import-not-found]

        self._session_id = uuid4().hex
        retries = 0

        while retries < self._max_retries:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(
                        f"{self._uri}/session/{self._session_id}",
                        timeout=aiohttp.ClientWSTimeout(ws_close=10.0),
                    ) as ws:
                        await ws.send_json({"type": "init", "session_id": self._session_id})
                        resp = await ws.receive(timeout=10.0)
                        data = json.loads(resp.data) if resp.data else {}
                        if data.get("status") == "ok":
                            logger.info("Browser session created: {}", self._session_id)
                            return self._session_id  # type: ignore[return-value]
                        raise RuntimeError(f"Session init failed: {data}")
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                retries += 1
                logger.warning(
                    "Session creation attempt {}/{} failed: {}",
                    retries,
                    self._max_retries,
                    exc,
                )
                if retries >= self._max_retries:
                    raise RuntimeError(f"Could not create session after {retries} retries") from exc
                await asyncio.sleep(self._reconnect_delay)

        raise RuntimeError("Session creation failed (exhausted retries)")

    async def stream_tokens(
        self,
        prompt: str,
        session_id: str | None = None,
        *,
        on_token: Callable[[str], Any] | None = None,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        temperature: float = _DEFAULT_TEMPERATURE,
    ) -> AsyncIterator[str]:
        """Stream generated tokens from the browser runtime.

        Args:
            prompt: Input text prompt.
            session_id: Session ID from :meth:`generate_session`. Falls back
                to the last created session.
            on_token: Optional synchronous callback invoked for each token.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Yields:
            Generated tokens as strings.

        Raises:
            RuntimeError: If connection fails after retries.
        """
        import aiohttp  # type: ignore[import-not-found]

        sid = session_id or self._session_id
        if sid is None:
            raise RuntimeError("No session available. Call generate_session() first.")

        retries = 0

        while retries < self._max_retries:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(
                        f"{self._uri}/infer/{sid}",
                        timeout=aiohttp.ClientWSTimeout(ws_close=10.0),
                    ) as ws:
                        await ws.send_json(
                            {
                                "type": "infer",
                                "prompt": prompt,
                                "max_tokens": max_tokens,
                                "temperature": temperature,
                            }
                        )

                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                data = json.loads(msg.data)
                                token_type = data.get("type")

                                if token_type == "token":
                                    token = data.get("text", "")
                                    if on_token:
                                        on_token(token)
                                    yield token
                                elif token_type == "done":
                                    return
                                elif token_type == "error":
                                    raise RuntimeError(data.get("message", "Unknown error"))
                            elif msg.type == aiohttp.WSMsgType.CLOSED:
                                break
                    # Normal exit — done streaming
                    return

            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                retries += 1
                logger.warning(
                    "Stream attempt {}/{} failed: {}",
                    retries,
                    self._max_retries,
                    exc,
                )
                if retries >= self._max_retries:
                    raise RuntimeError(f"Stream failed after {retries} retries") from exc
                await asyncio.sleep(self._reconnect_delay)

    async def close(self) -> None:
        """Close the current session."""
        self._session_id = None
        logger.info("Browser client session closed")


# ---------------------------------------------------------------------------
# Wisp — top-level orchestrator
# ---------------------------------------------------------------------------


class Wisp:
    """Orchestrates edge inference across WASM, split, and cluster modes.

    Combines :class:`WasmRuntime`, :class:`EdgeSplitExecutor`, and
    :class:`BrowserClient` behind a single interface.

    Usage::

        wisp = Wisp()
        output = await wisp.run(model_bytes, "Hello", mode="edge")
        print(output)
        print(wisp.stats())
    """

    def __init__(
        self,
        wasm_runtime: WasmRuntime | None = None,
        split_executor: EdgeSplitExecutor | None = None,
        browser_client: BrowserClient | None = None,
    ) -> None:
        self._runtime = wasm_runtime or WasmRuntime()
        self._split_executor = split_executor or EdgeSplitExecutor(self._runtime)
        self._browser_client = browser_client
        self._stats = EdgeStats()
        self._compiled_modules: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        model: bytes | str,
        input_text: str,
        *,
        mode: str = "edge",
        model_key: str = "default",
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        temperature: float = _DEFAULT_TEMPERATURE,
        cluster_send_fn: Callable[[dict[str, Any]], Any] | None = None,
        **kwargs: Any,
    ) -> str:
        """Run inference in the specified mode.

        Args:
            model: WASM bytecode (``bytes``) or a path/identifier (``str``)
                that was previously loaded.
            input_text: Input prompt.
            mode: Execution mode — ``"edge"``, ``"split"``, or ``"cluster"``.
            model_key: Key for caching compiled modules across calls.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            cluster_send_fn: Async callback for sending data to the cluster
                (required in ``"split"`` and ``"cluster"`` modes).
            **kwargs: Additional backend-specific arguments.

        Returns:
            Generated text.

        Raises:
            ValueError: If the mode is unknown.
            RuntimeError: If a required dependency is unavailable.
        """
        start = time.monotonic()

        # Ensure model is compiled
        compiled = self._get_or_compile(model, model_key)

        if mode == "edge":
            result = await self._run_edge(compiled, input_text, max_tokens, temperature)
        elif mode == "split":
            result = await self._run_split(
                compiled, input_text, max_tokens, temperature, cluster_send_fn, **kwargs
            )
        elif mode == "cluster":
            result = await self._run_cluster(
                input_text, max_tokens, temperature, cluster_send_fn
            )
        else:
            raise ValueError(
                f"Unknown mode '{mode}'. Expected 'edge', 'split', or 'cluster'."
            )

        elapsed = time.monotonic() - start
        logger.debug("Wisp.run {} mode completed in {:.3f}s", mode, elapsed)
        return result

    def stats(self) -> EdgeStats:
        """Return a snapshot of edge vs cluster distribution statistics."""
        return EdgeStats(
            edge_count=self._stats.edge_count,
            split_count=self._stats.split_count,
            cluster_count=self._stats.cluster_count,
            total_edge_layers=self._stats.total_edge_layers,
            total_cluster_layers=self._stats.total_cluster_layers,
            total_edge_time_s=self._stats.total_edge_time_s,
            total_cluster_time_s=self._stats.total_cluster_time_s,
        )

    def reset_stats(self) -> None:
        """Reset all runtime statistics to zero."""
        self._stats = EdgeStats()

    @property
    def runtime(self) -> WasmRuntime:
        """The underlying :class:`WasmRuntime` instance."""
        return self._runtime

    # ------------------------------------------------------------------
    # Internal mode runners
    # ------------------------------------------------------------------

    async def _run_edge(
        self,
        compiled: Any,
        input_text: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        start = time.monotonic()
        output = self._runtime.execute(compiled, input=input_text, max_tokens=max_tokens, temperature=temperature)
        elapsed = time.monotonic() - start

        self._stats.edge_count += 1
        self._stats.total_edge_layers += self._estimate_layers(compiled)
        self._stats.total_edge_time_s += elapsed

        return self._format_output(output)

    async def _run_split(
        self,
        compiled: Any,
        input_text: str,
        max_tokens: int,
        temperature: float,
        cluster_send_fn: Callable[[dict[str, Any]], Any] | None = None,
        **kwargs: Any,
    ) -> str:
        total_layers = self._estimate_layers(compiled)
        plan = self._split_executor.analyze(
            compiled,
            total_layers=total_layers,
            edge_flops_capacity=kwargs.get("edge_flops_capacity", 1e12),
            cluster_latency_ms=kwargs.get("cluster_latency_ms", 50.0),
        )

        start = time.monotonic()
        output = await self._split_executor.execute(
            {"input": input_text, "max_tokens": max_tokens, "temperature": temperature},
            plan,
            compiled,
            cluster_send_fn=cluster_send_fn,
        )
        elapsed = time.monotonic() - start

        self._stats.split_count += 1
        self._stats.total_edge_layers += len(plan.edge_layers)
        self._stats.total_cluster_layers += len(plan.cluster_layers)
        self._stats.total_edge_time_s += elapsed * (
            len(plan.edge_layers) / max(total_layers, 1)
        )
        self._stats.total_cluster_time_s += elapsed * (
            len(plan.cluster_layers) / max(total_layers, 1)
        )

        return self._format_output(output)

    async def _run_cluster(
        self,
        input_text: str,
        max_tokens: int,
        temperature: float,
        cluster_send_fn: Callable[[dict[str, Any]], Any] | None = None,
    ) -> str:
        if cluster_send_fn is None:
            raise RuntimeError("cluster_send_fn is required for cluster mode")

        start = time.monotonic()
        result = await cluster_send_fn(
            {
                "prompt": input_text,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )
        elapsed = time.monotonic() - start

        self._stats.cluster_count += 1
        self._stats.total_cluster_time_s += elapsed

        return self._format_output(result)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_compile(self, model: bytes | str, model_key: str) -> Any:
        """Return a cached compiled module or load and cache it."""
        if model_key in self._compiled_modules:
            return self._compiled_modules[model_key]

        if isinstance(model, str):
            # Interpret as a file path
            with open(model, "rb") as f:
                model_bytes = f.read()
        else:
            model_bytes = model

        compiled = self._runtime.load_model(model_bytes)
        self._compiled_modules[model_key] = compiled
        return compiled

    @staticmethod
    def _format_output(output: Any) -> str:
        """Convert raw inference output to a string."""
        if isinstance(output, str):
            return output
        if isinstance(output, dict):
            # Try common keys
            for key in ("generated_text", "text", "output", "response"):
                if key in output:
                    val = output[key]
                    if isinstance(val, str):
                        return val
                    if isinstance(val, list):
                        return " ".join(str(v) for v in val)
            return json.dumps(output, default=str)
        if isinstance(output, list):
            return " ".join(str(v) for v in output)
        return str(output)

    @staticmethod
    def _estimate_layers(compiled: Any) -> int:
        """Estimate the number of layers from a compiled module.

        If the module has no metadata, returns a sensible default.
        """
        if hasattr(compiled, "num_layers"):
            return compiled.num_layers  # type: ignore[union-attr]
        if isinstance(compiled, dict):
            return compiled.get("num_layers", 32)
        return 32
