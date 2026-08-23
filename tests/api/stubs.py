"""Lightweight stub classes for API tests -- no MagicMock, AsyncMock, or unittest.mock.patch.

Every class in this module is a plain Python class (or ``SimpleNamespace``) that
provides the subset of the interface that the route handlers actually touch.
"""

from __future__ import annotations

import functools
import sys
from types import SimpleNamespace
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Call tracker (replaces assert_called_once_with / assert_called)
# ---------------------------------------------------------------------------


class CallTracker:
    """Minimal call-recording helper.

    Usage::

        tracker = CallTracker()
        tracker("arg1", key="val")
        assert tracker.calls == [(("arg1",), {"key": "val"})]
        assert tracker.call_count == 1
    """

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append((args, kwargs))

    @property
    def call_count(self) -> int:
        return len(self.calls)


def call_recorder() -> CallTracker:
    """Return a ``CallTracker`` that also supports ``.assert_called_once_with(...)``."""
    return CallTracker()


def record_func(tracker: CallTracker) -> Callable:
    """Wrap *tracker* so its ``.assert_called_once_with`` works like mock."""

    def wrapped(*args: Any, **kwargs: Any) -> None:
        tracker(*args, **kwargs)

    wrapped.assert_called_once_with = _make_asserter(tracker)
    return wrapped


def _make_asserter(tracker: CallTracker) -> Callable:
    def asserter(*args: Any, **kwargs: Any) -> None:
        assert tracker.calls == [(args, kwargs)], (
            f"Expected call {args, kwargs}, got {tracker.calls}"
        )

    return asserter


# ---------------------------------------------------------------------------
# Simple call-tracking method builder
# ---------------------------------------------------------------------------


def make_method(return_value: Any = None) -> Callable:
    """Create a simple callable that returns *return_value*."""
    return lambda *args, **kwargs: return_value


# ---------------------------------------------------------------------------
# Coordinator stub
# ---------------------------------------------------------------------------


class CoordinatorStub:
    """Minimal coordinator stand-in.

    Provides the attributes and methods that route handlers reference.
    """

    def __init__(self) -> None:
        self.model_name = "test-model"
        self.total_layers = 24
        self.max_batch_size = 4
        self.max_tokens_per_batch = 4096
        self.nodes: dict[str, Any] = {}
        self.node_order: list[str] = []
        self._shutting_down = False
        self.scheduler: Any = None
        self.prefix_cache: Any = None
        self.metrics_exporter: Any = None
        self.tokenizer: Any = None
        self.local_partitioner: Any = None
        self._vlm_pipeline: Any = None
        self._spec_decoder: Any = None
        self._gossip_protocol: Any = None
        self._rag_pipeline: Any = None
        self._agent_loop: Any = None
        self._pipeline_composer: Any = None
        self._disagg_orchestrator: Any = None
        self._self_optimizing: Any = None
        self._replay_buffer: Any = SimpleNamespace(store=lambda *a, **kw: None, get=lambda *a, **kw: None, export=lambda: [])
        self._version_manager: Any = None
        self.adapter_manager: Any = None
        self._slora_manager: Any = None
        self._whisper_model: Any = None
        self._whisper_processor: Any = None
        self._diffusion_pipe: Any = None
        self.fine_tuning_backend: Any = None
        self._resource_mgr: Any = None

    def generate(self, prompt: str, *args: Any, **kwargs: Any) -> str:
        return "test response"

    def list_models(self) -> list[str]:
        return [self.model_name]

    def health_check(self) -> dict[str, Any]:
        return {}

    def get_metrics(self) -> dict[str, Any]:
        return {"requests_total": 42}

    def get_recent_requests(self, n: int = 10) -> list[Any]:
        return []

    def replay_request(self, request_id: str) -> Any:
        return None


# ---------------------------------------------------------------------------
# Scheduler stub
# ---------------------------------------------------------------------------

class SchedulerStub:
    """Minimal scheduler stand-in."""

    def __init__(self) -> None:
        self.default_temperature = 0.7
        self.default_top_p = 0.9
        self.default_top_k = 50


# ---------------------------------------------------------------------------
# NodeInfo stub
# ---------------------------------------------------------------------------


class NodeInfoStub:
    """Minimal node-info stand-in for admin tests."""

    def __init__(
        self,
        node_id: str = "node-0",
        host: str = "127.0.0.1",
        port: int = 50051,
        healthy: bool = True,
        start_layer: int = 0,
        end_layer: int = 11,
        gpu_name: str = "Tesla V100",
        gpu_memory_total: int = 16384,
        gpu_memory_free: int = 8192,
        gpu_sm_count: int = 80,
        role: str = "AUTO",
        cluster_id: str = "default",
        last_health_time: float = 1000.0,
    ) -> None:
        self.node_id = node_id
        self.host = host
        self.port = port
        self.healthy = healthy
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.gpu_name = gpu_name
        self.gpu_memory_total = gpu_memory_total
        self.gpu_memory_free = gpu_memory_free
        self.gpu_sm_count = gpu_sm_count
        self.role = role
        self.cluster_id = cluster_id
        self.last_health_time = last_health_time


# ---------------------------------------------------------------------------
# Tokenizer stub
# ---------------------------------------------------------------------------


def make_tokenizer_stub(vocab_size: int = 1000) -> SimpleNamespace:
    """Build a tokenizer stub with encode/decode that return canned tokens."""

    def encode(text: str, **kwargs: Any) -> list[int] | Any:
        tokens = [1, 2, 3, 4, 5]
        if kwargs.get("return_tensors") == "pt":
            import torch  # type: ignore[import-untyped]
            return torch.tensor([tokens])
        return tokens

    def decode(tokens: int | list[int] | Any, **kwargs: Any) -> str:
        if isinstance(tokens, int):
            lst = [tokens]
        elif isinstance(tokens, list):
            lst = tokens
        else:
            lst = tokens.tolist()
        return " ".join(f"tok-{t}" for t in lst)

    return SimpleNamespace(
        encode=encode,
        decode=decode,
        eos_token_id=0,
    )


# ---------------------------------------------------------------------------
# Torch model / output stubs
# ---------------------------------------------------------------------------


def make_model_stub() -> SimpleNamespace:
    """Create a stub ``model`` object with ``parameters()`` and ``__call__``."""
    return SimpleNamespace(
        parameters=lambda: iter([__import__("torch").randn(10, 10)]),
        return_value=SimpleNamespace(
            logits=__import__("torch").randn(1, 5, 1000),
            past_key_values=None,
        ),
        __call__=lambda *a, **kw: SimpleNamespace(
            logits=__import__("torch").randn(1, 5, 1000),
            past_key_values=None,
        ),
    )


def make_partitioner_stub() -> SimpleNamespace:
    """Create a stub ``local_partitioner`` with a ``full_model`` attribute."""
    return SimpleNamespace(full_model=make_model_stub())


# ---------------------------------------------------------------------------
# Patch dictionary helper (replaces ``with patch.dict("sys.modules", ...)``)
# ---------------------------------------------------------------------------


def patch_sys_modules(
    mod_map: dict[str, Any],
) -> dict[str, Any]:
    """Temporarily insert *mod_map* into ``sys.modules`` and return the originals.

    Use with ``try/finally``::

        saved = patch_sys_modules({"torch": object()})
        try:
            ...
        finally:
            for k in saved:
                sys.modules[k] = saved[k]
    """
    saved: dict[str, Any] = {}
    for mod_name, obj in mod_map.items():
        saved[mod_name] = sys.modules.get(mod_name)
        sys.modules[mod_name] = obj
    return saved


# ---------------------------------------------------------------------------
# Request stub (minimal FastAPI Request-like object)
# ---------------------------------------------------------------------------


def make_request_stub(
    role: str | None = "admin",
    key_id: str = "test-key",
) -> SimpleNamespace:
    """Build a minimal request-like object with ``state.api_key_role``."""
    return SimpleNamespace(
        state=SimpleNamespace(
            api_key_role=role,
            api_key_id=key_id,
        )
    )


# ---------------------------------------------------------------------------
# Adapter manager stub (simple callable-based)
# ---------------------------------------------------------------------------


class AdapterManagerStub:
    """Minimal adapter manager with call-recording on mutation methods."""

    def __init__(self) -> None:
        self.active_adapter: str | None = None
        self.load_calls: list[tuple] = []
        self.unload_calls: list[tuple] = []
        self.set_active_calls: list[tuple] = []

    def load_adapter(self, adapter_id: str, path: str, **kwargs: Any) -> None:
        self.load_calls.append((adapter_id, path, kwargs))

    def unload_adapter(self, adapter_id: str) -> bool:
        self.unload_calls.append((adapter_id,))
        return True

    def set_active(self, adapter_id: str) -> None:
        self.set_active_calls.append((adapter_id,))

    def list_adapters(self) -> list[str]:
        return []

    def rank_adapters(self) -> list[Any]:
        return []

    def get_stats(self) -> dict[str, Any]:
        return {}

    def get_adapter_info(self, adapter_id: str) -> Any:
        return None

    def warmup_adapters(self, adapters: dict[str, str], **kwargs: Any) -> int:
        return len(adapters)


# ---------------------------------------------------------------------------
# Version manager stub
# ---------------------------------------------------------------------------


class VersionManagerStub:
    """Minimal version manager."""

    def __init__(self) -> None:
        self.versions: dict[str, Any] = {}

    def register_version(self, version_id: str, model_path: str, **kwargs: Any) -> Any:
        obj = SimpleNamespace(
            version_id=version_id,
            model_id=kwargs.get("model_id", "model-1"),
            model_path=model_path,
            status=SimpleNamespace(value=kwargs.get("status", "stable")),
            created_at=kwargs.get("created_at", 1000.0),
            promoted_at=kwargs.get("promoted_at", 2000.0),
            traffic_weight=kwargs.get("traffic_weight", 0.5),
        )
        self.versions[version_id] = obj
        return obj

    def list_versions(self) -> list[Any]:
        return list(self.versions.values())

    def get_version(self, version_id: str) -> Any:
        return self.versions.get(version_id)

    def get_version_stats(self, version_id: str) -> Any:
        v = self.versions.get(version_id)
        if v is None:
            return None
        return {
            "version_id": version_id,
            "status": v.status.value,
            "traffic_weight": v.traffic_weight,
            "total_requests": 100,
            "error_rate": 0.01,
            "avg_latency_ms": 150.0,
        }

    def delete_version(self, version_id: str) -> bool:
        if version_id in self.versions:
            del self.versions[version_id]
            return True
        return False

    def promote_version(self, version_id: str) -> bool:
        return version_id in self.versions

    def evaluate_promotion(self, stable_version: str, candidate_version: str) -> dict[str, Any]:
        return {
            "sample_a": 50,
            "sample_b": 50,
            "sufficient_samples": True,
            "recommendation": "promote",
            "reason": "better performance",
        }

    def get_shadow_comparisons(self) -> list[dict[str, Any]]:
        return []

    def switch_color(self) -> str:
        return "green"

    def rollback_color(self) -> str:
        return "blue"


# ---------------------------------------------------------------------------
# RAG pipeline stub
# ---------------------------------------------------------------------------


class RagPipelineStub:
    """Minimal RAG pipeline."""

    def ingest(self, document_id: str, content: str, **kwargs: Any) -> int:
        return 3

    def retrieve(self, query: str, top_k: int = 5) -> list[Any]:
        return []

    def stats(self) -> dict[str, Any]:
        return {"documents": 5, "chunks": 20, "index_size": 4096}

    def save_index(self) -> None:
        pass

    def build_rag_prompt(self, query: str, base_prompt: str) -> str:
        return "enriched prompt"


# ---------------------------------------------------------------------------
# Fine-tuning backend stub
# ---------------------------------------------------------------------------


class FineTuningBackendStub:
    """Minimal fine-tuning backend."""

    def train(self, *args: Any, **kwargs: Any) -> int:
        return 0


# ---------------------------------------------------------------------------
# Async-method stub helper for disagg orchestrator etc.
# ---------------------------------------------------------------------------


async def _async_none(*args: Any, **kwargs: Any) -> Any:
    return None


async def _async_val(val: Any) -> Any:
    return val


class DisaggOrchestratorStub:
    """Minimal disaggregated orchestrator."""

    def __init__(self) -> None:
        self.router = SimpleNamespace(
            prefill_pool=SimpleNamespace(_nodes={}),
            decode_pool=SimpleNamespace(_nodes={}),
            add_prefill_node=_async_none,
            add_decode_node=_async_none,
        )

    async def submit(self, *args: Any, **kwargs: Any) -> str:
        return "disagg-1"

    async def get_result(self, request_id: str) -> Any:
        return [1, 2, 3]

    def health_check(self) -> dict[str, Any]:
        return {
            "healthy": True,
            "pending_requests": 0,
            "prefill_pool": {"total_nodes": 0, "active_nodes": 0},
            "decode_pool": {"total_nodes": 0, "active_nodes": 0},
        }


# ---------------------------------------------------------------------------
# Pipeline composer stub
# ---------------------------------------------------------------------------


class PipelineComposerStub:
    """Minimal pipeline composer."""

    def __init__(self) -> None:
        self._pipelines: dict[str, Any] = {}

    def execute(self, steps: list[Any], input_data: str) -> Any:
        async def _iter() -> Any:
            yield {
                "step_index": 0,
                "step_type": "transform",
                "output": "hello world",
                "latency_ms": 5.0,
                "error": None,
            }
            yield {
                "step_index": 1,
                "step_type": "complete",
                "output": "hello world",
                "latency_ms": 10.0,
                "error": None,
            }

        return _iter()

    def get(self, pipeline_id: str) -> Any:
        return self._pipelines.get(pipeline_id)

    def register(self, pipeline_id: str, steps: list[Any], **kwargs: Any) -> None:
        self._pipelines[pipeline_id] = {
            "pipeline_id": pipeline_id,
            "steps": steps,
            **kwargs,
        }


# ---------------------------------------------------------------------------
# Agent loop stub
# ---------------------------------------------------------------------------


class AgentLoopStub:
    """Minimal agent loop."""

    def run(self, goal: str, tools: list[Any]) -> dict[str, Any]:
        return {"result": "task done", "iterations": 3, "memory": [{"step": 1}]}

    def get_state(self) -> dict[str, Any]:
        return {"state": "idle", "memory": []}


# ---------------------------------------------------------------------------
# Optimization engine stub
# ---------------------------------------------------------------------------


class OptimizationEngineStub:
    """Minimal self-optimizing engine."""

    def __init__(self) -> None:
        self._running = True

    def stats(self) -> dict[str, Any]:
        return {"total_operations": 42}

    def get_suggestions(self) -> dict[str, Any]:
        return {
            "batch_size": 8,
            "kv_cache_quantization": False,
            "speculative_decoding": False,
        }


# ---------------------------------------------------------------------------
# Metrics exporter stub (supports .labels(**kw).inc(val) chain)
# ---------------------------------------------------------------------------


class _MetricRecorder:
    """Records ``.labels(**kw).inc(val)`` call sequences."""

    def __init__(self) -> None:
        self.label_kwargs: list[dict[str, Any]] = []
        self.inc_values: list[float] = []
        self._last_labels_return: _MetricIncHandle | None = None

    def labels(self, **kw: Any) -> "_MetricIncHandle":
        self.label_kwargs.append(kw)
        handle = _MetricIncHandle(self)
        self._last_labels_return = handle
        return handle

    @property
    def called(self) -> bool:
        return len(self.label_kwargs) > 0

    def assert_labels_called_once_with(self, **expected: Any) -> None:
        assert len(self.label_kwargs) == 1, (
            f"Expected 1 labels() call, got {len(self.label_kwargs)}: {self.label_kwargs}"
        )
        assert self.label_kwargs[0] == expected, (
            f"Expected labels({expected}), got labels({self.label_kwargs[0]})"
        )

    def assert_not_called(self) -> None:
        assert len(self.label_kwargs) == 0, (
            f"Expected no labels() calls, got {len(self.label_kwargs)}"
        )


class _MetricIncHandle:
    """Handle returned by labels(), tracking inc() calls."""

    def __init__(self, recorder: _MetricRecorder) -> None:
        self._recorder = recorder

    def inc(self, val: float = 1.0) -> None:
        self._recorder.inc_values.append(val)

    def assert_called_once_with(self, val: float) -> None:
        assert len(self._recorder.inc_values) == 1, (
            f"Expected 1 inc() call, got {len(self._recorder.inc_values)}"
        )
        assert self._recorder.inc_values[0] == val, (
            f"Expected inc({val}), got inc({self._recorder.inc_values[0]})"
        )


class MetricsExporterStub:
    """Minimal metrics exporter stand-in."""

    def __init__(self) -> None:
        self.requests_total = _MetricRecorder()
        self.request_latency = _MetricRecorder()
        self.request_duration_seconds = _MetricRecorder()
        self.errors_total = _MetricRecorder()
        self.request_cost_total = _MetricRecorder()
        self.request_gpu_hours = _MetricRecorder()


# ---------------------------------------------------------------------------
# Anomaly detector stub
# ---------------------------------------------------------------------------


class AnomalyDetectorStub:
    """Minimal anomaly detector stand-in."""

    def __init__(self) -> None:
        self.record_calls: list[tuple[str, float]] = []

    def record(self, name: str, value: float) -> None:
        self.record_calls.append((name, value))
