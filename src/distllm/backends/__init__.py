"""Inference backends for distributed-llm worker nodes.

Each backend adapter implements ``BackendAdapter`` (see ``protocol.py``),
providing a consistent ``forward()`` interface that the gRPC ``NodeService``
can call regardless of which inference engine is used.

Built-in backends are auto-registered on import. Third-party backends can
register via ``BackendRegistry.register(MyAdapterClass)``.

Usage:
    # Auto-select best backend for this machine
    from distllm.backends.registry import select_backend

    BackendCls = select_backend()
    adapter = BackendCls(model_name="HuggingFaceTB/SmolLM-135M")
    adapter.load_model()
    logits, kv = adapter.forward(input_ids=input_ids)

    # Named backend
    from distllm.backends import get_backend, list_available_backends

    vllm_cls = get_backend("vllm")
    print(list_available_backends())
"""

from __future__ import annotations

from loguru import logger

# ── BackendAdapter protocol ────────────────────────────────────────────
from .protocol import BackendAdapter

# ── Registry ───────────────────────────────────────────────────────────
from .registry import (
    BackendPlugin,
    BackendRegistry,
    get_backend,
    list_backends,
    list_available_backends,
    select_backend,
)

# ── Built-in backends (register eagerly) ───────────────────────────────
from .pytorch_backend import PyTorchNodeAdapter
from .vllm_backend import VLLMNodeAdapter
from .llamacpp_backend import LlamacppNodeAdapter
from .exllama_backend import ExLlamaV2NodeAdapter
from .onnx_backend import ONNXNodeAdapter
from .tensorrt_backend import TensorRTLLMAdapter

# ── Configuration ──────────────────────────────────────────────────────
from .config import BackendConfig

__all__ = [
    # Protocol
    "BackendAdapter",
    # Registry
    "BackendPlugin",
    "BackendRegistry",
    "get_backend",
    "list_backends",
    "list_available_backends",
    "select_backend",
    # Built-in backends
    "PyTorchNodeAdapter",
    "VLLMNodeAdapter",
    "LlamacppNodeAdapter",
    "ExLlamaV2NodeAdapter",
    "ONNXNodeAdapter",
    "TensorRTLLMAdapter",
    # Config
    "BackendConfig",
]

# ── Auto-register built-in backends ────────────────────────────────────

def _register_builtins() -> None:
    builtins = [
        (PyTorchNodeAdapter, "pytorch"),
        (VLLMNodeAdapter, "vllm"),
        (LlamacppNodeAdapter, "llamacpp"),
        (ExLlamaV2NodeAdapter, "exllama"),
        (ONNXNodeAdapter, "onnx"),
        (TensorRTLLMAdapter, "tensorrt"),
    ]
    for adapter_cls, name in builtins:
        try:
            BackendRegistry.register(adapter_cls, name=name, force=False)
        except KeyError:
            pass  # already registered (e.g. by a third-party plugin)
        except Exception as exc:
            logger.warning(
                f"Failed to register built-in backend '{name}': {exc}"
            )

_register_builtins()
