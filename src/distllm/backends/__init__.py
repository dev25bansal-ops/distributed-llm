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

# ── Built-in backends (lazy imports — some trigger circular chains) ────
# Deferred to _register_builtins() to avoid the import cycle:
#   backends.pytorch_backend → models.partitioner → dist.fsdp → dist.worker
#   → models.partitioner (circular)
# We define placeholders here for __all__ and import lazily in _register_builtins.
PyTorchNodeAdapter = None
VLLMNodeAdapter = None
LlamacppNodeAdapter = None
ExLlamaV2NodeAdapter = None
ONNXNodeAdapter = None
TensorRTLLMAdapter = None
TritonNodeAdapter = None
NimNodeAdapter = None
MLXNodeAdapter = None
WebGPUNodeAdapter = None
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
    "TritonNodeAdapter",
    "NimNodeAdapter",
    "MLXNodeAdapter",
    "WebGPUNodeAdapter",
    # Config
    "BackendConfig",
]

# ── Auto-register built-in backends ────────────────────────────────────

def _register_builtins() -> None:
    """Register all built-in backends with lazy imports.

    Import is deferred inside the loop to break the circular chain:
    pytorch_backend → models.partitioner → dist.fsdp → dist.worker → models.partitioner
    """
    _BACKEND_MODULES = [
        ("pytorch_backend", "PyTorchNodeAdapter", "pytorch"),
        ("vllm_backend", "VLLMNodeAdapter", "vllm"),
        ("llamacpp_backend", "LlamacppNodeAdapter", "llamacpp"),
        ("exllama_backend", "ExLlamaV2NodeAdapter", "exllama"),
        ("onnx_backend", "ONNXNodeAdapter", "onnx"),
        ("tensorrt_backend", "TensorRTLLMAdapter", "tensorrt"),
        ("triton_backend", "TritonNodeAdapter", "triton"),
        ("nim_backend", "NimNodeAdapter", "nim"),
        ("mlx_backend", "MLXNodeAdapter", "mlx"),
        ("webgpu_backend", "WebGPUNodeAdapter", "webgpu"),
    ]
    import importlib
    for mod_name, cls_name, reg_name in _BACKEND_MODULES:
        try:
            mod = importlib.import_module(f".{mod_name}", __package__)
            adapter_cls = getattr(mod, cls_name, None)
            if adapter_cls is None:
                logger.warning(f"Backend {mod_name} has no class {cls_name}")
                continue
            # Store on this module for direct access
            globals()[cls_name] = adapter_cls
            BackendRegistry.register(adapter_cls, name=reg_name, force=False)
        except KeyError:
            pass
        except Exception as exc:
            logger.debug(f"Backend '{reg_name}' not available: {exc}")

_register_builtins()
