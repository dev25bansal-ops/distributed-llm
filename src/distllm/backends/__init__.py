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
from .protocol import BackendAdapter, ForwardCapable, GenerateOnly

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
OllamaBackendAdapter = None
TGIBackendAdapter = None
from .config import BackendConfig

__all__ = [
    # Protocol
    "BackendAdapter",
    "ForwardCapable",
    "GenerateOnly",
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
    "OllamaBackendAdapter",
    "TGIBackendAdapter",
    # Config
    "BackendConfig",
]

# ── Known optional engine SDKs ───────────────────────────────────────────
# Maps backend registration name → module path of the engine SDK it imports.
# These are expected to fail when the SDK isn't installed; such failures stay
# at debug level. Any other exception indicates a real bug and gets logged
# at warning level (operator-visible at default log level).
_OPTIONAL_ENGINE_SDKS = {
    "vllm": "vllm",
    "llamacpp": "llama_cpp",
    "exllama": "exllamav2",
    "onnx": "onnxruntime",
    "tensorrt": "tensorrt_llm",
    "triton": "tritonclient",
    "mlx": "mlx",
    "webgpu": "aiortc",
    "nim": "",           # No required Python package; network-only dependency
    "ollama": "",         # HTTP-only; no client required
    "tgi": "",            # HTTP-only; no client required
}


def _is_optional_sdk_error(exc: BaseException, backend_name: str) -> bool:
    """Check whether *exc* is a missing optional SDK for the given backend.

    Returns ``True`` when the exception indicates an expected, non-critical
    dependency absence (operator should see this at debug level only):

      - ModuleNotFoundError where ``exc.name`` matches the known SDK module.
      - ImportError whose msg mentions the missing SDK module.

    All other exceptions (SyntaxError, AttributeError, ImportError for local
    modules, etc.) return ``False``, meaning they are likely bugs and should
    be surfaced at warning/error level.

    Args:
        exc: The caught exception.
        backend_name: The backend registration name (e.g. ``"vllm"``).

    Returns:
        ``True`` if this is an optional-sdk-not-found situation; ``False`` otherwise.
    """
    sdk_module = _OPTIONAL_ENGINE_SDKS.get(backend_name, "")
    if not sdk_module:
        # Backends without a required Python package (nim, ollama, tgi) always
        # get treated as "maybe-a-bug" so that real code errors become visible.
        return False

    if isinstance(exc, ModuleNotFoundError):
        # exc.name is the submodule that couldn't be imported (may be None).
        missing = getattr(exc, "name", None) or ""
        if missing == sdk_module or missing.startswith(sdk_module + "."):
            return True
        # Also check the base package (e.g. 'tritonclient.grpc' → 'tritonclient').
        if "." in missing and missing.split(".")[0] == sdk_module:
            return True
        # Fall back to the message when the exception was constructed without
        # a ``name`` argument (real import machinery always sets it).
        if not missing and f"'{sdk_module}'" in str(exc):
            return True

    if isinstance(exc, ImportError):
        msg = getattr(exc, "msg", None) or str(exc)
        # Soft-match: if the ImportError message references the SDK module.
        if sdk_module in msg:
            return True

    return False


# (module name, adapter class name, registration name) for each built-in backend.
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
    ("ollama_backend", "OllamaBackendAdapter", "ollama"),
    ("tgi_backend", "TGIBackendAdapter", "tgi"),
]

# ── Auto-register built-in backends ────────────────────────────────────


def _register_builtins() -> None:
    """Register all built-in backends with lazy imports.

    Import is deferred inside the loop to break the circular chain:
    pytorch_backend → models.partitioner → dist.fsdp → dist.worker → models.partitioner

    Built-in names are protected (see ``registry._PROTECTED_BACKEND_NAMES``),
    so this is the only path that may register them: the registry's
    built-in-registration flag is raised for the duration of the loop.

    Fail-open design: a broken backend never prevents process startup. However,
    we now distinguish operator-visible signals:

      • Optional SDK missing (e.g., ``vllm`` not installed) stays at DEBUG.
      • Any other exception (SyntaxError, AttributeError, ImportError for our
        own modules) is logged at WARNING so operators notice immediately.
    """
    from .registry import _set_allow_builtin_registration

    import importlib
    _set_allow_builtin_registration(True)
    try:
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
                # Intentional override by another registration — silently accept.
                pass
            except Exception as exc:
                if _is_optional_sdk_error(exc, reg_name):
                    # Optional SDK not installed — expected, stay at debug.
                    logger.debug(f"Backend '{reg_name}' not available: {exc}")
                else:
                    # Real bug (syntax error, missing internal import, attr error, etc.).
                    logger.warning(
                        f"Backend '{reg_name}' failed to register: {type(exc).__name__}: {exc}. "
                        "The backend will be unavailable; please check the backend's implementation."
                    )
    finally:
        _set_allow_builtin_registration(False)

_register_builtins()
