"""Model loading and partitioning for distributed LLM inference.

Uses lazy __getattr__ imports to break circular import chains
(e.g., models.partitioner → dist.fsdp → dist.worker → models.partitioner).
"""

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {}

def _register(module: str, *symbols: str) -> None:
    for sym in symbols:
        _LAZY_IMPORTS[sym] = (module, sym)

_register("distllm.models.partitioner", "ModelPartitioner", "partition_model_across_nodes", "get_model_info")
_register("distllm.models.model_hub", "ModelHub", "ModelInfo", "CachedModel", "ModelHubError", "ModelNotCachedError", "DownloadError")
_register("distllm.models.cache", "ModelCache")
_register("distllm.models.safetensors_index", "SafetensorsIndex")

def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib
        module_path, symbol = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path)
        value = getattr(module, symbol)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'distllm.models' has no attribute {name!r}")

__all__ = [
    "ModelPartitioner",
    "partition_model_across_nodes",
    "get_model_info",
    "ModelHub",
    "ModelInfo",
    "CachedModel",
    "ModelCache",
    "ModelHubError",
    "ModelNotCachedError",
    "DownloadError",
    "SafetensorsIndex",
]
