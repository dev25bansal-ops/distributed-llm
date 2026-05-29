"""Distributed LLM Inference System — lazy imports to avoid circular dependencies.

Heavy modules are imported on first attribute access via ``__getattr__``.
This breaks circular import chains and speeds up ``import distllm``.
"""

__version__ = "0.4.0"

# Map public names to their module paths for lazy loading.
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # core
    "Coordinator": ("distllm.core.coordinator", "Coordinator"),
    "KVCache": ("distllm.core.kv_cache", "KVCache"),
    "KVCacheManager": ("distllm.core.kv_cache", "KVCacheManager"),
    "BatchScheduler": ("distllm.core.batch_scheduler", "BatchScheduler"),
    "Sequence": ("distllm.core.batch_scheduler", "Sequence"),
    "SequenceStatus": ("distllm.core.batch_scheduler", "SequenceStatus"),
    "ScheduledBatch": ("distllm.core.batch_scheduler", "ScheduledBatch"),
    "JSONSchemaConstraint": ("distllm.core.structured_output", "JSONSchemaConstraint"),
    "SystemMonitor": ("distllm.core.monitor", "SystemMonitor"),
    "NodeRegistration": ("distllm.core.resource_manager", "NodeRegistration"),
    # models
    "ModelPartitioner": ("distllm.models.partitioner", "ModelPartitioner"),
    "partition_model_across_nodes": ("distllm.models.partitioner", "partition_model_across_nodes"),
    "get_model_info": ("distllm.models.partitioner", "get_model_info"),
    "AdapterManager": ("distllm.models.adapter", "AdapterManager"),
    # optional
    "ModelHotSwapManager": ("distllm.core.multi_model_serving", "ModelHotSwapManager"),
}

__all__ = ["__version__", *sorted(_LAZY_IMPORTS.keys())]


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        import importlib
        module = importlib.import_module(module_path)
        attr = getattr(module, attr_name)
        # Cache on the module so subsequent access is fast
        globals()[name] = attr
        return attr
    raise AttributeError(f"module 'distllm' has no attribute {name!r}")


def __dir__():
    return list(__all__)
