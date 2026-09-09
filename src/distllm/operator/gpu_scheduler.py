"""GPU scheduling helpers for the DistLLM operator.

Pure functions that build Kubernetes scheduling primitives (node
selectors, affinity, tolerations) for GPU workloads and pick the best
node for a model by VRAM. No kopf/k8s dependency — the operator layer
feeds these dicts straight into pod specs.
"""

from __future__ import annotations

from typing import Any


def build_gpu_node_selector(gpu_count: int = 1, gpu_product: str | None = None) -> dict[str, str]:
    """Node selector requiring NVIDIA GPU nodes.

    ``nvidia.com/gpu.count`` ensures K8s device-plugin nodes; an optional
    ``gpu_product`` (e.g. ``"A100"``) pins the hardware generation.
    """
    selector: dict[str, str] = {"nvidia.com/gpu.count": str(max(1, gpu_count))}
    if gpu_product:
        selector["nvidia.com/gpu.product"] = gpu_product
    return selector


def build_gpu_affinity(gpu_product: str | None = None) -> dict[str, Any]:
    """Node affinity: prefer GPU nodes, required when a product is pinned."""
    preferred = [
        {
            "weight": 1,
            "preference": {
                "matchExpressions": [
                    {"key": "nvidia.com/gpu.product", "operator": "Exists"}
                ]
            },
        }
    ]
    if gpu_product:
        preferred[0]["preference"]["matchExpressions"] = [
            {
                "key": "nvidia.com/gpu.product",
                "operator": "In",
                "values": [gpu_product],
            }
        ]
    return {"nodeAffinity": {"preferredDuringSchedulingIgnoredDuringExecution": preferred}}


def build_pod_anti_affinity(label_key: str = "app.kubernetes.io/component") -> dict[str, Any]:
    """Spread replicas across nodes (no two same-component pods per node)."""
    return {
        "podAntiAffinity": {
            "requiredDuringSchedulingIgnoredDuringExecution": [
                {
                    "labelSelector": {
                        "matchExpressions": [
                            {"key": label_key, "operator": "Exists"}
                        ]
                    },
                    "topologyKey": "kubernetes.io/hostname",
                }
            ]
        }
    }


def build_tensor_parallel_affinity(tp_size: int, label_key: str = "distllm.io/tp-group") -> dict[str, Any]:
    """Co-locate tensor-parallel shards on one node (NVLink locality)."""
    return {
        "podAffinity": {
            "requiredDuringSchedulingIgnoredDuringExecution": [
                {
                    "labelSelector": {
                        "matchExpressions": [
                            {"key": label_key, "operator": "Exists"}
                        ]
                    },
                    "topologyKey": "kubernetes.io/hostname",
                }
            ]
        }
    }


def build_gpu_tolerations() -> list[dict[str, Any]]:
    """Standard toleration allowing pods onto GPU-tainted nodes."""
    return [{"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}]


def _parse_gpu_memory_mb(value: str) -> int:
    """Parse the device plugin's gpu.memory value (MiB suffix) to MB."""
    v = str(value).strip()
    if v.endswith("Mi"):
        return int(v[:-2])
    if v.endswith("Ki"):
        return int(v[:-2]) // 1024
    if v.endswith("Gi"):
        return int(v[:-2]) * 1024
    return int(float(v)) if v else 0


def select_optimal_node_vram(required_vram_gb: float, nodes: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Pick the smallest node whose free GPU memory fits the requirement.

    Best-fit keeps big models off big GPUs when a smaller one suffices.
    Returns ``None`` when no node fits.
    """
    required_mb = required_vram_gb * 1024
    best: dict[str, Any] | None = None
    best_mb = None
    for node in nodes:
        allocatable = node.get("allocatable", {})
        mem = _parse_gpu_memory_mb(allocatable.get("nvidia.com/gpu.memory", "0"))
        if mem < required_mb:
            continue
        if best_mb is None or mem < best_mb:
            best, best_mb = node, mem
    return best


def estimate_gpu_requirements(model_size_params_b: float, quantization_bits: int = 16) -> dict[str, Any]:
    """Estimate VRAM and GPU count for a model.

    Weights dominate: params * bytes-per-param, plus ~20% for KV cache
    and activations. GPU count assumes 80GB A100/H100-class devices.
    """
    bytes_per_param = max(1, quantization_bits) / 8
    weights_gb = model_size_params_b * bytes_per_param
    total_gb = weights_gb * 1.2
    gpu_count = max(1, int(total_gb // 80) + (1 if total_gb % 80 else 0))
    return {"vram_gb": total_gb, "gpu_count": gpu_count, "weights_gb": weights_gb}
