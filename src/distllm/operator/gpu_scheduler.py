"""GPU-aware K8s pod scheduling and node selection.

Provides helpers for:
- Node affinity rules for GPU-aware pod placement
- GPU topology-aware scheduling (NVLink/NVSwitch interconnects)
- NUMA affinity for optimal GPU memory bandwidth
- GPU bin-packing across multiple model deployments
"""

try:
    from kubernetes import client as k8s
    HAS_K8S = True
except ImportError:
    HAS_K8S = False

# Well-known GPU topology labels
GPU_LABELS = {
    "nvidia.com/gpu.count": "nvidia.com/gpu.count",
    "nvidia.com/gpu.memory": "nvidia.com/gpu.memory",
    "nvidia.com/gpu.product": "nvidia.com/gpu.product",
    "nvidia.com/gpu.family": "nvidia.com/gpu.family",
    "topology.kubernetes.io/zone": "topology.kubernetes.io/zone",
    "nvidia.com/nvlink.domain": "nvidia.com/nvlink.domain",
    "nvidia.com/nvswitch.domain": "nvidia.com/nvswitch.domain",
}


def build_gpu_node_selector(
    min_gpu_count: int = 1,
    min_gpu_memory_gb: int = 40,
    gpu_product: str | None = None,
    gpu_family: str = "hopper",
) -> dict[str, str]:
    """Build node selector for GPU-capable nodes.

    Args:
        min_gpu_count: Minimum number of GPUs on the node.
        min_gpu_memory_gb: Minimum GPU memory in GB.
        gpu_product: Specific GPU product (e.g., "H100", "A100").
        gpu_family: GPU family ("hopper", "ampere", "turing").

    Returns:
        Node selector dict for use in PodSpec.
    """
    selector = {}
    if min_gpu_count > 0:
        selector["nvidia.com/gpu.count"] = str(min_gpu_count)
    if gpu_product:
        selector["nvidia.com/gpu.product"] = gpu_product
    else:
        selector["nvidia.com/gpu.family"] = gpu_family
    return selector


def build_gpu_affinity(
    preferred: bool = True,
    gpu_count_min: int = 1,
    gpu_memory_gb_min: int = 40,
    gpu_product: str | None = None,
    gpu_family: str = "hopper",
    topology_zone: str | None = None,
    require_nvlink: bool = False,
) -> dict:
    """Build pod affinity/anti-affinity for GPU topology-aware scheduling.

    Ensures pods are placed on nodes with sufficient GPU resources and,
    when applicable, colocated on the same NVLink domain for fast GPU-to-GPU
    communication (critical for tensor parallelism).

    Args:
        preferred: If True, use preferred (soft) scheduling; otherwise required.
        gpu_count_min: Minimum GPUs required.
        gpu_memory_gb_min: Minimum GPU memory in GB.
        gpu_product: Specific GPU product string.
        gpu_family: GPU family label.
        topology_zone: Required topology zone for NVLink locality.
        require_nvlink: If True, require NVLink-connected GPUs.

    Returns:
        Affinity dict for V1PodSpec.
    """
    match_expressions = [
        {
            "key": "nvidia.com/gpu.count",
            "operator": "Gt",
            "values": [str(gpu_count_min - 1)],
        },
    ]
    if gpu_product:
        match_expressions.append({
            "key": "nvidia.com/gpu.product",
            "operator": "In",
            "values": [gpu_product],
        })
    else:
        match_expressions.append({
            "key": "nvidia.com/gpu.family",
            "operator": "In",
            "values": [gpu_family],
        })

    if require_nvlink:
        match_expressions.append({
            "key": "nvidia.com/nvlink.domain",
            "operator": "Exists",
        })

    if topology_zone:
        match_expressions.append({
            "key": "topology.kubernetes.io/zone",
            "operator": "In",
            "values": [topology_zone],
        })

    node_selector_term = {
        "matchExpressions": match_expressions,
    }

    if preferred:
        return {
            "nodeAffinity": {
                "preferredDuringSchedulingIgnoredDuringExecution": [
                    {
                        "weight": 100,
                        "preference": node_selector_term,
                    }
                ]
            }
        }
    else:
        return {
            "nodeAffinity": {
                "requiredDuringSchedulingIgnoredDuringExecution": {
                    "nodeSelectorTerms": [node_selector_term]
                }
            }
        }


def build_pod_anti_affinity(
    topology_key: str = "kubernetes.io/hostname",
    max_per_host: int = 1,
    component_label: str = "coordinator",
) -> dict:
    """Build pod anti-affinity to spread model pods across hosts.

    Ensures at most `max_per_host` pods with the same component label
    run on the same node (for high availability / fault tolerance).

    Args:
        topology_key: Node topology key (default: hostname).
        max_per_host: Maximum pods per host.
        component_label: Label value identifying the component.

    Returns:
        Anti-affinity dict for V1PodSpec.
    """
    return {
        "podAntiAffinity": {
            "preferredDuringSchedulingIgnoredDuringExecution": [
                {
                    "weight": 100,
                    "podAffinityTerm": {
                        "labelSelector": {
                            "matchExpressions": [
                                {
                                    "key": "component",
                                    "operator": "In",
                                    "values": [component_label],
                                }
                            ]
                        },
                        "topologyKey": topology_key,
                    },
                }
            ]
        }
    }


def build_tensor_parallel_affinity(
    tp_size: int,
    require_same_node: bool = True,
) -> dict:
    """Build pod affinity to co-locate tensor-parallel shards on the same node.

    Tensor parallelism requires fast GPU-to-GPU communication (NVLink).
    Shards must be on the same node (or same NVLink domain).

    Args:
        tp_size: Tensor parallelism degree.
        require_same_node: If True, require same node (NVLink locality).

    Returns:
        Affinity dict.
    """
    if not require_same_node:
        return {}

    return {
        "podAffinity": {
            "requiredDuringSchedulingIgnoredDuringExecution": [
                {
                    "labelSelector": {
                        "matchExpressions": [
                            {
                                "key": "component",
                                "operator": "In",
                                "values": ["worker"],
                            }
                        ]
                    },
                    "topologyKey": "kubernetes.io/hostname",
                }
            ]
        }
    }


def build_gpu_tolerations() -> list[dict]:
    """Build tolerations for GPU nodes."""
    return [
        {
            "key": "nvidia.com/gpu",
            "operator": "Exists",
            "effect": "NoSchedule",
        }
    ]


def build_priority_class(
    priority: int = 1000,
    name: str = "distllm-high-priority",
    description: str = "High-priority class for distributed LLM inference pods",
) -> object | None:
    """Build a PriorityClass for model pods.

    Higher priority ensures model pods are scheduled before lower-priority
    workloads and are less likely to be preempted.

    Args:
        priority: Priority value (higher = more important).
        name: PriorityClass name.
        description: Description.

    Returns:
        V1PriorityClass object or None if K8s client not available.
    """
    if not HAS_K8S:
        return None
    return k8s.V1PriorityClass(
        metadata=k8s.V1ObjectMeta(name=name),
        value=priority,
        description=description,
    )


def select_optimal_node_vram(
    required_vram_gb: float,
    nodes: list[dict[str, object]],
) -> str | None:
    """Select the optimal node for a model based on available GPU memory.

    Uses best-fit (tightest fit) to maximize packing density.

    Args:
        required_vram_gb: Required GPU memory in GB.
        nodes: List of node info dicts with "name", "free_vram_gb" keys.

    Returns:
        Node name with the best fit, or None if no node has enough VRAM.
    """
    candidates = [
        n for n in nodes
        if n.get("free_vram_gb", 0) >= required_vram_gb
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda n: n["free_vram_gb"])
    return candidates[0]["name"]


def estimate_gpu_requirements(
    model_size_params_b: float,
    quantization_bits: int = 16,
    overhead_factor: float = 1.2,
) -> dict[str, float]:
    """Estimate GPU resource requirements for a model.

    Args:
        model_size_params_b: Model size in billions of parameters.
        quantization_bits: Bit width (16 for fp16, 8 for fp8, 4 for int4).
        overhead_factor: Memory overhead factor (KV cache, activations).

    Returns:
        Dict with "vram_gb", "gpu_count" recommended.
    """
    params_bytes = model_size_params_b * 1e9 * (quantization_bits / 8)
    vram_gb = params_bytes * overhead_factor / 1e9
    gpu_count = max(1, int(vram_gb // 40) + 1)
    return {
        "vram_gb": round(vram_gb, 1),
        "gpu_count": gpu_count,
    }
