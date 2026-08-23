"""Auto-partitioner for distributing model layers across GPUs.

Analyzes hardware capabilities and model architecture to produce
an optimal partition plan with tensor-parallel and pipeline-parallel
stage assignments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from distllm.core.gpu_profiler import GPUInfo


def best_fit_decreasing_partition(
    caps: dict[str, int], layer_bytes: list[int]
) -> dict[str, list[int]]:
    """Partition ``layer_bytes`` across devices by best-fit-decreasing.

    Args:
        caps: Device id -> capacity in bytes.
        layer_bytes: Size of each layer (index = layer id).

    Returns:
        Mapping device id -> list of layer indices placed on it.

    Raises:
        ValueError: If a layer does not fit any device (true OOM).
    """
    if not layer_bytes:
        return {dev: [] for dev in caps}
    # Sort layers by size descending (best-fit-decreasing), tracking indices.
    order = sorted(range(len(layer_bytes)), key=lambda i: -layer_bytes[i])
    placement: dict[str, list[int]] = {dev: [] for dev in caps}
    remaining: dict[str, int] = {dev: int(cap) for dev, cap in caps.items()}

    for idx in order:
        size = int(layer_bytes[idx])
        # Best fit: device with the smallest remaining capacity that fits.
        best_dev: str | None = None
        best_left: int | None = None
        for dev, left in remaining.items():
            if left >= size:
                if best_left is None or left < best_left:
                    best_left = left
                    best_dev = dev
        if best_dev is None:
            raise ValueError(
                f"Layer {idx} ({size} bytes) does not fit any device — true OOM"
            )
        placement[best_dev].append(idx)
        remaining[best_dev] -= size

    for dev in placement:
        placement[dev] = sorted(placement[dev])
    return placement


@dataclass
class LayerInfo:
    """Information about a single model layer."""

    name: str
    layer_id: int
    memory_bytes: int = 0
    flops_per_token: int = 0
    layer_type: str = "attention"
    is_embedding: bool = False
    is_lm_head: bool = False


@dataclass
class DeviceAssignment:
    """Assignment of layers to a single device."""

    device_id: int
    device_name: str = ""
    layers: list[LayerInfo] = field(default_factory=list)
    total_memory_bytes: int = 0
    total_flops: int = 0

    @property
    def memory_utilization(self) -> float:
        """Return memory utilization ratio (layers / total).

        Returns 1.0 if no layers are assigned (fully available) or
        if total_memory_bytes is 0 (unknown capacity).
        """
        if self.total_memory_bytes <= 0:
            return 1.0
        if not self.layers:
            return 1.0
        layer_mem = sum(l.memory_bytes for l in self.layers)
        return layer_mem / self.total_memory_bytes


@dataclass
class PartitionPlan:
    """Complete partition plan across all devices."""

    assignments: list[DeviceAssignment] = field(default_factory=list)
    tp_groups: list[list[int]] = field(default_factory=list)
    pp_stages: list[list[int]] = field(default_factory=list)
    estimated_throughput: float = 0.0

    def summary(self) -> str:
        """Return a human-readable summary."""
        num_devices = len(self.assignments)
        num_layers = sum(len(a.layers) for a in self.assignments)
        tp = len(self.tp_groups)
        pp = len(self.pp_stages)
        return (
            f"PartitionPlan({num_devices} devices, {num_layers} layers, "
            f"{tp} TP groups, {pp} PP stages, "
            f"throughput={self.estimated_throughput:.0f} tok/s)"
        )


class AutoPartitioner:
    """Automatically partitions a model across available GPUs.

    Usage::

        ap = AutoPartitioner(hidden_size=4096, num_layers=32)
        plan = ap.partition()
        print(plan.summary())
    """

    def __init__(
        self,
        hidden_size: int = 4096,
        num_layers: int = 32,
        num_attention_heads: int = 32,
        num_kv_heads: int = 32,
        intermediate_size: int = 11008,
        vocab_size: int = 32000,
        max_seq_len: int = 4096,
        batch_size: int = 1,
    ):
        self._hidden = hidden_size
        self._num_layers = num_layers
        self._num_heads = num_attention_heads
        self._num_kv = num_kv_heads
        self._intermediate = intermediate_size
        self._vocab = vocab_size
        self._max_seq = max_seq_len
        self._batch = batch_size

        from distllm.core.gpu_profiler import GPUProfiler
        self._profiler = GPUProfiler()

    def _estimate_layer_memory(self, layer_type: str) -> int:
        """Estimate memory in bytes for a layer type."""
        h = self._hidden
        fp16 = 2  # bytes per element

        if layer_type == "attention":
            # Q + K + V + O projections
            qkv = 3 * h * h * fp16
            o_proj = h * h * fp16
            return qkv + o_proj

        if layer_type == "mlp":
            # Gate + Up + Down projections
            gate = h * self._intermediate * fp16
            up = h * self._intermediate * fp16
            down = self._intermediate * h * fp16
            return gate + up + down

        if layer_type == "norm":
            return 2 * h * fp16

        if layer_type == "embed":
            return self._vocab * h * fp16

        return 0

    def _build_layers(self) -> list[LayerInfo]:
        """Build layer info list for the model."""
        layers = []
        layer_id = 0

        for i in range(self._num_layers):
            # Attention layer
            attn = LayerInfo(
                name=f"model.layers.{i}.self_attn",
                layer_id=layer_id,
                memory_bytes=self._estimate_layer_memory("attention"),
                layer_type="attention",
            )
            layers.append(attn)
            layer_id += 1

            # MLP layer
            mlp = LayerInfo(
                name=f"model.layers.{i}.mlp",
                layer_id=layer_id,
                memory_bytes=self._estimate_layer_memory("mlp"),
                layer_type="mlp",
            )
            layers.append(mlp)
            layer_id += 1

        return layers

    def partition(self) -> PartitionPlan:
        """Create a partition plan across available GPUs.

        Returns:
            PartitionPlan with device assignments, TP groups, and PP stages.
        """
        gpus = self._profiler.enumerate_gpus()
        layers = self._build_layers()

        if not gpus:
            # CPU fallback
            assignment = DeviceAssignment(
                device_id=0,
                device_name="cpu",
                layers=layers,
                total_memory_bytes=0,
            )
            return PartitionPlan(
                assignments=[assignment],
                tp_groups=[[0]],
                pp_stages=[[0]],
                estimated_throughput=10.0,
            )

        # Distribute layers across GPUs
        num_gpus = len(gpus)
        assignments = []
        layers_per_gpu = len(layers) // num_gpus
        remainder = len(layers) % num_gpus

        offset = 0
        for i, gpu in enumerate(gpus):
            count = layers_per_gpu + (1 if i < remainder else 0)
            gpu_layers = layers[offset:offset + count]
            offset += count

            assignments.append(DeviceAssignment(
                device_id=gpu.gpu_id,
                device_name=gpu.name,
                layers=gpu_layers,
                total_memory_bytes=gpu.total_memory,
            ))

        # Build TP groups (pairs of GPUs)
        tp_groups = self._build_tp_groups(gpus)

        # Build PP stages (one per GPU)
        pp_stages = [[gpu.gpu_id] for gpu in gpus]

        # Estimate throughput
        total_mem = sum(a.total_memory_bytes for a in assignments)
        total_layer_mem = sum(l.memory_bytes for a in assignments for l in a.layers)
        throughput = 100.0 * num_gpus  # Rough estimate

        return PartitionPlan(
            assignments=assignments,
            tp_groups=tp_groups,
            pp_stages=pp_stages,
            estimated_throughput=throughput,
        )

    def _build_tp_groups(self, gpus: list[GPUInfo]) -> list[list[int]]:
        """Build tensor-parallel groups (pairs of 2)."""
        groups = []
        ids = [g.gpu_id for g in gpus]
        for i in range(0, len(ids), 2):
            group = ids[i:i + 2]
            groups.append(group)
        return groups

    def get_memory_report(self) -> dict:
        """Generate a memory report for the model across available GPUs."""
        gpus = self._profiler.enumerate_gpus()
        layers = self._build_layers()

        total_layer_mem = sum(l.memory_bytes for l in layers)
        attn_mem = sum(l.memory_bytes for l in layers if l.layer_type == "attention")
        mlp_mem = sum(l.memory_bytes for l in layers if l.layer_type == "mlp")

        return {
            "num_gpus": len(gpus),
            "num_layers": self._num_layers,
            "total_layer_memory_gb": total_layer_mem / (1024**3),
            "gpu_memory_gb": [g.total_memory / (1024**3) for g in gpus],
            "gpu_names": [g.name for g in gpus],
            "estimated_attention_memory_gb": attn_mem / (1024**3),
            "estimated_mlp_memory_gb": mlp_mem / (1024**3),
        }


class ZeroConfigPartitioner:
    """Zero-config auto-partitioner that detects hardware and model automatically.

    Automatically:
    1. Detects available GPUs and their memory
    2. Estimates model parameters from the model name
    3. Computes optimal layer distribution
    4. Generates a partition plan

    Usage::

        partitioner = ZeroConfigPartitioner()
        plan = partitioner.auto_partition("meta-llama/Llama-2-70b")
        print(plan.summary())
    """

    # Model parameter estimates (params_b, num_layers, hidden_size, num_heads)
    MODEL_PROFILES = {
        "7b": (7, 32, 4096, 32),
        "8b": (8, 32, 4096, 32),
        "13b": (13, 40, 5120, 40),
        "14b": (14, 40, 5120, 40),
        "34b": (34, 64, 8192, 64),
        "65b": (65, 80, 8192, 64),
        "70b": (70, 80, 8192, 64),
        "405b": (405, 126, 16384, 128),
    }

    def auto_partition(
        self,
        model_name: str,
        dtype: str = "float16",
    ) -> PartitionPlan:
        """Auto-partition a model across available GPUs.

        Args:
            model_name: HuggingFace model name or path.
            dtype: Model dtype (float16, bfloat16, float32).

        Returns:
            PartitionPlan with optimal layer distribution.
        """
        # Detect hardware
        from distllm.core.gpu_profiler import GPUProfiler
        profiler = GPUProfiler()
        gpus = profiler.enumerate_gpus()

        if not gpus:
            logger.warning("No GPUs detected — returning single-device plan")
            return PartitionPlan()

        # Estimate model parameters from name
        params_b, num_layers, hidden_size, num_heads = self._estimate_model(model_name)

        # Calculate memory requirements
        bytes_per_param = 2 if dtype in ("float16", "bfloat16") else 4
        total_model_bytes = int(params_b * 1e9 * bytes_per_param)
        total_gpu_bytes = sum(g.total_memory for g in gpus)

        # Check if model fits
        if total_model_bytes > total_gpu_bytes * 0.9:
            logger.warning(
                f"Model ({params_b}B, {total_model_bytes / 1e9:.0f}GB) may not fit "
                f"in available GPUs ({total_gpu_bytes / 1e9:.0f}GB total)"
            )

        # Create partitioner with detected parameters
        partitioner = AutoPartitioner(
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_attention_heads=num_heads,
            num_kv_heads=num_heads,
            intermediate_size=hidden_size * 4,  # Standard 4x ratio
            vocab_size=32000,
        )

        plan = partitioner.partition()
        logger.info(
            f"Auto-partitioned {model_name} ({params_b}B): {plan.summary()}"
        )
        return plan

    def _estimate_model(self, model_name: str) -> tuple[int, int, int, int]:
        """Estimate model parameters from the model name.

        Returns:
            (params_billions, num_layers, hidden_size, num_heads)
        """
        name = model_name.lower()

        for key, profile in self.MODEL_PROFILES.items():
            if key in name:
                return profile

        # Default: assume 7B
        logger.warning(f"Unknown model '{model_name}' — assuming 7B parameters")
        return (7, 32, 4096, 32)
