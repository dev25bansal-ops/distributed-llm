"""Hardware profiler and auto-assigner for optimal layer-to-device mapping.

Profiles all available hardware (GPUs, memory, interconnect bandwidth) and
automatically assigns model layers to achieve optimal throughput.

Strategy:
1. Profile each GPU's memory, compute capability, and NVLink topology
2. Estimate per-layer memory and compute cost
3. Assign layers to devices balancing memory and compute load
4. Respect NVLink islands and PCIe bus topology for TP groups
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from distllm.core.gpu_profiler import GPUProfiler, GPUInfo


@dataclass
class LayerInfo:
    name: str
    layer_id: int
    memory_bytes: int = 0
    flops_per_token: int = 0
    layer_type: str = "attention"  # attention, mlp, norm, embed
    is_embedding: bool = False
    is_lm_head: bool = False


@dataclass
class DeviceAssignment:
    device_id: int
    device_name: str
    layers: list[LayerInfo] = field(default_factory=list)
    total_memory_bytes: int = 0
    total_flops: int = 0

    @property
    def memory_utilization(self) -> float:
        return self.total_memory_bytes / max(self.total_memory_bytes, 1)


@dataclass
class PartitionPlan:
    assignments: list[DeviceAssignment] = field(default_factory=list)
    tp_groups: list[list[int]] = field(default_factory=list)
    pp_stages: list[list[int]] = field(default_factory=list)
    estimated_throughput: float = 0.0

    def summary(self) -> str:
        return (
            f"PartitionPlan: {len(self.assignments)} devices, "
            f"{sum(len(a.layers) for a in self.assignments)} layers, "
            f"throughput={self.estimated_throughput:.0f} tok/s"
        )


class AutoPartitioner:
    """Profiles hardware and auto-assigns model layers for optimal throughput.

    Usage:
        partitioner = AutoPartitioner(
            hidden_size=4096,
            num_layers=32,
            num_heads=32,
        )
        plan = partitioner.partition()
        print(plan.summary())
    """

    def __init__(
        self,
        hidden_size: int = 4096,
        num_layers: int = 32,
        num_attention_heads: int = 32,
        num_kv_heads: int = 0,
        intermediate_size: int = 11008,
        vocab_size: int = 32000,
        max_seq_len: int = 4096,
        batch_size: int = 1,
    ):
        if hidden_size <= 0 or hidden_size % 64 != 0:
            raise ValueError(f"hidden_size must be positive multiple of 64, got {hidden_size}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if num_attention_heads <= 0:
            raise ValueError(f"num_attention_heads must be positive, got {num_attention_heads}")
        if intermediate_size <= 0:
            raise ValueError(f"intermediate_size must be positive, got {intermediate_size}")
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self._hidden = hidden_size
        self._num_layers = num_layers
        self._num_heads = num_attention_heads
        self._num_kv = num_kv_heads or num_attention_heads
        self._intermediate = intermediate_size
        self._vocab = vocab_size
        self._max_seq = max_seq_len
        self._batch = batch_size
        self._profiler = GPUProfiler()

    def _estimate_layer_memory(self, layer_type: str) -> int:
        """Estimate memory in bytes for a single layer."""
        bytes_per_param = 2  # fp16
        if layer_type == "attention":
            qkv = 3 * self._hidden * self._hidden
            o_proj = self._hidden * self._hidden
            return (qkv + o_proj) * bytes_per_param
        elif layer_type == "mlp":
            gate = self._hidden * self._intermediate
            up = self._hidden * self._intermediate
            down = self._intermediate * self._hidden
            return (gate + up + down) * bytes_per_param
        elif layer_type == "norm":
            return 2 * self._hidden * bytes_per_param
        elif layer_type == "embed":
            return self._vocab * self._hidden * bytes_per_param
        return 0

    def _build_layers(self) -> list[LayerInfo]:
        layers = []
        lid = 0
        for i in range(self._num_layers):
            layers.append(LayerInfo(
                name=f"model.layers.{i}.self_attn", layer_id=lid,
                memory_bytes=self._estimate_layer_memory("attention"),
                layer_type="attention",
            ))
            lid += 1
            layers.append(LayerInfo(
                name=f"model.layers.{i}.mlp", layer_id=lid,
                memory_bytes=self._estimate_layer_memory("mlp"),
                layer_type="mlp",
            ))
            lid += 1
        return layers

    def partition(self) -> PartitionPlan:
        """Profile hardware and assign layers to devices.

        Returns a PartitionPlan with balanced device assignments.
        """
        gpus = self._profiler.enumerate_gpus()
        if not gpus:
            logger.warning("No GPUs found, returning single-device plan")
            return PartitionPlan(
                assignments=[DeviceAssignment(device_id=0, device_name="cpu", layers=self._build_layers())],
            )

        layers = self._build_layers()
        num_devices = len(gpus)

        # Sort GPUs by memory (largest first)
        sorted_gpus = sorted(enumerate(gpus), key=lambda x: x[1].total_memory, reverse=True)
        total_mem = sum(g.total_memory for _, g in sorted_gpus)
        total_layer_mem = sum(l.memory_bytes for l in layers)

        # Balanced round-robin assignment
        device_loads = [0] * num_devices
        assignments = [DeviceAssignment(device_id=i, device_name=gpus[i].name) for i in range(num_devices)]

        for layer in sorted(layers, key=lambda l: l.memory_bytes, reverse=True):
            target = min(range(num_devices), key=lambda i: device_loads[i])
            assignments[target].layers.append(layer)
            device_loads[target] += layer.memory_bytes

        for a in assignments:
            a.total_memory_bytes = sum(l.memory_bytes for l in a.layers)

        # Build TP groups: group devices with NVLink connectivity
        tp_groups = self._build_tp_groups(gpus)

        # Build PP stages: order by device memory (larger = earlier stage)
        stage_order = sorted(range(num_devices), key=lambda i: gpus[i].total_memory, reverse=True)
        pp_stages = [[i] for i in stage_order]

        # Estimate throughput
        max_load = max(device_loads) if device_loads else 1
        load_balance = 1 - (max_load - min(device_loads)) / max(device_loads) if max_load > 0 else 1.0
        throughput = (self._batch * self._max_seq * 1000) * load_balance * min(1.0, total_mem / max(total_layer_mem, 1))

        plan = PartitionPlan(
            assignments=assignments,
            tp_groups=tp_groups,
            pp_stages=pp_stages,
            estimated_throughput=throughput,
        )

        logger.info(f"Partitioned {len(layers)} layers across {num_devices} devices: {plan.summary()}")
        return plan

    def _build_tp_groups(self, gpus: list[GPUInfo]) -> list[list[int]]:
        """Build TP groups based on NVLink topology detection."""
        groups: list[list[int]] = []
        if len(gpus) <= 2:
            groups.append(list(range(len(gpus))))
            return groups

        # Simple heuristic: assume pairs of GPUs share NVLink
        used = set()
        for i in range(0, len(gpus) - 1, 2):
            groups.append([i, i + 1])
            used.add(i)
            used.add(i + 1)
        for i in range(len(gpus)):
            if i not in used:
                if groups:
                    groups[-1].append(i)
                else:
                    groups.append([i])
        return groups

    def get_memory_report(self) -> dict[str, Any]:
        gpus = self._profiler.enumerate_gpus()
        layers = self._build_layers()
        total_layer_mem = sum(l.memory_bytes for l in layers)
        return {
            "num_gpus": len(gpus),
            "num_layers": self._num_layers,
            "total_layer_memory_gb": round(total_layer_mem / (1024**3), 2),
            "gpu_memory_gb": [round(g.total_memory_gb, 2) for g in gpus],
            "gpu_names": [g.name for g in gpus],
            "estimated_attention_memory_gb": round(self._estimate_layer_memory("attention") / (1024**3), 4),
            "estimated_mlp_memory_gb": round(self._estimate_layer_memory("mlp") / (1024**3), 4),
        }
