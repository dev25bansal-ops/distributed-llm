"""GPU profiling and VRAM estimation for GPU-aware partitioning."""

import torch
from dataclasses import dataclass
from typing import List, Optional
from loguru import logger


@dataclass
class GPUInfo:
    """Information about a single GPU."""
    gpu_id: int
    name: str
    total_memory: int  # bytes
    used_memory: int   # bytes
    free_memory: int   # bytes
    utilization: float  # 0.0-1.0

    @property
    def free_memory_gb(self) -> float:
        return self.free_memory / (1024 ** 3)

    @property
    def total_memory_gb(self) -> float:
        return self.total_memory / (1024 ** 3)


class GPUProfiler:
    """Profiles GPU hardware and estimates VRAM requirements for model layers."""

    def enumerate_gpus(self) -> List[GPUInfo]:
        """List all available GPUs with their memory and utilization."""
        if not torch.cuda.is_available():
            logger.warning("No CUDA GPUs available")
            return []

        gpus = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            mem_reserved = torch.cuda.memory_reserved(i)
            mem_allocated = torch.cuda.memory_allocated(i)

            utilization = self._get_gpu_utilization(i)

            gpus.append(GPUInfo(
                gpu_id=i,
                name=props.name,
                total_memory=props.total_memory,
                used_memory=mem_allocated,
                free_memory=props.total_memory - mem_reserved,
                utilization=utilization,
            ))

        return gpus

    def _get_gpu_utilization(self, gpu_id: int) -> float:
        """Get GPU utilization (0.0-1.0)."""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            pynvml.nvmlShutdown()
            return util.gpu / 100.0
        except Exception:
            return 0.0

    def estimate_layer_vram(
        self,
        model_name: str,
        layer_idx: int,
        total_layers: int,
        trust_remote_code: bool = False,
    ) -> int:
        """Estimate VRAM (bytes) for a single model layer.

        Uses model config to estimate memory from hidden_size, intermediate_size,
        num_attention_heads, and dtype.
        """
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        dtype_bytes = self._get_dtype_size(getattr(config, "torch_dtype", "float16"))

        hidden_size = getattr(config, "hidden_size", 768)
        intermediate_size = getattr(config, "intermediate_size", hidden_size * 4)
        num_heads = getattr(config, "num_attention_heads", 12)

        # Transformer layer memory estimate:
        # - Attention: Q, K, V, O projections (4 * hidden * hidden)
        # - MLP: gate, up, down projections (3 * hidden * intermediate)
        # - Layer norms (2 * hidden)
        # Parameters only (weights), per layer
        params_per_layer = (
            4 * hidden_size * hidden_size +  # Q, K, V, O
            3 * hidden_size * intermediate_size +  # MLP
            4 * hidden_size  # biases + norms
        )

        return params_per_layer * dtype_bytes

    def estimate_total_vram(
        self,
        model_name: str,
        total_layers: int,
        trust_remote_code: bool = False,
        safety_margin: float = 0.1,
    ) -> int:
        """Estimate total VRAM for the full model including KV cache overhead."""
        # Embedding + final norm + lm_head (shared across layers)
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        dtype_bytes = self._get_dtype_size(getattr(config, "torch_dtype", "float16"))
        hidden_size = getattr(config, "hidden_size", 768)
        vocab_size = getattr(config, "vocab_size", 50257)

        # Shared components
        shared_params = (
            vocab_size * hidden_size +  # embeddings
            hidden_size * hidden_size +  # lm_head
            2 * hidden_size  # final norm
        ) * dtype_bytes

        # Per-layer components
        layer_vram = self.estimate_layer_vram(
            model_name, 0, total_layers, trust_remote_code
        )
        total_params_vram = shared_params + layer_vram * total_layers

        # KV cache overhead (rough estimate: 2 * num_heads * head_dim * seq_len * batch)
        # Assume max seq_len=512, batch=1 for estimation
        num_heads = getattr(config, "num_attention_heads", 12)
        head_dim = hidden_size // num_heads
        kv_cache_per_layer = 2 * num_heads * head_dim * 512 * dtype_bytes  # K + V
        kv_cache_vram = kv_cache_per_layer * total_layers

        total_vram = total_params_vram + kv_cache_vram
        # Apply safety margin
        total_vram = int(total_vram * (1 + safety_margin))

        return total_vram

    def _get_dtype_size(self, dtype) -> int:
        """Get byte size for a dtype string or torch.dtype."""
        if isinstance(dtype, str):
            dtype_map = {"float32": 4, "float16": 2, "bfloat16": 2, "int8": 1, "int4": 0.5}
            return int(dtype_map.get(dtype, 2))
        elif isinstance(dtype, torch.dtype):
            return torch.tensor([], dtype=dtype).element_size()
        return 2  # default to float16
