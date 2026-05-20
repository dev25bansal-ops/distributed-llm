"""torch.compile support for 1.5-2x speedup via graph optimization.

Provides safe compilation wrappers that avoid Dynamo-tracing issues:
- DynamoFP8Linear: uses torch.float8_e4m3fn instead of torch._scaled_mm
- compile_model(): wraps model forward with torch.compile(mode="reduce-overhead")
- CompileSafeLoRAManager: pre-merges adapters without in-place mutation during compiled forward
"""

from typing import Any, Callable

import torch
import torch.nn as nn
from loguru import logger


def compile_model(
    model: nn.Module | Callable,
    mode: str = "reduce-overhead",
    fullgraph: bool = False,
    dynamic: bool = True,
) -> Any:
    """Compile a model or forward function with torch.compile.

    Args:
        model: PyTorch module or callable to compile.
        mode: Compilation mode - "default", "reduce-overhead", "max-autotune".
        fullgraph: Whether to compile the entire graph (stricter, faster).
        dynamic: Enable dynamic shape support.

    Returns:
        Compiled model or callable.
    """
    if not hasattr(torch, 'compile'):
        logger.warning("torch.compile not available (requires PyTorch >= 2.0), using eager mode")
        return model

    try:
        compiled = torch.compile(
            model,
            mode=mode,
            fullgraph=fullgraph,
            dynamic=dynamic,
        )
        logger.info(f"torch.compile: compiled with mode='{mode}'")
        return compiled
    except Exception as e:
        logger.warning(f"torch.compile failed: {e}, falling back to eager mode")
        return model


class DynamoFP8Linear(nn.Module):
    """FP8 linear layer compatible with torch.compile (Dynamo).

    Uses torch.float8_e4m3fn tensor type instead of torch._scaled_mm,
    which Dynamo can trace through.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Store weight as FP8
        self.register_buffer(
            "weight_fp8",
            torch.empty(out_features, in_features, dtype=torch.float8_e4m3fn),
        )
        self.register_buffer("weight_scale", torch.ones(1))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float16))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "DynamoFP8Linear":
        """Create from an existing nn.Linear layer."""
        layer = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
        )
        # Quantize weight to FP8
        fp8_max = 448.0
        scale = linear.weight.data.abs().max().clamp(min=1e-12) / fp8_max
        fp8_weight = (linear.weight.data / scale).to(torch.float8_e4m3fn)
        layer.weight_fp8.copy_(fp8_weight)
        layer.weight_scale.copy_(scale)
        if linear.bias is not None:
            layer.bias.data.copy_(linear.bias.data)
        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantize FP8 weight to fp16 for matmul
        weight_fp16 = self.weight_fp8.to(x.dtype) * self.weight_scale
        return torch.nn.functional.linear(x, weight_fp16, self.bias)


class CompileSafeLoRAManager:
    """Compile-safe LoRA adapter management.

    Instead of in-place weight mutation during compiled forward (which
    breaks Dynamo), this pre-merges adapter deltas into base weights
    before compilation. The compiled graph sees static weights.

    Usage:
        manager = CompileSafeLoRAManager(base_model)
        manager.register_adapter("adapter_a", A, B)
        # Pre-merge before compilation
        manager.apply_to_model()
        # Now safe to compile
        compiled = torch.compile(model)
    """

    def __init__(self, base_model: nn.Module):
        self.base_model = base_model
        self._adapters: dict[str, dict] = {}  # name -> {path, A, B, scaling}
        self._original_weights: dict[str, torch.Tensor] = {}

    def register_adapter(
        self,
        name: str,
        module_path: str,
        A: torch.Tensor,
        B: torch.Tensor,
        scaling: float = 1.0,
    ) -> None:
        """Register a LoRA adapter.

        Args:
            name: Adapter identifier.
            module_path: Dotted path to the target module (e.g., "layers.0.self_attn.q_proj").
            A: LoRA A matrix [rank, in_features].
            B: LoRA B matrix [out_features, rank].
            scaling: LoRA alpha scaling factor.
        """
        self._adapters[name] = {
            "module_path": module_path,
            "A": A,
            "B": B,
            "scaling": scaling,
        }

    def apply_to_model(self, adapter_names: list[str] | None = None) -> None:
        """Pre-merge adapter deltas into base model weights.

        This is done BEFORE torch.compile so the compiled graph sees
        static (non-mutating) weights.

        Args:
            adapter_names: List of adapter names to apply. None = all.
        """
        names = adapter_names or list(self._adapters.keys())

        for name in names:
            adapter = self._adapters[name]
            path = adapter["module_path"]
            module = self._get_module(path)
            if module is None:
                logger.warning(f"CompileSafeLoRA: module not found: {path}")
                continue

            # Save original weight if not already saved
            weight_key = f"{path}.weight"
            if weight_key not in self._original_weights:
                self._original_weights[weight_key] = module.weight.data.clone()

            # Apply delta: W' = W + B @ A * scaling
            delta = adapter["B"] @ adapter["A"] * adapter["scaling"]
            module.weight.data.add_(delta)

            logger.debug(f"CompileSafeLoRA: applied adapter '{name}' to {path}")

    def restore_original(self) -> None:
        """Restore original weights (undo adapter merge)."""
        for key, weight in self._original_weights.items():
            parts = key.rsplit(".", 1)
            if len(parts) == 2:
                path, attr = parts
                module = self._get_module(path)
                if module is not None:
                    setattr(module, attr, nn.Parameter(weight))
                    logger.debug(f"CompileSafeLoRA: restored {path}.{attr}")

        self._original_weights.clear()

    def _get_module(self, path: str) -> nn.Module | None:
        """Get a module by dotted path."""
        module = self.base_model
        for part in path.split("."):
            if hasattr(module, part):
                module = getattr(module, part)
            else:
                return None
        return module
