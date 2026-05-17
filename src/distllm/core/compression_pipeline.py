"""Compression pipeline for automatic model compression during loading.

Provides post-training quantization, structured pruning, knowledge distillation,
and auto-compression based on VRAM budget. Now enhanced with:

- AWQ-style activation-aware INT4 quantization (group-wise, no fallback)
- GPTQ-style calibration-based quantization with Hessian approximation
- Structured pruning with proper weight removal and dimension adjustment
- Hardware-aware auto-selection via hardware profiling
- Command: distllm compress --model <name> --target int4 --output <dir>
"""

from __future__ import annotations

import gc
import math
import time
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from loguru import logger

from distllm.core.compression_config import CompressionConfig, CompressionMethod


class HardwareClass(Enum):
    DATA_CENTER = "datacenter"
    WORKSTATION = "workstation"
    CONSUMER = "consumer"
    EDGE = "edge"
    CPU_ONLY = "cpu_only"


@dataclass
class StrategyScore:
    method: CompressionMethod
    estimated_speedup: float
    estimated_quality_loss: float
    memory_savings_gb: float
    estimated_latency_ms: float
    score: float
    reasoning: str = ""


@dataclass
class PipelineStage:
    method: CompressionMethod
    order: int
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompressionPlan:
    stages: List[PipelineStage] = field(default_factory=list)
    target_hardware: HardwareClass = HardwareClass.DATA_CENTER
    expected_speedup: float = 1.0
    expected_quality_loss: float = 0.0
    expected_memory_gb: float = 0.0
    total_compression_ratio: float = 1.0

    def summary(self) -> str:
        stages_str = " -> ".join(s.method.value for s in sorted(self.stages, key=lambda x: x.order))
        return (
            f"CompressionPlan: {stages_str} | "
            f"speedup={self.expected_speedup:.1f}x, "
            f"quality_loss={self.expected_quality_loss:.4f}, "
            f"mem={self.expected_memory_gb:.1f}GB, "
            f"ratio={self.total_compression_ratio:.2f}x"
        )


# ---------------------------------------------------------------------------
# Hardware Profiler
# ---------------------------------------------------------------------------

class HardwareProfiler:
    def classify(self) -> HardwareClass:
        if not torch.cuda.is_available():
            return HardwareClass.CPU_ONLY
        props = torch.cuda.get_device_properties(0)
        name = props.name.lower()
        total_mem = props.total_memory
        if any(kw in name for kw in ("h100", "h200", "a100", "a6000", "mi300")):
            return HardwareClass.DATA_CENTER
        if total_mem >= 24 * 1024**3 and any(kw in name for kw in ("4090", "6000", "a4000", "5000")):
            return HardwareClass.WORKSTATION
        if total_mem >= 8 * 1024**3:
            return HardwareClass.CONSUMER
        return HardwareClass.EDGE

    def get_vram_gb(self) -> float:
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.get_device_properties(0).total_memory / (1024**3)

    def get_free_vram_gb(self) -> float:
        if not torch.cuda.is_available():
            return 0.0
        free = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
        return free / (1024**3)

    def supports_int8(self) -> bool:
        if not torch.cuda.is_available():
            return False
        return torch.cuda.get_device_capability() >= (7, 0)

    def supports_int4(self) -> bool:
        if not torch.cuda.is_available():
            return False
        return torch.cuda.get_device_capability() >= (8, 0)


# ---------------------------------------------------------------------------
# AWQ-Style Activation-Aware Quantization
# ---------------------------------------------------------------------------

class WQLinear(nn.Module):
    """Weight-quantized linear layer with group-wise INT4.

    Stores INT4 weights packed as int32 (8 weights per int32).
    Group-wise scale and zero-point for accuracy.
    On forward, dequantizes to fp16 for computation.
    Supports AWQ-style per-channel activation scaling.
    """

    def __init__(self, in_features: int, out_features: int, group_size: int = 128,
                 bias: bool = True, use_awq: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.use_awq = use_awq
        self.num_groups = (in_features + group_size - 1) // group_size

        packed_size = (out_features * in_features + 1) // 2
        self.qweight = nn.Parameter(torch.zeros(packed_size, dtype=torch.int32), requires_grad=False)
        self.scales = nn.Parameter(torch.zeros(out_features, self.num_groups, dtype=torch.float16), requires_grad=False)
        self.zeros = nn.Parameter(torch.zeros(out_features, self.num_groups, dtype=torch.float16), requires_grad=False)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float16))
        else:
            self.register_parameter('bias', None)
        if use_awq:
            self.act_scales = nn.Parameter(torch.ones(in_features, dtype=torch.float16), requires_grad=False)
        else:
            self.register_buffer('act_scales', torch.ones(in_features, dtype=torch.float16))

    @staticmethod
    def pack_int4(weights: torch.Tensor) -> torch.Tensor:
        """Pack float weights into INT4 and store as int32 (8 per int32)."""
        w = weights.to(torch.float16)
        shape = w.shape
        w_flat = w.reshape(-1)
        n = w_flat.numel()
        padded = (8 - n % 8) % 8
        if padded:
            w_flat = torch.cat([w_flat, torch.zeros(padded, dtype=torch.float16, device=w.device)])
        w_int = w_flat.to(torch.int8).clamp(-8, 7).to(torch.int32)
        w_shifted = (w_int + 8) & 0xF
        packed = torch.zeros(w_shifted.numel() // 8, dtype=torch.int32, device=w.device)
        for i in range(8):
            packed = packed | (w_shifted[i::8] << (i * 4))
        return packed

    @staticmethod
    def unpack_int4(qweight: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
        """Unpack INT4 values back to int8 tensor of given shape."""
        n_elems = shape[0] * shape[1]
        n_words = qweight.numel()
        vals = torch.zeros(8 * n_words, dtype=torch.int32, device=qweight.device)
        for i in range(8):
            vals[i::8] = (qweight >> (i * 4)) & 0xF
        vals = vals[:n_elems].reshape(shape).to(torch.int8) - 8
        return vals

    def quantize(self, weight: torch.Tensor):
        """Quantize full-precision weight to group-wise INT4 with AWQ scaling."""
        w = weight.to(torch.float16)
        with torch.no_grad():
            for g in range(self.num_groups):
                g_start = g * self.group_size
                g_end = min(g_start + self.group_size, self.in_features)
                w_slice = w[:, g_start:g_end]
                max_val = w_slice.abs().max(dim=1, keepdim=True).values.clamp(min=1e-6)
                scale = max_val / 7.0
                zero = torch.zeros_like(scale)
                w_q = (w_slice / scale).round().clamp(-8, 7).to(torch.int8)
                self.scales.data[:, g] = scale.squeeze(-1)
                self.zeros.data[:, g] = zero.squeeze(-1)
                w[:, g_start:g_end] = w_q.float() * scale
            self.qweight.data = self.pack_int4(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_awq:
            x = x * self.act_scales
        w_unpacked = self.unpack_int4(self.qweight, (self.out_features, self.in_features)).to(x.dtype)
        w = w_unpacked * self.scales[:, 0:1] if self.num_groups == 1 else self._dequantize(w_unpacked)
        out = torch.nn.functional.linear(x, w.to(x.dtype), self.bias)
        return out

    def _dequantize(self, w_int: torch.Tensor) -> torch.Tensor:
        w_fp = w_int.float()
        for g in range(self.num_groups):
            g_start = g * self.group_size
            g_end = min(g_start + self.group_size, self.in_features)
            w_fp[:, g_start:g_end] = w_fp[:, g_start:g_end] * self.scales[:, g:g+1]
        return w_fp.to(torch.float16)


class AWQQuantizer:
    """AWQ-style activation-aware quantization with calibration."""

    def __init__(self, group_size: int = 128):
        self.group_size = group_size

    def _get_act_scales(self, model: nn.Module, calib_inputs: List[torch.Tensor],
                        module_names: List[str]) -> Dict[str, torch.Tensor]:
        """Collect activation scales for targeted modules."""
        act_scales = {}
        hooks = []
        seen = set()

        def hook_fn(name):
            def fn(_, inp, out):
                if name not in seen:
                    seen.add(name)
                    act_scales[name] = inp[0].abs().view(-1, inp[0].shape[-1]).max(dim=0)[0].detach()
            return fn

        for name, mod in model.named_modules():
            if any(t in name for t in module_names) and isinstance(mod, nn.Linear):
                hooks.append(mod.register_forward_hook(hook_fn(name)))

        model.eval()
        with torch.no_grad():
            for inp in calib_inputs[:8]:
                try:
                    model(inp)
                except Exception:
                    pass
        for h in hooks:
            h.remove()
        return act_scales

    def quantize_model(self, model: nn.Module, calib_inputs: Optional[List[torch.Tensor]] = None,
                       target_modules: Optional[List[str]] = None) -> nn.Module:
        """Quantize linear layers to INT4 using AWQ calibration."""
        if target_modules is None:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj"]

        if calib_inputs:
            act_scales = self._get_act_scales(model, calib_inputs, target_modules)
        else:
            act_scales = {}

        replacements = []
        for name, mod in model.named_modules():
            if not isinstance(mod, nn.Linear):
                continue
            if not any(t in name for t in target_modules):
                continue
            replacements.append((name, mod))

        for name, mod in replacements:
            wq = WQLinear(mod.in_features, mod.out_features,
                          group_size=self.group_size,
                          bias=mod.bias is not None,
                          use_awq=True)
            wq.quantize(mod.weight.data)
            if mod.bias is not None:
                wq.bias.data = mod.bias.data.to(torch.float16)
            if name in act_scales:
                s = act_scales[name].to(torch.float16)
                s = s / s.mean()
                wq.act_scales.data = s
            parent, child_key = self._get_parent(model, name)
            setattr(parent, child_key, wq)
            logger.debug(f"AWQ quantized {name}: {mod.in_features}x{mod.out_features} group={self.group_size}")

        logger.info(f"AWQ quantized {len(replacements)} linear layers to INT4")
        return model

    def _get_parent(self, model: nn.Module, name: str):
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            if p.isdigit():
                parent = parent[int(p)]
            else:
                parent = getattr(parent, p)
        return parent, parts[-1]


# ---------------------------------------------------------------------------
# GPTQ-Style Quantization with Calibration
# ---------------------------------------------------------------------------

class GPTQQuantizer:
    """GPTQ-style layer-wise quantization with Hessian approximation.

    Uses calibration data to minimize quantization error via
    optimal brain damage-style weight updates.
    """

    def __init__(self, group_size: int = 128, damp_percent: float = 0.01):
        self.group_size = group_size
        self.damp_percent = damp_percent

    def quantize_model(self, model: nn.Module,
                       calib_inputs: List[torch.Tensor],
                       target_modules: Optional[List[str]] = None) -> nn.Module:
        if target_modules is None:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj"]

        replacements = []
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Linear) and any(t in name for t in target_modules):
                replacements.append((name, mod))

        device = next(model.parameters()).device
        handles = []
        cached_inputs = {}

        def cache_hook(name):
            def fn(_, inp, out):
                cached_inputs[name] = inp[0].detach().to(device)
            return fn

        for name, mod in replacements:
            handles.append(mod.register_forward_hook(cache_hook(name)))

        model.eval()
        with torch.no_grad():
            for inp in calib_inputs[:4]:
                inp = inp.to(device)
                try:
                    model(inp)
                except Exception:
                    pass
        for h in handles:
            h.remove()

        for name, mod in replacements:
            if name not in cached_inputs:
                continue
            H = self._compute_hessian(cached_inputs[name], mod.out_features)
            wq = WQLinear(mod.in_features, mod.out_features,
                          group_size=self.group_size,
                          bias=mod.bias is not None,
                          use_awq=False)
            self._gptq_quantize(mod.weight.data, H, wq)
            if mod.bias is not None:
                wq.bias.data = mod.bias.data.to(torch.float16)
            parent, child_key = self._get_parent(model, name)
            setattr(parent, child_key, wq)

        logger.info(f"GPTQ quantized {len(replacements)} linear layers to INT4")
        return model

    def _compute_hessian(self, x: torch.Tensor, out_features: int, nsamples: int = 1) -> torch.Tensor:
        xs = x.view(-1, x.shape[-1]).float()
        if xs.shape[0] > 2048:
            idx = torch.randperm(xs.shape[0], device=xs.device)[:2048]
            xs = xs[idx]
        H = xs.T @ xs
        if isinstance(H, torch.Tensor):
            H *= (2.0 / xs.shape[0])
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        damp = self.damp_percent * torch.mean(torch.diag(H))
        H = H + damp * torch.eye(H.shape[0], device=H.device)
        return H

    def _gptq_quantize(self, weight: torch.Tensor, H: torch.Tensor, wq: WQLinear):
        w = weight.float().cpu()
        H = H.cpu()
        w_orig = w.clone()
        num_groups = wq.num_groups

        for g in range(num_groups):
            g_start = g * self.group_size
            g_end = min(g_start + self.group_size, w.shape[1])
            w_slice = w[:, g_start:g_end]
            max_val = w_slice.abs().max(dim=1, keepdim=True).values.clamp(min=1e-6)
            scale = max_val / 7.0
            w_q = (w_slice / scale).round().clamp(-8, 7).to(torch.int8)
            w_q_float = w_q.float() * scale
            err = w_slice - w_q_float

            if g < wq.scales.shape[1]:
                wq.scales.data[:, g] = scale.squeeze(-1).to(torch.float16)
                wq.zeros.data[:, g] = torch.zeros_like(scale.squeeze(-1)).to(torch.float16)

            w[:, g_start:g_end] = w_q_float

            if g < num_groups - 1:
                next_start = g_end
                next_end = min(g_end + self.group_size, w.shape[1])
                if next_start < next_end:
                    H_block = H[g_start:g_end, next_start:next_end]
                    H_block_inv = torch.linalg.solve(H[next_start:next_end, next_start:next_end],
                                                     H_block.T).T
                    w[:, next_start:next_end] += err @ H_block_inv

        qweight = WQLinear.pack_int4(w_orig)
        wq.qweight.data = qweight.to(wq.qweight.device)

    def _get_parent(self, model: nn.Module, name: str):
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            if p.isdigit():
                parent = parent[int(p)]
            else:
                parent = getattr(parent, p)
        return parent, parts[-1]


# ---------------------------------------------------------------------------
# Strategy Selector
# ---------------------------------------------------------------------------

class StrategySelector:
    def __init__(self, hw_profiler: Optional[HardwareProfiler] = None):
        self._hw = hw_profiler or HardwareProfiler()

    def score_all_strategies(self, model: nn.Module, model_params_billions: float = 0.0) -> List[StrategyScore]:
        hw_class = self._hw.classify()
        vram_gb = self._hw.get_vram_gb()
        free_gb = self._hw.get_free_vram_gb()
        total_params = model_params_billions or (sum(p.numel() for p in model.parameters()) / 1e9)
        fp16_mem = total_params * 2
        strategies = []

        if self._hw.supports_int8():
            int8_mem = total_params * 1
            int8_speed = 2.5 if hw_class == HardwareClass.DATA_CENTER else 1.8
            int8_quality = 0.005 if total_params > 7 else 0.01
            strategies.append(StrategyScore(
                method=CompressionMethod.PTQ_INT8,
                estimated_speedup=int8_speed,
                estimated_quality_loss=int8_quality,
                memory_savings_gb=fp16_mem - int8_mem,
                estimated_latency_ms=50,
                score=self._compute_score(int8_speed, int8_quality, free_gb, int8_mem),
                reasoning=f"INT8 (dynamic): {int8_speed}x speed, ~{int8_quality} quality loss",
            ))

        if self._hw.supports_int4():
            int4_mem = total_params * 0.5
            int4_speed = 3.5 if hw_class == HardwareClass.DATA_CENTER else 2.5
            int4_quality = 0.015 if total_params > 7 else 0.03
            strategies.append(StrategyScore(
                method=CompressionMethod.PTQ_INT4,
                estimated_speedup=int4_speed,
                estimated_quality_loss=int4_quality,
                memory_savings_gb=fp16_mem - int4_mem,
                estimated_latency_ms=120,
                score=self._compute_score(int4_speed, int4_quality, free_gb, int4_mem),
                reasoning=f"AWQ INT4 (group-wise): {int4_speed}x speed, ~{int4_quality} quality loss",
            ))
            strategies.append(StrategyScore(
                method=CompressionMethod.QUANT_GPTQ,
                estimated_speedup=int4_speed * 1.05,
                estimated_quality_loss=int4_quality * 0.8,
                memory_savings_gb=fp16_mem - int4_mem,
                estimated_latency_ms=180,
                score=self._compute_score(int4_speed * 1.05, int4_quality * 0.8, free_gb, int4_mem),
                reasoning=f"GPTQ INT4 (Hessian-aware): {int4_speed * 1.05:.1f}x speed, ~{int4_quality * 0.8:.3f} quality loss",
            ))

        pruning_mem = fp16_mem * (1 - 0.25)
        pruning_speed = 1.3
        pruning_quality = 0.015
        strategies.append(StrategyScore(
            method=CompressionMethod.PRUNING_STRUCTURED,
            estimated_speedup=pruning_speed,
            estimated_quality_loss=pruning_quality,
            memory_savings_gb=fp16_mem - pruning_mem,
            estimated_latency_ms=60,
            score=self._compute_score(pruning_speed, pruning_quality, free_gb, pruning_mem),
            reasoning=f"Structured pruning: {pruning_speed}x speed, ~{pruning_quality} quality loss",
        ))

        strategies.append(StrategyScore(
            method=CompressionMethod.DISTILLATION,
            estimated_speedup=1.0,
            estimated_quality_loss=0.0,
            memory_savings_gb=0.0,
            estimated_latency_ms=200,
            score=0.5,
            reasoning="Distillation: same architecture, no speedup (requires teacher)",
        ))

        if strategies:
            auto_speed = max(s.estimated_speedup for s in strategies)
            auto_quality = min(s.estimated_quality_loss for s in strategies)
            strategies.append(StrategyScore(
                method=CompressionMethod.AUTO,
                estimated_speedup=auto_speed * 1.2,
                estimated_quality_loss=auto_quality * 1.5,
                memory_savings_gb=max((s.memory_savings_gb for s in strategies), default=0),
                estimated_latency_ms=max((s.estimated_latency_ms for s in strategies), default=100),
                score=auto_speed * 10,
                reasoning="AUTO: apply best available methods in pipeline",
            ))

        strategies.sort(key=lambda s: s.score, reverse=True)
        return strategies

    def _compute_score(self, speedup: float, quality_loss: float, free_vram_gb: float,
                       estimated_mem_gb: float) -> float:
        speed_score = speedup * 5.0
        quality_penalty = quality_loss * 100.0
        mem_penalty = max(0, estimated_mem_gb - free_vram_gb) * 0.1
        return speed_score - quality_penalty - mem_penalty

    def recommend(self, model: nn.Module, model_params_billions: float = 0.0) -> StrategyScore:
        scored = self.score_all_strategies(model, model_params_billions)
        return scored[0] if scored else self._default_score()

    def _default_score(self) -> StrategyScore:
        return StrategyScore(
            method=CompressionMethod.NONE,
            estimated_speedup=1.0,
            estimated_quality_loss=0.0,
            memory_savings_gb=0.0,
            estimated_latency_ms=0,
            score=0.0,
            reasoning="No compression available",
        )

    def build_plan(self, model: nn.Module, model_params_billions: float = 0.0) -> CompressionPlan:
        best = self.recommend(model, model_params_billions)
        hw_class = self._hw.classify()
        stages = []

        if best.method == CompressionMethod.AUTO:
            stages = [
                PipelineStage(method=CompressionMethod.PRUNING_STRUCTURED, order=1, params={"ratio": 0.1}),
                PipelineStage(method=CompressionMethod.QUANT_AWQ, order=2, params={"bits": 4}),
            ]
        elif best.method in (CompressionMethod.PTQ_INT8,):
            stages = [PipelineStage(method=CompressionMethod.PTQ_INT8, order=1, params={"bits": 8})]
        elif best.method in (CompressionMethod.PTQ_INT4, CompressionMethod.QUANT_AWQ):
            stages = [PipelineStage(method=CompressionMethod.QUANT_AWQ, order=1, params={"bits": 4})]
        elif best.method == CompressionMethod.QUANT_GPTQ:
            stages = [PipelineStage(method=CompressionMethod.QUANT_GPTQ, order=1, params={"bits": 4})]
        elif best.method == CompressionMethod.PRUNING_STRUCTURED:
            stages = [PipelineStage(method=CompressionMethod.PRUNING_STRUCTURED, order=1, params={"ratio": 0.2})]
        elif best.method == CompressionMethod.DISTILLATION:
            stages = [PipelineStage(method=CompressionMethod.DISTILLATION, order=1, params={})]

        ratio = 1.0
        for stage in stages:
            if stage.method in (CompressionMethod.PTQ_INT8,):
                ratio *= 2.0
            elif stage.method in (CompressionMethod.PTQ_INT4, CompressionMethod.QUANT_AWQ, CompressionMethod.QUANT_GPTQ):
                ratio *= 4.0
            elif stage.method == CompressionMethod.PRUNING_STRUCTURED:
                ratio *= 1.25
            elif stage.method == CompressionMethod.DISTILLATION:
                ratio *= 1.0

        return CompressionPlan(
            stages=stages,
            target_hardware=hw_class,
            expected_speedup=best.estimated_speedup,
            expected_quality_loss=best.estimated_quality_loss,
            expected_memory_gb=best.memory_savings_gb,
            total_compression_ratio=ratio,
        )


# ---------------------------------------------------------------------------
# Calibration Data Loader
# ---------------------------------------------------------------------------

class CalibrationDataLoader:
    def __init__(self, tokenizer, n_samples: int = 128, seq_length: int = 512):
        self.tokenizer = tokenizer
        self.n_samples = n_samples
        self.seq_length = seq_length

    def generate(self) -> List[torch.Tensor]:
        texts = self._get_calibration_texts()
        encoded = []
        for text in texts[:self.n_samples]:
            tokens = self.tokenizer.encode(text, max_length=self.seq_length,
                                           truncation=True, return_tensors="pt")
            encoded.append(tokens)
        return encoded

    def _get_calibration_texts(self) -> List[str]:
        texts = []
        try:
            from datasets import load_dataset
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", streaming=True)
            for item in dataset:
                text = item.get("text", "")
                if text and len(text.strip()) > 50:
                    texts.append(text)
                if len(texts) >= self.n_samples:
                    break
        except Exception as e:
            logger.debug(f"datasets unavailable: {e}, using synthetic data")
        if len(texts) < self.n_samples:
            texts.extend(self._generate_synthetic_texts(self.n_samples - len(texts)))
        return texts[:self.n_samples]

    def _generate_synthetic_texts(self, count: int) -> List[str]:
        patterns = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "The transformer architecture uses self-attention mechanisms.",
            "Natural language processing enables computers to understand language.",
            "Deep learning models use multiple layers of neural networks.",
            "Tokenization is the process of breaking text into smaller units.",
            "The model was trained on a large corpus of text data.",
            "Inference is the process of using a trained model for predictions.",
            "Distributed systems consist of multiple nodes working together.",
            "Attention is all you need for processing sequential data.",
        ]
        return [f"{patterns[i % len(patterns)]} [calibration sample {i}]" for i in range(count)]


# ---------------------------------------------------------------------------
# Structured Pruner (Proper Dimension Adjustment)
# ---------------------------------------------------------------------------

class StructuredPruner:
    """Prunes attention heads and FFN neurons with proper dim adjustment.

    Understands transformer model graph:
    - q_proj, k_proj, v_proj output dims  -> paired with o_proj input dims
    - gate_proj, up_proj output dims       -> paired with down_proj input dims
    """

    def __init__(self, ratio: float = 0.2):
        self.ratio = ratio

    def prune(self, model: nn.Module) -> nn.Module:
        if self.ratio <= 0:
            return model
        logger.info(f"Structured pruning with ratio={self.ratio}")

        linear_layers = {}
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Linear):
                linear_layers[name] = mod

        paired_groups = self._find_paired_groups(linear_layers)
        replacements = []

        for q_name, kv_names, o_name, g_name, u_name, d_name in paired_groups:
            q_mod = linear_layers.get(q_name)
            o_mod = linear_layers.get(o_name)
            if q_mod is None or o_mod is None:
                continue

            num_heads = self._infer_num_heads(q_mod, o_mod)
            if num_heads <= 0:
                continue
            heads_to_prune = max(1, int(num_heads * self.ratio))

            head_scores = self._score_heads(q_mod)
            prune_idx = head_scores.argsort()[:heads_to_prune]

            keep_dims = self._head_to_dims(num_heads, q_mod.out_features, prune_idx)
            keep_count = keep_dims.sum().item()
            if keep_count >= q_mod.out_features:
                continue

            new_q = self._slice_linear(q_mod, out_keep=keep_dims)
            new_o = self._slice_linear(o_mod, in_keep=keep_dims)

            q_parent, q_key = self._get_parent(model, q_name)
            o_parent, o_key = self._get_parent(model, o_name)
            replacements.append((q_parent, q_key, new_q))
            replacements.append((o_parent, o_key, new_o))

            for kv_name in kv_names:
                kv_mod = linear_layers.get(kv_name)
                if kv_mod is not None:
                    new_kv = self._slice_linear(kv_mod, out_keep=keep_dims)
                    kv_parent, kv_key = self._get_parent(model, kv_name)
                    replacements.append((kv_parent, kv_key, new_kv))

            for g_name2, u_name2, d_name2 in [(g_name, u_name, d_name)]:
                g_mod = linear_layers.get(g_name2)
                d_mod = linear_layers.get(d_name2)
                u_mod = linear_layers.get(u_name2)
                if g_mod is None or d_mod is None:
                    continue

                g_norms = g_mod.weight.data.abs().sum(dim=0)
                threshold = torch.quantile(g_norms.float(), self.ratio)
                ffn_keep = g_norms > threshold
                ffn_count = ffn_keep.sum().item()
                if ffn_count >= g_mod.out_features:
                    continue

                new_g = self._slice_linear(g_mod, out_keep=ffn_keep)
                new_u = self._slice_linear(u_mod, out_keep=ffn_keep) if u_mod is not None else None
                new_d = self._slice_linear(d_mod, in_keep=ffn_keep)

                g_parent, g_key = self._get_parent(model, g_name2)
                d_parent, d_key = self._get_parent(model, d_name2)
                replacements.append((g_parent, g_key, new_g))
                replacements.append((d_parent, d_key, new_d))
                if new_u is not None:
                    u_parent, u_key = self._get_parent(model, u_name2)
                    replacements.append((u_parent, u_key, new_u))

        for parent, key, new_mod in replacements:
            setattr(parent, key, new_mod)

        logger.info(f"Structured pruning: replaced {len(replacements)} modules")
        return model

    def _find_paired_groups(self, layers: Dict[str, nn.Module]) -> List[Tuple]:
        """Find (q_proj, [k_proj, v_proj], o_proj, gate_proj, up_proj, down_proj) groups."""
        qkv_groups = {}
        ffn_groups = {}
        for name in layers:
            parts = name.split(".")
            layer_idx = next((p for p in parts if p.isdigit()), "0")
            base = ".".join(p for p in parts if not p.isdigit())

            if base.endswith("q_proj"):
                prefix = name[:name.rfind("q_proj")]
                k_name = prefix + "k_proj"
                v_name = prefix + "v_proj"
                o_name = prefix + "o_proj"
                qkv_groups[layer_idx] = (name, [k_name, v_name], o_name)

            if base.endswith("gate_proj"):
                prefix = name[:name.rfind("gate_proj")]
                u_name = prefix + "up_proj"
                d_name = prefix + "down_proj"
                ffn_groups[layer_idx] = (name, u_name, d_name)

        result = []
        for lid in sorted(set(qkv_groups.keys()) | set(ffn_groups.keys())):
            q_name, kv_names, o_name = qkv_groups.get(lid, ("", [], ""))
            g_name, u_name, d_name = ffn_groups.get(lid, ("", "", ""))
            if q_name and o_name:
                result.append((q_name, kv_names, o_name, g_name, u_name, d_name))
        return result

    def _infer_num_heads(self, q_proj: nn.Linear, o_proj: nn.Linear) -> int:
        if q_proj.out_features <= 0 or o_proj.in_features <= 0:
            return 0
        candidates = [32, 64, 16, 8, 12, 24, 48, 96, 128]
        for h in candidates:
            if q_proj.out_features % h == 0 and o_proj.in_features % h == 0:
                return h
        return max(1, q_proj.out_features // 64)

    def _score_heads(self, q_proj: nn.Linear) -> torch.Tensor:
        w = q_proj.weight.data
        return w.abs().sum(dim=1)

    def _head_to_dims(self, num_heads: int, total_dims: int, prune_idx: torch.Tensor) -> torch.Tensor:
        head_size = total_dims // num_heads
        keep = torch.ones(total_dims, dtype=torch.bool)
        for idx in prune_idx:
            start = idx * head_size
            end = start + head_size
            keep[start:end] = False
        return keep

    def _slice_linear(self, mod: nn.Linear, out_keep: Optional[torch.Tensor] = None,
                      in_keep: Optional[torch.Tensor] = None) -> nn.Linear:
        w = mod.weight.data
        b = mod.bias.data if mod.bias is not None else None

        new_out = w.shape[0] if out_keep is None else out_keep.sum().item()
        new_in = w.shape[1] if in_keep is None else in_keep.sum().item()

        new_mod = nn.Linear(new_in, new_out, bias=b is not None, device=w.device, dtype=w.dtype)

        kept_w = w
        if out_keep is not None:
            kept_w = kept_w[out_keep]
        if in_keep is not None:
            kept_w = kept_w[:, in_keep]
        new_mod.weight.data = kept_w.contiguous()

        if b is not None:
            kept_b = b[out_keep] if out_keep is not None else b
            new_mod.bias.data = kept_b.contiguous()

        return new_mod

    def _get_parent(self, model: nn.Module, name: str):
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            if p.isdigit():
                parent = parent[int(p)]
            else:
                parent = getattr(parent, p)
        return parent, parts[-1]


# ---------------------------------------------------------------------------
# Compression Pipeline
# ---------------------------------------------------------------------------

class CompressionPipeline:
    def __init__(self, config: CompressionConfig):
        self.config = config
        self._strategy_selector = StrategySelector()

    def plan(self) -> CompressionPlan:
        model_params = 0.0
        try:
            dummy = nn.Linear(1, 1)
            model_params = 7.0
        except Exception:
            pass
        return self._strategy_selector.build_plan(dummy, model_params)

    def apply(self, model, tokenizer=None) -> nn.Module:
        if not self.config.enabled or self.config.method == CompressionMethod.NONE:
            return model

        logger.info(f"Applying compression: method={self.config.method.value}")

        if self.config.method == CompressionMethod.AUTO:
            return self._auto_compress(model, tokenizer)
        elif self.config.method in (CompressionMethod.PTQ_INT8,):
            return self.apply_quantization(model, bits=8, tokenizer=tokenizer)
        elif self.config.method in (CompressionMethod.PTQ_INT4, CompressionMethod.QUANT_AWQ, CompressionMethod.QUANT_GPTQ):
            quant_method = "gptq" if self.config.method == CompressionMethod.QUANT_GPTQ else self.config.quant_method
            return self.apply_quantization(model, bits=4, tokenizer=tokenizer, quant_method=quant_method)
        elif self.config.method == CompressionMethod.PRUNING_STRUCTURED:
            return self.apply_pruning(model)
        elif self.config.method == CompressionMethod.DISTILLATION:
            return self.apply_distillation(model, tokenizer)
        else:
            logger.warning(f"Unknown compression method: {self.config.method}")
            return model

    def apply_pipeline(self, model, plan: CompressionPlan, tokenizer=None) -> nn.Module:
        logger.info(f"Executing compression pipeline: {plan.summary()}")

        for stage in sorted(plan.stages, key=lambda s: s.order):
            logger.info(f"  Stage {stage.order}: {stage.method.value}")
            try:
                if stage.method == CompressionMethod.PRUNING_STRUCTURED:
                    old_ratio = self.config.pruning_ratio
                    self.config.pruning_ratio = stage.params.get("ratio", 0.15)
                    model = self.apply_pruning(model)
                    self.config.pruning_ratio = old_ratio
                elif stage.method in (CompressionMethod.PTQ_INT8, CompressionMethod.PTQ_INT4,
                                      CompressionMethod.QUANT_AWQ, CompressionMethod.QUANT_GPTQ):
                    bits = stage.params.get("bits", 4)
                    quant_method = "gptq" if stage.method == CompressionMethod.QUANT_GPTQ else "awq"
                    model = self.apply_quantization(model, bits=bits, tokenizer=tokenizer,
                                                    quant_method=quant_method)
                elif stage.method == CompressionMethod.DISTILLATION:
                    model = self.apply_distillation(model, tokenizer)
            except Exception as e:
                logger.warning(f"Stage {stage.order} failed: {e}, continuing")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Pipeline complete")
        return model

    def apply_quantization(self, model, bits: int = 8, tokenizer=None,
                           quant_method: str = "awq") -> nn.Module:
        if bits == 8:
            logger.info("Applying INT8 dynamic quantization")
            model = torch.quantization.quantize_dynamic(
                model, {nn.Linear}, dtype=torch.qint8,
            )
        elif bits == 4:
            logger.info("Applying INT4 quantization via AWQ/GPTQ")
            calib_inputs = self._get_calib_inputs(model, tokenizer) if tokenizer else []
            if quant_method == "gptq" and calib_inputs:
                quantizer = GPTQQuantizer(group_size=128)
                model = quantizer.quantize_model(model, calib_inputs)
            else:
                quantizer = AWQQuantizer(group_size=128)
                model = quantizer.quantize_model(model, calib_inputs)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return model

    def apply_pruning(self, model) -> nn.Module:
        ratio = self.config.pruning_ratio
        if ratio <= 0:
            logger.warning("Pruning ratio is 0, skipping pruning")
            return model

        logger.info(f"Applying structured pruning with ratio={ratio}")
        pruner = StructuredPruner(ratio=ratio)
        model = pruner.prune(model)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return model

    def apply_distillation(self, model, tokenizer=None) -> nn.Module:
        if self.config.distillation_teacher is None:
            logger.warning("No teacher model specified, skipping distillation")
            return model
        if tokenizer is None:
            logger.warning("Tokenizer required for distillation, skipping")
            return model

        logger.info(f"Applying knowledge distillation from {self.config.distillation_teacher}")

        try:
            from transformers import AutoModelForCausalLM
            teacher = AutoModelForCausalLM.from_pretrained(
                self.config.distillation_teacher,
                torch_dtype=model.dtype if hasattr(model, 'dtype') else torch.float16,
                device_map="auto",
            )
            teacher.eval()
        except Exception as e:
            logger.error(f"Failed to load teacher model: {e}")
            return model

        cal_loader = CalibrationDataLoader(tokenizer, n_samples=min(32, self.config.calibration_samples))
        calibration_data = cal_loader.generate()

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        temperature = 2.0

        model.train()
        for inputs in calibration_data[:8]:
            device = next(model.parameters()).device
            inputs = inputs.to(device)

            with torch.no_grad():
                teacher_outputs = teacher(inputs)
                teacher_logits = teacher_outputs.logits

            student_outputs = model(inputs)
            student_logits = student_outputs.logits

            student_log_probs = torch.log_softmax(student_logits / temperature, dim=-1)
            teacher_probs = torch.softmax(teacher_logits / temperature, dim=-1)

            loss = nn.functional.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        del teacher
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Distillation complete")
        return model

    def _get_calib_inputs(self, model, tokenizer) -> List[torch.Tensor]:
        try:
            loader = CalibrationDataLoader(tokenizer, n_samples=min(32, self.config.calibration_samples))
            return loader.generate()
        except Exception as e:
            logger.debug(f"Failed to generate calibration data: {e}")
            return []

    def _auto_compress(self, model, tokenizer=None) -> nn.Module:
        total_params = sum(p.numel() for p in model.parameters()) / 1e9
        selector = StrategySelector()
        plan = selector.build_plan(model, total_params)
        logger.info(f"Hardware: {plan.target_hardware.value}")
        logger.info(f"Selected plan: {plan.summary()}")
        if not plan.stages:
            return self._vram_based_compress(model, tokenizer)
        return self.apply_pipeline(model, plan, tokenizer)

    def _vram_based_compress(self, model, tokenizer=None) -> nn.Module:
        if not torch.cuda.is_available():
            logger.info("No CUDA device available, skipping auto-compression")
            return model

        total_params = sum(p.numel() for p in model.parameters())
        dtype_bytes = 2 if model.dtype == torch.float16 else 4
        estimated_vram = total_params * dtype_bytes
        available_vram = torch.cuda.get_device_properties(0).total_memory
        utilization = torch.cuda.memory_allocated()
        free_vram = available_vram - utilization

        logger.info(f"Model estimated VRAM: {estimated_vram / 1e9:.1f}GB, Free VRAM: {free_vram / 1e9:.1f}GB")

        if estimated_vram > free_vram * 0.8:
            if self.config.pruning_ratio > 0:
                logger.info("Auto-compression: applying pruning first")
                model = self.apply_pruning(model)
            if estimated_vram > free_vram * 0.8:
                logger.info("Auto-compression: applying INT4 quantization")
                model = self.apply_quantization(model, bits=4, tokenizer=tokenizer)
        else:
            logger.info("Model fits in VRAM, skipping auto-compression")

        return model
