"""Quantized model inference backend for edge deployment.

Simulates int4/int8/NF4 quantized inference without requiring
bitsandbytes or other heavy quantization libraries at import time.
"""

import json
import struct
from pathlib import Path
from typing import Optional

from loguru import logger

from distllm.edge.models import QuantizationType


class QuantizationBackend:
    """Placeholder for quantization library integration.

    Can be extended to use bitsandbytes, auto-gptq, or
    llama.cpp for actual quantized inference.
    """

    @staticmethod
    def quantize_weights(
        weights: list[float],
        quant_type: QuantizationType,
    ) -> bytes:
        """Simulate weight quantization."""
        if quant_type == QuantizationType.INT8:
            packed = bytearray()
            for w in weights:
                packed.append(max(0, min(255, int((w + 1.0) * 127.5))))
            return bytes(packed)
        elif quant_type in (QuantizationType.INT4, QuantizationType.NF4, QuantizationType.FP4):
            packed = bytearray()
            for i in range(0, len(weights), 2):
                v0 = max(0, min(15, int((weights[i] + 1.0) * 7.5)))
                v1 = max(0, min(15, int((weights[i + 1] + 1.0) * 7.5))) if i + 1 < len(weights) else 0
                packed.append((v1 << 4) | v0)
            return bytes(packed)
        else:
            packed = bytearray()
            for w in weights:
                packed.extend(struct.pack("<e", w))
            return bytes(packed)

    @staticmethod
    def dequantize_weights(
        data: bytes,
        quant_type: QuantizationType,
        count: int,
    ) -> list[float]:
        """Simulate weight dequantization."""
        if quant_type == QuantizationType.INT8:
            return [(b / 127.5) - 1.0 for b in data[:count]]
        elif quant_type in (QuantizationType.INT4, QuantizationType.NF4, QuantizationType.FP4):
            values = []
            for byte in data:
                v0 = byte & 0x0F
                v1 = (byte >> 4) & 0x0F
                values.append((v0 / 7.5) - 1.0)
                if len(values) < count:
                    values.append((v1 / 7.5) - 1.0)
            return values[:count]
        else:
            result = []
            for i in range(0, min(len(data), count * 2), 2):
                (val,) = struct.unpack("<e", data[i:i + 2])
                result.append(val)
            return result


class QuantizedModel:
    """Represents a quantized model loaded on an edge device."""

    def __init__(
        self,
        model_name: str,
        quant_type: QuantizationType = QuantizationType.INT4,
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.quant_type = quant_type
        self.device = device
        self._loaded = False
        self._compression_ratio = 1.0
        self._backend = QuantizationBackend()

    def load(self, path: Optional[str] = None) -> None:
        if path:
            config_path = Path(path) / "quant_config.json"
            if config_path.exists():
                with open(config_path) as f:
                    cfg = json.load(f)
                self.quant_type = QuantizationType(cfg.get("quant_type", "int4"))
                self._compression_ratio = cfg.get("compression_ratio", 4.0)
        self._loaded = True
        logger.info(f"Quantized model {self.model_name} loaded on {self.device} (type={self.quant_type}, ratio={self._compression_ratio:.1f}x)")

    @property
    def loaded(self) -> bool:
        return self._loaded

    async def generate(self, messages: list[dict], max_tokens: int = 128, temperature: float = 0.7) -> dict:
        if not self._loaded:
            raise RuntimeError(f"Model {self.model_name} not loaded")
        content = f"Edge-quantized ({self.quant_type.value}) response for: {messages[-1].get('content', '')[:50]}..."
        return {
            "id": f"edge-{self.model_name}",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }

    async def generate_stream(self, messages: list[dict], max_tokens: int = 128, temperature: float = 0.7):
        if not self._loaded:
            raise RuntimeError(f"Model {self.model_name} not loaded")
        yield {
            "id": f"edge-{self.model_name}",
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
        content = f"Edge-quantized ({self.quant_type.value})"
        for token in content.split():
            yield {
                "id": f"edge-{self.model_name}",
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {"content": token + " "}, "finish_reason": None}],
            }
        yield {
            "id": f"edge-{self.model_name}",
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
