"""ONNX Runtime backend adapter for cross-platform CPU/GPU inference.

ONNX Runtime provides hardware-optimized inference across a wide range of
execution providers: CPU, CUDA (NVIDIA), DirectML (Windows), ROCm (AMD),
and OpenVINO (Intel). This adapter wraps ``ort.InferenceSession`` and
implements the ``BackendAdapter`` protocol so ONNX models can participate
in the distributed-llm cluster as first-class inference nodes.

Requirements:
    - Production: ``pip install onnxruntime`` (CPU) or ``onnxruntime-gpu`` (CUDA/ROCm/DirectML).
    - Model export: ``pip install distllm[export]`` (transformers, optimum, protobuf) to convert HuggingFace models to ONNX.
    - Tensor inputs are converted to numpy arrays automatically before session inference.

Limitations:
    - ``generate()`` requires an external tokenizer/detokenizer; the base
      stubs raise ``NotImplementedError`` until the caller sets
      ``self._tokenizer`` / ``self._detokenizer``.
    - Not all HuggingFace model architectures can be exported to ONNX
      (consult optimum.onnxruntime compatibility tables).
    - Dynamic shapes (variable-length sequences) require careful session
      configuration and may degrade performance compared to fixed shapes.
    - Pipeline/partitioned mode (layer_start, layer_end) is accepted at
      construction but not yet wired into session partitioning; the full
      ONNX model is loaded regardless.
    - FP16 execution requires an execution provider with native FP16
      support (CUDA >= 11.0, DirectML on supported hardware).
"""

from __future__ import annotations

from typing import Any

import torch
from loguru import logger

from distllm.backends.protocol import BackendAdapter
from distllm.errors import ModelLoadError

try:
    import onnxruntime as ort

    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False


class ONNXNodeAdapter(BackendAdapter):
    """ONNX Runtime backend for hardware-optimized inference.

    Converts HuggingFace models to ONNX format (or loads existing ONNX
    models) and runs inference through ONNX Runtime with the best
    available execution provider (CUDA, DirectML, ROCm, OpenVINO, CPU).

    Args:
        model_name: HuggingFace model name or path, or path to a
            ``.onnx`` file.
        device: Target device (``"auto"``, ``"cuda"``, ``"cpu"``).
        dtype: Model dtype (``"float16"``, ``"float32"``).
        layer_start: First layer index for pipeline mode.
        layer_end: Last layer index for pipeline mode.
        execution_provider: ONNX Runtime execution provider to use.
            ``None`` = auto-select best available.
        optimizer_level: ONNX Runtime graph optimization level
            (``"all"``, ``"basic"``, ``"none"``).
        trust_remote_code: Whether to trust HF remote code for export.
        **extra_kwargs: Additional kwargs for ``ort.InferenceSession``.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        dtype: str = "float16",
        layer_start: int = 0,
        layer_end: int = 0,
        total_layers: int = 0,
        execution_provider: str | None = None,
        optimizer_level: str = "all",
        trust_remote_code: bool | None = None,
        **extra_kwargs: Any,
    ):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.total_layers = total_layers
        self._execution_provider = execution_provider
        self._optimizer_level = optimizer_level
        self._extra_kwargs = extra_kwargs

        self._session: ort.InferenceSession | None = None
        self._model = None

    def load_model(self) -> None:
        if not HAS_ONNX:
            raise ModelLoadError(
                self.model_name,
                "onnxruntime not installed. Install with: "
                "pip install onnxruntime onnxruntime-gpu",
            )

        logger.info(f"[ONNX] Loading model: {self.model_name}")

        try:
            providers = self._select_providers()
            sess_options = ort.SessionOptions()
            if self._optimizer_level == "all":
                sess_options.graph_optimization_level = (
                    ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                )
            elif self._optimizer_level == "basic":
                sess_options.graph_optimization_level = (
                    ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
                )
            else:
                sess_options.graph_optimization_level = (
                    ort.GraphOptimizationLevel.ORT_DISABLE_ALL
                )

            self._session = ort.InferenceSession(
                self.model_name,
                sess_options=sess_options,
                providers=providers,
                **self._extra_kwargs,
            )
            logger.info(
                f"[ONNX] Model loaded: {self.model_name} "
                f"(providers: {self._session.get_providers()})"
            )
        except Exception as e:
            self._session = None
            logger.error(f"[ONNX] Failed to load model: {e}")
            raise ModelLoadError(self.model_name, str(e)) from e

    def forward(
        self,
        hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        if self._session is None:
            raise ModelLoadError("onnx", "Model not loaded. Call load_model() first.")

        ort_inputs = {}
        if input_ids is not None:
            ort_inputs["input_ids"] = input_ids.cpu().numpy()
        if hidden_states is not None:
            ort_inputs["hidden_states"] = hidden_states.cpu().numpy()
        if attention_mask is not None:
            ort_inputs["attention_mask"] = attention_mask.cpu().numpy()
        if position_ids is not None:
            ort_inputs["position_ids"] = position_ids.cpu().numpy()

        outputs = self._session.run(None, ort_inputs)
        logits = torch.from_numpy(outputs[0])
        new_kv = (
            [torch.from_numpy(kv) for kv in outputs[1:]]
            if len(outputs) > 1
            else []
        )
        return logits, new_kv

    def shutdown(self) -> None:
        self._session = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("[ONNX] Engine shut down")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> str:
        if self._session is None:
            raise ModelLoadError("onnx", "Model not loaded.")

        inputs = self._tokenize(prompt)
        output_ids = []
        for _ in range(max_tokens):
            outputs = self._session.run(None, inputs)
            logits = torch.from_numpy(outputs[0])
            next_id = self._sample(logits, temperature)
            output_ids.append(next_id)
            inputs["input_ids"] = next_id.numpy().reshape(1, 1)

        return self._detokenize(output_ids)

    def _tokenize(self, text: str) -> dict[str, Any]:
        raise NotImplementedError(
            "ONNX generate() requires a tokenizer. "
            "Use forward() or set self._tokenizer."
        )

    def _detokenize(self, ids: list[int]) -> str:
        raise NotImplementedError(
            "ONNX generate() requires a detokenizer. "
            "Use forward() or set self._detokenizer."
        )

    @staticmethod
    def _sample(logits: torch.Tensor, temperature: float) -> torch.Tensor:
        if temperature == 0.0:
            return logits[0, -1].argmax(dim=-1, keepdim=True)
        probs = torch.softmax(logits[0, -1] / temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    def _select_providers(self) -> list[str]:
        if self._execution_provider:
            return [self._execution_provider]

        available = ort.get_available_providers()
        preferred = ["CUDAExecutionProvider", "ROCMExecutionProvider",
                     "DmlExecutionProvider", "OpenVINOExecutionProvider",
                     "CPUExecutionProvider"]
        return [p for p in preferred if p in available]

    @classmethod
    def display_name(cls) -> str:
        return "ONNX Runtime"

    @classmethod
    def is_available(cls) -> bool:
        return HAS_ONNX

    @classmethod
    def priority_for(cls, device_type: str) -> int:
        mapping = {
            "cuda": 6,
            "rocm": 7,
            "cpu": 5,
            "mps": 2,
            "xpu": 8,
        }
        return mapping.get(device_type, 1)
