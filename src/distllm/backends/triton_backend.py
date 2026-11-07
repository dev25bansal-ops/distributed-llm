"""Triton Inference Server backend adapter for distributed inference.

Connects to a running Triton Inference Server instance via its HTTP/gRPC API
to serve models backed by TensorRT, ONNX, PyTorch, or any other framework
supported by Triton's multi-framework backend architecture.

Provides dynamic batching, concurrent model execution, and model config
generation for deployment automation.

Usage:
    backend = TritonNodeAdapter(
        model_name="my_model",
        triton_url="localhost:8000",
        protocol="http",
    )
    backend.load_model()
    logits, kv = backend.forward(input_ids=input_ids)
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, AsyncIterator, Iterator

import numpy as np
import torch
from loguru import logger

from distllm.backends.protocol import BackendAdapter
from distllm.errors import ModelLoadError


# ── Helpers ──────────────────────────────────────────────────────────

_TRITON_PROTOCOLS = ("http", "grpc")
"""Supported Triton client protocols."""

_DEFAULT_TIMEOUT_SECONDS = 30.0
"""Default timeout for Triton API calls."""


def _import_tritonclient(protocol: str):
    """Lazily import the appropriate ``tritonclient`` module.

    Returns:
        Tuple of ``(inference_server_client, grpc_service_pb2, service_pb2)``.
        The gRPC-specific imports return ``None`` for the HTTP protocol.
    """
    if protocol == "grpc":
        try:
            import tritonclient.grpc as client
            from tritonclient.grpc import (
                service_pb2,
            )
            from tritonclient.grpc.service_pb2 import ModelInferResponse
            return client, service_pb2, ModelInferResponse
        except ImportError as e:
            raise ImportError(
                "tritonclient[grpc] is required for gRPC protocol. "
                "Install with: pip install tritonclient[grpc]"
            ) from e
    else:
        try:
            import tritonclient.http as client
            return client, None, None
        except ImportError as e:
            raise ImportError(
                "tritonclient[http] is required for HTTP protocol. "
                "Install with: pip install tritonclient[http]"
            ) from e


def _np_to_triton_dtype(np_dtype: np.dtype) -> str:
    """Map a NumPy dtype to a Triton data-type string."""
    mapping = {
        np.dtype("bool"): "TYPE_BOOL",
        np.dtype("int8"): "TYPE_INT8",
        np.dtype("int16"): "TYPE_INT16",
        np.dtype("int32"): "TYPE_INT32",
        np.dtype("int64"): "TYPE_INT64",
        np.dtype("uint8"): "TYPE_UINT8",
        np.dtype("uint16"): "TYPE_UINT16",
        np.dtype("uint32"): "TYPE_UINT32",
        np.dtype("uint64"): "TYPE_UINT64",
        np.dtype("float16"): "TYPE_FP16",
        np.dtype("float32"): "TYPE_FP32",
        np.dtype("float64"): "TYPE_FP64",
        np.dtype("str"): "TYPE_STRING",
    }
    return mapping.get(np_dtype, "TYPE_FP32")


def _torch_dtype_to_str(dtype: torch.dtype) -> str:
    """Map a PyTorch dtype to a Triton config string."""
    return {
        torch.float16: "TYPE_FP16",
        torch.float32: "TYPE_FP32",
        torch.float64: "TYPE_FP64",
        torch.int8: "TYPE_INT8",
        torch.int16: "TYPE_INT16",
        torch.int32: "TYPE_INT32",
        torch.int64: "TYPE_INT64",
        torch.bool: "TYPE_BOOL",
        torch.uint8: "TYPE_UINT8",
    }.get(dtype, "TYPE_FP32")


# ── Adapter ──────────────────────────────────────────────────────────


class TritonNodeAdapter(BackendAdapter):
    """Wraps Triton Inference Server as a per-node inference backend.

    Communicates with a Triton server via HTTP or gRPC. Suitable for
    production deployments where models are pre-deployed on a Triton
    instance and inference is dispatched over the network.

    Args:
        model_name: Name of the model as deployed on Triton
            (matches the model repository directory name).
        triton_url: Triton server URL (e.g. ``"localhost:8000"`` for HTTP
            or ``"localhost:8001"`` for gRPC).
        protocol: Transport protocol — ``"http"`` (default) or ``"grpc"``.
        model_version: Model version to use (``""`` = latest).
        timeout: Timeout in seconds for Triton API calls.
        verbose: Enable verbose Triton client logging.
        ssl: Whether to use SSL for the connection.
        ssl_options: Dict of SSL options (``cert_file``, ``key_file``,
            ``ca_file``, etc.).
        client_kwargs: Additional keyword arguments forwarded to the
            Triton client constructor.
        tokenizer: Optional tokenizer object exposing ``encode(text) -> list[int]``
            and ``decode(ids, skip_special_tokens=True) -> str`` (e.g. a
            HuggingFace ``PreTrainedTokenizerBase``). Required for
            ``generate()``/``generate_stream()`` — Triton ships no tokenizer
            and the old byte-wise ``ord(c)``/``chr(id)`` fallback was removed
            (lossy: every token id >= 127 became ``" "``).
        tokenizer_name: Optional HuggingFace model id or local path used to
            lazily load a tokenizer in ``load_model()`` via
            ``AutoTokenizer.from_pretrained(..., trust_remote_code=False)``.
            Ignored when ``tokenizer`` is given.
        tokenizer_kwargs: Extra kwargs forwarded to
            ``AutoTokenizer.from_pretrained``.
        eos_token_id: Optional explicit EOS token id. Defaults to the
            tokenizer's ``eos_token_id``, falling back to ``0``.
        max_context: Sliding-window cap (in tokens) on the ids sent per
            decode step. Keeps long generations bounded the way the old
            512-character truncation did; raise it for models with a large
            context that do not manage KV cache server-side.
    """

    def __init__(
        self,
        model_name: str,
        triton_url: str = "localhost:8000",
        protocol: str = "http",
        model_version: str = "",
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        verbose: bool = False,
        ssl: bool = False,
        ssl_options: dict[str, Any] | None = None,
        tokenizer: Any = None,
        tokenizer_name: str | None = None,
        tokenizer_kwargs: dict[str, Any] | None = None,
        eos_token_id: int | None = None,
        max_context: int = 2048,
        **client_kwargs: Any,
    ):
        if protocol not in _TRITON_PROTOCOLS:
            raise ValueError(
                f"Unsupported Triton protocol {protocol!r}. "
                f"Choose from {_TRITON_PROTOCOLS}."
            )

        self.model_name = model_name
        self.triton_url = triton_url
        self.protocol = protocol
        self.model_version = model_version
        self.timeout = timeout
        self.verbose = verbose
        self.ssl = ssl
        self.ssl_options = ssl_options or {}
        self._client_kwargs = client_kwargs
        self._tokenizer_override = tokenizer
        self.tokenizer_name = tokenizer_name
        self._tokenizer_kwargs = tokenizer_kwargs or {}
        self._eos_token_id_override = eos_token_id
        self.max_context = max_context

        self._client = None
        self._model_metadata: dict[str, Any] = {}
        self._model_config: dict[str, Any] = {}
        self._tokenizer = tokenizer
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Core interface ───────────────────────────────────────────────

    def load_model(self) -> None:
        """Connect to Triton and verify the model is ready for inference.

        Raises:
            ModelLoadError: If connection fails or the model is not ready.
        """
        logger.info(
            f"[Triton] Connecting to {self.triton_url} "
            f"(protocol={self.protocol}, model={self.model_name})"
        )

        try:
            client_mod, grpc_pb2, infer_resp_cls = _import_tritonclient(self.protocol)
            self._grpc_pb2 = grpc_pb2
            self._infer_resp_cls = infer_resp_cls

            url = self.triton_url
            if self.protocol == "http":
                self._client = client_mod.InferenceServerClient(
                    url=url,
                    verbose=self.verbose,
                    ssl=self.ssl,
                    ssl_options=self.ssl_options,
                    **self._client_kwargs,
                )
            else:
                self._client = client_mod.InferenceServerClient(
                    url=url,
                    verbose=self.verbose,
                    ssl=self.ssl,
                    ssl_options=self.ssl_options,
                    **self._client_kwargs,
                )
        except ImportError:
            raise
        except Exception as e:
            self._client = None
            logger.error(f"[Triton] Failed to create client for {self.triton_url}: {e}")
            raise ModelLoadError(self.model_name, str(e)) from e

        # Verify server is live and model is ready
        if not self._probe_server_ready():
            self._client = None
            raise ModelLoadError(
                self.model_name,
                f"Triton server at {self.triton_url} is not ready",
            )

        if not self._probe_model_ready():
            self._client = None
            raise ModelLoadError(
                self.model_name,
                f"Model {self.model_name!r} is not ready on Triton at {self.triton_url}. "
                f"Verify the model is deployed in the Triton model repository.",
            )

        # Load model metadata and config
        try:
            if self.protocol == "http":
                self._model_metadata = self._client.get_model_metadata(
                    self.model_name, self.model_version
                )
                self._model_config = self._client.get_model_config(
                    self.model_name, self.model_version
                )
            else:
                meta_resp = self._client.get_model_metadata(
                    self.model_name, self.model_version
                )
                config_resp = self._client.get_model_config(
                    self.model_name, self.model_version
                )
                self._model_metadata = {
                    "name": meta_resp.name,
                    "versions": list(meta_resp.versions),
                    "platform": meta_resp.platform,
                    "inputs": [
                        {"name": i.name, "dtype": i.datatype, "shape": list(i.shape)}
                        for i in meta_resp.inputs
                    ],
                    "outputs": [
                        {"name": o.name, "dtype": o.datatype, "shape": list(o.shape)}
                        for o in meta_resp.outputs
                    ],
                }
                if config_resp.HasField("config"):
                    cfg = config_resp.config
                    self._model_config = {
                        "name": cfg.name,
                        "platform": cfg.platform,
                        "max_batch_size": cfg.max_batch_size,
                        "input": [
                            {"name": i.name, "dtype": i.data_type, "dims": list(i.dims)}
                            for i in cfg.input
                        ],
                        "output": [
                            {"name": o.name, "dtype": o.data_type, "dims": list(o.dims)}
                            for o in cfg.output
                        ],
                    }
                else:
                    self._model_config = {}
        except Exception as e:
            logger.warning(f"[Triton] Failed to load model metadata: {e}")

        logger.info(
            f"[Triton] Model {self.model_name} ready on {self.triton_url}"
        )
        self._load_tokenizer()

    # ── Tokenizer ────────────────────────────────────────────────────

    def _load_tokenizer(self) -> None:
        """Load a real tokenizer for ``generate()``.

        Preference order: an explicitly injected tokenizer, then
        ``tokenizer_name`` (or the model name) via
        ``AutoTokenizer.from_pretrained(..., trust_remote_code=False)``.
        A failure here is warned about, not raised — ``forward()`` with
        pre-tokenized ids does not need one — but ``generate()`` will
        then raise a clear error instead of silently falling back to
        byte-wise ``ord``/``chr``.
        """
        if self._tokenizer is not None:
            return
        name = self.tokenizer_name or self.model_name
        if not name:
            return
        try:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                name, trust_remote_code=False, **self._tokenizer_kwargs
            )
        except Exception as e:
            self._tokenizer = None
            logger.warning(
                f"[Triton] No tokenizer loaded for {name!r} ({e}). "
                "generate()/generate_stream() will raise; "
                "pass tokenizer= or tokenizer_name=."
            )

    def get_tokenizer(self) -> Any:
        """Return the backend's tokenizer, if available."""
        return self._tokenizer

    @property
    def _eos_token_id(self) -> int:
        if self._eos_token_id_override is not None:
            return self._eos_token_id_override
        if self._tokenizer is not None:
            eos = getattr(self._tokenizer, "eos_token_id", None)
            if eos is not None:
                return int(eos)
        return 0

    def _encode(self, text: str) -> list[int]:
        """Encode ``text`` to token ids — the only encoding path."""
        tokenizer = self._tokenizer
        if tokenizer is None:
            raise RuntimeError(
                "Triton generate() requires a real tokenizer: Triton ships "
                "none and the old byte-wise ord(c) fallback was removed "
                "(it silently corrupted every token id >= 127). Pass "
                "tokenizer=... or tokenizer_name=... (HF model id/path) "
                "to TritonNodeAdapter."
            )
        encode = getattr(tokenizer, "encode", None)
        if callable(encode):
            return [int(t) for t in encode(text, add_special_tokens=False)]
        try:
            out = tokenizer(text, add_special_tokens=False)["input_ids"]
        except Exception:
            # CQ-08: failure is recorded in the output value (visible to the caller); not propagated.
            out = tokenizer.encode(text)
        return [int(t) for t in out]

    def _decode(self, ids: list[int]) -> str:
        """Decode token ids — the only decoding path."""
        tokenizer = self._tokenizer
        if tokenizer is None:
            raise RuntimeError(
                "Triton generate() requires a real tokenizer: Triton ships "
                "none and the old chr(id) fallback was removed (it silently "
                "mapped every token id >= 127 to a space). Pass "
                "tokenizer=... or tokenizer_name=... to TritonNodeAdapter."
            )
        decode = getattr(tokenizer, "decode", None)
        if callable(decode):
            return decode(ids, skip_special_tokens=True)
        return tokenizer.decode(ids)

    def forward(
        self,
        hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Run inference via Triton's Infer API.

        Sends the input tensor to Triton and returns the output logits
        and an empty KV cache (Triton manages KV cache server-side).

        Args:
            input_ids: Token IDs (first pipeline node only).
            hidden_states: Activations from previous node (for pipeline
                parallelism; not yet supported via remote Triton).

        Returns:
            Tuple of ``(output_logits, [])`` where the KV cache is empty
            because Triton manages it internally.
        """
        if self._client is None:
            raise ModelLoadError(
                self.model_name,
                "Triton not connected. Call load_model() first.",
            )

        if input_ids is not None:
            return self._forward_input_ids(input_ids)
        if hidden_states is not None:
            raise NotImplementedError(
                "Pipeline-mode forward via hidden_states is not supported "
                "through remote Triton. Triton manages layer execution "
                "server-side. Use input_ids for full-model inference."
            )
        raise ValueError("Either input_ids or hidden_states must be provided")

    def _forward_input_ids(
        self, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Send ``input_ids`` to Triton and retrieve logits.

        Constructs an InferInput, performs the inference call, and
        extracts the output logits tensor.
        """
        import numpy as np

        # Convert input_ids to int32 numpy array (Triton standard)
        input_np = input_ids.cpu().numpy().astype(np.int32)

        if self.protocol == "http":
            return self._forward_http(input_np)
        return self._forward_grpc(input_np)

    def _forward_http(
        self, input_np: np.ndarray
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """HTTP protocol forward pass."""
        import tritonclient.http as http_client

        input_name = self._get_input_name("input_ids")
        output_name = self._get_output_name("logits")

        infer_input = http_client.InferInput(
            input_name, list(input_np.shape), "INT32"
        )
        infer_input.set_data_from_numpy(input_np)

        infer_output = http_client.InferRequestedOutput(output_name)

        try:
            results = self._client.infer(
                model_name=self.model_name,
                model_version=self.model_version,
                inputs=[infer_input],
                outputs=[infer_output],
                timeout=self.timeout,
            )
            output_np = results.as_numpy(output_name)
        except Exception as e:
            raise RuntimeError(
                f"Triton HTTP inference failed for {self.model_name}: {e}"
            ) from e

        logits = torch.from_numpy(output_np).to(
            dtype=torch.float32, device=self._device
        )
        return logits, []

    def _forward_grpc(
        self, input_np: np.ndarray
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """gRPC protocol forward pass."""
        import tritonclient.grpc as grpc_client

        input_name = self._get_input_name("input_ids")
        output_name = self._get_output_name("logits")

        infer_input = grpc_client.InferInput(
            input_name, list(input_np.shape), "INT32"
        )
        infer_input.set_data_from_numpy(input_np)

        infer_output = grpc_client.InferRequestedOutput(output_name)

        try:
            results = self._client.infer(
                model_name=self.model_name,
                model_version=self.model_version,
                inputs=[infer_input],
                outputs=[infer_output],
                timeout=self.timeout,
            )
            output_np = results.as_numpy(output_name)
        except Exception as e:
            raise RuntimeError(
                f"Triton gRPC inference failed for {self.model_name}: {e}"
            ) from e

        logits = torch.from_numpy(output_np).to(
            dtype=torch.float32, device=self._device
        )
        return logits, []

    def _get_input_name(self, default: str = "input_ids") -> str:
        """Derive the input tensor name from model metadata."""
        inputs = self._model_metadata.get("inputs", [])
        if inputs:
            return inputs[0].get("name", default) if isinstance(inputs[0], dict) else default
        return default

    def _get_output_name(self, default: str = "logits") -> str:
        """Derive the output tensor name from model metadata."""
        outputs = self._model_metadata.get("outputs", [])
        if outputs:
            return outputs[0].get("name", default) if isinstance(outputs[0], dict) else default
        return default

    # ── Health & load ────────────────────────────────────────────────

    def health_check(self) -> bool:
        """Ping the Triton server health endpoint.

        Returns:
            ``True`` if the server is live and the model is ready.
        """
        if self._client is None:
            return False
        try:
            if not self._client.is_server_live():
                return False
            if not self._client.is_server_ready():
                return False
            if not self._client.is_model_ready(self.model_name, self.model_version):
                return False
            return True
        except Exception:
            # CQ-08: intentional fail-open: return False on failure; callers treat False as 'not available/enabled'.
            return False

    def current_load(self) -> float:
        """Query the Triton server's inferred load.

        Returns a load factor in ``[0.0, 1.0]`` based on the ratio of
        inflight requests to max batch size. Falls back to ``0.0`` on
        error.
        """
        if self._client is None:
            return 0.0
        try:
            stats = self._client.get_inference_statistics(
                model_name=self.model_name, model_version=self.model_version
            )
            if self.protocol == "http":
                stat = stats.get("model_stats", [{}])[0]
            else:
                stat = stats.model_stats[0] if stats.model_stats else {}

            inflight = int(getattr(stat, "inference_count", 0) if not isinstance(stat, dict) else stat.get("inference_count", 0))
            max_batch = self._model_config.get("max_batch_size", 32)
            load = min(inflight / max(max_batch, 1), 1.0)
            return float(load)
        except Exception:
            # CQ-08: intentional fail-open: numeric/estimate fallback on failure.
            return 0.0

    # ── Generation ───────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **sampling_kwargs: Any,
    ) -> str:
        """Single-shot text generation via Triton's Infer API.

        Token ids are encoded once (``_encode``) and then extended by one
        id per decode step — each step sends only the sliding window of
        recent ids and appends the sampled id to the running list, so
        encode cost is O(prompt + 1) instead of the old O(n^2)
        re-encoding of the growing output string. Callers that need fewer
        round trips should use ``batch_generate()`` (one Infer per
        sequence step, but shared across sequences) or the ``sequence``
        passthrough the gRPC client supports.

        Args:
            prompt: Input text (encoded with the backend tokenizer).
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (unused: greedy argmax is
                used, matching the previous behaviour; pass a real sampler
                via ``sampling_kwargs`` later).
            top_p: Nucleus sampling threshold (unused, see above).

        Returns:
            Generated text (prompt excluded).
        """
        if self._client is None:
            raise ModelLoadError(
                self.model_name,
                "Triton not connected. Call load_model() first.",
            )
        input_ids = self._encode(prompt)
        if not input_ids:
            return ""
        full = self._decode_loop(input_ids, max_tokens)
        return self._decode(full[len(input_ids):])

    def batch_generate(
        self,
        prompts: list[str],
        max_tokens: int = 256,
        **sampling_kwargs: Any,
    ) -> list[str]:
        """Generate for many prompts, batching decode steps.

        All sequences share one Infer RPC per step: at each step, live
        sequences are padded to a common width, sent as a single batched
        input, and each sequence appends its own sampled token. This
        replaces ``len(prompts)`` independent per-token RPC streams
        (N sequences x T steps) with T RPCs total.

        Note: Triton models without server-side KV-cache management
        re-compute the full context per step; per-sequence history is
        still maintained here so no sequence is re-encoded.
        """
        if self._client is None:
            raise ModelLoadError(
                self.model_name,
                "Triton not connected. Call load_model() first.",
            )
        if not prompts:
            return []
        seqs: list[list[int]] = [self._encode(p) for p in prompts]
        prompt_lens = [len(s) for s in seqs]
        finished = [len(s) == 0 for s in seqs]
        for _ in range(max_tokens):
            live = [i for i, done in enumerate(finished) if not done]
            if not live:
                break
            width = max(len(seqs[i]) for i in live)
            width = min(width, self.max_context) if self.max_context else width
            batch = np.array(
                [
                    ([0] * (width - len(seqs[i][-width:]))) + seqs[i][-width:]
                    for i in live
                ],
                dtype=np.int32,
            )
            output_np = self._infer_once(batch)
            if output_np is None or output_np.size == 0:
                break
            for row, i in enumerate(live):
                token_id = int(np.argmax(output_np[row, -1, :]))
                if token_id == self._eos_token_id:
                    finished[i] = True
                else:
                    seqs[i].append(token_id)
        return [
            "" if len(s) == prompt_lens[i] else self._decode(s[prompt_lens[i]:])
            for i, s in enumerate(seqs)
        ]

    # ── Decode engine (shared by generate / generate_stream) ──────────

    def _infer_module(self):
        if self.protocol == "http":
            import tritonclient.http as client_mod

            return client_mod, client_mod.InferInput, client_mod.InferRequestedOutput
        import tritonclient.grpc as client_mod

        return client_mod, client_mod.InferInput, client_mod.InferRequestedOutput

    def _infer_once(self, input_np: np.ndarray) -> Any:
        """One Infer RPC for an int32 ``(batch, seq)`` array."""
        _, infer_input_cls, infer_output_cls = self._infer_module()
        input_name = self._get_input_name("input_ids")
        output_name = self._get_output_name("logits")

        infer_input = infer_input_cls(
            input_name, list(input_np.shape), "INT32"
        )
        infer_input.set_data_from_numpy(input_np)
        infer_output = infer_output_cls(output_name)

        results = self._client.infer(
            model_name=self.model_name,
            model_version=self.model_version,
            inputs=[infer_input],
            outputs=[infer_output],
            timeout=self.timeout,
        )
        return results.as_numpy(output_name)

    def _decode_loop(self, input_ids: list[int], max_tokens: int) -> list[int]:
        """Decode ``max_tokens`` ids, returning prompt + generated ids.

        Sends one Infer per step with the sliding ``max_context`` window
        and appends the sampled id — O(prompt + T) total work instead of
        the old O(n^2) string re-encode. Sampling is greedy argmax (the
        previous behaviour); ``temperature``/``top_p`` are accepted for
        signature compatibility and ignored.
        """
        hist = list(input_ids)
        for _ in range(max_tokens):
            window = hist[-self.max_context:] if self.max_context else hist
            input_np = np.array([window], dtype=np.int32)
            try:
                output_np = self._infer_once(input_np)
            except Exception as e:
                logger.error(f"[Triton] Generation failed: {e}")
                break
            if output_np is None or output_np.size == 0:
                break
            token_id = int(np.argmax(output_np[0, -1, :]))
            if token_id == self._eos_token_id:
                break
            hist.append(token_id)
        return hist

    def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **sampling_kwargs: Any,
    ) -> Iterator[str]:
        """Streaming generation via Triton's Infer API.

        Yields decoded token strings incrementally as they are generated,
        one per decode step. Uses the same single encode + sliding-window
        decode loop as ``generate()`` (no per-step string re-encoding);
        temporary decode buffers are flushed whenever the longest decoded
        segment looks complete enough, so multi-token pieces still yield
        promptly.

        Yields:
            Decoded text for each generated token, in order.
        """
        if self._client is None:
            raise ModelLoadError(
                self.model_name,
                "Triton not connected. Call load_model() first.",
            )
        input_ids = self._encode(prompt)
        if not input_ids:
            return
        hist = list(input_ids)
        emitted = 0  # number of generated ids already yielded
        for _ in range(max_tokens):
            window = hist[-self.max_context:] if self.max_context else hist
            input_np = np.array([window], dtype=np.int32)
            try:
                output_np = self._infer_once(input_np)
            except Exception as e:
                logger.error(f"[Triton] Stream generation failed: {e}")
                break
            if output_np is None or output_np.size == 0:
                break
            token_id = int(np.argmax(output_np[0, -1, :]))
            if token_id == self._eos_token_id:
                break
            hist.append(token_id)
            new_ids = hist[len(input_ids) + emitted:]
            if new_ids:
                yield self._decode(new_ids)
                emitted += len(new_ids)

    # ── Model config generator ───────────────────────────────────────

    @staticmethod
    def generate_config_pbtxt(
        model_name: str,
        platform: str = "onnx",
        max_batch_size: int = 32,
        input_name: str = "input_ids",
        input_dtype: str = "TYPE_INT32",
        input_dims: list[int] | None = None,
        output_name: str = "logits",
        output_dtype: str = "TYPE_FP32",
        output_dims: list[int] | None = None,
        dynamic_batching: bool = True,
        max_queue_delay_microseconds: int = 100,
        instance_count: int = 1,
        kind: str = "KIND_GPU",
        **extra_fields: Any,
    ) -> str:
        """Generate a Triton ``config.pbtxt`` from model metadata.

        Produces a complete model configuration file suitable for
        deployment in a Triton model repository.

        Args:
            model_name: Name of the model (directory name).
            platform: Triton platform string — ``"onnx"``, ``"tensorrt_plan"``,
                ``"pytorch"``, ``"tensorflow_savedmodel"``, etc.
            max_batch_size: Maximum batch size for dynamic batching.
            input_name: Name of the input tensor.
            input_dtype: Triton data type for the input.
            input_dims: Dimensions of the input tensor (excluding batch dim).
            output_name: Name of the output tensor.
            output_dtype: Triton data type for the output.
            output_dims: Dimensions of the output tensor (excluding batch dim).
            dynamic_batching: Enable Triton's dynamic batcher.
            max_queue_delay_microseconds: Maximum queue delay for the
                dynamic batcher.
            instance_count: Number of model instances.
            kind: Instance kind — ``"KIND_GPU"`` or ``"KIND_CPU"``.
            extra_fields: Additional fields to include in the config
                (e.g. ``optimization``, ``cc_model_filenames``).

        Returns:
            ``config.pbtxt`` content as a string.
        """
        input_dims = input_dims or [-1]
        output_dims = output_dims or [-1, -1]

        lines: list[str] = [
            f'name: "{model_name}"',
            f'platform: "{platform}"',
            f"max_batch_size: {max_batch_size}",
            "",
            "# ── Input ──",
            "input [",
            "  {",
            f'    name: "{input_name}"',
            f"    data_type: {input_dtype}",
            f"    dims: [{', '.join(str(d) for d in input_dims)}]",
            "  },",
            "]",
            "",
            "# ── Output ──",
            "output [",
            "  {",
            f'    name: "{output_name}"',
            f"    data_type: {output_dtype}",
            f"    dims: [{', '.join(str(d) for d in output_dims)}]",
            "  },",
            "]",
        ]

        if dynamic_batching:
            lines.extend([
                "",
                "# ── Dynamic batching ──",
                "dynamic_batching {",
                f"  max_queue_delay_microseconds: {max_queue_delay_microseconds}",
                "}",
            ])

        lines.extend([
            "",
            "# ── Instance group ──",
            "instance_group [",
            "  {",
            f"    count: {instance_count}",
            f"    kind: {kind}",
            "  },",
            "]",
        ])

        for key, value in extra_fields.items():
            if isinstance(value, str):
                lines.append(f'{key}: "{value}"')
            elif isinstance(value, bool):
                lines.append(f"{key}: {str(value).lower()}")
            elif isinstance(value, (int, float)):
                lines.append(f"{key}: {value}")
            elif isinstance(value, dict) or isinstance(value, list):
                lines.append(f"# {key}: {json.dumps(value)}")

        lines.append("")  # trailing newline
        return "\n".join(lines)

    # ── Shutdown ─────────────────────────────────────────────────────

    def shutdown(self) -> None:
        """Close the Triton client connection."""
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                # CQ-08: best-effort: failure is non-fatal (optional capability / telemetry / cleanup).
                pass
            self._client = None
            self._model_metadata = {}
            self._model_config = {}
            self._tokenizer = None
            logger.info(f"[Triton] Connection to {self.triton_url} closed")

    # ── Metadata classmethods ────────────────────────────────────────

    @classmethod
    def display_name(cls) -> str:
        return "Triton Inference Server"

    @classmethod
    def is_available(cls) -> bool:
        try:
            import tritonclient.http  # noqa: F401
            return True
        except ImportError:
            pass
        try:
            import tritonclient.grpc  # noqa: F401
            return True
        except ImportError:
            pass
        return False

    @classmethod
    def priority_for(cls, device_type: str) -> int:
        # Triton is ideal for GPU-backed production deployments with
        # multiple frameworks (TensorRT, ONNX, PyTorch).
        if device_type in ("cuda", "rocm"):
            return 12
        return 0

    # ── Internal helpers ─────────────────────────────────────────────

    def _probe_server_ready(self) -> bool:
        """Check that the Triton server is live and ready."""
        try:
            return bool(
                self._client.is_server_live()
                and self._client.is_server_ready()
            )
        except Exception:
            # CQ-08: intentional fail-open: return False on failure; callers treat False as 'not available/enabled'.
            return False

    def _probe_model_ready(self) -> bool:
        """Check that the model is loaded and ready on Triton."""
        try:
            return bool(
                self._client.is_model_ready(
                    self.model_name, self.model_version
                )
            )
        except Exception:
            # CQ-08: intentional fail-open: return False on failure; callers treat False as 'not available/enabled'.
            return False


__all__ = ["TritonNodeAdapter"]
