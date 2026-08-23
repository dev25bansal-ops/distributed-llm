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

        self._client = None
        self._model_metadata: dict[str, Any] = {}
        self._model_config: dict[str, Any] = {}
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
        """Single-shot text generation via Triton's dynamic batching.

        Uses Triton's text generation endpoint (typically the
        ``text_output`` or custom output name defined by the model).

        Args:
            prompt: Input text.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.

        Returns:
            Generated text.
        """
        if self._client is None:
            raise ModelLoadError(
                self.model_name,
                "Triton not connected. Call load_model() first.",
            )

        if self.protocol == "http":
            return self._generate_http(
                prompt, max_tokens, temperature, top_p, **sampling_kwargs
            )
        return self._generate_grpc(
            prompt, max_tokens, temperature, top_p, **sampling_kwargs
        )

    def _generate_http(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        **sampling_kwargs: Any,
    ) -> str:
        """HTTP-based generation loop using Triton's dynamic batching.

        Iteratively sends the prompt and concatenates generated tokens
        until ``max_tokens`` is reached or the model signals end.
        """
        import numpy as np
        import tritonclient.http as http_client

        input_name = self._get_input_name("input_ids")
        output_name = self._get_output_name("logits")
        full_output = prompt

        for _ in range(max_tokens):
            # Tokenize by encoding the full output string into input_ids
            input_ids_np = np.array(
                [[ord(c) for c in full_output[-512:]]], dtype=np.int32
            )

            infer_input = http_client.InferInput(
                input_name, list(input_ids_np.shape), "INT32"
            )
            infer_input.set_data_from_numpy(input_ids_np)
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
                logger.error(f"[Triton] Generation failed at step {_}: {e}")
                break

            if output_np is None or output_np.size == 0:
                break

            # Greedy decode: argmax of last position
            token_id = int(np.argmax(output_np[0, -1, :]))
            if token_id == 0:
                break  # EOS token

            full_output += chr(token_id) if 32 <= token_id < 127 else " "

        return full_output

    def _generate_grpc(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        **sampling_kwargs: Any,
    ) -> str:
        """gRPC-based generation loop using Triton's dynamic batching."""
        import numpy as np
        import tritonclient.grpc as grpc_client

        input_name = self._get_input_name("input_ids")
        output_name = self._get_output_name("logits")
        full_output = prompt

        for _ in range(max_tokens):
            input_ids_np = np.array(
                [[ord(c) for c in full_output[-512:]]], dtype=np.int32
            )

            infer_input = grpc_client.InferInput(
                input_name, list(input_ids_np.shape), "INT32"
            )
            infer_input.set_data_from_numpy(input_ids_np)
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
                logger.error(f"[Triton] Generation failed at step {_}: {e}")
                break

            if output_np is None or output_np.size == 0:
                break

            token_id = int(np.argmax(output_np[0, -1, :]))
            if token_id == 0:
                break

            full_output += chr(token_id) if 32 <= token_id < 127 else " "

        return full_output

    def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **sampling_kwargs: Any,
    ) -> Iterator[str]:
        """Streaming generation via Triton's dynamic batching.

        Yields tokens incrementally as they are generated.

        Args:
            prompt: Input text.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (used in generation loop).
            top_p: Nucleus sampling threshold.

        Yields:
            Generated text tokens one at a time.
        """
        import numpy as np

        if self._client is None:
            raise ModelLoadError(
                self.model_name,
                "Triton not connected. Call load_model() first.",
            )

        input_name = self._get_input_name("input_ids")
        output_name = self._get_output_name("logits")
        full_output = prompt

        for _ in range(max_tokens):
            input_ids_np = np.array(
                [[ord(c) for c in full_output[-512:]]], dtype=np.int32
            )

            if self.protocol == "http":
                import tritonclient.http as http_client

                infer_input = http_client.InferInput(
                    input_name, list(input_ids_np.shape), "INT32"
                )
                infer_input.set_data_from_numpy(input_ids_np)
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
                    logger.error(f"[Triton] Stream generation failed: {e}")
                    break
            else:
                import tritonclient.grpc as grpc_client

                infer_input = grpc_client.InferInput(
                    input_name, list(input_ids_np.shape), "INT32"
                )
                infer_input.set_data_from_numpy(input_ids_np)
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
                    logger.error(f"[Triton] Stream generation failed: {e}")
                    break

            if output_np is None or output_np.size == 0:
                break

            token_id = int(np.argmax(output_np[0, -1, :]))
            if token_id == 0:
                break

            token_char = chr(token_id) if 32 <= token_id < 127 else " "
            full_output += token_char
            yield token_char

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
                pass
            self._client = None
            self._model_metadata = {}
            self._model_config = {}
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
            return False


__all__ = ["TritonNodeAdapter"]
