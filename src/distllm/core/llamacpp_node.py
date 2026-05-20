"""llama.cpp-based worker node for distributed inference.

Provides a lightweight alternative to VLLMWorkerNode using llama-cpp-python
GGUF models. Supports CPU, CUDA, AMD ROCm, and Apple Metal inference
without requiring PyTorch/vLLM's heavy dependency chain.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any
from loguru import logger

from distllm.communication.grpc_client import GRPCServer
from distllm.communication.node_service import NodeService
from distllm.core.llamacpp_backend import LlamacppNodeAdapter


class LlamacppWorkerNode:
    """Worker node that runs llama.cpp for its assigned layers.

    Starts a gRPC server and exposes ForwardPass, HealthCheck,
    and other RPCs using llama.cpp as the inference backend.

    Args:
        node_id: Unique node identifier.
        model_path: Path to GGUF model file.
        start_layer: First layer index (unused in single-node mode).
        end_layer: Last layer index (unused in single-node mode).
        port: gRPC server port.
        n_gpu_layers: GPU layers to offload (0 = CPU only).
        n_ctx: Context size.
        n_threads: CPU thread count (None = auto).
        n_batch: Batch size for prompt processing.
        seed: Random seed.
        verbose: Enable verbose logging.
        device: Target device ("auto", "cuda", "cpu", "metal", "rocm").
    """

    def __init__(
        self,
        node_id: str,
        model_path: str,
        start_layer: int = 0,
        end_layer: int = 0,
        total_layers: int = 0,
        port: int = 50051,
        n_gpu_layers: int = 0,
        n_ctx: int = 2048,
        n_threads: int | None = None,
        n_batch: int = 512,
        seed: int = 0,
        verbose: bool = False,
        device: str = "auto",
        **extra_adapter_kwargs: Any,
    ):
        self.node_id = node_id
        self.model_path = model_path
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.total_layers = total_layers
        self.port = port
        self._device = device

        is_first = start_layer == 0
        is_last = end_layer == 0 or end_layer >= total_layers
        self._adapter = LlamacppNodeAdapter(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_batch=n_batch,
            seed=seed,
            verbose=verbose,
            layer_start=start_layer if not is_first else None,
            layer_end=end_layer if not is_last else None,
            **extra_adapter_kwargs,
        )
        self.server = None

    def load_model(self):
        """Initialize the llama.cpp model and load GGUF weights."""
        self._adapter.load_model()

    def start(
        self,
        use_tls: bool = True,
        cert_file: str | None = None,
        key_file: str | None = None,
        ca_cert: str | None = None,
    ) -> None:
        """Start the worker node gRPC server with llama.cpp backend."""
        self.load_model()

        servicer = NodeService(
            node_id=self.node_id,
            forward_fn=self._adapter.forward,
        )

        self.server = GRPCServer(
            port=self.port,
            servicer=servicer,
            use_tls=use_tls,
            cert_file=cert_file,
            key_file=key_file,
            ca_cert=ca_cert,
        )
        self.server.start()

        logger.info(
            f"[{self.node_id}] llama.cpp node started on port {self.port}"
            f" (model: {self.model_path})"
        )

        try:
            self.server.wait_for_termination()
        except KeyboardInterrupt:
            logger.info(f"[{self.node_id}] Shutting down...")
            self.stop()

    def stop(self) -> None:
        """Stop the worker node and release llama.cpp resources."""
        if self.server:
            self.server.stop()
        self._adapter.shutdown()
        logger.info(f"[{self.node_id}] Stopped")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for running a llama.cpp worker node."""
    parser = argparse.ArgumentParser(description="DistLLM llama.cpp Worker Node")
    parser.add_argument("--node-id", required=True, help="Unique node identifier")
    parser.add_argument("--model-path", required=True, help="Path to GGUF model file")
    parser.add_argument("--port", type=int, default=50051, help="gRPC server port")
    parser.add_argument("--n-gpu-layers", type=int, default=0, help="GPU layers to offload (0 = CPU)")
    parser.add_argument("--n-ctx", type=int, default=2048, help="Context size")
    parser.add_argument("--n-threads", type=int, default=None, help="CPU thread count")
    parser.add_argument("--n-batch", type=int, default=512, help="Batch size")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "metal", "rocm"],
        help="Target device",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point for llama.cpp worker node."""
    args = parse_args()
    logger.add(sys.stderr, level="DEBUG" if args.verbose else "INFO")

    node = LlamacppWorkerNode(
        node_id=args.node_id,
        model_path=args.model_path,
        port=args.port,
        n_gpu_layers=args.n_gpu_layers,
        n_ctx=args.n_ctx,
        n_threads=args.n_threads,
        n_batch=args.n_batch,
        seed=args.seed,
        verbose=args.verbose,
        device=args.device,
    )
    node.start()


if __name__ == "__main__":
    main()