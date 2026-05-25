"""vLLM-based worker node for distributed inference.

Replaces the legacy Node class with a vLLM-backed node that uses
VLLMNodeAdapter instead of ModelPartitioner for per-layer execution.
Provides production-quality PagedAttention, continuous batching,
and FlashAttention for free from vLLM.
"""

from __future__ import annotations

import argparse
from loguru import logger

from distllm.core.vllm_backend import VLLMNodeAdapter

try:
    from distllm.communication.grpc_client import GRPCServer
except ImportError:
    GRPCServer = None
try:
    from distllm.communication.node_service import NodeService
except ImportError:
    NodeService = None


class VLLMWorkerNode:
    """Worker node that runs vLLM for its assigned layers.

    Starts a gRPC server and exposes ForwardPass, HealthCheck,
    and other RPCs using vLLM as the inference backend.

    Args:
        node_id: Unique node identifier.
        model_name: HuggingFace model name or path.
        start_layer: First layer index for pipeline parallelism.
        end_layer: Last layer index for pipeline parallelism.
        port: gRPC server port.
        vllm_config: vLLM engine configuration dict.
        device: Target device ("auto", "cuda", "cpu").
        trust_remote_code: Whether to trust HuggingFace remote code.
    """

    def __init__(
        self,
        node_id: str,
        model_name: str,
        start_layer: int = 0,
        end_layer: int = 0,
        total_layers: int = 0,
        port: int = 50051,
        vllm_config: dict | None = None,
        device: str = "auto",
        trust_remote_code: bool | None = None,
    ):
        self.node_id = node_id
        self.model_name = model_name
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.total_layers = total_layers
        self.port = port
        self._vllm_config = vllm_config or {}
        self._device = device
        self._trust_remote_code = trust_remote_code

        is_first = start_layer == 0
        is_last = end_layer == 0 or end_layer >= total_layers

        self._adapter = VLLMNodeAdapter(
            model_name=model_name,
            vllm_config=self._vllm_config,
            layer_start=start_layer if not is_first else None,
            layer_end=end_layer if not is_last else None,
            trust_remote_code=trust_remote_code,
        )
        self.server = None

    def load_model(self):
        """Initialize the vLLM engine and load model weights."""
        self._adapter.load_model()

    def start(
        self,
        use_tls: bool = True,
        cert_file: str | None = None,
        key_file: str | None = None,
        ca_cert: str | None = None,
    ) -> None:
        """Start the worker node gRPC server with vLLM backend.

        Args:
            use_tls: Enable TLS encryption.
            cert_file: Path to TLS certificate file.
            key_file: Path to TLS key file.
            ca_cert: Path to CA certificate for client verification.
        """
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
            f"[{self.node_id}] vLLM node started on port {self.port}"
            f" (layers {self.start_layer}-{self.end_layer} of {self.total_layers})"
        )

        try:
            self.server.wait_for_termination()
        except KeyboardInterrupt:
            logger.info(f"[{self.node_id}] Shutting down...")
            self.stop()

    def stop(self) -> None:
        """Stop the worker node and release vLLM resources."""
        if self.server:
            self.server.stop()
        self._adapter.shutdown()
        logger.info(f"[{self.node_id}] Stopped")


def main():
    parser = argparse.ArgumentParser(description="vLLM Worker Node")
    parser.add_argument("--node-id", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--start-layer", type=int, default=0)
    parser.add_argument("--end-layer", type=int, default=0)
    parser.add_argument("--total-layers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--insecure", action="store_true", help="Disable TLS for development")

    args = parser.parse_args()

    vllm_config = {
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "dtype": args.dtype,
        "max_num_seqs": args.max_num_seqs,
        "trust_remote_code": args.trust_remote_code,
    }

    node = VLLMWorkerNode(
        node_id=args.node_id,
        model_name=args.model,
        start_layer=args.start_layer,
        end_layer=args.end_layer,
        total_layers=args.total_layers,
        port=args.port,
        vllm_config=vllm_config,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
    )

    node.start(use_tls=not args.insecure)


if __name__ == "__main__":
    main()
