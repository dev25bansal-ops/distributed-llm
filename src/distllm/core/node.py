"""Worker node for distributed LLM inference."""

import argparse

import torch
from loguru import logger
import uuid

from distllm.models.partitioner import ModelPartitioner
from distllm.communication.grpc import NodeService, GRPCServer
from distllm.config.settings import DistLLMSettings
from distllm.communication.grpc import set_debug_mode


class WorkerNode:
    """A worker node that runs a subset of model layers in the pipeline."""

    def __init__(
        self,
        node_id: str,
        model_name: str,
        start_layer: int,
        end_layer: int,
        total_layers: int,
        port: int,
        coordinator_host: str = "localhost",
        coordinator_port: int = 50050,
        device: str = "auto",
        dtype: str = "float16",
        quantization_config=None,
        expert_ids: list[int] | None = None,
        compression_config=None,
    ):
        self.node_id = node_id
        self.model_name = model_name
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.total_layers = total_layers
        self.port = port
        self.coordinator_host = coordinator_host
        self.coordinator_port = coordinator_port
        self.device = device
        self.dtype = dtype
        self.quantization_config = quantization_config
        self.expert_ids = expert_ids or []
        self.compression_config = compression_config

        self.partitioner: ModelPartitioner | None = None
        self.server: GRPCServer | None = None
        self.is_first = (start_layer == 0)
        self.is_last = (end_layer >= total_layers - 1)

    def load_model(self) -> None:
        """Load assigned model layers."""
        target_device = self._get_device()

        self.partitioner = ModelPartitioner(
            model_name=self.model_name,
            device=self.device,
            dtype=self.dtype,
            quantization_config=self.quantization_config,
            compression_config=self.compression_config,
        )

        self.partitioner.load_layer_subset(
            self.start_layer, self.end_layer, self.total_layers, device=target_device
        )

        logger.info(f"[{self.node_id}] Model loaded: layers {self.start_layer}-{self.end_layer}")
        logger.info(f"[{self.node_id}] Role: first={self.is_first}, last={self.is_last}")
        logger.info(f"[{self.node_id}] Device: {target_device}")

    def _get_device(self) -> str:
        """Determine the best device to use."""
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    def forward_fn(
        self,
        hidden_states: torch.Tensor | None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass through assigned layers with KV cache support.

        For first node: if input_ids provided, embed them first.
        For middle nodes: process hidden states directly.
        For last node: compute logits after layers.
        """
        # First node: embed token IDs if provided
        if input_ids is not None and self.partitioner is not None and self.partitioner.embed_input is not None:
            # Determine position offset from KV cache length
            position_offset = 0
            if past_key_values and len(past_key_values) > 0:
                position_offset = past_key_values[0][0].shape[-2]
            hidden_states = self.partitioner.embed_input(input_ids, position_offset=position_offset)

        assert self.partitioner is not None, "Model not loaded"
        output, new_kv = self.partitioner.forward(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )

        # If this is the last node, compute logits
        if self.is_last:
            output = self.partitioner.get_logits(output)

        return output, new_kv

    def start(self, use_tls: bool = True, cert_file: str | None = None,
              key_file: str | None = None, ca_cert: str | None = None) -> None:
        """Start the worker node gRPC server.

        Args:
            use_tls: Enable TLS encryption. Use --insecure to disable in dev.
            cert_file: Path to TLS certificate file.
            key_file: Path to TLS key file.
            ca_cert: Path to CA certificate for client verification.
        """
        self.load_model()

        servicer = NodeService(
            node_id=self.node_id,
            forward_fn=self.forward_fn,
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

        logger.info(f"[{self.node_id}] Worker node started on port {self.port}")
        logger.info(f"[{self.node_id}] Layers {self.start_layer}-{self.end_layer} of {self.total_layers}")

        try:
            self.server.wait_for_termination()
        except KeyboardInterrupt:
            logger.info(f"[{self.node_id}] Shutting down...")
            self.stop()

    def stop(self) -> None:
        """Stop the worker node."""
        if self.server:
            self.server.stop()
        logger.info(f"[{self.node_id}] Stopped")


def main():
    parser = argparse.ArgumentParser(description="Distributed LLM Worker Node")
    parser.add_argument("--node-id", type=str, required=True, help="Unique node identifier")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model name or path")
    parser.add_argument("--start-layer", type=int, required=True, help="First layer to run")
    parser.add_argument("--end-layer", type=int, required=True, help="Last layer to run")
    parser.add_argument("--total-layers", type=int, required=True, help="Total layers in model")
    parser.add_argument("--port", type=int, default=50051, help="gRPC port")
    parser.add_argument("--coordinator-host", type=str, default="localhost")
    parser.add_argument("--coordinator-port", type=int, default=50050)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32", "bfloat16"])
    parser.add_argument("--quantization-method", type=str, default="none", choices=["none", "bnb_4bit", "bnb_8bit"], help="Quantization method")
    parser.add_argument("--expert-ids", type=int, nargs="*", default=[], help="Expert IDs this node hosts for MoE inference")
    parser.add_argument("--compression-method", type=str, default="none", choices=["none", "ptq_int8", "ptq_int4", "pruning_structured", "distillation", "auto"], help="Compression method")
    parser.add_argument("--pruning-ratio", type=float, default=0.0, help="Fraction of weights to prune (0.0-1.0)")
    parser.add_argument("--distillation-teacher", type=str, default=None, help="Teacher model for knowledge distillation")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with tensor shape logging")
    parser.add_argument("--validate-config", action="store_true", help="Validate configuration at startup and exit")
    parser.add_argument("--insecure", action="store_true", help="Disable TLS for gRPC (development only)")
    parser.add_argument("--tls-cert", type=str, default=None, help="Path to TLS certificate file")
    parser.add_argument("--tls-key", type=str, default=None, help="Path to TLS key file")
    parser.add_argument("--tls-ca", type=str, default=None, help="Path to TLS CA certificate file")

    args = parser.parse_args()

    # Optional: validate config and exit
    if args.validate_config:
        DistLLMSettings.validate_startup()
        print("✅ Config validation passed")
        return

    if args.debug:
        set_debug_mode(True)
        logger.info("Debug mode enabled: tensor shape logging active")

    if args.quantization_method != "none":
        from distllm.config.loader import QuantizationConfig
        quant_config = QuantizationConfig(method=args.quantization_method)
    else:
        quant_config = None

    if args.compression_method != "none":
        from dataclasses import dataclass
        @dataclass
        class SimpleCompressionConfig:
            method: str
            enabled: bool
            target_bits: int
            pruning_ratio: float
            distillation_teacher: str | None
            calibration_samples: int
            pruning_targets: list
        comp_config = SimpleCompressionConfig(
            method=args.compression_method,
            enabled=True,
            target_bits=8,
            pruning_ratio=args.pruning_ratio,
            distillation_teacher=args.distillation_teacher,
            calibration_samples=128,
            pruning_targets=["q_proj", "v_proj"],
        )
    else:
        comp_config = None

    node = WorkerNode(
        node_id=args.node_id,
        model_name=args.model,
        start_layer=args.start_layer,
        end_layer=args.end_layer,
        total_layers=args.total_layers,
        port=args.port,
        coordinator_host=args.coordinator_host,
        coordinator_port=args.coordinator_port,
        device=args.device,
        dtype=args.dtype,
        quantization_config=quant_config,
        expert_ids=args.expert_ids or None,
        compression_config=comp_config,
    )

    use_tls = not args.insecure
    if args.insecure:
        logger.warning(
            "TLS DISABLED via --insecure flag. "
            "gRPC communication will be unencrypted. "
            "Only use this in isolated development environments."
        )

    node.start(
        use_tls=use_tls,
        cert_file=args.tls_cert,
        key_file=args.tls_key,
        ca_cert=args.tls_ca,
    )


if __name__ == "__main__":
    main()
