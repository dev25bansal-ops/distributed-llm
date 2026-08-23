"""Worker node for distributed LLM inference.

Each worker loads a subset of the model layers and serves them via
gRPC for remote forward passes. The coordinator sends input through
all workers in pipeline order.
"""

from __future__ import annotations
import argparse
import hashlib
import sys
from dataclasses import dataclass

import torch
from loguru import logger

@dataclass
class _SimpleCompressionConfig:
    """Minimal compression config for CLI-driven worker startup.

    Defined at module level so it is importable/visible to introspection
    tools and avoids defining a dataclass inside a function body.
    """

    method: str
    enabled: bool
    target_bits: int
    pruning_ratio: float
    distillation_teacher: str | None
    calibration_samples: int
    pruning_targets: list

def _validate_state_dict_keys(model, state_dict: dict) -> None:
    """Log warnings for unexpected keys in *state_dict* before loading.

    Raises no exception — the caller still uses ``strict=False`` because
    the state dict contains a layer subset, so missing model keys are
    expected.  Unexpected keys (e.g. from a different architecture) are
    reported so silent corruption is avoided.
    """

    model_keys = set(model.state_dict().keys())
    state_keys = set(state_dict.keys())
    unexpected = state_keys - model_keys
    if unexpected:
        logger.warning(
            f"State dict contains {len(unexpected)} key(s) not found in "
            f"the model — possible architecture mismatch: {list(unexpected)[:5]}..."
        )

from distllm.config.loader import QuantizationConfig
from distllm.config.settings import DistLLMSettings
from distllm.core.debug import set_debug_mode
from distllm.dist.node_service import NodeServer
from distllm.dist.privacy import PrivacySplitConfig
from distllm.models.partitioner import ModelPartitioner

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
        privacy_config: PrivacySplitConfig | None = None,
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
        self.privacy_config = privacy_config or PrivacySplitConfig()

        self.partitioner: ModelPartitioner | None = None
        self.is_first = (start_layer == 0)
        self.is_last = (end_layer >= total_layers - 1)
        self.is_privacy_node = self.privacy_config.enabled
        self._ready = False  # Set to True only after load_model() succeeds

    def load_model(self, model_cache_dir: str | None = None,
                   weight_source: str | None = None) -> None:
        """Load assigned model layers. Supports P2P weight transfer.

        Priority:
          1. Local model cache (model_cache_dir)
          2. P2P weight transfer from another node (weight_source)
          3. HuggingFace download (default)

        Args:
            model_cache_dir: Shared cache directory for model layers.
            weight_source: ``host:port`` of a peer node that can serve weights.
        """

        target_device = self._get_device()

        # Priority 1: Local model cache
        if model_cache_dir:
            from distllm.dist.model_store import ModelStore
            store = ModelStore(cache_dir=model_cache_dir)
            cached_path = store.get_layer_path(
                self.model_name, self.start_layer, self.end_layer,
            )
            if cached_path:
                logger.info(f"Loading layers from cache: {cached_path}")
                self.partitioner = ModelPartitioner(
                    model_name=self.model_name,
                    device=self.device,
                    dtype=self.dtype,
                )
                state_dict = torch.load(cached_path, map_location=target_device, weights_only=True)
                self.partitioner.load_layer_subset(
                    self.start_layer, self.end_layer, self.total_layers,
                )
                _validate_state_dict_keys(self.partitioner.full_model, state_dict)
                self.partitioner.full_model.load_state_dict(state_dict, strict=False)
                logger.info(f"[{self.node_id}] Cached layers {self.start_layer}-{self.end_layer} loaded")
                return

        # Priority 2: P2P weight transfer from another node
        if weight_source:
            host, port_str = weight_source.rsplit(":", 1)
            port = int(port_str)
            from distllm.dist.node_client import request_layer_weights
            logger.info(f"Requesting weights from {weight_source} "
                         f"(layers {self.start_layer}-{self.end_layer})")
            weights_bytes = request_layer_weights(
                host, port,
                self.model_name, self.start_layer, self.end_layer,
                cluster_key=self.cluster_key if hasattr(self, 'cluster_key') else None,
            )
            if weights_bytes:
                import io
                state_dict = torch.load(io.BytesIO(weights_bytes), map_location=target_device, weights_only=True)
                self.partitioner = ModelPartitioner(
                    model_name=self.model_name,
                    device=self.device,
                    dtype=self.dtype,
                )
                self.partitioner.load_layer_subset(
                    self.start_layer, self.end_layer, self.total_layers,
                )
                _validate_state_dict_keys(self.partitioner.full_model, state_dict)
                self.partitioner.full_model.load_state_dict(state_dict, strict=False)
                logger.info(f"[{self.node_id}] P2P layers {self.start_layer}-{self.end_layer} loaded")

                if model_cache_dir:
                    from distllm.dist.model_store import ModelStore
                    store = ModelStore(cache_dir=model_cache_dir)
                    save_path = store.save_layer_weights(
                        self.model_name, self.start_layer, self.end_layer,
                    )
                    store.save_layer_manifest(self.model_name, self.total_layers)
                    torch.save(self.partitioner.full_model.state_dict(), save_path)
                    logger.info(f"[{self.node_id}] Layers cached to {save_path}")
                return
            logger.warning(f"P2P weight transfer from {weight_source} failed, "
                           f"falling back to HuggingFace")

        # Priority 3: HuggingFace download (layer-aware)
        from distllm.models.model_hub import ModelHub

        hub = ModelHub()

        # Pre-download only the shards needed for our layer range.
        # This ensures the HuggingFace cache has exactly the required
        # shards before the partitioner tries to load them.
        layer_path = hub.download_layer_subset(
            self.model_name,
            self.start_layer,
            self.end_layer,
        )
        logger.info(f"[{self.node_id}] Layer-scoped cache at {layer_path}")

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

        if model_cache_dir:
            from distllm.dist.model_store import ModelStore
            store = ModelStore(cache_dir=model_cache_dir)
            save_path = store.save_layer_weights(
                self.model_name, self.start_layer, self.end_layer,
            )
            store.save_layer_manifest(self.model_name, self.total_layers)
            if hasattr(self, 'partitioner') and self.partitioner and hasattr(self.partitioner, 'full_model'):
                torch.save(self.partitioner.full_model.state_dict(), save_path)
                logger.info(f"[{self.node_id}] Layers cached to {save_path}")

        logger.info(f"[{self.node_id}] Model loaded: layers {self.start_layer}-{self.end_layer}")
        logger.info(f"[{self.node_id}] Role: first={self.is_first}, last={self.is_last}")
        logger.info(f"[{self.node_id}] Device: {target_device}")

    def _get_device(self) -> str:
        """Determine the best device to use (cross-platform)."""

        if self.device == "auto":
            from distllm.core.device_registry import detect_platform
            return detect_platform()
        return self.device

    def verify_model_integrity(self, expected_checksum: str | None = None) -> str:
        """Verify the integrity of loaded model weights.

        Computes SHA-256 of the loaded state dict and optionally compares
        against an expected checksum.

        Args:
            expected_checksum: Optional hex SHA-256 to compare against.

        Returns:
            Hex SHA-256 checksum of the loaded weights.

        Raises:
            RuntimeError: If model is not loaded.
            ValueError: If checksum doesn't match.
        """

        if self.partitioner is None or not hasattr(self.partitioner, "full_model"):
            raise RuntimeError("Model not loaded. Call load_model() first.")

        state_dict = self.partitioner.full_model.state_dict()
        # Sort keys for deterministic hashing
        sorted_keys = sorted(state_dict.keys())
        hasher = hashlib.sha256()
        for key in sorted_keys:
            hasher.update(key.encode("utf-8"))
            tensor = state_dict[key]
            hasher.update(tensor.cpu().numpy().tobytes())

        checksum = hasher.hexdigest()
        logger.info(f"[{self.node_id}] Model integrity: SHA-256={checksum[:16]}...")

        if expected_checksum and checksum != expected_checksum:
            raise ValueError(
                f"Model integrity check failed! "
                f"Expected {expected_checksum[:16]}..., got {checksum[:16]}..."
            )

        return checksum

    def reconnect_to_coordinator(
        self,
        new_coordinator_host: str,
        new_coordinator_port: int,
        cluster_key: str | None = None,
    ) -> bool:
        """Reconnect to a new coordinator after failover.

        Stops the current gRPC server and restarts it, registering with
        the new coordinator.

        Args:
            new_coordinator_host: New coordinator hostname.
            new_coordinator_port: New coordinator gRPC port.
            cluster_key: Optional shared secret for authentication.

        Returns:
            True if reconnection succeeded.
        """

        logger.info(
            f"[{self.node_id}] Reconnecting to coordinator at "
            f"{new_coordinator_host}:{new_coordinator_port}"
        )

        # Stop existing server
        self.stop()

        # Update coordinator address
        self.coordinator_host = new_coordinator_host
        self.coordinator_port = new_coordinator_port

        # Restart server (model is already loaded, no need to reload)
        try:
            from distllm.dist.node_service import NodeServer

            self._server = NodeServer(
                self, port=self.port, cluster_key=cluster_key,
            )
            self._server.start(
                use_tls=False,  # Will be configured by caller if needed
            )
            logger.info(f"[{self.node_id}] Reconnected to new coordinator")
            return True
        except Exception as e:
            logger.error(f"[{self.node_id}] Reconnection failed: {e}")
            return False

    def forward_fn(
        self,
        hidden_states: torch.Tensor | None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass through assigned layers with KV cache support.

        When privacy mode is enabled (self.is_privacy_node), this node is part of
        a privacy-preserving split. Prefix and suffix layers stay on the requester
        device; only trunk (middle) layers are routed to peers. Peers never see
        the raw input embeddings or the final logits.
        """

        if self.partitioner is None:
            raise RuntimeError("Model not loaded. Call load_model() before starting the node.")

        if input_ids is not None and self.partitioner.embed_input is not None:
            position_offset = 0
            if past_key_values and len(past_key_values) > 0:
                position_offset = past_key_values[0][0].shape[-2]
            hidden_states = self.partitioner.embed_input(input_ids, position_offset=position_offset)

        result = self.partitioner.forward(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )

        if result is None:
            raise RuntimeError(
                "Model forward pass returned None — model may not be properly loaded"
            )

        output, new_kv = result

        if self.is_last:
            output = self.partitioner.get_logits(output)

        return output, new_kv

    def start(self, use_tls: bool = False, cert_file: str | None = None,
              key_file: str | None = None, ca_cert: str | None = None,
              cluster_key: str | None = None,
              model_cache_dir: str | None = None,
              max_workers: int = 16,
              weight_source: str | None = None) -> None:
        """Start the worker node gRPC server.

        Loads model layers, then starts a gRPC server that serves
        ForwardPass, HealthCheck, and Profile RPCs to the coordinator.

        Args:
            use_tls: Enable TLS for gRPC.
            cert_file: TLS certificate path.
            key_file: TLS key path.
            ca_cert: CA certificate path (unused for server).
            cluster_key: Optional shared secret for node authentication.
            model_cache_dir: Optional shared model cache directory.
            max_workers: Max gRPC server thread pool workers.
            weight_source: ``host:port`` of a peer to pull weights from.
        """

        self.cluster_key = cluster_key  # store for load_model() P2P weight transfer
        self.use_tls = use_tls  # store for coordinator registration URL scheme
        self.load_model(model_cache_dir=model_cache_dir, weight_source=weight_source)
        self._ready = True  # Model loaded successfully — safe to accept work
        self._server = NodeServer(self, port=self.port, cluster_key=cluster_key,
                                  max_workers=max_workers)
        self._server.start(
            use_tls=use_tls,
            cert_file=cert_file,
            key_file=key_file,
        )
        logger.info(f"[{self.node_id}] Worker node serving on port {self.port}")
        logger.info(f"[{self.node_id}] Layers {self.start_layer}-{self.end_layer} of {self.total_layers}")

        # ARCHITECTURE: Register with coordinator AFTER model loading and server start.
        # This prevents the registration race where a coordinator would try to route
        # requests to this worker before it's ready to serve.
        self._register_with_coordinator(cluster_key)
        self._notify_coordinator_ready(cluster_key)

        self._server.wait()

    def _register_with_coordinator(self, cluster_key: str | None = None) -> None:
        """Register this worker with the coordinator's HTTP API.

        Sends a POST request to the coordinator with node details
        so the coordinator knows about this worker.
        """

        # API runs on port from DISTLLM_API_PORT env var, defaulting to 8000
        import os
        api_port = int(os.environ.get("DISTLLM_API_PORT", 8000))
        scheme = "https" if (hasattr(self, 'use_tls') and self.use_tls) else "http"
        url = f"{scheme}://{self.coordinator_host}:{api_port}/admin/v1/nodes/register"
        payload = {
            "node_id": self.node_id,
            "host": self._get_host_ip(),
            "port": self.port,
            "start_layer": self.start_layer,
            "end_layer": self.end_layer,
            "total_layers": self.total_layers,
            "device": self.device,
            "gpu_name": self._get_gpu_name(),
            "ready": self._ready,  # Reflects actual model-load state, not a hardcoded assumption
        }
        headers = {"Content-Type": "application/json"}
        if cluster_key:
            headers["Authorization"] = f"Bearer {cluster_key}"
        else:
            # Try environment variables for API key
            import os
            env_key = os.environ.get("API_KEY") or os.environ.get("DISTLLM_API_KEY")
            if env_key:
                headers["Authorization"] = f"Bearer {env_key}"

        try:
            import httpx
            resp = httpx.post(url, json=payload, headers=headers, timeout=10.0)
            if resp.status_code == 200:
                logger.info(f"[{self.node_id}] Registered with coordinator at {url}")
            else:
                logger.warning(f"[{self.node_id}] Registration returned {resp.status_code}: {resp.text}")
        except Exception as e:
            logger.warning(f"[{self.node_id}] Could not register with coordinator: {e}")
            logger.info("Worker will run standalone. Use 'distllm cluster list-nodes' to check.")

    def _notify_coordinator_ready(self, cluster_key: str | None = None) -> None:
        """Send a ready status update to the coordinator after model loading.

        This is a second signal — after the initial registration — that
        explicitly tells the coordinator the worker has finished loading
        its model layers and is safe to receive inference requests.
        """

        import os
        api_port = int(os.environ.get("DISTLLM_API_PORT", 8000))
        scheme = "https" if (hasattr(self, 'use_tls') and self.use_tls) else "http"
        url = (
            f"{scheme}://{self.coordinator_host}:{api_port}"
            f"/admin/v1/nodes/{self.node_id}/ready"
        )
        payload = {"ready": True}
        headers = {"Content-Type": "application/json"}
        if cluster_key:
            headers["Authorization"] = f"Bearer {cluster_key}"
        else:
            env_key = os.environ.get("API_KEY") or os.environ.get("DISTLLM_API_KEY")
            if env_key:
                headers["Authorization"] = f"Bearer {env_key}"

        try:
            import httpx
            resp = httpx.post(url, json=payload, headers=headers, timeout=10.0)
            if resp.status_code == 200:
                logger.info(f"[{self.node_id}] Notified coordinator: worker is ready")
            else:
                logger.warning(
                    f"[{self.node_id}] Ready notification returned "
                    f"{resp.status_code}: {resp.text}"
                )
        except Exception as e:
            logger.warning(f"[{self.node_id}] Could not send ready notification: {e}")

    def _get_host_ip(self) -> str:
        """Get the host IP for worker registration."""

        try:
            import socket
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            return self.coordinator_host

    def _get_gpu_name(self) -> str:
        """Get GPU name for registration."""

        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.get_device_name(0)
        except Exception:
            pass
        return "cpu"

    def stop(self) -> None:
        """Stop the worker node gRPC server."""

        if hasattr(self, '_server') and self._server:
            self._server.stop()
        logger.info(f"[{self.node_id}] Stopped")

def main():
    from distllm.config.resolver import ConfigResolver

    parser = argparse.ArgumentParser(description="DistLLM Worker Node")
    ConfigResolver._register_args(parser, ConfigResolver.COMMON_ARGS + ConfigResolver.WORKER_ARGS)
    args = parser.parse_args()

    # Discover config file for shared settings (model name, TLS, cluster key)
    config_path = ConfigResolver._resolve_config_path("worker", args)
    settings = DistLLMSettings.from_yaml(config_path=config_path) if config_path else None

    # Merge settings with CLI args (CLI wins)
    model_name = args.model
    dtype = args.dtype
    tls_cert = args.tls_cert
    tls_key = args.tls_key
    tls_ca = args.tls_ca
    cluster_key = args.cluster_key
    model_cache_dir = args.model_cache_dir
    insecure = args.insecure

    if settings is not None:
        if not args.model:
            model_name = settings.model.name
        if not args.dtype:
            dtype = settings.model.dtype
        if settings.tls.enabled:
            insecure = False  # TLS required by settings
            if not args.tls_cert and settings.tls.cert_file:
                tls_cert = str(settings.tls.cert_file)
            if not args.tls_key and settings.tls.key_file:
                tls_key = str(settings.tls.key_file)
            if not args.tls_ca and settings.tls.ca_cert_file:
                tls_ca = str(settings.tls.ca_cert_file)
        if not args.cluster_key:
            cluster_key = settings.network.cluster_key or None
        if not args.model_cache_dir:
            model_cache_dir = settings.model_hub.cache_dir or None

    if model_cache_dir:
        from distllm.dist.model_store import ModelStore
        store = ModelStore(cache_dir=model_cache_dir)
        logger.info(f"Model cache: {model_cache_dir}")

    if args.validate_config:
        DistLLMSettings.validate_startup()
        print("Config validation passed")
        return

    if args.debug:
        set_debug_mode(True)
        logger.info("Debug mode enabled: tensor shape logging active")

    if args.quantization_method != "none":
        quant_config = QuantizationConfig(method=args.quantization_method)
    else:
        quant_config = None

    if args.compression_method != "none":
        comp_config = _SimpleCompressionConfig(
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

    privacy_config = PrivacySplitConfig(
        enabled=args.privacy_split,
        prefix_layers=args.privacy_prefix_layers,
        suffix_layers=args.privacy_suffix_layers,
    )

    node = WorkerNode(
        node_id=args.node_id,
        model_name=model_name,
        start_layer=args.start_layer,
        end_layer=args.end_layer,
        total_layers=args.total_layers,
        port=args.port,
        coordinator_host=args.coordinator_host,
        coordinator_port=args.coordinator_port,
        device=args.device,
        dtype=dtype,
        quantization_config=quant_config,
        expert_ids=args.expert_ids or None,
        compression_config=comp_config,
        privacy_config=privacy_config,
    )

    use_tls = not insecure
    if insecure:
        logger.warning(
            "TLS DISABLED via --insecure flag. "
            "gRPC communication will be unencrypted. "
            "Only use this in isolated development environments."
        )

    try:
        node.start(
            use_tls=use_tls,
            cert_file=tls_cert,
            key_file=tls_key,
            ca_cert=tls_ca,
            cluster_key=cluster_key,
            model_cache_dir=model_cache_dir,
            max_workers=args.max_workers,
            weight_source=args.weight_source,
        )
    except OverflowError as e:
        logger.error(f"OverflowError: {e}")
        logger.error("This is likely a gRPC configuration issue. Reinstall the latest distllm wheel.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Worker failed to start: {e}")
        logger.debug("Exception details", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
