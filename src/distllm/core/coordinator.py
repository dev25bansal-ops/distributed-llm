"""Coordinator for distributed LLM inference across multiple devices.

Manages a cluster of worker nodes, orchestrates pipeline-parallel
inference across them, and provides OpenAI-compatible API endpoints.
"""

import argparse
import time
import threading
from typing import Any, Callable

from loguru import logger
import torch
from transformers import AutoTokenizer

from distllm.config.settings import NodeRole, DistLLMSettings
from distllm.core.debug import set_debug_mode
from distllm.core.resource_manager import ResourceManager
from distllm.core.cache_manager import CacheManager
from distllm.core.token_generator import TokenGenerator
from distllm.core.batch_scheduler import BatchScheduler
from distllm.core.request_replay import get_replay_buffer, RequestReplayBuffer, DeterministicMode
from distllm.dist.pipeline import PipelineOrchestrator
from distllm.dist.node_registrar import NodeRegistrar
from distllm.dist.recovery import NodeRecoveryManager
from distllm.dist.straggler import StragglerDetector
from distllm.dist.latency import LatencyTracker
from distllm.models.partitioner import ModelPartitioner, get_model_info
from distllm.security import hf_revision


class CoordinatorConfig:
    """Configuration for the distributed coordinator."""

    def __init__(
        self,
        model_name: str = "",
        port: int = 50050,
        dtype: str = "float16",
        trust_remote_code: bool | None = None,
        max_batch_size: int = 4,
        max_tokens_per_batch: int = 1024,
        pipeline_timeout: float = 30.0,
        cluster_key: str | None = None,
    ):
        self.model_name = model_name
        self.port = port
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code
        self.metrics_exporter = None
        self.discovery_mode = None
        self.max_batch_size = max_batch_size
        self.max_tokens_per_batch = max_tokens_per_batch
        self.pipeline_timeout = pipeline_timeout
        self.cluster_key = cluster_key
        self.prefix_cache_enabled = False
        self.prefix_cache_max_entries = 256
        self.prefix_cache_min_prefix_len = 4
        self.radix_tree_cache_enabled = False
        self.chunked_prefill_enabled = False
        self.chunked_prefill_chunk_size = 512


class Coordinator:
    """Orchestrates distributed inference across multiple worker nodes.

    Splits model layers across connected devices and runs pipeline-parallel
    inference. Supports dynamic node registration, failure recovery,
    straggler detection, and latency tracking.
    """

    def __init__(self, config: CoordinatorConfig | None = None):
        self.config = config or CoordinatorConfig(model_name="")
        self.model_name = self.config.model_name
        self.model_revision = hf_revision()
        self.port = self.config.port
        self.dtype = self.config.dtype
        self.trust_remote_code = self.config.trust_remote_code
        self.total_layers = 0

        self._resource_mgr = ResourceManager()
        self._cache_mgr = CacheManager()
        self._pipeline = PipelineOrchestrator(
            resource_mgr=self._resource_mgr,
            pipeline_timeout=self.config.pipeline_timeout,
        )
        self._node_registrar = NodeRegistrar(
            pipeline=self._pipeline,
            model_name=self.model_name,
            trust_remote_code=self.trust_remote_code,
        )
        self._token_gen = TokenGenerator()
        self._batch_scheduler = BatchScheduler(
            max_batch_size=self.config.max_batch_size,
            max_tokens_per_batch=self.config.max_tokens_per_batch,
        )
        self._latency_tracker = LatencyTracker()
        self._straggler_detector = StragglerDetector()
        self._recovery_manager = NodeRecoveryManager()

        self._pipeline.set_latency_tracker(self._latency_tracker)
        self._pipeline._latency_tracker = self._latency_tracker

        self.tokenizer: AutoTokenizer | None = None
        self.model_info: dict | None = None
        self.local_partitioner: ModelPartitioner | None = None
        self._replay_buffer: RequestReplayBuffer = get_replay_buffer(max_requests=100)
        self._deterministic_mode = DeterministicMode(seed=42, enabled=False)
        self._running = threading.Event()
        self._health_check_interval_s: float = 10.0
        self._health_thread: threading.Thread | None = None
        self._health_event = threading.Event()

        logger.info(f"Coordinator initialized for model: {self.model_name}")

    @property
    def nodes(self) -> dict:
        return self._pipeline.nodes

    @nodes.setter
    def nodes(self, value: dict):
        self._pipeline.nodes = value

    @property
    def node_order(self) -> list[str]:
        return self._pipeline.node_order

    @node_order.setter
    def node_order(self, value: list[str]):
        self._pipeline.node_order = value

    @property
    def scheduler(self) -> BatchScheduler | None:
        return self._batch_scheduler

    def auto_setup(self, nodes_config: list[dict]) -> None:
        model_info, total_layers = self._node_registrar.auto_setup(nodes_config)
        self.model_info = model_info
        self.total_layers = total_layers
        self._pipeline.total_layers = total_layers
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
            revision=self.model_revision,
        )
        logger.info(f"Auto-setup complete: {len(nodes_config)} nodes, {total_layers} layers")

    def manual_register(
        self,
        node_id: str,
        host: str,
        port: int,
        start_layer: int,
        end_layer: int,
        total_layers: int | None = None,
        role: NodeRole = NodeRole.AUTO,
        expert_ids: list[int] | None = None,
        cluster_id: str = "default",
        cluster_key: str | None = None,
    ) -> None:
        self._node_registrar.manual_register(
            node_id, host, port, start_layer, end_layer,
            total_layers=total_layers, role=role,
            expert_ids=expert_ids, cluster_id=cluster_id,
            cluster_key=cluster_key or self.config.cluster_key,
        )
        if total_layers:
            self._pipeline.total_layers = total_layers
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
                revision=self.model_revision,
            )
        if self.model_info is None:
            self.model_info = get_model_info(self.model_name, self.trust_remote_code)
            if total_layers is None:
                self.total_layers = self.model_info["num_layers"]
                self._pipeline.total_layers = self.total_layers

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        request_id: str | None = None,
        user_id: str = "default",
        speculative_config: dict | None = None,
    ) -> str:
        if self.local_partitioner is not None:
            return self._generate_local(prompt, max_new_tokens, temperature, top_p, top_k)
        return self._generate_distributed(prompt, max_new_tokens, temperature, top_p, top_k)

    def _generate_local(
        self, prompt: str, max_new_tokens: int,
        temperature: float, top_p: float, top_k: int,
    ) -> str:
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        device = next(self.local_partitioner.full_model.parameters()).device
        input_ids = input_ids.to(device)
        generated = input_ids

        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = self.local_partitioner.full_model(generated)
                logits = outputs.logits[:, -1, :]
                next_token = self._token_gen.sample(
                    logits, temperature=temperature, top_p=top_p, top_k=top_k
                )[0]
                if next_token.dim() == 0:
                    next_token = next_token.unsqueeze(0)
                if next_token.dim() == 1:
                    next_token = next_token.unsqueeze(-1)
                generated = torch.cat([generated, next_token], dim=-1)
                if next_token.item() == self.tokenizer.eos_token_id:
                    break

        return self.tokenizer.decode(generated[0, input_ids.shape[1]:], skip_special_tokens=True)

    def _generate_distributed(
        self, prompt: str, max_new_tokens: int,
        temperature: float, top_p: float, top_k: int,
    ) -> str:
        if not self.node_order:
            raise RuntimeError("No nodes registered in the pipeline")

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        generated_ids = input_ids.clone()
        node_kv_caches = self._pipeline.create_node_kv_caches()

        with torch.no_grad():
            for step in range(max_new_tokens):
                step_input = generated_ids if step == 0 else generated_ids[:, -1:]
                logits = self._pipeline.run_pipeline(
                    step_input, node_kv_caches, request_id=f"req_{step}"
                )
                logits_slice = logits[:, -1, :]
                next_token = self._token_gen.sample(
                    logits_slice, temperature=temperature, top_p=top_p, top_k=top_k
                )[0]
                if next_token.dim() == 0:
                    next_token = next_token.unsqueeze(0)
                if next_token.dim() == 1:
                    next_token = next_token.unsqueeze(-1)
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                if next_token.item() == self.tokenizer.eos_token_id:
                    break

        return self.tokenizer.decode(
            generated_ids[0, input_ids.shape[1]:], skip_special_tokens=True
        )

    def health_check(self) -> dict:
        nodes_status = {}
        for node_id, node in self.nodes.items():
            try:
                nodes_status[node_id] = {
                    "healthy": getattr(node, 'healthy', False),
                    "start_layer": node.start_layer,
                    "end_layer": node.end_layer,
                }
            except Exception:
                nodes_status[node_id] = {"healthy": False}
        return {
            "status": "ok" if self.nodes else "no_nodes",
            "num_nodes": len(self.nodes),
            "total_layers": self._pipeline.total_layers,
            "nodes": nodes_status,
        }

    def load_local_model(self) -> None:
        self.local_partitioner = ModelPartitioner(
            model_name=self.model_name,
            dtype=self.dtype,
            trust_remote_code=self.trust_remote_code,
        )
        self.local_partitioner.load_full_model()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
            revision=self.model_revision,
        )
        self.model_info = get_model_info(self.model_name, self.trust_remote_code)
        self.total_layers = self.model_info["num_layers"]
        logger.info(f"Local model loaded: {self.model_name}")

    def start(self, blocking: bool = True, on_stop: Callable | None = None,
              health_check_interval_s: float = 10.0) -> None:
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
                revision=self.model_revision,
            )
        self._health_check_interval_s = health_check_interval_s
        self._running.set()
        self._health_event.clear()
        self._health_thread = threading.Thread(
            target=self._health_probe_loop,
            daemon=True,
            name="health-probe",
        )
        self._health_thread.start()
        logger.info(f"Coordinator started on port {self.port} "
                     f"(health check every {health_check_interval_s}s)")
        if blocking:
            try:
                self._running.wait()
            except KeyboardInterrupt:
                logger.info("Coordinator shutting down...")
                self.stop()
        else:
            def _wait_and_callback():
                try:
                    self._running.wait()
                except KeyboardInterrupt:
                    pass
                finally:
                    if on_stop:
                        on_stop()
            threading.Thread(target=_wait_and_callback, daemon=True).start()

    def stop(self) -> None:
        logger.info("Initiating graceful shutdown...")
        self._running.clear()
        self._health_event.set()
        if self._health_thread and self._health_thread.is_alive():
            self._health_thread.join(timeout=3.0)
        for node_id, node in list(self.nodes.items()):
            node.close()
        self._pipeline.shutdown()
        logger.info("Graceful shutdown complete")

    def _health_probe_loop(self) -> None:
        """Background loop: periodically pings all registered nodes.

        Marks unresponsive nodes as unhealthy and triggers recovery.
        Runs every self._health_check_interval_s seconds.
        """
        while not self._health_event.is_set():
            self._health_event.wait(self._health_check_interval_s)
            if self._health_event.is_set():
                break
            if not self._running.is_set():
                break
            for node_id, node in list(self.nodes.items()):
                if node.client is None:
                    continue
                alive = node.health_check()
                if not alive and node.healthy:
                    logger.warning(
                        f"Node {node_id} at {node.host}:{node.port} "
                        f"failed health check — marking unhealthy"
                    )
                    node.healthy = False
                    self._resource_mgr.record_failure(node_id)
                    if self._recovery_manager is not None:
                        self._recovery_manager.on_node_failure(node_id)
                elif alive and not node.healthy:
                    logger.info(f"Node {node_id} recovered — marking healthy")
                    node.healthy = True
                    self._recovery_manager.mark_alive(node_id)

    def set_deterministic_mode(self, enabled: bool = True, seed: int = 42) -> None:
        if enabled:
            self._deterministic_mode.enable(seed)
        else:
            self._deterministic_mode.disable()

    def get_recent_requests(self, n: int = 10) -> list[Any]:
        return self._replay_buffer.list_recent(n)


def main():
    parser = argparse.ArgumentParser(description="Distributed LLM Coordinator")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--port", type=int, default=50050)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32", "bfloat16"])
    parser.add_argument("--nodes", type=str, nargs="+", help="host:port:start:end per node")
    parser.add_argument("--total-layers", type=int, help="Total layers in model")
    parser.add_argument("--local", action="store_true", help="Run full model locally")
    parser.add_argument("--chat", action="store_true", help="Start interactive chat mode")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--cluster-key", type=str, default=None, help="Shared cluster authentication key")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--validate-config", action="store_true", help="Validate configuration")

    args = parser.parse_args()

    if args.validate_config:
        DistLLMSettings.validate_startup()
        print("Config validation passed")
        return

    if args.debug:
        set_debug_mode(True)

    config = CoordinatorConfig(
        model_name=args.model,
        port=args.port,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code or None,
    )
    coordinator = Coordinator(config=config)

    if args.local:
        coordinator.load_local_model()
        if args.chat:
            print(f"Model loaded: {args.model}")
            while True:
                prompt = input("\nPrompt (or 'quit' to exit): ")
                if prompt.lower() in ('quit', 'exit'):
                    break
                result = coordinator.generate(prompt, max_new_tokens=128)
                print(f"\nResult: {result}")
        else:
            coordinator.start()
    else:
        if args.nodes:
            for i, node_str in enumerate(args.nodes):
                parts = node_str.split(":")
                coordinator.manual_register(
                    node_id=f"node_{i}",
                    host=parts[0],
                    port=int(parts[1]),
                    start_layer=int(parts[2]),
                    end_layer=int(parts[3]),
                    total_layers=args.total_layers,
                )
        coordinator.start()


if __name__ == "__main__":
    main()
