import torch
from loguru import logger

from distllm.communication.grpc import is_debug_mode
from distllm.core.batch_scheduler import ScheduledBatch
from transformers import AutoTokenizer
from distllm.config.loader import NodeRole


class OrchestratorService:
    """Pipeline orchestration: node topology, forward execution, routing."""

    def __init__(self, pipeline, resource_mgr, cache_mgr, container,
                 node_registrar, model_name, trust_remote_code,
                 spec_decoder=None, async_pipeline=None):
        self._pipeline = pipeline
        self._resource_mgr = resource_mgr
        self._cache_mgr = cache_mgr
        self._container = container
        self._node_registrar = node_registrar
        self._model_name = model_name
        self._trust_remote_code = trust_remote_code
        self._spec_decoder = spec_decoder
        self._async_pipeline = async_pipeline
        self._pipeline_schedule_type = "sequential"
        self._hybrid_parallel_executor = None
        self._latency_tracker = None
        self._rebalancer = None
        self._tokenizer = None
        self.nodes_info = {}

    def set_tokenizer(self, tokenizer):
        self._tokenizer = tokenizer

    def set_hybrid_executor(self, executor):
        self._hybrid_parallel_executor = executor

    def set_spec_decoder(self, decoder):
        self._spec_decoder = decoder

    # -- Pipeline properties --

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
    def prefill_nodes(self) -> dict:
        return self._pipeline.prefill_nodes

    @prefill_nodes.setter
    def prefill_nodes(self, value: dict):
        self._pipeline.prefill_nodes = value

    @property
    def decode_nodes(self) -> dict:
        return self._pipeline.decode_nodes

    @decode_nodes.setter
    def decode_nodes(self, value: dict):
        self._pipeline.decode_nodes = value

    @property
    def prefix_cache(self):
        return self._cache_mgr.prefix_cache if self._cache_mgr else None

    @prefix_cache.setter
    def prefix_cache(self, value):
        if self._cache_mgr:
            self._cache_mgr.prefix_cache = value

    # -- Pipeline scheduling --

    def init_pipeline_schedule(self, pipeline_schedule_config=None):
        self._async_pipeline = None
        self._pipeline_schedule_type = "sequential"
        if pipeline_schedule_config:
            schedule_type = getattr(pipeline_schedule_config, "schedule", "sequential")
            if schedule_type in ("1f1b", "interleaved"):
                from distllm.core.async_pipeline import AsyncPipelineEngine, AsyncPipelineConfig, ScheduleType
                async_config = AsyncPipelineConfig(
                    schedule=ScheduleType.ONE_F_ONE_B if schedule_type == "1f1b" else ScheduleType.INTERLEAVED,
                    num_micro_batches=getattr(pipeline_schedule_config, "num_micro_batches", 4),
                    num_stages=getattr(pipeline_schedule_config, "num_stages", 1),
                    overlap_allreduce=getattr(pipeline_schedule_config, "overlap_allreduce", True),
                    prefetch_next_batch=getattr(pipeline_schedule_config, "prefetch_next_batch", True),
                )
                self._async_pipeline = AsyncPipelineEngine(config=async_config)
                self._pipeline_schedule_type = schedule_type

    # -- Node registration --

    def auto_setup(self, nodes_config: list[dict]):
        model_info, total_layers = self._node_registrar.auto_setup(nodes_config)
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name, trust_remote_code=self._trust_remote_code)
        return model_info, total_layers

    def manual_register(self, node_id: str, host: str, port: int,
                        start_layer: int, end_layer: int,
                        total_layers: int | None = None,
                        role: NodeRole = NodeRole.AUTO,
                        expert_ids: list[int] | None = None,
                        cluster_id: str = "default"):
        self._node_registrar.manual_register(
            node_id, host, port, start_layer, end_layer,
            total_layers=total_layers, role=role,
            expert_ids=expert_ids, cluster_id=cluster_id,
        )
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name, trust_remote_code=self._trust_remote_code)

    def validate_layer_assignment(self, node_id: str, start_layer: int, end_layer: int):
        self._pipeline.validate_layer_assignment(node_id, start_layer, end_layer)

    def register_expert_on_node(self, node_id: str, expert_ids: list[int], layer_idx: int = 0):
        if not hasattr(self, '_expert_registry') or self._expert_registry is None:
            return
        for eid in expert_ids:
            self._expert_registry.register_expert(eid, node_id, layer_idx)
        logger.info(f"Registered experts {expert_ids} on {node_id}")

    # -- Pipeline forward passes --

    def run_distributed_pipeline_batch(self, batch: ScheduledBatch) -> None:
        if self._pipeline:
            self._pipeline.run_pipeline_batch(batch)

    def run_async_pipeline_batch(self, batch: ScheduledBatch,
                                 spec_decoder, continuous_trainer,
                                 scheduler, tokenizer,
                                 batch_kv_caches, batch_kv_caches_lock) -> None:
        if tokenizer is None:
            raise ValueError("Tokenizer not loaded")

        input_tensors = []
        for seq_idx, seq in enumerate(batch.sequences):
            if batch.is_prefill[seq_idx]:
                start = seq.prefix_match_len
                tokens = seq.prompt_tokens[start:]
            else:
                tokens = [seq.decode_input_token]
            input_tensors.append(torch.tensor([tokens], dtype=torch.long))

        max_len = max(t.shape[1] for t in input_tensors)
        padded_tensors = []
        for t in input_tensors:
            if t.shape[1] < max_len:
                padding = torch.zeros((1, max_len - t.shape[1]), dtype=torch.long)
                t = torch.cat([t, padding], dim=1)
            padded_tensors.append(t)

        batch_input = torch.cat(padded_tensors, dim=0)

        use_spec = (
            spec_decoder is not None
            and spec_decoder.is_enabled
            and hasattr(self, '_draft_model') and self._draft_model is not None
            and not all(batch.is_prefill)
        )

        draft_tokens_list = None
        if use_spec:
            draft_tokens_list, _ = spec_decoder.generate_batch_draft_tokens(
                self._draft_model, input_tensors
            )

        def stage_forward(micro_batch: torch.Tensor) -> torch.Tensor:
            micro_kv_caches = self._pipeline.create_node_kv_caches()
            if self._pipeline.enable_overlap:
                logits = self._pipeline.run_pipeline_overlap(
                    micro_batch, micro_kv_caches, request_id="async_micro"
                )
            else:
                logits = self._pipeline.run_pipeline(
                    micro_batch, micro_kv_caches, request_id="async_micro"
                )
            return logits

        if self._async_pipeline and not self._async_pipeline._stages:
            from distllm.core.async_pipeline import AsyncPipelineStage
            stage = AsyncPipelineStage(
                stage_id=0,
                forward_fn=stage_forward,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            self._async_pipeline.add_stage(stage)

        logits = self._async_pipeline.forward(batch_input)

        if use_spec and draft_tokens_list is not None:
            target_logits_list = [logits[i:i+1] for i in range(len(batch.sequences))]
            results = spec_decoder.verify_batch(
                draft_tokens_list=draft_tokens_list,
                target_logits_list=target_logits_list,
                tokenizer=tokenizer,
            )
            if continuous_trainer is not None:
                for idx, (_, accepted, _) in enumerate(results):
                    if accepted and idx < len(draft_tokens_list):
                        dt = draft_tokens_list[idx]
                        draft_ids = dt.tolist() if hasattr(dt, 'tolist') else list(dt) if dt else []
                        if draft_ids:
                            continuous_trainer.record(draft_ids, list(accepted))
            for i, seq in enumerate(batch.sequences):
                next_token = results[i][2]
                seq._async_next_token = torch.tensor([next_token], dtype=torch.long)
        else:
            for i, seq in enumerate(batch.sequences):
                seq_logits = logits[i:i+1, -1, :]
                if seq.constraint is not None:
                    mask = seq.constraint.get_logits_mask(seq_logits.shape[-1], tokenizer)
                    seq_logits = seq_logits.masked_fill(~mask, float('-inf'))
                token = seq.sample(seq_logits, temperature=seq.temperature, top_p=seq.top_p, top_k=seq.top_k)
                if not hasattr(seq, '_async_next_token'):
                    seq._async_next_token = token
                else:
                    seq._async_next_token = token

        next_tokens_list = [seq._async_next_token for seq in batch.sequences]
        next_tokens_tensor = torch.stack(next_tokens_list).squeeze(-1)
        with batch_kv_caches_lock:
            kv_copy = dict(batch_kv_caches)
        decoded = [
            tokenizer.decode([int(next_tokens_tensor[i])])
            if batch.sequences[i].constraint is not None else None
            for i in range(len(batch.sequences))
        ]
        scheduler.step(batch, next_tokens_tensor, kv_caches=kv_copy, decoded_tokens=decoded)

        if is_debug_mode():
            logger.debug(f"Async pipeline stats: {self._async_pipeline.summary()}")

    def unregister_node(self, node_id: str):
        self._pipeline.unregister_node(node_id)

    def create_node_kv_caches(self):
        return self._pipeline.create_node_kv_caches()

    # -- SLA helpers --

    def get_nodes_within_sla(self, max_latency_ms: float) -> set[str]:
        if self._latency_tracker is None:
            return set(self._pipeline.nodes.keys())
        sla_nodes = set()
        for node_id in self._pipeline.nodes:
            avg = self._latency_tracker.get_avg(node_id)
            if avg is None or avg < max_latency_ms:
                sla_nodes.add(node_id)
        return sla_nodes

    def set_latency_tracker(self, tracker):
        self._latency_tracker = tracker

    def set_rebalancer(self, rebalancer):
        self._rebalancer = rebalancer

    def shutdown(self):
        self._pipeline.shutdown()
