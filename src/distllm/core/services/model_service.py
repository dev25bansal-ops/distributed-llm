from typing import Any
import gc
import torch
from loguru import logger

from distllm.config.settings import MultiModelSettings, MoESettings
from distllm.core.speculative_decoder import SpeculativeDecoder
from distllm.core.coordinator_multi_model import MultiModelManager
from distllm.core.model_registry import ModelRegistry


class ModelService:
    """Model loading, adapters, speculative decoding, quantization, hot-swap."""

    def __init__(self, model_name: str, dtype: str, trust_remote_code: bool | None,
                 quantization_config=None, model_mgr=None, pipeline=None):
        self.model_name = model_name
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code
        self.quantization_config = quantization_config
        self._model_mgr = model_mgr
        self._pipeline = pipeline

        self.adapter_manager = None
        self._lora_adapters_config = None
        self._draft_model_name = None
        self.draft_model = None
        self.num_assistant_tokens = 5
        self._spec_decoder: SpeculativeDecoder | None = None
        self._spec_method = "draft_model"
        self._continuous_trainer = None
        self._continuous_trainer_config = None
        self._multi_model = None
        self._expert_registry = None
        self._moe_orchestrator = None

        self._version_manager = None
        self._model_hotswap = None
        self._paged_attention = None
        self._paged_kv_backend = None
        self._vlm_pipeline = None
        self._flash_attention = None
        self._plugin_manager = None
        self._hybrid_parallel_planner = None
        self._hybrid_parallel_executor = None
        self._zero_copy_engine = None
        self._adaptive_precision = None
        self._predictive_cache = None
        self._self_optimizing = None
        self._cuda_graph_batch_sizes = None
        self._cuda_graph_pool = None
        self._compile_enabled = False
        self._compile_mode = 'reduce-overhead'
        self._compile_fullgraph = False
        self._slora_max_adapters = None
        self._slora_manager = None
        self._rag_pipeline = None
        self._agent_loop = None
        self._disagg_orchestrator = None
        self._embedding_loader = None
        self._embedding_config = None
        self._version_config = None

        self.local_partitioner = None

    def get_spec_decoder(self):
        return self._spec_decoder

    def init_adapter(self, lora_config: Any = None):
        self.adapter_manager = None
        if lora_config and getattr(lora_config, "enabled", False):
            from distllm.models.adapter import AdapterManager
            self.adapter_manager = AdapterManager()
            self._lora_adapters_config = lora_config.adapters

    def init_speculative(self, speculative_config: Any = None):
        self._draft_model_name = None
        self.draft_model = None
        self.num_assistant_tokens = 5
        self._spec_decoder = None
        self._spec_method = "draft_model"
        self._continuous_trainer = None
        self._continuous_trainer_config = None
        if speculative_config:
            sc = speculative_config if isinstance(speculative_config, dict) else (
                speculative_config.model_dump() if hasattr(speculative_config, 'model_dump') else dict(speculative_config)
            )
            self._draft_model_name = sc.get("draft_model") or None
            self.num_assistant_tokens = sc.get("num_assistant_tokens", 5)
            self._spec_method = sc.get("method", "draft_model")
            self._spec_decoder = SpeculativeDecoder(
                num_assistant_tokens=self.num_assistant_tokens,
                min_acceptance_rate=sc.get("min_acceptance_rate", 0.3),
                warmup_steps=sc.get("warmup_steps", 10), method=self._spec_method,
                medusa_num_heads=sc.get("medusa_num_heads", 4),
                medusa_num_tokens_per_head=sc.get("medusa_num_tokens_per_head", 3),
                eagle_hidden_size=sc.get("eagle_hidden_size", 4096),
                eagle_vocab_size=sc.get("eagle_vocab_size", 32000),
                ngram_min_match=sc.get("ngram_min_match", 4),
            )
            eagle_checkpoint = sc.get("eagle_checkpoint")
            if eagle_checkpoint:
                self._spec_decoder.load_eagle_checkpoint(
                    eagle_checkpoint, variant=sc.get("eagle_variant", "eagle"),
                    hidden_size=sc.get("eagle_hidden_size"),
                    vocab_size=sc.get("eagle_vocab_size"),
                    num_layers=sc.get("eagle_num_layers", 2),
                )

    def init_multi_model(self, multi_model_config: MultiModelSettings | None):
        if multi_model_config and multi_model_config.enabled:
            model_registry = ModelRegistry(max_models=multi_model_config.max_models)
            model_registry._default_model = multi_model_config.default_model or self.model_name
            model_registry.register_version(self.model_name, "1", self.model_name, 0)
            for name, path in multi_model_config.models.items():
                model_registry.register_version(name, "1", path, 0)
            self._multi_model = MultiModelManager(
                model_name=self.model_name,
                model_registry=model_registry,
                pipeline=self._pipeline,
            )

    def init_moe(self, moe_config: MoESettings | None):
        if moe_config and moe_config.enabled:
            from distllm.core.expert_registry import ExpertRegistry
            from distllm.core.moe_orchestrator import MoEOrchestrator
            self._expert_registry = ExpertRegistry()
            self._moe_orchestrator = MoEOrchestrator(expert_registry=self._expert_registry)
            if self._multi_model is None:
                self._multi_model = MultiModelManager(
                    model_name=self.model_name,
                    pipeline=self._pipeline,
                    moe_orchestrator=self._moe_orchestrator,
                )
            else:
                self._multi_model.moe_orchestrator = self._moe_orchestrator

    def init_embedding_loader(self, embedding_config=None):
        self._embedding_loader = None
        if not embedding_config:
            return
        embed_model = getattr(embedding_config, "embedding_model", "") or ""
        rerank_model = getattr(embedding_config, "rerank_model", "") or ""
        if not embed_model and not rerank_model:
            return
        from distllm.core.embedding_loader import EmbeddingModelLoader
        self._embedding_loader = EmbeddingModelLoader(
            embedding_model=embed_model or None,
            rerank_model=rerank_model or None,
            device="auto",
            dtype=self.dtype,
            trust_remote_code=self.trust_remote_code,
        )
        if embed_model:
            self._embedding_loader.load_embedding_model()
        if rerank_model:
            self._embedding_loader.load_rerank_model()

    def init_version_manager(self, version_config=None):
        self._version_manager = None
        if not version_config or not getattr(version_config, "enabled", False):
            return
        from distllm.deploy.version_manager import VersionManager
        self._version_manager = VersionManager(
            max_versions=getattr(version_config, "max_versions", 4),
            shadow_enabled=getattr(version_config, "shadow_enabled", False),
            shadow_pct=getattr(version_config, "shadow_pct", 0.0),
            blue_green_enabled=getattr(version_config, "blue_green_enabled", False),
            ab_testing_enabled=getattr(version_config, "ab_testing_enabled", False),
            ab_test_split=getattr(version_config, "ab_test_split", 50.0),
            auto_promote_enabled=getattr(version_config, "auto_promote_enabled", False),
            min_samples=getattr(version_config, "min_samples", 100),
            significance_level=getattr(version_config, "significance_level", 0.05),
        )

    def init_model_hotswap(self, max_models: int = 4, total_gpu_memory_gb: float = 0.0):
        from distllm.core.multi_model_serving import ModelHotSwapManager
        from distllm.core.model_registry import ModelRegistry
        registry = ModelRegistry(max_models=max_models)
        registry.register(self.model_name, self.model_name, 0)

        def _load_model_callback(name, path):
            from distllm.models.partitioner import ModelPartitioner
            try:
                partitioner = ModelPartitioner(
                    model_name=path,
                    dtype=self.dtype,
                    trust_remote_code=self.trust_remote_code,
                    quantization_config=self.quantization_config,
                )
                partitioner.load_full_model()
                mem_gb = 0.0
                if torch.cuda.is_available():
                    mem_gb = torch.cuda.memory_allocated() / (1024 ** 3)
                return partitioner.full_model, partitioner.tokenizer, mem_gb
            except Exception as e:
                logger.error(f"Failed to load model '{name}' from {path}: {e}")
                raise RuntimeError(f"Failed to load model '{name}' for hot-swap: {e}") from e

        def _unload_model_callback(name, model, tokenizer):
            del model
            del tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self._model_hotswap = ModelHotSwapManager(
            model_registry=registry,
            total_gpu_memory_gb=total_gpu_memory_gb,
            max_models=max_models,
            on_load_model=_load_model_callback,
            on_unload_model=_unload_model_callback,
        )

    def init_paged_attention(self, num_blocks=256, block_size=16,
                             num_layers=12, num_heads=12, head_dim=64,
                             swap_to_cpu=False, max_swap_blocks=0):
        from distllm.core.paged_attention import PagedAttentionManager
        dtype = getattr(torch, self.dtype, torch.float16)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._paged_attention = PagedAttentionManager(
            num_blocks=num_blocks, block_size=block_size,
            num_layers=num_layers, num_heads=num_heads,
            head_dim=head_dim, dtype=dtype, device=device,
            swap_to_cpu=swap_to_cpu, max_swap_blocks=max_swap_blocks,
        )

    def init_vlm_pipeline(self, vision_model: str | None = None,
                          llm_hidden_size: int = 4096):
        if not vision_model:
            return
        from distllm.core.vlm_pipeline import VLMPipeline
        self._vlm_pipeline = VLMPipeline(
            vision_model_name=vision_model,
            llm_hidden_size=llm_hidden_size,
            device="auto",
            dtype=self.dtype,
            trust_remote_code=self.trust_remote_code,
        )
        self._vlm_pipeline.load_vision_tower()

    def init_flash_attention(self, causal: bool = True, enable_fa2: bool = True):
        if not enable_fa2:
            self._flash_attention = None
            return
        try:
            from distllm.core.flash_attention import FlashAttentionWrapper
            self._flash_attention = FlashAttentionWrapper(causal=causal)
            logger.info("FlashAttention-2 wrapper initialized")
        except ImportError:
            self._flash_attention = None
            logger.warning("FlashAttention module not available")

    def init_plugin_manager(self, coordinator):
        from distllm.core.plugin import PluginManager
        context = {
            "coordinator": coordinator,
            "model_name": self.model_name,
            "dtype": str(self.dtype),
            "trust_remote_code": self.trust_remote_code,
        }
        self._plugin_manager = PluginManager(context=context)

    def init_hybrid_parallel(self, config, total_layers, expert_registry, moe_orchestrator, pipeline):
        self._hybrid_parallel_planner = None
        self._hybrid_parallel_executor = None
        if config is None:
            return
        enabled = getattr(config, 'enabled', False) if not isinstance(config, bool) else config
        if not enabled:
            return
        from distllm.core.hybrid_parallel import (
            HardwareProber, HybridParallelPlanner, HybridParallelExecutor,
        )
        topology = HardwareProber.probe()
        self._hybrid_parallel_planner = HybridParallelPlanner(topology)
        plan = self._hybrid_parallel_planner.plan(
            total_layers=total_layers,
            num_experts=getattr(expert_registry, 'num_experts', 0) if expert_registry else 0,
            use_moe=moe_orchestrator is not None,
            pp_overlap=getattr(config, "pp_overlap", True),
            tp_enabled=getattr(config, "tp_enabled", True),
            ep_enabled=getattr(config, "ep_enabled", True),
        )
        force_tp = getattr(config, "force_tp_world_size", 0)
        if force_tp and force_tp > 1:
            from distllm.core.hybrid_parallel import ParallelStrategy
            plan.tp_world_size = force_tp
            if plan.strategy == ParallelStrategy.PP:
                plan.strategy = ParallelStrategy.TP_PP if plan.pp_num_stages > 1 else ParallelStrategy.TP
        self._hybrid_parallel_executor = HybridParallelExecutor(plan, coordinator=self)
        self._hybrid_parallel_executor.configure_pp(pipeline)
        self._hybrid_parallel_executor.launch_tp(
            model_name=self.model_name,
            dtype=self.dtype,
        )
        if hasattr(pipeline, 'enable_overlap') and plan.pp_num_stages > 1:
            pipeline.enable_overlap = True
        logger.info(f"Hybrid parallel plan: {plan.explanation}")

    def init_zero_copy(self, config):
        self._zero_copy_engine = None
        if config is None:
            return
        enabled = getattr(config, 'enabled', False) if not isinstance(config, bool) else config
        if not enabled:
            return
        from distllm.core.zero_copy_transfer import ZeroCopyTransferEngine
        self._zero_copy_engine = ZeroCopyTransferEngine()
        logger.info("Zero-copy transfer engine initialized")

    def init_adaptive_precision(self, config):
        self._adaptive_precision = None
        if config is None:
            return
        enabled = getattr(config, 'enabled', False) if not isinstance(config, bool) else config
        if not enabled:
            return
        cal_samples = getattr(config, 'calibration_samples', 64) if not isinstance(config, bool) else 64
        from distllm.core.adaptive_precision import AdaptivePrecisionEngine
        self._adaptive_precision = AdaptivePrecisionEngine(calibration_samples=cal_samples)
        logger.info("Adaptive precision engine initialized")

    def init_predictive_cache(self, config, cache_mgr):
        self._predictive_cache = None
        if config is None:
            return
        enabled = getattr(config, 'enabled', False) if not isinstance(config, bool) else config
        if not enabled:
            return
        gpu_mb = getattr(config, 'gpu_cache_mb', 512) if not isinstance(config, bool) else 512
        cpu_mb = getattr(config, 'cpu_cache_mb', 4096) if not isinstance(config, bool) else 4096
        compress_int = getattr(config, 'background_compress_interval_s', 300) if not isinstance(config, bool) else 300
        from distllm.core.predictive_cache import PredictiveCacheManager
        from distllm.core.prefix_cache import PrefixCache
        gpu_cache = PrefixCache(max_entries=0, memory_budget_bytes=gpu_mb * 1024 * 1024) if cache_mgr else None
        self._predictive_cache = PredictiveCacheManager(
            gpu_cache=gpu_cache,
            gpu_memory_bytes=gpu_mb * 1024 * 1024,
            cpu_memory_bytes=cpu_mb * 1024 * 1024,
        )
        self._predictive_cache.start_background_compression(compress_int)
        logger.info(f"Predictive cache initialized (GPU={gpu_mb}MB, CPU={cpu_mb}MB)")

    def init_self_optimizing(self, config, apply_params_fn):
        if config is None:
            return
        enabled = getattr(config, 'enabled', False) if not isinstance(config, bool) else config
        if not enabled:
            return
        from distllm.core.self_optimizing_engine import SelfOptimizingEngine
        self._self_optimizing = SelfOptimizingEngine(
            model_name=self.model_name,
            profile_dir=getattr(config, 'profile_dir', None),
            tune_interval_seconds=getattr(config, 'tune_interval_seconds', 60.0),
            warmup_seconds=getattr(config, 'warmup_seconds', 30.0),
            apply_params=apply_params_fn,
        )
        logger.info("Self-optimizing engine initialized")

    def init_cuda_graph(self, config):
        if config is None:
            return
        enabled = getattr(config, 'enabled', False) if not isinstance(config, bool) else config
        if not enabled:
            return
        self._cuda_graph_batch_sizes = getattr(config, 'batch_sizes', [1, 2, 4, 8, 16, 32])
        logger.info("CUDA graph capture enabled")

    def init_compile_support(self, config):
        self._compile_enabled = False
        if config is None:
            return
        self._compile_enabled = getattr(config, 'enabled', False) if not isinstance(config, bool) else config
        if self._compile_enabled:
            self._compile_mode = getattr(config, 'mode', 'reduce-overhead')
            self._compile_fullgraph = getattr(config, 'fullgraph', False)
            logger.info("torch.compile enabled")

    def init_slora(self, config):
        if config is None:
            return
        enabled = getattr(config, 'enabled', False) if not isinstance(config, bool) else config
        if not enabled:
            return
        self._slora_max_adapters = getattr(config, 'max_adapters', 64)
        logger.info("SLoRA multi-adapter serving enabled")

    def init_rag(self, config):
        if config is None:
            return
        enabled = getattr(config, 'enabled', False) if not isinstance(config, bool) else config
        if not enabled:
            return
        embedding_fn = getattr(self._embedding_loader, 'encode', None) if self._embedding_loader else None
        if embedding_fn is None:
            logger.warning("RAG enabled but embedding_loader not available")
            return
        from distllm.core.rag_pipeline import RAGPipeline
        self._rag_pipeline = RAGPipeline(
            embedding_fn=embedding_fn,
            dimension=getattr(config, 'dimension', 768),
            chunk_size=getattr(config, 'chunk_size', 512),
            chunk_overlap=getattr(config, 'chunk_overlap', 50),
            index_path=getattr(config, 'index_path', None),
        )
        logger.info("RAG pipeline initialized")

    def init_agent(self, config, generate_fn):
        if config is None:
            return
        enabled = getattr(config, 'enabled', False) if not isinstance(config, bool) else config
        if not enabled:
            return
        from distllm.core.agent_loop import AgentLoop
        self._agent_loop = AgentLoop(
            llm_fn=generate_fn,
            max_iterations=getattr(config, 'max_iterations', 10),
            reflection_enabled=getattr(config, 'reflection_enabled', True),
        )
        logger.info("Agent loop initialized")

    def init_disagg(self, config, local_coordinator):
        if config is None:
            return
        enabled = getattr(config, 'enabled', False) if not isinstance(config, bool) else config
        if not enabled:
            return
        from distllm.core.disagg_serving import DisaggRouter, DisaggOrchestrator
        router = DisaggRouter(local_coordinator=local_coordinator, local_model_name=self.model_name)
        for node in getattr(config, 'prefill_nodes', []):
            router.add_prefill_node(**node)
        for node in getattr(config, 'decode_nodes', []):
            router.add_decode_node(**node)
        self._disagg_orchestrator = DisaggOrchestrator(router=router)
        logger.info("Disaggregated orchestrator initialized")

    def load_local_model(self, coordinator):
        self._model_mgr.load_local_model(coordinator)

    def load_draft_model(self, coordinator):
        self._model_mgr.load_draft_model(coordinator)

    def apply_flash_attention(self, local_partitioner):
        if local_partitioner is not None and local_partitioner.full_model is not None:
            model = local_partitioner.full_model
            try:
                from distllm.core.flash_attention import apply_flash_attention_to_model
                patched = apply_flash_attention_to_model(model)
                if patched > 0:
                    logger.info(f"FlashAttention-2: patched {patched} attention modules")
            except ImportError:
                logger.debug("FlashAttention module not available, skipping patch")

    def apply_rope_scaling(self, local_partitioner):
        if local_partitioner is not None and local_partitioner.full_model is not None:
            model = local_partitioner.full_model
            config = getattr(model, "config", None)
            if config is not None:
                max_pos = getattr(config, "max_position_embeddings", 4096)
                target_ctx = 131072
                if target_ctx > max_pos:
                    from distllm.models.partitioner import apply_rope_scaling
                    apply_rope_scaling(model, target_context_len=target_ctx, scaling_type="yarn")
                    logger.info(f"RoPE scaling applied: {max_pos} -> {target_ctx}")

    def wire_paged_attention(self, local_partitioner):
        if local_partitioner is not None and local_partitioner.full_model is not None:
            model = local_partitioner.full_model
            config = getattr(model, "config", None)
            if config is not None:
                num_layers = getattr(config, "num_hidden_layers", 32)
                num_heads = getattr(config, "num_attention_heads", 32)
                head_dim = getattr(config, "hidden_size", 4096) // num_heads
                self.init_paged_attention(
                    num_blocks=512, block_size=16,
                    num_layers=num_layers, num_heads=num_heads, head_dim=head_dim,
                )
                if self._paged_attention is not None:
                    from distllm.core.kv_cache import PagedKVCacheBackend
                    self._paged_kv_backend = PagedKVCacheBackend(self._paged_attention)

    def apply_adaptive_precision(self, local_partitioner):
        engine = getattr(self, '_adaptive_precision', None)
        if engine is None or local_partitioner is None:
            return
        model = local_partitioner.full_model
        if model is None:
            return
        try:
            sample_input = torch.randint(0, 100, (1, 64), device=next(model.parameters()).device)
            engine.profile_model(model, sample_input)
            converted = engine.apply_precision(model)
            logger.info(f"Adaptive precision: profiled & converted {converted} layers")
        except Exception as e:
            logger.warning(f"Adaptive precision profiling failed: {e}")

    def enable_continuous_training(self, base_model, draft_head=None, config=None):
        head = draft_head or (
            self._spec_decoder._eagle_heads
            if self._spec_decoder and self._spec_decoder.has_eagle_heads
            else None
        )
        if head is None:
            logger.warning("Continuous training requires a draft head module")
            return
        device = next(base_model.parameters()).device
        from distllm.core.speculative_trainer import ContinuousSpeculativeTrainer, ContinuousTrainConfig
        self._continuous_trainer = ContinuousSpeculativeTrainer(
            base_model=base_model,
            draft_head=head,
            config=config or ContinuousTrainConfig(),
            device=str(device),
        )
        self._continuous_trainer.start_background()
        logger.info("Continuous speculative training enabled")

    def cuda_graph_capture(self, local_partitioner):
        if getattr(self, '_cuda_graph_batch_sizes', None) and local_partitioner is not None:
            model = local_partitioner.full_model
            config = model.config
            from distllm.core.cuda_graph import CUDAGraphPool
            self._cuda_graph_pool = CUDAGraphPool(
                model=model,
                batch_sizes=self._cuda_graph_batch_sizes,
                num_layers=getattr(config, 'num_hidden_layers', 0),
                num_heads=getattr(config, 'num_attention_heads', 0),
                head_dim=getattr(config, 'hidden_size', 4096) // getattr(config, 'num_attention_heads', 32),
            )
            self._cuda_graph_pool.capture_all()

    def compile_model(self, local_partitioner):
        if getattr(self, '_compile_enabled', False) and local_partitioner is not None:
            from distllm.core.compile_support import compile_model
            local_partitioner.full_model = compile_model(
                local_partitioner.full_model,
                mode=getattr(self, '_compile_mode', 'reduce-overhead'),
                fullgraph=getattr(self, '_compile_fullgraph', False),
            )

    def setup_slora(self, local_partitioner):
        if getattr(self, '_slora_max_adapters', None) and local_partitioner is not None:
            from distllm.core.slora_manager import SLoRAManager
            self._slora_manager = SLoRAManager(
                base_model=local_partitioner.full_model,
                max_adapters=self._slora_max_adapters,
            )

    @property
    def hybrid_parallel_executor(self):
        return self._hybrid_parallel_executor

    @property
    def predictive_cache(self):
        return self._predictive_cache

    def register_model(self, name, path, total_layers):
        if self._multi_model is None:
            self._multi_model = MultiModelManager(
                model_name=self.model_name,
                pipeline=self._pipeline,
            )
        return self._multi_model.register_model(name, path, total_layers)

    def list_models(self, chat_router=None):
        if self._multi_model is None:
            models = [self.model_name]
        else:
            models = self._multi_model.list_models()
        if chat_router is not None:
            for hname in chat_router.list_hybrid_models():
                if hname and hname not in models:
                    models.append(hname)
        return models

    def get_model_name(self, requested: str | None = None) -> str:
        if self._multi_model is None:
            return self.model_name
        return self._multi_model.get_model_name(requested)

    def moe_forward(self, hidden_states, moe_router):
        if self._multi_model is None or self._multi_model.moe_orchestrator is None:
            raise RuntimeError("MoE orchestrator not initialized")
        return self._multi_model.moe_forward(hidden_states, moe_router)
