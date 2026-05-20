from distllm.core.batch_scheduler import BatchScheduler, ScheduledBatch
from distllm.core.preemption_policy import PreemptionPolicy, GPUMemoryMonitor, SLATracker
import torch


class SchedulerService:
    """Batch scheduling, priority queue, preemption, and sampling."""

    def __init__(self, cache_mgr=None):
        self.scheduler: BatchScheduler | None = None
        self._cache_mgr = cache_mgr
        self._preemption_policy: PreemptionPolicy | None = None
        self.chunked_prefill_enabled = False
        self.chunked_prefill_chunk_size = 512

    def init_scheduler(self, max_batch_size: int, max_tokens_per_batch: int,
                       model_info=None):
        self.scheduler = None
        if max_batch_size > 1:
            self.scheduler = BatchScheduler(
                max_batch_size=max_batch_size,
                max_tokens_per_batch=max_tokens_per_batch,
                model_info=model_info,
            )
        if self.scheduler is not None and self._cache_mgr:
            self.scheduler.set_cache_manager(self._cache_mgr)
        self._preemption_policy = None
        if max_batch_size > 1:
            self._preemption_policy = PreemptionPolicy(
                gpu_monitor=GPUMemoryMonitor(),
                sla_tracker=SLATracker(max_violations=3, sla_deadline_ms=5000.0),
                max_queue_depth=100, max_checkpoints=10, checkpoint_memory_limit_mb=4096,
            )

    def sample(self, logits: torch.Tensor, temperature: float = 1.0,
               top_p: float = 1.0, top_k: int = 0) -> torch.Tensor:
        if self.scheduler is not None:
            return self.scheduler.sample(logits, temperature, top_p, top_k)
        from distllm.core.token_generator import TokenGenerator
        return TokenGenerator.sample(logits, temperature, top_p, top_k)

    def sample_batch(self, logits: torch.Tensor, batch: ScheduledBatch) -> torch.Tensor:
        if self.scheduler is not None:
            return self.scheduler.sample_batch(logits, batch)
        from distllm.core.token_generator import TokenGenerator
        return TokenGenerator.sample_batch(logits, batch)

    def generate_batch(self, pipeline_runner, timeout: float = 120.0,
                       max_steps: int = 0) -> None:
        pipeline_runner.generate_batch(timeout=timeout, max_steps=max_steps)

    def get_model_info(self):
        if self.scheduler is not None and hasattr(self.scheduler, '_model_info'):
            return self.scheduler._model_info
        return None

    def set_model_info(self, model_info):
        if self.scheduler is not None:
            self.scheduler._model_info = model_info
            self.scheduler._use_length_grouping = True

    @property
    def preemption_policy(self):
        return self._preemption_policy
