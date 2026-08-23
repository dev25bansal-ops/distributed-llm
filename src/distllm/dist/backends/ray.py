"""Ray-based pipeline execution for distributed LLM inference.

Chains RayWorkerNode actor calls through pipeline stages using
Ray ObjectRefs for zero-copy tensor transfer between stages
on the same node.
"""

from __future__ import annotations
try:
    import ray
except ImportError:
    ray = None
import torch
from loguru import logger

class RayPipeline:
    def __init__(self) -> None:
        self.workers: list[ray.actor.ActorHandle] = []
        self.worker_ids: list[str] = []
        self._num_gpus = 0

    def add_worker(self, worker: ray.actor.ActorHandle, node_id: str) -> None:
        self.workers.append(worker)
        self.worker_ids.append(node_id)
        logger.info(f"RayPipeline: added worker {node_id} ({len(self.workers)} total)")

    @property
    def num_stages(self) -> int:
        return len(self.workers)

    def run_pipeline(self, input_ids: torch.Tensor, request_id: str) -> torch.Tensor:
        if not self.workers:
            raise RuntimeError("No workers registered in pipeline")

        hidden_ref = ray.put(input_ids)

        for i, worker in enumerate(self.workers):
            is_first = (i == 0)
            (i == len(self.workers) - 1)

            if is_first:
                hidden_ref = worker.forward.remote(
                    input_ids=hidden_ref,
                    request_id=request_id,
                )
            else:
                hidden_ref = worker.forward.remote(
                    hidden_states=hidden_ref,
                    request_id=request_id,
                )

        logits = ray.get(hidden_ref)
        return logits

    def run_pipeline_async(self, input_ids: torch.Tensor, request_id: str):
        if not self.workers:
            raise RuntimeError("No workers registered in pipeline")

        ref = ray.put(input_ids)

        for i, worker in enumerate(self.workers):
            is_first = (i == 0)
            if is_first:
                ref = worker.forward.remote(
                    input_ids=ref,
                    request_id=request_id,
                )
            else:
                ref = worker.forward.remote(
                    hidden_states=ref,
                    request_id=request_id,
                )

        return ref

    def clear_kv_cache(self, request_id: str) -> None:
        for worker in self.workers:
            worker.clear_kv_cache.remote(request_id)

    def clear_all_kv_caches(self) -> None:
        for worker in self.workers:
            worker.clear_all_kv_caches.remote()

    def health_check_all(self) -> list[dict]:
        refs = [worker.health.remote() for worker in self.workers]
        return ray.get(refs)

    def get_num_stages(self) -> int:
        return len(self.workers)
