from __future__ import annotations

import asyncio
import uuid


from distllm.core.disagg.types import PrefillRequest
from distllm.core.disagg.router import DisaggRouter


class DisaggOrchestrator:
    """End-to-end lifecycle for disaggregated serving.

    Flow:
        1. Receive request -> route to prefill pool
        2. Prefill processes prompt, returns KV cache
        3. KV cache transferred to decode pool node
        4. Decode loop runs on decode node until complete
        5. Resources released
    """

    def __init__(self, router: DisaggRouter):
        self.router = router
        self._running_tasks: dict[str, asyncio.Task] = {}
        self._is_healthy = True

    async def submit(self, prompt_tokens: list[int], max_new_tokens: int = 256, **kwargs) -> str:
        """Submit a generation request through the disagg pipeline.

        Returns a request_id that can be used to poll for results.
        """
        request_id = str(uuid.uuid4())
        request = PrefillRequest(
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )
        task = asyncio.create_task(self._execute_pipeline(request))
        self._running_tasks[request_id] = task
        return request_id

    async def _execute_pipeline(self, request: PrefillRequest) -> list[int]:
        result = await self.router.prefill(request)
        if result is None or result.first_token is None:
            return []

        generated: list[int] = [result.first_token]
        token = result.first_token
        position = result.prompt_len

        for step in range(1, request.max_new_tokens):
            next_token = await self.router.decode(
                request.request_id, token, position,
            )
            if next_token is None:
                break
            generated.append(next_token)
            token = next_token
            position += 1

        await self.router.complete_request(request.request_id)
        return generated

    async def get_result(self, request_id: str, timeout: float = 30.0) -> list[int] | None:
        """Get the result of a generation request."""
        task = self._running_tasks.get(request_id)
        if task is None:
            return None
        try:
            result = await asyncio.wait_for(task, timeout=timeout)
            self._running_tasks.pop(request_id, None)
            return result
        except asyncio.TimeoutError:
            return None

    @property
    def pending_count(self) -> int:
        return len([t for t in self._running_tasks.values() if not t.done()])

    def health_check(self) -> dict:
        prefill_stats = self.router.prefill_pool.get_stats()
        decode_stats = self.router.decode_pool.get_stats()
        self._is_healthy = (
            prefill_stats["active_nodes"] > 0 and decode_stats["active_nodes"] > 0
        )
        return {
            "healthy": self._is_healthy,
            "pending_requests": self.pending_count,
            "prefill_pool": prefill_stats,
            "decode_pool": decode_stats,
        }
