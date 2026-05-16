"""Streaming SSE response helpers for chat and completion endpoints.

Deduplicates the nearly-identical _stream_chat and _stream_completion
functions into a single _stream_response() that parameterizes the
object_type and request_id prefix.
"""

import asyncio
import json
import time
import uuid
from typing import Any, AsyncGenerator, Dict, Optional, List

import torch

from .api_state import g


def _sample_token(logits: torch.Tensor, temperature: float, top_p: float, top_k: int = 0) -> torch.Tensor:
    """Sample next token from logits with temperature, top-k, and top-p filtering."""
    if temperature > 0:
        probs = torch.softmax(logits / temperature, dim=-1)
        if top_k > 0:
            top_k_indices = torch.topk(probs[0], top_k, dim=-1).indices
            mask = torch.zeros(probs.shape[-1], dtype=torch.bool, device=probs.device)
            mask[top_k_indices] = True
            probs = probs.masked_fill(~mask, 0.0)
            probs = probs / probs.sum(dim=-1, keepdim=True)
        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            probs = probs.masked_fill(indices_to_remove, 0.0)
            probs = probs / probs.sum(dim=-1, keepdim=True)
        return torch.multinomial(probs, 1)
    else:
        return torch.argmax(logits, dim=-1, keepdim=True)


def _stream_event(request_id: str, object_type: str, model: str, token_text: str) -> str:
    """Format a single streaming SSE event for chat or completion."""
    d: Dict[str, Any] = {
        "id": request_id,
        "object": object_type,
        "created": int(time.time()),
        "model": model,
    }
    if object_type == "chat.completion.chunk":
        d["choices"] = [{"index": 0, "delta": {"content": token_text}}]
    else:
        d["choices"] = [{"index": 0, "text": token_text}]
    return f"data: {json.dumps(d)}\n\n"


def _stream_start_event(request_id: str, object_type: str, model: str) -> str:
    """Format the initial streaming SSE event with role/text start."""
    d: Dict[str, Any] = {
        "id": request_id,
        "object": object_type,
        "created": int(time.time()),
        "model": model,
    }
    if object_type == "chat.completion.chunk":
        d["choices"] = [{"index": 0, "delta": {"role": "assistant"}}]
    else:
        d["choices"] = [{"index": 0, "text": ""}]
    return f"data: {json.dumps(d)}\n\n"


def _stream_stop_event(request_id: str, object_type: str, model: str) -> str:
    """Format the final streaming SSE stop event."""
    d: Dict[str, Any] = {
        "id": request_id,
        "object": object_type,
        "created": int(time.time()),
        "model": model,
    }
    if object_type == "chat.completion.chunk":
        d["choices"] = [{"index": 0, "finish_reason": "stop", "delta": {}}]
    else:
        d["choices"] = [{"index": 0, "finish_reason": "stop", "text": ""}]
    return f"data: {json.dumps(d)}\n\n"


async def _generate_tokens(
    prompt: str,
    request_id: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> AsyncGenerator[str, None]:
    """Core token generation loop shared by chat and completion streaming.

    Yields token text strings as they are generated, handling both
    local and distributed modes with dynamic param updates.
    """
    coord = g.coordinator
    if not coord:
        return

    if coord.local_partitioner:
        model = coord.local_partitioner.full_model
        tokenizer = coord.tokenizer
        device = next(model.parameters()).device

        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        past_key_values = None

        with torch.no_grad():
            for step in range(max_tokens):
                temp = temperature
                tp = top_p
                tk = top_k
                if coord:
                    from distllm.core.param_update_channel import GenerationParams
                    params = coord._param_update_channel.get(request_id)
                    if params is not None and isinstance(params, GenerationParams):
                        temp = params.temperature
                        tp = params.top_p
                        tk = params.top_k

                if step == 0:
                    outputs = await asyncio.to_thread(model, input_ids, use_cache=True)
                else:
                    outputs = await asyncio.to_thread(model, next_token, past_key_values=past_key_values, use_cache=True)

                logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values

                next_token = _sample_token(logits, temp, tp, tk)
                token_text = tokenizer.decode(next_token[0], skip_special_tokens=True)
                yield token_text

                if next_token.item() == tokenizer.eos_token_id:
                    break

    elif coord.node_order:
        tokenizer = coord.tokenizer
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        generated_ids = input_ids.clone()

        node_kv_caches: Dict[str, Optional[List]] = {
            nid: None for nid in coord.node_order
        }

        for step in range(max_tokens):
            temp = temperature
            tp = top_p
            tk = top_k
            if coord:
                from distllm.core.param_update_channel import GenerationParams
                params = coord._param_update_channel.get(request_id)
                if params is not None and isinstance(params, GenerationParams):
                    temp = params.temperature
                    tp = params.top_p
                    tk = params.top_k

            step_input = generated_ids if step == 0 else generated_ids[:, -1:]

            logits = await asyncio.to_thread(
                coord._pipeline.run_pipeline,
                step_input, node_kv_caches, request_id,
            )

            next_token = _sample_token(logits[:, -1, :], temp, tp, tk)
            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=1)

            token_text = tokenizer.decode(next_token[0], skip_special_tokens=True)
            yield token_text

            if next_token.item() == tokenizer.eos_token_id:
                break


async def _stream_response(
    prompt: str,
    request: Any,  # ChatCompletionRequest or CompletionRequest
    object_type: str,
    request_id_prefix: str,
) -> AsyncGenerator[str, None]:
    """Unified streaming response generator for both chat and completion.

    Args:
        prompt: The text prompt to generate from.
        request: The pydantic request object (chat or completion).
        object_type: SSE object type, e.g. "chat.completion.chunk" or "text_completion.chunk".
        request_id_prefix: Prefix for the request ID, e.g. "chatcmpl-" or "cmpl-".
    """
    request_id = f"{request_id_prefix}{uuid.uuid4().hex[:12]}"

    coord = g.coordinator
    if coord:
        coord._param_update_channel.register(request_id)

    yield _stream_start_event(request_id, object_type, request.model)

    if not coord:
        yield _stream_stop_event(request_id, object_type, request.model)
        yield "data: [DONE]\n\n"
        return

    async for token_text in _generate_tokens(
        prompt, request_id, request.max_tokens,
        request.temperature, request.top_p, request.top_k,
    ):
        yield _stream_event(request_id, object_type, request.model, token_text)

    yield _stream_stop_event(request_id, object_type, request.model)
    yield "data: [DONE]\n\n"

    if coord:
        coord._param_update_channel.unregister(request_id)
