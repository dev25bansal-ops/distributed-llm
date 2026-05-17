"""Streaming SSE response helpers for chat and completion endpoints.

Deduplicates the nearly-identical _stream_chat and _stream_completion
functions into a single _stream_response() that parameterizes the
object_type and request_id prefix.

Supports:
- include_usage: true (usage data in final chunk)
- logprobs in streaming (per-token logprobs)
"""

import asyncio
import json
import math
import time
import uuid
from contextlib import asynccontextmanager
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


def _compute_logprobs(
    logits: torch.Tensor,
    token_id: int,
    tokenizer,
    top_logprobs: int = 0,
    temperature: float = 1.0,
) -> Dict[str, Any]:
    """Compute logprobs for a sampled token.

    Args:
        logits: Raw logits [batch, vocab].
        token_id: The sampled token ID.
        tokenizer: Tokenizer for decoding token strings.
        top_logprobs: Number of top alternatives to return.
        temperature: Sampling temperature used.

    Returns:
        Dict with token logprob and top alternatives.
    """
    probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
    log_probs = torch.log(probs + 1e-10)

    token_logprob = log_probs[0, token_id].item()
    token_str = tokenizer.decode([token_id])

    result = {
        "token": token_str,
        "logprob": token_logprob,
        "bytes": list(token_str.encode('utf-8')) if token_str else None,
    }

    if top_logprobs > 0:
        top_indices = torch.topk(log_probs[0], min(top_logprobs, log_probs.shape[-1])).indices
        result["top_logprobs"] = []
        for idx in top_indices:
            alt_str = tokenizer.decode([idx.item()])
            result["top_logprobs"].append({
                "token": alt_str,
                "logprob": log_probs[0, idx].item(),
                "bytes": list(alt_str.encode('utf-8')) if alt_str else None,
            })

    return result


def _stream_event(
    request_id: str,
    object_type: str,
    model: str,
    token_text: str,
    logprob_data: Optional[Dict] = None,
) -> str:
    """Format a single streaming SSE event for chat or completion."""
    d: Dict[str, Any] = {
        "id": request_id,
        "object": object_type,
        "created": int(time.time()),
        "model": model,
    }
    if object_type == "chat.completion.chunk":
        choice = {"index": 0, "delta": {"content": token_text}}
        if logprob_data:
            choice["logprobs"] = {"content": [logprob_data]}
        d["choices"] = [choice]
    else:
        choice = {"index": 0, "text": token_text}
        if logprob_data:
            choice["logprobs"] = logprob_data
        d["choices"] = [choice]
    return f"data: {json.dumps(d)}\n\n"


def _stream_usage_event(
    request_id: str,
    object_type: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> str:
    """Format usage data as a streaming SSE event (final chunk when include_usage=true)."""
    d: Dict[str, Any] = {
        "id": request_id,
        "object": object_type,
        "created": int(time.time()),
        "model": model,
        "choices": [],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
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
) -> AsyncGenerator[tuple, None]:
    """Core token generation loop shared by chat and completion streaming.

    Yields (token_text, logprob_data, ttft) tuples as they are generated.
    logprob_data is None unless logprobs are requested.
    ttft is set only on the first token yield, None thereafter.
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

        prefill_start = time.monotonic()

        with torch.no_grad():
            for step in range(max_tokens):
                temp = temperature
                tp = top_p
                tk = top_k
                include_logprobs = False
                top_logprobs_n = 0
                if coord:
                    from distllm.core.param_update_channel import GenerationParams
                    params = coord._param_update_channel.get(request_id)
                    if params is not None and isinstance(params, GenerationParams):
                        temp = params.temperature
                        tp = params.top_p
                        tk = params.top_k
                        include_logprobs = getattr(params, 'include_logprobs', False)
                        top_logprobs_n = getattr(params, 'top_logprobs', 0)

                if step == 0:
                    outputs = await asyncio.to_thread(model, input_ids, use_cache=True)
                else:
                    outputs = await asyncio.to_thread(model, next_token, past_key_values=past_key_values, use_cache=True)

                logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values

                next_token = _sample_token(logits, temp, tp, tk)
                token_text = tokenizer.decode(next_token[0], skip_special_tokens=True)

                logprob_data = None
                if include_logprobs:
                    logprob_data = _compute_logprobs(logits, next_token[0].item(), tokenizer, top_logprobs_n, temp)

                ttft = None
                if step == 0:
                    ttft = time.monotonic() - prefill_start

                yield token_text, logprob_data, ttft

                if next_token.item() == tokenizer.eos_token_id:
                    break

    elif coord.node_order:
        tokenizer = coord.tokenizer
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        generated_ids = input_ids.clone()

        node_kv_caches: Dict[str, Optional[List]] = {
            nid: None for nid in coord.node_order
        }

        prefill_start = time.monotonic()

        for step in range(max_tokens):
            temp = temperature
            tp = top_p
            tk = top_k
            include_logprobs = False
            top_logprobs_n = 0
            if coord:
                from distllm.core.param_update_channel import GenerationParams
                params = coord._param_update_channel.get(request_id)
                if params is not None and isinstance(params, GenerationParams):
                    temp = params.temperature
                    tp = params.top_p
                    tk = params.top_k
                    include_logprobs = getattr(params, 'include_logprobs', False)
                    top_logprobs_n = getattr(params, 'top_logprobs', 0)

            step_input = generated_ids if step == 0 else generated_ids[:, -1:]

            logits = await asyncio.to_thread(
                coord._pipeline.run_pipeline,
                step_input, node_kv_caches, request_id,
            )

            next_token = _sample_token(logits[:, -1, :], temp, tp, tk)
            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=1)

            token_text = tokenizer.decode(next_token[0], skip_special_tokens=True)

            logprob_data = None
            if include_logprobs:
                logprob_data = _compute_logprobs(logits[:, -1, :], next_token[0].item(), tokenizer, top_logprobs_n, temp)

            ttft = None
            if step == 0:
                ttft = time.monotonic() - prefill_start

            yield token_text, logprob_data, ttft

            if next_token.item() == tokenizer.eos_token_id:
                break


async def _stream_response(
    prompt: str,
    request: Any,  # ChatCompletionRequest or CompletionRequest
    object_type: str,
    request_id_prefix: str,
) -> AsyncGenerator[str, None]:
    """Unified streaming response generator for both chat and completion.

    Supports:
    - include_usage: true (sends usage data in final chunk)
    - logprobs: per-token logprobs in streaming
    - OTel generation span with TTFT tracking
    - Cost tracking completion
    """
    request_id = f"{request_id_prefix}{uuid.uuid4().hex[:12]}"
    include_usage = getattr(request, 'stream_options', None)
    if include_usage and isinstance(include_usage, dict):
        include_usage = include_usage.get('include_usage', False)
    else:
        include_usage = getattr(request, 'include_usage', False)

    coord = g.coordinator
    if coord:
        coord._param_update_channel.register(request_id)

    model_name = request.model
    tenant = getattr(request, 'user', None) or 'default'
    prompt_len = len(coord.tokenizer.encode(prompt)) if coord and coord.tokenizer else 0

    from distllm.observability.spans import async_span_generation, record_ttft

    stream_start = time.monotonic()
    ttft_recorded = None

    async with async_span_generation(request_id, model_name, prompt_len, tenant) as gen_span:
        yield _stream_start_event(request_id, object_type, model_name)

        if not coord:
            yield _stream_stop_event(request_id, object_type, model_name)
            yield "data: [DONE]\n\n"
            return

        completion_tokens = 0
        async for token_text, logprob_data, ttft in _generate_tokens(
            prompt, request_id, request.max_tokens,
            request.temperature, request.top_p, request.top_k,
        ):
            completion_tokens += 1
            # Record TTFT on first token
            if ttft is not None:
                ttft_recorded = ttft
                record_ttft(gen_span, ttft)
                gen_span.set_attribute("generation.ttft_s", ttft)

            yield _stream_event(request_id, object_type, model_name, token_text, logprob_data)

        yield _stream_stop_event(request_id, object_type, model_name)

        # Usage event if requested
        if include_usage and coord and coord.tokenizer:
            prompt_tokens = len(coord.tokenizer.encode(prompt))
            yield _stream_usage_event(request_id, object_type, model_name, prompt_tokens, completion_tokens)

        yield "data: [DONE]\n\n"

        # Record final metrics on the generation span
        duration = time.monotonic() - stream_start
        gen_span.set_attribute("generation.duration_s", duration)
        gen_span.set_attribute("generation.completion_tokens", completion_tokens)
        gen_span.add_event("generation_complete", {
            "duration": duration,
            "ttft": ttft_recorded or 0.0,
            "tokens": completion_tokens,
        })

        # Prometheus metrics via exporter
        if coord.metrics_exporter:
            coord.metrics_exporter.tokens_generated.inc(completion_tokens)
            coord.metrics_exporter.token_latency.observe(duration)
            if duration > 0:
                coord.metrics_exporter.tokens_per_second.set(completion_tokens / duration)

        # Cost tracking completion
        cost_tracker = getattr(g, 'cost_tracker', None)
        if cost_tracker and request_id:
            try:
                cost_tracker.complete_request(request_id)
            except Exception:
                pass  # Non-fatal

    if coord:
        coord._param_update_channel.unregister(request_id)
