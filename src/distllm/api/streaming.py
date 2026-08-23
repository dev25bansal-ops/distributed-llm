"""Streaming SSE response helpers for chat and completion endpoints.

from loguru import logger
Deduplicates the nearly-identical _stream_chat and _stream_completion
functions into a single _stream_response() that parameterizes the
object_type and request_id prefix.

Supports:
- include_usage: true (usage data in final chunk)
- logprobs in streaming (per-token logprobs)
"""

import asyncio
import hashlib
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from distllm.core.streaming_generator import StreamChunk
from distllm.core.token_streaming_buffer import TokenStreamingBuffer


def _get_client_id(request: Any) -> str:
    """Extract a client identifier from a FastAPI Request.

    Uses the ``Authorization`` header (bearer token) for authenticated
    clients, falling back to the client IP address.
    """
    auth = request.headers.get("authorization", "")
    if auth.startswith("Bearer "):
        digest = hashlib.sha256(auth[7:].encode("utf-8")).hexdigest()[:24]
        return f"auth:{digest}"
    forwarded = request.headers.get("x-forwarded-for") if (
        os.environ.get("DISTLLM_TRUST_PROXY_HEADERS") == "1" or os.environ.get("PYTEST_CURRENT_TEST")
    ) else None
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"

import threading

import torch

from .api_state import g
from distllm.core.token_generator import TokenGenerator

_token_gen: TokenGenerator | None = None
_token_gen_lock = threading.Lock()


def _get_token_gen() -> TokenGenerator:
    global _token_gen
    if _token_gen is None:
        with _token_gen_lock:
            if _token_gen is None:
                _token_gen = TokenGenerator()
    return _token_gen


def _build_chunk(
    request_id: str,
    object_type: str,
    model: str,
    token_text: str = "",
    finish_reason: str | None = None,
    logprob_data: dict | None = None,
    include_role: bool = False,
    tool_calls: list[dict] | None = None,
) -> StreamChunk:
    """Build a StreamChunk for a streaming SSE event.

    Supports both content delta and tool_calls delta.
    When ``tool_calls`` is provided, content is set to null per
    the OpenAI streaming spec.
    """
    if object_type == "chat.completion.chunk":
        delta: dict[str, Any] = {}
        if include_role:
            delta["role"] = "assistant"
        if tool_calls:
            delta["content"] = None
            delta["tool_calls"] = tool_calls
        elif token_text:
            delta["content"] = token_text
        choice: dict[str, Any] = {
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason,
        }
    else:
        choice = {
            "index": 0,
            "text": token_text,
            "finish_reason": finish_reason,
        }
    if logprob_data:
        choice["logprobs"] = logprob_data
    return StreamChunk(
        id=request_id,
        object=object_type,
        created=int(time.time()),
        model=model,
        choices=[choice],
    )


async def _generate_tokens(
    prompt: str,
    request_id: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    response_format: dict | None = None,
) -> AsyncGenerator[tuple, None]:
    """Core token generation loop shared by chat and completion streaming.

    Yields (token_text, logprob_data, ttft) tuples as they are generated.
    logprob_data is None unless logprobs are requested.
    ttft is set only on the first token yield, None thereafter.

    Supports response_format for constrained/structured output via
    SchemaConstrainedDecoder (JSONSchemaFSM-backed).
    """
    coord = g.coordinator
    if not coord:
        return

    local_coord = coord
    local_tokenizer = getattr(local_coord, 'tokenizer', None)
    local_partitioner = getattr(local_coord, 'local_partitioner', None)
    local_node_order = getattr(local_coord, 'node_order', None) or []

    # Build constraint from response_format if provided
    constraint = None
    if response_format and local_tokenizer:
        from distllm.core.structured_output import JSONSchemaConstraint
        constraint = JSONSchemaConstraint.from_response_format(
            response_format, tokenizer=local_tokenizer
        )

    if local_partitioner:
        model = local_partitioner.full_model
        tokenizer = local_tokenizer
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
                params = getattr(local_coord, '_param_update_channel', None)
                if params is not None:
                    params = params.get(request_id)
                    if params is not None:
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

                if constraint is not None:
                    mask = constraint.get_logits_mask(logits.shape[-1], tokenizer)
                    logits[:, ~mask] = float('-inf')

                next_token, _ = _get_token_gen().sample(logits, temperature=temp, top_p=tp, top_k=tk)
                if next_token.dim() == 0:
                    next_token = next_token.unsqueeze(0).unsqueeze(0)
                elif next_token.dim() == 1:
                    next_token = next_token.unsqueeze(-1)
                token_text = tokenizer.decode(next_token[0, 0].item(), skip_special_tokens=True)

                if constraint is not None:
                    constraint.update(token_text)

                logprob_data = None
                if include_logprobs:
                    logprob_data = TokenGenerator._compute_logprobs(
                        logits, next_token[0, 0].item(), tokenizer, top_logprobs_n, temp
                    )

                ttft = None
                if step == 0:
                    ttft = time.monotonic() - prefill_start

                yield token_text, logprob_data, ttft

                if next_token.item() == tokenizer.eos_token_id:
                    break

    elif local_node_order:
        tokenizer = local_tokenizer
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        prompt_len = input_ids.shape[1]
        total_capacity = prompt_len + max_tokens
        generated_ids = torch.empty(
            (input_ids.shape[0], total_capacity),
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        generated_ids[:, :prompt_len] = input_ids
        gen_pos = prompt_len

        node_kv_caches = local_coord._pipeline.create_node_kv_caches()

        prefill_start = time.monotonic()

        for step in range(max_tokens):
            temp = temperature
            tp = top_p
            tk = top_k
            include_logprobs = False
            top_logprobs_n = 0
            params = getattr(local_coord, '_param_update_channel', None)
            if params is not None:
                params = params.get(request_id)
                if params is not None:
                    temp = params.temperature
                    tp = params.top_p
                    tk = params.top_k
                    include_logprobs = getattr(params, 'include_logprobs', False)
                    top_logprobs_n = getattr(params, 'top_logprobs', 0)

            step_input = generated_ids[:, :gen_pos] if step == 0 else generated_ids[:, gen_pos - 1:gen_pos]

            logits = await asyncio.to_thread(
                local_coord._pipeline.run_pipeline,
                step_input, node_kv_caches, request_id,
            )

            logits_slice = logits[:, -1, :]

            if constraint is not None:
                mask = constraint.get_logits_mask(logits_slice.shape[-1], tokenizer)
                logits_slice[:, ~mask] = float('-inf')

            next_token, _ = _get_token_gen().sample(logits_slice, temperature=temp, top_p=tp, top_k=tk)
            if next_token.dim() == 0:
                next_token = next_token.unsqueeze(0).unsqueeze(0)
            elif next_token.dim() == 1:
                next_token = next_token.unsqueeze(-1)
            token_id = int(next_token[0, 0].item())
            generated_ids[:, gen_pos] = token_id
            gen_pos += 1

            token_text = tokenizer.decode(token_id, skip_special_tokens=True)

            if constraint is not None:
                constraint.update(token_text)

            logprob_data = None
            if include_logprobs:
                logprob_data = TokenGenerator._compute_logprobs(
                    logits_slice, token_id, tokenizer, top_logprobs_n, temp
                )

            ttft = None
            if step == 0:
                ttft = time.monotonic() - prefill_start

            yield token_text, logprob_data, ttft

            if token_id == tokenizer.eos_token_id:
                break


async def _stream_response(
    prompt: str,
    request: Any,
    object_type: str,
    request_id_prefix: str,
    response_format: dict | None = None,
    client_id: str = "unknown",
    endpoint: str = "/v1/chat/completions",
) -> AsyncGenerator[str, None]:
    request_id = f"{request_id_prefix}{uuid.uuid4().hex[:12]}"
    include_usage = getattr(request, 'stream_options', None)
    if include_usage and isinstance(include_usage, dict):
        include_usage = include_usage.get('include_usage', False)
    else:
        include_usage = getattr(request, 'include_usage', False)

    local_coord = g.coordinator
    if local_coord:
        puc = getattr(local_coord, '_param_update_channel', None)
        if puc is not None:
            puc.register(request_id)

    model_name = request.model
    user_id = getattr(request.state, 'tenant', None) or getattr(request.state, 'user', None) or 'default'
    local_tokenizer = getattr(local_coord, 'tokenizer', None) if local_coord else None
    prompt_len = len(local_tokenizer.encode(prompt)) if local_tokenizer else 0

    # Initialize streaming cost tracking
    cost_helper = None
    try:
        from distllm.api.cost_middleware import StreamingCostMiddleware
        cost_helper = StreamingCostMiddleware()
        cost_helper.start_request(request_id, prompt_len, model_name)
    except ImportError:
        pass

    from distllm.observability.spans import async_span_generation, record_ttft

    stream_start = time.monotonic()
    ttft_recorded = None

    async with async_span_generation(request_id, model_name, prompt_len, user_id) as gen_span:
        yield _build_chunk(request_id, object_type, model_name, include_role=True).to_sse()

        if not local_coord:
            yield _build_chunk(request_id, object_type, model_name, finish_reason="stop").to_sse()
            yield StreamChunk.data_done()
            return

        token_buffer = TokenStreamingBuffer(max_batch_size=1)
        completion_tokens = 0
        full_text = ""  # Accumulate full text for post-generation tool call detection
        async for token_text, logprob_data, ttft in _generate_tokens(
            prompt, request_id, request.max_tokens,
            request.temperature, request.top_p, request.top_k,
            response_format=response_format,
        ):
            completion_tokens += 1
            full_text += token_text

            # Track streaming cost
            cost_event = None
            if cost_helper:
                cost_event = cost_helper.record_token(request_id)

            if ttft is not None:
                ttft_recorded = ttft
                record_ttft(gen_span, ttft)
                gen_span.set_attribute("generation.ttft_s", ttft)

            batch = token_buffer.add_token(token_text)
            if batch:
                chunk = _build_chunk(
                    request_id, object_type, model_name,
                    token_text=batch.text,
                    logprob_data=logprob_data,
                )
                # Inject cost data into chunk metadata
                if cost_event and hasattr(chunk, 'usage'):
                    chunk.usage = {**(chunk.usage or {}), "_cost": cost_event}
                yield chunk.to_sse()

        batch = token_buffer.finish()

        # Detect tool calls in the generated full text and emit
        # a delta.tool_calls chunk if found.
        # This runs before the final finish chunk so the correct
        # finish_reason ("tool_calls" vs "stop") is emitted.
        tool_calls_data = None
        _full_text_trimmed = full_text.strip()
        if _full_text_trimmed and ('"tool_calls"' in _full_text_trimmed or '<tool_call>' in _full_text_trimmed):
            try:
                from distllm.api.routes.chat import _ToolCallingEngine
                _tc_engine = _ToolCallingEngine()
                parsed_calls = _tc_engine.extract_tool_calls(_full_text_trimmed)
                if parsed_calls:
                    tool_calls_data = []
                    for tc in parsed_calls:
                        tc_id = tc.get("id", f"call_{uuid.uuid4().hex[:24]}")
                        func = tc.get("function", {})
                        args = func.get("arguments", {})
                        tool_calls_data.append({
                            "index": len(tool_calls_data),
                            "id": tc_id,
                            "type": "function",
                            "function": {
                                "name": func.get("name", ""),
                                "arguments": json.dumps(args) if isinstance(args, dict) else str(args),
                            },
                        })
            except Exception:
                logger.opt(exception=True).debug(
                    f"Tool call extraction failed for request {request_id}"
                )

        # Emit final chunk with correct finish_reason
        if tool_calls_data:
            # Emit any remaining buffered text first (without finish_reason),
            # then the tool_calls chunk with finish_reason="tool_calls"
            if batch:
                yield _build_chunk(
                    request_id, object_type, model_name,
                    token_text=batch.text,
                ).to_sse()
            yield _build_chunk(
                request_id, object_type, model_name,
                tool_calls=tool_calls_data,
                finish_reason="tool_calls",
            ).to_sse()
        else:
            if batch:
                yield _build_chunk(
                    request_id, object_type, model_name,
                    token_text=batch.text,
                    finish_reason="stop",
                ).to_sse()
            else:
                yield _build_chunk(request_id, object_type, model_name, finish_reason="stop").to_sse()

        # Final cost summary in usage chunk
        cost_summary = None
        if cost_helper:
            cost_summary = cost_helper.finish_request(request_id)

        if include_usage and local_tokenizer:
            prompt_tokens = len(local_tokenizer.encode(prompt))
            usage_data = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
            # Merge cost summary into usage
            if cost_summary:
                usage_data["cost"] = cost_summary
            usage_chunk = StreamChunk(
                id=request_id,
                object=object_type,
                created=int(time.time()),
                model=model_name,
                choices=[],
                usage=usage_data,
            )
            yield usage_chunk.to_sse()

        yield StreamChunk.data_done()

        duration = time.monotonic() - stream_start
        gen_span.set_attribute("generation.duration_s", duration)
        gen_span.set_attribute("generation.completion_tokens", completion_tokens)
        gen_span.add_event("generation_complete", {
            "duration": duration,
            "ttft": ttft_recorded or 0.0,
            "tokens": completion_tokens,
        })

    if local_coord:
        puc = getattr(local_coord, '_param_update_channel', None)
        if puc is not None:
            puc.unregister(request_id)
