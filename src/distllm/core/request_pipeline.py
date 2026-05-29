"""Request pipeline: generation, batch scheduling, and speculative decoding.

Extracted from coordinator.py to reduce the 3000+ line monolith.
Contains all generate(), generate_async(), generate_batch() methods
and their supporting functions (sampling, speculative decoding, pipeline batch).
"""

from __future__ import annotations

import time
import uuid
from contextvars import ContextVar
from typing import Any

import torch
from loguru import logger

from distllm.core.batch_scheduler import ScheduledBatch, Sequence
from distllm.core.structured_output import JSONSchemaConstraint
from distllm.core.constrained_decoder import SchemaConstrainedDecoder
from distllm.core.debug import is_debug_mode
from distllm.core.graceful_degradation import LoadSnapshot
from distllm.errors.types import (
    NodeError, NodeUnreachableError, OOMError, GRPCTimeoutError, BatchError,
)

_current_request_id_ctx: ContextVar[str | None] = ContextVar(
    "current_request_id", default=None
)


class RequestPipeline:
    """Handles text generation, batch scheduling, and speculative decoding.

    Composes the coordinator's specialized components into a generation pipeline.
    Holds a reference to the coordinator for shared state access.
    """

    def __init__(self, coord):
        self._coord = coord

    # -- Token Sampling --

    def _sample(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
    ) -> torch.Tensor:
        c = self._coord
        c._token_gen.tokenizer = c.tokenizer
        request_id = _current_request_id_ctx.get()
        if request_id is not None:
            params = c._param_update_channel.get(request_id)
            if params is not None:
                temperature = params.temperature
                top_p = params.top_p
                top_k = params.top_k
        return c._token_gen.sample(logits, temperature=temperature, top_p=top_p, top_k=top_k)

    def _sample_batch(self, logits: torch.Tensor, batch: ScheduledBatch) -> torch.Tensor:
        c = self._coord
        c._token_gen.tokenizer = c.tokenizer
        for seq in batch.sequences:
            params = c._param_update_channel.get(seq.request_id)
            if params is not None:
                seq.temperature = params.temperature
                seq.top_p = params.top_p
                seq.top_k = params.top_k
                seq.include_logprobs = params.include_logprobs
                seq.top_logprobs = params.top_logprobs
                seq.logit_bias = params.logit_bias
                seq.presence_penalty = params.presence_penalty
                seq.frequency_penalty = params.frequency_penalty
        next_tokens, logprobs_list = c._token_gen.sample_batch(
            logits, batch.sequences, tokenizer=c.tokenizer
        )
        if logprobs_list:
            for i, seq in enumerate(batch.sequences):
                if i < len(logprobs_list) and logprobs_list[i] is not None:
                    if not hasattr(seq, '_collected_logprobs'):
                        seq._collected_logprobs = []
                    seq._collected_logprobs.append(logprobs_list[i])
                    token_id = next_tokens[i].item() if next_tokens.dim() > 0 else next_tokens.item()
                    seq.token_counts[token_id] = seq.token_counts.get(token_id, 0) + 1
        return next_tokens

    # -- Speculative Decode Helpers --

    @staticmethod
    def _speculative_tokens_to_append(
        draft_tokens: torch.Tensor | list[int],
        target_logits: torch.Tensor,
        accepted_count: int,
        accepted_tokens: list[int],
        next_token: int,
    ) -> list[int]:
        tokens = list(accepted_tokens)
        draft_len = int(draft_tokens.numel()) if isinstance(draft_tokens, torch.Tensor) else len(draft_tokens)
        if target_logits.dim() == 3:
            verification_steps = target_logits.shape[1]
        elif target_logits.dim() == 2:
            verification_steps = target_logits.shape[0]
        else:
            verification_steps = 1
        if accepted_count >= draft_len and verification_steps > accepted_count and next_token >= 0:
            tokens.append(next_token)
        elif not tokens and next_token >= 0:
            tokens.append(next_token)
        return tokens

    # -- Generation (distributed and local) --

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        request_id: str | None = None,
        user_id: str = "default",
        speculative_config: dict[str, Any] | None = None,
    ) -> str:
        c = self._coord
        if not c.node_order and c.local_partitioner is None:
            raise NodeError("No nodes registered and no local model loaded")

        if c.model_info:
            max_pos = c.model_info.get("max_position_embeddings", 4096)
            if max_new_tokens >= max_pos:
                logger.warning(
                    f"max_new_tokens={max_new_tokens} exceeds model context window ({max_pos}). "
                    f"Capping to {max_pos - 1}."
                )
                max_new_tokens = max_pos - 1

        request_id = request_id or str(uuid.uuid4())
        c._param_update_channel.register(request_id)
        token = _current_request_id_ctx.set(request_id)
        req_log = logger.bind(request_id=request_id, mode="distributed" if c.node_order else "local")

        c.record_metric("total_requests", 1)
        start_time = time.time()

        if c._rate_limiter is not None:
            allowed = c._rate_limiter.check(f"user:{user_id}", 1.0)
            if not allowed:
                c._request_tracker.set_result(request_id, "[Rate limited]")
                c.record_metric("rate_limited", 1)
                raise NodeError("Rate limit exceeded")

        fp_fprint = None
        fp_params_dict = {"max_new_tokens": max_new_tokens, "temperature": temperature, "top_p": top_p, "top_k": top_k}
        if c._request_fingerprinter is not None:
            fp_fprint = c._request_fingerprinter.fingerprint(prompt, fp_params_dict)
            if c._request_fingerprinter.is_in_flight(fp_fprint):
                cached = c._request_fingerprinter.wait_for_result(fp_fprint)
                if cached is not None:
                    return cached
            c._request_fingerprinter.mark_in_flight(fp_fprint, request_id)
            c._request_fingerprinter.store(fp_fprint, request_id, "", prompt=prompt, params=fp_params_dict)

        if c._prompt_cache_service is not None:
            cache_hit = c._prompt_cache_service.lookup(prompt, model=c.model_name, params={"max_new_tokens": max_new_tokens, "temperature": temperature})
            if cache_hit is not None:
                result = cache_hit.response
                if c._request_auditor is not None:
                    c._request_auditor.record(request_id=request_id, prompt=prompt, response=result, model=c.model_name, duration_ms=0.0, status="cache_hit")
                if c._request_fingerprinter is not None and fp_fprint is not None:
                    c._request_fingerprinter.clear_in_flight(fp_fprint, request_id)
                return result

        if c._graceful_degradation is not None:
            queue_depth = c.scheduler.pending_count if c.scheduler else 0
            load = LoadSnapshot(queue_depth=queue_depth)
            degradation_plan = c._graceful_degradation.evaluate(load)
            if degradation_plan.level.value >= 3:
                partial = c._graceful_degradation.get_partial_response(request_id)
                result = partial["choices"][0]["text"] if partial["choices"] else ""
                if c._request_auditor is not None:
                    c._request_auditor.record(request_id=request_id, prompt=prompt, response=result, model=c.model_name, duration_ms=0.0, status="degraded")
                if c._request_fingerprinter is not None and fp_fprint is not None:
                    c._request_fingerprinter.clear_in_flight(fp_fprint, request_id)
                return result

        if c._request_auditor is not None:
            c._request_auditor.record(request_id=request_id, prompt=prompt, model=c.model_name, status="processing")

        prompt_len = len(c.tokenizer.encode(prompt)) if c.tokenizer else 0

        # Classify workload for auto-speculative method selection
        spec_decoder = c._model_svc.get_spec_decoder() if hasattr(c, '_model_svc') else c._spec_decoder
        if spec_decoder is not None and spec_decoder.method == "auto":
            spec_decoder.set_workload_type(prompt)

        try:
            if c.tokenizer is None:
                raise ValueError("Tokenizer not loaded")
            input_ids = c.tokenizer.encode(prompt, return_tensors="pt")

            if c.node_order:
                req_log.info(f"Starting distributed generation: {max_new_tokens} tokens max")
                input_ids = input_ids.to("cpu")
                prompt_len = input_ids.shape[1]
                total_capacity = min(prompt_len + max_new_tokens, c.config.max_context_length)
                generated_ids = torch.zeros(1, total_capacity, dtype=torch.long)
                generated_ids[:, :prompt_len] = input_ids
                gen_pos = prompt_len

                node_kv_caches = c._pipeline.create_node_kv_caches()

                predictive_cache = c._predictive_cache
                if predictive_cache is not None:
                    token_ids = input_ids[0].tolist()
                    predictions = predictive_cache.observe_request(token_ids)
                    for pred in predictions:
                        if pred.should_prefetch and pred.confidence > 0.3:
                            c._cache_mgr.lookup_prefix(list(pred.prefix_tokens))

                active_method = c._spec_decoder.get_active_method(c.draft_model) if c._spec_decoder else None
                use_speculative = c._spec_decoder is not None and c._spec_decoder.is_enabled

                if use_speculative:
                    draft_tokens_count = c.num_assistant_tokens
                    req_log.info(f"Speculative decoding enabled ({active_method}): {draft_tokens_count} draft tokens")

                debug_on = is_debug_mode()
                step = 0
                while step < max_new_tokens:
                    if gen_pos == prompt_len:
                        step_input = generated_ids[:, :gen_pos]
                    else:
                        step_input = generated_ids[:, gen_pos-1:gen_pos]

                    draft_tokens = None
                    draft_logits = None
                    if use_speculative:
                        active_method = c._spec_decoder.get_active_method(c.draft_model)
                        if active_method == "draft_model" and c.draft_model is not None:
                            draft_tokens, _, draft_logits = c._spec_decoder.generate_draft_tokens(c.draft_model, step_input)
                        elif active_method == "ngram":
                            generated_list = generated_ids[0].tolist() if generated_ids.dim() == 2 else generated_ids.tolist()
                            draft_tokens, _, _ = c._spec_decoder.generate_draft_tokens(None, step_input, generated_ids=generated_list)

                    zc_engine = c._zero_copy_engine
                    zc_input = step_input
                    if zc_engine is not None and step_input.is_cuda and c.node_order:
                        for nid in c.node_order:
                            zc_engine.send(nid, step_input, peer_is_local=False, tag=f"input_{request_id}")

                    hybrid_executor = c._hybrid_parallel_executor
                    if hybrid_executor is not None:
                        logits = hybrid_executor.execute(zc_input, node_kv_caches, request_id=request_id, draft_tokens=draft_tokens if use_speculative else None)
                    else:
                        logits = c._pipeline.run_pipeline(zc_input, node_kv_caches, request_id=request_id, draft_tokens=draft_tokens if use_speculative else None)

                    if use_speculative and active_method == "medusa":
                        draft_tokens, _, _ = c._spec_decoder.generate_draft_tokens(None, step_input, target_logits=logits)

                    if use_speculative and draft_tokens:
                        accepted_count, accepted_tokens, next_token = c._spec_decoder.verify_and_accept(draft_tokens, logits, c.tokenizer, temperature=temperature, draft_logits=draft_logits)
                        if c._continuous_trainer is not None and accepted_tokens:
                            draft_ids = draft_tokens.tolist() if hasattr(draft_tokens, 'tolist') else list(draft_tokens) if draft_tokens else []
                            c._continuous_trainer.record(draft_ids, list(accepted_tokens))
                        tokens_to_append = self._speculative_tokens_to_append(draft_tokens, logits, accepted_count, accepted_tokens, next_token)
                        tokens_to_append = tokens_to_append[: max_new_tokens - step]
                        hit_eos = False
                        for token_id in tokens_to_append:
                            generated_ids[:, gen_pos] = token_id
                            gen_pos += 1
                            if token_id == c.tokenizer.eos_token_id:
                                hit_eos = True
                                break
                        if hit_eos:
                            break
                        step += len(tokens_to_append)
                        c._spec_decoder.record_generated_tokens(generated_ids[0, :gen_pos].tolist())
                    else:
                        next_token = self._sample(logits[:, -1, :], temperature=temperature, top_p=top_p, top_k=top_k)
                        generated_ids[:, gen_pos] = next_token.item()
                        gen_pos += 1
                        step += 1
                        if next_token.item() == c.tokenizer.eos_token_id:
                            break
                        if c._spec_decoder:
                            c._spec_decoder.record_generated_tokens(generated_ids[0, :gen_pos].tolist())

                result = c.tokenizer.decode(generated_ids[0, :gen_pos], skip_special_tokens=True)
                tokens_generated = gen_pos - prompt_len
            else:
                req_log.info(f"Starting local generation: {max_new_tokens} tokens max")
                model_device = next(c.local_partitioner.full_model.parameters()).device
                input_ids = input_ids.to(model_device)

                if c._spec_decoder is not None and c._spec_method == "eagle" and c._spec_decoder.has_eagle_heads:
                    result = self._generate_local_eagle_sync(input_ids, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, top_k=top_k)
                    tokens_generated = len(c.tokenizer.encode(result)) - prompt_len
                    elapsed = time.time() - start_time
                    c.record_metric("total_tokens_generated", tokens_generated)
                    c.record_metric("total_generation_time", elapsed)
                    req_log.info(f"Generated {tokens_generated} tokens in {elapsed:.2f}s ({tokens_generated/max(elapsed, 0.001):.1f} tok/s)")
                    return result

                gen_kwargs = {
                    "max_new_tokens": max_new_tokens, "temperature": temperature, "top_p": top_p,
                    "do_sample": temperature > 0, "pad_token_id": c.tokenizer.eos_token_id,
                }
                if c.draft_model is not None:
                    gen_kwargs["assistant_model"] = c.draft_model
                    gen_kwargs["num_assistant_tokens"] = c.num_assistant_tokens
                    req_log.info(f"Speculative decoding enabled with {c.num_assistant_tokens} assistant tokens")

                with torch.no_grad():
                    output = c.local_partitioner.full_model.generate(input_ids, **gen_kwargs)
                result = c.tokenizer.decode(output[0], skip_special_tokens=True)
                tokens_generated = output.shape[1] - input_ids.shape[1]

            elapsed = time.time() - start_time

            if c._self_optimizing:
                c._self_optimizing.record_operation("decode", duration_ms=elapsed * 1000, batch_size=1, seq_len=prompt_len + tokens_generated)

            c.record_metric("total_tokens_generated", tokens_generated)
            c.record_metric("total_generation_time", elapsed)

            if c.metrics_exporter:
                c.metrics_exporter.tokens_generated.inc(tokens_generated)
                c.metrics_exporter.token_latency.observe(elapsed)
                if elapsed > 0:
                    c.metrics_exporter.tokens_per_second.set(tokens_generated / elapsed)

            req_log.info(f"Generated {tokens_generated} tokens in {elapsed:.2f}s ({tokens_generated/max(elapsed, 0.001):.1f} tok/s)")

            if c._request_auditor is not None:
                c._request_auditor.update_response(request_id=request_id, response=result, duration_ms=elapsed * 1000, status="success")

            if c._request_fingerprinter is not None and fp_fprint is not None:
                c._request_fingerprinter.store(fp_fprint, request_id, result, prompt=prompt, params=fp_params_dict)
                c._request_fingerprinter.clear_in_flight(fp_fprint, request_id)

            return result

        except (NodeUnreachableError, OOMError, GRPCTimeoutError, NodeError) as e:
            c.record_metric("errors", 1)
            if c.metrics_exporter:
                c.metrics_exporter.errors_total.labels(type=type(e).__name__).inc()
            raise
        except Exception as e:
            c.record_metric("errors", 1)
            if c.metrics_exporter:
                c.metrics_exporter.errors_total.labels(type=type(e).__name__).inc()
            raise
        finally:
            c._param_update_channel.unregister(request_id)
            _current_request_id_ctx.reset(token)

    def generate_async(
        self,
        prompt: str,
        request_id: str | None = None,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        schema: dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
        priority: int = 2,
        adapter_id: str | None = None,
        include_logprobs: bool = False,
        top_logprobs: int = 0,
        logit_bias: dict[str, float] | None = None,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        max_latency_ms: float | None = None,
        user_id: str = "default",
    ) -> str:
        c = self._coord
        if c.scheduler is None:
            raise BatchError("Batch scheduler not configured. Use generate() instead.")

        if c.model_info:
            max_pos = c.model_info.get("max_position_embeddings", 4096)
            if max_new_tokens >= max_pos:
                max_new_tokens = max_pos - 1

        request_id = request_id or str(uuid.uuid4())
        c._param_update_channel.register(request_id)
        token = _current_request_id_ctx.set(request_id)

        c.record_metric("total_requests", 1)

        if c._rate_limiter is not None:
            allowed = c._rate_limiter.check(f"user:{user_id}", 1.0)
            if not allowed:
                c._request_tracker.set_result(request_id, "[Rate limited]")
                c.record_metric("rate_limited", 1)
                raise NodeError("Rate limit exceeded")

        input_ids = c.tokenizer.encode(prompt, return_tensors="pt")
        prefix_match_len = 0
        if c.prefix_cache:
            prefix_match_len, _ = c._cache_mgr.lookup_prefix(input_ids)

        constraint = None
        if response_format:
            constraint = SchemaConstrainedDecoder.from_response_format(
                response_format, tokenizer=c.tokenizer
            )
            if constraint is None:
                constraint = JSONSchemaConstraint.from_response_format(
                    response_format, tokenizer=c.tokenizer
                )
        elif schema:
            constraint = JSONSchemaConstraint(schema=schema)

        chunk_state = c._cache_mgr.maybe_chunk(input_ids)

        seq = Sequence(
            request_id=request_id, prompt_tokens=input_ids, max_new_tokens=max_new_tokens,
            temperature=temperature, top_p=top_p, top_k=top_k, constraint=constraint,
            prefix_match_len=prefix_match_len, priority=priority, adapter_id=adapter_id,
            include_logprobs=include_logprobs, top_logprobs=top_logprobs,
            logit_bias=logit_bias, presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty, max_latency_ms=max_latency_ms,
        )
        if chunk_state:
            seq.chunk_state = chunk_state
        if c.tokenizer.eos_token_id is not None:
            seq.stop_token_ids = [c.tokenizer.eos_token_id]

        c.scheduler.add(seq)
        c._batch_event.set()
        c._request_tracker.register_request(request_id)
        c.record_metric("total_requests", 1)
        return request_id

    def wait_for_result(self, request_id: str, timeout: float = 120.0) -> str:
        return self._coord._request_tracker.wait_for_result(request_id, timeout)

    def get_logprobs(self, request_id: str) -> dict[str, Any] | None:
        return self._coord._request_tracker.get_logprobs(request_id)

    # -- Batch Generation Loop --

    def generate_batch(self, timeout: float = 120.0, max_steps: int = 0) -> None:
        c = self._coord
        if c.scheduler is None:
            raise BatchError("Batch scheduler not configured. Use generate() instead.")

        step = 0
        idle_time = 0.0

        while c.scheduler.has_pending:
            if c._preemption_policy is not None:
                if c.scheduler._latency_tracker is not None:
                    for seq in list(c.scheduler.active):
                        info = c.scheduler._latency_tracker._requests.get(seq.request_id)
                        if info is not None and info.elapsed_ms > 0:
                            if info.elapsed_ms > info.sla_target_ms * 0.8:
                                logger.debug(f"SLA urgency preemption for {seq.request_id}: {info.elapsed_ms:.0f}/{info.sla_target_ms:.0f} ms")
                                preempted = c.scheduler.preempt_lowest(min_priority=seq.priority + 1, kv_cache_state=c._batch_kv_caches)
                                if preempted is not None:
                                    c._preemption_policy.create_checkpoint(request_id=preempted.request_id, kv_cache=preempted.generated_tokens, sequence=preempted)

                should_preempt = c._preemption_policy.should_preempt(pending_count=c.scheduler.pending_count)
                if should_preempt and c.scheduler.active_count > 0:
                    preempted_seq = c.scheduler.preempt_lowest(min_priority=3, kv_cache_state=c._batch_kv_caches)
                    if preempted_seq is not None:
                        with c._batch_kv_caches_lock:
                            node_kv = c._batch_kv_caches.get(preempted_seq.request_id)
                        if node_kv:
                            all_kv = []
                            for _nid, kv_list in node_kv.items():
                                if kv_list is not None:
                                    all_kv.extend(kv_list)
                            if all_kv:
                                c._preemption_policy.create_checkpoint(request_id=preempted_seq.request_id, kv_cache=all_kv, sequence=preempted_seq)
                            with c._batch_kv_caches_lock:
                                c._batch_kv_caches.pop(preempted_seq.request_id, None)
                        logger.info(f"Preempted {preempted_seq.request_id} (priority={preempted_seq.priority}, generated={len(preempted_seq.generated_tokens)} tokens)")
                        continue

                if c.scheduler.get_preempted_count() > 0:
                    restored = c.scheduler.restore_preempted(kv_cache_state=c._batch_kv_caches)
                    for seq in restored:
                        checkpoint = c._preemption_policy.restore_checkpoint(seq.request_id)
                        if checkpoint is not None:
                            seq.generated_tokens = list(checkpoint.generated_tokens)
                            logger.info(f"Restored {seq.request_id} ({len(checkpoint.generated_tokens)} tokens)")

            batch = c.scheduler.schedule()
            if batch is None:
                c._batch_event.wait(timeout=0.01)
                c._batch_event.clear()
                idle_time += 0.01
                if idle_time > timeout:
                    logger.warning(f"Batch scheduler idle timeout ({timeout}s) exceeded. Completing {c.scheduler.pending_count} pending requests with error.")
                    pending_seqs = [s for _, _, s in c.scheduler._pending_heap]
                    c._request_tracker.complete_batch_requests(c.scheduler.active, pending_seqs, c.tokenizer)
                    with c._request_tracker._lock:
                        for seq in pending_seqs:
                            c._request_tracker._results[seq.request_id] = "[Error: Request timed out in scheduler]"
                            event = c._request_tracker._events.pop(seq.request_id, None)
                            if event:
                                event.set()
                        c.scheduler._pending_heap.clear()
                    break
                continue

            idle_time = 0.0
            batch_request_ids = [seq.request_id for seq in batch.sequences]
            try:
                if c.local_partitioner is not None:
                    self._generate_local_batch(batch)
                else:
                    self._run_distributed_pipeline_batch(batch)
            except Exception:
                with c._batch_kv_caches_lock:
                    for rid in batch_request_ids:
                        c._batch_kv_caches.pop(rid, None)
                raise

            step += 1
            if max_steps > 0 and step >= max_steps:
                break

        if c.scheduler is not None:
            c._request_tracker.complete_batch_requests(c.scheduler.active, [s for _, _, s in c.scheduler._pending_heap], c.tokenizer)
            with c._batch_kv_caches_lock:
                for rid in list(c._batch_kv_caches.keys()):
                    if rid not in c.scheduler.active:
                        c._batch_kv_caches.pop(rid, None)

    def _generate_local_batch(self, batch: ScheduledBatch) -> None:
        c = self._coord
        batch_size = batch.batch_size
        device = next(c.local_partitioner.full_model.parameters()).device

        if c._cache_mgr.prefix_cache is not None:
            for seq in batch.sequences:
                if seq.prefix_match_len == 0 and len(seq.generated_tokens) == 0:
                    match_len, _ = c._cache_mgr.lookup_prefix(seq.prompt_tokens)
                    if match_len > 0:
                        seq.prefix_match_len = match_len
                        logger.debug(f"Radix cache hit for {seq.request_id}: {match_len} tokens")

        input_ids, _, seq_starts, seq_lengths, position_offsets, _ = batch.build_inputs()
        attention_mask = batch.build_attention_mask()

        if c._spec_decoder is not None and c._spec_decoder.is_enabled:
            seq_inputs = [input_ids[:, start:start + length] for start, length in zip(seq_starts, seq_lengths, strict=False)]
            draft_tokens_list, _ = c._spec_decoder.generate_batch_draft_tokens(c.draft_model, seq_inputs)
            with torch.no_grad():
                outputs = c.local_partitioner.full_model(input_ids=input_ids, attention_mask=attention_mask)
                all_logits = outputs.logits
            target_logits_list = [all_logits[i:i+1] for i in range(batch.batch_size)]
            results = c._spec_decoder.verify_batch(draft_tokens_list=draft_tokens_list, target_logits_list=target_logits_list, tokenizer=c.tokenizer)
            if c._continuous_trainer is not None and draft_tokens_list:
                for idx, (_, accepted, _) in enumerate(results):
                    if accepted and idx < len(draft_tokens_list):
                        dt = draft_tokens_list[idx]
                        draft_ids = dt.tolist() if hasattr(dt, 'tolist') else list(dt) if dt else []
                        if draft_ids:
                            c._continuous_trainer.record(draft_ids, list(accepted))
            next_tokens = torch.tensor([r[2] for r in results], device=device)
        else:
            with torch.no_grad():
                outputs = c.local_partitioner.full_model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits[:, -1, :]
            next_tokens = self._sample_batch(logits, batch)

        with c._batch_kv_caches_lock:
            kv_copy = dict(c._batch_kv_caches)
        decoded = [c.tokenizer.decode([int(next_tokens[i])]) if batch.sequences[i].constraint is not None else None for i in range(len(batch.sequences))]
        c.scheduler.step(batch, next_tokens, kv_caches=kv_copy, decoded_tokens=decoded)

    def _run_distributed_pipeline_batch(self, batch: ScheduledBatch) -> None:
        c = self._coord
        next_tokens = []

        if c._cache_mgr.prefix_cache is not None:
            for seq in batch.sequences:
                if seq.prefix_match_len == 0 and len(seq.generated_tokens) == 0:
                    match_len, _ = c._cache_mgr.lookup_prefix(seq.prompt_tokens)
                    if match_len > 0:
                        seq.prefix_match_len = match_len
                        logger.debug(f"Radix cache hit for {seq.request_id}: {match_len} tokens")

        if c._async_pipeline is not None and c._pipeline_schedule_type in ("1f1b", "interleaved"):
            self._run_async_pipeline_batch(batch)
            return

        for seq_idx, seq in enumerate(batch.sequences):
            if batch.is_prefill[seq_idx]:
                start = seq.prefix_match_len
                tokens = seq.prompt_tokens[start:]
            else:
                tokens = [seq.decode_input_token]

            input_ids = torch.tensor([tokens], dtype=torch.long)

            with c._batch_kv_caches_lock:
                if seq.request_id in c._batch_kv_caches:
                    node_kv_caches = c._batch_kv_caches[seq.request_id]
                else:
                    node_kv_caches = c._pipeline.create_node_kv_caches()
                    c._batch_kv_caches[seq.request_id] = node_kv_caches

            use_spec = c._spec_decoder is not None and c._spec_decoder.is_enabled and c.draft_model is not None and not batch.is_prefill[seq_idx]

            if use_spec:
                draft_tokens, _, draft_logits = c._spec_decoder.generate_draft_tokens(c.draft_model, input_ids)
                if c._pipeline.enable_overlap:
                    logits = c._pipeline.run_pipeline_overlap(input_ids, node_kv_caches, request_id=seq.request_id)
                else:
                    logits = c._pipeline.run_pipeline(input_ids, node_kv_caches, request_id=seq.request_id)
                _, accepted, next_token = c._spec_decoder.verify_and_accept(draft_tokens, logits, c.tokenizer, temperature=seq.temperature, draft_logits=draft_logits)
                if c._continuous_trainer is not None and accepted:
                    draft_ids = draft_tokens.tolist() if hasattr(draft_tokens, 'tolist') else list(draft_tokens) if draft_tokens else []
                    c._continuous_trainer.record(draft_ids, list(accepted))
                next_tokens.append(torch.tensor([next_token], dtype=torch.long))
            else:
                if c._pipeline.enable_overlap:
                    logits = c._pipeline.run_pipeline_overlap(input_ids, node_kv_caches, request_id=seq.request_id)
                else:
                    logits = c._pipeline.run_pipeline(input_ids, node_kv_caches, request_id=seq.request_id)
                seq_logits = logits[:, -1, :]
                if seq.constraint is not None:
                    mask = seq.constraint.get_logits_mask(seq_logits.shape[-1], c.tokenizer)
                    seq_logits = seq_logits.masked_fill(~mask, float('-inf'))
                token = self._sample(seq_logits, temperature=seq.temperature, top_p=seq.top_p, top_k=seq.top_k)
                next_tokens.append(token)

        next_tokens_tensor = torch.stack(next_tokens).squeeze(-1)
        decoded = [c.tokenizer.decode([int(next_tokens[i])]) if batch.sequences[i].constraint is not None else None for i in range(len(batch.sequences))]
        c.scheduler.step(batch, next_tokens_tensor, kv_caches=dict(c._batch_kv_caches), decoded_tokens=decoded)

    def _run_async_pipeline_batch(self, batch: ScheduledBatch) -> None:
        c = self._coord
        if c.tokenizer is None:
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

        use_spec = c._spec_decoder is not None and c._spec_decoder.is_enabled and c.draft_model is not None and not all(batch.is_prefill)

        draft_tokens_list = None
        draft_logits_list = None
        if use_spec:
            draft_tokens_list, _, draft_logits_list = c._spec_decoder.generate_batch_draft_tokens(c.draft_model, input_tensors)

        def stage_forward(micro_batch):
            stage_logits = c._async_pipeline.forward_stage(micro_batch, None)
            return stage_logits

        outputs = c._async_pipeline.run_batch(batch_input)
        last_logits = outputs[:, -1, :] if outputs.dim() == 3 else outputs

        if use_spec and draft_tokens_list:
            results = c._spec_decoder.verify_batch(draft_tokens_list=draft_tokens_list, target_logits_list=[last_logits[i:i+1] for i in range(batch.batch_size)], tokenizer=c.tokenizer, draft_logits_list=draft_logits_list)
            if c._continuous_trainer is not None:
                for idx, (_, accepted, _) in enumerate(results):
                    if accepted and idx < len(draft_tokens_list):
                        dt = draft_tokens_list[idx]
                        if dt:
                            c._continuous_trainer.record(
                                dt.tolist() if hasattr(dt, 'tolist') else list(dt),
                                list(accepted),
                            )
            next_tokens = torch.tensor([r[2] for r in results], device=last_logits.device)
        else:
            next_tokens = self._sample_batch(last_logits, batch)

        with c._batch_kv_caches_lock:
            kv_copy = dict(c._batch_kv_caches)
        decoded = [c.tokenizer.decode([int(next_tokens[i])]) if batch.sequences[i].constraint is not None else None for i in range(len(batch.sequences))]
        c.scheduler.step(batch, next_tokens, kv_caches=kv_copy, decoded_tokens=decoded)

    def _generate_local_sync(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        c = self._coord
        input_ids = c.tokenizer.encode(prompt, return_tensors="pt")
        model_device = next(c.local_partitioner.full_model.parameters()).device
        input_ids = input_ids.to(model_device)

        if c._spec_decoder is not None and c._spec_method == "eagle" and c._spec_decoder.has_eagle_heads:
            return self._generate_local_eagle_sync(input_ids, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p)

        gen_kwargs = {
            "max_new_tokens": max_new_tokens, "temperature": temperature, "top_p": top_p,
            "do_sample": temperature > 0, "pad_token_id": c.tokenizer.eos_token_id,
        }
        if c.draft_model is not None:
            gen_kwargs["assistant_model"] = c.draft_model
            gen_kwargs["num_assistant_tokens"] = c.num_assistant_tokens

        with torch.no_grad():
            output = c.local_partitioner.full_model.generate(input_ids, **gen_kwargs)
        return c.tokenizer.decode(output[0], skip_special_tokens=True)

    def _generate_local_eagle_sync(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int = 0,
    ) -> str:
        c = self._coord
        if c._spec_decoder is None:
            raise RuntimeError("Speculative decoder is not configured")

        model = c.local_partitioner.full_model
        prompt_len = input_ids.shape[1]
        total_capacity = prompt_len + max_new_tokens
        generated_ids = torch.empty(
            (input_ids.shape[0], total_capacity),
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        generated_ids[:, :prompt_len] = input_ids
        gen_pos = prompt_len
        tokens_generated = 0

        while tokens_generated < max_new_tokens:
            current_ids = generated_ids[:, :gen_pos]
            with torch.no_grad():
                outputs = model(input_ids=current_ids, output_hidden_states=True, use_cache=False)

            hidden_states = getattr(outputs, "hidden_states", None)
            hidden_states = hidden_states[-1] if hidden_states else None
            draft_tokens, _, _ = c._spec_decoder.generate_draft_tokens(None, current_ids[:, -1:], hidden_states=hidden_states)

            if draft_tokens:
                draft_tensor = torch.tensor([draft_tokens], device=generated_ids.device, dtype=generated_ids.dtype)
                verify_input = torch.empty(
                    (current_ids.shape[0], gen_pos + len(draft_tokens)),
                    device=generated_ids.device,
                    dtype=generated_ids.dtype,
                )
                verify_input[:, :gen_pos] = current_ids
                verify_input[:, gen_pos:gen_pos + len(draft_tokens)] = draft_tensor
                with torch.no_grad():
                    verify_outputs = model(input_ids=verify_input, use_cache=False)
                target_logits = verify_outputs.logits[:, gen_pos - 1:, :]
                accepted_count, accepted_tokens, next_token = c._spec_decoder.verify_and_accept(draft_tokens=draft_tokens, target_logits=target_logits, tokenizer=c.tokenizer, temperature=temperature)
                if c._continuous_trainer is not None and accepted_tokens:
                    draft_ids = draft_tokens.tolist() if hasattr(draft_tokens, 'tolist') else list(draft_tokens) if draft_tokens else []
                    c._continuous_trainer.record(draft_ids, list(accepted_tokens))
                tokens_to_append = self._speculative_tokens_to_append(draft_tokens, target_logits, accepted_count, accepted_tokens, next_token)
            else:
                next_token_tensor = self._sample(outputs.logits[:, -1, :], temperature=temperature, top_p=top_p, top_k=top_k)
                tokens_to_append = [int(next_token_tensor.item())]

            tokens_to_append = tokens_to_append[: max_new_tokens - tokens_generated]
            if not tokens_to_append:
                break

            append_len = len(tokens_to_append)
            next_tensor = torch.tensor([tokens_to_append], device=generated_ids.device, dtype=generated_ids.dtype)
            generated_ids[:, gen_pos:gen_pos + append_len] = next_tensor
            gen_pos += append_len
            tokens_generated += append_len
            c._spec_decoder.record_generated_tokens(generated_ids[0, :gen_pos].tolist())

            if any(token_id == c.tokenizer.eos_token_id for token_id in tokens_to_append):
                break

        return c.tokenizer.decode(generated_ids[0, :gen_pos], skip_special_tokens=True)
