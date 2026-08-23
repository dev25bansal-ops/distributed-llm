---
tags:
  - core
  - audit
  - issues
date: 2026-08-05
---

# Core Audit 02 — Issues & Required Fixes

**← [[Core Comprehensive Audit 2026-08-05]]**
Full catalog of what is broken or insufficient in `src/distllm/core/**`. Each issue: severity, priority, effort, dependencies, timeline, and (for bugs) repro + expected-vs-actual.

Severity scale: **Critical** = blocks start / corrupts data / false security guarantee; **High** = load-bearing feature silently broken or dead; Medium/Low = correctness/quality/tech-debt.

---

## Critical (fix first)

### C1 — Coordinator cannot start: `_subsystem_mgr` read before assignment  — `coordinator.py:126`  ✅ FIXED (2026-08-05; regression test: `test_coordinator_startup_order.py`)
- **Repro:** `Coordinator(config=CoordinatorConfig(model_name=''))` → `AttributeError: 'Coordinator' object has no attribute '_subsystem_mgr'`.
- **Expected:** Coordinator constructs and the platform boots. **Actual:** `self._subsystem_mgr` is only assigned at `coordinator.py:217`, but line 126 passes `on_straggler_cb=self._subsystem_mgr._on_straggler_detected`. Every server/CLI `distllm-coordinator` boot hits this. Tests pass **only** because `tests/conftest.py` injects a class-level `MagicMock`.
- **Impact:** Total outage — no coordinator, no API server can start.
- **Effort:** <1h · **Timeline:** release-blocker · **Dep:** none
- **Fix:** Move `self._subsystem_mgr = SubsystemManager(self)` above line 124, or pass a lazy callback `lambda r: self._subsystem_mgr._on_straggler_detected(r)`; then delete the conftest masking workaround so a true regression test runs. ✅ *Verified by this audit (direct read).*

### C2 — Quantized-KV path crashes: import of deleted functions  — `kv_cache.py:683` / `quantization_selector.py`  ✅ FIXED (2026-08-05; functions restored in `quantization_selector.py`; test: `test_kv_cache_quantized_path.py`)
- **Repro:** Enable KV quantization → call `append()`/`get_all()` → `ImportError: cannot import name 'apply_kv_cache_quantization'`.
- **Expected:** FP8/INT KV quantize + dequantize. **Actual:** `quantization_selector.py` was refactored to a pure selector and now defines only `QuantizationChoice`/`QuantizationSelector`; `kv_cache.py:683` still does `from distllm.core.quantization_selector import (apply_kv_cache_quantization, dequantize_kv_cache)`, neither of which exists. The documented feature ships green because no test enables the quantized path.
- **Effort:** 1–3h · **Timeline:** release-blocker · **Dep:** none
- **Fix:** Restore/re-export `apply_kv_cache_quantization(qk,sk,bits)` + `dequantize_kv_cache(...)` (from `kv_cache` or a new quant module) and add a test toggling `_quantized=True`. ✅ *Verified by this audit (direct read).*

### C3 — Structured output emits empty/invalid JSON on first call  — `structured_output/__init__.py:92`  ✅ FIXED (2026-08-05; synchronous token-index build + no-op-mask fallback; test: `test_structured_output_first_call.py`)
- **Repro:** First `json_schema`/`json_object` request after a restart → generation terminates immediately on EOS with no output.
- **Expected:** Valid constrained JSON. **Actual:** `_build_token_index()` spawns a daemon thread and synchronously `return {}`. `get_logits_mask()` consumes the empty index → every token forbidden except `eos_token_id` → the model emits EOS at step 1. Self-heals only after the async 32k-token decode build lands. Wired into `request_pipeline`, `token_generator`, and the API — user-visible.
- **Effort:** 1–2h · **Timeline:** release-blocker · **Dep:** none
- **Fix:** Build the token index synchronously when empty (or await via a barrier) before computing the mask; fall back to all-True mask if `first_chars` is empty rather than all-False. ✅ *Verified by this audit (direct read).*

### C4 — Pipelined speculative verifier never runs; every draft accepted unverified  — `async_pipelined_speculative.py:407`  ✅ FIXED (2026-08-05; verifier wired + fail-safe rejection + progress fallback; test: `test_async_pipelined_speculative_verify.py`)
- **Repro:** Decoder with a verifier that rejects all drafts still accepts 100% (`verify_calls` stays 0).
- **Expected:** Verifier gates acceptance. **Actual:** slots are enqueued with `accepted=None` and `_collect_verifications` accepts any slot whose `accepted is not False` — the verifier (`_verify_worker`/`_verifier_pool`) is never invoked. Silent wrong-output risk.
- **Effort:** 3–5h · **Timeline:** next sprint · **Dep:** none
- **Fix:** Submit each popped slot to `_verifier_pool.submit(_verify_worker, slot)` and honor the returned flag; never treat `accepted is None` as True.

### C5 — Go-to-market rests on unproven distributed-70B claims (strategic)  — `benchmarks/results/throughput-dist.json`
- **Summary:** "Pool GPUs to run 70B" has **zero measured datapoints** — the stored JSON is `nodes:1 / tokens_per_sec:80 / total_tokens:0 / samples:0`, and the sole real number is one single-node RTX-5060 TinyLlama row. Marketing claims distributed 70B with no distributed 70B result.
- **Impact:** Single hardest gate on downloads, stars, enterprise POCs (see [[Core Audit 01 Strategic & Opportunities]]).
- **Effort:** 1–2w · **Timeline:** release-blocker
- **Fix:** Stand up a reproducible multi-node benchmark harness (2×/3× consumer GPU, 34B/70B, hardware fingerprint, TTFT + ITL P50/P99, N≥30), wire nightly into CI, and pivot positioning to where distribution genuinely wins.

### C6 — Test suite fakes the entire package graph (test gap)  — `tests/_import_helper.py:84`
- **Summary:** 114 of 241 core test files run under a `sys.modules` **fake** for `distllm.*`, with hand-written stubs replacing real classes (e.g. `JSONSchemaConstraint` → FSM stub, `validate_structured_output` → identity). `scheduler`/`advanced_scheduling`/`structured_output`/`vectorstore` green results therefore **prove nothing about the real package**; real import bugs (the C1–C3 above) are invisible to CI.
- **Effort:** 5–10d · **Timeline:** release-blocker · **Fix:** split pure-unit (keep bootstrap) from a real-import integration suite; add a CI job for `from distllm.core.coordinator import Coordinator`.

### C7 — 78 of 287 core modules have zero test references (test gap)  — `coordinator_subsystem.py`, `evaluation/runner.py`, `dp_inference/accounting.py`, `kv_cache_paged.py`, `kraken_chaos.py`, `priority_heap.py`, …
- **Summary:** 27% of core ships with no direct or aggregate coverage, including the production coordinator start path, the eval-quality gate, and DP math.
- **Effort:** 1–2d tooling + 3–4w backfill · **Timeline:** 3–4w · **Fix:** coverage-frontier CI gate (parse import graph, fail when a core module has no test ref) + tracked frontier doc.

---

## High — bugs (confirmed; see verification ledger)

### B1 — `advanced_scheduling` policies out-of-contract with `batch_scheduler` (4 broken DI sites) — `cost_aware.py:15`  ✅ FIXED (2026-08-05; `set_nodes`/`detect_wan_mode`/constructor kwargs/`stats()` aligned; test: `test_advanced_scheduling_contract.py`; existing `advanced_scheduling/` + `test_batch_scheduler.py` suites still pass)
WAN/energy/cost/heterogeneous scheduling crash on toggle: `WANSchedulingPolicy(WANConfig(...))` → `TypeError` (no `chunk_multiplier`); `HeterogeneousBudgetComputer().set_nodes(...)` → `AttributeError` (method is `update_nodes`); no `stats()` anywhere → `scheduler.stats()` crashes. **Effort:** 3–5h. **Fix:** align constructor kwargs + method names + add `stats()`; add a contract test with the exact kwargs `batch_scheduler` passes.

### B2 — LoRA merge omits alpha/rank scaling — `aether_federated.py:277`  ✅ FIXED (2026-08-05; rank/alpha persisted at `create()`, `merge()` applies `alpha/rank`; test: `test_aether_lora_merge.py`)
`delta = b @ a.T` with no `alpha/rank` scale; `create()` persists only `(a,b)`, not rank/alpha. Merging an adapter yields unscaled deltas (wrong weights). Unwired today so blast radius limited. **Effort:** 1–2h. **Fix:** persist rank/alpha; always scale by `alpha/rank`; add regression test.

### B3 — `TieredMemoryPool` L3 round-trip corrupts KV tensors to bytes — `tiered_store.py:421`  ✅ FIXED (2026-08-05; tensors persisted with dtype/shape metadata and reconstructed on load; tests: `test_tiered_store_l3_roundtrip.py`)
NVMe demote does `data.cpu().numpy().tobytes()` (shape/dtype lost), get() returns opaque `bytes` as a "tensor". **Effort:** 2–3h. **Fix:** persist tensor metadata (shape/dtype/device) or forbid bytes in HOT/WARM tiers; add round-trip test.

### B4 — AtlasMesh/LPSolverRouter score-key mismatch silently disables multi-objective scoring — `atlas_mesh.py:1200` ✅ confirmed
`AtlasMesh._req_key` = `model:prompt:max`; `LPSolverRouter._req_key` always appends `:idx`. Every `scores.get()` misses → base score constant 0.5 in all routing modes → latency/cost/reliability/carbon scorer is dead; corrupts bandit training reward. **Effort:** 2–3h. **Fix:** one canonical key.

### B5 — AgenticRouter DPO trains on empty prompts — `agentic_router.py:499`
`PreferenceExample(prompt="", ...)` → judge fine-tunes on empty-prompt pairs; actively **degrades** the judge. Wired via `coordinator_subsystem.py:380`. **Effort:** 2h. **Fix:** thread the real prompt through `record_outcome()`; reject empty-prompt examples.

### B6 — HA standby replication is a permanent no-op — `coordinator_election.py:126`
`_is_standby` initialized `False` and **never set True** anywhere → `apply_state_snapshot()` always returns early → standby replicas never warm state → leader crash forces cold rebuild, defeating HA. **Effort:** 1–2h. **Fix:** set `_is_standby=True` when HA is enabled without leadership; regression test.

### B7 — `CarbonMigrationEngine` NameError → carbon migration never fires — `carbon_migration.py:206` ✅ confirmed
Log line uses undefined `current.gco2_per_kwh` (var is `current_gco2`); NameError is swallowed in `_monitor_loop` → migration callback never invoked, `_last_migration` stays 0. **Effort:** <1h. **Fix:** `current_gco2`.

### B8 — `CompressedSpeculativeDecoder` rejection path is a no-op — `compressed_speculative.py:274` ✅ confirmed
On rejection it re-runs the **identical** `self._target(generated, **kwargs)`; `_kv_cache`/`_re_run_threshold`/`_max_re_runs` never referenced → the "re-run with uncompressed cache" contract is unimplemented; verifier even defaults `None`. Zero correctness recovery. **Effort:** 2–4h. **Fix:** implement re-run with compression disabled (or document verifier as advisory).

### B9 — `dp_inference.generate()` else-branch UnboundLocalError — `dp_inference/__init__.py:594`
For engines exposing only `generate()` (no `generate_stream`), `text`/`token_count` are undefined in that branch → guaranteed crash on the documented fallback. **Effort:** 1–2h. **Fix:** compute them in the `generate()` branch; add fake-engine test.

### B10 — EventBus async queue written from multiple threads — `event_bus.py:250`
`asyncio.Queue.put_nowait()` called from arbitrary threads (marketplace workers) while a loop thread drains → non-thread-safe, can drop/corrupt lifecycle events. **Effort:** 1–2h. **Fix:** `call_soon_threadsafe` or a dedicated `threading.Lock`.

### B11 — HA evicts a peer forever after one transient timeout — `ha_coordinator.py:381`
`_run_election_round` does `del self._peers[pid]`; neither `handle_heartbeat_request` nor `_probe_peer` re-adds an unknown peer → a network blip permanently removes a node from quorum until manual `add_peer()`. **Effort:** 3–4h. **Fix:** mark stale peers `offline` + re-admit on inbound heartbeat.

### B12 — MemoryDefragmenter async path drops both locks — `memory_defragmenter.py:533`
`defragment_async()`/`_with_tier_async()` run `_defragment_impl` in a pool without `mgr._lock`/`self._lock` (sync `defragment()` acquires both) → concurrent compaction can corrupt the KV page table. **Effort:** 0.5–1h. **Fix:** acquire both locks in the executor path.

### B13 — ResourceManager draining-set leak after circuit-breaker cooldown — `resource_manager.py:344`
`record_failure()` adds to `_draining_nodes`; cooldown expiry allows retries but never removes the node → recovered nodes excluded indefinitely (capacity loss, skewed routing). **Effort:** 1–2h. **Fix:** discard from `_draining_nodes` on cooldown (+ clear failure counters).

### B14 — Request dedup + audit + prompt-cache silently `None` — `coordinator.py:238`
`_request_fingerprinter`/`_request_auditor`/`_prompt_cache_service` are hardcoded `None` and never constructed → ~1,000 LOC of dedup/coalescing/audit/prompt-cache is dead in the running server. **Effort:** 4–8h. **Fix:** wire behind config flags (see [[Core Audit 03 Enhancements & Modifications]]).

### B15 — `prompt_caching_service` Redis tier broken by design — `prompt_caching_service.py:184`
`initialize()` sets `_redis_available=True` but never constructs `_redis_cache` (stays `None`); `store()` calls `store_prefix` (does not exist on `RedisPromptCache`); `lookup()` reads the KV-ref second return value as the user `response`. **Effort:** 4–6h. **Fix:** construct the Redis client, use `store`/`lookup`, store response under its own key.

### B16 — `SemanticCache.invalidate()` can never remove anything — `semantic_cache.py:294`
`store`/`lookup` key by `'{scope}:{hash}'`; `invalidate()` pops bare `hash` → never matches → invalidation dead-on-arrival (stale/poisoned responses can't be purged). **Effort:** 1–2h. **Fix:** accept scope and pop the same constructed key.

### B17 — `shadow_eval_runner` uses `random` without importing it — `shadow_eval_runner.py:186`
Every `run_eval()` crashes `NameError`. (Also unwired stub returning random draws.) **Effort:** 1h. **Fix:** `import random`; wire to a real eval or delete.

### B18 — `shared_layer_pool` dedups by hashing first 1024 bytes → wrong weights shared — `shared_layer_pool.py:247`
Two layers equal in shape whose first 1KB coincide (but differ later) are declared identical → second model runs with the first's weights (CWE-345). **Effort:** 2–4h. **Fix:** hash full tensor + require `torch.equal` before pinning a shared layer.

### B19 — `MultiDraftSpeculativeDecoder._verify_tokens` off-by-one — `speculative_decoder.py:634`
`prefix_len = prefix.shape[1] - 1` reads the last prompt token's logits for the first draft token → every accept/reject decision shifted by one. **Effort:** 30–60m. **Fix:** `prefix_len = prefix.shape[1]` (matches `SpeculativeDecoder:210`); regression test.

### B20 — Telemetry deadlocks every 50th event — `telemetry.py:185`
`_add_event()` holds a non-reentrant `threading.Lock` and calls `self.flush()` at the BATCH_SIZE boundary; `flush()` re-acquires the same lock → guaranteed deadlock when telemetry enabled. **Effort:** 15–30m. **Fix:** use `RLock` or flush outside the held lock.

### B21 — Legacy `chroma/qdrant/pgvector *_store.py` import nonexistent `VectorStore` — `chroma_store.py:18`
`from distllm.core.vectorstore.base import VectorStore` — only `VectorDBInterface` exists → ImportError. (Dead legacy adapters, unreachable via factory.) **Effort:** 1h. **Fix:** delete or re-point to `VectorDBInterface`.

### B22 — `UsageMeter` record-id collision → silent underbilling — `usage_meter.py:240`
`record_id = f"usage-{int(time.time())}-{self._record_counter}"` with an in-memory counter that **resets on restart** → two records in the same second collide on the PRIMARY KEY; the `IntegrityError` is swallowed → the billable record is never persisted. **Effort:** 30m. **Timeline:** release-blocker. **Fix:** UUID or DB-persisted monotonic sequence.

### B23 — Webhook treats all HTTP <500 as success — `webhook_manager.py:296`
400/404/401/429 are recorded `success=True` → broken Slack/Discord URLs and expired tokens never retry or deactivate; alerting silently degrades. **Effort:** 1–2h. **Fix:** success only on 2xx/3xx; retry 5xx/429; treat other 4xx as terminal failure (deactivate after N).

### B24 — `resource_manager` quorum node draining leak — *see B13.*

---

## High — security

### S1 — "DP inference" never adds DP noise to output — `dp_inference/__init__.py:566`
The flagship (ε,δ)-DP guarantee for generated text is **not delivered**: `generate()` returns the plain decoded text, the module only warns "budget tracked but output tokens do not have DP guarantees". `dp_noise_injection`/`gumbel_noise_mechanism`/`_dp_sample` are never invoked on a real path. **Impact:** false privacy assurance for privacy-sensitive tenants → reputational/compliance risk (CWE-200; privacy-guarantee-not-enforced). **Effort:** 1–2d · **Fix:** apply noise at logits in `_sample()`, or prominently mark as budget-tracking-only and refuse to claim DP.

### S2 — `differential_privacy.py` adds noise to unclipped tensors — no bounded sensitivity — `differential_privacy.py:57`
`add_noise_to_tensor`/`add_noise_to_kv_cache` apply `sigma` derived from `max_grad_norm` but **never clip** the tensor L2 norm first (`clip_tensor` exists at L94, unused in the noise path) → the Gaussian mechanism's (ε,δ) guarantee doesn't hold. Wired via `privacy_budget.py`. **Effort:** 0.5d · **CVSS/CWE:** CWE-200/privacy-mechanism-incorrect. **Fix:** clip-then-noise; test `||output|| ≤ max_grad_norm`.

### S3 — `shared_layer_pool` prefix-hash dedup shares wrong weights (also B18) — `shared_layer_pool.py:247` (CWE-345).
See B18.

### Additional Medium security items (CVSS graded in prior `SECURITY_AUDIT_CORE.md`)
- `backup_manager.py:137` — backup integrity/verify gap (Medium).
- `correctness_cert.py:41` — cert verification weakness (Medium).
- `model_router.py:724` — path/route injection surface (Medium).
- `request_auditor.py:160` — raw-field redaction gap (Medium).
- `plugin_sandbox.py:654` — capability-model enforcement gap (Medium).
- `aegis_compliance.py:203` — PHI in plaintext SQLite while its own rule demands encryption-at-rest (Medium, PRAGMA WAL/synchronous=NORMAL, sync commit per record).

---

## High — architecture / tech-debt (see [[Core Audit 07 Dead Code & Consolidation]] for depth)
- `coordinator_subsystem.py:239` — `SubsystemManager` (620 LOC) is **dead**; `Coordinator.start/stop` re-implement lifecycle inline. Duplication should be merged.
- `cortex_multimodel.py:803` — **dead** (939 LOC) AND its served prefix-sharing caches an empty `dummy_kv` (stub).
- `evaluation/runner.py:120` — eval pipeline duplicated: live monolith (`evaluation_harness.py`) vs dead `evaluation/` refactor carrying a `_SecretStr`-wrapper API-key bug.
- `grammar_constrained.py:522` — grammar decoding split across 3 unwired modules + a never-invoked `_patch_schema_constrained_decoder`.
- `kv_cache.py:781` — KV-cache triple/duplicate cluster (`KVCacheManager` ×3, each sibling dead); `kv_cache.py` still 1254 LOC.
- `pipeline_executor.py:110` — ~10 modules (several thousand LOC) unwired.
- `starvation_monitor.py:16` — ~5,500 LOC of dead modules (list in [[Core Audit 07 Dead Code & Consolidation]]).

---

## Security (Medium) — key concrete items
| Finding | File:Line | CVSS-ish / CWE | Effort |
|---------|-----------|----------------|--------|
| PHI stored plaintext in audit SQLite vs `encrypt_phi` rule | `aegis_compliance.py:203` | CWE-311 (encryption missing) | 0.5–1d |
| Backup integrity not verified | `backup_manager.py:137` | CWE-494 | 2h |
| Certificate verification gap | `correctness_cert.py:41` | CWE-295 | 2h |
| Router path/injection surface | `model_router.py:724` | CWE-20 | 2h |
| Audit raw-field redaction gap | `request_auditor.py:160` | CWE-200 | 2h |
| Plugin sandbox capability-model gap | `plugin_sandbox.py:654` | CWE-270 | 4h |

---

## Performance (measurable)
| Finding | File:Line | Metric / impact | Effort |
|---------|-----------|-----------------|--------|
| `redis_prompt_cache.py` O(N) sequential GETs | `redis_prompt_cache.py:215` | ~4096 round-trips for a 4K prompt | 2h |
| Telemetry deadlock stalls request at 50th event | `telemetry.py:185` (see B20) | hang → p50+ spike under telemetry | 30m |
| Autoscaler `hybrid_cache.py:129` cache-miss recompute | `hybrid_cache.py:129` | repeated cold-key work | 3h |
| `autonomous_healer.py:405` busy polling | `autonomous_healer.py:405` | interval-bound CPU | 2h |
| `api_key_store.py:145` per-request hashing cost | `api_key_store.py:145` | hash on hot path | 2h |
| `moe_orchestrator.py:127` per-route overhead | `moe_orchestrator.py:127` | router latency | 3h |
| `federated_incentives.py:223` O(N²) accumulation | `federated_incentives.py:223` | scales with node count | 3h |
| `semantic_cache.py:172` eviction scan O(n) | `semantic_cache.py:172` | large-cache latency | 2h |
| `coordinator_health.py:115` sync probe | `coordinator_health.py:115` | event-loop block | 2h |
| `tree_speculative_decoder.py:210` tree expansion | `tree_speculative_decoder.py:210` | accept-rate/work | 3h |
| `tenant_billing.py:260` per-request DB write | `tenant_billing.py:260` | write amplification | 2h |
| `request_latency.py:77` per-event timestamp | `request_latency.py:77` | overhead | 1h |
| `adaptive_batching.py:193` re-batch cost | `adaptive_batching.py:193` | low | 2h |

---

## Medium — catalog (compact)
*bug*
| File:Line | Title | Effort |
|-----------|-------|--------|
| `heterogeneous.py:54` | All-zero TFLOPS crash + budget mutated in place | 2–3h |
| `adaptive_cache_compressor.py:62` | Compression metric drift on quantized tensors | 2h |
| `adaptive_compression.py:327` | Ref-count leak in cached references | 2h |
| `ab_test_coordinator.py:157` | A/B assignment race across workers | 2h |
| `cache_eviction.py:121` | LFU `_maybe_decay` double-decay race | 1h |
| `batch_scheduler.py:419` | Waiting-queue budget accounting under preemption | 2h |
| `cache_template_warmer.py:43` | Uses fake token IDs (`range(len//4)`) — never warms | 30m |
| `constrained_decoder.py:516,593` | Checks first byte only (invalid multibyte tokens); rebuild race | 1d |
| `coordinator_request.py:48` | In-flight leak on early-return paths | 2h |
| `cost_dashboard.py:326` | Query window off-by-one | 1h |
| `cross_model_prefix_sharing.py:128` | Sharing keyed on prefix not semantics | 2h |
| `draft_model_router.py:305` | Draft model stale fallback | 2h |
| `gpu_resource_manager.py:231` | Free VRAM accounting race | 1h |
| `learning_router.py:630` | Cross-entropy on bandit data ignores reward | 1d |
| `micro_batch_scheduler.py:82` | Zero-length batch Edge case | 1h |
| `multi_model_serving.py:187,597` | `remove_model()` does `pass` — memory never freed | 1d |
| `model_version_manager.py:250` | Version rollback doesn't reload | 2h |
| `preference_learning.py:305` | Reward normalization edge | 2h |
| `request_pipeline.py:162` | Dedup key ignores client context | 2h |
| `prompt_caching_service.py:130` | anyio.from_thread deadlock risk | 2h |
| `usage_meter.py:399,349` | Aggregation window drift / counter reset | 2h |
| `unified_sla_router.py:205` | SLA tier not re-validated on fallback | 2h |
| `token_generator.py:336` | Streaming tokenizer stray state | 1h |
| `webhook_manager.py:258` | Dedup window leakage | 1h |
| `hybrid_cache.py:129` | Tier promotion lossy | 2h |
| `structured_output/streaming.py:97` | Streaming mask not applied per-chunk | 2h |

--- 

## Medium/Low — code-quality/tech-debt (compact)
See [[Core Audit 07 Dead Code & Consolidation]] for the dead-module map. Representative mental-health items:
- `adaptive_compression.py:138` — silent partial-failure swallowing.
- `__init__.py:70` — `_register()` grows unboundedly (242 modules).
- `cost_dashboard.py:182` — duplicated metric aggregation.
- `agentic_router.py:649` / `batch_scheduler.py:1134` — 100+ line methods.
- `gpu_profiler.py:45` — probe collisions.
- `health_manager.py:248` — blocking probe in async post.
- `multi_model_serving.py:450` — dead `model_pool` field.
- `coordinator_config.py:187` — config defaults diverge from `coordinator_config_wiring.py`.

---
**← [[Core Comprehensive Audit 2026-08-05]]** · Next: [[Core Audit 03 Enhancements & Modifications]]