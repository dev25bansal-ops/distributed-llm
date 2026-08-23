---
tags:
  - audit
  - exhaustive
  - verified
date: 2026-08-11
---

# Exhaustive Audit 01 — Verified Critical/High Bug & Security Findings

**← [[Exhaustive Audit 2026-08-11]]**

Every Critical/High **bug** and **security** finding from the exhaustive read was adversarially verified: a fresh agent opened the actual file, read around the cited line, and independently judged `isReal` (does the defect genuinely exist as described) and `mustFix` (is it load-bearing enough to block release). **46 of 59 are confirmed real and must-fix.**

## Fix status (2026-08-11 → )

Verified findings below that are **FIXED + regression-tested** (each adversarially reviewed; see **[[Action Plan 2026-08-11]]**):

- ✅ **JWT HS256-fallback auth bypass** (F-00x) — `plugins/auth_plugin.py` — tests `tests/security/test_jwt_hs256_fallback.py`
- ✅ **Exact-match cache key not tenant-scoped** — `plugins/cache_plugin.py` — tests `tests/security/test_cache_plugin_tenant_isolation.py`
- ✅ **HA heartbeat/snapshot fail-open** — `api/server.py` — tests `tests/api/test_ha_heartbeat.py`
- ✅ **ApiKeyRotator never retires old key** — `core/api_key_store.py` + `core/cert_rotation.py` — tests `tests/core/test_api_key_rotation.py`
- ✅ **RED metrics never recorded** — `api/observability_middleware.py` (fixed `.inc()/.observe()` + mounted) — tests `tests/api/test_auth_middleware.py`
- ✅ **Federated LoRA merge untrusted adapter/dataset_size** — `dist/federated_merge.py` (signature auth + cap) — tests `tests/dist/test_federated_merge.py`
- ✅ **DHT STORE unauthenticated** — `dist/p2p/kademlia_dht.py` (shared-secret HMAC token) — tests `tests/dist/test_kademlia_dht.py`
- ✅ **QUIC CERT_NONE MITM** — `dist/p2p/quic_transport.py` (CA-gated CERT_REQUIRED) — tests `tests/dist/test_quic_verify.py` (skips w/o aioquic)
- ✅ **Pipeline gRPC plaintext** — `dist/node_client.py` + `dist/pipeline/orchestrator.py` (use_tls threaded) — tests `tests/dist/test_node_client_tls.py`
- ✅ **DP 'inference' charges budget without noise** — `core/dp_inference.py` + `core/dp_inference/__init__.py` (fail-closed raise) — tests `tests/core/test_dp_inference.py`
- ✅ **DedupMiddleware auth bypass + cross-tenant leak** — `api/dedup.py` (fingerprint namespaced by `api_key_id`; middleware reordered so Auth runs before Dedup) — tests `tests/api/test_dedup_middleware.py` (cross-key isolation + unauthenticated-replay)
- ✅ **RedundantExecutor._run_redundant stub** — `dist/redundant.py` (real fan-out via `forward_request`; `_find_redundant_nodes`/`get_node_groups`/`promote_standby` use pipeline internals; `PipelineNode.kv_cache` field) — tests `tests/dist/test_redundant.py`
- ✅ **Generated API key logged in cleartext** — `api/middleware.py` (fingerprint via logger; full key printed once to stdout only) — tests `tests/api/test_api_key_not_logged.py`
- ✅ **SSO fork security bugs** (`api/auth/` unwired fork) — `auth/oidc.py` stores+validates the actual nonce (replay protection); `auth/__init__.py` imports hashlib (was NameError). Consolidation of the two SSO stacks still pending (Audit 05). — tests `tests/security/test_auth_fork_nonce.py`
- ✅ **GPUResourceManager.snapshot used/free swapped** — `core/gpu_resource_manager.py` — tests `tests/core/test_quick_audit_fixes.py`
- ✅ **NeuralBanditRouter missing `import torch`** — `core/learning_router.py` — tests `tests/core/test_quick_audit_fixes.py`
- ✅ **`distllm system doctor` never ran** — `cli/main.py` (strips Typer subcommand tokens before calling doctor) + `cli/doctor.py` (rich.group→rich.console Group import; main accepts argv) — tests `tests/core/test_quick_audit_fixes.py`
- ✅ **telemetry.flush() deadlock** — `core/telemetry.py` (`_add_event` releases the non-reentrant lock before flushing) — tests `tests/core/test_telemetry.py`
- ✅ **JSONSchema FSM truncates multi-digit numbers** — `core/structured_output/__init__.py` (`_valid_next_chars` gained `in_number`; comma-after-number transitions) — tests `tests/core/test_structured_output_numbers.py`
- ✅ **Off-by-one token-position indexing in 4 speculative verifiers** — `core/speculative_decoder.py` (tree batched `_verify_tokens`, prefix_len now `shape[1]-1`), `core/multi_draft_verifier.py`, `core/mtp_head.py` — tests `tests/core/test_multi_draft_speculative_decoder.py` (20 pass). NOTE: 3 pre-existing failures in `test_speculative_decoder.py` mock the wrong token convention (`logits[pos]`→`input[pos]` instead of `logits[k]`→`token[k+1]`); fail identically at HEAD.
- ✅ **Prompt-exchange acquire IDOR** — `api/routes/exchange.py` (user_id now bound to authenticated `api_key_owner`; 403 on mismatch) — tests `tests/api/test_exchange.py`, `tests/security/test_idor_exchange.py`
- ✅ **`g.get`/`reset_app_state_for_testing` missing (F-007 collection blocker)** — `api/api_state.py` gained dict-style `get`/`__getitem__`/`set`, and `reset_app_state_for_testing` now mutates AppState in place (stale-binding fix); unblocked exchange/idor test collection — 76 tests pass
- ✅ **Streaming broken across langchain/crewai/llamaindex adapters (F-005)** — removed `stream=True` from `chat_completions_stream` calls (langchain, llamaindex, crewai); added `completions_stream` to `distllm.sdk` + `distllm_sdk` (sync + async) so text-completion streaming resolves. Also added `Coordinator.generate_batch` (delegates to the request pipeline's batch loop; raises `BatchError` when unwired) — `tests/integration/test_streaming_generation.py` 9 pass.
- ✅ **Tool-calling contract broken (F-006)** — `distllm.sdk` + `distllm_sdk` `chat_completions`/`chat_completions_stream` now accept `tools` + `federation_strategy`/`preferred_regions`/`spillover_enabled` and propagate them via `_build_chat_payload`; langchain/llamaindex bind_tools() payloads no longer TypeError
- ✅ **Kademlia routing table trusts sender IP:port (F-050)** — all 4 DHT handlers (PING/STORE/FIND_NODE/FIND_VALUE) now use the packet source addr instead of the sender-declared ip/port, preventing routing-table eclipsing — tests `tests/dist/test_kademlia_dht.py` (73 pass)
- ✅ **request_latency elapsed grows with wall clock (F-022)** — `core/request_latency.py` freezes elapsed at `completed_at` for completed rows so fast requests don't become overdue as time passes; compliance percentiles stable — tests `tests/core/test_request_latency_frozen.py`
- ✅ **UsageMeter window under-reports after 100k cap (F-024)** — `dist/daas/usage_meter.py` queries SQLite for the full window when the in-memory cap is hit (DB is authoritative) so billing/quota aggregation stays correct — tests `tests/dist/test_usage_meter_window.py`
- ✅ **PagedAttention KV blocks leak per completed sequence (F-041)** — `core/batch_scheduler.py` step() now frees a completed sequence's PagedAttention blocks when pruning it from `active` — tests `tests/core/test_batch_scheduler.py`, `tests/core/test_advanced_scheduling_contract.py` (55 pass)
- ✅ **Gossip HMAC rotation breaks shared key (F-027)** — `dist/p2p/gossip.py` no longer rotates a deployment-shared `DISTLLM_GOSSIP_HMAC_KEY` (only node-local dev keys rotate); gossip auth no longer breaks after ~24h — tests `tests/dist/test_gossip_hmac_rotation.py`
- ✅ **HA leader election never gates request processing (F-015)** — `core/coordinator.py` `generate()` now refuses when HA is enabled and this coordinator is not the leader (standby can't act as writer; split-brain protection enforced); single-node unaffected — tests `tests/core/test_coordinator_startup_order.py`, `test_coordinator_failover.py` (15 pass)
- ✅ **Local-model generation leaks a generator instead of a string + token-loop tuple crash** (demo blocker, surfaced during investor-demo prep) — `core/inference_engine.py` `_PromptLookupStrategy.generate` now collects the generator into a joined str (protocol-compliant) with the per-token loop in `_generate_tokens`; `token_gen.sample()` tuple unwrapped + dim-normalized at both sample sites; `core/coordinator.py` defines `_agentic_router=None` in `__init__` so the pre-routing check never AttributeErrors. Verified end-to-end: `sshleifer/tiny-gpt2` loads and `generate("Hello")` returns real text. Tests `tests/core/test_local_generation_str.py` (2 pass).

**Batch fixed 2026-08-21** (15-agent workflow; each fix independently re-run by main session):

- ✅ **F-037 E2E ratchet divergence under asymmetric traffic** — `security/e2e.py`: dropped the divergent local-counter ratchet; per-message fresh salt → unique box key per message, transmitted with the ciphertext; `derive_box_key` uses correct PyNaCl keyword signature (+HKDF-SHA256 fallback) — tests `tests/security/test_e2e_ratchet_f037.py`
- ✅ **F-009 MLX adapter scores tokens as isolated length-1 sequences** — `backends/mlx_backend.py` evaluates full sequence per row (causal attention over context), unpacks mlx-lm tuple logits, keeps batch rows — tests `tests/backends/test_mlx_forward_full_sequence_f-009.py`
- ✅ **F-010 NIM fabricates synthetic logits on API fallback** — ALREADY FIXED (raises NotImplementedError); regression test exists and passes
- ✅ **F-011 Azure availability check fail-open + literal `{subscriptionId}` URL** — `cloud/azure.py`: real ARM call via DefaultAzureCredential, interpolated region/subscription, fails CLOSED on every error path — tests `tests/cloud/test_azure_availability_f-011.py` (88 cloud tests pass)
- ✅ **F-012 Aether "LoRA" training trains full base weights / zero delta merged** — `core/aether_federated.py`: real LoRA train_lora (frozen base, federated rounds over W+s·BAᵀ, aggregated grads converted to grad_A/grad_B), persisted via update(); merge applies trained delta — tests `tests/core/test_aether_lora_training_f-012.py`
- ✅ **F-014 IntelligentAutoscaler instantiated but never evaluated/actuated** — `core/coordinator.py`+`coordinator_subsystem.py`: background `_autoscaler_loop` collects live ScalingMetrics each tick, calls evaluate(), actuates via `set_scale_callback` hook (env-tunable interval) — tests `tests/core/test_intelligent_autoscaler_wiring_f-014.py`
- ✅ **F-018 Federated fine-tuner averages stragglers' stale gradients forever** — `core/federated_finetuner.py`: round-tagged contributions, pruned at round start, non-current-round messages dropped at receive, defense-in-depth round check + shape guard in aggregation; FedProx now snapshots w_local correctly — tests `tests/core/test_federated_stale_gradients_f-018.py`
- ✅ **F-019 FedProx proximal term uses wrong w (post-update params)** — `core/federated_finetuner.py` snapshots local params BEFORE applying the update — tests `tests/core/test_fedprox_term_f-019.py`
- ✅ **F-021 int8 KV dequant reads whole cache per token** — bulk dequant path added so long contexts don't re-dequantize per step — tests `tests/core/test_kv_cache_bulk_dequant_f-021.py` (82 KV-cache tests pass)
- ✅ **F-040 paged speculative decoding breaks across block boundaries** — pipelined spec decode handles block-resident KV — tests `tests/core/test_pipelined_speculative_f-040.py` (25 module tests pass)
- ✅ **F-048 dynamic sharder mis-splits on heterogeneous nodes** — `core/dynamic_sharder` corrected — tests `tests/core/test_dynamic_sharder_f-048.py`
- ✅ **F-051 learned cost model never trains/serves consistently (train/serve skew)** — `dist/partition/learned_cost.py` + `network_cost_model.py` — tests `tests/dist/test_learned_cost_train_serve_f-051.py` (136 related pass)
- ✅ **F-052 congestion control ignores bandwidth feedback** — `dist/pipeline/` bandwidth controller wired — tests `tests/dist/pipeline/test_bandwidth_controller_f-052.py` (98 pipeline tests pass)
- ✅ **F-053 streaming KV transfer corrupts bf16 blocks** — bf16-safe streaming KV transfer — tests `tests/dist/test_streaming_kv_transfer_f-053.py` (56 incl existing)
- ✅ **F-054 model hub download_layer_subset fetches wrong slice** — `models/model_hub.py` layer-subset range math fixed — tests `tests/models/test_model_hub_layer_subset_f-054.py` (78 hub tests pass)
- ✅ **F-055 diffusion pipeline ignores scheduler/timestep contract** — `core/hydra_diffusion.py` — tests `tests/core/test_hydra_diffusion_f-055.py`

## Confirmed must-fix (46)

### F-001 — [Critical] 🔒 VERIFIED · MUST-FIX RED metrics (requests/latency/duration/errors) are never recorded — .labels() handles are discarded
✅ **FIXED** — `api/observability_middleware.py` (`.inc()`/`.observe()` now chained), mounted in `server.py` (was never mounted), and `routes/health.py /metrics` now serves the API exporter registry (RED metrics visible at `/metrics`). Tests: `tests/api/test_auth_middleware.py` (TestObservabilityREDMetricsRegression + TestObservabilityMiddlewareMounted).

`src/distllm/api/observability_middleware.py:124` · zone=`core-perf-obs` · category=`bug`

- **Summary:** ObservabilityMiddleware is the sole wiring for the Prometheus exporter's request metrics, but every metric call constructs a labeled handle via .labels(...) and throws it away without .inc()/.observe(). requests_total, request_latency, request_duration_seconds, and errors_total are thus permanently empty — the whole rate/errors/duration promise is dead. Only request_cost_total and request_gpu_hours (which call .inc()) emit anything.
- **Evidence (verbatim):**
```
exporter.requests_total.labels(method=method...)     exporter.request_latency.labels(method=method...)     exporter.request_duration_seconds.labels(method=method...)     if error ...: exporter.errors_total.labels(type='http_500'...)   # only cost metrics: .inc(cost_val) / .inc(gpu_val)
```
- **Impact:** Every dashboard/alert reading request count, p95 latency, duration, or error rate from the Prometheus exporter shows 0 forever — SLO and capacity graphs silently blank, and any alert threshold on these is unreachable.
- **Effort:** 1-2 hours
- **Reliability:** Run two requests through the app with the exporter attached; generate_latest() shows distllm_requests_total and distllm_request_latency_seconds all zero despite traffic. Repro is deterministic because the labeled child object returned by .labels() is discarded before .inc()/.observe() is possible.
- **Recommendation:** Add the terminating calls: exporter.requests_total.labels(...).inc(); exporter.request_latency.labels(...).observe(duration); exporter.request_duration_seconds.labels(...).observe(duration); and exporter.errors_total.labels(...).inc() on error. Add a regression test that instantiates DistLLMPrometheusExporter, runs one request, and asserts requests_total/re-request_duration_seconds counters are non-zero via generate_latest().
- **Verdict (real):** mustFix=True. Verified by reading observability_middleware.py L124-136 and exporter.py. The four RED calls — requests_total (Counter), request_latency (Histogram), request_duration_seconds (Histogram), errors_total (Counter) — call .labels(...) and discard the returned handle with NO .inc()/.observe() chained, so nothing is ever recorded from this middleware. The neighboring cost block (L143-149) correctly chains .inc(cost_val)/.inc(gpu_val), proving the intended pattern was known but not applied to the RED metrics. Grep confirms requests_total/request_latency/request_duration_seconds have no other recording site; tests (test_auth_middleware.py TestObservabilityREDMetrics, L856-899) only assert .labels.assert_called_once_with(...), i.e. handle creation not value mutation, so they pass despite dead metrics. Fix (add .inc()/.observe(), correct the tests) is trivial and low-risk; leaving it means the documented RATE/ERRORS/DURATION promise and the Grafana/Prometheus dashboards/alerts feeding those series stay permanently empty.
  - *Correction:* Two caveats refine, but do not negate, the finding: (1) errors_total is NOT globally dead — request_pipeline.py L348/353 increments it via .labels(type=type(e).__name__).inc(); only the middleware's type="http_500" call is a dead handle. (2) ObservabilityMiddleware is never instantiated in src/distllm (constructor grep finds only the class def; server.py add_middleware omits it), so the entire middleware — including the live cost metrics — appears unmounted in production; that compounds the issue rather than refuting the defect. The core claim (the four .labels() handles for requests_total/request_latency/request_duration_seconds/errors_total are discarded, so request rate/latency/duration are never recorded from this middleware) stands as written.

---

### F-002 — [Critical] 🔒 VERIFIED · MUST-FIX Cross-model sibling cache lookup returns KV data for a DIFFERENT prompt (wrong/injected tokens)

`src/distllm/core/cross_model_prefix_sharing.py:169` · zone=`core-training` · category=`bug`

- **Summary:** CrossModelPrefixSharing.lookup() sibling branch iterates every cache entry of models sharing a base_model and returns the FIRST one without ever comparing token_ids/prefix_hash to the query. Direct (case 1) and base (case 2) branches key on the computed prefix_hash, but case 3 does not, so any same-family query can be answered with cached KV from an unrelated prompt, silently injecting wrong/poisoned tokens into generation.
- **Evidence (verbatim):**
```
for key, entry in self._cache.items():     if self._is_expired(entry): continue     sibling = self._models.get(entry.source_model)     if sibling and sibling.base_model == model.base_model:         shared = min(entry.shared_layers, model.shared_layers)         if shared > 0: ... return entry
```
- **Impact:** Wrong/cross-tenant generations in shared-base serving; correctness + data-exposure risk whenever a cached same-family entry exists for any other prompt.
- **Effort:** 2-4 hours
- **Reliability:** Register two variants sharing a base_model; store(model A, tokens_A); lookup(model B, tokens_B != tokens_A) returns tokens_A KV entry.
- **Recommendation:** Require the sibling's token_ids / prefix_hash to equal the query. Iterate only keys f"{sibling}:{prefix_hash}" (or compare entry.prefix_hash == prefix_hash and entry.token_ids == token_ids) before returning, mirroring case 1/2. Add a regression test asserting a sibling with a different prompt hash is never returned.
- **Verdict (real):** mustFix=True. Confirmed by reading cross_model_prefix_sharing.py. The sibling branch (lines 168-181) iterates every entry in self._cache and returns the FIRST entry whose source_model has the same base_model and shared_layers>0, with NO comparison of entry.token_ids or entry.prefix_hash to the query's computed prefix_hash. This is in explicit contrast to case 1 (line 145, key f"{model_id}:{prefix_hash}") and case 2 (line 155, key f"{model.base_model}:{prefix_hash}"), both of which match on the prefix_hash. So case 3 silently returns KV computed for a DIFFERENT prompt of any same-family model, injecting wrong/poisoned tokens into generation. The claim's cited evidence matches the code exactly. Load-bearing: this is a silent data-integrity defect in a core cache component documented as delivering 30-50% TTFT reduction, and the existing sibling tests only exercise the happy path so they would not catch the wrong-hash injection. Caveat that tempers severity but does not negate it: the module and its only consumer (ClusterPredictivePrefetcher, whose cross_model_sharing param defaults to None) are currently not instantiated in any src/ production call path (neither is imported by coordinator/inference_engine/__init__), so the path is dormant right now. Still a genuine Critical/High correctness bug that must be fixed (add prefix_hash/token_ids comparison in case 3) before the feature is considered release-ready.

---

### F-003 — [High] 🔒 VERIFIED · MUST-FIX Dify integration double-prefixes /v1 and probes the wrong health path

`integrations/dify/distllm_provider.py:104` · zone=`integrations` · category=`bug`
✅ **FIXED** — base URL default is now the bare origin (no trailing `/v1`), request paths stay `/v1/...`; `invoke` honors the per-call `model` argument. Tests: `tests/security/test_dify_provider_urls.py` (5 pass).

- **Summary:** DistLLMProvider._get_client sets base_url to `DISTLLM_API_BASE` which defaults to `http://localhost:8000/v1`, but every request then prepends `/v1/...` again: `client.post("/v1/chat/completions")` -> `http://localhost:8000/v1/v1/chat/completions`, and `client.get("/v1/models")` -> `/v1/v1/models`. validate_credentials() calls `client.get("/health")` -> `/v1/health` which 404s. The integration is unusable as shipped.
- **Evidence (verbatim):**
```
return httpx.Client(             base_url=self._api_base,   # '.../v1' by default             ...)
```
- **Impact:** Dify custom-provider plugin always fails credential validation and every inference/embed call hits a 404/405, so the advertised Dify integration cannot run.
- **Effort:** 1-2 hours
- **Reliability:** base_url env default 'http://localhost:8000/v1' (line 99); _invoke_sync posts '/v1/chat/completions' (line 162) -> concatenated '/v1/v1/chat/completions'; validate_credentials GET '/health' (line 114) -> '/v1/health'.
- **Recommendation:** Drop the trailing '/v1' from the default base_url (keep it bare host) and keep '/v1/...' path prefixes, mirroring the fastapi router.py which correctly posts to '/v1/chat/completions'. Also honor the per-call `model` argument in invoke() instead of always using self._model.
- **Verdict (real):** mustFix=True. I opened integrations/dify/distllm_provider.py. DistLLMProvider.__init__ (line 99) sets _api_base default to 'http://localhost:8000/v1', and _get_client (line 105) passes it as httpx base_url. Then _invoke_sync (line 162), _invoke_stream (line 184), and get_models (line 219) call http-client-paths that themselves start with /v1/. Because httpx concatenates base_path with the request path (it does not RFC-3986 slash-replace), I ran the actual client and got '.../v1/v1/chat/completions' and '.../v1/v1/models'. The FastAPI server mounts chat_router and health_router at root with no extra prefix and serves only '/v1/chat/completions' and '/v1/models', so the double-prefixed URL 404s -> the Dify chat invocation path is genuinely broken (integration unusable for its core function). get_models hides its 404 behind a try/except fallback. This is load-bearing enough to be a real High: the integration cannot make any chat/completion call as shipped. One detail in the original claim was incorrect (the /health check does not 404 because the server also serves /v1/health), but it does not change the severity of the core double-prefix bug.
  - *Correction:* The double-prefix defect is real and confirmed by executing httpx (base_url='http://localhost:8000/v1' + '/v1/chat/completions' => 'http://localhost:8000/v1/v1/chat/completions'; + '/v1/models' => '/v1/v1/models'). The server serves the single-prefixed paths only (routes/chat.py POST /v1/chat/completions, routes/health.py GET /v1/models), so chat invocations 404 and get_models silently falls back to a hardcoded model. However, the health sub-claim is wrong: routes/health.py lines 62-63 explicitly serve GET /v1/health (same handler as /health), so validate_credentials' client.get('/health')->/v1/health returns 200 when a model is loaded — it does NOT 404.

---

### F-004 — [High] 🔒 VERIFIED · MUST-FIX gRPC client is non-functional: imports a proto package that is never shipped and falls back to a service name that does not exist on the server

`integrations/grpc_client/src/distllm_grpc/client.py:218` · zone=`integrations` · category=`bug`

- **Summary:** DistLLMGrpcClient tries `from distllm_grpc.proto import inference_pb2_grpc` (client.py:91) but no `proto/` module exists in the grpc_client package (only client.py, cli.py, __init__.py), so `_stub` is always None. The JSON-over-channel fallback then calls generic channel unary_unary/unary_stream on `/distllm.InferenceService/ChatCompletion`, `/distllm.InferenceService/ChatCompletionStream`, `/distllm.InferenceService/Embeddings`. The actual DistLLM gRPC service (node_pb2_grpc.py) only exposes `/distllm.NodeService` (ForwardPass, HealthCheck, Profile, TransferWeights, ...). Every call returns UNIMPLEMENTED.
- **Evidence (verbatim):**
```
resp = await self._channel.unary_unary(             "/distllm.InferenceService/ChatCompletion", ...)
```
- **Impact:** The advertised high-performance gRPC client cannot complete a single request against the real server. The 'falls back to REST if gRPC unavailable' docstring claim is also false (there is no REST fallback, only the broken JSON-over-gRPC path).
- **Effort:** 1-2 days
- **Reliability:** grep node_pb2_grpc.py shows only '/distllm.NodeService/...' routes; distllm_grpc has no proto/ dir. Even the optional stub path is dead because the package ships no generated stubs.
- **Recommendation:** Either (a) generate and ship inference.proto stubs that match the real NodeService or add a gRPC gateway serving /distllm.InferenceService, or (b) reimplement the client against the existing NodeService methods, or (c) implement a genuine REST fallback via httpx as documented. Add a live test against grpc_bridge.py.
- **Verdict (real):** mustFix=True. Verified the finding exactly as described. (1) The package integrations/grpc_client/src/distllm_grpc/ contains only __init__.py, cli.py, client.py — no proto/ module and no inference_pb2_grpc anywhere. Line 91 `from distllm_grpc.proto import inference_pb2_grpc` always raises ImportError, so `self._stub` is always None and every public method (chat_completion, chat_completion_stream, embeddings) always takes the _via_json fallback. (2) The JSON fallback at lines 218-222, 234-238, 243-247 issues generic channel calls to /distllm.InferenceService/ChatCompletion, /ChatCompletionStream, and /Embeddings. (3) The only shipped gRPC service, src/distllm/dist/node_pb2_grpc.py, registers distllm.NodeService (ForwardPass, HealthCheck, Profile, TransferWeights, TransferWeightsStream, AdvertiseModels) — no InferenceService exists anywhere in src (grep confirmed; the grpc_bridge uses NodeServiceStub). A generic unary_unary to /distllm.InferenceService/* against a server registering only NodeService returns UNIMPLEMENTED on every call. This breaks the integration's sole advertised purpose (gRPC chat/stream/embeddings), so it is load-bearing and must be fixed before release.

---

### F-005 — [High] 🔒 VERIFIED · MUST-FIX Streaming is fully broken across langchain/crewai/llamaindex adapters (TypeError on stream=True + AttributeError on completions_stream)

`integrations/langchain/src/distllm_langchain/chat_models.py:285` · zone=`integrations` · category=`bug`

- **Summary:** All chat/complete streaming paths in the three main SDK adapters crash at runtime because they pass `stream=True` to `DistLLMClient.chat_completions_stream(...)`, a method whose signature accepts no `stream` parameter (streaming is implicit). Additionally, the text-completion streaming paths (DistLLM BaseLLM._stream and llamaindex stream_complete) call `self._client.completions_stream(...)`, a method that does not exist anywhere in the SDK (only `chat_completions_stream` exists). Nothing against the real SDK works.
- **Evidence (verbatim):**
```
for chunk in self._client.chat_completions_stream(**payload, stream=True):
```
- **Impact:** Every LangChain, CrewAI and LlamaIndex app that uses .stream()/.astream()/stream_chat()/stream_complete() raises at call time instead of streaming tokens. Tests mask it because crewai/tests/test_all.py & langchain/tests/test_chat_advanced.py monkeypatch chat_completions_stream with MagicMock (which swallows unknown kwargs) and feed dict chunks, so the real SDK invocation is never exercised.
- **Effort:** 2-4 hours
- **Reliability:** sdk/client.py:856 signature has no `stream` and no **kwargs; grep for `completions_stream` returns only chat_completions_stream. Reproduce: llm.stream([HumanMessage('hi')]) -> TypeError unexpected keyword 'stream'; llm.completions-stream completes LLM -> AttributeError.
- **Recommendation:** Remove `stream=True` (and any tools/federation kwargs that the SDK method rejects) from these calls. Rewrite the async streaming adapters to consume the content strings that `DistLLMClient.chat_completions_stream` async actually yields (sdk/client.py:576 `yield content`), and drop the `isinstance(chunk, dict)` guard. Add an integration test against the real SDK stream rather than a MagicMock.
- **Verdict (real):** mustFix=True. Verified by reading the SDK and all three adapters. (1) In src/distllm/sdk/client.py, chat_completions_stream is defined at line 549 (async) and 856 (sync) with signature (messages, model, temperature, top_p, max_tokens, response_format, adapter, logprobs, include_usage, timeout) — no `stream` param and no `**kwargs`. Yet langchain chat_models.py:285 & :318-319, llamaindex llms.py:185 & :197, and crewai llm.py:66 & :104 all call `chat_completions_stream(..., stream=True)`, which raises TypeError at runtime (confirmed by a captured traceback "...got an unexpected keyword argument 'stream'"). (2) `completions_stream` is called at langchain llms.py:113 & :138 and llamaindex llms.py:221 & :230, but no such method exists anywhere in the SDK (grep confirms only `chat_completions_stream` is defined, in both src/distllm/sdk/client.py and the legacy sdk/src/distllm_sdk/client.py); llamaindex test_advanced.py:171 relies on `create=True` when mocking it, confirming it does not exist on the real client. Both the chat and text-completion streaming paths in all three flagship adapters (langchain, crewai, llamaindex) crash against the real SDK. This is a first-class feature (streaming) that is completely non-functional, so it is a genuine Critical/High release blocker. Minor nuance: the finding understates it slightly — even without `stream=True`, the langchain payload also forwards `stop`/`tools`/federation fields not in the SDK signature, but that does not change the verdict. Streaming=true is also redundant since chat_completions_stream hardcodes stream True internally.

---

### F-006 — [High] 🔒 VERIFIED · MUST-FIX Tool-calling contract broken: bind_tools() and framework function-calling inject `tools`/federation kwargs the SDK does not accept

`integrations/langchain/src/distllm_langchain/chat_models.py:430` · zone=`integrations` · category=`bug`

- **Summary:** LangChain's bind_tools() builds a payload with `payload["tools"] = bound_tools`, and when federation hints are set it also adds `federation_strategy`/`preferred_regions`/`spillover_enabled`. These are then passed positionally into `DistLLMClient.chat_completions(**payload)`, but the SDK chat_completions signature (sdk/client.py:513) accepts neither `tools` nor any of those keyword args (grep finds zero `tools` tokens in sdk/client.py). Every tool-calling call raises TypeError. LlamaIndex advertises `is_function_calling_model=True` (llms.py:112) yet never sends a tools array, so function calls silently return prose.
- **Evidence (verbatim):**
```
bound_tools = kwargs.pop("tools", None)         if bound_tools:             payload["tools"] = bound_tools
```
- **Impact:** The platform's core tool-calling story fails in production for every adapter that consumes it, contradicting LLMMetadata.is_function_calling_model=True and the `distllm_chat` tool provider contract.
- **Effort:** 3-5 hours
- **Reliability:** Trace: bind_tools() -> RunnableBinding(kwargs={tools:...}) -> _generate -> _build_payload puts 'tools' into payload -> self._client.chat_completions(**payload) -> TypeError (no 'tools' param, no **kwargs). Same for federation keys.
- **Recommendation:** Add `tools: list | None = None` (and the federation kwargs) to DistLLMClient.chat_completions / chat_completions_stream and forward into `_build_chat_payload`'s body. In LlamaIndex, actually pass `tools` from the framework or drop the is_function_calling_model flag until supported.
- **Verdict (real):** mustFix=True. Confirmed by reading both files. integrations/langchain/.../chat_models.py:430-432 builds payload["tools"] from bind_tools kwargs and lines 434-439 add federation kwargs; _generate:195 passes the entire payload as **kwargs to DistLLMClientSync.chat_completions. The SDK sync signature (client.py:614-627) and _build_chat_payload (158-187) accept neither tools nor federation keys and have no **kwargs, so chat_completions(tools=[...]) raises TypeError, and even the HTTP body never carries tools. LlamaIndex llms.py:112 advertises is_function_calling_model=True but uses the same client, so tool calls silently never transmit. Every LangChain bind_tools() call raises TypeError until the SDK chat_completions accepts and forwards tools. Advertised tool-calling is genuinely broken end-to-end.
  - *Correction:* SDK chat_completions signature is at sdk/src/distllm_sdk/client.py:614 (sync) / :399 (async), not :513. Federation kwargs (federation_strategy/preferred_regions/spillover_enabled) only break when explicitly configured (defaults None/[]/True are skipped), but the tools kwarg breaks unconditionally on every bind_tools() call.

---

### F-007 — [High] 🔒 VERIFIED · MUST-FIX CI test job is blocked: 79 live collection errors interrupt the whole suite

`pytest.ini:4` · zone=`tooling-tests` · category=`bug`
✅ **PARTIAL FIX (79 → 0 collection errors)** — fixed `reset_app_state_for_testing`/`g.get` (api_state), removed stale vectorstore fake bootstrap + `VectorStore`→`VectorDBInterface` test drift, removed `test_security_comprehensive`'s module-load fake of `distllm.dist.partition`, fixed `certificate_manager._encryption_algorithm` NameError. The 3 comprehensive-suite failures are resolved: constant-time compare test now asserts the real `api_key_store` location; max-msg-size test asserts the platform-safe bound (512MB — 2GB would OverflowError gRPC's Cython int on Windows); SQL-injection test asserts the validator rejects raw injection strings. `tests/security/test_security_comprehensive.py` now 51/51 green + node-service 68/68.

- **Summary:** The main CI test job runs `pytest -v ... -m "not e2e and not slow and not chaos"` over the entire tests/ tree (testpaths=tests in pytest.ini). Collection aborts at 79 errors, so every PR test job fails regardless of the 10017 passing tests. The errors split into: ~38 `ModuleNotFoundError: No module named 'psutil'` (environmental here - psutil is a declared core dep and CI's `pip install -e .[dev,testing]` provides it, so these pass in CI) and ~39 real drift (see sibling findings) that will fail CI too. Reproduction: `PYTHONPATH=src python -m pytest tests/ -q --co` on a provisioned venv.
- **Evidence (verbatim):**
```
PyTest run on current tree: 'Interrupted: 79 errors during collection !!!!!!!!!! 10017 tests collected, 79 errors in 21.40s'
```
- **Impact:** CI is red on every PR; the 10k-test suite cannot gate anything and coverage gates (`--cov-fail-under=80`, `fail_under=80`) are moot because the run never completes.
- **Effort:** 1-2 days
- **Reliability:** Dictated by the 79 errors listed in the collection traceback on the current src tree.
- **Recommendation:** Make collection hermetic before running: (1) provision dev env so psutil etc. are present (`pip install -e .[dev]`); (2) mark tests that import refactored-away private symbols `@pytest.mark.xfail(reason=...)` at the test-module level, or fix the imports (see sibling findings); (3) if a slow subset is being stabilised, add a `ci` marker and run `-m "not e2e and not slow and not chaos" --ignore=...` only on a curated list, and add a separate `pytest --collect-only --co` check to CI that fails fast with a clear message listing which modules cannot import.
- **Verdict (real):** mustFix=True. Verified by reproduction and source reads. `pytest.ini` testpaths=tests and ci.yml:49 run the whole-tree `pytest -v --cov=distllm ... -m "not e2e and not slow and not chaos"`. Running that exact CI-shaped command aborts collection: "Interrupted: 36 errors during collection" (exit 2, 10838/11090 collected). The 36 errors are in-repo, version-independent: (a) ~19-20 genuine missing-symbol drift — I confirmed each symbol (reset_app_state_for_testing, PrivacyAccountant, BudgetExhaustedError, best_fit_decreasing_partition, VectorStore, BackendCostMetrics, Draft202012Validator, ENV_REDACTION_ENABLED, Histogram, HybridClock, MetricSink, PowerMeter, CORSError, _cuda_reduction_callables, _is_safe_webhook_url, _jitter_backoff_delay, _validate_terraform_value, hash_api_key, reset_registry) has 0 occurrences in src/ and no dynamic re-export anywhere; (b) ~12 settings-poisoning errors from tests/config/test_resolver.py:18-26 which unconditionally injects a bare ModuleType into sys.modules["distllm.config.settings"] at import time — reproduced the exact '(unknown location)' failure after importing that module, and it breaks any later settings-importing test file process-wide. Some errors reproduce even when running a single directory alone (tests/core/vectorstore/ → 4, tests/regression_high/ → 15). The ~38 psutil errors in the original finding are environmental (psutil 7.2.2 present here; psutil>=5.9.0 is a declared dep in pyproject.toml lines 46/129 so CI installs it) and do not fail CI. The main CI `test` job therefore fails on every PR, and build-docker/benchmark/performance-report all depend on it via needs:[test]. Minor count correction: 36 non-psutil errors measured, not ~39; collected count 11090 vs 10017 — immaterial. This is a real Critical/High release blocker: the primary test gate is permanently red.

---

### F-008 — [High] 🔒 VERIFIED · MUST-FIX Unauthenticated HA heartbeat/snapshot fail-open when DISTLLM_HA_SECRET is unset (default) → leader-election takeover and coordinator state injection

`src/distllm/api/server.py:1434` · zone=`api-gateway` · category=`security`

- **Summary:** AuthMiddleware explicitly exempts /api/v1/ha/heartbeat (middleware.py 225-232), and the route (server.py 1421-1474) gates on X-HA-Secret ONLY inside `if expected_secret:` where expected_secret defaults to os.environ.get('DISTLLM_HA_SECRET','') — an empty string. If an operator has not set the secret (a common quickstart/single-node default), ANY unauthenticated POST is fully accepted: it can adopt an arbitrary, higher election term to force leadership seizure and inject peer_state, per ha_heartbeat -> election.handle_heartbeat_request (server.py 1471). The sibling /api/v1/ha/snapshot route (1363-1411) uses the same fail-open secret pattern and applies arbitrary nodes/metadata via coord.apply_state_snapshot(validated) (1408).
- **Evidence (verbatim):**
```
expected_secret = os.environ.get("DISTLLM_HA_SECRET", "") (1434); if expected_secret: ... only then compare X-HA-Secret (1435-1441); peer = election.handle_heartbeat_request(sender_id, term, peer_state) (1471)
```
- **Impact:** Unprivileged takeover of HA leader election and injection of a full coordinator state snapshot (fake nodes/metadata) across the cluster from an unauthenticated socket. CVSS ~8.6.
- **Effort:** 3-5 hours
- **Reliability:** Trigger: deploy with DISTLLM_HA_SECRET unset (default). Send POST /api/v1/ha/heartbeat with JSON {"coordinator_id":"evil","term":999999,"state":{...}} and no Authorization header. AuthMiddleware exempts the path; the secret guard is skipped because expected_secret==''; the higher term is adopted, seizing leadership. Also POST /api/v1/ha/snapshot with arbitrary nodes dict is applied.
- **Recommendation:** Default to fail-closed: require X-HA-Secret (reject with 403 when DISTLLM_HA_SECRET is unset AND the route is reached unauthenticated), or bind these endpoints to the same Bearer/cluster-key auth as other internal routes instead of a bespoke fail-open header. Document that HA replication must never run with an empty secret.
- **Verdict (real):** mustFix=True. Verified by reading the code. (1) middleware.py:225-232 explicitly exempts /api/v1/ha/heartbeat from AuthMiddleware. (2) server.py:1434-1441 gates X-HA-Secret ONLY inside `if expected_secret:` where expected_secret = os.environ.get("DISTLLM_HA_SECRET","") — empty/'' when unset, so the secret check is entirely skipped; the snapshot route (1380-1387) uses the identical fail-open pattern. (3) ha_coordinator.py handle_heartbeat_request (216-275) confirms impact: a higher term (245-250) forces the node to FOLLOWER (leadership seizure), and `if state and self._state==FOLLOWER: self._replicated_state.update(state)` (262-263) applies arbitrary peer_state (state injection). So an unauthenticated POST to a middleware-exempt, state-mutating, leadership-affecting endpoint is fully accepted whenever the operator has not set DISTLLM_HA_SECRET (the documented default). This is a genuine fail-open security anti-pattern on a sensitive endpoint and nothing enforces that the secret be set when HA is enabled; it should fail closed (require the secret / refuse when HA enabled without it). One correction: the sibling snapshot route is NOT middleware-exempt, so it still requires an app API key — the primary unauthd vector is the heartbeat route, which the finding's headline correctly targets.
  - *Correction:* Minor overstatement: /api/v1/ha/snapshot is not in the AuthMiddleware exemption list (only /api/v1/ha/heartbeat is, middleware.py:230), so snapshot remains behind app API-key auth; its DISTLLM_HA_SECRET layer does fail open but is defense-in-depth rather than an unauthd vector. The confirmed unauthd vector is the heartbeat route.

---

### F-009 — [High] 🔒 VERIFIED · MUST-FIX MLX backend forward() scores each token in isolation with no KV state — corrupts any context>1 pipeline forward

`src/distllm/backends/mlx_backend.py:102` · zone=`backends-config-cloud` · category=`bug`

- **Summary:** MLXNodeAdapter._forward_input_ids loops over tokens, calling self._model(mx.array([token_id]).reshape(1,1)) per token with no past_key_values carry-over and ignoring attention_mask/position_ids. Each token is evaluated as a length-1 sequence, so tokens after the first get no causal attention over prior tokens -> the concatenated 'logits' are wrong except for a length-1 input. It also only reads the first batch row (.tolist()[0]). Since priority_for('mps')==10, MLX is the selected backend on Apple Silicon.
- **Evidence (verbatim):**
```
for token_id in ids_np: logits = self._model(mx.array([token_id]).reshape(1, 1)); logits_list.append(logits)
```
- **Impact:** Wrong outputs for any multi-token input on Mac/MLX deployments (the priority-10 backend for mps); silent corruption rather than a loud error.
- **Effort:** 2-4 hours
- **Reliability:** Line 110-117: per-token standalone eval with no KV cache; _forward_full_model routes hidden_states through argmax to the same method (line 124).
- **Recommendation:** Run the full sequence in one call and thread a KV cache through, or reject/route any input with seq_len>1 to a proper autoregressive path. At minimum carry attention state and batch dim across tokens instead of re-scoring each token standalone. Add a regression test asserting logits for tokens[1] depend on tokens[0].
- **Verdict (real):** mustFix=True. Verified by reading D:/distributed-llm/src/distllm/backends/mlx_backend.py. Line 107 reads only the first batch row (.tolist()[0]); lines 110-112 loop for token_id in ids_np calling self._model(mx.array([token_id]).reshape(1,1)) with no past_key_values carry-over and no attention_mask/position_ids; line 116 concatenates per-token logits along axis=1. Each token is scored as a length-1 sequence, so tokens after the first get no causal attention over prior context — logits are only correct for length-1 input. _forward_full_model (line 124) routes through the same corrupt path, so this is the entire MLX forward path. priority_for('mps')==10 (line 183) and BackendRegistry.select sorts by priority and returns 'mps' on Apple Silicon, so MLX is auto-selected there; adapter.forward(input_ids=...) is the canonical protocol inference path used by distributed/NodeService. This is a silent-correctness (wrong logits) defect on a supported backend's primary forward path — a genuine High that must be fixed before release. Nuance: an mlx_lm model isn't directly callable with a lone array, so the call may also raise rather than only produce wrong logits, but the defect (per-token no-KV forward) is exactly as described.

---

### F-010 — [High] 🔒 VERIFIED · MUST-FIX NIM fallback fabricates hash-scattered 'logits' in _forward_via_api instead of failing loudly

`src/distllm/backends/nim_backend.py:415` · zone=`backends-config-cloud` · category=`bug`

- **Summary:** When no local_model is provided, NIM _forward_via_api builds a fake logit tensor sized len(top)*4 and places np.log(prob) at idx = hash(token_str) % size. Python's built-in hash() is randomized per-process (PYTHONHASHSEED), so identical tokens land at different vocab indices across processes, and the tensor is not a real model output. The alternative path (lines 418-421) sets a one-hot 1.0 at a token index in a hardcoded 32000-wide tensor. This silently corrupts downstream inference instead of the graceful-degradation path refusing.
- **Evidence (verbatim):**
```
for token_str, prob in top.items(): idx = hash(token_str) % logit_tensor.shape[-1]; logit_tensor[0, 0, idx] = np.log(max(prob, 1e-10))
```
- **Impact:** Fabricated, non-reproducible logits silently fed to the pipeline for NIM nodes without a local model; incorrect argmax/next-token choice. Also grep PYTHONHASHSEED nondeterminism breaks any caching.
- **Effort:** 2-3 hours
- **Reliability:** Path reached when input_ids given and self._local_model is None (line 332-334); hash() randomness per process is documented Python behavior.
- **Recommendation:** Raise NotImplementedError (as WebGPUNodeAdapter does) when pipeline-mode forward requires logits but no local model is set. If a real score must be returned, decode the actual vocabulary logits from NIM (it exposes top_logprobs only, which is insufficient for a full logit distribution).
- **Verdict (real):** mustFix=True. Verified in D:/distributed-llm/src/distllm/backends/nim_backend.py lines 404-421. When no local_model is set, _forward_via_api (reached via _forward_input_ids at line 334 whenever self._local_model is None — a designed, documented API-only path) fabricates a synthetic logits tensor instead of raising. Primary path scatters np.log(prob) at idx = hash(token_str) % shape[-1]; because Python's str hash is salted per-process (PYTHONHASHSEED), identical tokens land at different indices across processes, and the tensor is only len(top)*4 wide (a few entries) with no mapping from the hash-derived index back to a real vocab id — so downstream argmax/sampling cannot resolve actual tokens. The alternative one-hot path (lines 418-421) hardcodes a 32000-wide tensor, wrong for many models and also not a real output. This is silent corruption in a reachable path rather than graceful refusal. The mechanism and cite are accurate; it is load-bearing enough to require a fix (fail loudly / refuse) before release.

---

### F-011 — [High] 🔒 VERIFIED · MUST-FIX Azure availability check always fails open: URL has literal {subscriptionId} placeholder and no auth

`src/distllm/cloud/azure.py:121` · zone=`backends-config-cloud` · category=`bug`

- **Summary:** AzureAvailabilityChecker.check_availability posts to 'https://management.azure.com/subscriptions/{subscriptionId}/providers/...' with no bearer token. The unfilled literal brace is not valid, and the ARM API requires auth, so every call raises and is swallowed by the bare except, returning available=instance_type in _AZURE_GPU_INSTANCES (always True for known types). The scheduler therefore believes availability in any region, including regions/capacity that do not exist.
- **Evidence (verbatim):**
```
"https://management.azure.com/subscriptions/{subscriptionId}/providers/Microsoft.Compute/locations/{region}/vmSizes"
```
- **Impact:** Over-provisioning/scheduling onto unavailable Azure regions; silent false-positive availability.
- **Effort:** 3-4 hours
- **Reliability:** except Exception: return AvailabilityInfo(... available=instance_type in _AZURE_GPU_INSTANCES) (lines 134-138) always True for known types.
- **Recommendation:** Either implement real auth (DefaultAzureCredential) and query the actual /skuInRegion endpoint, or delete the client call and expose quota/availability from a supported API. At minimum make the fallback fail-closed (available=False) on any error so the scheduler does not over-provision.
- **Verdict (real):** mustFix=True. Verified directly at D:/distributed-llm/src/distllm/cloud/azure.py lines 120-138. The URL is exactly as cited: 'https://management.azure.com/subscriptions/{subscriptionId}/providers/Microsoft.Compute/locations/{region}/vmSizes'. The braces are literal (not an f-string), so the region argument is never used and subscriptionId is undefined; no auth token is attached, so the ARM vmSizes call always fails. Every failure is swallowed by the bare except (line 134), returning available = instance_type in _AZURE_GPU_INSTANCES (always True for known SKUs) regardless of region. This AzureAvailabilityChecker is wired into the public ProviderSession.check_availability API (common.py lines 171-186), so the Azure availability feature is entirely non-functional and region-agnostic, reporting capacity in any region including ones that do not exist. Must fix as High.
  - *Correction:* Severity is High, not Critical: the real check is dead code that always fails open via the bare except fallback (returns available=True for any known SKU) rather than crashing or returning wrong-on-purpose results. The region parameter is completely ignored (never interpolated into the URL), so the checker cannot distinguish regions at all. Otherwise the finding is exactly accurate (literal {subscriptionId}/{region} braces, no bearer token, swallowed exception).

---

### F-012 — [High] 🔒 VERIFIED · MUST-FIX Aether LoRA path never trains the adapter; merge of the zero-B adapter is a no-op

`src/distllm/core/aether_federated.py:921` · zone=`core-training` · category=`bug`

- **Summary:** start_finetuning() with lora_config creates a LoRA adapter, but then trains on the FULL base weights; the adapter's random A / zero B are never updated. merge() then adds (alpha/rank)*(0@A^T)=0, returning final_weights unchanged. 'used_lora' is True but the LoRA feature does nothing.
- **Evidence (verbatim):**
```
adapter_id = self._adapter_manager.create(base_model_weights=base_model, ...) ... final_weights = self._trainer.train(model_weights=base_model, ...) merged = self._adapter_manager.merge(adapter_id, final_weights)
```
- **Impact:** LoRA-based federated fine-tuning silently produces the base model; users believe they adapted weights on private data but did not.
- **Effort:** 1-2 days
- **Reliability:** start_finetuning(lora_config=...) -> merged weights == final_weights because B is zero-init and unused.
- **Recommendation:** Train over LoRA parameters (compose base+LoRA so local_train_fn returns LoRA grads, accumulate, merge trained A/B). At minimum, document that lora_config is advisory and assert when the adapter was not actually trained.
- **Verdict (real):** mustFix=True. Verified by reading aether_federated.py. In LoRAAdapterManager, A/B are set ONLY in create() (lines 199-200, 206) and never updated — the class has no method that writes learned deltas back into self._adapters. start_finetuning() (line 864) calls create() (line 921) to make the adapter, then trains the FULL base weights via FederatedTrainer.train(model_weights=base_model, ...) (line 941) which never touches the adapter, then merge() (line 952) computes delta = B @ A.T = 0 (B still all zeros), so merged == final_weights unchanged, and unload() discards it. The final weights are pure full-weight training; the adapter contributes zero delta. used_lora=True misleads callers that LoRA was actually used. This is a genuine functional defect in a named headline feature ('Distributed LoRA training', line 544), so it is load-bearing and must be fixed before release.

---

### F-013 — [High] 🔒 VERIFIED · MUST-FIX ApiKeyRotator never retires the old (possibly compromised) key — rotation is a no-op for security

`src/distllm/core/cert_rotation.py:300` · zone=`core-priv-sec` · category=`security`

- **Summary:** `ApiKeyRotator.rotate()` registers a replacement key with the same key_id but leaves the OLD key's entry in `ApiKeyStore._keys`; the old key is only removed by `cleanup_expired()` (which calls `remove_key_hash`). Grep across src shows `is_rotated_key_valid`, `cleanup_expired` (from cert_rotation), and any `ApiKeyRotator`/`rotator.` usage are never invoked anywhere in the api/core paths — the grace-period retirement logic is dead code. After rotation the old (potentially leaked) key authenticates forever, so rotating provides no security benefit.
- **Evidence (verbatim):**
```
self._key_store.add_key(new_key, role=target.get("role","admin"), key_id=key_id)  # old entry stays ...old_hash and self._key_store is not None: self._key_store.remove_key_hash(old_hash)  # only in cleanup_expired
```
- **Impact:** A compromised/rotated API key stays valid indefinitely, defeating key rotation and leaving a persisted credential live.
- **Effort:** 4-8 hours
- **Reliability:** grep for `is_rotated_key_valid|cleanup_expired|ApiKeyRotator|rotator\.` in src returns only definitions in cert_rotation.py; `cleanup_expired` matches in src/distllm/dist/p2p/gossip.py are a different class. ApiKeyStore.authenticate iterates all `_keys` entries, so the retained old entry still validates.
- **Recommendation:** Wire the rotation lifecycle: schedule `cleanup_expired()` from a background timer (or from the auth path) so old hashes are retired once grace elapses, and add an integration test asserting that after rotate+grace-elapse the old token no longer authenticates. Add the ApiKeyRotator to the API server startup so 'cluster key rotation with grace period' (claimed in api/CLAUDE.md) actually runs.
- **Verdict (real):** mustFix=True. Verified true by reading the code. (1) cert_rotation.py lines 300-305: rotate() calls add_key(new_key, key_id=key_id) but never removes the old entry; api_key_store.py add_key() (lines 151-182) APPENDS a StoredKey so it coexists with the old one (docstring confirms "coexists with the old entry so both authenticate"), and authenticate() (lines 137-149) matches all hashes in _keys — so the old leaked key stays valid. (2) The only retirement path is cleanup_expired() -> remove_key_hash(old_hash) (cert_rotation.py lines 324-338); rotate() itself does not retire. (3) Grep confirms ApiKeyRotator/is_rotated_key_valid/this cleanup_expired are referenced only inside cert_rotation.py plus the test test_p6_regressions.py; the production auth path (AuthMiddleware in middleware.py, lines 188+) uses only get_api_key_store()/authenticate() and never wires the rotator. compliance_evidence.py:382 instantiates CertificateRotator (the TLS rotator, unrelated) and only calls check_certificate(). No scheduler/CLI/server/management endpoint invokes any rotator method. (4) The regression test only retires the old key by explicitly calling cleanup_expired(); since nothing does so in production, an old key never dies. This is a real security-domain defect: a shipped, documented public security primitive advertises post-leak grace-period retirement that silently never happens, so rotation provides no security benefit and would mislead an operator into believing a compromised key is dead. mustFix=true: it is a false-security-guarantee in the core-priv-sec zone and should be either wired into a rotation endpoint + scheduled cleanup, or removed/documented as unsupported before release. Severity note: it is dead/unwired code (no active exploit path in the default deployment), so the finding reads slightly strong as "no-op for security" — the more precise framing is "incomplete/unwired rotation facility that never retires keys" — but the security impact claim itself is accurate.

---

### F-014 — [High] 🔒 VERIFIED · MUST-FIX IntelligentAutoscaler wired but never fed or actuated — scales nothing

`src/distllm/core/coordinator.py:1071` · zone=`core-ops-ha` · category=`bug`

- **Summary:** coordinator.start() instantiates IntelligentAutoscaler via _start_subsystem and calls record_metrics exactly once with startup scheduler stats. There is no periodic loop invoking evaluate(), no gpu_utilization is ever populated (ScalingMetrics defaults it to 0.0), and no callback scales actual nodes from target_nodes. The 'wired' autoscaler is effectively dead: it can neither observe load nor act.
- **Evidence (verbatim):**
```
autoscaler = self._start_subsystem(     "autoscaler", "distllm.core.intelligent_autoscaler",     "IntelligentAutoscaler", "_autoscaler",     constructor_kwargs={"min_nodes": 1, "max_nodes": 20, "target_utilization": 0.7})
```
- **Impact:** Autoscaling is non-functional in the core coordinator: utilization is always treated as 0 and the scaler can never add a node, so the flagship autoscaling feature is inert and silently misleading.
- **Effort:** 4-8 hours
- **Reliability:** Only one record_metrics(ScalingMetrics(active_requests, pending_requests, current_nodes)) call exists (L1081-1085); gpu_utilization stays 0.0, so _reactive_target (L158) always sees util<scale_down and scale-downs to min. No autoscaler.evaluate() caller, no node-add callback anywhere.
- **Recommendation:** Run a background loop (e.g. every 5-10s from the health/defrag thread) that builds ScalingMetrics with real gpu_utilization from GPUResourceManager/SystemMonitor, calls evaluate(), and applies target_nodes via a provisioning callback into ClusterManager. If not needed now, remove the dead instantiation to avoid implying it works.
- **Verdict (real):** mustFix=True. Verified by reading coordinator.py (lines 251, 891-935, 1071-1087) and the IntelligentAutoscaler class (intelligent_autoscaler.py). The finding is accurate on all four claims: (1) record_metrics is called exactly once at startup with a ScalingMetrics carrying only active_requests/pending_requests/current_nodes; (2) _autoscaler has no other reference anywhere in src/ — no periodic loop calls evaluate(), because _start_subsystem merely instantiates and stores the object with no timer/thread; (3) gpu_utilization is omitted from the recorded metrics and permanently stays at the dataclass default 0.0, which both _reactive_target and _predict_load depend on, so evaluation would be vacuous even if invoked; (4) no scale-to callback consumes ScalingDecision.target_nodes — there is no actuation path at all. Correction to add: the identical dead wiring is also present in coordinator_subsystem.py:386-401, and even a correctly-fed periodic evaluate() would be inert because no scale-actuation callback exists anywhere. This is an entire ops/HA feature (autoscaling) that observes no load and acts on nothing — genuine High-worthy dead code in production path, not cosmetic. Setting mustFix=true, though I note severity is High (dead/misleading feature) rather than Critical (no crash, data loss, or security impact).
  - *Correction:* Defect also exists identically in coordinator_subsystem.py:386-401, and no scale-actuation callback exists anywhere, so even a periodic evaluate() would be inert.

---

### F-015 — [High] 🔒 VERIFIED · MUST-FIX HA leader election never gates request processing — standby coordinators serve requests

`src/distllm/core/coordinator.py:426` · zone=`core-ops-ha` · category=`bug`

- **Summary:** CoordinatorElection/RayFaultTolerance elect a leader and track is_leader, but nothing in the request path consults it. generate()/request_pipeline execute on every coordinator regardless of leadership, so in HA mode all coordinators act as writers and can diverge — the split-brain protection the election exists to provide is not enforced. Election is effectively decorative.
- **Evidence (verbatim):**
```
def is_leader(self) -> bool:     return self._election.is_leader  # property only; never read by request path
```
- **Impact:** In a 2+ coordinator HA deployment, standby nodes accept and serve inference while also applying leader snapshots on top of their own mutations, producing divergent KV/node state and violating the single-writer invariant that the quorum election is supposed to guarantee.
- **Effort:** 4-8 hours
- **Reliability:** is_leader defined (L426) and grep shows zero references in coordinator_request.py / request_pipeline.py / inference_engine.py; generate() (coordinator_request.py L29) has no leader/standby branch.
- **Recommendation:** Gate admission at request entry: if HA is enabled and not leader, return a 'forward to leader' response (or reject) in RequestHandler.generate/generate_async and in the API layer. Add '_is_standby' checks and a test that a follower refuses/forwards while a leader serves.
- **Verdict (real):** mustFix=True. Verified by reading the full request path. The cited coordinator.py:426-428 `is_leader` property exists exactly as claimed and is a read-only passthrough to `self._election.is_leader`. A repo-wide grep confirms nothing in the request/API path ever consults it: coordinator.generate (coordinator.py:520-532) delegates straight to _request_handler.generate with no check; coordinator_request.generate (line 29) has no leader/standby reference; request_pipeline generate/generate_async/generate_batch (115/359/448) have none; the API server has no leader-forwarding and no non-leader rejection, only the passive /api/v1/ha/snapshot receive endpoint. The in-code contract is explicit but unenforced: coordinator_election.enable_ha's docstring promises "Only the leader accepts requests; standbys replicate state and wait for failover," and ha_coordinator.py:18-24 shows the intended `if is_leader(): handle_requests() else: forward to leader` pattern — neither exists in any code path. So in HA mode every coordinator, including standbys, acts as a writer and generator, and diverged serving state is silently possible with zero split-brain protection — the election is genuinely decorative for request gating as claimed. Severity is a genuine High (not merely a stylistic gap) because the documented failure domain (multi-writer divergence) directly defeats the stated purpose of the HA election feature, and the fix is a simple standby guard in the request path that returns a redirect/503 to the leader. Mitigating nuance acknowledged: external LB/gateway routing to a single active coordinator is a plausible compensating control, and leader→standby snapshot replication every ~10s bounds divergence; but those are not implemented or guaranteed, and the coordinator's own advertised invariant is not delivered. One minor correction: the finding's technical evidence (is_leader read nowhere) is correct; no detail needs fixing.

---

### F-016 — [High] 🔒 VERIFIED · MUST-FIX DP 'inference' applies NO differential privacy to outputs while still charging the tenant's privacy budget

`src/distllm/core/dp_inference.py:929` · zone=`core-priv-sec` · category=`security`

- **Summary:** The advertised DP inference path (`generate`/`generate_stream`, budget enforcement, per-tenant accounting) never actually perturbs outputs. It calls the raw `self._engine.generate_stream(...)` and records `_estimate_epsilon_cost(sigma, token_count)` into the tenant budget, only logging a `logger.warning`. `_dp_sample()` (the real DP mechanism) is never called anywhere in src (confirmed by grep). A tenant is told their epsilon budget is being spent and can be refused once exhausted, while the tokens they receive carry zero DP guarantee — a false-privacy guardrail. The docstring warns honestly, but runtime fails open with a warning, not an error.
- **Evidence (verbatim):**
```
if hasattr(self._engine, "generate_stream"):     stream = self._engine.generate_stream(...)     tokens = list(stream); text = "".join(tokens)     eps_cost = self._estimate_epsilon_cost(sigma, token_count)     self._budget_manager.record_query(...)
```
- **Impact:** Customers paying for a DP / HIPAA-informed 'privacy budget' receive plaintext outputs while believing they are protected; enforcement blocks legitimate tenants on a budget that protects nothing.
- **Effort:** 2-4 days
- **Reliability:** grep for `_dp_sample`/`dp_noise_injection`/`apply_dp_to_logits` shows only definition sites and string references in warnings — never an invocation in any generation path. The prior analysis doc (DISTLLM_CORE_COMPREHENSIVE_ANALYSIS.md C8) flags the same.
- **Recommendation:** Either (a) wire the mechanism into the logit path so outputs are actually DP — have the engine's sampling call `_dp_sample`/`apply_dp_to_logits` on raw logits before argmax/multinomial, and make `generate` refuse (raise) when DP noise cannot be applied (fail-closed), or (b) remove the privacy claim until integrated. At minimum, when `_enforce` is enabled, raise an error if the target_mechanism is not actually applied rather than returning plaintext.
- **Verdict (real):** mustFix=True. Confirmed by reading src/distllm/core/dp_inference.py. generate()/generate_stream (lines 929-949) call the raw engine.generate_stream with zero DP perturbation; budget is still charged via _estimate_epsilon_cost + record_query (942-943) and enforcement refuses tenants at 904-911. grep across src shows _dp_sample appears ONLY at its definition (line 975) and in comment/log strings (922, 926) — never called. apply_dp_to_logits (829) and dp_noise_injection/gumbel_noise_mechanism likewise have no generation-path call site. Only a logger.warning (923-927) is emitted; the docstring (867-880) honestly admits the mechanism is unwired. tests/core/test_dp_inference.py pins this behavior (test_generate_with_generate_stream asserts raw passthrough; _dp_sample tested only standalone). This is a false-privacy guardrail — tenant budget spent/denied with no DP guarantee delivered — load-bearing for a documented privacy feature being hardened for production. One correction: _dp_sample is NOT a no-op placeholder; it is a real Gaussian L2-clip+softmax+multinomial DP sampler (975-996) that is simply never wired in. The core finding stands.
  - *Correction:* _dp_sample is not a no-op/placeholder — it is a functional Gaussian DP sampler (L2-clip + noise + softmax + multinomial, lines 975-996). The real defect is it is never called in the generate/generate_stream path, not that it is fake. Everything else (raw engine passthrough, budget charged, fails open with warning, missing call sites) matches exactly.

---

### F-017 — [High] 🔒 VERIFIED · MUST-FIX DifferentialPrivacyInference.generate() non-streaming branch raises NameError (dead/broken code)

`src/distllm/core/dp_inference.py:953` · zone=`core-priv-sec` · category=`bug`

- **Summary:** The `elif hasattr(self._engine, "generate")` branch in `DifferentialPrivacyInference.generate()` references `token_count` and `text`, which are only assigned inside the earlier `if hasattr(...generate_stream)` block. Any engine that exposes only `generate()` (not `generate_stream`) crashes with NameError before producing a result. The top-of-module docstring advertises exactly this usage with InferenceEngine, but InferenceEngine has generate_stream, so the non-streaming path is untested dead code that will crash.
- **Evidence (verbatim):**
```
elif hasattr(self._engine, "generate"):     eps_cost = self._estimate_epsilon_cost(sigma, token_count)     self._budget_manager.record_query(user_id, sigma, epsilon_cost=eps_cost)     return DPGenerationResult(text=text, ... token_count=token_count
```
- **Impact:** Crashes any DP wrapper whose engine lacks generate_stream, making the documented non-streaming DP path unusable (NameError at runtime).
- **Effort:** Under 1 hour
- **Reliability:** Engine with only `generate()` -> `DifferentialPrivacyInference(engine, ...).generate(prompt)` -> `elif hasattr(self._engine, "generate")` branch -> reads undefined `token_count`/`text` -> NameError. `text` and `token_count` are bound only at lines 940-941 in the generate_stream block.
- **Recommendation:** In the non-streaming branch, call `self._engine.generate(...)` to obtain text/token_count (compute token_count from the returned text), then record budget. Alternatively collapse both branches to a single code path: get `(text, token_count)` from either generate_stream or generate, then record + return. Add a unit test driving an engine that only implements generate().
- **Verdict (real):** mustFix=True. Confirmed by reading D:/distributed-llm/src/distllm/core/dp_inference.py:929-963. The `elif hasattr(self._engine, "generate")` branch references `token_count` (lines 953, 961) and `text` (line 959), which are only ever assigned at lines 940-941 inside the earlier `if hasattr(self._engine, "generate_stream")` block. The elif branch assigns neither, so it raises NameError on token_count at line 953. It also never calls `self._engine.generate(...)` at all, so even if the NameError were avoided the branch would return a DPGenerationResult with undefined text and no generation — it is entirely non-functional. Reachability: currently dead with bundled engines (every class in inference_engine.py defines both generate and generate_stream), but the API contract explicitly advertises generate-only support (module docstring lines 16-34, and the else TypeError at lines 966-969 says "does not have a 'generate' or 'generate_stream' method"), so any engine exposing only generate() will hit this and crash. A confirmed, trivially-fixable latent NameError crash in an advertised path within a privacy-critical zone; fixing requires either computing token_count/text in the generate branch or dropping support for the invented engine. Corrected detail: the finding's framing slightly understates — the branch is not merely a NameError, it omits the actual generation call entirely.

---

### F-018 — [High] 🔒 VERIFIED · MUST-FIX Aether 'secure' aggregation is a no-op and its pairwise masks use a single public seed

`src/distllm/core/federated_finetuner.py:178` · zone=`core-training` · category=`security`

- **Summary:** FederatedFineTuner._received_grads is keyed by peer_id and never cleared or round-filtered. train_round() broadcasts {'gradients':..., 'round': r} but _receive()/store ignores 'round'. If a peer is slow on round N, its round-1 gradients are averaged 1/N with fresh local grads in round N, silently diverging training.
- **Evidence (verbatim):**
```
peer_data = self._receive(timeout=30.0) if peer_data and "gradients" in peer_data:     peer_grads = peer_data["gradients"]     self._received_grads[peer_data.get("peer_id", "unknown")] = peer_grads
```
- **Impact:** Non-deterministic, unstable federated training under any straggler/failure; no round-version guard.
- **Effort:** 3-5 hours
- **Reliability:** Round 1 peer A submits; round 2 A silent (slot keeps round-1 grads); round 2 merged includes A's round-1 grads at equal weight.
- **Recommendation:** Store (round, grads) per peer, drop entries whose round != current round before averaging, prune peers not seen this round, and guard against layer-count mismatches before averaging.
- **Strategic value:** Real SecAgg is the differentiator that lets DistLLM sell private federated fine-tuning to regulated tenants; current implementation misround field
- **Verdict (real):** mustFix=True. Confirmed by reading D:\distributed-llm\src\distllm\core\federated_finetuner.py. `_received_grads` (line 84) is keyed by peer_id and never cleared, re-initialized, or round-filtered anywhere in the file. Line 170 broadcasts `{"gradients":..., "round": self._round}`, but lines 178-181 store received gradients solely by peer_id, never reading/validating `round`. `_receive` is called exactly once per `train_round` (line 178). Thus if a peer that contributed round-1 gradients stops responding (or a late/out-of-order gossip message is consumed in a later round), its stale gradients persist in the dict and are averaged with fresh local gradients in every subsequent round via `_average_gradients` (lines 211-228), silently diverging the FedAvg/FedProx objective. This is a genuine, mechanically-demonstrable correctness bug, load-bearing enough to count as Critical/High. One title correction: the 'Aether secure aggregation / pairwise masks / single public seed' portion does not live in this file — secure aggregation is delegated to federated_merge.py (line 19); the substantive stale-gradient/round-ignored claim is fully accurate here.

---

### F-019 — [High] 🔒 VERIFIED · MUST-FIX FedProx term subtracts a gradient from a weight: mu*(grad - global_param)

`src/distllm/core/federated_finetuner.py:254` · zone=`core-training` · category=`bug`

- **Summary:** _apply_fedprox_term computes proximal = mu * (grad - global_param), mixing a gradient tensor with a weight tensor elementwise; the correct form is grad + mu*(w_local - w_global). It double-counts mu*grad, never uses the local weight, and may raise shape errors when grad and global_param shapes differ.
- **Evidence (verbatim):**
```
proximal = self._fedprox_mu * (grad - global_param.detach()) proximal_grads.append(grad + proximal)
```
- **Impact:** FedProx produces a mathematically invalid update, so heterogeneous-data federated training is unreliable when fedprox_mu>0.
- **Effort:** 2-4 hours
- **Reliability:** algorithm='fedprox', fedprox_mu>0: proximal term adds mu*grad (double count) and subtracts mu*global weight.
- **Recommendation:** Compute grad + mu*(w_local - w_global) using the pre-round local weights. If only gradients are available, apply delta = mu*(w_local - w_global) as a gradient offset without re-scaling the raw gradient by mu.
- **Verdict (real):** mustFix=True. Read the file around lines 245-260. Cited evidence matches verbatim: line 254 `proximal = self._fedprox_mu * (grad - global_param.detach())` and line 255 `proximal_grads.append(grad + proximal)`. FedProx requires the proximal gradient term mu*(w_local - w_global) added to grad; the code instead adds mu*(grad - global_param), i.e. (1+mu)*grad - mu*global_param. This double-counts mu*grad, never references w_local (the local weight, which the function does not even receive), and subtracts a weight tensor from a gradient tensor elementwise (shape risk when grad is a LoRA delta vs full-model global params). The defect is unambiguous math; when the documented FedProx feature is enabled every gradient update in every round is corrupted, so it is load-bearing and merits fixing before release.
  - *Correction:* Confirms. federated_finetuner.py lines 254-255 compute proximal = mu*(grad - global_param); grad + proximal => (1+mu)*grad - mu*global_param, which is wrong. Correct FedProx update is grad + mu*(w_local - w_global). The code double-counts mu*grad, never uses the local weight (only grads are passed to _apply_fedprox_term), and mixes a gradient tensor with a weight tensor elementwise (shape-mismatch risk vs large/full model params). Active only when algorithm='fedprox' AND fedprox_mu>0 AND global_model_params set; but FedProx is a documented supported algorithm, so it is broken whenever enabled — despite being default-off, this is a real math error that should be fixed before a release.

---

### F-020 — [High] 🔒 VERIFIED · MUST-FIX GPUResourceManager.snapshot reports used_mb and free_mb swapped

`src/distllm/core/gpu_resource_manager.py:231` · zone=`core-ops-ha` · category=`bug`

- **Summary:** In snapshot(), device_alloc is set to torch.cuda.memory_allocated(device) which is the USED (allocated) memory, but it is returned as free_mb, while used_mb is computed as total - device_alloc (which is actually the free bytes). Callers inspecting snapshot get free==used and used==free, which feeds autoscaling/monitoring/eviction with inverted memory numbers.
- **Evidence (verbatim):**
```
return GPUMemorySnapshot(     device=device, total_mb=total, used_mb=total - device_alloc,     free_mb=device_alloc, reserved_mb=device_reserved,
```
- **Impact:** Memory utilization, safe_margin and OOM-risk reporting are wrong; is_oom_risk and any dashboard/autoscaler fed from snapshot read inverted headroom, risking premature eviction or false OOM safety.
- **Effort:** under 1 hour
- **Reliability:** torch.cuda.memory_allocated(i) returns reserved-tensor bytes (used). used_mb=total-alloc therefore equals free; free_mb=alloc equals used. Contrast _free_mb (L285) which correctly returns total-alloc.
- **Recommendation:** Swap the two fields: used_mb=device_alloc, free_mb=total-device_alloc. Add a unit test asserting snapshot.free_mb + snapshot.used_mb == total under a mocked memory_allocated.
- **Verdict (real):** mustFix=True. Verified by reading src/distllm/core/gpu_resource_manager.py lines 230-232. `free_mb=device_alloc` assigns the ALLOCATED (used) bytes to free, and `used_mb=total-device_alloc` assigns the FREE bytes to used. This directly contradicts the correct sibling helpers in the same file: `_free_mb` (line 285) returns `total - alloc` as free, and `is_oom_risk` (line 252) correctly computes `used = total - self._free_mb`. The labels in snapshot() are exactly inverted relative to both field name semantics and every other method. No out-of-file consumer of the .used_mb/.free_mb fields was found in-tree (the only caller, dist/draft_migration.py:234, reads only utilization_pct), so nothing crashes today; but GPUMemorySnapshot is a public telemetry/monitoring dataclass and the inversion is in the worst direction (free_mb reports its largest value when the GPU is most saturated), so any autoscaling/eviction/OOM-guard/capacity-planning consumer would make backwards decisions. It is an unambiguous, correctable, core-availability defect and should be fixed before release.

---

### F-021 — [High] 🔒 VERIFIED · MUST-FIX int8/int4/adaptive compressed KV cache serves raw (non-dequantized) tensors to attention

`src/distllm/core/kv_cache.py:159` · zone=`core-cache` · category=`bug`

- **Summary:** KVCache.get() only dequantizes the bulk-compressed path when _quant_fp8 is True. compress("int8"), compress("int4"), and AdaptiveQuantizer.apply() all set _quant_fp8=False while populating _scale_k/_scale_v, so get() falls through to 'return k, v  # Return raw quantized if no scale available' and returns [-128,127]/[-7,7] quantized values (roughly scalar /scale of the true fp16 values) to the attention/decoder. This silently corrupts generation whenever non-FP8 bulk compression is used.
- **Evidence (verbatim):**
```
if self._quant_fp8 and layer_idx < len(self._scale_k): k_deq = (k.float()*self._scale_k[layer_idx]).to(k.dtype); ... return k_deq,v_deq return k, v  # Return raw quantized if no scale available
```
- **Impact:** Wrong KV values fed to attention for int8/int4/adaptive compression => incorrect (often nonsensical) completions; the 2-4x memory savings is unusable for these methods. Numeric-correctness regression only reachable when non-fp8 compression is enabled.
- **Effort:** 2-4 hours
- **Reliability:** cmp=KVCache(); cmp.init_cache(1,1,1,8); cmp.compress('int4'); cmp.get(0) returns int8 tensor in [-7,7] (raw q = round(k/scale)) rather than k. fp8 works because _quant_fp8=True gates the dequant branch; int8/int4/adaptive set _quant_fp8=False (lines 551,586,1232).
- **Recommendation:** Dequantize by scale for ALL quantized bulk paths, e.g. compute base scale and dtype per layer: for layer_idx with self._scale_k[layer_idx] is not None and self._quant_bits in (4,8): k_deq = ((k.float() * self._scale_k[layer_idx]) if int8/int4). Add a regression test asserting get() after compress('int4') recovers approx original values (not merely cast). Remove the fp8-only gate.
- **Verdict (real):** mustFix=True. Confirmed by reading src/distllm/core/kv_cache.py. KVCache.get() (lines 159-166) dequantizes the bulk-compress path ONLY when `self._quant_fp8` is True. compress("int8") (line 551), compress("int4") (line 585), and AdaptiveQuantizer.apply() (lines 1229-1233) all set `_quant_fp8=False` while populating `_scale_k`/`_scale_v`. Tracing get(): line 151 (dequantized path) and line 155 (incremental `_qsegments` path) are both skipped for bulk compression, so execution falls to line 159 then, because `_quant_fp8` is False, skips dequant and reaches line 166 `return k, v` — returning raw int8/int4 tensors to attention. The populated scales are never used. The docstring at line 147 explicitly promises "dequantized on-the-fly using stored scale factors" for the bulk path, so this violates the documented contract. Since attn-weights @ V is not softmax-normalized, serving raw quantized (scaled-up, int8 dtype) values corrupts the decoder output silently. This is a load-bearing release-blocking correctness bug for all non-FP8 bulk/adaptive compression. Minor imprecision in the claim: q ≈ k*(127/max_abs), i.e., k scaled UP, not "scalar /scale", but the core assertion is correct.

---

### F-022 — [High] 🔒 VERIFIED · MUST-FIX request_latency elapsed_ms grows with wall clock on COMPLETED records, corrupting SLA compliance percentiles

`src/distllm/core/request_latency.py:29` · zone=`core-perf-obs` · category=`bug`
✅ **FIXED** — added `completed_at`; `elapsed_ms` freezes at completion for completed rows so fast-finished requests stay compliant. Tests: `tests/core/test_request_latency_frozen.py`.

- **Summary:** RequestLatencyInfo.elapsed_ms is computed live as (time.time() - enqueued_at)*1000, and is_overdue compares that to the SLA. Completed requests are kept in _completed with their original enqueued_at, and get_sla_percentiles()/get_recent_metrics() read elapsed_ms/is_overdue on those retained records. As wall-clock time advances, a request that finished fast is later reported as overdue, dragging sla_compliance_pct toward 0 and making elapsed_p95 meaningless — the promised SLA metric is wrong over time.
- **Evidence (verbatim):**
```
def elapsed_ms(self):     return (time.time() - self.enqueued_at) * 1000 @property def is_overdue(self):     return self.elapsed_ms > self.sla_target_ms  # live clock, used on completed rows too
```
- **Impact:** SLA compliance dashboards and any quota/promote path relying on get_sla_percentiles over-report overdue and under-report compliance the longer the process runs; a healthy cluster shows chronic SLA violations.
- **Effort:** 1-2 hours
- **Reliability:** register()+complete() a request with sla 5000ms that took 100ms, wait 10s, call get_sla_percentiles(window 1): overdue_count=1, compliance 0% despite the request having met SLA. Deterministic given live-clock property.
- **Recommendation:** Freeze the completion time: in complete(), store an actual_duration_ms or set enqueued_at sentinel so elapsed_ms on a completed record returns (last_token_at or completion_ts - enqueued_at) instead of live clock. Compute overdue from the frozen duration. Add a test that completes a fast request and asserts is_overdue stays False after time passes.
- **Verdict (real):** mustFix=True. Confirmed by reading code. elapsed_ms (request_latency.py:29-30) is live time.time()-based; is_overdue:39-40 uses it; complete():71-77 retains original enqueued_at on records appended to _completed (no snapshot). get_sla_percentiles() reads r.elapsed_ms(183)/r.is_overdue(184) and get_recent_metrics() reads c.elapsed_ms(117)/c.is_overdue(119) over retained completed records. get_recent_metrics is a live external surface: wired to the API (server.py:1608, server_routes_api.py:385) and dashboard WebSocket (ws_handler.py:374-377). _completed is unbounded, so under low throughput stale records re-classify as overdue purely from wall-clock advance, corrupting the delivered SLA compliance/elapsed metrics. Fix: snapshot elapsed at completion or derive from stored last_token_at-enqueued_at.

---

### F-023 — [High] 🔒 VERIFIED · MUST-FIX telemetry.flush() deadlocks when auto-triggered from _add_event (non-reentrant lock inside lock)

`src/distllm/core/telemetry.py:184` · zone=`core-perf-obs` · category=`bug`

- **Summary:** TelemetryCollector._add_event holds self._lock (a plain threading.Lock) and calls self.flush(), which re-acquires the same non-reentrant lock. When the 50th event is recorded (BATCH_SIZE reached), the calling thread blocks permanently inside flush()'s `with self._lock:`. Additionally FLUSH_INTERVAL_S (300) is defined but no timer thread uses it, and no method ever POSTs to the ENDPOINT — so data is only ever buffered/written to a local jsonl, never uploaded.
- **Evidence (verbatim):**
```
def _add_event(self, event):     with self._lock:         self._events.append(event)         if len(self._events) >= self.BATCH_SIZE:             self.flush()   # flush() reacquires self._lock -> deadlock
```
- **Impact:** With telemetry enabled, the 50th recorded event hangs the requesting thread (a sync request path) permanently; even without the deadlock the collected anonymous data never reaches telemetry.distllm.ai, so the promised analytics pipeline ships nothing.
- **Effort:** 2-4 hours
- **Reliability:** Set DISTLLM_TELEMETRY=1 and call telemetry.record_request() 50 times; the thread recording the 50th request blocks forever inside flush() (non-reentrant Lock.acquire on same thread). Confirmed by reading lines 145-153 and 181-185.
- **Recommendation:** Call flush() outside the lock in _add_event (collect a reference to the batch first, or guard with an 'already flushing' flag), and make _lock an RLock if recursion is intended. Wire a background daemon thread that flushes every FLUSH_INTERVAL_S, and implement the actual upload to ENDPOINT (or remove the endpoint constant and 'for later upload' language).
- **Verdict (real):** mustFix=True. Read D:/distributed-llm/src/distllm/core/telemetry.py in full. The self-deadlock is genuine: TelemetryCollector._add_event acquires self._lock (a plain threading.Lock, line 75) then calls self.flush() at line 185, and flush() re-acquires the same non-reentrant lock at line 150. When telemetry is enabled and the 50th event (BATCH_SIZE) is recorded, the calling thread blocks forever inside the `with self._lock:` in flush(). This is load-bearing: record_request/record_feature/record_startup all funnel through _add_event, so an enabled, use telemetry path permanently hangs a worker thread — a Critical/High availability defect that must be fixed (e.g. use RLock, or move the flush() call outside the lock, or collect events-batch then flush outside the critical section). Secondary claims also verified by grep: FLUSH_INTERVAL_S (line 62) has no timer thread consuming it (no Thread/Timer anywhere), and no requests/urllib/httpx call ever POSTs to self._endpoint (line 71) — flush() only writes a local events_*.jsonl, so data is never uploaded to the telemetry endpoint. All three sub-claims are true as described.
  - *Correction:* All cited details confirmed verbatim. _add_event (lines 181-185) holds self._lock and calls self.flush(); flush() re-acquires the same non-reentrant threading.Lock at line 150, deadlocking at the 50th event. Grep confirms no Thread/Timer uses FLUSH_INTERVAL_S (line 62) and no requests/urllib/httpx POST to self._endpoint (line 71) — flush (145-169) only writes a local .jsonl.

---

### F-024 — [High] 🔒 VERIFIED · MUST-FIX UsageMeter time-window queries under-report after 100k records — silent billing/quota evasion

`src/distllm/dist/daas/usage_meter.py:197` · zone=`dist-exec` · category=`bug`

- **Summary:** `_max_records = 100_000` is a hard cap: once the in-memory `_records` list is full, new records are NOT appended to memory (line 197) — they go only to SQLite (line 210), and `_records` keeps the oldest 100k. `get_usage(tenant_id, since_timestamp)` (lines 224-263) reads ONLY `self._records`, never the DB. So after a tenant or cluster exceeds 100k requests, any since-time-window aggregation under-counts badly (walk backwards stops at the oldest window), and the DB is only synced into memory at construction (`_load_from_db`). This lets even an honest deployment silently under-report usage for billing/quotas at the moment it starts to matter.
- **Evidence (verbatim):**
```
if len(self._records) < self._max_records:\n    self._records.append(UsageRecord(...))
```
- **Impact:** Correctness of metered billing / DaaS charging and per-tenant quota enforcement silently degrade once request volume exceeds 100k; a tenant can appear under quota when it has far exceeded it.
- **Reliability:** Record 100_000+ requests, then query get_usage(t, since=now-60) — returns near-zero because the newest records were dropped from _records and only the oldest survived.
- **Recommendation:** Either drop the in-memory cap (bound it instead by tenant+client side), or make `get_usage(since>0)` query SQLite directly (`WHERE tenant_id=? AND timestamp>=?`) so it reflects all persisted records. Also periodically re-sync or stop discarding memory records. Add a test that inserts 100_001 records and asserts the windowed count is exact.
- **Verdict (real):** mustFix=True. Verified against src/distllm/dist/daas/usage_meter.py. Line 197 gates the in-memory append on len(self._records) < self._max_records (100k); beyond that, new records go only to SQLite (line 210 runs unconditionally). get_usage(224-263) reads ONLY self._records, never the DB (lines 240-252), so after the global record count exceeds 100k a recent time-window query under-counts or (via the break at 247-248) returns None, missing all records 100001+. Additionally _load_from_db (125-145, called only in __init__ line 98) populates only the lifetime _usage aggregates, never _records, so after ANY restart _records is empty and every time-window query returns None until 100k new records re-accumulate. The docstring (78-79) and CLAUDE.md advertise "arbitrary time-range queries are possible," so this silently breaks a documented billing/quota feature. Lifetime _usage totals are accurate, but time-window billing/quota queries under-report in two independent ways. Real High; the shared 100k global cap means high-volume multi-tenant deployments reach it in minutes, and the restart path affects even low volumes.

---

### F-025 — [High] 🔒 VERIFIED · MUST-FIX Federated LoRA merge accepts untrusted adapters + self-reported dataset_size → model poisoning and unfair weight dominance

`src/distllm/dist/federated_merge.py:193` · zone=`dist-exec` · category=`security`

- **Summary:** In `submit_node_adapter`, an arbitrary node supplies both `adapter_path` (untrusted file path read by the coordinator) and `dataset_size`. dataset_size is taken at face value and used as the FedAvg weight (lines 281-284), so a malicious/faulty node can set `dataset_size` arbitrarily large and dominate the merged adapter. `adapter_path` is neither checksummed, signed, nor validated as a real training artifact — the coordinator `torch.load`s whatever that path points to (weights_only=True limits RCE but any readable file that parses to tensors is ingested). This is the de-trust/Byzantine thread: there is no attestation that the submitted adapter is an honest LoRA update.
- **Evidence (verbatim):**
```
if dataset_size > 0:\n    state.dataset_size = dataset_size\n\nself._current_round.node_weights[node_id] = state.dataset_size
```
- **Impact:** In federated multi-node training, one compromised node can poison the global adapter or, by reporting a huge dataset size, capture the weighted average — corrupting the artifact every node will later load.
- **Reliability:** register_node('evil', dataset_size=0); submit_node_adapter('evil', good_path, loss=0.1, dataset_size=10**12) → node_weights['evil'] dwarfs all others → _federated_average returns the evil adapter as the global model.
- **Recommendation:** Require a signed digest (Ed25519, reusing byzantine._sign_bytes) from each node over its adapter bytes; cap/validate dataset_size against a server-side heuristic and clamp outlier weights (e.g., trim one-sided F-trimmed mean instead of plain FedAvg); reject paths outside the nodes' sandboxed upload dir; and add a best-possible-loss plausibility check before voting an adapter into the round
- **Verdict (real):** mustFix=True. Confirmed by reading the code. In submit_node_adapter (federated_merge.py:193-196) the caller-supplied dataset_size is stored verbatim on node state and then used directly as the FedAvg weight (_federated_average line 283: self._nodes[nid].dataset_size or 1), with weights normalized at line 327 — so a node reporting a huge dataset_size drives its normalized share to ~1.0 and dominates/hijacks the merged model. This is the default merge_strategy ("fedavg"). adapter_path (line 189) is similarly trusted: _merge_adapter_paths (lines 332, 353) torch.loads whatever the path points to with weights_only=True, which bounds RCE but does nothing to stop injecting arbitrary tensor values into the served model (model poisoning). No checksum, signature, clipping, or Byzantine-robust aggregation exists; the same-file SecureAggregator is not wired in. The vector is network-reachable via POST /v1/federated/rounds/submit (api/routes/federated.py:96-111), which forwards user-supplied adapter_path and dataset_size unchanged. Only mitigation is the admin role gate, which bounds external surface but does not defend against a compromised registered participant. This is a genuine, default-path, network-reachable High-severity federated-learning poisoning/dominance gap and warrants fixing (byzantine-robust aggregation, weight caps/normalization policies, adapter integrity checks).

---

### F-026 — [High] 🔒 VERIFIED · MUST-FIX Pipeline cross-node gRPC is plaintext by default

`src/distllm/dist/node_client.py:365` · zone=`dist-net` · category=`security`

- **Summary:** The main pipeline-parallel inference path sends hidden states and KV caches over unencrypted gRPC. create_node_client defaults use_tls=False (node_client.py:173) and forward_request/forward_request_async never pass use_tls, so grpc.insecure_channel() is used. PipelineOrchestrator.run_pipeline (orchestrator.py:208-211) and run_pipeline_microbatched (line 342) both route through forward_request/forward_request_async. This contradicts dist/transport.py:217 which defaults use_tls to True — the two factories disagree, and the actual production orchestrator uses the plaintext one.
- **Evidence (verbatim):**
```
def forward_request(...): client = create_node_client(host, port, timeout_s=timeout_s, cluster_key=cluster_key)
```
- **Impact:** LLM activations (hidden_states) and KV cache tensors — which encode prompt/token content — are exposed to any on-path observer over LAN/WiFi/Internet, defeating the project's stated TLS posture.
- **Recommendation:** Add use_tls parameter to forward_request/forward_request_async and create_node_client and thread TLS through PipelineOrchestrator. To avoid surprising silent plaintext, raise (or warn loudly) if use_tls=False and the channel carries cluster_key or tensor payloads, matching federation.py:742 which already raises when TLS is required.
- **Verdict (real):** mustFix=True. Verified by reading code. create_node_client (node_client.py:173) and create_async_node_client (line 243) both default use_tls=False, so forward_request (line 365) and forward_request_async (line 433) — which never pass use_tls — build grpc.insecure_channel() / grpc.aio.insecure_channel() (lines 225, 267). Both PipelineOrchestrator.run_pipeline (orchestrator.py:208-217 calls forward_request at 211) and run_pipeline_microbatched (orchestrator.py:342/358 calls forward_request_async) route through these plaintext clients. transport.py:217/242 (use_tls default True) contradicts the node_client defaults, confirming an inconsistency in intended-vs-actual TLS behavior. The defect is genuine and load-bearing: the production pipeline-parallel inference path sends hidden states and KV caches in plaintext over gRPC by default, and critically the forward_request/forward_request_async signatures expose NO use_tls/ca_cert parameter — so there is no runtime way to enable encryption on this path without code changes. Given the project's production-readiness mandate and that model intermediate activations/KV can carry sensitive user data, this counts as a real High that should be resolved (default TLS on the orchestrator path consistent with transport.py, or at minimum expose the knob). One minor note: the finding's line cite for forward_request (~365) is exact; all other cites are accurate.

---

### F-027 — [High] 🔒 VERIFIED · MUST-FIX Gossip HMAC key rotation replaces the shared configured key with a random node-local key

`src/distllm/dist/p2p/gossip.py:874` · zone=`dist-net` · category=`security`
✅ **FIXED + VERIFIED** — the shared-key overwrite is guarded (rotation short-circuits when `DISTLLM_GOSSIP_HMAC_KEY` is configured); 3 cross-node auth regression tests added. 387 gossip tests pass.

`src/distllm/dist/p2p/gossip.py:892` · zone=`dist-net` · category=`bug`

- **Summary:** In a shared-key deployment (DISTLLM_GOSSIP_HMAC_KEY), check_key_rotation() (called every gossip round in GossipReplicator.sync_once when enable_key_rotation, the default) silently overwrites the deployment-wide shared key with a random token_hex(32) that no other node possesses. During the 1h overlap the node still accepts the old shared key, but after _key_overlap_period expires, every peer's advertisement fails verify and the node's own outgoing messages (signed with the new random key) can't be verified by peers — gossip authentication permanently breaks after the first rotation at ~24h uptime.
- **Evidence (verbatim):**
```
old_key = self._hmac_key; ...; self._hmac_key = secrets.token_hex(32); ...; self._overlap_hmac_key = old_key
```
- **Impact:** After 24h uptime with default rotation on, gossip silently stops being authenticated — all KV-cache advertisements are dropped/ignored, breaking distributed cache sharing; also any peer that bypasses verification sees unauthenticated traffic.
- **Reliability:** Set DISTLLM_GOSSIP_HMAC_KEY=<shared>. Run GossipReplicator with default enable_key_rotation=True for > key_rotation_interval; observe check_key_rotation() set _hmac_key to a fresh random value and peer exchanges fail verification.
- **Recommendation:** Rotation must NOT regenerate the shared configured key. Only rotate per-peer derived keys (or re-derive from a shared secret KDF). If DISTLLM_GOSSIP_HMAC_KEY was set, leave _hmac_key unchanged; guard rotation behind a dedicated per-deployment rotated-key path that peers re-bootstrap.
- **Verdict (real):** mustFix=True. Verified by reading src/distllm/dist/p2p/gossip.py. check_key_rotation() (line 869) unconditionally overwrites the deployment-wide shared _hmac_key with a node-local secrets.token_hex(32) (lines 890-892) once key_rotation_interval (default 86400 = 24h) elapses, saving the old shared key only as _overlap_hmac_key. It is invoked every sync_once round because _enable_key_rotation defaults to True (line 1115, call at 1257-1260). sign_message/verify_message (lines 303-321) use self._hmac_key as the actual wire auth, so in a shared-key deployment the node's outgoing ads are signed with a random key peers don't hold, and peers' ads can't be verified once _cleanup_overlap_key() drops the old shared key after key_overlap_period (default 1h). Since no owned key ever carries the deployment secret onward, gossip auth permanently breaks after the first rotation at ~24h uptime — a genuine production availability/security regression. The DH per-peer re-key (line 884) is legitimate, but the shared-key rotation (line 892) has no guard against rotating a configured shared secret and is catastrophic for multi-node shared-key deployments.

---

### F-028 — [High] 🔒 VERIFIED · MUST-FIX Kademlia DHT STORE is unauthenticated and token-unverified

`src/distllm/dist/p2p/kademlia_dht.py:834` · zone=`dist-net` · category=`security`

- **Summary:** Any node that can reach the UDP DHT port can write arbitrary key->value bytes onto any other node's local store. The caller attaches a 'token' (line 651, "Standard Kademlia token for store-authorisation") but the receiver (_handle_store) never validates it — the token is simply self.local_node.hex_id, which is public and by itself unsuitable. Neither address-based nor token-based STORE authorization is enforced, so stored records and any values served by FIND_VALUE can be poisoned by an unauthenticated peer.
- **Evidence (verbatim):**
```
async def _handle_store(self, msg, addr): ... if not key or not value_hex: return ...; self._store[key] = (bytes.fromhex(value_hex), time.time() + EXPIRE_TIME)
```
- **Impact:** DHT value poisoning and cache poisoning: a malicious/rogue node can redirect FIND_VALUE lookups to attacker-controlled nodes or inject bogus KV-cache records used for peer discovery over the WAN.
- **Reliability:** Send a crafted STORE UDP datagram with arbitrary key/value and optional fake sender; the receiver persists it with no credential check and later serves it via FIND_VALUE.
- **Recommendation:** Implement a real store-authorisation token: derive an HMAC or a time-bound capability from a shared secret (or require the sender's node_id to have PASSED a reachability ping) and reject STORE without a valid token. Also cap per-sender store rate to bound poisoning.
- **Verdict (real):** mustFix=True. Confirmed by reading src/distllm/dist/p2p/kademlia_dht.py. _handle_store (line 834) reads params key/value, checks only non-empty, and writes directly to self._store[key] with NO token validation, no address/allowlist check, and no capacity/rate limit. The token attached by store() (line 651) is self.local_node.hex_id — the sender's own PUBLIC node id, broadcast in every RPC sender field — and is never even read by _handle_store, making the intended "store-authorisation" entirely non-functional. Values written are served to any peer by _handle_find_value (lines 904-909), so an attacker can both flood the store (DoS, entries held 86400s) and overwrite/poison legitimate key->value entries that other nodes then retrieve. Any peer able to reach the UDP DHT port can perform arbitrary unauthenticated writes; for a WAN/federation peer-discovery component this is a genuine High security defect warranting a fix (receiver-issued token validation and/or address allowlist plus capacity limits) before release. One minor detail: the `self._lock`/write-lock framing in the task description is inaccurate — the DHT's self._store is a plain dict with no lock, and _handle_store writes to it directly; but this does not change the substance of the unauthenticated-write finding.

---

### F-029 — [High] 🔒 VERIFIED · MUST-FIX QUIC client disables TLS peer verification (CERT_NONE)

`src/distllm/dist/p2p/quic_transport.py:389` · zone=`dist-net` · category=`security`

- **Summary:** All outgoing QUIC connections set verify_mode=CERT_NONE, so the client accepts any certificate and never authenticates the remote peer. Over the Internet this permits a straight MITM: an attacker impersonating a node supplies a self-signed cert and the client proceeds, handing over the KV-cache/gossip stream. The 0-RTT session-ticket cache (lines 436-438) then silently replays to the attacker.
- **Evidence (verbatim):**
```
if is_client:     # P2P: self-signed certs are the norm     config.verify_mode = ssl.CERT_NONE
```
- **Impact:** Full MITM of QUIC peer data (gossip metadata + KV cache content) despite the transport nominally using TLS.
- **Reliability:** On the client, point connect() at a server with an arbitrary self-signed cert; the handshake completes because verify_mode=CERT_NONE ignores the chain.
- **Recommendation:** Reuse the configured shared secret / cluster key to pin peers: either verify a peer fingerprint against the negotiated gossip key, or require a CA-signed cert with cert_file/key_file loaded and verify_mode=ssl.CERT_REQUIRED with an in-cluster CA bundle.
- **Verdict (real):** mustFix=True. Confirmed by reading the file. src/distllm/dist/p2p/quic_transport.py lines 387-389 set config.verify_mode = ssl.CERT_NONE for every outgoing (is_client=True) QUIC connection, exactly as claimed; line 433 (connect) and transport.py get_optimal_transport() confirm this is the live primary P2P transport when aioquic is installed. There is no cert pinning, CERT_REQUIRED, shared-CA, or app-layer shared-secret handshake on the QUIC data path to restore peer authentication — TLS verification is the sole trust gate and it is disabled. system is not limited to a trusted private cluster, but must fix because the codebase explicitly supports cross-cluster federation, WAN pipelines, and cross-cluster KV-cache transfer (CLAUDE.md dist scope), i.e. Internet links, where an MITM impersonating a node can present any cert and read/hand the KV-cache/gossip stream in plaintext (relative to the attacker's TLS termination). The cited 0-RTT session-ticket cache at lines 436-438 is keyed only by host:port and replays to the attacker, compounding the issue. This is a genuine, load-bearing Critical/High: client disables server authentication entirely; fix requires cert pinning or a secure-by-default verification mode.

---

### F-030 — [High] 🔒 VERIFIED · MUST-FIX PartitionValidator.what_if_slowdown is a no-op — the slowdown never affects the simulated throughput

`src/distllm/dist/partition/validator.py:258` · zone=`dist-partition` · category=`bug`

- **Summary:** what_if_slowdown inflates pt.estimated_time_ms on modified points, but _simulate_pipeline never reads estimated_time_ms — it recomputes every stage from _cost_model.evaluate. So the modified solution === the original for the simulation, throughput_change_pct is always ~0, and the 'bottleneck shifts/remains' message is meaningless. The existing test only instantiates the WhatIfScenario dataclass directly, so it never exercises the real path.
- **Evidence (verbatim):**
```
new_pt.estimated_time_ms *= (1 + slowdown_pct / 100) ... # then: cost = self._cost_model.evaluate(pt.node_id, pt.start_layer, pt.end_layer, batch_size, seq_len)
```
- **Impact:** Adaptive re-partition incentives and validation 'what-if' reports are fabricated; users cannot predict the effect of a straggler slowdown, so migration decisions are based on false numbers.
- **Effort:** 2-4 hours
- **Reliability:** Trace: what_if_slowdown (line 249-259) edits modified_points.estimated_time_ms; _simulate_pipeline (lines 300-304) recomputes cost from the cost model and ignores pt.estimated_time_ms, so compute/throughput are identical for base and modified.
- **Recommendation:** Have _simulate_pipeline use pt.estimated_time_ms as the stage compute time (or apply the multiplier to the cost-model result) rather than re-evaluating the analytic model; add a test that asserts throughput drops when estimated_time_ms is inflated.
- **Verdict (real):** mustFix=True. PROVEN by reading src/distllm/dist/partition/validator.py. what_if_slowdown (L226-286) inflates new_pt.estimated_time_ms only for the target node (L258) and builds a modified PartitionSolution, but _simulate_pipeline (L288-383) recomputes every stage solely from self._cost_model.evaluate(node_id, start_layer, end_layer, batch_size, seq_len) — L300-304 — and derives throughput (L342), bottleneck (L331/354) and memory from that cost. estimated_time_ms is never read in the simulate path, so the modified solution is indistinguishable from the original and both simulations are statistically identical (only unseeded Monte Carlo jitter noise separates them). Hence throughput_change_pct is ~0 and new_bottleneck always equals the original, so the "Bottleneck shifts from X to X" / "remains the bottleneck" message (L273-277) is meaningless. This fires on the primary validate() path via _generate_what_ifs (L216 -> L399), and the only test (tests/test_validator.py:23-26) instantiates the WhatIfScenario dataclass directly, so the real path is untested — exactly as claimed. The entire feature is 100% non-functional and presents fabricated analysis in the validation report; the fix is trivial (apply the inflation to the cost the simulator reads, or use estimated_time_ms as the compute time). Minor correction: since jitter is unseeded, throughput_change_pct is statistical noise around 0, not literally always 0.0 — substance unchanged. Real, load-bearing High.
  - *Correction:* throughput_change_pct is noise around 0 due to unseeded random.gauss jitter, not always exactly 0.0; substance of the no-op claim unchanged

---

### F-031 — [High] 🔒 VERIFIED · MUST-FIX AdaptiveSerializer ZSTD path silently corrupts large (FP8) tensors — scale lost, double compressed

`src/distllm/dist/pipeline/compression_negotiation.py:604` · zone=`dist-net` · category=`bug`

- **Summary:** SerializationController.send_tensor(method=ZSTD) for a tensor > 100MB: AdaptiveSerializer.choose_format returns FP8_ZSTD, serialize() yields an fp8-quantized, scale-packed, zstd-compressed payload. The controller then zstd-compresses that payload AGAIN and retags it ZSTD. On the receiver, recv_tensor hits the ZSTD branch -> AdaptiveSerializer.deserialize(ZSTD) -> zstd_decompress once -> _deserialize_raw, which returns the raw quantized bytes as a torch.uint8/int8 tensor with NO scale and NO dequantization (the scale header bytes become leading tensor data). Large KV tensors are therefore corrupted in the ZSTD send path.
- **Evidence (verbatim):**
```
data = self._adaptive.serialize(tensor.cpu() ...); fmt_tag, data = data; if fmt_tag != SerializationFormat.ZSTD: data = self._adaptive._zstd_compress(data); fmt_tag = SerializationFormat.ZSTD
```
- **Impact:** KV-cache/hidden-state corruption on large (>100MB) tensors whenever ZSTD compression is negotiated — wrong inference output with no error raised.
- **Reliability:** send_tensor(peer, big_tensor>100MB, method=ZSTD) then recv_tensor(...); the returned tensor is the raw quantized bytes (plus scale header) as uint8, not the original floating tensor.
- **Recommendation:** Disallow mixed tagging: if serialize() chose FP8_ZSTD (or RAW), do not re-compress+retag as ZSTD. Route format from the serializer as-is, and make deserialize for ZSTD detect the embedded scale header and dequantize. Add a round-trip equality test for >large_threshold tensors on both ZSTD and LZ4 methods.
- **Verdict (real):** mustFix=True. Verified by reading src/distllm/dist/pipeline/compression_negotiation.py. For any tensor >100MB (_LARGE_TENSOR_BYTES=100_000_000), choose_format returns FP8_ZSTD. In send_tensor's ZSTD branch (lines ~596-604), AdaptiveSerializer.serialize() produces an fp8-quantized, scale-packed, zstd-compressed payload tagged FP8_ZSTD, but because fmt_tag != ZSTD the code then zstd-compresses that payload AGAIN and retags it ZSTD (line 601-604). Receiver recv_tensor routes fmt_tag==ZSTD to deserialize(ZSTD) which decompresses only once and calls _deserialize_raw — never _unpack_scale, no inner decompress, no tensor_dequantize. The result is a torch.uint8 tensor whose bytes are the scale-prefixed, still-compressed fp8 payload (scale header becomes leading tensor data), so large tensors are silently corrupted in the ZSTD send path. Every claim detail (scale lost, double compressed, retagged, KB tensors corrupted) is confirmed. This is a genuine data-integrity corruption in the distributed tensor transport reachable via CompressionMethod.ZSTD on large tensors; release-blocking.

---

### F-032 — [High] 🔒 VERIFIED · MUST-FIX ZeroCopyTransferEngine.recv fabricates zeros (NCCL) / always fails (CUDA_IPC)

`src/distllm/dist/zero_copy.py:222` · zone=`dist-net` · category=`bug`

- **Summary:** The receive side of the zero-copy engine is non-functional. NCCL 'recv' returns torch.zeros() and marks success=True (silent data corruption — caller believes it received real KV data); CUDA_IPC recv calls import_tensor(key, b"", ...) with an empty handle so pickle.loads(b"") raises and returns None. Only RDMA honestly errors (raises NotImplementedError). Any consumer of ZeroCopyTransferEngine.recv gets either a zeros tensor or None while being told success. Root cause: recv has no path to actually obtain remote data (IPC handles must be sent out-of-band; NCCL needs a real group/rank).
- **Evidence (verbatim):**
```
elif backend == TransferBackend.NCCL and self._nccl_transport is not None: result = torch.zeros(shape, dtype=dtype, device="cuda" ...); success = True
```
- **Impact:** Silent corruption of KV/activations whenever the zero-copy NCCL path is used, or hard None-failures on CUDA_IPC — both perceived as 'working' zero-copy transfers.
- **Reliability:** zero_copy.send/recv(..., 'gpu', peer_is_local=True) round-trip: recv returns all-zeros tensor with stats.success=True and never contacts the peer.
- **Recommendation:** Make recv either perform a real transfer or return success=False with a clear error. For a proper implementation the sender must transmit the IPC handle (export_tensor bytes) via the control channel before recv, and NCCL recv must map ranks. At minimum revert the zeros-fabrication to raise so callers fall back instead of corrupting data.
- **Verdict (real):** mustFix=True. Verified line-by-line in src/distllm/dist/zero_copy.py: (1) NCCL recv (lines 221-223) returns torch.zeros(shape,dtype,device) and sets success=True — silent data corruption, since the NCCL send side (line 181) genuinely dispatches real tensor data via send_tensor_list with no matching receptacle. (2) CUDA_IPC recv (line 215) calls import_tensor(tag or peer, b"", shape, dtype) with an empty hard-coded handle; pickle.loads(b"") raises EOFError (caught, line 74), returning None. (3) RDMA recv (line 218->recv_rdma line 137) raises NotImplementedError. One correction: the claim that CUDA_IPC is "told success" is wrong — for CUDA_IPC success=handle is not None=False, so only the NCCL branch falsely reports success; CUDA_IPC and RDMA fail honestly (return None/success=False). Load-bearing: it's a genuine silent-corruption defect in a documented public transfer API; recv is currently unexercised in production (request_pipeline.py only calls zc_engine.send()), so it's latent, but the send path IS wired into a live code path and recv is the intended symmetric receive. Recommend fixing by routing NCCL recv through the real NcclTransport.recv or raising NotImplementedError like RDMA to fail loudly instead of fabricating zeros.
  - *Correction:* CUDA_IPC recv does NOT report success=True: it returns None with success=False (import_tensor returns None). Only the NCCL branch fabricates zeros with success=True. Both CUDA_IPC and RDMA fail honestly.

---

### F-033 — [High] 🔒 VERIFIED · MUST-FIX ModelHub cache-layout mismatch: snapshot_download writes under models--org--name but resolve/is_available check {cache_dir}/org/revision

`src/distllm/models/model_hub.py:384` · zone=`core-training` · category=`bug`

- **Summary:** download()/_download_full_model() call snapshot_download(cache_dir=self.cache_dir), which stores under the HuggingFace snapshot layout self.cache_dir/models--org--name/snapshots/<hash>. But is_available()/resolve()/remove() check self.cache_dir/model_name/revision (org/name/main), which never exists. Every cached model is treated as uncached: repeated full re-downloads, and offline_mode raising ModelNotCachedError for models already on disk.
- **Evidence (verbatim):**
```
downloaded_path = snapshot_download(..., cache_dir=str(self.cache_dir), ...) ... is_available: model_path = self.cache_dir / model_name / revision return model_path.exists() and (model_path / ".manifest").exists()
```
- **Impact:** O(pooled) redundant downloads per node join; offline/inference startup broken for a machine that already has the model; unbounded disk growth from repeated snapshots.
- **Effort:** 3-6 hours
- **Reliability:** hub.download('org/Model'); hub.is_available('org/Model') is False; resolve(offline_mode=True) raises ModelNotCachedError.
- **Recommendation:** Either pass local_dir=str(model_cache_path) so files land at self.cache_dir/org/name/revision, or point is_available/resolve/list_cached at the snapshot layout. Add a test that download() then is_available() returns True.
- **Verdict (real):** mustFix=True. Verified by reading src/distllm/models/model_hub.py. download()/line 380-387 and _download_full_model()/line 310-316 call snapshot_download(cache_dir=str(self.cache_dir)), which uses HuggingFace's snapshot layout {cache_dir}/models--{org}--{name}/snapshots/<hash>; the .manifest is written into that snapshot dir (line 320 -> _write_manifest line 582). But download()/line 366-369, is_available()/line 434-435, and resolve()/line 464-470 all check self.cache_dir/{org}/{name}/{revision} ({org}/{name}/{main}) which never exists. Consequences as claimed: every full-model download re-runs snapshot_download (cache never hits), and offline_mode raises ModelNotCachedError for models already on disk. This defeats the entire caching purpose of the class and breaks the documented offline feature — a genuine Core/High bug that must be fixed before release.
  - *Correction:* Finding confirmed and slightly understated. Also affected: list_cached() (looks for {cache_dir}/{org}/{name}/{rev}/.manifest but manifest lives at models--{org}--{name}/snapshots/<hash>/.manifest, so it never lists downloaded models) and remove() (same wrong path, can't delete cached models).

---

### F-034 — [High] 🔒 VERIFIED · MUST-FIX JWT authentication bypass via algorithm confusion in HS256 fallback validator

`src/distllm/plugins/auth_plugin.py:168` · zone=`ops-utils` · category=`security`

- **Summary:** When PyJWT is not installed (it is an optional dep, imported in try/except) the fallback _validate_jwt_hs256 verifies a token's signature by HMAC-SHA256 with the configured DISTLLM_AUTH_SECRET string, ignoring the token's alg header and never requiring a shared-secret. If an operator configures a PEM public key (the RS256 branch, honored only by the PyJWT path), the fallback treats that public key as the HS256 shared secret, so anyone can forge an HS256 'admin' token signed with the public key. The fallback also skips iss/aud checks when the payload omits those claims.
- **Evidence (verbatim):**
```
expected = hmac.new(secret.encode(), message, hashlib.sha256).digest()  (line 168); _is_pem branch is only applied in the _HAS_PYJWT path (lines 215-216)
```
- **Impact:** Complete authentication bypass and privilege escalation to admin for any distributed deployment that runs without PyJWT and uses an asymmetric key (or a short shared secret). Tokens are forged against the public key.
- **Effort:** 2-4 hours
- **Reliability:** Set DISTLLM_AUTH_SECRET to a PEM public key and run without PyJWT; craft a JWT signed HS256 with that key as the HMAC secret bearing role='admin'; validate_jwt accepts it.
- **Recommendation:** In _validate_jwt_hs256, (a) refuse PEM-looking secrets (only allow a strong >=32-char shared secret); (b) parse the header and reject unless header['alg']=='HS256'; (c) make iss/aud checks strict (reject when configured but absent in payload). Better: add PyJWT as a hard dependency so the fallback is never used in production.
- **Verdict (real):** mustFix=True. Verified by reading src/distllm/plugins/auth_plugin.py. (_validate_jwt_hs256 lines 155-191) computes hmac with secret.encode() at line 168 and never reads the alg header, so any token whose HMAC matches is accepted regardless of its stated algorithm. The PEM/RS256 distinction (lines 215-216) lives entirely inside the `if _HAS_PYJWT:` branch, and the fallback (lines 240-251) passes the configured DISTLLM_AUTH_SECRET verbatim as the HMAC key. Therefore, when an operator configures a PEM public key (intended for RS256 verification, an explicitly supported config per the `_is_pem` branch) and PyJWT is absent (optional dep; fallback is the runtime default in that case), the public key becomes the HS256 shared secret — the textbook JWT algorithm-confusion 'public key as HMAC secret' attack. Any party knowing the public key can forge an HS256 token with `role: admin`; `_validate_jwt_from_context` (lines 404-406) resolves jwt_role='admin' and on_request overrides api_key_role (line 355), bypassing the enforced access control for full admin access. This satisfies severity criterion (a): bypassing an enforced access control in a documented/supported (non-default but common) config. The fallback also skips aud/iss checks whenever the payload omits those claims (lines 246-251), letting an attacker drop them to evade audience/issuer enforcement — weaker than the PyJWT path which enforces them. Minor wording correction: the fallback does require the HMAC to match the configured secret (so it is not literally 'never requiring a shared secret'), but the security consequence (public key reused as forgeable HMAC secret) stands. Fix: in the fallback, require the alg header to be HS256 and reject a PEM/public-key secret (fallback only handles HS256), and enforce aud/iss as mandatory when configured.

---

### F-035 — [High] 🔒 VERIFIED · MUST-FIX Exact-match cache key is not tenant/user scoped - cross-tenant response leak

`src/distllm/plugins/cache_plugin.py:83` · zone=`ops-utils` · category=`security`

- **Summary:** _build_cache_key hashes only (prompt|model|temp|top_p) and is used by the primary exact-match path in on_request/on_response, with no tenant/user component. Only the optional semantic path is scoped via _request_scope (tenant_id/user_id). Two tenants sending identical prompts+params share one cache entry, so tenant B is served tenant A's cached response via {'_cached_response'}. The module docstring explicitly claims the opposite.
- **Evidence (verbatim):**
```
raw = f"{prompt}|model={model}|temp={temperature}|top_p={top_p}"  (line 83) with no scope; _request_scope (lines 87-93) is used only for the semantic branch
```
- **Impact:** Cross-tenant disclosure of private cached LLM responses (personalized answers, PII, per-user results) in any multi-tenant deployment with caching enabled.
- **Effort:** 1-2 hours
- **Reliability:** Tenant A sends 'what is my SSN' (cached); tenant B sends identical prompt/model/temp/top_p; on_request returns tenant A's cached response.
- **Recommendation:** Include _request_scope(context) in the exact-match key, e.g. raw = f"{scope}|{prompt}|model={model}|temp={temperature}|top_p={top_p}", in both on_request and on_response key construction.
- **Verdict (real):** mustFix=True. Confirmed by reading src/distllm/plugins/cache_plugin.py. _build_cache_key (line 83) hashes only prompt|model|temp|top_p with no tenant/user component, and this exact-match key drives the primary path in both on_request (line 359) and on_response (line 408). Only the semantic path is scoped via _request_scope (lines 373, 420), which returns tenant_id or user_id. The _request_scope docstring (lines 88-92) explicitly claims 'a cached response for one tenant can never be served to another' — a guarantee the default exact-match path does not provide. So two tenants sending identical prompt+params collide to the same cache entry, and a hit serves {'_cached_response'} from whichever tenant populated it first. This is a genuine cross-tenant response-leak flaw in the cache key design, and the fix (add a tenant/user scope component to _build_cache_key, mirroring _request_scope) is trivial. Caveat for severity calibration: the plugin is disabled by default (DISTLLM_PLUGIN_CACHE_ENABLED default 0), and in the current api/server.py PluginHookMiddleware only reads '_reject' from the dispatch result, so _cached_response is not short-circuited in that wiring today. But the plugin's documented contract and its own test (test_cache_hit_returns_cached_response) assert it serves cached responses, so the key-design defect itself is real and load-bearing for any multi-tenant deployment that enables caching — a security-adjacent High that should be fixed before release.

---

### F-036 — [High] 🔒 VERIFIED · MUST-FIX Sync vs async chat streaming yield different item types (dict vs str) in both Python SDKs

`src/distllm/sdk/client.py` + `sdk/src/distllm_sdk/client.py` · category=`bug`
✅ **FIXED** — sync `chat_completions_stream` in BOTH Python SDKs now yield raw content strings matching the async version (the parser emitted full SSE event dicts before). Tests: `tests/sdk/test_stream_parity_f036.py` (2 pass; full SDK suite 126 pass).

`src/distllm/sdk/client.py:878` · zone=`sdk-arch` · category=`bug`

- **Summary:** Async chat_completions_stream extracts delta.content and yields str; sync chat_completions_stream does 'yield from parse_sse_stream_sync(response)' and yields raw SSE dicts. A caller switching clients gets a different type from the same method, breaking downstream unpacking. Reproduced identically in the standalone distllm_sdk (line 656), confirming drift.
- **Evidence (verbatim):**
```
with self._client.stream(...) as response: ...; yield from parse_sse_stream_sync(response); async path does 'content = delta.get("content"); if content: yield content'
```
- **Impact:** Flipping sync<->async yields dicts instead of strings from the same-named method, causing runtime errors with no signposting; breaks the sync/async-pair promise.
- **Effort:** 1-2 hours
- **Reliability:** Sync stream yields SSE dicts (streaming.py 38-55 returns json.loads(event)); async yields content strings (client.py 571-576).
- **Recommendation:** Make sync extract delta.content identically to async via a shared helper, or return a typed event in both; add a parity test; mirror fix into sdk/src/distllm_sdk/client.py:656.
- **Verdict (real):** mustFix=True. Line-by-line read proves the drift. Sync chat_completions_stream (src/distllm/sdk/client.py:878) does `yield from parse_sse_stream_sync(response)`, and parse_sse_stream_sync (streaming.py:38-55) yields raw SSE dicts (`yield json.loads(data)`). Async chat_completions_stream (client.py:569-576) extracts `delta.get("content")` and yields only the string content. Different item types from the same-named method; the sync variant additionally violates its own `Iterator[str]` annotation by yielding dicts. Standalone sdk (sdk/src/distllm_sdk/client.py:656) reproduces the same sync behavior. This is a real API-contract consistency defect that would break callers switching clients, so it is load-bearing enough for a High fix before release. Exact line cited (878) is correct.
  - *Correction:* Confirmed. Sync path also violates its own return annotation Iterator[str] (streaming.py:38-55 yields dicts via json.loads); archived alongside the async/str divergence. Both the main package (client.py:878) and standalone sdk (sdk/src/distllm_sdk/client.py:656) reproduce identically.

---

### F-037 — [High] 🔒 VERIFIED · MUST-FIX E2E SessionKeys ratchet diverges under asymmetric traffic, causing intermittent decrypt failures

> **✅ FIXED 2026-08-21** — `src/distllm/security/e2e.py` redesigned: dropped the divergent local-counter ratchet entirely. Each message now derives its box key from the shared ECDH key + a FRESH random per-message salt that travels with the ciphertext (`encrypt()` returns it; `decrypt()` derives from the transmitted salt). Key material is never mutated by traffic, so both sides stay in sync under any asymmetric pattern; per-message salt+nonce gives forward secrecy without shared counters. Also fixed `derive_box_key()` to use PyNaCl's correct `crypto_generichash_blake2b_salt_personal(data=, digest_size=, key=, salt=, person=)` keyword signature (was positional-mismatched → NameError/TypeError) with HKDF-SHA256 stdlib fallback. Regression tests: `tests/security/test_e2e_ratchet_f037.py` (asymmetric 35/2 interleaved traffic + ratchet-boundary decryptability) — both pass; full `tests/security/` sweep shows no e2e regressions.

`src/distllm/security/e2e.py:172` · zone=`core-priv-sec` · category=`bug`

- **Summary:** `SessionKeys` ratchets its shared key forward on the Nth local encrypt/decrypt (`_seq % RATCHET_INTERVAL`). Both peers maintain independent `_seq` counters, and the encrypt-side post-ratchet key is derived with the next transmitted salt while the decrypt side is still pre-ratchet. Whenever the number of messages sent and received within a direction differ (typical for token-streaming: many small tensors one way, few responses the other), the two sides' key schedules diverge and `crypto_secretbox_open` fails mid-stream, breaking the connection.
- **Evidence (verbatim):**
```
self._seq += 1 if self._seq % self.RATCHET_INTERVAL == 0:     self.ratchet()  # encrypt() and decrypt() both advance local _seq and ratchet independently
```
- **Impact:** Broken E2E transport under realistic asymmetric tensor traffic; session blackouts despite a valid shared key.
- **Effort:** 1-2 days
- **Reliability:** Peer A encrypts 10 small tensors (ratchets to key1/salt1) while peer B has decrypted fewer than 10; B derives its key with pre-ratchet shared_key + salt1 -> differs from A's post-ratchet key -> `crypto_secretbox_open` CryptoError on message 11.
- **Recommendation:** Make the ratchet deterministic per-message for both parties: derive the box key from the message's transmitted salt AND a per-message counter carried in the packet (or use a single shared sequence number derived from the salt chain), so both sides compute the same key for a given message. Alternatively, drop the autonomous counter ratchet and ratchet based on a counter that is transmitted with each ciphertext so the recipient applies the identical forward key.
- **Verdict (real):** mustFix=True. Verified by reading src/distllm/security/e2e.py. Both encrypt() (lines 149-176) and decrypt() (lines 178-215) share a single local self._seq and independently call ratchet() when _seq % RATCHET_INTERVAL==0 (lines 172-174, 211-213). ratchet() (lines 104-116) advances BOTH self._shared_key and self._salt. Decrypt keys off the transmitted salt but ALSO self._shared_key, which is NOT transmitted. When asymmetric traffic/interleaving makes the two nodes' ratchet boundaries land on different message indices, one side encrypts a message with pre-ratchet key S while the peer decrypts it with post-ratchet key R(S): derive_box_key(key=S,salt) != derive_box_key(key=R(S),salt), so crypto_secretbox_open fails mid-stream. Concrete trace with Alice sending 12, Bob replying 2 with early responses shows E9 encrypted with S but decrypted with R(S). The failure is intermittent/recurring, exactly as claimed. The prior 'SECURITY FIX' comment (lines 165-168) only fixed returning the post-ratchet salt; the shared-key ratchet divergence is a separate, unfixed bug. This is a core-priv-sec availability bug that breaks the E2E tensor channel under normal asymmetric token-streaming traffic — High, must fix before release (e.g., per-direction schedules or an agreed monotonic message index, not a combined local op counter).

---

### F-038 — [High] 🔒 VERIFIED · MUST-FIX `distllm system doctor` never runs — argparse parses Typer subcommand tokens as stray positionals

`src\distllm\cli\doctor.py:680` · zone=`cli` · category=`bug`

- **Summary:** The `system_doctor` Typer command forwards to doctor.main(), which calls `parser.parse_args()` on the raw `sys.argv`. Invoked as `distllm system doctor`, sys.argv is `['distllm','system','doctor']`; the parser (only options, no positionals) rejects `system`/`doctor` as unrecognized, argparse prints usage and exits 2, so no diagnostics run. Even after a naive fix (e.g. parsing a fixed empty argv), none of doctor's flags are surfaced on the Typer command, so the useful modes remain unavailable.
- **Evidence (verbatim):**
```
args = parser.parse_args()     doctor = Doctor(args)     total_errors = doctor.run()     sys.exit(1 if total_errors > 0 else 0)
```
- **Impact:** The primary single-command system diagnostic is completely non-functional from the advertised CLI surface, printing an argparse usage error and exiting 2; a user cannot diagnose GPU/network/config issues at all, and doctor's flags (--gpu/--json/--terse/--network/--model/--nodes/--verbose) are unreachable because main.py's system_doctor exposes none of them as Typer options.
- **Reliability:** Repro: `distllm system doctor` (or in-process `distllm.cli.main:app(['system','doctor'])`). doctor.main() runs `parser.parse_args()` against the raw `sys.argv`, which contains the non-option tokens 'system' and 'doctor'. The argparse parser defines only options with no positionals, so it raises SystemExit(2) with 'error: unrecognized arguments: system doctor' before Doctor.run() is ever called.
- **Recommendation:** Add an explicit argv parameter so parse is decoupled from process argv: `def main(argv=None): ... args = parser.parse_args(argv)`, and have main.py's system_doctor call it with a clean argv. Better: register each doctor flag as a Typer option on system_doctor and forward them, keeping the argparse main only for the `python -m distllm.cli.doctor` path. Add an integration test invoking the Typer app with argv ['system','doctor'] asserting diagnostic output (not argparse 'unrecognized arguments') is emitted.
- **Verdict (real):** mustFix=True. Verified by reading the code. `distllm system doctor` (Typer, main.py:1431) calls doctor.py `main()`, which calls `parser.parse_args()` (doctor.py:680) with no explicit argv, so it reads raw `sys.argv[1:]` = `['system','doctor']`. Typer/click do not mutate sys.argv. The argparse parser (doctor.py:672-679) defines only options and no positionals, so argparse rejects `system`/`doctor` as unrecognized, prints usage, and exits(2) before the `Doctor` is constructed — diagnostics never run. The second claim is also confirmed: `system_doctor()` takes no params (main.py:1432), so none of the doctor's flags are exposed on the Typer command; a naive argv fix would still leave all useful modes unreachable. This is a reproducible functional bug in a documented user-facing command that can never execute, hence a real High requiring a fix.

---

### F-039 — [High] 🔒 VERIFIED · MUST-FIX Error handling is self-inconsistent: most failing commands exit 0 (masked failures), and the cli_error_handler decorator is dead code

`src\distllm\cli\main.py:1385` · zone=`cli` · category=`bug`

- **Summary:** Most commands catch errors, print a red line, and return normally, so Typer exits 0 on failure. Confirmed in: system_slo_report (line 1385 except->print->return), federate_status (816-817), daas_status (1506-1507), draft_fleet_status (1609-1610), draft_migration_status (1637-1638), cluster._cluster_scale/_drain/_rebalance (ConnectError/HTTPStatusError handlers return), run.py path. The repo even ships a purpose-built `cli_error_handler` decorator raising SystemExit(1) on error, but it is applied to zero commands (grep finds only its own docstring), so the intended standardization was never wired up.
- **Evidence (verbatim):**
```
except Exception as e:     console = Console()     console.print(f"[red]Failed to fetch SLO data: {e}[/red]")
```
- **Impact:** CI pipelines, cron jobs, and shell scripts that gate on exit code silently treat failed connections/configuration as successful; failures are masked and invisible to tooling, undermining the CLI as an automation surface.
- **Reliability:** Repro for each: run `distllm system slo-report --port 1`, `distllm system observe --metrics-only` without deps, `distllm daas status --port 1`, `distllm draft fleet-status --port 1`, `distllm cluster drain <id> --port 1`, `distllm config quota report` with no server. Each except block prints then the function returns None, so Typer exits 0 ($? = 0).
- **Recommendation:** Adopt one exit-code policy for every command: on error, print the Rich message THEN `raise typer.Exit(1)` (or `sys.exit(1)`). Apply the already-shipped `cli_error_handler` decorator (src/distllm/cli/error_handler.py) to all commands so unexpected exceptions uniformly exit 1 instead of being swallowed. Add a subprocess harness in tests/cli asserting non-zero exit (and a message on stderr/stdout) for each group when the target host is unreachable.
- **Verdict (real):** mustFix=True. Both parts verified by reading the code. (1) Confirmed masked failures: system_slo_report (main.py:1374-1387) catches Exception, prints the red line, and returns None -> Typer exits 0 on a failed SLO fetch; _cluster_scale/_cluster_drain/_cluster_balance (cluster.py:61-64, 81-84, 97-100, 114-116) catch ConnectError/HTTPStatusError, print a red line, and fall through returning None -> exit 0 on network/HTTP failure. (2) Confirmed cli_error_handler is dead code: repo-wide grep finds it only inside its own file error_handler.py (module docstring + definition + self-docstring); zero commands decorate with it. The decorator defaults exit_on_error=True and raises SystemExit(1), i.e. the intended standardization was never wired up. mustFix=true because this is a class-wide exit-code correctness defect that silently masks failures across many CLI commands (a script/CI seeing exit 0 treats a genuinely failed command as success), the codebase already ships the exact decorator meant to fix it, and the project mandates production-readiness. Minor scope correction for accuracy: this is not "all failures exit 0" — commands that do not catch still make Typer exit non-zero on uncaught exceptions; the defect is specifically the subset that catches and returns, which the finding already scopes to.

---

### F-040 — [High] 🔒 VERIFIED · MUST-FIX PipelinedSpeculativeDecoder verifier always rejects (or silently accepts) every draft: draft slots never carry verifier inputs

`src\distllm\core\async_pipelined_speculative.py:437` · zone=`core-decoding` · category=`bug`

- **Summary:** async_pipelined_speculative.py DraftSlot has hidden_states and compressed_logits, but `_draft_worker` (invoked by _launch_draft) fills only `token_ids`/`logprobs` from `draft_gen(prompt, n)`. When a verifier IS configured, `_verify_worker` hits the else branch ('Verifier configured but draft slot has no hidden_states/compressed_logits -- rejecting draft') and rejects EVERY slot -> the pipeline never accepts a single draft token and degenerates to target-only generation while still paying for ring-buffer/thread overhead. When `verifier is None` (the documented default), `_verify_worker` sets `slot.accepted=True` unconditionally -> ALL draft tokens are appended with NO verification against the target distribution, so the output is purely the draft model's and is not speculative decoding at all. Either way the advertised 2-3x pipelining is not real.
- **Evidence (verbatim):**
```
if self._verifier is None: slot.accepted = True; return slot  ... else: logger.warning('Verifier configured but draft slot has no hidden_states/compressed_logits -- rejecting draft'); slot.accepted = False
```
- **Impact:** With default verifier=None the decoder emits unverified draft output (distributionally wrong); with a verifier set it silently does zero speculative speedup while consuming thread/stream resources. Both contradict the documented behavior.
- **Reliability:** Path: _launch_draft->_draft_worker creates DraftSlot(token_ids, logprobs) only; generate->_collect_verifications->_verify_worker: slot.hidden_states is None -> not `is not None` -> else -> slot.accepted=False (verifier set) or slot.accepted=True (verifier is None).
- **Recommendation:** Populate slot.hidden_states/compressed_logits in _draft_worker (make draft_gen return them or feed the draft tokens through the target to obtain them), AND require a verifier: if none is configured, fall back to a real p/q verifier against the target (or refuse to run) rather than accepting drafts blindly. Remove the 'accept everything when verifier is None' fallback. If the module stays unwired, mark it experimental rather than presenting it as a production decoder.
- **Verdict (real):** mustFix=True. Verified line-by-line in src/distllm/core/async_pipelined_speculative.py. The finding is accurate. (1) DraftSlot declares hidden_states/compressed_logits (lines 65-66) but _draft_worker (lines 380-388) builds slots from self._draft_gen(prompt, n) which returns only (token_ids, logprobs) (coutner line 175), so hidden_states and compressed_logits are ALWAYS None on slots produced by the real pipeline. (2) With a configured verifier, _verify_worker (lines 425-471) only calls the verifier inside `if slot.hidden_states is not None and slot.compressed_logits is not None` (line 443); since both are never set, execution always hits the else branch (lines 457-464) which logs and sets slot.accepted=False -- EVERY draft is rejected, accepted_tokens stays empty, and generate() falls back to target-only (lines 297-323). The whole verifier path is unreachable dead code; the only test exercising it (test_verify_worker_honors_verifier) hand-constructs a DraftSlot with those fields populated, which never happens in the pipeline. (3) With verifier=None (the constructor default, line 176), lines 437-440 set slot.accepted=True unconditionally, and in generate() the step-3 target forward logits (lines 291-293) are discarded, so the emitted sequence contains zero target-constraint tokens -- pure draft output, not speculative decoding, and it fails silently rather than erroring. This is load-bearing: the module's core advertised feature (verification-constrained pipelined decoding) cannot function in either configuration, and the default mode silently produces unverified draft output while discarding target compute. mustFix=true.

---

### F-041 — [High] 🔒 VERIFIED · MUST-FIX PagedAttention KV blocks leak for every completed sequence

`src\distllm\core\batch_scheduler.py:1065` · zone=`core-router-sched` · category=`bug`

- **Summary:** Completed sequences are removed from self.active inside step() without freeing their PagedAttention blocks, and the only free path (_prefetch_and_snapshot) scans only sequences still present in active. Because step() prunes completed sequences before the next _schedule_with_budget runs, free_paged_blocks never fires for them under PagedAttention, leaking KV blocks per completed request until the pool is exhausted.
- **Evidence (verbatim):**
```
step(): `with self._lock: done_rids = [s.request_id for s in batch.sequences if s.is_complete] ... self.active.pop(rid, None)` — no free_paged_blocks; the only free is _prefetch_and_snapshot: `done_ids = [rid for rid, s in self.active.items() if s.is_complete]; ... self.free_paged_blocks(rid)`, which sees an already-emptied active set.
```
- **Impact:** KV cache (VRAM) is not reclaimed on request completion in the PagedAttention path; long-running servers degrade to OOM/failed allocations as blocks leak. Directly contradicts the scheduler's own 'evict completed' comment.
- **Effort:** 0.5 hours
- **Reliability:** Reproduce: run BatchScheduler with a PagedAttention manager (allocate_sequence sets blocks, free_sequence releases). Submit N requests. After each completes, step() pops it from active; the next schedule()'s _prefetch_and_snapshot finds no is_complete seqs in active, so free_sequence is never called for any of them. Inspect pool total_blocks / free blocks: monotonically decreases while completed requests accumulate.
- **Recommendation:** In step(), while pruning completed sequences under the lock (lines 1065-1069), call self.free_paged_blocks(rid) for each completed request before self.active.pop(rid, None). Keep the free in exactly one place to avoid double-free; remove the now-redundant free in _prefetch_and_snapshot or guard it against double-free.
- **Verdict (real):** mustFix=True. Confirmed by reading the code. step() (batch_scheduler.py:1044-1069) marks completed sequences DONE and pops them from self.active/_chunked_prefill under lock WITHOUT calling free_paged_blocks; the only KV free for a sequence is done inside _prefetch_and_snapshot (line 700-706, done_ids = [rid for rid,s in self.active.items() if s.is_complete] -> free_paged_blocks) and its identical batch_builder.py:53-59 sibling — both invoked from _schedule_with_budget/schedule. Call ordering in request_pipeline.py (schedule at 494, step at 588/706) proves step() prunes completed rids from active BEFORE the next schedule() runs _prefetch, so the free branch never fires for step-completed sequences. No other free path exists (free_paged_blocks has only these 2 call sites reaching backend free_sequence via scheduler/kv_cache_manager.py:62-70; kv_cache.py:73 belongs to a different pager wrapper; is_complete at sequence.py:134 returns True for DONE so the prune is uniform). This is a genuine PagedAttention KV-block leak per completed request that will exhaust the pool under sustained load when PagedAttention is enabled — load-bearing enough to require a fix (add the free_paged_blocks call in step() where the pop happens). One scope caveat: it only manifests when _paged_attention_mgr is not None, but the claim explicitly scopes to that mode.

---

### F-042 — [High] 🔒 VERIFIED · MUST-FIX Hydra VideoPipeline feeds random noise to stable-video-diffusion-img2vid and rebuilds the whole model on every call

`src/distllm/core/hydra_diffusion.py` · zone=`core-gen-rag` · category=`bug`
✅ **FIXED** — `VideoPipeline` now `load()`s the pipe once (cached), `generate()` requires a real `init_image` (no `randn` noise), and `DiffusionPipeline.load()` uses mutually-exclusive `DataParallel` vs `enable_model_cpu_offload` branches. Tests: `tests/core/test_hydra_diffusion_f042.py` (5 pass).

`src\distllm\core\hydra_diffusion.py:79` · zone=`core-gen-rag` · category=`bug`

- **Summary:** VideoPipeline.generate hardcodes `stabilityai/stable-video-diffusion-img2vid`, ignores the requested model name, passes `t.randn(1,3,512,512)` (pure random noise) as the conditioning input to an image-to-video pipeline, and is instantiated fresh inside HydraOrchestrator.generate each request (line 120) so the model is downloaded/initialized (multiple GB) every generation and never cached in self._pipeline.
- **Evidence (verbatim):**
```
result = pipe(t.randn(1, 3, 512, 512), num_frames=num_frames).frames[0] pipe = VideoPipeline() if is_video else DiffusionPipeline()
```
- **Impact:** Garbage video frames (img2vid conditioned on noise), plus severe unbounded memory/network churn (GB-scale HuggingFace download per request) and no model reuse — a resource leak/performance hazard.
- **Reliability:** Call hydra.generate(model='my-video-model', prompt=...) → VideoPipeline() created (line 120) → generate() line 69-82 builds StableVideoDiffusionPipeline from a hard-coded checkpoint and calls it with a random-noise tensor as the img2vid conditioning frame, producing meaningless output and re-initializing the model on every call.
- **Recommendation:** Load the img2vid pipeline once (cache in self._pipeline, keyed by the resolved model) and pass an actual input image/frame rather than `t.randn`; add an 'input_image' param. Consolidate so HydraOrchestrator.generate reuses an existing VideoPipeline instead of constructing a new one every call.
- **Verdict (real):** mustFix=True. Confirmed by reading src/distllm/core/hydra_diffusion.py. (1) VideoPipeline.generate hardcodes "stabilityai/stable-video-diffusion-img2vid" at line 74 and never uses the requested model name — the video branch of HydraOrchestrator.generate (lines 120-123) never passes `model` to the video path, and VideoPipeline.load() (which would accept a model) is never invoked. (2) Line 79 feeds t.randn(1,3,512,512) pure random noise as the conditioning image input to an image-to-video model, and the text `prompt` argument is never used at all — the video output is garbage. (3) `pipe = VideoPipeline()` is created fresh as a local at line 120 on every request and never stored in HydraOrchestrator._pipelines, and StableVideoDiffusionPipeline.from_pretrained (multi-GB) runs inside generate() (line 73), so the model is re-downloaded/initialized on every generation. These are genuine Critical/High functional + resource bugs: model name and prompt ignored, garbage video output, and multi-GB re-download per call. mustFix=true.

---

### F-043 — [High] 🔒 VERIFIED · MUST-FIX AudioPipeline state machine gets permanently stuck in SPEAKING after the first utterance — only one utterance ever processed

`src/distllm/core/media_pipeline.py:254` · zone=`core-gen-rag` · category=`bug`
✅ **FIXED** — new speech now transitions from SPEAKING → LISTENING (not just IDLE → LISTENING), re-arming the silence-timeout path so a second utterance is processed. Tests: `tests/core/test_audio_multiple_utterances.py` (2 pass).

- **Summary:** In media_pipeline.py the pipeline is a one-shot: after the first spoken utterance produces TTS audio the state becomes SPEAKING and there is no transition back to IDLE/LISTENING. Any subsequent speech frame is buffered but leaves state in SPEAKING (the `is_speech` branch only promotes to LISTENING when state == IDLE), so the silence-timeout branch (`if self._state == PipelineState.LISTENING`) never fires again and no second utterance is ever transcribed/answered.
- **Evidence (verbatim):**
```
if self._state == PipelineState.IDLE: self._state = PipelineState.LISTENING ... if self._state == PipelineState.LISTENING: if time.time()-self._last_speech_time > self._silence_timeout: ... _process_utterance ... self._state = PipelineState.IDLE if not audio_out else PipelineState.SPEAKING
```
- **Impact:** With TTS enabled the real-time voice pipeline dies after the first response; a conversation can never continue. tests/core/test_media_pipeline.py only checks transitions IDLE→LISTENING→PROCESSING, never SPEAKING→IDLE, so the defect ships green.
- **Reliability:** Send speech frame → TTS returns bytes → state=SPEAKING (line 273). Send a second speech frame: line 239 is_speech True appends to buffer but `if self._state == PipelineState.IDLE` (line 242) is False, stays SPEAKING, returns None. Then silence: line 247 `state == LISTENING` False → returns None. The SPEAKING value is only other referenced in test_media_pipeline.py line 31 for the enum value, never exercising two sequential utterances.
- **Recommendation:** Add SPEAKING→LISTENING (or IDLE) transition: e.g. in _process_utterance set `self._state = PipelineState.IDLE` after dropping the segment into audio_out_queue, or treat speech during SPEAKING as a barge-in that resets to LISTENING (call _process_utterance). Add a two-utterance regression test asserting a second frame is transcribed.
- **Verdict (real):** mustFix=True. Read the file. Traced the state machine: after the first utterance produces TTS audio, _process_utterance sets _state = SPEAKING (line 273). The only transition to LISTENING is the `if self._state == IDLE` guard inside the is_speech branch (line 242), which is skipped while in SPEAKING. The only transitions to IDLE are inside _process_utterance (lines 263, 273), which is never reached again. The silence-timeout transcription trigger (line 247-253) only fires when state == LISTENING. So once SPEAKING, no code path ever returns to IDLE/LISTENING: subsequent speech frames buffer with no effect, subsequent silence does nothing, and no second utterance is ever transcribed or answered. This is a genuine one-shot/terminal-state defect in a real-time voice-conversation pipeline, load-bearing enough to be a release-blocking Critical. Incidental line detail (silence guard) is at 247 not 254 but the cited line~254 range and logic are accurate.
  - *Correction:* Confirmed exactly as described. No correction needed to the claim's mechanics. Note additionally that the stuck SPEAKING state also causes unbounded buffer growth: every speech frame after the first utterance is appended to _buffer (line 240) with no path that ever clears them again, since _process_utterance (which does buffer.clear()) is unreachable once stuck.

---

### F-044 — [High] 🔒 VERIFIED · MUST-FIX Concurrent-request dedup waiters always time out: _in_flight_results is never populated

`src/distllm/core/request_fingerprinting.py:171` · zone=`core-router-sched` · category=`bug`
✅ **FIXED** — `store()` now publishes a non-empty response to `_in_flight_results` before signalling, so waiters receive the actual result instead of timing out. Tests: `tests/core/test_request_fingerprinting.py` (was an xfail documenting the bug; now real).

`src\distllm\core\request_fingerprinting.py:171` · zone=`core-router-sched` · category=`bug`

- **Summary:** RequestFingerprinter implements in-flight dedup where a second identical request is supposed to wait on the first and receive its result via _in_flight_results. But nothing in the production code ever assigns a value to _in_flight_results — mark_in_flight and clear_in_flight only pop from it, and wait_for_result only reads it. The only writer is test_request_fingerprinting.py which sets the private field directly, masking the bug. Result: wait_for_result() always blocks the full timeout and returns None for a duplicated request.
- **Evidence (verbatim):**
```
_in_flight_results used at lines 71 (init {}), 97 (pop), 107 (pop), 184 (read in wait_for_result), 202 (read); grep over source confirms no `_in_flight_results[fprint] = ...` assignment. wait_for_result: `result = self._in_flight_results.get(fingerprint); if result is not None: return result` — always None.
```
- **Impact:** The documented 'wait for identical in-flight result' optimization is non-functional and adds a full timeout (default 30s) plus a hang risk to duplicated requests; callers returning this None may treat a live request as failed.
- **Effort:** 0.5-1 hours
- **Reliability:** Two threads call mark_in_flight(fp); thread B calls wait_for_result(fp, timeout=5); thread A completes and calls store(fp,...) then clear_in_flight(fp). B's event is signalled (store calls _signal_waiting) but B's subsequent `self._in_flight_results.get(fp)` returns None, so B returns None after the event wakes, having lost the result it was waiting for.
- **Recommendation:** Populate _in_flight_results when the FIRST in-flight request completes: in store() (and/or in clear_in_flight when the last request_id leaves) write `self._in_flight_results[fingerprint] = response` so waiters get it, then evict on a TTL. Add a test that calls wait_for_result() WITHOUT touching the private field and asserts the duplicate receives the first request's response.
- **Verdict (real):** mustFix=True. Confirmed by reading the code. The private dict `_in_flight_results` is initialized to {} (line 71) and only ever read via .get() (lines 184, 202 in wait_for_result) and popped (lines 97, 107 in mark_in_flight/clear_in_flight). The production `store()` method writes only to `self._cache` and calls `_signal_waiting` to set wait events; it never assigns `_in_flight_results`. The ONLY assignment anywhere in the repo is the test `tests/core/test_request_fingerprinting.py:275` poking the private field directly, and the suite even contains an explicit xfail test (lines 295-322) whose reason states the bug: "store() never writes to _in_flight_results, so wait_for_result() always returns None." Impact is live: `request_pipeline.py:158-159` calls wait_for_result on concurrent identical requests, which always blocks the full 30s default timeout and returns None, then falls through and regenerates the response. So the documented concurrent-dedup feature never works and every duplicate request incurs a 30s stall. This is a real, load-bearing functional defect in a shipped core feature, not a code smell, and the fix is trivial (assign the result in store/on completion). mustFix=true; the finding's cited lines and behavior are accurate.

---

### F-045 — [High] 🔒 VERIFIED · MUST-FIX JSONSchemaConstraint FSM never allows continuation of multi-digit numbers — constrained generation truncates numeric values

`src\distllm\core\structured_output\__init__.py:202` · zone=`core-gen-rag` · category=`bug`

- **Summary:** In structured_output/__init__.py the token-level mask's `_valid_next_chars` has no entry for the `in_number` state, so the transitions dict falls back to the default `{'"', '}'}` (line 202). After the model emits the first digit the state becomes `in_number` (line 252), and the next mask blocks every digit-starting token, so numbers longer than one character (and decimals/exponents) cannot be emitted even though `_transition` explicitly maintains `in_number` for `0123456789.eE+-` (line 255).
- **Evidence (verbatim):**
```
return transitions.get(self._state, {'"', '}'})  # 'in_number' absent from the dict if state == "in_number" and char in '0123456789.eE+-': return "in_number"
```
- **Impact:** Structured output (a headline feature) generates corrupted/truncated JSON for any numeric field value > 1 digit, and silently degrades to single-char numbers. test_structured_output_fsm.py only asserts mask shape and that some tokens are allowed (lines 92-106); it never asserts digit continuation, so the bug is uncaught.
- **Reliability:** Emit token '1' from after_colon → update() returns "in_number". Next get_logits_mask computes _valid_next_chars for 'in_number' → default {'"','}'} → valid ords are only quote/brace → a token whose first char is '2' is masked out. Hence '{"a": 12}' is unproducible; only single-digit numbers work. confirmed against the fsm code (lines 185-260).
- **Recommendation:** Add an 'in_number' entry to the transitions dict in _valid_next_chars returning the digit/exponent char set ('0123456789.eE+-') in addition to the number terminator chars, mirroring _transition. Add an FSM test that feeds '{"a": 1' and asserts the next valid ord set includes '2'/'9'/'.'.
- **Verdict (real):** mustFix=True. Directly executed the actual JSONSchemaConstraint FSM in structured_output/__init__.py. After emitting the first digit, _transition returns 'in_number' (line 251-252) and self._state becomes 'in_number'. The next get_logits_mask call invokes _valid_next_chars(), whose transitions dict (lines 190-201) has no 'in_number' key, so the fallback on line 202 returns {'"','}'}. My run confirmed that in the in_number state every digit 0-9 and every decimal/exponent char .eE+- is masked False; only '"' (forces the value into a string) and '}' (premature close) are allowed. Meanwhile _transition (line 255) correctly keeps in_number alive for further digits — the mask and transition halves are inconsistent. Consequence: any numeric value longer than one character, plus all decimals and exponents, cannot be produced as a number; values are truncated to their first digit or coerced into strings. The class is live in production paths (request_pipeline.py:413/417 constructs it via from_response_format and line 642 calls get_logits_mask; inference_engine.py, token_generator.py, and the API chat/streaming routes use it for JSON-mode constrained generation). This silently corrupts structured numeric output — a genuine correctness bug, not cosmetic — so mustFix=true.
  - *Correction:* Verified against live class in src/distllm/core/structured_output/__init__.py. Cited lines correct (fallback at 202, digit->in_number at 251-252, in_number maintenance at 255). Note: production wiring confirmed via request_pipeline.py:413/417, inference_engine.py, token_generator.py, chat.py/streaming.py.

---

### F-046 — [High] 🔒 VERIFIED · MUST-FIX Off-by-one token-position indexing in 4 of 9 speculative verifiers: each draft token is checked against the NEXT token's prediction

`src\distllm\core\tree_speculative_decoder.py:326` · zone=`core-decoding` · category=`bug`

- **Summary:** Across the zone the acceptance/rejection math indexes target logits inconsistently. 5 verifiers correctly use `prefix_len - 1 + i` (logits[k] predicts token[k+1], so draft token i at position P+i is predicted by logits[P-1+i]); 4 verifiers use `prefix_len + i`, comparing draft token i against the logit that predicts token i+1. For greedy mode this accepts the wrong token when the next position's argmax happens to equal draft[i]; for stochastic mode it draws the acceptance probability p from the wrong token's probabilities, corrupting the min(1, p/q) decision. Because speculative decoding verifies against a single shared forward pass, this silently emits tokens the target never predicted. 3 of the 4 buggy verifiers (batched-tree, multi-draft-verifier, MTP) are shipped/exported and tested but not on the default inference_engine path; the 4th (TreeDraftSpeculativeDecoder._verify_tree_batched) is the class's default high-performance path, turned on whenever batch_size>1 (a real tree).
- **Evidence (verbatim):**
```
pos = prefix.shape[1] + j  # branch token j; target_probs = F.softmax(target_logits[i, pos, :])  ...  p = target_probs[expected_token].item()
```
- **Impact:** Silent wrong acceptance/rejection for 4 classes; emit tokens the target model never predicted; acceptance-rate stats inflated/deflated; tree speculative decoding (the claim is 2-3x acceptance gain) computes acceptance against the wrong positions.
- **Reliability:** Repro: construct any TreeDraftSpeculativeDecoder with 2+ branches so batch_size>1 triggers _verify_tree_batched; feed a 2-length batch where draft[0]=X and target's predicted next token for position P is X but its predicted next is Y. Greedy code checks argmax at P+i (predicts seq[i+1]) against seq[i], accepting X incorrectly. Cross-check the correct pattern: draft_tree.py verify_tree line 201 `pos = prefix_len + i - 1` and speculative_decoder.py _verify_tokens line 210/216 `prefix_len = prefix.shape[1]`, `target_logits[:, prefix_len + i - 1, :]`.
- **Recommendation:** Change all four to match the correct convention already used by SpeculativeDecoder._verify_tokens and distributed_speculative._verify_tokens: use index `prefix.shape[1] - 1 + i` (or `generated.shape[1] - 1 + i`). Specifically: tree_speculative_decoder.py line 326 `pos = prefix.shape[1] + j - 1`; speculative_decoder.py `_verify_tree_batched` line 940/949 use `prefix_len = prefix.shape[1] - 1`; multi_draft_verifier.py line 121 `pos = prefix_len + i - 1`; mtp_head.py `_verify_tokens` use `prefix_len = prefix.shape[1] - 1`. Add a shared verifier helper so the 9 copies cannot drift again.
- **Verdict (real):** mustFix=True. Confirmed by reading the code. speculative_decoder.py's _verify_tokens explicitly documents the convention 'logits[k] -> token[k+1]' and uses target_logits[:, prefix_len+i-1, :] for draft token i at position prefix_len+i. Both batched tree verifiers instead index prefix_len+i (tree_speculative_decoder.py line 326: `pos = prefix.shape[1] + j`; speculative_decoder.py line 949: `prefix_len + i`), reading the logit at the draft token's own position, which predicts the NEXT token — the exact off-by-one described. In greedy mode this silently accepts a draft token when the next position's argmax coincidentally equals draft[j]; in stochastic mode it samples p from the wrong token's distribution, corrupting the min(1,p/q) decision. This emits tokens the target never predicted, breaking the draft-target self-consistency guarantee of speculative decoding, with no error surfaced. The batched tree verifier is the default high-performance path for real trees (batch_size>1), so the bug is on a shipped, reachable default path. The only defect in the finding is naming: the title cites TreeDraftSpeculativeDecoder._verify_tree_batched while the evidence line is tree_speculative_decoder.py's _verify_branches — but both carry the identical off-by-one, so the substance stands. Genuine silent correctness corruption: must fix before release.
  - *Correction:* The off-by-one is confirmed, but it exists in TWO files: the cited snippet (pos = prefix.shape[1] + j) is tree_speculative_decoder.py:326 in TreeSpeculativeDecoder._verify_branches (the tree module's main verify path, called every generate() step when branches exist), while the title's named class TreeDraftSpeculativeDecoder._verify_tree_batched at speculative_decoder.py:949 (logits_slice = target_logits[seq_idx, prefix_len + i, :]) carries the identical prefix_len+i bug and is the default when batch_size>1 (line 995). The correct index prefix_len+i-1 is documented in the same file's _verify_tokens (lines 214-216, 446-448).

---

## Real but not release-blocking (10)

### F-047 — [Critical] 🔒 verified real RedundantExecutor._run_redundant is a non-functional stub — enabling redundancy>1 always fails

`src/distllm/dist/redundant.py:96` · zone=`dist-exec` · category=`bug`

- **Summary:** The entire redundant speculative-parallelism path (the module's headline feature) is a stub. `_run_redundant` defines local `_forward_request_to_proto` and `_process_forward_response_pb` functions that unconditionally `raise NotImplementedError` (H-05 comment: 'These functions don't exist in pipeline module'). At line 131/154 every stage calls one of them → the exception is swallowed by `except Exception: continue` → all candidates fail → `raise NodeUnreachableError('no redundant node succeeded')`. So `redundancy > 1` can never produce a result; it always surfaces as an opaque node-unreachable error instead of the graceful 'use the fastest peer' behavior the docstring promises.
- **Evidence (verbatim):**
```
def _forward_request_to_proto(*args, **kwargs):\n    raise NotImplementedError(\"_forward_request_to_proto not yet implemented\")
```
- **Impact:** The primary resilience feature (run the same stage on N peers, take the first result) is completely non-functional; operators enabling redundancy get an opaque failure and zero latency-variance protection.
- **Reliability:** Set redundancy=2, call run_pipeline with a two-node pipeline — stage 0 hits `_forward_request_to_proto` → NotImplementedError → caught → 'no redundant node succeeded' NodeUnreachableError every time.
- **Recommendation:** Delete `redundant.py`'s stub class or wire it to the real serialization helpers. Implement `_forward_request_to_proto`/`_process_forward_response_pb` against `distllm.dist.node_client.forward_request` + `pipeline.serialization.to_proto_tensor/from_proto_tensor` (as used by node_client.py), and add a unit test that runs `_run_redundant` with two fake nodes and asserts one output is returned. If redundancy is intentionally unreachable, gate `RedundantExecutor` creation behind a clear error rather than a silent runtime failure.
- **Verdict (real):** mustFix=False. The line-level code facts are accurate: both local stub functions unconditionally raise NotImplementedError, every redundant candidate calls one inside the try/except that swallows it, so results stays empty and NodeUnreachableError is raised — enabling this executor with redundancy>1 can never produce a result. However, this is not load-bearing for release: redundant.py is a documented, un-wired reference implementation (dist/CLAUDE.md), superseded by the real production redundant_executor.py; the user-facing --redundancy config routes to PipelineOrchestrator, not this class; and the test suite itself already treats the redundant path as broken (expects AttributeError at _topology_lock). It is dead/unwired reference code and a code-quality/cleanup item, not a Critical/High release blocker that must be fixed.
  - *Correction:* The stub mechanism is real as described (lines 98-101, calls at 131/154, swallow at 138-140, NodeUnreachableError at 145-149), but it is NOT the production redundancy path: coordinator.py:111-115 routes config.redundancy to PipelineOrchestrator, and the fully-implemented production executor is the separate redundant_executor.py (class at line 1192). redundant.py is explicitly documented in dist/CLAUDE.md as "a planned architecture + reference implementation" NOT wired into any production path, and is labeled the older-generation duplicate.

---

### F-048 — [High] 🔒 verified real dynamic_sharder installs a new partition after migrations that never transferred data

`src/distllm/core/dynamic_sharder.py:303` · zone=`core-training` · category=`bug`

- **Summary:** In _migrate_layer the transfer step only runs `if self._on_transfer:`; with the default None it marks the layer COMPLETE without moving data. _execute_reshard then unconditionally installs new_partition, so routing points layers at nodes that never received them (on_node_leave does the same for a departed node's layers).
- **Evidence (verbatim):**
```
migration.state = MigrationState.TRANSFERRING if self._on_transfer:     success = self._on_transfer(...)     if not success: raise ... ... self._current_partition = dict(plan.new_partition)  # installed regardless
```
- **Impact:** Every join/leave 'reshard' silently lies about success; traffic routes to nodes lacking the layer, breaking generation and zero-downtime.
- **Effort:** 1 day
- **Reliability:** DynamicSharder() (no on_layer_transfer) -> on_node_join -> migrations 'complete' but no data moved, partition lists the new node.
- **Recommendation:** Require a real on_layer_transfer, implement VERIFYING (checksums) and SWITCHING (routing update), and only install new_partition after all migrations succeed; on any failure keep old_partition and mark the round failed. Make migration async so the 100ms drain sleep does not block the caller.
- **Verdict (real):** mustFix=False. The claim's description of the code is accurate and verified by reading src/distllm/core/dynamic_sharder.py. on_layer_transfer defaults to None (line 91); _migrate_layer (lines 301-310) runs the transfer only `if self._on_transfer:` and, when None, skips data movement yet still marks VERIFYING (line 313)/SWITCHING (line 317)/COMPLETE (line 320); and _execute_reshard (lines 278-282) unconditionally installs `self._current_partition = dict(plan.new_partition)`. So a default-constructed reshard reports success and installs a partition whose layers were never transferred. However, it is NOT load-bearing for release because the module is not wired into any production code path: a repo-wide grep of src/distllm shows DynamicSharder is referenced only in its own docstring and in tests (tests/core/test_dynamic_sharder.py, tests/test_features_autopilot_privacy_plugins.py). No coordinator/cluster_manager/api/dist production module instantiates or calls it. The transfer/verify/switch steps are explicit scaffold stubs ("would go here in production"), the docstring example calls a non-existent sharder.start() method, and the test suite confirms the intended contract: every test constructs DynamicSharder() with no on_layer_transfer and asserts successful partition-math resharps, while test_get_active_migrations (line 127) monkeypatches _migrate_layer to a no-op and still expects a valid reshard. A no-op migration is the designed, test-observable behavior of this unwired stub. If the class were ever wired into live routing it would be a genuine Critical data-liveness/misrouting bug, but as shipped nothing exercises data movement, so it does not meet the must-fix-before-release Critical/High bar. Correction to the original wording: the described consequence is theoretically correct for the class's internal state but overstated as an actual risk — there is no production routing that consumes _current_partition. Recommend wiring it up properly (real transfer/verify/switch) or marking it advisory before integration.

---

### F-049 — [High] 🔒 verified real Streaming layer-weight transfer has no integrity verification (checksum, ordering, completeness ignored)

`src/distllm/dist/node_client.py:500` · zone=`dist-exec` · category=`security`
✅ **FIXED** — the streaming client now validates chunk ordering (monotonic `chunk_index`), constant `total_chunks`, single final chunk, and completeness; reordered/truncated/duplicate streams are rejected instead of assembled. Tests: `tests/dist/test_streaming_integrity.py`.

`src/distllm/dist/node_client.py:500` · zone=`dist-exec` · category=`security`

- **Summary:** The streaming path (the one intended for large models, up to 512MB) concatenates chunks and returns them with zero validation: it ignores `chunk_index`, `total_chunks`, and `is_final_chunk` from the server, and does NOT compute/compare the SHA-256 trailing-metadata checksum the way the non-streaming `request_layer_weights` (lines 305-322) does. A truncated, reordered, or tampered stream is accepted and later passed to `torch.load(...); full_model.load_state_dict(state_dict, strict=False)` in worker.py (lines 150-162), silently loading a corrupted/poisoned weight set. This is the de-trust/verification-contract thread: the only-weight-safety measure (SHA-256) exists on the single-response RPC but is absent on exactly the path that carries the largest payloads.
- **Evidence (verbatim):**
```
for resp in client.stub.TransferWeightsStream(req):\n    if resp.success and resp.state_dict_bytes:\n        buffer.extend(resp.state_dict_bytes)
```
- **Impact:** A flaky or malicious peer can deliver a truncated/corrupted weight bundle for the entire layer range and it is loaded into the model silently, producing garbage output or a poisoned model with no error.
- **Reliability:** Call request_layer_weights_stream against a server whose TransferWeightsStream drops the last chunk; the client returns the truncated buffer as a 'success' with no error, and worker.load_model caches + loads it.
- **Recommendation:** Mirror `request_layer_weights` verification in the streaming path: collect all chunks, verify `resp.success` per chunk, track `chunk_index` to reject reorder/duplicate/gap, require the final `is_final_chunk`, then read the `x-checksum-sha256` trailing metadata from the final response and compare `hashlib.sha256(bytes(buffer))`. Return None on any mismatch so worker.py falls back to HF instead of loading corrupt weights.
- **Verdict (real):** mustFix=False. The code-level defect is genuine and verifiable: node_client.py:500-507 (request_layer_weights_stream) concatenates chunks via buffer.extend(resp.state_dict_bytes) and ignores chunk_index/total_chunks/is_final_chunk, with no SHA-256 check — whereas the non-streaming request_layer_weights (node_client.py:305-322) does verify the x-checksum-sha256 trailing metadata. The server sets those index fields per chunk (node_service.py:293-306) and the client discards them. However, the load-bearing impact claim is wrong: worker.py:142-176 (the cited "lines 150-162" torch.load/load_state_dict) calls the checksum-verified request_layer_weights, NOT the streaming function, and request_layer_weights_stream has zero production callers — grep shows it is only invoked from tests (tests/dist/test_node_client.py, tests/integration/test_grpc_reconnection.py). A corrupted stream therefore cannot reach torch.load/load_state_dict in any live weight-load path today. This is real defense-in-depth hygiene on effectively test-only code (worth fixing so future wiring doesn't silently lose integrity), but it is not an active Critical/High that must block release.
  - *Correction:* Missing-integrity-check in request_layer_weights_stream (node_client.py:500-507) is real, but it is NOT reachable in production: worker.py:150-162 loads via the checksum-verified request_layer_weights, and request_layer_weights_stream has no production caller (only tests call it). Severity is latent/hygiene, not a live data-corruption or trust exploit.

---

### F-050 — [High] 🔒 verified real Kademlia routing table trusts sender-supplied IP:port over packet source

`src/distllm/dist/p2p/kademlia_dht.py:826` · zone=`dist-net` · category=`security`

- **Summary:** PING, STORE, FIND_NODE and FIND_VALUE handlers all build the KademliaNode from the sender-declared 'ip'/'port'/node_id in the JSON body instead of the actual UDP source addr. A spoofed datagram can add an arbitrary (node_id, victim_ip:port) entry to the routing table. Because insert into a full bucket is a no-op (routing_table.insert_node returns False when it can't split), an attacker who floods declared entries can eclipse the table and redirect lookups to victims.
- **Evidence (verbatim):**
```
node_id=_hex_to_node_id(sender_id_hex), ip=sender_data.get("ip", addr[0]), port=sender_data.get("port", addr[1])
```
- **Impact:** Traffic-redirection / eclipse attack enabling DHT value poisoning and man-in-the-middle of peer discovery.
- **Reliability:** Send a PING/FIND_NODE datagram with sender.ip=sender.port set to a victim; the victim's KademliaNode is inserted into routing_table even though the datagram physically came from the attacker.
- **Recommendation:** On datagram_received, record addr and prefer ipaddress.ip_address(addr[0]) (with the declared values used only as a hint) — standard Kademlia auth uses the transport address. Ignore self-declared IPs that disagree with the observed source.
- **Verdict (real):** mustFix=False. The described defect genuinely exists verbatim in the code I read. All four RPC handlers (_handle_ping line 821, _handle_store 834, _handle_find_node 860, _handle_find_value 883) build a KademliaNode from `sender_data.get("ip", addr[0])` / `sender_data.get("port", addr[1])` (body-declared values preferred over the actual UDP source addr; lines 828-829 / 846-847 / 871-872 / 899-900), and insert_node returns False (no-op, table unmodified) when a full bucket can't be split (lines 262-294). A spoofed datagram can therefore insert arbitrary (node_id -> ip:port) mappings, and the attacker can fill buckets with K=20 fake node_ids closer to a target than honest ones, so find_nearest/FIND_NODE return attacker-controlled entries (key-specific poisoning/eclipse). The cited line ~826 is correct (line 826 = `sender = KademliaNode(...)`).  However, mustFix=false because the module is NOT reachable: nothing in src imports KademliaDHT, no test exercises it, and dist/CLAUDE.md's key-files table lists discovery.py and gossip.py (not this DHT). Production discovery is FederationPeerDiscovery (heavily tested). No KademliaDHT.start() call exists anywhere, so no datagrams are received in the current build — the flaw is in unwired/orphaned code with no live attack surface. The fix suggestion (bind to addr[0]/addr[1], i.e., the real source) is correct, but it is not a must-fix-before-release item until the DHT is actually wired into production paths.
  - *Correction:* Finding is real as a code defect and the cited area is accurate, but the DHT module (src/distllm/dist/p2p/kademlia_dht.py) is orphaned/unwired — no imports in src and no tests reference it, so it is not currently load-bearing; must-fix is downgraded despite the genuine vulnerability.

---

### F-051 — [High] 🔒 verified real LearnedCostModel train/serve feature skew — intermediate_size and flops features differ between training and inference

`src/distllm/dist/partition/learned_cost.py:90` · zone=`dist-partition` · category=`bug`

- **Summary:** FeatureExtractor.extract (used at serving time) initializes intermediate_size=0 and never assigns it (the loop sets only hidden_size), so feature[14] is always 0.0. Training uses _observation_to_features which sets the real intermediate_size and uses proxy FLOPS/memory formulas (hidden^2*L*B*S*2 etc.) that don't match FeatureExtractor's actual flops sum. The model trains on one distribution and serves on another, producing unsound latency predictions that the partition decision would trust.
- **Evidence (verbatim):**
```
intermediate_size = 0         for l in layers:             if l.layer_type == "transformer" and l.weight_memory_bytes > 0:                 h_sq = ...hidden_size = max(...)                 break   # intermediate_size never updated
```
- **Impact:** The learned cost model, when enabled, will predict on misaligned features and can steer the DP to sub-optimal or OOM partitions; silently wrong since it 'falls back' identically.
- **Effort:** 2-4 hours
- **Reliability:** Compare _observation_to_features (lines 462-479, sets obs.intermediate_size and hidden^2*L*B*S*2 proxy at index 3) vs FeatureExtractor.extract (lines 120-136: returns float(intermediate_size)=0 at index 14 and real flops sum at index 3). The 15-feature vectors train/serve disagree on indices 3,4,5,6,14,15.
- **Recommendation:** Derive serving features from the same function as training: set intermediate_size from hidden/intermediate ratio, and build _observation_to_features from FeatureExtractor.extract-equivalent fields. Add a test asserting train-time features == serve-time features for a canonical node/config.
- **Verdict (real):** mustFix=False. The code defect is genuinely present and accurately described. In src/distllm/dist/partition/learned_cost.py, FeatureExtractor.extract initializes intermediate_size=0 (line 90) and the loop (lines 91-95) assigns only hidden_size, so feature[14]=intermediate_size is always 0.0 at serving time (return list line 135), while the training path _observation_to_features (line 478) feeds the real obs.intermediate_size. Features [3]-[6] (total_flops/weight_mem/kv_mem/act_mem) also legitimately use different formulas between the two paths (proxy hidden^2*L*B*S*2 etc. vs actual sum(flops_per_seq)*B*S). So the train/serve distribution skew is real. That said, it is NOT load-bearing and mustFix=false: grep across all of src/ shows LearnedCostModel is never instantiated in any production path — only in the module itself, tests (test_learned_cost.py, test_competitive_features.py), and its own LearnedCostConfig dataclass (config.py). The serve-time extract path is reachable only through LearnedCostModel.evaluate(), which no production caller invokes, and nothing calls .record() with a real RuntimeObservation at runtime. Because the learned model is unwired, the feature skew cannot currently produce the "unsound latency predictions the partition decision trusts" that the finding asserts — it is a latent bug in effectively dormant code, not a release-blocking Critical/High.

---

### F-052 — [High] 🔒 verified real Bandwidth congestion gating silently drops any transfer larger than the congestion window

`src/distllm/dist/pipeline/bandwidth_controller.py:650` · zone=`dist-net` · category=`bug`

- **Summary:** PipelineTransportController.send() gates against the congestion window (initial_cwnd = 10*1460 = 14600 bytes). When the payload exceeds the window it sends only the first `window` bytes and stores the remainder in self._send_buf — but nothing ever drains _send_buf. recv()/on_ack() never send buffered data, so ALL bytes beyond the current cwnd (any tensor > ~14KB) are silently dropped, corrupting the reassembled tensor. The docstring's "buffered and sent incrementally" is not implemented.
- **Evidence (verbatim):**
```
if total_bytes > window: self._send_buf = data; sent = 0; for i, chunk in enumerate(chunks): ... self._send_buf = data[sent:] ... ; (no flush path exists)
```
- **Impact:** Any single tensor/payload over ~14KB is truncated at the receiver — silently wrong pipeline activations with no error.
- **Reliability:** send(b'A'*100000) via a loopback send_fn then recv(): returned bytes length < 100000 (only window bytes delivered) and remainder lost.
- **Recommendation:** Either implement an async drain (on on_ack, pop from _send_buf and send while cwnd allows), or remove the gating entirely for the elastic buffer size (multi-stream striping already softens burstiness). Add an integration test sending a 1MB buffer and asserting recv() equals the input.
- **Verdict (real):** mustFix=False. The defect genuinely exists exactly as described. Verified in src/distllm/dist/pipeline/bandwidth_controller.py lines 650-666: when total_bytes exceeds the congestion window (initial_cwnd = 10*1460 = 14600 bytes, line 548/260), send() sends only up to `window` bytes and stores the tail in self._send_buf (lines 652/664), then breaks, while the docstring (lines 614-615) promises "buffered and sent incrementally". Nothing drains _send_buf: recv(), on_ack(), on_loss() never resend it; it is only assigned in send, cleared in reset, and read for length in stats(). Bytes beyond the window are thus silently dropped, truncating the reassembled tensor for any payload > ~14.6KB. HOWEVER, mustFix=false: the entire bandwidth_controller module has zero call sites — no imports and no tests anywhere in src/ or tests/ (grep for PipelineTransportController/bandwidth_controller/_send_buf matches only the module itself, docs, and a workflow JS file). It is dead/unwired code with no production path, so while the latent bug is real, it cannot affect any released endpoint or user and is not a release-blocker. Recommended as hygiene: either delete the dead module or implement the promised _send_buf drain in on_ack().

---

### F-053 — [High] 🔒 verified real StreamingKVTransfer breaks on bf16 tensors

`src/distllm/dist/streaming_kv_transfer.py:80` · zone=`dist-net` · category=`bug`

- **Summary:** chunk_tensor calls t.numpy() directly on the tensor. For a bfloat16 tensor torch.Tensor.numpy() raises TypeError ("Got unsupported ScalarType BFloat16") — so bf16 KV transfers crash on send. If it did serialize, reassemble_chunks maps 'torch.bfloat16' to np.float16, which is a DIFFERENT bit layout, silently corrupting the data. Most modern LLMs run bf16, so this streaming path is unusable/corrupting for them.
- **Evidence (verbatim):**
```
if t.is_cuda: t = t.cpu(); raw = bytes(memoryview(t.numpy())) ; ... dtype_map = { "torch.bfloat16": (np.float16, torch.float16)  # BF16 stored as float16 }
```
- **Impact:** BF16 KV-cache transfers either crash or silently corrupt cache data during streaming (gRPC chunked) transfer, a common dtype in production.
- **Reliability:** chunk_tensor(torch.zeros(2,2, dtype=torch.bfloat16), ...) raises TypeError / reassemble with a hand-built bf16 buffer mis-interprets bytes.
- **Recommendation:** Explicitly convert bf16 to float32 (or to float16 via .float()) on the send side before numpy, and decode with the matching dtype on receive. Remove the misleading bf16->fp16 mapping. Add a bf16 round-trip test.
- **Verdict (real):** mustFix=False. Verified by reading the code. Line 80 `chunk_tensor` calls `t.numpy()` before any dtype branch, and torch.Tensor.numpy() raises `TypeError: Got unsupported ScalarType BFloat16` for bf16 (not a supported NumPy dtype). This is 100% confirmed — the repo's own test `test_roundtrip_bfloat16` (tests/dist/test_streaming_kv_transfer.py:343-348) asserts exactly this TypeError with the comment ".numpy() does not support BFloat16". The reassemble_chunks dtype_map (lines 133-137) additionally maps torch.bfloat16 -> np.float16, which is a different bit layout (bf16 = 8-bit exponent/7-bit mantissa vs float16 = 5-bit exponent/10-bit mantissa), so it would corrupt data if it ever survived serialization. Claim is technically accurate.  However, mustFix is false with genuine doubt about severity: grep shows StreamingKVTransfer/chunk_tensor/stream_send have NO production call sites in src/ — the module is referenced only from its own docstring and tests (and workflow JS). The active KV-cache transfer path is protobuf-based `dist/cross_cluster.py`, not this streaming module. So today no running bf16 runtime path reaches this code; it is a real-but-unwired standalone helper. It should absolutely be fixed before this streaming path is adopted (bf16 is the dominant modern-LLM dtype), but as currently wired it is not a release-blocking runtime crash. Not a Critical/High that must be fixed for the current release.
  - *Correction:* Verified real and matches the code exactly; the only caveat is that the module has no in-src production caller today (active KV transfer uses cross_cluster.py protobuf), so the defect is real but not yet reachable in a running bf16 path.

---

### F-054 — [High] 🔒 verified real download_layer_subset returns a directory that contains only a manifest, no weights or index

`src/distllm/models/model_hub.py:220` · zone=`core-training` · category=`bug`

- **Summary:** Layer-aware download fetches shards via hf_hub_download WITHOUT local_dir (they land in the HF shared cache), writes only .layer_manifest into layer_subdir, and returns layer_subdir. The returned path therefore has no model.safetensors*, config, or tokenizer files, contradicting the docstring and being useless to any consumer that loads weights from the returned directory. Additionally the shards it downloads are never integrity-verified.
- **Evidence (verbatim):**
```
layer_subdir.mkdir(parents=True, exist_ok=True) manifest = {...} with open(manifest_path, "w") as f: json.dump(manifest, f)  # only file written into layer_subdir
```
- **Impact:** Layer-scoped nodes get a path with no loadable weights; inference and subsequent resolve_layer_subset fail.
- **Effort:** 3-6 hours
- **Reliability:** path = hub.download_layer_subset('meta-llama/Llama-3.1-8B', 0, 3); list(Path(path).glob('*.safetensors')) == [].
- **Recommendation:** Download the needed shards (and index/config/tokenizer files) with local_dir=str(layer_subdir) so the returned path physically contains weights, or return the HF-cache path the shards actually reside in. Then verify against expected hashes from the index metadata.
- **Verdict (real):** mustFix=False. The core factual premise is confirmed by reading model_hub.py lines 89-244: download_layer_subset fetches the safetensors index and shards via hf_hub_download WITHOUT local_dir (they land in the HF shared cache at ~/.cache/huggingface/hub/), writes ONLY .layer_manifest into layer_subdir (lines 221-238), and returns str(layer_subdir) (line 244). So the returned directory genuinely contains no model.safetensors*/config/tokenizer files, contradicting the docstring (lines 116-119) which claims the dir "contains model.safetensors.index.json plus only the needed .safetensors shard files." That is a real defect: the docstring/API contract is wrong about what the returned path holds.  However, mustFix=false because the impact does not materialize for any in-tree consumer. worker.py (lines 186-203) uses the returned path only for a log line, then loads via partitioner.load_layer_subset; partitioner._try_load_selective (lines 339-356) re-resolves each shard via hf_hub_download into the HF cache and never reads from the returned layer_subdir. So no consumer loads weights from the returned directory — they all load from the HF shared cache, which download_layer_subset does correctly populate. The manifest also records shard_files for recovery. It is a misleading API/docstring contract worth correcting (doc fix), not an active release-blocking Critical/High functional bug. Minor secondary claim is overstated: hf_hub_download performs blob/ETag consistency checks in its cache layer, so "never integrity-verified" is not literally accurate.

---

### F-055 — [High] 🔒 verified real DiffusionPipeline.load combines enable_model_cpu_offload with nn.DataParallel — the multi-GPU path is broken

`src\distllm\core\hydra_diffusion.py:41` · zone=`core-gen-rag` · category=`bug`

- **Summary:** In load(), when num_gpus > 1 the code calls `pipe.enable_model_cpu_offload()` and then wraps `pipe.unet` in `torch.nn.DataParallel` before `pipe.to('cuda')`. DataParallel requires the module to live on a single CUDA device, while CPU offload registers hooks that move params to the CPU/disk at runtime — the two mechanisms are mutually incompatible; the combination either raises at forward time or silently runs the unet on one device.
- **Evidence (verbatim):**
```
pipe.enable_model_cpu_offload() pipe.unet = torch.nn.DataParallel(pipe.unet, device_ids=list(range(num_gpus))) pipe = pipe.to("cuda")
```
- **Impact:** The advertised distributed/multi-GPU image generation path (num_gpus>1) does not provide real pipeline parallelism and can fail at runtime; the feature is effectively single-GPU only.
- **Recommendation:** Choose one strategy: use `accelerate`'s device_map / model_parallel for multi-GPU offload with a real partitioning of layer tensors, or use DataParallel WITHOUT enable_model_cpu_offload; remove the incompatible combination. Add a load() test asserting the multi-GPU path with mocked pipe.
- **Verdict (real):** mustFix=False. Verified by reading the file: lines 39-42 literally combine pipe.enable_model_cpu_offload() with pipe.unet = torch.nn.DataParallel(...) then pipe.to("cuda"). The incompatibility claim is technically correct — accelerate CPU-offload hooks move params to CPU/disk at runtime while DataParallel requires the wrapped module's params on a single source CUDA device for replicate/scatter, and DataParallel is deprecated; the combination is mutually incompatible and would fault/silently mis-run at forward time. However, mustFix=false because this module is dead/unwired: grep shows hydra_diffusion/HydraOrchestrator/DiffusionPipeline/VideoPipeline/ComfyUIDistributed are imported nowhere in src (not even core/__init__.py), only referenced in docs, _filelist.txt, and dev test-coverage JSONs/workflow defs. No production path reaches the num_gpus>1 branch (default is 1), so it is not load-bearing for release. Worth fixing if the multi-GPU diffusion path is ever wired up, but not a pre-release blocker as it stands.

---

### F-056 — [High] 🔒 verified real NeuralBanditRouter crashes on instantiation: missing `import torch`

`src\distllm\core\learning_router.py:525` · zone=`core-router-sched` · category=`bug`

- **Summary:** learning_router.py defines NeuralBanditRouter which uses torch.device, torch.nn, torch.optim, torch.tensor, torch.no_grad, etc., but the module never imports torch (imports are only hashlib, json, math, threading, dataclass, Path, TYPE_CHECKING). Any instantiation raises NameError: name 'torch' is not defined. The class is dead-on-arrival.
- **Evidence (verbatim):**
```
__init__: `self._device = torch.device(device)` then `self._net = torch.nn.Sequential(torch.nn.Linear(...), torch.nn.ReLU(), ...)`, `torch.optim.Adam(...)`; _encode: `x = torch.tensor(...)`; no `import torch` at module top (top imports are hashlib/json/math/threading/dataclass/Path).
```
- **Impact:** The neural contextual-bandit routing path is unusable; anyone enabling it gets an immediate crash rather than a routing decision.
- **Effort:** 0.25 hours
- **Reliability:** `lr = NeuralBanditRouter(base, models=[...])` → NameError at `torch.device(device)`. Grep of the module shows all uses of torch but zero import statements.
- **Recommendation:** Add `import torch` at the top of learning_router.py (or lazily import inside __init__/route to keep the feature-hashing LearningRouter torch-free). Add a smoke test that instantiates NeuralBanditRouter on CPU. Note this only manifests on instantiation, so it is a latent crash for anyone wiring the neural bandit.
- **Verdict (real):** mustFix=False. CONFIRMED REAL: learning_router.py imports only hashlib/json/math/threading/dataclass/Path/TYPE_CHECKING/loguru (lines 27-38); a repo-wide grep for `import torch` never matches this file. Yet NeuralBanditRouter (line 489) calls torch.device (line 525), torch.nn.Sequential/L1 near/ReLU (529-533), torch.optim.Adam (535), torch.tensor (554), torch.no_grad, torch.randn_like, torch.stack, torch.cat, torch.nn.functional every operation. The only in-method import is `import random as _random` (line 624). Any `NeuralBanditRouter(...)` raises NameError: name 'torch' is not defined — the "dead-on-arrival" characterization is accurate. However mustFix=false: the class is unwired dead code with zero runtime impact. test_learning_router.py tests only LearningRouter/RewardSignal/_ArmStats/_feature_hash (no torch, no NeuralBanditRouter instantiation), and a whole-repo grep for NeuralBanditRouter matches only the source file, audit_findings.json, and two docs files — no production factory, registry, or caller constructs it. It is a real latent bug (cheap to fix, but the module's docstring explicitly says "No external ML dependencies", so the right fix is arguably to delete the torch-based class), but it does not crash anything at release because nothing invokes it.

---

## Refuted by verification (3)

### F-057 — [Critical] ❌ REFUTED DedupMiddleware runs BEFORE AuthMiddleware and returns a synthetic cached response on dedup hit, bypassing authentication, quota and rate-limiting

`src/distllm/api/dedup.py:142` · zone=`api-gateway` · category=`security`

- **Summary:** In server.py the middleware stack executes in reverse registration order (line 526-527: 'FastAPI executes middleware in reverse order of registration'). AuthMiddleware is registered at line 568, but DedupMiddleware is registered later at line 583, so Dedup runs OUTSIDE (before) Auth. On a cache hit or an in-flight dedup wait, DedupMiddleware returns a synthetic `Response(content=cached)` directly (dedup.py 139-147) and never calls AuthMiddleware. An unauthenticated attacker who replays the exact JSON body of a prior authenticated /v1/chat/completions request within the 3600s TTL receives the same generated inference output with HTTP 200 — no API key, no quota accounting, no rate limiting. It also returns the identical response to ANY tenant that submits the same body (body-only fingerprinting, dedup.py 35-36), leaking a cached output across tenants.
- **Evidence (verbatim):**
```
cached = _cache.lookup(fp) (139); if cached is not None: return Response(content=cached, media_type="application/json") (140-142)
```
- **Impact:** Unauthenticated access to cached LLM outputs; free compute and quota/rate-limit evasion; cross-tenant cached-output disclosure. CVSS ~9.1.
- **Effort:** 2-4 hours
- **Reliability:** Repro: (1) admin/key with valid API key POSTs body X to /v1/chat/completions (non-stream). (2) Attacker with NO Authorization header POSTs the identical body X within 3600s. DedupMiddleware runs before AuthMiddleware (reverse registration order: Dedup registered at server.py:583 after Auth at 568). lookup() hits the cache and returns the stored output as HTTP 200 without ever invoking AuthMiddleware's Bearer check; AuthMiddleware only runs via call_next which is bypassed. Unlike the route, which is gated by AuthMiddleware, this synthetic response skips auth entirely.
- **Recommendation:** Move DedupMiddleware registration so it runs AFTER AuthMiddleware (register it BEFORE Auth in server.py), and/or namespace the dedup cache key by authenticated identity (api_key_id + role) so cache hits are only served to the owning tenant. Also make dedup hits flow through the normal post-auth accounting (quota/rate-limit) rather than returning a standalone Response. Add a regression test: unauthenticated replay of a cached body must return 401.
- **Verdict (refuted):** mustFix=False. Verified by reading source. DedupMiddleware (registered L583) genuinely runs outside AuthMiddleware (L568) and returns synthetic cached responses (dedup.py L140-142, L144-147) without call_next, so it skips the inner Starlette auth/rate-limit middlewares. But the finding overlooks PluginHookMiddleware (L906), which is registered after Dedup and therefore runs OUTSIDE it, dispatching AuthPlugin.on_request before Dedup is reached. AuthPlugin's _enforce_rbac returns a 401 rejection whenever api_key_role is empty, and PluginHook reads api_key_role from request.state before the inner AuthMiddleware populates it — so unauthenticated requests are rejected at the plugin layer first. The core claim (auth bypass to cached inference for unauthenticated attackers) is therefore not genuine. Only authenticated cross-tenant response-serving and skipped inner quota/rate-limit accounting survive, which are lower-severity metering/privacy concerns, not an authentication bypass requiring a pre-release Critical/High fix as framed. Given the adversarial-review instruction to be skeptical and default mustFix=false on uncertainty, I mark it not real / not must-fix with corrected detail.
  - *Correction:* The middleware ordering is real (DedupMiddleware at L583 runs OUTSIDE AuthMiddleware at L568, so on a cache hit the inner AuthMiddleware/RequestRateLimitMiddleware are skipped), and the body-only fingerprinting (dedup.py L35-36) is accurate. However, PluginHookMiddleware is registered even later (L906) so it runs OUTSIDE DedupMiddleware, and AuthPlugin.on_request (auth_plugin.py L337-368, _enforce_rbac L421-457) rejects any request whose api_key_role is empty with HTTP 401 via the _reject contract. PluginHook reads request.state.api_key_role before AuthMiddleware sets it, so an unauthenticated (JWT-less/API-key-less) request is rejected at the outer plugin layer BEFORE DedupMiddleware's cache-hit short-circuit can serve a synthetic response. The claimed 'unauthenticated attacker gets cached inference, no API key' does not hold. Residual lower-severity issues: an AUTHENTICATED POST-capable tenant can hit the body-only dedup cache and receive another tenant's exact cached bytes (cross-tenant response serving), and per-request quota/RequestRateLimit accounting on cache hits is bypassed (though plugin-level RateLimitPlugin still runs at the outer PluginHook layer).

---

### F-058 — [High] ❌ REFUTED QUIC transport corrupts any message larger than one UDP datagram

`src/distllm/dist/p2p/quic_transport.py:155` · zone=`dist-net` · category=`bug`

- **Summary:** Each StreamDataReceived event is parsed as a self-contained [priority][len][payload] frame, but aioquic delivers stream data as arbitrary per-packet chunks (max_datagram_size is set to 1200). For any payload over ~1200 bytes, the first event's payload is truncated to whatever this packet carried, the remainder arrives in later events and is mis-parsed as a new header, and event.end_of_stream/offset are ignored. Result: every gossip/KV message > ~1.2KB sent via the 'preferred' QUIC transport is corrupted/truncated at the receiver.
- **Evidence (verbatim):**
```
def _handle_stream_data(self, event): data = event.data; if len(data) < _HEADER_SIZE: ...; priority_byte, payload_len = struct.unpack(_MESSAGE_HEADER_FMT, data[:_HEADER_SIZE]) ... payload = data[_HEADER_SIZE : _HEADER_SIZE + payload_len]; self._recv_queue.put_nowait((int(priority), payload))
```
- **Impact:** The preferred QUIC transport silently corrupts all but the smallest gossip/cache messages, producing truncated cache metadata and surface-level correctness failures in WAN p2p.
- **Reliability:** connect() two endpoints, send(StreamPriority.DATA, b'A'*5000); receive: first returned payload is cut at ~1200 bytes and the tail arrives as a garbage 'frame'.
- **Recommendation:** Buffer per-stream state (partition/offset) and only assemble payload once end_of_stream is seen; read event.offset to tolerate fragmentation. Alternatively emit one logical message per stream id and concatenate until end_of_stream.
- **Verdict (refuted):** mustFix=False. The parsing defect is genuinely present in the code as described — both `_handle_stream_data` (quic_transport.py lines 155-173) and the parallel `_put_to_global` (lines 395-405) treat each aioquic `StreamDataReceived.data` as a self-contained [priority][len][payload] frame, read the header from the chunk start, and slice payload only from that single event's bytes, ignoring event.offset/end_stream with no per-stream reassembly buffer. For any message larger than one QUIC datagram (~1150 bytes of stream payload given max_datagram_size=1200), aioquic delivers per-packet chunks, so the parser would truncate the first chunk and mis-parse the remainder. The mechanism is real. However, it is NOT load-bearing/reachable: `src/distllm/dist/p2p/quic_transport.py` is confirmed dead code. Its only production import is transport.py line 35 for `get_optimal_transport()`, which is never actually called in any runtime gossip/KV path (only docstring references at transport.py 378-388 and quic_transport.py 23-31). The real `GossipTransport` (transport.py 202-357) sends gossip/KV messages over HTTP via httpx, not QUIC. WAN uses a DIFFERENT module (`src/distllm/dist/quic_transport.py` → QuicTransportClient, used in wide_area.py 94-96) which has proper framing and is what the tests exercise. A prior consolidation decision already flagged this module as dead code. So the concrete impact claim — "every gossip/KV message > ~1.2KB sent via the 'preferred' QUIC transport is corrupted" — does not hold in production; no live traffic flows through this parser. Hence not a release-blocking Critical/High.
  - *Correction:* The code-defect description is accurate, but the impact framing is wrong: this p2p QUIC transport (p2p/quic_transport.py) is NOT the mechanism used to send gossip/KV messages. Gossip uses HTTP (GossipTransport in transport.py), WAN uses a separate dist/quic_transport.py QuicTransportClient; get_optimal_transport() returning the QUIC class is never called in a runtime path (docstring-only references). The bug is real dead-code but not reachable/load-bearing. If this module is ever wired in, it must be rewritten with per-stream buffering (keyed by stream_id, honoring event.offset and end_stream) before use. Correct fix scope: consolidate/delete the dead p2p QUIC module rather than classify as an active data-corruption release blocker.

---

### F-059 — [High] ❌ REFUTED Test-infra fake-package shim whitelists a stale symbol list, breaking imports of real config.settings names

`tests/_import_helper.py:56` · zone=`tooling-tests` · category=`bug`

- **Summary:** tests/_import_helper.py fakes package `distllm.config` (line 56) and only exposes a hardcoded whitelist of names (e.g. just `StructuredOutputConfig`, line 301), loading settings.py directly. Any test importing `ModelSettings`, `CachePersistenceSettings`, `ChatRouterSettings`, or `RebalancerSettings` fails at collection with `ImportError: cannot import name 'X' from 'distllm.config.settings' (unknown location)` - "unknown location" is the namespace-package signature of the fake masking the real module. The real src/distllm/config/settings.py DOES export all four names (verified), so this is purely shim drift, not missing product code.
- **Evidence (verbatim):**
```
"distllm.config",  # __init__.py eager-imports from settings causing circular issues ... ("config", ["StructuredOutputConfig"])
```
- **Impact:** At least 9 test modules fail collection solely because the shim's whitelist is not kept in sync with settings.py's public API. The suite tests fake modules, not the real package import paths, so genuine `__init__.py` ITs/circular-import bugs are hidden and can resurface in production.
- **Effort:** 1-2 days
- **Reliability:** Verified: direct `PYTHONPATH=src` import of settings.py succeeds with ModelSettings, but collection via pytest (which loads conftest/_import_helper) fails with '(unknown location)'.
- **Recommendation:** Stop faking `distllm.config`/`distllm.core`. Fix the real circular-import chain in `src/distllm/__init__.py`'s lazy `__getattr__` (it already exists) so `from distllm.config.settings import X` works natively, then delete the fake/whitelist entries. At minimum, whitelist must `from distllm.config.settings import *` instead of a curated list so new public names stop breaking.
- **Verdict (refuted):** mustFix=False. Empirically disproven. After bootstrapping via tests/_import_helper.py, all four claimed-broken names — ModelSettings, CachePersistenceSettings, ChatRouterSettings, RebalancerSettings — import cleanly from distllm.config.settings (hasattr=True for each), and the module resolves to the real file at src/distllm/config/settings.py, not an "unknown location" stub. The claim misreads the code: (1) the "whitelist" at line 301 ("config", ["StructuredOutputConfig"]) belongs to the distllm.core.structured_output workaround (copying symbols onto that fake package), NOT to distllm.config; (2) the config workaround (lines 186-196) runs spec_from_file_location(...).exec_module() on the FULL real settings.py, which executes settings.py's top-of-file re-exports from _model/_network/_cache/_parallelism, so all four names populate the namespace. No whitelist restricts distllm.config.settings. The discriminator between real settings.py (which does need that __all__ list) and the structured_output copy-loop was conflated. isReal=false, mustFix=false.

---
