---
tags:
  - audit
  - exhaustive
date: 2026-08-11
---

# Exhaustive Audit 04 — Test Gaps & Code Quality

**← [[Exhaustive Audit 2026-08-11]]**

All findings in category `test_gap, code_quality` (Medium/Low and non-verified severities).

**30 findings** — High: 2 · Medium: 12 · Low: 16

---

### F-158 — [High] GPU benchmark regression gate is dead: `runner.environment` is not a valid GitHub context

`.github/workflows/ci.yml:202` · zone=`tooling-tests` · category=`test_gap`

- **Summary:** ci.yml benchmark job gates the REAL GPU benchmark behind `if: ${{ runner.environment == 'gpu' }}` (line 202) and the CPU fallback behind `!= 'gpu'` (lines 213, 218). `runner.environment` is not a defined context key in GitHub Actions, so the GPU branch is ALWAYS false on every runner and the CPU-compatible branch is ALWAYS true. GPU benchmark/regression checking (regression_check --threshold 0.10) silently never runs on PRs; the workflow still reports green.
- **Evidence (verbatim):**
```
if: ${{ runner.environment == 'gpu' }}  # `runner.environment` never resolves on any runner
```
- **Impact:** Regression detection for the actual product (GPU inference) is effectively disabled; only toy CPU benchmarks gate PRs, giving false confidence. This is a high-severity gap for the audit's CI matrix breadth.
- **Effort:** 2-4 hours
- **Reliability:** Confirmed by `grep -rn runner.environment .github/`: only referenced in ci.yml lines 202/213/218, no runner defines this context.
- **Recommendation:** Branch on a real GPU runner label: add `runs-on: [self-hosted, linux, gpu]` (or a `gpu` job in the matrix) and use `if: ${{ contains(github.event.pull_request.labels.*.name, 'gpu') }}` only as an extra clamp, never as the sole gate. Ensure the non-GPU path reports clearly that GPU regression was skipped, e.g. set an explicit job success with a notice rather than an empty condition.

---

### F-159 — [High] ~17 regression_high/security/features tests import symbols that were refactored away or never existed

`tests/regression_high/test_m3_api_key_argon2.py:1` · zone=`tooling-tests` · category=`test_gap`

- **Summary:** A cluster of hand-written high-severity/regression/security tests import symbols that do not exist in current src: `reset_registry`/`BackendCostMetrics` (distllm.backends.registry), `hash_api_key` (core.api_key_store), `CORSError` (config._network), `HybridClock` (dist.p2p.gossip), `ENV_REDACTION_ENABLED` (security.log_redaction), `Draft202012Validator` (core.structured_output.validator), `PrivacyAccountant`/`BudgetExhaustedError` (core.differential_privacy), `PowerMeter` (core.advanced_scheduling), `_jitter_backoff_delay` (errors.retry), `_cuda_reduction_callables` (dist.zero_copy), `best_fit_decreasing_partition` (core.auto_partitioner), `reset_app_state_for_testing` (api.api_state), `Histogram` (core.coordinator_metrics), `_validate_terraform_value`/`_is_safe_webhook_url` (core.*), and `VectorStore`/`VectorDBFactory`/`RAGPipeline` (core.vectorstore - src only has `VectorDBInterface`). These are real product/test drift and block collection (part of the 79).
- **Evidence (verbatim):**
```
ImportError: cannot import name 'hash_api_key' from 'distllm.core.api_key_store'
```
- **Impact:** High-severity fixes advertised by these test names get no verification; the suite's perceived coverage overstates reality. Since collection fails, these tests never run at all.
- **Effort:** 2-3 days
- **Reliability:** Exact error list captured by `pytest --co -rE` on current tree.
- **Recommendation:** Triage each: (a) symbols that genuinely exist under a renamed module - update the test import; (b) symbols that are expected product API - implement or mark `xfail(strict=False)` with a tracked issue; (c) aspirational feature tests with no implementation - move to a `tests/aspirational/` dir deselected by default. Add a hard CI rule that `pytest --collect-only` must be clean. Enable a linter check that every `distllm.*` import in tests resolves (e.g. run `check_real_imports.py` already in ci.yml over the whole suite, not just a subset).

---

### F-160 — [Medium] CI matrix breadth is thin vs declared support and chaos/load jobs also collect the broken suite

`.github/workflows/ci.yml:41` · zone=`tooling-tests` · category=`test_gap`

- **Summary:** The main test matrix covers only python 3.10/3.11 on ubuntu+windows; 3.12 and 3.13 are declared as supported in pyproject classifiers but never tested; no macos. The `chaos-tests` job runs `pytest -v -m chaos` over the whole tree (testpaths=tests) and `load-test`/`integration-test` also collect, so all of them are blocked by the same 79 collection errors. The mutation job runs only 2 modules with `|| true`, so it can never gate.
- **Evidence (verbatim):**
```
strategy: matrix: os: [ubuntu-latest, windows-latest]; python-version: ["3.10", "3.11"]  (pyproject.classifiers declare 3.10-3.13)
```
- **Impact:** Python 3.12/3.13 compatibility is unverified (a common source of packaging/dep pain); chaos/load CI jobs are non-functional because collection errors abort them.
- **Effort:** 2-4 hours
- **Reliability:** Matrix read directly from ci.yml; chaos job runs against full suite.
- **Recommendation:** Fix collection (top finding) first; then extend matrix to include 3.12 and 3.13 (and add py313 to lint). Scope chaos/load jobs to their own directories (`pytest tests/chaos/ tests/load/`), remove `|| true` from mutation and make it assert on a survival threshold.

---

### F-161 — [Medium] Testing-strategy depth is unit-heavy; chaos/load/stress/soak and GPU classes are thin or disabled

`Makefile:151` · zone=`tooling-tests` · category=`test_gap`

- **Summary:** 716 test files (10017 collected) skew heavily toward unit tests of core/dist modules. Cluster/integration (19), e2e (17), chaos (8), load (5), stress (1) are thin relative to the unit base. There is no sustained soak/peak or real-cluster performance test gated in CI (GPU one is dead per the runner.environment finding), fuzz runs only 500 iterations in pytest mode (no atheris libFuzzer natively), and backup/restore/failure-injection coverage is minimal.
- **Evidence (verbatim):**
```
Test file counts: core=242, dist=136, regression_high=70, api=45, integration=19, e2e=17, security=22, chaos=8, load=5, stress=1, mutation=1, fuzz=1(pytest-mode)
```
- **Impact:** High-confidence-unit yet low 'does it survive a real cluster' verification; chaos/load classes are effectively smoke tests, not adversarial or capacity coverage.
- **Effort:** 3-5 days
- **Reliability:** Directory counts from find; job definitions from ci.tsv+workflows.
- **Recommendation:** Prioritise depth: (1) restore the GPU perf gate; (2) grow integration/e2e to cover coordinator failover, multi-node recovery, and partition during generation; (3) make chaos tests run real scenarios in a cluster (tests/chaos already has the infrastructure) rather than unit-mocked fault injection; (4) promote fuzz to native atheris for the parsers (grammar, JSON schema) that already exist.

---

### F-162 — [Medium] Streaming broker tests monkeypatch the SDK stream and feed dicts, so the real stream signature/return-shape bugs ship green

`integrations/crewai/tests/test_all.py:117` · zone=`integrations` · category=`test_gap`

- **Summary:** crewai/tests/test_all.py (lines 108-120), langchain/tests/test_chat_advanced.py (175-335), and llamaindex/tests/test_advanced.py / test_basic.py replace `chat_completions_stream` with a MagicMock returning dict chunks and patched async generators. These stubs accept any kwargs (masking the stream=True TypeError) and return dicts (masking the async method's str-yield), so the adapter tests pass while production streaming crashes. llamaindex test_basic.py:171 even needs `create=True` to patch the nonexistent `completions_stream`, proving the method is absent.
- **Evidence (verbatim):**
```
mock_client.chat_completions_stream.return_value = iter(chunks)
```
- **Impact:** The core regression suite gives false confidence for the platform's headline streaming + tool-calling features across its most-used adapters.
- **Effort:** 4-6 hours
- **Recommendation:** Write at least one test per adapter that runs the real DistLLMClient against the real parser (monkeypatch only the transport, not the SDK method), asserting the sync stream returns dicts and the async stream yields the content strings the adapters consume.

---

### F-163 — [Medium] No tests exercise the security/correctness-critical federated merge and cross-model prefix paths

`src/distllm/core/aether_federated.py:481` · zone=`core-training` · category=`test_gap`

- **Summary:** The confirmed defects (cross-model sibling KV reuse, SecureAggregator reconstruction, federated_finetuner round versioning, aether secure-mask no-op, model_hub cache-layout mismatch, download_layer_subset empty dir) each lack regression coverage, so they shipped silently.
- **Evidence (verbatim):**
```
return {k: v.detach().clone() for k, v in self._global_weights.items()}  # no test asserts masking/round-versioning/reconstruction
```
- **Impact:** Without regression coverage the high-severity issues recur on any refactor and have no CI safety net.
- **Effort:** 1-2 days
- **Recommendation:** Add tests: secure aggregate path; 3-node share/reconstruct equals sum of raw grads; cross-model lookup returns only same-prefix entries; download() then is_available() True; download_layer_subset path contains weights.

---

### F-164 — [Medium] DisaggManager.decode is a placeholder that never calls the decode node

`src/distllm/dist/disagg/__init__.py:119` · zone=`dist-net` · category=`test_gap`

- **Summary:** The disaggregated prefill/decode manager returns fabricated tokens `[42]*10` without ever invoking the decode node — it only resolves handle.decode_node_id and releases the node. So DisaggManager, the headline split-Prefill+Decode feature, is not functional for actual inference; any test asserting real output would need to mock gRPC. The architecture (pools, scheduler, handles) is sound but the decode execution path is unwired.
- **Evidence (verbatim):**
```
await asyncio.sleep(0.01)  # placeholder for actual gRPC call; output_tokens = input_ids[-1:] + [42] * 10  # placeholder generation
```
- **Impact:** Disaggregated inference advertised as working but emits placeholder tokens; production adoption of disagg would return garbage output.
- **Reliability:** mgr.decode(input_ids, handle) returns a synthetic token list without any network call to decode_node.address.
- **Recommendation:** Wire decode() to the real node gRPC forward (mirror PipelineOrchestrator._execute_node_grpc / node_client) and assert KV cache handle arrives on the correct node. Add a test that provides a fake decode node stub and asserts it is actually called with the prefilled KV handle.

---

### F-165 — [Medium] exporter.py defines many metrics that nothing ever populates (node_gpu_util, token_latency, cost_per_hour, anomaly, recovery histograms)

`src/distllm/observability/exporter.py:200` · zone=`core-perf-obs` · category=`test_gap`

- **Summary:** DistLLMPrometheusExporter registers ~28 metrics, but populate_gauges() only fills active_nodes, circuit_breaker_state, node_health, draining_nodes, dead_nodes, coordinator gauges. node_gpu_utilization, node_gpu_memory_bytes, node_latency_p50/p99, kv_cache_usage_ratio, cost_per_hour_total, budget_remaining, spot_interruptions_total, anomaly_detected_total and the recovery_* histograms/counters are declared with no caller in src — dashboard panels and alerts for GPU utilization, KV pressure, and per-node latency read zeros. There is no test asserting these are ever set.
- **Evidence (verbatim):**
```
def populate_gauges(self, coordinator=None):     ... self.active_nodes.set(...); self.circuit_breaker_state.labels(...).set(...)         self.node_health.labels(...).set(...); draining_nodes/dead_nodes.set(...)   # node_gpu_util, kv_cache_usage_ratio, cost_per_hour, anomaly, recovery never set
```
- **Impact:** Capacity planning, alerting on KV pressure/GPU utilization, and recovery dashboards are silent (zeros) exactly where ops need them — a metric-completeness GA blocker for 'are promised SLAs/infra metrics truly measured?'.
- **Effort:** 2-4 hours
- **Reliability:** Instantiate DistLLMPrometheusExporter and call generate_latest(); node_gpu_utilization_percent, kv_cache_usage_ratio, cost_per_hour_total, recovery_total all report 0 with no code path able to set them. Confirmed by reading the sole populate_gauges.
- **Recommendation:** Wire populate_gauges to real sources (SystemMonitor for GPU util/mem, kv_cache manager ratio, CostTracker for cost_per_hour) or delete the dead metric definitions; add a completeness test that greps each registered metric name for a producer call, and expose the /metrics handler with this registry.

---

### F-166 — [Medium] Zero/weak coverage for the concurrency-critical paths just audited

`src\distllm\core\request_fingerprinting.py:275` · zone=`core-router-sched` · category=`test_gap`

- **Summary:** The threads the zone claims to be about — cache-aware routing, preemption/restore, dedup result-sharing, load accounting, KV block free, latency windowing — have no (or only self-masking) tests. Notably test_request_fingerprinting.py manually sets fp._in_flight_results, which is precisely the bug in finding 2 and hid it from CI.
- **Evidence (verbatim):**
```
test_request_fingerprinting.py:275: `fp._in_flight_results[fprint] = "cached_result"` — the test writes the field production code never writes, so the missing-writer bug passes. No tests exist for BatchScheduler KV-free-on-completion, DisaggregatedRouter fallback load release, _promote_pending prefill-budget rejection, or RequestLatencyTracker bounded memory.
```
- **Impact:** Concurrency and memory-accounting regressions in the scheduler/routing stack are not caught by CI, and the one test that touches this area actively masks a real bug.
- **Effort:** 1-2 hours
- **Reliability:** Run the existing test suite; the in-flight dedup test only passes because it injects state that production never creates, leaving the timeout behavior unasserted.
- **Recommendation:** Add tests: (1) in-flight dedup returns first request's response via public API (don't touch _in_flight_results); (2) BatchScheduler with a mock paged_attention_mgr asserts free_sequence is called on completion exactly once; (3) DisaggregatedRouter fallback then release keeps the fallback pool's load balanced to zero; (4) _promote_pending honors remain_p; (5) RequestLatencyTracker._completed stays bounded after >max completes.

---

### F-167 — [Medium] No test asserts schema compliance of the token-level 'json_schema' constraint; tests validate only JSON-syntax, masking the 'schema ignored' and 'first-byte mask' defects

`src\distllm\core\structured_output\validator.py:99` · zone=`core-decoding` · category=`test_gap`

- **Summary:** tests/ for constrained decoding (test_structured_output_fsm.py, test_constrained_decoder_fsm.py, test_gbnf.py, test_structured_output_engine.py) assert that constrained tokens are valid JSON and that FSM states advance, but none feed a real schema ({required:[...], enum:[...]}) and assert the generated token stream can only produce schema-conformant objects, nor that a multi-byte (non-first-char) invalid token is blocked. As a result the two production defects above (schema discarded, first-byte-only masking) pass CI. grammar schemas are only exercised through gbnf_to_regex, not the sms-IBM grammar constants.
- **Evidence (verbatim):**
```
if expected is None: return True  # unknown schema keyword types silently pass — no enum/anyOf handling; test files never assert schema compliance
```
- **Impact:** Without these, the central structured-output guarantees (valid JSON syntax + schema conformance) are unverified and regressions (the off-by-one, all-ones mask, first-char-only mask) land silently.
- **Reliability:** test_gap: no existing test in tests/core references schema-driven constraints on the token mask or bool-vs-int type validation; existing tests assert JSON syntax only.
- **Recommendation:** Add adversarial tests: (1) build JSONSchemaConstraint for schema {'type':'object','required':['id'],'properties':{'id':{'type':'integer'}}} and assert the allowed-token mask can never permit '} ' (object close) before 'id':'<int>' is present; (2) feed token `}x` while state=after_value and assert it is blocked; (3) feed `true` against an integer schema via SchemaValidator and assert valid=False; (4) SchemaConstrainedDecoder.json_schema({'enum':['a','b']}) then generate and assert only a/b appear.

---

### F-168 — [Medium] Pinecone query returns metadata even when include_metadata=False, unlike every other provider

`src\distllm\core\vectorstore\providers\pinecone.py:118` · zone=`core-gen-rag` · category=`code_quality`

- **Summary:** _PineconeStore.query builds every hit dict with `'metadata': m.get('metadata', {})` unconditionally. include_metadata=False is forwarded to the SDK (which stops the server sending payloads) but the returned dict still contains a metadata key, whereas qdrant/milvus/weaviate all honor include_metadata and omit/empty it. Inconsistent interface behavior across providers.
- **Evidence (verbatim):**
```
"score": m.get("score", 0.0), "metadata": m.get("metadata", {})) ..., include_metadata=include_metadata,
```
- **Impact:** Providers are not drop-in interchangeable under the VectorDBInterface contract: code that reads hit['metadata'] unconditionally after include_metadata=False behaves differently on pinecone vs the others.
- **Recommendation:** Guard the metadata key behind include_metadata (`if include_metadata: entry['metadata'] = m.get('metadata', {})`), matching the other providers, and add a provider test.

---

### F-169 — [Medium] Critical partition/quant bugs are untested — tests construct fixtures instead of exercising the real paths

`tests/dist/test_validator.py:23` · zone=`dist-partition` · category=`test_gap`

- **Summary:** tests/dist/test_validator.py only instantiates the WhatIfScenario/ValidationReport dataclasses and never calls what_if_slowdown or asserts a throughput change, so the no-op slowdown bug ships green. tests/dist/test_learned_cost.py asserts intermediate_size is in feature_names but never asserts its VALUE, so the always-0 serve feature is missed. There is no test for QuantizedPagedAttentionManager INT4 width, base get_kv_cache num_tokens slicing, or calibration-delta non-zero.
- **Evidence (verbatim):**
```
def test_what_if_scenario(self):             scenario = WhatIfScenario(..., throughput_change_pct=20.0)  # fixture only, real path not exercised
```
- **Impact:** The highest-impact correctness bugs (what-if no-op, learned-cost skew, INT4 packing, KV gather padding) are invisible to CI.
- **Effort:** 2-4 hours
- **Reliability:** grep tests: test_validator builds the dataclass directly; test_learned_cost checks feature_names membership not the value; no test imports paged_attention_quantized.
- **Recommendation:** Add: (1) validator test that inflates estimated_time_ms and asserts throughput drops; (2) learned-cost test that asserts Extract.extract feature[14]==training intermediate_size for a configured model; (3) paged_attention num_tokens slicing test; (4) INT4 storage-width test asserting actual bytes/slot.

---

### F-170 — [Medium] AudioPipeline two-utterance flow, multi-digit-number FSM masking, and structured-output partial-validation are untested (each hides a real bug)

`tests\core\test_media_pipeline.py:152` · zone=`core-gen-rag` · category=`test_gap`

- **Summary:** tests/core/test_media_pipeline.py never drives two sequential utterances (SPEAKING re-entry), so the pipeline deadlock ships green. tests/core/test_structured_output_fsm.py only checks mask shape and 'some tokens allowed'; it never asserts digit continuation in the in_number state, so the multi-digit-number truncation goes uncaught. tests/dist/structured_output/test_repair.py only exercises completed outputs, not per-token partial validation.
- **Evidence (verbatim):**
```
only transitions to IDLE/LISTENING/PROCESSING asserted (lines 150-184); no SPEAKING→next-utterance assertion; fsm test asserts mask.sum()>0 but never that a digit token remains allowed in 'in_number'
```
- **Impact:** The highest-severity functional defects in this zone are invisible to CI.
- **Recommendation:** Add (1) a media test: two speech frames + pointer advancing; assert second utterance is transcribed. (2) an FSM test feeding '{"a": 1' and asserting the allowed-char set includes digits. (3) a streaming/partial repair test feeding incremental JSON and asserting validate_token is not False on a normal in-progress prefix.

---

### F-171 — [Medium] The four factory-registered vectorstore providers (and RAGPipeline retrieve semantics) have zero dedicated tests

`tests\core\vectorstore:1` · zone=`core-gen-rag` · category=`test_gap`

- **Summary:** tests/core/vectorstore/ covers base.py, legacy chroma/pgvector/qdrant_store, and rag_pipeline, but nothing exercises the actual providers registered in VectorDBFactory (pinecone, qdrant, weaviate, milvus in providers/). Consequently the delete-contract bug (qdrant returns UpdateResult.status), the namespace inconsistency, the pinecone include_metadata deviation, and the milvus alias cross-talk are all unguarded. RAG/vector correctness (chunking, retrieval k, empty-query handling) is untested.
- **Evidence (verbatim):**
```
Only test_base.py, test_chroma_store.py, test_pgvector_store.py, test_qdrant_store.py, test_rag_pipeline.py, test_interface.py, test_vectorstore.py exist; no test_pinecone/test_qdrant/test_weaviate/test_milvus for the providers/ modules
```
- **Impact:** The exact seam the audit flagged (provider semantics divergence) is the least-covered surface; regressions ship silently.
- **Recommendation:** Add tests/ that run the shared interface conformance suite against each provider using a fake/local client (monkeypatched SDK), asserting upsert/query/delete return conventions, namespace handling, include_metadata, and empty-query behavior.

---

### F-172 — [Low] Two duplicate/unverified Helm charts and stale security-scan targets

`deploy/helm/distllm-operator/Chart.yaml:1` · zone=`tooling-tests` · category=`test_gap`

- **Summary:** Two separate Helm charts exist (deploy/helm/* used by the Makefile helm-* targets and by ci.yml validate-k8s, and integrations/kubernetes/helm/distllm used by the pre-commit helm-lint hook and CODEOWNERS). This duplication invites drift between charts. The pre-commit hook targets the integrations copy; the Makefile targets deploy/helm. Minor but should be reconciled. Also the Makefile `security` target calls `bash scripts/security_scan.sh`, which is not portable to the win32 dev environment.
- **Evidence (verbatim):**
```
deploy/helm/distllm-operator/Chart.yaml (name: distllm-operator) and integrations/kubernetes/helm/distllm/Chart.yaml (name: distllm) both exist
```
- **Impact:** Chart fixes can be applied to the wrong copy, so lint/CI validates a chart that is not the one deployed; security scanning is unavailable to Windows devs.
- **Effort:** 1-2 hours
- **Reliability:** Confirmed both Chart.yaml files exist with different names.
- **Recommendation:** Pick one canonical chart location (suggest deploy/helm), update the pre-commit hook and CODEOWNERS to it, and add a CI check that both locations are in sync if both must remain. Make the Makefile security/scan targets platform-aware (look for gmake/shell or invoke python for the portable pieces).

---

### F-173 — [Low] Stray Windows 'nul' file and server_middleware.py ghost binary in src/distllm/api indicate leftover artifacts from a deleted/moved module

`src/distllm/api/nul:1` · zone=`api-gateway` · category=`code_quality`

- **Summary:** src/distllm/api contains a 102-byte file literally named 'nul' (a Windows reserved-name artifact, likely from a redirect mishap `> nul`) and a __pycache__/server_middleware.cpython-314.pyc with no corresponding server_middleware.py source — matching the memory note that 'server_middleware unwired' was a dead-code finding. The presence of a compiled bytecode without source means some module referenced server_middleware and was removed, but its cache and possibly imports remain. These artifacts pollute the audit surface and can confuse future audits and packaging.
- **Evidence (verbatim):**
```
api/nul exists as a 102-byte file; __pycache__/server_middleware.cpython-314.pyc exists with no source server_middleware.py
```
- **Impact:** Repository hygiene / audit noise; risk that a stale compiled module is accidentally relied on. No runtime effect to the wired server.
- **Effort:** <1 hour
- **Reliability:** `ls src/distllm/api` shows 'nul' (102 bytes) alongside the Python sources, and the pyc under __pycache__ has no matching .py.
- **Recommendation:** Delete the 'nul' file and strip stray __pycache__/*.pyc without a source file; grep the tree for any remaining `import server_middleware` and remove them; add 'nul'/'con' to .gitignore to prevent recurrence on Windows dev machines.

---

### F-174 — [Low] ws_chat.py WSChatHandler is unwired and, if ever mounted, performs no authentication and no per-user isolation (user_id='default')

`src/distllm/api/ws_chat.py:143` · zone=`api-gateway` · category=`test_gap`

- **Summary:** src/distllm/api/ws_chat.py defines a full multiplexed WebSocket chat handler, but no route registers it — server.py mounts only /ws and /ws/metrics (which authenticate via Authorization header, server.py 984-1001). WSChatHandler.handle() accepts the socket and immediately runs the message loop with user_id defaulting to 'default' and no token/API-key check (ws_chat.py 129-147). Its _build_prompt concatenates client messages verbatim. If an operator or future change mounts this handler, it is an unauthenticated, multi-session-generation surface with no rate limiting, no ownership, and no prompt-safety hookup (it bypasses AuthMiddleware, CSRF, and content moderation), plus per-connection asyncio task fan-out that is unbounded other than max_concurrent_sessions.
- **Evidence (verbatim):**
```
await self._ws.accept() (143); user_id: str = 'default' (129); no Authorization check before the receive loop (146)
```
- **Impact:** Dead-but-hazardous code: an immediately exploitable unauthenticated compute surface if ever adopted. Current impact none because unmounted.
- **Effort:** 1-2 days
- **Reliability:** Grep for WSChatHandler shows it is referenced only inside ws_chat.py itself; no server route constructs it. Mount @app.websocket('/ws/chat') returning WSChatHandler(websocket) and any client can stream generations without credentials.
- **Recommendation:** Either delete ws_chat.py, or integrate it through AuthMiddleware-compatible header auth (mirror the /ws endpoint's explicit API-key check), constrain message/session sizes, and add it to CSRF/skip-path handling before wiring. Add a test asserting unauthenticated WebSocket is rejected if mounted.

---

### F-175 — [Low] registry _check_health/_get_load instantiate adapters via object.__new__ without __init__ — fragile and masks backends

`src/distllm/backends/registry.py:412` · zone=`backends-config-cloud` · category=`code_quality`

- **Summary:** BackendRegistry._check_health (registry.py:392-416) builds probe = object.__new__(adapter_class) and calls probe.health_check() for subclasses that override health_check. Any override reading instance state set in __init__ (like WebGPUNodeAdapter.health_check reading self._model_loaded) raises AttributeError -> caught -> returns False, silently excluding that backend from selection even when it is perfectly usable. Same pattern in _get_load.
- **Evidence (verbatim):**
```
probe = object.__new__(adapter_class) return bool(probe.health_check())
```
- **Impact:** Health filtering can silently blacklist working backends and is opaque to diagnose; future health_check overrides inherit the landmine.
- **Effort:** 2-3 hours
- **Reliability:** WebGPUNodeAdapter.health_check (webgpu_backend.py:228-229) reads self._model_loaded/self._ready_workers which object.__new__ never sets -> AttributeError path (registry.py:415-416).
- **Recommendation:** Prefer a @classmethod probe_health() on adapters (the code already checks for it) and require it for health-aware selection; stop object.__new__-ing instances. Add a test that a health_check override reading __init__ state does not silently de-select the backend.

---

### F-176 — [Low] ClusterStateStore and coordinator_failover exported/registered but unreachable from coordinator runtime

`src/distllm/core/__init__.py:20` · zone=`core-ops-ha` · category=`code_quality`

- **Summary:** core/__init__.py exports ClusterStateStore, and the module advertises itself as 'coordinator HA'; but the coordinator's HA path is CoordinatorElection->RayFaultTolerance. ClusterStateStore (Redis lock leader election) never runs in the coordinator, nor does CoordinatorFailoverHandler, giving operators three HA entry points where only one is live.
- **Evidence (verbatim):**
```
_register("distllm.core.cluster_state_store", "ClusterNodeState", "ClusterState", "ClusterStateStore")
```
- **Impact:** Public API surface exposes a Redis HA store and a failover handler that do nothing in the real coordinator, so users configuring them get silent no-ops.
- **Effort:** 2-4 hours
- **Reliability:** grep: ClusterStateStore only referenced in __init__.py export and its own module; no coordinator/API call sites. Coordinator HA uses RayFaultTolerance (coordinator_election.py L62-65).
- **Recommendation:** Either wire ClusterStateStore into CoordinatorElection as the persistence backend or remove the public export; same for CoordinatorFailoverHandler. Keep a single documented HA path.

---

### F-177 — [Low] HealthChecker (coordinator_health.py) is dead code — live health lives in the dist-layer HealthManager

`src/distllm/core/coordinator_health.py:18` · zone=`core-ops-ha` · category=`code_quality`

- **Summary:** HealthChecker provides a clean, deadline-bounded sync/async node health dispatch, but it is never instantiated. The coordinator composes HealthManager (imported from dist via health_manager) instead, so HealthChecker's bounded-probe and circuit-breaker integration is unreachable and risks confusion with the version that actually runs.
- **Evidence (verbatim):**
```
class HealthChecker:     """Dispatches health checks across registered nodes."""
```
- **Impact:** Duplicate health-check logic (with differing timeout semantics) drifts from the live path; any bug fix localized to HealthChecker has zero effect.
- **Effort:** 1-2 hours
- **Reliability:** grep: 'HealthChecker' appears only inside coordinator_health.py; Coordinator uses distllm.core.health_manager.HealthManager (coordinator.py L33, L186).
- **Recommendation:** Delete coordinator_health.py or fold its deadline-bounded check_all_async into the real HealthManager; add vulture to CI to catch future one-off dead modules.

---

### F-178 — [Low] neural_partition_optimizer: exploration_beta and Optuna are dead (documented but unused)

`src/distllm/core/neural_partition_optimizer.py:888` · zone=`core-training` · category=`code_quality`

- **Summary:** BayesianOptimizationLoop stores self._exploration_beta and the module detects _OPTUNA_AVAILABLE, but neither is used: acquisition is plain Thompson sampling or cold-start noise, and optuna is imported but unreferenced. Docstrings promise LCB exploration and 'Optuna backed' sampling that do not exist.
- **Evidence (verbatim):**
```
self._exploration_beta = exploration_beta ... # exploration_beta never referenced; _OPTUNA_AVAILABLE never referenced after import
```
- **Impact:** Applied config has no effect and the 'Bayesian via Optuna' claim is misleading; exploration control is inert.
- **Effort:** 2-4 hours
- **Recommendation:** Implement LCB acquisition (scores = gp_mean - beta*gp_std) and gate an Optuna TPE path on _OPTUNA_AVAILABLE, or remove the field/import and correct the docstrings.

---

### F-179 — [Low] ray.py dead statement, unused field, and fire-and-forget KV-cache clear() races

`src/distllm/dist/backends/ray.py:39` · zone=`backends-config-cloud` · category=`code_quality`

- **Summary:** dist/backends/ray.py line 39 evaluates (i == len(self.workers) - 1) and discards the result (likely intended is_last logic that never happens), self._num_gpus is initialized and never read, and clear_kv_cache / clear_all_kv_caches invoke worker.clear_kv_cache.remote(...) without collecting/awaiting the ObjectRefs, so a subsequent inference can read stale KV before the clear completes on a worker.
- **Evidence (verbatim):**
```
is_first = (i == 0) (i == len(self.workers) - 1)
```
- **Impact:** Latent stale-KV correctness bug under Ray when clears race forwards; dead code confusion.
- **Effort:** 1-2 hours
- **Reliability:** Lines 39, 20 unused; lines 76-82 fire .remote() and ignore return refs — no ordering barrier before the next run_pipeline.
- **Recommendation:** Delete the dead is_last expression; implement clear_kv_cache to ray.get() the clear ObjectRefs (or track the future) so clears are ordered before the next forward; drop _num_gpus or use it.

---

### F-180 — [Low] SecureAggregator claims secure aggregation but has no masking/unmasking or malicious-input handling (SecAgg is overstated)

`src/distllm/dist/federated_merge.py:449` · zone=`dist-exec` · category=`test_gap`

- **Summary:** `SecureAggregator` is a plain additive secret-sharing sum. The module docstring concedes 'simplified SecAgg ... WITHOUT the masking/unmasking phase'. Without the mask, two colluding peer share-holders can subtract their own shares to recover a third party's raw gradient (g = self_share + sum(peer_shares) is trivially reconstructible by the recipient), there is no input-length/shape validation on received_shares, and no dropout/timeout handling for Byzantine or crashed peers — a crashed peer's missing shares silently zero a gradient. Fully automated tests plus the 'secure' name overstate the privacy guarantee.
- **Evidence (verbatim):**
```
self_share = grad - running_sum\nself_shares.append(self_share)
```
- **Impact:** Privacy-preserving training marketing is not matched by the protocol: a single colluding peer can recover another node's gradient, defeating the stated guarantee.
- **Reliability:** aggregate_received_shares with received_shares containing a peer's share set reconstructs that peer's gradient exactly since each node's self_share = grad - sum(peer_shares).
- **Recommendation:** Downgrade the claim to 'private aggregation for honest-but-curious' and add adversarial coverage: (1) unit test that an eavesdropping peer cannot reconstruct another's gradient without masking, (2) reject received shares whose count/shape mismatch self_gradients, (3) require a full round of share acknowledgements so missing peers abort rather than corrupt the aggregate.

---

### F-181 — [Low] Divergent config-loading path in cli/run.py bypasses DistLLMSettings validation and precedence

`src\distllm\cli\run.py:39` · zone=`cli` · category=`code_quality`

- **Summary:** run_inference loads config.yaml by raw yaml.safe_load and only reads model.name/model.dtype, diverging from the sanctioned DistLLMSettings.from_yaml/validate_startup precedence chain used elsewhere, so `distllm system run --config` can behave differently from `distllm system api`/`cluster start` and is never validated.
- **Evidence (verbatim):**
```
if config:     with open(config) as f:         cfg = yaml.safe_load(f)     model = cfg.get("model", {}).get("name", model)     dtype = cfg.get("model", {}).get("dtype", dtype)
```
- **Impact:** `distllm system run` silently bypasses the canonical DistLLMSettings validation/precedence chain (env vars, defaults, CLI overrides), so an invalid or partial config.yaml is used with raw .get() defaults and never validated; two different CLI-served run paths can therefore yield different resolved configs.
- **Reliability:** run.py:39-43 loads config.yaml with yaml.safe_load and reads cfg['model']['name']/['dtype'] with bare .get() defaults; by contrast main.py config_validate and the server use DistLLMSettings.validate_startup/from_yaml (config/settings.py:423+). No validation or fallback handling occurs in run.py's path.
- **Recommendation:** Route `system run` config handling through DistLLMSettings.from_yaml(config_path=..., cli_overrides=...) (the same path validate_startup uses) instead of ad-hoc yaml.safe_load + .get() in run_inference, so defaults, env precedence, and type validation are consistent across main.py, cluster start, and the server. Mirror this in the cluster._cluster_start path.

---

### F-182 — [Low] BatchScheduler aging boost cannot actually re-prioritize starved pending requests

`src\distllm\core\batch_scheduler.py:775` · zone=`core-router-sched` · category=`code_quality`

- **Summary:** _promote_pending iterates the pending heap in strict (priority, counter) order and only inspects candidates that a high-priority request has NOT already consumed slots for. _aging_boost and latency boost adjust the effective_priority of an already-dequeued candidate, but never re-sort the candidates among themselves, so a deeply-starved low-priority request that sits behind many higher-priority items is never selected before slots run out. The advertised starvation prevention is largely ineffective against sustained higher-priority load.
- **Evidence (verbatim):**
```
Loop: `while self._pending_heap and len(batch_seqs) + len(rejected) < max_examine: pri, cnt, candidate = heapq.heappop(self._pending_heap) ... effective_pri = self._latency_tracker...  aging = self._aging_boost(candidate)`. effective_pri is only used to accept/reject; there is no reordering of the dequeued set by effective_pri (acceptance is FIFO in heap order: first-fit). _check_starvation() only samples the top 20 heap items (line 419), i.e. exactly the most-likely-to-schedule items, so it does not detect buried victims.
```
- **Impact:** Priority fairness under load is weaker than documented; low-priority requests can be starved arbitrarily long, and the starvation watchdog provides a false sense of detection.
- **Effort:** 1-2 hours
- **Reliability:** Feed 100 high-priority prefill requests, then 1 low-priority one that ages well past _starvation_threshold_s; observe the low-priority request is still not selected until the high-priority heap drains, despite aging. The aging boost only lowers its effective_priority to 0, which is meaningless once slots are consumed by earlier candidates.
- **Recommendation:** Collect the set of examined candidates that fit, then sort by effective_priority (incorporating aging/latency boost) before filling remain_slots, or apply a priority-aging mechanism that periodically promotes aged pending IDs in the heap itself. Also make _check_starvation() sample across priority strata, not just the top of the heap. Add a test: continuous high-priority arrivals must not indefinitely starve a low-priority request.

---

### F-183 — [Low] StructuredOutputConfig.schema_config / resolve_refs is dead configuration, never read anywhere

`src\distllm\core\structured_output\config.py:14` · zone=`core-gen-rag` · category=`code_quality`

- **Summary:** config.py defines `SchemaConfig(resolve_refs=True, allow_additional_properties=False)` and `StructuredOutputConfig.schema_config`, but a grep across src shows neither `resolve_refs` nor `schema_config` is referenced by any consumer (engine/validator ignore it). SchemaValidator performs only shallow type/required/properties checks and never resolves $ref or honors allow_additional_properties, yet structured output is documented as 'schema validation' that pulls the ~16 MB jsonschema dependency for json_schema mode.
- **Evidence (verbatim):**
```
resolve_refs: bool = True allow_additional_properties: bool = False  # never consumed
```
- **Impact:** Config options advertise capability ($ref resolution, additional-property control) that does not exist, and schema validation is shallower than promised; users with $ref-based schemas get wrong validation results.
- **Recommendation:** Either wire schema_config into SchemaValidator (use jsonschema with $ref resolver + additionalProperties from config when jsonschema is available) or delete the dead fields; add a test that a $ref schema validates correctly under resolve_refs=True.

---

### F-184 — [Low] Late dangling `import threading` at module end and Missing/race in routing stats counters

`src\distllm\core\unified_router.py:351` · zone=`core-router-sched` · category=`code_quality`

- **Summary:** unified_router.py imports threading only at line 425, after DisaggregatedRouter is defined (works only because module import runs before instantiation — fragile/confusing). Separately, several routers increment shared stats dicts without locks: CrossCloudRouter.select_provider `self._stats["routes"] += 1`, UnifiedRouter.route `self._stats["routes"] += 1`, CacheAwareRouter.route `self._route_stats[best_node]["routed"] += 1`, and SpeculativePreWarmer.predict_and_warm increments _pre_warms outside the lock. These read-modify-writes race under concurrent routing.
- **Evidence (verbatim):**
```
DisaggregatedRouter.__init__ uses `threading.Lock()`; `import threading` is at line 425 (bottom). CrossCloudRouter line 663: `self._stats["routes"] += 1`; UnifiedRouter line 227: `self._stats["routes"] += 1`; CacheAwareRouter line 74: `self._route_stats[best_node]["routed"] += 1`.
```
- **Impact:** Metrics/reporting and LRU eviction counters can lose updates under concurrent load, and the misplaced import invites a future refactor to break it.
- **Effort:** 0.5 hours
- **Reliability:** Run two threads calling route() concurrently on the same router and observe stats under-count/over-count (lost updates) due to unsynchronized +=. The late import is not a runtime crash but is a latent ordering trap.
- **Recommendation:** Move `import threading` to the module top. Guard stats increments with the existing locking (or use an atomic/Counter) in CrossCloudRouter/UnifiedRouter/CacheAwareRouter route paths, and move SpeculativePreWarmer._pre_warms under self._lock (both read and write).

---

### F-185 — [Low] RAG.retrieve has no empty-query guard and assumes embedder always returns a non-empty list

`src\distllm\core\vectorstore\rag.py:116` · zone=`core-gen-rag` · category=`code_quality`

- **Summary:** rag.py line 116 does `query_vector = self._embedder([query])[0]` with no check on `query` or on the embedder's return length; an empty string or an embedder returning [] raises IndexError, and there is no metadata_filter/top_k bound at the pipeline layer. Also `_embedder([query])[0]` implicitly assumes single-vector output without validation.
- **Evidence (verbatim):**
```
if self._store is None: raise RuntimeError(...) query_vector = self._embedder([query])[0]
```
- **Impact:** Empty/normalized queries crash rather than returning an informative error; embedder contract violations surface as low-level IndexError instead of a clear message.
- **Recommendation:** Guard `if not query: raise ValueError('empty query')` and validate `len(self._embedder([query])) == 1` (or take the last-safe index) with a descriptive error; add a rag_pipeline test for empty query.

---

### F-186 — [Low] ParallelEncoderPipeline uses executor.shutdown(wait=False) and can leak threads on early-exit paths

`src\distllm\core\voyager_multimodal.py:1009` · zone=`core-gen-rag` · category=`code_quality`

- **Summary:** Execute() creates a ThreadPoolExecutor, drains futures via as_completed, then calls shutdown(wait=False). This is only safe because every future is consumed before shutdown; any future refactor that returns early between submit and shutdown leaks worker threads. No context-manager use. When all non-text encoders fail, _safe_encode still returns a row so futures isn't empty, but the pattern is fragile.
- **Evidence (verbatim):**
```
executor.shutdown(wait=False) # ... text encoded synchronously after
```
- **Impact:** Latent thread-resource leak under partial/exceptional execution; with up to max_workers=4 threads leaked per call in a reused Voyager, contributor GPU capacity degrades over time.
- **Recommendation:** Use `with concurrent.futures.ThreadPoolExecutor(...) as executor:` (exits wait/join and shuts down cleanly) and collect results inside the with-block.

---

### F-187 — [Low] Voyager._generate tautological ternary: both branches return the same value, and TEXT_ONLY echo path is counted as real generation

`src\distllm\core\voyager_multimodal.py:1286` · zone=`core-gen-rag` · category=`code_quality`

- **Summary:** voyager_multimodal.py line 1286 `return prompt if plan.route_type == RouteType.TEXT_ONLY else prompt` — both arms are identical (dead conditional). Because there is no default generator, a text-only request with no registered generation_fn returns the input text itself (echo), yet process() still records it as a real request with tokens_generated = word count (line 1179), inflating stats.
- **Evidence (verbatim):**
```
return prompt if plan.route_type == RouteType.TEXT_ONLY else prompt tokens = max(1, len(response_text.split()))
```
- **Impact:** Misleading telemetry: echo responses are counted as successful multimodal generations with 1+ tokens; the dead ternary obscures control flow.
- **Recommendation:** Collapse the ternary to `return prompt`, and when no generation_fn is registered for a TEXT_ONLY request either return a non-success VoyagerResponse (valid=False) or skip _record_request so echoes are not counted as generations; use a real tokenizer for tokens_generated.

---
