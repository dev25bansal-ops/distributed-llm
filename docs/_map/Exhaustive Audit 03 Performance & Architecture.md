---
tags:
  - audit
  - exhaustive
date: 2026-08-11
---

# Exhaustive Audit 03 — Performance & Architecture

**← [[Exhaustive Audit 2026-08-11]]**

All findings in category `performance, architecture` (Medium/Low and non-verified severities).

**11 findings** — Critical: 1 · High: 3 · Medium: 6 · Low: 1

---

### F-147 — [Critical] Single biggest adoption risk: measured single-node throughput is uncompetitive AND the latency tables are empty, so the 'pooling = faster' claim is weakly supported

`docs/PERFORMANCE_COMPARISON.md:21` · zone=`strategic` · category=`performance`

- **Summary:** The natural first-run is 'install on my one machine, then scale up.' On that on-ramp DistLLM is measured at 92 tok/s for Llama-3.1-8B on an RTX 4090 — far below what Ollama/vLLM deliver on the same card (vLLM typically 150-250+ tok/s), and the doc itself says the TTFT and ITL latency tables are still filled with em-dashes. Two-node scaling is 153 tok/s = only 1.66x for 2x the GPUs (83% efficiency) — which undermines 'pipeline parallelism = faster generation' in the README's own Why table. This is a trust-and-performance composite risk: an evaluator either (a) tries one device, loses to Ollama, and never reaches the pooling step, or (b) reads an evaluation and finds no latency numbers at all.
- **Evidence (verbatim):**
```
DistLLM's measured speed on consumer hardware (92 tok/s single GPU, 153 tok/s across two 1 GbE nodes) ... the TTFT and ITL tables there are filled with `—`
```
- **Impact:** Directly gates adoption (first-run loses to Ollama), gate checks (enterprise evaluation rejects missing latency), and credibility; it is the difference between 'show HN' momentum and a stall.
- **Effort:** 2-3 weeks (benchmark harness + published report)
- **Reliability:** PERFORMANCE_COMPARISON.md lines 11-15: TTFT/ITL table cells are '—'; BENCHMARKS.md contains 16 placeholder '—' cells; TASK-015 (benchmark blog) is unstarted in TASKS.md.
- **Recommendation:** Before any paid tier exists, do the TASK-015 benchmark sprint as a blocker: fill TTFT/ITL per config on the exact claim matrix, publish honest single-node numbers, and reframe the benefit from 'faster' to 'runs models you can't otherwise run + data stays local + near-zero marginal cost.' If single-node is uncompetitive, say so and win on the 70B-on-consumer-hardware axis — do not let an empty latency table stand in a competitive document.

---

### F-148 — [High] Two parallel, already-diverged Python SDKs (distllm.sdk vs distllm_sdk) duplicate all internals

`sdk/src/distllm_sdk/__init__.py:1` · zone=`sdk-arch` · category=`architecture`

- **Summary:** Repo ships two Python client stacks: src/distllm/sdk (distllm.sdk) and sdk/src/distllm_sdk (distllm_sdk 1.0.0). They duplicate client/transport/streaming/circuit_breaker/types; RetryConfig is defined twice (client.py:52 and transport.py:17); the same streaming bug appears in both, proving divergence.
- **Evidence (verbatim):**
```
both expose 'DistLLMClient' (from distllm.sdk vs distllm_sdk); duplicate RetryConfig at client.py:52 and transport.py:17; same sync-stream bug in both
```
- **Impact:** Two sources of truth; a fix in one may not land in the other; pip users cannot tell which is canonical.
- **Effort:** 0.5-1 day
- **Reliability:** Both packages exist with identical module sets and both carry the identical sync-vs-async streaming bug.
- **Recommendation:** Make src/distllm/sdk the single implementation and have distllm_sdk thin-reexport it; delete duplicate RetryConfig (import from transport); run one CI parity job.

---

### F-149 — [High] Two divergent SSO implementations under src/distllm/api: sso_auth.py (wired) vs auth/ package (unwired, with security defects)

`src/distllm/api/auth/oidc.py:114` · zone=`api-gateway` · category=`architecture`

- **Summary:** src/distllm/api/sso_auth.py is used by sso_middleware.py (wired into server.py via setup_sso at server.py:954). A second, near-duplicate auth stack lives in src/distllm/api/auth/ (oidc.py, oauth2.py, saml.py, __init__.py) that is imported nowhere in the codebase except its own docstring (no `from distllm.api.auth import` in src/). The two have already diverged on security-critical details: auth/oidc.py stores only an expiry timestamp in _nonce_store (line 114: `self._nonce_store[state] = time.time() + self._nonce_ttl`), so the issued nonce can never be re-verified and OIDC nonce replay protection is effectively dead; auth/__init__.py calls hashlib.sha256 (line 156) without importing hashlib, so validate_token would raise NameError. Any future 'fix' applied to one fork silently leaves the other broken or misaligned.
- **Evidence (verbatim):**
```
self._nonce_store[state] = time.time() + self._nonce_ttl (114) — stores only a float expiry, never the nonce value; _nonce_store is later never validated against the token nonce
```
- **Impact:** Maintenance split-risk on an auth boundary; latent, untestable bugs (dead OIDC nonce, NameError) in a to-be-wired replacement; higher long-term risk of a security regression landing in the 'other' fork.
- **Effort:** 1-2 days
- **Reliability:** Compare auth/oidc.py get_login_url: issues nonce into the URL but records only a timestamp, so handle_callback's `received_nonce != expected_nonce` check (line 187) can never be satisfied against a stored value — nonce verification is inert. Also `from . import ...` usage of auth/__init__.validate_token hits hashlib.sha256 (line 156) with hashlib not imported in that module (only asyncio/os/threading/time), → NameError. Grep shows no src/ importer of distllm.api.auth.
- **Recommendation:** Consolidate on the wired sso_auth.py/sso_middleware.py path: delete the auth/ package (or, if the richer PKCE handling there is wanted, promote it into the live path and remove sso_auth.py). Add an import-linter test that enforces exactly one SSO handler entry point so the forks cannot silently diverge again. Fix the nonce store to retain the nonce value and import hashlib before deleting.
- **Strategic value:** Consolidation is a hygiene prerequisite before adding multi-provider/OAuth-concentrator features; shipping two divergent IdP stacks is an anti-pattern that will surface as an auth outage or bypass once the unwired fork is mounted.

---

### F-150 — [High] Partition/quantization logic is fragmented across four DP solvers plus un-integrated learned-cost and core/models duplicates

`src/distllm/dist/partition/optimizer.py:102` · zone=`dist-partition` · category=`architecture`

- **Summary:** Within the zone there are three independent DP solvers that each re-implement the partition DP — PartitionOptimizer (optimizer.py, incl. beam search), QuantAwarePartitionSolver (quant_partition.py), and ParetoPartitionOptimizer (pareto_optimizer.py). LearnedCostModel is never referenced from partitioner.py/optimizer.py, so it is dead code in the solve path (config flag exists but the wrapper is never wired in). Cross-zone, core/auto_partitioner.py, core/neural_partition_optimizer.py (its own NN+Bayesian 'learned cost'), models/partition_planner.py, and core/quantization_selector.QuantizationSelector provide overlapping, mutually inconsistent implementations.
- **Evidence (verbatim):**
```
def solve(self, num_layers): ...  # PartitionOptimizer DP -- plus quant_partition.QuantAwarePartitionSolver.solve, pareto_optimizer.ParetoPartitionOptimizer.solve; LearnedCostModel never imported by partitioner/optimizer
```
- **Impact:** No single source of truth: the quant-aware solve (quant_partition) and the quant-integrated PartitionOptimizer produce different plans, and users choosing different entry points get divergent results; the learned-cost system is inert despite being configurable.
- **Effort:** 2-5 days
- **Reliability:** grep 'def solve(' returns optimizer.py, pareto_optimizer.py, quant_partition.py; grep for LearnedCostModel in partitioner/optimizer returns nothing; core/quantization_selector + core/neural_partition_optimizer + models/partition_planner provide overlapping logic.
- **Recommendation:** Adopt a single solver core (recommend QuantAwarePartitionSolver for quant, extend to Pareto), route learned cost through PartitionCostModel so the flag is honored, and deprecate core/quantization_selector apps above the dist tuner; document which entry point is canonical.

---

### F-151 — [Medium] Windows asyncio flakiness under pytest-timeout=60: IOCP event loop hangs and trips 60s abort

`pytest.ini:3` · zone=`tooling-tests` · category=`performance`

- **Summary:** broad_run.log (a full-suite sweep) shows many async tests hanging on Windows where asyncio uses the ProactorEventLoop backed by IOCP (GetQueuedCompletionStatus). Combined with the global `timeout = 60` in pytest.ini, long/first-run async tests get killed and appear as F/error clusters - real environmental flakiness, not product bugs. pytest-timeout on Windows uses the thread method by default, which cannot reliably interrupt a blocking Proactor call, so the abort can leave the worker unusable.
- **Evidence (verbatim):**
```
(broad_run.log tail) asyncio\windows_events.py ... GetQueuedCompletionStatus ...  '++++++++++ Timeout ++++++++++  EXIT=1'
```
- **Impact:** Windows CI (which is in the matrix: os: [ubuntu, windows]) produces noisy, non-reproducible failures; threaded timeout can't cleanly abort IOCP waits, worsening the flakiness the logs exhibit.
- **Effort:** 2-4 hours
- **Reliability:** Erroneous timeout traceback captured at tail of broad_run.log.
- **Recommendation:** On windows use `--timeout-method=thread` deliberately and raise per-marker timeouts for integration/chaos (`slow`, `integration` markers), or run async-heavy tests on the ubuntu leg only. Consider `asyncio_mode=auto` already set; add a `--timeout` override in the windows test step and separate slow from default. Set `timeout_method` via pyproject to avoid the abort leaving hung workers.

---

### F-152 — [Medium] Webhook delivery performs blocking sync httpx.post on the event loop while an AsyncClient is created but never used

`src/distllm/api/webhooks/delivery.py:270` · zone=`api-gateway` · category=`performance`

- **Summary:** In webhooks/delivery.py, WebhookManager.__init__ constructs `self._client = httpx.AsyncClient(timeout=30.0)` (152), but _submit_delivery (270) calls the SYNCHRONOUS httpx.post(url, ..., timeout=30.0) directly inside what is dispatched from async contexts. Each delivery can block the event loop for up to 30s per webhook, and dispatch() fire-and-forgets one delivery per matched webhook (242-250), so a single event can stall the server for N webhooks x 30s in sequence. The created AsyncClient is dead code. Additionally, register() accepts arbitrary urls with no SSRF/internal-address allowlist, so any low-privilege registration enables SSRF to internal services via retry (1,2,4,8,16s) fan-out.
- **Evidence (verbatim):**
```
self._client = httpx.AsyncClient(timeout=30.0) (152) never used; resp = httpx.post(reg.url, content=body, headers=headers, timeout=30.0) (270-275)
```
- **Impact:** Event-loop starvation / availability under webhook load; undocumented SSRF primitive if webhook registration is exposed.
- **Effort:** 2-4 hours
- **Reliability:** Call WebhookManager.dispatch('job.completed', {...}) from an async route with a slow/unresponsive reg.url. _submit_delivery runs sync httpx.post on the current thread; with the event loop on that same thread, all other requests stall until the 30s timeout. The async client (152) is unused, showing the intended non-blocking path was abandoned.
- **Recommendation:** Use the existing self._client (await self._client.post(...)) for all deliveries and retries so I/O yields to the loop; run in a worker if strict order is required. Add a URL scheme/private-IP allowlist to register() (reusing _reject_private_address-style checks) and per-webhook concurrency bounds.

---

### F-153 — [Medium] Cross-backend forward(input_ids=...) returns token-index tensor for some backends but vocab logits for others

`src/distllm/backends/vllm_backend.py:231` · zone=`backends-config-cloud` · category=`architecture`

- **Summary:** The BackendAdapter.forward contract (protocol.py) declares output_tensor is hidden states/logits, but in single-node mode vLLM, llama.cpp and TensorRT's _forward_with_input_ids return torch.tensor([[next_token_id]]) (a (1,1) token index), while PyTorch/MLX/ONNX/NIM return vocab-dimension logits. LlamacppNodeAdapter even computes next_token = output["choices"][0]["text"] (line 131) then never uses it. Callers that branch on logits vs token idx cannot rely on a stable output type.
- **Evidence (verbatim):**
```
next_token = torch.tensor([[token_ids[0]]]) if token_ids else torch.tensor([[0]]) ... return next_token, []
```
- **Impact:** Backend selection between vLLM/llama.cpp (token index) and PyTorch/NIM (logits) yields different output semantics for the same logical operation; subtle integration bugs.
- **Effort:** 1-2 days
- **Reliability:** vllm_backend.py:231, llamacpp_backend.py:131-134, tensorrt_backend.py:212 return token idx; mlx_backend.py:117-118 and onnx_backend.py:158 return logits.
- **Recommendation:** Standardize the forward contract: have single-node input_ids paths return the full logits (SampingParams logprobs), or document and enforce that single-node forward returns the argmax token index across ALL backends uniformly. Add a conformance test that asserts the shape/semantics of forward(input_ids=...) for every registered backend.

---

### F-154 — [Medium] swap_providers migrates cloud instances with no data-plane handoff — drops in-flight model/KV state

`src/distllm/cloud/spot_orchestrator.py:1570` · zone=`backends-config-cloud` · category=`architecture`

- **Summary:** SpotOrchestrator.swap_providers bids on target instances via launch_cluster then cancels the old instances (lines 1570-1579). For a distributed-LLM cluster, migrating GPU nodes without transferring the loaded model weights / KV cache / active requests means the new instances start cold and all in-flight generation is lost. The docstring claims 'Migrate all running instances', but no workload/state transfer exists.
- **Evidence (verbatim):**
```
# Bid on target instances bid_results = self.launch_cluster(target_instances) # Cancel old instances (best-effort) for inst in migrating: success = self._market.cancel_bid(current, inst.instance_id)
```
- **Impact:** Silent interruption of active distributed inference during provider migration; cost and availability surprises.
- **Effort:** 1-2 days
- **Reliability:** Lines 1571-1579: no readiness gate between launching target and cancelling source.
- **Recommendation:** Either take a dry-run/plan phase that stages model weights and checkpoint on the target before cancel (model-export + reload flow), or explicitly document that this is a 'reprovision the cluster' operation, not a live failover, and refuse to cancel old instances until new ones report ready/loaded.

---

### F-155 — [Medium] cache_manager._tier_store eviction is O(n^2): recomputes min() over all entries once per eviction

`src/distllm/core/cache_manager.py:221` · zone=`core-cache` · category=`performance`

- **Summary:** In the production-core CacheManager multi-tier tier (GPU/CPU/SSD), _tier_store() loops 'while tier used_bytes + size > max_bytes', and in each iteration re-scans the ENTIRE entries dict via min(..., key=_tier_eviction_score) (which itself touches the ghost cache dict) to pick one victim, then evicts one entry. Under heavy pressure this is quadratic, and the per-call _prune_ghost_cache() rebuilds the whole ghost dict (and re-sorts it when >max_ghost) on every store. The tier is a leak/boundedness risk as the sole user-facing eviction in the cache path.
- **Evidence (verbatim):**
```
while tier["used_bytes"] + size > tier["max_bytes"] and tier["entries"]:     victim_hash,(victim_blob,_,victim_size) = min(tier["entries"].items(), key=lambda kv:self._tier_eviction_score(kv[0],kv[1],tier_name))
```
- **Impact:** Predictable latency spike and CPU burn during cache-pressure bursts (e.g. cold-start churn of many prompts); grows linearly with cached-entry count.
- **Effort:** 2-4 hours
- **Reliability:** Every eviction scans all entries; with N entries and K evictions per store => O(NK). _prune_ghost_cache rebuilds dict comprehension and does sorted() on every _tier_store call whenever ghost cache is hot.
- **Recommendation:** Build a sorted eviction queue (heapkeyed by _tier_eviction_score) refreshed lazily, or bulk-select the k worst victims once and evict in a batch; mantain _prune_ghost_cache() amortized (e.g. prune by tw prefix-len watermark instead of full dict rebuild + sorted() each call). Add a micro-benchmark caching thousands of prefixes to bound eviction latency.

---

### F-156 — [Medium] The 2,000-line PBFT/Byzantine subsystem (byzantine.py) is not wired into any production path

`src/distllm/dist/byzantine.py:1789` · zone=`dist-exec` · category=`architecture`

- **Summary:** Grep across src/distllm shows no production caller for PBFTNode, ByzantineCoordinator, SplitBrainDetector, or QuorumManager outside byzantine.py's own docstrings/tests. The Byzantine handling claimed for the cluster (membership, crash-eviction, fork detection) is an inert, highly complex (45KB) asset. Meanwhile the actual membership/trust decisions in the execution path rest only on a single shared cluster_key (node_service._check_auth) with no quorum, no per-node identity, and no fault model — so the 'Byzantine-node handling' thread is effectively unimplemented in the running system while the code that looks like it is, is dead.
- **Evidence (verbatim):**
```
msg = PBFTMessage(phase=PBFTPhase.PRE_PREPARE, **kwargs)\nreturn self._handle_pre_prepare(msg, from_primary=False)
```
- **Impact:** Reputation for 'Byzantine fault tolerance' is unbacked by runtime behavior; a malicious peer holding the shared key can act unilaterally (e.g., trigger TransferWeights on every node to drain the model), which BFT was supposed to prevent.
- **Reliability:** `grep -rn 'PBFTNode\|ByzantineCoordinator\|QuorumManager' src/distllm --include=*.py` — hits only inside byzantine.py.
- **Recommendation:** Either (a) integrate a thin consensus path (register membership changes, KV-cache invalidation, log-off tuples) through ByzantineCoordinator so eviction/fork decisions actually use quorums, or (b) if BFT is out of scope for v1, document it as experimental and add an explicit call-graph test asserting no accidental use. As-is the maintenance burden runs ahead of the guarantee.

---

### F-157 — [Low] MultiTenantSLOEnforcer does O(n log n) sort of the full latency window on every request, and p99/SLO breach only update after 100 samples

`src/distllm/dist/multi_tenant.py:244` · zone=`dist-exec` · category=`performance`

- **Summary:** Every `record_request_end` sorts the entire (up to 1000-entry) deque to compute p99, so per-request latency accounting is O(1000 log 1000) even though only one sample was added — a performance tax on the hot path. Further, p99 and the resulting `slo_breaches` increment are only recalculated once history length crosses 100 and thereafter; a single request breaching the SLO (or a spike in p50) is never counted, so SLO breach rate undercounts real violations. `sorted(history)[:999]` sorts 1000 items then trims to 999 (no-op).
- **Evidence (verbatim):**
```
if len(history) >= 100:\n    sorted_lats = sorted(history)[:999]\n    p99_idx = int(len(sorted_lats) * 0.99)
```
- **Impact:** Redundant sort work on the scheduling hot path plus systematically understated p99/SLO-breach statistics that feed priority boosting.
- **Reliability:** With a 1000-sample window, record_request_end sorts 1000 floats per request; and 100 requests each above the SLO but pushing p99 below threshold record zero breaches.
- **Recommendation:** Use a bucketed/ordered-statistic approximation (e.g., a small reservoir or the existing rows as a quantile sketch) updated in O(1), and compute p99 on the full window without the slice. Count an SLO breach per request whose own latency exceeds the target, not only when the p99 crosses it.

---
