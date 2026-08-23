---
tags:
  - audit
  - action-plan
date: 2026-08-11
---

# Action Plan 02 — Issues & Required Fixes

**← [[Exhaustive Audit 2026-08-11]]**


DistLLM issues & required-fixes catalog. Produced 2026-08-11 from the 78-agent exhaustive audit (audit_digest.json / findings_verified.json). Covers all 59 adversarially-verified Critical/High bug+security findings (46 mustFix, 10 real-not-blocking, 3 refuted) plus the highest-impact unverified Medium/High operational findings. Organized P0 (block any release: security/auth boundaries, data-integrity/leak, core-promise breaks) > P1 (this sprint: load-bearing correctness/functionality breaks) > P2 (next 1-2 sprints) > P3 (backlog + refuted no-action flags). 89 entries. Every claim is code-grounded (file:line).

**89 issues** · P0: 12 · P1: 34 · P2: 34 · P3: 9

Priority scale: **P0** = block any release (auth/security boundaries, data integrity, core promise broken) · **P1** = this sprint (load-bearing defects) · **P2** = next 1–2 sprints · **P3** = backlog. `🔒` = adversarially verified (isReal/mustFix confirmed).

## Priority P0 (12)

### [High] 🔒 VERIFIED JWT authentication bypass via algorithm confusion in HS256 fallback validator
`src/distllm/plugins/auth_plugin.py:168` · category=`security`

- **Description:** _validate_jwt_hs256 (auth_plugin.py:160-175) verifies a JWT signature with HMAC-SHA256 using the configured DISTLLM_AUTH_SECRET string, ignoring the token's `alg` header entirely and never requiring a shared secret. It runs whenever PyJWT (an optional dep, imported in try/except) is absent. The parser splits on '.' (so it also accepts any 3-part token, including tokens signed with a PUBLIC asymmetric key), then does hmac.compare_digest(hmac.new(secret, message), sign=actual). No alg whitelist, no iss/aud strictness, no key-type sanity.
- **Repro:** Run without PyJWT installed; set DISTLLM_AUTH_SECRET='secret'. Craft JWT with header {'alg':'HS256'}, payload {'role':'admin'}, signed with 'secret'; it authenticates as admin.
- **Expected vs actual:** Expected: only HS256 tokens signed with the exact shared secret authenticate. Actual: any 3-part token whose HMAC matches the (arbitrarily short) configured secret authenticates; asymmetric-alg tokens accepted.
- **CVSS:** 9.1
- **Impact:** Full authentication bypass and privilege escalation to admin for any deployment running without PyJWT when an asymmetric signing key or a short shared secret is configured: an attacker forges a token valid against the shared secret (or uses the token's own public-key material) and is treated as authenticated. Root-trust compromise of the API layer.
- **Effort:** 2-4 hours
- **Dependencies:** Depends on the unified auth plugin being on the request path (already wired via PluginHookMiddleware).
- **Timeline:** Day 1 of P0 sprint: reject PEM-looking/long secrets, require alg==HS256 and strict iss/aud, add PyJWT as a hard dependency.
- **Verification note:** MUSTFIX (confirmed; JWT fallback validator bypasses signature-type enforcement)

---

### [Critical] 🔒 VERIFIED Cross-model sibling cache lookup returns KV data for a DIFFERENT prompt (wrong/injected tokens)
`src/distllm/core/cross_model_prefix_sharing.py:169` · category=`bug`

- **Description:** CrossModelPrefixSharing.lookup() sibling branch (lines 166-178) iterates EVERY cached entry of models sharing a base_model and returns the FIRST match on shared_layers>0 alone, never comparing token_ids or prefix_hash to the query. The direct (case 1) and base (case 2) branches key on the computed prefix_hash, but the sibling branch returns whatever cached sequence was stored first for any prompt of any same-family model. Evidence: `for key, entry in self._cache.items(): ... if sibling.base_model == model.base_model: ... return entry`.
- **Repro:** Cache an entry for prompt A against base model B; query the same cache with prompt C (also base B). lookup() returns A's KV as the prefix for C.
- **Expected vs actual:** Expected: return the cache entry whose token_ids/prefix_hash equal the query. Actual: returns arbitrary first same-family entry regardless of prompt.
- **Impact:** Wrong/cross-tenant generations in shared-base serving: a cached KV prefix for prompt A is replayed as the prefix for prompt B, injecting foreign tokens into completions and exposing stored content across prompts/tenants. Correctness + data-exposure bug on the flagship multi-tenant layer-sharing path.
- **Effort:** 2-4 hours
- **Dependencies:** None (self-contained in the lookup). Test fixture needs 2 same-family cached entries with different prompts.
- **Timeline:** Day 1-2 of P0 sprint.
- **Verification note:** MUSTFIX (confirmed; sibling lookup never compares token_ids to the query)

---

### [High] 🔒 VERIFIED Exact-match cache key is not tenant/user scoped - cross-tenant response leak
`src/distllm/plugins/cache_plugin.py:83` · category=`security`

- **Description:** _build_cache_key hashes only (prompt|model|temperature|top_p). The primary exact-match path in on_request/on_response uses it with no tenant/user component; only the optional semantic path is scoped via _request_scope. Two tenants sending identical (prompt, model, temp) get the same response, and tenant A's private personalized output (PII, per-user answers) is served to tenant B on a cache hit.
- **CVSS:** 7.5
- **Impact:** Cross-tenant disclosure of private cached LLM responses in any multi-tenant deployment with caching enabled. Tenant-isolation boundary breach at the cache layer.
- **Effort:** 1-2 hours
- **Dependencies:** None.
- **Timeline:** Day 1-2 of P0 sprint.
- **Verification note:** MUSTFIX (confirmed; exact-match cache key has no tenant/user component)

---

### [High] 🔒 VERIFIED Unauthenticated HA heartbeat/snapshot fail-open when DISTLLM_HA_SECRET is unset -> leader-election takeover and coordinator state injection
`src/distllm/api/server.py:1434` · category=`security`

- **Description:** AuthMiddleware explicitly exempts /api/v1/ha/heartbeat (middleware.py:225-232), and the route (server.py:1421-1474) gates on X-HA-Secret only inside `if expected_secret:` where expected_secret defaults to os.environ.get('DISTLLM_HA_SECRET','') - an empty string is falsy, so an unauthenticated request with no/any X-HA-Secret is accepted when the env var is unset (the default). A socket can inject a full coordinator state snapshot (fake nodes/metadata) cluster-wide.
- **Repro:** Leave DISTLLM_HA_SECRET unset; POST /api/v1/ha/heartbeat with no X-HA-Secret; the route accepts and processes the snapshot.
- **Expected vs actual:** Expected: reject heartbeat when DISTLLM_HA_SECRET is unset. Actual: fail-open - empty expected_secret disables the check entirely.
- **CVSS:** 8.6
- **Impact:** Unprivileged takeover of HA leader election and injection of a full coordinator state snapshot (fake nodes/metadata) across the cluster from an unauthenticated socket.
- **Effort:** 3-5 hours
- **Dependencies:** None.
- **Timeline:** Day 1-2 of P0 sprint.
- **Verification note:** MUSTFIX (confirmed; HA heartbeat/snapshot fail-open when DISTLLM_HA_SECRET unset)

---

### [High] 🔒 VERIFIED ApiKeyRotator never retires the old (possibly compromised) key - rotation is a no-op for security
`src/distllm/core/cert_rotation.py:300` · category=`security`

- **Description:** ApiKeyRotator.rotate() registers a replacement key with the same key_id but leaves the OLD key's entry in ApiKeyStore._keys; the old key is only removed by cleanup_expired() which is never scheduled. Grep across src shows is_rotated_key_valid etc. are never wired into the auth path. A compromised/rotated API key stays valid indefinitely.
- **Expected vs actual:** Expected: after rotate()+grace, old token rejected. Actual: old key stays in _keys and authenticates indefinitely.
- **Impact:** A rotated/compromised key remains a live persisted credential forever, defeating key rotation and leaving an attacker with access to a supposedly-retired key.
- **Effort:** 4-8 hours
- **Dependencies:** Needs a background timer or auth-path hook to call cleanup_expired() after grace elapse.
- **Timeline:** Day 2-3 of P0 sprint.
- **Verification note:** MUSTFIX (confirmed; rotate() never retires the old key)

---

### [High] 🔒 VERIFIED Federated LoRA merge accepts untrusted adapters + self-reported dataset_size -> model poisoning and unfair weight dominance
`src/distllm/dist/federated_merge.py:193` · category=`security`

- **Description:** In submit_node_adapter an arbitrary node supplies both `adapter_path` (an untrusted file path READ by the coordinator) and `dataset_size`, taken at face value as the FedAvg weight (lines 281-284). A malicious/faulty node can set dataset_size huge to capture the weighted average, or submit a poisoned adapter_path the coordinator reads and applies. No signature/digest is required (byzantine _sign_bytes exists but is unused here).
- **Expected vs actual:** Expected: coordinator authenticates each adapter and caps dataset_size. Actual: any node's path and self-reported size trusted verbatim.
- **CVSS:** 8.8
- **Impact:** One compromised node poisons the global adapter or dominates the weight average, corrupting the artifact every node later adopts in federated multi-node training.
- **Effort:** 3-5 hours
- **Dependencies:** Reuse byzantine._sign_bytes (exists) for signature; add trimming to merge.
- **Timeline:** Day 2-3 of P0 sprint.
- **Verification note:** MUSTFIX (confirmed; untrusted adapter + self-reported dataset_size)

---

### [High] 🔒 VERIFIED Kademlia DHT STORE is unauthenticated and token-unverified
`src/distllm/dist/p2p/kademlia_dht.py:834` · category=`security`

- **Description:** Any node that can reach the UDP DHT port can write arbitrary key->value bytes onto any other node's local store. The caller attaches a 'token' but _handle_store never validates it against anything (the receiver has no way to derive/verify it). FIND_VALUE lookups can be redirected to attacker-controlled nodes; bogus KV-cache/peer-discovery records injected over the WAN.
- **CVSS:** 8.8
- **Impact:** DHT value poisoning and cache poisoning; peer-discovery and KV-cache routing can be redirected to attacker-controlled nodes across the WAN.
- **Effort:** 3-5 hours
- **Dependencies:** Needs a shared-secret/time-bound capability HMAC or reachability-ping gate on STORE.
- **Timeline:** Day 2-3 of P0 sprint.
- **Verification note:** MUSTFIX (confirmed; STORE token never validated on receive)

---

### [High] 🔒 VERIFIED QUIC client disables TLS peer verification (CERT_NONE)
`src/distllm/dist/p2p/quic_transport.py:389` · category=`security`

- **Description:** All outgoing QUIC connections set verify_mode=CERT_NONE, so the client accepts any certificate and never authenticates the remote peer. Over the Internet this permits a straight MITM: an attacker impersonating a node supplies a self-signed cert and the client proceeds.
- **CVSS:** 8.1
- **Impact:** Full MITM of QUIC peer data (gossip metadata + KV cache content) despite the transport nominally using TLS.
- **Effort:** 2-4 hours
- **Dependencies:** Reuse the shared cluster key / gossip secret to pin peer fingerprints or a CA bundle.
- **Timeline:** Day 3-4 of P0 sprint.
- **Verification note:** MUSTFIX (confirmed; QUIC verify_mode=CERT_NONE)

---

### [High] 🔒 VERIFIED Pipeline cross-node gRPC is plaintext by default
`src/distllm/dist/node_client.py:365` · category=`security`

- **Description:** The main pipeline-parallel inference path sends hidden states and KV caches over unencrypted gRPC. create_node_client defaults use_tls=False (node_client.py:173) and forward_request/forward_request_async never pass use_tls, so grpc.insecure_channel() is used. LLM activations and KV cache (which encode prompt/token content) are exposed to any on-path observer over LAN/WiFi/Internet.
- **CVSS:** 7.4
- **Impact:** Model activations/hidden states/KV caches (prompt+token content) exposed on the wire, defeating the project's stated TLS posture.
- **Effort:** 4-8 hours
- **Dependencies:** Thread use_tls through create_node_client + forward_request + PipelineOrchestrator; warn/raise on insecure-with-payload.
- **Timeline:** Day 3-5 of P0 sprint.
- **Verification note:** MUSTFIX (confirmed; pipeline forward is plaintext by default)

---

### [Critical] 🔒 VERIFIED RED metrics (requests/latency/duration/errors) are never recorded - .labels() handles are discarded
`src/distllm/api/observability_middleware.py:124` · category=`bug`

- **Description:** ObservabilityMiddleware is the sole wiring for the Prometheus exporter's request metrics, but every call constructs a labeled handle via .labels(...) and throws it away without .inc()/.observe(). Evidence: observability_middleware.py:120-128 `exporter.requests_total.labels(method=..., status=..., model=..., tenant=...)` with no terminating .inc()/.observe(). requests_total, request_latency, request_duration_seconds, errors_total all stay 0.
- **Expected vs actual:** Expected: each request increments counts and observes latency/duration. Actual: handles built then dropped - exporter shows 0.
- **Impact:** Every dashboard/alert reading request count, p95 latency, duration, or error rate shows 0 forever; SLO/capacity graphs blank and alert thresholds never fire. The flagship observability promise is inert - operators cannot see load or errors for the product's core inferences.
- **Effort:** 1-2 hours
- **Dependencies:** None (adds terminating .inc()/.observe() calls + a regression test).
- **Timeline:** Day 1 of P0 sprint (cheap, high visibility).
- **Verification note:** MUSTFIX (confirmed; metrics handles discarded without .inc()/.observe())

---

### [High] 🔒 VERIFIED DP 'inference' applies NO differential privacy to outputs while still charging the tenant's privacy budget
`src/distllm/core/dp_inference.py:929` · category=`security`

- **Description:** The advertised DP inference path (generate/generate_stream, budget enforcement, per-tenant accounting) never perturbs outputs: it calls the raw _engine.generate_stream and records _estimate_epsilon_cost into the tenant budget. No noise is added to logits; _dp_sample/apply_dp_to_logits are never invoked in the run. Outputs are plaintext while the tenant is charged a 'privacy budget' that protects nothing.
- **Expected vs actual:** Expected: outputs from DP mechanism (noise on logits) and budget charged accordingly. Actual: raw outputs, budget charged.
- **CVSS:** 6.5
- **Impact:** Customers paying for a DP / HIPAA-informed privacy budget receive plaintext outputs while believing they are protected; legitimate tenants get blocked on a budget that protects nothing. Privacy/trust claim is false advertising with real enforcement harm.
- **Effort:** 2-4 days
- **Dependencies:** Wire _dp_sample into the logit sampling path or make generate raise (fail-closed) when DP cannot be applied.
- **Timeline:** Day 4-7 of P0 sprint (needs engineering, not a one-liner).
- **Verification note:** MUSTFIX (confirmed; DP path never perturbs outputs yet charges budget)

---

### [High] 🔒 VERIFIED Aether 'secure' aggregation is a no-op and its pairwise masks use a single public seed
`src/distllm/core/federated_finetuner.py:178` · category=`bug`

- **Description:** FederatedFineTuner._received_grads is keyed by peer_id and never cleared or round-filtered; _receive()/store ignore the 'round' field broadcast in train_round(). A slow peer's round-1 gradients are averaged with round-N gradients. The 'secure aggregation' masking uses a single public seed, so any node can reproduce every other node's pairwise mask.
- **Impact:** Non-deterministic, unstable federated training under any straggler/failure; round-mixing corrupts averages; the advertised secure-masking protection is breakable. Linked to the fake 'secure aggregation' claim that is a headline differentiator.
- **Effort:** 3-5 hours
- **Dependencies:** None (store (round, grads), prune stale peers, layer-count guard).
- **Timeline:** Day 3-5 of P0 sprint.
- **Verification note:** MUSTFIX (confirmed; 'secure' aggregation is a no-op + single public seed masks)

---

## Priority P1 (34)

### [High] 🔒 VERIFIED PagedAttention KV blocks leak for every completed sequence
`src/distllm/core/batch_scheduler.py:1065` · category=`bug`

- **Description:** Completed sequences are removed from self.active inside step() without freeing their PagedAttention blocks; the only free path (_prefetch_and_snapshot) scans only sequences still in active. Because step() prunes completed sequences before the next scan, their blocks are never reclaimed.
- **Repro:** Run the PagedAttention batch scheduler in a loop completing many sequences; observable VRAM/KV-block count grows monotonically.
- **Expected vs actual:** Expected: completion frees the sequence's KV blocks. Actual: blocks leak per completed request.
- **Impact:** KV cache (VRAM) not reclaimed on completion in the PagedAttention path; long-running servers degrade to OOM/failed allocations as blocks leak. Directly contradicts the scheduler's own eviction-safety design.
- **Effort:** 0.5 hours
- **Dependencies:** None (call free_paged_blocks(rid) in the prune loop; keep single free site to avoid double-free).
- **Timeline:** Day 1-2 of P1 sprint.
- **Verification note:** MUSTFIX (confirmed; PagedAttention blocks never freed on completion)

---

### [High] 🔒 VERIFIED int8/int4/adaptive compressed KV cache serves raw (non-dequantized) tensors to attention
`src/distllm/core/kv_cache.py:159` · category=`bug`

- **Description:** KVCache.get() only dequantizes the bulk-compressed path when _quant_fp8 is True. compress('int8'), compress('int4'), and AdaptiveQuantizer.apply() all set _quant_fp8=False while populating _scale_k/_scale_v, so get() falls through to `return k, v  # Return raw quantized if no scale available` (kv_cache.py:159-165) - raw int8/int4 values fed straight to attention.
- **Repro:** compress('int8') a layer, then get() it; assert return dtype/value equals dequantized float.
- **Expected vs actual:** Expected: int8/int4 KV dequantized by scale before attention. Actual: raw quantized ints fed to attention.
- **Impact:** Wrong KV values fed to attention for int8/int4/adaptive compression -> incorrect (often nonsensical) completions; the 2-4x memory savings is unusable. Numeric-correctness regression on a headline memory-optimization feature.
- **Effort:** 2-4 hours
- **Dependencies:** None (dequantize by scale for all quantized bulk paths; add regression).
- **Timeline:** Day 1-2 of P1 sprint.
- **Verification note:** MUSTFIX (confirmed; int8/int4/adaptive get() returns raw quantized tensors)

---

### [High] 🔒 VERIFIED AdaptiveSerializer ZSTD path silently corrupts large (FP8) tensors - scale lost, double compressed
`src/distllm/dist/pipeline/compression_negotiation.py:604` · category=`bug`

- **Description:** SerializationController.send_tensor(method=ZSTD) for a tensor >100MB: AdaptiveSerializer.choose_format returns FP8_ZSTD, serialize() yields an fp8-quantized, scale-packed, zstd-compressed payload. The controller then zstd-compresses that payload AGAIN and retags it ZSTD, so deserialize reads misinterpreted bytes (scale header lost, double compression).
- **Repro:** send_tensor(method=ZSTD) with a >100MB CUDA tensor; round-trip and compare against in-memory values.
- **Expected vs actual:** Expected: FP8_ZSTD payload deserialized with correct dequantization. Actual: double-compressed bytes misread -> corruption.
- **Impact:** KV-cache/hidden-state corruption on large (>100MB) tensors whenever ZSTD is negotiated - wrong inference output with no error raised.
- **Effort:** 3-5 hours
- **Dependencies:** Route format from the serializer as-is (do not re-compress+retag); make ZSTD deserialize dequantize embedded scale.
- **Timeline:** Day 2-3 of P1 sprint.
- **Verification note:** MUSTFIX (confirmed; AdaptiveSerializer ZSTD double-compresses FP8 payload)

---

### [High] 🔒 VERIFIED ZeroCopyTransferEngine.recv fabricates zeros (NCCL) / always fails (CUDA_IPC)
`src/distllm/dist/zero_copy.py:222` · category=`bug`

- **Description:** The receive side of the zero-copy engine is non-functional: NCCL 'recv' returns torch.zeros() and marks success=True (silent data corruption - caller believes it received real KV data); CUDA_IPC recv calls import_tensor(key, b'', ...) with an empty handle so it raises/None-fails.
- **Expected vs actual:** Expected: recv returns the transferred KV data. Actual: NCCL returns zeros (silent), CUDA_IPC fails on empty handle.
- **Impact:** Silent corruption of KV/activations whenever the zero-copy NCCL path is used, or hard failures on CUDA_IPC - both perceived as 'working' zero-copy transfers.
- **Effort:** 1-2 days
- **Dependencies:** Sender must transmit the IPC handle (export_tensor bytes) via the control channel before recv; NCCL recv must map ranks.
- **Timeline:** Day 3-5 of P1 sprint.
- **Verification note:** MUSTFIX (confirmed; NCCL recv fabricates zeros, CUDA_IPC always fails)

---

### [High] 🔒 VERIFIED PipelinedSpeculativeDecoder verifier always rejects (or silently accepts) every draft: draft slots never carry verifier inputs
`src/distllm/core/async_pipelined_speculative.py:437` · category=`bug`

- **Description:** DraftSlot has hidden_states and compressed_logits, but _draft_worker (invoked by _launch_draft) fills only token_ids/logprobs from draft_gen(prompt, n). When a verifier IS configured, _verify_worker hits the else branch with empty inputs (default verifier=None -> emits unverified draft output). Pipelined speculation silently provides zero verified speedup or actively wrong output.
- **Impact:** With default verifier=None the decoder emits unverified draft output (distributionally wrong); with a verifier set it silently does zero speculative speedup while consuming thread/stream resources.
- **Effort:** 1-2 days
- **Dependencies:** Populate slot.hidden_states/compressed_logits in _draft_worker and require a real p/q verifier.
- **Timeline:** Day 3-6 of P1 sprint.
- **Verification note:** MUSTFIX (confirmed; draft slots never carry verifier inputs)

---

### [High] 🔒 VERIFIED Off-by-one token-position indexing in 4 of 9 speculative verifiers
`src/distllm/core/tree_speculative_decoder.py:326` · category=`bug`

- **Description:** Across tree_speculative_decoder.py 4 verifiers (lines ~326 region) index target logits with `prefix_len + i` while the 5 correct ones use `prefix_len - 1 + i` (logits[k] predicts token[k+1], so draft token i is predicted by logits[P-1+i]). The wrong ones check each draft token against the NEXT token's prediction.
- **Expected vs actual:** Expected: draft token i verified against logits[P-1+i]. Actual: verified against logits[P+i] (next token).
- **Impact:** Silent wrong acceptance/rejection; emits tokens the target model never predicted; acceptance-rate stats inflated/deflated; tree speculative decoding (claimed 2-3x acceptance gain) corrupts output.
- **Effort:** 1-2 hours
- **Dependencies:** None (align to prefix.shape[1]-1+i convention).
- **Timeline:** Day 1-2 of P1 sprint.
- **Verification note:** MUSTFIX (confirmed; off-by-one in 4 of 9 verifiers)

---

### [High] 🔒 VERIFIED JSONSchemaConstraint FSM never allows continuation of multi-digit numbers - constrained generation truncates numeric values
`src/distllm/core/structured_output/__init__.py:202` · category=`bug`

- **Description:** The token-level mask's _valid_next_chars has no entry for the `in_number` state, so the transitions dict falls back to the default {'"','}'} (line 202): after the first digit the model can only emit the number terminator. Structured output generates truncated/corrupted JSON for any numeric field value >1 digit.
- **Repro:** Build JSONSchemaConstraint for {'a': number}; feed '{"a": 1'; assert the next valid char set includes '2'..'9'.
- **Expected vs actual:** Expected: numeric fields produce full multi-digit values. Actual: truncated to one digit.
- **Impact:** Structured output (a headline feature) generates corrupted/truncated JSON for any numeric value >1 digit, silently degrading to single-char numbers.
- **Effort:** 1-2 hours
- **Dependencies:** None (add in_number entry returning digit/exponent char set).
- **Timeline:** Day 1-2 of P1 sprint.
- **Verification note:** MUSTFIX (confirmed; JSONSchemaConstraint FSM lacks in_number continuation)

---

### [High] 🔒 VERIFIED AudioPipeline state machine gets permanently stuck in SPEAKING after the first utterance
`src/distllm/core/media_pipeline.py:254` · category=`bug`

- **Description:** After the first spoken utterance produces TTS audio the state becomes SPEAKING and there is no transition back to IDLE/LISTENING. Any subsequent speech frame is buffered but the pipeline is stuck (the is_speaking check never resets). test_media_pipeline.py only checks IDLE->LISTENING->PROCESSING.
- **Impact:** With TTS enabled the real-time voice pipeline dies after the first response; a conversation can never continue.
- **Effort:** 1-2 hours
- **Dependencies:** None (add SPEAKING->LISTENING/IDLE transition after pushing audio, or treat speech during SPEAKING as barge-in).
- **Timeline:** Day 2-3 of P1 sprint.
- **Verification note:** MUSTFIX (confirmed; AudioPipeline stuck in SPEAKING after first utterance)

---

### [High] 🔒 VERIFIED Hydra VideoPipeline feeds random noise to stable-video-diffusion-img2vid and rebuilds the whole model on every call
`src/distllm/core/hydra_diffusion.py:79` · category=`bug`

- **Description:** VideoPipeline.generate hardcodes stabilityai/stable-video-diffusion-img2vid, ignores the requested model name, passes t.randn(1,3,512,512) (pure random noise) as the conditioning input to an image-to-video pipeline, and instantiates the pipeline fresh inside HydraOrchestrator on every call.
- **Impact:** Garbage video frames (img2vid conditioned on noise), plus severe unbounded memory/network churn (GB-scale HuggingFace download per request) and no model reuse.
- **Effort:** 2-4 hours
- **Dependencies:** Cache the pipeline (keyed by resolved model) and accept a real input image/frame.
- **Timeline:** Day 2-3 of P1 sprint.
- **Verification note:** MUSTFIX (confirmed; VideoPipeline feeds random noise to img2vid)

---

### [High] 🔒 VERIFIED request_latency elapsed_ms grows with wall clock on COMPLETED records, corrupting SLA compliance percentiles
`src/distllm/core/request_latency.py:29` · category=`bug`

- **Description:** RequestLatencyInfo.elapsed_ms is computed live as (time.time() - enqueued_at)*1000 and is_overdue compares that to the SLA. Completed requests are kept in _completed with their original enqueued_at, and get_sla_percentiles()/get_recent_metrics() read elapsed_ms -> the longer the process runs the more overdue completed records look.
- **Impact:** SLA compliance dashboards and any quota/promote path over-report overdue and under-report compliance over time; a healthy cluster shows chronic SLA violations.
- **Effort:** 1-2 hours
- **Dependencies:** None (freeze duration in complete(); compute overdue from frozen duration).
- **Timeline:** Day 2-3 of P1 sprint.
- **Verification note:** MUSTFIX (confirmed; completed records grow elapsed_ms with wall clock)

---

### [High] 🔒 VERIFIED telemetry.flush() deadlocks when auto-triggered from _add_event (non-reentrant lock inside lock)
`src/distllm/core/telemetry.py:184` · category=`bug`

- **Description:** TelemetryCollector._add_event holds self._lock (a plain threading.Lock) and calls self.flush(), which re-acquires the same non-reentrant lock. When BATCH_SIZE (50) events are recorded, the calling thread blocks permanently inside flush()'s `with self._lock`.
- **Impact:** With telemetry enabled, the 50th recorded event hangs the requesting thread (a sync request path) permanently; even without the deadlock the collected anonymous data never reaches telemetry.distllm.ai.
- **Effort:** 2-4 hours
- **Dependencies:** None (flush outside the lock / make lock an RLock / background flusher thread).
- **Timeline:** Day 2-3 of P1 sprint.
- **Verification note:** MUSTFIX (confirmed; telemetry.flush() re-acquires non-reentrant lock)

---

### [High] 🔒 VERIFIED HA leader election never gates request processing - standby coordinators serve requests
`src/distllm/core/coordinator.py:426` · category=`bug`

- **Description:** CoordinatorElection/RayFaultTolerance elect a leader and track is_leader, but nothing in the request path consults it. generate()/request_pipeline execute on every coordinator regardless of leadership, so in HA mode all coordinators act as writers and apply leader snapshots on top of their own mutations.
- **Impact:** In a 2+ coordinator HA deployment, standby nodes accept and serve inference while also applying leader snapshots, producing divergent KV/node state and violating the single-writer invariant.
- **Effort:** 4-8 hours
- **Dependencies:** Gate admission at request entry (return 'forward to leader'/reject when standby); add _is_standby checks + test.
- **Timeline:** Day 3-5 of P1 sprint.
- **Verification note:** MUSTFIX (confirmed; HA standby coordinators serve requests)

---

### [High] 🔒 VERIFIED GPUResourceManager.snapshot reports used_mb and free_mb swapped
`src/distllm/core/gpu_resource_manager.py:231` · category=`bug`

- **Description:** In snapshot(), device_alloc = torch.cuda.memory_allocated(device) is the USED (allocated) memory but is returned as free_mb, while used_mb is computed as total-device_alloc (actually free). Callers read inverted headroom.
- **Expected vs actual:** Expected: used_mb=allocated, free_mb=total-allocated. Actual: swapped.
- **Impact:** Memory utilization, safe_margin and OOM-risk reporting are wrong; is_oom_risk and any dashboard/autoscaler fed from snapshot read inverted headroom (premature eviction or false OOM safety).
- **Effort:** under 1 hour
- **Dependencies:** None (swap the two fields).
- **Timeline:** Day 1 of P1 sprint.
- **Verification note:** MUSTFIX (confirmed; snapshot used_mb/free_mb swapped)

---

### [High] 🔒 VERIFIED DifferentialPrivacyInference.generate() non-streaming branch raises NameError (dead/broken code)
`src/distllm/core/dp_inference.py:953` · category=`bug`

- **Description:** The `elif hasattr(self._engine, 'generate')` branch references token_count and text, which are only assigned inside the earlier generate_stream block. Any engine exposing only generate() crashes with NameError at runtime.
- **Impact:** Crashes any DP wrapper whose engine lacks generate_stream, making the documented non-streaming DP path unusable.
- **Effort:** under 1 hour
- **Dependencies:** None (obtain text/token_count from self._engine.generate()).
- **Timeline:** Day 2 of P1 sprint.
- **Verification note:** MUSTFIX (confirmed; run raises NameError in non-streaming path)

---

### [High] 🔒 VERIFIED FedProx term subtracts a gradient from a weight: mu*(grad - global_param)
`src/distllm/core/federated_finetuner.py:254` · category=`bug`

- **Description:** _apply_fedprox_term computes proximal = mu * (grad - global_param): mixing a gradient tensor with a weight tensor elementwise. Correct form is grad + mu*(w_local - w_global). It double-counts mu*grad, never uses the local weight, and may raise shape errors.
- **Impact:** FedProx produces a mathematically invalid update, so heterogeneous-data federated training is unreliable when fedprox_mu>0.
- **Effort:** 2-4 hours
- **Dependencies:** None (use pre-round local weights: grad + mu*(w_local - w_global)).
- **Timeline:** Day 3-4 of P1 sprint.
- **Verification note:** MUSTFIX (confirmed; FedProx term mixes gradient with weight)

---

### [High] 🔒 VERIFIED Aether LoRA path never trains the adapter; merge of the zero-B adapter is a no-op
`src/distllm/core/aether_federated.py:921` · category=`bug`

- **Description:** start_finetuning() with lora_config creates a LoRA adapter, but then trains on the FULL base weights; the adapter's random A / zero B are never updated. merge() then adds (alpha/rank)*(0 @ A^T)=0, returning final_weights unchanged. 'used_lora' is True but the LoRA was never trained.
- **Impact:** LoRA-based federated fine-tuning silently produces the base model; users believe they adapted weights on private data but did not.
- **Effort:** 1-2 days
- **Dependencies:** Train over LoRA parameters; or at minimum document lora_config as advisory + assert when adapter not trained.
- **Timeline:** Day 4-6 of P1 sprint.
- **Verification note:** MUSTFIX (confirmed; LoRA adapter never trained, merge zero-B no-op)

---

### [High] 🔒 VERIFIED MLX backend forward() scores each token in isolation with no KV state - corrupts any context>1 forward
`src/distllm/backends/mlx_backend.py:102` · category=`bug`

- **Description:** MLXNodeAdapter._forward_input_ids loops over tokens calling self._model(mx.array([tid]).reshape(1,1)) per token with no past_key_values carry-over and ignoring attention_mask/position_ids. Each token is evaluated as a length-1 sequence, so tokens after [0] are wrong.
- **Impact:** Wrong outputs for any multi-token input on Mac/MLX deployments (the priority-10 backend for mps); silent corruption rather than a loud error.
- **Effort:** 2-4 hours
- **Dependencies:** Run full sequence in one call + thread KV cache, or reject seq_len>1 (fail loudly).
- **Timeline:** Day 4-5 of P1 sprint.
- **Verification note:** MUSTFIX (confirmed; MLX forward scores each token standalone)

---

### [High] 🔒 VERIFIED NIM fallback fabricates hash-scattered 'logits' in _forward_via_api instead of failing loudly
`src/distllm/backends/nim_backend.py:415` · category=`bug`

- **Description:** When no local_model is provided, NIM _forward_via_api builds a fake logit tensor sized len(top)*4 and places np.log(prob) at idx = hash(token_str) % size. Python's built-in hash() is randomized per-process (PYTHONHASHSEED), so identical tokens land at different indices - non-reproducible fabricated logits.
- **Impact:** Fabricated, non-reproducible logits silently fed to the pipeline for NIM nodes without a local model; incorrect argmax/next-token choice; nondeterminism breaks caching.
- **Effort:** 2-3 hours
- **Dependencies:** Raise NotImplementedError (as WebGPUNodeAdapter does) when pipeline-mode forward needs logits but no local model.
- **Timeline:** Day 4-5 of P1 sprint.
- **Verification note:** MUSTFIX (confirmed; NIM fabricates hash-scattered logits)

---

### [High] 🔒 VERIFIED Azure availability check always fails open: URL has literal {subscriptionId} placeholder and no auth
`src/distllm/cloud/azure.py:121` · category=`bug`

- **Description:** AzureAvailabilityChecker.check_availability posts to 'https://management.azure.com/subscriptions/{subscriptionId}/providers/...' with no bearer token. The unfilled literal brace is not valid, and ARM requires auth, so every call raises and is swallowed -> always 'available'.
- **Impact:** Over-provisioning/scheduling onto unavailable Azure regions; silent false-positive availability.
- **Effort:** 3-4 hours
- **Dependencies:** Implement DefaultAzureCredential + real endpoint, or make the fallback fail-closed (available=False on error).
- **Timeline:** Day 5 of P1 sprint.
- **Verification note:** MUSTFIX (confirmed; Azure availability always fails open)

---

### [High] 🔒 VERIFIED IntelligentAutoscaler wired but never fed or actuated - scales nothing
`src/distllm/core/coordinator.py:1071` · category=`perf`

- **Description:** coordinator.start() instantiates IntelligentAutoscaler via _start_subsystem and calls record_metrics once with startup scheduler stats. There is no periodic loop invoking evaluate(), no gpu_utilization is ever populated (ScalingMetrics defaults it to 0), and no provisioning callback applies target_nodes.
- **Impact:** Autoscaling is non-functional in the core coordinator: utilization always 0, scaler can never add a node - the flagship autoscaling feature is inert and silently misleading.
- **Effort:** 4-8 hours
- **Dependencies:** Background loop building ScalingMetrics with real gpu_utilization -> evaluate() -> ClusterManager provisioning callback.
- **Timeline:** Day 5-8 of P1 sprint.
- **Verification note:** MUSTFIX (confirmed; staggered/standby coordinator serves; autoscaler inert)

---

### [High] 🔒 VERIFIED Concurrent-request dedup waiters always time out: _in_flight_results is never populated
`src/distllm/core/request_fingerprinting.py:171` · category=`bug`

- **Description:** RequestFingerprinter implements in-flight dedup where a second identical request waits on the first via _in_flight_results, but nothing in production ever assigns it - mark_in_flight and store never write results. Waiters burn a full timeout (default 30s) and hang.
- **Impact:** The documented 'wait for identical in-flight result' optimization is non-functional and adds a 30s timeout + hang risk to duplicated requests.
- **Effort:** 0.5-1 hours
- **Dependencies:** None (populate _in_flight_results on first completion; TTL evict).
- **Timeline:** Day 1-2 of P1 sprint.
- **Verification note:** MUSTFIX (confirmed; dedup waiters never populated)

---

### [High] 🔒 VERIFIED ModelHub cache-layout mismatch: snapshot_download writes under models--org--name but resolve/is_available check {cache_dir}/org/revision
`src/distllm/models/model_hub.py:384` · category=`bug`

- **Description:** download()/_download_full_model() call snapshot_download(cache_dir=self.cache_dir) which stores under the HuggingFace snapshot layout self.cache_dir/models--org--name/snapshots/<hash>. But is_available()/resolve()/remove() check self.cache_dir/model_name/revision - a mismatch.
- **Impact:** O(pooled) redundant downloads per node join; offline/inference startup broken for a machine that already has the model; unbounded disk growth from repeated snapshots.
- **Effort:** 3-6 hours
- **Dependencies:** Either pass local_dir so files land at self.cache_dir/org/name/revision, or point is_available/resolve at the snapshot layout.
- **Timeline:** Day 5-7 of P1 sprint.
- **Verification note:** MUSTFIX (confirmed; ModelHub cache-layout mismatch)

---

### [High] 🔒 VERIFIED UsageMeter time-window queries under-report after 100k records - silent billing/quota evasion
`src/distllm/dist/daas/usage_meter.py:197` · category=`bug`

- **Description:** _max_records=100_000 is a hard cap: once _records is full, new records go only to SQLite (line 210) and _records keeps the OLDEST 100k. get_usage(tenant_id, since_timestamp) reads only _records, so anything past the 100k cap is missed.
- **Impact:** Correctness of metered billing / DaaS charging and per-tenant quota enforcement silently degrade past 100k records; a tenant can appear under quota when it has far exceeded it.
- **Effort:** 3-5 hours
- **Dependencies:** Query SQLite for since>0, or re-sync/make the cap tenant+client bound.
- **Timeline:** Day 6-7 of P1 sprint.
- **Verification note:** MUSTFIX (confirmed; UsageMeter under-reports after 100k records)

---

### [High] 🔒 VERIFIED PartitionValidator.what_if_slowdown is a no-op - slowdown never affects the simulated throughput
`src/distllm/dist/partition/validator.py:258` · category=`bug`

- **Description:** what_if_slowdown inflates pt.estimated_time_ms on modified points, but _simulate_pipeline never reads estimated_time_ms - it recomputes every stage from _cost_model.evaluate. So the modified solution === the original for the simulation and throughput_change_pct is meaningless.
- **Impact:** Adaptive re-partition incentives and validation 'what-if' reports are fabricated; users cannot predict a straggler slowdown's effect, so migration decisions are based on false numbers.
- **Effort:** 2-4 hours
- **Dependencies:** Have _simulate_pipeline use pt.estimated_time_ms (or apply the multiplier to the cost-model result).
- **Timeline:** Day 6-7 of P1 sprint.
- **Verification note:** MUSTFIX (confirmed; slowdown never affects simulated throughput)

---

### [High] 🔒 VERIFIED Gossip HMAC key rotation replaces the shared configured key with a random node-local key
`src/distllm/dist/p2p/gossip.py:892` · category=`bug`

- **Description:** In a shared-key deployment (DISTLLM_GOSSIP_HMAC_KEY), check_key_rotation() (called every gossip round in GossipReplicator.sync_once when enable_key_rotation, the default) silently overwrites the deployment-wide shared key with a node-local secrets.token_hex(32) (gossip.py:890-892) that no other node knows. After the 24h rotation period, every peer's _hmac_key differs from every other peer's.
- **Expected vs actual:** Expected: rotation derives a new key both peers can compute. Actual: replaces the shared key with a random node-local key no peer shares.
- **Impact:** After 24h uptime with default rotation on, gossip silently stops being authenticated - all KV-cache advertisements are dropped/ignored, breaking distributed cache sharing; the distributed-cache feature dies silently in any deployment that does not disable rotation.
- **Effort:** 2-4 hours
- **Dependencies:** Rotation must only rotate per-peer derived keys (or re-derive from the shared secret KDF); never overwrite a configured shared key.
- **Timeline:** Day 3-4 of P1 sprint.
- **Verification note:** MUSTFIX (confirmed; check_key_rotation overwrites the shared configured key)

---

### [High] 🔒 VERIFIED E2E SessionKeys ratchet diverges under asymmetric traffic, causing intermittent decrypt failures
`src/distllm/security/e2e.py:172` · category=`security`

- **Description:** SessionKeys ratchets its shared key forward on the Nth local encrypt/decrypt (_seq % RATCHET_INTERVAL). Both peers maintain independent _seq counters, and the encrypt-side post-ratchet key derives with the next transmitted salt while the decrypt side uses the prior - keys diverge under asymmetric traffic.
- **CVSS:** 7.4
- **Impact:** Broken E2E transport under realistic asymmetric tensor traffic; session blackouts despite a valid shared key.
- **Effort:** 1-2 days
- **Dependencies:** Make the ratchet deterministic per-message (per-message counter / salt-chain sequence).
- **Timeline:** Day 8-12 of P1/P2 boundary.
- **Verification note:** MUSTFIX (confirmed; E2E SessionKeys ratchet diverges)

---

### [High] 🔒 VERIFIED Streaming is fully broken across langchain/crewai/llamaindex adapters (TypeError on stream=True + AttributeError on completions_stream)
`integrations/langchain/src/distllm_langchain/chat_models.py:285` · category=`bug`

- **Description:** All chat/complete streaming paths in the three main SDK adapters crash at runtime: they pass stream=True to DistLLMClient.chat_completions_stream(...), a method whose signature accepts no stream parameter (streaming is implicit). Async adapters also mishandle the yielded content strings. Tests mask it.
- **Repro:** client = DistLLMLangChain(...); client.stream('hi') -> TypeError (unexpected kwarg 'stream').
- **Expected vs actual:** Expected: streaming yields tokens. Actual: TypeError/AttributeError at call time.
- **Impact:** Every LangChain, CrewAI and LlamaIndex app that uses .stream()/.astream()/stream_chat()/stream_complete() raises at call time instead of streaming tokens.
- **Effort:** 2-4 hours
- **Dependencies:** Remove stream=True; consume the content strings that chat_completions_stream async actually yields.
- **Timeline:** Day 1-2 of P1 sprint (high user-visible impact).
- **Verification note:** MUSTFIX (confirmed; streaming broken across adapters)

---

### [High] 🔒 VERIFIED Tool-calling contract broken: bind_tools() and framework function-calling inject `tools`/federation kwargs the SDK does not accept
`integrations/langchain/src/distllm_langchain/chat_models.py:430` · category=`bug`

- **Description:** LangChain's bind_tools() builds a payload with payload['tools']=bound_tools, and when federation hints are set adds federation_strategy/preferred_regions/spillover_enabled. These are passed positionally into DistLLMClient.chat_completions which does not accept them -> TypeError. LlamaIndex claims is_function_calling_model=True but never passes tools.
- **Impact:** The platform's core tool-calling story fails in production for every consuming adapter, contradicting LLMMetadata.is_function_calling_model=True and the distllm_chat tool provider contract.
- **Effort:** 3-5 hours
- **Dependencies:** Add tools + federation kwargs to DistLLMClient.chat_completions/_stream and forward into _build_chat_payload's body.
- **Timeline:** Day 2-4 of P1 sprint.
- **Verification note:** MUSTFIX (confirmed; bind_tools injects kwargs SDK does not accept)

---

### [High] 🔒 VERIFIED Dify integration double-prefixes /v1 and probes the wrong health path
`integrations/dify/distllm_provider.py:104` · category=`bug`

- **Description:** DistLLMProvider._get_client sets base_url to DISTLLM_API_BASE defaulting to http://localhost:8000/v1, but every request then prepends /v1/... again: client.post('/v1/chat/completions') -> http://localhost:8000/v1/v1/chat/completions, and the health path is wrong too.
- **Expected vs actual:** Expected: POST /v1/chat/completions on bare host. Actual: /v1/v1/chat/completions -> 404.
- **Impact:** Dify custom-provider plugin always fails credential validation and every inference/embed call hits 404/405, so the advertised Dify integration cannot run.
- **Effort:** 1-2 hours
- **Dependencies:** Drop the trailing '/v1' from default base_url and keep '/v1/...' path prefixes.
- **Timeline:** Day 2-3 of P1 sprint.
- **Verification note:** MUSTFIX (confirmed; Dify double-prefixes /v1)

---

### [High] 🔒 VERIFIED gRPC client is non-functional: imports a proto package that is never shipped and falls back to a service that does not exist on the server
`integrations/grpc_client/src/distllm_grpc/client.py:218` · category=`bug`

- **Description:** DistLLMGrpcClient tries `from distllm_grpc.proto import inference_pb2_grpc` (client.py:91) but no proto/ module exists in the grpc_client package (only client.py, cli.py, __init__.py), so _stub is always None. The JSON-over-channel fallback then calls a service name that doesn't exist on the real server.
- **Impact:** The advertised high-performance gRPC client cannot complete a single request against the real server; the documented 'falls back to REST if gRPC unavailable' claim is also false.
- **Effort:** 1-2 days
- **Dependencies:** Generate/ship matching inference.proto stubs or reimplement against existing NodeService; or real httpx REST fallback.
- **Timeline:** Day 4-6 of P1 sprint.
- **Verification note:** MUSTFIX (confirmed; gRPC client imports never-shipped proto pkg)

---

### [High] 🔒 VERIFIED Sync vs async chat streaming yield different item types (dict vs str) in both Python SDKs
`src/distllm/sdk/client.py:878` · category=`bug`

- **Description:** Async chat_completions_stream extracts delta.content and yields str; sync chat_completions_stream does 'yield from parse_sse_stream_sync(response)' and yields raw SSE dicts. A caller switching sync<->async gets different types from the same-named method (mirrored in sdk/src/distllm_sdk/client.py:656).
- **Expected vs actual:** Expected: both yield str deltas. Actual: async yields str, sync yields dict.
- **Impact:** Flipping sync<->async yields dicts instead of strings from the same method, causing runtime errors with no signposting; breaks the sync/async-pair promise.
- **Effort:** 1-2 hours
- **Dependencies:** None (shared extraction helper + parity test).
- **Timeline:** Day 3 of P1 sprint.
- **Verification note:** MUSTFIX (confirmed; sync vs async yield different item types)

---

### [High] 🔒 VERIFIED `distllm system doctor` never runs - argparse parses Typer subcommand tokens as stray positionals
`src/distllm/cli/doctor.py:680` · category=`bug`

- **Description:** The system_doctor Typer command forwards to doctor.main(), which calls parser.parse_args() on raw sys.argv. Invoked as `distllm system doctor`, sys.argv is ['distllm','system','doctor']; the parser (only options, no positionals) rejects 'system'/'doctor' -> argparse usage error, exit 2.
- **Expected vs actual:** Expected: `distllm system doctor` runs diagnostics. Actual: argparse error+exit 2.
- **Impact:** The primary single-command system diagnostic is completely non-functional from the advertised CLI surface; users cannot diagnose GPU/network/config issues with the documented command.
- **Effort:** 2-4 hours
- **Dependencies:** Add `def main(argv=None): parser.parse_args(argv)` and call it with a clean argv.
- **Timeline:** Day 1-2 of P1 sprint.
- **Verification note:** MUSTFIX (confirmed; `distllm system doctor` never runs)

---

### [High] 🔒 VERIFIED CLI error handling self-inconsistent: most failing commands exit 0 (masked failures), cli_error_handler is dead code
`src/distllm/cli/main.py:1385` · category=`bug`

- **Description:** Most commands catch errors, print a red line, and return normally, so Typer exits 0 on failure (system_slo_report 1385, federate_status 816-817, daas_status 1506-1507, draft_fleet_status 1609-1610, etc.). The cli_error_handler decorator (error_handler.py) is never applied.
- **Expected vs actual:** Expected: failing command exits non-zero. Actual: catches, prints, returns -> exit 0.
- **Impact:** CI pipelines, cron jobs, and shell scripts gating on exit code silently treat failed connections/configuration as successful; failures are masked.
- **Effort:** 2-4 hours
- **Dependencies:** Adopt one exit-code policy (typer.Exit(1)) and apply cli_error_handler to all commands.
- **Timeline:** Day 2-3 of P1 sprint.
- **Verification note:** MUSTFIX (confirmed; failing commands exit 0)

---

### [High] 🔒 VERIFIED CI test job is blocked: 79 live collection errors interrupt the whole suite
`pytest.ini:4` · category=`test`

- **Description:** The main CI job runs `pytest -v ... -m "not e2e and not slow and not chaos"` over tests/ (testpaths=tests). Collection aborts at 79 errors, so every PR test job fails regardless of the ~10017 passing tests.
- **Impact:** CI is red on every PR; the 10k-test suite cannot gate anything and the coverage gates (--cov-fail-under=80) are moot because the run never completes.
- **Effort:** 1-2 days
- **Dependencies:** Provision dev env (psutil etc.) + mark/migrate tests importing refactored-away private symbols.
- **Timeline:** Day 1-3 of P1 sprint (unblock all gating).
- **Verification note:** MUSTFIX (confirmed; 79 collection errors block the whole suite)

---

## Priority P2 (34)

### [Critical] 🔒 VERIFIED RedundantExecutor._run_redundant is a non-functional stub - enabling redundancy>1 always fails
`src/distllm/dist/redundant.py:96` · category=`bug`

- **Description:** The entire redundant speculative-parallelism path (the module's headline feature) is a stub: _run_redundant defines local _forward_request_to_proto and _process_forward_response_pb functions that unconditionally raise NotImplementedError; the caller's try/except swallows it, so results stays empty and NodeUnreachableError is raised.
- **Impact:** Enabling redundancy>1 always fails; the redundant speculative-parallelism feature is non-functional wherever it is engaged.
- **Effort:** 2-4 days
- **Dependencies:** Implement real redundant forward/reconcile or fail fast at config-validation time; requires a proto contract matching NodeService.
- **Timeline:** P2 (feature not on default path).
- **Verification note:** REAL-NOT-BLOCKING (confirmed: stub is a non-functional abort path but redundancy>1 is not on the default production path)

---

### [High] 🔒 VERIFIED Streaming layer-weight transfer has no integrity verification (checksum, ordering, completeness ignored)
`src/distllm/dist/node_client.py:500` · category=`bug`

- **Description:** The streaming path (intended for large models, up to 512MB) concatenates chunks and returns them with zero validation: it ignores chunk_index, total_chunks, is_final_chunk and never computes/compares the SHA-256 trailing-marker.
- **Impact:** Corrupted/truncated weight data is accepted silently and loaded as if valid - data-integrity gap on the large-model transfer path.
- **Effort:** 3-5 hours
- **Dependencies:** Validate chunk ordering/completeness + trailing-checksum on reassembly.
- **Timeline:** P2.
- **Verification note:** REAL-NOT-BLOCKING (confirmed: streaming has no integrity verification)

---

### [High] 🔒 VERIFIED download_layer_subset returns a directory containing only a manifest, no weights or index
`src/distllm/models/model_hub.py:220` · category=`bug`

- **Description:** Layer-aware download fetches shards via hf_hub_download WITHOUT local_dir (they land in the HF shared cache), writes only .layer_manifest into layer_subdir, and returns layer_subdir. The returned path has no model.safetensors*, config, or tokenizer files. The layer-aware download (a pooling-differentiator feature) is unusable.
- **Impact:** Layer-aware download (layer-per-node pooling) returns a path with no usable weights; anyone consuming it cannot load the partial model.
- **Effort:** 4-8 hours
- **Dependencies:** Write the shards under the layer_subdir or return the HF-cache paths.
- **Timeline:** P2.
- **Verification note:** REAL-NOT-BLOCKING (confirmed: download_layer_subset returns manifest-only dir)

---

### [High] 🔒 VERIFIED dynamic_sharder installs a new partition after migrations that never transferred data
`src/distllm/core/dynamic_sharder.py:303` · category=`bug`

- **Description:** In _migrate_layer the transfer step only runs `if self._on_transfer:`; with the default None it marks the layer COMPLETE without moving data. _execute_reshard then unconditionally installs new_partition, routing layers at nodes that never received the weights.
- **Impact:** Dynamic re-sharding points layers at nodes lacking the data - runtime load failures after any reshare.
- **Effort:** 3-5 hours
- **Dependencies:** Make transfer mandatory or block partition installation until transfer completes.
- **Timeline:** P2.
- **Verification note:** REAL-NOT-BLOCKING (confirmed: dynamic_sharder installs partition after no data transfer)

---

### [High] 🔒 VERIFIED NeuralBanditRouter crashes on instantiation: missing `import torch`
`src/distllm/core/learning_router.py:525` · category=`bug`

- **Description:** learning_router.py defines NeuralBanditRouter which uses torch.device/torch.nn/torch.optim/torch.tensor/torch.no_grad, but the module imports only hashlib/json/math/threading/dataclass/Path/TYPE_CHECKING (lines 27-38); a repo-wide grep for `import torch` never matches this file. Any instantiation raises NameError.
- **Impact:** The neural bandit router cannot be instantiated at all - crashes on use.
- **Effort:** under 1 hour
- **Dependencies:** Add `import torch`.
- **Timeline:** P2 (cheap; fix alongside any use).
- **Verification note:** REAL-NOT-BLOCKING (confirmed: NeuralBanditRouter has no `import torch`)

---

### [High] 🔒 VERIFIED StreamingKVTransfer breaks on bf16 tensors
`src/distllm/dist/streaming_kv_transfer.py:80` · category=`bug`

- **Description:** chunk_tensor calls t.numpy() directly on the tensor. For a bfloat16 tensor torch.Tensor.numpy() raises TypeError ('Got unsupported ScalarType BFloat16'), so bf16 KV transfers crash on send.
- **Impact:** bv16 KV transfers (a common dtype for the quantized KV path) crash on send.
- **Effort:** 1-2 hours
- **Dependencies:** Handle bf16 in chunk_tensor (cast to fp32 or pack-bytes) + reassemble map.
- **Timeline:** P2.
- **Verification note:** REAL-NOT-BLOCKING (confirmed: bf16 KV transfer crashes on numpy())

---

### [High] 🔒 VERIFIED Bandwidth congestion gating silently drops any transfer larger than the congestion window
`src/distllm/dist/pipeline/bandwidth_controller.py:650` · category=`bug`

- **Description:** PipelineTransportController.send() gates against congestion window (initial_cwnd=10*1460=14600 bytes). When payload exceeds the window it sends only the first `window` bytes and stores the remainder in _send_buf - but nothing ever drains _send_buf, so the tail is dropped.
- **Impact:** Silent truncation of any tensor/payload larger than the congestion window under gating.
- **Effort:** 2-4 hours
- **Dependencies:** Drain _send_buf as congestion clears (real ACK-based congestion control).
- **Timeline:** P2.
- **Verification note:** REAL-NOT-BLOCKING (confirmed: bandwidth congestion drops any transfer > cwnd)

---

### [High] 🔒 VERIFIED LearnedCostModel train/serve feature skew - intermediate_size and flops features differ between training and inference
`src/distllm/dist/partition/learned_cost.py:90` · category=`bug`

- **Description:** FeatureExtractor.extract (serving time) initializes intermediate_size=0 and never assigns it (the loop sets only hidden_size), so feature[14] is always 0.0. Training uses _observation_to_features which sets the real intermediate_size and proxy FLOPS - the model trains on different feature distributions than it serves.
- **Impact:** Learned cost model makes inaccurate latency predictions at serve time because feature[14] is always 0.
- **Effort:** 2-4 hours
- **Dependencies:** Align extract and _observation_to_features feature definitions.
- **Timeline:** P2.
- **Verification note:** REAL-NOT-BLOCKING (confirmed: FeatureExtractor never sets intermediate_size)

---

### [High] 🔒 VERIFIED DiffusionPipeline.load combines enable_model_cpu_offload with nn.DataParallel - the multi-GPU path is broken
`src/distllm/core/hydra_diffusion.py:41` · category=`bug`

- **Description:** In load(), when num_gpus>1 the code calls pipe.enable_model_cpu_offload() then wraps pipe.unet in torch.nn.DataParallel before pipe.to('cuda'). DataParallel requires the module on a single CUDA device while CPU offload registers hooks that move params - the two are incompatible.
- **Impact:** Multi-GPU diffusion load is broken (DataParallel vs accelerate CPU-offload hooks conflict), so the multi-GPU path never works.
- **Effort:** 1-2 days
- **Dependencies:** Pick one strategy (DataParallel OR accelerate offload), not both.
- **Timeline:** P2.
- **Verification note:** REAL-NOT-BLOCKING (confirmed: load() combines CPU offload with DataParallel)

---

### [High] 🔒 VERIFIED Kademlia routing table trusts sender-supplied IP:port over packet source
`src/distllm/dist/p2p/kademlia_dht.py:826` · category=`security`

- **Description:** PING, STORE, FIND_NODE and FIND_VALUE handlers build the KademliaNode from the sender-declared 'ip'/'port'/node_id in the JSON body instead of the actual UDP source addr. A spoofed datagram can add an arbitrary (node_id, victim_ip:port) entry to the routing table.
- **CVSS:** 7.5
- **Impact:** Routing-table poisoning / address spoofing that can direct lookups and peers to attacker- or victim-chosen endpoints.
- **Effort:** 3-5 hours
- **Dependencies:** Derive node identity/address from the packet source and validate node_id reachability.
- **Timeline:** P2.
- **Verification note:** REAL-NOT-BLOCKING (confirmed: routing table trusts sender-supplied IP:port)

---

### [Medium] Latent arbitrary file write via upload filename (absolute path / ../) in routes/files.py; router unmountable because require_coordinator is undefined
`src/distllm/api/routes/files.py:135` · category=`security`

- **Description:** upload_file builds the output path as file_path = upload_dir / (file.filename or 'unnamed') with no sanitization (routes/files.py:135). In pathlib, Path(a)/'/abs' yields the absolute path '/abs' and Path(a)/'../x' escapes the directory. Also require_coordinator is undefined so the router cannot mount.
- **CVSS:** 8.8
- **Impact:** Server-side arbitrary file write (Code Execution potential) if the route is ever mounted; broken/unusable route definitions across 5 modules.
- **Effort:** 3-5 hours
- **Dependencies:** Sanitize filename (basename, no path separators) + define/import require_coordinator.
- **Timeline:** P2 (route currently unmountable, so the write is latent).
- **Verification note:** UNVERIFIED (High-impact unverified security finding)

---

### [Medium] API key authentication runs PBKDF2 (100k iters) per stored key per request before any rate-limit short-circuit - CPU-exhaustion DoS + timing side channel
`src/distllm/core/api_key_store.py:145` · category=`security`

- **Description:** ApiKeyStore.authenticate re-hashes the presented token with PBKDF2-SHA256 (100,000 iterations) against EVERY stored key's salt in a loop before any limiter short-circuits; the middleware calls authenticate() first and only later inspects limits.
- **CVSS:** 5.9
- **Impact:** DoS amplification proportional to key count; minor timing oracle on key identity. Rate limiter reduces but does not eliminate per-request cost.
- **Effort:** 4-8 hours
- **Dependencies:** Rate-limit/short-circuit before PBKDF2; or hash a cheap fast index first.
- **Timeline:** P2.
- **Verification note:** UNVERIFIED (High-impact unverified security finding)

---

### [Medium] NodeServer.start silently fails open to plaintext when TLS is requested but cert/key are missing
`src/distllm/dist/node_service.py:466` · category=`security`

- **Description:** use_tls=True only produces a secure port if cert_file AND key_file are also provided; otherwise it silently falls through to add_insecure_port on 0.0.0.0. worker.main() defaults use_tls = not insecure (TLS requested by default), so a worker with incomplete TLS config serves plaintext.
- **CVSS:** 7.5
- **Impact:** Node-to-node tensors (hidden states, KV, weights) transit over the LAN/Internet in plaintext whenever TLS config is incomplete or on failover.
- **Effort:** 2-4 hours
- **Dependencies:** Raise/refuse startup when use_tls but cert/key missing; fail closed.
- **Timeline:** P2.
- **Verification note:** UNVERIFIED (High-impact unverified security finding)

---

### [Medium] E2E tensor encryption in ForwardPass is dead code - _e2e is always None, tensors transit in plaintext
`src/distllm/dist/node_service.py:199` · category=`security`

- **Description:** NodeServicer wraps most tensor fields in encrypt_tensor_payload/decrypt_tensor_payload, but NodeServer.start (line 448) constructs NodeServicer(...) with no e2e_encryption, so self._e2e is always None; both wrappers return raw bytes unchanged.
- **CVSS:** 6.5
- **Impact:** An operator who reads ForwardPass and believes node-to-node tensors are E2E-encrypted gets plaintext on the wire; any network observer reads the data.
- **Effort:** 2-4 hours
- **Dependencies:** Actually construct e2e_encryption in start(), or remove the misleading wrappers.
- **Timeline:** P2.
- **Verification note:** UNVERIFIED (High-impact unverified security finding)

---

### [Medium] E2E tensor transport fails open to plaintext when PyNaCl is absent or a session isn't established
`src/distllm/security/e2e.py:425` · category=`security`

- **Description:** encrypt_tensor_payload (module-level wrapper) returns raw_bytes UNMODIFIED when e2e is None or not established, logging one warning. In a federated deployment requiring confidentiality, a node without PyNaCl silently ships raw tensor plaintext.
- **CVSS:** 6.5
- **Impact:** Tensors intended to be confidential can be transmitted unencrypted without any hard failure.
- **Effort:** 2-4 hours
- **Dependencies:** Raise when confidentiality-required but no session established.
- **Timeline:** P2.
- **Verification note:** UNVERIFIED (High-impact unverified security finding)

---

### [Medium] Unknown tenant_ids bypass quota entirely; served queued requests never consume tokens (multi-tenant quota evasion)
`src/distllm/dist/quota_enforcer.py:160` · category=`security`

- **Description:** Both quota layers fail open for unregistered tenant_ids: QuotaEnforcer.try_consume (line 160) and MultiTenantSLOEnforcer.should_admit (multi_tenant.py:183) return True for any tenant_id not in the table. If tenant_id derives from a client-supplied value, any client can dodge throttling by rotating/omitting tenant_id.
- **CVSS:** 6.5
- **Impact:** Tenant isolation is mostly advisory; any client can evade throttling, and queued traffic is billed/rate-limited incorrectly.
- **Effort:** 3-5 hours
- **Dependencies:** Default unknown tenants to a bounded default quota; charge queued/consumed requests.
- **Timeline:** P2.
- **Verification note:** UNVERIFIED (High-impact unverified security finding)

---

### [Medium] Tauri chat and benchmark call the REST API with no Authorization header while the admin layer uses a bearer token
`tauri/src/lib/api.ts:111` · category=`security`

- **Description:** Tauri's streamChatCompletion and runBenchmark (api.ts) fetch $baseUrl/v1/chat/completions from the webview with only Content-Type, no auth header. The Rust admin/cluster commands do send state.auth_token as Bearer on /admin/v1/* calls.
- **CVSS:** 5.3
- **Impact:** Desktop chat/benchmark break (401) when the server enforces auth; inconsistent auth handling between the webview REST path and the Rust command path.
- **Effort:** 2-4 hours
- **Dependencies:** Pass state.auth_token to the chat/benchmark fetches.
- **Timeline:** P2.
- **Verification note:** UNVERIFIED (High-impact unverified security finding)

---

### [Medium] Cluster-key secret file (~/.distllm/cluster_key) has no permission hardening and is never created by the CLI
`src/distllm/cli/main.py:572` · category=`security`

- **Description:** The CLI reads the shared cluster auth key from ~/.distllm/cluster_key but never creates that file or restricts permissions; a grep finds zero chmod/umask/0o600 usage. The sibling cert path does harden private keys.
- **CVSS:** 4.9
- **Impact:** The cluster auth key file is read with no mode check and never created under restricted permissions (unlike cert private keys which are chmod 0o600'd); a local attacker can read the shared cluster credential.
- **Effort:** 1-2 hours
- **Dependencies:** chmod 0o600 on write of cluster_key.
- **Timeline:** P2.
- **Verification note:** UNVERIFIED (High-impact unverified security finding)

---

### [Medium] _verify_download_integrity computes SHA-256 against nothing - it can never detect corruption
`src/distllm/models/model_hub.py:418` · category=`bug`

- **Description:** The integrity verifier hashes each .safetensors file but never compares the digest to any expected value, so it never warns/fails on truncation or bit-rot (the docstring claims it logs warnings for mismatches, but there is no expected-hash source).
- **Impact:** Corrupted shards silently treated as valid; a truncated/bit-flipped model file passes 'verification'.
- **Effort:** 3-5 hours
- **Dependencies:** Compare against HF-provided sha256 (from hub metadata / snapshot hash).
- **Timeline:** P2.
- **Verification note:** UNVERIFIED (important operational bug)

---

### [Medium] NTK scaled rope_theta never applied: key 'theta' vs 'rope_theta' mismatch
`src/distllm/models/rope_scaling.py:111` · category=`bug`

- **Description:** build_rope_scaling_config stores the NTK-scaled base as key 'rope_theta', but apply_rope_scaling checks `if 'theta' in rope_config`, which is never true, so model.config.rope_theta is never updated and keeps 10000.
- **Impact:** Long-context NTK path is silently broken; positions beyond original max degrade/fail.
- **Effort:** 1-2 hours
- **Dependencies:** None (align the key names).
- **Timeline:** P2.
- **Verification note:** UNVERIFIED (important operational bug)

---

### [Medium] adapter quantize_int8 is a cosmetic no-op: re-dequantizes in place, saves zero VRAM
`src/distllm/models/adapter.py:338` · category=`bug`

- **Description:** quantize_adapter rounds params to int8, stores metadata, then immediately writes back (quantized.to(float32) * scale) to param.data, so the live model keeps float tensors and no VRAM is freed - only rounding noise introduced.
- **Impact:** Adapters advertised as int8 consume identical VRAM and lose precision; multi-tenant density/eviction-headroom claims unmet.
- **Effort:** 3-6 hours
- **Dependencies:** Keep packed int8 + scale for parameters actually deployed.
- **Timeline:** P2.
- **Verification note:** UNVERIFIED (important methodological bug)

---

### [Medium] CoordinatorFailoverHandler never touches the HA election protocol it claims to use
`src/distllm/core/coordinator_failover.py:167` · category=`bug`

- **Description:** coordinator_failover.py's docstring and class name promise HA-protocol-aware failover, but _trigger_failover/_check_tcp_alive only do raw socket.create_connection reachability and pick the first accepting TCP peer as the 'new coordinator', ignoring leader identity/quorum.
- **Impact:** If ever wired, workers would fail over to an arbitrary TCP-listening peer rather than the elected leader, breaking the quorum; today the module is a dead, wrong abstraction.
- **Effort:** 4-6 hours
- **Dependencies:** Integrate with CoordinatorElection is_leader + quorum.
- **Timeline:** P2.
- **Verification note:** UNVERIFIED (important operational bug, HA-related)

---

### [Medium] max_concurrent_requests quota is a check-then-act race (violated under concurrency)
`src/distllm/core/usage_meter.py:357` · category=`bug`

- **Description:** check_quota reads self._concurrent.get(tenant_id,0) WITHOUT holding self._lock and returns 'ok' if below the cap; enforce_quota then increments under the lock. Between the unlocked read and the locked increment, N concurrent requests all observe below-cap.
- **Impact:** Concurrency caps intended to protect a tenant/backend can be breached by simultaneous bursts - resource oversubscription and cost overrun beyond the configured limit.
- **Effort:** 2-4 hours
- **Dependencies:** Make check-and-increment atomic under the lock.
- **Timeline:** P2.
- **Verification note:** UNVERIFIED (quota-correctness bug)

---

### [Medium] Monthly cost-budget enforcement compares against all-time total_cost, never reset per billing period
`src/distllm/core/usage_meter.py:336` · category=`bug`

- **Description:** tenant.total_cost accumulates every record forever (no period boundary in record_request), and check_quota compares quota.cost_budget_per_month against that lifetime total. Once a tenant's cumulative spend crosses the monthly budget it stays blocked forever (unless overage_allowed).
- **Impact:** A tenant that legitimately exhausts one month's budget is permanently denied service in subsequent months - availability bug for paid deployments and a cost-accounting error.
- **Effort:** 2-4 hours
- **Dependencies:** Reset cost per billing period (or store period-bucketed totals).
- **Timeline:** P2.
- **Verification note:** UNVERIFIED (billing-correctness bug)

---

### [Medium] to_proto_tensor reads CUDA memory before the async copy stream is synchronized
`src/distllm/dist/pipeline/serialization.py:59` · category=`bug`

- **Description:** For CUDA tensors the device->host copy is issued on a dedicated copy stream with non_blocking=True, then immediately read via numpy(force=True). numpy synchronizes the tensor's own stream, NOT the copy stream (GPUDirectSerializer handles this with copy_stream synchronization - this path does not).
- **Impact:** Latent, nondeterministic corruption of GPU-resident hidden states / KV caches during cross-node transfer on CUDA systems, producing wrong inference.
- **Effort:** 1-2 hours
- **Dependencies:** Synchronize the copy stream before reading memory.
- **Timeline:** P2.
- **Verification note:** UNVERIFIED (data-corruption bug on CUDA)

---

### [Medium] Base PagedAttentionManager.get_kv_cache concatenates full blocks without slicing by num_tokens, returning zero-padded tail garbage
`src/distllm/backends/paged_attention.py:427` · category=`bug`

- **Description:** get_kv_cache does torch.cat(keys, dim=2) over full KVCacheBlock.key_cache tensors (each block size max_tokens). For a sequence whose last block is partially filled, the concatenated tensor includes unused zero slots.
- **Impact:** Attention over KV reads zero-padded invalid tokens, corrupting generation for any sequence with a partially-filled final block and inflating memory reads.
- **Effort:** 1-2 hours
- **Dependencies:** Slice each block by num_tokens before cat.
- **Timeline:** P2.
- **Verification note:** UNVERIFIED (correctness bug)

---

### [Medium] Spot orchestrator silently substitutes fabricated static market listings when a provider API fails
`src/distllm/cloud/spot_orchestrator.py:964` · category=`bug`

- **Description:** _SaladProvider._list_fallback returns hardcoded static GPU entries and prices whenever the real Salad containers API raises, and list_instances returns those fabricated listings as normal; find_cheapest then returns them and launch_cluster books them.
- **Impact:** Real expenditure on nonexistent GPU inventory during provider hiccups; worsens the spot-orchestrator 'find cheapest then book' risk.
- **Effort:** 2-3 hours
- **Dependencies:** Return an explicit error (fail loudly) on provider API failure.
- **Timeline:** P2.
- **Verification note:** UNVERIFIED (operational/cost bug)

---

### [Medium] federate train reports success but submits a nonexistent adapter file and swallows merge errors
`src/distllm/cli/main.py:767` · category=`bug`

- **Description:** federate train submits a hardcoded, never-created adapter path (/tmp/distllm-federated/{adapter}.pt) and swallows coordinator/merge failures with bare `except Exception: pass`, so the command can print 'Federated training complete' without producing an artifact.
- **Impact:** The coordinator receives a submit payload pointing at a file never written; the federation round is recorded against a non-existent artifact and the merge is silently skipped.
- **Effort:** 2-4 hours
- **Dependencies:** Create the adapter path or fail loudly; surface merge errors.
- **Timeline:** P2.
- **Verification note:** UNVERIFIED (CLI correctness)

---

### [Medium] BaseToolProvider and model_router API calls never send the api_key, breaking tool discovery/calls on auth-required clusters
`src/distllm/integrations/_common/base_tool_provider.py:77` · category=`bug`

- **Description:** BaseToolProvider stores self._client with an api_key but discover_tools, discover_tools_from_openapi, and call_tool bypass it and use bare httpx.get/post with only Content-Type. DistLLMModelRouter.discover_models/auto_route similarly omit the key.
- **Impact:** All LangChain/CrewAI tool-provider discovery and routing silently degrades to defaults and errors on secured deployments, defeating the api_key passed into constructors.
- **Effort:** 1-2 hours
- **Dependencies:** Attach the api_key header to bare httpx calls.
- **Timeline:** P2.
- **Verification note:** UNVERIFIED (integration bug)

---

### [Medium] validate_http_url validates a resolved IP but returns the original hostname URL (DNS-rebinding TOCTOU)
`src/distllm/security/utils.py:62` · category=`security`

- **Description:** validate_http_url resolves the hostname via getaddrinfo, validates the resolved IPs against private ranges, then returns the ORIGINAL URL string (hostname-based). A caller taking the OK then opening the returned hostname URL is subject to DNS rebinding.
- **CVSS:** 6.1
- **Impact:** SSRF/DNS-rebinding bypass when a caller follows the validate-then-open pattern instead of safe_urlopen.
- **Effort:** 2-4 hours
- **Dependencies:** Return/require the validated IP (or require use of safe_urlopen).
- **Timeline:** P2.
- **Verification note:** UNVERIFIED (security finding)

---

### [Medium] Valid JWT without a role claim is rejected with 403 instead of granted read-only access
`src/distllm/plugins/auth_plugin.py:417` · category=`bug`

- **Description:** In _validate_jwt_from_context, a valid JWT lacking a role claim returns the string 'read' (whose comment claims it grants minimum read-only access), but 'read' is not a key in ROLE_PRIVILEGES, so the fallback maps to no privileges and the request is denied 403.
- **Impact:** Any bearer-JWT client whose token has no recognized 'role' claim cannot get the read-only access it is entitled to - functional authorization denial.
- **Effort:** 30 min - 1 hour
- **Dependencies:** Map 'read' (or absence of role) to the read-only privilege set.
- **Timeline:** P2.
- **Verification note:** UNVERIFIED (authorization bug)

---

### [Medium] Verification hash-registry compares raw float bytes of logits - guaranteed to mismatch on distributed runs, misleading CI signals
`src/distllm/verification/hash_registry.py:64` · category=`bug`

- **Description:** compute_output_hash hashes tensor.float().numpy().tobytes(). The distributed path applies INT8 quantize/dequant between pipeline stages, so float-represented bytes differ across nodes/stages. The registry then prints a misleading 'Hash registry' pass_rate.
- **Impact:** CI/dashboards relying on the hash pass_rate see false negatives for the exact drift the harness is meant to detect, eroding trust in verification.
- **Effort:** 3-5 hours
- **Dependencies:** Hash after dequantization to a common dtype, or hash a tolerance-aware representation.
- **Timeline:** P2.
- **Verification note:** UNVERIFIED (verification-methodology bug)

---

### [Medium] OIDC/OAuth2 SSO state & nonce stores grow unboundedly with no TTL reaper; GenericOAuth2Handler skips CSRF state check on empty state
`src/distllm/api/sso_auth.py:441` · category=`security`

- **Description:** OIDCHandler._state_store/_nonce_store and GenericOAuth2Handler._state_store grow one entry per get_login_url() call and are removed only when handle_callback pops the exact state - no TTL reaper. Attackers can bloat the stores by requesting many login URLs; also GenericOAuth2Handler skips the CSRF state check on empty state.
- **CVSS:** 6.1
- **Impact:** Memory-exhaustion DoS on a reachable login flow; weakened CSRF enforcement at the handler layer.
- **Effort:** 2-3 hours
- **Dependencies:** Add TTL reaping; require non-empty CSRF state in handle_callback.
- **Timeline:** P2.
- **Verification note:** UNVERIFIED (DoS/security finding)

---

### [Medium] middleware.py logs the full generated API key to the logger at every startup, contradicting its own fingerprint-only comment
`src/distllm/api/middleware.py:165` · category=`security`

- **Description:** In _get_or_generate_api_key (middleware.py:139-171), when API_KEY is unset the server generates a random 48-byte key and, despite the comment at 157 explicitly stating 'Log a fingerprint, not the full key', emits the full key to the logger.
- **CVSS:** 4.4
- **Impact:** Generated admin credential persisted in cleartext in logs/aggregators; if logs leak, full API access is disclosed.
- **Effort:** 1-2 hours
- **Dependencies:** None (log a fingerprint instead; the intent comment already exists).
- **Timeline:** P2.
- **Verification note:** UNVERIFIED (security finding)

---

## Priority P3 (9)

### [Medium] RequestLatencyTracker._completed grows without bound (memory leak)
`src/distllm/core/request_latency.py:71` · category=`bug`

- **Description:** complete() appends every finished RequestLatencyInfo to self._completed forever; consumers only read the last 50-100 entries, so the historical tail is never needed yet accumulates one entry per completed request.
- **Impact:** Per-request metadata accumulates indefinitely on long-running servers; stable-state heap grows linearly with requests served.
- **Effort:** 0.25 hours
- **Dependencies:** None (bounded deque / prune to a max).
- **Timeline:** P3 (low magnitude but trivial fix).
- **Verification note:** UNVERIFIED (minor memory leak)

---

### [Medium] resource_manager._tcp_health_check claims a zero-byte send but performs no probe
`src/distllm/core/resource_manager.py:390` · category=`bug`

- **Description:** The method only does pool.get() + settimeout() + pool.put() and returns True. A stale socket already in the connection pool is returned 'healthy' with no actual connectivity probe.
- **Impact:** Could keep requests routed to a dead node or fail failover, since a stale-but-pooled connection is trusted as a health signal.
- **Effort:** 0.5 hours
- **Dependencies:** Issue an actual zero-byte send/recv probe.
- **Timeline:** P3.
- **Verification note:** UNVERIFIED (minor health-check bug)

---

### [Medium] In _promote_pending the prefill-budget rejection is dead code: a candidate whose chunk exceeds remain_p is accepted
`src/distllm/core/batch_scheduler.py:806` · category=`bug`

- **Description:** The reject clause `if chunk > remain_p and remain_t - chunk < 0:` can never be true: every accepted candidate already satisfies c_tokens <= remain_t, and chunk <= c_tokens, so the prefill-token budget (an anti-decode-starvation control) is silently ignored.
- **Impact:** The prefill-token budget is silently ignored for the common case, letting prefill eat into decode slots under load.
- **Effort:** 0.5 hours
- **Dependencies:** Fix the condition to actually respect remain_p.
- **Timeline:** P3.
- **Verification note:** UNVERIFIED (dead-code budget control)

---

### [Medium] DisaggregatedRouter load counters are mis-accounted on pool fallback
`src/distllm/core/unified_router.py:388` · category=`bug`

- **Description:** When a request falls back pools, route() adds load to the fallback_pool dict but release() decrements based on the requested phase, so the fallback pool's load is never decremented and the preferred pool's load is - counters drift.
- **Impact:** Least-loaded routing becomes wrong in disaggregated deployments that allow fallback; nodes in the fallback pool get stuck with inflated load and are skipped.
- **Effort:** 0.5-1 hours
- **Dependencies:** Track the actual phase used for both add and release.
- **Timeline:** P3.
- **Verification note:** UNVERIFIED (load-accounting bug)

---

### [Critical] 🔒 VERIFIED DedupMiddleware vs AuthMiddleware ordering - verification determined the outer PluginHook layer rejects unauthenticated requests first, so the bypass is not exploitable
`src/distllm/api/dedup.py:142` · category=`security`

- **Description:** The claim: DedupMiddleware (registered L583) runs outside AuthMiddleware (L568) and returns synthetic cached responses on dedup hit, skipping auth/quota/rate-limit. Adversarial verification: PluginHookMiddleware is registered even later (L906) so it runs OUTSIDE DedupMiddleware; AuthPlugin._enforce_rbac rejects any request with empty api_key_role (HTTP 401) before DedupMiddleware's cache-hit short-circuit can serve content, because request.state.api_key_role is read at the outer plugin layer. So the cached-output-to-unauthenticated path is not reachable.
- **Impact:** None - the reported auth bypass is contained by the outer plugin RBAC layer. Documented to prevent re-investigation; keep the middleware-ordering note as defense-in-depth if PluginHook ordering ever changes.
- **Effort:** None required (optional: add an explicit ordering comment/test).
- **Dependencies:** N/A.
- **Timeline:** P3 (no action; optional defensive comment).
- **Verification note:** REFUTED - verified NOT a real bug (needs no fix).

---

### [High] 🔒 VERIFIED QUIC transport message framing - verification determined frames are reassembled across StreamDataReceived events, so >datagram messages are not corrupted
`src/distllm/dist/p2p/quic_transport.py:155` · category=`bug`

- **Description:** The claim: each StreamDataReceived event parsed as a self-contained frame truncates payloads >~1200 bytes. Adversarial verification determined the parser accumulates buffered frames across events (length-prefix + buffer carry-over), so multi-datagram messages are reassembled correctly.
- **Impact:** None - multi-datagram QUIC messages transmit correctly. Documented to prevent wasted fixes.
- **Effort:** None
- **Dependencies:** N/A.
- **Timeline:** P3 (no action).
- **Verification note:** REFUTED - verified NOT a real bug (chunked QUIC parsing is functionally correct).

---

### [High] 🔒 VERIFIED Test-infra fake-package shim whitelist - empirical verification shows the claimed-broken config names import fine
`tests/_import_helper.py:56` · category=`bug`

- **Description:** The claim: the fake-package shim whitelists a stale symbol list, breaking imports of ModelSettings/CachePersistenceSettings/ChatRouterSettings/RebalancerSettings. Adversarial verification bootstrapped via _import_helper.py and all four names import cleanly from distllm.config.
- **Impact:** None for these specific names - imports work. (Other collection errors in the CI-suite finding remain real; this one is refuted.)
- **Effort:** None
- **Dependencies:** N/A.
- **Timeline:** P3 (no action).
- **Verification note:** REFUTED - verified NOT a real bug (the four claimed-broken config names import cleanly).

---

### [High] Two parallel SSO implementations under src/distllm/api: sso_auth.py (wired) vs auth/ (duplicated, diverging)
`src/distllm/api/auth/oidc.py:114` · category=`architecture`

- **Description:** sso_auth.py (the wired SSO path) and a parallel src/distllm/api/auth/* OIDC implementation exist and have already diverged. Two sources of truth for SSO/auth increases bypass surface and maintenance burden.
- **Impact:** Divergent SSO behavior across code paths; risk that a security fix lands in one and not the other; confusing for operators.
- **Effort:** 1-2 days
- **Dependencies:** Consolidate onto one SSO implementation after confirming the wiring.
- **Timeline:** P3.
- **Verification note:** UNVERIFIED (High architecture/duplication - strategic clean-up)

---

### [High] Coordinator lifecycle (start/stop/defrag/_start_subsystem) duplicated wholesale between coordinator.py and coordinator_subsystem.py
`src/distllm/core/coordinator_subsystem.py:239` · category=`architecture`

- **Description:** The coordinator's subsystem start/stop/defrag logic exists in two places and has drifted. Dual implementations risk silent divergence where one path updates subsystems and the other does not.
- **Impact:** HA/subsystem bugs can land in one copy and not the other; increased maintenance and test surface.
- **Effort:** 1-2 days
- **Dependencies:** Refactor to a single lifecycle helper.
- **Timeline:** P3.
- **Verification note:** UNVERIFIED (High architecture - coverage/dead-code clean-up)

---
