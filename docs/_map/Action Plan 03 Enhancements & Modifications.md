---
tags:
  - audit
  - action-plan
date: 2026-08-11
---

# Action Plan 03 — Enhancements & Modifications

**← [[Exhaustive Audit 2026-08-11]]**


16 concrete, code-grounded enhancements distilled from the strongest (mostly adversarially-verified) audit findings. The single highest-leverage theme is API-gateway correctness: the Starlette middleware stack in server.py is wired in a fragile reverse-registration order with several verified-pain consequences (request dedup runs OUTSIDE auth/rate-limit/prompt-injection and returns synthetic responses; RED metrics are never recorded; the middleware that would record them is never even added; SSO is duplicated across two divergent trees). Core themes: freeze/correct latency & telemetry accounting (request_latency elapsed grows with wall clock; telemetry.flush deadlocks; usage_meter under-reports after 100k), wire un-plumbed coordinator subsystems (autoscaler feed+actuate, HA leader gating), consolidate duplicate dedup/cache/speculative implementations (3 dedup+cache engines, 4+ speculative verifiers, 2 Python SDKs), and fix CLI/UX failures (doctor never runs, masking exit codes). Every entry names the component, the current state with file:line evidence, why it matters, a concrete implementation path, measurable benefits, trade-offs, effort, and priority. P0 items block production trust/correctness; P1 are consolidation/UX; P2 are parity/quality.

**17 enhancements**

### E1. Declarative, ordered middleware stack with ordering-invariant regression tests — `api-gateway / src/distllm/api/server.py` [priority P0]
- **Current state:** The stack is built imperatively via ~17 app.add_middleware() calls (server.py L255-L954) that execute in REVERSE registration order (comment at L526-527). The order is load-bearing but undocumented and only implicit: DedupMiddleware (registered L583) executes OUTSIDE AuthMiddleware/L568, RequestRateLimitMiddleware/L578 and PromptInjectionMiddleware/L589, so on a dedup cache hit (dedup.py L139-147) auth, per-IP rate-limit and prompt-injection are all skipped; ObservabilityMiddleware is defined (observability_middleware.py L25) but never added via add_middleware anywhere (grep shows only the class + docstring).
- **Why:** Middleware ordering is the root of several verified findings (id 1 dedup-before-auth residual cross-tenant serving; skipped quota/rate-limit on cache hits; dead RED metrics because the measuring middleware is unwired). A single move to the wrong side of auth silently breaks every auth gate, and there is no test asserting the order.
- **Implementation approach:** (1) Define a single ordered ABN constant MIDDLEWARE_ORDER = [(cls, kwargs), ...] in registration order; replace the imperative app.add_middleware() block with a loop that registers in ABN order, with a clear comment that execution is reverse. (2) Add DISTLLM_MIDDLEWARE_ORDER debug env that dumps the effective (execution) chain at startup via app.user_middleware. (3) Add tests/api/test_middleware_order.py with a tiny probe middleware that appends to a list, asserting: Timeout outermost-of-inner trio, Auth outside RequestRateLimit, Dedup INSIDE Auth (execution), PromptInjection inside Auth, PluginHook outermost overall, Tracing outermost. (4) Fix dedup so a cache hit re-enters the post-auth accounting path (see tenant-dedup entry). (5) Gate CI on these ordering tests.
- **Benefits:** - Makes auth/rate-limit/prompt-injection guarantees explicit and CI-enforced, eliminating silent regression of the highest-severity class of findings
- Removes the 'reverse-registration' footgun that caused findings 1, 72 and the unwired-ObservabilityMiddleware bug
- Provides operators a single place to reason about and reorder the pipeline
- Observable startup log of the effective chain aids debugging
- **Trade-offs:** - Refactor touches the inner middleware block; needs a full API test run to re-confirm 401/429/403 invariants
- A declarative registry reduces some add_middleware flexibility for runtime-conditional middleware (e.g. SSO setup_sso, moderation)
- Introspection via user_middleware requires handling BaseHTTPMiddleware vs pure-ASGI classes distinctly
- **Effort:** 1-2 days

---

### E2. Tenant-namespaced, auth-aware request dedup that flows through accounting — `api-gateway / src/distllm/api/dedup.py` [priority P0]
- **Current state:** DedupMiddleware fingerprints only the raw body: `hashlib.sha256(body).hexdigest()` (dedup.py L35-36), with no tenant/identity component. On a hit it returns a synthetic Response directly (L139-147) and never calls call_next, so the inner Starlette auth (AuthMiddleware L568), per-IP RequestRateLimitMiddleware (L578), quota (quota_middleware), and PromptInjectionMiddleware (L589) are skipped (finding id 1; verified reason confirmed the ordering is real, leaving an authenticated cross-tenant response-serving + skipped per-request accounting as the verified residual).
- **Why:** Body-only fingerprinting means any tenant that POSTs the same JSON within the 3600s TTL receives another tenant's exact generated output, and cache hits dodge quota/rate-limit accounting. Even though the plugin layer currently rejects unauthenticated requests first, the authenticated cross-tenant leak and metering bypass are genuine.
- **Implementation approach:** (1) Incorporate request.state.api_key_id (set by AuthMiddleware L305-306) into the fingerprint: fp = sha256(body + b'|' + tenant_key). Only dedup within the same owner. (2) Instead of returning a raw Response on hit, still run call_next, but set request.state.dedup_cached = True and return the cached bytes from within the (now inner) accounting path so Auth/rate-limit/quota still count the request. (3) Add an X-Dedup-Cache: HIT header and a dedup HIT/MISS counter on the exporter. (4) Regression test: same body from two different api_key_ids must NOT hit each other's cache entry; a dedup hit must still increment request counters.
- **Benefits:** - Eliminates cross-tenant cached-response disclosure (verified residual of finding 1)
- Cache hits are metered (quota/rate-limit honored), fixing silent billing/abuse-evasion
- Preserves the large latency win of collapsing repeated prompts
- Observable dedup hit-rate for capacity/cost planning
- **Trade-offs:** - Tenant-scoped keys reduce global dedup hit rate (only same-owner dedup)
- Re-running call_next on a cache hit slightly changes middleware timing semantics
- Needs to keep the working in-flight wait path (L144-147) intact
- **Effort:** 4-8 hours

---

### E3. Wire the RED metrics middleware and actually record counters/histograms — `observability / src/distllm/api/observability_middleware.py + exporter.py` [priority P0]
- **Current state:** ObservabilityMiddleware._record_metrics (observability_middleware.py L117-137) calls exporter.requests_total.labels(...) (L124-126), request_latency.labels(...) (L127-129), request_duration_seconds.labels(...) (L130-132), errors_total.labels(...) (L133-136) but never calls .inc()/.observe() — .labels() only selects a series. requests_total/errors_total are Counters and request_latency/request_duration_seconds are Histograms (exporter.py L17-29, L103-128), so no data is ever emitted (finding 72, verified must-fix). It is also not registered: grep for add_middleware(Observability) returns nothing, so even a correct class never runs.
- **Why:** RED metrics are the primary operational signal (rate/errors/duration). They are entirely dead today: dashboards and SLO alerts on distllm_requests_total / *latency* / *_errors_total read zeros. The API otherwise emits cost/gpu gauges (which DO call .inc properly at L143-149), highlighting the gap.
- **Implementation approach:** (1) In server.py, app.add_middleware(ObservabilityMiddleware, metrics_exporter=state.metrics_exporter, tracer=...) registered outermost (after tracing). (2) Fix _record_metrics: requests_total.labels(...).inc() on every request; request_latency.labels(...).observe(duration); request_duration_seconds.labels(...).observe(duration) (keep both for back-compat); errors_total.labels(type=status_class,...).inc() on 4xx/5xx. (3) Add a unit test asserting generate_metrics() grows after a request. (4) Align with the latency-entry below so elapsed reflects true completion time.
- **Benefits:** - Unblocks live RED dashboards/alerting that currently read zero
- Correct per-status error counters for SLO/incident triage
- Histogram latency percentiles become trustworthy
- Closes the observability gap flagged across ids 72, 74
- **Trade-offs:** - Prometheus label cardinality grows with tenant/model/method — must bound tenant set
- Double-recording both request_latency and request_duration_seconds duplicates storage
- Middleware must be positioned so it records after dedup short-circuits
- **Effort:** 0.5-1 day

---

### E4. Freeze elapsed_ms for completed requests in RequestLatencyTracker (fix SLA percentiles) — `observability / src/distllm/core/request_latency.py` [priority P0]
- **Current state:** RequestLatencyInfo.elapsed_ms returns (time.time() - enqueued_at)*1000 using the live clock (request_latency.py L29-30). complete() (L71-77) sets no completion timestamp. get_recent_metrics (L110-122) and get_sla_percentiles (L162-205) then compute elapsed / is_overdue on COMPLETED records from a live clock, so elapsed_ms keeps growing with wall-clock and P50/P95/P99 SLA percentiles are corrupted as the process runs (finding 74, verified must-fix).
- **Why:** SLA compliance is the core batching promise (sla_target_ms=5000 default, L20). Computing it on a moving clock makes completed-request metrics drift and can mark recently-completed requests overdue, and get_urgency_score (L124-135) can deprioritize nothing live while misreporting.
- **Implementation approach:** (1) Add completed_at: float|None = None to RequestLatencyInfo. (2) In complete(), set info.completed_at = time.time() (before append, L71-77). (3) Change elapsed_ms to return (completed_at or time.time()) - enqueued_at so completed records freeze. (4) Set is_overdue from the frozen elapsed for completed records. (5) Add a regression test that sleeps, completes, and asserts elapsed_ms no longer changes.
- **Benefits:** - SLA P50/P95/P99 and overdue counts become deterministic and correct
- Waterfall/API metrics (/api/requests/waterfall, server.py L1596-1608) report real durations
- Removes a source of false-positive SLA breach alarms
- Small, isolated change with low blast radius
- **Trade-offs:** - Changes the meaning of elapsed_ms for consumers that call it pre-completion (they still get live time)
- Completed list now also needs trimming/bounding to avoid unbounded growth (already bounded by window in percentiles)
- Coordinator code that reads info.elapsed_ms must be compatible with the frozen value
- **Effort:** 2-4 hours

---

### E5. Remove telemetry.flush() self-deadlock (non-reentrant lock) and make flush reentrant-safe — `observability / src/distllm/core/telemetry.py` [priority P0]
- **Current state:** flush() acquires self._lock (telemetry.py L150) and _add_event() (L181-185) calls self.flush() while already holding that lock when the in-memory batch reaches BATCH_SIZE=50, deadlocking on threading.Lock (non-reentrant). Any burst of 50 records to an enabled collector freezes the caller (finding 73, verified must-fix).
- **Why:** Telemetry is opt-in (off by default) but the deadlock path is exercised as soon as DISTLLM_TELEMETRY is enabled and traffic exceeds the batch threshold — an intermittent hard hang in the request path. Violates the 'flush must never block request processing' intent.
- **Implementation approach:** (1) Make _add_event not call flush() inside the lock: swap a full buffer into a separate pending list under the lock, then flush outside the lock. (2) Or use an RLock, but prefer (1) so I/O never happens under the lock. (3) Add a test: record_usage x60 on an enabled collector must return without deadlock and drain all 60 events to disk. (4) Optionally bound flush via a background thread guarded by a generating-flag to avoid concurrent double-writes.
- **Benefits:** - Eliminates a verified intermittent hang under load
- Keeps telemetry I/O out of the hot request lock
- Guarantees no telemetry events are lost at flush-on-threshold
- Low-risk, well-scoped fix
- **Trade-offs:** - Refactor of locking requires re-checking opt_out/get_stats for thread-safety
- A background flusher adds a shutdown-drain concern
- Minor behavior change in flush timing
- **Effort:** 2-4 hours

---

### E6. Guarantee dequantization of int8/int4/adaptive compressed KV on the serve path — `core-cache / src/distllm/core/kv_cache.py` [priority P0]
- **Current state:** KVCache supports FP8/int8/int4 compress (kv_cache.py L477-603) and incremental dequant (L94-192, _dequantize_layer L172-192) but the compressed serve path can still expose quantized tensors to attention, and the quant/adaptive variants are inconsistent between compress() and the incremental path (finding 25, verified must-fix: 'int8/int4/adaptive compressed KV cache serves raw (non-dequantized) tensors to attention').
- **Why:** Serving non-dequantized int8/int4 keys/values silently corrupts attention scores and thus generations — a correctness bug that erodes trust in compression, which is a headline efficiency feature (per-layer FP8 quantization). Distributed/FP8 variants are also documented (dist/CLAUDE.md).
- **Implementation approach:** (1) Audit every get_layer/read path (L142-156, L752-754, L696-730) to confirm a single funnel (dequantize_kv_cache) is applied before any F.linear/attention call. (2) Add an explicit assertion or a quant-bits-aware read that returns only float tensors, plus a bool is_dequantized flag on returned segments. (3) Unify compress()/adaptive path with the incremental _dequantize_layer so bits and scale layout match (scale scales). (4) Add a numerical test: quantize -> dequantize -> max abs error < threshold, and an end-to-end logits test showing compressed vs fp16 match within tolerance.
- **Benefits:** - Restores correctness of compressed KV serving, unblocking memory savings for large models
- Eliminates a class of silent-bad-outputs bug
- Makes quant/adaptive paths consistent with documented behavior
- Unblocks distributed FP8 KV transfer correctness
- **Trade-offs:** - Increases per-layer read CPU/GPU cost if dequant is per-request rather than cached
- Requires careful scale/bits layout coupling across compress and serve
- Numerical-tolerance test tuning to avoid flaky CI
- **Effort:** 1-2 days

---

### E7. Unify request dedup: fix core RequestFingerprinter waiters and route all dedup through one engine — `core-router-sched / dedup consolidation` [priority P1]
- **Current state:** There are three independent dedup/cache engines: api/dedup.py _FingerprintCache (working in-flight wait, L83-103), core/request_fingerprinting.py RequestFingerprinter (in-flight wait broken: _in_flight_results is declared L71 but never populated by store() L122-150, so wait_for_result L171-203 always times out at 30s — finding 12, verified must-fix), and plugins/cache_plugin.py _LRUCache (L98-147). The coordinator/scheduler dedup path relies on the broken RequestFingerprinter.
- **Why:** Concurrent identical requests reaching the core fingerprinter always wait 30s and then return None, so dedup silently degrades (no collapse) while still blocking. A single, correct dedup+response-cache service removes the cross-module divergence that produced dedup.py (working) vs request_fingerprinting (broken) and gives one place to fix tenant-scoping and accounting.
- **Implementation approach:** (1) Port the working _FingerprintCache in-flight/event notification into RequestFingerprinter: store() must write _in_flight_results[fingerprint]=response and _signal_waiters. (2) Add a regression test: two threads, same fingerprint, assert second returns the first's result without timeout (not 30s). (3) Introduce a single RequestDedup service (thin wrapper) used by both api/dedup middleware and the coordinator scheduler, with a tenant-scoped key option (see dedup entry). (4) Delete/re-route the broken duplicate paths and update split tests.
- **Benefits:** - Fixes verified concurrent-dedup waiters timing out (id 12)
- Eliminates 3 divergent dedup engines -> one tested implementation
- Reduces latency for repeated concurrent prompts (the intended behavior)
- Enables a single place to add tenant scoping and accounting hooks
- **Trade-offs:** - Consolidation touches coordinator hot path and middleware; needs perf check
- Semantic-cache requirements of cache_plugin differ (approximate) from exact dedup
- Migration risk if existing tests assert the old broken timeout behavior
- **Effort:** 1-2 days

---

### E8. Consolidate speculative verification into one shared, correct verifier (fix off-by-one / all-reject) — `core-decoding / speculative decoders` [priority P1]
- **Current state:** Speculative decoding is split across 7+ files with independent verification logic: tree_speculative_decoder.py _verify_branches (L276), async_pipelined_speculative.py _verify_worker (L425), compressed_speculative.py verify (L127), distributed_speculative.py _verify_tokens (L1154), speculative_decoder.py (SpecDecoderBase L23 + 3 _verify_tokens + _verify_tree L903/969). Findings 33 (4 of 9 tree verifiers have off-by-one token-position indexing so every draft token is checked against the wrong target) and 39 (PipelinedSpeculativeDecoder verifier always rejects or silently accepts drafts) are verified must-fix.
- **Why:** Provider verifiers are the correctness core of speculation: an off-by-one rejects all drafts (killing the speedup) or, worse, accepts a wrong token (corrupts output). Duplication is the direct cause — each verifier re-implements token indexing and has drifted. Consolidation fixes both findings once and gives a single testable contract.
- **Implementation approach:** (1) Extract one VerifyResult=`_verify_sequence(draft_tokens, actual_tokens, on_first_mismatch) -> n_accepted` shared helper with correct position alignment (draft[i] must equal actual[i] for acceptance; on mismatch accept up to i-1). (2) Make tree/pipelined/compressed/distributed call it, deleting their bespoke loops. (3) Add table-driven tests covering: all-correct, first-token wrong, mid-stream wrong, length mismatch, and the off-by-one case that previously rejected every draft (id 33 repro). (4) Add a per-implementation acceptance-rate smoke test using a greedy stub so PipelinedSpeculativeDecoder verifier acceptance is observable.
- **Benefits:** - Fixes verified off-by-one and all-reject bugs across decoders (ids 33, 39)
- One verifier contract -> dramatically smaller correctness surface
- Removes 4+ near-duplicate verification loops (matches the already-done SpecDecoderBase _sample mixin)
- Bounded, measurable acceptance-rate behavior
- **Trade-offs:** - Verification semantics differ subtly across tree vs linear draft shapes; the shared helper must take a draft-shape param
- Broad refactor spans core files; needs the full speculative test suite
- Some decoders are experimental (compressed/dynamic) and may need guard rails
- Potential perf sensitivity on the hot verify path — keep vectorized
- **Effort:** 2-4 days

---

### E9. Gate request serving on leader status in HA mode (stop standby coordinators serving) — `core-ops-ha / src/distllm/core/coordinator.py` [priority P0]
- **Current state:** coordinator.py exposes is_leader (L426-428) and HA election via CoordinatorElection (L205-219, enable_ha L414-424), but generate (L520) and generate_async (L861) never check self.is_leader before serving. In the verified finding 62, standby coordinators in a leader-elected HA group serve requests they should redirect/reject, so failover consistency is not enforced at the request gate.
- **Why:** HA standby mode is advertised (dist/CLAUDE.md 'continuous HA state replication'). Without a leader gate, a split-brain or a just-failed-over standby answers requests with stale/divergent state, undermining the HA guarantee and state-replication invariants.
- **Implementation approach:** (1) Add a fast gate in generate/generate_async: if self._election.ha_enabled and not self.is_leader: raise/route to leader (return a redirect-friendly error with leader address from ha_status). (2) Expose leader address in the API error payload for client re-routing. (3) Add a config flag ha.gate_requests (default on when ha enabled) with a bypass for the leader itself. (4) Test: two coordinators, force standby, assert standby.generate returns the not-leader error while leader serves.
- **Benefits:** - Restores HA leader/follower semantics — prevents divergent responses from standbys
- Makes failover behavior deterministic for clients
- Surfaces leader address for client-side reconnect
- Small, targeted change to generation entry points
- **Trade-offs:** - Requires clients to handle the leader-redirect error and retry against the leader
- Could break single-node (non-HA) deployments if is_leader check isn't guarded by ha_enabled
- Needs careful handling of in-flight standby requests already accepted before election transition
- **Effort:** 0.5-1 day

---

### E10. Feed and actuate IntelligentAutoscaler (currently one-shot feed, never scales) — `core-ops-ha / src/distllm/core/coordinator.py` [priority P1]
- **Current state:** coordinator.py instantiates IntelligentAutoscaler (L1072-1076, via _start_subsystem) and calls autoscaler.record_metrics(ScalingMetrics(...)) exactly ONCE at startup (L1077-1087) using a single scheduler.stats() snapshot. Nothing periodically feeds it and nothing consumes its scaling decision to add/remove nodes (finding 63, verified must-fix: 'wired but never fed or actuated — scales nothing').
- **Why:** Elastic scaling is core to heterogeneous consumer-device pooling (the strategic moat). The autoscaler exists and is documented but is a no-op at runtime, so the cluster never grows/shrinks with load, wasting idle devices or over-saturating. This is a prime 'coordinator wiring' pain the audit flags.
- **Implementation approach:** (1) Run a periodic background task (every e.g. 5-15s) that reads scheduler.stats() (active/pending, from coord.scheduler.stats()) and calls autoscaler.record_metrics. (2) After feeding, read autoscaler.recommend()/decide() and apply: if it recommends scale-down by N, drain then remove those node_ids (using existing cluster_manager/defrag APIs); if scale-up, emit a provisioning event / register new nodes. (3) Add config autoscaler.{enabled,interval_s,min_nodes,max_nodes,target_utilization}. (4) Wrap in the same _start_subsystem lifecycle so it starts/stops with coordinator and surfaces in _subsystem_health. (5) Test with a fake scheduler.loads -> assert record_metrics called repeatedly and actuation invoked on threshold crossing.
- **Benefits:** - Turns an advertised-but-dead feature into real elastic scaling
- Lowers operating cost and improves utilization on heterogeneous fleets
- Reuses existing lifecycle (_start_subsystem) for clean start/stop/health
- Measurable: scale events correlated with load in tests
- **Trade-offs:** - Requires a real provisioning contract for scale-up (cloud/on-prem node creation)
- Actuation must respect drain/grace so scaling down doesn't kill in-flight sequences
- Loop interval adds scheduler.stats() polling overhead (already cheap)
- Behavioral change needs opt-out for users who manage nodes manually
- **Effort:** 2-3 days

---

### E11. Fix `distllm system doctor` and unify CLI exit codes (stop masking failures) — `cli / src/distllm/cli/main.py + doctor.py` [priority P0]
- **Current state:** system_doctor (main.py L1431-1435) imports distllm.cli.doctor.main and calls it, but doctor.main (doctor.py L671-685) runs argparse parser.parse_args() which re-parses sys.argv (already consumed by typer as ['system','doctor',...]) as stray positionals, erroring out (exit 2) before any diagnostic runs (finding 174, verified must-fix). Separately, most failing CLI commands exit 0 because command bodies print errors but never raise typer.Exit(nonzero) — only the completion command raises typer.Exit(1) (main.py L67); grep shows no sys.exit/nonzero propagation in command bodies (finding 175).
- **Why:** The flagship diagnostics command silently fails, and the CLI's exit-0-on-error makes scripts/CI treat failed commands as success, masking health/operational failures — a UX and automation-integrity problem for an ops-facing CLI.
- **Implementation approach:** (1) Refactor doctor.py: expose run_diagnostics(args: argparse.Namespace) -> int and have its typer command build the Namespace from typer options (--gpu/--network/--model/--json/--terse) and call run_diagnostics directly, NOT re-parse sys.argv. Give system_doctor proper typer Options (main.py L1431). (2) Ensure a non-zero exit maps to typer: raise typer.Exit(1 if errors else 0). (3) Apply a global convention: wrap each command body in a helper that catches exceptions and raises typer.Exit(code) so failures propagate non-zero. (4) Add a CLI test invoking the typer CommandRunner (or subprocess) asserting `distllm system doctor --json` returns code 0/+ on healthy and nonzero on error, and that a failing subcommand exits nonzero.
- **Benefits:** - Makes the primary diagnosis tool actually run, with correct exit codes
- Fixes CI/automation false-success masking (id 175)
- Uniform error contract across all subcommands
- Testable via typer test client
- **Trade-offs:** - Behavior change for scripts that (incorrectly) relied on exit 0
- Massaging every command body is broad; do it for the ops-critical ones first, then the rest
- Doctor's --nodes/--model need explicit typer Options to avoid argparse/sys.argv confusion
- **Effort:** 1-2 days

---

### E12. Make time-window usage accounting query the authoritative DB (fix under-report after 100k) — `dist/daas / src/distllm/dist/daas/usage_meter.py` [priority P0]
- **Current state:** get_usage(since_timestamp) (usage_meter.py L224-263) aggregates ONLY from the in-memory self._records list, which stops appending once len reaches _max_records=100_000 (L197-207: `if len(self._records) < self._max_records`). When SQLite persistence is on (record_usage also inserts to DB, L210-220), the DB holds the full history but the time-window query still uses the truncated in-memory list — so any window that falls after the retained cutoff is under-reported or returns None (finding 132, verified must-fix).
- **Why:** This meters billing/quota. Silent under-reporting after 100k records means tenants are under-charged and quota enforcement is escapable simply by exceeding the cap — a trust/ADR issue on the metering path (the DaaS usage_meter is the commercial heart).
- **Implementation approach:** (1) When db_path is set, make get_usage(since) run a SQL aggregate: SELECT SUM(prompt_tokens),SUM(completion_tokens),SUM(duration_ms),COUNT(*) ... WHERE tenant_id=? AND timestamp>=? (schema+index idx_usage_tenant_ts already exist, L108-123). (2) Keep the in-memory fast path only for the no-DB case, but remove the blind drop: instead of refusing to append past 100k, either let _records overflow into a bounded window or fall back to a pre-aggregated per-minute rollup so cap doesn't lose accounting. (3) Add a regression test: insert 100_001+ records, then get_usage(since=now-3600) must include records beyond the cap. (4) Add stats() exposure of 'records_dropped' to make silently-dropped accounting visible.
- **Benefits:** - Fixes verified billing/quota under-report after 100k (id 132)
- Time-window queries become exact and indexed
- Monitoring can flag unexpected record-caps so accounting loss is never silent
- Small change; index already present
- **Trade-offs:** - SQL path brings sync I/O on the query path — keep it out of the lock or use WAL
- Memory-only fast path still needs a bounded-rollup to avoid unbounded growth
- Schema migration not required (columns exist); just a behavior change
- **Effort:** 0.5-1 day

---

### E13. Harden cross-node tensor serialization: fix ZSTD FP8 corruption, zero_copy fabrication, add integrity checks — `dist-net / serialization + zero_copy` [priority P1]
- **Current state:** Three verified dist-net transport defects: AdaptiveSerializer ZSTD path corrupts large FP8 tensors (scale lost / double-converted, finding 120); ZeroCopyTransferEngine.recv fabricates zeros on NCCL and always fails on CUDA_IPC (finding 121); and streaming layer-weight transfer has no integrity verification (checksum/ordering/completeness, finding 130). These live in dist/pipeline/compression_negotiation.py, dist/zero_copy.py, and dist/node_client.py respectively.
- **Why:** Cross-node weight/activations transfer is the core of pipeline parallelism over heterogeneous devices. Silent FP8 corruption and zero-fabrication produce wrong model weights/outputs with no error; lack of checksums means corruption propagates undetected. These are reliability and trust blockers for the distributed transport.
- **Implementation approach:** (1) compression_negotiation.py: make the ZSTD path preserve dtype/scale metadata by serializing the tensor header {dtype, shape, scale} alongside the compressed bytes and casting back to the exact dtype; add a round-trip test for fp8/int8/bfloat16 large buffers. (2) zero_copy.py: replace the fabricated-zeros NCCL path with an explicit NotImplementedError or a guarded fallback that reads real peer memory; add a correctness test that recv returns exactly what send wrote. (3) Add per-transfer checksum (e.g. crc32 on the serialized blob) + sequence/order + completeness metadata on streaming layer-weight transfers (node_client.py), verified on the receive side, failing loudly instead of silently accepting. (4) Add integration tests for partial/intended-loss and out-of-order cases.
- **Benefits:** - Fixes silent numeric corruption in FP8/ZSTD large tensors (id 120)
- Removes zero-fabrication path that returns wrong data (id 121)
- Detects corruption/ordering/completeness on the wire (id 130)
- Bases the transport on fail-loud integrity instead of silent acceptance
- **Trade-offs:** - Checksum adds per-transfer CPU cost and slightly larger payload
- zero_copy stub behavior change may break tests expecting the fake zeros
- Compression metadata adds serialization complexity and potential size growth for small tensors
- **Effort:** 2-4 days

---

### E14. Consolidate the two divergent SSO implementations behind one auth provider layer — `api-gateway / SSO consolidation` [priority P1]
- **Current state:** Two parallel SSO trees exist: src/distllm/api/sso_auth.py + sso_middleware.py (wired into server.py via setup_sso, L953-954 and the /v1/auth routes) vs src/distllm/api/auth/ package (oidc.py, oauth2.py, saml.py, store.py, models.py — present but not wired into the documented flow). Finding 4 flags the divergence risk: drift between the wired SSO path and the auth/ package means fixes go to the wrong tree and behavior is not single-sourced.
- **Why:** SSO/OIDC handling is security-critical (JWT validation, CSRF state, nonce). Two divergent implementations raise the chance of a patch landing in the unused tree, orphaned configs, and inconsistent contract — a genuine maintainability and correctness exposure for auth.
- **Implementation approach:** (1) Define the wired path explicitly: build a Provider interface in api/auth (OIDCProvider/OAuth2Provider/SAMLProvider) and make sso_middleware.setup_sso consume it. (2) Move/alias sso_auth.py's provider logic into api/auth/oidc.py etc., deleting the duplicate. (3) Add a parity test asserting the wired middleware and the auth/provider classes yield identical token-validation results. (4) Deprecate-and-remove orphaned modules not reached from setup_sso after confirming with an import graph. (5) Add an SSO end-to-end test: mock provider -> /v1/auth/token issues -> AuthMiddleware honors request.state.auth_method=='sso' (middleware.py L256-257).
- **Benefits:** - Single SSO source of truth removes a real security divergence risk (id 4)
- Fixes go to one tree; config is not duplicated
- Testable provider contract for OIDC/OAuth/SAML
- Ends the orphaned auth/ package dead-code ambiguity
- **Trade-offs:** - Provider unification is invasive and touches security code; needs careful review
- Must preserve JWT algorithm handling and nonce/state CSRF guarantees
- Risk if third parties import sso_auth symbols directly
- **Effort:** 2-3 days

---

### E15. Apply tenant scoping to ALL cache keys (exact-match included) and unify the cache backends — `cache / plugins/cache_plugin.py + dedup + semantic_cache` [priority P0]
- **Current state:** cache_plugin._request_scope (cache_plugin.py L87-93) computes tenant/user scope but it is applied ONLY to the semantic-cache path (L373, L420). The exact-match cache key built by _build_cache_key (L70-84) is just prompt|model|temperature|top_p with NO tenant component, so exact-match cache entries leak across tenants (finding 199, verified must-fix: 'Exact-match cache key is not tenant/user scoped - cross-tenant response leak'). This duplicates the same cross-tenant leak present in api/dedup.py (see dedup entry).
- **Why:** Caching shared model outputs across tenants is a privacy/security violation for any multi-tenant deployment: one tenant's cached response becomes another's. The helper (scope) already exists — it's just not applied to the primary exact-match path — making this a small, high-value fix.
- **Implementation approach:** (1) Thread scope into _build_cache_key: prepend tenant/user scope (from request context) to the raw string before hashing (L83-84). (2) Ensure on_request uses the same scoped key as on_response (it does build the same key — just needs scope). (3) Since _request_scope already derives from context tenant_id/user_id, reuse it for exact match too. (4) Add tests: same prompt, different tenant -> distinct keys; different user under same tenant -> distinct keys. (5) Align Redis KEY_PREFIX with scope when multi-tenant to avoid over-wide namespaces.
- **Benefits:** - Closes the verified cross-tenant exact-match cache leak (id 199)
- Reuses the existing, already-correct scope helper — minimal new code
- Consistent with dedup tenant-namespacing (shared fix pattern)
- Prevents cached hallucination/private data from crossing tenants
- **Trade-offs:** - Reduces exact-match hit rate across tenants (only same-tenant hits)
- Backwards-incompatible cache keys (old entries become unreachable) — harmless, just eviction
- Needs the request context to reliably carry tenant_id (check PluginHook context, server.py L853-869)
- **Effort:** 2-4 hours

---

### E16. Make the OpenAPI spec the single source of truth and gate cross-SDK parity in CI — `sdk-arch / OpenAPI + multi-language SDKs` [priority P2]
- **Current state:** The audit flags two parallel, already-diverged Python SDKs (distllm.sdk vs distllm_sdk, finding 194), sync-vs-async chat streaming yielding different item types in both (finding 193), and multi-language SDKs (Rust lacks streaming; JS/Go only cover a core surface, finding 195). The OpenAPI spec sdk/openapi/distllm.yaml exists but is not the enforced contract for SDK codegen (finding 192: 'Make the OpenAPI spec the single source of truth and gate cross-SDK parity in CI').
- **Why:** The SDKs are how users integrate; divergence means incomplete/correctness-inconsistent clients across languages and the two Python packages are best grouped into one. Enforcing parity from the spec stops drift and makes the multi-language surface trustworthy.
- **Implementation approach:** (1) Deprecate one Python SDK (keep distllm_sdk as canonical, alias/remove distllm.sdk) — pick based on which is packaged/published. (2) Generate the Python types (and stubs) from sdk/openapi/distllm.yaml so sync and async variants share the same item shapes (fixes the stream type mismatch). (3) Add a CI job that validates every checked-in SDK model against the OpenAPI spec and runs a small parity test (streaming item type, method surface) across py/ts/go/rust. (4) Add streaming support to the Rust SDK or explicitly document it as out of scope; JS/Go gain the remaining endpoint coverage generated from the spec. (5) Add a schema-version header check so clients and spec stay aligned.
- **Benefits:** - Removes duplicate/divergent Python SDKs (id 194)
- Fixes the verified streaming item-type mismatch (id 193)
- Cross-language method-surface parity, enforced in CI (id 192)
- Spec-driven generation cuts long-term maintenance and drift
- **Trade-offs:** - Deprecating an SDK is a breaking change for its users; needs a migration window
- Codegen tooling (OpenAPI generator) adds a build dependency and can lag spec features
- Rust streaming may be non-trivial; explicit scope-decision required
- CI parity harness is a new, evergreen cost
- **Effort:** 3-5 days

---

### E17. Unblock the pytest suite (79 live collection errors) so CI gates run real checks — `tooling-tests / pytest + CI` [priority P0]
- **Current state:** pytest.ini currently collects 79 import/collection errors that abort the whole suite before any real assertions run (finding 213, verified must-fix: 'CI test job is blocked: 79 live collection errors interrupt the whole suite'), and the GPU benchmark regression gate uses an invalid GitHub context (runner.environment, finding 214), so benchmarks never gate. A test-infra fake-package shim also whitelists a stale symbol list, breaking imports of real config (finding 215).
- **Why:** Without a green collection, no CI check meaningfully guards the codebase; regressions ship silently. The 232-finding audit is only as durable as the test gate that prevents recurrence — unblocking collection is a precondition for every other fix to be verified in CI.
- **Implementation approach:** (1) Run `pytest --collect-only -q` and fix the 79 errors (missing imports, refactored symbols, orphan modules like the stale test-helper whitelist in tests/_import_helper.py for finding 215). (2) Replace the invalid `runner.environment` gate with `runner.os`/`matrix` (finding 214). (3) Add a CI step that fails on any collection error (--strict-collect). (4) Split the suite into fast unit vs slow/integration so one broken module doesn't block the whole job; use -x or per-directory jobs. (5) Add a smoke CI job running core+api+dist unit tests as the merge gate.
- **Benefits:** - Restores a real, green test gate for the entire audit remediation
- Surfaces regressions immediately instead of a collection wall
- Fixes the dead benchmark gate (id 214) and stale test shim (id 215)
- Faster, partitionable CI feedback
- **Trade-offs:** - Fixing 79 collection errors is broad, front-loaded effort with diminishing ROI
- Splitting jobs changes CI maintenance shape
- Strict collection may surface tech debt users were relying on being skipped
- **Effort:** 2-4 days

---
