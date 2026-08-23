---
tags:
  - core
  - audit
  - enhancements
date: 2026-08-05
---

# Core Audit 03 — Enhancements & Modifications

**← [[Core Comprehensive Audit 2026-08-05]]**
Category 3: concrete, actionable upgrades to existing core components. High-ROI first. The common theme: **a large body of already-tested code is wired to `None` or unwired — activating it is the cheapest real enhancement in the repo.**

## E1 — Activate dormant request-dedup + compliance-audit behind config flags — `coordinator.py:238`, `request_fingerprinting.py:59` (High/P1)
`RequestFingerprinter` and `RequestAuditor` are complete, tested, and referenced throughout `request_pipeline` (mark_in_flight/store/lookup/clear_in_flight, audit.record/update_response), but `Coordinator.__init__` leaves their instances `None`, so nothing runs.
- **Do:** add `enable_request_dedup` / `enable_request_audit` to `coordinator_config`; construct the collaborators when enabled (bounded cache size, audit `log_dir`); add an E2E test proving (a) two identical concurrent prompts coalesce to one generation and (b) an audited request lands in the JSONL trail.
- **Benefit:** concrete multi-tenant cost+latency win on duplicate/retry bursts + an audit trail — reusing ~1,000 LOC that already works.
- **Trade-off:** bounded cache memory; make the in-flight path crash-safe (fix the in-flight leak) or a crash mid-coalesce could stall.

## E2 — Activate per-tenant QoS isolation + route audit for enterprise governance — `sentinel_qos.py`, `route_audit.py` (Medium/P2)
`Sentinel` (token buckets, WFQ, per-tenant KV partitioning, SLO/SLI + alerting) and `RouteAuditLog` (routing-decision JSONL: provider/region/cost/carbon/outcome) are fully implemented but unwired.
- **Do:** wire `Sentinel` into `batch_scheduler` admission; wire `RouteAuditLog` into `model_router` decisions; expose tenant SLI and cost/carbon reports on the dashboard.
- **Benefit:** verifiable tenant isolation + per-query cost/carbon attribution — the exact governance surface enterprise/regulated buyers pay for (SOC2/ISO27001-aligned), where vLLM/TGI have nothing.
- **Trade-off:** per-request overhead + scheduler admission complexity; keep opt-in per tenant.

## E3 — Converge the speculative-decoding variants onto one base — `speculative_decoder.py`, `distributed_speculative.py`, `compressed_speculative.py`, `async_pipelined_speculative.py`, `tree_speculative_decoder.py`
Five+ implementations share ~30% duplicated logic (off-by-one in `prefix_len` is a recurring class of bug — see B19). Consolidate a `SpecDecoderBase` (the repo already started one) and route each strategy through it.
- **Benefit:** kills recurring verification/prefix math bugs, halves maintenance. This is both an enhancement (cut duplication) and the fix for C4/B8/B19.
- **Trade-off:** refactor risk; needs an acceptance-rate parity test matrix across strategies.

## E4 — Consolidate the four prefix-cache implementations behind `protocols.ICacheBackend` — `protocols.py:80`, `prefix_cache`, `radix_tree_cache`, `redis_prompt_cache`, `prompt_caching_service`
`ICacheBackend` is declared but unused; four cache implementations expose divergent APIs. 
- **Do:** implement `ICacheBackend` for PrefixCache (canonical memory), RadixTreeCache, and a Redis backend; make `cache_manager` build exactly one backend from config (`memory_engine: prefix|radix`, `distributed: none|redis`); delete test-only modules.
- **Benefit:** one enforced cache contract → cross-node (Redis) prompt sharing becomes first-class → real WAN throughput win; also fixes the broken Redis tier (B15).
- **Trade-off:** behavior-drift risk across backends is mitigated by a parity test matrix.

## E5 — Real cloud SDK calls for spot bidding — `bargaining_engine.py`, `cross_cloud_router.py`
The DQN spot-bidding/routing is implemented but not connected to real cloud pricing. 
- **Do:** wire live provider metadata (AWS/Azure/GCP spot price feeds) through a `MetadataProvider` interface; single-source pricing for `CostComparison`/`CostOptimizer`/`CrossCloudRouter` (currently three consumers read data independently).
- **Benefit:** delivers the flagship "$60–80% GPU cost reduction" promise; carbon-aware routing becomes trustworthy (F52).
- **Trade-off:** needs a live data dependency (electricityMap/WattTime, provider APIs) and must be optional for air-gapped users.

## E6 — Make structured-output token-index build synchronous — `structured_output/__init__.py`
Once C3 is fixed, add caching + incremental builds so the first constrained request doesn't pay a 32k-token decode synchronously on every restart. Consider persisting the token-index cache across processes.

## E7 — Fix + wire the dormant **predictive-failure** and **Pareto multi-objective routing** pillars — `predictive_failure.py`, `pareto_optimizer.py` (Medium/P3)
Both are complete and tested but unwired. Wire `PredictiveFailureDetector` into `health_manager` and `ParetoOptimizer` into `model_router` behind a documented feature flag.
- **Benefit:** turns inert tested code into the customer-visible "autonomous, self-optimizing inference fabric" positioning — a real differentiator vs single-objective vLLM/SGLang routers.
- **Trade-off:** wiring adds a runtime path to maintain; gate honestly so you don't over-claim.

## E8 — Wire `NeuralPartitionOptimizer` as the default heterogeneous partition planner — `neural_partition_optimizer.py:1199`
The learned-cost (MLP) + Bayesian-optimizer partitioner is exported from `core/__init__` but invoked by nothing. Integrate `auto_optimize()` into heterogeneous scheduling on node-registry change; persist the cost-model `state_dict`.
- **Benefit:** genuine "AutoPlacement" — data-driven layer→GPU assignment no static-partition tool has.
- **Trade-off:** GP/Optuna bootstrap latency; it already ships a cold-start heuristic fallback.

## E9 — Multimodal: actually encode media — `multimodal_engine.py:168`
The non-text path only prepends `[IMAGE]`/`[AUDIO]` markers and drops the tensors. Wire a real vision encoder (CLIP/fusion) + image→token fusion, or log "media dropped" explicitly and treat markers as an "unavailable" mode.
- **Benefit:** genuine vision/audio/doc gateway ≠ text-only shell.
- **Trade-off:** encoder GPUs raise fleet cost; make opt-in per node.

## E10 — Fix the PagerDuty webhook formatter — `webhook_formatters.py:196`
`pagerduty_formatter` defaults `routing_key=""`, and `WebhookManager` calls the formatter with two args only → empty routing key → PagerDuty always rejects. Thread `target.secret`/a `routing_key` param through `register()`; add a test with a non-empty key.

## E11 — Correct the `feature_flags` gating order — `feature_flags.py:170`
`is_enabled()` early-returns `True` when a user is allow-listed, skipping the time-window and rollout-% checks → rollouts overshoot/ignore expiry. Evaluate targeting/rollout/time-window as one composite gate.

---
**← [[Core Comprehensive Audit 2026-08-05]]** · Next: [[Core Audit 04 Advanced Features]]