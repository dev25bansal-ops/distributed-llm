---
tags:
  - core
  - audit
  - advanced
date: 2026-08-05
---

# Core Audit 04 — Advanced Features

**← [[Core Comprehensive Audit 2026-08-05]]**
Category 4: sophisticated capabilities that could elevate the core and differentiate versus competitors. These are real code paths that are either stubbed, unwired, or under-built — completing them to a **measured, production** surface is the advanced-feature work with the most business value.

## A1 — Remote-draft speculative decoding (cloud drafts, local verifies) — headline hybrid — `distributed_speculative.py`
A **genuinely novel architecture**: use a cheap small remote model (OpenAI-compatible over HTTP/gRPC) as the *draft* for a local big model, hiding per-token network latency behind a parallel draft stream. No competitor (vLLM/Ollama/Petals) ships a hosted-draft hybrid.
- **Implementation depth:** already has `RemoteDraftConfig`, `DraftLatencyStats`, pydantic response models, retry/raw fallback, async client, gRPC stub.
- **Make it:**
  1. a default-candidate mode for 7B–14B local + small draft;
  2. **measured** acceptance-rate + TTFT/ITL vs pure-local and pure-distributed;
  3. framed as "data stays local, latency hides behind the draft" (pairs with DistLLM Cloud as its own revenue stream).
- **Trade-off:** a paid cloud draft per request reopens the cloud dependency the zero-cost narrative rejects — message it as an optional small cost or the cloud draft tier.

## A2 — Cost-arbitrage control plane (spot + carbon + failover) — `spot_forecasting.py`, `spot_failover.py`, `cross_cloud_router.py`, `carbon_migration.py`
`SpotFailover`, `SpotPriceForecaster`, `CarbonBudgetEnforcer`, `CostForecaster` are complete and tested but **unwired**. Together they implement the "$60–80% GPU cost reduction" pitch.
- **Do:** wire `SpotFailover` into `coordinator_subsystem.py` via `SubsystemRegistry`; feed GCP/AWS/Azure price metadata through a `MetadataProvider`; expose `/admin/cost-forecast`; gate `CarbonBudgetEnforcer` on the existing quota middleware; finish `carbon_migration` (fix B7) and validate carbon routing isn't greenwash (needs live electricityMap/WattTime).
- **Trade-off:** new operational surface (metadata polling, alerting); forecast models are Holt-Winters heuristics needing real price telemetry — ship with explicit "no guarantee" UX.

## A3 — True hierarchical cache coherence (Bloom→Merkle→exact) — `hierarchical_digest.py`
The current L2 "which blocks differ" is an **all-or-nothing stub** ([[Core Audit 02 Issues & Required Fixes|hierarchical_digest]]): any bloom match collapses straight to a full sync, so the advertised "95%+ gossip bandwidth reduction" isn't realized.
- **Implement:** real per-block narrowing (walk the Merkle tree, return only mismatching leaf ranges), a level-3 exact-index exchange, and `build_exchange_payload` advancing through levels 2/3.
- **Benefit:** near-linear→O(1) cross-cluster prefix sync — a strong WWAN multi-cluster differentiator with no comparable OSS implementation.
- **Trade-off:** correctness of narrowing *is* the entire feature; ship it properly or remove the 95% claim.

## A4 — Predictive KV-cache pre-warming — `predictive_cache_warming.py`, `cluster_predictive_prefetcher.py`
Unwired but tested: Markov/prefix-based prediction of cache reuse. Wire into `cache_manager` to prefetch likely-reused prefixes ahead of requests, cutting TTFT on repeated/related workloads.
- **Trade-off:** prefetch accuracy vs wasted memory/fabric — needs a hit-rate guard.

## A5 — Self-healing autopilot — `autonomous_healer.py`, `predictive_failure.py`, `graceful_degradation.py`
Combine predictive failure detection with the autonomous healer and (currently unwired) graceful degradation into a zero-human-intervention failure pre-emption loop (predict → pre-empt → rebalance). Wire the predictive/pareto pillars first (E7), then layer healing.

## A6 — DPO self-improving router — `preference_learning.py`, `agentic_router.py`
The router's judge is meant to self-improve via DPO on routing outcomes, but today it trains on **empty prompts** (B5). Fix B5, then expose the router as an "autonomous, self-improving" control plane: online preference capture → periodic DPO retrain → A/B-validated swap. This is a differentiating narrative ("the router gets smarter with your traffic") no OSS competitor ships.

## A7 — Multi-objective Pareto scheduling (latency × throughput × cost × carbon × fairness) — `pareto_optimizer.py` (unwired)
`ParetoOptimizer` computes the price/latency/carbon frontier but nothing calls it. Wire it into `model_router` as the decision engine (replacing single-objective routing), and fix the AtlasMesh scorer wiring (B4) so the multi-objective weights are real. No mainstream LLM server does Pareto-optimal scheduling.

## A8 — Cross-cluster KV-cache mesh — `atlas_mesh.py`, `cache_migration.py`, `gossip_cache_bridge.py`, `hierarchical_digest.py`
Chain the coherence protocol (A3) with the existing migration/gossip bridges into a real **global KV mesh**: users' prompts reuse cache that exists anywhere on the fabric. Fix B4 (scorer), B7 (migration), and the `cache_migration` SSRF/TLS gap, then wire `cross_model_prefix_sharing` (currently keys on prefix, not semantics — B-adjacent) to make it correct.

## A9 — `Cortex` cross-model prefix sharing + expert parallelism — `cortex_multimodel.py` (dead + stubbed)
Either wire or delete. If kept, make `serve()` actually copy the matched cached KV tensors into the model run (today it caches an empty `dummy_kv` and logs only), and stop returning `utilization=1.0` unconditionally.

---
**← [[Core Comprehensive Audit 2026-08-05]]** · Next: [[Core Audit 05 New Additions]]