---
tags:
  - audit
  - action-plan
date: 2026-08-11
---

# Action Plan 04 — Advanced Features

**← [[Exhaustive Audit 2026-08-11]]**


Twelve advanced features that amplify DistLLM's defensible moat (heterogeneous consumer multi-device pooling, data-local inference, distributed/cross-device KV sharing, speculative decoding at scale, cost & federated arbitrage, privacy-preserving inference). Each is grounded in existing scaffolding in src/distllm/core and src/distllm/dist and the 2026-08-11 audit (per Digest IDs 46, 77, 78, 80, 81, 88, 100, 111, 127, 129, 225, 226, 229, 230, 232). Sequenced to first close the strategic gaps the audit flagged as adoption blockers (perf honesty, monetization linchpin, correctness of cross-model/cross-node KV). Honest split between near-term (weeks, buildable now on real scaffolding) and moonshot (needs new distributed infra / research). All 59 Critical/High audit findings these depend on are captured as explicit dependencies so implementation ships as a correctness fix as well as a feature.

**12 advanced features**

### A1. Real WAN disaggregated prefill/decode via KV transfer (fix the DisaggManager placeholder)
- **Description:** Complete the disaggregated prefill+decode split so prefill (compute-bound) and decode (memory-bound) run on different node pools connected by KV-cache transfer. Today DisaggManager.prefill() hands a handle to the transfer scheduler, but decode() never calls the decode node: it returns placeholder output via `await asyncio.sleep(0.01)` and `input_ids[-1:] + [42]*10` (src/distllm/dist/disagg/__init__.py:103-125), Digest 127. Wire decode() to real gRPC/QUIC forward on the decode node, plumb the KVCacheHandle through distributed_speculative's KV passthrough, and surface aggregation to wide_area. NEAR-TERM.
- **Differentiation:** Competitors (vLLM disaggregated prefill, SGLang RadixAttention) do prefill/decode split inside one datacenter; DistLLM's edge is doing it over LAN/WAN across heterogeneous consumer devices with the existing streaming_kv_transfer and wide_area QUIC path. No other platform lets a CPU/edge node prefill and a remote gaming GPU decode.
- **Implementation technique:** Replace the decode() placeholder with a real client call (extend node_client / node_service forwards); add a chunked KV transfer lifecycle that resolves the KVCacheHandle prefill cache before decode starts; re-use KVCacheTransferScheduler (disagg/transfer.py) for in-flight reconciliation; expose pool-level capacity-aware routing via decode_pool.select_node(). Add end-to-end integration test (tests/dist/disagg) asserting real token output, not sentinels.
- **Business value:** Decodes a whole class of 'can't fit the model' and 'low latency streaming' workloads; separates bursty compute from steady memory so consumer fleets use idle GPU for decode while cheap CPU preflils.
- **Market angle:** Anchor feature for the 'pool your GPUs' wedge (Digest 225): lets a single 8GB consumer GPU decode a 70B-class model whose prefill ran on a neighbor's machine.
- **Complexity:** medium · **Effort:** M (2-4 weeks incl. tests)
- **Dependencies:** Fix/verify KV-cache serialization (Digest 120 ZSTD FP8 corruption is a blocker for correct tensor transfer); node_client/node_service forward path; streaming_kv_transfer correctness
- **Builds on:** dist/disagg/__init__.py; dist/disagg/pool.py; dist/disagg/transfer.py; core/distributed_speculative.py; dist/wide_area.py; dist/streaming_kv_transfer.py
- **Evidence:** src/distllm/dist/disagg/__init__.py:103-125 (decode placeholder), :66-101 (prefill returns handle); Digest 127 (DisaggManager.decode never calls decode node)

---

### A2. Heterogeneous Draft-Fleet speculative decoding + Draft-as-a-Service monetization
- **Description:** Operationalize the DraftModelFleet in core/distributed_speculative.py (register multiple remote draft endpoints across cheaper devices — CPU laptops, edge boxes, browser WebGPU) and package the draft model as a billable Draft-as-a-Service via dist/daas_server.py, which already exposes OpenAI /v1/completions, token logprobs, rate limiting, and cost_per_hour billing (src/distllm/dist/daas_server.py:18-25, :46-59). NEAR-TERM (fleet routing exists; monetization and auto-routing do not).
- **Differentiation:** vLLM/Medusa self-speculate on the same GPU; nobody runs the draft on OTHER people's cheap devices and bills for it. This turns the platform's own distributed ethos into a two-sided market (draft providers + target owners).
- **Implementation technique:** Add a fleet-aware asynchronous draft scheduler that round-robins/selects draft endpoints by measured RTT (extend DistributedSpeculativeDecoder.agenerate overlap per docs lines 8-41); surface DaaS as a first-class coordinator backend (register_draft in subsystem_registry); emit per-endpoint acceptance-rate telemetry (speculative_profiler/speculative_dashboard) so coordinator picks the fastest draft; wire marketplace.GPUListing as the discovery source for draft providers.
- **Business value:** New revenue line (draft-model serving) with near-zero marginal cost; improves target throughput for all customers when drafts are fast.
- **Market angle:** Addresses monetization linchpin the audit says is unimplemented (Digest 229: 7-tier MARKETING relies on 'billing/' that doesn't exist). DaaS is the smallest shippable paid unit.
- **Complexity:** high · **Effort:** L (4-8 weeks)
- **Dependencies:** Correct the 9 speculative verifier off-by-one / consensus sampling bugs first (Digest 33, 34, 39, 40, 43) so acceptance math is trustworthy; speculative_profiler telemetry; daas_server billing/rate-limit path
- **Builds on:** core/distributed_speculative.py (DraftModelFleet); dist/daas_server.py; core/speculative_profiler.py; core/speculative_dashboard.py; core/subsystem_registry.py; dist/marketplace.py
- **Evidence:** src/distllm/core/distributed_speculative.py:34-41 (fleet), :13-16 (KV passthrough); src/distllm/dist/daas_server.py:46-59 (DaaSConfig cost_per_hour), :18-25 (features)

---

### A3. Self-improving speculative decoding (online adaptive draft selection + acceptance feedback)
- **Description:** Add a closed-loop controller that measures real draft acceptance per prefix/model/device (speculative_profiler, speculative_dashboard, adaptive_speculator) and continuously tunes num_candidates, draft endpoint, and KV passthrough depth. Grounded in core/distributed_speculative.py (adaptive candidate count, docs line 12) and dist/adaptive_speculator.py + dist/draft_bank.py. NEAR-TERM.
- **Differentiation:** Static speculative config is the norm; a controller that self-tunes against each tenant's traffic mix (and retrains draft banks when acceptance drifts) is a live-platform advantage, and it parks colder simply behind a telemetry loop rather than new inference research.
- **Implementation technique:** Wrap verification in the existing 9 verifiers behind one interface; collect (prefix, draft_model, n_candidates, accepted_k, latency) tuples; run a Bayesian/multi-armed-bandit selector over draft endpoints and candidate counts (reuse arbitrage_engine/opportunity scoring pattern); persist learned policies to a tunable store and emit to speculative_dashboard. Guard with an acceptance-floor so a bad draft is dropped before it harms TPS.
- **Business value:** Directly increases tokens/sec and reduces wasteful verification on every server, compounding the 'cost per token' story that prices the DaaS/marketplace tiers.
- **Market angle:** 'Self-tuning speculative decode' is a measurable, demoable latency win that offsets the weak single-node bench (Digest 226) with a cluster-scale strength.
- **Complexity:** high · **Effort:** L (4-8 weeks)
- **Dependencies:** Spec-verifier correctness fixes (Digest 33, 34, 43); speculative_profiler/dashboard telemetry wiring; draft_bank persistence
- **Builds on:** core/speculative_profiler.py; core/speculative_dashboard.py; core/speculative_adaptor.py; dist/adaptive_speculator.py; dist/draft_bank.py; core/distributed_speculative.py
- **Evidence:** src/distllm/core/distributed_speculative.py:12 (adaptive candidate count), :34-41 (fleet); src/distllm/dist/adaptive_speculator.py (draft bank adaptor)

---

### A4. Correct cross-model + cross-node prefix sharing (fix Critical 46 and cross-node gap 28)
- **Description:** Rebuild KV prefix sharing so a request for model B can reuse the shared-layers KV of model A, and so a request can reuse another NODE's prefix blocks. Current sibling lookup returns a cached entry matched only on source-model base and shared_layers with NO verification that tokens match — the CRITICAL wrong/injected-token bug (Digest 46) at src/distllm/core/cross_model_prefix_sharing.py:169-181; and dist/prefix_cache.find_best_node matches only a full-sequence hash, so genuine cross-node prefix sharing never happens (Digest 28). NEAR-TERM but correctness-critical.
- **Differentiation:** True token-guaranteed cross-variant + cross-node prefix reuse is what gives 30-50% TTFT cuts on fine-tuned model families (docstring lines 6-7) — a concrete, hard-to-clone latency advantage that datacenter KV caches don't generalize to heterogeneous node memory.
- **Implementation technique:** Store the actual token_ids (not just a truncated 16-hex hash, current _hash_tokens at :185-189) and verify token-id equality before returning any entry; key entries by (model_variant, token_hash) and add a shared-layer offset so only the shared prefix region is reused; in dist/prefix_cache, extend find_best_node to match on shared-prefix hash ranges rather than full sequence; add a dedicated security test that a sibling lookup for a different prompt returns None (Digest 61 flags no such test today).
- **Business value:** Protects correctness/secrecy (today it can serve another tenant's prompt content) and unlocks TTFT wins that become a headline metric.
- **Market angle:** 'Same model family, 40% faster first token, cross-device' is a crisp enterprise value prop that also de-risks the project's credibility (Digest 231).
- **Complexity:** medium · **Effort:** M (2-4 weeks)
- **Dependencies:** Harden cache store overwrite memory budgeting (Digest 26); consolidate the ~15 overlapping cache zones (Digest 27); add the missing cross-model/path tests (Digest 61)
- **Builds on:** core/cross_model_prefix_sharing.py; core/prefix_cache.py; dist/prefix_cache.py
- **Evidence:** src/distllm/core/cross_model_prefix_sharing.py:169-181 (sibling lookup returns unverified entry), :185-189 (_hash_tokens truncates to 16 hex); Digest 46 (CRITICAL), Digest 28 (cross-node full-hash only)

---

### A5. Federated cross-cluster KV-cache sharing with per-cluster privacy budget (DP layer)
- **Description:** Expose a cross-cluster cache tier that shares prefix/KV blocks between federated clusters (dist/cross_cluster.py forwarder, dist/redis_cache, dist/cache_digest) and protect shared tensors with the differential-privacy noise + PII anonymization already scaffolded (DifferentialPrivacy.add_noise_to_kv_cache at cross-cluster boundary; InputAnonymizer strips PII before leaving the node). NEAR-TERM (scaffolding exists; DP semantic wiring and cross-cluster cache index do not).
- **Differentiation:** Rivals share caches only inside a trusted datacenter. DistLLM can legally/meaningfully share KV across untrusted clusters because it applies calibrated DP noise (core/differential_privacy.py) and PII redaction (InputAnonymizer) at the trust boundary — a fabric-clustering privacy story no competitor offers.
- **Implementation technique:** Add a cache_digest-based advertisement of sharable prefix blocks across clusters (reuse CrossClusterForwarder._call_ray_worker and gossip); route cross-cluster hits through dp_inference privacy budget accounting (core/dp_inference.py set_tenant_budget) so a tenant's aggregate epsilon is enforced; featurize IP/carbon for cross-region placement via marketplace.GPUListing region+carbon fields.
- **Business value:** Monetizes idle KV blocks of a whole federated fleet rather than one machine; opens enterprise 'share across regions but keep data local' contracts.
- **Market angle:** The star privacy angle of the honest narrative (Digest 232: 'data that never leaves your devices') — shipped as a real DP guarantee rather than a slogan.
- **Complexity:** high · **Effort:** L (4-8 weeks)
- **Dependencies:** Fix DP inference correctness first (Digest 80 NameError, 81 no-op noise, 88 sensitivity/sigma mismatch, 87 advanced-composition under-report); E2E tensor encryption (Digest 92, 134) so cross-cluster is not plaintext; cross_cluster ray forwarder
- **Builds on:** core/differential_privacy.py (DifferentialPrivacy, InputAnonymizer); core/dp_inference.py; dist/cross_cluster.py; dist/redis_cache.py; dist/cache_digest.py; dist/federation.py
- **Evidence:** src/distllm/core/differential_privacy.py:83-99 (add_noise_to_kv_cache), :139-183 (InputAnonymizer PII patterns); src/distllm/core/dp_inference.py:29-31 (per-tenant budget); src/distllm/dist/cross_cluster.py:23-44 (forwarder)

---

### A6. Cost/ROI arbitrage engine with reconciled accounting and honest carbon/spot data
- **Description:** Turn the existing arbitrage + cost stack into a trustworthy optimizer: arbitrage_engine.py (OpportunityType PRICE_DROP/CHEAPER_REGION/CHEAPER_PROVIDER/CARBON_SWITCH, MigrationRisk) + cost_optimizer.py (ModelROI, CostAlert budget escalation, check_budgets) + cross_cloud_router + pricing_providers. Today two independent cost engines disagree and zero-cost requests misclassify (Digest 77), AWS static spot lists price ABOVE on-demand (Digest 78), and monthly budgets compare against all-time totals (Digest 76). NEAR-TERM; correctness + data-integrity first.
- **Differentiation:** Most cost tools just bill. DistLLM is positioned to show real 'price of this model if run here vs cloud' and auto-migrate in-flight workloads across providers/regions/carbon — arbitrage as a product, not a bill.
- **Implementation technique:** Unify the two cost engines into one recorded ledger (cost_tracker is the source of truth; streaming_cost reconciles token-price vs GPU-hour), reset monthly budgets per period in usage_meter, fix pricing_provider AWS static fallback, and add carbon-awareness (marketplace.GPUListing carbon_intensity/renewable_pct) to migration scoring. Gate auto-migration behind MigrationRisk.risk tiers so high-risk moves require confirmation.
- **Business value:** Direct cost reduction headline ('save 40% running on spot'), plus the ROI dashboard is the sales artifact that justifies the paid tiers.
- **Market angle:** Compelling land-and-expand for cloud-native teams; 'carbon-optimized inference' is a differentiation wedge ESG buyers will pay for.
- **Complexity:** medium · **Effort:** M (3-5 weeks)
- **Dependencies:** Fix cost reconciliation (Digest 77), spot pricing (78), monthly budget reset (76); cloud_selector/cross_cloud_router live pricing; arbitrage_engine migration safety
- **Builds on:** core/arbitrage_engine.py; core/cost_optimizer.py; core/cost_tracker.py; core/streaming_cost.py; core/cross_cloud_router.py; core/pricing_providers.py; dist/marketplace.py (carbon fields)
- **Evidence:** src/distllm/core/arbitrage_engine.py:26-39 (OpportunityType/MigrationRisk); src/distllm/core/cost_optimizer.py:146-193 (get_roi_report), :195-239 (check_budgets); src/distllm/dist/marketplace.py:88-90 (carbon_intensity/renewable_pct); Digest 76/77/78

---

### A7. Privacy-preserving inference gateway (DP generation + PII + E2E + watermark)
- **Description:** Package the privacy stack as a single opt-in inference gateway: DifferentialPrivacyInference with per-tenant epsilon budgets, PII anonymization (InputAnonymizer) before any tensor leaves the node, E2E AES/PyNaCl tensor encryption (security/e2e), and model/IP output watermarking. Today the 'DP inference' non-streaming branch raises NameError, 'DP' applies no noise to outputs, and sensitivity/sigma are mismatched ~4x (Digests 80, 81, 88, 87); e2e fails open to plaintext (Digest 92). NEAR-TERM; safety fixes are the deliverable.
- **Differentiation:** Competitors treat privacy as a passthrough claim. DistLLM can ship a mathematically accountable budget (RDP accounting per tenant) over heterogeneous multi-owner nodes — the 'bring your own data' trust story that cloud inference cannot match.
- **Implementation technique:** Fix differential_privacy.privacy_budget_used advanced-composition term (87); fix dp_inference sensitivity to match sigma (88), repair the non-streaming generate branch (80) and make outputs actually noised (81); never charge a tenant's budget when no DP was applied; make security/e2e fail-closed when PyNaCl/session is absent; integrate a maintained privacy library (opacus/autodp) per Digest 95 recommendation and add an accounting unit+integration test.
- **Business value:** The 'data never leaves your devices' narrative (Digest 232/225) is the platform's core differentiator — it only monetizes if it's actually true and testable.
- **Market angle:** Compliance-driven enterprise (HIPAA/GDPR) inference and federated-analytics buyers; the fix becomes a 'we're provably private, not just claiming it' trust asset.
- **Complexity:** high · **Effort:** L (4-8 weeks)
- **Dependencies:** DP correctness fixes (Digest 80, 81, 87, 88); E2E fail-closed enforcement (Digest 92, 134); watermark key management (Digest 84, 85, 86)
- **Builds on:** core/dp_inference.py; core/differential_privacy.py; core/dp_inference/accounting.py; security/e2e.py; security/watermark.py; core/aegis_compliance.py
- **Evidence:** src/distllm/core/dp_inference.py:1-34 (three mechanisms + tenant budgets); src/distllm/core/differential_privacy.py:108-136 (privacy_budget_used advanced composition); src/distllm/dist/privacy.py; Digest 80/81/87/88/92

---

### A8. Bring-Your-Own-GPU marketplace with reputation + carbon ranking (monetization linchpin)
- **Description:** Ship the peer GPU marketplace (dist/marketplace.py) as the real paid tier: GPUListing with hardware specs, price per hour/token, region, carbon_intensity, reputation_score and the effective_score matcher (src/distllm/dist/marketplace.py:50-116), coupled to job lifecycle (MarketplaceJob statuses below line 120), usage metering, and the reputation system (dist/reputation.py). The audit confirms the 7-tier MARKETING matrix's linchpin ('billing/') is unimplemented and no marketplace is buildable yet (Digest 229, 230). NEAR-TERM for v1 matching + billing ledger; MOONSHOT for real-money settlement.
- **Differentiation:** Crowdsourced GPU exchange with trust — the only one of its kind for consumer hardware — turns idle home GPUs into income and idle demand into cheaper compute, while reputation+DP keeps it safe. This is the category DistLLM invented claim on.
- **Implementation technique:** Wire marketplace listings into coordinator provisioning (dist/provisioning, dist/autoscaler) so matched jobs actually schedule; add billing ledger that records provider earnings per job (price_per_hour) and consumer charges, and make it the SAME ledger cost_tracker consumes to satisfy the billing tier; gate high-risk providers by reputation threshold and carbon-aware ranking; add load/stress and state-replication tests for job lifecycle.
- **Business value:** The one credible paid model; turns the network effect into revenue and gives GPU owners a reason to stay on the platform.
- **Market angle:** 'Your gaming PC can earn money while you sleep' — a viral consumer hook that also feeds the DaaS and cross-cluster tiers.
- **Complexity:** high · **Effort:** XL (8-16 weeks for real money / billing)
- **Dependencies:** Implement the billing ledger (Digest 229 linchpin); reputation.py trust scoring; provisioning scheduling path; DP + E2E for cross-owner data (above)
- **Builds on:** dist/marketplace.py; dist/reputation.py; dist/provisioning.py; dist/autoscaler.py; core/cost_tracker.py; dist/daas_server.py
- **Evidence:** src/distllm/dist/marketplace.py:50-116 (GPUListing fields, effective_score), :119+ (MarketplaceJob); Digest 229 (billing linchpin unimplemented), 225 (moat = consumer multi-device pooling)

---

### A9. Chaos engineering suite with learned recovery policies and drill automation
- **Description:** Operationalize the autonomous chaos subsystem (dist/chaos/chaos_orchestrator.py injects NODE_KILL/NETWORK_PARTITION/LATENCY_INJECTION/OOM/STRAGGLER via FaultScenario+FaultTarget at lines 16-38 and learns recovery policies for the autonomous_healer) plus chaos_simulator, recovery_drill and the RedundantExecutor (which is a non-functional stub today — Digest 129). NEAR-TERM for simulation+dri/chaos; MOONSHOT for 'recovers autonomously better than humans'.
- **Differentiation:** Distributed-inference reliability is a trust moat: nobody else ships 'fault drills' that auto-tune the healer. Enterprises adopt the platform that can prove 99.9% under node churn of heterogeneous devices.
- **Implementation technique:** Make RedundantExecutor actually execute (Digest 129) so redundancy>1 works; define an ExperimentResult→policy learning loop (chaos_simulator) that feeds the autonomous_healer thresholds; add a determinism-safe /dev-mode gate so drills run safely in staging then promote to prod; surface scenario library as a `distllm chaos run` CLI with a pass/fail report.
- **Business value:** Warranties/SLAs for the marketplace and DaaS tiers require demonstrated survivability; reduces support burden on self-healing consumer fleets.
- **Market angle:** 'Enterprise-grade reliability for hardware you already own' — differentiator vs hobbyist and cloud-on-prem alternatives.
- **Complexity:** high · **Effort:** L (5-8 weeks)
- **Dependencies:** Fix RedundantExecutor stub (Digest 129); CI chaos/load matrix (Digest 220); unblock test collection (Digest 213/216) so drills can assert pass/fail; autonomous_healer + node_recovery wiring
- **Builds on:** dist/chaos/chaos_orchestrator.py; dist/simulation/chaos_simulator.py; dist/recovery_drill.py; dist/redundant.py; core/autonomous_healer.py; core/predictive_failure.py; core/node_recovery.py
- **Evidence:** src/distllm/dist/chaos/chaos_orchestrator.py:1-38 (autonomous injection, FaultType/FaultScenario); src/distllm/dist/simulation/chaos_simulator.py; src/distllm/dist/redundant.py; Digest 129 (Critical stub), 220 (thin chaos/load CI)

---

### A10. Digital twin capacity/cost planner (what-if simulator for cluster sizing)
- **Description:** Productize the digital twin (dist/simulation/digital_twin.py) that snapshots a production cluster, applies what-if mutations (node type/count/region), and simulates throughput/latency/cost/failure (DigitalTwin.run_simulation, WhatIfEngine.compare_with_baseline at lines 11-19; SimClusterNode/SimRequest lines 36+), paired with schedule_simulator and pipeline simulator. NEAR-TERM (sim core exists; production-snapshot and recommendation UX do not).
- **Differentiation:** 'Should I buy 2x4090 or 1x A100? Where should I add nodes?' is a question no competitor answers for heterogeneous hardware. DistLLM can give a concrete ROI forecast grounded in its own honest scaling-efficiency data (Digest 232).
- **Implementation technique:** Connect DigitalTwin.add_nodes to real coordinator topology snapshot and learned cost/throughput models (dist/partition/learned_cost.py) instead of hardcoded params; add a recommendation engine that maps a target p99/throughput to a minimal-cost node mix; publish the scaling-efficiency curve (tok/s vs node count) as the planner's prior; add cost-optimizer integration so recommendations reconcile with budget alerts.
- **Business value:** Sells the 'try before you buy / plan before you scale' motion and justifies hardware purchase decisions — a consulting-grade deliverable from OSS.
- **Market angle:** Data-center and AI-budget buyers get a defensible, data-backed capacity plan, differentiating from 'just rent from us' cloud alternatives.
- **Complexity:** medium · **Effort:** M (3-6 weeks)
- **Dependencies:** Fill honest perf/TTFT/ITL numbers (Digest 226) to seed the simulator prior; fix learned_cost train/serve feature skew (Digest 143); connect to partition cost_model
- **Builds on:** dist/simulation/digital_twin.py; core/schedule_simulator.py; dist/pipeline/simulator.py; dist/partition/learned_cost.py; dist/partition/cost_model.py; core/cost_optimizer.py
- **Evidence:** src/distllm/dist/simulation/digital_twin.py:11-19 (DigitalTwin/WhatIfEngine usage), :36-69 (SimClusterNode/SimRequest); docs/PERFORMANCE_COMPARISON.md; Digest 226/232 (honest numbers as prior)

---

### A11. Zero-install WebGPU/edge contribution that actually executes compute (consumer pooling wedge)
- **Description:** Fix the decorative WebGPU contribution path so a browser's GPU genuinely runs a slice/forward in the cluster, and add the 'join my machines' zero-install flow. Today webgpu_manager registers BrowserGPU nodes (lines 38-60) but Digest 111 confirms registered browser GPUs never execute any compute, and webgpu_backend.is_available()==True while every forward raises NotImplementedError (Digest 166). Layered on web-llm (docstring lines 20-22). NEAR-TERM for browser-inference; MOONSHOT for browser as a real multi-layer pipeline worker at scale.
- **Differentiation:** The single best on-ramp for the consumer multi-device pooling moat (Digest 225): a visitor's browser tab adds compute with zero install, mirroring the Ollama 'cluster all your devices' gap (Digest 230).
- **Implementation technique:** Either (a) route drafts/first-layers to browser nodes via web-llm forward with real worker protocol, or (b) cut to an honest 'browser runs isolated small models+draft' path that contributes real tokens; implement a WebGPU/SIMD compute kernel path in webgpu_backend so forward() no longer raises; add contribution health/security gates (sandbox, DP for contributed compute, e2e).
- **Business value:** Drops the activation barrier to seconds and makes 'pool your GPUs' literally URL-clickable, feeding every other paid tier.
- **Market angle:** 'Cluster Ollama-style, across all your machines and even a browser tab' — the viral distribution wedge the competitive analysis explicitly names but hasn't built (Digest 230).
- **Complexity:** high · **Effort:** L (6-10 weeks)
- **Dependencies:** Implement webgpu_backend compute (Digest 166, 111); browser↔coordinator worker protocol; DP/E2E for contributed compute (privacy stack above); multi-tenant/geolocation for edge workers
- **Builds on:** core/webgpu_manager.py; backends/webgpu_backend.py; dist/edge_federation.py; dist/edge_cloud.py; dist/p2p/webrtc.py; dist/wide_area.py
- **Evidence:** src/distllm/core/webgpu_manager.py:38-60 (BrowserGPU/WebGPUNode registration), :20-22 (web-llm for in-browser); src/distllm/backends/webgpu_backend.py; Digest 111 (decorative), 166 (NotImplementedError), 230 (Ollama wedge), 225 (consumer pooling moat)

---

### A12. Distributed-inference evaluation harness (honest benchmark, regression-gated)
- **Description:** Ship an evaluation/perf harness that measures TTFT, ITL, scaling-efficiency (tok/s vs node count at 1GbE), and correctness across heterogeneous configs, and gates regressions in CI. Scaffolding exists (core/evaluation_harness.py, benchmarks/evaluation_harness.py, cli/eval.py, api/routes/eval.py), but the audit flags the single biggest adoption risk as empty latency tables and a weak 92 tok/s single-node number vs vLLM 150-250 (Digest 226) and tells the project to publish honest scaling curves (Digest 232). NEAR-TERM.
- **Differentiation:** Trust via measurement: while rivals hand-wave speedups, DistLLM publishes a reproducible efficiency curve and an 'runs models you can't otherwise run + data stays local' framing — the honest positioning the audit says is the actual product story (Digest 232).
- **Implementation technique:** Implement the TASK-015 benchmark sprint (Digest 226/227 recommend it as a blocker): fill TTFT/ITL per config across the claim matrix, generate the tok/s-vs-nodes curve, wire evaluation_harness into a CI regression gate (fix test-collection blockers Digest 213/216 first), and render results into PERFORMANCE_COMPARISON and the digital-twin planner prior.
- **Business value:** Unblocks all adoption and tier pricing: every claim becomes measurable and demoable, and the eval harness is itself a sellable benchmarking product for customers comparing backends.
- **Market angle:** 'We prove it' is the counter to the cloneable-speed claims; turns the honest scaling narrative (not faster-by-Nx, but runs-what-you-can't) into the brand.
- **Complexity:** medium · **Effort:** M (3-5 weeks for a credible v1 curve + regression gate)
- **Dependencies:** Fix CI test-collection/post-import errors (Digest 213, 216); fill the perf matrix (Digest 226); wire eval into digital-twin prior and marketing
- **Builds on:** core/evaluation_harness.py; benchmarks/evaluation_harness.py; cli/eval.py; api/routes/eval.py; api/services/eval_service.py; docs/PERFORMANCE_COMPARISON.md
- **Evidence:** docs/PERFORMANCE_COMPARISON.md (empty latency tables, 92 tok/s vs vLLM); Digest 226 (Critical adoption risk), 232 (honest scaling narrative), 220 (CI matrix breadth), 213/216 (broken test collection)

---
