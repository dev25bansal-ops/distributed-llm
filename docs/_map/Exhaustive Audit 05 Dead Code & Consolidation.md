---
tags:
  - audit
  - exhaustive
date: 2026-08-11
---

# Exhaustive Audit 05 — Dead Code & Consolidation

**← [[Exhaustive Audit 2026-08-11]]**

All findings in category `tech_debt` (Medium/Low and non-verified severities).

**27 findings** — High: 6 · Medium: 11 · Low: 10

---

### F-188 — [High] 3 most over-engineered areas (by business value) vs 3 most under-built

`src/distllm/core` · zone=`strategic` · category=`tech_debt`

- **Summary:** OVER-ENGINEERED: (1) The 'moonshot' themed core modules — wisp_wasm.py (934 LOC), atlas_mesh.py (1437), kraken_chaos.py (1407), voyageur_multimodal.py (1354), aether_federated.py (1069), gaia_cache.py (1330), cortex_multimodel.py (939), hydra_diffusion, arbitrage_engine.py (606), neural_partition_optimizer.py (1398), autoq.py (1545), pulse_performance_model.py (1558), faas_7b.py — ~20-30 modules, thousands of lines each, zero evidence of any user; they are the bulk of the 89K-line core/ and a permanent maintenance + review drain. (2) The speculative-decoding sprawl (dozens of *spec/*draft/*verifier/*msd + dist/speculative/) — a real feature massively over-built relative to the empty single-node latency baseline. (3) The multi-cloud costing/procurement stack (cloud/{aws,azure,gcp}.py, spot_*, arbitrage_engine, pricing_providers) — building cost-arbitrage purchase optimization before product-market fit. UNDER-BUILT: (1) live end-user proof — one-command install, HF Spaces demo, TCO calculator, RBAC matrix docs, benchmark blog are ALL unstarted (TASK-011..015). (2) real monetization metering/billing — MARKETING.md references 'src/distllm/billing/' which does not exist. (3) latency benchmarking + single-node qualifier (TTFT/ITL = em-dashes).
- **Evidence (verbatim):**
```
wisp_wasm 934 LOC, atlas_mesh 1437, kraken_chaos 1407, voyageur_multimodal 1354, gaia_cache 1330, autoq 1545 — with no user evidence; meanwhile TASKS 011-015 (demo, install, benchmark) are all unstarted
```
- **Impact:** Releases the single-founding-team bottleneck: ~30% of core LOC currently consume review/CI/maintenance attention while the monetization, onboarding and benchmark artifacts the business depends on are unbuilt.
- **Effort:** 3-4 weeks to park+reorganize; ongoing redirected effort thereafter
- **Recommendation:** Park (do not delete — they contain salvageable ideas) the moonshot + speculative + procurement modules into an 'experimental/parked' tree that is excluded from CI/release, tier the 6 backends (TASK-017) to surface only vLLM+llama.cpp+Pytorch, and redirect that engineering time to: (a) filling latency/perf benchmarks, (b) one-command install + live demo, (c) the actual billing/metering tier. Reorganize core/ into subdirs (TASK-018) so the maintained surface is obvious to contributors.
- **Strategic value:** Focus is the differentiation lever here. Cutting surface area to the maintained core makes the codebase reviewable (contributor-friendly), reduces reported security surface (the TASKS list is dominated by gaps in exactly these expansive modules), and frees the two things the market tests: perf numbers and a working demo. Tradeoff: parked modules are potential future features, so the park must be documented for revival, not silently deleted.

---

### F-189 — [High] Coordinator lifecycle (start/stop/defrag/_start_subsystem) duplicated wholesale between Coordinator and SubsystemManager

`src/distllm/core/coordinator_subsystem.py:239` · zone=`core-ops-ha` · category=`tech_debt`

- **Summary:** coordinator.py still contains its own full start(), stop(), _defrag_loop, _get_active_temperature, _start_subsystem, _default_utilization_fn, record_metric, _cleanup_stale_results — byte-for-byte near-duplicates of SubsystemManager.start/stop/_defrag_loop/etc. in coordinator_subsystem.py. Only SubsystemManager._on_straggler_detected is ever called (via the StragglerDetector callback). The ~300 lines of SubsystemManager lifecycle methods are dead and prone to drift.
- **Evidence (verbatim):**
```
def start(self, blocking: bool = True, on_stop=None, health_check_interval_s=10.0):     coord = self.coordinator  # full duplicate of Coordinator.start (coordinator.py L939)
```
- **Impact:** Two sources of truth for lifecycle/hot-paths must be kept in sync by hand; a fix applied to one side (e.g. shutdown cordon ordering) silently leaves the other stale, and the dead duplicate misleads maintainers into thinking it is exercised.
- **Effort:** 2-4 hours
- **Reliability:** subsystem_registry.py is registered but start_all is never invoked; SubsystemManager.start/stop/_defrag_loop have zero callers (grep: only _on_straggler_detected and the two delegated election stubs reference it). Both files carry identical tokenizer loading, defrag thread fallback, and shutdown sequences.
- **Recommendation:** Delete the dead lifecycle methods from SubsystemManager and have Coordinator delegate to the retained copies (or vice-versa choose one). Keep only the straggler callback + election delegates that are actually wired. Add a dead-code lint (vulture) to the CI gate.

---

### F-190 — [High] profile_layer_precision / assign_mixed_precision / LayerPrecisionProfile / LayerPrecisionResult duplicated verbatim between quantization_tuner.py and quantization_metrics.py

`src/distllm/dist/partition/quantization_metrics.py:32` · zone=`dist-partition` · category=`tech_debt`

- **Summary:** LayerPrecisionProfile, LayerPrecisionResult(+best_precision), profile_layer_precision, and assign_mixed_precision are implemented identically in quantization_tuner.py (lines 486-776) and quantization_metrics.py (lines 32-355). quantization_metrics even imports SensitivityAnalyzer/MixedPrecisionPlan from quantization_tuner yet re-implements the same functions. This is a classic duplicate-implementation hazard: a numeric fix in one copy silently misses the other.
- **Evidence (verbatim):**
```
@dataclass class LayerPrecisionProfile: ... def profile_layer_precision(model, layer_idx, layer_name, sample_input, ...):  # same as tuner.py:546
```
- **Impact:** The int8/fp8 numeric-correctness bug above (on profile_layer_precision) exists in both copies; keeping them in sync double-manages the same logic.
- **Effort:** 2-4 hours
- **Reliability:** Reading both modules shows identical signatures/bodies for profile_layer_precision and assign_mixed_precision; the module docstring even says it 'depends on quantization_tuner'.
- **Recommendation:** Make quantization_metrics import profile_layer_precision/assign_mixed_precision from quantization_tuner (single source of truth) and keep only the dataclass types or drop the module; add a test that the two don't drift.

---

### F-191 — [High] AutoMixedPrecisionPipeline is defined twice, nearly verbatim — quantization_search.py copy is dead code

`src/distllm/dist/partition/quantization_search.py:29` · zone=`dist-partition` · category=`tech_debt`

- **Summary:** The identical class AutoMixedPrecisionPipeline (with _parse_dtype, run/_run_sequential/_run_microbatched, apply_to_model_weights) exists in both quantization_tuner.py (lines 783-1045) and quantization_search.py (lines 29-290). __init__.py does not export the quantization_search copy, and the tests import the tuner one; the quantization_search.py version is 260+ lines of dead duplication. That module is also misleadingly named — it contains no search.
- **Evidence (verbatim):**
```
class AutoMixedPrecisionPipeline:     def __init__(self, orchestrator, precision_plan, model=None, device='cuda'):         ...  # byte-for-byte same class as quantization_tuner.py:783
```
- **Impact:** Two divergent copies risk drift/fixes applied to only one; dead module adds audit surface and confusion.
- **Effort:** 1-2 hours
- **Reliability:** grep: tests import from quantization_tuner; quantization_search not referenced by __init__ or tests; identical body confirmed by reading both files.
- **Recommendation:** Delete quantization_search.py or re-point it to re-export from quantization_tuner; add a single source of truth and a consumption test so future edits are caught.

---

### F-192 — [High] 9 dead command modules duplicate commands inlined in the 76KB main.py (single-source-of-truth drift)

`src\distllm\cli\main.py:341` · zone=`cli` · category=`tech_debt`

- **Summary:** main.py (76KB, ~44 commands) inlines every command, while src/distllm/cli also contains config_commands.py, system_commands.py, daas_commands.py, federate_commands.py, draft_commands.py, defrag_commands.py, setup_wizard.py, output.py, profile_presets.py that are never imported by main.py or any other module — each duplicates a command main.py already implements (e.g. config_commands.config_validate/config_reference/config_openapi vs main.py 341-443; system_commands.system_run/schedule_viz/coordinator/api/slo_report/cost_avoid/doctor/observe vs main.py 289-339,1248-1435,1673-1702; daas/federate/draft/defrag equivalents). The separate files were clearly intended as a refactor that was never wired in.
- **Evidence (verbatim):**
```
@config_app.command("setup") def config_setup(...):     from distllm.cli.setup import run_setup     run_setup(config_path, console)
```
- **Impact:** Two sources of truth: the 'extracted' modules and main.py drift independently, so a fix applied to the wrong copy (e.g. the dead config_commands.config_validate) ships unexercised; ~9 modules of dead code remain on-disk and confuse contributors and readers about which implementation is authoritative.
- **Reliability:** Evidence: `grep -rn` for `import config_commands|import system_commands|import daas_commands|import federate_commands|import draft_commands|import defrag_commands|import setup_wizard|import output|import profile_presets` across src/ and tests/ returns only self-references. pyproject.toml registers only `distllm = distllm.cli.main:main` as the console entry point, so main.py is the sole command surface; these modules are unreachable. status.py is referenced only by tests (test_cli.py, test_cli_modules.py), not by main.py.
- **Recommendation:** Consolidation: (1) delete dead modules whose logic is fully inlined in main.py (config_commands, system_commands, daas_commands, federate_commands, draft_commands, defrag_commands, setup_wizard, output, profile_presets), and/or (2) split main.py into per-group register functions (register_cluster_app(app), register_config_app(app), ...) and have main import them, so each command has exactly one implementation. Wire status.show_status into `cluster status` or delete status.py. Add a test that enumerates all registered commands and asserts no name is registered twice from both a module and main.py.
- **Strategic value:** A single registered command tree removes the risk that error-handling/security fixes land only in the dead copy and halves the CLI maintenance surface, directly supporting the production-readiness mandate; the refactor also unlocks per-module unit tests without the huge main.py import cost.

---

### F-193 — [High] Three parallel vectorstore implementations with incompatible interfaces; the factory-registered providers and legacy adapters must be consolidated

`src\distllm\core\vectorstore\chroma_store.py:61` · zone=`core-gen-rag` · category=`tech_debt`

- **Summary:** There are two disjoint vectorstore stacks. (1) factory-registered `vectorstore/providers/{pinecone,qdrant,weaviate,milvus}.py` with the `VectorDBInterface` signature `upsert(vectors, metadata, *, namespace, batch_size)`; (2) legacy `vectorstore/{chroma_store,pgvector_store,qdrant_store}.py` which alias `VectorDBInterface as VectorStore` but override EVERY method with a different signature (`upsert(embeddings, ids, metadata)`, `query(embedding, top_k=10)`), so they are not interchangeable with the interface — a caller using documented kwargs (metadata_filter/namespace) on ChromaStore/PGVectorStore/QdrantStore raises TypeError. The legacy trio is not registered in the factory and is referenced only by the back-compat tests. Qdrant additionally has TWO implementations with different id-hashing (_to_point_id vs the provider's raw ids).
- **Evidence (verbatim):**
```
class ChromaStore(VectorStore):     def upsert(self, embeddings, ids, metadata=None) -> None:  # VS ABC upsert(vectors, metadata, *, namespace, batch_size)
```
- **Impact:** Maintainers must keep two divergent RAG stacks in sync; the legacy classes silently violate the abstraction they claim to implement, producing confusing type errors for anyone using the interface uniformly.
- **Recommendation:** Consolidation story: promote the provider interface as the single contract; either rework chroma/pgvector/qdrant_store to the ABC signature or delete them (audit finding B21 already tracks deletion) once no call site references them (grep shows only tests). Add a shared conformance suite run against all providers to lock the interface.

---

### F-194 — [Medium] Massive skip cluster resolved by deleting ~74 broken test files instead of fixing the underlying modules

`allskip.txt:1` · zone=`tooling-tests` · category=`tech_debt`

- **Summary:** The audit asked to characterise the enormous prior test-skipping. Two distinct old clusters: (A) ~83 files imported `distllm.core.*` modules (agent_loop, rag_pipeline, hybrid_parallel, moe_*, geo_router, gossip_protocol, model_registry, etc.) that did not exist -> the response was to delete 74 of those files and guard a handful with module-level `pytest.skip(...)` (add_skips.py), rather than implement/rename the modules. (B) individual `*_error*`/`*_no_error*` tests skip-annotated en masse. The 11 current module-level guards (test_high_severity_gpu.py, security/test_ssrf_federation.py, etc.) are legitimate feature/env-gated skips.
- **Evidence (verbatim):**
```
of 83 files listed in allskip.txt (Jul 14), 74 no longer exist in tests/ today; survivors reassigned to feature/env-gated skips
```
- **Impact:** Coverage for those featured modules is gone entirely, not deferred; the audit's '695 source modules' no longer matches reality (current src has 736 non-__init__ .py), and the deletion hides which areas were never implemented.
- **Effort:** 2-4 hours analysis
- **Reliability:** Cross-referenced allskip.txt vs current tree; add_skips.py shows the guard mechanism.
- **Recommendation:** Re-introduce the deleted test files' intent as lightweight smoke tests against whatever module now provides the capability (many were re-homed, e.g. rag -> core/vectorstore, gossip -> dist/p2p/gossip). Track the genuinely unimplemented surfaces (agent_loop, hybrid_parallel, moe_orchestrator) as explicit tickets; do not let 'delete tests' be the default remediation for missing implementation.

---

### F-195 — [Medium] Three parallel PagedAttention KV managers (single-node, quantized drop-in, distributed) with no shared core

`src/distllm/backends/paged_attention_quantized.py:249` · zone=`backends-config-cloud` · category=`tech_debt`

- **Summary:** backends/paged_attention.py (PagedAttentionManager), backends/paged_attention_quantized.py (QuantizedPagedAttentionManager), and dist/attention.py (BlockPool-based manager) each reimplement block allocation/refcount/gather. paged_attention.py's own docstring directs distributed use to dist/attention.py, confirming the split. Quantized manager stores K/V as (2,heads,block,head_dim) with channel 0/1 selectors duplicated across write_kv/gather_kv.
- **Evidence (verbatim):**
```
block.key_quantized[0, :, start:start + take, :] = k_q block.value_quantized[1, :, start:start + take, :] = v_q
```
- **Impact:** 3x surface to audit/fix for KV bugs; subtle drift (e.g. ref-count vs LRU eviction) already forcing doc notecalls to the reader.
- **Effort:** 2-3 days
- **Reliability:** Three distinct managers in the same package implementing the same allocate/write/gather lifecycle.
- **Recommendation:** Extract a common block-pool/refcount core; make the quantized manager a storage-strategy over it and dist/attention a distributed-storage strategy, sharing allocation/gather logic. Add conformance tests for equivalent behavior across the three.

---

### F-196 — [Medium] Four independent autoscaling implementations with no shared decision model

`src/distllm/core/aria_autoscaler.py:1025` · zone=`core-ops-ha` · category=`tech_debt`

- **Summary:** Autoscaling is spread across IntelligentAutoscaler (core,intelligent_autoscaler.py, wired-but-inert), Aria + TrafficForecaster + CarbonAwareScaler + PredictiveScaler + HPAMetrics (aria_autoscaler.py, unwired), dist/autoscaler.py, and PredictiveScaler nodes. Each rebuilds the same min/max/cooldown/threshold logic with different, incompatible ScalingDecision/ScaleDecision/ScalingPlan types and units (util as 0-1 vs 0-100, forecast as load vs node count).
- **Evidence (verbatim):**
```
def _predict_load(self) -> int:     ...     if trend > 10 and avg_util > 60:         return recent[-1].current_nodes + 1     return recent[-1].current_nodes  # returns NODE COUNT, not a load prediction
```
- **Impact:** Autoscaling decisions are inconsistent across subsystems, forecasts and reactive logic are in different units (a classic unit-confusion source), and the richer Aria/carbon logic is stranded and unreachable.
- **Effort:** 2-3 days
- **Reliability:** aria_autoscaler.py has no importers in prod. intelligent_autoscaler._predict_load returns a node-count but is used as 'predicted_load' in max(reactive_target, predicted_load) — mixing load and count semantics. Four files each re-implement cooldown/threshold scaling.
- **Recommendation:** Consolidate on Aria's components as the single decision engine (it has forecast+carbon+cooldown), expose it behind one interface returning a unified scaling decision, and wire IntelligentAutoscaler as a thin compatibility wrapper. Delete the third copy.

---

### F-197 — [Medium] Cache zone is ~15 overlapping implementations; only 3 are wired to production (consolidation map)

`src/distllm/core/cache_manager.py:141` · zone=`core-cache` · category=`tech_debt`

- **Summary:** Grep across the repo shows only dist/prefix_cache.PrefixCache (via core/cache_manager.py), core/semantic_cache.SemanticCache (via coordinator._start_subsystem line 1003 + plugins/cache_plugin.py), and core/kv_cache.KVCache are imported by production code. The following are DEAD (zero importers outside their own file): core/radix_tree_cache (a full alternate trie prefix cache), core/hybrid_cache, core/block_eviction_policy, core/block_affinity_tracker, core/gaia_cache, core/redis_prompt_cache, core/cache_coherence, core/dynamic_memory_budget, core/cache_template_warmer, core/cache_bench, core/cache_doctor, core/cache_snapshot, core/cache_eviction. TTLPolicy+SemanticGrouping are duplicated in BOTH core/cache_eviction.py and dist/cache.py; prefix_cache.py (core) is a deprecation shim re-exporting dist's. core/cache_manager.py also instantiates its own 3-tier GPU/CPU/SSD hash cache (tiers dict, _tier_store/_tier_lookup) plus a separate ghost cache, duplicating the dist PrefixCache it also holds.
- **Evidence (verbatim):**
```
radix_tree_cache/hybrid_cache/block_eviction_policy/block_affinity_tracker/gaia_cache/redis_prompt_cache/cache_coherence/dynamic_memory_budget/cache_template_warmer/cache_bench: 0 importers in src. TTLPolicy defined in core/cache_eviction.py AND dist/cache.py. core/prefix_cache.py re-exports dist PrefixCache.
```
- **Impact:** Maintenance hazard: 5+ independent eviction / prefix / semantic cache designs, each with its own bugs and memory accounting; ~7.2k lines to maintain; contradictory behavior depending on which one an integration happens to import.
- **Effort:** 3-5 days
- **Reliability:** grep -rl 'distllm.core.radix_tree_cache|hybrid_cache|block_eviction_policy|gaia_cache|redis_prompt_cache|cache_coherence|dynamic_memory_budget|cache_template_warmer|cache_bench|cache_doctor' src yields only the defining file; semantic_cache importers = coordinator.py, coordinator_subsystem.py, plugins/cache_plugin.py; PrefixCache import = cache_manager.py.
- **Recommendation:** Adopt ONE prefix-cache abstraction (keep dist/prefix_cache.PrefixCache; it has the most features: tenant scope, bloom filter, hybrid LFU+LRU, TTL) and ONE eviction module (keep dist/cache policy; delete core/cache_eviction equivalent). Delete or explicitly deprecate: radix_tree_cache, hybrid_cache, block_eviction_policy, block_affinity_tracker, gaia_cache, redis_prompt_cache, cache_coherence, dynamic_memory_budget, cache_template_warmer, cache_bench, cache_doctor, cache_snapshot, core/cache_eviction. Move cache_manager's bespoke GPU/CPU/SSD tier dict behind the same interface or drop it.

---

### F-198 — [Medium] Four leader-election / split-brain mechanisms with no coordination

`src/distllm/core/cluster_state_store.py:196` · zone=`core-ops-ha` · category=`tech_debt`

- **Summary:** Split-brain and leadership are independently re-implemented in RayFaultTolerance (core/ha_coordinator.py, used by CoordinatorElection), core/split_brain.py SplitBrainDetector (used by FederationCoordinator), ClusterStateStore (Redis lock leader, registered in __init__ but not wired into coordinator prod paths), and a second SplitBrainDetector in dist/byzantine.py. Each has different quorum/fence semantics; a single coordinator HA setup will never reconcile them with the federation detector.
- **Evidence (verbatim):**
```
def elect_leader(self) -> bool:     if self._backend.acquire_lock(self._lock_key(), self._node_id, ttl=self._ttl): ... self._leader_id = self._node_id
```
- **Impact:** Coordinator-level and federation-level partition decisions can disagree (one says quorum met, other says split), undermining the fencing guarantee; operators get no single coherent HA story.
- **Effort:** 2-3 days
- **Reliability:** grep: ClusterStateStore referenced only in core/__init__.py export; RayFaultTolerance used by CoordinatorElection; core/split_brain.SplitBrainDetector used by dist/federation.py; dist/byzantine.SplitBrainDetector is a vector-clock variant. No shared interface or cross-check between coordinator-election quorum and federation quorum.
- **Recommendation:** Pick RayFaultTolerance (or ClusterStateStore) as the coordinator authority and have FederationCoordinator surface its partition verdict as input to it. Extract a common election/quorum interface and make the others thin adapters; document the fence-token contract once.

---

### F-199 — [Medium] Two independent cost engines (GPU-hour estimate vs token-price) with no reconciliation; zero-cost requests misclassified

`src/distllm/core/cost_tracker.py:302` · zone=`core-perf-obs` · category=`tech_debt`

- **Summary:** cost_tracker.estimate_cost() prices via GPU_COST_PER_HOUR * gpu_seconds (idealized throughput, ignores actual duration_ms), while usage_meter.record_request() prices via input*input_price + output*output_price per 1000 tokens. The API middleware records into cost_tracker, while billing/quota uses usage_meter — a request can be reported at materially different dollar figures in headers vs. the invoice. Also both use truthiness (`cost_usd > 0`, `input_cost_per_token or ...`) so a legitimately zero cost cannot be represented and falls back to an estimate, double- or wrongly charging.
- **Evidence (verbatim):**
```
for r in self._completed ... estimate = self.estimate_cost(input_tokens, output_tokens, model_name, gpu_type) # cost from idealized total/max(tps) GPU seconds; usage_meter uses token prices if cost_usd > 0: cost = cost_usd else: cost = (input/1000)*input_price + (output/1000)*output_price
```
- **Impact:** Customers see one number in X-DistLLM-Cost headers and another on the invoice; auditors can't reconcile. Zero-cost and promo requests are misbilled. Undermines the cost-accounting correctness thread that is a GA sign-off blocker.
- **Effort:** 2-3 days
- **Reliability:** Same request (512 in/128 out, A100, 5s real) returns different USD from CostTracker.record_request (GPU-hour, duration-independent) vs UsageMeter.record_request (token-price). Reading both formulas confirms divergence.
- **Recommendation:** Designate one system of record (usage_meter, which has real gpu_time_seconds + cost_usd fields). Have cost_tracker.record_request feed cost_usd from usage_meter rather than recomputing, and change the zero-guards to explicit is None checks so a free/zero-cost record can be stored. Add a reconciliation test asserting header cost == billed cost for the same request.

---

### F-200 — [Medium] Three divergent, duplicated PII/anonymizer pattern sets across core and security

`src/distllm/core/request_auditor.py:24` · zone=`core-priv-sec` · category=`tech_debt`

- **Summary:** PII detection/redaction is re-implemented in at least three places with slightly different coverage: `core/request_auditor.py` PII_PATTERNS (7 patterns, no token run), `security/log_redaction.py` _PII_PATTERNS (adds long_token + aws_key), and `core/differential_privacy.py` InputAnonymizer._PATTERNS (emails/phones/SSN/card/IP only, different substitutions). They drift independently (as evidenced by the long_token gap). A single source of truth should be shared.
- **Evidence (verbatim):**
```
PII_PATTERNS = {... "api_key": re.compile(r"(?i)(sk-[a-zA-Z0-9]{20,}|...)") ...}  # vs log_redaction adds "long_token"/"aws_key"
```
- **Impact:** Maintenance risk and inconsistent redaction behavior across the stack; a secret redacted in logs may leak through the auditor and vice versa.
- **Effort:** Half day
- **Reliability:** Set DISTLLM auto-admin key; log_redaction contains_pii() true (long_token matches) while request_auditor PIIInspector.inspect() returns [] for the same string.
- **Recommendation:** Extract one `distllm.security.secrets` module exposing a canonical pattern registry + inspect(s)/redact() and have request_auditor, log_redaction, and the InputAnonymizer delegate to it; keep annotations for which detector/tokenization is used.

---

### F-201 — [Medium] BlockTransferClient is a silent stub that always fails

`src/distllm/dist/block_transfer_service.py:219` · zone=`dist-net` · category=`tech_debt`

- **Summary:** BlockTransferClient.fetch_blocks references BlockTransferStub, which is not defined anywhere in the module — so even when grpc IS importable, `stub = BlockTransferStub(channel)` raises NameError, caught by the blanket except and returned as None. Similarly BlockTransferServer.start() (lines 139-143) only sets a flag and logs 'listening on port' but never binds a socket or serves requests; handle_request is in-process only. The advertised 'gRPC transfer layer' is a stub on both sides and the docstring doc only claims it as a stub.
- **Evidence (verbatim):**
```
import grpc ...; stub = BlockTransferStub(channel); request = BlockTransferRequest(block_ids=block_ids); response = stub.FetchBlocks(request, timeout=self._timeout)
```
- **Impact:** Peer block/transfer features silently never work in production; a caller gets None and may fall back to local (broken) cache, or worse continue with partial state believing a transfer happened.
- **Reliability:** fetch_blocks always returns None because BlockTransferStub is undefined -> NameError swallowed line 236-238.
- **Recommendation:** Either implement real gRPC (define BlockTransferStub from a .proto, bind the server with server.start()), or delete the misleading fetch_blocks/start and keep only the pure in-process BlockData helpers. Make the stub fail loudly (raise) instead of silently returning None when grpc is present.

---

### F-202 — [Medium] Two complete parallel prompt registries; the domain-file/prompt_def/management tree is dead

`src/distllm/prompts/library.py:17` · zone=`ops-utils` · category=`tech_debt`

- **Summary:** library.py is the live registry (self-contained SystemPromptDef + _reg into the SYSTEM_PROMPTS dict, imported by __init__, api/routes/prompts.py, cli/prompts.py). Independently, prompt_def.py defines a second SystemPromptDef dataclass + _reg appending to a separate _PROMPTS list, and the 13 domain files (code.py, writing.py, analysis.py...) register the same prompt IDs through it, with management.py mirroring library's lookup helpers on that dead list. Nothing in src imports the domain modules, so _PROMPTS stays empty and the whole subtree is inert; the duplicated dataclass is a divergence risk (library has version field, prompt_def doesn't).
- **Evidence (verbatim):**
```
SystemPromptDef + SYSTEM_PROMPTS dict + inline _reg (lines 6-24); prompt_def.py duplicates SystemPromptDef/_reg/_PROMPTS (lines 10-33); domain files import from distllm.prompts.prompt_def (e.g. code.py line 1)
```
- **Impact:** ~1200 lines of dead/duplicated code; a maintainer editing the shared prompt content in one tree silently diverges from the other; confusing for contributors.
- **Effort:** 3-6 hours
- **Reliability:** grep confirms only library/templates/engine are imported by live consumers; prompt_def/domain modules have no importers in src.
- **Recommendation:** Consolidate to one canonical registry: keep library.py, delete prompt_def.py + the 13 domain modules + management.py (their content is already in library.py), or invert - make domain modules _reg into library.SYSTEM_PROMPTS and drop prompt_def. Delete the orphan SystemPromptDef duplicate and add a test asserting SYSTEM_PROMPTS count/ids to guard the merge.

---

### F-203 — [Medium] Audit callback fires only via resolve(); route() and route_with_context() never populate the audit trail

`src\distllm\core\model_router.py:398` · zone=`core-router-sched` · category=`tech_debt`

- **Summary:** set_audit_callback registers a hook intended to record every routing decision, but _record_audit is invoked only inside resolve() (lines 398, 416, 420). route() and route_with_context() — the primary conversation path — never call it, so the audit/monitoring hook is silently silent for the majority of traffic. This is both a behavioral inconsistency and a duplication smell across three near-identical routing methods.
- **Evidence (verbatim):**
```
_record_audit called only in resolve(): `self._record_audit("", text, rule.name, target, 0.9)` (and 416, 420). route() (423-501) and route_with_context() (503-681) contain the same rule loop but no _record_audit call.
```
- **Impact:** Compliance/observability hook silently loses per-decision metadata for the main path; future fixes must be applied in three places, inviting drift.
- **Effort:** 1-2 hours
- **Reliability:** Router users exercise route_with_context() in production (the docstring calls it 'the primary integration point with the ModelRouter ecosystem'); the audit callback receives zero events even under heavy routing. Verify by set_audit_callback + calling route()/route_with_context() and observing the callback never fires.
- **Recommendation:** Add _record_audit(...) calls to the matched-rule, workload, and fallback branches of route() and route_with_context() (mirroring resolve). Then consolidate: resolve/route/route_with_context share ~40 lines of identical match+stats+record logic per branch; extract a single _match_and_record(text_lower, available_models, start) helper returning a RouteMatch and have all three call it, removing the triple duplication.

---

### F-204 — [Medium] 9 parallel speculative-verifier implementations share acceptance math but disagree on the core indexing convention (primary root cause of the off-by-one above)

`src\distllm\core\speculative_decoder.py:23` · zone=`core-decoding` · category=`tech_debt`

- **Summary:** speculative_decoder.py (SpeculativeDecoder, SelfSpeculativeDecoder, MultiDraftSpeculativeDecoder, TreeDraftSpeculativeDecoder with both a sequential _verify_tree and a batched _verify_tree_batched), tree_speculative_decoder.py (own TreeNode/TreeSpecStats/TreeSpeculativeDecoder), multi_draft_verifier.py (MultiDraftVerifier + TreeMultiDraftVerifier), distributed_speculative.py, mtp_head.py MTPDecoder, draft_tree.py DraftTree, and compressed_speculative.py each re-implement target logits indexing and rejection sampling independently. 4 chose `P+i`, 5 chose `P-1+i`, so the same logical operation is correct in one file and wrong in its sibling. The SpecDecoderBase mixin (commit 2a7e561) consolidated only `_sample`, not the verifier used on the critical path.
- **Evidence (verbatim):**
```
class SpecDecoderBase:  # Mixin with shared _sample and stats ... Eliminates 7x duplicated ``_sample`` methods
```
- **Impact:** The divergence IS the bug source: each of the 4 off-by-one verifiers is a copy that omitted the -1. Consolidation eliminates the entire class of index-drift bugs and makes acceptance rate comparable across strategies.
- **Reliability:** Code trace: correct = draft_tree.py:201 (`prefix_len + i - 1`), speculative_decoder.py:210/216, distributed_speculative.py:1171 (prefix-1 then +i); wrong = tree_speculative_decoder.py:326, speculative_decoder.py:940/949, multi_draft_verifier.py:121, mtp_head.py:407-412. Two conventions coexist untouched since they were added.
- **Recommendation:** Extract one `verify_draft_tokens(prefix, draft_tokens, target_logits, draft_logprobs, temperature)` pure function (given `full_input` it computes positions internally) and have every class call it. Move the tree-specific flatten/verify into DraftTree so tree_speculative_decoder.py, TreeDraftSpeculativeDecoder, TreeMultiDraftVerifier, and TreeSpeculativeDecoder delegate to one verified implementation. Delete the dead duplicates or route them through the shared helper.

---

### F-205 — [Low] Three near-identical streaming/delta parsers duplicated across adapters drift from the SDK's actual shapes

`integrations/crewai/src/distllm_crewai/llm.py:75` · zone=`integrations` · category=`tech_debt`

- **Summary:** The `delta = chunk if isinstance(chunk, dict) else {}; content = delta.get('choices',[{}])[0].get('delta',{}).get('content','')` idiom is copy-pasted in langchain chat_models._stream/_astream, llamaindex stream_chat/astream_chat, and crewai generate_stream/agenerate_stream. Because the async SDK yields content strings rather than this nested-choice dict, the copied consumer logic is wrong for async in every copy. This is convergence evidence that a single shared stream-normalizer in distllm.sdk (mirroring BaseToolProvider) would have caught the bug once instead of four times.
- **Evidence (verbatim):**
```
delta = chunk if isinstance(chunk, dict) else {}             content = (                 delta.get("choices", [{}])[0].get("delta", {}).get("content", "")             )
```
- **Impact:** Four separate copies diverge from the SDK contract and each needs an independent fix; no single place to correct the chunk contract.
- **Effort:** 2-3 hours
- **Recommendation:** Add a `normalize_stream_chunk(chunk) -> str` helper in distllm.sdk.streaming and consume it from all adapters, eliminating the isinstance(dict) branch that is wrong for the async SDK.

---

### F-206 — [Low] require_coordinator is referenced by 5 route modules but never defined anywhere, silently guaranteeing those routers can never be mounted

`src/distllm/api/routes/api_keys.py:19` · zone=`api-gateway` · category=`tech_debt`

- **Summary:** routes/files.py:29, api_keys.py:19, experiments.py:37, fine_tuning.py:28, webhooks.py:31 all import `require_coordinator` from ..auth_deps, but auth_deps.py (the whole file) defines only require_role. Grep for 'def require_coordinator' across src/distllm returns nothing. These routers are therefore un-importable (ImportError) and can never be wired into the app — they are dead by construction. This also means none of these files' endpoints (files, api keys, experiments, fine-tuning, webhooks) are reachable, and their intended authorization contract is entirely unimplemented.
- **Evidence (verbatim):**
```
from ..auth_deps import require_coordinator, require_role (19); dependencies=[Depends(require_coordinator), Depends(require_role("admin"))] (24); 'def require_coordinator' not found in src
```
- **Impact:** API-key management, file upload, experiments, fine-tuning and webhook surface are all unimplemented/unreachable; misleading auth contract; prevents operators from enabling these features safely.
- **Effort:** 2-4 hours
- **Reliability:** Importing any of the 5 modules raises ImportError (NameError/ImportError on the missing name), so server.py would fail to boot if they were added to routes/__init__.py. They are currently simply absent from the include list, masking the defect.
- **Recommendation:** Define require_coordinator in auth_deps.py as a dependency that checks coordinator availability (and pairs with require_role for admin gating), or remove the references and mount the routers deliberately with explicit role checks. Add a startup test that every route module in routes/ imports cleanly so dead/broken routers surface instead of silently vanishing.

---

### F-207 — [Low] VLLMNodeAdapter and TensorRTLLMAdapter duplicate _extract_inner_model + pipeline forward + single-node forward

`src/distllm/backends/tensorrt_backend.py:183` · zone=`backends-config-cloud` · category=`tech_debt`

- **Summary:** TensorRTLLMAdapter replicates VLLMNodeAdapter's _extract_inner_model (version-path guessing) and _forward_hidden_states/_forward_with_input_ids verbatim (tensorrt_backend.py:183-262 vs vllm_backend.py:201-261). Both wrap an LLM(...) engine and differ only by package import. The device-guessing + extract paths must be maintained twice and will drift with SDK version changes.
- **Evidence (verbatim):**
```
def _forward_with_input_ids(...): if self._is_pipeline_mode: raise NotImplementedError("TensorRT-LLM pipeline mode does not support input_ids-based forward.")
```
- **Impact:** Version-compat bugs fixed in one engine silently remain in the other; higher maintenance cost.
- **Effort:** 1 day
- **Reliability:** tensorrt_backend.py:244-262 mirrors vllm_backend.py:168-199.
- **Recommendation:** Parameterize a shared 'LLM-Engine-Adapter' base by (import config, model-factory) and have both adapters subclass it, overriding only package-specific load_model/generate salt.

---

### F-208 — [Low] Two unrelated 'RedundantExecutor' classes share a name in one package; export resolves to the broken stub

`src/distllm/dist/__init__.py:31` · zone=`dist-exec` · category=`tech_debt`

- **Summary:** Both `redundant.py` (the 165-line, stub-broken pipeline redundancy, exports `RedundantExecutor`) and `redundant_executor.py` (the ~55KB training-oriented FaultTolerantCanvas executor, also defines `class RedundantExecutor`) exist. `__init__.py`/`__init__.pyi` export the former (the broken one), while the substantial implementation in redundant_executor.py is the one that has real machine-learning recovery semantics. Two classes with the identical public name in the same package is a maintenance trap: importers can silently get either implementation.
- **Evidence (verbatim):**
```
_register("distllm.dist.redundant", "RedundantExecutor")  # vs redundant_executor.py also defines class RedundantExecutor
```
- **Impact:** Confusion and accidental import of the stub; one 'RedundantExecutor' fails at runtime while a working, differently-purposed class is shadowed.
- **Reliability:** from distllm.dist import RedundantExecutor resolves to redundant.py's stub (fails when redundancy>1), while redundant_executor.RedundantExecutor (a distinct, working training executor) is unreachable by that name.
- **Recommendation:** Rename one class (e.g., redundant.py → keep generic namespace but name its class `PipelineRedundantExecutor`, and export redundant_executor's `RedundantExecutor` as the canonical name), or delete the broken redundant.py stub entirely and route its callers to the working executor. Consolidate to a single redundancy abstraction with one documented import.

---

### F-209 — [Low] Several parallel, overlapping KV-transfer / serialization implementations with divergent behavior

`src/distllm/dist/zero_copy.py:145` · zone=`dist-net` · category=`tech_debt`

- **Summary:** At least five modules independently serialize KV cache tensors to bytes with different format/error semantics: block_transfer_service (numpy.tobytes + BlockTransferStub stub), streaming_kv_transfer (chunked numpy with broken bf16), zero_copy (IPC/RDMA/NCCL with broken recv), pipeline/serialization (protobuf, CUDA race), and compression_negotiation (adaptive zstd/fp8, double-compress bug). They use different chunking, different dtype handling, and disagree on wire format — the bugs above (bf16, zeros, double-compress, race) are all manifestations of this fragmentation. Gossip also re-defines select_peer twice (gossip.py:564 and :639), the first dead.
- **Evidence (verbatim):**
```
class ZeroCopyTransferEngine: ... (block_transfer_service / streaming_kv_transfer / cross_cluster / pipeline.serialization / compression_negotiation each serialize KV to bytes independently)
```
- **Impact:** Redundant maintenance, inconsistent security/correctness, and five places to fix the same class of tensor-integrity bug.
- **Recommendation:** Consolidate onto one serialization/transfer abstraction (recommend the protobuf path as the canonical wire format, with a single optional compression stage) and have block_transfer_service/streaming_kv_transfer/zero_copy delegate to it. Single shared code fixes bf16, scale, and chunk-framing consistently.

---

### F-210 — [Low] Built-in plugins are never auto-registered and no distllm.plugins entry points are declared

`src/distllm/plugins/__init__.py:1` · zone=`ops-utils` · category=`tech_debt`

- **Summary:** plugins/__init__.py is docstring-only (no re-exports, no registration of builtin plugins), and pyproject declares no [project.entry-points."distllm.plugins"]. Consequently PluginRegistry.discover()/PluginSystem.discover_entry_points(group='distllm.plugins') enumerate nothing by default, and the builtin AuthPlugin/RateLimitPlugin/etc. must be wired manually through config 'module' strings. The package docstrings promise entry-point discovery and a global registry that are inert out of the box.
- **Evidence (verbatim):**
```
no [project.entry-points] table in pyproject; __init__.py contains only a docstring (lines 1-14); registry.discover() default group 'distllm.plugins' (registry.py line 29)
```
- **Impact:** Operators who expect builtin plugins to activate (per docs) find them silent; unsigned discover-by-entry-point path is also the RCE surface noted separately.
- **Effort:** 2-4 hours
- **Reliability:** Call PluginRegistry().discover() on a fresh install -> returns 0; no builtin is registered.
- **Recommendation:** Declare [project.entry-points."distllm.plugins"] for each builtin plugin (name -> module:Factory) and re-export the plugin classes from plugins/__init__, or document the explicit config 'module' wiring as the only supported path.

---

### F-211 — [Low] _SlidingWindowCounter is duplicated (builtin.py and auth_plugin.py) despite a shared utils package

`src/distllm/plugins/auth_plugin.py:100` · zone=`ops-utils` · category=`tech_debt`

- **Summary:** The exact same thread-safe sliding-window counter class is copy-pasted in plugins/builtin.py (lines 29-55) and plugins/auth_plugin.py (lines 100-141), the latter even commenting it 'Mirrors the implementation in builtin.RateLimitPlugin to keep the plugin self-contained'. This is exactly what src/utils exists for; retention logic can drift.
- **Evidence (verbatim):**
```
class _SlidingWindowCounter identical in builtin.py (lines 29-55) and auth_plugin.py (lines 100-141)
```
- **Impact:** Two copies must be kept in sync; any rate-limit fix (e.g. per-key windows) must be duplicated, inviting divergence bugs.
- **Effort:** 1 hour
- **Reliability:** Grep shows the class body repeated verbatim in both files with no shared import.
- **Recommendation:** Move _SlidingWindowCounter to src/distllm/utils/scheduling.py (or a new utils/rate_limit.py) and import from both plugins.

---

### F-212 — [Low] Duplicate n-gram feature hashing: routing_extensions._simple_embed and learning_router._feature_hash

`src\distllm\core\learning_router.py:45` · zone=`core-router-sched` · category=`tech_debt`

- **Summary:** Two near-identical feature-hashing embedders compute 2/3/4-char n-gram hashes with sign from h//dim%2 and L2 normalization. They are maintained separately and can drift (e.g. one uses md5, the other sha256), which changes feature space and bandit buckets across routers unexpectedly.
- **Evidence (verbatim):**
```
learning_router._feature_hash: `for n in (2,3,4): ... h = int(hashlib.md5(ngram).hexdigest(),16); idx = h % num_buckets; sign = 1.0 if (h // num_buckets) % 2 == 0 else -1.0` → L2 normalize. routing_extensions._simple_embed (lines 230-246) is the same using sha256.
```
- **Impact:** Redundant code; divergent hash choices silently change context-bucket assignments and model-selection behavior between routers.
- **Effort:** 0.5 hours
- **Reliability:** Both modules define the identical algorithm; grep confirms the duplicated n-gram/idx/sign/L2 normalization code blocks.
- **Recommendation:** Extract a single shared feature_hasher(text, num_buckets, hash_fn) used by both LearningRouter and SemanticRouter, parameterizing the hash function; add a test asserting both produce identical vectors for the same input when given the same hash_fn.

---

### F-213 — [Low] priority_heap.py is an orphaned duplicate of the inlined promote_request in BatchScheduler

`src\distllm\core\priority_heap.py:12` · zone=`core-router-sched` · category=`tech_debt`

- **Summary:** The scheduler decomposition plan extracted priority_heap.promote_request/rebuild_pending_index, but BatchScheduler still has its own identical inlined promote_request (heap _siftdown/_siftup + _pending_index invalidation) and never imports priority_heap.py. The standalone module is dead code that will drift from the live implementation.
- **Evidence (verbatim):**
```
priority_heap defines `def promote_request(pending_heap, request_id, new_priority, pending_index)`; batch_scheduler.py defines its own `def promote_request(self, request_id, new_priority)` at line 1101 with the same _siftdown/_siftup logic and imports at top of batch_scheduler do not include priority_heap (grep shows batch_scheduler never imports it; only tests/_map references exist).
```
- **Impact:** Two copies of the same priority-promote logic; a future fix applied to one will not be reflected in the other.
- **Effort:** 0.5 hours
- **Reliability:** Search `from distllm.core.priority_heap import` across src — no production import exists; the module appears only in decomposition plans and coverage reports.
- **Recommendation:** Either delete priority_heap.py or make batch_scheduler.promote_request delegate to it (passing self._pending_heap, self._pending_index) so there is one authoritative O(log n) implementation with one test suite (push/pop/promote invariants).

---

### F-214 — [Low] Two independent structured-output repair stacks exist (core OutputRepairer vs dist RepairOrchestrator) with divergent heuristics

`src\distllm\dist\structured_output\engine.py:44` · zone=`core-gen-rag` · category=`tech_debt`

- **Summary:** core/structured_output/validator.py OutputRepairer closes unmatched braces/brackets/quotes heuristically, while dist/structured_output/engine.py RepairOrchestrator implements heuristic/truncate/regenerate strategies AND a per-token validate_token that uses `json.loads(prefix+token)` (a streaming prefix is rarely complete JSON, so it reports 'invalid' on almost every token — undermining its stated 'on each token validates against the schema' behavior). These are separate repair implementations with different success semantics and no shared spec.
- **Evidence (verbatim):**
```
candidate = prefix + token  try: json.loads(candidate); return True except ...: return False
```
- **Impact:** Maintaining two repair stacks invites behavioral drift (e.g. one fixes trailing commas, the other strips them), and the dist validate_token cannot actually validate partial JSON, so any consumer relying on it will over-reject in-progress tokens.
- **Recommendation:** Consolidation story: make core OutputRepairer the canonical repairer and have RepairOrchestrator delegate to it; replace validate_token's json.loads guess with a partial-JSON validator (reuse the streaming engine's state machine or a tolerant incremental parser). Add a test that a valid in-progress JSON prefix does not fail validate_token.

---
