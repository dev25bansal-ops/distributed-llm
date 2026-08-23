# HI-30 File Decomposition Plan

**Date:** 2026-07-18
**Scope:** 10 monolithic files across 4 modules (API, CLI, Core, SDK/Prompts)

---

## 1. Summary

| # | File | Current Lines | Target Lines | Delta | New Files |
|---|------|--------------|-------------|-------|-----------|
| 1 | `api/server.py` | 1,727 | 230 | -1,497 | 8 |
| 2 | `cli/main.py` | 1,738 | 750 | -988 | 10 |
| 3 | `core/evaluation_harness.py` | 1,578 | 1,650* | +72 | 8 + `__init__.py` |
| 4 | (no standalone file — eval_harness.py not found) | — | — | — | — |
| 5 | `core/batch_scheduler.py` | 1,336 | 900 | -436 | 5 |
| 6 | `core/kv_cache.py` | 1,257 | 420 | -837 | 6 |
| 7 | `sdk/client.py` | 1,171 | 550 | -621 | 3 |
| 8 | `prompts/library.py` | 1,198 | 1,265* | +66 | 14 |
| 9 | `core/dp_inference.py` | 1,051 | 990* | -61 | 6 |
| 10 | `core/coordinator.py` | 1,698 | 350 | -1,348 | 6 |

> Lines marked * increase marginally due to required boilerplate (imports, `__init__.py` stubs).
> Not-found files (eval_harness.py, dist_speculative.py) are believed to have been renamed or integrated — no extraction possible without locating them.

---

## 2. Priority Order

1. **Phase 1 — Low-effort extractions (trivially safe)** — self-contained code blocks that can be moved without changing behavior.
2. **Phase 2 — Medium-effort extractions** — well-bounded blocks that depend on a few external references.
3. **Phase 3 — High-effort refactors** — deeply interleaved code that requires abstraction layers or dependency injection.

---

## 3. File-by-File Plan

### 3.1 `api/server.py` (1,727 -> ~230)

**Current state:** Monolithic FastAPI server with middleware, error handlers, state proxy, config loading,
lifespan management, coordinator creation, operational API endpoints, dashboard routes, and the `main()` entrypoint.

**Target:** Orchestrator that wires together sub-modules via `app.add_middleware()` and `app.include_router()`.

| New File | Content | Effort | Lines |
|----------|---------|--------|-------|
| `server_middleware.py` | SecurityHeaders, Timeout, RequestSizeLimit, Backpressure, PluginHook | Medium | ~200 |
| `server_errors.py` | Error response model, 3 exception handlers, `register_exception_handlers(app)` | Low | ~80 |
| `server_state.py` | State proxy | Low | ~60 |
| `server_config.py` | CORS config, API versioning constants, `get_cors_origins_lazy()`, `load_settings(args)` | Low | ~100 |
| `server_lifespan.py` | Lifespan async generator, observability init, plugin init, WS broadcaster | Medium | ~150 |
| `server_coordinator.py` | `create_coordinator` + `_CoordinatorNode` | Medium | ~120 |
| `server_routes_api.py` | 13 ad-hoc operational endpoints behind `register_api_routes(app)` | High | ~400 |
| `server_routes_dashboard.py` | Dashboard websockets, static pages, prometheus behind `register_dashboard_routes(app)` | High | ~380 |

**Dependencies:** `__init__.py` must re-export `app`, `main`, and `__version__`.
Test fixtures that import `server.app` must continue to work.

---

### 3.2 `cli/main.py` (1,738 -> ~750)

**Current state:** Typer CLI with 10 top-level command groups (76 command functions total).
58 commands already extracted to dedicated modules; 18 remain inline.

**Target:** Routing registry with 2-4 line stubs per command.

| New File | Content | Effort | Lines |
|----------|---------|--------|-------|
| `cli/completion.py` | `completion` command | **Low** | ~70 |
| `cli/config_commands.py` | `config validate`, `config reference`, `config openapi` | **Low/Med** | ~70 |
| `cli/defrag_commands.py` | `defrag status`, `defrag run`, `defrag stats` | Medium | ~120 |
| `cli/federate_commands.py` | `federate train`, `federate status` | High | ~90 |
| `cli/system_commands.py` | `system coordinator`, `system api`, `system schedule-viz`, `system slo-report` | Medium/High | ~250 |
| `cli/daas_commands.py` | `daas serve`, `daas status`, `daas benchmark` | High | ~150 |
| `cli/draft_commands.py` | `draft fleet-status`, `draft migration-status` | Medium | ~80 |

**Deprecations:** `_display_qr` (14 lines) stays in `main.py`.

---

### 3.3 `core/evaluation_harness.py` (1,578 -> ~1,650 across package)

**Current state:** 9 separable concerns in one file.

**Target:** `core/evaluation/` package with 8 modules + `__init__.py`.

| New File | Content | Effort | Lines |
|----------|---------|--------|-------|
| `evaluation/constants.py` | Constants, enums (EvalBenchmark, EvalStatus), `_SecretStr` | **Low** | ~60 |
| `evaluation/models.py` | Data models (EvalSample, EvalResult, EvalReport) | **Low** | ~70 |
| `evaluation/db.py` | EvalDB (SQLite persistence) | Medium | ~165 |
| `evaluation/loaders.py` | 5 dataset loaders + embedded fallback data | Medium | ~280 |
| `evaluation/formatters.py` | 3 prompt formatters (Heim, MT-Bench, Arena) | **Low** | ~65 |
| `evaluation/scorers.py` | 3 scorers (ExactMatch, MTBench LLM-judge, Arena) | Medium | ~255 |
| `evaluation/report.py` | ReportGenerator | **Low** | ~65 |
| `evaluation/worker.py` | WorkerPool + `_count_tokens` helper | **Low** | ~55 |
| `evaluation/runner.py` | EvalRunner + `run_all_heim` | High | ~480 |
| `evaluation/__init__.py` | Re-exports public API | Low | ~20 |

**Backward compatibility:** `evaluation_harness.py` becomes a thin re-export shim, or consumers update imports to `distllm.core.evaluation`.

---

### 3.4 (eval_harness.py) — File not found on disk

The analysis describes a 1,527-line file `eval_harness.py` containing RemoteDraftModel and
DistributedSpeculativeDecoder. No such file exists at `core/eval_harness.py`. The content may have
been merged into `core/distributed_speculative.py` or split during a previous refactor.

---

### 3.5 `core/batch_scheduler.py` (1,336 -> ~900)

**Current state:** Monolithic `BatchScheduler` class with embedded batch-building, starvation detection,
priority management, stats collection, and step processing. 8 methods already extracted into `scheduler/`.

**Target:** `BatchScheduler` as orchestrator calling sub-managers.

| New File | Content | Effort | Lines |
|----------|---------|--------|-------|
| `core/starvation_monitor.py` | `_check_starvation`, `_aging_boost`, `_priority_weight` | **Low** | ~50 |
| `core/priority_heap.py` | `promote_request`, `_pending_index` caching | **Low** | ~45 |
| `core/stats_collector.py` | `stats()` method | **Low** | ~40 |
| `core/batch_builder.py` | 8-method batch construction pipeline | Medium | ~200 |
| `core/step_processor.py` | `step()` + `_record_step_metrics` | Medium | ~85 |

**Retained in batch_scheduler.py:** `BatchScheduler.__init__`, public API (`add`, `schedule`, `step`
delegation), config/setup methods, advanced scheduling integration hooks, and thin wrappers.

---

### 3.6 `core/kv_cache.py` (1,257 -> ~420)

**Current state:** KVCache class + PagedKVCacheBackend + KVCacheManager + serialization + AdaptiveQuantizer.

**Target:** KVCache stays (~420 lines); all auxiliary classes/functions extracted.

| New File | Content | Effort | Lines |
|----------|---------|--------|-------|
| `core/kv_cache_paged.py` | PagedKVCacheBackend | **Low** | ~68 |
| `core/kv_cache_serialization.py` | Serialization functions (tensor_to_bytes, serialize/deserialize, save/load disk) | **Low** | ~154 |
| `core/kv_cache_adaptive_quantizer.py` | AdaptiveQuantizer | **Low** | ~150 |
| `core/kv_cache_manager.py` | KVCacheManager | Medium | ~166 |
| `core/kv_cache_migration.py` | CPU/GPU swap helper mixin | Medium | ~100 |

**Backward compatibility:** `__init__.py` and `kv_cache.py` re-exports. All external importers
(`dist/wide_area.py`, `dist/node_service.py`, backends, `cache_manager.py`, `kv_backup.py`)
must continue to work via re-exports.

---

### 3.7 `sdk/client.py` (1,171 -> ~550)

**Current state:** HTTP client with heavy sync/async transport duplication (~200 lines verbatim
`_request`/`_request_raw`), domain twin methods (~280 lines `_*_async`/`_*_sync` pairs),
observability/stats, and domain wrappers.

**Target:** Streamlined client with shared transport.

| New File | Content | Effort | Lines |
|----------|---------|--------|-------|
| `sdk/transport.py` | HTTP transport layer: `_request`, `_request_raw`, retry + circuit-breaker, streaming helpers | Medium | ~120 |
| `sdk/observability.py` | `_record_call`, `_parse_usage` | Low | ~60 |
| `sdk/streaming.py` | Streaming response handling | Low | ~60 |

**Retained in client.py:** Base client class, thin client classes (SyncClient, AsyncClient),
auth, and domain wrapper methods (chat, embeddings, batch, audio, images, files, fine-tuning).

---

### 3.8 `prompts/library.py` (1,198 -> ~1,265 across package)

**Current state:** Monolithic prompt library with `SystemPromptDef` dataclass + registry + 11 per-category
sections in a single 1,052-line file.

**Target:** `prompts/` package with 14 focused files.

| New File | Content | Effort | Lines |
|----------|---------|--------|-------|
| `prompts/prompt_def.py` | SystemPromptDef dataclass + registry + `_reg` helper | Low | ~40 |
| `prompts/management.py` | Query functions (get_prompt, list_categories, list_by_category, search_prompts) | Low | ~40 |
| `prompts/code.py` | Code-related prompts (one section) | Low | ~90 |
| `prompts/writing.py` | Writing-related prompts | Low | ~90 |
| `prompts/analysis.py` | Analysis prompts | Low | ~90 |
| `prompts/language.py` | Language prompts | Low | ~90 |
| `prompts/education.py` | Education prompts | Low | ~90 |
| `prompts/professional.py` | Professional prompts | Low | ~90 |
| `prompts/reasoning.py` | Reasoning prompts | Low | ~90 |
| `prompts/specialized.py` | Specialized prompts | Low | ~90 |
| `prompts/creative.py` | Creative prompts | Low | ~90 |
| `prompts/productivity.py` | Productivity prompts | Low | ~90 |
| `prompts/__init__.py` | Re-export all public names | Low | ~20 |

**Import migration:** 3 downstream consumers need path updates: `cli/prompts.py`,
`api/routes/prompts.py`, `tests/prompts/test_library.py`.

---

### 3.9 `core/dp_inference.py` (1,051 -> ~990 across package)

**Current state:** DP inference with embedded config, RDP accounting, noise mechanisms, budget manager,
and the main `DifferentialPrivacyInference` class.

**Target:** `core/dp_inference/` package with 6 files.

| New File | Content | Effort | Lines |
|----------|---------|--------|-------|
| `dp_inference/config.py` | DPConfig, BudgetEntry, DPGenerationResult dataclasses | **Low** | ~60 |
| `dp_inference/accounting.py` | RDPAccounting | Medium | ~110 |
| `dp_inference/mechanisms.py` | clip_gradients, dp_noise_injection, gumbel_noise_mechanism | **Low** | ~110 |
| `dp_inference/budget.py` | PrivacyBudgetManager | Medium | ~190 |
| `dp_inference/engine.py` | DifferentialPrivacyInference + wrap_with_dp | High | ~400 |
| `dp_inference/__init__.py` | Re-export public API | Low | ~15 |

**Caveat:** The generation-path privacy caveat (warning about unapplied noise) must be preserved.

---

### 3.10 `core/coordinator.py` (1,698 -> ~350)

**Current state:** Composees InferenceEngine, ClusterManager, HealthManager, MetricsCollector via delegation.
5 major remaining code blocks need extraction.

**Target:** Facade (~350 lines) with property delegations and thin public-API wrappers.

| New File | Content | Effort | Lines |
|----------|---------|--------|-------|
| `core/coordinator_election.py` | HA election + state replication (lines 432-597) | Medium | ~166 |
| `core/coordinator_lifecycle.py` | Node lifecycle callbacks: drain, mark_dead, redistribute, recover (lines 263-390, 601-607) | Medium | ~135 |
| `core/coordinator_config_wiring.py` | Config wiring, init methods, hot-swap, adaptive compression, defrag, graceful degradation (lines 82-260, 392-1092) | High | ~540 |
| `core/coordinator_request.py` | Request generation: sync/async, wait_for_result (lines 689-743, 1128-1259) | Medium | ~185 |
| `core/coordinator_subsystem.py` | Subsystem lifecycle: register, start, stop, save_state (lines 274-290, 1263-1690) | High | ~430 |

**Existing supporting files (not extractions):** `coordinator_health.py`, `coordinator_failover.py`,
`coordinator_lifecycle.py`, `coordinator_metrics.py`, `coordinator_state.py`, `protocols.py`.

---

## 4. Phase 1 — Quick Wins (Low Effort)

These are pure code moves with no dependency injection, no interface changes, and no behavioral change.

### Completed in this pass:

| Source File | Target File | Lines Moved | Section |
|-------------|-------------|-------------|---------|
| `core/kv_cache.py` | `core/kv_cache_paged.py` | 68 | PagedKVCacheBackend class |
| `core/kv_cache.py` | `core/kv_cache_serialization.py` | ~154 | `serialize_kv_cache`, `deserialize_kv_cache`, `_tensor_to_bytes`, `_bytes_to_tensor`, `save_kv_cache_to_disk`, `load_kv_cache_from_disk`, `serialize_kv_cache_async` |
| `core/kv_cache.py` | `core/kv_cache_adaptive_quantizer.py` | ~150 | `AdaptiveQuantizer` class |
| `core/dp_inference.py` | `core/dp_inference/config.py` | ~60 | `DPConfig`, `BudgetEntry`, `DPGenerationResult` dataclasses |
| `core/dp_inference.py` | `core/dp_inference/mechanisms.py` | ~110 | `clip_gradients`, `dp_noise_injection`, `gumbel_noise_mechanism` |
| `core/evaluation_harness.py` | `core/evaluation/models.py` | ~130 | Constants, enums, data models |
| `core/evaluation_harness.py` | `core/evaluation/constants.py` | ~45 | Constants (separated from models) |
| `core/evaluation_harness.py` | `core/evaluation/formatters.py` | ~65 | PromptFormatter, Heim/MTBench/Arena formatters |
| `core/evaluation_harness.py` | `core/evaluation/report.py` | ~65 | ReportGenerator |
| `core/evaluation_harness.py` | `core/evaluation/worker.py` | ~55 | WorkerPool + `_count_tokens` |
| `core/batch_scheduler.py` | `core/starvation_monitor.py` | ~50 | `_check_starvation`, `_aging_boost`, `_priority_weight` |
| `core/batch_scheduler.py` | `core/priority_heap.py` | ~45 | `promote_request`, `_pending_index` |
| `core/batch_scheduler.py` | `core/stats_collector.py` | ~40 | `stats()` method |
| `cli/main.py` | `cli/completion.py` | ~70 | `completion_command` |
| `cli/main.py` | `cli/config_commands.py` | ~70 | Config validate, reference, openapi |

---

## 5. Phase 2 — Medium Effort

| Source File | Target File | Key Challenge |
|-------------|-------------|---------------|
| `core/evaluation_harness.py` | `core/evaluation/db.py` | Threading, lock management |
| `core/evaluation_harness.py` | `core/evaluation/loaders.py` | 5 loaders with embedded fallback |
| `core/evaluation_harness.py` | `core/evaluation/scorers.py` | LLM-as-judge API integration |
| `core/kv_cache.py` | `core/kv_cache_manager.py` | Import registration in `__init__.py` |
| `core/kv_cache.py` | `core/kv_cache_migration.py` | Internal state access (`self._lock`, `self.cache`) |
| `core/batch_scheduler.py` | `core/batch_builder.py` | 8 methods forming closed call graph |
| `core/batch_scheduler.py` | `core/step_processor.py` | Multiple dependency references |
| `core/dp_inference.py` | `core/dp_inference/accounting.py` | Renyi accountant (self-contained) |
| `core/dp_inference.py` | `core/dp_inference/budget.py` | Thread-safe per-tenant tracking |
| `core/coordinator.py` | `coordinator_election.py` | HA election/replication |
| `core/coordinator.py` | `coordinator_lifecycle.py` | Node lifecycle callbacks |
| `core/coordinator.py` | `coordinator_request.py` | Request generation |
| `cli/main.py` | `cli/defrag_commands.py` | ~120 lines of inline defrag commands |
| `cli/main.py` | `cli/draft_commands.py` | ~80 lines of inline draft commands |

---

## 6. Phase 3 — Deep Refactors (High Effort)

| Source File | Target File | Key Challenge |
|-------------|-------------|---------------|
| `api/server.py` | `server_routes_api.py`, `server_routes_dashboard.py` | 13 ad-hoc endpoints, dashboard routes |
| `api/server.py` | `server_middleware.py` | 5 middleware classes with various deps |
| `cli/main.py` | `cli/federate_commands.py`, `cli/daas_commands.py`, `cli/system_commands.py` | Commands with more complex logic |
| `core/evaluation_harness.py` | `core/evaluation/runner.py` | Orchestrates all other components |
| `core/dp_inference.py` | `core/dp_inference/engine.py` | Main class depends on all others |
| `core/coordinator.py` | `coordinator_config_wiring.py`, `coordinator_subsystem.py` | Deeply interleaved with coordinator internals |
| `sdk/client.py` | `sdk/transport.py` | Transport abstraction layer to eliminate sync/async duplication |
| `prompts/library.py` | 11 per-category files | Mechanical but requires careful section delimiting |
| `server.py` | `server_lifespan.py`, `server_coordinator.py` | Lifespan + coordinator creation |

---

## 7. Verification Strategy

### 7.1 After Each Extraction

```bash
# 1. Syntax check
python -c "import ast; ast.parse(open('new_file.py').read())"

# 2. Import check (for modules within the project)
python -c "from distllm.core.kv_cache_paged import PagedKVCacheBackend"

# 3. Verify original imports still work (re-exports preserved)
python -c "from distllm.core.kv_cache import KVCache, PagedKVCacheBackend, serialize_kv_cache"
```

### 7.2 Full Verification

```bash
# 1. Run module-level tests
python -m pytest tests/core/ -x -q --timeout=60

# 2. Verify no broken imports
python -c "
from distllm.core.kv_cache import KVCache, PagedKVCacheBackend, serialize_kv_cache
from distllm.core import kv_cache
from distllm.core.dp_inference import DifferentialPrivacyInference, DPConfig
"

# 3. Git diff review
git diff --stat
```

### 7.3 Post-Extraction Checks

- `git diff` shows moved code, no behavioral changes
- `__all__` exports in all new files match the public API
- Original `__all__` exports still resolve through re-exports
- No circular imports introduced
- All importers of original module continue to work
- Test suite passes (100% of previously passing tests still pass)

### 7.4 Rollback Plan

- Each extraction is a single commit with a clear message
- If a test breaks, revert the commit, diagnose, and retry
- No cross-file dependency changes in the same commit
- Re-exports ensure backward compatibility even if extraction is incomplete
