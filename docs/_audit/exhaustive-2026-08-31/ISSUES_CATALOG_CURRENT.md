# DistLLM — Issues & Required Fixes Catalog (Definitive)
**Version**: 2026-09-08 (post-Sprint-1 security closure + O-10/O-13 quick wins)
**Method**: Every item verified against the live working tree on this date. Fixed items appear in §5 as a traceability ledger; only genuinely open work is in §1–§4. Candidate-count sources: full core suite (6,383 passing), git-stash round-trip baseline.

Legend — Severity: business impact × risk. Priority: P0 = block release, P1 = within sprint, P2 = within 30 days, P3 = next release. Effort: single-engineer hours/days.

---

## §1. OPEN — Software Bugs

| ID | Description + Example | Severity | Priority | Effort | Depends on | Timeline |
|---|---|---|---|---|---|---|
| **BUG-FN** | **`test_split_brain`, `test_quick_audit_fixes`, `test_pipeline_orchestrator`, `test_framework_adapters`, `test_audio_multiple_utterances`, `test_prefix_cache*` (≈18) assert stale APIs from earlier refactors.** Repro: `pytest tests/core/test_*.py` → AssertionError/ImportError on contracts that moved (e.g. quorum helper). Expected: green; actual: red on master (verified). The suite is the product's regression radar; a permanently-red baseline buries new failures. **2026-09-08: dist/ red set measured exactly — 38 failures across 10 files (13 cluster-key, 12 federation-get-load, rest transport/wide-area/observability), all verified pre-existing via git-stash baseline.** | Medium | P2 | 1–2d | contract review session | 2 weeks |

## §2. OPEN — Performance Bottlenecks

| ID | Description + Measurable Impact | Severity | Priority | Effort | Depends on | Timeline |
|---|---|---|---|---|---|---|
| **O-08** | ~~**SSE emission duplicates + per-token deep-copy**~~ **CLOSED 2026-09-08** — `to_sse()` builds the payload dict directly (no `asdict` deep-copy per token); dead `format_sse_response`/`format_done` duplicates removed (zero callers, verified). Tests: `test_streaming_generator.py` rewritten static-method class pins wire-format identity (4). | Medium | P2 | ✅ done | none | closed |
| **O-09** | **Multimodal = zero capability** (C-01 made it fail-loud). Any image/audio/document request raises `NotImplementedError`. Product gap: no vision/audio/doc encoders wired to the pipeline. Impact: the advertised multimodal endpoint does nothing. | Medium (product, not a defect) | P2 | 2–3 wk | encoder backends + embedding injection seam | roadmap (v1.0) |

## §3. OPEN — Security Vulnerabilities

| ID | Description + CVSS | Severity | Priority | Effort | Depends on | Timeline |
|---|---|---|---|---|---|---|
| **O-01** | **DHT fail-closed env var documented but NOT enforced.** `kademlia_dht.py:84` defines `DISTLLM_DHT_REQUIRE_SECRET` with an operator-policy comment block; `grep` across `src/` shows zero consumers. Operator sets the kill-switch → no behavior change; DHT STORE stays open per the old default. CVSS 6.5 (AV:A/AC:L/PR:N/UI:N/S:U/C:N/I:H/A:N). **DOCUMENTED BUT NOT ENFORCED.** | High | P1 | 0.5d | gossip already has `authorize_fetch_request` to mirror | This week |
| **O-02** | **`trust_remote_code` default unresolved → potential RCE from model download.** `inference_engine.py:99,114` stores `trust_remote_code: bool\|None = None` verbatim; no `DISTLLM_TRUST_REMOTE_CODE=1` defaulting. Forwarded `None` to `AutoConfig.from_pretrained` lets HF execute bundled repo code. CVSS 7.5 (AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:N) when paired with an unpinned HF id. Pin test exists (`test_inference_engine_trust_remote_code_default.py`) but core defaulting never landed. | High | P1 | 2h | none | This week |
| **O-04** | **Node-to-node gRPC still plaintext by default.** `node_service.py:491-561`: `DISTLLM_TLS_ENABLED=1` works; unset → explicit log "no TLS", but the default posture is plaintext. On-prem (the core market) on shared LAN/VPC → tensor/prompt MITM. CVSS 5.9 (AV:A/AC:H/PR:N/UI:N/S:U/C:H/I:H/A:N). **2026-09-08 PARTIAL: TLS-requested-but-missing-certs silent downgrade closed (fail-closed ConfigError); full default-flip remains next release.** | High | P1 | 0.5d remaining (default-flip + cert bootstrap + 2 C5 signature tests) | O-01-style env plumbing exists | Next release |
| **O-11** | ~~**`TreeNode.logprob` dual semantics**~~ **CLOSED 2026-09-08** — greedy mode now stores `log_softmax` values (was raw logit); matches `tree_speculative_decoder.py` semantics. Tests: `test_draft_tree.py::TestLogprobSemanticsO11` (3). | Low | P3 | ✅ done | none | closed |

## §4. OPEN — Code Quality / Architecture / Tech Debt

| ID | Description + Example / Impact | Severity | Priority | Effort | Depends on | Timeline |
|---|---|---|---|---|---|---|
| **O-06** | **PipelineOrchestrator god class (~3,400 LOC)** — owns 1F1B scheduling, transport dispatch, node health, checkpoint bookkeeping. Extensibility bottleneck; M-06 proved the registry pattern works → apply same seam extraction. | Medium | P2 | 1 wk | proven by M-05 strategy seam | next month |
| **O-07** | **server.py god assembly (~2,160 LOC)** — 20-middleware registration + 30 route includes at module scope with import-time side effects (O-13 import-time key print now closed; remaining side effects are config/CORS reads). TestCollection reuse is fragile. Enh #4 assertion suite makes reorder safe during refactor. | Medium | P2 | 3–4d | Enh #4 | next month |
| **O-05** | **26 pre-existing red tests on master** (25 confirmed + 1 flake): 8 = missing optional deps (`pgvector`/`chromadb` — should be `pytest.importorskip`); ~18 = stale contracts. A permanently red suite erodes CI trust. | Medium | P2 | 1–2d | none | 2 weeks |
| **O-12** | **Legacy audit/fleet-plan docs assert completed work as future** (`FLEET-PLAN*.md`, `WAVE2-PLAN.md`, 40 root-level scratch files). Confuses contributors and the public-launch footprint. | Low | P3 | 3h | none | before public launch |
| **O-14** | **Two dependency-management systems** — Python 3.10 retained; `requirements-dev.txt` coexists with PEP 735 `[dependency-groups]`. | Low | P3 | 2h | none | next release |

---

## §5. FIXED — Traceability Ledger

| ID | Defect | Remedy | Regression lock |
|---|---|---|---|
| O-01 | DHT `DISTLLM_DHT_REQUIRE_SECRET` documented but never read — kill-switch was a no-op | `KademliaDHT.__init__` raises `ConfigError` when env=1 and no secret; policy comment updated to match code | `tests/dist/test_dht_require_secret_enforced.py` (9) |
| O-02 | `trust_remote_code=None` forwarded to HF loaders → potential RCE via repo code | Engine resolves None → `DISTLLM_TRUST_REMOTE_CODE=1` env check → else `False` (fail closed) | `tests/api/test_inference_engine_trust_remote_code_default.py` (5) now pass for real |
| O-03 | `VectorClock.merge`/`happens_before` TypeError on None/negative timestamps from hostile peer | Skip non-int/negative in merge; coerce to 0 in happens_before | `tests/dist/test_vectorclock_none_guard.py` (9) |
| O-04a | gRPC `use_tls=True` + missing certs silently downgraded to plaintext | `start()` raises `ConfigError` instead of binding insecure port | `tests/dist/test_node_service_tls.py::TestTLSMissingCertsFailsClosed` (2) |
| O-10 | Quarantine had no expiry policy — rot risk | 2026-10-01 review-by date documented in `_quarantine/core/__init__.py`; unmoved modules deleted wholesale at sweep | commit 126309b |
| O-13 | Import-time API key print (server.py module scope) + the printed key was a DECOY (discarded; ApiKeyStore generates its own) | Lazy `ensure_api_key_resolved()` at lifespan startup; real auto-generated admin key printed once, only when it exists | 154 auth/key tests green; API suite delta vs stash baseline = 0 (5 pre-existing order-pollution failures both ways) |
| C-01 | Multimodal silent-vision stub | `NotImplementedError` fail-loud | `test_multimodal_fail_loud.py` + rewritten `test_multimodal_engine.py` (25) |
| C-02 | `verify_chain` bounds-break asserts unverified tokens | `return i` on bounds exhaustion; batched kernel | `test_spec_verify_bounds.py` (5) |
| C-03 | `rand ≥ p` instead of `min(1, p/q)` (4 sites) | All acceptance routed through `spec_verify.accept_token` | `test_rejection_sampling.py` + AST lint (3) |
| C-04 | Tuple-cache crop desync | `_crop_tuple_cache` dim-2 slice fallback | `test_kv_cache_crop_fallback.py` |
| C-05 | Streaming backpressure infinite hang | bounded `backpressure_timeout_s` flush | `test_streaming_backpressure.py` |
| C-06 | `app.add_event_handler` removed in Starlette≥0.40 | FastAPI lifespan | `test_lifespan_no_add_event_handler.py` |
| C-07 | Coordinator circular import | tail-import removed | import guard |
| H-01 | KV cache serialized 1,640× per request | hoisted out of node loop; real counts | `test_htier_mtier_fixes.py` (17) |
| H-02 | 648 GPU↔CPU syncs/round tree verify | batched argmax precompute (81× less) | H-tier file |
| H-03 | O(n²) n-gram slice scan (~2M allocs/gen) | incremental suffix dict index | 6 n-gram tests |
| H-04 | P99 = max of 100-window | nearest-rank on 500-window | 4 percentile tests |
| H-05 | All speculative requests shared id "spec" | per-generation unique ids | source-guard |
| H-06 | Buffer race dropped tokens | drain inside critical section | race test |
| H-07 | cost_middleware 500 on non-dict bodies | 400 short-circuit | `test_cost_middleware_invalid_body.py` (12) |
| H-08 | `bitsandbytes_4bit` vs `bnb_4bit` crash | field_validator normalizes | `test_quantization_method_validator.py` (13) |
| M-01 | Word-count billed as tokens | `prompt_tokens_estimated` flag | H-tier |
| M-02 | Kernel invariant unenforced | AST lint; 7 decoders import kernel | `tests/lint/test_spec_verify_invariant.py` |
| M-03 | Draft tree `_seed` never set | init sets seed | H-tier |
| M-04 | throughput metric ≈ 0 forever | monotonic per-model windows | 2 tests |
| M-05 | InferenceEngine god class (1,405 LOC) | 5 strategy modules + `EngineContext` seam (773 LOC) | 76 engine/strategy tests |
| M-06 | Layer topology duplicated (core+dist) | `TopologyRegistry` shared | `test_topology_registry.py` (20) |
| W5-A | DocsAuth ran OUTSIDE Auth (docs auth always failed — 12 red tests) | registration moved before Auth | `test_middleware_order_assertion.py` (7) |
| W5-B | CSRF ran INSIDE Auth (auth work before CSRF block) | registration moved after Auth | same |
| W5-C | MagicMock `_param_update_channel` crash in streaming | `isinstance(puc, ParamUpdateChannel)` guard | streaming tests |
| W5-D | Dead code: 21 modules / ~5,100 LOC | 11 deleted + 10 quarantined | package-import guard |

---

## §6. Rolls-Up Summary

2026-09-08 status after Sprint-1 + quick-wins: **all 4 High security items addressed** — O-01/O-02/O-03 fully closed with regression locks, O-04 half-closed (fail-closed cert handling landed; default-flip next release). O-10 (quarantine expiry) and O-13 (import-time key print + decoy-key bug) also closed. Full API suite delta verified = 0 via stash baseline (5 pre-existing order-pollution failures both ways); full `tests/dist/` verified = 0 via stash baseline (38 pre-existing failures both ways).

| Severity | Open count | Effort |
|---|---|---|
| Critical | 0 | — |
| High | 1 (O-04 default-flip, scoped next release) | ~0.5 day |
| Medium | 5 (O-05, O-06, O-07, O-08, O-09) | ~2–6 wks (O-09 dominates) |
| Low | 2 (O-11, O-14) | ~3 days |

**Sprint-1 (done 2026-09-08):**
1. ✅ O-03 (1h) — gossip None-guard
2. ✅ O-02 (2h) — `trust_remote_code` env defaulting, fail-closed
3. ✅ O-01 (4h) — enforce `DISTLLM_DHT_REQUIRE_SECRET` (ConfigError at construction)
4. ◐ O-04 — missing-certs downgrade closed; full default-flip with cert bootstrap next release
5. ✅ O-10 (1h) — quarantine 2026-10-01 expiry policy
6. ✅ O-13 (1h) — import-time key print removed; decoy-key bug fixed (real key now surfaced once at startup)

That closes every "documented but not enforced" and silent-crash security gap in the tree before the next public milestone.