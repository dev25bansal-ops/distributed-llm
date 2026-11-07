---
tags:
  - tests
  - quality
  - verification
aliases:
  - Tests
  - Test Suite
---
# Test Suite — `tests/`

**788 .py files · 152,219 LOC · 51 packages.**

> The full verification surface, exercising every subtree of `src/distllm`. Heavily **stubbed** so suites run on CPU without real GPUs (`_import_helper.py`, `helpers.py`, per-package `_stubs.py`/`stubs.py`). `regression_high/` holds 70 letter-indexed regression gates (`a1–a8, c2–c16, e1–e13, m2–m17, n1–n10, h3–h14, pbt-*`) mapping to verified senior-audit findings.
>
> **Run:** `pytest` (config: `pytest.ini`, coverage ≥ 80%). **CI:** [[09 Infrastructure]] gates.

## Per-package map

| Package | files | LOC | covers |
|---------|------:|-----:|--------|
| (root) | 48 | 8,712 | cross-suite feature/integration smoke, high/medium-severity fixes |
| `core` | 299 | 65,483 | engine subsystems: scheduler, coordinator, kv_cache, spec, routers, autoscaler, multimodal, structured output |
| `dist` | 158 | 57,925 | distributed layer: pipeline, p2p, partition/quant, topology, federation, transport |
| `api` | 65 | 17,199 | HTTP+WS API, auth, middleware, rate-limit, batch |
| `regression_high` | 70 | 9,477 | letter-indexed audit gates (a/h/c/e/m/n/pbt) |
| `integration` | 22 | 3,894 | distributed pipeline, spec e2e, gRPC reconnect, LoRA, TLS |
| `security` | 34 | 5,748 | JWT/SSRF/CSRF/poisoning/moderation/vulnerabilities |
| `cli` | 16 | 4,371 | CLI commands + modules |
| `comprehensive` | 13 | 1,916 | broad sweeps (kv-cache, serialization, NAT, rate-limit) |
| `e2e` | 22 | 3,694 | full flows: disaster recovery, cross-machine, graceful shutdown |
| `chaos` | 16 | 2,802 | fault injection, node failure, split-brain |
| `correctness` | 10 | 1,590 | determinism, spec-vs-greedy parity, quant quality |
| `benchmark` | 15 | 1,679 | perf + regression gates |
| `property` | 8 | 1,405 | property-based invariants |
| `introspection` (`verification`) | 3 | 1,163 | project-wide walk |
| `fuzz` | 14 | 3,554 | API/auth/CLI/config/grammar/grpc/proto fuzzers |
| `load` (+`locust`) | 17 | 1,857+ | Locust load, SLO, draft contention |
| `partition` | 13 | 1,644 | partitioner, optimizer, quant, topology |
| **… other suites** | — | — | backends, client, cloud, dashboard, deploy, errors, evaluation, models, observability, plugins, prompts, sdk, ui, utils, stability/soak, stress, mutation, infra … |

<!-- recounted 2026-11-06 by hygiene pass -->

> Full per-file table (all 788) is maintained in the source report at `.claude/obsidian-grasp/reports/tests.md`; the per-package table above is the summary index.

## Key harnesses
- `tests/fixtures/draft_model_server.py` — shared draft-model HTTP server fixture.
- `tests/_import_helper.py`, `tests/helpers.py` — import-bootstrap + generic helpers.
- `_stubs.py` doubles in `api`, `cli`, `core`, `integration`, `verification`.
- `tests/regression_high/` — themed regressions (a1 spec-invariant … h14 retry-storm, e12 metered billing, m13/14).

## Suites to run by area
| Area | Command |
|------|---------|
| Core | `python -m pytest tests/core -v` |
| Distributed | `python -m pytest tests/dist -v` |
| API | `python -m pytest tests/api -v` |
| Security | `python -m pytest tests/security tests/security_pkg -v` |
| CLI/config | `python -m pytest tests/cli tests/config -v` |
| Full + quality gates | `make test` / `make cov` / `make check` (bandit, coverage, flaky, mutation gates) |