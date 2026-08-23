# CLAUDE.md — Test Suite

## Your Scope
You have ownership of `tests/` — all test files across 30+ directories, including unit tests, integration tests, e2e tests, chaos tests, fuzz tests, property tests, load tests, and security tests.

## Do NOT Touch
Any source files in `src/` — focus entirely on test quality and coverage.

## Key Directories

| Directory | Purpose | File Count |
|-----------|---------|------------|
| `tests/core/` | Core engine tests | 100+ |
| `tests/api/` | API route tests | 30 |
| `tests/dist/` | Distributed layer tests | 17 |
| `tests/e2e/` | End-to-end tests | 20+ |
| `tests/integration/` | Integration tests | 20+ |
| `tests/security/` | Security tests | 6+ |
| `tests/chaos/` | Chaos engineering | 8 |
| `tests/fuzz/` | Fuzz testing | 7 |
| `tests/cli/` | CLI tests | 5 |
| `tests/distributed/` | Real multi-GPU tests (NEW) | 1 |
| `tests/benchmark/` | Performance benchmarks | 4 |
| `tests/correctness/` | Output correctness | 8 |
| `tests/partition/` | Partitioner tests | 8 |
| `tests/load/` | Load testing | 8 |
| `tests/property/` | Property-based testing | 8 |

## Key Test Files Added Recently

| File | What It Covers |
|------|----------------|
| `tests/dist/pipeline/test_1f1b_scheduling.py` | 1F1B pipeline scheduling |
| `tests/distributed/test_real_multi_gpu.py` | Real multi-GPU inference |
| `tests/security/test_jwt_auth.py` | JWT auth end-to-end |
| `tests/security/test_oauth_state_csrf.py` | OAuth CSRF protection |
| `tests/core/test_cert_rotation.py` | Certificate expiry |
| `tests/core/test_node_recovery.py` | Checkpoint replay |
| `tests/core/test_kv_cache_fp8.py` | FP8 quantization |
| `tests/core/test_coordinator_state_replication.py` | HA replication |
| `tests/core/test_cost_tracker_all.py` | Cost tracking (pytest-style) |
| `tests/core/test_gbnf.py` | GBNF grammar (pytest-style) |
| `tests/api/test_model_load_rbac.py` | Model RBAC |
| `tests/api/test_blocking_warmup.py` | Non-blocking warmup |
| `tests/dist/test_federation_heartbeat.py` | Federation HMAC |
| `tests/fuzz/fuzz_auth_bypass.py` | Auth bypass fuzzing |

## Commands
- `python -m pytest tests/core/ -v` — core tests
- `python -m pytest tests/api/ -v` — API tests
- `python -m pytest tests/dist/ -v` — distributed tests
- `python -m pytest tests/security/ -v` — security tests
- `python -m pytest tests/fuzz/ -v` — fuzz tests
- `pytest -v --cov=distllm --cov-report=xml --cov-fail-under=80` — full coverage
