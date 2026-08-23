---
tags:
  - tests
---
# Test Suite

**Location:** `tests/` — **~5 MB, ~300 files across 30+ directories**

## Test Directories
| Directory | Files | Type |
|-----------|-------|------|
| `core/` | 100+ | Unit |
| `api/` | 30 | Unit/Integration |
| `dist/` | 17 | Unit/Integration |
| `e2e/` | 20+ | E2E |
| `integration/` | 20+ | Integration |
| `security/` | 6+ | Security |
| `fuzz/` | 7 | Fuzz |
| `load/` | 8 | Load/Locust |

## Key New Tests
| File | What It Covers |
|------|----------------|
| `tests/dist/pipeline/test_1f1b_scheduling.py` | 1F1B scheduling + bubble ratio (18 test cases) |
| `tests/distributed/test_real_multi_gpu.py` | Real multi-GPU inference |
| `tests/security/test_jwt_auth.py` | JWT HS256 end-to-end |
| `tests/security/test_oauth_state_csrf.py` | OAuth CSRF protection |
| `tests/core/test_kv_cache_fp8.py` | FP8 per-step quantization |
| `tests/core/test_coordinator_state_replication.py` | HA replication |
| `tests/core/test_cost_tracker_all.py` | Cost tracking (pytest-style) |
| `tests/core/test_gbnf.py` | GBNF grammar (pytest-style) |
| `tests/fuzz/fuzz_auth_bypass.py` | Auth bypass fuzzing (81 cases) |
