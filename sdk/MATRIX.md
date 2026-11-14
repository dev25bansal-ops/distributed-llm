# SDK Integration Matrix

Status of every DistLLM SDK: how it installs, what was actually executed on this
machine (2026-11-09), and where each stands for publishing.

Smoke legend: **live-smoke** = an automated test spins an in-process mock
OpenAI-compatible HTTP server (`/v1/models`, `/v1/chat/completions`, ...) and
asserts the SDK parses the canned responses into typed values. No external
network required.

Read this matrix for what the code *does*, not what it aspires to. Coverage is
per-language and deliberately uneven: the Python client is the reference
surface; JS/Go/Rust implement a small common core.

**Test coverage is narrower than "the suite is green" suggests.** Every executed
test passes, but only five HTTP routes are ever exercised by any client:
`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/models`,
`/health`. Python's batch / moderations / audio / image / file / fine-tuning
methods are implemented but **no test calls them**. See the coverage table
below for the per-method truth.

| SDK | Language | Install method | Executed version | Toolchain | Tests actually run | Publish status |
|-----|----------|----------------|------------------|-----------|--------------------|----------------|
| `distllm-sdk` (Python, async + sync) | Python >= 3.10 | PyPI `pip install distllm-sdk` · source `pip install -e sdk/` | 1.0.0 | Python 3.14.4, httpx 0.28.1, pytest 9.0.2 | **149 passed, 0 failed** — `PYTHONPATH=src:sdk/src python -m pytest sdk/tests/ -q` (10 hermetic smoke in `test_smoke_live.py` via httpx `MockTransport`; 139 other tests: 55 `test_sdk`, 38 `test_openai_compat`, 23 `test_client_extended`, 15 `test_grpc_client`, 8 `test_completions_stream_parity`) | Ready. Local `sdk/dist/distllm_sdk-0.5.0.tar.gz` is **stale** (0.5.0 vs pyproject 1.0.0) — rebuild before next PyPI upload |
| `distllm-sdk` (JS/TS) | Node >= 18 | npm `npm i distllm-sdk` (**not yet published**) · git subdir `sdk/js` | 1.0.0 | Node 24.12.0, vitest, tsc (ranges `^1.0.0` / `^5.0.0`, no lockfile) | **11 passed, 0 failed** — `cd sdk/js && npm install && npm run build && npm test` (`smoke.test.ts` 6 + `stream-timeout.test.ts` 5); `tsc --noEmit` clean | **Ready to publish** — `README.md` present, `exports`/`repository`/`bugs`/`homepage` restored; `npm pack --dry-run` = 8 files, 5.8 kB (6 dist + package.json + README). Remaining: actual `npm publish` |
| `distllm-sdk` (Go) | Go >= 1.21 | `go get github.com/distributed-llm/distllm-sdk-go` (**needs git tag**) | 1.0.0-dev (go.mod, untagged) | Go 1.26.2 windows/amd64 | **5 passed, 0 failed** — `cd sdk/go && go test ./...` (`TestSmokeListModels`, `TestSmokeChatCompletion`, `TestSmokeChatCompletionRequestBody`, `TestSmokeHealth`, `TestSmokeChatCompletionStream`; `httptest` server). `go vet ./...` clean; `gofmt` clean. **Caveat:** `Embedding` and `Completion` are implemented but never called by a test | Module path correct in `go.mod`; package docs in `doc.go`. Publish = push tag `v1.0.0` |
| `distllm-sdk` (Rust) | Rust 2021 edition | crates.io `cargo add distllm-sdk` (**not yet published**) · git subdir `sdk/rust` | — **not tested** | ⚠ cargo/rustc **not installed** on this machine | ✗ review-only — the crate contains 12 `#[test]` cases, but they are serde/serialization unit tests only (no HTTP, no server) and **have never been executed** | **Blocked** until `cargo check` / `cargo test` can run. `Cargo.toml` + `lib.rs` reviewed by hand. `src/generated/types.rs` is **not wired into the crate** (no `mod generated;`) — uncompiled, so review-only |

## How to run the suites

```bash
# Python (149 tests, hermetic)
PYTHONPATH=src:sdk/src python -m pytest sdk/tests/ -q

# JS (11 tests, hermetic; needs one npm install)
cd sdk/js && npm install && npm run build && npm test

# Go (5 tests, hermetic)
cd sdk/go && go test ./...
```

## Per-language coverage — what each client actually wraps

The Python client is the reference surface (20 public async methods). The other
clients implement a **common core only**; this is a capability statement, not a
bug list. What is absent, is absent.

Legend: **✓ = implemented AND exercised over HTTP by an automated test**;
**~ = implemented, but only covered by type/unit tests (no request is made)**;
**✗ = not implemented**.

| Capability | Python | JS | Go | Rust |
|---|---|---|---|---|
| `chat_completions` | ✓ | ✓ | ✓ | ~ |
| `chat_completions_stream` (yields `str` deltas) | ✓ | ✓ | ✓ | ✗ |
| `completions` | ✓ | ✓ | ~ | ~ |
| `completions_stream` (yields `str` deltas) | ✓ | ✗ | ✗ | ✗ |
| `list_models` | ✓ | ✓ | ✓ | ~ |
| `health_check` | ✓ | ✗ (type declared, no method) | ✓ | ~ |
| `embeddings` | ✓ | ✓ | ~ | ~ |
| batch `submit` / `status` / `stream` / `cancel` | ~ | ✗ | ✗ | ✗ |
| `moderations` | ~ | ✗ | ✗ | ✗ |
| audio `transcribe` / `speech` | ~ | ✗ | ✗ | ✗ |
| `generate_images` | ~ | ✗ | ✗ | ✗ |
| files `upload` / `list` / `delete` | ~ | ✗ | ✗ | ✗ |
| fine-tuning `create` / `list` / `cancel` | ~ | ✗ | ✗ | ✗ |

**What the "~" cells mean, precisely** (verified 2026-11-10 against the merge tip):

- **Python:** every one of the 149 tests passes, but the HTTP mock only ever
  serves five routes — `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`,
  `/v1/models`, `/health`. The batch, moderations, audio, image, file and
  fine-tuning methods have **no test that calls them**; the `TestTypes` cases
  only construct their dataclasses. Treat those eight rows as *implemented,
  untested* — the wire format is unverified.
- **Go:** `Embedding` and `Completion` exist and compile, but `client_test.go`
  only hits `/v1/chat/completions`, `/v1/models` and `/health`. No test calls
  `client.Embedding` or `client.Completion`.
- **Rust:** the crate's 12 in-crate tests are **pure serde/serialization tests**
  — they build `ChatRequest`, parse canned JSON, and check the `Error` display.
  None performs an HTTP call (no server is started). All five methods are
  `~`. `cargo` is not installed here, so even those 12 have not been executed.

Client-side extras present **only in Python**: automatic retry with
`Retry-After` honouring, circuit breaker, usage/cost stats, sync mirror client,
<25 framework adapters (`*_adapter.py`, `portkey_integration.py`,
`openai_agents.py`, ...) — those adapters are **untested here** (they need their
host frameworks installed) and are out of scope for this matrix.

## Batch API — the one contract every client must agree on

The server exposes the **items-based background batch API**: `POST
/v1/batch/submit`, `GET /v1/batch/{id}/status`, `GET /v1/batch/{id}/stream`,
`POST /v1/batch/{id}/cancel` (`sdk/openapi/distllm.yaml`, `src/distllm/api/routes/batch.py`).
The old `/v1/batches` + `input_file_id` OpenAI-style API is **not implemented by
the server**. Only the Python client wraps the real endpoints; the generated
constants in all four SDKs now point at the correct paths.

## Environment notes & caveats

- All executed suites are hermetic: each starts its own in-process server
  (httpx.MockTransport / vitest + node:http / net/http/httptest) returning
  canned OpenAI-compatible payloads.
- **Python:** 149/149 green. Includes regression coverage for Retry-After
  backoff, `_compute_delay(headers=...)`, public-API exports, gRPC advertise
  error, string-parity streaming, and openai_compat ignore-extra parsing.
- **JS:** build (`tsc`) and tests (`vitest`) both pass once `npm install` has
  run; `node_modules` is gitignored. Two-phase stream timeout (connect deadline
  then re-arming idle timer) is covered by `stream-timeout.test.ts`.
- **Go:** generated code lives under `sdk/go/generated/` and is a separate
  import path (`go test ./...` reports it with no test files). It is not imported
  by the root package — the root `client.go` defines its own request types.
- **Rust:** honest disclosure — `cargo check` could **not** be executed (no Rust
  toolchain on this machine). Verification was by manual review only; do not
  publish to crates.io before a CI run of `cargo check && cargo test`.
- Stale artifacts worth cleaning before release: `sdk/build/` (leftover 0.x-era
  copy missing current modules), `sdk/dist/distllm_sdk-0.5.0.tar.gz`.
- `sdk/compat/openai_compat.py` is a standalone copy of
  `src/distllm_sdk/compat/openai_compat.py`; keep them byte-identical (both are
  exercised by `test_openai_compat.py`'s packaged/standalone parametrisation).