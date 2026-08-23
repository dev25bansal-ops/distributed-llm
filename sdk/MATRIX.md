# SDK Integration Matrix

Status of every DistLLM SDK: how it installs, what was actually tested on this
machine (2026-08-24), and where each stands for publishing.

Smoke legend: **live-smoke** = an automated test spins an in-process mock
OpenAI-compatible HTTP server (`/v1/models`, `/v1/chat/completions`, ...) and
asserts the SDK parses the canned responses into typed values. No external
network required.

| SDK | Language | Install method | Tested version | Toolchain | Live-smoke | Publish status |
|-----|----------|----------------|----------------|-----------|------------|----------------|
| `distllm-sdk` (Python, async + sync clients) | Python >= 3.10 | PyPI `pip install distllm-sdk` · source `pip install -e sdk/` | 1.0.0 (from `sdk/src`) | Python 3.14.4, httpx 0.28.1, pytest 9.0.2 | ✓ 10/10 — `sdk/tests/test_smoke_live.py` (httpx `MockTransport`; models, chat, completions, embeddings, health, SSE stream, auth/payload shape) | Ready. Local sdist `sdk/dist/distllm_sdk-0.5.0.tar.gz` is **stale** (0.5.0 vs pyproject 1.0.0) — rebuild before next PyPI upload |
| `distllm-sdk` (JS/TS) | Node >= 18 | npm `npm i distllm-sdk` (**not yet published**) · git subdir `sdk/js` | 1.0.0 (local) | Node 24.12.0, vitest 1.6.1, tsc 5.9.3 | ✓ 6/6 — `sdk/js/test/smoke.test.ts` (`npm test`; node:http mock server; models, chat create + SSE stream, completions, embeddings, 404 error mapping, no-retry-on-4xx) | **Ready** — `npm pack --dry-run` verified (7 files, 3.5 kB). Blocked only on adding `sdk/js/README.md` (recommended; npm auto-includes it once present) and the actual `npm publish` |
| `distllm-sdk-go` | Go >= 1.21 | `go get github.com/distributed-llm/distllm-sdk-go` (**needs git tag**) | 1.0.0-dev (go.mod, untagged) | Go 1.26.2 windows/amd64 | ✓ 5/5 — `sdk/go/client_test.go` (`go test ./...`; httptest server; models, chat + request-body shape, health, SSE stream) | Module path correct in `go.mod`; package docs in `doc.go`; gofmt/vet clean. Publish = push tag `v1.0.0` |
| `distllm-sdk` (Rust) | Rust 2021 edition | crates.io `cargo add distllm-sdk` (**not yet published**) · git subdir `sdk/rust` | — **not tested** | ⚠ cargo/rustc **not installed** on this machine | ✗ review-only | **Blocked** until `cargo check` / `cargo test` can run. `Cargo.toml` + `lib.rs` reviewed by hand (structs/serde/thiserror consistent); metadata completed |

## How to run the smokes

```bash
# Python (10 tests)
python -m pytest sdk/tests/test_smoke_live.py -v

# JS (6 tests)
cd sdk/js && npm test

# Go (5 tests)
cd sdk/go && go test ./...
```

## Environment notes & caveats

- All smokes are hermetic: each starts its own in-process server
  (httpx.MockTransport / vitest + node:http / net/http/httptest) returning
  canned OpenAI-compatible payloads — same fixtures across all three languages.
- **Python:** full suite green — `python -m pytest sdk/tests/` → 103 passed
  (includes the 4 formerly-failing tests now fixed: Retry-After backoff,
  `_compute_delay(headers=...)`, public-API exports, gRPC advertise error).
- **Go:** generated code lives under `sdk/go/generated/` (separate package);
  `gofmt`, `go vet`, `go test` all clean.
- **Rust:** honest disclosure — `cargo check` could **not** be executed
  (no Rust toolchain on this machine). Verification was by manual review only;
  do not publish to crates.io before a CI run of `cargo check && cargo test`.
- Stale artifacts worth cleaning before release: `sdk/build/` (leftover 0.x-era
  copy missing current modules), `sdk/dist/distllm_sdk-0.5.0.tar.gz`,
  `sdk/compat/` (superseded by `src/distllm_sdk/compat/`).
- Framework integrations (~25 adapters under `src/distllm_sdk/*_adapter.py`,
  `portkey_integration.py`, `openai_agents.py`, `vercel_adapter.py`, ...) are
  **untested** here — they need their host frameworks installed and are out of
  scope for this matrix update.
