# DistLLM Tauri Desktop — Comprehensive Analysis Report

**Date:** 2026-06-02
**Version Analyzed:** v0.4.0
**Stack:** Tauri v2 (Rust) + Svelte 5 + TypeScript
**Codebase Size:** ~1,915 lines across 12 source files

---

## Table of Contents

1. [Project Analysis & Strategic Opportunities](#1-project-analysis--strategic-opportunities)
2. [Issues & Required Fixes](#2-issues--required-fixes)
3. [Enhancements & Modifications](#3-enhancements--modifications)
4. [Advanced Features](#4-advanced-features)
5. [New Additions](#5-new-additions)
6. [Verification & Testing Strategy](#6-verification--testing-strategy)
7. [Business & Product Strategy](#7-business--product-strategy)
8. [Developer Experience & Workflow](#8-developer-experience--workflow)
9. [Deployment & Distribution](#9-deployment--distribution)
10. [Community & Ecosystem](#10-community--ecosystem)
11. [Accessibility & Inclusivity](#11-accessibility--inclusivity)
12. [Documentation & Knowledge Management](#12-documentation--knowledge-management)
13. [Monitoring & Observability](#13-monitoring--observability)
14. [Implementation Roadmap](#14-implementation-roadmap)

---

## 1. Project Analysis & Strategic Opportunities

### What This Project Is

DistLLM Tauri is a desktop GUI for distributed LLM inference. It pools GPUs across multiple consumer machines to run models no single device can handle. The Rust backend spawns Python subprocesses (`distllm.cli.main`) to manage clusters, while the Svelte 5 frontend provides dashboard, cluster management, model browsing, and friend invitation features.

### Competitive Landscape

| Competitor | Their Strength | DistLLM's Advantage | DistLLM's Gap |
|---|---|---|---|
| **Ollama** | Zero-config single-machine, massive model library | Multi-GPU pooling across machines | No integrated chat UI, no model registry |
| **LM Studio** | Polished GUI, local model discovery, chat UI | Distributed inference | Single-machine only |
| **vLLM** | Production-grade serving, PagedAttention | Consumer-friendly, multi-device | No production serving story |
| **LocalAI** | Drop-in OpenAI API replacement | Multi-device pooling | More mature API compatibility |
| **llama.cpp** | Universal hardware support, smallest footprint | GUI + orchestration layer | llama.cpp is the actual inference engine |

### Key Differentiator

**No competitor solves "I have 3 consumer GPUs and want to run 70B" well.** This is DistLLM's real moat. The positioning should be:

- **"Netflix for your GPUs"** — friends pool hardware to run models none could run alone
- **"LAN party for AI"** — the social computing angle
- **Privacy-first distributed inference** — data never leaves your devices

### Strategic Opportunities

1. **Integrated Chat UI** — The #1 gap. Users can set up clusters but can't actually talk to a model. This is the single highest-impact feature.
2. **One-click model sharing** — When a friend joins, they should see available models and download with one click.
3. **Shared chat sessions** — Multiple users in the same cluster chatting with the same model simultaneously. Turns DistLLM into a social experience.
4. **Web dashboard** — The Svelte frontend can be extracted as a web app served from the coordinator node.
5. **VS Code / JetBrains extension** — Code completion using pooled GPUs.

---

## 2. Issues & Required Fixes

### CRITICAL (Must fix before any release)

#### C1: No Content Security Policy (CSP)
- **Location:** `src-tauri/tauri.conf.json:26` — `"csp": null`
- **Impact:** The webview runs with zero content security restrictions. Any XSS (e.g., via the Grafana iframe) can escalate to full RCE on the host machine.
- **Fix:**
  ```json
  "security": {
    "csp": "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self' http://127.0.0.1:*; frame-src http://localhost:3000; object-src 'none'; base-uri 'self'"
  }
  ```
- **Effort:** 30 minutes

#### C2: Shell Plugin `open: true` — Arbitrary Command Execution
- **Location:** `src-tauri/tauri.conf.json:49-51`
- **Impact:** Frontend JS can call `shell.open()` with arbitrary URLs or file paths. Combined with null CSP, this is RCE.
- **Fix:** Set `"open": false`. If needed, use scoped permissions restricting to specific URL patterns.
- **Effort:** 30 minutes

#### C3: `unwrap()` on Mutex Lock Will Panic and Crash
- **Location:** `src-tauri/src/lib.rs:603-604`
- **Impact:** `handle_tray_menu` uses `.lock().unwrap()` while the same logic in `leave_cluster` correctly uses `.lock().map_err()`. If the mutex is poisoned, the tray handler panics silently.
- **Fix:** Replace with `if let Ok(mut guard) = state.cluster_id.lock() { *guard = None; }`
- **Effort:** 15 minutes

#### C4: UI-Blocking `thread::sleep(3 seconds)`
- **Location:** `src-tauri/src/lib.rs:228`
- **Impact:** `create_cluster` blocks the Tauri main thread for 3 seconds, freezing the entire application window.
- **Fix:** Mark command `async`, use `tokio::time::sleep`, or poll the API in a loop with short timeouts.
- **Effort:** 1 hour

### HIGH (Should fix before release)

#### H1: Unvalidated `model` Parameter — Argument Injection
- **Location:** `src-tauri/src/lib.rs:212-213`
- **Impact:** User-supplied model string passed directly as CLI argument. Values starting with `-` could inject additional flags.
- **Fix:** Validate model name (alphanumeric, `/`, `-`, `_`, `.` only, max 256 chars, no leading `-`). Use `--` separator before user values.
- **Effort:** 30 minutes

#### H2: No Validation on `host` in `join_cluster`
- **Location:** `src-tauri/src/lib.rs:250-258`
- **Impact:** Host string passed directly to subprocess with zero validation. Argument injection risk.
- **Fix:** Validate host is a valid IP/hostname, reject values starting with `-`.
- **Effort:** 30 minutes

#### H3: iframe Injection via User-Controlled Grafana URL
- **Location:** `src/lib/Dashboard.svelte:230`
- **Impact:** `grafanaUrl` bound to text input, directly interpolated into iframe `src`. Can be set to `data:text/html,...` for code execution.
- **Fix:** Validate URL starts with `http://` or `https://`. Add `sandbox` attribute to iframe.
- **Effort:** 30 minutes

#### H4: Unscoped Shell Capabilities
- **Location:** `src-tauri/capabilities/default.json`
- **Impact:** Grants `shell:allow-open`, `shell:allow-execute`, `shell:allow-spawn` without scope restrictions. Any JS in the webview can execute arbitrary system commands.
- **Fix:** Scope capabilities to specific Python binary path. Remove unnecessary permissions.
- **Effort:** 1 hour

#### H5: No TLS, No Auth on Local API Calls
- **Location:** `src-tauri/src/lib.rs:141`
- **Impact:** All API traffic is plaintext HTTP with no authentication. Any local process can query the admin API.
- **Fix:** Generate random auth token on cluster creation, include in all API requests.
- **Effort:** 2 hours

#### H6: `fetch_from_api` Returns `running: true` When API Fails
- **Location:** `src-tauri/src/lib.rs:300-338`
- **Impact:** If the API call fails, the function still returns `ClusterStatus { running: true }`. UI shows "Cluster Running" when the coordinator has crashed.
- **Fix:** Return `running: false` if the API call fails.
- **Effort:** 15 minutes

### MEDIUM (Address in next sprint)

#### M1: Zombie Process Risk — No Cleanup on Crash
- **Location:** `src-tauri/src/lib.rs:608-618`
- **Impact:** No `Drop` impl on `AppState`. If the app crashes or is force-killed, Python subprocesses become orphans.
- **Fix:** Implement `Drop` for `AppState`, use job objects on Windows for process tree cleanup.
- **Effort:** 2 hours

#### M2: No Rate Limiting on Subprocess Spawning
- **Location:** `src-tauri/src/lib.rs:511-535`
- **Impact:** `download_model` spawns a new process on every call with no guard. Rapid invoke calls could exhaust system resources.
- **Fix:** Track active downloads, prevent duplicate downloads, add global semaphore.
- **Effort:** 1 hour

#### M3: Direct `fetch()` to Localhost Bypasses Tauri IPC
- **Location:** `src/lib/Dashboard.svelte:29`
- **Impact:** `autoDetectCoordinator()` uses raw `fetch("http://localhost:8000/health")` bypassing Tauri invoke layer.
- **Fix:** Route through Rust backend via a new Tauri command.
- **Effort:** 30 minutes

#### M4: Dead Code — `_model` Variable Never Used
- **Location:** `src-tauri/src/lib.rs:324-328`
- **Impact:** `/health` endpoint called on every poll, result discarded. Wasted network I/O.
- **Fix:** Either remove the call or add `model` field to `ClusterStatus`.
- **Effort:** 15 minutes

#### M5: Unused `chrono` Dependency
- **Location:** `src-tauri/Cargo.toml:22`
- **Impact:** Adds compile time and binary size for no benefit.
- **Fix:** Remove `chrono = "0.4"` from Cargo.toml.
- **Effort:** 2 minutes

#### M6: Event Listener Never Cleaned Up
- **Location:** `src/App.svelte:14-21`
- **Impact:** `listen()` returns `unlisten` function that is never called.
- **Fix:** Use `onDestroy(() => { unlisten.then(fn => fn()); })`.
- **Effort:** 5 minutes

#### M7: NVML Re-initialized on Every Poll
- **Location:** `src-tauri/src/lib.rs:377`
- **Impact:** `get_gpu_metrics` re-initializes NVML on every call (every 3 seconds). 50-100ms overhead per call.
- **Fix:** Initialize NVML once at app start, store in state.
- **Effort:** 1 hour

### LOW (Track and fix opportunistically)

#### L1: Error Messages Leak Internal Details
- **Location:** `src-tauri/src/lib.rs:145,505`
- **Impact:** Python tracebacks, file paths, and network details forwarded to frontend.
- **Fix:** Log detailed errors server-side, return sanitized messages to frontend.
- **Effort:** 1 hour

#### L2: Hardcoded Version String
- **Location:** `src-tauri/src/lib.rs:566`
- **Impact:** `distllm_version` hardcoded as `"0.4.0"`, may not match actual installed version.
- **Fix:** Query actual version via `pip show distributed-llm`.
- **Effort:** 15 minutes

#### L3: `console.log` in Production Code
- **Location:** `src/lib/Models.svelte:28`
- **Impact:** Debug noise in production.
- **Fix:** Remove or replace with user-visible notification.
- **Effort:** 2 minutes

#### L4: Weak Invite Code (32 bits of entropy)
- **Location:** `src-tauri/src/lib.rs:538-547`
- **Impact:** First 8 hex chars of UUID v4 = 32 bits, brute-forceable in seconds.
- **Fix:** Use full UUID or at least 128 bits.
- **Effort:** 15 minutes

#### L5: `user-select: none` on Body
- **Location:** `src/app.css:39`
- **Impact:** Users cannot select/copy any text — error messages, node IDs, GPU names.
- **Fix:** Remove from body, apply selectively to UI chrome only.
- **Effort:** 5 minutes

#### L6: No Confirmation on Leave Cluster
- **Location:** `src/lib/Cluster.svelte:112`
- **Impact:** `leaveCluster()` called immediately with no confirmation dialog.
- **Fix:** Add a simple confirm dialog.
- **Effort:** 15 minutes

---

## 3. Enhancements & Modifications

### 3.1 Rust Backend Modularization

**Current:** Everything in one 668-line `lib.rs` file.

**Target structure:**
```
src-tauri/src/
  main.rs          -- just calls lib::run()
  lib.rs           -- re-exports, run() builder
  types.rs         -- GpuInfo, ClusterStatus, PeerInfo, ModelInfo, InviteInfo, SystemInfo
  error.rs         -- AppError enum with thiserror derive
  state.rs         -- AppState definition with Drop impl
  commands/
    mod.rs         -- re-exports
    cluster.rs     -- create_cluster, join_cluster, leave_cluster, get_cluster_status
    gpu.rs         -- get_gpu_metrics
    models.rs      -- list_models, download_model
    system.rs      -- get_system_info, generate_invite
  process.rs       -- find_python, ensure_distllm, spawn/kill helpers
  tray.rs          -- make_tray_rgba, update_tray_icon, build_tray_menu, handle_tray_menu
  api_client.rs    -- api_get, fetch_from_api
```

**Impact:** HIGH | **Effort:** 4 hours | **Priority:** P1

### 3.2 Error Handling Overhaul

**Current:** Flat `String` errors, inconsistent types (`Result<T, String>` vs `Result<T, ApiError>`).

**Target:** Enum-based errors with `thiserror`:
```rust
#[derive(Debug, thiserror::Error, Serialize)]
pub enum AppError {
    #[error("Python not found. Install Python 3.8+ or activate your virtual environment.")]
    PythonNotFound,
    #[error("distllm package not installed. Run: pip install distributed-llm")]
    PackageNotFound,
    #[error("Cluster already running")]
    ClusterAlreadyRunning,
    #[error("Process spawn failed: {0}")]
    SpawnFailed(String),
    #[error("API request failed: {0}")]
    ApiRequestFailed(String),
}
```

**Impact:** HIGH | **Effort:** 3 hours | **Priority:** P1

### 3.3 Shared Component Library

**Current:** CSS duplicated across all 4 page components (~111 lines of identical styles).

**Target:** Extract reusable components:
```
src/lib/ui/
  Card.svelte
  Button.svelte
  Input.svelte
  ErrorBanner.svelte
  StatusDot.svelte
  Toast.svelte
  Skeleton.svelte
  styles.css       -- shared component styles
```

**Impact:** HIGH | **Effort:** 3 hours | **Priority:** P1

### 3.4 State Management Consolidation

**Current:** Dashboard and Cluster both independently poll `getClusterStatus()` every 3 seconds.

**Target:** Shared store module:
```typescript
// src/lib/stores/cluster.ts
import { writable } from 'svelte/store';
// Single source of truth for cluster state
// Single polling coordinator
// Visibility-based pause (pause when window hidden)
```

**Impact:** MEDIUM | **Effort:** 2 hours | **Priority:** P2

### 3.5 Process Lifecycle Management

**Current:** No health monitoring, no crash detection, no restart logic.

**Target:**
- Background thread calling `try_wait()` every 5 seconds
- Automatic crash detection and UI notification
- Optional auto-restart policy
- Proper process tree cleanup on Windows via job objects

**Impact:** HIGH | **Effort:** 4 hours | **Priority:** P1

### 3.6 Download Progress Streaming

**Current:** `download_model` blocks until complete. No progress, no cancel.

**Target:**
- Tauri event channel for download progress (bytes, total, speed)
- Progress bar per model card
- ETA display
- Cancel button
- Multiple simultaneous downloads

**Impact:** HIGH | **Effort:** 6 hours | **Priority:** P1

### 3.7 System Info Completion

**Current:** `cpu: "Unknown"`, `ram_gb: 0` in `get_system_info`.

**Target:** Use `sysinfo` crate to get:
- CPU name, core count, architecture
- Total/free RAM
- OS version details
- Disk space for model storage

**Impact:** MEDIUM | **Effort:** 2 hours | **Priority:** P2

### 3.8 NVML Initialization Optimization

**Current:** NVML re-initialized on every `get_gpu_metrics` call (every 3 seconds).

**Target:** Initialize once at app start, store NVML handle in `AppState`. Use background thread for periodic metric collection.

**Impact:** MEDIUM | **Effort:** 2 hours | **Priority:** P2

### 3.9 Cache Python Discovery

**Current:** `find_python()` spawns up to 3 processes on every call. `ensure_distllm()` called on every command.

**Fix:** Cache both with `OnceLock`. Reduces startup from ~1800ms to ~300ms.

**Impact:** MEDIUM | **Effort:** 30 minutes | **Priority:** P2

### 3.10 Use `ureq::Agent` with Connection Pooling

**Current:** Per-request `ureq::get()` creates a new TCP connection every time.

**Fix:** Use a shared `ureq::Agent` that maintains a connection pool. Eliminates TCP handshake overhead on every poll.

**Impact:** MEDIUM | **Effort:** 30 minutes | **Priority:** P2

### 3.11 Add Keys to `{#each}` Blocks

**Location:** `Dashboard.svelte:119,153`

**Impact:** Lack of keyed iteration causes unnecessary DOM thrashing on every poll update.

**Fix:** Add `(node.node_id)` and `(gpu.index)` keys.

**Impact:** MEDIUM | **Effort:** 10 minutes | **Priority:** P2

### 3.12 Remove Double GPU Query

**Location:** `lib.rs:559` — `get_system_info()` calls `get_gpu_metrics()` internally. Dashboard polls both, causing GPU metrics collected twice per cycle.

**Fix:** Remove GPU query from `get_system_info()`, let frontend combine the data.

**Impact:** MEDIUM | **Effort:** 15 minutes | **Priority:** P2

### 3.13 Make Commands Async

**Current:** All 9 Tauri commands are synchronous. `create_cluster` blocks for 3s, `download_model` blocks for minutes.

**Fix:** Mark all I/O commands as `async fn`. Use `tauri::async_runtime::spawn` for background tasks. Change `ureq` to `reqwest` (async) or wrap in `spawn_blocking`.

**Impact:** HIGH | **Effort:** 3 hours | **Priority:** P1

### 3.14 Exponential Backoff on Poll Failures

**Current:** Dashboard polls every 3 seconds forever even when endpoint is unreachable.

**Fix:** Backoff: 3s → 6s → 12s → max 30s. Reset on success.

**Impact:** MEDIUM | **Effort:** 30 minutes | **Priority:** P2

---

## 4. Advanced Features

### 4.1 Integrated Chat Interface
- **Description:** A full chat UI that sends requests to the OpenAI-compatible API endpoint exposed by the coordinator
- **Features:** Streaming token display, conversation history, system prompt configuration, temperature/top-p controls
- **Impact:** CRITICAL — This is the #1 feature gap. Without it, users set up clusters but can't use them.
- **Effort:** 2-3 days
- **Priority:** P0

### 4.2 Real-Time Inference Metrics
- **Description:** Display tokens/second, time-to-first-token, inter-token latency, and throughput comparison (single-node vs multi-node)
- **Impact:** HIGH — Users need to see the value of distributed inference
- **Effort:** 1 day
- **Priority:** P1

### 4.3 Model Performance Benchmarking
- **Description:** Built-in benchmark suite that tests inference speed across different configurations
- **Features:** Compare single-GPU vs multi-GPU, different quantizations, different model sizes
- **Impact:** MEDIUM — Valuable for power users and community sharing
- **Effort:** 2 days
- **Priority:** P2

### 4.4 Cluster Topology Visualization
- **Description:** Visual graph showing how model layers are distributed across nodes
- **Features:** Drag-and-drop layer reassignment, real-time data flow visualization
- **Impact:** MEDIUM — Helps users understand and optimize their cluster
- **Effort:** 2 days
- **Priority:** P2

### 4.5 Multi-Model Serving
- **Description:** Run multiple smaller models simultaneously across the cluster
- **Features:** Model routing, load balancing, per-model resource limits
- **Impact:** HIGH — Unlocks new use cases (e.g., coding model + chat model simultaneously)
- **Effort:** 1 week
- **Priority:** P2

### 4.6 Plugin System
- **description:** Allow users to add custom model backends, authentication providers, and monitoring integrations
- **Impact:** MEDIUM — Extensibility for power users
- **Effort:** 1 week
- **Priority:** P3

### 4.7 Web Dashboard Extraction
- **Description:** Serve the Svelte frontend from the coordinator node as a web dashboard
- **Impact:** HIGH — Enables headless server management and mobile companion access
- **Effort:** 2 days
- **Priority:** P2

### 4.8 Auto-Discovery via mDNS
- **Description:** Use `mdns-sd` crate to broadcast and discover coordinators on LAN
- **Impact:** HIGH — Eliminates manual IP:port entry
- **Effort:** 1 week
- **Priority:** P2

### 4.9 Ollama Compatibility Layer
- **Description:** Accept `OLLAMA_HOST` env var, serve the same API shape
- **Impact:** HIGH — Users can drop DistLLM into any Ollama workflow
- **Effort:** 3 days
- **Priority:** P2

### 4.10 Keyboard Shortcuts
- **Description:** Ctrl+1-4 for page navigation, Ctrl+G for Grafana toggle, Escape to dismiss errors
- **Impact:** LOW — Power user convenience
- **Effort:** 1 hour
- **Priority:** P3

---

## 5. New Additions

### 5.1 First-Launch Onboarding Wizard
- **Description:** 3-step guided flow on first launch: detect GPUs → recommend model → create/join cluster
- **Impact:** HIGH — Critical for first-time user retention
- **Effort:** 1 day
- **Priority:** P1

### 5.2 Toast Notification System
- **Description:** Non-blocking notifications for success, info, warning, and error states
- **Position:** Bottom-right, auto-dismiss after 4 seconds
- **Impact:** HIGH — Users currently get zero feedback on most operations
- **Effort:** 2 hours
- **Priority:** P1

### 5.3 Settings Page
- **Description:** Centralized configuration for:
  - Default cluster port
  - Grafana URL
  - Theme (dark/light/auto)
  - Auto-join behavior
  - Download directory
  - Notification preferences
  - Python path override
- **Impact:** MEDIUM
- **Effort:** 1 day
- **Priority:** P2

### 5.4 Activity/Logs Viewer
- **Description:** Scrollable log viewer showing cluster events, model downloads, inference requests
- **Features:** Filter by severity, search, export
- **Impact:** MEDIUM — Essential for debugging
- **Effort:** 1 day
- **Priority:** P2

### 5.5 QR Code Generation
- **Description:** Currently a placeholder. Implement actual QR code rendering for invite links.
- **Fix:** Use a pure JS QR library (no Python dependency needed)
- **Impact:** LOW — Nice-to-have for the invite flow
- **Effort:** 1 hour
- **Priority:** P3

### 5.6 Dark/Light Theme Toggle
- **Description:** Currently hardcoded dark theme. Add theme switching with CSS custom properties.
- **Impact:** MEDIUM — Many users prefer light themes
- **Effort:** 2 hours
- **Priority:** P2

### 5.7 System Tray Enhancements
- **Description:** Show cluster status, connected nodes, active model in tray tooltip
- **Features:** Quick actions (start/stop cluster), recent clusters list
- **Impact:** LOW
- **Effort:** 2 hours
- **Priority:** P3

### 5.8 Deep Link Handler
- **Description:** Handle `distllm://connect/...` URLs to auto-join clusters from invite links
- **Impact:** HIGH — Critical for the friend invitation flow
- **Effort:** 2 hours
- **Priority:** P1

---

## 6. Verification & Testing Strategy

### 6.1 Current State: ZERO Tests

There are no tests anywhere in the project. This is the most critical technical debt.

### 6.2 Rust Backend Testing

**Framework:** Built-in `#[cfg(test)]` + `cargo test`

**What to test:**
- `find_python()` — mock `Command` execution
- `ensure_distllm()` — mock subprocess check
- `api_get()` — mock HTTP responses (use `mockito` or `wiremock`)
- `fetch_from_api()` — test with various JSON responses
- `make_tray_rgba()` — pixel-level verification
- Input validation functions (once added)
- `AppError` serialization/deserialization

**Mocking strategy:** Use trait-based dependency injection for `Command` execution and HTTP calls.

### 6.3 Frontend Testing

**Framework:** Vitest + @testing-library/svelte

**What to test:**
- `api.ts` — mock Tauri `invoke`, verify correct command names and args
- `types.ts` — type compatibility with Rust structs
- Component rendering — smoke tests for each page
- `Models.svelte` — search filtering logic
- `Cluster.svelte` — form validation (once added)
- `Dashboard.svelte` — `fmtBytes`, `fmtPct`, `statusColor` utilities

### 6.4 Integration Testing

**Framework:** Tauri's built-in test harness + Playwright for E2E

**Scenarios:**
- Create cluster → verify status polling shows running
- Join cluster → verify node appears in dashboard
- Leave cluster → verify cleanup
- Download model → verify list updates
- Generate invite → verify link format
- GPU metrics → verify response structure

### 6.5 Security Testing

- **Static analysis:** `cargo audit` for Rust dependencies, `npm audit` for JS
- **Dynamic testing:** Attempt XSS via Grafana URL input, attempt argument injection via model/host fields
- **Fuzzing:** Fuzz all Tauri command inputs with random strings, extreme values, Unicode

### 6.6 Cross-Platform Matrix

| Platform | GPU | Tests |
|----------|-----|-------|
| Windows 10/11 | NVIDIA | Full suite |
| Windows 10/11 | AMD | GPU detection only |
| Ubuntu 22.04+ | NVIDIA | Full suite |
| Ubuntu 22.04+ | AMD (ROCm) | GPU detection |
| macOS 14+ | Apple Silicon | GPU detection (system_profiler) |
| macOS 14+ | Intel | Fallback |

### 6.7 CI/CD Pipeline

```yaml
# .github/workflows/test.yml
jobs:
  lint:
    - cargo clippy -- -D warnings
    - pnpm eslint src/
    - pnpm tsc --noEmit
  test:
    - cargo test
    - pnpm vitest run
  build:
    - cargo build --release (Windows, Linux, macOS)
    - pnpm tauri build
  security:
    - cargo audit
    - npm audit
```

### 6.8 Coverage Targets

| Layer | Target | Tool |
|-------|--------|------|
| Rust commands | 80% | `cargo tarpaulin` |
| TypeScript API | 90% | Vitest coverage |
| Components | 70% | Vitest + testing-library |
| E2E critical paths | 100% | Playwright |

---

## 7. Business & Product Strategy

### 7.1 Monetization Tiers

| Tier | Price | Features |
|------|-------|----------|
| **Free** | $0 | Local cluster (same LAN), up to 3 nodes, community support |
| **Pro** | $9/mo | WAN clusters (NAT traversal), unlimited nodes, priority downloads, advanced monitoring |
| **Team** | $29/mo/node | Shared cluster management, RBAC, audit logs, SSO |
| **Enterprise** | Custom | On-prem, air-gapped, compliance certs, dedicated support |

### 7.2 Additional Revenue Streams

- **Model marketplace** — Curated, pre-quantized model bundles optimized for DistLLM clusters. 15-20% commission.
- **GPU time sharing** — Users rent idle GPU time to others. DistLLM takes a cut.
- **Managed relay servers** — Hosted TURN/relay for WAN clusters. $5/mo per cluster.

### 7.3 Go-to-Market Strategy

1. **Phase 1 (Now):** Open-source core, build community on Discord/Reddit
2. **Phase 2 (v1.0):** Launch Pro tier with WAN support
3. **Phase 3 (v2.0):** Enterprise features, model marketplace
4. **Phase 4:** GPU time sharing marketplace

### 7.4 Key Metrics to Track

- Daily active clusters
- Average nodes per cluster
- Model download volume
- Inference requests per day
- User retention (D1, D7, D30)
- Cluster uptime

---

## 8. Developer Experience & Workflow

### 8.1 Build System

**Current state:** Basic `npm run dev` / `npm run build` scripts.

**Improvements needed:**
- Add `cargo clippy` to dev workflow
- Add `cargo fmt` check to CI
- Add pre-commit hooks for linting
- Add `pnpm` as package manager (faster than npm)

### 8.2 Development Environment

**Improvements:**
- Add `.env.example` for configuration
- Add `Makefile` or `justfile` for common tasks
- Add hot-reload for Rust changes (Tauri supports this)
- Add mock mode for testing without actual GPUs

### 8.3 Code Quality Tools

**Add:**
- `clippy` with strict lints
- `rustfmt` configuration
- ESLint rules (already configured, good)
- Prettier for consistent formatting
- `cargo-deny` for dependency auditing

### 8.4 Version String Centralization

**Current:** Version `0.4.0` is hardcoded in 4 places: `Cargo.toml`, `package.json`, `tauri.conf.json`, and `Nav.svelte` line 34.

**Fix:** Use `app.getVersion()` from Tauri API or a build-time constant.

**Impact:** LOW | **Effort:** 15 minutes

### 8.5 Add Logging

**Current:** Zero logging in the Rust backend. Any production issue is invisible.

**Fix:** Add `tracing` crate with structured logging. Use `tauri-plugin-log` to expose logs in the UI.

**Impact:** HIGH | **Effort:** 2 hours

### 8.6 Dependency Injection for Testability

**Current:** All commands directly spawn processes and access hardware. Untestable without mocking.

**Fix:** Define traits for external dependencies:
- `CommandRunner` for process spawning
- `HttpClient` for API calls
- `GpuProvider` for NVML/hardware access

**Impact:** HIGH | **Effort:** 4 hours

### 8.7 Documentation

**Add:**
- Architecture Decision Records (ADRs) for key decisions
- API documentation for Tauri commands
- Component storybook or documentation
- Contributing guide specific to the Tauri app

---

## 9. Deployment & Distribution

### 9.1 Current State

- `tauri.conf.json` has `"targets": "all"` for bundling
- Icons are generated via Python script (solid color placeholders)
- No CI/CD for building releases

### 9.2 Improvements Needed

1. **Real app icons** — Replace solid-color placeholders with proper branded icons
2. **Auto-update** — Tauri's built-in updater plugin for seamless updates
3. **Code signing** — Windows (Authenticode), macOS (notarization), Linux (GPG)
4. **CI/CD releases** — GitHub Actions to build and publish on tag push
5. **Installer customization** — Custom NSIS installer for Windows, DMG for macOS, AppImage/deb/rpm for Linux
6. **Version management** — Sync version across `package.json`, `Cargo.toml`, and `tauri.conf.json`

### 9.3 Distribution Channels

- GitHub Releases (primary)
- Homebrew (macOS)
- winget / Chocolatey (Windows)
- Snap Store / Flatpak (Linux)
- Direct download from website

---

## 10. Community & Ecosystem

### 10.1 Community Building Features

- **Cluster leaderboard** — Aggregate tokens/sec, model benchmarks, cluster uptime
- **Community model packs** — Curated collections: "Best models for creative writing under 70B"
- **Discord bot integration** — Connect a Discord bot to a DistLLM cluster
- **Shared chat sessions** — Multiple users chatting with the same model

### 10.2 Open Source Strategy

- Keep core distributed inference open source (Apache 2.0)
- Open source the Tauri desktop app
- Proprietary: WAN relay servers, enterprise features, model marketplace

### 10.3 Documentation for Contributors

- Architecture overview
- How to add a new Tauri command
- How to add a new page
- How to test with mock GPUs
- Release process

---

## 11. Accessibility & Inclusivity

### 11.1 Critical Accessibility Issues

| Issue | Location | Severity | Fix |
|-------|----------|----------|-----|
| Focus outlines removed globally | `app.css:57-66` | CRITICAL | Remove `outline: none`, add `focus-visible` styles |
| No ARIA labels on navigation | `Nav.svelte` | HIGH | Add `role="navigation"`, `aria-label`, `aria-current` |
| Active nav contrast fails WCAG AA | `Nav.svelte:87-89` | HIGH | Darken accent or use border indicator |
| No keyboard navigation for sidebar | `Nav.svelte` | HIGH | Add arrow-key support, visible focus indicators |
| No skip-to-content link | `App.svelte` | MEDIUM | Add skip link for keyboard users |
| Error banners have no role | All components | MEDIUM | Add `role="alert"` to error banners |
| No reduced-motion support | All animations | MEDIUM | Add `prefers-reduced-motion` media query |
| Color-only status indicators | `Dashboard.svelte` | MEDIUM | Add text labels alongside color dots |

### 11.2 Accessibility Fixes Priority

1. **P0:** Fix focus outlines (affects all keyboard users)
2. **P1:** Add ARIA attributes to navigation
3. **P1:** Fix color contrast on active nav item
4. **P2:** Add keyboard navigation
5. **P2:** Add reduced-motion support

---

## 12. Documentation & Knowledge Management

### 12.1 Missing Documentation

- **Architecture overview** — How the Rust backend, Svelte frontend, and Python subprocess interact
- **API reference** — All 9 Tauri commands with input/output types
- **Setup guide** — Development environment setup
- **Deployment guide** — How to build and distribute
- **Troubleshooting** — Common issues and solutions
- **Security model** — What the app can and cannot do

### 12.2 In-Code Documentation

- Rust functions lack doc comments
- TypeScript interfaces lack JSDoc
- Complex logic (tray icon rendering, NVML fallback) needs explanatory comments

---

## 13. Monitoring & Observability

### 13.1 Current State

- Grafana iframe embed (user must set up Grafana separately)
- No built-in metrics collection
- No error reporting
- No usage analytics

### 13.2 Improvements

1. **Built-in metrics dashboard** — Don't require external Grafana. Show key metrics natively.
2. **Error reporting** — Capture and report errors (with user consent) for debugging
3. **Usage analytics** — Track feature usage to inform product decisions
4. **Health checks** — Periodic checks for Python availability, GPU health, network connectivity
5. **Performance profiling** — Built-in profiling for inference latency and throughput

---

## 14. Implementation Roadmap

### Phase 1: Security & Stability (Week 1-2)
- [ ] Fix CSP (C1) — 30 min
- [ ] Fix shell plugin (C2) — 30 min
- [ ] Fix unwrap panic (C3) — 15 min
- [ ] Fix UI-blocking sleep (C4) — 1 hour
- [ ] Add input validation (H1, H2) — 1 hour
- [ ] Fix iframe injection (H3) — 30 min
- [ ] Scope shell capabilities (H4) — 1 hour
- [ ] Add auth token (H5) — 2 hours
- [ ] Fix fetch_from_api false positive (H6) — 15 min
- [ ] Add Drop impl for AppState (M1) — 2 hours
- [ ] Remove dead code (M4, M5) — 15 min

**Total: ~10 hours**

### Phase 2: Architecture & Quality (Week 3-4)
- [ ] Modularize Rust backend — 4 hours
- [ ] Error handling overhaul — 3 hours
- [ ] Extract shared UI components — 3 hours
- [ ] Toast notification system — 2 hours
- [ ] State management consolidation — 2 hours
- [ ] NVML initialization optimization — 2 hours
- [ ] Process health monitoring — 4 hours
- [ ] Fix event listener cleanup (M6) — 5 min
- [ ] Complete system info (M7) — 2 hours

**Total: ~22 hours**

### Phase 3: Core Features (Week 5-8)
- [ ] Integrated chat interface — 3 days
- [ ] Download progress streaming — 1 day
- [ ] First-launch onboarding wizard — 1 day
- [ ] Deep link handler — 2 hours
- [ ] Settings page — 1 day
- [ ] Logs viewer — 1 day
- [ ] Dark/light theme toggle — 2 hours

**Total: ~8 days**

### Phase 4: Testing & Polish (Week 9-10)
- [ ] Rust unit tests — 2 days
- [ ] Frontend unit tests — 2 days
- [ ] Integration tests — 1 day
- [ ] E2E tests — 1 day
- [ ] Accessibility fixes — 1 day
- [ ] Cross-platform testing — 1 day
- [ ] CI/CD pipeline — 1 day

**Total: ~9 days**

### Phase 5: Advanced Features (Week 11+)
- [ ] Real-time inference metrics — 1 day
- [ ] Model performance benchmarking — 2 days
- [ ] Cluster topology visualization — 2 days
- [ ] Multi-model serving — 1 week
- [ ] Plugin system — 1 week
- [ ] QR code generation — 1 hour
- [ ] System tray enhancements — 2 hours

---

## Summary Statistics

| Category | Count |
|----------|-------|
| CRITICAL issues | 4 |
| HIGH issues | 13 |
| MEDIUM issues | 18 |
| LOW issues | 9 |
| Total issues | 44 |
| Enhancement opportunities | 22 |
| New feature proposals | 19 |
| Estimated total effort (Phase 1-4) | ~6 weeks |

**The most impactful single feature is the integrated chat interface.** Without it, the app is a cluster management tool with no user-facing value proposition. With it, the app becomes a complete local AI experience.

**The most impactful security fix is implementing CSP + scoping shell capabilities.** Together, these close the most dangerous attack surface: XSS → RCE escalation.

**The most impactful refactoring is the module structure split of `lib.rs`.** This is the prerequisite for testing, maintainability, and all subsequent improvements.

**Estimated effort to reach MVP quality:** 2-3 weeks of focused work (fix criticals, add chat UI, add basic tests, add logging).

---

## What Is Done Well

These areas should be preserved and built upon:

1. **Clean type mirroring** between `types.ts` and `lib.rs` — field names, Option types, and serde serialization are consistent and correct
2. **Svelte 5 runes usage** — proper `$state`, `$derived`, `$props`, `$effect` patterns throughout
3. **Consistent API layer** — `api.ts` is a thin, well-typed wrapper with no business logic leakage
4. **Tray icon rendering** — programmatic RGBA circle with anti-aliased edges is creative and dependency-free
5. **Platform-aware GPU detection** — Linux (NVML + lspci), Windows (NVML), macOS (system_profiler) with appropriate fallbacks
6. **Component structure** — page-per-component with shared Nav scales well
7. **Release profile** — LTO, strip, single codegen unit, abort on panic. Production-ready binary optimization
8. **Minimum window size** — already configured at 800x600
