# DistLLM — Focus Areas

Divide-and-conquer map of the project. Each area is independently improvable.
Work them in the recommended order at the bottom; ignore areas outside your
current focus without guilt — the boundaries are clean by design.

---

## 1 · Core Inference Engine — `src/distllm/core/`
**What:** coordinator, batch scheduler, KV cache, speculative decoding,
inference engine, model router. The brain of the product.
**State:** strongest area. ~11,900-test suite green; end-to-end local
inference verified; paged KV cache + defrag + three speculative engines done.
**Next up:**
1. Replace placeholder benchmark numbers with real suite runs (blocks public launch credibility)
2. Cluster Planner: device picker → which models fit → expected tok/s
3. Continuous batching throughput tuning (measure before/after)

## 2 · Distributed Transport — `src/distllm/dist/`
**What:** pipeline orchestrator (1F1B), gRPC node service/client, P2P
gossip + DHT, federation, recovery, straggler detection, bandwidth control.
**State:** functional; multi-node join verified live; security hardened
(HMAC gossip/DHT, Ed25519 federation).
**Next up:**
1. Real two-laptop demo recording (marketing asset + e2e proof)
2. WAN scenario tests currently need live servers — make them hermetic or dockerized
3. Node-representation cleanup (dict vs PipelineNode inconsistency found in tests)

## 3 · API Server — `src/distllm/api/`
**What:** FastAPI app, 20+ route groups, middleware stack, auth, SSE streaming,
rate limiting, persistent store.
**State:** production-shaped after hardening round — plugins admin-gated,
fail-closed auth everywhere, connection-pooled store, unified rate limiter.
**Next up:**
1. OpenAPI spec polish → publish interactive docs link on the website
2. API versioning strategy (/v1 freeze promise) before external users arrive
3. Load test with realistic concurrency (locust script → real numbers for benchmarks page)

## 4 · Security Layer — `src/distllm/security/` + auth surfaces
**What:** E2E encryption (X25519), cert rotation, PBKDF2 key store, HMAC
gossip/DHT auth, SSRF guards, prompt-injection defense.
**State:** hardened through adversarial review rounds; all bypasses removed;
fail-closed patterns throughout.
**Next up:**
1. External security.txt + vulnerability-disclosure email
2. Dependency scanning gate (pip-audit / safety in CI)
3. Threat-model doc for the security page (turns claims into evidence)

## 5 · Training & Privacy — federated fine-tuning, DP
**What:** FederatedFineTuner (FedProx + DP), RDP accounting, LoRA training.
**State:** implemented + tested but invisible — no docs page, no marketing mention.
**Next up:**
1. Docs section: "Private fine-tuning on pooled devices"
2. Blog post: DP training demo walkthrough
3. Wire a minimal example into examples/

## 6 · SDKs & Integrations — `sdk/`, `integrations/`
**What:** Python ×2, JS, Go, Rust SDKs; ~25 framework integrations.
**State:** all compile; JS/Go/Rust lack live-call smoke coverage; Go needed a go.mod (added).
**Next up:**
1. One live-call integration test per SDK against a spinning server
2. Publish JS SDK to npm + Go module tag (parity with PyPI)
3. Integration matrix table on /integrations (tested ✓ vs untested)

## 7 · Website & Docs — `website-astro/`
**What:** Astro 7 site — home, 17-page docs, playground, blog, changelog.
Deployed on Vercel. Dark terminal theme.
**State:** strong visually; content thin spots remain (docs word counts,
placeholder benchmark data).
**Next up:**
1. Truth-in-benchmarks fix (blocked on Area 1 item 1)
2. Mobile docs nav drawer (currently hidden below lg)
3. Cmd+K global search across blog/glossary

## 8 · CLI & Config — `cli/`, `config/`
**What:** Typer CLI (system api/node/doctor/config keys), YAML config resolver.
**State:** working; doctor fixed; config precedence env>YAML tested.
**Next up:**
1. `distllm cluster` subcommands (list-nodes, status) polish — used by demos
2. Shell completions
3. Config validation errors with file:line pointers

## 9 · Observability — metrics, tracing, telemetry
**What:** Prometheus exporter (RED metrics wired), tracing middleware,
telemetry flush fixes, Grafana dashboards dir.
**State:** metrics visible at /metrics; tracing active; dashboards untested.
**Next up:**
1. Ship one reference Grafana dashboard JSON
2. Docs page with metric dictionary

---

## Recommended focus order (solo-founder, pre-public)

| Sprint | Focus | Why now |
|--------|-------|---------|
| 1 | Area 1 item 1 + Area 7 truth fix | Nothing else matters until published numbers are real |
| 2 | Area 2 item 1 (demo recording) | Your investor/launch asset |
| 3 | Area 3 + 8 (API + CLI polish) | First-impression surfaces for new users |
| 4 | Area 6 item 2 (npm/Go publish) | Multi-language reach |
| ongoing | Security (Area 4) | Continuous, not a sprint |

**Rule of thumb:** one area per week. Finish its "next up" list before touching
another — the boundaries above are deliberately independent so progress in one
never waits on another.
