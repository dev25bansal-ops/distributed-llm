---
tags:
  - core
  - audit
  - new
date: 2026-08-05
---

# Core Audit 05 — New Additions

**← [[Core Comprehensive Audit 2026-08-05]]**
Category 5: new features, modules, components, and third-party integrations that add tangible value and align with the architecture.

## P0 — trust & distribution first
### N1 — Reproducible distributed benchmark harness (the #1 addition)
A CLI (`distllm bench`) + nightly CI job that measures TTFT + ITL P50/P99 across 1/2/3-node consumer-GPU clusters (34B/70B), fingerprints hardware, fixes seeds, archives JSON, and **fails CI on regression**. Every downstream claim (marketing, SLA tiers, enterprise POCs) depends on this. **Effort:** 1–2w.
### N2 — Ollama Cluster plugin — the lead distribution wedge
`distllm ollama add` — pool any device already running Ollama into a DistLLM cluster, priced $5/$50 already (MARKETING.md). Converts the ~175K★ Ollama audience, who demonstrably cannot cluster devices, into the funnel for Cloud/Pro/Enterprise. **Effort:** 1–2w. Must be shipped before marketplace/enterprise.
### N3 — Clean `pip install distllm` + one-command demo
Remove the src-path hacks (PyPI must install without `PYTHONPATH=src`), add single-node mode, and a `distllm cluster demo` that spins a 2-node Docker cluster in one command. Measure time-to-first-token-in-cluster <5 min. The funnel converter. **Effort:** 1–2d.
### N4 — Real-import integration test suite + coverage-frontier gate
Separate the fake-import unit tests from a new suite that imports the real package (`from distllm.core.coordinator import Coordinator`), and add a CI gate that fails when a core module has no test reference. Turns invisible regressions (C1–C3) into caught ones. **Effort:** 5–10d.

## P1 — product & governance
### N5 — Real cloud SDK providers for spot bidding + carbon
A `MetadataProvider` interface with AWS/Azure/GCP implementations feeding `bargaining_engine`/`cross_cloud_router`; optional electricityMap/WattTime for live carbon routing. Delivers the flagship cost/ESG claims with real data. **Effort:** 3–5d.
### N6 — Enterprise compliance evidence package
SOC2 report access + HIPAA BAA + immutable audit-log export + air-gap bundle, wired to the already-claimed `aegis_compliance`/`compliance_evidence` modules (after fixing the plaintext-PHI gap). Target one regulated reference client (healthcare/financial). **Effort:** 4–8w.
### N7 — Prometheus-ready observability contract for core metrics
Expose the metrics the marketing promises (acceptance-rate, cache hit-rate, per-route cost/carbon, tenant SLI) from the wired modules (E2, E7) into `/metrics` + dashboard — currently many of these are promised but nothing wires them.

## P2 — ecosystem
### N8 — Kubernetes operator CRD (`distllm cluster install --k8s`)
The enterprise adoption gate (vLLM/Ray already have operators; DistLLM's is minimal). **Effort:** 3–4w.
### N9 — Language SDKs parity + a no-ML-deps Python client package
Clean install, auth-header handling, streaming parity (README already points at SDKs; verify against the real API).
### N10 — Plugin signing + real sandbox isolation
Upgrade `plugin_sandbox` from advisory `DISTLLM_SANDBOX_NO_NET` flag to an OS-level net namespace / seccomp boundary (or gate on the WASM path); add signature verification for marketplace plugins. **Effort:** 2w.

## Third-party integration matrix
| Integration | Value | Effort | Note |
|-------------|-------|--------|------|
| **Ollama** | Distribution wedge (N2) | 1–2w | Lead funnel |
| **Prometheus/Grafana** | Observability + enterprise | 2–3d | N7 |
| **Redis** (real) | Cross-node prompt cache (fix B15) | 0.5–1d | E4 |
| **electricityMap/WattTime** | Live carbon routing | 1–2d | N5 |
| **Kubernetes** | Enterprise adoption | 3–4w | N8 |
| **LangChain/LlamaIndex/CrewAI** | Developer radar | existing | ensure tool-contract parity |
| **HuggingFace Hub** (verified downloads) | Model provenance | 1–2d | integrity-first `trust_remote_code` |

---
**← [[Core Comprehensive Audit 2026-08-05]]** · Next: [[Core Audit 06 Verification & Testing]]