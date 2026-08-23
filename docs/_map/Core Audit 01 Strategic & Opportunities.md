---
tags:
  - core
  - audit
  - strategic
date: 2026-08-05
---

# Core Audit 01 — Project Analysis & Strategic Opportunities

**← [[Core Comprehensive Audit 2026-08-05]]**
Category 1: market differentiation, competitive advantage, and strategic value propositions — grounded in the actual state of `src/distllm/core/**` (many differentiators are real code, but unproven/unwired).

## The honest starting point
DistLLM has **unusually deep engineering** (P2P GPU pooling, 6+ speculative decoders, DQN spot bidding, tiered KV cache, cross-cluster cache coherence, DP accounting, RAG vectorstore, HA + PBFT) — but the go-to-market currently rests on **zero published distributed benchmarks** ([[Core Audit 02 Issues & Required Fixes#C5]]), ~20 unwired "feature-shelf" modules ([[Core Audit 07 Dead Code & Consolidation]]), and 242 core modules that contradict a consumer-first "pool your GPUs" story (F180). The strongest asset is the **docs/compliance body** (institutional-grade marketing, financials, GDPR/SOC2 mapping); its main risk is that capability-vs-claims gaps turn that asset into a liability.

## 1. Killer features + defensible moat
| Feature | Status | Moat vs | Notes |
|---------|--------|---------|-------|
| **Remote-draft speculative decoding** over public APIs (cloud small model drafts, local big model verifies) | Implemented, unwired | vLLM/Ollama/Petals — nobody ships hosted-draft hybrid | **Headline technical differentiator.** Data stays local, latency hides behind draft (F177). |
| **WAN + heterogeneous prefill/decode disaggregation** | Implemented | Petals' documented per-hop RTT weakness | Attacker of the exact P2P distributed-inference niche (F185). |
| **Peer-to-peer GPU pooling over LAN/Internet** | Implemented | Ollama (single-device), vLLM (single-node) | The core wedge; needs the benchmark to prove it. |
| **Tiered KV cache + cross-cluster cache coherence (Bloom→Merkle→full)** | Implemented | None mainstream | L2 "which-blocks-differ" is a stub today ([[Core Audit 02 Issues & Required Fixes|hierarchical_digest]]) — fix before claiming 95% savings. |
| **Carbon-aware + cost-aware cross-cloud routing** | Implemented, fragmented | Cloud LLM gateways | Unique ESG/cost joint routing; consolidate routers (F52). |
| **DP inference + privacy accounting** | Implemented, **guarantee not enforced** | None mainstream | Must fix the false-guarantee (S1/S2) before marketing it. |
| **Consumer-GPU spot arbitrage (DQN bidding, spot failover, cost forecast)** | Implemented, unwired | None | TCO wedge for the audience clouds can't serve at spot prices (F154). |
| **Per-tenant QoS isolation + route audit** (`sentinel_qos`, `route_audit`) | Implemented, unwired | vLLM/TGI lack it | The enterprise/regulated-buyer governance surface (F143). |
| **Request dedup + in-flight coalescing** | Implemented, unwired | None ship it | Concrete multi-tenant cost/latency win on duplicate prompts (F131). |
| Ollama Cluster plugin ($5/$50 priced already) | Spec'd, unshipped | Ollama (~175K★) can't cluster | The distribution wedge (F178). |

## 2. Competitive comparison (with the caveat that "YES" ≠ "measured")
DistLLM uniquely combines pipeline parallelism, multi-GPU pooling, speculative decoding, spot bidding, carbon routing, and cross-cluster cache sync. vLLM/Ray win on **production monitoring, K8s operator, docs, adoption, published benchmarks**. DistLLM's plan is credible only when the "YES" cells are backed by a reproducible measurement spine.

## 3. The single biggest adoption risk: **trust + complexity**
A prospect who reproduces a missing/bad distributed result abandons. Mitigate by sequencing (F186): **measure first, then promise.** Never ship "provisional SLA tiers" in sales/compliance collateral (F181) — in regulated procurement, "estimated SLA in writing" is worse than none.

## 4. Five sequenced strategic moves (F186)
1. **P0 now — publish the reproducible distributed benchmark program** (2×/3× consumer-GPU 34B/70B, TTFT + ITL P50/P99, N≥30, hardware-fingerprinted, CI-regression-gated). This is the trust gate for everything downstream.
2. **P0 now — ship the Ollama Cluster plugin** as the lead distribution wedge into the largest install base of target users.
3. **P1 — bulletproof first-run DX**: clean `pip install distllm` (no src-path hacks), single-node mode, `distllm cluster demo` (2-node Docker in one command), time-to-first-token-in-cluster <5 min.
4. **P1/pipeline — headline remote-draft + WAN-vs-Petals latency** as the technical moat, backed by measured ITL-over-RTT curves.
5. **P2 (after ≥500 active clusters) — enterprise data-sovereignty/compliance packaging + marketplace**, not before liquidity exists.

## 5. Monetization angles (from FINANCIALS/MARKETING)
- **Cloud GPU-hour** ($0.15/GPU-hr) + **Ollama plugin** ($5/$50) earn day-one revenue — sequence these first.
- **Enterprise self-hosted** is the ACV anchor (Y1 $40K → Y4 $12M) and the one axis cloud structurally can't meet (data + weights never leave premises). Package the **compliance evidence trail** (SOC2/HIPAA/SSO/air-gap, already claimed) before selling it (F179).
- **Defer the reputation marketplace** (chicken-and-egg, projected Y1 $2K vs Enterprise $12M); a two-sided hub won't boot on zero liquidity (F183).
- Generous Apache-2.0 core + usage-based (not per-seat) pricing is a legitimate differentiator that drives virality.

## 6. What to cut vs keep (focus is positioning)
**Keep** as moat: KV quantization, speculative decoding (converge the 5 variants), tiered cache, routing, discovery, coordinator/scheduler.
**Move behind feature-flags/EXPERIMENTAL** to shrink the 242-module surface: marketplace/hub, DP, federated finetuning, watermarking, A/B testing, kv-cache marketplace, wisp/wasm, hydra-diffusion (F180). Target: a new adopter understands the product from 5 modules, not 242.

## 7. Documentation: strength and risk
Docs are institutional-grade (competitive-analysis, financials with cohorts/CAC, GDPR, air-gap, migration guides) — a recruiting/enterprise/partner moat. But README roadmap is out of date (auto-discovery still listed as Q3-2026 while shipped) and benchmark tables are "Coming soon". **Reconcile roadmap with shipped reality** and tag every capability `state: implemented / benchmarked / provisional` (F182).

---
**← [[Core Comprehensive Audit 2026-08-05]]** · Next: [[Core Audit 02 Issues & Required Fixes]]