# DistLLM Hybrid Managed Control Plane — "DistLLM Cloud"

> **Status: STRATEGY + ARCHITECTURE BRIEF (E13).** This document plus the
> `distllm.cloud.worker_agent` scaffold are the *only* deliverables. The
> managed control-plane SaaS itself is **NOT built** here — this is the
> architecture, the go-to-market positioning, and a thin, dependency-light
> worker-agent stub that shows how a customer node would join a hosted
> coordinator. Everything network-facing is a **SCAFFOLD**.

---

## 1. Problem — the sovereignty gap

The incumbent managed GPU/inference platforms (RunPod, Modal, Together, Fireworks,
etc.) all share one property: **your compute and your data run on *their* infra.**
Prompts, weights, fine-tunes, and KV caches all transit and reside in a vendor
account you do not own. For regulated, sovereign, or IP-sensitive workloads that
is a non-starter.

DistLLM's differentiator (M15 North Star) is **data-sovereign distributed
inference on hardware the customer owns** — the coordinator, backends, and
federated fine-tuning all run on the customer's nodes, and nothing leaves their
trust boundary.

But pure self-hosting has a real UX cost: there is **no managed control plane**.
Customers must run and babysit their own coordinator, build their own billing/
metering, wire up their own observability, and curate their own plugin catalog.
That is exactly the operational burden the managed vendors remove — and the
reason teams accept the sovereignty trade-off.

**The gap:** managed UX *without* giving up compute/data sovereignty.

## 2. Solution — hosted control plane, customer-owned data plane

DistLLM Cloud is a **split-plane** architecture:

- **Control plane (hosted by DistLLM Cloud, managed SaaS):** the coordinator's
  orchestration brain, the scheduling/placement UI, tenant billing, the plugin
  catalog, and cross-fleet observability. This is stateless with respect to
  customer *content* — it only ever sees control metadata and billing meters.
- **Data plane (runs on CUSTOMER hardware or the customer's OWN cloud account /
  VPC):** the model workers, GPU nodes, weights, KV cache, prompts, and
  completions. **None of this leaves the customer's infra.**

The control plane talks to workers over an **authenticated, mTLS-ready channel**
(reusing the existing `distllm.dist.p2p` gossip/transport machinery). A customer
worker *dials out* to the cloud coordinator and registers its capabilities; the
cloud coordinator schedules onto it but never receives prompt/weight payloads.

### What crosses the trust boundary (and what never does)

| Crosses boundary → DistLLM Cloud | NEVER crosses boundary (stays on customer infra) |
| --- | --- |
| Worker registration (node id, capabilities, backend list) | Prompts / user inputs |
| Health & availability heartbeats | Model weights / adapters / fine-tunes |
| Scheduling hints, placement decisions | KV cache / activations |
| Billing meters (token counts, GPU-seconds, cost) | Completions / model outputs |
| Plugin catalog lookups | Training data / gradients |

The billing meters are **aggregate counters** (from E12 metering), not content —
so even the billing path preserves sovereignty.

## 3. Reference architecture

```
        ┌──────────────────────── DistLLM Cloud (HOSTED SaaS) ────────────────────────┐
        │                                                                              │
        │   ┌─────────────┐   ┌──────────────┐   ┌────────────┐   ┌────────────────┐  │
        │   │ Control-    │   │ Scheduling / │   │  Tenant    │   │ Plugin Catalog │  │
        │   │ plane       │   │ Placement UI │   │  Billing   │   │ (marketplace)  │  │
        │   │ Coordinator │   │              │   │  (metering)│   │                │  │
        │   └──────┬──────┘   └──────┬───────┘   └─────┬──────┘   └───────┬────────┘  │
        │          │                 │                 │                  │           │
        │          └────────── control-plane bus (metadata only) ────────┘           │
        │                              │  ▲                                           │
        └──────────────────────────────┼──┼───────────────────────────────────────────┘
                                        │  │   mTLS-ready authenticated channel
             control metadata +         │  │   (reuse dist/p2p transport + gossip)
             billing meters ONLY  ──────▼──┴──────  worker dials OUT, registers caps
        ┌────────────────────────── CUSTOMER TRUST BOUNDARY ──────────────────────────┐
        │  (customer HW or customer's OWN cloud account / VPC — compute stays here)    │
        │                                                                              │
        │   ┌────────────────┐   ┌────────────────┐   ┌────────────────┐              │
        │   │  Worker Agent  │   │  Worker Agent  │   │  Worker Agent  │   ...         │
        │   │ (register_     │   │                │   │                │              │
        │   │  worker())     │   │                │   │                │              │
        │   ├────────────────┤   ├────────────────┤   ├────────────────┤              │
        │   │ BackendRegistry│   │ GPU / model    │   │ Federated      │              │
        │   │ + local model  │   │ workers        │   │ finetuner      │              │
        │   │ weights + KV   │   │ (weights, KV,  │   │ (gradients     │              │
        │   │ (NEVER leave)  │   │  prompts here) │   │  stay local)   │              │
        │   └────────────────┘   └────────────────┘   └────────────────┘              │
        └──────────────────────────────────────────────────────────────────────────────┘
```

### Component mapping to EXISTING modules

The hosted control plane is not a rewrite — it is the existing coordinator brain
lifted into a managed tier. Each control-plane responsibility maps to a module
that already exists in this repo:

| Control-plane responsibility | Existing module | Notes |
| --- | --- | --- |
| Orchestration / node lifecycle | `distllm.core.coordinator` (`Coordinator`, `ClusterManager`) | Runs hosted; schedules onto remote workers instead of local nodes. |
| Worker scheduling / placement | `distllm.backends.registry` (`BackendRegistry`), `distllm.core.placement` (`select_placement`, `NodeTopology`, `LinkInfo`) | Capabilities registered by the worker agent feed placement scoring. |
| Tenant billing | `distllm.core.metering` (`MeteringStore`, `BillingExporter`) — from **E12** | Meters are aggregate counters; Stripe is still an E12 stub. |
| Plugin catalog | `distllm.core.plugin_marketplace`, `distllm.dist.marketplace` | Hosted catalog; settlement is M14-partial. |
| Observability | `distllm.observability.*` (metrics, tracing, exporter) | Aggregate metrics only; no prompt/output content. |
| Transport / auth channel | `distllm.dist.p2p.transport` (`GossipTransport`), `distllm.dist.p2p.gossip` | Already HMAC-signed; mTLS is the productionization path. |
| Federated fine-tuning (data plane) | `distllm.core.federated_finetuner` | Stays entirely customer-side; only meters/health surface to cloud. |

## 4. Sovereignty guarantees & how existing security work protects the data plane

The split-plane model is only credible if the data plane is genuinely hardened.
DistLLM already ships the relevant controls (M-series milestones):

- **M2 — trusted `X-Forwarded-For` handling** (`test_m2_xff_trust`): the worker's
  local API only trusts proxy headers from configured hops, so a hosted
  coordinator cannot spoof client identity into the data plane.
- **M3 — Argon2 API-key hashing** (`test_m3_api_key_argon2`): worker↔control-plane
  auth tokens are stored/verified with Argon2, not plaintext.
- **M4 — strict CORS** (`test_m4_cors`): the worker's local surface does not
  accept arbitrary cross-origin calls.
- **M5 — file-secret permissions** (`test_m5_file_secret_perms`): auth tokens and
  keys on the worker are written with locked-down perms.
- **M6 — prompt-injection confidence scoring** (`test_m6_prompt_injection_*`):
  content defense stays *on the data plane*, where the content actually lives.
- **mTLS-ready channel:** the p2p transport already signs messages with
  HMAC-SHA256; the productionization step is mutual TLS so the control plane and
  worker cryptographically authenticate each other and no third party can inject
  scheduling commands.

Because prompts, weights, KV cache, and gradients **never cross the boundary**,
the customer's compliance posture (data residency, IP protection, sovereignty)
is preserved even while the control plane is a shared multi-tenant SaaS.

## 5. Packaging — how a customer deploys a worker

A customer joins the hosted coordinator by running a worker agent that dials out.
The intended one-liners (the agent itself is the scaffold below):

**Docker:**

```bash
docker run --gpus all \
  -e DISTLLM_CLOUD_URL=https://cloud.distllm.ai \
  -e DISTLLM_CLOUD_TOKEN=$MY_TOKEN \
  distllm/worker-agent:latest
```

**Helm (Kubernetes / customer VPC):**

```bash
helm install distllm-worker distllm/worker \
  --set cloud.url=https://cloud.distllm.ai \
  --set cloud.token=$MY_TOKEN \
  --set gpu.count=8
```

Both entrypoints call `distllm.cloud.worker_agent.register_worker(...)`, which
gathers local capabilities (from `BackendRegistry` + `placement.LinkInfo`) and
registers them with the cloud coordinator over the mTLS-ready channel. The worker
then receives scheduling decisions but keeps all compute and data local.

## 6. Honest gaps — scaffolded vs real

This is a strategy + scaffold deliverable. Being explicit about what is and is
not real:

| Piece | Status |
| --- | --- |
| **Managed control-plane SaaS** | **NOT built.** This doc + the worker-agent scaffold only. No hosted coordinator, UI, or multi-tenant deployment exists. |
| **Worker agent (`distllm.cloud.worker_agent`)** | **SCAFFOLD.** `register_worker()` is callable, gathers real capability shapes from existing types, and posts over an mTLS-ready HTTP channel — but there is no live coordinator to accept it. |
| **Billing / metering** | E12 metering store is real (aggregate counters); Stripe settlement is an **E12 stub** (no real charges). |
| **Plugin marketplace settlement** | **M14-partial** — catalog exists; on-chain/real settlement is incomplete. |
| **mTLS** | Channel is **mTLS-*ready*** (HMAC-signed today via p2p transport); mutual TLS is the productionization step, not shipped here. |
| **Placement / scheduling onto remote workers** | Placement math (`select_placement`) is real and pure; wiring the hosted coordinator to schedule onto *remote* registered workers is **not implemented**. |

### Summary

DistLLM Cloud's wedge is simple: **give teams the managed UX of RunPod/Modal/
Together without moving their compute or data off infra they own.** The control
plane is a thin, metadata-only SaaS built from modules that already exist in this
repo; the data plane stays sovereign. This document and the `worker_agent`
scaffold establish the architecture and the join protocol — not a shipping SaaS.
