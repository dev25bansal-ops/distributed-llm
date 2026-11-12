# Startup Buildathon — DistLLM application

Deadline **13 November 2026**. Drafted 2026-11-12 against the real repository
state (branch `worktree-bubbly-pondering-whisper`, the 0.7.0 release candidate).

Every technical claim below is traceable to a file in this repository. Where a
fact is not yet true — chiefly Google Cloud deployment — it says "planned" and
the Google Cloud answer names the work as work. Do not upgrade those to
present tense; the jury includes engineers and the product's whole position is
that it can be verified.

---

## 1. One-sentence summary

DistLLM pools the GPUs you already own — across a LAN, a home, an office, or a
Kubernetes cluster — into one inference cluster, so a model too large for any
single machine (Llama 3.1 70B) runs anyway, on hardware you have already paid
for.

## 2. Main problem

Running a modern 70B model needs roughly 140 GB of GPU memory in FP16 — four
A100s, or a cloud bill in the thousands per month — while the machines most
developers and small teams already own sit at 8–24 GB each and idle. Buying or
renting one big GPU is the only answer the market offers, so teams either pay
for capacity that is idle most of the day or send their data to a third party's
API.

## 3. Typical user

Three, in order of who is hurting most:

1. **The small team with consumer/prosumer GPUs.** A startup or research group
   with three or four RTX 4090/3090-class machines that cannot individually run
   the model they want. This is the README's own scenario and the design centre.
2. **The privacy-bound operator.** Healthcare, legal, or internal-tooling teams
   who cannot send prompts to a hosted API and have no budget for an A100
   cluster.
3. **The platform engineer.** Teams that already run Kubernetes with a
   heterogeneous GPU pool and want one OpenAI-compatible endpoint in front of
   it rather than per-machine setups.

## 4. Solution

DistLLM splits a language model into layer ranges and assigns each range to a
different machine, using **pipeline parallelism**: tokens flow through the
cluster stage by stage, and every device only ever holds a fraction of the
weights. On top of that layer:

- **One command to start, one to join.** `distllm cluster start` on the primary
  machine, `distllm cluster join` everywhere else — with mDNS auto-discovery,
  so there is no config file to hand-edit.
- **Automatic partitioning and rebalancing.** A cost model decides where the
  layer boundaries fall given each node's memory and interconnect.
- **An OpenAI-compatible API.** Clients point at it with an API key and speak
  `/v1/chat/completions`, streaming included, with no code change.
- **It survives losing a node.** Nodes can be drained, taken offline, and
  deregistered from the pool without restarting the cluster.
- **Optional compression.** Quantization, KV-cache FP8, and speculative
  decoding to fit larger models into the pool you have.

**Measured, not asserted:** 92.0 tok/s on a single RTX 4090 (Llama 3.1 8B,
FP16), and **153.1 tok/s across two nodes on 1 GbE — 1.66× single-GPU
throughput at 83.2% scaling efficiency** (`docs/BENCHMARKS.md`, table at
"1 GPU (RTX 4090) / 2 nodes (1 GbE)", measured 2026-06-26;
`docs/PERFORMANCE_COMPARISON.md` cites the same figures). The comparison
document deliberately removed the older "30–90 tok/s" marketing range as
unmeasured; the numbers that remain are ones we can defend and re-run.

## 5. Stage

**Minimal Viable Product.**

Evidence, so the answer is checkable rather than a claim:

- Published on PyPI as `distributed-llm` (0.6.2.2 live; **0.7.0 is built,
  every test suite green, and awaiting the release tag**).
- A full OpenAI-compatible server that runs a real model on real hardware — the
  0.7.0 release candidate was exercised end-to-end on an NVIDIA A10: install,
  `selftest` 7/7, `/v1/chat/completions` 200, streaming with first token at
  0.77 s, a worker joining the pool, and a node leaving it cleanly.
- 13,600+ automated tests across the API, core engine, and distributed layers,
  all passing on the release candidate.

**What makes it MVP rather than "fully launched":** it is not yet multi-tenant
at scale, has no self-serve onboarding or billing, and — the honest gap — **runs
on Kubernetes/Helm/Terraform today but not yet on Google Cloud specifically.**

## 6. Business model

Open-core, three layers:

1. **Free and open source (Apache 2.0).** The engine and CLI, which is what
   drives adoption and credibility.
2. **Paid self-hosted "Team" licence.** SSO, audit, quota and cost governance,
   and support SLAs for organisations that must keep inference on their own
   hardware — the privacy-bound segment above. Sold per cluster per year.
3. **Managed control plane (later).** We run the coordination layer — model
   distribution, cluster health, upgrades — while the customer's GPUs do the
   work and their data never leaves their network.

The wedge is the free tier: teams adopt it because it turns hardware they
already own into a working endpoint, then pay when they need the compliance and
support features that a production deployment requires.

## 7. Competitors, and the difference

| | What they do | Where DistLLM differs |
|---|---|---|
| **vLLM / SGLang / TensorRT-LLM** | Excellent single-node inference engines; they assume the GPUs are in one machine | DistLLM's problem is the GPUs are in *different* machines. It is complementary — a vLLM-style engine can back a node. |
| **Ray / Ray Serve** | General distributed compute | A framework you build a system with, not an inference product you install. No model partitioning, no OpenAI API, no node lifecycle. |
| **Petals** | Closest in spirit — distributed inference over the internet | Research-grade and largely dormant; not something a team deploys and supports. DistLLM ships packaging, a CLI, a stable API, and a release process. |
| **Cloud APIs (OpenAI, Bedrock, Vertex)** | Zero setup, someone else's GPUs | Data leaves your network and the bill is per token forever. DistLLM's whole premise is the opposite trade. |
| **Buying an A100** | One machine that can do it | Capex and idle capacity. DistLLM uses capacity the customer already owns. |

The defensible difference is not "faster inference" — it is **the deployment
model**: verifiable, open-source, data-never-leaves, on heterogeneous hardware
that already exists.

## 8. Target countries

- **Primary: India.** The Buildathon's home market, a very large population of
  developers and small studios with consumer GPUs, and GPU cloud pricing that
  makes self-hosting attractive.
- **Secondary: United States and the EU.** Specifically regulated and
  privacy-bound teams (health, legal, financial tooling) where sending prompts
  to a third-party API is a compliance problem rather than a preference.

## 9. How AI is used

Two distinct answers, and it is worth separating them:

**AI is the product.** DistLLM runs large language models — that is the whole
of it. Internally it also applies AI to its own scheduling problem: a learned
routing/partitioning layer picks placement and model selection from observed
latency and reward signals, and speculative decoding uses a small draft model to
propose tokens that a larger model verifies.

**AI is in how it is built.** The 0.7.0 hardening pass — the six defects fixed
in the last two days, the test-suite analysis, the release gates — was carried
out with AI agents doing the reading, the test triage, and the verification
against real hardware. That is not a marketing line; it is why a
one-person-scale effort produced 13,600 passing tests and a reproducible
hardware validation in the time it did.

## 10. Google Cloud deployment and scaling

**Being direct: this is the least-built part of the answer, and the plan is the
substance.**

**Today.** DistLLM deploys on Kubernetes — Helm chart, Kustomize overlays, a
custom resource definition and operator, Terraform modules, and an AWS CDK
integration. It is cloud-portable by construction: anything that can run a
container with a GPU can run a node.

**On Google Cloud, in this order:**

1. **GKE first.** The existing Helm chart and operator are the deployment
   mechanism; the work is a GKE-specific overlay — GPU node pools, the NVIDIA
   device plugin, and node-pool autoscaling — not a rewrite. Coordination
   becomes a Deployment; worker nodes become a GPU node pool joining the
   cluster.
2. **GCE with L4/A100 for the single-machine-to-pool upgrade path.** A customer
   with two GCE GPU instances gets a cluster in two commands, which is the
   fastest demonstration that the product works on Cloud.
3. **Cloud Storage for model distribution.** Model shards are large and are
   fetched once per node; object storage is the right place for them, and it
   makes cold-start predictable.
4. **Vertex AI Model Garden as the model source,** so a customer picks a model
   there and DistLLM partitions it across their pool.
5. **Cloud Monitoring and Cloud Logging** for cluster telemetry — the
   observability layer already emits Prometheus metrics and OpenTelemetry
   traces, so this is an exporter, not new instrumentation.

**Why it scales on Cloud specifically.** DistLLM's unit of scale is the *node*,
not the machine, and GKE's node-pool autoscaler matches that exactly: demand
adds nodes to the pool, DistLLM rebalances layer ranges, and throughput grows
sub-linearly in cost because each added node only has to hold its own shard. The
measured 1.66× across two nodes at 83.2% efficiency is the number that makes
this a real scaling story rather than an aspiration — and the 0.7.0 release
added the node lifecycle (drain, offline, deregister) that autoscaling requires.

**What the $5,000 in Cloud credits would fund:** a GKE cluster with two
L4 GPU nodes to run the multi-node benchmark on Cloud hardware and publish the
numbers, plus the Vertex and Cloud Storage integration work above.

---

## Claims to double-check before submitting

- [ ] Confirm the PyPI version live on the day of submission (0.6.2.2 today;
      0.7.0 is built but **not yet tagged**).
- [ ] The 92.0 / 153.1 tok/s figures come from `BENCHMARKS.md`; confirm they
      were measured on the hardware the file states.
- [ ] Google Cloud numbers are **planned**, not measured. Nothing above claims
      otherwise; keep it that way.
