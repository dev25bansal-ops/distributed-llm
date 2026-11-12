# Startup Buildathon — DistLLM application (paragraph form)

Deadline **13 November 2026**. Every answer below is a paragraph you can paste
straight into the form field.

Each technical claim is traceable to this repository. Google Cloud is the one
area that is planned rather than built, and the answer says so — the jury has
engineers on it, and the product's whole position is that its claims can be
checked.

---

## One sentence summary

DistLLM pools the GPUs you already own — across a LAN, a home, an office, or a
Kubernetes cluster — into a single inference cluster, so a language model too
large for any one machine runs anyway, on hardware you have already paid for.

---

## What is the main problem you are solving?

Running a modern 70B-parameter model needs roughly 140 GB of GPU memory in FP16,
which means four A100s or a cloud bill in the thousands per month, while the
machines most developers and small teams already own sit at 8 to 24 GB each and
spend most of the day idle. The market offers exactly one answer to this: buy
or rent a bigger GPU, or send your prompts to someone else's API. So teams
either pay for capacity they use a fraction of the time, or they hand their data
to a third party and accept a per-token bill that grows forever. Meanwhile the
total GPU memory already sitting on desks in any small office is usually more
than enough to run the model they actually want — it is just spread across
machines that cannot see each other.

---

## Who would be a typical user?

The first is a small team or research group with three or four consumer or
prosumer GPUs — RTX 4090s and 3090s — who cannot individually run the model they
need but collectively have the memory to. This is the scenario the product was
designed around. The second is a privacy-bound operator: a healthcare, legal, or
internal-tooling team that cannot send prompts to a hosted API for compliance
reasons and has no budget for an A100 cluster. The third is a platform engineer
who already runs Kubernetes with a mixed GPU pool and wants one
OpenAI-compatible endpoint in front of all of it, rather than a separate setup
per machine.

---

## What is your solution to this problem?

DistLLM splits a language model into consecutive layer ranges and assigns each
range to a different machine, so tokens flow through the cluster stage by stage
and no device ever holds more than a fraction of the weights. That technique —
pipeline parallelism — is what turns a collection of ordinary machines into
something that can run a model none of them could run alone. On top of it we put
the things that make it usable rather than a research demo: one command to start
a cluster and one to join, with mDNS discovery so there is no config file to
hand-edit; a cost model that decides automatically where the layer boundaries
should fall given each machine's memory and interconnect; an OpenAI-compatible
API with streaming, so existing clients work unchanged with just an API key; and
a node lifecycle that lets machines be drained, taken offline, and deregistered
without restarting the cluster. Measured on real hardware, this gives 92.0
tokens per second on a single RTX 4090 running Llama 3.1 8B in FP16, and 153.1
tokens per second across two nodes on a 1 GbE link — 1.66 times the single-GPU
throughput at 83.2% scaling efficiency.

---

## What is the stage of your product?

Minimal Viable Product. The engine is published on PyPI as `distributed-llm`,
with 0.6.2.2 live today and 0.7.0 built, fully tested, and awaiting its release
tag. The server is a working OpenAI-compatible API rather than a prototype: on
the 0.7.0 release candidate we installed it on an NVIDIA A10 and exercised it
end to end — self-test passing all seven checks, a 200 on
`/v1/chat/completions` against a real model, streaming with the first token at
0.77 seconds, a worker joining the pool, and a node leaving it cleanly. The
codebase carries more than 13,600 automated tests across the API, core engine,
and distributed layers, all passing on the release candidate. What keeps it at
MVP rather than fully launched is that it is not yet multi-tenant at scale, has
no self-serve onboarding or billing, and — the honest gap — runs on Kubernetes,
Helm, Terraform and AWS CDK today but not yet on Google Cloud specifically.

---

## What is the potential business model?

The model is open-core, in three layers. The engine and CLI are Apache 2.0 and
free, because adoption and credibility both depend on people being able to read
and verify what runs on their own machines. On top of that sits a paid
self-hosted Team licence carrying the things a production deployment actually
needs — single sign-on, audit logging, per-tenant quotas, cost governance, and a
support SLA — sold per cluster per year, aimed squarely at the privacy-bound
organisations that must keep inference inside their own network. Later, a
managed control plane lets us run the coordination layer, model distribution,
cluster health and upgrades, while the customer's GPUs do the work and their
data never leaves their premises. The wedge is the free tier: a team adopts it
because it turns hardware they already own into a working endpoint, and starts
paying when they need the compliance and support that make it production-grade.

---

## Who are your primary competitors and how are you different?

The closest technical competitors are single-node inference engines — vLLM,
SGLang, TensorRT-LLM — and they are excellent, but they assume the GPUs are
inside one machine, which is precisely the assumption our customer cannot
satisfy; they are complementary rather than rival, and one can back a DistLLM
node. Ray and Ray Serve are general distributed-compute frameworks you build a
system with, not an inference product you install, so they offer no model
partitioning, no OpenAI API, and no node lifecycle. Petals is nearest in spirit,
distributed inference across machines, but it is research-grade and largely
dormant, not something a team deploys and supports. Cloud APIs such as OpenAI,
Bedrock and Vertex remove all setup by sending your data away and charging per
token forever, which is the exact trade our users are trying to escape; and
buying an A100 answers the memory problem with capex and permanent idle
capacity. Our defensible difference is not raw speed — it is the deployment
model: open-source, verifiable, data-never-leaves, running on heterogeneous
hardware that already exists.

---

## What will be your primary and secondary target countries?

India is the primary market. It is the Buildathon's home country, it has an
enormously large population of developers and small studios who own consumer
GPUs, and its GPU cloud pricing makes self-hosting an attractive alternative
rather than a hobbyist's compromise. The United States and the European Union
are the secondary markets, and specifically the regulated and privacy-bound
teams inside them — healthcare, legal, and financial tooling — where sending
prompts to a third-party API is a compliance problem rather than a matter of
preference.

---

## How are you planning to use AI for this solution?

There are two distinct answers and they deserve separating. AI is the product:
DistLLM exists to run large language models, and it also applies AI to its own
scheduling problem, using a learned routing layer that picks model placement
from observed latency and reward signals, and speculative decoding in which a
small draft model proposes tokens that a larger model then verifies. Separately,
AI is how the product is built. The 0.7.0 hardening pass — the six defects
fixed in the final two days before this application, the triage of three
test failures that only appeared under full-suite load, and the verification
against real GPU hardware — was carried out with AI agents doing the reading,
the testing, and the checking. That is not a marketing line; it is the reason a
one-person-scale effort produced more than 13,600 passing tests and a
reproducible hardware validation in the time it did.

---

## How do you intend to deploy and scale on Google Cloud?

I want to be direct that this is the least-built part of the answer, and that
the plan is the substance. DistLLM deploys on Kubernetes today, with a Helm
chart, Kustomize overlays, a custom resource definition and operator, Terraform
modules and an AWS CDK integration, so it is cloud-portable by construction —
anything that can run a container with a GPU can run a node. On Google Cloud
the order of work is GKE first: the existing Helm chart and operator are the
deployment mechanism, so what is needed is a GKE-specific overlay covering GPU
node pools, the NVIDIA device plugin and node-pool autoscaling, with the
coordinator as a Deployment and the workers as a GPU node pool joining the
cluster. Second comes GCE with L4 and A100 instances, which gives a customer
with two GPU instances a working cluster in two commands and is the fastest
demonstration that the product works on Cloud. Third, Cloud Storage for model
shard distribution, since shards are large and fetched once per node, which also
makes cold start predictable. Fourth, Vertex AI Model Garden as the model
source, so a customer selects a model there and DistLLM partitions it across
their pool. Fifth, Cloud Monitoring and Cloud Logging fed by the Prometheus
metrics and OpenTelemetry traces the observability layer already emits, which is
an exporter rather than new instrumentation. It scales on Cloud specifically
because DistLLM's unit of scale is the node rather than the machine, which is
exactly what GKE's node-pool autoscaler manages: demand adds nodes to the pool,
DistLLM rebalances layer ranges, and throughput grows while cost grows
sub-linearly because each new node only carries its own shard. The measured 1.66
times across two nodes at 83.2% efficiency is what makes that a real scaling
story rather than an aspiration, and the 0.7.0 release added the node lifecycle
— drain, offline, deregister — that autoscaling depends on. The $5,000 in Cloud
credits would fund a two-node GKE cluster on L4 GPUs to run the multi-node
benchmark on Cloud hardware and publish the numbers, alongside the Vertex and
Cloud Storage integration work.

---

## Before submitting

Confirm the PyPI version live that day (0.6.2.2 today; 0.7.0 is built but not
yet tagged). The 92.0 and 153.1 tok/s figures are recorded in
`docs/BENCHMARKS.md`, measured 2026-06-26 on an RTX 4090 over 1 GbE. Everything
about Google Cloud above is stated as plan, not measurement — keep it that way.
