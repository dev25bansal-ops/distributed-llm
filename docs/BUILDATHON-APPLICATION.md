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
something that can run a model none of them could run alone. The important
property is that memory, not compute, is what usually stops a team: a 70B model
in FP16 needs roughly 140 GB of weights, and once those weights are divided
across four 24 GB cards the model fits, each card holding only the layers it
owns. Decoding then happens as a relay. The first machine runs its layers, hands
the resulting activations to the second, and so on down the chain, so the
per-token cost on any one device is the cost of its own slice plus the transfer.
On a slow link the transfer dominates and the whole exercise is pointless, which
is why the partitioning decision has to be made against the actual hardware
rather than a rule of thumb. DistLLM measures each machine's GPU model, memory,
compute throughput and the link between machines, and then solves for the layer
boundaries that minimise end-to-end latency. The solver is a dynamic program
over the number of layers and the number of stages rather than a greedy split,
so it accounts for the fact that layers are not equally expensive — attention
and MLP layers differ in both weight size and compute — and it models the
pipeline's own inefficiency: with N stages the first and last stages sit idle
during warm-up and cool-down, giving a bubble of (N−1) stage-times that is
amortised across micro-batches, so the model prefers fewer, larger stages on a
slow interconnect and more, smaller stages when the network is fast. That
single decision is the difference between a cluster that beats one machine and
one that is slower than the best card in it.

On top of the partitioning we put the things that make it usable rather than a
research demo. Starting a cluster is one command and joining it is another, with
mDNS discovery so the other machines find the coordinator on the LAN without
anyone editing a config file or tracking down an IP address. Every node is
enrolled with an API key, the coordinator mints one on first start and writes it
where the CLI reads it automatically, so the join actually completes rather than
silently failing. The server exposes an OpenAI-compatible API — the same
`/v1/chat/completions` shape and the same streaming semantics — so an existing
application moves over by changing a base URL and adding a key, with no client
library rewrite. Around that sit the operational pieces a real deployment
needs: a node lifecycle in which machines can be drained so in-flight requests
finish, taken offline, and deregistered from the pool entirely without
restarting the cluster or dropping the endpoint; health checks and circuit
breaking so a sick node stops receiving work; and optional compression —
quantization, FP8 KV cache and speculative decoding — to fit a larger model into
the pool that already exists.

The compression layer deserves a word because it is what makes the difference
between fitting and comfortably fitting. Quantization reduces the weight
precision so the same physical GPUs hold more parameters, and the pipeline
planner treats that as a per-layer choice rather than one global setting: a
layer that tolerates 8-bit weights and one that needs 16-bit can sit on the same
node, and the mixed-precision tuner decides that per layer against measured
quality rather than uniformly. The KV cache is quantized to FP8 separately,
because during long-context decoding the cache rather than the weights becomes
the memory ceiling, and a cache that grows with every token will exhaust a card
that comfortably holds the model. Speculative decoding attacks latency from the
other direction: a small draft model proposes several tokens cheaply, the larger
model verifies them in a single forward pass, and the accepted prefix is kept,
so the expensive model runs fewer times for the same output. Each of these is
optional and each is measurable, which matters because the honest answer to
"will this fit and how fast" depends on the specific model, the specific cards
and the specific link — so DistLLM measures the hardware it is given and reports
what it actually achieves rather than quoting a headline figure that assumes a
configuration the customer does not have.

Measured on real hardware rather than asserted, this gives 92.0 tokens per
second on a single RTX 4090 running Llama 3.1 8B in FP16, and 153.1 tokens per
second across two nodes on a 1 GbE link — 1.66 times the single-GPU throughput
at 83.2% scaling efficiency. Those two numbers together are the whole argument
in miniature: adding a second ordinary machine made the system half again as
fast rather than marginally faster, and it did so over consumer Ethernet, which
is the worst realistic interconnect; on 10 GbE or NVLink the same arrangement
loses far less to transfer and scales further. Just as importantly, the system
degrades honestly. Losing a node in the middle of generation does not corrupt
output or hang the request; the pipeline detects the failure, the remaining
nodes are re-partitioned, and the endpoint keeps answering, which is the
behaviour a team running this on hardware in an office actually needs, because
that hardware gets rebooted, unplugged and lent to a colleague. The figures are
recorded in the repository with their measurement date and hardware, and an
earlier set of marketing numbers was deliberately deleted from the comparison
document because it could not be reproduced. The two figures above are reported
measurements whose raw logs live outside the repository rather than CI
artifacts, and the Benchmarks document says so at the table itself — the
first thing published when the Cloud credits land is the Cloud-measured
scaling curve that re-earns these numbers on a platform anyone can rent. Taken together, what the system offers is not a faster
way to run a single GPU, but a way to make a decision that was previously
impossible: whether a given model runs at all on hardware a team already owns,
and if so how fast, measured before they commit to buying anything.

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
chart, Kustomize overlays, a custom resource definition and a matching operator,
Terraform modules and an AWS CDK integration, so it is cloud-portable by
construction — anything that can run a container with a GPU can run a node,
because a node is a process with an attached GPU that announces itself to the
coordinator and is then handed a layer range. Nothing in that contract is
specific to a cloud provider, which is what makes the Google Cloud work a matter
of integration rather than re-architecture.

On Google Cloud the order of work is GKE first, because it requires the least
new code and delivers the most. The existing Helm chart and operator are already
the deployment mechanism, so what is needed is a GKE-specific overlay: GPU node
pools with the right machine types, the NVIDIA device plugin and driver
installation, and node-pool autoscaling wired to the coordinator's health
endpoint so the pool grows and shrinks against real demand rather than CPU
utilisation, which is meaningless for an inference workload. The coordinator
becomes a Deployment with the existing pod disruption budget; the workers become
a GPU node pool whose pods join the cluster and announce themselves, at which
point the cost model re-solves the layer boundaries against the new set of
machines. Second comes GCE with L4 and A100 instances, which gives a customer
with two GPU instances a working cluster in two commands and is the fastest
possible demonstration that the product runs on Cloud at all; this is also the
cheapest way for us to produce Cloud-measured benchmark numbers instead of
extrapolating from consumer hardware. Third, Cloud Storage for model shard
distribution: model shards are large and are fetched once per node, which makes
object storage the correct place for them and makes cold start predictable
rather than dependent on how fast Hugging Face is answering that day. Fourth,
Vertex AI Model Garden as the model source, so a customer selects a model there
and DistLLM partitions that model across their pool, which removes the step
where a team has to know which repository to pull and which revisions are safe.
Fifth, Cloud Monitoring and Cloud Logging fed by the Prometheus metrics and
OpenTelemetry traces the observability layer already emits — the work there is
an exporter and a set of dashboards, not new instrumentation.

It scales on Cloud specifically because DistLLM's unit of scale is the node
rather than the machine, which is exactly the unit GKE's node-pool autoscaler
manages. Under load, demand adds nodes to the pool; each arriving node announces
itself and is given a layer range; the pipeline rebalances and throughput rises.
Cost rises sub-linearly because each added node only ever carries its own shard
— the marginal node does not need to hold the whole model, which is the property
that makes the economics work at all. The measured 1.66 times across two nodes
at 83.2% efficiency is what turns that from an aspiration into a scaling story
with a number behind it, and the 0.7.0 release added the node lifecycle — drain,
offline, deregister — that autoscaling fundamentally depends on, because a pool
that can only grow and never shrink is a bill, not an autoscaler. The $5,000 in
Cloud credits would fund a two-node GKE cluster on L4 GPUs to run the multi-node
benchmark on Cloud hardware and publish those numbers, alongside the Vertex
Model Garden and Cloud Storage integration work described above, which together
convert the plan in this answer into measured results on the platform the
Buildathon is built around.

It is worth being concrete about effort, because a plan without numbers is a
wish. The GKE overlay is the smallest piece of work — the chart, the CRD and the
operator exist, so the overlay is GPU node-pool definitions, the NVIDIA device
plugin, the autoscaler policy and a worked example, which is days rather than
weeks. GCE is smaller still: the container image and the two commands already
work, so it is a startup script and a documented machine-type table. Cloud
Storage shard distribution is a new backend behind an interface the model loader
already has, since it already supports fetching from a remote origin. Vertex
Model Garden is the largest single piece because it is an integration with
someone else's catalogue and its access controls, and it is also the one we
would defer until the first three are proven. Cloud Monitoring is an exporter
configuration against metrics that are already emitted in Prometheus format, so
it is genuinely small. None of these require changing how the pipeline works,
which is the point: the portability was designed in, and what remains is mapping
that design onto one provider's primitives.

There is also a reason to believe the Cloud version will be easier to operate
than the consumer version rather than merely equivalent. A GKE node pool is
homogeneous and its network is predictable, so the cost model has less variance
to reason about than across a home LAN where one machine is on Wi-Fi and another
on powerline Ethernet. Cloud also allows testing failure deliberately — delete a
node and watch the pool replace it — at a cost of minutes rather than a trip to
someone's desk, which is how the drain, offline and deregister paths added in
0.7.0 get exercised continuously rather than manually. Continuous exercise of
failure paths is what makes a distributed system trustworthy in front of a
customer.

The first milestone the credits would buy is a published Cloud-measured scaling
curve — one node, two, four, on L4 hardware over GKE's network — replacing the
1.66-times figure from consumer Ethernet with a number from the platform
customers will actually run on. The second is a reproducible walkthrough in
which a reader creates a GKE cluster, installs DistLLM with Helm, and gets an
OpenAI-compatible endpoint serving a model too large for any single node in that
pool, in under an hour and without reading source code.

---

## Before submitting

Confirm the PyPI version live that day (0.6.2.2 today; 0.7.0 is built but not
yet tagged). The 92.0 and 153.1 tok/s figures are recorded in
`docs/BENCHMARKS.md` as measurements dated 2026-06-26 on an RTX 4090 over
1 GbE, with an explicit provenance note at the table: the raw logs live
outside the repository, and the 92.0 value is also the estimator's input
constant, so it is reported, not independently reproduced. Everything about
Google Cloud above is stated as plan, not measurement — keep it that way.
