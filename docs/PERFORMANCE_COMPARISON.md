# DistLLM Performance Comparison

> Last updated: 2026-11-10. Numbers here come from three places, and the
> distinction matters more than the number: `docs/BENCHMARKS.md` (measured on
> named hardware, with the command recorded), `docs/competitive-analysis.md`
> (third-party market and pricing data, not verified by us), and
> `website/comparisons.html` (published pricing only — its throughput claims
> were removed on 2026-11-10 because they were never measurements).
>
> Anything measured is marked "measured". Anything from a competitor is marked
> "third-party, not verified by us". Anything we have not measured says so
> instead of quoting a number.

---

## 1. Latency / Speed (tok/s or ms/token)

DistLLM's own **latency** (TTFT / ITL) is still pending in `BENCHMARKS.md` (the TTFT
and ITL tables there are filled with `—`). The table below therefore mixes the
measured **throughput** numbers DistLLM does have with published speed/latency
figures for competitors.

| Solution | Metric | Value | Source |
|---|---|---:|---|
| **DistLLM — 1× RTX 4090** (Llama 3.1 8B, FP16) | Throughput | 92.0 tok/s | BENCHMARKS.md (measured) |
| **DistLLM — 2 nodes @ 1 GbE** (Llama 3.1 8B) | Throughput | 153.1 tok/s (1.66×, 83.2% eff.) | BENCHMARKS.md (measured) |
| **DistLLM — consumer GPUs** (qualitative) | Throughput | not published | — (the 30–90 tok/s range was removed from comparisons.html as unmeasured) |
| **DistLLM + vLLM — 3 nodes** (Llama 3.1 70B) | Throughput | not measured | — (was sourced to a comparisons.html figure that was never measurement) |
| **DistLLM + vLLM — 4 nodes** (Llama 3.1 70B) | Throughput | not measured | — (same) |
| **vLLM — 1× A100** (Llama 3.1 70B) | Throughput | 18 tok/s | competitive-analysis.md (third-party, not verified by us) |
| **Groq** (Llama 70B, LPU) | Throughput / TTFT | 300–800 tok/s, <10 ms TTFT | competitive-analysis.md |
| **Fireworks.ai** (Llama 70B) | TTFT | sub-100 ms TTFT | competitive-analysis.md |
| **Petals** (distributed P2P) | Latency | Variable; high (each hop adds network RTT) | competitive-analysis.md |

**Takeaway:** DistLLM's measured speed on consumer hardware (92 tok/s single GPU,
153 tok/s across two 1 GbE nodes) is what this project can defend. Earlier
revisions of this document compared it against "30–90 tok/s on consumer GPUs",
52 tok/s at 3 nodes and 68 tok/s at 4 nodes — all three sourced to
website/comparisons.html and none of them a measurement. Those pages have been
corrected to say so, and the figures are removed here rather than repeated.
DistLLM does **not** compete with Groq's custom-LPU speeds (300–800 tok/s), a gap
its own positioning explicitly avoids competing on (competitive-analysis.md,
"Avoid competing on raw throughput").

---

## 2. Throughput

| Configuration | tok/s | Speedup vs single node | Source |
|---|---:|---:|---|
| DistLLM 1× RTX 4090 (Llama 3.1 8B) | 92.0 | 1.00× | BENCHMARKS.md (measured) |
| DistLLM 2 nodes @ 1 GbE (Llama 3.1 8B) | 153.1 | 1.66× (83.2% eff.) | BENCHMARKS.md (measured) |
| DistLLM + vLLM 3 nodes (Llama 3.1 70B) | not measured | — | removed: sourced to comparisons.html, never a measurement |
| DistLLM + vLLM 4 nodes (Llama 3.1 70B) | not measured | — | removed: same |
| vLLM 1× A100 (Llama 3.1 70B) | 18 | 1.0× | competitive-analysis.md (third-party, not verified by us) |

> Note: DistLLM's 4-node and InfiniBand results, and the 70B FP16 single/2-/4-node
> results, are marked "pending" in BENCHMARKS.md. The scaling numbers above for
> 70B come from the published website figure (illustrative, not a BENCHMARKS.md run).

---

## 3. Cost

DistLLM's core cost advantage is **zero per-token cost on owned hardware**
(electricity only). The table contrasts self-hosted DistLLM against cloud/serverless
providers.

### Per 1M tokens (or equivalent)

| Option | Cost per 1M tokens | Source |
|---|---:|---|
| **DistLLM (owned hardware)** | ~$0.01 (electricity) | comparisons.html |
| **RunPod** (RTX 4090, ~$0.44/hr) | ~$0.30–1.50 | comparisons.html; competitive-analysis.md |
| **Modal** (H100) | ~$0.50–2.00 | comparisons.html |
| **Together.ai** (Llama 70B) | ~$0.88 | competitive-analysis.md |
| **Fireworks.ai** (Llama 70B) | ~$0.90 | competitive-analysis.md |
| **Groq** (Llama 70B) | ~$0.59–0.79 | competitive-analysis.md |
| **Cerebras** (Llama 70B) | ~$0.60 | competitive-analysis.md |

### Sustained monthly cost (70B model, continuous workload)

| Option | Monthly Cost | Data Sovereignty | Source |
|---|---:|---|---|
| **DistLLM (owned hardware)** | $0 (electricity only) | Full | competitive-analysis.md |
| **DistLLM (2× RTX 4090)** | ~$0 (one-time $3,200) | Full | competitive-analysis.md |
| DistLLM self-host (70B 24/7) | ~$50/mo (electricity) | Full | comparisons.html |
| RunPod (70B 24/7) | ~$450/mo | None | comparisons.html |
| Modal (H100 dedicated) | ~$2,870–8,532 | None | competitive-analysis.md |
| Lambda (H100 on-demand) | ~$2,880 | None | competitive-analysis.md |
| Together.ai (serverless) | ~$600–2,000+ | None | competitive-analysis.md |
| Fireworks.ai (serverless) | ~$650–2,200+ | None | competitive-analysis.md |
| AWS Bedrock (Llama 70B) | ~$800–3,000+ | None | competitive-analysis.md |
| Azure AI (GPT-4o equiv) | ~$1,500–5,000+ | None | competitive-analysis.md |

> Break-even: competitive-analysis.md notes RTX 4090 self-hosting breaks even
> "in weeks" vs cloud, and Modal's 24/7 H100 cost breaks even vs self-host in
> "3–6 months." Any workload running 4+ hours/day saves money self-hosting within
> months (cloud adds 2–10× margins).

---

## 4. When to Choose DistLLM

DistLLM occupies a unique axis — **heterogeneous multi-device aggregation on
consumer hardware with OpenAI-compatible serving** — where it has zero direct
competition (competitive-analysis.md, "Unique Position"). Choose DistLLM when:

- **You own multiple consumer GPUs/devices** (RTX 3060–4090, AMD, Apple Silicon,
  CPU) across machines and want to pool them into one model no single device can
  run. *No other production tool does this* — Petals tries but is research-grade
  and unreliable.
- **Data sovereignty matters** — healthcare, finance, government, legal, or
  privacy-conscious workloads where data cannot leave your premises. Every cloud
  provider requires data to transit their servers.
- **You run sustained or 24/7 workloads** and want to avoid per-token cloud
  margins (2–10×). Self-hosting breaks even within weeks–months.
- **You need internet/WAN-capable inference** with auto-discovery (mDNS) and
  node recovery across LAN, WiFi, and internet.
- **You want an OpenAI-compatible API** on your own hardware with zero per-token
  cost.

**Do NOT choose DistLLM when:**

- You need maximum raw throughput / lowest TTFT on datacenter hardware — vLLM,
  SGLang, Groq, or Cerebras win there (competitive-analysis.md explicitly says to
  avoid competing on raw throughput).
- You want the widest model catalog with zero setup — cloud providers offer 100+
  models; Ollama wins on single-device simplicity.
- You have no GPUs and want to borrow others' — Petals' public swarm fits that.

**Recommended complement, not competitor:** Run DistLLM + vLLM — vLLM per-node,
DistLLM cross-node. The throughput this reaches on 70B across 3–4 machines is
not measured; earlier revisions quoted 52–68 tok/s from comparisons.html, which
was never a measurement and has been removed from both places.

---

## Sources

- `docs/BENCHMARKS.md` — measured DistLLM throughput (92.0 / 153.1 tok/s), pending TTFT/ITL.
- `docs/competitive-analysis.md` — market positioning, pricing tables, per-token cloud rates (Groq, Together, Fireworks, Cerebras), break-even analysis.
- `website/comparisons.html` — published **pricing** only (cost-per-1M-token, $50/mo self-host). Its throughput figures (30–90, 18/52/68 tok/s) were removed on 2026-11-10 as unmeasured and must not be cited from here.
