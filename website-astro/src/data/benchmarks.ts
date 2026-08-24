/**
 * Benchmark data for the public benchmarks page.
 *
 * SOURCE OF TRUTH — every measured number below is copied verbatim from one of:
 *   - docs/benchmark-results.md   (measured run of 2026-08-24)
 *   - benchmarks/results/*-cuda.json and *-cpu.json (raw suite output)
 *
 * Rig: a single laptop (Intel Core i7-14700HX + RTX 5060 Laptop GPU, 8 GB),
 * model roneneldan/TinyStories-1M. Nothing on this page is multi-node.
 *
 * Numbers that could NOT be measured on that rig are never rendered as
 * measurements. Aspirational content lives in DESIGN_TARGETS / SUITE_TARGETS
 * and is flagged with IS_ILLUSTRATIVE so pages must render a visible badge.
 */

/** Flag for rows/sections that describe goals or suite thresholds, not measurements. */
export const IS_ILLUSTRATIVE = true;

export const MEASURED_DATE = "2026-08-24";
export const MODEL_NAME = "roneneldan/TinyStories-1M";
export const RIG_SUMMARY =
  "Single laptop — Intel Core i7-14700HX, RTX 5060 Laptop GPU (8 GB GDDR7), Windows 11, PyTorch 2.13.0+cu130. fp16 on CUDA, fp32 on CPU.";
export const RESULTS_DOC_URL =
  "https://github.com/distributed-llm/distributed-llm/blob/master/docs/benchmark-results.md";

export interface MeasuredBar {
  device: string;
  dtype: string;
  /** Numeric value in this chart's unit — all rows in a chart share one unit. */
  value: number;
  /** Verbatim rendering of the value, including its unit. */
  display: string;
}

/**
 * Throughput — built-in suite (benchmarks/run.py): HF generate() baseline,
 * batched prefill+decode, 8 prompts left-padded, 50 new tokens each.
 * Aggregate decoded tokens per second of wall clock.
 */
export const THROUGHPUT: MeasuredBar[] = [
  { device: "CUDA — RTX 5060 Laptop", dtype: "fp16", value: 1200.7, display: "1200.7 tok/s" },
  { device: "CPU — Core i7-14700HX", dtype: "fp32", value: 652.4, display: "652.4 tok/s" },
];

/**
 * Time to first token (median over 8 prompts, ~5-token prompt). All values are
 * milliseconds — do not mix units here; bar length is linear in `value`.
 */
export const TTFT: MeasuredBar[] = [
  { device: "CPU — Core i7-14700HX", dtype: "fp32", value: 5.2, display: "5.2 ms" },
  { device: "CUDA — RTX 5060 Laptop", dtype: "fp16", value: 8.6, display: "8.6 ms" },
];

export interface LatencyRow {
  metric: string;
  cuda: string;
  cpu: string;
}

/** Median latency over 8 prompts (ranges span repeat runs). */
export const LATENCY_ROWS: LatencyRow[] = [
  { metric: "TTFT — prefill + first token (~5-token prompt)", cuda: "8.6 ms", cpu: "5.2 ms" },
  { metric: "ITL — inter-token latency", cuda: "6.9–7.1 ms", cpu: "4.3–4.7 ms" },
];

/** Observed run-to-run spread on the CUDA throughput number. */
export const THROUGHPUT_VARIANCE =
  "Run-to-run variance observed on CUDA: first run 1125.3 tok/s, second 1200.7 tok/s (~7%).";

/** Why CPU beats GPU on latency at this model size. */
export const LATENCY_CAVEAT =
  "CPU beats GPU on latency here because the model is tiny (3.7M params): kernel-launch and sync overhead dominate on GPU at this scale.";

export interface SpecAcceptRow {
  method: string;
  acceptance: string;
}

/**
 * Speculative-decoding verifier acceptance, measured by the built-in
 * spec-accept-rate benchmark. This exercises the real SpeculativeDecoder
 * verify_and_accept logic against synthetic logits — it is NOT a real-model
 * acceptance measurement; real-model rates depend on draft/target match.
 */
export const SPEC_ACCEPTANCE = {
  caveat:
    "Measured with synthetic logits (8 samples, noisy draft) against the real SpeculativeDecoder.verify_and_accept logic — not real-model acceptance. Real-model accept rates depend on draft/target match quality and are not yet measured.",
  rows: [
    { method: "n-gram style verification", acceptance: "47.5%" },
    { method: "EAGLE-style verification", acceptance: "78%" },
  ] as SpecAcceptRow[],
};

export interface EngineMetricRow {
  metric: string;
  cuda: string;
  cpu: string;
}

/**
 * DistLLM engine path (benchmarks/engine_bench.py): InferenceEngine end-to-end
 * — ModelPartitioner load, TokenGenerator sampling, prompt-lookup speculative
 * decoding (the default local strategy). Sequential single-stream requests,
 * 8 prompts, up to 64 new tokens, temperature 0.7.
 */
export const ENGINE_PATH = {
  intro:
    "The full DistLLM engine path — ModelPartitioner load, TokenGenerator sampling, prompt-lookup speculative decoding (the default strategy for local models). Sequential single-stream requests, 8 prompts, up to 64 new tokens, temperature 0.7.",
  rows: [
    { metric: "TTFT (median)", cuda: "64.2 ms", cpu: "15.0 ms" },
    { metric: "Throughput (aggregate)", cuda: "13.5 tok/s", cpu: "30.0 tok/s" },
    { metric: "Throughput (mean per request)", cuda: "13.4 tok/s", cpu: "37.0 tok/s" },
    { metric: "Generated tokens (8 prompts)", cuda: "336", cpu: "434" },
    { metric: "Model load time", cuda: "11.1 s", cpu: "11.1 s" },
    { metric: "Warmup (8 tokens)", cuda: "5.6 ms", cpu: "402 ms" },
  ] as EngineMetricRow[],
};

/** Honest finding about the engine-vs-baseline gap on GPU. */
export const ENGINE_GAP =
  "The engine's local generation strategy is ~90× slower than the raw HF baseline on GPU (13.5 vs 1200.7 tok/s). Root cause: the prompt-lookup strategy re-forwards the entire sequence every step — no KV-cache reuse, O(n²) total work — plus multiple GPU→CPU syncs per step. KV-cache reuse in _PromptLookupStrategy would be step one.";

export interface StatCard {
  label: string;
  value: string;
  note: string;
}

/** Other measured capabilities from the same run. */
export const CAPABILITY_STATS: StatCard[] = [
  {
    label: "Max sustained concurrency",
    value: "16 req/GPU",
    note: "Ramp ceiling reached, no OOM. Peak GPU memory 551 MB.",
  },
  {
    label: "KV-cache prefix hit rate",
    value: "100%",
    note: "16 requests sharing a 64-token prefix — component test of the PrefixCache, not end-to-end serving.",
  },
];

/**
 * Suite pass/fail thresholds embedded in benchmarks/run.py results. These were
 * written for 1.5B-class models — the TinyStories-1M numbers above clear them
 * trivially and say nothing about larger-model performance.
 */
export const SUITE_TARGETS = [
  { name: "throughput-small", target: "≥ 100 tok/s", result: "1200.7 CUDA / 652.4 CPU", met: true },
  { name: "latency-ttft", target: "≤ 500 ms", result: "8.6 ms CUDA / 5.2 ms CPU", met: true },
  { name: "latency-itl", target: "≤ 50 ms", result: "6.9–7.1 ms CUDA / 4.3–4.7 ms CPU", met: true },
  { name: "spec-accept-rate", target: "≥ 60%", result: "47.5% ngram / 78% eagle", met: false },
  { name: "memory-efficiency", target: "≥ 8 req/GPU", result: "16 req/GPU", met: true },
  { name: "kv-cache-hit-rate", target: "≥ 40%", result: "100%", met: true },
];

/**
 * Design goals with NO measured number behind them yet. Rendered under a
 * visible "Illustrative" badge; never presented as results.
 */
export const DESIGN_TARGETS = [
  {
    area: "70B-class models on pooled consumer GPUs",
    body: "Pipeline/tensor parallel serving of 70B-class models across pooled consumer GPUs is DistLLM's headline design goal. It requires multi-GPU, multi-node hardware we do not have, so no measured number exists. An older analytical estimate in benchmarks/results/throughput-dist.json is flagged do-not-publish and is deliberately absent from this page.",
  },
  {
    area: "Real-model speculative speedups",
    body: "End-to-end speedups for realistic draft/target pairs (135M-class drafts, 1B+ targets) need hardware beyond the current rig. Only synthetic verifier acceptance has been measured so far.",
  },
  {
    area: "HTTP end-to-end serving throughput",
    body: "Server-side load testing of the API boundary is planned follow-up work. Every measured number above exercises library-level generation, not HTTP serving.",
  },
];

export interface MethodCard {
  title: string;
  body: string;
}

export const METHODOLOGY: MethodCard[] = [
  {
    title: "Hardware",
    body: "One laptop: Intel Core i7-14700HX, 15.7 GB usable RAM, RTX 5060 Laptop GPU (8 GB GDDR7, Blackwell), Windows 11, PyTorch 2.13.0+cu130. No datacenter hardware in any published number.",
  },
  {
    title: "Workload",
    body: "roneneldan/TinyStories-1M (GPT-Neo, 3.7M params) from a local offline HF cache. Suite: batched prefill+decode, 8 left-padded prompts × 50 generated tokens. Engine bench: sequential single-stream, up to 64 tokens, temperature 0.7. fp16 on CUDA, fp32 on CPU.",
  },
  {
    title: "Measurement",
    body: "TTFT from request send to first streamed token, median over 8 prompts; throughput counts decoded tokens per second of wall clock. Engine-path token counts use instrumented ground truth, not stream-event count.",
  },
  {
    title: "Reproducibility",
    body: "Exact commands and raw JSON ship in the repo: docs/benchmark-results.md plus benchmarks/results/*.json. Re-run the suite on your own machine to compare.",
  },
  {
    title: "What was NOT measured",
    body: "Multi-node anything, 70B models, real-model speculative acceptance, and HTTP end-to-end load could not be measured on this rig. Those appear below as labeled design targets only.",
  },
];
