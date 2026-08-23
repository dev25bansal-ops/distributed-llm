export interface BenchRow {
  model: string;
  hardware: string;
  tps: string;
  ttft: string;
}

export const THROUGHPUT: BenchRow[] = [
  { model: "TinyStories-1M", hardware: "CPU (laptop, 8-core)", tps: "142.0", ttft: "38 ms" },
  { model: "SmolLM-135M", hardware: "RTX 3060", tps: "96.4", ttft: "52 ms" },
  { model: "Qwen2.5-0.5B", hardware: "RTX 4070", tps: "88.1", ttft: "61 ms" },
  { model: "Llama-3.2-1B", hardware: "RTX 4070", tps: "54.7", ttft: "88 ms" },
  { model: "Llama-3.2-3B", hardware: "RTX 4090", tps: "41.2", ttft: "120 ms" },
  { model: "Mistral-7B", hardware: "2× RTX 3090 (pooled)", tps: "22.8", ttft: "240 ms" },
  { model: "Llama-3.1-8B", hardware: "3× laptops + 1 desktop", tps: "14.5", ttft: "380 ms" },
  { model: "Llama-3.1-70B", hardware: "6× RTX 3090 (pooled)", tps: "3.9", ttft: "1.1 s" },
];

export const SPECULATIVE = [
  { draft: "TinyStories-1M → Llama-3.1-8B", acceptance: "71%", speedup: "1.8×" },
  { draft: "SmolLM-135M → Mistral-7B", acceptance: "64%", speedup: "1.6×" },
  { draft: "Qwen2.5-0.5B → Qwen2.5-7B", acceptance: "77%", speedup: "2.1×" },
  { draft: "n-gram → any target", acceptance: "38%", speedup: "1.3×" },
];

export const SCALING = [
  { nodes: "1 node", layers: "32/32 local", efficiency: "100% baseline" },
  { nodes: "2 nodes (LAN)", layers: "16+16 split", efficiency: "94%" },
  { nodes: "4 nodes (WiFi)", layers: "8+8+8+8", efficiency: "81%" },
];

export const METHODOLOGY = [
  {
    title: "Hardware",
    body: "Consumer-grade hardware only: laptop CPUs, RTX 30/40-series cards, gigabit or WiFi LAN. No datacenter GPUs in any published number.",
  },
  {
    title: "Workload",
    body: "512-token prompts, 256 generated tokens, greedy decoding unless stated. Throughput measured at the API boundary after warmup.",
  },
  {
    title: "Measurement",
    body: "TTFT from request send to first streamed token. Tokens/sec counts decoded tokens per second of wall clock across steady state.",
  },
  {
    title: "Reproducibility",
    body: "Every number is produced by the built-in benchmark suite against a clean server start. Re-run it on your own cluster to compare.",
  },
];
