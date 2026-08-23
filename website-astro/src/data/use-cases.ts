export const USE_CASES = [
  {
    icon: "🏠",
    industry: "Home AI Lab",
    headline: "A 70B assistant on hardware you already own",
    body: "Combine the gaming PC, the old laptop, and the mini-PC under the desk into one private LLM endpoint. No subscriptions, no telemetry, no cloud bills — your conversations never leave the house.",
    points: ["Zero per-token cost", "Works over home WiFi", "OpenAI-compatible for any client"],
  },
  {
    icon: "🏥",
    industry: "Healthcare",
    headline: "HIPAA-aligned inference inside your network",
    body: "Clinical notes and records stay on hospital hardware. Differential-privacy training means model updates can't leak a single patient's data, and every node authenticates with shared cluster keys.",
    points: ["Data never leaves premises", "DP-protected fine-tuning", "Audit logging built in"],
  },
  {
    icon: "🏦",
    industry: "Finance",
    headline: "Deterministic, air-gapped document analysis",
    body: "Run contract review and summarization entirely on-premises. Quantized models fit commodity servers, and the benchmark suite proves throughput before you commit to SLAs.",
    points: ["Air-gapped deployment", "Quantization cuts memory 4–8×", "Reproducible benchmarks"],
  },
  {
    icon: "🎓",
    industry: "Education",
    headline: "A teaching cluster out of lab machines",
    body: "Turn a classroom of heterogeneous laptops into a live distributed-systems demo. Students see pipeline parallelism, straggler handling, and federated learning happen in front of them.",
    points: ["Runs on existing lab hardware", "Real distributed-systems lessons", "Free and open source"],
  },
  {
    icon: "⚖️",
    industry: "Legal",
    headline: "Privilege-preserving summarization",
    body: "Client documents are summarized by local models with privacy projections on sensitive layers. Nothing is sent to third-party APIs, so privilege is preserved end to end.",
    points: ["No third-party API calls", "Privacy layer projections", "Per-tenant adapter isolation"],
  },
  {
    icon: "🛠️",
    industry: "Edge & Field Ops",
    headline: "Inference where the internet isn't",
    body: "Ships, remote sites, factory floors: pool whatever compute is on site into one serving node. Models sync from a cache dir; the cluster self-heals when devices come and go.",
    points: ["Offline-first design", "Node join/leave without restarts", "Checkpoint replay after failures"],
  },
];
