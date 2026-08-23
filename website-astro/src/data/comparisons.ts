export const COMPARISONS = [
  {
    vs: "DistLLM vs Petals",
    summary:
      "Petals pools public volunteers over the internet to serve big open models. DistLLM pools YOUR hardware over YOUR network — same idea, private by default, with training and an OpenAI-compatible API included.",
    rows: [
      { aspect: "Network", distllm: "Your LAN / your machines", other: "Public swarm (internet)" },
      { aspect: "Privacy", distllm: "Data never leaves your devices", other: "Requests traverse volunteer nodes" },
      { aspect: "Fine-tuning", distllm: "Federated LoRA + FedProx built in", other: "Inference-focused" },
      { aspect: "API", distllm: "OpenAI-compatible REST + SSE streaming", other: "Custom client libraries" },
      { aspect: "Auth", distllm: "Per-node cluster keys, API keys, JWT/SSO", other: "Trust-based public network" },
    ],
  },
  {
    vs: "DistLLM vs Modal / RunPod",
    summary:
      "Cloud GPU platforms rent you capacity by the hour. DistLLM makes hardware you already own behave like that platform — zero marginal cost, zero data egress.",
    rows: [
      { aspect: "Cost model", distllm: "Electricity only — hardware you own", other: "Per-second GPU rental" },
      { aspect: "Cold start", distllm: "None — always resident on your LAN", other: "Scale-from-zero latency" },
      { aspect: "Data locality", distllm: "Fully local processing", other: "Data leaves your premises" },
      { aspect: "Heterogeneous pool", distllm: "Mix CPUs and consumer GPUs freely", other: "Instance-type constrained" },
      { aspect: "Open source", distllm: "Apache-2.0 engine", other: "Proprietary control plane" },
    ],
  },
  {
    vs: "DistLLM + vLLM",
    summary:
      "Not competitors — layers. vLLM is a superb single-node engine; DistLLM is the pooling layer that coordinates several engines into one endpoint and can drive vLLM as a backend on capable nodes.",
    rows: [
      { aspect: "Scope", distllm: "Multi-node orchestration + API", other: "(vLLM) Single-node execution" },
      { aspect: "Small devices", distllm: "CPUs and laptops participate in the pipeline", other: "(vLLM) Needs one large GPU" },
      { aspect: "Training", distllm: "Federated LoRA fine-tuning included", other: "(vLLM) Inference only" },
      { aspect: "Together", distllm: "DistLLM schedules across nodes; vLLM serves where it fits best", other: "—" },
    ],
  },
];
