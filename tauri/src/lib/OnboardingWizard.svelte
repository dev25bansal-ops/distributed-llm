<script lang="ts">
  import { getGpuMetrics, createCluster, joinCluster } from "./api";
  import { Card, Button, Input, StatusDot } from "./ui";
  import type { GpuInfo } from "./types";

  let { oncomplete }: { oncomplete: () => void } = $props();

  let step = $state(1);
  let gpus = $state<GpuInfo[]>([]);
  let loading = $state(false);
  let error = $state<string | null>(null);

  // Step 2: Model recommendation
  let recommendedModel = $state("");
  let selectedModel = $state("");
  const modelOptions = [
    { id: "HuggingFaceTB/SmolLM-135M", label: "SmolLM 135M", vram_mb: 512, desc: "Fast, minimal VRAM" },
    { id: "HuggingFaceTB/SmolLM-360M", label: "SmolLM 360M", vram_mb: 1024, desc: "Balanced small model" },
    { id: "Qwen/Qwen2.5-0.5B", label: "Qwen2.5 0.5B", vram_mb: 1200, desc: "Good multilingual" },
    { id: "Qwen/Qwen2.5-1.5B", label: "Qwen2.5 1.5B", vram_mb: 2048, desc: "Strong reasoning" },
    { id: "Qwen/Qwen2.5-3B", label: "Qwen2.5 3B", vram_mb: 4096, desc: "High quality" },
    { id: "meta-llama/Llama-3.2-1B", label: "Llama 3.2 1B", vram_mb: 1500, desc: "Meta's efficient model" },
    { id: "meta-llama/Llama-3.2-3B", label: "Llama 3.2 3B", vram_mb: 4096, desc: "Meta's balanced model" },
  ];

  // Step 3: Cluster
  let createPort = $state(8000);
  let joinHost = $state("");
  let joinPort = $state(8000);
  let mode = $state<"create" | "join">("create");

  async function detectGpus() {
    loading = true;
    error = null;
    try {
      gpus = await getGpuMetrics();
      recommendModel();
      step = 2;
    } catch (e: unknown) {
      error = `GPU detection failed: ${e}`;
    } finally {
      loading = false;
    }
  }

  function recommendModel() {
    const totalVram = gpus.reduce((sum, g) => sum + g.memory_total, 0);
    const vramMb = totalVram / (1024 * 1024);

    // Pick the largest model that fits in 60% of VRAM
    const fit = modelOptions
      .filter((m) => m.vram_mb <= vramMb * 0.6)
      .sort((a, b) => b.vram_mb - a.vram_mb)[0];

    recommendedModel = fit?.id ?? modelOptions[0].id;
    selectedModel = recommendedModel;
  }

  function fmtVram(bytes: number): string {
    const mb = bytes / (1024 * 1024);
    if (mb >= 1024) return (mb / 1024).toFixed(1) + " GB";
    return mb.toFixed(0) + " MB";
  }

  async function finishCluster() {
    loading = true;
    error = null;
    try {
      if (mode === "create") {
        await createCluster(createPort, selectedModel || undefined);
      } else {
        await joinCluster(joinHost, joinPort);
      }
      localStorage.setItem("distllm_onboarded", "1");
      oncomplete();
    } catch (e: unknown) {
      error = String(e);
    } finally {
      loading = false;
    }
  }

  function skipWizard() {
    localStorage.setItem("distllm_onboarded", "1");
    oncomplete();
  }
</script>

<div class="wizard-overlay">
  <div class="wizard-modal">
    <div class="wizard-header">
      <div class="wizard-steps">
        <span class="step" class:active={step >= 1} class:done={step > 1}>1</span>
        <span class="step-line" class:done={step > 1}></span>
        <span class="step" class:active={step >= 2} class:done={step > 2}>2</span>
        <span class="step-line" class:done={step > 2}></span>
        <span class="step" class:active={step >= 3}>3</span>
      </div>
      <h2 class="wizard-title">
        {#if step === 1}Detect Hardware{:else if step === 2}Choose Model{:else}Set Up Cluster{/if}
      </h2>
      <p class="wizard-desc">
        {#if step === 1}
          We'll scan for NVIDIA GPUs to recommend the best model for your hardware.
        {:else if step === 2}
          Based on your GPU memory, we recommend a model that fits comfortably.
        {:else}
          Create a new cluster on this machine, or join an existing one.
        {/if}
      </p>
    </div>

    {#if error}
      <div class="wizard-error">{error}</div>
    {/if}

    <div class="wizard-body">
      {#if step === 1}
        <div class="detect-step">
          {#if gpus.length > 0}
            <div class="gpu-results">
              {#each gpus as gpu (gpu.index)}
                <div class="gpu-item">
                  <div class="gpu-header">
                    <span class="gpu-name">{gpu.name}</span>
                    <span class="gpu-vram">{fmtVram(gpu.memory_total)}</span>
                  </div>
                  <div class="gpu-bar-track">
                    <div class="gpu-bar-fill" style="width: {gpu.utilization}%"></div>
                  </div>
                </div>
              {/each}
            </div>
          {:else}
            <div class="gpu-empty">
              <span class="gpu-empty-icon">◎</span>
              <span>No NVIDIA GPUs detected</span>
              <span class="gpu-empty-hint">You can still use CPU inference or connect to remote GPUs.</span>
            </div>
          {/if}
        </div>
      {:else if step === 2}
        <div class="model-step">
          {#if gpus.length > 0}
            <div class="recommendation">
              <StatusDot variant="green" />
              <span>Recommended: <strong>{modelOptions.find(m => m.id === recommendedModel)?.label}</strong></span>
            </div>
          {/if}
          <div class="model-list">
            {#each modelOptions as model (model.id)}
              <button
                class="model-option"
                class:selected={selectedModel === model.id}
                onclick={() => (selectedModel = model.id)}
              >
                <div class="model-info">
                  <span class="model-label">{model.label}</span>
                  <span class="model-desc">{model.desc}</span>
                </div>
                <span class="model-vram">~{model.vram_mb} MB</span>
              </button>
            {/each}
          </div>
        </div>
      {:else}
        <div class="cluster-step">
          <div class="mode-toggle">
            <button
              class="mode-btn"
              class:active={mode === "create"}
              onclick={() => (mode = "create")}
            >
              Create Cluster
            </button>
            <button
              class="mode-btn"
              class:active={mode === "join"}
              onclick={() => (mode = "join")}
            >
              Join Cluster
            </button>
          </div>

          {#if mode === "create"}
            <div class="form-row">
              <label class="form-label" for="wiz-port">Port</label>
              <Input id="wiz-port" type="number" bind:value={createPort} min={1024} max={65535} />
            </div>
            <div class="form-hint">
              Model: <strong>{modelOptions.find(m => m.id === selectedModel)?.label ?? "None"}</strong>
            </div>
          {:else}
            <div class="form-row">
              <label class="form-label" for="wiz-host">Coordinator Host</label>
              <Input id="wiz-host" placeholder="192.168.1.100" bind:value={joinHost} />
            </div>
            <div class="form-row">
              <label class="form-label" for="wiz-join-port">Port</label>
              <Input id="wiz-join-port" type="number" bind:value={joinPort} min={1024} max={65535} />
            </div>
          {/if}
        </div>
      {/if}
    </div>

    <div class="wizard-footer">
      <button class="skip-btn" onclick={skipWizard}>Skip Setup</button>
      <div class="footer-actions">
        {#if step > 1}
          <Button variant="ghost" onclick={() => step--}>Back</Button>
        {/if}
        {#if step === 1}
          <Button onclick={detectGpus} disabled={loading}>
            {loading ? "Scanning..." : "Detect GPUs"}
          </Button>
        {:else if step === 2}
          <Button onclick={() => (step = 3)}>Continue</Button>
        {:else}
          <Button onclick={finishCluster} disabled={loading || (mode === "join" && !joinHost)}>
            {loading ? "Setting up..." : mode === "create" ? "Create Cluster" : "Join Cluster"}
          </Button>
        {/if}
      </div>
    </div>
  </div>
</div>

<style>
  .wizard-overlay {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.7);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 10000;
    backdrop-filter: blur(4px);
  }
  .wizard-modal {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    width: 520px;
    max-width: 95vw;
    max-height: 90vh;
    overflow-y: auto;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
  }
  .wizard-header {
    padding: 28px 28px 0;
  }
  .wizard-steps {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0;
    margin-bottom: 20px;
  }
  .step {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 13px;
    font-weight: 600;
    background: var(--bg-input);
    color: var(--text-muted);
    border: 2px solid var(--border);
    transition: all 0.2s;
  }
  .step.active {
    background: var(--accent);
    color: white;
    border-color: var(--accent);
  }
  .step.done {
    background: var(--success);
    color: white;
    border-color: var(--success);
  }
  .step-line {
    width: 48px;
    height: 2px;
    background: var(--border);
    transition: background 0.2s;
  }
  .step-line.done {
    background: var(--success);
  }
  .wizard-title {
    font-size: 20px;
    font-weight: 700;
    margin-bottom: 6px;
    text-align: center;
  }
  .wizard-desc {
    font-size: 14px;
    color: var(--text-secondary);
    text-align: center;
    margin-bottom: 0;
  }
  .wizard-error {
    margin: 16px 28px 0;
    padding: 10px 14px;
    background: color-mix(in srgb, var(--danger) 15%, transparent);
    color: var(--danger);
    border-radius: 8px;
    font-size: 13px;
  }
  .wizard-body {
    padding: 24px 28px;
  }
  .wizard-footer {
    padding: 16px 28px 24px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .footer-actions {
    display: flex;
    gap: 8px;
  }
  .skip-btn {
    background: none;
    border: none;
    color: var(--text-muted);
    font-size: 13px;
    cursor: pointer;
    padding: 8px 12px;
    border-radius: 6px;
  }
  .skip-btn:hover {
    color: var(--text-secondary);
    background: var(--bg-input);
  }

  /* Step 1: GPU detection */
  .gpu-results {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }
  .gpu-item {
    background: var(--bg-input);
    border-radius: 8px;
    padding: 12px;
  }
  .gpu-header {
    display: flex;
    justify-content: space-between;
    margin-bottom: 8px;
  }
  .gpu-name {
    font-size: 14px;
    font-weight: 500;
  }
  .gpu-vram {
    font-size: 13px;
    color: var(--accent);
    font-family: var(--font-mono);
  }
  .gpu-bar-track {
    height: 6px;
    background: var(--border);
    border-radius: 3px;
    overflow: hidden;
  }
  .gpu-bar-fill {
    height: 100%;
    background: var(--accent);
    border-radius: 3px;
    transition: width 0.5s ease;
  }
  .gpu-empty {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
    padding: 24px;
    color: var(--text-muted);
    font-size: 14px;
  }
  .gpu-empty-icon {
    font-size: 32px;
    opacity: 0.4;
  }
  .gpu-empty-hint {
    font-size: 12px;
    opacity: 0.7;
  }

  /* Step 2: Model selection */
  .recommendation {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 14px;
    background: color-mix(in srgb, var(--success) 10%, transparent);
    border-radius: 8px;
    margin-bottom: 14px;
    font-size: 13px;
  }
  .model-list {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .model-option {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 14px;
    background: var(--bg-input);
    border: 2px solid transparent;
    border-radius: 8px;
    cursor: pointer;
    text-align: left;
    width: 100%;
    transition: border-color 0.15s;
  }
  .model-option:hover {
    border-color: var(--border);
  }
  .model-option.selected {
    border-color: var(--accent);
    background: color-mix(in srgb, var(--accent) 8%, var(--bg-input));
  }
  .model-info {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }
  .model-label {
    font-size: 14px;
    font-weight: 500;
    color: var(--text);
  }
  .model-desc {
    font-size: 12px;
    color: var(--text-muted);
  }
  .model-vram {
    font-size: 12px;
    font-family: var(--font-mono);
    color: var(--text-secondary);
  }

  /* Step 3: Cluster */
  .mode-toggle {
    display: flex;
    gap: 4px;
    background: var(--bg-input);
    border-radius: 8px;
    padding: 4px;
    margin-bottom: 16px;
  }
  .mode-btn {
    flex: 1;
    padding: 8px 12px;
    border: none;
    border-radius: 6px;
    background: transparent;
    color: var(--text-secondary);
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.15s;
  }
  .mode-btn.active {
    background: var(--accent);
    color: white;
  }
  .form-hint {
    font-size: 13px;
    color: var(--text-secondary);
    padding: 8px 0;
  }
</style>
