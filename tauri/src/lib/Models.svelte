<script lang="ts">
  import { listModels, downloadModel } from "./api";
  import type { ModelInfo } from "./types";

  let models = $state<ModelInfo[]>([]);
  let loading = $state(true);
  let error = $state<string | null>(null);
  let searchQuery = $state("");
  let downloading = $state<Set<string>>(new Set());

  async function loadModels() {
    loading = true;
    error = null;
    try {
      models = await listModels();
    } catch (e: unknown) {
      error = String(e);
    } finally {
      loading = false;
    }
  }

  async function handleDownload(modelId: string) {
    downloading.add(modelId);
    error = null;
    try {
      const msg = await downloadModel(modelId);
      console.log(msg);
      await loadModels();
    } catch (e: unknown) {
      error = String(e);
    } finally {
      downloading.delete(modelId);
    }
  }

  const popularModels: { id: string; name: string; size: string; gpu: string }[] = [
    { id: "HuggingFaceTB/SmolLM-135M", name: "SmolLM 135M", size: "270 MB", gpu: "Any" },
    { id: "HuggingFaceTB/SmolLM-360M", name: "SmolLM 360M", size: "720 MB", gpu: "Any" },
    { id: "HuggingFaceTB/SmolLM-1.7B", name: "SmolLM 1.7B", size: "3.4 GB", gpu: "6 GB+" },
    { id: "Qwen/Qwen2.5-7B-Instruct", name: "Qwen 2.5 7B", size: "14 GB", gpu: "12 GB+" },
    { id: "mistralai/Mistral-7B-Instruct-v0.3", name: "Mistral 7B", size: "14 GB", gpu: "12 GB+" },
    { id: "meta-llama/Llama-3.1-8B", name: "Llama 3.1 8B", size: "16 GB", gpu: "16 GB+" },
    { id: "Qwen/Qwen2.5-32B-Instruct", name: "Qwen 2.5 32B", size: "64 GB", gpu: "48 GB+ (multi-node)" },
    { id: "mistralai/Mixtral-8x7B-Instruct-v0.1", name: "Mixtral 8x7B", size: "90 GB", gpu: "48 GB+ (multi-node)" },
  ];

  let filtered = $derived(
    searchQuery.trim()
      ? popularModels.filter(
          (m) =>
            m.id.toLowerCase().includes(searchQuery.toLowerCase()) ||
            m.name.toLowerCase().includes(searchQuery.toLowerCase()),
        )
      : popularModels,
  );

  // Load downloaded models on mount
  import { onMount } from "svelte";
  onMount(loadModels);
</script>

<div class="models-page">
  <h1 class="page-title">Model Browser</h1>

  {#if error}
    <div class="error-banner">{error}</div>
  {/if}

  <section class="card">
    <h2 class="card-title">Available Models</h2>
    <p class="card-desc">
      One-click download from Hugging Face. Models are automatically split across cluster nodes.
    </p>

    <div class="search-bar">
      <span class="search-icon">🔍</span>
      <input
        type="text"
        class="search-input"
        placeholder="Search models..."
        bind:value={searchQuery}
      />
    </div>

    <div class="model-grid">
      {#each filtered as model}
        <div class="model-card">
          <div class="model-header">
            <span class="model-name">{model.name}</span>
            {#if model.id.startsWith("meta-llama")}
              <span class="model-badge">Gated</span>
            {/if}
          </div>
          <div class="model-meta">
            <span class="meta-item">{model.size}</span>
            <span class="meta-item">GPU: {model.gpu}</span>
          </div>
          <div class="model-id mono">{model.id}</div>
          <button
            class="btn btn-download"
            disabled={downloading.has(model.id)}
            onclick={() => handleDownload(model.id)}
          >
            {downloading.has(model.id) ? "Downloading..." : "Download"}
          </button>
        </div>
      {/each}
    </div>

    {#if filtered.length === 0}
      <div class="empty-state">No models match your search.</div>
    {/if}
  </section>
</div>

<style>
  .models-page { max-width: 900px; }
  .page-title { font-size: 22px; font-weight: 700; margin-bottom: 20px; }
  .error-banner {
    background: color-mix(in srgb, var(--danger) 15%, transparent);
    color: var(--danger);
    padding: 10px 14px;
    border-radius: 8px;
    margin-bottom: 16px;
    font-size: 13px;
  }
  .card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 16px;
  }
  .card-title { font-size: 15px; font-weight: 600; margin-bottom: 6px; }
  .card-desc { font-size: 13px; color: var(--text-secondary); margin-bottom: 16px; }
  .search-bar {
    display: flex;
    align-items: center;
    gap: 8px;
    background: var(--bg-input);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 8px 12px;
    margin-bottom: 16px;
  }
  .search-icon { font-size: 14px; opacity: 0.5; }
  .search-input {
    flex: 1;
    background: transparent;
    border: none;
    color: var(--text-primary);
    font-size: 14px;
  }
  .search-input::placeholder { color: var(--text-muted); }
  .model-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 12px; }
  .model-card {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  .model-header { display: flex; align-items: center; gap: 8px; }
  .model-name { font-weight: 600; font-size: 14px; }
  .model-badge {
    font-size: 10px;
    background: var(--warning);
    color: #000;
    padding: 1px 6px;
    border-radius: 4px;
    font-weight: 600;
  }
  .model-meta { display: flex; gap: 12px; font-size: 12px; color: var(--text-secondary); }
  .model-id { font-size: 11px; color: var(--text-muted); word-break: break-all; }
  .btn-download {
    margin-top: 4px;
    padding: 8px 16px;
    background: var(--accent);
    color: #fff;
    border-radius: 6px;
    font-size: 13px;
    font-weight: 600;
    transition: background 0.15s;
  }
  .btn-download:hover:not(:disabled) { background: var(--accent-hover); }
  .btn-download:disabled { opacity: 0.5; cursor: not-allowed; }
  .mono { font-family: var(--font-mono); }
  .empty-state { color: var(--text-muted); font-size: 13px; padding: 20px 0; text-align: center; }
</style>
