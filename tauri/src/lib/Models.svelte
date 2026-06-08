<script lang="ts">
  import { listModels, downloadModel } from "./api";
  import { logStore } from "./stores";
  import { Card, ErrorBanner, toastStore } from "./ui";
  import type { ModelInfo } from "./types";
  import { onMount, onDestroy } from "svelte";
  import { listen } from "@tauri-apps/api/event";

  let models = $state<ModelInfo[]>([]);
  let loading = $state(true);
  let error = $state<string | null>(null);
  let searchQuery = $state("");
  let downloading = $state<Set<string>>(new Set());
  // 3.6: Track download progress per model
  let downloadProgress = $state<Map<string, string>>(new Map());
  let unlistenProgress: (() => void) | undefined;

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

  // 3.6: Listen for download progress events
  onMount(async () => {
    await loadModels();
    unlistenProgress = await listen<{ model_id: string; status: string; detail: string }>(
      "download-progress",
      (e) => {
        const { model_id, status, detail } = e.payload;
        if (status === "completed") {
          downloadProgress = new Map([...downloadProgress].filter(([k]) => k !== model_id));
          downloading = new Set([...downloading].filter((id) => id !== model_id));
          toastStore.success("Downloaded " + model_id.split("/").pop());
          logStore.info("models", `Model downloaded: ${model_id}`);
          loadModels();
        } else if (status === "failed") {
          downloadProgress = new Map([...downloadProgress].filter(([k]) => k !== model_id));
          downloading = new Set([...downloading].filter((id) => id !== model_id));
          error = detail;
          logStore.error("models", `Model download failed: ${model_id} - ${detail}`);
        } else {
          downloadProgress = new Map(downloadProgress).set(model_id, detail);
        }
      },
    );
  });

  onDestroy(() => {
    unlistenProgress?.();
  });

  async function handleDownload(modelId: string) {
    downloading = new Set([...downloading, modelId]);
    downloadProgress = new Map(downloadProgress).set(modelId, "Starting...");
    error = null;
    logStore.info("models", `Starting download: ${modelId}`);
    try {
      await downloadModel(modelId);
    } catch (e: unknown) {
      error = String(e);
      downloading = new Set([...downloading].filter((id) => id !== modelId));
      downloadProgress = new Map([...downloadProgress].filter(([k]) => k !== modelId));
      logStore.error("models", `Download failed: ${modelId} - ${e}`);
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

  onMount(loadModels);
</script>

<div class="models-page">
  <h1 class="page-title">Model Browser</h1>

  <ErrorBanner message={error ?? ""} ondismiss={() => (error = null)} />

  <Card title="Available Models" description="One-click download from Hugging Face. Models are automatically split across cluster nodes.">
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
      {#each filtered as model (model.id)}
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
          {#if downloading.has(model.id)}
            <div class="download-status">
              <div class="download-spinner"></div>
              <span class="download-detail">{downloadProgress.get(model.id) ?? "Downloading..."}</span>
            </div>
          {:else}
            <button
              class="btn btn-download"
              onclick={() => handleDownload(model.id)}
            >
              Download
            </button>
          {/if}
        </div>
      {/each}
    </div>

    {#if filtered.length === 0}
      <div class="empty-state">No models match your search.</div>
    {/if}
  </Card>
</div>

<style>
  .models-page { max-width: 900px; }
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
    outline: none;
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
    cursor: pointer;
    border: none;
    transition: background 0.15s;
  }
  .btn-download:hover:not(:disabled) { background: var(--accent-hover); }
  .download-status {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 0;
    font-size: 12px;
    color: var(--text-secondary);
  }
  .download-spinner {
    width: 14px;
    height: 14px;
    border: 2px solid var(--border);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }
  .download-detail {
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  @keyframes spin {
    to { transform: rotate(360deg); }
  }
</style>
