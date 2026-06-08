<script lang="ts">
  import { onMount } from "svelte";
  import {
    getOllamaConfig,
    checkOllama,
    listOllamaModels,
    pullOllamaModel,
  } from "./api";
  import { Card, ErrorBanner } from "./ui";
  import type { OllamaConfig, OllamaModel } from "./types";

  let config = $state<OllamaConfig>({
    host: "127.0.0.1",
    port: 11434,
    enabled: false,
  });
  let reachable = $state<boolean | null>(null);
  let models = $state<OllamaModel[]>([]);
  let loading = $state(false);
  let pulling = $state<string | null>(null);
  let error = $state<string | null>(null);
  let success = $state<string | null>(null);

  onMount(async () => {
    try {
      config = await getOllamaConfig();
      if (config.enabled) {
        await testConnection();
      }
    } catch {
      // ignore
    }
  });

  async function testConnection() {
    loading = true;
    error = null;
    try {
      reachable = await checkOllama(config);
      if (reachable) {
        models = await listOllamaModels(config);
      }
    } catch (e: unknown) {
      error = String(e);
      reachable = false;
    } finally {
      loading = false;
    }
  }

  async function handlePull(modelName: string) {
    pulling = modelName;
    error = null;
    success = null;
    try {
      await pullOllamaModel(config, modelName);
      success = `Model "${modelName}" pulled successfully`;
      await testConnection();
    } catch (e: unknown) {
      error = `Failed to pull ${modelName}: ${e}`;
    } finally {
      pulling = null;
    }
  }

  function fmtSize(bytes: number): string {
    const gb = bytes / (1024 * 1024 * 1024);
    if (gb >= 1) return `${gb.toFixed(1)} GB`;
    const mb = bytes / (1024 * 1024);
    return `${mb.toFixed(0)} MB`;
  }
</script>

<div class="ollama-section">
  <div class="section-header">
    <h3 class="section-title">Ollama Compatibility</h3>
    <span class="section-badge" class:active={config.enabled}>
      {config.enabled ? "Enabled" : "Disabled"}
    </span>
  </div>
  <p class="section-desc">
    Connect to a local or remote Ollama server. Set <code>OLLAMA_HOST</code> env
    var or configure below.
  </p>

  <ErrorBanner message={error ?? ""} ondismiss={() => (error = null)} />
  {#if success}
    <div class="success-banner">{success}</div>
  {/if}

  <div class="config-row">
    <label class="toggle-label">
      <input
        type="checkbox"
        bind:checked={config.enabled}
        class="toggle-input"
      />
      <span class="toggle-switch"></span>
      <span>Enable Ollama</span>
    </label>
  </div>

  {#if config.enabled}
    <div class="config-grid">
      <div class="form-field">
        <label class="field-label" for="ollama-host">Host</label>
        <input
          id="ollama-host"
          class="field-input"
          bind:value={config.host}
          placeholder="127.0.0.1"
        />
      </div>
      <div class="form-field">
        <label class="field-label" for="ollama-port">Port</label>
        <input
          id="ollama-port"
          class="field-input"
          type="number"
          bind:value={config.port}
          min={1}
          max={65535}
        />
      </div>
    </div>

    <div class="actions">
      <button class="btn btn-primary" onclick={testConnection} disabled={loading}>
        {loading ? "Testing..." : "Test Connection"}
      </button>
    </div>

    {#if reachable === true}
      <div class="status-msg ok">Connected to Ollama at {config.host}:{config.port}</div>
    {:else if reachable === false}
      <div class="status-msg err">Cannot reach Ollama at {config.host}:{config.port}</div>
    {/if}

    {#if models.length > 0}
      <div class="models-section">
        <h4 class="models-title">Available Models ({models.length})</h4>
        <div class="models-list">
          {#each models as model (model.name)}
            <div class="model-item">
              <div class="model-info">
                <span class="model-name">{model.name}</span>
                <span class="model-meta">{fmtSize(model.size)}</span>
              </div>
            </div>
          {/each}
        </div>
      </div>
    {/if}
  {/if}
</div>

<style>
  .ollama-section {
    padding: 16px 0;
  }
  .section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 4px;
  }
  .section-title {
    font-size: 15px;
    font-weight: 600;
  }
  .section-badge {
    font-size: 10px;
    padding: 2px 8px;
    border-radius: 4px;
    background: rgba(128, 128, 128, 0.2);
    color: var(--text-muted);
  }
  .section-badge.active {
    background: rgba(34, 197, 94, 0.2);
    color: #22c55e;
  }
  .section-desc {
    font-size: 13px;
    color: var(--text-muted);
    margin-bottom: 12px;
  }
  .section-desc code {
    font-family: var(--font-mono);
    font-size: 12px;
    background: rgba(128, 128, 128, 0.15);
    padding: 1px 4px;
    border-radius: 3px;
  }
  .success-banner {
    background: rgba(34, 197, 94, 0.1);
    color: #22c55e;
    padding: 10px 14px;
    border-radius: 8px;
    margin-bottom: 12px;
    font-size: 13px;
  }
  .config-row {
    margin-bottom: 14px;
  }
  .toggle-label {
    display: flex;
    align-items: center;
    gap: 10px;
    cursor: pointer;
    font-size: 14px;
  }
  .toggle-input {
    display: none;
  }
  .toggle-switch {
    width: 36px;
    height: 20px;
    background: rgba(128, 128, 128, 0.3);
    border-radius: 10px;
    position: relative;
    transition: background 0.2s;
  }
  .toggle-switch::after {
    content: "";
    position: absolute;
    top: 2px;
    left: 2px;
    width: 16px;
    height: 16px;
    background: white;
    border-radius: 50%;
    transition: transform 0.2s;
  }
  .toggle-input:checked + .toggle-switch {
    background: var(--accent);
  }
  .toggle-input:checked + .toggle-switch::after {
    transform: translateX(16px);
  }
  .config-grid {
    display: grid;
    grid-template-columns: 1fr 120px;
    gap: 12px;
    margin-bottom: 14px;
  }
  .form-field {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .field-label {
    font-size: 12px;
    color: var(--text-muted);
  }
  .field-input {
    padding: 8px 10px;
    background: var(--bg-input);
    border: 1px solid var(--border);
    border-radius: 6px;
    color: var(--text);
    font-size: 13px;
  }
  .field-input:focus {
    outline: none;
    border-color: var(--accent);
  }
  .actions {
    margin-bottom: 12px;
  }
  .btn {
    padding: 8px 16px;
    border-radius: 6px;
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    border: none;
  }
  .btn-primary {
    background: var(--accent);
    color: white;
  }
  .btn-primary:hover:not(:disabled) {
    opacity: 0.9;
  }
  .btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  .status-msg {
    font-size: 13px;
    padding: 8px 12px;
    border-radius: 6px;
    margin-bottom: 12px;
  }
  .status-msg.ok {
    background: rgba(34, 197, 94, 0.1);
    color: #22c55e;
  }
  .status-msg.err {
    background: rgba(239, 68, 68, 0.1);
    color: #ef4444;
  }
  .models-section {
    margin-top: 8px;
  }
  .models-title {
    font-size: 13px;
    font-weight: 600;
    margin-bottom: 8px;
    color: var(--text-secondary);
  }
  .models-list {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .model-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 10px;
    background: var(--bg-input);
    border-radius: 6px;
  }
  .model-info {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }
  .model-name {
    font-size: 13px;
    font-weight: 500;
  }
  .model-meta {
    font-size: 11px;
    color: var(--text-muted);
    font-family: var(--font-mono);
  }
</style>
