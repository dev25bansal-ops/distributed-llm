<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import { multiModelStore, clusterStore } from "./stores";
  import { ErrorBanner, StatusDot } from "./ui";
  import { listModels } from "./api";
  import type { ModelSlot, ModelRoutingRule, ModelInfo, ClusterStatus } from "./types";

  let slots = $state<ModelSlot[]>([]);
  let rules = $state<ModelRoutingRule[]>([]);
  let loading = $state(true);
  let error = $state<string | null>(null);
  let cluster = $state<ClusterStatus | null>(null);
  let availableModels = $state<ModelInfo[]>([]);

  let showAddRule = $state(false);
  let newRule = $state<ModelRoutingRule>({
    id: "",
    pattern: "",
    target_slot: "",
    priority: 0,
  });

  let unsubscribe: (() => void) | undefined;
  let unsubCluster: (() => void) | undefined;

  onMount(async () => {
    unsubscribe = multiModelStore.subscribe((d) => {
      slots = d.slots;
      rules = d.rules;
      loading = d.loading;
      error = d.error;
    });
    unsubCluster = clusterStore.subscribe((d) => {
      cluster = d.cluster;
    });
    try {
      availableModels = await listModels();
    } catch {
      // ignore
    }
  });

  onDestroy(() => {
    unsubscribe?.();
    unsubCluster?.();
  });

  function handleLoad(slotId: string, modelId: string) {
    multiModelStore.loadModel(slotId, modelId);
  }

  function handleUnload(slotId: string) {
    multiModelStore.unloadModel(slotId);
  }

  function handleAddRule() {
    if (!newRule.pattern || !newRule.target_slot) return;
    const rule = { ...newRule, id: `rule-${Date.now()}` };
    multiModelStore.addRule(rule);
    newRule = { id: "", pattern: "", target_slot: "", priority: 0 };
    showAddRule = false;
  }

  function handleRemoveRule(ruleId: string) {
    multiModelStore.removeRule(ruleId);
  }

  function statusColor(status: ModelSlot["status"]): string {
    switch (status) {
      case "ready": return "green";
      case "loading": return "gray";
      case "error": return "red";
      case "unloaded": return "gray";
    }
  }

  function fmtMb(mb: number): string {
    if (mb >= 1024) return (mb / 1024).toFixed(1) + " GB";
    return mb + " MB";
  }
</script>

<div class="multimodel-page">
  <h1 class="page-title">Multi-Model Serving</h1>

  <ErrorBanner message={error ?? ""} ondismiss={() => (error = null)} />

  {#if loading}
    <div class="card">
      <div class="empty-state">Loading model slots...</div>
    </div>
  {:else}
    <!-- Model Slots -->
    <div class="card">
      <h2 class="card-title">Model Slots</h2>
      <p class="card-desc">Load multiple models simultaneously across your cluster.</p>

      {#if !cluster?.running}
        <div class="empty-state">Start a cluster to manage model slots.</div>
      {:else if slots.length === 0}
        <div class="empty-state">No model slots configured. The coordinator will create slots as models are loaded.</div>
      {:else}
        <div class="slots-grid">
          {#each slots as slot (slot.id)}
            <div class="slot-card" class:ready={slot.status === "ready"} class:error={slot.status === "error"}>
              <div class="slot-header">
                <div class="slot-name">
                  <StatusDot variant={statusColor(slot.status)} />
                  <span>{slot.model_name || slot.model_id || "Empty"}</span>
                </div>
                <span class="slot-status badge {slot.status}">{slot.status}</span>
              </div>

              {#if slot.status === "ready"}
                <div class="slot-stats">
                  <div class="stat">
                    <span class="stat-label">VRAM</span>
                    <span class="stat-value">{fmtMb(slot.vram_allocated_mb)}</span>
                  </div>
                  <div class="stat">
                    <span class="stat-label">Context</span>
                    <span class="stat-value">{slot.max_context.toLocaleString()}</span>
                  </div>
                  <div class="stat">
                    <span class="stat-label">tok/s</span>
                    <span class="stat-value">{slot.avg_tokens_per_sec.toFixed(1)}</span>
                  </div>
                  <div class="stat">
                    <span class="stat-label">Requests</span>
                    <span class="stat-value">{slot.requests_served}</span>
                  </div>
                </div>
                <button class="btn btn-danger btn-sm" onclick={() => handleUnload(slot.id)}>
                  Unload
                </button>
              {:else if slot.status === "error"}
                <div class="slot-error">{slot.error_message}</div>
                <button class="btn btn-ghost btn-sm" onclick={() => handleLoad(slot.id, slot.model_id)}>
                  Retry
                </button>
              {:else if slot.status === "loading"}
                <div class="loading-bar">
                  <div class="loading-fill"></div>
                </div>
              {:else}
                <div class="form-row">
                  <select class="input" bind:value={slot.model_id}>
                    <option value="">Select model...</option>
                    {#each availableModels.filter(m => m.downloaded) as model (model.id)}
                      <option value={model.id}>{model.name} ({model.size})</option>
                    {/each}
                  </select>
                </div>
                <button
                  class="btn btn-primary btn-sm"
                  disabled={!slot.model_id}
                  onclick={() => handleLoad(slot.id, slot.model_id)}
                >
                  Load Model
                </button>
              {/if}
            </div>
          {/each}
        </div>
      {/if}
    </div>

    <!-- Routing Rules -->
    <div class="card">
      <div class="card-header-row">
        <h2 class="card-title">Routing Rules</h2>
        <button class="btn btn-ghost btn-sm" onclick={() => showAddRule = !showAddRule}>
          {showAddRule ? "Cancel" : "+ Add Rule"}
        </button>
      </div>
      <p class="card-desc">Route incoming requests to specific model slots based on patterns.</p>

      {#if showAddRule}
        <div class="add-rule-form">
          <div class="form-row">
            <label class="form-label" for="rule-pattern">Pattern (regex or model name)</label>
            <input id="rule-pattern" class="input" bind:value={newRule.pattern} placeholder="e.g. coding|code" />
          </div>
          <div class="form-row">
            <label class="form-label" for="rule-target">Target Slot ID</label>
            <input id="rule-target" class="input" bind:value={newRule.target_slot} placeholder="e.g. slot-1" />
          </div>
          <div class="form-row">
            <label class="form-label" for="rule-priority">Priority (higher = checked first)</label>
            <input id="rule-priority" class="input" type="number" bind:value={newRule.priority} min={0} />
          </div>
          <button class="btn btn-primary btn-sm" onclick={handleAddRule}>Add Rule</button>
        </div>
      {/if}

      {#if rules.length === 0}
        <div class="empty-state">No routing rules configured. All requests go to the default model.</div>
      {:else}
        <div class="rules-table">
          <div class="table-header">
            <span>Pattern</span>
            <span>Target Slot</span>
            <span>Priority</span>
            <span></span>
          </div>
          {#each rules as rule (rule.id)}
            <div class="table-row">
              <span class="mono">{rule.pattern}</span>
              <span class="mono">{rule.target_slot}</span>
              <span>{rule.priority}</span>
              <button class="btn-icon" onclick={() => handleRemoveRule(rule.id)} title="Remove">✕</button>
            </div>
          {/each}
        </div>
      {/if}
    </div>
  {/if}
</div>

<style>
  .multimodel-page { max-width: 900px; }
  .slots-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 12px; }
  .slot-card {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px;
    display: flex;
    flex-direction: column;
    gap: 10px;
  }
  .slot-card.ready { border-color: color-mix(in srgb, var(--success) 30%, transparent); }
  .slot-card.error { border-color: color-mix(in srgb, var(--danger) 30%, transparent); }
  .slot-header { display: flex; justify-content: space-between; align-items: center; }
  .slot-name { display: flex; align-items: center; gap: 8px; font-weight: 600; font-size: 13px; }
  .slot-status {
    font-size: 10px;
    padding: 2px 8px;
    border-radius: 4px;
    font-weight: 600;
    text-transform: uppercase;
  }
  .slot-status.ready { background: color-mix(in srgb, var(--success) 20%, transparent); color: var(--success); }
  .slot-status.loading { background: color-mix(in srgb, var(--text-muted) 20%, transparent); color: var(--text-muted); }
  .slot-status.error { background: color-mix(in srgb, var(--danger) 20%, transparent); color: var(--danger); }
  .slot-status.unloaded { background: color-mix(in srgb, var(--text-muted) 20%, transparent); color: var(--text-muted); }
  .slot-stats { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
  .stat { display: flex; flex-direction: column; }
  .stat-label { font-size: 11px; color: var(--text-muted); text-transform: uppercase; }
  .stat-value { font-size: 13px; font-family: var(--font-mono); font-weight: 600; }
  .slot-error { font-size: 12px; color: var(--danger); }
  .btn-sm { padding: 6px 12px; font-size: 12px; }
  .loading-bar { height: 4px; background: var(--bg-input); border-radius: 2px; overflow: hidden; }
  .loading-fill { height: 100%; background: var(--accent); animation: loading-progress 1.5s ease infinite; }
  @keyframes loading-progress {
    0% { width: 0; }
    50% { width: 70%; }
    100% { width: 100%; }
  }
  .card-header-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }
  .add-rule-form { background: var(--bg-secondary); border: 1px solid var(--border); border-radius: 8px; padding: 12px; margin-bottom: 12px; }
  .rules-table { font-size: 13px; }
  .table-header, .table-row { display: grid; grid-template-columns: 2fr 1.5fr 0.8fr 0.5fr; gap: 8px; padding: 8px 0; align-items: center; }
  .table-header { color: var(--text-muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; border-bottom: 1px solid var(--border); }
  .table-row { border-bottom: 1px solid color-mix(in srgb, var(--border) 50%, transparent); }
  .btn-icon { background: none; border: none; color: var(--text-muted); cursor: pointer; font-size: 14px; padding: 4px; }
  .btn-icon:hover { color: var(--danger); }
</style>
