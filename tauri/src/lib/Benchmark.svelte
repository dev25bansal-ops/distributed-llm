<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import { benchmarkStore, clusterStore } from "./stores";
  import { ErrorBanner } from "./ui";
  import type { BenchmarkRun, BenchmarkConfig, ClusterStatus } from "./types";

  let runs = $state<BenchmarkRun[]>([]);
  let running = $state(false);
  let error = $state<string | null>(null);
  let cluster = $state<ClusterStatus | null>(null);

  let config = $state<BenchmarkConfig>({
    model: "HuggingFaceTB/SmolLM-135M",
    prompt_length: 200,
    max_tokens: 128,
    num_runs: 3,
    quantization: "fp16",
  });

  let sortBy = $state<"tokens_per_sec" | "ttft" | "timestamp">("timestamp");
  let sortDir = $state<"asc" | "desc">("desc");

  let unsubscribe: (() => void) | undefined;
  let unsubCluster: (() => void) | undefined;

  onMount(() => {
    unsubscribe = benchmarkStore.subscribe((d) => {
      runs = d.runs;
      running = d.running;
      error = d.error;
    });
    unsubCluster = clusterStore.subscribe((d) => {
      cluster = d.cluster;
    });
  });

  onDestroy(() => {
    unsubscribe?.();
    unsubCluster?.();
  });

  async function handleRun() {
    const addr = cluster?.coordinator_addr;
    if (!addr) {
      error = "No cluster running. Start a cluster first.";
      return;
    }
    await benchmarkStore.start(config, addr);
  }

  function handleStop() {
    benchmarkStore.stop();
  }

  function handleClear() {
    if (window.confirm("Clear all benchmark results?")) {
      benchmarkStore.clear();
    }
  }

  let sorted = $derived(
    [...runs].sort((a, b) => {
      const mul = sortDir === "asc" ? 1 : -1;
      return (a[sortBy] - b[sortBy]) * mul;
    }),
  );

  let avgByModel = $derived(() => {
    const grouped = new Map<string, BenchmarkRun[]>();
    for (const r of runs) {
      const key = r.model;
      if (!grouped.has(key)) grouped.set(key, []);
      grouped.get(key)!.push(r);
    }
    const avgs: { model: string; avg_tps: number; avg_ttft: number; runs: number }[] = [];
    for (const [model, items] of grouped) {
      const valid = items.filter((r) => r.tokens_per_sec > 0);
      if (valid.length === 0) continue;
      avgs.push({
        model,
        avg_tps: valid.reduce((s, r) => s + r.tokens_per_sec, 0) / valid.length,
        avg_ttft: valid.reduce((s, r) => s + r.ttft, 0) / valid.length,
        runs: valid.length,
      });
    }
    return avgs.sort((a, b) => b.avg_tps - a.avg_tps);
  });

  function toggleSort(col: typeof sortBy) {
    if (sortBy === col) {
      sortDir = sortDir === "asc" ? "desc" : "asc";
    } else {
      sortBy = col;
      sortDir = "desc";
    }
  }
</script>

<div class="benchmark-page">
  <h1 class="page-title">Benchmark</h1>

  <ErrorBanner message={error ?? ""} ondismiss={() => (error = null)} />

  <div class="config-card card">
    <h2 class="card-title">Run Benchmark</h2>
    <p class="card-desc">Test inference speed across different configurations.</p>

    <div class="config-grid">
      <div class="form-row">
        <label class="form-label" for="bench-model">Model</label>
        <input id="bench-model" class="input" bind:value={config.model} />
      </div>
      <div class="form-row">
        <label class="form-label" for="bench-prompt">Prompt Length (chars)</label>
        <input id="bench-prompt" class="input" type="number" bind:value={config.prompt_length} min={50} max={2000} />
      </div>
      <div class="form-row">
        <label class="form-label" for="bench-tokens">Max Tokens</label>
        <input id="bench-tokens" class="input" type="number" bind:value={config.max_tokens} min={16} max={4096} />
      </div>
      <div class="form-row">
        <label class="form-label" for="bench-runs">Runs</label>
        <input id="bench-runs" class="input" type="number" bind:value={config.num_runs} min={1} max={20} />
      </div>
      <div class="form-row">
        <label class="form-label" for="bench-quant">Quantization</label>
        <select id="bench-quant" class="input" bind:value={config.quantization}>
          <option value="fp16">FP16</option>
          <option value="int8">INT8</option>
          <option value="int4">INT4</option>
        </select>
      </div>
    </div>

    <div class="button-row">
      {#if running}
        <button class="btn btn-danger" onclick={handleStop}>Stop</button>
      {:else}
        <button
          class="btn btn-primary"
          onclick={handleRun}
          disabled={!cluster?.running}
        >
          Run Benchmark
        </button>
      {/if}
      <button class="btn btn-ghost" onclick={handleClear} disabled={runs.length === 0}>
        Clear Results
      </button>
    </div>

    {#if !cluster?.running}
      <p class="empty-state" style="margin-top: 8px;">Start a cluster to run benchmarks.</p>
    {/if}
  </div>

  {#if avgByModel().length > 0}
    <div class="card">
      <h2 class="card-title">Summary by Model</h2>
      <div class="summary-grid">
        {#each avgByModel() as avg (avg.model)}
          <div class="summary-card">
            <div class="summary-model">{avg.model.split("/").pop()}</div>
            <div class="summary-stats">
              <div class="stat">
                <span class="stat-value">{avg.avg_tps.toFixed(1)}</span>
                <span class="stat-label">tok/s</span>
              </div>
              <div class="stat">
                <span class="stat-value">{avg.avg_ttft.toFixed(0)}ms</span>
                <span class="stat-label">TTFT</span>
              </div>
              <div class="stat">
                <span class="stat-value">{avg.runs}</span>
                <span class="stat-label">runs</span>
              </div>
            </div>
          </div>
        {/each}
      </div>
    </div>
  {/if}

  {#if sorted.length > 0}
    <div class="card">
      <h2 class="card-title">All Results</h2>
      <div class="results-table">
        <div class="table-header">
          <span class="col-model">Model</span>
          <button class="col-header" onclick={() => toggleSort("tokens_per_sec")}>
            tok/s {sortBy === "tokens_per_sec" ? (sortDir === "asc" ? "↑" : "↓") : ""}
          </button>
          <button class="col-header" onclick={() => toggleSort("ttft")}>
            TTFT {sortBy === "ttft" ? (sortDir === "asc" ? "↑" : "↓") : ""}
          </button>
          <span class="col-header">Tokens</span>
          <span class="col-header">Time</span>
          <span class="col-header">Quant</span>
          <button class="col-header" onclick={() => toggleSort("timestamp")}>
            Date {sortBy === "timestamp" ? (sortDir === "asc" ? "↑" : "↓") : ""}
          </button>
        </div>
        {#each sorted as run (run.id)}
          <div class="table-row" class:error={run.tokens_per_sec === 0}>
            <span class="col-model mono">{run.model.split("/").pop()}</span>
            <span class="col-val">{run.tokens_per_sec > 0 ? run.tokens_per_sec.toFixed(1) : "ERR"}</span>
            <span class="col-val">{run.ttft > 0 ? run.ttft.toFixed(0) + "ms" : "—"}</span>
            <span class="col-val">{run.completion_tokens}</span>
            <span class="col-val">{run.total_time.toFixed(1)}s</span>
            <span class="col-val mono">{run.quantization}</span>
            <span class="col-val">{new Date(run.timestamp).toLocaleTimeString()}</span>
          </div>
        {/each}
      </div>
    </div>
  {:else if !running}
    <div class="card">
      <div class="empty-state">No benchmark results yet. Configure and run a benchmark above.</div>
    </div>
  {/if}
</div>

<style>
  .benchmark-page { max-width: 900px; }
  .config-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    margin-bottom: 16px;
  }
  .button-row { display: flex; gap: 8px; }
  .summary-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 12px; }
  .summary-card {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px;
  }
  .summary-model { font-weight: 600; font-size: 14px; margin-bottom: 10px; }
  .summary-stats { display: flex; gap: 16px; }
  .stat { display: flex; flex-direction: column; }
  .stat-value { font-size: 16px; font-weight: 700; color: var(--accent); font-family: var(--font-mono); }
  .stat-label { font-size: 11px; color: var(--text-muted); text-transform: uppercase; }
  .results-table { font-size: 13px; }
  .table-header, .table-row { display: grid; grid-template-columns: 2fr 1fr 1fr 0.8fr 0.8fr 0.8fr 1fr; gap: 8px; padding: 8px 0; align-items: center; }
  .table-header { color: var(--text-muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; border-bottom: 1px solid var(--border); }
  .col-header { background: none; border: none; color: var(--text-muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; cursor: pointer; padding: 0; text-align: left; }
  .col-header:hover { color: var(--text-primary); }
  .table-row { border-bottom: 1px solid color-mix(in srgb, var(--border) 50%, transparent); }
  .table-row.error { opacity: 0.5; }
  .col-model { font-weight: 500; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .col-val { font-family: var(--font-mono); font-size: 12px; }
  select.input { appearance: auto; }
</style>
