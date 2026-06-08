<script lang="ts">
  import type { InferenceMetrics } from "../types";

  let { metrics, streaming }: { metrics: InferenceMetrics | null; streaming: boolean } = $props();
</script>

{#if metrics || streaming}
  <div class="metrics-bar">
    {#if streaming}
      <div class="metric">
        <span class="metric-label">Generating</span>
        <span class="metric-value streaming-indicator">●●●</span>
      </div>
    {/if}
    {#if metrics}
      <div class="metric">
        <span class="metric-label">TTFT</span>
        <span class="metric-value">{metrics.ttft !== null ? metrics.ttft.toFixed(0) + "ms" : "—"}</span>
      </div>
      <div class="metric">
        <span class="metric-label">tok/s</span>
        <span class="metric-value">{metrics.tokens_per_sec.toFixed(1)}</span>
      </div>
      <div class="metric">
        <span class="metric-label">inter-token</span>
        <span class="metric-value">{metrics.inter_token_latency.toFixed(1)}ms</span>
      </div>
      <div class="metric">
        <span class="metric-label">tokens</span>
        <span class="metric-value">{metrics.total_tokens}</span>
      </div>
      <div class="metric">
        <span class="metric-label">time</span>
        <span class="metric-value">{metrics.total_time.toFixed(1)}s</span>
      </div>
    {/if}
  </div>
{/if}

<style>
  .metrics-bar {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 6px 12px;
    background: var(--bg-secondary);
    border-top: 1px solid var(--border);
    font-size: 11px;
    font-family: var(--font-mono);
    flex-shrink: 0;
  }
  .metric {
    display: flex;
    align-items: center;
    gap: 4px;
  }
  .metric-label {
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  .metric-value {
    color: var(--accent);
    font-weight: 600;
  }
  .streaming-indicator {
    color: var(--success);
    animation: blink 1s steps(1) infinite;
  }
  @keyframes blink {
    50% { opacity: 0.3; }
  }
</style>
