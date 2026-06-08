<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import { clusterStore } from "./stores";
  import { ErrorBanner, StatusDot } from "./ui";
  import type { ClusterStatus, TopologyNode, TopologyLink } from "./types";
  import { buildTopologyFromCluster } from "./api";

  let cluster = $state<ClusterStatus | null>(null);
  let error = $state<string | null>(null);
  let canvasEl = $state<HTMLCanvasElement | null>(null);
  let containerEl = $state<HTMLDivElement | null>(null);

  let unsubscribe: (() => void) | undefined;

  let topology = $derived(buildTopologyFromCluster(cluster));
  let animFrame = $state(0);
  let flowOffset = $state(0);

  onMount(() => {
    unsubscribe = clusterStore.subscribe((d) => {
      cluster = d.cluster;
      if (d.error) error = d.error;
    });
    startAnimation();
  });

  onDestroy(() => {
    unsubscribe?.();
    cancelAnimationFrame(animFrame);
  });

  function startAnimation() {
    function frame() {
      flowOffset = (flowOffset + 0.5) % 20;
      draw();
      animFrame = requestAnimationFrame(frame);
    }
    animFrame = requestAnimationFrame(frame);
  }

  function draw() {
    const canvas = canvasEl;
    const container = containerEl;
    if (!canvas || !container) return;

    const rect = container.getBoundingClientRect();
    canvas.width = rect.width * devicePixelRatio;
    canvas.height = rect.height * devicePixelRatio;
    canvas.style.width = rect.width + "px";
    canvas.style.height = rect.height + "px";

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.scale(devicePixelRatio, devicePixelRatio);
    ctx.clearRect(0, 0, rect.width, rect.height);

    const nodes = topology.nodes;
    const links = topology.links;

    if (nodes.length === 0) {
      ctx.fillStyle = "#606080";
      ctx.font = "14px -apple-system, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("No cluster running. Start or join a cluster to see topology.", rect.width / 2, rect.height / 2);
      return;
    }

    // Layout: coordinator on left, workers in column on right
    const cx = 120;
    const cy = rect.height / 2;
    const wx = rect.width - 180;
    const workerSpacing = Math.min(100, (rect.height - 80) / Math.max(nodes.length - 1, 1));

    const positions = new Map<string, { x: number; y: number }>();
    positions.set("coordinator", { x: cx, y: cy });

    nodes.forEach((n, i) => {
      if (n.type === "worker") {
        const totalWorkers = nodes.filter((nd) => nd.type === "worker").length;
        const wi = nodes.filter((nd) => nd.type === "worker").indexOf(n);
        const startY = cy - ((totalWorkers - 1) * workerSpacing) / 2;
        positions.set(n.id, { x: wx, y: startY + wi * workerSpacing });
      }
    });

    // Draw links
    for (const link of links) {
      const s = positions.get(link.source);
      const t = positions.get(link.target);
      if (!s || !t) continue;

      ctx.beginPath();
      ctx.moveTo(s.x + 40, s.y);
      ctx.lineTo(t.x - 40, t.y);

      if (link.active) {
        ctx.strokeStyle = "rgba(108, 92, 231, 0.4)";
        ctx.lineWidth = 2;
        ctx.setLineDash([]);

        // Animated flow dots
        ctx.save();
        ctx.strokeStyle = "#6c5ce7";
        ctx.lineWidth = 2;
        ctx.setLineDash([4, 16]);
        ctx.lineDashOffset = -flowOffset;
        ctx.stroke();
        ctx.restore();
      } else {
        ctx.strokeStyle = "rgba(96, 96, 128, 0.3)";
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]);
        ctx.stroke();
      }
      ctx.setLineDash([]);
    }

    // Draw nodes
    for (const node of nodes) {
      const pos = positions.get(node.id);
      if (!pos) continue;

      const isCoord = node.type === "coordinator";
      const r = isCoord ? 36 : 30;

      // Glow
      if (node.healthy) {
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, r + 4, 0, Math.PI * 2);
        ctx.fillStyle = isCoord
          ? "rgba(108, 92, 231, 0.15)"
          : `rgba(34, 204, 102, ${0.05 + (node.gpu_utilization / 100) * 0.1})`;
        ctx.fill();
      }

      // Node circle
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, r, 0, Math.PI * 2);
      ctx.fillStyle = isCoord ? "#1e1e36" : "#1a1a2e";
      ctx.fill();
      ctx.strokeStyle = node.healthy
        ? isCoord
          ? "#6c5ce7"
          : utilColor(node.gpu_utilization)
        : "#ff4466";
      ctx.lineWidth = 2;
      ctx.stroke();

      // GPU utilization arc
      if (!isCoord && node.gpu_utilization > 0) {
        const angle = (node.gpu_utilization / 100) * Math.PI * 2;
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, r - 4, -Math.PI / 2, -Math.PI / 2 + angle);
        ctx.strokeStyle = utilColor(node.gpu_utilization);
        ctx.lineWidth = 3;
        ctx.stroke();
      }

      // Label
      ctx.fillStyle = "#e0e0f0";
      ctx.font = `bold ${isCoord ? 11 : 10}px -apple-system, sans-serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(isCoord ? "COORD" : node.gpu_name.split(" ").slice(-1)[0], pos.x, pos.y - 4);

      // Utilization text
      ctx.fillStyle = "#9090b0";
      ctx.font = "9px monospace";
      if (isCoord) {
        ctx.fillText(`${nodes.length - 1} nodes`, pos.x, pos.y + 10);
      } else {
        ctx.fillText(`${node.gpu_utilization.toFixed(0)}%`, pos.x, pos.y + 8);
        // Layer range
        ctx.fillStyle = "#606080";
        ctx.font = "8px monospace";
        ctx.fillText(`L${node.layers.start}-${node.layers.end}`, pos.x, pos.y + 18);
      }
    }
  }

  function utilColor(util: number): string {
    if (util > 80) return "#ff4466";
    if (util > 50) return "#ffaa33";
    return "#22cc66";
  }
</script>

<div class="topology-page">
  <h1 class="page-title">Cluster Topology</h1>

  <ErrorBanner message={error ?? ""} ondismiss={() => (error = null)} />

  <div class="topology-layout">
    <div class="canvas-container" bind:this={containerEl}>
      <canvas bind:this={canvasEl}></canvas>
    </div>

    <div class="topology-sidebar">
      <div class="card">
        <h2 class="card-title">Nodes</h2>
        {#if topology.nodes.length === 0}
          <div class="empty-state">No nodes in cluster.</div>
        {:else}
          {#each topology.nodes as node (node.id)}
            <div class="node-info">
              <div class="node-header-row">
                <StatusDot variant={node.healthy ? "green" : "red"} />
                <span class="node-label">{node.type === "coordinator" ? "Coordinator" : node.label}</span>
              </div>
              {#if node.type === "worker"}
                <div class="node-details">
                  <span class="detail">{node.gpu_name}</span>
                  <span class="detail">{node.gpu_utilization.toFixed(1)}% util</span>
                  <span class="detail mono">Layers {node.layers.start}-{node.layers.end}</span>
                  <span class="detail mono">{node.host}:{node.port}</span>
                </div>
              {:else}
                <div class="node-details">
                  <span class="detail mono">{node.host}:{node.port}</span>
                </div>
              {/if}
            </div>
          {/each}
        {/if}
      </div>

      <div class="card">
        <h2 class="card-title">Legend</h2>
        <div class="legend">
          <div class="legend-item">
            <span class="legend-dot" style="background: #6c5ce7;"></span>
            <span>Coordinator</span>
          </div>
          <div class="legend-item">
            <span class="legend-dot" style="background: #22cc66;"></span>
            <span>Healthy worker</span>
          </div>
          <div class="legend-item">
            <span class="legend-dot" style="background: #ff4466;"></span>
            <span>Unhealthy worker</span>
          </div>
          <div class="legend-item">
            <span class="legend-line animated"></span>
            <span>Active data flow</span>
          </div>
          <div class="legend-item">
            <span class="legend-line dashed"></span>
            <span>Inactive link</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>

<style>
  .topology-page { max-width: 1100px; }
  .topology-layout { display: flex; gap: 16px; }
  .canvas-container {
    flex: 1;
    min-height: 400px;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
  }
  .canvas-container canvas { display: block; }
  .topology-sidebar { width: 260px; flex-shrink: 0; }
  .node-info { margin-bottom: 12px; padding-bottom: 12px; border-bottom: 1px solid color-mix(in srgb, var(--border) 50%, transparent); }
  .node-info:last-child { border-bottom: none; margin-bottom: 0; padding-bottom: 0; }
  .node-header-row { display: flex; align-items: center; gap: 8px; margin-bottom: 4px; }
  .node-label { font-weight: 600; font-size: 13px; }
  .node-details { display: flex; flex-direction: column; gap: 2px; padding-left: 18px; }
  .detail { font-size: 12px; color: var(--text-secondary); }
  .legend { display: flex; flex-direction: column; gap: 8px; }
  .legend-item { display: flex; align-items: center; gap: 8px; font-size: 12px; color: var(--text-secondary); }
  .legend-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
  .legend-line { width: 24px; height: 2px; flex-shrink: 0; }
  .legend-line.animated { background: #6c5ce7; }
  .legend-line.dashed { background: repeating-linear-gradient(90deg, #606080 0, #606080 4px, transparent 4px, transparent 8px); }
</style>
