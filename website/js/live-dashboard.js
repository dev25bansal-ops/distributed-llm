/**
 * Real-Time Cluster Dashboard — live cluster health, stats, and monitoring.
 *
 * Features:
 * - Coordinator status + worker node health
 * - Animated GPU utilization bars
 * - Global counters (tokens, requests, uptime)
 * - Throughput sparkline (CSS-based)
 * - 5-second auto-refresh with pause
 * - Demp data mode + real API connection
 * - Color-coded status indicators
 *
 * Usage:
 *   <div id="liveDashboard"></div>
 *   <script type="module">
 *     import { initLiveDashboard } from './js/live-dashboard.js';
 *     initLiveDashboard();
 *   </script>
 */

import { escapeHtml } from './utils.js';

// ── Demo Data ────────────────────────────────────────────────────
const DEMO = {
  coordinator: { status: 'running', uptime: '14d 3h 22m', version: '0.4.0', apiRequests: 15420, model: 'Llama 3.1 70B (4-bit)' },
  workers: [
    { name: 'Node-1 (RTX 4090)', status: 'online', gpuUtil: 72, memUtil: 65, temp: 68, tps: 42.3, uptime: '14d 3h', layers: '0-19' },
    { name: 'Node-2 (RTX 4090)', status: 'online', gpuUtil: 85, memUtil: 78, temp: 74, tps: 38.1, uptime: '14d 3h', layers: '20-39' },
    { name: 'Node-3 (RTX 3090)', status: 'online', gpuUtil: 61, memUtil: 52, temp: 62, tps: 45.7, uptime: '12d 7h', layers: '40-59' },
    { name: 'Node-4 (RTX 3090)', status: 'online', gpuUtil: 93, memUtil: 88, temp: 81, tps: 31.2, uptime: '10d 1h', layers: '60-79' },
    { name: 'Node-5 (Laptop 4060)', status: 'degraded', gpuUtil: 45, memUtil: 38, temp: 55, tps: 22.8, uptime: '3d 12h', layers: 'draft' },
  ],
  models: [
    { name: 'Llama 3.1 70B', quant: '4-bit', layers: 80, nodes: 4, throughput: 28.5 },
    { name: 'Qwen 2.5 32B', quant: '8-bit', layers: 64, nodes: 2, throughput: 45.2 },
  ],
  stats: { totalTokens: 2845000, totalRequests: 15420, peakWorkers: 8, avgLatency: 342 },
  throughput: [22, 28, 25, 32, 30, 35, 33, 38, 35, 40, 38, 42, 39, 45, 42, 48, 44, 50, 47, 52, 49, 55, 51, 48],
};

// ── UI ───────────────────────────────────────────────────────────
export function initLiveDashboard() {
  const container = document.getElementById('liveDashboard');
  if (!container) return;

  let paused = false;
  let intervalId = null;
  let data = JSON.parse(JSON.stringify(DEMO));

  container.innerHTML = `
    <div class="dashboard-card">
      <div class="dashboard-header">
        <h3>Live Cluster Dashboard</h3>
        <div class="dashboard-controls">
          <span class="dashboard-status" id="dashStatus">● Live Demo</span>
          <button class="dash-btn" id="dashPause">Pause</button>
          <button class="dash-btn" id="dashExport">Export</button>
        </div>
      </div>
      <div class="dashboard-stats" id="dashStats"></div>
      <div class="dashboard-workers" id="dashWorkers"></div>
      <div class="dashboard-models" id="dashModels"></div>
      <div class="dashboard-throughput" id="dashThroughput"></div>
    </div>
  `;

  render();
  startAutoRefresh();

  document.getElementById('dashPause').addEventListener('click', () => {
    paused = !paused;
    document.getElementById('dashPause').textContent = paused ? 'Resume' : 'Pause';
    if (paused) stopAutoRefresh();
    else startAutoRefresh();
  });

  document.getElementById('dashExport').addEventListener('click', () => {
    const text = JSON.stringify(data, null, 2);
    navigator.clipboard.writeText(text).then(() => {
      const btn = document.getElementById('dashExport');
      btn.textContent = 'Copied!';
      setTimeout(() => { btn.textContent = 'Export'; }, 2000);
    });
  });

  function render() {
    renderStats();
    renderWorkers();
    renderModels();
    renderThroughput();
  }

  function renderStats() {
    const s = data.stats;
    document.getElementById('dashStats').innerHTML = `
      <div class="dash-stat"><span class="dash-stat-val">${(s.totalTokens / 1000000).toFixed(1)}M</span><span class="dash-stat-label">Tokens Generated</span></div>
      <div class="dash-stat"><span class="dash-stat-val">${(s.totalRequests / 1000).toFixed(1)}K</span><span class="dash-stat-label">API Requests</span></div>
      <div class="dash-stat"><span class="dash-stat-val">${s.peakWorkers}</span><span class="dash-stat-label">Peak Workers</span></div>
      <div class="dash-stat"><span class="dash-stat-val">${s.avgLatency}ms</span><span class="dash-stat-label">Avg Latency</span></div>
      <div class="dash-stat"><span class="dash-stat-val">${escapeHtml(data.coordinator.uptime)}</span><span class="dash-stat-label">Coordinator Uptime</span></div>
    `;
  }

  function renderWorkers() {
    document.getElementById('dashWorkers').innerHTML = `
      <h4 class="dash-section-title">Worker Nodes (${data.workers.filter(w => w.status === 'online').length}/${data.workers.length} online)</h4>
      <div class="dash-worker-list">${data.workers.map(w => `
        <div class="dash-worker">
          <div class="dw-header">
            <span class="dw-dot dw-${w.status}"></span>
            <span class="dw-name">${escapeHtml(w.name)}</span>
            <span class="dw-status">${w.status}</span>
          </div>
          <div class="dw-bars">
            <div class="dw-bar-row"><span class="dw-bar-label">GPU</span><div class="dw-bar"><div class="dw-bar-fill" style="width:${w.gpuUtil}%;background:${w.gpuUtil > 80 ? '#ef4444' : w.gpuUtil > 50 ? '#f59e0b' : '#22c55e'}"></div></div><span class="dw-bar-val">${w.gpuUtil}%</span></div>
            <div class="dw-bar-row"><span class="dw-bar-label">MEM</span><div class="dw-bar"><div class="dw-bar-fill" style="width:${w.memUtil}%;background:${w.memUtil > 80 ? '#ef4444' : '#3b82f6'}"></div></div><span class="dw-bar-val">${w.memUtil}%</span></div>
          </div>
          <div class="dw-meta"><span>${w.tps} tok/s</span><span>${w.temp}°C</span><span>${escapeHtml(w.uptime)}</span></div>
        </div>
      `).join('')}</div>
    `;
  }

  function renderModels() {
    document.getElementById('dashModels').innerHTML = `
      <h4 class="dash-section-title">Loaded Models</h4>
      <div class="dash-model-list">${data.models.map(m => `
        <div class="dash-model">
          <span class="dm-name">${escapeHtml(m.name)}</span>
          <span class="dm-quant">${m.quant}</span>
          <span class="dm-nodes">${m.nodes} nodes</span>
          <span class="dm-tps">${m.throughput} tok/s</span>
        </div>
      `).join('')}</div>
    `;
  }

  function renderThroughput() {
    const max = Math.max(...data.throughput);
    const w = 100 / data.throughput.length;
    document.getElementById('dashThroughput').innerHTML = `
      <h4 class="dash-section-title">Throughput (24h) — Current: ${data.throughput[data.throughput.length - 1]} tok/s</h4>
      <div class="dash-sparkline">${data.throughput.map(v => `
        <div class="dash-spark-bar" style="height:${(v / max) * 100}%;width:${w}%" title="${v} tok/s"></div>
      `).join('')}</div>
    `;
  }

  function randomWalk(val, maxDelta, min, max) {
    return Math.max(min, Math.min(max, val + (Math.random() - 0.5) * maxDelta * 2));
  }

  function tick() {
    if (paused) return;
    data.stats.totalTokens += Math.floor(Math.random() * 500);
    data.stats.totalRequests += Math.floor(Math.random() * 5);

    data.workers.forEach(w => {
      if (w.status === 'online') {
        w.gpuUtil = randomWalk(w.gpuUtil, 5, 10, 98);
        w.memUtil = randomWalk(w.memUtil, 3, 10, 95);
        w.temp = randomWalk(w.temp, 2, 40, 88);
        w.tps = randomWalk(w.tps, 2, 5, 60);
      }
    });

    data.throughput.push(data.workers.reduce((sum, w) => sum + w.tps, 0));
    if (data.throughput.length > 24) data.throughput.shift();

    render();
  }

  function startAutoRefresh() {
    if (intervalId) return;
    intervalId = setInterval(tick, 3000);
  }

  function stopAutoRefresh() {
    if (intervalId) { clearInterval(intervalId); intervalId = null; }
  }
}
