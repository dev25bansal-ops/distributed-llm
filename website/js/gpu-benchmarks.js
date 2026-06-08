/**
 * GPU Benchmark Suite — interactive GPU × Model performance comparison.
 *
 * Features:
 * - 10 GPUs × 9 Models compatibility matrix
 * - Sortable columns, filter by GPU/model
 * - Cluster mode: combine GPUs
 * - CSS bar charts, color-coded compatibility
 * - "What GPU do I need for [model]?" auto-filter
 *
 * Usage:
 *   <div id="gpuBenchmarks"></div>
 *   <script type="module">
 *     import { initGpuBenchmarks } from './js/gpu-benchmarks.js';
 *     initGpuBenchmarks();
 *   </script>
 */

import { escapeHtml } from './utils.js';

// ── GPU Database ─────────────────────────────────────────────────
const GPUS = [
  { id: 'h100', name: 'H100', vram: 80, price: 30000, tier: 'datacenter' },
  { id: 'a100', name: 'A100', vram: 80, price: 15000, tier: 'datacenter' },
  { id: 'rtx-4090', name: 'RTX 4090', vram: 24, price: 1600, tier: 'consumer' },
  { id: 'a6000', name: 'RTX A6000', vram: 48, price: 4500, tier: 'pro' },
  { id: 'rtx-3090', name: 'RTX 3090', vram: 24, price: 1500, tier: 'consumer' },
  { id: 'rtx-4080', name: 'RTX 4080', vram: 16, price: 1200, tier: 'consumer' },
  { id: 'mac-m3-max', name: 'M3 Max', vram: 64, price: 3500, tier: 'apple' },
  { id: 'mac-m2-ultra', name: 'M2 Ultra', vram: 192, price: 7000, tier: 'apple' },
  { id: 'rtx-4060', name: 'RTX 4060', vram: 8, price: 300, tier: 'consumer' },
  { id: 'arc-a770', name: 'Arc A770', vram: 16, price: 350, tier: 'consumer' },
];

// ── Model Requirements ───────────────────────────────────────────
const MODELS = [
  { id: 'phi-3', name: 'Phi-3 Mini 3.8B', fp16: 8, int8: 4, int4: 2, tps: 70 },
  { id: 'qwen-2.5-7b', name: 'Qwen 2.5 7B', fp16: 14, int8: 7, int4: 3.5, tps: 55 },
  { id: 'mistral-7b', name: 'Mistral 7B', fp16: 14, int8: 7, int4: 3.5, tps: 50 },
  { id: 'llama-3-8b', name: 'Llama 3.1 8B', fp16: 16, int8: 8, int4: 4, tps: 45 },
  { id: 'falcon-2-11b', name: 'Falcon 2 11B', fp16: 22, int8: 11, int4: 5.5, tps: 40 },
  { id: 'deepseek-v2', name: 'DeepSeek V2 16B', fp16: 32, int8: 16, int4: 8, tps: 35 },
  { id: 'gemma-2-27b', name: 'Gemma 2 27B', fp16: 54, int8: 27, int4: 14, tps: 22 },
  { id: 'mixtral-8x7b', name: 'Mixtral 8x7B', fp16: 90, int8: 45, int4: 22, tps: 25 },
  { id: 'llama-3-70b', name: 'Llama 3.1 70B', fp16: 140, int8: 70, int4: 35, tps: 12 },
];

// ── Compatibility Check ─────────────────────────────────────────
function checkCompatibility(gpuVram, modelReqs) {
  if (gpuVram >= modelReqs.fp16) return { status: 'full', quant: 'FP16', quality: '100%' };
  if (gpuVram >= modelReqs.int8) return { status: 'good', quant: 'INT8', quality: '96%' };
  if (gpuVram >= modelReqs.int4) return { status: 'partial', quant: 'INT4', quality: '90%' };
  return { status: 'no', quant: '-', quality: '-' };
}

// ── UI ───────────────────────────────────────────────────────────
export function initGpuBenchmarks() {
  const container = document.getElementById('gpuBenchmarks');
  if (!container) return;

  let clusterMode = false;
  let clusterGpuCount = 1;
  let sortCol = 'name';
  let sortDir = 1;
  let filterGpu = 'all';
  let filterModel = 'all';

  container.innerHTML = `
    <div class="benchmark-card">
      <div class="benchmark-header">
        <h3>GPU Benchmark Suite</h3>
        <div class="benchmark-controls">
          <label class="benchmark-toggle">
            <input type="checkbox" id="bmClusterToggle">
            <span>Cluster Mode</span>
          </label>
          <div class="bm-gpu-count" id="bmGpuCountWrap" style="display:none">
            <label>GPUs: <input type="number" id="bmGpuCount" value="2" min="2" max="16" class="bm-num"></label>
          </div>
        </div>
      </div>
      <div class="benchmark-filters">
        <select class="bm-filter" id="bmFilterGpu"><option value="all">All GPUs</option>${GPUS.map(g => `<option value="${g.id}">${g.name}</option>`).join('')}</select>
        <select class="bm-filter" id="bmFilterModel"><option value="all">All Models</option>${MODELS.map(m => `<option value="${m.id}">${m.name}</option>`).join('')}</select>
        <span class="bm-count" id="bmCount"></span>
      </div>
      <div class="benchmark-table-wrap">
        <table class="benchmark-table" id="bmTable">
          <thead><tr id="bmHeader"><th data-col="name">Model ↓<th data-col="tps">Tok/s<th data-col="fp16">FP16<th data-col="int8">INT8<th data-col="int4">INT4</tr></thead>
          <tbody id="bmBody"></tbody>
        </table>
      </div>
    </div>
  `;

  // Event handlers
  document.getElementById('bmClusterToggle').addEventListener('change', (e) => {
    clusterMode = e.target.checked;
    document.getElementById('bmGpuCountWrap').style.display = clusterMode ? 'flex' : 'none';
    render();
  });

  document.getElementById('bmGpuCount').addEventListener('input', (e) => {
    clusterGpuCount = Math.max(2, parseInt(e.target.value, 10) || 2);
    render();
  });

  document.getElementById('bmFilterGpu').addEventListener('change', (e) => { filterGpu = e.target.value; render(); });
  document.getElementById('bmFilterModel').addEventListener('change', (e) => { filterModel = e.target.value; render(); });

  document.getElementById('bmHeader').addEventListener('click', (e) => {
    const col = e.target.dataset.col;
    if (!col) return;
    if (sortCol === col) sortDir *= -1;
    else { sortCol = col; sortDir = 1; }
    render();
  });

  render();

  function getGpuVram(gpuId) {
    const gpu = GPUS.find(g => g.id === gpuId);
    if (!gpu) return 0;
    return clusterMode ? gpu.vram * clusterGpuCount : gpu.vram;
  }

  function render() {
    const body = document.getElementById('bmBody');
    const gpuses = filterGpu === 'all' ? GPUS : GPUS.filter(g => g.id === filterGpu);
    const models = filterModel === 'all' ? MODELS : MODELS.filter(m => m.id === filterModel);

    // Update header
    const headerCells = document.querySelectorAll('#bmHeader th');
    headerCells.forEach(th => {
      th.textContent = th.textContent.replace(/[↓↑]/g, '');
      if (th.dataset.col === sortCol) th.textContent += sortDir > 0 ? ' ↓' : ' ↑';
    });

    // Build rows
    const rows = [];
    for (const model of models) {
      for (const gpu of gpuses) {
        const vram = getGpuVram(gpu.id);
        const compat = checkCompatibility(vram, model);
        const name = gpu.id === filterGpu ? model.name : `${model.name} on ${gpu.name}`;
        const score = compat.status === 'full' ? 3 : compat.status === 'good' ? 2 : compat.status === 'partial' ? 1 : 0;
        rows.push({ name, score, compat: compat.status, quant: compat.quant, tps: model.tps, sortKey: sortCol === 'name' ? name : sortCol === 'tps' ? model.tps : compat.status === 'full' ? 100 : compat.status === 'good' ? 80 : compat.status === 'partial' ? 50 : 0 });
      }
    }

    // Sort
    rows.sort((a, b) => {
      const valA = a.sortKey;
      const valB = b.sortKey;
      if (typeof valA === 'string') return valA.localeCompare(valB) * sortDir;
      return (valA - valB) * sortDir;
    });

    document.getElementById('bmCount').textContent = `${rows.length} configurations`;

    body.innerHTML = rows.map(r => `
      <tr class="bm-row bm-${r.compat}">
        <td class="bm-name">${escapeHtml(r.name)}</td>
        <td class="bm-cell">~${r.tps}</td>
        <td class="bm-cell bm-qual"><span class="bm-dot bm-${r.compat === 'full' ? 'green' : r.compat === 'good' ? 'amber' : r.compat === 'partial' ? 'yellow' : 'gray'}"></span>${r.quant || '-'}</td>
        <td class="bm-cell"><div class="bm-bar"><div class="bm-bar-fill ${r.compat}" style="width:${r.compat === 'full' ? 100 : r.compat === 'good' ? 66 : r.compat === 'partial' ? 33 : 0}%"></div></div></td>
        <td class="bm-cell bm-status">${r.compat === 'no' ? '✗' : '✓'}</td>
      </tr>
    `).join('');
  }
}
