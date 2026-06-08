/**
 * Model Optimization Service — recommends optimal model config for user hardware.
 *
 * Features:
 * - GPU/VRAM input → top 3 model recommendations
 * - Quantization recommendation with reasoning
 * - "What if I add more GPUs?" scaling slider
 * - CLI config generation
 * - Download config as JSON
 *
 * Usage:
 *   <div id="modelOptimizer"></div>
 *   <script type="module">
 *     import { initModelOptimizer } from './js/model-optimizer.js';
 *     initModelOptimizer();
 *   </script>
 */

import { escapeHtml } from './utils.js';

// ── Hardware Database ────────────────────────────────────────────
const GPUS = [
  { id: 'rtx-4090', name: 'RTX 4090', vram: 24, tflops: 82.6, price: 1600 },
  { id: 'rtx-4080', name: 'RTX 4080', vram: 16, tflops: 48.7, price: 1200 },
  { id: 'rtx-4070-ti', name: 'RTX 4070 Ti', vram: 12, tflops: 40.1, price: 800 },
  { id: 'rtx-4070', name: 'RTX 4070', vram: 12, tflops: 29.1, price: 550 },
  { id: 'rtx-4060', name: 'RTX 4060', vram: 8, tflops: 22.1, price: 300 },
  { id: 'rtx-3090', name: 'RTX 3090', vram: 24, tflops: 35.6, price: 1500 },
  { id: 'rtx-3080', name: 'RTX 3080', vram: 10, tflops: 29.8, price: 700 },
  { id: 'rtx-3070', name: 'RTX 3070', vram: 8, tflops: 20.3, price: 500 },
  { id: 'rtx-3060', name: 'RTX 3060', vram: 12, tflops: 12.7, price: 330 },
  { id: 'a100-80', name: 'A100 80GB', vram: 80, tflops: 312, price: 15000 },
  { id: 'h100', name: 'H100 80GB', vram: 80, tflops: 989, price: 30000 },
  { id: 'a6000', name: 'RTX A6000', vram: 48, tflops: 38.7, price: 4500 },
  { id: 'mac-m2-ultra', name: 'M2 Ultra (76-core)', vram: 192, tflops: 27.2, price: 7000 },
  { id: 'mac-m3-max', name: 'M3 Max (40-core)', vram: 64, tflops: 18.1, price: 3500 },
  { id: 'arc-a770', name: 'Arc A770 16GB', vram: 16, tflops: 19.7, price: 350 },
];

// ── Model Requirements Database ─────────────────────────────────
const MODELS = [
  { id: 'llama-3-8b', name: 'Llama 3.1 8B', provider: 'Meta', fp16: 16, int8: 8, int4: 4, baseTPS: 45, useCase: ['chat', 'code', 'rag'] },
  { id: 'llama-3-70b', name: 'Llama 3.1 70B', provider: 'Meta', fp16: 140, int8: 70, int4: 35, baseTPS: 12, useCase: ['chat', 'code', 'rag', 'agent'] },
  { id: 'qwen-2.5-7b', name: 'Qwen 2.5 7B', provider: 'Alibaba', fp16: 14, int8: 7, int4: 3.5, baseTPS: 55, useCase: ['chat', 'code', 'rag'] },
  { id: 'qwen-2.5-32b', name: 'Qwen 2.5 32B', provider: 'Alibaba', fp16: 64, int8: 32, int4: 16, baseTPS: 20, useCase: ['chat', 'code', 'rag', 'agent'] },
  { id: 'mistral-7b', name: 'Mistral 7B v0.3', provider: 'Mistral', fp16: 14, int8: 7, int4: 3.5, baseTPS: 50, useCase: ['chat', 'code', 'rag'] },
  { id: 'mixtral-8x7b', name: 'Mixtral 8x7B', provider: 'Mistral', fp16: 90, int8: 45, int4: 22, baseTPS: 25, useCase: ['chat', 'rag', 'agent'] },
  { id: 'phi-3', name: 'Phi-3 Mini', provider: 'Microsoft', fp16: 8, int8: 4, int4: 2, baseTPS: 70, useCase: ['chat', 'code'] },
  { id: 'deepseek-v2-lite', name: 'DeepSeek V2 Lite', provider: 'DeepSeek', fp16: 32, int8: 16, int4: 8, baseTPS: 35, useCase: ['chat', 'code', 'rag'] },
  { id: 'gemma-2-27b', name: 'Gemma 2 27B', provider: 'Google', fp16: 54, int8: 27, int4: 14, baseTPS: 22, useCase: ['chat', 'rag'] },
  { id: 'falcon-2-11b', name: 'Falcon 2 11B', provider: 'TII', fp16: 22, int8: 11, int4: 5.5, baseTPS: 40, useCase: ['chat', 'rag'] },
];

// ── Recommendation Engine ────────────────────────────────────────
function getRecommendations(gpuModel, gpuCount, useCase, qualityPref) {
  const gpu = GPUS.find(g => g.id === gpuModel);
  if (!gpu) return [];
  const totalVram = gpu.vram * gpuCount;

  const scored = MODELS
    .map(m => {
      // Check if model fits in any quantization
      const options = [];
      if (m.fp16 <= totalVram) options.push({ quant: 'FP16', vram: m.fp16, quality: 100, speedFactor: 1.0 });
      if (m.int8 <= totalVram) options.push({ quant: 'INT8', vram: m.int8, quality: 96, speedFactor: 1.1 });
      if (m.int4 <= totalVram) options.push({ quant: 'INT4', vram: m.int4, quality: 90, speedFactor: 1.3 });

      if (options.length === 0) return null;

      // Pick best quantization for quality preference
      const best = qualityPref === 'speed'
        ? options[options.length - 1] // lowest quality, highest speed
        : qualityPref === 'quality'
        ? options[0] // highest quality
        : options[Math.floor(options.length / 2)]; // balanced

      const tps = m.baseTPS * best.speedFactor * (1 + (gpuCount - 1) * 0.6);
      const score = best.quality * tps;
      const fits = best.vram <= totalVram;
      const vramUtil = ((best.vram / totalVram) * 100).toFixed(0);

      return { ...m, bestQuant: best.quant, tps: Math.round(tps), quality: best.quality, score, fits, vramUtil, vramUsed: best.vram };
    })
    .filter(Boolean)
    .filter(m => useCase === 'any' || m.useCase.includes(useCase))
    .sort((a, b) => b.score - a.score);

  return scored.slice(0, 3);
}

// ── UI ───────────────────────────────────────────────────────────
export function initModelOptimizer() {
  const container = document.getElementById('modelOptimizer');
  if (!container) return;

  container.innerHTML = `
    <div class="optimizer-card">
      <div class="optimizer-header">
        <h3>Model Optimizer</h3>
        <span class="optimizer-badge">Find your perfect config</span>
      </div>
      <div class="optimizer-layout">
        <div class="optimizer-form" id="optForm">
          <div class="opt-field">
            <label class="opt-label">GPU Model</label>
            <select class="opt-select" id="optGpu">
              ${GPUS.map(g => `<option value="${g.id}">${g.name} (${g.vram}GB, ~$${g.price})</option>`).join('')}
            </select>
          </div>
          <div class="opt-field">
            <label class="opt-label">Number of GPUs: <strong id="optGpuCountLabel">1</strong></label>
            <input type="range" class="opt-slider" id="optGpuCount" min="1" max="8" step="1" value="1" aria-label="Number of GPUs">
          </div>
          <div class="opt-field">
            <label class="opt-label">Use Case</label>
            <select class="opt-select" id="optUseCase">
              <option value="any">Any</option>
              <option value="chat">Chat</option>
              <option value="code">Code</option>
              <option value="rag">RAG</option>
              <option value="agent">Agent</option>
            </select>
          </div>
          <div class="opt-field">
            <label class="opt-label">Priority</label>
            <div class="opt-priority" id="optPriority">
              <button class="opt-prio-btn" data-val="speed">Speed</button>
              <button class="opt-prio-btn active" data-val="balanced">Balanced</button>
              <button class="opt-prio-btn" data-val="quality">Quality</button>
            </div>
          </div>
          <button class="btn btn-primary opt-submit" id="optSubmit">Analyze Hardware</button>
        </div>
        <div class="optimizer-results" id="optResults">
          <div class="opt-placeholder">Enter your hardware specs above and click Analyze to see recommended configurations.</div>
        </div>
      </div>
    </div>
  `;

  // Event handlers
  document.getElementById('optGpuCount').addEventListener('input', () => {
    document.getElementById('optGpuCountLabel').textContent = document.getElementById('optGpuCount').value;
  });

  document.querySelectorAll('.opt-prio-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.opt-prio-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
    });
  });

  document.getElementById('optSubmit').addEventListener('click', analyze);

  function analyze() {
    const gpuModel = document.getElementById('optGpu').value;
    const gpuCount = parseInt(document.getElementById('optGpuCount').value, 10);
    const useCase = document.getElementById('optUseCase').value;
    const qualityPref = document.querySelector('.opt-prio-btn.active')?.dataset.val || 'balanced';

    const results = getRecommendations(gpuModel, gpuCount, useCase, qualityPref);
    const resultsDiv = document.getElementById('optResults');
    const gpu = GPUS.find(g => g.id === gpuModel);

    if (results.length === 0) {
      resultsDiv.innerHTML = `<div class="opt-empty">No compatible models found for this hardware. Try adding more GPUs or selecting a different use case.</div>`;
      return;
    }

    const totalVram = gpu.vram * gpuCount;
    resultsDiv.innerHTML = `
      <div class="opt-summary">
        <span class="opt-summary-label">Hardware: <strong>${escapeHtml(gpu.name)} × ${gpuCount}</strong> (${totalVram}GB total)</span>
        <span class="opt-summary-label">Estimated GPU cost: <strong>$${(gpu.price * gpuCount).toLocaleString()}</strong></span>
      </div>
      <div class="opt-cards">
        ${results.map((r, i) => `
          <div class="opt-card ${i === 0 ? 'opt-recommended' : ''}">
            ${i === 0 ? '<div class="opt-badge-recommended">★ Best Match</div>' : ''}
            <div class="opt-card-header">
              <h4>${escapeHtml(r.name)}</h4>
              <span class="opt-provider">${escapeHtml(r.provider)}</span>
            </div>
            <div class="opt-card-specs">
              <div class="opt-spec"><span>Quantization</span><strong>${r.bestQuant}</strong></div>
              <div class="opt-spec"><span>VRAM Used</span><strong>${r.vramUsed}GB / ${totalVram}GB (${r.vramUtil}%)</strong></div>
              <div class="opt-spec"><span>Throughput</span><strong>~${r.tps} tok/s</strong></div>
              <div class="opt-spec"><span>Quality</span><strong>${r.quality}%</strong></div>
            </div>
            <div class="opt-vram-bar">
              <div class="opt-vram-fill" style="width:${Math.min(r.vramUtil, 100)}%"></div>
            </div>
            <div class="opt-card-actions">
              <button class="opt-copy-btn" data-config='${JSON.stringify({ model: r.id, quant: r.bestQuant, gpus: gpuCount })}'>Copy CLI Config</button>
            </div>
          </div>
        `).join('')}
      </div>
      <div class="opt-scaling">
        <h4>What if I add more GPUs?</h4>
        <p>With ${gpuCount} GPU${gpuCount > 1 ? 's' : ''}: top model is <strong>${escapeHtml(results[0].name)}</strong> at ~${results[0].tps} tok/s</p>
        <p>With ${gpuCount + 1} GPUs: you could run <strong>${escapeHtml(getRecommendations(gpuModel, gpuCount + 1, useCase, qualityPref)[0]?.name || 'the same model')}</strong></p>
        <p>With ${gpuCount + 2} GPUs: you could run <strong>${escapeHtml(getRecommendations(gpuModel, gpuCount + 2, useCase, qualityPref)[0]?.name || 'the same model')}</strong></p>
      </div>
    `;

    // Copy CLI config buttons
    resultsDiv.querySelectorAll('.opt-copy-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const config = JSON.parse(btn.dataset.config);
        const cli = `distllm deploy --hf ${config.model} --quantization bitsandbytes_${config.quant.toLowerCase()} --nodes ${config.gpus}`;
        navigator.clipboard.writeText(cli).then(() => {
          btn.textContent = 'Copied!';
          setTimeout(() => { btn.textContent = 'Copy CLI Config'; }, 2000);
        });
      });
    });
  }
}
