/**
 * Model Recommendation Engine — intelligent model + GPU + quantization selection.
 *
 * Takes user constraints (hardware, budget, use case) and recommends
 * the optimal model + quantization + GPU combo.
 *
 * Usage:
 *   <div id="modelRec"></div>
 *   <script type="module">
 *     import { initModelRec } from './js/model-rec.js';
 *     initModelRec();
 *   </script>
 */

// ── Knowledge Base ─────────────────────────────────────────────────────

const MODELS = [
    { name: 'Qwen2.5 0.5B', family: 'Qwen', params: 0.5, layers: 24, hidden: 896, vram_fp16: 1, vram_int8: 0.5, quality: 0.3, speed: 120, use: ['chat', 'simple'] },
    { name: 'Phi-3 mini 3.8B', family: 'Phi', params: 3.8, layers: 32, hidden: 2560, vram_fp16: 8, vram_int8: 4, quality: 0.6, speed: 80, use: ['chat', 'code', 'simple'] },
    { name: 'Llama 3.2 3B', family: 'Llama', params: 3, layers: 28, hidden: 3072, vram_fp16: 6, vram_int8: 3, quality: 0.55, speed: 90, use: ['chat', 'simple'] },
    { name: 'Mistral 7B v0.3', family: 'Mistral', params: 7, layers: 32, hidden: 4096, vram_fp16: 14, vram_int8: 7, quality: 0.7, speed: 55, use: ['chat', 'code', 'reasoning'] },
    { name: 'Llama 3.1 8B', family: 'Llama', params: 8, layers: 32, hidden: 4096, vram_fp16: 16, vram_int8: 8, quality: 0.75, speed: 50, use: ['chat', 'code', 'reasoning'] },
    { name: 'Qwen2.5 7B', family: 'Qwen', params: 7, layers: 28, hidden: 4096, vram_fp16: 14, vram_int8: 7, quality: 0.72, speed: 55, use: ['chat', 'code', 'reasoning', 'multilingual'] },
    { name: 'CodeLlama 13B', family: 'Llama', params: 13, layers: 40, hidden: 5120, vram_fp16: 26, vram_int8: 13, quality: 0.78, speed: 35, use: ['code'] },
    { name: 'Llama 3.1 13B', family: 'Llama', params: 13, layers: 40, hidden: 5120, vram_fp16: 26, vram_int8: 13, quality: 0.8, speed: 35, use: ['chat', 'code', 'reasoning'] },
    { name: 'CodeLlama 34B', family: 'Llama', params: 34, layers: 48, hidden: 8192, vram_fp16: 68, vram_int8: 34, quality: 0.85, speed: 25, use: ['code'] },
    { name: 'Mixtral 8x7B', family: 'Mistral', params: 47, layers: 32, hidden: 4096, vram_fp16: 94, vram_int8: 47, quality: 0.82, speed: 30, use: ['chat', 'code', 'reasoning'] },
    { name: 'Qwen2.5 72B', family: 'Qwen', params: 72, layers: 80, hidden: 8192, vram_fp16: 144, vram_int8: 72, quality: 0.92, speed: 15, use: ['chat', 'code', 'reasoning', 'multilingual'] },
    { name: 'Llama 3.1 70B', family: 'Llama', params: 70, layers: 80, hidden: 8192, vram_fp16: 140, vram_int8: 70, quality: 0.9, speed: 18, use: ['chat', 'code', 'reasoning'] },
    { name: 'Llama 3.1 405B', family: 'Llama', params: 405, layers: 126, hidden: 16384, vram_fp16: 810, vram_int8: 405, quality: 0.98, speed: 8, use: ['chat', 'code', 'reasoning'] },
];

const GPUS = [
    { name: 'RTX 4060', vram: 8, price: 300, band: 272, tier: 'consumer' },
    { name: 'RTX 4070 Ti', vram: 12, price: 600, band: 504, tier: 'consumer' },
    { name: 'RTX 4090', vram: 24, price: 1600, band: 1008, tier: 'consumer' },
    { name: 'RTX 3090', vram: 24, price: 800, band: 936, tier: 'consumer' },
    { name: 'A6000', vram: 48, price: 4500, band: 768, tier: 'pro' },
    { name: 'A100 40GB', vram: 40, price: 8000, band: 1555, tier: 'datacenter' },
    { name: 'A100 80GB', vram: 80, price: 15000, band: 2039, tier: 'datacenter' },
    { name: 'H100 80GB', vram: 80, price: 30000, band: 3352, tier: 'datacenter' },
];

const USE_CASES = {
    chat: { label: 'Chat / Assistant', weight: { quality: 0.4, speed: 0.3, cost: 0.3 } },
    code: { label: 'Code Generation', weight: { quality: 0.5, speed: 0.3, cost: 0.2 } },
    reasoning: { label: 'Reasoning / Analysis', weight: { quality: 0.6, speed: 0.2, cost: 0.2 } },
    simple: { label: 'Simple Tasks', weight: { quality: 0.2, speed: 0.5, cost: 0.3 } },
    multilingual: { label: 'Multilingual', weight: { quality: 0.5, speed: 0.3, cost: 0.2 } },
};

// ── Recommendation Engine ──────────────────────────────────────────────

function recommend(gpuNames, budget, useCase) {
    const gpus = gpuNames.map(n => GPUS.find(g => g.name === n)).filter(Boolean);
    if (!gpus.length) return [];

    const totalVram = gpus.reduce((s, g) => s + g.vram, 0);
    const totalCost = gpus.reduce((s, g) => s + g.price, 0);
    const weights = USE_CASES[useCase]?.weight || USE_CASES.chat.weight;

    const results = [];

    for (const model of MODELS) {
        // Check if model fits in VRAM with different quantizations
        const quants = [
            { name: 'FP16', vram: model.vram_fp16, quality: 1.0 },
            { name: 'INT8', vram: model.vram_int8, quality: 0.95 },
            { name: 'INT4', vram: model.vram_int8 / 2, quality: 0.85 },
        ];

        for (const q of quants) {
            if (q.vram > totalVram) continue;
            if (budget && totalCost > budget) continue;

            // Score
            const qualityScore = model.quality * q.quality;
            const speedScore = Math.min(model.speed / 100, 1);
            const costScore = 1 - (totalCost / 50000);

            // Use-case fitness
            const useFit = model.use.includes(useCase) ? 1.0 : 0.3;

            const score = (
                qualityScore * weights.quality +
                speedScore * weights.speed +
                costScore * weights.cost
            ) * useFit;

            const gpusNeeded = Math.ceil(q.vram / gpus[0].vram);

            results.push({
                model: model.name,
                quant: q.name,
                vramNeeded: q.vram,
                gpusNeeded,
                gpuName: gpus[0].name,
                quality: qualityScore,
                speed: model.speed,
                cost: totalCost,
                score,
                fits: true,
            });
        }
    }

    results.sort((a, b) => b.score - a.score);
    return results.slice(0, 5);
}

// ── UI ─────────────────────────────────────────────────────────────────

export function initModelRec() {
    const container = document.getElementById('modelRec');
    if (!container) return;

    const gpuOptions = GPUS.map(g => `<option value="${g.name}">${g.name} (${g.vram}GB, $${g.price})</option>`).join('');
    const useOptions = Object.entries(USE_CASES).map(([k, v]) => `<option value="${k}">${v.label}</option>`).join('');

    container.innerHTML = `
        <div class="model-rec-card">
            <h3>Model Recommendation Engine</h3>
            <p class="model-rec-desc">Tell us your hardware and use case. We'll recommend the optimal model.</p>
            <div class="model-rec-form">
                <div class="model-rec-row">
                    <label for="recGpu">Your GPU(s)</label>
                    <select id="recGpu" multiple>${gpuOptions}</select>
                </div>
                <div class="model-rec-row">
                    <label for="recGpuCount">Number of GPUs</label>
                    <input type="number" id="recGpuCount" value="1" min="1" max="16">
                </div>
                <div class="model-rec-row">
                    <label for="recUse">Use Case</label>
                    <select id="recUse">${useOptions}</select>
                </div>
                <div class="model-rec-row">
                    <label for="recBudget">Budget (USD, optional)</label>
                    <input type="number" id="recBudget" placeholder="e.g. 5000">
                </div>
                <button id="recBtn" class="model-rec-btn">Get Recommendations</button>
            </div>
            <div id="recResults" class="model-rec-results"></div>
        </div>
    `;

    document.getElementById('recBtn').addEventListener('click', () => {
        const gpuSelect = document.getElementById('recGpu');
        const gpuCount = parseInt(document.getElementById('recGpuCount').value) || 1;
        const useCase = document.getElementById('recUse').value;
        const budget = parseInt(document.getElementById('recBudget').value) || 0;

        const selectedGpu = gpuSelect.value;
        const gpuNames = Array(gpuCount).fill(selectedGpu);

        const results = recommend(gpuNames, budget || null, useCase);
        const resultsDiv = document.getElementById('recResults');

        if (!results.length) {
            resultsDiv.innerHTML = '<p style="color:#888;font-size:13px;">No models fit your constraints. Try a larger GPU or lower quality.</p>';
            return;
        }

        resultsDiv.innerHTML = results.map((r, i) => `
            <div class="rec-card ${i === 0 ? 'best' : ''}">
                <div>
                    <div class="rec-model">
                        ${r.model}
                        ${i === 0 ? '<span class="rec-badge">Best Match</span>' : ''}
                    </div>
                    <div class="rec-quant">${r.quant} · ${r.vramNeeded}GB VRAM · ${r.gpusNeeded} GPU${r.gpusNeeded > 1 ? 's' : ''}</div>
                    <div class="rec-details">
                        Quality: ${(r.quality * 100).toFixed(0)}% · 
                        Speed: ~${r.speed} tok/s · 
                        GPU: ${r.gpuName} · 
                        Cost: $${r.cost.toLocaleString()}
                    </div>
                </div>
                <div class="rec-score">
                    <div class="rec-score-val">${(r.score * 100).toFixed(0)}</div>
                    <div class="rec-score-label">Score</div>
                </div>
            </div>
        `).join('');
    });
}
