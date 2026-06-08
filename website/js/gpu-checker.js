/**
 * A3. GPU Compatibility Checker — Enter GPU model → shows compatible models.
 */
const GPU_DATABASE = [
    { name: 'RTX 4090', vram: 24, tier: 'consumer' },
    { name: 'RTX 4080', vram: 16, tier: 'consumer' },
    { name: 'RTX 4070 Ti', vram: 12, tier: 'consumer' },
    { name: 'RTX 4070', vram: 12, tier: 'consumer' },
    { name: 'RTX 4060 Ti', vram: 16, tier: 'consumer' },
    { name: 'RTX 4060', vram: 8, tier: 'consumer' },
    { name: 'RTX 3090', vram: 24, tier: 'consumer' },
    { name: 'RTX 3080', vram: 10, tier: 'consumer' },
    { name: 'RTX 3070', vram: 8, tier: 'consumer' },
    { name: 'A100 80GB', vram: 80, tier: 'datacenter' },
    { name: 'A100 40GB', vram: 40, tier: 'datacenter' },
    { name: 'H100 80GB', vram: 80, tier: 'datacenter' },
    { name: 'L40S', vram: 48, tier: 'datacenter' },
    { name: 'A6000', vram: 48, tier: 'pro' },
    { name: 'A40', vram: 48, tier: 'pro' },
];

const MODEL_DATABASE = [
    { name: 'Llama 3.1 8B', family: 'Llama', requirements: { FP16: 16, INT8: 8, INT4: 4 }, tokPerSec: '40-60' },
    { name: 'Llama 3.1 70B', family: 'Llama', requirements: { FP16: 140, INT8: 70, INT4: 35 }, tokPerSec: '20-30' },
    { name: 'Llama 3.1 405B', family: 'Llama', requirements: { FP16: 810, INT8: 405 }, tokPerSec: '8-15' },
    { name: 'Mistral 7B v0.3', family: 'Mistral', requirements: { FP16: 14, INT8: 7, INT4: 4 }, tokPerSec: '45-65' },
    { name: 'Mixtral 8x7B', family: 'Mistral', requirements: { FP16: 94, INT8: 47 }, tokPerSec: '25-35' },
    { name: 'Mixtral 8x22B', family: 'Mistral', requirements: { FP16: 282, INT8: 141 }, tokPerSec: '12-22' },
    { name: 'Qwen2 7B', family: 'Qwen', requirements: { FP16: 14, INT8: 7, INT4: 4 }, tokPerSec: '45-65' },
    { name: 'Qwen2.5 72B', family: 'Qwen', requirements: { FP16: 144, INT8: 72 }, tokPerSec: '18-28' },
    { name: 'Phi-3 mini', family: 'Phi', requirements: { FP16: 8, INT8: 4, INT4: 2 }, tokPerSec: '60-80' },
    { name: 'Phi-3 medium', family: 'Phi', requirements: { FP16: 28, INT8: 14 }, tokPerSec: '35-55' },
    { name: 'Falcon 40B', family: 'Falcon', requirements: { FP16: 80, INT8: 40 }, tokPerSec: '18-30' },
    { name: 'DeepSeek V2', family: 'DeepSeek', requirements: { FP16: 472, INT8: 236 }, tokPerSec: '8-18' },
    { name: 'CodeLlama 34B', family: 'Llama', requirements: { FP16: 68, INT8: 34 }, tokPerSec: '25-40' },
    { name: 'Gemma 2 27B', family: 'Gemma', requirements: { FP16: 54, INT8: 27 }, tokPerSec: '30-45' },
];

export function initGpuChecker() {
    const container = document.getElementById('gpuChecker');
    if (!container) return;

    const gpuOptions = GPU_DATABASE.map(g =>
        `<option value="${g.name}">${g.name} (${g.vram}GB)</option>`
    ).join('');

    container.innerHTML = `
        <div class="gpu-checker-card">
            <div class="gpu-checker-row">
                <label for="gpuSelect" class="calc-label"><span>Select your GPU</span></label>
                <select id="gpuSelect" class="gpu-select">${gpuOptions}</select>
            </div>
            <div class="gpu-checker-row">
                <label for="gpuCountInput" class="calc-label"><span>Number of GPUs</span></label>
                <input type="number" id="gpuCountInput" class="gpu-count-input" min="1" max="16" value="1">
            </div>
            <div class="gpu-checker-results" id="gpuResults"></div>
        </div>
    `;

    const select = document.getElementById('gpuSelect');
    const countInput = document.getElementById('gpuCountInput');
    const results = document.getElementById('gpuResults');

    const update = () => {
        const gpu = GPU_DATABASE.find(g => g.name === select.value);
        const count = parseInt(countInput.value) || 1;
        if (!gpu) return;

        const totalVram = gpu.vram * count;
        const resultsHtml = MODEL_DATABASE.map(m => {
            const preferredQuant = getPreferredQuantization(m.requirements, totalVram);
            const cheapestRequirement = Math.min(...Object.values(m.requirements));
            const quant = preferredQuant || '—';
            const statusClass = preferredQuant ? 'gpu-fit-yes' : 'gpu-fit-no';
            const gpusNeeded = preferredQuant
                ? Math.ceil(m.requirements[preferredQuant] / gpu.vram)
                : Math.ceil(cheapestRequirement / gpu.vram);

            return `<div class="gpu-model-row ${statusClass}">
                <span class="gpu-model-name">${m.name}</span>
                <span class="gpu-model-family">${m.family}</span>
                <span class="gpu-model-quant">${quant}</span>
                <span class="gpu-model-gpus">${preferredQuant ? `${gpusNeeded} GPU${gpusNeeded > 1 ? 's' : ''}` : `${gpusNeeded} needed`}</span>
                <span class="gpu-model-tps">${preferredQuant ? `~${m.tokPerSec} tok/s` : '—'}</span>
            </div>`;
        }).join('');

        results.innerHTML = `
            <div class="gpu-summary">Total VRAM: <strong>${totalVram}GB</strong> across ${count}x ${gpu.name}</div>
            <div class="gpu-model-header">
                <span>Model</span><span>Family</span><span>Quant</span><span>GPUs</span><span>Speed</span>
            </div>
            ${resultsHtml}
        `;
    };

    select.addEventListener('change', update);
    countInput.addEventListener('input', update);
    update();
}

function getPreferredQuantization(requirements, totalVram) {
    return ['FP16', 'INT8', 'INT4'].find(quant => requirements[quant] && totalVram >= requirements[quant]) || null;
}
