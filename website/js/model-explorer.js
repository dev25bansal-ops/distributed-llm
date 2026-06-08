/**
 * A6. Model Explorer — Interactive grid with filtering by VRAM, family, quantization.
 */
const MODELS = [
    { name: 'Llama 3.1 8B', family: 'Llama', vram: 16, quant: ['FP16', 'INT8', 'INT4'], params: '8B', desc: 'General-purpose model for chat, code, and local clusters.' },
    { name: 'Llama 3.1 70B', family: 'Llama', vram: 140, quant: ['FP16', 'INT8', 'INT4'], params: '70B', desc: 'High-quality reasoning; practical on two or more GPUs with quantization.' },
    { name: 'Llama 3.1 405B', family: 'Llama', vram: 810, quant: ['FP16', 'INT8'], params: '405B', desc: 'Frontier-scale model for large multi-node clusters.' },
    { name: 'Llama 3 8B Instruct', family: 'Llama', vram: 16, quant: ['FP16', 'INT8', 'INT4'], params: '8B', desc: 'Instruction-tuned Llama model for assistant workloads.' },
    { name: 'Llama 3 70B Instruct', family: 'Llama', vram: 140, quant: ['FP16', 'INT8', 'INT4'], params: '70B', desc: 'Instruction-tuned 70B model for advanced chat and reasoning.' },
    { name: 'CodeLlama 34B', family: 'Llama', vram: 68, quant: ['FP16', 'INT8'], params: '34B', desc: 'Code generation and understanding.' },
    { name: 'Mistral 7B v0.3', family: 'Mistral', vram: 14, quant: ['FP16', 'INT8', 'INT4'], params: '7B', desc: 'Fast, efficient, and strong for production assistants.' },
    { name: 'Mixtral 8x7B', family: 'Mistral', vram: 94, quant: ['FP16', 'INT8'], params: '47B', desc: 'MoE architecture with strong quality per active token.' },
    { name: 'Mixtral 8x22B', family: 'Mistral', vram: 282, quant: ['FP16', 'INT8'], params: '141B', desc: 'Large MoE model for multi-GPU reasoning workloads.' },
    { name: 'Qwen2 7B', family: 'Qwen', vram: 14, quant: ['FP16', 'INT8', 'INT4'], params: '7B', desc: 'Strong multilingual model for English and Chinese.' },
    { name: 'Qwen2 72B', family: 'Qwen', vram: 144, quant: ['FP16', 'INT8'], params: '72B', desc: 'Large multilingual reasoning model.' },
    { name: 'Qwen2.5 72B', family: 'Qwen', vram: 144, quant: ['FP16', 'INT8'], params: '72B', desc: 'Latest Qwen 72B support in the repository matrix.' },
    { name: 'DeepSeek V2', family: 'DeepSeek', vram: 472, quant: ['FP16', 'INT8'], params: '236B', desc: 'MoE architecture for high-capability distributed inference.' },
    { name: 'DeepSeek Coder V2', family: 'DeepSeek', vram: 472, quant: ['FP16', 'INT8'], params: '236B', desc: 'Code-focused MoE model for development workloads.' },
    { name: 'Falcon 7B', family: 'Falcon', vram: 14, quant: ['FP16', 'INT8'], params: '7B', desc: 'Supported Falcon baseline for smaller deployments.' },
    { name: 'Falcon 40B', family: 'Falcon', vram: 80, quant: ['FP16', 'INT8'], params: '40B', desc: 'Larger Falcon model for distributed clusters.' },
    { name: 'Phi-3 mini', family: 'Phi', vram: 8, quant: ['FP16', 'INT8', 'INT4'], params: '3.8B', desc: 'Small but capable model for lightweight inference.' },
    { name: 'Phi-3 medium', family: 'Phi', vram: 28, quant: ['FP16', 'INT8'], params: '14B', desc: 'Efficient mid-size reasoning model.' },
    { name: 'Gemma 2 9B', family: 'Gemma', vram: 18, quant: ['FP16', 'INT8'], params: '9B', desc: 'Efficient Google model for general workloads.' },
    { name: 'Gemma 2 27B', family: 'Gemma', vram: 54, quant: ['FP16', 'INT8'], params: '27B', desc: 'Stronger Gemma option for analysis and reasoning.' },
];

const FAMILIES = [...new Set(MODELS.map(m => m.family))].sort();

export function initModelExplorer() {
    const container = document.getElementById('modelExplorer');
    if (!container) return;

    const familyOptions = FAMILIES.map(f => `<option value="${f}">${f}</option>`).join('');

    container.innerHTML = `
        <div class="explorer-filters">
            <div class="explorer-filter">
                <label for="explorerFamily">Family</label>
                <select id="explorerFamily" class="gpu-select">
                    <option value="all">All families</option>
                    ${familyOptions}
                </select>
            </div>
            <div class="explorer-filter">
                <label for="explorerVram">Max VRAM (GB)</label>
                <input type="range" id="explorerVram" class="calc-slider" min="4" max="900" value="900">
                <span id="explorerVramLabel">900GB</span>
            </div>
            <div class="explorer-filter">
                <label for="explorerQuant">Quantization</label>
                <select id="explorerQuant" class="gpu-select">
                    <option value="all">Any</option>
                    <option value="FP16">FP16 only</option>
                    <option value="INT8">INT8 or better</option>
                    <option value="INT4">INT4 or better</option>
                </select>
            </div>
        </div>
        <div class="explorer-grid" id="explorerGrid"></div>
    `;

    const familySel = document.getElementById('explorerFamily');
    const vramSlider = document.getElementById('explorerVram');
    const vramLabel = document.getElementById('explorerVramLabel');
    const quantSel = document.getElementById('explorerQuant');
    const grid = document.getElementById('explorerGrid');

    const update = () => {
        const family = familySel.value;
        const maxVram = parseInt(vramSlider.value);
        const quant = quantSel.value;

        vramLabel.textContent = maxVram + 'GB';

        const filtered = MODELS.filter(m => {
            if (family !== 'all' && m.family !== family) return false;
            if (quant === 'FP16' && !m.quant.includes('FP16')) return false;
            if (quant === 'INT8' && !m.quant.includes('INT8')) return false;
            if (quant === 'INT4' && !m.quant.includes('INT4')) return false;
            const required = getRequiredVram(m, quant);
            if (required > maxVram) return false;
            return true;
        });

        grid.innerHTML = filtered.length === 0
            ? '<div class="explorer-empty">No models match your filters. Try increasing VRAM or changing quantization.</div>'
            : filtered.map(m => {
                const vramRows = m.quant.map(q => `<span>${q}: ${getRequiredVram(m, q)}GB</span>`).join('');
                return `<div class="explorer-card">
                    <div class="explorer-card-header">
                        <span class="explorer-card-name">${m.name}</span>
                        <span class="explorer-card-params">${m.params}</span>
                    </div>
                    <div class="explorer-card-family">${m.family}</div>
                    <p class="explorer-card-desc">${m.desc}</p>
                    <div class="explorer-card-vram">${vramRows}</div>
                    <div class="explorer-card-quant">${m.quant.join(' · ')}</div>
                </div>`;
            }).join('');
    };

    familySel.addEventListener('change', update);
    vramSlider.addEventListener('input', update);
    quantSel.addEventListener('change', update);
    update();
}

function getRequiredVram(model, quantization) {
    if (quantization === 'all') {
        return Math.min(...model.quant.map(q => getRequiredVram(model, q)));
    }
    if (quantization === 'FP16') return model.vram;
    if (quantization === 'INT8') return Math.ceil(model.vram / 2);
    if (quantization === 'INT4') return Math.ceil(model.vram / 4);
    return model.vram;
}
