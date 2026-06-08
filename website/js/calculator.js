/**
 * Savings calculator — estimates cost savings vs. cloud APIs.
 * Uses rAF throttle for smooth slider interaction.
 *
 * Features:
 * - Monthly savings estimate (cloud vs self-hosted electricity)
 * - TCO comparison (1/3/5 year hardware depreciation)
 * - Cloud provider price comparison table
 * - CSS bar chart visualization
 *
 * Electricity: GPU TDP + system overhead (CPU/RAM/PSU inefficiency)
 * RTX 4090: 450W GPU + ~150W system = ~0.6kW total
 * A100 80GB: 300W GPU + ~200W system = ~0.5kW total
 * Average consumer GPU setup: ~0.55kW (rounded to 0.55)
 */

export const MODEL_SIZES = [
    { label: '1.5B', tpm: 0.0002 },
    { label: '3B', tpm: 0.0004 },
    { label: '7B', tpm: 0.0008 },
    { label: '13B', tpm: 0.0015 },
    { label: '70B', tpm: 0.002 },
];

// ── Cloud Provider Pricing Database ──────────────────────────────
const CLOUD_PROVIDERS = [
    { name: 'DistLLM (Self-Hosted)', color: '#00e676', desc: 'Your own GPUs', per1K: 0.0005, type: 'self' },
    { name: 'Groq', color: '#f97316', desc: 'Llama 3 70B (LPU)', per1K: 0.00059, type: 'api' },
    { name: 'Together AI', color: '#7c3aed', desc: 'Llama 3 70B', per1K: 0.0009, type: 'api' },
    { name: 'Replicate', color: '#1b1b1b', desc: 'Llama 3 70B', per1K: 0.0013, type: 'api' },
    { name: 'GCP Vertex AI', color: '#4285f4', desc: 'Llama 3 70B', per1K: 0.00240, type: 'api' },
    { name: 'AWS Bedrock', color: '#ff9900', desc: 'Llama 3 70B', per1K: 0.00265, type: 'api' },
    { name: 'Anthropic', color: '#d97706', desc: 'Claude 3', per1K: 0.015, type: 'api' },
    { name: 'OpenAI', color: '#10a37f', desc: 'GPT-4o', per1K: 0.01, type: 'api' },
    { name: 'Azure OpenAI', color: '#0078d4', desc: 'GPT-4', per1K: 0.03, type: 'api' },
];

export function initCalculator() {
    const ids = ['gpuCount', 'tokens', 'modelSize', 'hours', 'elecCost'];
    if (!ids.every(id => document.getElementById(id))) return;

    let rafId;
    const throttledUpdate = () => {
        if (rafId) return;
        rafId = requestAnimationFrame(() => {
            update();
            rafId = null;
        });
    };

    ids.forEach(id => document.getElementById(id).addEventListener('input', throttledUpdate));
    update();
}

function update() {
    const gpus = parseInt(document.getElementById('gpuCount').value) || 0;
    const tokens = parseInt(document.getElementById('tokens').value) || 0;
    const mi = parseInt(document.getElementById('modelSize').value) || 0;
    const hours = parseInt(document.getElementById('hours').value) || 0;
    const m = MODEL_SIZES[mi];

    document.getElementById('gpuCountLabel').textContent = gpus;
    document.getElementById('tokensLabel').textContent = tokens + 'M';
    document.getElementById('modelSizeLabel').textContent = m.label;
    document.getElementById('hoursLabel').textContent = hours + 'h';
    document.getElementById('elecCostLabel').textContent = '$' + parseFloat(document.getElementById('elecCost').value || 0.12).toFixed(2);

    const elecRate = parseFloat(document.getElementById('elecCost').value) || 0.12;
    const totalTokens = tokens * 1000000;
    const cloud = totalTokens * m.tpm;
    const elec = gpus * 0.55 * hours * elecRate * 30;
    const save = Math.max(0, cloud - elec);

    document.getElementById('savingsValue').textContent = '$' + Math.round(save).toLocaleString();
    document.getElementById('savingsSub').textContent =
        `Cloud: $${Math.round(cloud).toLocaleString()}/mo · DistLLM: $${Math.round(elec).toLocaleString()}/mo`;

    // TCO comparison
    const gpuCost = gpus * 1600;
    const monthlyCost = elec;
    const cloudMonthly = cloud;

    for (const years of [1, 3, 5]) {
        const months = years * 12;
        const tcoDistLLM = gpuCost + (monthlyCost * months);
        const tcoCloud = cloudMonthly * months;
        const el = document.getElementById(`tco${years}y`);
        const sub = document.getElementById(`tco${years}ySub`);
        if (el) el.textContent = '$' + Math.round(tcoDistLLM).toLocaleString();
        if (sub) sub.textContent = `vs $${Math.round(tcoCloud).toLocaleString()} cloud`;
    }

    // Cloud provider comparison table
    updateProviderComparison(totalTokens, monthlyCost);
}

function updateProviderComparison(totalTokens, selfHostedCost) {
    const container = document.getElementById('providerComparison');
    if (!container) return;

    const rows = CLOUD_PROVIDERS.map(p => {
        const monthly = p.type === 'self' ? selfHostedCost : totalTokens / 1000 * p.per1K;
        return { ...p, monthly };
    });

    const maxMonthly = Math.max(...rows.map(r => r.monthly), 1);

    container.innerHTML = `
        <h4 style="margin:24px 0 12px;font-size:15px;">Cloud Provider Comparison (monthly cost)</h4>
        <div class="provider-table">
            ${rows.map(r => `
                <div class="provider-row">
                    <div class="provider-name">
                        <span class="provider-dot" style="background:${r.color}"></span>
                        <span>${r.name}</span>
                        <span class="provider-desc">${r.desc}</span>
                    </div>
                    <div class="provider-bar-wrap">
                        <div class="provider-bar" style="width:${(r.monthly / maxMonthly) * 100}%;background:${r.color};opacity:${r.type === 'self' ? 1 : 0.7}"></div>
                    </div>
                    <div class="provider-cost" style="color:${r.type === 'self' ? 'var(--green)' : 'var(--text)'};font-weight:${r.type === 'self' ? 700 : 400}">
                        $${Math.round(r.monthly).toLocaleString()}/mo
                    </div>
                </div>
            `).join('')}
        </div>
    `;

    // Update chart style to show comparison
    const style = container.querySelector('style') || document.createElement('style');
    style.textContent = `
        .provider-table { display:flex;flex-direction:column;gap:8px; }
        .provider-row { display:grid;grid-template-columns:1fr 2fr 120px;align-items:center;gap:12px;padding:8px 12px;background:var(--card);border-radius:var(--radius-sm); }
        .provider-name { display:flex;align-items:center;gap:6px;font-size:13px; }
        .provider-dot { width:8px;height:8px;border-radius:50%;flex-shrink:0; }
        .provider-desc { color:var(--dim);font-size:11px;margin-left:4px; }
        .provider-bar-wrap { height:20px;background:var(--surface);border-radius:4px;overflow:hidden; }
        .provider-bar { height:100%;border-radius:4px;transition:width 0.3s;min-width:2px; }
        .provider-cost { font-size:13px;text-align:right;font-family:var(--font-mono); }
    `;
    container.appendChild(style);
}
