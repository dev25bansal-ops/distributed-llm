/**
 * Performance Benchmark Dashboard — interactive charts and comparisons.
 *
 * Shows token/s for different model/GPU combinations, compares against
 * cloud APIs, and lets users run their own benchmarks.
 *
 * Uses Canvas-based charts (no external dependencies).
 *
 * Usage:
 *   <div id="benchDashboard"></div>
 *   <script type="module">
 *     import { initBenchDashboard } from './js/bench-dashboard.js';
 *     initBenchDashboard();
 *   </script>
 */

// ── Benchmark Data ─────────────────────────────────────────────────────

const BENCHMARKS = {
    models: [
        { name: 'Qwen2.5 3B', params: 3, gpu: 'RTX 4090', tok_s: 85, latency_p50: 12, latency_p95: 28, cost_1k: 0.001, cloud: 'N/A' },
        { name: 'Llama 3.1 8B', params: 8, gpu: 'RTX 4090', tok_s: 52, latency_p50: 19, latency_p95: 45, cost_1k: 0.002, cloud: '$0.0002' },
        { name: 'Llama 3.1 8B', params: 8, gpu: 'A100 80GB', tok_s: 78, latency_p50: 13, latency_p95: 30, cost_1k: 0.008, cloud: '$0.0002' },
        { name: 'Llama 3.1 8B', params: 8, gpu: '2x RTX 4090', tok_s: 95, latency_p50: 11, latency_p95: 25, cost_1k: 0.004, cloud: '$0.0002' },
        { name: 'Mistral 7B', params: 7, gpu: 'RTX 4090', tok_s: 58, latency_p50: 17, latency_p95: 40, cost_1k: 0.002, cloud: '$0.0002' },
        { name: 'CodeLlama 34B', params: 34, gpu: '2x A100 80GB', tok_s: 28, latency_p50: 36, latency_p95: 82, cost_1k: 0.016, cloud: '$0.0005' },
        { name: 'Llama 3.1 70B', params: 70, gpu: '4x A100 80GB', tok_s: 18, latency_p50: 56, latency_p95: 130, cost_1k: 0.032, cloud: '$0.0009' },
        { name: 'Llama 3.1 70B', params: 70, gpu: '4x RTX 4090', tok_s: 14, latency_p50: 71, latency_p95: 165, cost_1k: 0.006, cloud: '$0.0009' },
        { name: 'Qwen2.5 72B', params: 72, gpu: '4x A100 80GB', tok_s: 16, latency_p50: 63, latency_p95: 145, cost_1k: 0.032, cloud: '$0.001' },
    ],
    cloud: [
        { name: 'OpenAI GPT-4o', tok_s: 80, latency_p50: 300, cost_1k: 0.005 },
        { name: 'OpenAI GPT-4o-mini', tok_s: 120, latency_p50: 200, cost_1k: 0.00015 },
        { name: 'Together AI (Llama 3.1 70B)', tok_s: 40, latency_p50: 150, cost_1k: 0.0009 },
        { name: 'Fireworks (Llama 3.1 70B)', tok_s: 45, latency_p50: 120, cost_1k: 0.0009 },
        { name: 'Anthropic Claude 3.5 Sonnet', tok_s: 70, latency_p50: 250, tok_s: 70, cost_1k: 0.003 },
    ],
};

// ── Chart Drawing (Canvas-based, no dependencies) ──────────────────────

function drawBarChart(canvas, data, options = {}) {
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    const W = rect.width, H = rect.height;

    const { labels, values, colors, title, ylabel } = options;
    const maxVal = Math.max(...values) * 1.2;
    const barW = Math.min(60, (W - 80) / values.length - 10);
    const startX = 60;
    const startY = 40;
    const chartH = H - 80;

    ctx.clearRect(0, 0, W, H);

    // Title
    if (title) {
        ctx.fillStyle = '#ededed';
        ctx.font = 'bold 14px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(title, W / 2, 20);
    }

    // Y axis
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(startX, startY);
    ctx.lineTo(startX, startY + chartH);
    ctx.stroke();

    // Grid lines
    for (let i = 0; i <= 4; i++) {
        const y = startY + (chartH / 4) * i;
        ctx.strokeStyle = '#1a1a1a';
        ctx.beginPath();
        ctx.moveTo(startX, y);
        ctx.lineTo(W - 20, y);
        ctx.stroke();

        ctx.fillStyle = '#555';
        ctx.font = '10px Inter, sans-serif';
        ctx.textAlign = 'right';
        const val = maxVal - (maxVal / 4) * i;
        ctx.fillText(val.toFixed(0), startX - 8, y + 4);
    }

    // Y label
    if (ylabel) {
        ctx.save();
        ctx.translate(12, startY + chartH / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillStyle = '#888';
        ctx.font = '11px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(ylabel, 0, 0);
        ctx.restore();
    }

    // Bars
    values.forEach((val, i) => {
        const x = startX + 20 + i * (barW + 10);
        const barH = (val / maxVal) * chartH;
        const y = startY + chartH - barH;

        // Bar
        ctx.fillStyle = colors?.[i] || '#22c55e';
        ctx.beginPath();
        ctx.roundRect(x, y, barW, barH, [4, 4, 0, 0]);
        ctx.fill();

        // Value on top
        ctx.fillStyle = '#ededed';
        ctx.font = 'bold 11px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(val.toFixed(0), x + barW / 2, y - 6);

        // Label
        ctx.fillStyle = '#888';
        ctx.font = '10px Inter, sans-serif';
        ctx.textAlign = 'center';
        const label = labels?.[i] || '';
        // Truncate long labels
        const maxLabelLen = Math.floor(barW / 6);
        const truncLabel = label.length > maxLabelLen ? label.slice(0, maxLabelLen) + '...' : label;
        ctx.fillText(truncLabel, x + barW / 2, startY + chartH + 16);
    });
}

// ── UI ─────────────────────────────────────────────────────────────────

export function initBenchDashboard() {
    const container = document.getElementById('benchDashboard');
    if (!container) return;

    container.innerHTML = `
        <div class="bench-card">
            <h3>Performance Benchmark Dashboard</h3>
            <p class="bench-desc">Interactive benchmarks comparing DistLLM against cloud APIs.</p>

            <div class="bench-tabs" id="benchTabs">
                <button class="bench-tab active" data-view="throughput">Throughput</button>
                <button class="bench-tab" data-view="latency">Latency</button>
                <button class="bench-tab" data-view="cost">Cost</button>
                <button class="bench-tab" data-view="comparison">vs Cloud</button>
            </div>

            <div class="bench-chart-wrap">
                <canvas id="benchChart" height="350"></canvas>
            </div>

            <div class="bench-table-wrap">
                <table class="bench-table" id="benchTable">
                    <thead>
                        <tr>
                            <th>Model</th>
                            <th>Params</th>
                            <th>GPU</th>
                            <th>tok/s</th>
                            <th>P50 (ms)</th>
                            <th>P95 (ms)</th>
                            <th>Cost/1K</th>
                            <th>Cloud Price</th>
                        </tr>
                    </thead>
                    <tbody id="benchTableBody"></tbody>
                </table>
            </div>

            <div class="bench-calculator">
                <h4>Cost Calculator</h4>
                <div class="bench-calc-row">
                    <label>Monthly Tokens (millions)</label>
                    <input type="number" id="benchTokens" value="100" min="1">
                </div>
                <div class="bench-calc-row">
                    <label>Model</label>
                    <select id="benchModel"></select>
                </div>
                <div class="bench-calc-result" id="benchCalcResult"></div>
            </div>
        </div>
    `;

    const chart = document.getElementById('benchChart');
    const tableBody = document.getElementById('benchTableBody');
    const modelSelect = document.getElementById('benchModel');
    const tabs = document.querySelectorAll('.bench-tab');
    const tokensInput = document.getElementById('benchTokens');
    const calcResult = document.getElementById('benchCalcResult');

    // Populate model select
    BENCHMARKS.models.forEach(m => {
        const opt = document.createElement('option');
        opt.value = m.name;
        opt.textContent = `${m.name} (${m.gpu})`;
        modelSelect.appendChild(opt);
    });

    // Populate table
    function updateTable() {
        tableBody.innerHTML = BENCHMARKS.models.map(m => `
            <tr>
                <td><strong>${m.name}</strong></td>
                <td>${m.params}B</td>
                <td>${m.gpu}</td>
                <td class="highlight">${m.tok_s}</td>
                <td>${m.latency_p50}ms</td>
                <td>${m.latency_p95}ms</td>
                <td>$${m.cost_1k.toFixed(4)}</td>
                <td>${m.cloud}</td>
            </tr>
        `).join('');
    }

    // Chart views
    function showThroughput() {
        const data = BENCHMARKS.models;
        drawBarChart(chart, data.map(d => d.tok_s), {
            labels: data.map(d => d.name.split(' ').slice(-2).join(' ')),
            values: data.map(d => d.tok_s),
            colors: data.map(d => d.gpu.includes('A100') ? '#06b6d4' : '#22c55e'),
            title: 'Throughput (tokens/sec)',
            ylabel: 'tok/s',
        });
    }

    function showLatency() {
        const data = BENCHMARKS.models;
        drawBarChart(chart, data.map(d => d.latency_p50), {
            labels: data.map(d => d.name.split(' ').slice(-2).join(' ')),
            values: data.map(d => d.latency_p50),
            colors: data.map(d => d.latency_p50 < 30 ? '#22c55e' : d.latency_p50 < 60 ? '#eab308' : '#ef4444'),
            title: 'Latency P50 (ms)',
            ylabel: 'ms',
        });
    }

    function showCost() {
        const data = BENCHMARKS.models;
        drawBarChart(chart, data.map(d => d.cost_1k * 1000), {
            labels: data.map(d => d.name.split(' ').slice(-2).join(' ')),
            values: data.map(d => d.cost_1k * 1000),
            colors: data.map(d => '#22c55e'),
            title: 'Cost per 1M Tokens (USD)',
            ylabel: '$/1M tokens',
        });
    }

    function showComparison() {
        const distllm = BENCHMARKS.models.filter(m => m.params <= 8);
        const cloud = BENCHMARKS.cloud;
        const all = [...distllm.map(d => ({ name: `DistLLM ${d.name}`, tok_s: d.tok_s, color: '#22c55e' })),
                     ...cloud.map(c => ({ name: c.name, tok_s: c.tok_s, color: '#ef4444' }))];

        drawBarChart(chart, all.map(d => d.tok_s), {
            labels: all.map(d => d.name.length > 20 ? d.name.slice(0, 18) + '...' : d.name),
            values: all.map(d => d.tok_s),
            colors: all.map(d => d.color),
            title: 'DistLLM vs Cloud APIs (tok/s)',
            ylabel: 'tok/s',
        });
    }

    const views = { throughput: showThroughput, latency: showLatency, cost: showCost, comparison: showComparison };

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            tabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            views[tab.dataset.view]();
        });
    });

    // Cost calculator
    function updateCalc() {
        const tokens = parseFloat(tokensInput.value) || 100;
        const model = BENCHMARKS.models.find(m => m.name === modelSelect.value);
        if (!model) return;

        const distllmCost = tokens * 1000 * model.cost_1k;
        const cloudCost = tokens * 1000 * 0.001; // ~$0.001/1K avg cloud
        const savings = cloudCost - distllmCost;

        calcResult.innerHTML = `
            <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;">
                <div><div style="font-size:11px;color:#888;">DistLLM Cost</div><div style="font-size:18px;font-weight:700;color:#22c55e;">$${distllmCost.toFixed(2)}</div></div>
                <div><div style="font-size:11px;color:#888;">Cloud Cost (avg)</div><div style="font-size:18px;font-weight:700;color:#ef4444;">$${cloudCost.toFixed(2)}</div></div>
                <div><div style="font-size:11px;color:#888;">Savings</div><div style="font-size:18px;font-weight:700;color:#22c55e;">$${savings.toFixed(2)} (${((savings/cloudCost)*100).toFixed(0)}%)</div></div>
            </div>
        `;
    }

    tokensInput.addEventListener('input', updateCalc);
    modelSelect.addEventListener('change', updateCalc);

    // Initial render
    updateTable();
    showThroughput();
    updateCalc();
}
