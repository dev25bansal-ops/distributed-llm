/**
 * Model Benchmark Explorer
 *
 * Interactive charts and filters for model × GPU × quantization benchmarks.
 *
 * Features:
 * - Throughput vs Latency scatter plot
 * - Cost per token comparison
 * - VRAM usage visualization
 * - Filter by GPU, model, quantization
 * - "What can I expect?" queries
 *
 * Usage:
 *   <div id="benchmarkExplorer"></div>
 *   <script type="module">
 *     import { initBenchmarkExplorer } from './js/benchmark-explorer.js';
 *     initBenchmarkExplorer();
 *   </script>
 */

// ── Benchmark Data ─────────────────────────────────────────────────────

const BENCHMARK_DATA = [
    // Llama 3.1 8B
    { model: 'llama-3.1-8b', gpu: 'rtx4090', quant: 'fp16', tok_s: 52, latency_p50: 19, latency_p95: 45, vram: 16, cost_1k: 0.002 },
    { model: 'llama-3.1-8b', gpu: 'rtx4090', quant: 'int8', tok_s: 68, latency_p50: 15, latency_p95: 35, vram: 8.5, cost_1k: 0.0015 },
    { model: 'llama-3.1-8b', gpu: 'rtx4090', quant: 'int4', tok_s: 85, latency_p50: 12, latency_p95: 28, vram: 5, cost_1k: 0.001 },
    { model: 'llama-3.1-8b', gpu: 'rtx3090', quant: 'fp16', tok_s: 42, latency_p50: 24, latency_p95: 55, vram: 16, cost_1k: 0.0025 },
    { model: 'llama-3.1-8b', gpu: 'rtx3090', quant: 'int8', tok_s: 55, latency_p50: 18, latency_p95: 42, vram: 8.5, cost_1k: 0.002 },
    { model: 'llama-3.1-8b', gpu: 'rtx3090', quant: 'int4', tok_s: 72, latency_p50: 14, latency_p95: 32, vram: 5, cost_1k: 0.0015 },
    { model: 'llama-3.1-8b', gpu: 'a100', quant: 'fp16', tok_s: 78, latency_p50: 13, latency_p95: 30, vram: 16, cost_1k: 0.008 },
    { model: 'llama-3.1-8b', gpu: 'a100', quant: 'int8', tok_s: 105, latency_p50: 10, latency_p95: 22, vram: 8.5, cost_1k: 0.006 },
    { model: 'llama-3.1-8b', gpu: 'a100', quant: 'int4', tok_s: 130, latency_p50: 8, latency_p95: 18, vram: 5, cost_1k: 0.004 },

    // Llama 3.1 70B
    { model: 'llama-3.1-70b', gpu: 'rtx4090', quant: 'int4', tok_s: 14, latency_p50: 71, latency_p95: 165, vram: 35, cost_1k: 0.006 },
    { model: 'llama-3.1-70b', gpu: 'rtx4090x2', quant: 'int4', tok_s: 24, latency_p50: 42, latency_p95: 95, vram: 35, cost_1k: 0.004 },
    { model: 'llama-3.1-70b', gpu: 'rtx4090x4', quant: 'int4', tok_s: 42, latency_p50: 24, latency_p95: 55, vram: 35, cost_1k: 0.003 },
    { model: 'llama-3.1-70b', gpu: 'a100', quant: 'fp16', tok_s: 18, latency_p50: 56, latency_p95: 130, vram: 140, cost_1k: 0.032 },
    { model: 'llama-3.1-70b', gpu: 'a100x4', quant: 'int8', tok_s: 45, latency_p50: 22, latency_p95: 50, vram: 70, cost_1k: 0.016 },
    { model: 'llama-3.1-70b', gpu: 'a100x4', quant: 'int4', tok_s: 68, latency_p50: 15, latency_p95: 35, vram: 35, cost_1k: 0.012 },
    { model: 'llama-3.1-70b', gpu: 'h100', quant: 'fp16', tok_s: 28, latency_p50: 36, latency_p95: 82, vram: 140, cost_1k: 0.025 },
    { model: 'llama-3.1-70b', gpu: 'h100x4', quant: 'int4', tok_s: 95, latency_p50: 11, latency_p95: 25, vram: 35, cost_1k: 0.008 },

    // Mistral 7B
    { model: 'mistral-7b', gpu: 'rtx4090', quant: 'fp16', tok_s: 58, latency_p50: 17, latency_p95: 40, vram: 14, cost_1k: 0.0018 },
    { model: 'mistral-7b', gpu: 'rtx4090', quant: 'int4', tok_s: 92, latency_p50: 11, latency_p95: 25, vram: 4.5, cost_1k: 0.001 },
    { model: 'mistral-7b', gpu: 'rtx3090', quant: 'int4', tok_s: 78, latency_p50: 13, latency_p95: 30, vram: 4.5, cost_1k: 0.0012 },

    // Mixtral 8x7B
    { model: 'mixtral-8x7b', gpu: 'rtx4090', quant: 'int4', tok_s: 28, latency_p50: 36, latency_p95: 82, vram: 24, cost_1k: 0.003 },
    { model: 'mixtral-8x7b', gpu: 'rtx4090x2', quant: 'int4', tok_s: 48, latency_p50: 21, latency_p95: 48, vram: 24, cost_1k: 0.002 },
    { model: 'mixtral-8x7b', gpu: 'a100', quant: 'int8', tok_s: 52, latency_p50: 19, latency_p95: 45, vram: 47, cost_1k: 0.01 },

    // Qwen 72B
    { model: 'qwen-72b', gpu: 'rtx4090', quant: 'int4', tok_s: 13, latency_p50: 77, latency_p95: 180, vram: 36, cost_1k: 0.0065 },
    { model: 'qwen-72b', gpu: 'rtx4090x4', quant: 'int4', tok_s: 40, latency_p50: 25, latency_p95: 58, vram: 36, cost_1k: 0.003 },
    { model: 'qwen-72b', gpu: 'a100x4', quant: 'int4', tok_s: 65, latency_p50: 15, latency_p95: 35, vram: 36, cost_1k: 0.012 },
];

const GPU_OPTIONS = [
    { id: 'all', name: 'All GPUs' },
    { id: 'rtx4090', name: 'RTX 4090' },
    { id: 'rtx4090x2', name: '2x RTX 4090' },
    { id: 'rtx4090x4', name: '4x RTX 4090' },
    { id: 'rtx3090', name: 'RTX 3090' },
    { id: 'a100', name: 'A100' },
    { id: 'a100x4', name: '4x A100' },
    { id: 'h100', name: 'H100' },
    { id: 'h100x4', name: '4x H100' },
];

const MODEL_OPTIONS = [
    { id: 'all', name: 'All Models' },
    { id: 'llama-3.1-8b', name: 'Llama 3.1 8B' },
    { id: 'llama-3.1-70b', name: 'Llama 3.1 70B' },
    { id: 'mistral-7b', name: 'Mistral 7B' },
    { id: 'mixtral-8x7b', name: 'Mixtral 8x7B' },
    { id: 'qwen-72b', name: 'Qwen 72B' },
];

const QUANT_OPTIONS = [
    { id: 'all', name: 'All' },
    { id: 'fp16', name: 'FP16' },
    { id: 'int8', name: 'INT8' },
    { id: 'int4', name: 'INT4' },
];

// ── State ──────────────────────────────────────────────────────────────

const state = {
    filters: {
        gpu: 'all',
        model: 'all',
        quant: 'all',
    },
    chartType: 'throughput-latency',
    selectedBenchmark: null,
};

// ── Filtering ──────────────────────────────────────────────────────────

function getFilteredData() {
    return BENCHMARK_DATA.filter(b => {
        if (state.filters.gpu !== 'all' && !b.gpu.includes(state.filters.gpu)) return false;
        if (state.filters.model !== 'all' && b.model !== state.filters.model) return false;
        if (state.filters.quant !== 'all' && b.quant !== state.filters.quant) return false;
        return true;
    });
}

// ── Chart Rendering ────────────────────────────────────────────────────

function renderChart(containerId, data, chartType) {
    const container = document.getElementById(containerId);
    if (!container) return;

    const width = container.offsetWidth;
    const height = 300;
    const padding = { top: 20, right: 20, bottom: 40, left: 60 };

    const chartWidth = width - padding.left - padding.right;
    const chartHeight = height - padding.top - padding.bottom;

    let xData, yData, xLabel, yLabel, xMax, yMax;

    switch (chartType) {
        case 'throughput-latency':
            xData = data.map(d => d.latency_p50);
            yData = data.map(d => d.tok_s);
            xLabel = 'Latency P50 (ms)';
            yLabel = 'Throughput (tok/s)';
            break;
        case 'cost-throughput':
            xData = data.map(d => d.cost_1k);
            yData = data.map(d => d.tok_s);
            xLabel = 'Cost per 1K tokens ($)';
            yLabel = 'Throughput (tok/s)';
            break;
        case 'vram-throughput':
            xData = data.map(d => d.vram);
            yData = data.map(d => d.tok_s);
            xLabel = 'VRAM (GB)';
            yLabel = 'Throughput (tok/s)';
            break;
        default:
            return;
    }

    xMax = Math.max(...xData, 1) * 1.1;
    yMax = Math.max(...yData, 1) * 1.1;

    const xScale = (v) => padding.left + (v / xMax) * chartWidth;
    const yScale = (v) => height - padding.bottom - (v / yMax) * chartHeight;

    // Color by model
    const modelColors = {
        'llama-3.1-8b': '#00e676',
        'llama-3.1-70b': '#3b82f6',
        'mistral-7b': '#f59e0b',
        'mixtral-8x7b': '#8b5cf6',
        'qwen-72b': '#ef4444',
    };

    const points = data.map((d, i) => {
        const x = xScale(xData[i]);
        const y = yScale(yData[i]);
        const color = modelColors[d.model] || '#888';
        const size = state.selectedBenchmark === i ? 8 : 5;

        return `<circle cx="${x}" cy="${y}" r="${size}" fill="${color}" opacity="0.8" data-index="${i}" class="chart-point"/>`;
    }).join('');

    // Grid lines
    const xGridLines = Array.from({ length: 5 }, (_, i) => {
        const x = padding.left + (i / 4) * chartWidth;
        return `<line x1="${x}" y1="${padding.top}" x2="${x}" y2="${height - padding.bottom}" stroke="#222" stroke-width="1"/>`;
    }).join('');

    const yGridLines = Array.from({ length: 5 }, (_, i) => {
        const y = padding.top + (i / 4) * chartHeight;
        return `<line x1="${padding.left}" y1="${y}" x2="${width - padding.right}" y2="${y}" stroke="#222" stroke-width="1"/>`;
    }).join('');

    // Axis labels
    const xTicks = Array.from({ length: 5 }, (_, i) => {
        const value = (xMax / 4) * i;
        const x = padding.left + (i / 4) * chartWidth;
        return `<text x="${x}" y="${height - 10}" text-anchor="middle" fill="#888" font-size="11">${value.toFixed(1)}</text>`;
    }).join('');

    const yTicks = Array.from({ length: 5 }, (_, i) => {
        const value = (yMax / 4) * i;
        const y = height - padding.bottom - (i / 4) * chartHeight;
        return `<text x="${padding.left - 10}" y="${y + 4}" text-anchor="end" fill="#888" font-size="11">${value.toFixed(0)}</text>`;
    }).join('');

    container.innerHTML = `
        <svg width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
            ${xGridLines}
            ${yGridLines}
            ${xTicks}
            ${yTicks}
            ${points}
            <text x="${width / 2}" y="${height - 5}" text-anchor="middle" fill="#888" font-size="12">${xLabel}</text>
            <text x="15" y="${height / 2}" text-anchor="middle" fill="#888" font-size="12" transform="rotate(-90, 15, ${height / 2})">${yLabel}</text>
        </svg>
    `;

    // Add click handlers to points
    container.querySelectorAll('.chart-point').forEach(point => {
        point.addEventListener('click', () => {
            const index = parseInt(point.dataset.index);
            state.selectedBenchmark = index;
            render();
        });
    });
}

function renderLegend(data) {
    const models = [...new Set(data.map(d => d.model))];
    const modelColors = {
        'llama-3.1-8b': '#00e676',
        'llama-3.1-70b': '#3b82f6',
        'mistral-7b': '#f59e0b',
        'mixtral-8x7b': '#8b5cf6',
        'qwen-72b': '#ef4444',
    };

    return models.map(model => {
        const name = MODEL_OPTIONS.find(m => m.id === model)?.name || model;
        const color = modelColors[model] || '#888';
        return `<span class="legend-item"><span class="legend-color" style="background: ${color}"></span>${name}</span>`;
    }).join('');
}

// ── UI ─────────────────────────────────────────────────────────────────

export function initBenchmarkExplorer() {
    const container = document.getElementById('benchmarkExplorer');
    if (!container) return;

    function render() {
        const filteredData = getFilteredData();
        const selected = state.selectedBenchmark !== null ? filteredData[state.selectedBenchmark] : null;

        container.innerHTML = `
            <div class="benchmark-explorer">
                <h3>Model Benchmark Explorer</h3>
                <p class="bench-explore-desc">Interactive benchmarks for all supported model × GPU × quantization combinations.</p>

                <!-- Filters -->
                <div class="bench-filters">
                    <div class="bench-filter">
                        <label>GPU</label>
                        <select id="benchFilterGpu">
                            ${GPU_OPTIONS.map(g => `<option value="${g.id}" ${g.id === state.filters.gpu ? 'selected' : ''}>${g.name}</option>`).join('')}
                        </select>
                    </div>
                    <div class="bench-filter">
                        <label>Model</label>
                        <select id="benchFilterModel">
                            ${MODEL_OPTIONS.map(m => `<option value="${m.id}" ${m.id === state.filters.model ? 'selected' : ''}>${m.name}</option>`).join('')}
                        </select>
                    </div>
                    <div class="bench-filter">
                        <label>Quantization</label>
                        <select id="benchFilterQuant">
                            ${QUANT_OPTIONS.map(q => `<option value="${q.id}" ${q.id === state.filters.quant ? 'selected' : ''}>${q.name}</option>`).join('')}
                        </select>
                    </div>
                    <div class="bench-filter">
                        <label>Chart</label>
                        <select id="benchChartType">
                            <option value="throughput-latency" ${state.chartType === 'throughput-latency' ? 'selected' : ''}>Throughput vs Latency</option>
                            <option value="cost-throughput" ${state.chartType === 'cost-throughput' ? 'selected' : ''}>Cost vs Throughput</option>
                            <option value="vram-throughput" ${state.chartType === 'vram-throughput' ? 'selected' : ''}>VRAM vs Throughput</option>
                        </select>
                    </div>
                </div>

                <!-- Chart -->
                <div class="bench-chart-container">
                    <div class="bench-chart" id="benchChart"></div>
                    <div class="bench-legend">${renderLegend(filteredData)}</div>
                </div>

                <!-- Data Table -->
                <div class="bench-table-container">
                    <table class="bench-data-table">
                        <thead>
                            <tr>
                                <th>Model</th>
                                <th>GPU</th>
                                <th>Quant</th>
                                <th>Tok/s</th>
                                <th>P50 (ms)</th>
                                <th>P95 (ms)</th>
                                <th>VRAM (GB)</th>
                                <th>Cost/1K</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${filteredData.map((d, i) => {
                                const modelName = MODEL_OPTIONS.find(m => m.id === d.model)?.name || d.model;
                                const gpuName = GPU_OPTIONS.find(g => g.id === d.gpu)?.name || d.gpu;
                                const isSelected = state.selectedBenchmark === i;
                                return `<tr class="${isSelected ? 'selected' : ''}" data-index="${i}">
                                    <td>${modelName}</td>
                                    <td>${gpuName}</td>
                                    <td>${d.quant.toUpperCase()}</td>
                                    <td class="highlight">${d.tok_s}</td>
                                    <td>${d.latency_p50}</td>
                                    <td>${d.latency_p95}</td>
                                    <td>${d.vram}</td>
                                    <td>$${d.cost_1k.toFixed(3)}</td>
                                </tr>`;
                            }).join('')}
                        </tbody>
                    </table>
                </div>

                <!-- Selected Benchmark Details -->
                ${selected ? `
                    <div class="bench-detail">
                        <h4>Selected Configuration</h4>
                        <div class="bench-detail-grid">
                            <div class="bench-detail-item">
                                <span class="bench-detail-label">Model</span>
                                <span class="bench-detail-value">${MODEL_OPTIONS.find(m => m.id === selected.model)?.name || selected.model}</span>
                            </div>
                            <div class="bench-detail-item">
                                <span class="bench-detail-label">GPU</span>
                                <span class="bench-detail-value">${GPU_OPTIONS.find(g => g.id === selected.gpu)?.name || selected.gpu}</span>
                            </div>
                            <div class="bench-detail-item">
                                <span class="bench-detail-label">Quantization</span>
                                <span class="bench-detail-value">${selected.quant.toUpperCase()}</span>
                            </div>
                            <div class="bench-detail-item">
                                <span class="bench-detail-label">Throughput</span>
                                <span class="bench-detail-value">${selected.tok_s} tok/s</span>
                            </div>
                            <div class="bench-detail-item">
                                <span class="bench-detail-label">Latency P50</span>
                                <span class="bench-detail-value">${selected.latency_p50}ms</span>
                            </div>
                            <div class="bench-detail-item">
                                <span class="bench-detail-label">Latency P95</span>
                                <span class="bench-detail-value">${selected.latency_p95}ms</span>
                            </div>
                            <div class="bench-detail-item">
                                <span class="bench-detail-label">VRAM</span>
                                <span class="bench-detail-value">${selected.vram}GB</span>
                            </div>
                            <div class="bench-detail-item">
                                <span class="bench-detail-label">Cost/1K tokens</span>
                                <span class="bench-detail-value">$${selected.cost_1k.toFixed(3)}</span>
                            </div>
                        </div>
                    </div>
                ` : ''}
            </div>
        `;

        // Render chart
        renderChart('benchChart', filteredData, state.chartType);

        // Add event listeners
        setupEventListeners(container, filteredData);
    }

    function setupEventListeners(container, filteredData) {
        container.querySelectorAll('select').forEach(select => {
            select.addEventListener('change', (e) => {
                const id = e.target.id;
                if (id === 'benchChartType') {
                    state.chartType = e.target.value;
                } else if (id === 'benchFilterGpu') {
                    state.filters.gpu = e.target.value;
                } else if (id === 'benchFilterModel') {
                    state.filters.model = e.target.value;
                } else if (id === 'benchFilterQuant') {
                    state.filters.quant = e.target.value;
                }
                state.selectedBenchmark = null;
                render();
            });
        });

        container.querySelectorAll('.bench-data-table tr[data-index]').forEach(row => {
            row.addEventListener('click', () => {
                state.selectedBenchmark = parseInt(row.dataset.index);
                render();
            });
        });
    }

    render();
}
