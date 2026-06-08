/**
 * Live Cluster Status Dashboard — real-time metrics from DistLLM demo cluster.
 *
 * Shows:
 * - Connected GPUs with utilization
 * - Current model loaded
 * - Active requests
 * - Tokens/sec throughput
 * - Latency p50/p95/p99
 *
 * Usage:
 *   <div id="liveCluster"></div>
 *   <script type="module">
 *     import { initLiveCluster } from './js/live-cluster.js';
 *     initLiveCluster();
 *   </script>
 */

// ── Configuration ──────────────────────────────────────────────────────

const CONFIG = {
    // API endpoint for cluster status (configurable via data attribute)
    endpoint: null,
    // Refresh interval in ms
    refreshInterval: 5000,
    // Max data points for charts
    maxDataPoints: 60,
    // Colors
    colors: {
        green: '#00e676',
        amber: '#f59e0b',
        red: '#ef4444',
        blue: '#3b82f6',
        purple: '#8b5cf6',
        dim: '#888',
        surface: '#111',
        border: '#222',
    },
};

// ── State ──────────────────────────────────────────────────────────────

const state = {
    metrics: {
        gpus: [],
        model: null,
        activeRequests: 0,
        tokensPerSec: 0,
        latency: { p50: 0, p95: 0, p99: 0 },
        uptime: 0,
        totalRequests: 0,
        totalTokens: 0,
    },
    history: {
        tokensPerSec: [],
        latency: [],
        requests: [],
    },
    intervalId: null,
    isLive: false,
};

// ── Data Fetching ──────────────────────────────────────────────────────

async function fetchClusterStatus(endpoint) {
    if (!endpoint) {
        // Return demo data if no endpoint configured
        return generateDemoData();
    }

    try {
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 3000);

        const response = await fetch(`${endpoint}/status`, {
            signal: controller.signal,
        });

        clearTimeout(timeout);

        if (!response.ok) throw new Error('API error');
        return await response.json();
    } catch (e) {
        console.warn('[LiveCluster] Fetch failed, using demo data:', e.message);
        return generateDemoData();
    }
}

function generateDemoData() {
    const now = Date.now();
    const gpus = [
        { id: 'gpu-0', name: 'RTX 4090', util: 0.72 + Math.random() * 0.1, mem: 0.65, temp: 68 + Math.random() * 5, vram: { used: 15.6, total: 24 } },
        { id: 'gpu-1', name: 'RTX 4090', util: 0.85 + Math.random() * 0.1, mem: 0.78, temp: 74 + Math.random() * 5, vram: { used: 18.7, total: 24 } },
        { id: 'gpu-2', name: 'RTX 3090', util: 0.61 + Math.random() * 0.1, mem: 0.52, temp: 62 + Math.random() * 5, vram: { used: 12.5, total: 24 } },
        { id: 'gpu-3', name: 'RTX 3090', util: 0.93 + Math.random() * 0.05, mem: 0.88, temp: 81 + Math.random() * 5, vram: { used: 21.1, total: 24 } },
    ];

    return {
        status: 'online',
        model: 'Llama-3.1-70B',
        quantization: 'GPTQ-4bit',
        gpus,
        activeRequests: Math.floor(Math.random() * 10) + 1,
        tokensPerSec: 42.3 + Math.random() * 10,
        latency: {
            p50: 28 + Math.random() * 10,
            p95: 65 + Math.random() * 20,
            p99: 120 + Math.random() * 30,
        },
        uptime: 864000 + Math.floor(Math.random() * 86400),
        totalRequests: 15420 + Math.floor(Math.random() * 1000),
        totalTokens: 2847563 + Math.floor(Math.random() * 100000),
        timestamp: now,
    };
}

// ── UI Rendering ───────────────────────────────────────────────────────

function formatUptime(seconds) {
    const days = Math.floor(seconds / 86400);
    const hours = Math.floor((seconds % 86400) / 3600);
    const mins = Math.floor((seconds % 3600) / 60);

    if (days > 0) return `${days}d ${hours}h`;
    if (hours > 0) return `${hours}h ${mins}m`;
    return `${mins}m`;
}

function formatNumber(num) {
    if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
    if (num >= 1000) return (num / 1000).toFixed(1) + 'k';
    return String(Math.floor(num));
}

function getStatusColor(util) {
    if (util > 0.9) return CONFIG.colors.red;
    if (util > 0.7) return CONFIG.colors.amber;
    return CONFIG.colors.green;
}

function renderDashboard(container) {
    const { metrics, history, isLive } = state;

    container.innerHTML = `
        <div class="live-cluster">
            <!-- Header -->
            <div class="live-header">
                <div class="live-title">
                    <h3>Live Cluster Dashboard</h3>
                    <div class="live-status ${isLive ? 'online' : 'offline'}">
                        <span class="live-dot"></span>
                        ${isLive ? 'Live' : 'Connecting...'}
                    </div>
                </div>
                <div class="live-meta">
                    <span class="live-model">${metrics.model || 'No model loaded'}</span>
                    ${metrics.quantization ? `<span class="live-quant">${metrics.quantization}</span>` : ''}
                    <span class="live-uptime">Uptime: ${formatUptime(metrics.uptime)}</span>
                </div>
            </div>

            <!-- Key Metrics -->
            <div class="live-metrics">
                <div class="metric-card">
                    <div class="metric-label">Tokens/sec</div>
                    <div class="metric-value">${metrics.tokensPerSec.toFixed(1)}</div>
                    <div class="metric-chart" id="tpsChart"></div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Active Requests</div>
                    <div class="metric-value">${metrics.activeRequests}</div>
                    <div class="metric-sub">Total: ${formatNumber(metrics.totalRequests)}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Latency P50</div>
                    <div class="metric-value">${metrics.latency.p50.toFixed(0)}<span class="metric-unit">ms</span></div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Latency P95</div>
                    <div class="metric-value">${metrics.latency.p95.toFixed(0)}<span class="metric-unit">ms</span></div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Latency P99</div>
                    <div class="metric-value">${metrics.latency.p99.toFixed(0)}<span class="metric-unit">ms</span></div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Total Tokens</div>
                    <div class="metric-value">${formatNumber(metrics.totalTokens)}</div>
                </div>
            </div>

            <!-- GPU Grid -->
            <div class="live-gpus">
                <h4>Connected GPUs</h4>
                <div class="gpu-grid">
                    ${metrics.gpus.map(gpu => `
                        <div class="gpu-card">
                            <div class="gpu-header">
                                <span class="gpu-name">${gpu.name}</span>
                                <span class="gpu-id">${gpu.id}</span>
                            </div>
                            <div class="gpu-util-bar">
                                <div class="gpu-util-fill" style="width: ${gpu.util * 100}%; background: ${getStatusColor(gpu.util)}"></div>
                            </div>
                            <div class="gpu-stats">
                                <div class="gpu-stat">
                                    <span class="gpu-stat-label">Util</span>
                                    <span class="gpu-stat-value">${(gpu.util * 100).toFixed(0)}%</span>
                                </div>
                                <div class="gpu-stat">
                                    <span class="gpu-stat-label">VRAM</span>
                                    <span class="gpu-stat-value">${gpu.vram.used.toFixed(1)}/${gpu.vram.total}GB</span>
                                </div>
                                <div class="gpu-stat">
                                    <span class="gpu-stat-label">Temp</span>
                                    <span class="gpu-stat-value">${gpu.temp.toFixed(0)}°C</span>
                                </div>
                            </div>
                        </div>
                    `).join('')}
                </div>
            </div>

            <!-- Latency Chart -->
            <div class="live-chart-section">
                <h4>Latency History</h4>
                <div class="latency-chart" id="latencyChart"></div>
            </div>
        </div>
    `;

    // Render mini charts
    renderMiniChart('tpsChart', history.tokensPerSec, CONFIG.colors.green);
    renderLatencyChart('latencyChart', history.latency);
}

function renderMiniChart(containerId, data, color) {
    const container = document.getElementById(containerId);
    if (!container || data.length < 2) return;

    const width = container.offsetWidth;
    const height = 40;
    const max = Math.max(...data, 1);
    const step = width / (data.length - 1);

    const points = data.map((val, i) => {
        const x = i * step;
        const y = height - (val / max) * (height - 4);
        return `${x},${y}`;
    }).join(' ');

    container.innerHTML = `
        <svg width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
            <polyline
                points="${points}"
                fill="none"
                stroke="${color}"
                stroke-width="2"
                stroke-linecap="round"
                stroke-linejoin="round"
            />
        </svg>
    `;
}

function renderLatencyChart(containerId, data) {
    const container = document.getElementById(containerId);
    if (!container || data.length < 2) return;

    const width = container.offsetWidth;
    const height = 120;
    const max = Math.max(...data.map(d => d.p99), 100);
    const step = width / (data.length - 1);

    const createPath = (key) => {
        return data.map((d, i) => {
            const x = i * step;
            const y = height - (d[key] / max) * (height - 10);
            return `${i === 0 ? 'M' : 'L'} ${x} ${y}`;
        }).join(' ');
    };

    container.innerHTML = `
        <svg width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
            <path d="${createPath('p99')}" fill="none" stroke="${CONFIG.colors.red}" stroke-width="1.5" opacity="0.5"/>
            <path d="${createPath('p95')}" fill="none" stroke="${CONFIG.colors.amber}" stroke-width="1.5" opacity="0.7"/>
            <path d="${createPath('p50')}" fill="none" stroke="${CONFIG.colors.green}" stroke-width="2"/>
        </svg>
        <div class="chart-legend">
            <span><span class="legend-dot" style="background: ${CONFIG.colors.green}"></span> P50</span>
            <span><span class="legend-dot" style="background: ${CONFIG.colors.amber}"></span> P95</span>
            <span><span class="legend-dot" style="background: ${CONFIG.colors.red}"></span> P99</span>
        </div>
    `;
}

// ── Update Loop ────────────────────────────────────────────────────────

async function updateMetrics(endpoint) {
    const data = await fetchClusterStatus(endpoint);

    // Update metrics
    state.metrics = {
        gpus: data.gpus || [],
        model: data.model || null,
        quantization: data.quantization || null,
        activeRequests: data.activeRequests || 0,
        tokensPerSec: data.tokensPerSec || 0,
        latency: data.latency || { p50: 0, p95: 0, p99: 0 },
        uptime: data.uptime || 0,
        totalRequests: data.totalRequests || 0,
        totalTokens: data.totalTokens || 0,
    };

    state.isLive = data.status === 'online';

    // Update history
    state.history.tokensPerSec.push(state.metrics.tokensPerSec);
    state.history.latency.push(state.metrics.latency);
    state.history.requests.push(state.metrics.activeRequests);

    // Trim to max data points
    if (state.history.tokensPerSec.length > CONFIG.maxDataPoints) {
        state.history.tokensPerSec.shift();
        state.history.latency.shift();
        state.history.requests.shift();
    }
}

// ── Initialization ─────────────────────────────────────────────────────

export function initLiveCluster() {
    const container = document.getElementById('liveCluster');
    if (!container) return;

    // Get endpoint from data attribute
    const endpoint = container.dataset.endpoint || CONFIG.endpoint;

    // Initial render
    renderDashboard(container);

    // Start update loop
    const update = async () => {
        await updateMetrics(endpoint);
        renderDashboard(container);
    };

    // Initial fetch
    update();

    // Set interval
    state.intervalId = setInterval(update, CONFIG.refreshInterval);

    // Cleanup on page unload
    window.addEventListener('beforeunload', () => {
        if (state.intervalId) {
            clearInterval(state.intervalId);
        }
    });
}
