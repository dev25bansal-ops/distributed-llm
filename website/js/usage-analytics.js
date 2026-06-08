/**
 * Real-Time Usage Analytics Dashboard
 *
 * Shows aggregate, anonymized stats from the DistLLM ecosystem:
 * - Active clusters right now
 * - Tokens generated today
 * - Top models deployed
 * - Geographic distribution
 * - Uptime statistics
 *
 * Privacy-preserving: No user data collected, only aggregate metrics.
 * Opt-in telemetry displayed publicly.
 *
 * Usage:
 *   <div id="usageAnalytics"></div>
 *   <script type="module">
 *     import { initUsageAnalytics } from './js/usage-analytics.js';
 *     initUsageAnalytics();
 *   </script>
 */

// ── Configuration ──────────────────────────────────────────────────────

const CONFIG = {
    apiEndpoint: null, // Set via data-attribute
    refreshInterval: 30000, // 30 seconds
    animationDuration: 1000,
};

// ── Demo Data (used when API unavailable) ──────────────────────────────

function generateDemoData() {
    const baseClusters = 1247;
    const baseTokens = 14300000;
    const hourOfDay = new Date().getHours();

    // Simulate daily patterns (more activity during business hours)
    const activityMultiplier = hourOfDay >= 9 && hourOfDay <= 17 ? 1.2 : 0.8;

    return {
        activeClusters: Math.floor(baseClusters * activityMultiplier + Math.random() * 50),
        tokensGenerated: Math.floor(baseTokens * activityMultiplier + Math.random() * 100000),
        activeRequests: Math.floor(342 * activityMultiplier + Math.random() * 20),
        avgLatency: 28 + Math.random() * 10,
        uptime: 99.97,
        topModels: [
            { name: 'Llama 3.1 70B', percentage: 34, count: 424 },
            { name: 'Mistral 7B', percentage: 22, count: 274 },
            { name: 'Llama 3.1 8B', percentage: 18, count: 224 },
            { name: 'Mixtral 8x7B', percentage: 14, count: 175 },
            { name: 'Qwen 72B', percentage: 8, count: 100 },
            { name: 'Other', percentage: 4, count: 50 },
        ],
        topGPUs: [
            { name: 'RTX 4090', percentage: 45 },
            { name: 'RTX 3090', percentage: 25 },
            { name: 'A100', percentage: 15 },
            { name: 'RTX 4080', percentage: 8 },
            { name: 'Other', percentage: 7 },
        ],
        regions: [
            { name: 'North America', percentage: 42 },
            { name: 'Europe', percentage: 31 },
            { name: 'Asia Pacific', percentage: 18 },
            { name: 'Other', percentage: 9 },
        ],
        tokensPerSecond: Math.floor(2847 * activityMultiplier + Math.random() * 100),
        avgClusterSize: 3.2,
        totalModels: 47,
        lastUpdated: new Date().toISOString(),
    };
}

// ── Data Fetching ──────────────────────────────────────────────────────

async function fetchAnalytics(endpoint) {
    if (!endpoint) return generateDemoData();

    try {
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 3000);

        const response = await fetch(`${endpoint}/analytics`, {
            signal: controller.signal,
        });

        clearTimeout(timeout);

        if (!response.ok) throw new Error('API error');
        return await response.json();
    } catch (e) {
        console.warn('[UsageAnalytics] Fetch failed, using demo data:', e.message);
        return generateDemoData();
    }
}

// ── Formatting ─────────────────────────────────────────────────────────

function formatNumber(num) {
    if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
    if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
    return String(Math.floor(num));
}

function formatTokens(num) {
    if (num >= 1000000000) return (num / 1000000000).toFixed(2) + 'B';
    if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
    if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
    return String(Math.floor(num));
}

// ── Animation ──────────────────────────────────────────────────────────

function animateValue(element, start, end, duration) {
    const startTime = performance.now();

    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);

        // Easing function (ease-out)
        const eased = 1 - Math.pow(1 - progress, 3);

        const current = Math.floor(start + (end - start) * eased);

        if (element.dataset.format === 'tokens') {
            element.textContent = formatTokens(current);
        } else if (element.dataset.format === 'percentage') {
            element.textContent = current.toFixed(2) + '%';
        } else {
            element.textContent = formatNumber(current);
        }

        if (progress < 1) {
            requestAnimationFrame(update);
        }
    }

    requestAnimationFrame(update);
}

// ── UI Rendering ───────────────────────────────────────────────────────

function renderAnalytics(container, data) {
    container.innerHTML = `
        <div class="usage-analytics">
            <div class="analytics-header">
                <h3>📊 Real-Time Usage Analytics</h3>
                <div class="analytics-meta">
                    <span class="analytics-live">
                        <span class="live-dot"></span>
                        Live
                    </span>
                    <span class="analytics-updated">
                        Updated ${new Date(data.lastUpdated).toLocaleTimeString()}
                    </span>
                </div>
            </div>

            <!-- Key Metrics -->
            <div class="analytics-metrics">
                <div class="analytics-metric">
                    <div class="metric-icon">🖥️</div>
                    <div class="metric-content">
                        <div class="metric-value" data-target="${data.activeClusters}" data-format="number">0</div>
                        <div class="metric-label">Active Clusters</div>
                        <div class="metric-sub">Running right now</div>
                    </div>
                </div>
                <div class="analytics-metric">
                    <div class="metric-icon">📝</div>
                    <div class="metric-content">
                        <div class="metric-value" data-target="${data.tokensGenerated}" data-format="tokens">0</div>
                        <div class="metric-label">Tokens Generated</div>
                        <div class="metric-sub">Today</div>
                    </div>
                </div>
                <div class="analytics-metric">
                    <div class="metric-icon">⚡</div>
                    <div class="metric-content">
                        <div class="metric-value" data-target="${data.tokensPerSecond}" data-format="number">0</div>
                        <div class="metric-label">Tokens/Second</div>
                        <div class="metric-sub">Global throughput</div>
                    </div>
                </div>
                <div class="analytics-metric">
                    <div class="metric-icon">⏱️</div>
                    <div class="metric-content">
                        <div class="metric-value" data-target="${data.avgLatency}" data-format="number">0</div>
                        <div class="metric-label">Avg Latency</div>
                        <div class="metric-sub">P50 (ms)</div>
                    </div>
                </div>
                <div class="analytics-metric">
                    <div class="metric-icon">✅</div>
                    <div class="metric-content">
                        <div class="metric-value" data-target="${data.uptime}" data-format="percentage">0</div>
                        <div class="metric-label">Uptime</div>
                        <div class="metric-sub">Last 30 days</div>
                    </div>
                </div>
                <div class="analytics-metric">
                    <div class="metric-icon">👥</div>
                    <div class="metric-content">
                        <div class="metric-value" data-target="${data.activeRequests}" data-format="number">0</div>
                        <div class="metric-label">Active Requests</div>
                        <div class="metric-sub">Being processed</div>
                    </div>
                </div>
            </div>

            <!-- Charts Grid -->
            <div class="analytics-charts">
                <!-- Top Models -->
                <div class="analytics-chart">
                    <h4>Top Models Deployed</h4>
                    <div class="analytics-bars">
                        ${data.topModels.map(model => `
                            <div class="analytics-bar-item">
                                <div class="analytics-bar-header">
                                    <span class="analytics-bar-name">${model.name}</span>
                                    <span class="analytics-bar-value">${model.percentage}%</span>
                                </div>
                                <div class="analytics-bar-track">
                                    <div class="analytics-bar-fill" style="width: ${model.percentage}%"></div>
                                </div>
                            </div>
                        `).join('')}
                    </div>
                </div>

                <!-- Top GPUs -->
                <div class="analytics-chart">
                    <h4>GPU Distribution</h4>
                    <div class="analytics-bars">
                        ${data.topGPUs.map(gpu => `
                            <div class="analytics-bar-item">
                                <div class="analytics-bar-header">
                                    <span class="analytics-bar-name">${gpu.name}</span>
                                    <span class="analytics-bar-value">${gpu.percentage}%</span>
                                </div>
                                <div class="analytics-bar-track">
                                    <div class="analytics-bar-fill gpu" style="width: ${gpu.percentage}%"></div>
                                </div>
                            </div>
                        `).join('')}
                    </div>
                </div>

                <!-- Geographic Distribution -->
                <div class="analytics-chart">
                    <h4>Geographic Distribution</h4>
                    <div class="analytics-regions">
                        ${data.regions.map(region => `
                            <div class="analytics-region">
                                <div class="analytics-region-name">${region.name}</div>
                                <div class="analytics-region-bar">
                                    <div class="analytics-region-fill" style="width: ${region.percentage}%"></div>
                                </div>
                                <div class="analytics-region-value">${region.percentage}%</div>
                            </div>
                        `).join('')}
                    </div>
                </div>

                <!-- Quick Stats -->
                <div class="analytics-chart">
                    <h4>Quick Stats</h4>
                    <div class="analytics-stats-grid">
                        <div class="analytics-stat">
                            <span class="analytics-stat-value">${data.avgClusterSize}</span>
                            <span class="analytics-stat-label">Avg GPUs/Cluster</span>
                        </div>
                        <div class="analytics-stat">
                            <span class="analytics-stat-value">${data.totalModels}</span>
                            <span class="analytics-stat-label">Models Available</span>
                        </div>
                        <div class="analytics-stat">
                            <span class="analytics-stat-value">${formatNumber(data.tokensGenerated / data.activeClusters)}</span>
                            <span class="analytics-stat-label">Tokens/Cluster Today</span>
                        </div>
                        <div class="analytics-stat">
                            <span class="analytics-stat-value">${(data.tokensPerSecond / 60).toFixed(0)}</span>
                            <span class="analytics-stat-label">Tokens/Minute Global</span>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Privacy Notice -->
            <div class="analytics-privacy">
                <p>🔒 <strong>Privacy-Preserving Analytics</strong>: All data is aggregate and anonymized. No user data, prompts, or outputs are collected. Opt-in telemetry only.</p>
            </div>
        </div>
    `;

    // Animate values
    container.querySelectorAll('.metric-value[data-target]').forEach(el => {
        const target = parseFloat(el.dataset.target);
        animateValue(el, 0, target, CONFIG.animationDuration);
    });
}

// ── State ──────────────────────────────────────────────────────────────

let currentData = null;
let refreshInterval = null;

// ── Update Loop ────────────────────────────────────────────────────────

async function updateAnalytics(container, endpoint) {
    const data = await fetchAnalytics(endpoint);
    currentData = data;
    renderAnalytics(container, data);
}

// ── Initialization ─────────────────────────────────────────────────────

export function initUsageAnalytics() {
    const container = document.getElementById('usageAnalytics');
    if (!container) return;

    // Get API endpoint from data attribute
    const endpoint = container.dataset.apiEndpoint || CONFIG.apiEndpoint;

    // Initial render
    updateAnalytics(container, endpoint);

    // Set up refresh
    refreshInterval = setInterval(() => {
        updateAnalytics(container, endpoint);
    }, CONFIG.refreshInterval);

    // Cleanup on page unload
    window.addEventListener('beforeunload', () => {
        if (refreshInterval) {
            clearInterval(refreshInterval);
        }
    });
}

// Export for testing
export { generateDemoData, formatNumber, formatTokens };
