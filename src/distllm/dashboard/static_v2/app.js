/* DistLLM Dashboard v2 - Frontend JavaScript */

(function () {
    'use strict';

    const wsStatus = document.getElementById('ws-status');
    const modelName = document.getElementById('model-name');
    const nodeCount = document.getElementById('node-count');
    const wsConnections = document.getElementById('ws-connections');
    const nodeList = document.getElementById('node-list');
    const schedulerStats = document.getElementById('scheduler-stats');
    const rawMetrics = document.getElementById('raw-metrics');
    const waterfallContainer = document.getElementById('waterfall-container');

    let ws = null;
    let reconnectTimer = null;
    let waterfallTimer = null;

    function connect() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;

        wsStatus.textContent = 'Connecting...';
        wsStatus.className = 'status-badge connecting';

        ws = new WebSocket(wsUrl);

        ws.onopen = () => {
            wsStatus.textContent = 'Connected';
            wsStatus.className = 'status-badge connected';
        };

        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                updateDashboard(data);
            } catch (e) {
                console.error('Failed to parse WebSocket message:', e);
            }
        };

        ws.onclose = () => {
            wsStatus.textContent = 'Disconnected';
            wsStatus.className = 'status-badge disconnected';
            // Reconnect after 5 seconds
            reconnectTimer = setTimeout(connect, 5000);
        };

        ws.onerror = () => {
            ws.close();
        };
    }

    function updateDashboard(data) {
        if (data.type !== 'metrics' || !data.data) return;

        const d = data.data;

        // System overview
        if (d.model) modelName.textContent = d.model;
        if (d.nodes !== undefined) nodeCount.textContent = typeof d.nodes === 'object' ? Object.keys(d.nodes).length : d.nodes;

        // Connections
        wsConnections.textContent = 'active';

        // Node health
        if (d.nodes && typeof d.nodes === 'object') {
            const html = Object.entries(d.nodes).map(([id, info]) => {
                const status = info.healthy ? 'healthy' : 'unhealthy';
                return `<div class="node-item">
                    <span class="node-id">${id}</span>
                    <span class="node-status ${status}">${status}</span>
                </div>`;
            }).join('');
            nodeList.innerHTML = html || '<p class="placeholder">No nodes connected</p>';
        }

        // Scheduler stats
        if (d.scheduler) {
            const s = d.scheduler;
            schedulerStats.innerHTML = `
                <div class="stat-row"><span class="stat-label">Active</span><span class="stat-value">${s.active || 0}</span></div>
                <div class="stat-row"><span class="stat-label">Pending</span><span class="stat-value">${s.pending || 0}</span></div>
                <div class="stat-row"><span class="stat-label">Completed</span><span class="stat-value">${s.completed || 0}</span></div>
            `;
        }

        // Raw metrics
        rawMetrics.textContent = JSON.stringify(d, null, 2);
    }

    // Start connection when page loads
    connect();

    // Fetch waterfall data periodically
    function fetchWaterfall() {
        fetch('/api/requests/waterfall?limit=50')
            .then(r => r.json())
            .then(data => {
                if (data && data.length > 0) {
                    waterfallContainer.innerHTML = renderWaterfall(data);
                }
            })
            .catch(() => {});
    }

    function renderWaterfall(items) {
        const maxElapsed = Math.max(...items.map(i => i.elapsed_ms || 0), 1);
        const rows = items.map(item => {
            const ttft = item.ttft_ms || 0;
            const elapsed = item.elapsed_ms || 0;
            const ttftPct = Math.min((ttft / maxElapsed) * 100, 100);
            const totalPct = Math.min((elapsed / maxElapsed) * 100, 100);
            const decodePct = Math.max(totalPct - ttftPct, 0);

            const statusClass = item.is_overdue ? 'waterfall-overdue' : 'waterfall-ok';
            return `<div class="waterfall-row ${statusClass}">
                <div class="waterfall-label" title="${item.request_id}">${item.request_id.substring(0, 12)}...</div>
                <div class="waterfall-bar">
                    <div class="waterfall-segment waterfall-prefill" style="width: ${ttftPct}%" title="Prefill: ${ttft.toFixed(0)}ms"></div>
                    <div class="waterfall-segment waterfall-decode" style="width: ${decodePct}%" title="Decode: ${(elapsed - ttft).toFixed(0)}ms"></div>
                </div>
                <div class="waterfall-time">${elapsed.toFixed(0)}ms</div>
            </div>`;
        }).join('');

        return `<div class="waterfall-header">
            <span>0ms</span><span>${maxElapsed.toFixed(0)}ms</span>
        </div>
        <div class="waterfall-rows">${rows}</div>`;
    }

    // Start waterfall polling
    fetchWaterfall();
    waterfallTimer = setInterval(fetchWaterfall, 3000);
})();
