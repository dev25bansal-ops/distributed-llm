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

    function esc(str) {
        if (str === null || str === undefined) return '';
        const s = String(str);
        const map = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
        return s.replace(/[&<>"']/g, c => map[c]);
    }

    function connect() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = protocol + '//' + window.location.host + '/ws';

        wsStatus.textContent = 'Connecting...';
        wsStatus.className = 'status-badge connecting';

        ws = new WebSocket(wsUrl);

        ws.onopen = function () {
            wsStatus.textContent = 'Connected';
            wsStatus.className = 'status-badge connected';
        };

        ws.onmessage = function (event) {
            try {
                var data = JSON.parse(event.data);
                updateDashboard(data);
            } catch (e) {
                console.error('Failed to parse WebSocket message:', e);
            }
        };

        ws.onclose = function () {
            wsStatus.textContent = 'Disconnected';
            wsStatus.className = 'status-badge disconnected';
            reconnectTimer = setTimeout(connect, 5000);
        };

        ws.onerror = function () {
            ws.close();
        };
    }

    function updateDashboard(data) {
        if (data.type !== 'metrics' || !data.data) return;

        var d = data.data;

        // System overview
        if (d.model) modelName.textContent = d.model;
        if (d.nodes !== undefined) nodeCount.textContent = typeof d.nodes === 'object' ? Object.keys(d.nodes).length : d.nodes;

        // Connections
        wsConnections.textContent = 'active';

        // Node health
        if (d.nodes && typeof d.nodes === 'object') {
            nodeList.innerHTML = '';
            var entries = Object.entries(d.nodes);
            if (entries.length === 0) {
                var p = document.createElement('p');
                p.className = 'placeholder';
                p.textContent = 'No nodes connected';
                nodeList.appendChild(p);
            } else {
                entries.forEach(function (entry) {
                    var id = entry[0];
                    var info = entry[1];
                    var status = info.healthy ? 'healthy' : 'unhealthy';

                    var div = document.createElement('div');
                    div.className = 'node-item';

                    var idSpan = document.createElement('span');
                    idSpan.className = 'node-id';
                    idSpan.textContent = id;

                    var statusSpan = document.createElement('span');
                    statusSpan.className = 'node-status ' + status;
                    statusSpan.textContent = status;

                    div.appendChild(idSpan);
                    div.appendChild(statusSpan);
                    nodeList.appendChild(div);
                });
            }
        }

        // Scheduler stats
        if (d.scheduler) {
            var s = d.scheduler;
            schedulerStats.innerHTML = '';
            var stats = [
                { label: 'Active', value: s.active || 0 },
                { label: 'Pending', value: s.pending || 0 },
                { label: 'Completed', value: s.completed || 0 }
            ];
            stats.forEach(function (stat) {
                var row = document.createElement('div');
                row.className = 'stat-row';
                var label = document.createElement('span');
                label.className = 'stat-label';
                label.textContent = stat.label;
                var value = document.createElement('span');
                value.className = 'stat-value';
                value.textContent = stat.value;
                row.appendChild(label);
                row.appendChild(value);
                schedulerStats.appendChild(row);
            });
        }

        // Raw metrics
        rawMetrics.textContent = JSON.stringify(d, null, 2);
    }

    // Start connection when page loads
    connect();

    // Fetch waterfall data periodically
    function fetchWaterfall() {
        fetch('/api/requests/waterfall?limit=50')
            .then(function (r) { return r.json(); })
            .then(function (data) {
                if (data && data.length > 0) {
                    renderWaterfall(data);
                }
            })
            .catch(function () {});
    }

    function renderWaterfall(items) {
        var maxElapsed = Math.max.apply(null, items.map(function (i) { return i.elapsed_ms || 0; }).concat([1]));

        waterfallContainer.innerHTML = '';
        var header = document.createElement('div');
        header.className = 'waterfall-header';
        var hStart = document.createElement('span');
        hStart.textContent = '0ms';
        var hEnd = document.createElement('span');
        hEnd.textContent = maxElapsed.toFixed(0) + 'ms';
        header.appendChild(hStart);
        header.appendChild(hEnd);
        waterfallContainer.appendChild(header);

        var rowsDiv = document.createElement('div');
        rowsDiv.className = 'waterfall-rows';

        items.forEach(function (item) {
            var ttft = item.ttft_ms || 0;
            var elapsed = item.elapsed_ms || 0;
            var ttftPct = Math.min((ttft / maxElapsed) * 100, 100);
            var totalPct = Math.min((elapsed / maxElapsed) * 100, 100);
            var decodePct = Math.max(totalPct - ttftPct, 0);

            var statusClass = item.is_overdue ? 'waterfall-overdue' : 'waterfall-ok';
            var row = document.createElement('div');
            row.className = 'waterfall-row ' + statusClass;

            var label = document.createElement('div');
            label.className = 'waterfall-label';
            label.title = esc(item.request_id);
            label.textContent = esc((item.request_id || '').substring(0, 12)) + '...';

            var bar = document.createElement('div');
            bar.className = 'waterfall-bar';

            var prefill = document.createElement('div');
            prefill.className = 'waterfall-segment waterfall-prefill';
            prefill.style.width = ttftPct + '%';
            prefill.title = 'Prefill: ' + ttft.toFixed(0) + 'ms';

            var decode = document.createElement('div');
            decode.className = 'waterfall-segment waterfall-decode';
            decode.style.width = decodePct + '%';
            decode.title = 'Decode: ' + (elapsed - ttft).toFixed(0) + 'ms';

            bar.appendChild(prefill);
            bar.appendChild(decode);

            var time = document.createElement('div');
            time.className = 'waterfall-time';
            time.textContent = elapsed.toFixed(0) + 'ms';

            row.appendChild(label);
            row.appendChild(bar);
            row.appendChild(time);
            rowsDiv.appendChild(row);
        });

        waterfallContainer.appendChild(rowsDiv);
    }

    // Start waterfall polling
    fetchWaterfall();
    waterfallTimer = setInterval(fetchWaterfall, 3000);
})();
