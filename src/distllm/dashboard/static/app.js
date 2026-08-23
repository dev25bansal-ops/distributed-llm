/* ======================================================================
   DistLLM Dashboard — App JavaScript
   Features:
     - Auto-detect + manual theme toggle (localStorage persisted)
     - Skeleton loader show/hide
     - Exponential backoff WebSocket (1s, 2s, 4s, 8s, max 30s)
     - Connection quality indicator (green / yellow / red dot)
     - Chart.js real-time GPU utilization and memory line charts
     - GPU detail expandable sections (temp, power, processes)
   ====================================================================== */
(function () {
  'use strict';

  // ------------------------------------------------------------------
  // DOM refs
  // ------------------------------------------------------------------
  const $ = (id) => document.getElementById(id);
  const body = document.body;

  // --- Theme ---
  const themeToggle = $('theme-toggle');
  const themeIcon = $('theme-icon');
  const themeLabel = $('theme-label');

  // --- Connection ---
  const connDot = $('connection-dot');
  const connLabel = $('ws-label');

  // --- Skeleton containers ---
  const skeletonAreas = {
    'skeleton-overview': $('skeleton-overview'),
    'skeleton-latency': $('skeleton-latency'),
    'skeleton-histogram': $('skeleton-histogram'),
    'skeleton-kv': $('skeleton-kv'),
    'skeleton-spec': $('skeleton-spec'),
    'skeleton-cost': $('skeleton-cost'),
    'skeleton-stream': $('skeleton-stream'),
    'skeleton-throughput': $('skeleton-throughput'),
    'skeleton-nodes': $('skeleton-nodes'),
    'skeleton-waterfall': $('skeleton-waterfall'),
    'skeleton-tenants': $('skeleton-tenants'),
  };

  // ------------------------------------------------------------------
  // 1. THEME
  // ------------------------------------------------------------------
  function initTheme() {
    const saved = localStorage.getItem('distllm-theme');
    if (saved === 'light' || saved === 'dark') {
      document.documentElement.setAttribute('data-theme', saved);
    } else {
      // Auto-detect: no attribute means CSS media query handles it
      document.documentElement.removeAttribute('data-theme');
    }
    updateThemeUI();
  }

  function updateThemeUI() {
    const theme = getEffectiveTheme();
    if (themeIcon) {
      themeIcon.textContent = theme === 'dark' ? '☀' : '🌙';
    }
    if (themeLabel) {
      themeLabel.textContent = theme === 'dark' ? 'Light' : 'Dark';
    }
  }

  function getEffectiveTheme() {
    const attr = document.documentElement.getAttribute('data-theme');
    if (attr === 'light') return 'light';
    if (attr === 'dark') return 'dark';
    return window.matchMedia('(prefers-color-scheme: light)').matches
      ? 'light'
      : 'dark';
  }

  function toggleTheme() {
    const current = getEffectiveTheme();
    const next = current === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('distllm-theme', next);
    updateThemeUI();
    // Notify Chart.js instances about theme change
    if (window.dashboardCharts) {
      Object.values(window.dashboardCharts).forEach(function (chart) {
        if (chart && chart.update) chart.update();
      });
    }
  }

  if (themeToggle) {
    themeToggle.addEventListener('click', toggleTheme);
  }

  // Listen for OS theme changes when no manual preference is saved
  window.matchMedia('(prefers-color-scheme: light)').addEventListener('change', function () {
    if (!localStorage.getItem('distllm-theme')) {
      updateThemeUI();
      if (window.dashboardCharts) {
        Object.values(window.dashboardCharts).forEach(function (chart) {
          if (chart && chart.update) chart.update();
        });
      }
    }
  });

  initTheme();

  // ------------------------------------------------------------------
  // 2. SKELETON LOADER
  // ------------------------------------------------------------------
  function showSkeleton(areaId) {
    const el = skeletonAreas[areaId] || $(areaId);
    if (el) {
      el.style.display = '';
    }
  }

  function hideSkeleton(areaId) {
    const el = skeletonAreas[areaId] || $(areaId);
    if (el) {
      el.style.display = 'none';
    }
  }

  function hideAllSkeletons() {
    Object.keys(skeletonAreas).forEach(function (id) {
      const el = skeletonAreas[id];
      if (el) el.style.display = 'none';
    });
  }

  function showContentSections() {
    // Show corresponding content sections when skeleton hidden
    var contentIds = [
      'content-overview', 'content-latency', 'content-histogram',
      'content-kv', 'content-spec', 'content-cost', 'content-stream',
      'content-throughput', 'content-nodes', 'content-waterfall', 'content-tenants',
    ];
    contentIds.forEach(function (id) {
      var el = $(id);
      if (el) el.style.display = '';
    });
  }

  // ------------------------------------------------------------------
  // 3. EXPONENTIAL BACKOFF WEBSOCKET
  // ------------------------------------------------------------------
  var ws = null;
  var reconnectAttempt = 0;
  var reconnectTimer = null;
  var wsConnected = false;
  var lastMessageTime = 0;
  var connectionCheckInterval = null;

  var BACKOFF_INITIAL = 1000;    // 1s
  var BACKOFF_MAX = 30000;       // 30s
  var STALE_THRESHOLD = 10000;   // 10s no message = degraded

  function getBackoffDelay(attempt) {
    var delay = BACKOFF_INITIAL * Math.pow(2, attempt);
    return Math.min(delay, BACKOFF_MAX);
  }

  function updateConnectionState(state, reason) {
    // state: 'connected' | 'degraded' | 'disconnected' | 'connecting'
    if (!connDot || !connLabel) return;

    connDot.className = 'connection-dot';

    switch (state) {
      case 'connected':
        connDot.classList.add('connected');
        connLabel.textContent = 'Connected';
        break;
      case 'degraded':
        connDot.classList.add('degraded');
        connLabel.textContent = 'Degraded';
        break;
      case 'disconnected':
        connDot.classList.add('disconnected');
        connLabel.textContent = reason || 'Disconnected';
        break;
      case 'connecting':
        connDot.classList.add('connecting');
        connLabel.textContent = 'Connecting...';
        break;
      default:
        connDot.classList.add('disconnected');
        connLabel.textContent = 'Unknown';
    }
  }

  function startConnectionMonitor() {
    if (connectionCheckInterval) clearInterval(connectionCheckInterval);
    connectionCheckInterval = setInterval(function () {
      if (!wsConnected) return;
      var elapsed = Date.now() - lastMessageTime;
      if (elapsed > STALE_THRESHOLD) {
        updateConnectionState('degraded');
      } else {
        updateConnectionState('connected');
      }
    }, 2000);
  }

  function connectWebSocket() {
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
      return;
    }

    var protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    var wsUrl = protocol + '//' + window.location.host + '/ws';

    updateConnectionState('connecting');
    wsConnected = false;

    ws = new WebSocket(wsUrl);

    ws.onopen = function () {
      wsConnected = true;
      reconnectAttempt = 0;
      lastMessageTime = Date.now();
      updateConnectionState('connected');
      // Subscribe to all metrics with 1s interval
      ws.send(JSON.stringify({
        type: 'subscribe',
        metrics: [
          'latency', 'ttft', 'throughput', 'kv_cache',
          'speculative', 'cost', 'scheduler', 'nodes', 'gpu',
        ],
        interval: 1.0,
      }));
    };

    ws.onmessage = function (event) {
      lastMessageTime = Date.now();
      if (updateConnectionState && wsConnected) {
        updateConnectionState('connected');
      }
      try {
        var msg = JSON.parse(event.data);
        if (msg.type === 'subscribed') {
          console.log('Subscribed to metrics:', msg.metrics);
          return;
        }
        if (msg.type === 'pong') return;
        if (msg.type === 'error') {
          console.error('WS error:', msg.detail);
          return;
        }
        updateDashboard(msg);
      } catch (e) {
        console.error('Parse error:', e);
      }
    };

    ws.onclose = function () {
      wsConnected = false;
      updateConnectionState('disconnected', 'Reconnecting...');
      scheduleReconnect();
    };

    ws.onerror = function () {
      // onclose will fire after this
      ws.close();
    };
  }

  function scheduleReconnect() {
    if (reconnectTimer) {
      clearTimeout(reconnectTimer);
      reconnectTimer = null;
    }
    var delay = getBackoffDelay(reconnectAttempt);
    reconnectAttempt++;
    console.log('WebSocket reconnecting in ' + delay + 'ms (attempt ' + reconnectAttempt + ')');
    reconnectTimer = setTimeout(function () {
      connectWebSocket();
    }, delay);
  }

  function cancelReconnect() {
    if (reconnectTimer) {
      clearTimeout(reconnectTimer);
      reconnectTimer = null;
    }
  }

  // ------------------------------------------------------------------
  // 4. UTILITY FUNCTIONS
  // ------------------------------------------------------------------
  function fmt(v, d) {
    if (d === undefined) d = 0;
    return typeof v === 'number' ? v.toFixed(d) : (v ?? '--');
  }

  function fmtPct(v) {
    return typeof v === 'number' ? (v * 100).toFixed(1) + '%' : '--';
  }

  function fmtMoney(v) {
    return typeof v === 'number' ? '$' + v.toFixed(6) : '--';
  }

  function fmtTime(ts) {
    if (!ts) return '--';
    var s = Math.floor(ts);
    var m = Math.floor(s / 60);
    var h = Math.floor(m / 60);
    return h > 0
      ? h + 'h ' + (m % 60) + 'm'
      : m > 0
        ? m + 'm ' + (s % 60) + 's'
        : s + 's';
  }

  function esc(str) {
    if (str === null || str === undefined) return '';
    var s = String(str);
    var map = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
    return s.replace(/[&<>"']/g, function (c) { return map[c]; });
  }

  // ------------------------------------------------------------------
  // 5. GPU CHARTS (Chart.js)
  // ------------------------------------------------------------------
  var gpuCharts = {};
  var gpuHistory = {};
  var MAX_HISTORY = 60; // 60 seconds of data

  function ensureGpuHistory(nodeId) {
    if (!gpuHistory[nodeId]) {
      gpuHistory[nodeId] = {
        labels: [],
        utilization: [],
        memory: [],
      };
    }
    return gpuHistory[nodeId];
  }

  function getChartColors() {
    var isDark = getEffectiveTheme() === 'dark';
    return {
      grid: isDark ? 'rgba(148, 163, 184, 0.15)' : 'rgba(100, 116, 139, 0.2)',
      text: isDark ? '#94a3b8' : '#64748b',
      utilLine: isDark ? '#38bdf8' : '#0ea5e9',
      utilFill: isDark ? 'rgba(56, 189, 248, 0.15)' : 'rgba(14, 165, 233, 0.15)',
      memLine: isDark ? '#a78bfa' : '#8b5cf6',
      memFill: isDark ? 'rgba(167, 139, 250, 0.15)' : 'rgba(139, 92, 246, 0.15)',
    };
  }

  function createGpuChart(canvasId, label, lineColor, fillColor) {
    var canvas = $(canvasId);
    if (!canvas) return null;

    var ctx = canvas.getContext('2d');
    var colors = getChartColors();

    var chart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: [],
        datasets: [{
          label: label,
          data: [],
          borderColor: lineColor || colors.utilLine,
          backgroundColor: fillColor || colors.utilFill,
          borderWidth: 1.5,
          pointRadius: 0,
          pointHitRadius: 6,
          tension: 0.3,
          fill: true,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 300 },
        plugins: {
          legend: { display: false },
          tooltip: {
            mode: 'index',
            intersect: false,
            backgroundColor: isDarkBackground() ? '#1e293b' : '#ffffff',
            titleColor: isDarkBackground() ? '#e2e8f0' : '#1e293b',
            bodyColor: isDarkBackground() ? '#94a3b8' : '#64748b',
            borderColor: isDarkBackground() ? '#334155' : '#e2e8f0',
            borderWidth: 1,
            padding: 8,
            cornerRadius: 6,
            displayColors: false,
          },
        },
        scales: {
          x: {
            display: true,
            grid: { color: colors.grid, drawBorder: false },
            ticks: {
              color: colors.text,
              maxTicksLimit: 6,
              font: { size: 9 },
              maxRotation: 0,
            },
          },
          y: {
            display: true,
            min: 0,
            max: 100,
            grid: { color: colors.grid, drawBorder: false },
            ticks: {
              color: colors.text,
              font: { size: 9 },
              callback: function (val) { return val + '%'; },
            },
          },
        },
      },
    });

    return chart;
  }

  function isDarkBackground() {
    return getEffectiveTheme() === 'dark';
  }

  function updateGpuCharts(nodeId, utilPct, memPct) {
    var now = new Date();
    var timeStr = now.getHours().toString().padStart(2, '0') + ':' +
                  now.getMinutes().toString().padStart(2, '0') + ':' +
                  now.getSeconds().toString().padStart(2, '0');

    var history = ensureGpuHistory(nodeId);
    history.labels.push(timeStr);
    history.utilization.push(Math.round(utilPct * 10) / 10);

    if (memPct !== undefined && memPct !== null) {
      history.memory.push(Math.round(memPct * 10) / 10);
    }

    // Trim to max history
    while (history.labels.length > MAX_HISTORY) {
      history.labels.shift();
      history.utilization.shift();
      if (history.memory.length > MAX_HISTORY) history.memory.shift();
    }

    // Update util chart
    var utilChart = gpuCharts[nodeId + '-util'];
    if (utilChart && history.utilization.length > 0) {
      utilChart.data.labels = history.labels.slice();
      utilChart.data.datasets[0].data = history.utilization.slice();
      var colors = getChartColors();
      utilChart.data.datasets[0].borderColor = colors.utilLine;
      utilChart.data.datasets[0].backgroundColor = colors.utilFill;
      utilChart.options.scales.x.grid.color = colors.grid;
      utilChart.options.scales.x.ticks.color = colors.text;
      utilChart.options.scales.y.grid.color = colors.grid;
      utilChart.options.scales.y.ticks.color = colors.text;
      utilChart.update('none');
    }

    // Update memory chart
    var memChart = gpuCharts[nodeId + '-mem'];
    if (memChart && history.memory.length > 0) {
      memChart.data.labels = history.labels.slice();
      memChart.data.datasets[0].data = history.memory.slice();
      var colors2 = getChartColors();
      memChart.data.datasets[0].borderColor = colors2.memLine;
      memChart.data.datasets[0].backgroundColor = colors2.memFill;
      memChart.update('none');
    }
  }

  function initNodeCharts(nodeId) {
    // Create utilization chart
    var utilCanvas = $(nodeId + '-util-chart');
    if (utilCanvas && !gpuCharts[nodeId + '-util']) {
      var colors = getChartColors();
      gpuCharts[nodeId + '-util'] = createGpuChart(
        nodeId + '-util-chart',
        'GPU Util %',
        colors.utilLine,
        colors.utilFill
      );
    }

    // Create memory chart
    var memCanvas = $(nodeId + '-mem-chart');
    if (memCanvas && !gpuCharts[nodeId + '-mem']) {
      var colors2 = getChartColors();
      gpuCharts[nodeId + '-mem'] = createGpuChart(
        nodeId + '-mem-chart',
        'Memory %',
        colors2.memLine,
        colors2.memFill
      );
    }
  }

  // Expose charts for theme updates
  window.dashboardCharts = gpuCharts;

  // ------------------------------------------------------------------
  // 6. GPU DETAIL EXPANDABLE SECTIONS
  // ------------------------------------------------------------------
  function toggleGpuDetail(nodeId) {
    var detail = $(nodeId + '-detail');
    var header = $(nodeId + '-header');
    var btn = $(nodeId + '-expand-btn');
    if (!detail) return;

    var isOpen = detail.classList.contains('open');
    if (isOpen) {
      detail.classList.remove('open');
      if (header) header.classList.remove('expanded');
      if (btn) btn.textContent = 'Expand';
    } else {
      detail.classList.add('open');
      if (header) header.classList.add('expanded');
      if (btn) btn.textContent = 'Collapse';
      // Initialize charts when first expanded
      initNodeCharts(nodeId);
    }
  }

  // ------------------------------------------------------------------
  // 7. MAIN DASHBOARD UPDATE
  // ------------------------------------------------------------------
  var dataReceived = false;

  function updateDashboard(msg) {
    if (msg.type !== 'metrics' || !msg.data) return;

    var d = msg.data;

    // First data received: hide skeletons, show content
    if (!dataReceived) {
      dataReceived = true;
      hideAllSkeletons();
      showContentSections();
    }

    // Clock
    var clockEl = $('clock');
    if (clockEl && msg.timestamp) {
      clockEl.textContent = new Date(msg.timestamp * 1000).toLocaleTimeString();
    }

    // System overview
    setText('model-name', d.model || '--');
    setText('node-count', d.nodes ?? 0);
    setText('uptime', fmtTime(d.uptime));
    setText('ws-connections', (d.ws_connections !== undefined ? d.ws_connections : 'active'));
    var topo = d.topology || {};
    setText('pipeline-mode', topo.pipeline_parallel ? 'Pipeline Parallel' : (topo.nodes && topo.nodes > 1 ? 'Distributed' : 'Single Node'));

    var m = d.metrics_summary || {};

    // Latency
    var lat = m.latency || {};
    setText('lat-p50', lat.p50 ? fmt(lat.p50, 1) + 'ms' : '--');
    setText('lat-p95', lat.p95 ? fmt(lat.p95, 1) + 'ms' : '--');
    setText('lat-p99', lat.p99 ? fmt(lat.p99, 1) + 'ms' : '--');
    setText('lat-avg', lat.avg ? fmt(lat.avg, 1) + 'ms' : '--');
    var ttft = m.ttft || {};
    setText('ttft-avg', ttft.avg ? fmt(ttft.avg, 1) + 'ms' : '--');
    var tp = m.throughput || {};
    setText('tokens-per-sec', tp.tokens_per_sec_avg ? fmt(tp.tokens_per_sec_avg, 1) : '--');

    // Latency histogram
    if (lat.histogram) {
      renderHistogram(lat.histogram);
    }

    // KV Cache
    var kv = m.kv_cache || {};
    setText('kv-hit-rate', fmtPct(kv.hit_rate));
    setText('kv-hits', kv.hits ?? 0);
    setText('kv-misses', kv.misses ?? 0);
    var kvPct = (kv.hit_rate || 0) * 100;
    setStyle('kv-gauge', 'width', kvPct + '%');
    setText('kv-gauge-val', kvPct.toFixed(1) + '%');

    // Speculative
    var spec = m.speculative || {};
    setText('spec-rate', fmtPct(spec.acceptance_rate));
    setText('spec-drafts', spec.drafts ?? 0);
    setText('spec-accepted', spec.accepted ?? 0);
    var specPct = (spec.acceptance_rate || 0) * 100;
    setStyle('spec-gauge', 'width', specPct + '%');
    setText('spec-gauge-val', specPct.toFixed(1) + '%');

    // Cost
    var cost = m.cost || {};
    setText('cost-total', fmtMoney(cost.total));
    setText('cost-avg', fmtMoney(cost.avg_per_request));

    // Scheduler
    var sched = d.scheduler || {};
    setText('sched-active', sched.active ?? sched.active_requests ?? 0);
    setText('sched-pending', sched.pending ?? sched.pending_requests ?? 0);
    setText('sched-completed', sched.completed ?? 0);

    // Throughput
    setText('throughput-rpm', (sched.completed || 0) + ' total');
    var byModel = m.requests_by_model || {};
    setText('requests-by-model', Object.keys(byModel).length ? Object.entries(byModel).map(function (e) { return e[0] + ': ' + e[1]; }).join(', ') : '--');

    // Nodes & GPU
    renderNodes(d);

    // Tenants
    renderTenants(d);

    // Raw data
    var rawEl = $('raw-metrics');
    if (rawEl) rawEl.textContent = JSON.stringify(d, null, 2);
  }

  function setText(id, val) {
    var el = $(id);
    if (el) el.textContent = val;
  }

  function setStyle(id, prop, val) {
    var el = $(id);
    if (el) el.style[prop] = val;
  }

  // ------------------------------------------------------------------
  // 7a. Histogram
  // ------------------------------------------------------------------
  function renderHistogram(hist) {
    var labels = Object.keys(hist);
    var values = Object.values(hist);
    var maxVal = Math.max.apply(null, values.concat([1]));
    var container = $('latency-histogram');
    var labelContainer = $('histogram-labels');
    if (!container) return;

    container.innerHTML = labels.map(function (l, i) {
      var pct = (values[i] / maxVal) * 100;
      return '<div class="histogram-bar" style="height:' + Math.max(pct, 3) + '%" title="' + l + 'ms: ' + values[i] + '"></div>';
    }).join('');

    if (labelContainer) {
      var step = Math.max(1, Math.floor(labels.length / 5));
      labelContainer.innerHTML = labels.filter(function (_, i) { return i % step === 0; }).map(function (l) { return '<span>' + l + 'ms</span>'; }).join('');
    }
  }

  // ------------------------------------------------------------------
  // 7b. Nodes & GPU
  // ------------------------------------------------------------------
  var nodesData = {};

  function renderNodes(d) {
    var gpuUtil = d.gpu_utilization || {};
    var nodes = d.nodes || {};
    var nodeContainer = $('node-list');
    var topoContainer = $('topology-container');
    if (!nodeContainer) return;

    nodesData = nodes;

    if (Object.keys(nodes).length === 0) {
      nodeContainer.innerHTML = '<p class="placeholder">No nodes connected</p>';
      if (topoContainer) topoContainer.innerHTML = '<p class="placeholder">No topology data</p>';
      return;
    }

    // Topology
    if (topoContainer) {
      var nodeIds = Object.keys(nodes);
      topoContainer.innerHTML = nodeIds.map(function (id, i) {
        var info = nodes[id];
        var healthy = info.healthy ? 'healthy' : 'unhealthy';
        var arrow = i < nodeIds.length - 1 ? '<span class="topology-arrow"> → </span>' : '';
        return '<span class="topology-node"><span class="dot ' + healthy + '"></span>' + esc(id) + '</span>' + arrow;
      }).join('');
    }

    // Node list with expandable GPU detail
    var html = '';
    Object.keys(nodes).forEach(function (id) {
      var info = nodes[id];
      var status = info.healthy ? 'healthy' : 'unhealthy';
      var gpu = gpuUtil[id] || 0;
      var gpuClass = gpu > 80 ? 'red' : gpu > 50 ? 'yellow' : 'green';

      // Update GPU charts
      updateGpuCharts(id, gpu, info.gpu_memory_pct);

      // Build detail content
      var detailContent = buildGpuDetail(id, info);

      html += '<div class="node-item">'
        + '<div class="node-header" id="' + id + '-header" onclick="toggleGpuDetail(\'' + id + '\')">'
        + '  <span class="node-id"><span class="expand-icon">▶</span>' + esc(id) + '</span>'
        + '  <div style="display:flex;align-items:center;gap:8px;">'
        + '    <span class="node-status ' + status + '">' + status + '</span>'
        + '    <button class="node-expand-btn" id="' + id + '-expand-btn" onclick="event.stopPropagation();toggleGpuDetail(\'' + id + '\')">Expand</button>'
        + '  </div>'
        + '</div>'
        + '<div class="gauge">'
        + '  <span class="gauge-label">GPU: ' + esc(info.host || '') + ':' + (info.port || '') + '</span>'
        + '  <div class="gauge-bar"><div class="gauge-fill ' + gpuClass + '" style="width:' + gpu + '%"></div></div>'
        + '  <span class="gauge-value">' + fmt(gpu, 1) + '%</span>'
        + '</div>'
        + '<div style="font-size:0.7rem;color:var(--text-muted);">Layers: ' + (info.layers || '--') + ' | Role: ' + (info.role || 'auto') + '</div>'
        + detailContent
        + '</div>';
    });

    nodeContainer.innerHTML = html;
  }

  function buildGpuDetail(nodeId, info) {
    var kv = info.kv_cache || {};
    var gpuName = info.gpu_name || 'N/A';
    var vramTotal = info.gpu_memory_total ? (info.gpu_memory_total / (1024 * 1024 * 1024)).toFixed(1) + ' GB' : 'N/A';
    var vramFree = info.gpu_memory_free ? (info.gpu_memory_free / (1024 * 1024 * 1024)).toFixed(1) + ' GB' : 'N/A';
    var gpuTemp = info.gpu_temp !== undefined ? info.gpu_temp + '°C' : 'N/A';
    var gpuPower = info.gpu_power !== undefined ? info.gpu_power + ' W' : 'N/A';
    var gpuSmCount = info.gpu_sm_count || 'N/A';

    var kvHitRate = kv.hit_rate !== undefined ? fmtPct(kv.hit_rate) : 'N/A';
    var kvHits = kv.hits !== undefined ? kv.hits : 'N/A';

    return '<div class="node-detail" id="' + nodeId + '-detail">'
      + '  <div class="node-detail-grid">'
      + '    <div class="node-detail-item"><span class="node-detail-label">GPU</span><span class="node-detail-value">' + esc(gpuName) + '</span></div>'
      + '    <div class="node-detail-item"><span class="node-detail-label">Temperature</span><span class="node-detail-value">' + gpuTemp + '</span></div>'
      + '    <div class="node-detail-item"><span class="node-detail-label">VRAM</span><span class="node-detail-value">' + vramFree + ' / ' + vramTotal + '</span></div>'
      + '    <div class="node-detail-item"><span class="node-detail-label">Power</span><span class="node-detail-value">' + gpuPower + '</span></div>'
      + '    <div class="node-detail-item"><span class="node-detail-label">SM Count</span><span class="node-detail-value">' + gpuSmCount + '</span></div>'
      + '    <div class="node-detail-item"><span class="node-detail-label">KV Cache Hits</span><span class="node-detail-value">' + kvHits + '</span></div>'
      + '    <div class="node-detail-item"><span class="node-detail-label">KV Hit Rate</span><span class="node-detail-value">' + kvHitRate + '</span></div>'
      + '    <div class="node-detail-item"><span class="node-detail-label">Role</span><span class="node-detail-value">' + esc(info.role || 'auto') + '</span></div>'
      + '  </div>'
      + '  <div class="gpu-charts-row">'
      + '    <div class="gpu-chart-box">'
      + '      <h3>GPU Utilization (60s)</h3>'
      + '      <div class="chart-container"><canvas id="' + nodeId + '-util-chart"></canvas></div>'
      + '    </div>'
      + '    <div class="gpu-chart-box">'
      + '      <h3>Memory Usage (60s)</h3>'
      + '      <div class="chart-container"><canvas id="' + nodeId + '-mem-chart"></canvas></div>'
      + '    </div>'
      + '  </div>'
      + '</div>';
  }

  // Expose toggle for inline onclick
  window.toggleGpuDetail = toggleGpuDetail;

  // ------------------------------------------------------------------
  // 7c. Tenants
  // ------------------------------------------------------------------
  function renderTenants(d) {
    var tenants = d.tenants || [];
    var tenantList = $('tenant-list');
    if (!tenantList) return;

    if (tenants.length) {
      tenantList.innerHTML = tenants.map(function (t) {
        return '<div class="tenant-item"><span>' + esc(t.name || t.tenant_id || '') + '</span><span style="color:var(--text-secondary);">' + esc(t.tier || '') + ' · ' + esc(t.tenant_id || '') + '</span></div>';
      }).join('');
    }
  }

  // ------------------------------------------------------------------
  // 8. REST API POLLING (Waterfall, Streaming Cost, Cost Summary)
  // ------------------------------------------------------------------
  function fetchWaterfall() {
    fetch('/api/requests/waterfall?limit=30')
      .then(function (r) { return r.json(); })
      .then(function (data) {
        if (!data || !data.length) return;
        var container = $('waterfall-container');
        if (!container) return;
        var maxElapsed = Math.max.apply(null, data.map(function (i) { return i.elapsed_ms || 0; }).concat([1]));
        var rows = data.slice(0, 20).map(function (item) {
          var ttft = item.ttft_ms || 0;
          var elapsed = item.elapsed_ms || 0;
          var ttftPct = Math.min((ttft / maxElapsed) * 100, 100);
          var decodePct = Math.max(Math.min((elapsed / maxElapsed) * 100, 100) - ttftPct, 0);
          return '<div class="waterfall-row">'
            + '<span class="waterfall-label" title="' + esc(item.request_id) + '">' + esc((item.request_id || '???').substring(0, 10)) + '</span>'
            + '<div class="waterfall-bar">'
            + '  <div class="waterfall-prefill" style="width:' + ttftPct + '%" title="Prefill: ' + ttft.toFixed(0) + 'ms"></div>'
            + '  <div class="waterfall-decode" style="width:' + decodePct + '%" title="Decode: ' + (elapsed - ttft).toFixed(0) + 'ms"></div>'
            + '</div>'
            + '<span class="waterfall-time">' + elapsed.toFixed(0) + 'ms</span>'
            + '</div>';
        }).join('');
        container.innerHTML = '<div class="waterfall-header"><span>0ms</span><span>' + maxElapsed.toFixed(0) + 'ms</span></div>' + rows;
      })
      .catch(function () {});
  }

  function fetchStreamingCost() {
    fetch('/api/streaming-cost/stats')
      .then(function (r) { return r.json(); })
      .then(function (data) {
        if (!data || data.status === 'not_available') return;
        setText('stream-active', data.active_streams || 0);
        setText('stream-cost-per-sec', fmtMoney(data.active_cost_per_second));
        var totalTracked = (data.total_cost_tracked || 0) + (data.total_savings_tracked || 0);
        setText('stream-cloud-cost', fmtMoney(totalTracked));
        setText('stream-savings', fmtMoney(data.total_savings_tracked));
        setText('stream-tokens-tracked', (data.total_tokens_tracked || 0).toLocaleString());
        var totalCost = (data.total_cost_tracked || 0) + (data.total_savings_tracked || 0);
        var savingsPct = totalCost > 0 ? ((data.total_savings_tracked || 0) / totalCost * 100) : 0;
        setStyle('savings-gauge', 'width', Math.min(savingsPct, 100) + '%');
        setText('savings-gauge-val', savingsPct.toFixed(1) + '%');
      })
      .catch(function () {});
  }

  function fetchCostSummary() {
    fetch('/api/cost/summary')
      .then(function (r) { return r.json(); })
      .then(function (data) {
        if (!data || data.status === 'not_available') return;
        setText('cost-total', fmtMoney(data.total_cost_usd));
        setText('cost-avg', fmtMoney(data.avg_cost_per_request));
      })
      .catch(function () {});
  }

  function fetchClusterNodes() {
    fetch('/api/cluster/nodes')
      .then(function (r) { return r.json(); })
      .then(function (data) {
        if (!data || !data.nodes || !data.nodes.length) return;
        // Enrich node info with cluster data if node-list already populated
        // This is supplementary to WebSocket data
      })
      .catch(function () {});
  }

  // ------------------------------------------------------------------
  // 9. INITIALIZATION
  // ------------------------------------------------------------------
  function init() {
    // Connect WebSocket
    connectWebSocket();
    startConnectionMonitor();

    // Initial REST fetches
    fetchWaterfall();
    fetchStreamingCost();
    fetchCostSummary();
    fetchClusterNodes();

    // Periodic REST polling
    setInterval(fetchWaterfall, 3000);
    setInterval(fetchStreamingCost, 2000);
    setInterval(fetchCostSummary, 10000);
    setInterval(fetchClusterNodes, 5000);

    // Clock
    setInterval(function () {
      var clockEl = $('clock');
      if (clockEl) clockEl.textContent = new Date().toLocaleTimeString();
    }, 1000);
  }

  // Wait for DOM and Chart.js to be ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();
