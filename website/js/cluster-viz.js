/**
 * Live Cluster Visualization — animated GPU nodes with data flow.
 *
 * Renders an interactive cluster diagram showing:
 * - GPUs as nodes with real-time utilization bars
 * - Animated data flow between nodes (hidden states)
 * - Click-to-inspect node details (specs, layers, throughput)
 * - Pipeline parallelism visualization
 *
 * Usage:
 *   <div id="clusterViz"></div>
 *   <script type="module">
 *     import { initClusterViz } from './js/cluster-viz.js';
 *     initClusterViz();
 *   </script>
 */

// ── Configuration ──────────────────────────────────────────────────────

const COLORS = {
    bg: '#0a0a0a',
    node: '#18181b',
    nodeStroke: '#333',
    nodeActive: '#22c55e',
    nodeIdle: '#555',
    flow: '#22c55e',
    flowDim: 'rgba(34,197,94,0.3)',
    text: '#ededed',
    textDim: '#888',
    barBg: '#222',
    barFill: '#22c55e',
    barHigh: '#ef4444',
    barMed: '#eab308',
    coordinator: '#06b6d4',
};

// ── Data ───────────────────────────────────────────────────────────────

function generateCluster() {
    const gpus = [
        { id: 'gpu-0', name: 'RTX 4090', vram: 24, layers: [0, 7], util: 0.72, mem: 0.65, temp: 68, throughput: 42.3 },
        { id: 'gpu-1', name: 'RTX 4090', vram: 24, layers: [8, 15], util: 0.85, mem: 0.78, temp: 74, throughput: 38.1 },
        { id: 'gpu-2', name: 'RTX 3090', vram: 24, layers: [16, 23], util: 0.61, mem: 0.52, temp: 62, throughput: 45.7 },
        { id: 'gpu-3', name: 'RTX 3090', vram: 24, layers: [24, 31], util: 0.93, mem: 0.88, temp: 81, throughput: 31.2 },
    ];
    return gpus;
}

// ── Visualization ──────────────────────────────────────────────────────

export function initClusterViz() {
    const container = document.getElementById('clusterViz');
    if (!container) return;

    container.innerHTML = `
        <div class="cluster-viz-card">
            <div class="cluster-viz-header">
                <h3>Live Cluster</h3>
                <span class="cluster-viz-status" id="clusterStatus">● Running</span>
            </div>
            <canvas id="clusterCanvas" width="800" height="400"></canvas>
            <div class="cluster-viz-details" id="clusterDetails">
                <span class="cluster-viz-hint">Click a GPU node to inspect</span>
            </div>
        </div>
    `;

    const canvas = document.getElementById('clusterCanvas');
    const ctx = canvas.getContext('2d');
    const details = document.getElementById('clusterDetails');
    const gpus = generateCluster();

    // Scale canvas for retina
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    const W = rect.width;
    const H = rect.height;

    // Node positions
    const nodeW = 140, nodeH = 160, gap = 40;
    const startX = (W - (gpus.length * (nodeW + gap) - gap)) / 2;
    const nodeY = H / 2 - nodeH / 2;
    const nodes = gpus.map((g, i) => ({
        ...g,
        x: startX + i * (nodeW + gap),
        y: nodeY,
        w: nodeW,
        h: nodeH,
        flowParticles: [],
        selected: false,
    }));

    // Click handler
    function handleClick(e) {
        const rect = canvas.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;

        let clicked = null;
        for (const n of nodes) {
            if (mx >= n.x && mx <= n.x + n.w && my >= n.y && my <= n.y + n.h) {
                clicked = n;
                break;
            }
        }

        nodes.forEach(n => n.selected = false);
        if (clicked) {
            clicked.selected = true;
            details.innerHTML = `
                <div class="cluster-node-info">
                    <div class="info-item"><div class="info-label">GPU</div><div class="info-value">${clicked.name}</div></div>
                    <div class="info-item"><div class="info-label">VRAM</div><div class="info-value">${clicked.vram} GB</div></div>
                    <div class="info-item"><div class="info-label">Layers</div><div class="info-value">${clicked.layers[0]}-${clicked.layers[1]}</div></div>
                    <div class="info-item"><div class="info-label">Utilization</div><div class="info-value">${(clicked.util * 100).toFixed(0)}%</div></div>
                    <div class="info-item"><div class="info-label">Memory</div><div class="info-value">${(clicked.mem * clicked.vram).toFixed(1)} / ${clicked.vram} GB</div></div>
                    <div class="info-item"><div class="info-label">Temperature</div><div class="info-value">${clicked.temp}°C</div></div>
                    <div class="info-item"><div class="info-label">Throughput</div><div class="info-value">${clicked.throughput.toFixed(1)} tok/s</div></div>
                    <div class="info-item"><div class="info-label">Status</div><div class="info-value" style="color:#22c55e">Active</div></div>
                </div>
            `;
        } else {
            details.innerHTML = '<span class="cluster-viz-hint">Click a GPU node to inspect</span>';
        }
    }
    canvas.addEventListener('click', handleClick);

    // Animation state
    let animTime = 0;
    const flowSpeed = 2;

    function spawnParticle(fromNode, toNode) {
        return {
            x: fromNode.x + fromNode.w,
            y: fromNode.y + fromNode.h / 2,
            tx: toNode.x,
            ty: toNode.y + toNode.h / 2,
            progress: 0,
            speed: 0.015 + Math.random() * 0.01,
            size: 3 + Math.random() * 2,
        };
    }

    function drawNode(n) {
        const { x, y, w, h, name, util, mem, vram, selected } = n;

        // Node background
        ctx.fillStyle = COLORS.node;
        ctx.strokeStyle = selected ? COLORS.nodeActive : COLORS.nodeStroke;
        ctx.lineWidth = selected ? 2 : 1;
        ctx.beginPath();
        ctx.roundRect(x, y, w, h, 8);
        ctx.fill();
        ctx.stroke();

        // Selection glow
        if (selected) {
            ctx.strokeStyle = 'rgba(34,197,94,0.3)';
            ctx.lineWidth = 4;
            ctx.beginPath();
            ctx.roundRect(x - 2, y - 2, w + 4, h + 4, 10);
            ctx.stroke();
        }

        // GPU icon
        ctx.fillStyle = COLORS.text;
        ctx.font = 'bold 13px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('GPU', x + w / 2, y + 20);

        // Name
        ctx.fillStyle = COLORS.textDim;
        ctx.font = '11px Inter, sans-serif';
        ctx.fillText(name, x + w / 2, y + 36);

        // Utilization bar
        const barX = x + 12, barY = y + 50, barW = w - 24, barH = 8;
        ctx.fillStyle = COLORS.barBg;
        ctx.beginPath();
        ctx.roundRect(barX, barY, barW, barH, 4);
        ctx.fill();

        const fillW = barW * util;
        const barColor = util > 0.9 ? COLORS.barHigh : util > 0.7 ? COLORS.barMed : COLORS.barFill;
        ctx.fillStyle = barColor;
        ctx.beginPath();
        ctx.roundRect(barX, barY, fillW, barH, 4);
        ctx.fill();

        ctx.fillStyle = COLORS.text;
        ctx.font = '10px Inter, sans-serif';
        ctx.textAlign = 'right';
        ctx.fillText(`${(util * 100).toFixed(0)}%`, x + w - 12, barY + barH + 12);

        // Memory bar
        const memY = barY + 28;
        ctx.fillStyle = COLORS.barBg;
        ctx.beginPath();
        ctx.roundRect(barX, memY, barW, barH, 4);
        ctx.fill();

        const memW = barW * mem;
        ctx.fillStyle = mem > 0.9 ? COLORS.barHigh : COLORS.barFill;
        ctx.beginPath();
        ctx.roundRect(barX, memY, memW, barH, 4);
        ctx.fill();

        ctx.fillStyle = COLORS.textDim;
        ctx.font = '10px Inter, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText(`${(mem * vram).toFixed(0)}GB`, barX, memY + barH + 12);
        ctx.textAlign = 'right';
        ctx.fillText(`${vram}GB`, x + w - 12, memY + barH + 12);

        // Layers badge
        ctx.fillStyle = 'rgba(34,197,94,0.15)';
        ctx.beginPath();
        ctx.roundRect(x + w / 2 - 25, y + h - 32, 50, 20, 4);
        ctx.fill();
        ctx.fillStyle = COLORS.nodeActive;
        ctx.font = '10px JetBrains Mono, monospace';
        ctx.textAlign = 'center';
        ctx.fillText(`L${n.layers[0]}-${n.layers[1]}`, x + w / 2, y + h - 18);
    }

    function drawCoordinator() {
        const cx = W / 2, cy = 30;
        ctx.fillStyle = COLORS.coordinator;
        ctx.font = 'bold 12px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('Coordinator', cx, cy);
        ctx.fillStyle = COLORS.textDim;
        ctx.font = '10px Inter, sans-serif';
        ctx.fillText('API :8000  gRPC :50050', cx, cy + 14);

        // Lines from coordinator to each node
        for (const n of nodes) {
            ctx.strokeStyle = 'rgba(6,182,212,0.15)';
            ctx.lineWidth = 1;
            ctx.setLineDash([4, 4]);
            ctx.beginPath();
            ctx.moveTo(cx, cy + 20);
            ctx.lineTo(n.x + n.w / 2, n.y);
            ctx.stroke();
            ctx.setLineDash([]);
        }
    }

    function drawFlowParticles() {
        for (const n of nodes) {
            for (const p of n.flowParticles) {
                p.progress += p.speed;
                const t = p.progress;
                const x = p.x + (p.tx - p.x) * t;
                const y = p.y + (p.ty - p.y) * t + Math.sin(t * Math.PI * 2) * 5;
                const alpha = t < 0.1 ? t * 10 : t > 0.9 ? (1 - t) * 10 : 1;

                ctx.fillStyle = `rgba(34,197,94,${alpha * 0.8})`;
                ctx.beginPath();
                ctx.arc(x, y, p.size, 0, Math.PI * 2);
                ctx.fill();

                // Glow
                ctx.fillStyle = `rgba(34,197,94,${alpha * 0.2})`;
                ctx.beginPath();
                ctx.arc(x, y, p.size * 3, 0, Math.PI * 2);
                ctx.fill();
            }
            n.flowParticles = n.flowParticles.filter(p => p.progress < 1);
        }
    }

    let animFrameId = null;
    let animTime = 0;
    let isActive = true;

    function animate() {
        if (!isActive) return;
        ctx.clearRect(0, 0, W, H);

        drawCoordinator();
        drawFlowParticles();
        nodes.forEach(drawNode);

        // Spawn particles between adjacent nodes
        animTime++;
        if (animTime % 20 === 0) {
            for (let i = 0; i < nodes.length - 1; i++) {
                if (nodes[i].util > 0.3) {
                    nodes[i].flowParticles.push(spawnParticle(nodes[i], nodes[i + 1]));
                }
            }
        }

        // Slowly vary utilization for realism
        if (animTime % 60 === 0) {
            for (const n of nodes) {
                n.util = Math.max(0.1, Math.min(0.99, n.util + (Math.random() - 0.5) * 0.1));
                n.mem = Math.max(0.1, Math.min(0.95, n.mem + (Math.random() - 0.5) * 0.05));
                n.temp = Math.max(40, Math.min(90, n.temp + (Math.random() - 0.5) * 3));
                n.throughput = Math.max(10, Math.min(60, n.throughput + (Math.random() - 0.5) * 5));
            }
        }

        animFrameId = requestAnimationFrame(animate);
    }

    // Pause animation when off-screen to save GPU/CPU
    const visibilityObserver = new IntersectionObserver((entries) => {
        for (const entry of entries) {
            if (entry.isIntersecting) {
                isActive = true;
                animFrameId = requestAnimationFrame(animate);
            } else {
                isActive = false;
                if (animFrameId) {
                    cancelAnimationFrame(animFrameId);
                    animFrameId = null;
                }
            }
        }
    }, { rootMargin: '200px' });
    visibilityObserver.observe(canvas);

    // Handle canvas resize
    const resizeObserver = new ResizeObserver(() => {
        const rect = canvas.getBoundingClientRect();
        canvas.width = rect.width * dpr;
        canvas.height = rect.height * dpr;
        ctx.scale(dpr, dpr);
    });
    resizeObserver.observe(canvas);

    // Start animation
    isActive = true;
    animFrameId = requestAnimationFrame(animate);

    // Store cleanup on container for potential teardown
    container._clusterVizCleanup = () => {
        isActive = false;
        if (animFrameId) {
            cancelAnimationFrame(animFrameId);
            animFrameId = null;
        }
        visibilityObserver.disconnect();
        resizeObserver.disconnect();
        canvas.removeEventListener('click', handleClick);
    };
}
