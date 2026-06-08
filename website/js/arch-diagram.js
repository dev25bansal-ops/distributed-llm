/**
 * Interactive Architecture Diagram — canvas-based animated pipeline visualization.
 *
 * Renders:
 * - Coordinator, Workers, API layer, Redis, Client apps, Model Hub
 * - Animated data flow particles between components
 * - Click-to-inspect info panel
 * - Zoom controls
 * - Auto-pause when off-screen
 *
 * Usage:
 *   <div id="archDiagram"></div>
 *   <script type="module">
 *     import { initArchDiagram } from './js/arch-diagram.js';
 *     initArchDiagram();
 *   </script>
 */

// ── Component Definitions ────────────────────────────────────────
const COMPONENTS = [
  { id: 'clients', label: 'Client Apps', x: 0.1, y: 0.08, w: 0.18, h: 0.16, color: '#3b82f6', desc: 'Browser, SDK, CLI, LangChain, LlamaIndex, and any OpenAI-compatible client connects via REST API.' },
  { id: 'api', label: 'REST API', x: 0.35, y: 0.08, w: 0.18, h: 0.16, color: '#8b5cf6', desc: 'FastAPI server on port 8000. OpenAI-compatible endpoints: /v1/chat/completions, /v1/models, /v1/completions.' },
  { id: 'grpc', label: 'gRPC', x: 0.6, y: 0.08, w: 0.18, h: 0.16, color: '#06b6d4', desc: 'Internal gRPC protocol on port 50050. Handles ForwardPass, HealthCheck, and WeightTransfer between nodes.' },
  { id: 'coordinator', label: 'Coordinator', x: 0.35, y: 0.38, w: 0.3, h: 0.2, color: '#00e676', desc: 'Central orchestrator: pipeline scheduler, batch scheduler, KV cache manager, health monitor, straggler detector, node recovery.' },
  { id: 'redis', label: 'Redis Cache', x: 0.04, y: 0.38, w: 0.18, h: 0.16, color: '#f59e0b', desc: 'Prompt cache, rate limiter token buckets, cluster state store, session management.' },
  { id: 'modelHub', label: 'Model Hub', x: 0.78, y: 0.38, w: 0.18, h: 0.16, color: '#ef4444', desc: 'HuggingFace model registry. Downloads, caches, and distributes model weights to workers.' },
];

function generateWorkers(count) {
  const workers = [];
  const startX = 0.1;
  const spacing = 0.23;
  const maxDisplay = 4;
  for (let i = 0; i < Math.min(count, maxDisplay); i++) {
    const x = startX + i * spacing;
    workers.push({
      id: `worker-${i}`,
      label: `Worker ${i + 1}`,
      sublabel: `Layers ${i * 20}-${(i + 1) * 20 - 1}`,
      x, y: 0.7, w: 0.18, h: 0.18,
      color: '#22c55e',
      util: 0.6 + Math.random() * 0.35,
      desc: `GPU Node ${i + 1}: hosts model layers. Receives activations, computes forward pass, passes to next node.`,
    });
  }
  return workers;
}

// ── Canvas Rendering ─────────────────────────────────────────────
export function initArchDiagram() {
  const container = document.getElementById('archDiagram');
  if (!container) return;

  const workers = generateWorkers(4);
  const allNodes = [...COMPONENTS, ...workers];
  let selectedNode = null;
  let particles = [];
  let animFrame = null;
  let isActive = true;
  let scale = 1;
  let offsetX = 0, offsetY = 0;

  container.innerHTML = `
    <div class="arch-card">
      <div class="arch-header">
        <h3>Architecture</h3>
        <div class="arch-controls">
          <button class="arch-btn" id="archZoomIn" title="Zoom in">+</button>
          <button class="arch-btn" id="archZoomOut" title="Zoom out">−</button>
          <button class="arch-btn" id="archZoomReset" title="Reset">⟲</button>
        </div>
      </div>
      <div class="arch-canvas-wrap">
        <canvas id="archCanvas" width="800" height="520"></canvas>
        <div class="arch-legend" id="archLegend">
          <span class="arch-legend-item"><span class="arch-dot" style="background:#3b82f6"></span>Client</span>
          <span class="arch-legend-item"><span class="arch-dot" style="background:#8b5cf6"></span>API</span>
          <span class="arch-legend-item"><span class="arch-dot" style="background:#06b6d4"></span>gRPC</span>
          <span class="arch-legend-item"><span class="arch-dot" style="background:#00e676"></span>Coordinator</span>
          <span class="arch-legend-item"><span class="arch-dot" style="background:#22c55e"></span>Worker</span>
          <span class="arch-legend-item"><span class="arch-dot" style="background:#f59e0b"></span>Cache</span>
          <span class="arch-legend-item"><span class="arch-dot" style="background:#ef4444"></span>Model Hub</span>
        </div>
      </div>
      <div class="arch-info" id="archInfo">
        <span class="arch-info-hint">Click any component to learn more</span>
      </div>
    </div>
  `;

  const canvas = document.getElementById('archCanvas');
  const ctx = canvas.getContext('2d');
  const info = document.getElementById('archInfo');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);

  const W = () => rect.width;
  const H = () => rect.height;

  function toCanvasX(relX) { return (relX * W()) + offsetX; }
  function toCanvasY(relY) { return (relY * H()) + offsetY; }
  function toRelX(absX) { return (absX - offsetX) / W(); }
  function toRelY(absY) { return (absY - offsetY) / H(); }

  function drawNode(node) {
    const x = toCanvasX(node.x) * scale;
    const y = toCanvasY(node.y) * scale;
    const w = node.w * W() * scale;
    const h = node.h * H() * scale;
    const isSelected = selectedNode === node;

    // Shadow
    ctx.shadowColor = 'rgba(0,0,0,0.3)';
    ctx.shadowBlur = isSelected ? 20 : 8;
    ctx.shadowOffsetY = isSelected ? 4 : 2;

    // Background
    ctx.fillStyle = node.color || '#333';
    ctx.globalAlpha = isSelected ? 1 : 0.85;
    const radius = 8;
    ctx.beginPath();
    ctx.roundRect(x, y, w, h, radius);
    ctx.fill();
    ctx.globalAlpha = 1;
    ctx.shadowBlur = 0;

    // Border
    ctx.strokeStyle = isSelected ? '#fff' : 'rgba(255,255,255,0.15)';
    ctx.lineWidth = isSelected ? 2 : 1;
    ctx.beginPath();
    ctx.roundRect(x, y, w, h, radius);
    ctx.stroke();

    // GPU utilization bar (for workers)
    if (node.util !== undefined) {
      const barY = y + h - 12;
      const barH = 6;
      const barW = w - 16;
      ctx.fillStyle = 'rgba(0,0,0,0.3)';
      ctx.beginPath();
      ctx.roundRect(x + 8, barY, barW, barH, 3);
      ctx.fill();
      ctx.fillStyle = node.util > 0.8 ? '#ef4444' : node.util > 0.5 ? '#f59e0b' : '#22c55e';
      ctx.beginPath();
      ctx.roundRect(x + 8, barY, barW * node.util, barH, 3);
      ctx.fill();
    }

    // Label
    ctx.fillStyle = '#fff';
    ctx.font = `bold ${13 * scale}px Inter, sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(node.label, x + w / 2, y + h / 2 - (node.sublabel ? 8 : 0));

    if (node.sublabel) {
      ctx.fillStyle = 'rgba(255,255,255,0.7)';
      ctx.font = `${10 * scale}px Inter, sans-serif`;
      ctx.fillText(node.sublabel, x + w / 2, y + h / 2 + 12);
    }
  }

  // Particle system
  function spawnParticle(from, to) {
    return {
      x: toCanvasX(from.x) + from.w * W() / 2,
      y: toCanvasY(from.y) + from.h * H() / 2,
      targetX: toCanvasX(to.x) + to.w * W() / 2,
      targetY: toCanvasY(to.y) + to.h * H() / 2,
      progress: 0,
      speed: 0.008 + Math.random() * 0.012,
      size: 2 + Math.random() * 2,
    };
  }

  function drawParticles() {
    // Spawn new particles
    if (Math.random() < 0.3) {
      const fromIdx = Math.floor(Math.random() * COMPONENTS.length);
      const toIdx = Math.floor(Math.random() * workers.length);
      particles.push(spawnParticle(COMPONENTS[fromIdx], workers[toIdx]));
    }

    particles = particles.filter(p => p.progress < 1);
    for (const p of particles) {
      p.progress += p.speed;
      const x = p.x + (p.targetX - p.x) * p.progress;
      const y = p.y + (p.targetY - p.y) * p.progress;
      const alpha = Math.sin(p.progress * Math.PI);

      ctx.fillStyle = `rgba(0, 230, 118, ${alpha * 0.8})`;
      ctx.beginPath();
      ctx.arc(x, y, p.size, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = `rgba(0, 230, 118, ${alpha * 0.2})`;
      ctx.beginPath();
      ctx.arc(x, y, p.size * 3, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  function animate() {
    if (!isActive) return;
    ctx.clearRect(0, 0, W(), H());

    // Background grid
    ctx.strokeStyle = 'rgba(255,255,255,0.03)';
    ctx.lineWidth = 1;
    for (let x = 0; x < W(); x += 40) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, H());
      ctx.stroke();
    }
    for (let y = 0; y < H(); y += 40) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(W(), y);
      ctx.stroke();
    }

    drawParticles();

    for (const node of allNodes) {
      drawNode(node);
    }

    animFrame = requestAnimationFrame(animate);
  }

  // Click handler
  function handleClick(e) {
    const rect = canvas.getBoundingClientRect();
    const mx = (e.clientX - rect.left - offsetX) / scale;
    const my = (e.clientY - rect.top - offsetY) / scale;

    let clicked = null;
    for (const node of allNodes) {
      const x = node.x * W();
      const y = node.y * H();
      const w = node.w * W();
      const h = node.h * H();
      if (mx >= x && mx <= x + w && my >= y && my <= y + h) {
        clicked = node;
        break;
      }
    }

    selectedNode = clicked;
    if (clicked) {
      info.innerHTML = `
        <div class="arch-info-content">
          <strong style="color:${clicked.color}">${escapeHtml(clicked.label)}</strong>
          <p>${escapeHtml(clicked.desc)}</p>
        </div>`;
    } else {
      info.innerHTML = '<span class="arch-info-hint">Click any component to learn more</span>';
    }
  }

  // Zoom controls
  document.getElementById('archZoomIn').addEventListener('click', () => { scale = Math.min(scale + 0.2, 2); });
  document.getElementById('archZoomOut').addEventListener('click', () => { scale = Math.max(scale - 0.2, 0.5); });
  document.getElementById('archZoomReset').addEventListener('click', () => { scale = 1; offsetX = 0; offsetY = 0; });

  canvas.addEventListener('click', handleClick);

  // Off-screen pause
  const obs = new IntersectionObserver((entries) => {
    isActive = entries[0].isIntersecting;
    if (isActive && !animFrame) animFrame = requestAnimationFrame(animate);
    else if (!isActive && animFrame) { cancelAnimationFrame(animFrame); animFrame = null; }
  }, { rootMargin: '200px' });
  obs.observe(canvas);

  // Resize
  const resizeObs = new ResizeObserver(() => {
    const r = canvas.getBoundingClientRect();
    canvas.width = r.width * dpr;
    canvas.height = r.height * dpr;
    ctx.scale(dpr, dpr);
  });
  resizeObs.observe(canvas);

  // Start
  isActive = true;
  animFrame = requestAnimationFrame(animate);

  // Helper
  function escapeHtml(str) {
    if (!str) return '';
    const d = document.createElement('div');
    d.textContent = str;
    return d.innerHTML;
  }
}

// polyfill roundRect if needed
if (!CanvasRenderingContext2D.prototype.roundRect) {
  CanvasRenderingContext2D.prototype.roundRect = function(x, y, w, h, r) {
    if (r > w / 2) r = w / 2;
    if (r > h / 2) r = h / 2;
    this.moveTo(x + r, y);
    this.arcTo(x + w, y, x + w, y + h, r);
    this.arcTo(x + w, y + h, x, y + h, r);
    this.arcTo(x, y + h, x, y, r);
    this.arcTo(x, y, x + w, y, r);
    return this;
  };
}
