/**
 * Screenshot Gallery — for Grafana dashboards, CLI output, architecture diagrams.
 *
 * Shows placeholder cards with icons. Replace icons with <img> when real screenshots are available.
 *
 * Usage:
 *   <div class="screenshot-gallery" id="gallery"></div>
 *   <script type="module">
 *     import { initScreenshotGallery } from './js/screenshot-gallery.js';
 *     initScreenshotGallery();
 *   </script>
 */

const GALLERY_ITEMS = [
    { title: 'Grafana Dashboard', desc: 'Real-time GPU utilization and throughput monitoring', icon: '📊' },
    { title: 'CLI Output', desc: 'DistLLM cluster status and node information', icon: '💻' },
    { title: 'Architecture', desc: 'Pipeline parallelism across multiple nodes', icon: '🏗️' },
    { title: 'Web UI', desc: 'Interactive chat playground and model browser', icon: '🌐' },
    { title: 'Kubernetes', desc: 'Pod status and resource utilization', icon: '☸️' },
    { title: 'Cost Dashboard', desc: 'Per-model cost tracking and savings report', icon: '💰' },
];

export function initScreenshotGallery() {
    const container = document.getElementById('gallery');
    if (!container) return;

    container.innerHTML = `
        <div class="ss-gallery">
            <h3>Screenshots</h3>
            <p class="ss-desc">Visual overview of DistLLM dashboards, CLI output, and architecture.</p>
            <div class="ss-grid" id="ssGrid"></div>
        </div>
    `;

    const grid = document.getElementById('ssGrid');
    grid.innerHTML = GALLERY_ITEMS.map(item => `
        <div class="ss-item" role="img" aria-label="${item.title} screenshot placeholder">
            <div class="ss-icon">${item.icon}</div>
            <div class="ss-title">${item.title}</div>
            <div class="ss-desc">${item.desc}</div>
            <div class="ss-placeholder-badge">Preview</div>
        </div>
    `).join('');
}
