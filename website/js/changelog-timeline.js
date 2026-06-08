/**
 * Changelog Timeline — visual timeline of releases.
 *
 * Usage:
 *   <div id="changelogTimeline"></div>
 *   <script type="module">
 *     import { initChangelogTimeline } from './js/changelog-timeline.js';
 *     initChangelogTimeline();
 *   </script>
 */

const TIMELINE = [
    { version: 'v0.4.0', date: '2026-05-16', title: 'Production Readiness', desc: 'Graceful shutdown, structured errors, K8s probes, backpressure, YAML config, security hardening.', breaking: false, color: '#22c55e' },
    { version: 'v0.3.0', date: '2026-05-14', title: 'Security & Features', desc: 'CORS, security headers, auth + rate limiting, LoRA adapters, prefix cache, continuous batching.', breaking: true, color: '#ef4444' },
    { version: 'v0.2.0', date: '2026-04-20', title: 'Multi-Node', desc: 'Multi-node cluster support, pipeline parallelism, gRPC communication, worker registration.', breaking: false, color: '#22c55e' },
    { version: 'v0.1.0', date: '2026-03-01', title: 'Initial Release', desc: 'Single-node inference, OpenAI-compatible API, HuggingFace model support, basic CLI.', breaking: false, color: '#22c55e' },
];

export function initChangelogTimeline() {
    const container = document.getElementById('changelogTimeline');
    if (!container) return;

    container.innerHTML = `
        <div class="tl-card">
            <h3>Release Timeline</h3>
            <div class="tl-list">${
                TIMELINE.map((r, i) => `
                    <div class="tl-item">
                        <div class="tl-dot" style="background:${r.color}"></div>
                        ${i < TIMELINE.length - 1 ? '<div class="tl-line"></div>' : ''}
                        <div class="tl-content">
                            <div class="tl-header">
                                <span class="tl-version">${r.version}</span>
                                <span class="tl-date">${r.date}</span>
                                ${r.breaking ? '<span class="tl-breaking">BREAKING</span>' : ''}
                            </div>
                            <div class="tl-title">${r.title}</div>
                            <div class="tl-desc">${r.desc}</div>
                        </div>
                    </div>
                `).join('')
            }</div>
        </div>
    `;
}
