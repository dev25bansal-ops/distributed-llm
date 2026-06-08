/**
 * Newsletter Archive — past newsletter editions.
 *
 * Usage:
 *   <div id="newsletterArchive"></div>
 *   <script type="module">
 *     import { initNewsletterArchive } from './js/newsletter-archive.js';
 *     initNewsletterArchive();
 *   </script>
 */

const ARCHIVE = [
    { id: 1, date: '2026-05-28', subject: 'DistLLM v0.4.0 Release', preview: 'Production readiness: graceful shutdown, K8s probes, structured errors, YAML config.' },
    { id: 2, date: '2026-05-14', subject: 'v0.3.0 — Security & Features', preview: 'Breaking changes: API endpoints moved to /v1/, auth required by default, config format changed.' },
    { id: 3, date: '2026-04-20', subject: 'Multi-Node Clusters Are Here', preview: 'Run models across multiple machines with pipeline parallelism. Docker Compose setup included.' },
    { id: 4, date: '2沺26-03-01', subject: 'Welcome to DistLLM', preview: 'First release! Single-node inference with OpenAI-compatible API. Getting started guide inside.' },
];

export function initNewsletterArchive() {
    const container = document.getElementById('newsletterArchive');
    if (!container) return;

    container.innerHTML = `
        <div class="na-card">
            <h3>Newsletter Archive</h3>
            <p class="na-desc">Past newsletter editions.</p>
            <div class="na-list">${
                ARCHIVE.map(n => `
                    <div class="na-item">
                        <div class="na-date">${n.date}</div>
                        <div class="na-content">
                            <div class="na-subject">${n.subject}</div>
                            <div class="na-preview">${n.preview}</div>
                        </div>
                    </div>
                `).join('')
            }</div>
        </div>
    `;
}
