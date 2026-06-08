/**
 * Changelog & Migration Guide System.
 *
 * Parses GitHub releases into structured format, shows breaking change
 * alerts, and provides migration guides between versions.
 *
 * Usage:
 *   <div id="changelog"></div>
 *   <script type="module">
 *     import { initChangelog } from './js/changelog.js';
 *     initChangelog();
 *   </script>
 */

// ── Release Data ───────────────────────────────────────────────────────

const RELEASES = [
    {
        version: 'v0.4.0',
        date: '2026-05-16',
        codename: 'Production Readiness',
        breaking: false,
        highlights: [
            'Graceful shutdown with request draining',
            'Structured error codes with troubleshooting links',
            'Kubernetes native gRPC health probes',
            'Backpressure middleware for overload protection',
            'YAML config validation at startup',
            'Security hardening (CSP, HSTS, CSRF)',
        ],
        breaking_changes: [],
        migration: null,
    },
    {
        version: 'v0.3.0',
        date: '2026-05-14',
        codename: 'Security & Features',
        breaking: true,
        highlights: [
            'CORS middleware for browser access',
            'Security headers middleware',
            'Authentication with API key',
            'Rate limiting with token bucket',
            'LoRA adapter management',
            'Prefix cache for shared prompts',
            'Continuous batching scheduler',
        ],
        breaking_changes: [
            'API endpoints moved from `/` to `/v1/` prefix',
            'Auth now required by default (set API_KEY env var)',
            'Config file format changed from TOML to YAML',
        ],
        migration: {
            from: 'v0.2.x',
            steps: [
                'Update API endpoint URLs: `/chat/completions` → `/v1/chat/completions`',
                'Set `API_KEY` environment variable or use `--no-auth` for development',
                'Convert `config.toml` to `config.yaml` (see migration tool)',
                'Update SDK to v0.3.0+',
            ],
        },
    },
    {
        version: 'v0.2.0',
        date: '2026-04-20',
        codename: 'Multi-Node',
        breaking: false,
        highlights: [
            'Multi-node cluster support',
            'Pipeline parallelism across GPUs',
            'gRPC inter-node communication',
            'Worker registration and discovery',
        ],
        breaking_changes: [],
        migration: null,
    },
    {
        version: 'v0.1.0',
        date: '2026-03-01',
        codename: 'Initial Release',
        breaking: false,
        highlights: [
            'Single-node inference',
            'OpenAI-compatible API',
            'HuggingFace model support',
            'Basic CLI',
        ],
        breaking_changes: [],
        migration: null,
    },
];

// ── UI ─────────────────────────────────────────────────────────────────

export function initChangelog() {
    const container = document.getElementById('changelog');
    if (!container) return;

    container.innerHTML = `
        <div class="cl-card">
            <h3>Changelog & Migration Guide</h3>
            <p class="cl-desc">Track changes, breaking changes, and migration paths between versions.</p>
            <div id="clContent"></div>
        </div>
    `;

    const content = document.getElementById('clContent');

    // "What's New" banner for returning visitors
    const lastSeen = localStorage.getItem('distllm-last-version');
    const latest = RELEASES[0].version;
    if (lastSeen && lastSeen !== latest) {
        const banner = document.createElement('div');
        banner.className = 'cl-what-new';
        banner.innerHTML = `
            <h4>🆕 What's New in ${latest}</h4>
            <p>${RELEASES[0].highlights.slice(0, 3).join(' • ')}</p>
        `;
        content.appendChild(banner);
    }
    localStorage.setItem('distllm-last-version', latest);

    // Render releases
    for (const rel of RELEASES) {
        const div = document.createElement('div');
        div.className = 'cl-release';

        let html = `
            <div class="cl-release-header">
                <span class="cl-version">${rel.version}</span>
                <span class="cl-date">${rel.date}</span>
                <span class="cl-codename">${rel.codename}</span>
                ${rel.breaking ? '<span class="cl-breaking-badge">BREAKING</span>' : ''}
            </div>
            <ul class="cl-highlights">
                ${rel.highlights.map(h => `<li>${h}</li>`).join('')}
            </ul>
        `;

        // Breaking changes
        if (rel.breaking_changes.length > 0) {
            html += `
                <div class="cl-breaking">
                    <h4>⚠ Breaking Changes</h4>
                    <ul>${rel.breaking_changes.map(b => `<li>${b}</li>`).join('')}</ul>
                </div>
            `;
        }

        // Migration guide
        if (rel.migration) {
            html += `
                <div class="cl-migration">
                    <h4>🔄 Migration Guide: ${rel.migration.from} → ${rel.version}</h4>
                    ${rel.migration.steps.map((s, i) => `
                        <div class="step"><span class="step-num">${i + 1}.</span><span>${s}</span></div>
                    `).join('')}
                </div>
            `;
        }

        div.innerHTML = html;
        content.appendChild(div);

        // Divider
        const hr = document.createElement('div');
        hr.className = 'cl-divider';
        content.appendChild(hr);
    }
}
