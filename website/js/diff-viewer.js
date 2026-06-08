/**
 * Code Diff Viewer — shows migration steps between versions.
 *
 * Usage:
 *   <div id="diffViewer"></div>
 *   <script type="module">
 *     import { initDiffViewer } from './js/diff-viewer.js';
 *     initDiffViewer();
 *   </script>
 */

const MIGRATIONS = [
    {
        title: 'v0.2.x → v0.3.0: API Endpoints',
        description: 'All endpoints moved from / to /v1/ prefix',
        before: `curl http://localhost:8000/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{"messages":[{"role":"user","content":"Hello"}]}'`,
        after: `curl http://localhost:8000/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer $API_KEY" \\
  -d '{"messages":[{"role":"user","content":"Hello"}]}'`,
    },
    {
        title: 'v0.2.x → v0.3.0: Config Format',
        description: 'Config changed from TOML to YAML',
        before: `[model]
name = "llama-3.1-8b"
dtype = "float16"

[network]
port = 8000
host = "0.0.0.0"`,
        after: `model:
  name: llama-3.1-8b
  dtype: float16

network:
  port: 8000
  host: 0.0.0.0`,
    },
    {
        title: 'v0.2.x → v0.3.0: Authentication',
        description: 'API key now required by default',
        before: `# No auth required
curl http://localhost:8000/v1/models`,
        after: `# Set API key
export API_KEY="your-key"
curl -H "Authorization: Bearer $API_KEY" \\
  http://localhost:8000/v1/models

# Or disable for dev
distllm system api --no-auth`,
    },
    {
        title: 'v0.3.0 → v0.4.0: Error Handling',
        description: 'Error responses now include code and troubleshooting URL',
        before: `{
  "error": "Model not found"
}`,
        after: `{
  "error": {
    "code": "MODEL_NOT_FOUND",
    "message": "Model 'xyz' not found",
    "troubleshooting_url": "https://distllm.dev/docs/troubleshooting#2-model-loading-failures"
  }
}`,
    },
];

export function initDiffViewer() {
    const container = document.getElementById('diffViewer');
    if (!container) return;

    container.innerHTML = `
        <div class="diff-card">
            <h3>Migration Guide</h3>
            <p class="diff-desc">Code changes required when upgrading between versions.</p>
            <div id="diffContent"></div>
        </div>
    `;

    const content = document.getElementById('diffContent');

    content.innerHTML = MIGRATIONS.map(m => `
        <div class="diff-item">
            <div class="diff-title">${m.title}</div>
            <div class="diff-subtitle">${m.description}</div>
            <div class="diff-pair">
                <div class="diff-pane before">
                    <div class="diff-pane-header">Before</div>
                    <pre>${escapeHtml(m.before)}</pre>
                </div>
                <div class="diff-pane after">
                    <div class="diff-pane-header">After</div>
                    <pre>${escapeHtml(m.after)}</pre>
                </div>
            </div>
        </div>
    `).join('');
}

function escapeHtml(text) {
    return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
