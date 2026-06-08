/**
 * Contributor Grid — pulls from GitHub API, shows avatar grid.
 *
 * Usage:
 *   <div id="contribGrid"></div>
 *   <script type="module">
 *     import { initContribGrid } from './js/contrib-grid.js';
 *     initContribGrid();
 *   </script>
 */

function escapeAttr(str) {
    return String(str).replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/'/g, '&#39;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function safeUrl(url) {
    try {
        const parsed = new URL(url);
        return ['https:', 'http:'].includes(parsed.protocol) ? url : '#';
    } catch { return '#'; }
}

function safeText(str) {
    return document.createTextNode(String(str)).textContent;
}

export function initContribGrid() {
    const container = document.getElementById('contribGrid');
    if (!container) return;

    // Build initial structure using safe DOM methods
    const card = document.createElement('div');
    card.className = 'cg-card';

    const h3 = document.createElement('h3');
    h3.textContent = 'Contributors';
    card.appendChild(h3);

    const desc = document.createElement('p');
    desc.className = 'cg-desc';
    desc.textContent = 'People who helped build DistLLM.';
    card.appendChild(desc);

    const grid = document.createElement('div');
    grid.className = 'cg-grid';
    grid.id = 'cgGrid';
    const loading = document.createElement('p');
    loading.style.cssText = 'color:#888;font-size:13px;';
    loading.textContent = 'Loading...';
    grid.appendChild(loading);
    card.appendChild(grid);

    container.appendChild(card);

    fetch('https://api.github.com/repos/distributed-llm/distributed-llm/contributors?per_page=30')
        .then(r => r.ok ? r.json() : [])
        .then(contribs => {
            const g = document.getElementById('cgGrid');
            g.innerHTML = '';
            if (!contribs.length) {
                const empty = document.createElement('p');
                empty.style.cssText = 'color:#888;font-size:13px;';
                empty.textContent = 'No contributors yet. Be the first!';
                g.appendChild(empty);
                return;
            }
            for (const c of contribs) {
                const login = escapeAttr(c.login || '');
                const href = safeUrl(c.html_url);
                const avatar = safeUrl(c.avatar_url);
                const commits = Number(c.contributions) || 0;

                const a = document.createElement('a');
                a.href = href;
                a.target = '_blank';
                a.rel = 'noopener noreferrer';
                a.className = 'cg-avatar';
                a.title = `${c.login || 'contributor'} (${commits} commits)`;

                const img = document.createElement('img');
                img.src = avatar;
                img.alt = login;
                img.loading = 'lazy';
                a.appendChild(img);

                g.appendChild(a);
            }
        })
        .catch(() => {
            const g = document.getElementById('cgGrid');
            g.innerHTML = '';
            const err = document.createElement('p');
            err.style.cssText = 'color:#888;font-size:13px;';
            err.textContent = 'Could not load contributors.';
            g.appendChild(err);
        });
}
