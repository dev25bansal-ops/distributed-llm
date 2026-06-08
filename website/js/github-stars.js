/**
 * Fetch live GitHub star count with localStorage caching (1 hour TTL).
 * Shows skeleton placeholder while loading, smooth transition to value.
 */
export function initGitHubStars() {
    const el = document.getElementById('starCount');
    if (!el) return;

    if (document.documentElement.dataset.liveStats !== 'true') {
        el.textContent = 'GitHub';
        return;
    }

    const CACHE_KEY = 'distllm-stars';
    const CACHE_TTL = 3600000; // 1 hour

    // Show skeleton placeholder
    el.textContent = '...';
    el.setAttribute('aria-busy', 'true');

    // Show cached value immediately if available
    try {
        const cached = localStorage.getItem(CACHE_KEY);
        if (cached) {
            const { count, ts } = JSON.parse(cached);
            if (Date.now() - ts < CACHE_TTL) {
                el.textContent = count;
                el.removeAttribute('aria-busy');
                return; // Fresh enough, skip fetch
            }
            el.textContent = count; // Show stale while revalidating
        }
    } catch {}

    fetch('https://api.github.com/repos/distributed-llm/distributed-llm')
        .then(r => {
            if (!r.ok) throw new Error(r.status);
            return r.json();
        })
        .then(d => {
            if (d.stargazers_count !== undefined) {
                const count = d.stargazers_count >= 1000
                    ? (d.stargazers_count / 1000).toFixed(1) + 'k'
                    : String(d.stargazers_count);
                el.textContent = count;
                el.removeAttribute('aria-busy');
                try { localStorage.setItem(CACHE_KEY, JSON.stringify({ count, ts: Date.now() })); } catch {}
                window.dispatchEvent(new CustomEvent('distllm-stars-updated'));
            }
        })
        .catch(() => {
            el.removeAttribute('aria-busy');
            if (el.textContent === '...') el.textContent = 'GitHub';
        });
}
