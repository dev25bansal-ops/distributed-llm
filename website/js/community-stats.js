/**
 * Community stats — fetch PyPI downloads, Docker pulls, GitHub stars, contributors.
 *
 * Caching strategy:
 * - Service worker caches API responses (network-first + cache-fallback)
 * - localStorage cache with 1-hour TTL as fallback
 * - GitHub stars reads from localStorage cache populated by github-stars.js
 */
export function initCommunityStats() {
    const liveStats = document.documentElement.dataset.liveStats === 'true';
    const staticFallbacks = {
        githubStars: 'GitHub',
        pypiDownloads: 'PyPI',
        dockerPulls: 'Docker',
        contributors: 'GitHub',
    };

    if (!liveStats) {
        Object.entries(staticFallbacks).forEach(([id, value]) => {
            const el = document.getElementById(id);
            if (el) el.textContent = value;
        });
        return;
    }

    const CACHE_TTL = 3600000; // 1 hour
    const formatCount = value => value >= 1000 ? (value / 1000).toFixed(1) + 'k' : String(value);

    // Show skeleton placeholders
    ['githubStars', 'pypiDownloads', 'dockerPulls', 'contributors'].forEach(id => {
        const el = document.getElementById(id);
        if (el) {
            el.textContent = '...';
            el.setAttribute('aria-busy', 'true');
        }
    });

    // Helper: fetch with localStorage cache + timeout + service worker fallback
    function cachedFetch(key, url, extract, timeoutMs = 5000) {
        // Check localStorage first
        try {
            const cached = localStorage.getItem(`distllm-stat-${key}`);
            if (cached) {
                const { value, ts } = JSON.parse(cached);
                if (Date.now() - ts < CACHE_TTL) return Promise.resolve(value);
            }
        } catch {}

        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), timeoutMs);

        return fetch(url, { signal: controller.signal })
            .then(r => {
                clearTimeout(timeout);
                if (!r.ok) throw new Error(r.status);
                return r.json();
            })
            .then(d => {
                const value = extract(d);
                // Cache in localStorage
                try {
                    localStorage.setItem(`distllm-stat-${key}`, JSON.stringify({
                        value,
                        ts: Date.now(),
                        source: 'network'
                    }));
                } catch {}
                return value;
            })
            .catch(() => {
                clearTimeout(timeout);
                // Try to get from service worker cache
                return getFromServiceWorkerCache(key, url, extract);
            });
    }

    // Helper: get from service worker cache
    async function getFromServiceWorkerCache(key, url, extract) {
        if (!('caches' in window)) return null;

        try {
            const cache = await caches.open('distllm-api-v1');
            const response = await cache.match(url);

            if (response) {
                const data = await response.json();
                const value = extract(data);

                // Also cache in localStorage for faster access
                try {
                    localStorage.setItem(`distllm-stat-${key}`, JSON.stringify({
                        value,
                        ts: Date.now(),
                        source: 'sw-cache'
                    }));
                } catch {}

                return value;
            }
        } catch (e) {
            // Service worker cache not available
        }

        return null;
    }

    // GitHub stars — read from cache, listen for update event from github-stars.js
    const starsEl = document.getElementById('githubStars');
    if (starsEl) {
        try {
            const cached = localStorage.getItem('distllm-stars');
            if (cached) {
                const { count } = JSON.parse(cached);
                starsEl.textContent = count;
            }
        } catch {}

        // Listen for stars update event (dispatched by github-stars.js)
        const updateStars = () => {
            try {
                const c = localStorage.getItem('distllm-stars');
                if (c) starsEl.textContent = JSON.parse(c).count;
            } catch {}
        };
        window.addEventListener('distllm-stars-updated', updateStars);
        // Also check once after a short delay in case event already fired
        setTimeout(updateStars, 500);
    }

    // PyPI downloads
    const pypiEl = document.getElementById('pypiDownloads');
    if (pypiEl) {
        cachedFetch('pypi', 'https://pypistats.org/api/packages/distributed-llm/recent',
            d => d?.data?.last_month || 0
        ).then(v => {
            pypiEl.textContent = v !== null ? formatCount(v) : 'PyPI';
            pypiEl.removeAttribute('aria-busy');
        });
    }

    // Docker pulls
    const dockerEl = document.getElementById('dockerPulls');
    if (dockerEl) {
        cachedFetch('docker', 'https://hub.docker.com/v2/repositories/distributed-llm/coordinator/',
            d => d?.pull_count || 0
        ).then(v => {
            dockerEl.textContent = v !== null ? formatCount(v) : 'Docker Hub';
            dockerEl.removeAttribute('aria-busy');
        });
    }

    // GitHub contributors
    const contribEl = document.getElementById('contributors');
    if (contribEl) {
        cachedFetch('contributors', 'https://api.github.com/repos/distributed-llm/distributed-llm/contributors?per_page=1&anon=true',
            d => Array.isArray(d) ? d.length : 0
        ).then(v => {
            contribEl.textContent = (v !== null && v > 0) ? String(v) : 'GitHub';
            contribEl.removeAttribute('aria-busy');
        });
    }
}
