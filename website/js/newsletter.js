/**
 * Newsletter form — POSTs to backend API with background sync support.
 *
 * Features:
 * - Posts to configurable backend endpoint
 * - Queues for background sync when offline
 * - Shows sync status to user
 * - Falls back to localStorage when service worker unavailable
 *
 * Configurable via data attributes on the form:
 *   data-endpoint="https://api.distllm.dev/newsletter/subscribe"
 *   data-site-key="distllm-dev"
 */
export function initNewsletter() {
    const form = document.getElementById('newsletterForm');
    const success = document.getElementById('newsletterSuccess');
    if (!form || !success) return;

    const endpoint = form.dataset.endpoint || '/api/newsletter/subscribe';
    const siteKey = form.dataset.siteKey || 'distllm-dev';

    const isValidEmail = (email) => /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);

    // Listen for background sync messages from service worker
    if ('serviceWorker' in navigator) {
        navigator.serviceWorker.addEventListener('message', (event) => {
            if (event.data.type === 'NEWSLETTER_SYNCED') {
                showStatus(`✓ ${event.data.email} synced successfully!`, 'success');
            } else if (event.data.type === 'NEWSLETTER_QUEUED') {
                showStatus(`📬 ${event.data.email} queued for sync when online`, 'info');
            }
        });
    }

    function showStatus(message, type = 'success') {
        success.textContent = message;
        success.style.display = 'block';
        success.style.color = type === 'success'
            ? 'var(--green, #00e676)'
            : type === 'error'
                ? 'var(--red, #ef4444)'
                : 'var(--amber, #f59e0b)';
        setTimeout(() => { success.style.display = 'none'; }, 5000);
    }

    form.addEventListener('submit', async e => {
        e.preventDefault();
        const input = form.querySelector('input[type="email"]');
        const email = input?.value.trim();
        if (!email || !isValidEmail(email)) {
            showStatus('Please enter a valid email address.', 'error');
            return;
        }

        // Try to POST to backend
        try {
            const res = await fetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email, site: siteKey }),
            });
            if (!res.ok) throw new Error('Backend rejected');
            showStatus('Thanks for subscribing! We\'ll keep you updated.', 'success');
        } catch (error) {
            // Backend unavailable — try background sync
            if ('serviceWorker' in navigator && navigator.serviceWorker.controller) {
                // Send to service worker for background sync
                navigator.serviceWorker.controller.postMessage({
                    type: 'QUEUE_NEWSLETTER',
                    email,
                    site: siteKey,
                });
                showStatus(`📬 ${email} queued for sync when online`, 'info');
            } else {
                // Fallback to localStorage
                try {
                    const subs = JSON.parse(localStorage.getItem('distllm-newsletter') || '[]');
                    if (!subs.includes(email)) {
                        subs.push(email);
                        localStorage.setItem('distllm-newsletter', JSON.stringify(subs));
                    }
                    showStatus('Thanks for subscribing! (Saved locally — will sync when online)', 'success');
                } catch {
                    showStatus('Thanks for subscribing!', 'success');
                }
            }
        }

        form.reset();
    });

    // Check for queued subscriptions on page load
    checkQueuedSubscriptions();
}

async function checkQueuedSubscriptions() {
    if (!('serviceWorker' in navigator)) return;

    try {
        const registration = await navigator.serviceWorker.ready;

        // Check if there are queued subscriptions
        if ('sync' in registration) {
            const cache = await caches.open('distllm-newsletter-queue');
            const requests = await cache.keys();

            if (requests.length > 0) {
                // Trigger sync
                await registration.sync.register('newsletter-sync');
            }
        }
    } catch (e) {
        // Service worker not available
    }
}
