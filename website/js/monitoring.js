/**
 * Error Tracking & Monitoring
 *
 * Integrates with Sentry for error tracking and provides:
 * - Automatic error capture
 * - Performance monitoring
 * - User feedback collection
 * - Health check endpoint
 *
 * Usage:
 *   import { initMonitoring } from './js/monitoring.js';
 *   initMonitoring();
 */

// ── Configuration ──────────────────────────────────────────────────────

const CONFIG = {
    sentryDsn: null, // Set via data attribute or environment
    environment: 'production',
    release: '0.4.0',
    sampleRate: 1.0,
    tracesSampleRate: 0.1,
};

// ── Early return if no DSN configured ───────────────────────────────────
// This prevents Sentry from loading when not configured.
// Set data-sentry-dsn on the script tag to enable.

// ── Sentry Integration ─────────────────────────────────────────────────

async function initSentry(dsn) {
    if (!dsn) {
        return; // Silently skip — Sentry not configured
    }

    try {
        // Dynamic import of Sentry
        const Sentry = await import('https://browser.sentry-cdn.com/7.119.0/bundle.min.js');

        Sentry.init({
            dsn,
            environment: CONFIG.environment,
            release: CONFIG.release,
            sampleRate: CONFIG.sampleRate,
            tracesSampleRate: CONFIG.tracesSampleRate,
            integrations: [
                new Sentry.BrowserTracing(),
                new Sentry.Replay(),
            ],
            replaysSessionSampleRate: 0.1,
            replaysOnErrorSampleRate: 1.0,
            beforeSend(event) {
                // Filter out known non-errors
                if (event.exception?.values?.[0]?.type === 'ChunkLoadError') {
                    return null;
                }
                return event;
            },
        });

        console.log('[Monitoring] Sentry initialized');
    } catch (e) {
        console.warn('[Monitoring] Failed to load Sentry:', e);
    }
}

// ── Real User Monitoring (RUM) ─────────────────────────────────────────

function initRUM() {
    // Track Core Web Vitals
    if ('PerformanceObserver' in window) {
        // Largest Contentful Paint (LCP)
        new PerformanceObserver((list) => {
            const entries = list.getEntries();
            const lastEntry = entries[entries.length - 1];
            trackMetric('LCP', lastEntry.startTime);
        }).observe({ type: 'largest-contentful-paint', buffered: true });

        // First Input Delay (FID)
        new PerformanceObserver((list) => {
            const entries = list.getEntries();
            entries.forEach(entry => {
                trackMetric('FID', entry.processingStart - entry.startTime);
            });
        }).observe({ type: 'first-input', buffered: true });

        // Cumulative Layout Shift (CLS)
        let clsValue = 0;
        new PerformanceObserver((list) => {
            const entries = list.getEntries();
            entries.forEach(entry => {
                if (!entry.hadRecentInput) {
                    clsValue += entry.value;
                }
            });
            trackMetric('CLS', clsValue);
        }).observe({ type: 'layout-shift', buffered: true });

        // Time to First Byte (TTFB)
        const navEntry = performance.getEntriesByType('navigation')[0];
        if (navEntry) {
            trackMetric('TTFB', navEntry.responseStart - navEntry.requestStart);
        }
    }

    // Track page load time
    window.addEventListener('load', () => {
        const loadTime = performance.now();
        trackMetric('PageLoad', loadTime);
    });

    // Track JavaScript errors
    window.addEventListener('error', (event) => {
        trackError({
            message: event.message,
            filename: event.filename,
            lineno: event.lineno,
            colno: event.colno,
            error: event.error,
        });
    });

    // Track unhandled promise rejections
    window.addEventListener('unhandledrejection', (event) => {
        trackError({
            message: 'Unhandled Promise Rejection',
            reason: event.reason,
        });
    });
}

// ── Metric Tracking ────────────────────────────────────────────────────

const metrics = {};

function trackMetric(name, value) {
    metrics[name] = value;

    // Send to analytics if available
    if (window.plausible) {
        window.plausible('Web Vital', {
            props: { name, value: Math.round(value) },
        });
    }

    // Log in development
    if (CONFIG.environment === 'development') {
        console.log(`[RUM] ${name}: ${Math.round(value)}ms`);
    }
}

function trackError(error) {
    console.error('[Error]', error);

    // Send to Sentry if available
    if (window.Sentry) {
        window.Sentry.captureException(error.error || new Error(error.message));
    }

    // Send to analytics if available
    if (window.plausible) {
        window.plausible('Error', {
            props: { message: error.message },
        });
    }
}

// ── Health Check ────────────────────────────────────────────────────────

function getHealthStatus() {
    return {
        status: 'ok',
        version: CONFIG.release,
        timestamp: new Date().toISOString(),
        uptime: performance.now(),
        metrics: {
            ...metrics,
            memory: performance.memory ? {
                usedJSHeapSize: Math.round(performance.memory.usedJSHeapSize / 1024 / 1024),
                totalJSHeapSize: Math.round(performance.memory.totalJSHeapSize / 1024 / 1024),
            } : null,
        },
    };
}

// ── Public API ──────────────────────────────────────────────────────────

export function initMonitoring(options = {}) {
    // Merge options
    Object.assign(CONFIG, options);

    // Get Sentry DSN from data attribute
    const scriptTag = document.querySelector('script[data-sentry-dsn]');
    if (scriptTag) {
        CONFIG.sentryDsn = scriptTag.dataset.sentryDsn;
    }

    // Initialize Sentry
    initSentry(CONFIG.sentryDsn);

    // Initialize RUM
    initRUM();

    // Expose health check
    window.distllmHealth = getHealthStatus;

    // console.log removed for production — uncomment for debugging
}


export { trackMetric, trackError, getHealthStatus };
