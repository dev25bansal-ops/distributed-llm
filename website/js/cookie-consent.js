/**
 * Cookie Consent Banner
 *
 * GDPR-compliant cookie consent banner.
 * Even though Plausible is privacy-friendly, GDPR requires disclosure.
 *
 * Features:
 * - Minimal, non-intrusive design
 * - Remembers user choice in localStorage
 * - Links to privacy policy
 * - Accessible (keyboard navigation, ARIA labels)
 *
 * Usage:
 *   import { initCookieConsent } from './js/cookie-consent.js';
 *   initCookieConsent();
 */

// ── Configuration ──────────────────────────────────────────────────────

const CONFIG = {
    storageKey: 'distllm-cookie-consent',
    privacyUrl: '/privacy.html',
    position: 'bottom', // 'bottom' or 'top'
};

// ── State ──────────────────────────────────────────────────────────────

let consentGiven = false;

// ── Check Existing Consent ─────────────────────────────────────────────

function hasConsent() {
    return localStorage.getItem(CONFIG.storageKey) === 'accepted';
}

function saveConsent(value) {
    localStorage.setItem(CONFIG.storageKey, value || 'accepted');
    consentGiven = value === 'accepted';
}

// ── Banner UI ──────────────────────────────────────────────────────────

function createBanner() {
    const banner = document.createElement('div');
    banner.id = 'cookieConsent';
    banner.className = 'cookie-consent';
    banner.setAttribute('role', 'dialog');
    banner.setAttribute('aria-label', 'Cookie consent');
    banner.setAttribute('aria-describedby', 'cookieConsentText');

    banner.innerHTML = `
        <div class="cookie-consent-inner">
            <div class="cookie-consent-text">
                <p id="cookieConsentText">
                    <strong>🍪 Privacy-Friendly Analytics</strong><br>
                    We use <a href="https://plausible.io" target="_blank" rel="noopener noreferrer">Plausible Analytics</a>, 
                    a privacy-friendly alternative to Google Analytics. No cookies, no personal data, GDPR compliant. 
                    <a href="${CONFIG.privacyUrl}">Privacy Policy</a>
                </p>
            </div>
            <div class="cookie-consent-actions">
                <button class="cookie-consent-accept" id="cookieAccept">
                    Accept
                </button>
                <button class="cookie-consent-reject" id="cookieReject">
                    Reject
                </button>
                <button class="cookie-consent-learn-more" id="cookieLearnMore">
                    Learn More
                </button>
            </div>
        </div>
    `;

    return banner;
}

// ── Event Handlers ─────────────────────────────────────────────────────

function handleAccept() {
    saveConsent('accepted');
    removeBanner();
}

function handleReject() {
    saveConsent('rejected');
    removeBanner();
    // Disable analytics when consent is rejected
    window.disableAnalytics = true;
}

function handleLearnMore() {
    window.open(CONFIG.privacyUrl, '_blank');
}

function removeBanner() {
    const banner = document.getElementById('cookieConsent');
    if (banner) {
        banner.style.opacity = '0';
        banner.style.transform = 'translateY(100%)';
        setTimeout(() => banner.remove(), 300);
    }
}

// ── Initialization ─────────────────────────────────────────────────────

export function initCookieConsent() {
    // Don't show if consent already given
    if (hasConsent()) {
        consentGiven = true;
        return;
    }

    // Create and show banner
    const banner = createBanner();
    document.body.appendChild(banner);

    // Add event listeners
    document.getElementById('cookieAccept')?.addEventListener('click', handleAccept);
    document.getElementById('cookieReject')?.addEventListener('click', handleReject);
    document.getElementById('cookieLearnMore')?.addEventListener('click', handleLearnMore);

    // Keyboard navigation: Escape dismisses without accepting or rejecting
    banner.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            removeBanner(); // Dismiss without setting consent — banner reappears next visit
        }
    });

    // Show banner with animation
    requestAnimationFrame(() => {
        banner.style.opacity = '1';
        banner.style.transform = 'translateY(0)';
    });
}

export function getConsentStatus() {
    return consentGiven || hasConsent();
}
