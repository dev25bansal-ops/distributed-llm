/**
 * Notification Toast — shows temporary notifications.
 *
 * Usage:
 *   import { showToast } from './js/toast.js';
 *   showToast('New version available!', 'info');
 *   showToast('Settings saved', 'success');
 *   showToast('Connection failed', 'error');
 */

let container = null;

function ensureContainer() {
    if (container) return container;
    container = document.createElement('div');
    container.id = 'toast-container';
    container.setAttribute('aria-live', 'polite');
    container.style.cssText = 'position:fixed;top:72px;right:16px;z-index:9999;display:flex;flex-direction:column;gap:8px;pointer-events:none;';
    document.body.appendChild(container);
    return container;
}

/**
 * Show a toast notification.
 * @param {string} message - The message to display.
 * @param {'info'|'success'|'error'|'warning'} type - Toast type.
 * @param {number} duration - Auto-dismiss in ms (default 4000).
 */
export function showToast(message, type = 'info', duration = 4000) {
    const c = ensureContainer();
    const toast = document.createElement('div');
    const colors = {
        info: { bg: '#1a2a3a', border: '#306090', text: '#58a6ff' },
        success: { bg: '#1a3a1a', border: '#2a6a2a', text: '#3fb950' },
        error: { bg: '#3a1a1a', border: '#6a2a2a', text: '#f85149' },
        warning: { bg: '#3a3a1a', border: '#6a6a2a', text: '#d29922' },
    };
    const c_ = colors[type] || colors.info;

    toast.style.cssText = `
        pointer-events: auto;
        background: ${c_.bg};
        border: 1px solid ${c_.border};
        color: ${c_.text};
        padding: 10px 16px;
        border-radius: 8px;
        font-size: 13px;
        font-family: Inter, sans-serif;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        animation: toastIn 0.3s ease;
        max-width: 360px;
    `;
    toast.textContent = message;
    c.appendChild(toast);

    // Add animation keyframes once
    if (!document.getElementById('toast-styles')) {
        const style = document.createElement('style');
        style.id = 'toast-styles';
        style.textContent = `
            @keyframes toastIn { from { opacity: 0; transform: translateX(20px); } to { opacity: 1; transform: translateX(0); } }
            @keyframes toastOut { from { opacity: 1; transform: translateX(0); } to { opacity: 0; transform: translateX(20px); } }
        `;
        document.head.appendChild(style);
    }

    setTimeout(() => {
        toast.style.animation = 'toastOut 0.3s ease forwards';
        setTimeout(() => toast.remove(), 300);
    }, duration);
}

// Auto-show "new version" toast for returning visitors
export function checkNewVersion() {
    const current = 'v0.4.0';
    const last = localStorage.getItem('distllm-last-seen-version');
    if (last && last !== current) {
        showToast(`New version ${current} available! See changelog.`, 'info', 6000);
    }
    localStorage.setItem('distllm-last-seen-version', current);
}
