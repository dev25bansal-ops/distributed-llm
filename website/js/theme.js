/**
 * Theme toggle — persisted in localStorage.
 * Uses textContent with Unicode chars instead of innerHTML for security.
 */
export function initTheme() {
    const btn = document.getElementById('themeBtn');
    if (!btn) return;

    const saved = localStorage.getItem('distllm-theme') || 'dark';
    if (saved === 'light') {
        document.documentElement.setAttribute('data-theme', 'light');
        btn.textContent = '☀'; // sun
    }

    btn.addEventListener('click', () => {
        const isLight = document.documentElement.getAttribute('data-theme') === 'light';
        if (isLight) {
            document.documentElement.removeAttribute('data-theme');
            btn.textContent = '☾'; // crescent moon
            localStorage.setItem('distllm-theme', 'dark');
        } else {
            document.documentElement.setAttribute('data-theme', 'light');
            btn.textContent = '☀'; // sun
            localStorage.setItem('distllm-theme', 'light');
        }
    });
}
