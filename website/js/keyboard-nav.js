/**
 * A5. Keyboard-Driven Navigation — j/k sections, / search, ? shortcuts overlay.
 */
const SECTIONS = ['hero', 'quickstart', 'how', 'features', 'code', 'compare', 'testimonials', 'pricing', 'faq'];
let currentSectionIndex = 0;

const SHORTCUTS = [
    { key: 'j / ↓', desc: 'Next section' },
    { key: 'k / ↑', desc: 'Previous section' },
    { key: '/', desc: 'Search (command palette)' },
    { key: '?', desc: 'Show this help' },
    { key: 'Escape', desc: 'Close dialog / help' },
    { key: 't', desc: 'Toggle theme' },
];

export function initKeyboardNav() {
    // Create help overlay
    const overlay = document.createElement('div');
    overlay.id = 'keyboardHelp';
    overlay.className = 'keyboard-help-overlay';
    overlay.innerHTML = `
        <div class="keyboard-help-card">
            <h3>Keyboard Shortcuts</h3>
            <div class="keyboard-help-list">
                ${SHORTCUTS.map(s => `<div class="keyboard-help-row"><kbd>${s.key}</kbd><span>${s.desc}</span></div>`).join('')}
            </div>
            <p class="keyboard-help-dismiss">Press <kbd>?</kbd> or <kbd>Esc</kbd> to close</p>
        </div>
    `;
    document.body.appendChild(overlay);

    document.addEventListener('keydown', (e) => {
        // Don't capture when typing in inputs
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.isContentEditable) return;

        // Don't capture when command palette is open
        const cmdPalette = document.getElementById('cmdPalette');
        if (cmdPalette && cmdPalette.classList.contains('open')) return;

        switch (e.key) {
            case 'j':
            case 'ArrowDown':
                e.preventDefault();
                navigateSection(1);
                break;
            case 'k':
            case 'ArrowUp':
                e.preventDefault();
                navigateSection(-1);
                break;
            case '?':
                e.preventDefault();
                toggleHelp();
                break;
            case 'Escape':
                if (overlay.classList.contains('open')) {
                    overlay.classList.remove('open');
                }
                break;
            case 't':
                document.getElementById('themeBtn')?.click();
                break;
        }
    });

    // Track current section via IntersectionObserver
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const idx = SECTIONS.indexOf(entry.target.id);
                if (idx !== -1) currentSectionIndex = idx;
            }
        });
    }, { threshold: 0.3 });

    SECTIONS.forEach(id => {
        const el = document.getElementById(id);
        if (el) observer.observe(el);
    });
}

function navigateSection(direction) {
    currentSectionIndex = Math.max(0, Math.min(SECTIONS.length - 1, currentSectionIndex + direction));
    const target = document.getElementById(SECTIONS[currentSectionIndex]);
    if (target) target.scrollIntoView({ behavior: 'smooth' });
}

function toggleHelp() {
    document.getElementById('keyboardHelp')?.classList.toggle('open');
}
