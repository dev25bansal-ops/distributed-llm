/**
 * Command palette — Ctrl+K / Cmd+K to open, searchable, keyboard navigable.
 * All state is encapsulated in the initCmdPalette closure.
 */
const ITEMS = [
    { icon: '⚡', label: 'Features', action: () => scrollTo('features') },
    { icon: '✅', label: 'Repository-backed Proof', action: () => scrollTo('proof') },
    { icon: '🚀', label: 'Quick Start', action: () => scrollTo('quickstart') },
    { icon: '🔧', label: 'How It Works', action: () => scrollTo('how') },
    { icon: '💬', label: 'Chat Demo', action: () => scrollTo('chat-demo') },
    { icon: '🖥', label: 'GPU Checker', action: () => scrollTo('gpu-checker') },
    { icon: '📦', label: 'Model Explorer', action: () => scrollTo('models') },
    { icon: '🚀', label: 'Deploy Wizard', action: () => scrollTo('deploy') },
    { icon: '⚖', label: 'Compare', action: () => scrollTo('compare') },
    { icon: '💰', label: 'Pricing & Calculator', action: () => scrollTo('pricing') },
    { icon: '🗺', label: 'Roadmap', action: () => scrollTo('roadmap') },
    { icon: '❓', label: 'FAQ', action: () => scrollTo('faq') },
    { icon: '📖', label: 'Documentation', action: () => { window.location.href = '/docs.html'; } },
    { icon: '📝', label: 'Blog', action: () => { window.location.href = '/blog.html'; } },
    { icon: '🔗', label: 'Open GitHub', action: () => { window.open('https://github.com/distributed-llm/distributed-llm', '_blank', 'noopener,noreferrer'); } },
    { icon: '💬', label: 'GitHub Discussions', action: () => { window.open('https://github.com/distributed-llm/distributed-llm/discussions', '_blank', 'noopener,noreferrer'); } },
];

function scrollTo(id) {
    document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' });
}

export function initCmdPalette() {
    const overlay = document.getElementById('cmdPalette');
    const input = document.getElementById('cmdInput');
    const results = document.getElementById('cmdResults');

    if (!overlay || !input || !results) return;

    let selected = 0;
    let filteredItems = [];

    function render(filter) {
        const f = filter.toLowerCase();
        filteredItems = ITEMS.filter(i => i.label.toLowerCase().includes(f));
        selected = Math.min(selected, Math.max(filteredItems.length - 1, 0));
        results.textContent = '';

        if (filteredItems.length === 0) {
            const empty = document.createElement('div');
            empty.className = 'cmd-item';
            empty.textContent = 'No matches';
            results.appendChild(empty);
            return;
        }

        filteredItems.forEach((item, i) => {
            const el = document.createElement('button');
            el.type = 'button';
            el.className = `cmd-item${i === selected ? ' selected' : ''}`;
            el.dataset.idx = String(i);

            const icon = document.createElement('span');
            icon.className = 'cmd-item-icon';
            icon.textContent = item.icon;

            const label = document.createElement('span');
            label.textContent = item.label;

            el.append(icon, label);
            el.addEventListener('click', () => {
                item.action();
                close();
            });
            results.appendChild(el);
        });
    }

    function open() {
        overlay.classList.add('open');
        input.value = '';
        selected = 0;
        render('');
        setTimeout(() => input.focus(), 50);
    }

    function close() {
        overlay.classList.remove('open');
    }

    input.addEventListener('input', () => { selected = 0; render(input.value); });
    overlay.addEventListener('click', e => { if (e.target === overlay) close(); });

    document.addEventListener('keydown', e => {
        // Open/close
        if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
            e.preventDefault();
            overlay.classList.contains('open') ? close() : open();
        }
        if (e.key === 'Escape' && overlay.classList.contains('open')) close();

        // Navigation
        if (!overlay.classList.contains('open')) return;

        if (e.key === 'Enter') {
            if (filteredItems[selected]) {
                filteredItems[selected].action();
                close();
            }
        }
        if (e.key === 'ArrowDown' && filteredItems.length) {
            e.preventDefault();
            selected = (selected + 1) % filteredItems.length;
            render(input.value);
        }
        if (e.key === 'ArrowUp' && filteredItems.length) {
            e.preventDefault();
            selected = (selected - 1 + filteredItems.length) % filteredItems.length;
            render(input.value);
        }
    });
}
