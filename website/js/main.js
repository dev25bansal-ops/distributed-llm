/**
 * Main entry point — initializes all modules.
 *
 * CRITICAL modules: Loaded synchronously (theme, scroll, navigation)
 * LAZY modules: Loaded when scrolled into view via IntersectionObserver
 * DEFERRED modules: Loaded after page is interactive
 */

// ── CRITICAL: Must load immediately (prevent FOUC, enable interaction) ──
import { initTheme } from './theme.js';
import { initScroll } from './scroll.js';
import { initKeyboardNav } from './keyboard-nav.js';
import { initCmdPalette } from './cmd-palette.js';
import { initGitHubStars } from './github-stars.js';
import { initNewsletter } from './newsletter.js';
import { initCommunityStats } from './community-stats.js';
import { initToast, checkNewVersion } from './toast.js';

// ── LAZY: Load when scrolled into view (IntersectionObserver) ──
const LAZY_MODULES = [
    { name: 'gpuChecker', loader: () => import('./gpu-checker.js'), init: 'initGpuChecker' },
    { name: 'modelExplorer', loader: () => import('./model-explorer.js'), init: 'initModelExplorer' },
    { name: 'deployWizard', loader: () => import('./deploy-wizard.js'), init: 'initDeployWizard' },
    { name: 'clusterViz', loader: () => import('./cluster-viz.js'), init: 'initClusterViz' },
    { name: 'modelRec', loader: () => import('./model-rec.js'), init: 'initModelRec' },
    { name: 'apiPlayground', loader: () => import('./api-playground.js'), init: 'initApiPlayground' },
    { name: 'benchDashboard', loader: () => import('./bench-dashboard.js'), init: 'initBenchDashboard' },
    { name: 'communityHub', loader: () => import('./community-hub.js'), init: 'initCommunityHub' },
    { name: 'calculator', loader: () => import('./calculator.js'), init: 'initCalculator' },
    { name: 'terminal', loader: () => import('./terminal.js'), init: 'initTerminal' },
    { name: 'chatDemo', loader: () => import('./chat-demo.js'), init: 'initChatDemo' },
    { name: 'changelog', loader: () => import('./changelog.js'), init: 'initChangelog' },
    { name: 'diffViewer', loader: () => import('./diff-viewer.js'), init: 'initDiffViewer' },
    { name: 'videoEmbeds', loader: () => import('./video-embed.js'), init: 'initVideoEmbeds' },
    { name: 'screenshotGallery', loader: () => import('./screenshot-gallery.js'), init: 'initScreenshotGallery' },
    { name: 'testimonialCarousel', loader: () => import('./testimonial-carousel.js'), init: 'initTestimonialCarousel' },
    { name: 'contribGrid', loader: () => import('./contrib-grid.js'), init: 'initContribGrid' },
    { name: 'changelogTimeline', loader: () => import('./changelog-timeline.js'), init: 'initChangelogTimeline' },
    { name: 'newsletterArchive', loader: () => import('./newsletter-archive.js'), init: 'initNewsletterArchive' },
    { name: 'feedback', loader: () => import('./feedback.js'), init: 'initFeedback' },
    { name: 'modelPlayground', loader: () => import('./model-playground.js'), init: 'initModelPlayground' },
    { name: 'gpuBenchmarks', loader: () => import('./gpu-benchmarks.js'), init: 'initGpuBenchmarks' },
    { name: 'liveDashboard', loader: () => import('./live-dashboard.js'), init: 'initLiveDashboard' },
    { name: 'archDiagram', loader: () => import('./arch-diagram.js'), init: 'initArchDiagram' },
    { name: 'modelOptimizer', loader: () => import('./model-optimizer.js'), init: 'initModelOptimizer' },
    { name: 'quickstartWizard', loader: () => import('./quickstart-wizard.js'), init: 'initQuickstartWizard' },
    { name: 'videoTutorials', loader: () => import('./video-tutorials.js'), init: 'initVideoTutorials' },
];

// ── LAZY-LOAD CHATBOT: Only load on user interaction ──
const CHATBOT_MODULE = { name: 'aiChatbot', loader: () => import('./ai-chatbot.js'), init: 'initAiChatbot' };

// ── Inline utility functions ──

function initMobileMenu() {
    const btn = document.getElementById('menuBtn');
    const links = document.getElementById('navLinks');
    if (!btn || !links) return;

    btn.addEventListener('click', () => {
        const isOpen = links.classList.toggle('open');
        btn.classList.toggle('open');
        btn.setAttribute('aria-expanded', isOpen);
    });

    links.querySelectorAll('a').forEach(link => {
        link.addEventListener('click', () => {
            links.classList.remove('open');
            btn.classList.remove('open');
            btn.setAttribute('aria-expanded', 'false');
        });
    });
}

function initCodeTabs() {
    const tabs = Array.from(document.querySelectorAll('.code-tab'));
    if (!tabs.length) return;

    const activate = (tab, focus = false) => {
        tabs.forEach(t => {
            const isActive = t === tab;
            t.classList.toggle('active', isActive);
            t.setAttribute('aria-selected', String(isActive));
            t.setAttribute('tabindex', isActive ? '0' : '-1');
        });

        document.querySelectorAll('.code-content').forEach(c => c.classList.remove('active'));
        document.getElementById('tab-' + tab.dataset.tab)?.classList.add('active');
        if (focus) tab.focus();
    };

    tabs.forEach((tab, index) => {
        tab.setAttribute('tabindex', tab.classList.contains('active') ? '0' : '-1');
        tab.addEventListener('click', () => activate(tab));
        tab.addEventListener('keydown', e => {
            let nextIndex = index;
            if (e.key === 'ArrowRight') nextIndex = (index + 1) % tabs.length;
            else if (e.key === 'ArrowLeft') nextIndex = (index - 1 + tabs.length) % tabs.length;
            else if (e.key === 'Home') nextIndex = 0;
            else if (e.key === 'End') nextIndex = tabs.length - 1;
            else return;

            e.preventDefault();
            activate(tabs[nextIndex], true);
        });
    });
}

function initCopyButtons() {
    document.querySelectorAll('.code-copy').forEach(btn => {
        btn.addEventListener('click', () => {
            const dataCopy = btn.getAttribute('data-copy');
            const text = dataCopy || (() => {
                const active = document.querySelector('.code-content.active');
                return active ? active.innerText.replace(/^\n|\n$/g, '') : '';
            })();

            if (!text) return;

            navigator.clipboard.writeText(text)
                .then(() => {
                    btn.textContent = 'Copied!';
                    setTimeout(() => { btn.textContent = 'Copy'; }, 2000);
                })
                .catch(() => {
                    btn.textContent = 'Failed';
                    setTimeout(() => { btn.textContent = 'Copy'; }, 2000);
                });
        });
    });
}

function initFAQ() {
    document.querySelectorAll('.faq-q').forEach(btn => {
        btn.addEventListener('click', () => {
            const item = btn.parentElement;
            const wasOpen = item.classList.contains('open');

            document.querySelectorAll('.faq-item').forEach(i => {
                i.classList.remove('open');
                i.querySelector('.faq-q')?.setAttribute('aria-expanded', 'false');
            });

            if (!wasOpen) {
                item.classList.add('open');
                btn.setAttribute('aria-expanded', 'true');
            }
        });
    });
}

// Load Google Fonts (replaces inline onload handler for CSP compliance)
function loadFonts() {
    const preload = document.getElementById('fontPreload');
    if (preload) {
        preload.rel = 'stylesheet';
    }
}

// ── IntersectionObserver for lazy modules ──

function initLazyModules() {
    const safeInit = (name, fn) => {
        try { fn(); } catch (e) { console.warn(`[DistLLM] ${name} init failed:`, e); }
    };

    const loadModule = async (mod) => {
        try {
            const module = await mod.loader();
            if (module[mod.init]) {
                safeInit(mod.name, module[mod.init]);
            }
        } catch (e) {
            console.warn(`[DistLLM] ${mod.name} load failed:`, e);
        }
    };

    // Use IntersectionObserver to load modules when scrolled into view
    if ('IntersectionObserver' in window) {
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const element = entry.target;
                    const moduleName = element.dataset.lazyModule;

                    if (moduleName) {
                        const mod = LAZY_MODULES.find(m => m.name === moduleName);
                        if (mod) {
                            // Load the module
                            loadModule(mod);
                            // Stop observing this element
                            observer.unobserve(element);
                            // Remove the lazy attribute
                            element.removeAttribute('data-lazy-module');
                        }
                    }
                }
            });
        }, {
            // Load when element is 200px before entering viewport
            rootMargin: '200px 0px',
            threshold: 0
        });

        // Observe all elements with data-lazy-module attribute
        document.querySelectorAll('[data-lazy-module]').forEach(el => {
            observer.observe(el);
        });
    } else {
        // Fallback: load all lazy modules after page is interactive
        const scheduleLoad = (fn) => {
            if ('requestIdleCallback' in window) {
                requestIdleCallback(fn, { timeout: 2000 });
            } else {
                setTimeout(fn, 100);
            }
        };

        LAZY_MODULES.forEach(mod => {
            const container = document.getElementById(mod.name);
            if (container) {
                scheduleLoad(() => loadModule(mod));
            }
        });
    }
}

// ── Lazy-load chatbot on user interaction ──

function initChatbotLazy() {
    const chatbotEl = document.getElementById('aiChatbot');
    if (!chatbotEl) return;

    // Create a minimal FAB button for lazy loading
    const fab = document.createElement('div');
    fab.className = 'chat-fab';
    fab.title = 'Chat with DistLLM AI';
    fab.innerHTML = `
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
        </svg>
    `;
    chatbotEl.appendChild(fab);

    // Load chatbot when user clicks the FAB
    const loadChatbot = async () => {
        try {
            const module = await CHATBOT_MODULE.loader();
            if (module[CHATBOT_MODULE.init]) {
                module[CHATBOT_MODULE.init]();
            }
        } catch (e) {
            console.warn('[DistLLM] Chatbot load failed:', e);
        }
        // Remove the temporary FAB after chatbot loads
        fab.remove();
    };

    fab.addEventListener('click', loadChatbot, { once: true });

    // Also load on keyboard shortcut (Ctrl+Shift+C)
    document.addEventListener('keydown', (e) => {
        if (e.ctrlKey && e.shiftKey && e.key === 'C') {
            loadChatbot();
        }
    }, { once: true });
}

// ── Initialize everything on DOM ready ──

document.addEventListener('DOMContentLoaded', () => {
    // Enable JS-dependent animations (progressive enhancement)
    document.documentElement.classList.add('js-enabled');

    // Load fonts early
    loadFonts();

    // Register service worker for PWA
    if ('serviceWorker' in navigator) {
        navigator.serviceWorker.register('/sw.js').catch(() => {});
    }

    // Safe init — each module wrapped in try/catch so one failure doesn't break others
    const safeInit = (name, fn) => {
        try { fn(); } catch (e) { console.warn(`[DistLLM] ${name} init failed:`, e); }
    };

    // CRITICAL: Load immediately
    safeInit('theme', initTheme);
    safeInit('scroll', initScroll);
    safeInit('mobileMenu', initMobileMenu);
    safeInit('codeTabs', initCodeTabs);
    safeInit('copyButtons', initCopyButtons);
    safeInit('faq', initFAQ);
    safeInit('cmdPalette', initCmdPalette);
    safeInit('githubStars', initGitHubStars);
    safeInit('newsletter', initNewsletter);
    safeInit('communityStats', initCommunityStats);
    safeInit('toast', () => { initToast(); checkNewVersion(); });

    // LAZY: Load when scrolled into view
    initLazyModules();

    // CHATBOT: Load on user interaction
    initChatbotLazy();

    // Platform detection for Cmd/Ctrl hint
    const cmdHint = document.getElementById('cmdKHint');
    if (cmdHint && navigator.platform) {
        const isMac = /Mac|iPod|iPhone|iPad/.test(navigator.platform);
        cmdHint.textContent = isMac ? '⌘K' : 'Ctrl+K';
    }
});
