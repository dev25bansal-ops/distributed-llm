/**
 * Scroll progress bar, active nav highlighting, fade-up animations.
 * Combines all scroll-related logic into a single handler for performance.
 */
export function initScroll() {
    const progress = document.getElementById('scrollProgress');
    const nav = document.querySelector('.nav');

    // Cache scrollHeight, recalculate on resize
    let cachedHeight = document.documentElement.scrollHeight - window.innerHeight;
    window.addEventListener('resize', () => {
        cachedHeight = document.documentElement.scrollHeight - window.innerHeight;
    }, { passive: true });

    // Single scroll handler with rAF batching
    if (progress || nav) {
        let ticking = false;
        window.addEventListener('scroll', () => {
            if (!ticking) {
                requestAnimationFrame(() => {
                    const y = window.scrollY;

                    // Progress bar
                    if (progress && cachedHeight > 0) {
                        progress.style.width = (y / cachedHeight * 100) + '%';
                    }

                    // Nav shadow
                    if (nav) {
                        nav.classList.toggle('scrolled', y > 20);
                    }

                    ticking = false;
                });
                ticking = true;
            }
        }, { passive: true });
    }

    // Active nav highlighting
    const sections = document.querySelectorAll('section[id]');
    const navItems = document.querySelectorAll('.nav-links a[data-section]');
    if (sections.length && navItems.length) {
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const id = entry.target.id;
                    navItems.forEach(a => a.classList.toggle('active', a.dataset.section === id));
                }
            });
        }, { threshold: 0.3, rootMargin: '-80px 0px -60% 0px' });
        sections.forEach(s => observer.observe(s));
    }

    // Fade-up on scroll
    const fadeObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
                fadeObserver.unobserve(entry.target);
            }
        });
    }, { threshold: 0.1, rootMargin: '0px 0px -40px 0px' });
    document.querySelectorAll('.fade-up').forEach(el => fadeObserver.observe(el));

    // Pause off-screen CSS animations (architecture particles, CTA orbs)
    const animatedSections = document.querySelectorAll('.arch-diagram, .cta-section');
    if (animatedSections.length) {
        const animObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                const state = entry.isIntersecting ? 'running' : 'paused';
                entry.target.style.setProperty('--anim-state', state);
            });
        }, { threshold: 0 });
        animatedSections.forEach(s => animObserver.observe(s));
    }

    // Smooth scroll for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            const selector = this.getAttribute('href');
            if (!selector || selector.length <= 1) return;

            const target = document.querySelector(selector);
            if (target) {
                e.preventDefault();
                target.scrollIntoView({ behavior: 'smooth' });
            }
        });
    });
}
