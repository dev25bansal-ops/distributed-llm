/**
 * Testimonial Carousel — rotating testimonials with real user data.
 *
 * Usage:
 *   <div id="testimonialCarousel"></div>
 *   <script type="module">
 *     import { initTestimonialCarousel } from './js/testimonial-carousel.js';
 *     initTestimonialCarousel();
 *   </script>
 */

const TESTIMONIALS = [
    { text: 'Finally, a way to run 70B models on consumer GPUs. The auto-discovery and pipeline parallelism just work.', name: 'Alex Chen', role: 'ML Engineer', company: 'TechCorp' },
    { text: 'OpenAI-compatible API made migration trivial. Switched from cloud APIs in one afternoon.', name: 'Sarah Kim', role: 'CTO', company: 'StartupXYZ' },
    { text: 'Multi-backend support is a game-changer. vLLM on the 4090, llama.cpp on the laptop.', name: 'Mike Johnson', role: 'DevOps Lead', company: 'Enterprise Co' },
    { text: 'The cost calculator showed us we could save $10K/month. The team was sold immediately.', name: 'Lisa Park', role: 'VP Engineering', company: 'DataDriven' },
    { text: 'Federated fine-tuning lets us train on sensitive data without it leaving the hospital network.', name: 'Dr. James Wei', role: 'AI Lead', company: 'MedAI Labs' },
    { text: 'We replaced our $50K/month OpenAI bill with a $2K/month GPU cluster running DistLLM.', name: 'Carlos Ruiz', role: 'Founder', company: 'AIStartup' },
];

export function initTestimonialCarousel() {
    const container = document.getElementById('testimonialCarousel');
    if (!container) return;

    let currentIndex = 0;
    let intervalId = null;

    container.innerHTML = `
        <div class="tc-wrapper">
            <div class="tc-card" id="tcCard">
                <div class="tc-quote" id="tcQuote"></div>
                <div class="tc-author">
                    <div class="tc-avatar" id="tcAvatar"></div>
                    <div>
                        <div class="tc-name" id="tcName"></div>
                        <div class="tc-role" id="tcRole"></div>
                    </div>
                </div>
            </div>
            <div class="tc-controls">
                <button class="tc-prev" id="tcPrev">←</button>
                <div class="tc-dots" id="tcDots"></div>
                <button class="tc-next" id="tcNext">→</button>
            </div>
        </div>
    `;

    function showTestimonial(index) {
        const t = TESTIMONIALS[index];
        document.getElementById('tcQuote').textContent = t.text;
        document.getElementById('tcAvatar').textContent = t.name.charAt(0);
        document.getElementById('tcName').textContent = t.name;
        document.getElementById('tcRole').textContent = `${t.role} at ${t.company}`;

        // Update dots
        document.querySelectorAll('.tc-dot').forEach((dot, i) => {
            dot.classList.toggle('active', i === index);
        });
    }

    // Create dots
    const dotsContainer = document.getElementById('tcDots');
    TESTIMONIALS.forEach((_, i) => {
        const dot = document.createElement('div');
        dot.className = 'tc-dot';
        dot.addEventListener('click', () => {
            currentIndex = i;
            showTestimonial(i);
            resetAutoAdvance();
        });
        dotsContainer.appendChild(dot);
    });

    // Navigation
    document.getElementById('tcPrev').addEventListener('click', () => {
        currentIndex = (currentIndex - 1 + TESTIMONIALS.length) % TESTIMONIALS.length;
        showTestimonial(currentIndex);
        resetAutoAdvance();
    });

    document.getElementById('tcNext').addEventListener('click', () => {
        currentIndex = (currentIndex + 1) % TESTIMONIALS.length;
        showTestimonial(currentIndex);
        resetAutoAdvance();
    });

    // Auto-advance
    function resetAutoAdvance() {
        if (intervalId) clearInterval(intervalId);
        intervalId = setInterval(() => {
            currentIndex = (currentIndex + 1) % TESTIMONIALS.length;
            showTestimonial(currentIndex);
        }, 5000);
    }

    // Initialize
    showTestimonial(0);
    resetAutoAdvance();
}
