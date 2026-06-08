/**
 * Feedback Widget — "Was this page helpful?" on docs pages.
 *
 * Usage:
 *   <div id="feedbackWidget"></div>
 *   <script type="module">
 *     import { initFeedback } from './js/feedback.js';
 *     initFeedback();
 *   </script>
 */

export function initFeedback() {
    const container = document.getElementById('feedbackWidget');
    if (!container) return;

    const page = window.location.pathname;
    const storageKey = `distllm-feedback-${page}`;

    // Check if already submitted
    if (localStorage.getItem(storageKey)) {
        container.innerHTML = '<p style="font-size:12px;color:#555;">Thanks for your feedback!</p>';
        return;
    }

    container.innerHTML = `
        <div class="feedback-widget">
            <p class="feedback-question">Was this page helpful?</p>
            <div class="feedback-buttons">
                <button class="fb-btn fb-yes" data-v="yes">👍 Yes</button>
                <button class="fb-btn fb-no" data-v="no">👎 No</button>
            </div>
            <div class="feedback-comment" id="fbComment" style="display:none;">
                <textarea id="fbText" rows="2" placeholder="How can we improve? (optional)"></textarea>
                <button class="fb-submit" id="fbSubmit">Submit</button>
            </div>
            <p class="feedback-thanks" id="fbThanks" style="display:none;">Thanks! Your feedback helps us improve.</p>
        </div>
    `;

    // Button handlers
    container.querySelectorAll('.fb-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const val = btn.dataset.v;
            localStorage.setItem(storageKey, val);

            if (val === 'no') {
                // Show comment box
                document.getElementById('fbComment').style.display = 'flex';
            } else {
                // Show thanks
                container.innerHTML = '<p class="feedback-thanks" style="font-size:13px;color:#22c55e;text-align:center;padding:12px;">Thanks! Your feedback helps us improve.</p>';
            }
        });
    });

    // Submit handler
    const submitBtn = document.getElementById('fbSubmit');
    if (submitBtn) {
        submitBtn.addEventListener('click', () => {
            const text = document.getElementById('fbText')?.value || '';
            localStorage.setItem(storageKey, 'no:' + text);
            container.innerHTML = '<p style="font-size:13px;color:#22c55e;text-align:center;padding:12px;">Thanks! We\'ll use your feedback to improve.</p>';
        });
    }
}
