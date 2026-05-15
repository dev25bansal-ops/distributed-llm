/* DistLLM UI JavaScript */

document.addEventListener('DOMContentLoaded', function() {
    // Auto-resize textarea
    const textarea = document.getElementById('chat-input');
    if (textarea) {
        textarea.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 150) + 'px';
        });
    }

    // Show loading indicator for HTMX requests
    document.body.addEventListener('htmx:beforeRequest', function() {
        console.log('Request started');
    });

    document.body.addEventListener('htmx:afterRequest', function() {
        console.log('Request completed');
    });
});
