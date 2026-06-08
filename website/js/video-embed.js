/**
 * Video Embed — lazy-loaded YouTube/custom video player.
 *
 * Usage:
 *   <div class="video-embed" data-src="https://youtube.com/watch?v=xxx"></div>
 *   <script type="module">
 *     import { initVideoEmbeds } from './js/video-embed.js';
 *     initVideoEmbeds();
 *   </script>
 */

export function initVideoEmbeds() {
    document.querySelectorAll('.video-embed').forEach(el => {
        const src = el.dataset.src;
        if (!src) return;

        // Extract YouTube ID
        const ytMatch = src.match(/(?:youtube\.com\/watch\?v=|youtu\.be\/)([a-zA-Z0-9_-]+)/);
        const ytId = ytMatch?.[1];

        if (ytId) {
            // Lazy load YouTube iframe
            const wrapper = document.createElement('div');
            wrapper.style.cssText = 'position:relative;padding-bottom:56.25%;height:0;overflow:hidden;border-radius:12px;background:#111;';
            
            const placeholder = document.createElement('div');
            placeholder.style.cssText = 'position:absolute;inset:0;display:flex;align-items:center;justify-content:center;cursor:pointer;background:#111;border:1px solid #222;border-radius:12px;';
            placeholder.innerHTML = `
                <img src="https://img.youtube.com/vi/${ytId}/hqdefault.jpg" 
                     style="width:100%;height:100%;object-fit:cover;border-radius:12px;" 
                     alt="Video thumbnail"
                     loading="lazy">
                <div style="position:absolute;width:64px;height:64px;background:rgba(0,0,0,0.7);border-radius:50%;display:flex;align-items:center;justify-content:center;">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="white"><polygon points="8,5 19,12 8,19"/></svg>
                </div>
            `;

            placeholder.addEventListener('click', () => {
                const iframe = document.createElement('iframe');
                iframe.src = `https://www.youtube-nocookie.com/embed/${ytId}?autoplay=1&rel=0`;
                iframe.style.cssText = 'position:absolute;top:0;left:0;width:100%;height:100%;border:none;';
                iframe.allow = 'accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture';
                iframe.allowFullscreen = true;
                wrapper.innerHTML = '';
                wrapper.appendChild(iframe);
            });

            wrapper.appendChild(placeholder);
            el.appendChild(wrapper);
        }
    });
}
