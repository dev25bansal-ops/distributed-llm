/**
 * Video Tutorial Library — embedded YouTube tutorials with search, filter, and modal.
 *
 * Usage:
 *   <div id="videoTutorials"></div>
 *   <script type="module">
 *     import { initVideoTutorials } from './js/video-tutorials.js';
 *     initVideoTutorials();
 *   </script>
 */

import { escapeHtml } from './utils.js';

const TUTORIALS = [
  { id: 'setup', title: 'Install DistLLM in 5 Minutes', duration: '5:23', difficulty: 'Beginner', desc: 'Step-by-step installation guide covering pip and Docker setup across all platforms.', youtube: 'https://www.youtube.com/watch?v=placeholder1' },
  { id: 'cluster', title: 'Multi-Node Cluster Setup', duration: '8:45', difficulty: 'Intermediate', desc: 'Connect multiple machines on your LAN to pool GPUs for larger models like Llama 3.1 70B.', youtube: 'https://www.youtube.com/watch?v=placeholder2' },
  { id: 'models', title: 'Running Models & Quantization', duration: '6:12', difficulty: 'Beginner', desc: 'How to load HuggingFace models, configure INT4/INT8 quantization, and optimize VRAM usage.', youtube: 'https://www.youtube.com/watch?v=placeholder3' },
  { id: 'api', title: 'OpenAI-Compatible API', duration: '7:30', difficulty: 'Intermediate', desc: 'Integrate DistLLM with LangChain, LlamaIndex, and any OpenAI SDK by changing just the base URL.', youtube: 'https://www.youtube.com/watch?v=placeholder4' },
  { id: 'advanced', title: 'Speculative Decoding & Tuning', duration: '10:15', difficulty: 'Advanced', desc: 'Optimize throughput 2-3x with speculative decoding, pipeline overlap, and KV cache tuning.', youtube: 'https://www.youtube.com/watch?v=placeholder5' },
  { id: 'troubleshoot', title: 'Common Issues & Debugging', duration: '9:00', difficulty: 'Intermediate', desc: 'Fix OOM errors, connection timeouts, GPU detection issues, and performance bottlenecks.', youtube: 'https://www.youtube.com/watch?v=placeholder6' },
];

const DIFFICULTIES = ['All', 'Beginner', 'Intermediate', 'Advanced'];
const COLORS = { Beginner: '#22c55e', Intermediate: '#f59e0b', Advanced: '#ef4444' };

export function initVideoTutorials() {
  const container = document.getElementById('videoTutorials');
  if (!container) return;

  let filter = 'All';
  let searchQuery = '';
  let bookmarks = JSON.parse(localStorage.getItem('distllm-bookmarks') || '[]');

  function render() {
    container.innerHTML = '';

    const header = document.createElement('div');
    header.className = 'vt-header';

    const h3 = document.createElement('h3');
    h3.textContent = 'Video Tutorials';
    header.appendChild(h3);
    container.appendChild(header);

    // Filters
    const filters = document.createElement('div');
    filters.className = 'vt-filters';

    const searchInput = document.createElement('input');
    searchInput.className = 'vt-search';
    searchInput.type = 'text';
    searchInput.placeholder = 'Search tutorials...';
    searchInput.value = searchQuery;
    searchInput.addEventListener('input', () => { searchQuery = searchInput.value; render(); });
    filters.appendChild(searchInput);

    const diffBar = document.createElement('div');
    diffBar.className = 'vt-diff-bar';
    for (const d of DIFFICULTIES) {
      const btn = document.createElement('button');
      btn.className = 'vt-diff-btn' + (filter === d ? ' active' : '');
      btn.textContent = d;
      btn.addEventListener('click', () => { filter = d; render(); });
      diffBar.appendChild(btn);
    }
    filters.appendChild(diffBar);
    container.appendChild(filters);

    // Grid
    const grid = document.createElement('div');
    grid.className = 'vt-grid';

    const filtered = TUTORIALS.filter(t => {
      if (filter !== 'All' && t.difficulty !== filter) return false;
      if (searchQuery && !t.title.toLowerCase().includes(searchQuery.toLowerCase())) return false;
      return true;
    });

    for (const t of filtered) {
      const card = document.createElement('div');
      card.className = 'vt-card';

      // Thumbnail placeholder
      const thumb = document.createElement('div');
      thumb.className = 'vt-thumb';
      thumb.innerHTML = `<div class="vt-play-icon">▶</div><span class="vt-duration">${t.duration}</span>`;
      card.appendChild(thumb);

      // Info
      const info = document.createElement('div');
      info.className = 'vt-info';

      const title = document.createElement('h4');
      title.textContent = t.title;
      info.appendChild(title);

      const meta = document.createElement('div');
      meta.className = 'vt-meta';

      const badge = document.createElement('span');
      badge.className = 'vt-badge';
      badge.style.background = COLORS[t.difficulty] + '22';
      badge.style.color = COLORS[t.difficulty];
      badge.textContent = t.difficulty;
      meta.appendChild(badge);

      const dur = document.createElement('span');
      dur.className = 'vt-dur';
      dur.textContent = t.duration;
      meta.appendChild(dur);
      info.appendChild(meta);

      const desc = document.createElement('p');
      desc.className = 'vt-desc';
      desc.textContent = t.desc;
      info.appendChild(desc);

      card.appendChild(info);

      // Actions
      const actions = document.createElement('div');
      actions.className = 'vt-actions';

      const watchBtn = document.createElement('button');
      watchBtn.className = 'vt-watch-btn';
      watchBtn.textContent = '▶ Watch';
      watchBtn.addEventListener('click', () => openModal(t));
      actions.appendChild(watchBtn);

      const bmBtn = document.createElement('button');
      const isBookmarked = bookmarks.includes(t.id);
      bmBtn.className = 'vt-bm-btn' + (isBookmarked ? ' bookmarked' : '');
      bmBtn.textContent = isBookmarked ? '★' : '☆';
      bmBtn.title = isBookmarked ? 'Remove bookmark' : 'Bookmark for later';
      bmBtn.addEventListener('click', () => {
        if (bookmarks.includes(t.id)) {
          bookmarks = bookmarks.filter(b => b !== t.id);
        } else {
          bookmarks.push(t.id);
          if (bookmarks.length > 20) bookmarks.shift();
        }
        localStorage.setItem('distllm-bookmarks', JSON.stringify(bookmarks));
        render();
      });
      actions.appendChild(bmBtn);

      card.appendChild(actions);
      grid.appendChild(card);
    }

    if (filtered.length === 0) {
      grid.innerHTML = '<div class="vt-empty">No tutorials match your search. Try different keywords.</div>';
    }

    container.appendChild(grid);
  }

  function openModal(tutorial) {
    const overlay = document.createElement('div');
    overlay.className = 'vt-modal-overlay';
    overlay.addEventListener('click', (e) => { if (e.target === overlay) closeModal(); });

    const modal = document.createElement('div');
    modal.className = 'vt-modal';

    const closeBtn = document.createElement('button');
    closeBtn.className = 'vt-close';
    closeBtn.innerHTML = '✕';
    closeBtn.setAttribute('aria-label', 'Close');
    closeBtn.addEventListener('click', closeModal);
    modal.appendChild(closeBtn);

    const title = document.createElement('h3');
    title.textContent = tutorial.title;
    modal.appendChild(title);

    const player = document.createElement('div');
    player.className = 'vt-player';
    player.innerHTML = `
      <div class="vt-player-placeholder">
        <div class="vt-big-play">▶</div>
        <p>${escapeHtml(tutorial.title)}</p>
        <p class="vt-player-hint">Video would load here: ${escapeHtml(tutorial.youtube)}</p>
      </div>
    `;
    modal.appendChild(player);

    const desc = document.createElement('p');
    desc.className = 'vt-modal-desc';
    desc.textContent = tutorial.desc;
    modal.appendChild(desc);

    overlay.appendChild(modal);
    document.body.appendChild(overlay);
    overlay.focus();

    function closeModal() { overlay.remove(); }
  }

  render();
}
