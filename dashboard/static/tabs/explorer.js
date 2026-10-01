// ===========================================================================
// tabs/explorer.js — node search + browse.
// ---------------------------------------------------------------------------
// Click a node-card → opens the right-side detail panel via lib/node_detail.js.
// The type-filter dropdown is populated by app.js's loadStats() (cross-tab
// concern), this module only reads the dropdown's current value on search.
// ===========================================================================

import { api } from '/static/lib/api.js';
import { el, localTime } from '/static/lib/dom.js';
import { loadNodeDetail } from '/static/lib/node_detail.js';

export async function searchNodes() {
  const search = document.getElementById('search-input').value;
  const type = document.getElementById('type-filter').value;
  try {
    const d = await api.nodes({ limit: 100, search: search || undefined, type: type || undefined });
    document.getElementById('node-list').replaceChildren(...d.nodes.map(n =>
      el('div', { class: 'node-card', onclick: () => loadNodeDetail(n.id) },
        el('div', { class: 'node-title' },
          el('span', { class: 'type-badge type--' + (n.type || '') }, n.type || ''),
          n.locked ? el('span', { class: 'locked-icon' }, '\u{1f512}') : null,
          ' ', n.title || '(untitled)'),
        el('div', { class: 'node-meta' },
          el('span', null, 'conf: ' + (n.confidence || 0).toFixed(2)),
          el('span', null, 'accessed: ' + n.access_count + 'x'),
          el('span', null, n.encoding_source || ''),
          el('span', null, localTime(n.created_at))))));
  } catch(e) { console.error('[dashboard] searchNodes failed:', e); }
}

// ── Lifecycle ─────────────────────────────────────────────────────────

export function init() {
  // No polling — explorer is search-on-demand. The dropdown is wired via
  // index.html's onkeyup/onchange handlers calling window.searchNodes.
}

export function activate() {
  searchNodes();
}

export function deactivate() {}
