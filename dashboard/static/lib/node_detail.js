// ===========================================================================
// lib/node_detail.js — the right-side node detail panel.
// ---------------------------------------------------------------------------
// Called from many tabs (Graph onNodeClick, Explorer card click, Live entries
// clicking a candidate, recursive open-correction clicks). Lives in lib/ —
// not a tab module — because it's a shared overlay, not tab-scoped.
//
// Sections (top → bottom):
//   1. Close button
//   2. Title (type badge + lock glyph + node title)
//   3. Meta strip (id, conf, access, source)
//   4. Timestamps (created / last accessed / revised, relative + absolute)
//   5. Content (n.content)
//   6. Fields (every non-empty top-level + metadata KV pair — deny-list,
//      not allow-list, so new keys surface automatically)
//   7. Encoded-from traces (source_refs, clickable → Traces tab)
//   8. Corrections (aspect-edge walk)
//   9. Connections (top 20 by edge weight)
//
// Renders via el() — auto-escaping, addEventListener wiring, no inline
// `onclick="..."` strings in rendered HTML.
// ===========================================================================

import { api } from './api.js';
import { el, html, mount, localTime, relativeTime } from './dom.js';
import { SCALE_COLORS } from './scales.js';

// ── Field rendering policy ────────────────────────────────────────────
// Fields rendered elsewhere (title, content, timestamps, the meta strip)
// don't appear in the Fields section. Everything else does — including
// metadata KV keys the encoder may add over time. New keys surface
// automatically, no curation needed.

const NODE_FIELDS_HIDDEN_IN_LIST = new Set([
  'id', 'type', 'title', 'content', 'locked', 'metadata',
  'created_at', 'last_accessed', 'revised_at',
  'access_count', 'confidence', 'encoding_source',
]);

// Pretty labels for known keys. Anything not listed renders with the raw
// snake_case key — never silently drops. Every field renders.
const FIELD_LABELS = {
  emotion:            'emotion',
  personal:           'personal',
  personal_context:   'personal context',
  evolution_status:   'evolution status',
  situation:          'situation',
  reasoning:          'reasoning',
  question:           'question',
  their_raw_quote:     'user raw quote',
  my_raw_quote:   'anchor raw quote',
  correction_of:      'correction_of',
  correction_pattern: 'correction pattern',
  source_context:     'source context',
};

function _labelFor(key) { return FIELD_LABELS[key] || key.replace(/_/g, ' '); }

// ── Section helpers (each returns an element or array of elements) ────

function _closeButton() {
  return el('div', {
    class: 'nd-close',
    onclick: () => { document.getElementById('node-detail').style.display = 'none'; },
  }, html('&times;'));
}

function _title(n) {
  return el('div', { class: 'nd-title' },
    el('span', { class: 'type-badge type--' + (n.type || '') }, n.type || ''),
    ' ',
    n.locked ? html('&#x1f512; ') : null,
    n.title || '',
  );
}

function _metaStrip(n) {
  return el('div', { class: 'nd-meta' },
    el('span', null, 'id: ' + (n.id || '').substring(0, 8)),
    el('span', null, 'accessed: ' + n.access_count + 'x'),
    el('span', null, 'conf: ' + (n.confidence || 0).toFixed(2)),
    el('span', null, 'source: ' + (n.encoding_source || '?')),
  );
}

// Three timestamp pills — created / last_accessed / revised — each with a
// big relative time and a small absolute below it. `title=` on the pill
// shows the full ISO so power users can copy-paste.
function _timestampsSection(n) {
  const pills = [];
  const addPill = (label, iso) => {
    if (!iso) return;
    pills.push(el('div', { class: 'nd-time-pill', title: iso },
      el('span', { class: 'nd-time-label' }, label),
      el('span', { class: 'nd-time-rel' }, relativeTime(iso) || '—'),
      el('span', { class: 'nd-time-abs' }, localTime(iso)),
    ));
  };
  addPill('CREATED',       n.created_at);
  addPill('LAST ACCESSED', n.last_accessed);
  addPill('REVISED',       n.revised_at);
  if (!pills.length) return null;
  return el('div', { class: 'nd-times' }, ...pills);
}

// Render a single field value — string, number, bool, null. Special case
// for correction_of: render as a recursive-open link.
function _fieldValue(key, value) {
  if (key === 'correction_of' && value) {
    return el('a', {
      class: 'nd-correction-of',
      onclick: () => loadNodeDetail(String(value)),
    }, String(value));
  }
  if (typeof value === 'string') {
    if (key === 'their_raw_quote' || key === 'my_raw_quote') {
      return el('em', null, '"' + value + '"');
    }
    return value;
  }
  // numbers, booleans → monospaced rendering so a sea of fields scans cleanly.
  return el('span', { class: 'nd-field-mono' }, String(value));
}

function _fieldsSection(n, meta) {
  const fields = [];
  const pushField = (key, value) => {
    if (value === null || value === undefined || value === '' || value === false) return;
    fields.push(el('div', { class: 'nd-field' },
      el('span', { class: 'nd-fk' }, _labelFor(key) + ':'),
      ' ',
      _fieldValue(key, value),
    ));
  };

  // 1. Top-level node fields (deny-list).
  for (const [k, v] of Object.entries(n)) {
    if (NODE_FIELDS_HIDDEN_IN_LIST.has(k)) continue;
    pushField(k, v);
  }
  // 2. Metadata KV pairs — same policy, never silently drop a key.
  for (const [k, v] of Object.entries(meta || {})) {
    // `situation` is sometimes promoted to top-level by the query — skip
    // the duplicate if we already rendered it from `n.situation`.
    if (k === 'situation' && n.situation) continue;
    pushField(k, v);
  }

  if (!fields.length) return [];
  return [el('div', { class: 'nd-section' }, 'Fields'), ...fields];
}

// Navigate to a specific trace event. Three steps:
//   1. Switch to the Traces tab
//   2. Apply the session filter (narrows the chain list)
//   3. Once loadTraces resolves, scroll the target trace into view and
//      flash-highlight it so the operator sees WHERE they landed.
//
// _flashTargetTrace handles the "render until found" case: if the trace
// isn't in the first batch of 30 chains, we call _loadMoreTraces() once
// before giving up. Most refs land in the first batch (newest-first),
// but very old nodes can sit further down.
async function _openTraceInTracesTab(ref) {
  const sessionId = ref && ref.session_id;
  const traceId = ref && ref.trace_id;
  if (!sessionId || typeof window.switchTab !== 'function') return;
  window.switchTab('traces');
  if (typeof window.loadTraces !== 'function') return;
  // Pass session as an explicit opts param — `sel.value = sessionId` no
  // longer works to force a filter, because the dropdown's <option> for
  // this session may not exist yet (loadTraces populates options AFTER
  // reading the current value). When the option is missing, setting
  // .value silently becomes '' and the filter is lost. opts.session
  // bypasses the dropdown read entirely; loadTraces then syncs the
  // dropdown back to match.
  try { await window.loadTraces({ session: sessionId }); }
  catch (_) { /* keep going to fallback */ }
  if (traceId) _flashTargetTrace(traceId);
}

// Trace-flash state. The Traces tab polls every 5s and rewrites
// #traces-content from scratch, which destroys any class we added. We
// stash the target trace_id at module scope and let traces.js re-apply
// the class after each render (see _reapplyFlashOnRender bound in
// node_detail.js init below). Cleared after the animation completes
// so subsequent polls don't keep flashing the same row forever.
let _pendingFlashTraceId = null;
let _pendingFlashUntil = 0;

function _flashTargetTrace(traceId) {
  _pendingFlashTraceId = traceId;
  _pendingFlashUntil = Date.now() + 2500;   // a bit beyond the 2.2s animation
  _applyFlashIfPending();
}

function _applyFlashIfPending() {
  if (!_pendingFlashTraceId) return;
  if (Date.now() > _pendingFlashUntil) {
    _pendingFlashTraceId = null;
    return;
  }
  // A rendered row can still be out of the DOM: traces.js collapses runs of
  // same-ref_type events and builds their members only when opened. So a miss
  // asks traces.js to open the run holding this id, then looks once more.
  const find = () => {
    const sel = '#traces-content .trace-event[data-trace-id="' + _pendingFlashTraceId + '"]';
    let hit = document.querySelector(sel);
    if (!hit && typeof window._revealTrace === 'function') {
      try {
        if (window._revealTrace(_pendingFlashTraceId)) hit = document.querySelector(sel);
      } catch (_) {}
    }
    return hit;
  };
  let el = find();
  // Not in the first batch? Click "Load more" once and retry. Two batches
  // cover ~60 chains, enough for nearly every realistic source-ref.
  if (!el && typeof window._loadMoreTraces === 'function') {
    try { window._loadMoreTraces(); } catch (_) {}
    el = find();
  }
  if (!el) {
    console.warn('[node-detail] trace not visible after Load More:', _pendingFlashTraceId);
    return;
  }
  el.scrollIntoView({ behavior: 'smooth', block: 'center' });
  // Re-trigger the CSS animation by toggling the class. Removing first
  // (in case the same event was already flashed in this session) and
  // re-adding on the next frame forces a restart.
  el.classList.remove('trace-event--target');
  // void offsetHeight forces a reflow so the second classList.add
  // registers as a new animation, not a continuation.
  void el.offsetHeight;
  el.classList.add('trace-event--target');
}

// Public: traces.js calls this after every render to re-apply the flash
// if one is still pending. Bound on window for the cross-module call
// to stay loosely coupled (no import needed in traces.js).
export function reapplyTraceFlashIfPending() {
  _applyFlashIfPending();
}

// Episodic refs — which traces this node was encoded from
// (node_source_refs). Each card is clickable → Traces tab + session filter.
function _sourceRefsSection(refs) {
  if (!refs.length) return [];
  // Source-refs are the most actionable thing in the panel — they
  // connect a node back to the conversation event that produced it.
  // The --source modifiers (vs the generic .nd-section / .nd-conn used
  // by other panels) bump visual weight: brighter section heading,
  // accent-tinted card background, the action link styled as a real
  // button-chip rather than the muted italic afterthought it was.
  const out = [el('div', { class: 'nd-section nd-section--source' },
    'Encoded from ' + refs.length + ' trace(s)')];
  for (const ref of refs) {
    if (ref.missing) {
      out.push(el('div', { class: 'nd-field nd-conn-missing' },
        'trace ' + (ref.trace_id || '') + ' (not found — log-rotated or archived)'));
      continue;
    }
    const sc = SCALE_COLORS[ref.scale] || '#666';
    const sess = ref.session_id ? ref.session_id.substring(0, 8) : '';
    const tagText = ref.scale + ' · ' + (ref.event_type || '')
                  + (ref.ref_type ? ' · ' + ref.ref_type : '')
                  + (sess ? ' · ' + sess : '');
    out.push(el('div', {
      class: 'nd-conn nd-conn--clickable nd-conn--source',
      style: { borderLeftColor: sc },
      onclick: () => _openTraceInTracesTab(ref),
    },
      el('div', { class: 'nd-conn-tag', style: { color: sc } }, tagText),
      el('div', { class: 'nd-conn-summary' }, (ref.summary || '').substring(0, 180)),
      el('div', { class: 'nd-conn-trace', title: ref.trace_id || '' },
        'trace ' + (ref.trace_id || '').substring(0, 8)
        + ' · pos ' + (ref.position || 1)
        + ' · ' + (ref.trace_created_at ? localTime(ref.trace_created_at) : '?')
        + (ref.trace_created_at ? ' (' + relativeTime(ref.trace_created_at) + ')' : '')),
      ref.chain_id
        ? el('div', { class: 'nd-conn-trace', title: ref.chain_id }, 'chain: ' + ref.chain_id)
        : null,
      el('div', { class: 'nd-conn-action' }, '→ Open in Traces tab'),
    ));
  }
  return out;
}

// Corrections — aspect-edge walk via daemon. `direction` is the neighbor's
// role (servers/brain_corrections.py): 'corrector' corrected this node, null
// a verb with no orientation. Nodes this one corrected are ordinary edges.
const _CORRECTION_DIR = {
  corrector: ['#ff8866', '⤴ corrects this'],
};
function _correctionsSection(corrections) {
  if (!corrections.length) return [];
  const out = [el('div', { class: 'nd-section' },
    'Corrections (' + corrections.length + ')')];
  for (const c of corrections) {
    const [dirColor, dirLabel] = _CORRECTION_DIR[c.direction] || ['#aaaaaa', '⚠ correction'];
    out.push(el('div', {
      class: 'nd-conn nd-conn--clickable',
      style: { borderLeftColor: dirColor },
      onclick: () => loadNodeDetail(c.id || ''),
    },
      el('div', { class: 'nd-conn-tag', style: { color: dirColor } },
        dirLabel + ' · ' + (c.relation || '')),
      el('div', { class: 'nd-conn-title' },
        el('span', { class: 'type-badge type--' + (c.type || '') }, c.type || ''),
        ' ',
        (c.title || '').substring(0, 80)),
      c.edge_description
        ? el('div', { class: 'nd-conn-edge-why' },
            'why: ' + c.edge_description.substring(0, 180))
        : null,
      c.content
        ? el('div', { class: 'nd-conn-content' }, c.content.substring(0, 200))
        : null,
      c.their_raw_quote
        ? el('div', { class: 'nd-conn-quote' },
            '"' + c.their_raw_quote.substring(0, 150) + '"')
        : null,
    ));
  }
  return out;
}

function _connectionsSection(conns) {
  const out = [el('div', { class: 'nd-section' }, 'Connections (' + conns.length + ')')];
  for (const c of conns) {
    out.push(el('div', {
      class: 'nd-conn nd-conn--clickable',
      onclick: () => loadNodeDetail(c.id),
    },
      el('div', { class: 'nd-conn-title' },
        el('span', { class: 'type-badge type--' + (c.type || '') }, c.type || ''),
        ' ',
        (c.title || '').substring(0, 60)),
      el('div', { class: 'nd-conn-meta' },
        (c.relation || '') + ' · weight ' + (c.weight || 0).toFixed(2)),
    ));
  }
  if (!conns.length) {
    out.push(el('div', { class: 'nd-conn-empty' }, 'No connections'));
  }
  return out;
}

// ── Public entry point ─────────────────────────────────────────────────

export async function loadNodeDetail(nodeId) {
  const panel = document.getElementById('node-detail');
  panel.style.display = 'block';
  mount(panel, el('div', { class: 'nd-loading' }, 'Loading...'));
  try {
    const [d, crData, srrData] = await Promise.all([
      api.node(nodeId),
      api.nodeCorrections(nodeId).catch(() => ({ corrections: [] })),
      api.nodeSourceRefs(nodeId).catch(() => ({ refs: [] })),
    ]);
    const corrections = (crData && crData.corrections) || [];
    const sourceRefs  = (srrData && srrData.refs) || [];
    const n           = d.node;
    const conns       = d.connections || [];
    const meta        = n.metadata || {};

    mount(panel,
      _closeButton(),
      _title(n),
      _metaStrip(n),
      _timestampsSection(n),
      el('div', { class: 'nd-section' }, 'Content'),
      el('div', { class: 'nd-content' }, n.content || '(empty)'),
      _fieldsSection(n, meta),
      _sourceRefsSection(sourceRefs),
      _correctionsSection(corrections),
      _connectionsSection(conns),
    );
  } catch(e) {
    mount(panel, el('div', { class: 'feed-empty feed-empty--error' },
      'Failed to load: ' + (e && e.message ? e.message : String(e))));
  }
}
