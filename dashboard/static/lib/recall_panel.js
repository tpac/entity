// ===========================================================================
// lib/recall_panel.js — ask the brain, and read the answer.
// ---------------------------------------------------------------------------
// The graph's search box did substring matching over titles already in the
// browser. Useful, but it answers a question the brain never asks. Enter now
// runs the REAL recall pipeline through the daemon — LAF scoring, graph
// expansion, the same ranking a live session gets — and shows what came back
// as a ranked list with scores and discovery lanes, not as dots that light up
// and leave you guessing which ones.
//
// Two modes on one control, which is why they don't fight:
//   typing → local filter (instant, dims non-matching nodes)
//   Enter  → recall probe (network, ranked panel + graph highlight)
//
// The probe is read-only by construction: the server sends mark_accessed=False,
// so the operator's searching never becomes part of the brain's own record of
// what it recalled. Nothing here writes.
// ===========================================================================

import { api } from '/static/lib/api.js';
import { el, relativeTime } from '/static/lib/dom.js';
import { loadNodeDetail } from '/static/lib/node_detail.js';
import * as graph from '/static/tabs/graph.js';

// Discovery lane → how this node was found. Recall has one lane (laf_v1);
// any other tag renders raw below, which makes a regression visible.
const LANE = {
  laf_v1: { label: 'laf', title: 'learned field weighting — the recall ranker' },
};

let _panel = null, _list = null, _meta = null, _busy = false;

function _ensurePanel() {
  if (_panel) return _panel;
  // Mounted on .graph-container, NOT #graph-3d: buildScaffold() clears
  // #graph-3d's innerHTML on every graph load, which would silently delete
  // this panel the first time the operator hit Refresh or switched shape.
  const host = document.querySelector('.graph-container');
  if (!host) return null;
  _meta = el('div', { class: 'recall-panel-meta' });
  _list = el('div', { class: 'recall-panel-list' });
  const close = el('button', { class: 'recall-panel-close', title: 'Close (clears the highlight)' }, '×');
  close.addEventListener('click', () => closeRecallPanel());
  _panel = el('div', { class: 'recall-panel' },
    el('div', { class: 'recall-panel-head' },
      el('span', { class: 'recall-panel-title' }, 'Recall'),
      _meta,
      close,
    ),
    _list,
  );
  // Inside the graph container so it overlays the galaxy it is describing —
  // the list and the lit nodes are the same answer, and separating them across
  // the page would make you look twice.
  host.appendChild(_panel);
  return _panel;
}

export function closeRecallPanel() {
  if (_panel) _panel.classList.remove('open');
  // Drop the spotlight, keep the layout — closing the panel shouldn't cost a
  // graph reload.
  graph.clearHighlight();
}

function _scoreBar(score) {
  const v = Math.max(0, Math.min(1, Number(score) || 0));
  return el('span', { class: 'recall-score', title: 'blended score ' + v.toFixed(3) },
    el('span', { class: 'recall-score-fill', style: { width: Math.round(v * 100) + '%' } }));
}

function _row(n, rank) {
  const lane = LANE[n.discovery] || (n.discovery ? { label: n.discovery, title: n.discovery } : null);
  const row = el('div', {
    class: 'recall-row',
    onclick: () => loadNodeDetail(n.id),
    title: 'Open ' + n.id,
  },
    el('div', { class: 'recall-row-head' },
      el('span', { class: 'recall-rank' }, String(rank)),
      n.type ? el('span', { class: 'type-badge type--' + n.type }, n.type) : null,
      el('span', { class: 'recall-title' }, n.title || n.id),
      lane ? el('span', { class: 'recall-lane', title: lane.title }, lane.label) : null,
      _scoreBar(n.score),
    ),
    // Situation is the field that says WHEN this memory is relevant — for a
    // person judging whether recall picked well, it's worth more than content.
    n.situation ? el('div', { class: 'recall-situation' }, n.situation) : null,
    n.content ? el('div', { class: 'recall-content' }, n.content) : null,
    el('div', { class: 'recall-row-foot' },
      n.created_at ? el('span', null, 'created ' + relativeTime(n.created_at)) : null,
      n.access_count ? el('span', null, 'recalled ' + n.access_count + '×') : null,
      n.confidence != null ? el('span', null, 'conf ' + Number(n.confidence).toFixed(2)) : null,
    ),
  );
  // Hovering a row isolates that one node in the galaxy — the list and the
  // picture stay tied together, so "which dot is that" is never a question.
  row.addEventListener('mouseenter', () =>
    graph.previewRecallOnGraph({ id: 'probe-hover', used_ids: [n.id], returned_ids: [n.id] }));
  row.addEventListener('mouseleave', () => graph.clearRecallPreview());
  return row;
}

/** Run the probe. Called on Enter in the graph search box. */
export async function runRecallProbe(query) {
  const q = (query || '').trim();
  if (!q || _busy) return;
  if (!_ensurePanel()) return;
  _busy = true;
  _panel.classList.add('open');
  _meta.textContent = 'asking…';
  _list.replaceChildren(el('div', { class: 'recall-panel-empty' }, 'Running the recall pipeline…'));
  try {
    const d = await api.recall({ query: q, limit: 14 });
    const results = d.results || [];
    // .filter(Boolean) — replaceChildren renders a null child as the literal
    // text "null" (el() filters, this does not).
    _meta.replaceChildren(...[
      el('span', { class: 'recall-meta-q' }, '\u201c' + q + '\u201d'),
      el('span', null, results.length + ' of ' + (d.candidates ?? '?') + ' candidates'),
      d.recall_ms != null ? el('span', null, Math.round(d.recall_ms) + 'ms') : null,
      d.mode ? el('span', { class: 'recall-meta-mode', title: 'ranking mode' }, d.mode) : null,
      el('span', { class: 'recall-meta-readonly',
        title: 'This probe runs with mark_accessed=False — it does not touch '
             + 'access counts, fatigue, or recall heat' }, 'read-only'),
    ].filter(Boolean));
    if (!results.length) {
      _list.replaceChildren(el('div', { class: 'recall-panel-empty' },
        'Nothing surfaced. That is an answer too — the brain has no memory it '
        + 'considers relevant to this.'));
      return;
    }
    _list.replaceChildren(...results.map((n, i) => _row(n, i + 1)));
    // Light exactly what came back. Top third are the "used" tier (the ones a
    // surfacer would most likely pick), the rest are returned candidates —
    // reusing the same highlight tiers a real recall paints, so the galaxy
    // means the same thing whether the recall came from a session or from here.
    const ids = results.map(n => n.id).filter(Boolean);
    const top = ids.slice(0, Math.max(1, Math.ceil(ids.length / 3)));
    graph.pinRecallToGraph({ id: 'probe:' + q, used_ids: top, returned_ids: ids, activation_ids: [] });
  } catch (e) {
    console.error('[recall-panel] probe failed:', e);
    _meta.textContent = '';
    _list.replaceChildren(el('div', { class: 'recall-panel-empty recall-panel-error' },
      String(e && e.message || e)));
  } finally {
    _busy = false;
  }
}

export default { runRecallProbe, closeRecallPanel };
