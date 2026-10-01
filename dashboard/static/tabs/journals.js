// ===========================================================================
// tabs/journals.js — the encoders' inner voice, in one place.
// ---------------------------------------------------------------------------
// Every encoder writes residue: what it noticed that its actions don't record.
// Per-run, that residue is a chip on a card. Across runs it is something else —
// the only place the brain says what it is confused about, what keeps coming
// back, and what it cannot resolve without a person.
//
// Which is the point. A journal note tagged `open` and re-flagged across a
// dozen runs is the encoder asking a question with nowhere to send it
// (the S2 consolidation journal is a closed loop —
// "operator confirmation still needed", written over and over, read by nobody).
// This view is the route out. Standing items lead; everything else follows.
//
// It reads the same rows the run cards read, through the same one reader
// (queries/journals.py) and renders with the same one renderer
// (lib/journal.js) — this tab adds a lens, not a second implementation.
// ===========================================================================

import { api } from '/static/lib/api.js';
import { poll } from '/static/lib/poll.js';
import { el, localTime } from '/static/lib/dom.js';
import { loadNodeDetail } from '/static/lib/node_detail.js';
import { journalRow, unitLabel, isStanding } from '/static/lib/journal.js';
import { sessionChip } from '/static/lib/sessions.js';

let _hours = 168;
let _unit = '';
let _tag = '';
let _query = '';

// A subject is free text — a node id, a cluster label, a tool name. When it
// LOOKS like a node id (8 hex chars, the brain's id format), make it open the
// node; otherwise it's just a label.
const _NODE_ID = /^[0-9a-f]{8}$/;
function _onSubject(note) {
  const first = (note.subject || '').trim().split(/\s+/)[0] || '';
  if (_NODE_ID.test(first)) loadNodeDetail(first);
}

function _matchesQuery(n) {
  if (!_query) return true;
  const q = _query.toLowerCase();
  return (n.note || '').toLowerCase().includes(q)
      || (n.subject || '').toLowerCase().includes(q);
}

export async function loadJournals() {
  const feed = document.getElementById('feed-journals');
  if (!feed) return;
  try {
    const [d, summary] = await Promise.all([
      api.journals({ hours: _hours, unit: _unit, tag: _tag, limit: 400 }),
      api.journalSummary({ hours: _hours }),
    ]);
    const notes = (d.notes || []).filter(_matchesQuery);
    _renderFilters(summary);
    _renderStanding(summary.standing || []);

    const count = document.getElementById('journals-count');
    if (count) count.textContent = notes.length + ' notes';

    if (!notes.length) {
      feed.replaceChildren(el('div', { class: 'feed-empty' },
        'No notes match. A clean encoder writes nothing — an empty fence is a '
        + 'valid review.'));
      return;
    }
    // Grouped by run: notes from one run are one thought, and reading them
    // interleaved with another encoder's would scramble both.
    const byChain = new Map();
    for (const n of notes) {
      if (!byChain.has(n.chain_id)) byChain.set(n.chain_id, []);
      byChain.get(n.chain_id).push(n);
    }
    const groups = [...byChain.entries()].sort(
      (a, b) => (b[1][0].created_at || '').localeCompare(a[1][0].created_at || ''));
    feed.replaceChildren(...groups.map(([chain, rows]) => el('div', { class: 'journal-group' },
      el('div', { class: 'journal-group-head' },
        el('span', { class: 'journal-group-unit' }, unitLabel(rows[0].unit)),
        el('span', { class: 'journal-group-time', title: rows[0].created_at },
          localTime(rows[0].created_at)),
        rows[0].session_id ? sessionChip(rows[0].session_id, { compact: true }) : null,
        el('span', { class: 'journal-group-chain', title: chain }, chain),
        el('span', { class: 'journal-group-count' },
          rows.length + (rows.length === 1 ? ' note' : ' notes')),
      ),
      ...rows.map(n => journalRow(n, { showUnit: false, onSubject: _onSubject })),
    )));
  } catch (e) {
    console.error('[journals] load failed:', e);
  }
}

// Standing items — the encoder asked N times and nothing answered. This block
// is why the tab exists, so it sits above everything and never scrolls away
// behind a filter.
function _renderStanding(standing) {
  const box = document.getElementById('journals-standing');
  if (!box) return;
  const live = (standing || []).filter(isStanding);
  if (!live.length) { box.replaceChildren(); return; }
  box.replaceChildren(
    el('div', { class: 'journals-standing-head' },
      el('span', { class: 'journal-standing-flag' }, 'needs you'),
      el('span', null, live.length + ' item' + (live.length === 1 ? '' : 's')
        + ' the encoders have re-raised without resolution'),
    ),
    ...live.map(n => journalRow(n, { showUnit: true, onSubject: _onSubject })),
  );
}

function _renderFilters(summary) {
  const bar = document.getElementById('journals-filters');
  if (!bar) return;
  const units = summary.units || {};
  const tags = summary.tags || {};
  const pills = [];

  const mk = (label, count, active, onclick, cls) => el('button', {
    class: ['journal-filter', active && 'journal-filter--active', cls].filter(Boolean),
    onclick,
  }, label, count != null ? el('span', { class: 'journal-filter-count' }, String(count)) : null);

  pills.push(mk('all encoders', summary.total ?? null, !_unit,
    () => { _unit = ''; loadJournals(); }));
  for (const [u, c] of Object.entries(units).sort((a, b) => b[1] - a[1])) {
    pills.push(mk(unitLabel(u), c, _unit === u,
      () => { _unit = (_unit === u ? '' : u); loadJournals(); }));
  }
  // Tags render second, visually separated — they're a different axis (what
  // KIND of note) from the encoder axis (who wrote it).
  pills.push(el('span', { class: 'journal-filter-sep' }));
  for (const [t, c] of Object.entries(tags).sort((a, b) => b[1] - a[1]).slice(0, 8)) {
    pills.push(mk(t, c, _tag === t,
      () => { _tag = (_tag === t ? '' : t); loadJournals(); }, 'journal-filter--tag'));
  }
  bar.replaceChildren(...pills);
}

export function onJournalsHoursChange() {
  const sel = document.getElementById('journals-hours');
  _hours = parseInt(sel ? sel.value : '168', 10) || 168;
  loadJournals();
}

export function onJournalsSearch() {
  const input = document.getElementById('journals-search');
  _query = (input ? input.value : '').trim();
  loadJournals();
}

export function init() {
  poll.register({
    key: 'journals',
    interval: 30000,
    activeWhen: () => document.getElementById('tab-journals').classList.contains('active'),
    fetcher: loadJournals,
  });
}

export function activate() { loadJournals(); }
export function deactivate() { /* poll self-gates */ }
