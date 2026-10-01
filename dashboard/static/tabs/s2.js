// ===========================================================================
// tabs/s2.js — S2: the brain integrating itself between sessions.
// ---------------------------------------------------------------------------
// S1 is a turn; S2 is what happened while nobody was looking — a different
// time-scale, so its own tab. Live is the session, this is the graph
// maintaining itself.
//
// Four units, one card shape (lib/cards.js primitives + lib/journal.js
// residue), differing only in what they PRODUCED:
//   aspect_integration  — taxonomy: new type/relation strings classified
//   community_detection — narrative: communities formed and enriched
//   healer              — repair: missing fields generated
//   consolidation       — settlement: clusters folded, links drawn
//
// The unit table is the single source for label + color + endpoint; adding a
// fifth unit is one row here plus its query, not a new render path.
// ===========================================================================

import { api } from '/static/lib/api.js';
import { poll } from '/static/lib/poll.js';
import { el, localTime, relativeTime } from '/static/lib/dom.js';
import { loadNodeDetail } from '/static/lib/node_detail.js';
import { promptSection, subRow, tierRow } from '/static/lib/cards.js';
import { journalBlock, journalChip, isStanding } from '/static/lib/journal.js';

// ── The unit table ─────────────────────────────────────────────────────────
// `fetch` returns that unit's runs. `color` is the unit's identity everywhere
// it appears (badge, card border, tier labels). Order = run order in
// coordinator.run_s2, so the tab reads in the sequence the brain executes.
const UNITS = [
  { key: 'aspect_integration',  label: 'ASPECTS',       color: '#b8ff7e',
    fetch: (p) => api.aspectRuns(p),
    blurb: 'new type + relation strings folded into the taxonomy' },
  { key: 'community_detection', label: 'COMMUNITY',     color: '#45B7D1',
    fetch: (p) => api.communityRuns(p),
    blurb: 'clusters given a narrative and a maturity' },
  { key: 'healer',              label: 'HEALER',        color: '#ff66aa',
    fetch: (p) => api.healerRuns(p),
    blurb: 'missing fields on existing memories filled in' },
  { key: 'consolidation',       label: 'CONSOLIDATION', color: '#33ff88',
    fetch: (p) => api.consolidationRuns(p),
    blurb: 'near-duplicates settled — folded, superseded, or kept apart' },
];

const UNIT_BY_KEY = Object.fromEntries(UNITS.map(u => [u.key, u]));

// Consolidation link relations → display. similar_to is symmetric (↔);
// supersedes/corrects/depends_on are directional (source → target). An
// unlisted relation renders its own verb rather than being dropped — the
// point of showing links is seeing what the run actually decided.
const REL_DISPLAY = {
  similar_to: { label: 'KEPT',       glyph: '↔', cls: 'enc-kind--kept' },
  supersedes: { label: 'SUPERSEDES', glyph: '→', cls: 'enc-kind--evolved' },
  corrects:   { label: 'CORRECTS',   glyph: '→', cls: 'enc-kind--corrects' },
  depends_on: { label: 'DEPENDS',    glyph: '→', cls: 'enc-kind--kept' },
};

let _hours = 48;
let _unitFilter = '';
let _loaded = false;

// ── Per-unit bodies — only what the unit produced ──────────────────────────

function _consolidationBody(run) {
  const out = [];
  for (const n of (run.synthesized || [])) {
    out.push(subRow({ kindClass: 'created', kindLabel: 'ENRICHED',
                      typeName: n.type, title: n.title, content: n.content, nodeId: n.id }));
  }
  for (const n of (run.archived || [])) {
    out.push(subRow({ kindClass: 'enc-kind--archived enc-sub-row--archived',
                      kindLabel: 'FOLDED IN', typeName: n.type, title: n.title,
                      content: n.content, contentDim: true, nodeId: n.id }));
  }
  for (const lnk of (run.links || [])) {
    const d = REL_DISPLAY[lnk.relation]
      || { label: (lnk.relation || 'LINKED').toUpperCase(), glyph: '→', cls: 'enc-kind--kept' };
    out.push(el('div', { class: 'enc-entry enc-sub-row' },
      el('span', { class: 'enc-kind ' + d.cls }, d.label),
      ' ', lnk.source || '',
      ' ', el('span', { class: d.cls }, d.glyph),
      ' ', lnk.target || '',
      lnk.description
        ? el('div', { class: 'enc-meta-line enc-meta-line--dim' }, lnk.description.substring(0, 250))
        : null,
    ));
  }
  return out;
}

function _communityBody(run) {
  const out = [
    tierRow('O', '', run.o_summary, { accentClass: 'enc-tier-label--o' }),
    tierRow('K', '', run.k_summary, { accentClass: 'enc-tier-label--k' }),
    tierRow('Δ', '', run.summary,   { accentClass: 'enc-tier-label--delta' }),
  ];
  for (const c of (run.communities || [])) {
    const matColor = c.maturity === 'settled' ? '#33ff88'
                   : c.maturity === 'active'  ? '#ffcc00'
                   : c.maturity === 'forming' ? '#45B7D1' : '#888';
    out.push(el('div', {
      class: 'enc-entry enc-community-row',
      onclick: (e) => { e.stopPropagation(); loadNodeDetail(c.id || ''); },
    },
      el('span', { class: 'enc-kind created' }, 'COMMUNITY'),
      ' ',
      el('span', { class: 'enc-community-maturity', style: { color: matColor } },
        (c.maturity || '?').toUpperCase()),
      el('span', { class: 'enc-title' }, c.title || ''),
      el('span', { class: 'enc-community-members' }, (c.members || 0) + ' members'),
    ));
  }
  return out;
}

function _okDeltaBody(run, color) {
  // Healer + aspects: the product IS the summaries. Tier labels take the
  // unit color so the card reads as one thing.
  return [
    tierRow('O', run.o_ref_type, run.o_summary, { accentStyle: { color } }),
    tierRow('K', run.k_ref_type, run.k_summary, { accentClass: 'enc-tier-label--k' }),
    tierRow('Δ', run.ref_type,   run.summary,   { accentClass: 'enc-tier-label--delta' }),
  ];
}

// ── The card ───────────────────────────────────────────────────────────────

function _headline(unitKey, run) {
  if (unitKey === 'consolidation') {
    const n = (run.synthesized || []).length + (run.archived || []).length
            + (run.links || []).length;
    return n + (n === 1 ? ' decision' : ' decisions');
  }
  if (unitKey === 'community_detection') {
    const n = (run.communities || []).length;
    return n ? n + ' communities in view' : (run.summary || '').substring(0, 60);
  }
  return (run.summary || '').substring(0, 90);
}

function _renderRunCard(unitKey, run) {
  const unit = UNIT_BY_KEY[unitKey];
  const notes = run.journal_notes || [];

  const bodyRows = unitKey === 'consolidation'       ? _consolidationBody(run)
                 : unitKey === 'community_detection' ? _communityBody(run)
                 :                                     _okDeltaBody(run, unit.color);
  const rows = bodyRows.filter(Boolean);
  // Journal FIRST: what the unit thought, above what it did.
  const jb = journalBlock(notes);
  if (!rows.length && !jb) {
    rows.push(el('div', { class: 'enc-empty-note' }, '(no write actions)'));
  }
  const body = el('div', { class: 'hook-body hook-body--padded' }, jb, ...rows);

  // Consolidation is the only unit whose prompt the dashboard can re-derive
  // (its chain carries the payload pointer); the others have no stored prompt.
  const prompt = unitKey === 'consolidation'
    ? promptSection(async () => {
        const d = await api.consolidationPrompt({ chain_id: run.chain_id });
        return d.user_content || d.error || '(no prompt available)';
      }, { label: 'Prompt' })
    : null;

  const header = el('div', { class: 'hook-header' },
    el('span', { class: 'hook-badge', style: { background: unit.color, color: '#000' } }, unit.label),
    el('span', { class: 'hook-time', title: run.timestamp || '' }, localTime(run.timestamp, 'time')),
    el('span', { class: 'hook-size' }, _headline(unitKey, run)),
    el('span', { class: 'card-header-tail' },
      journalChip(notes, { title: unit.label.toLowerCase() + ' · ' + localTime(run.timestamp, 'time') }),
      prompt ? prompt.button : null,
    ),
  );
  header.addEventListener('click', () => body.classList.toggle('open'));

  return el('div', {
    class: ['hook-entry', 'enc-entry', notes.some(isStanding) && 'card--standing'].filter(Boolean),
    dataset: { unit: unitKey, ts: run.timestamp || '' },
    style: { borderLeftColor: unit.color },
  },
    header,
    (run.o_summary || run.k_summary) && unitKey === 'consolidation'
      ? el('div', { class: 'hook-prompt' }, run.k_summary || run.o_summary || '')
      : null,
    body,
    prompt ? prompt.body : null,
  );
}

// ── Load ───────────────────────────────────────────────────────────────────

export async function loadS2() {
  const container = document.getElementById('feed-s2');
  if (!container) return;

  // ALWAYS fetch every unit, even when one is filtered. The pill row is a
  // summary as well as a filter — "has this unit gone silent?" is the question
  // it answers, and the idle styling is that answer. Fetching only the selected
  // unit made the other three render 0-and-idle, i.e. the tab reported three
  // healthy units as dead the moment you filtered. Four cheap endpoints in
  // parallel; the filter applies to the card list only.
  const settled = await Promise.all(UNITS.map(async (u) => {
    try {
      const d = await u.fetch({ hours: _hours });
      return (d.runs || []).map(run => ({ unitKey: u.key, run }));
    } catch (e) {
      console.error('[s2] ' + u.key + ' load failed:', e);
      return [];
    }
  }));
  const all = settled.flat();
  all.sort((a, b) => (b.run.timestamp || '').localeCompare(a.run.timestamp || ''));
  const shown = _unitFilter ? all.filter(i => i.unitKey === _unitFilter) : all;

  // Fingerprint short-circuit — same convention as the encoding feed and the
  // Boot view: skip the rebuild when nothing changed, so an expanded card
  // stays expanded across polls.
  const fingerprint = all.length + ':' + (all[0]?.run.timestamp || '') + ':' + _unitFilter + ':' + _hours;
  if (_loaded && container.dataset.fingerprint === fingerprint) return;
  container.dataset.fingerprint = fingerprint;
  _loaded = true;

  // Summary from the UNFILTERED set; cards from the filtered one.
  _renderSummaryBar(all);
  if (!shown.length) {
    container.replaceChildren(el('div', { class: 'feed-empty' },
      _unitFilter
        ? 'No ' + (UNIT_BY_KEY[_unitFilter]?.label || _unitFilter).toLowerCase()
          + ' runs in the last ' + _hours + 'h.'
        : 'No S2 runs in the last ' + _hours + 'h — the graph has been resting.'));
    return;
  }
  container.replaceChildren(...shown.map(item => _renderRunCard(item.unitKey, item.run)));
}

// Per-unit counts + last-run time. This is the tab's answer to "is S2 even
// running?" — a unit silent for days is the signal, so a unit with no runs
// still gets a row.
function _renderSummaryBar(all) {
  const bar = document.getElementById('s2-summary');
  if (!bar) return;
  bar.replaceChildren(...UNITS.map(u => {
    const runs = all.filter(i => i.unitKey === u.key);
    const last = runs[0]?.run.timestamp || '';
    const notes = runs.reduce((s, i) => s + (i.run.journal_notes || []).length, 0);
    const active = _unitFilter === u.key;
    return el('button', {
      class: ['s2-unit-pill', active && 's2-unit-pill--active', !runs.length && 's2-unit-pill--idle']
        .filter(Boolean),
      style: { '--unit-color': u.color },
      title: u.blurb + (last ? '\nlast run ' + relativeTime(last) : '\nno run in this window'),
      onclick: () => { _unitFilter = active ? '' : u.key; _loaded = false; loadS2(); },
    },
      el('span', { class: 's2-unit-dot', style: { background: u.color } }),
      el('span', { class: 's2-unit-name' }, u.label.toLowerCase()),
      el('span', { class: 's2-unit-count' }, String(runs.length)),
      notes ? el('span', { class: 's2-unit-notes', title: notes + ' journal notes' }, '📓' + notes) : null,
    );
  }));
}

export function onS2HoursChange() {
  const sel = document.getElementById('s2-hours');
  _hours = parseInt(sel ? sel.value : '48', 10) || 48;
  _loaded = false;
  loadS2();
}

// ── Lifecycle ──────────────────────────────────────────────────────────────

export function init() {
  poll.register({
    key: 's2-runs',
    interval: 20000,
    activeWhen: () => document.getElementById('tab-s2').classList.contains('active'),
    fetcher: loadS2,
  });
}

export function activate() { loadS2(); }
export function deactivate() { /* poll self-gates on activeWhen */ }
