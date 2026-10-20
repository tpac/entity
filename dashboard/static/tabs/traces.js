// ===========================================================================
// tabs/traces.js — trace chain rendering for arbitrary time-window + scale
// + session filter combos.
// ---------------------------------------------------------------------------
// Renders chains incrementally in batches of _TRACE_BATCH; a "Load more"
// button fetches the next page client-side (the server already returned
// the full result).
// ===========================================================================

import { api } from '/static/lib/api.js';
import { poll } from '/static/lib/poll.js';
import { escapeHtml, localTime, identityChipHTML, modelChipHTML } from '/static/lib/dom.js';
import { SCALE_COLORS } from '/static/lib/scales.js';
import { reapplyTraceFlashIfPending, loadNodeDetail } from '/static/lib/node_detail.js';
import { renderTraceDetail, collapsedBadges, isMergeableRefType, groupSummary } from '/static/lib/trace_detail.js';
import { renderFriendlyChain } from '/static/lib/trace_friendly.js';
import { sessionShort } from '/static/lib/sessions.js';

let _traceChainEntries = [];
let _traceRendered = 0;
const _TRACE_BATCH = 30;

// View mode: 'friendly' (jargon-free activity digest, one card per chain) or
// 'technical' (the full O/K/Δ event drill-down). Persisted so each operator
// keeps their choice; defaults to technical (hardening is the current use).
let _traceMode = (() => {
  try { return localStorage.getItem('traceMode') || 'technical'; } catch (_) { return 'technical'; }
})();

// Friendly relabelling of the jargon controls. Same filter values (s0/s1/s2),
// human words on top — a newcomer picks "Remembering", not "S1".
const _FRIENDLY_SCALE_LABELS = {
  '': 'All activity', s0: 'Conversations', s1: 'Remembering & learning',
  s2: 'Organizing', s3: 'Reflecting', s4: 'Growing',
};
const _TECH_SCALE_LABELS = {
  '': 'All scales', s0: 'S0 (Exchange)', s1: 'S1 (Turn)',
  s2: 'S2 (Graph)', s3: 'S3 (Sleep)', s4: 'S4 (Growth)',
};

// Expand-on-click state. Tracked by trace id (not DOM) so expansion survives
// the 5s poll that rewrites #traces-content from scratch — the same reason
// the source-ref flash is re-applied after each render. _traceEventById maps
// id → row so the delegated click handler can build detail without re-fetching.
const _traceExpanded = new Set();
let _traceEventById = {};
// Same deal for merged runs (see _groupRuns): open-state by group key, and a
// key → events map so the delegated handler can build the member rows without
// re-fetching. A group key is chain + ref_type + the run's first event id —
// stable across polls, so an open run stays open.
const _traceGroupsOpen = new Set();
let _traceGroupByKey = {};
// A run shorter than this reads fine as individual rows; merging two rows into
// a row-plus-caret saves nothing.
const _MERGE_MIN = 3;
// Signature of the last rendered result. The 5s poll calls loadTraces
// unconditionally; without this guard it rebuilt #traces-content every tick —
// blanking innerHTML resets the page scroll (jumps to top) and wipes the scroll
// position inside any expanded <pre> the operator is reading. We skip the
// rebuild when nothing relevant changed. Expansions go through the click
// handler (direct DOM), not loadTraces, so skipping never strands them.
let _lastTraceFp = null;

function _detailHTML(ev) {
  return '<div class="trace-detail" data-detail-for="' + escapeHtml(String(ev.id)) + '" '
    + 'style="background:#070710;border-top:1px solid #15151f;padding:6px 14px 10px 22px">'
    + renderTraceDetail(ev) + '</div>';
}

export function onTraceScaleChange() {
  const scale = document.getElementById('trace-scale-filter').value;
  const hoursEl = document.getElementById('trace-hours-filter');
  if (scale && scale >= 's2' && parseInt(hoursEl.value) < 168) {
    hoursEl.value = '168';
  }
  loadTraces();
}

export async function loadTraces(opts = {}) {
  try {
    const scaleFilter = document.getElementById('trace-scale-filter').value;
    const hours = document.getElementById('trace-hours-filter').value;
    // Callers (e.g. node_detail's "Open in Traces tab") can force a
    // session that isn't in the dropdown yet — setting the dropdown's
    // .value to a missing option silently resets to '', so reading from
    // it would lose the override. Explicit opts.session wins.
    const sessionFilter = (opts.session !== undefined)
      ? opts.session
      : document.getElementById('trace-session-filter').value;
    const traces = await api.traces({
      hours,
      scale: scaleFilter || undefined,
      session: sessionFilter || undefined,
    });
    // Skip the rebuild when nothing changed (traces use absolute timestamps, so
    // identical data → identical render). Newest id+count+filters+mode capture
    // every structural change; a new trace flips the newest id. Bypass via
    // opts.force for explicit refreshes that must re-render regardless.
    const fp = [_traceMode, scaleFilter, hours, sessionFilter,
      traces.length, traces[0] && traces[0].id, traces[traces.length - 1] && traces[traces.length - 1].id].join('|');
    if (!opts.force && fp === _lastTraceFp) return;
    _lastTraceFp = fp;
    const el = document.getElementById('traces-content');
    const techLabel = hours <= 1 ? '1h' : hours <= 6 ? '6h' : hours <= 24 ? '24h' : '7d';
    const friendlyLabel = hours <= 1 ? 'the last hour' : hours <= 6 ? 'the last 6 hours'
      : hours <= 24 ? 'the last day' : 'the last week';
    const chainCount = new Set(traces.map(t => t.chain_id)).size;
    document.getElementById('trace-count').textContent = _traceMode === 'friendly'
      ? chainCount + ' things the brain did in ' + friendlyLabel
      : traces.length + ' events (' + techLabel + ')';

    const sessSelect = document.getElementById('trace-session-filter');
    // Sync the dropdown to the resolved sessionFilter — without this,
    // an override passed via opts.session wouldn't visually appear in
    // the dropdown, so the user couldn't tell which session they're
    // looking at.
    try {
      const sessions = await api.sessions();
      // Note: not `opts` — that's the function parameter (opts.force/.session).
      // Sessions read by their handle (worktree / branch tail), resolved once
      // server-side in queries/sessions.py so the Traces dropdown, the session
      // rail, and activity cards all say the same name; a session with no
      // handle reads by its short.
      const optsHtml = '<option value="">All sessions</option>' + sessions.map(s =>
        '<option value="' + s.id + '"' + (s.id === sessionFilter ? ' selected' : '') + '>'
        + escapeHtml(s.handle || sessionShort(s.id)) + ' · ' + s.events + ' events</option>'
      ).join('');
      sessSelect.innerHTML = optsHtml;
    } catch(e) { /* keep existing options */ }

    if (!traces.length) {
      el.innerHTML = '<div style="color:#888;text-align:center;padding:40px">No trace events yet. Traces will appear after your next prompt.</div>';
      _traceChainEntries = [];
      return;
    }

    const chains = {};
    traces.forEach(t => {
      if (!chains[t.chain_id]) chains[t.chain_id] = [];
      chains[t.chain_id].push(t);
    });
    const chainEntries = Object.entries(chains);
    chainEntries.forEach(([_, events]) => events.sort(
      (a, b) => (a.created_at || '').localeCompare(b.created_at || '')));
    chainEntries.sort((a, b) => {
      const aMax = a[1][a[1].length - 1]?.created_at || '';
      const bMax = b[1][b[1].length - 1]?.created_at || '';
      return bMax.localeCompare(aMax);
    });
    _traceChainEntries = chainEntries;
    _traceRendered = 0;
    _traceEventById = {};
    _traceGroupByKey = {};
    el.innerHTML = '';
    _renderTracesBatch(el);
    // Re-apply the source-ref flash if one is still pending. The 5s
    // poll rewrites #traces-content from scratch, which would otherwise
    // destroy the .trace-event--target class added by node_detail.js.
    // reapplyTraceFlashIfPending is a no-op when nothing's pending.
    reapplyTraceFlashIfPending();
  } catch(e) { console.error('loadTraces', e); }
}

function _traceChainLabel(chainId) {
  // Map chain IDs to readable labels. Chain IDs remain string-prefixed
  // (s0-/s1r-/s1e-/s2-) even after trace_events.id became hex — those are
  // different identifiers.
  if (chainId.startsWith('s0-')) { const p = chainId.split('-'); return 'S0 Exchange #' + (p[2] || '?'); }
  if (chainId.startsWith('s1r-')) { const p = chainId.split('-'); return 'S1 Recall (Surface) #' + (p[2] || '?'); }
  if (chainId.startsWith('s1e-')) { const p = chainId.split('-'); return 'S1 Encode #' + (p[2] || '?'); }
  if (chainId.startsWith('s2-')) {
    const op = chainId.split('-').slice(2).join('-');
    const labels = {community_detection:'S2 Community Detection', consolidation:'S2 Consolidation', healer:'S2 Healer', aspect_integration:'S2 Aspect Integration'};
    return labels[op] || 'S2 ' + op.replace(/_/g, ' ');
  }
  if (chainId.startsWith('s3-')) return 'S3 ' + chainId.split('-').slice(2).join(' ');
  return chainId;
}

const _TYPE_LABELS = {O:'Observed', K:'Selected', delta:'Changed', outcome:'Outcome'};
const _TYPE_COLORS = {O:'#45B7D1', K:'#ffaa33', delta:'#33ff88', outcome:'#aa66ff'};

/** Split a chain's events into rows, each an array: a run of ≥_MERGE_MIN
 *  consecutive events sharing a mergeable ref_type stays whole (rendered as one
 *  group row), everything else comes back as a 1-element array. Length is the
 *  discriminant — a 2-element array is never produced. Runs (not whole-chain
 *  buckets) so the chain keeps its narrative order — an S2 unit's edge writes
 *  stay between the journal notes they happened between. */
function _groupRuns(events) {
  const rows = [];
  let run = [];
  const flush = () => {
    if (!run.length) return;
    if (run.length >= _MERGE_MIN) rows.push(run);
    else run.forEach(ev => rows.push([ev]));
    run = [];
  };
  for (const ev of events) {
    if (run.length && ev.ref_type === run[0].ref_type) { run.push(ev); continue; }
    flush();
    if (isMergeableRefType(ev.ref_type)) run = [ev];
    else rows.push([ev]);
  }
  flush();
  return rows;
}

function _eventRowHTML(ev, nested) {
  if (ev.id) _traceEventById[ev.id] = ev;
  const tColor = _TYPE_COLORS[ev.event_type] || '#666';
  const tLabel = _TYPE_LABELS[ev.event_type] || ev.event_type;
  const traceIdAttr = ev.id ? ' data-trace-id="' + escapeHtml(String(ev.id)) + '"' : '';
  const expanded = ev.id && _traceExpanded.has(ev.id);
  const caret = '<span class="trace-caret" style="flex-shrink:0;width:10px;color:#556;font-size:9px;margin-top:3px">' + (expanded ? '▾' : '▸') + '</span>';
  // Whole row is the expand affordance — cursor:pointer signals it; the
  // delegated #traces-content handler toggles on click.
  let h = '<div class="trace-event"' + traceIdAttr + ' style="padding:4px 12px 4px '
    + (nested ? 8 : 16) + 'px;border-top:1px solid #111;display:flex;gap:6px;align-items:flex-start;cursor:pointer">';
  h += caret;
  h += '<span style="flex-shrink:0;font-size:10px;font-weight:bold;color:' + tColor + ';min-width:55px">' + tLabel + '</span>';
  h += '<div style="flex:1;min-width:0">';
  if (ev.ref_type && !nested) h += '<span style="color:#666;font-size:10px;background:#1a1a2a;padding:1px 4px;border-radius:2px;margin-right:4px">' + ev.ref_type + '</span>';
  h += collapsedBadges(ev);
  h += '<div style="color:#ccc;font-size:12px;margin-top:2px;white-space:pre-wrap;word-break:break-word">' + escapeHtml((ev.summary || '').substring(0, 300)) + '</div>';
  h += '</div>';
  h += '<span style="color:#444;font-size:9px;flex-shrink:0;white-space:nowrap">' + localTime(ev.created_at, 'time') + '</span>';
  h += '</div>';
  if (expanded) h += _detailHTML(ev);
  return h;
}

function _groupKey(chainId, run) {
  return chainId + '|' + run[0].ref_type + '|' + (run[0].id || run[0].created_at || '');
}

// Always rendered/inserted directly after its header, so the header's next
// sibling IS the member list — no selector round-trip through the key (which
// would have to match the escaping the attribute was written with).
function _groupMembersHTML(run) {
  return '<div class="trace-group-members" style="margin-left:22px;border-left:1px solid #1b1b28">'
    + run.map(ev => _eventRowHTML(ev, true)).join('') + '</div>';
}

function _membersOf(headerEl) {
  const next = headerEl.nextElementSibling;
  return next && next.classList.contains('trace-group-members') ? next : null;
}

/** Open a group in place: remember it, insert its members, flip the caret.
 *  The one writer of that three-part contract — the click handler and
 *  revealTrace both come here. */
function _openGroup(headerEl, key, run) {
  _traceGroupsOpen.add(key);
  if (!_membersOf(headerEl)) headerEl.insertAdjacentHTML('afterend', _groupMembersHTML(run));
  const caret = headerEl.querySelector('.trace-caret');
  if (caret) caret.textContent = '▾';
}

/** One collapsed row standing for a run of same-ref_type events. */
function _groupRowHTML(chainId, run) {
  const key = _groupKey(chainId, run);
  _traceGroupByKey[key] = run;
  // Members register in _traceEventById even while collapsed, so expanding the
  // run and then a member finds the event without a re-render.
  run.forEach(ev => { if (ev.id) _traceEventById[ev.id] = ev; });
  const open = _traceGroupsOpen.has(key);
  const ev0 = run[0];
  const tColor = _TYPE_COLORS[ev0.event_type] || '#666';
  const tLabel = _TYPE_LABELS[ev0.event_type] || ev0.event_type;

  let h = '<div class="trace-group" data-group-key="' + escapeHtml(key) + '" '
    + 'style="padding:4px 12px 4px 16px;border-top:1px solid #111;display:flex;gap:6px;'
    + 'align-items:flex-start;cursor:pointer;background:#0c0c17">';
  h += '<span class="trace-caret" style="flex-shrink:0;width:10px;color:#7a7a90;font-size:9px;margin-top:3px">' + (open ? '▾' : '▸') + '</span>';
  h += '<span style="flex-shrink:0;font-size:10px;font-weight:bold;color:' + tColor + ';min-width:55px">' + tLabel + '</span>';
  h += '<div style="flex:1;min-width:0">';
  h += '<span style="color:#8a8aa0;font-size:10px;background:#1a1a2a;padding:1px 4px;border-radius:2px;margin-right:4px">'
    + escapeHtml(ev0.ref_type) + ' <b style="color:#bcd">×' + run.length + '</b></span>';
  h += collapsedBadges(run);
  h += '<div style="color:#ccc;font-size:12px;margin-top:2px;white-space:pre-wrap;word-break:break-word">'
    + escapeHtml(groupSummary(run).substring(0, 300)) + '</div>';
  h += '</div>';
  h += '<span style="color:#444;font-size:9px;flex-shrink:0;white-space:nowrap">' + localTime(run[run.length - 1].created_at, 'time') + '</span>';
  h += '</div>';
  if (open) h += _groupMembersHTML(run);
  return h;
}

function _renderTracesBatch(el) {
  const end = Math.min(_traceRendered + _TRACE_BATCH, _traceChainEntries.length);

  let html = '';
  for (let i = _traceRendered; i < end; i++) {
    const [chainId, events] = _traceChainEntries[i];

    // Friendly mode: one plain-language card per chain, no event drill-down.
    if (_traceMode === 'friendly') {
      html += renderFriendlyChain(chainId, events);
      continue;
    }

    const firstTime = events[0].created_at;
    const chainScale = events[0].scale;
    const color = SCALE_COLORS[chainScale] || '#666';
    const label = _traceChainLabel(chainId);
    const sessionId = events[0].session_id || '';
    const sessionTag = sessionId ? '<span style="color:#444;font-size:9px;margin-left:6px">' + sessionShort(sessionId) + '</span>' : '';
    // Identity per chain: first event with non-empty identity wins. Chains
    // often span multiple writes from the same speaker pair so one rendering
    // per chain is enough; a per-event chip would just repeat ad nauseam.
    let chainHi = '', chainAi = '';
    for (const ev of events) {
      if (ev.human_identity || ev.agent_identity) {
        chainHi = ev.human_identity; chainAi = ev.agent_identity; break;
      }
    }
    const identityTag = (chainHi || chainAi)
      ? '<span style="margin-left:6px">' + identityChipHTML(chainHi, chainAi) + '</span>'
      : '';
    // Which model produced this turn. The LAST stamped event wins (events are
    // sorted ascending): the assistant_message written at Stop names this
    // turn's model exactly, while the prompt-time user_message can only carry
    // the previous turn's — so on a mid-session model switch the later row is
    // the truthful one. Empty for chains from before the stamp existed.
    let chainModel = '', chainHost = '';
    for (const ev of events) {
      if (ev.model) { chainModel = ev.model; chainHost = ev.host || ''; }
    }
    const modelTag = chainModel
      ? '<span style="margin-left:6px">' + modelChipHTML(chainModel, chainHost) + '</span>'
      : '';

    // data-chain-id on the wrapper + data-trace-id on each event row let
    // node_detail.js scroll a specific trace into view after navigating
    // from a source-ref card. Without these attributes there'd be no
    // selector to target the right event.
    html += '<div class="trace-chain" data-chain-id="' + escapeHtml(chainId) + '" style="background:#0a0a12;border-radius:8px;margin:6px 0;border-left:3px solid ' + color + '">';
    html += '<div style="padding:8px 12px;display:flex;justify-content:space-between;align-items:center">';
    html += '<div><span style="color:' + color + ';font-size:12px;font-weight:bold">' + label + '</span>' + sessionTag + identityTag + modelTag + '</div>';
    html += '<span style="color:#555;font-size:10px">' + localTime(firstTime) + '</span>';
    html += '</div>';

    for (const run of _groupRuns(events)) {
      html += run.length > 1 ? _groupRowHTML(chainId, run) : _eventRowHTML(run[0], false);
    }

    html += '</div>';
  }
  el.insertAdjacentHTML('beforeend', html);
  _traceRendered = end;

  if (_traceRendered < _traceChainEntries.length) {
    el.insertAdjacentHTML('beforeend', '<div id="trace-load-more" style="text-align:center;padding:12px"><button onclick="_loadMoreTraces()" style="background:#1a1a2a;color:#7eb8ff;border:1px solid #3a3a5a;padding:4px 16px;border-radius:4px;cursor:pointer">Load more (' + (_traceChainEntries.length - _traceRendered) + ' remaining)</button></div>');
  }
}

/** Open the collapsed run holding `traceId`, if one is hiding it. A source-ref
 *  jump (node_detail) looks the row up by data-trace-id; a merged run keeps its
 *  members out of the DOM until opened, so the jump asks here first. */
export function revealTrace(traceId) {
  const want = String(traceId);
  for (const header of document.querySelectorAll('.trace-group')) {
    const key = header.getAttribute('data-group-key');
    const run = _traceGroupByKey[key];
    if (!run || _traceGroupsOpen.has(key)) continue;
    if (!run.some(ev => String(ev.id) === want)) continue;
    _openGroup(header, key, run);
    return true;
  }
  return false;
}

export function _loadMoreTraces() {
  const btn = document.getElementById('trace-load-more');
  if (btn) btn.remove();
  _renderTracesBatch(document.getElementById('traces-content'));
}

// Delegated click handler on the stable #traces-content container (its
// innerHTML is rewritten every poll, but the element itself persists, so one
// listener outlives every re-render). Two targets:
//   .trace-nodeid → open that node in the detail panel (no inline onclick, so
//                   arbitrary node-id chars never get interpolated into JS).
//   .trace-event  → toggle the expanded technical detail for that event.
// The detail panel is a SIBLING of .trace-event (inserted afterend), not a
// child, so clicks inside it (e.g. selecting text in a code block) don't
// match `.trace-event` and won't collapse the row.
function _onTracesClick(e) {
  const nodeEl = e.target.closest('.trace-nodeid');
  if (nodeEl) {
    const id = nodeEl.dataset.nodeid;
    if (id) loadNodeDetail(id);
    return;
  }
  // Group header → toggle its member rows. Members live in a SIBLING div, so a
  // click on one matches .trace-event (below) and never this branch.
  const groupEl = e.target.closest('.trace-group');
  if (groupEl) {
    const key = groupEl.getAttribute('data-group-key');
    if (_traceGroupsOpen.has(key)) {
      _traceGroupsOpen.delete(key);
      const members = _membersOf(groupEl);
      if (members) members.remove();
      const caret = groupEl.querySelector('.trace-caret');
      if (caret) caret.textContent = '▸';
    } else if (_traceGroupByKey[key]) {
      _openGroup(groupEl, key, _traceGroupByKey[key]);
    }
    return;
  }
  const row = e.target.closest('.trace-event');
  if (!row) return;
  const id = row.getAttribute('data-trace-id');
  if (!id) return;
  const caret = row.querySelector('.trace-caret');
  const existing = document.querySelector('.trace-detail[data-detail-for="' + id + '"]');
  if (_traceExpanded.has(id)) {
    _traceExpanded.delete(id);
    if (existing) existing.remove();
    if (caret) caret.textContent = '▸';
  } else {
    const ev = _traceEventById[id];
    if (!ev) return;   // not in the rendered batch's map — don't flip the caret with no detail to show
    _traceExpanded.add(id);
    if (!existing) row.insertAdjacentHTML('afterend', _detailHTML(ev));
    if (caret) caret.textContent = '▾';
  }
}

// ── View mode (Friendly ⇄ Technical) ──────────────────────────────────

// Relabel the jargon controls + sync the toggle buttons to the active mode.
function _applyModeLabels() {
  const labels = _traceMode === 'friendly' ? _FRIENDLY_SCALE_LABELS : _TECH_SCALE_LABELS;
  const sel = document.getElementById('trace-scale-filter');
  if (sel) [...sel.options].forEach(o => { if (o.value in labels) o.textContent = labels[o.value]; });
  ['friendly', 'technical'].forEach(m => {
    const b = document.getElementById('trace-mode-' + m);
    if (b) b.classList.toggle('active', _traceMode === m);
  });
}

export function setTraceMode(mode) {
  if (mode !== 'friendly' && mode !== 'technical' || mode === _traceMode) return;
  _traceMode = mode;
  try { localStorage.setItem('traceMode', mode); } catch (_) {}
  // Expanded-detail state is technical-only; clear on switch so it's not
  // stranded behind the friendly cards.
  _traceExpanded.clear();
  _traceGroupsOpen.clear();
  _applyModeLabels();
  loadTraces();
}

// ── Lifecycle ─────────────────────────────────────────────────────────

export function init() {
  // One delegated listener on the persistent container — survives every
  // poll-driven innerHTML rewrite (see _onTracesClick).
  const container = document.getElementById('traces-content');
  if (container && !container._traceClickBound) {
    container.addEventListener('click', _onTracesClick);
    container._traceClickBound = true;
  }
  _applyModeLabels();
  // 5s refresh while the Traces tab is active, via poll.register like the
  // other tabs.
  poll.register({
    key: 'traces',
    interval: 5000,
    activeWhen: () => {
      const tab = document.getElementById('tab-traces');
      return tab && tab.classList.contains('active');
    },
    fetcher: loadTraces,
  });
}

export function activate() {
  _applyModeLabels();
  loadTraces();
}

export function deactivate() {}
