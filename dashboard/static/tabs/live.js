// ===========================================================================
// tabs/live.js — the session, as it happens.
// ---------------------------------------------------------------------------
// One chronological feed of S1 activity (lib/activity.js): surface runs and
// encode runs, interleaved on one time axis but rendered as themselves.
//
// Recognition runs every turn, encoding every Nth: two card kinds, one axis.
// S2 has its own tab; focus lives on the rail.
//
// This module owns:
//   • the split layout (graph ⇄ stream) + graph visibility
//   • the insights panel
//   • the stream rail — who is live, and which stream the feed is focused on
//   • the activity feed: fetch, assemble, incremental render
//   • the graph's live coupling (hover previews, click pins, live firing)
//
// It owns no card markup: cards render via lib/activity.js, residue via
// lib/journal.js, session identity via lib/sessions.js.
// ===========================================================================

import { api } from '/static/lib/api.js';
import { poll } from '/static/lib/poll.js';
import bus from '/static/lib/bus.js';
import { el, escapeHtml } from '/static/lib/dom.js';
import * as graph from './graph.js';
import { assembleActivity, renderActivity, fingerprint, graphEvent } from '/static/lib/activity.js';
import * as sessions from '/static/lib/sessions.js';

// Cursor is an ISO timestamp: trace_events.id is unordered 8-char hex;
// created_at is monotonic per writer.
let lastRecallTs = '';
const MAX_ITEMS = 110;
const RECALL_POLL_MS = 2000;
const ENCODE_POLL_MS = 6000;
// 48h/200 covers normal browsing without pagination. The per-run payload is
// ~1KB (prompt is lazy, no inline blobs), so the window is cheap.
const ENCODE_WINDOW = { limit: 200, hours: 48 };

// Raw halves, keyed so a re-fetch replaces rather than duplicates. Recall
// events arrive incrementally (2s cursor poll); encode runs arrive as a full
// window each time (they mutate — nodes/edges/journal land after the run's
// first trace, so a run must be replaceable, not append-only).
const _recallsById = new Map();
const _encodesByChain = new Map();

// Rendered cards: key → { el, fingerprint }. The feed patches per card instead
// of rebuilding, so an expanded card stays expanded and the scroll position
// holds under a 2s poll.
const _rendered = new Map();

let _focusSession = '';
// Whether the collapsed resting-stream group is expanded. Page-session state:
// a rail that reopened itself on every poll would fight the operator.
let _restingOpen = false;

// ── Stream rail ────────────────────────────────────────────────────────────
// Streams are shown, not listed: one pill per stream in its own hue. Clicking a
// pill focuses the feed on it; clicking again clears. With several streams
// interleaving, the hue is what makes a row readable at a glance — the pill is
// where you learn which hue is whom.
//
// Dormant streams collapse behind a count. A machine accumulates them, and at
// full width they would wrap the rail onto several rows and push the feed
// down — so the ones actually thinking get the space, and the rest are one
// click away.

function _renderStreamRail() {
  const rail = document.getElementById('stream-rail');
  if (!rail) return;

  // Streams worth a pill: live ones, plus any that own a card on screen
  // (a stream that just ended is still the thing you're reading).
  const live = sessions.liveSessions();
  const seen = new Set(live.map(s => s.id));
  const inFeed = [];
  for (const key of _rendered.keys()) {
    const sid = _rendered.get(key).el?.dataset.session || '';
    if (sid && !seen.has(sid)) { seen.add(sid); inFeed.push(sessions.sessionInfo(sid)); }
  }
  const rows = [...live, ...inFeed];

  // Awake = actively thinking. Everything else (dormant on the roster, or ended
  // but still owning cards) collapses behind a count.
  const isAwake = (s) => s.live && s.state === 'active';
  const awake = rows.filter(isAwake);
  const resting = rows.filter(s => !isAwake(s));

  const pill = (s) => {
    const color = sessions.sessionColor(s.id);
    const active = _focusSession === s.id;
    const awakeNow = isAwake(s);
    return el('button', {
      class: ['stream-pill', active && 'stream-pill--active', !awakeNow && 'stream-pill--dormant']
        .filter(Boolean),
      style: { '--stream-color': color },
      title: sessions.sessionTooltip(s.id),
      onclick: () => setFocusSession(active ? '' : s.id),
    },
      el('span', {
        class: ['stream-pill-dot', awakeNow && 'stream-pill-dot--live'].filter(Boolean),
        style: { background: color },
      }),
      el('span', { class: 'stream-pill-name' }, sessions.sessionLabel(s.id)),
      s.turns ? el('span', { class: 'stream-pill-count' }, String(s.turns)) : null,
    );
  };

  const pills = [
    el('button', {
      class: ['stream-pill', 'stream-pill--all', !_focusSession && 'stream-pill--active']
        .filter(Boolean),
      title: 'Every stream, newest first',
      onclick: () => setFocusSession(''),
    }, 'all streams', el('span', { class: 'stream-pill-count' }, String(rows.length))),
    ...awake.map(pill),
  ];

  // Keep a focused resting stream visible even while collapsed — hiding the
  // pill you just clicked would leave the filter on with nothing indicating it.
  const focusedResting = resting.find(s => s.id === _focusSession);
  if (focusedResting) pills.push(pill(focusedResting));

  const hidden = resting.filter(s => s.id !== _focusSession);
  if (hidden.length) {
    pills.push(el('button', {
      class: ['stream-pill', 'stream-pill--more', _restingOpen && 'stream-pill--active']
        .filter(Boolean),
      title: hidden.map(s => sessions.sessionLabel(s.id)).join('\n'),
      onclick: () => { _restingOpen = !_restingOpen; _renderStreamRail(); },
    }, (_restingOpen ? '▾ ' : '▸ ') + hidden.length + ' resting'));
    if (_restingOpen) pills.push(...hidden.map(pill));
  }
  rail.replaceChildren(...pills);
}

export function setFocusSession(sessionId) {
  if (_focusSession === sessionId) return;
  _focusSession = sessionId || '';
  // Focus is a filter over what we already hold — no refetch, so switching
  // streams is instant. Re-assert visibility on every rendered card.
  _applyFocus();
  _renderStreamRail();
}

function _applyFocus() {
  for (const { el: node } of _rendered.values()) {
    const sid = node.dataset.session || '';
    node.style.display = (!_focusSession || sid === _focusSession) ? '' : 'none';
  }
  _renderFeedEmptyState();
}

function _renderFeedEmptyState() {
  const feed = document.getElementById('feed-activity');
  if (!feed) return;
  const visible = [..._rendered.values()].filter(r => r.el.style.display !== 'none').length;
  let note = feed.querySelector('.feed-empty');
  if (visible) { if (note) note.remove(); return; }
  if (!note) {
    note = el('div', { class: 'feed-empty' });
    feed.appendChild(note);
  }
  note.textContent = _focusSession
    ? 'Nothing from ' + sessions.sessionLabel(_focusSession) + ' in this window.'
    : 'Waiting for brain activity…';
}

// ── Live firing ────────────────────────────────────────────────────────────
// The graph lights for what the brain does while you watch: a recall once, as
// it lands with its picks; an encode once, when its run finishes. This module
// decides what is new because it owns the fetch: an item fires on the delivery
// that first shows it settled, and only if that delivery is live. Anything else
// is history and settles silently, or every tab switch would look like the
// brain had just thought something. A delivery is live when
//   • the feed was watching: no gap since the last good poll — time on another
//     tab, a hidden window, fetches failing. A blip under a poll or two still
//     counts; what landed in it is seconds old.
//   • it is measured against what the feed already holds. Recalls: the fetch
//     was cursored (the newest-N backfill is history by definition). Encodes:
//     the cache already held runs. A failed query answers [] exactly like an
//     idle brain, so an empty first fetch primes nothing. The cost: a brain's
//     very first recall lands quietly, and so does the first encode after a
//     page load that found no run in the encode window (48h).

// A poll starts within its interval plus one scheduler tick of the previous
// one ending; anything past that plus slack is a gap.
const WATCH_SLACK_MS = 4000;
let _recallsOkAt = 0;   // when each poll last succeeded
let _encodesOkAt = 0;

function _watching(okAt, interval) {
  return okAt > 0 && Date.now() - okAt < interval + WATCH_SLACK_MS;
}

// Settled = what the graph lights has arrived: a recall's picks; an encode
// run's closing delta, which carries its writes.
const _recallSettled = (evt) => (evt.used_ids || []).length > 0;
const _encodeSettled = (run) => !!run.delta_ts;

// Hold `next` under `key`; true when this delivery is the one that settled it.
// "Not held" reads as unseen, so the caches must never evict what a fetch can
// deliver again (see the trims in _syncFeed).
function _hold(map, key, next, isSettled) {
  const prev = map.get(key);
  map.set(key, next);
  return isSettled(next) && !(prev && isSettled(prev));
}

// Oldest first, so where two waves overlap the newer one's tint wins.
function _fire(items) {
  for (const item of items.reverse()) bus.publish('activity:new', { event: graphEvent(item) });
}

// ── Fetch ──────────────────────────────────────────────────────────────────

// The first fetch pulls a backfill: the encode window is 48h/200, so a short
// first page would leave the feed almost all encode runs. After that the
// cursor pages newest-first, so the page is as deep as the feed: whatever lands
// beyond a full page — a burst, or a gap spent on another tab — is older than
// everything the feed shows, and the cursor may jump past it.
async function pollRecalls() {
  const live = !!lastRecallTs && _watching(_recallsOkAt, RECALL_POLL_MS);
  try {
    const d = await api.recalls(lastRecallTs
      ? { limit: MAX_ITEMS, since_ts: lastRecallTs }
      : { limit: 300 });
    const settled = [];
    for (const evt of (d.events || [])) {
      if (evt.id && _hold(_recallsById, evt.id, evt, _recallSettled)) settled.push(evt);
    }
    if (d.latest_ts) lastRecallTs = d.latest_ts;
    _recallsOkAt = Date.now();
    _syncFeed();
    if (live) _fire(assembleActivity(settled, []));
  } catch (e) { console.error('[live] recall poll failed:', e); }
}

async function pollEncodes() {
  const live = _encodesByChain.size > 0 && _watching(_encodesOkAt, ENCODE_POLL_MS);
  try {
    const d = await api.encodingRuns(ENCODE_WINDOW);
    const settled = [];
    for (const run of (d.runs || [])) {
      if (run.chain_id && _hold(_encodesByChain, run.chain_id, run, _encodeSettled)) {
        settled.push(run);
      }
    }
    _encodesOkAt = Date.now();
    _syncFeed();
    if (live) _fire(assembleActivity([], settled));
  } catch (e) { console.error('[live] encode poll failed:', e); }
}

// ── Render ─────────────────────────────────────────────────────────────────

const _graphHooks = {
  onHover: (item) => graph.previewRecallOnGraph(graphEvent(item)),
  onLeave: () => graph.clearRecallPreview(),
  onPin:   (item) => graph.pinRecallToGraph(graphEvent(item)),
  // Collapsing the card that holds the graph hands it back to the default view.
  onUnpin: (item) => graph.clearHighlight(graphEvent(item).id),
};

function _syncFeed() {
  const feed = document.getElementById('feed-activity');
  if (!feed) return;
  const items = assembleActivity([..._recallsById.values()], [..._encodesByChain.values()]);

  const live = new Set();
  for (const item of items.slice(0, MAX_ITEMS)) {
    live.add(item.key);
    const fp = fingerprint(item);
    const prior = _rendered.get(item.key);
    if (prior && prior.fingerprint === fp) continue;
    const wasOpen = prior?.el.classList.contains('open');
    const node = renderActivity(item, { showSession: true, hooks: _graphHooks });
    if (wasOpen) {
      // Body is built lazily on expand — a card that was already open has to
      // be populated here, or replacing it would leave the operator staring
      // at an empty body after a data change.
      node._buildActivityBody?.();
      node.classList.add('open');
    }
    if (_focusSession && (item.session_id || '') !== _focusSession) node.style.display = 'none';
    if (prior) {
      prior.el.replaceWith(node);
    } else {
      // Insert in timestamp order — items arrive out of order (an encode run
      // for an older turn lands after newer recalls).
      const at = [...feed.querySelectorAll('.act')]
        .find(n => (n.dataset.ts || '') < (item.ts || ''));
      if (at) feed.insertBefore(node, at); else feed.appendChild(node);
    }
    _rendered.set(item.key, { el: node, fingerprint: fp });
  }

  // Evict cards that fell out of the window — DOM and map together, so the
  // page's memory stays flat over a long session.
  for (const [key, rec] of _rendered) {
    if (!live.has(key)) { rec.el.remove(); _rendered.delete(key); }
  }
  // Bound the raw caches, or they grow all day. The encode cache holds a whole
  // fetch window: every poll re-delivers it, and a run evicted from under it
  // would come back looking unseen — and fire as new.
  _trim(_recallsById, MAX_ITEMS * 3, e => e.timestamp || '');
  _trim(_encodesByChain, ENCODE_WINDOW.limit, r => r.start_ts || '');

  _renderFeedEmptyState();
  _renderStreamRail();
}

// Drop the OLDEST entries past `max`, by timestamp.
//
// Not by insertion order: both feeds arrive newest-first (ORDER BY created_at
// DESC), and Map.set keeps a re-set key at its ORIGINAL position — so deleting
// from the front evicts the newest rows. Sorting only runs when actually over
// the cap.
function _trim(map, max, tsOf) {
  if (map.size <= max) return;
  const byAge = [...map.entries()].sort(
    (a, b) => (tsOf(a[1]) || '').localeCompare(tsOf(b[1]) || ''));
  for (let i = 0; i < byAge.length - max; i++) map.delete(byAge[i][0]);
}

// ── Insights panel ─────────────────────────────────────────────────────────
// Pure subscriber on `insights:tick`. Dismissal is page-session-scoped only —
// no localStorage, so a reload restores the signal and a real problem can't be
// permanently silenced.
const _dismissedInsights = new Set();
let _lastInsights = [];

function _renderInsightsPanel({ insights }) {
  _lastInsights = insights || [];
  _redrawInsightsPanel();
}

function _redrawInsightsPanel() {
  const panel = document.getElementById('insights-panel');
  if (!panel) return;
  const visible = _lastInsights.filter(i => !_dismissedInsights.has(i.title || ''));
  if (!visible.length) { panel.innerHTML = ''; return; }
  panel.innerHTML = visible.map(i => {
    const sev = (i.severity || 'low').toLowerCase();
    return '<div class="insights-card insights-card--' + escapeHtml(sev) + '">' +
      '<div class="insights-icon">' + (i.icon || '') + '</div>' +
      '<div class="insights-body">' +
        '<div class="insights-title">' + escapeHtml(i.title || '') + '</div>' +
        '<div class="insights-detail">' + escapeHtml(i.detail || '') + '</div>' +
      '</div>' +
      '<button class="insights-dismiss" title="Dismiss until reload" ' +
              'data-title="' + escapeHtml(i.title || '') + '">&times;</button>' +
    '</div>';
  }).join('');
}

// Delegated once on the panel — per-render listeners would leak.
function _wireInsightsDismiss() {
  const panel = document.getElementById('insights-panel');
  if (!panel) return;
  panel.addEventListener('click', (e) => {
    const btn = e.target.closest('.insights-dismiss');
    if (!btn) return;
    _dismissedInsights.add(btn.dataset.title || '');
    _redrawInsightsPanel();
  });
}

// ── Live split layout ──────────────────────────────────────────────────────
// Four orientations: graph-left / graph-right / graph-top / graph-bottom.
// _applyLayout is the single state-driven writer: it sets the grid template on
// the active axis, the child order (graph-first / graph-last), and the axis
// class that drives cursor + drag direction. graphPct is always "the graph
// pane's share of the split, 0-100", whatever the orientation — so the drag
// handler inverts when the graph sits on the far side.

const LAYOUT_MODE_KEY = 'dashboard.liveLayoutMode';
const LAYOUT_PCT_KEY  = 'dashboard.liveSplitPct';
const LAYOUT_DEFAULT_MODE = 'graph-left';
const LAYOUT_DEFAULT_PCT  = 60;

const LAYOUTS = {
  'graph-left':   { axis: 'columns', graphFirst: true  },
  'graph-right':  { axis: 'columns', graphFirst: false },
  'graph-top':    { axis: 'rows',    graphFirst: true  },
  'graph-bottom': { axis: 'rows',    graphFirst: false },
};

let _layoutMode = LAYOUT_DEFAULT_MODE;
let _graphPct = LAYOUT_DEFAULT_PCT;

// Graph visibility — orthogonal to the four orientations. Default is
// width-driven: below GRAPH_AUTOLOAD_MIN_WIDTH the dashboard is almost
// certainly in a narrow/embedded pane (e.g. inside Claude), where a continuous
// render loop competes with the host, so we don't mount it at all. An explicit
// operator toggle wins over the width default from then on.
const GRAPH_VISIBLE_KEY = 'dashboard.graphVisible';
const GRAPH_AUTOLOAD_MIN_WIDTH = 1000;
let _graphVisible = true;

function _applyLayout(mode, graphPct) {
  const split = document.getElementById('live-split');
  if (!split) return;
  const cfg = LAYOUTS[mode] || LAYOUTS[LAYOUT_DEFAULT_MODE];
  const pct = Math.max(0, Math.min(100, graphPct));
  _layoutMode = mode;
  _graphPct = pct;

  const sizing = cfg.graphFirst
    ? `${pct}fr 6px ${100 - pct}fr`
    : `${100 - pct}fr 6px ${pct}fr`;
  if (cfg.axis === 'columns') {
    split.style.gridTemplateColumns = sizing;
    split.style.gridTemplateRows = '1fr';
  } else {
    split.style.gridTemplateRows = sizing;
    split.style.gridTemplateColumns = '1fr';
  }

  split.classList.remove('layout--horizontal', 'layout--vertical', 'graph-first', 'graph-last');
  split.classList.add(cfg.axis === 'columns' ? 'layout--horizontal' : 'layout--vertical');
  split.classList.add(cfg.graphFirst ? 'graph-first' : 'graph-last');

  document.querySelectorAll('.live-layout-btn').forEach(b => {
    b.classList.toggle('active', b.dataset.layout === mode);
  });
  // The graph resizes itself via a ResizeObserver on its container, which
  // catches this grid-template change along with window resizes and drags.
}

function _restoreLayout() {
  let mode = LAYOUT_DEFAULT_MODE;
  let pct  = LAYOUT_DEFAULT_PCT;
  try {
    const savedMode = localStorage.getItem(LAYOUT_MODE_KEY);
    if (savedMode && LAYOUTS[savedMode]) mode = savedMode;
    const savedPct = parseFloat(localStorage.getItem(LAYOUT_PCT_KEY));
    if (!Number.isNaN(savedPct) && savedPct >= 0 && savedPct <= 100) pct = savedPct;
  } catch (e) { /* localStorage may be blocked; defaults are fine */ }
  _applyLayout(mode, pct);
}

function _persistLayout() {
  try {
    localStorage.setItem(LAYOUT_MODE_KEY, _layoutMode);
    localStorage.setItem(LAYOUT_PCT_KEY, String(_graphPct));
  } catch (e) { /* blocked storage → silently skip */ }
}

function _restoreGraphVisibility() {
  let visible;
  try {
    const saved = localStorage.getItem(GRAPH_VISIBLE_KEY);
    if (saved === '1') visible = true;
    else if (saved === '0') visible = false;
  } catch (e) { /* blocked → fall through to width default */ }
  if (visible === undefined) visible = window.innerWidth >= GRAPH_AUTOLOAD_MIN_WIDTH;
  _graphVisible = visible;
  _applyGraphVisibility();
}

// Reflect _graphVisible in the DOM. Does NOT mount/destroy — callers do that,
// so this stays a pure view-sync safe to call before the graph module is ready.
function _applyGraphVisibility() {
  const split = document.getElementById('live-split');
  if (split) split.classList.toggle('graph-hidden', !_graphVisible);
  const btn = document.getElementById('graph-toggle-btn');
  if (btn) {
    btn.classList.toggle('active', _graphVisible);
    btn.textContent = _graphVisible ? 'Hide graph' : 'Show graph';
    btn.title = _graphVisible
      ? 'Hide the graph — frees its render loop'
      : 'Show the graph';
  }
}

/** Toolbar toggle. Full teardown, not pause: a hidden graph costs zero. */
export function toggleGraph() {
  _graphVisible = !_graphVisible;
  try { localStorage.setItem(GRAPH_VISIBLE_KEY, _graphVisible ? '1' : '0'); } catch (e) { /* blocked */ }
  _applyGraphVisibility();
  if (_graphVisible) graph.activate();
  else graph.destroy();
}

export function setLiveLayout(mode) {
  if (!LAYOUTS[mode]) return;
  _applyLayout(mode, _graphPct);
  _persistLayout();
}

// Idempotency guard — _setupDivider attaches document-level listeners. If
// init() ever fires twice (hot reload, future double-wire bug) we'd get N
// parallel handlers on every mouse move.
let _dividerWired = false;
function _setupDivider() {
  if (_dividerWired) return;
  const divider = document.getElementById('live-divider');
  const split = document.getElementById('live-split');
  if (!divider || !split) return;
  _dividerWired = true;
  let dragging = false;

  divider.addEventListener('mousedown', (e) => {
    dragging = true;
    divider.classList.add('dragging');
    split.classList.add('dragging');
    e.preventDefault();
  });
  document.addEventListener('mousemove', (e) => {
    if (!dragging) return;
    const cfg = LAYOUTS[_layoutMode];
    const rect = split.getBoundingClientRect();
    const raw = cfg.axis === 'columns'
      ? ((e.clientX - rect.left) / rect.width) * 100
      : ((e.clientY - rect.top)  / rect.height) * 100;
    // Graph on the FAR side → dragging toward it shrinks the first pane.
    _applyLayout(_layoutMode, cfg.graphFirst ? raw : 100 - raw);
  });
  document.addEventListener('mouseup', () => {
    if (!dragging) return;
    dragging = false;
    divider.classList.remove('dragging');
    split.classList.remove('dragging');
    _persistLayout();
  });
}

// ── Lifecycle ──────────────────────────────────────────────────────────────

export function init() {
  _restoreLayout();
  _restoreGraphVisibility();
  _setupDivider();
  _renderFeedEmptyState();

  bus.subscribe('insights:tick', _renderInsightsPanel);
  bus.subscribe('sessions:tick', () => { _renderStreamRail(); });
  _wireInsightsDismiss();

  // Pinned-card highlight. graph.js publishes `graph:pinned` when a pin is
  // set or cleared; mirroring it here means graph.js never reaches across the
  // module boundary to mutate this feed's DOM.
  bus.subscribe('graph:pinned', ({ eventId }) => {
    document.querySelectorAll('.act--pinned')
      .forEach(n => n.classList.remove('act--pinned'));
    if (!eventId) return;
    const target = document.querySelector('.act[data-graph-id="' + eventId + '"]');
    if (target) target.classList.add('act--pinned');
  });

  const liveVisible = () => document.getElementById('tab-live').classList.contains('active');

  poll.register({ key: 'recalls', interval: RECALL_POLL_MS, activeWhen: liveVisible, fetcher: pollRecalls });
  poll.register({ key: 'encodes', interval: ENCODE_POLL_MS, activeWhen: liveVisible, fetcher: pollEncodes });
  poll.register({
    key: 'insights-live',
    interval: 60000,
    activeWhen: liveVisible,
    fetcher: async () => {
      try {
        const d = await api.insightsLive();
        bus.publish('insights:tick', { insights: (d && d.insights) || [] });
      } catch (e) { console.error('[live] insights fetch failed:', e); }
    },
  });
}

export function activate() {
  // Live owns the graph — drive its activate() so the scene mounts (first
  // time) or resizes (after). Skipped entirely when hidden so no render loop
  // spins; toggleGraph() mounts on demand.
  if (_graphVisible) {
    try { graph.activate(); } catch (e) { console.error('[live] graph activate failed:', e); }
  }
  // Through the scheduler, so it coalesces with a poll already in flight: two
  // overlapping window fetches can answer out of order, and the older one would
  // un-settle a run that then fires twice.
  poll.fireNow('encodes');
}

export function deactivate() {
  // Graph stays mounted in the hidden tab; its canvas keeps state warm so
  // re-activating is instant.
}
