// ===========================================================================
// app.js — bootstrap + switchTab orchestrator.
// ---------------------------------------------------------------------------
// Per-tab logic lives in /static/tabs/{name}.js — each module exports
// {init, activate, deactivate}.
//
//   init()        called once on app boot (below). Registers polls + bus
//                 subs; does NOT fetch data.
//   activate()    called when this tab becomes visible. Lazy-loads data.
//   deactivate()  called when leaving this tab. Most modules no-op
//                 (poll.js auto-gates on activeWhen + document.hidden).
//
// `tab:active` bus topic fires on every successful switchTab, carrying
// {name} so any cross-cutting subscriber (e.g. a layout helper that needs
// to resize a panel) can react.
//
// app.js also owns "global" page chrome that no single tab can claim:
// the top stats bar, the daemon banner, the session-filter dropdown.
// ===========================================================================

import { api } from '/static/lib/api.js';
import { poll } from '/static/lib/poll.js';
import bus from '/static/lib/bus.js';
import sessionRegistry from '/static/lib/sessions.js';
import { runRecallProbe } from '/static/lib/recall_panel.js';

import * as live     from '/static/tabs/live.js';
import * as s2       from '/static/tabs/s2.js';
import * as journals from '/static/tabs/journals.js';
import * as graph    from '/static/tabs/graph.js';
import * as explorer from '/static/tabs/explorer.js';
import * as logs     from '/static/tabs/logs.js';
import * as overview from '/static/tabs/overview.js';
import * as traces   from '/static/tabs/traces.js';
import * as streams  from '/static/tabs/streams.js';
import * as thalamus from '/static/tabs/thalamus.js';

// ── Tab registry ──────────────────────────────────────────────────────
// `graph` is NOT a top-level tab — it lives inside Live as the left pane.
// It's still a module because Live needs to drive its activate/resize
// lifecycle.
// Primary tabs (visible top-of-page): live, streams, thalamus, s2, traces, logs.
// Overflow tabs (in the ⋯ dropdown):  journals, explorer, overview.
const TABS = { live, s2, journals, graph, explorer, logs, overview, traces, streams, thalamus };
const OVERFLOW_TABS = ['journals', 'explorer', 'overview'];
let activeTab = 'live';

function switchTab(name) {
  if (!TABS[name]) return;
  const prev = activeTab;
  try { TABS[prev]?.deactivate?.(); } catch(e) { console.error('[app] deactivate failed:', e); }
  activeTab = name;

  // Tabs are matched by data-tab (not array index), so the ⋯ button
  // with no data-tab stays untouched here — its active state is set below.
  document.querySelectorAll('.tab[data-tab]').forEach(t => {
    t.classList.toggle('active', t.dataset.tab === name);
  });
  // Overflow menu items: same data-tab convention.
  document.querySelectorAll('.tab-overflow-item').forEach(t => {
    t.classList.toggle('active', t.dataset.tab === name);
  });
  // ⋯ toggle is "active" iff the current tab is one of its menu items.
  const overflowToggle = document.querySelector('.tab-overflow-toggle');
  if (overflowToggle) overflowToggle.classList.toggle('active', OVERFLOW_TABS.includes(name));
  closeOverflowMenu();

  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');

  try { TABS[name]?.activate?.(); } catch(e) { console.error('[app] activate failed:', e); }
  bus.publish('tab:active', { name });
}

// ── Overflow menu ─────────────────────────────────────────────────────
function toggleOverflowMenu(event) {
  if (event) event.stopPropagation();
  const menu = document.getElementById('tab-overflow-menu');
  if (!menu) return;
  // The menu is position:fixed (it has to escape `.tabs`'s overflow clip), so
  // its coordinates come from the toggle's live rect rather than from CSS.
  const toggle = document.querySelector('.tab-overflow-toggle');
  if (toggle) {
    const r = toggle.getBoundingClientRect();
    menu.style.top = r.bottom + 'px';
    // Right-align to the toggle when a left-aligned menu would overflow the
    // viewport — the ⋯ sits at the end of the tab bar on a narrow window.
    const width = menu.offsetWidth || 140;
    menu.style.left = Math.max(4, Math.min(r.left, window.innerWidth - width - 4)) + 'px';
  }
  menu.classList.toggle('open');
}
function closeOverflowMenu() {
  const menu = document.getElementById('tab-overflow-menu');
  if (menu) menu.classList.remove('open');
}
// Click anywhere outside the menu closes it.
document.addEventListener('click', (e) => {
  if (!e.target.closest('.tab-overflow')) closeOverflowMenu();
});

// ── Page-chrome state (stats bar, session dropdown) ───────────────────
// These don't belong to any one tab — they appear on multiple tabs and are
// owned at the app level. The stats poll feeds Live's stats-bar AND the
// Explorer type-filter dropdown.

let daemonAlive = false;

async function loadStats() {
  try {
    const d = await api.stats();
    daemonAlive = d.daemon === 'alive';
    const statusClass = daemonAlive ? 'alive' : 'unavailable';
    const statusText = daemonAlive ? 'Daemon: alive' : 'Daemon: offline';
    document.getElementById('stats-bar').innerHTML =
      `<div class="stat"><span class="stat-value">${d.nodes}</span><span class="stat-label">Nodes</span></div>
       <div class="stat"><span class="stat-value">${d.edges}</span><span class="stat-label">Edges</span></div>
       <div class="stat"><span class="stat-value">${d.locked}</span><span class="stat-label">Locked</span></div>
       <div class="stat"><span class="stat-value">${d.recent_24h}</span><span class="stat-label">24h</span></div>
       <div class="stat"><span class="stat-value">${d.orphans}</span><span class="stat-label">Orphans</span></div>
       <div class="daemon-status ${statusClass}">${statusText}</div>
       <div class="daemon-status alive" style="font-size:10px;padding:3px 8px">${d.encoding ? 'Encode: ' + d.encoding.turns_since + '/' + (d.encoding.every ?? '?') + (d.encoding.due ? ' ⚡' : '') : ''}</div>`;

    const banner = document.getElementById('daemon-banner');
    if (!daemonAlive) {
      banner.innerHTML = '<div class="no-daemon-banner">Daemon is not running — showing read-only data from database. Live events unavailable.</div>';
    } else {
      banner.innerHTML = '';
    }

    const sel = document.getElementById('type-filter');
    const current = sel.value;
    sel.replaceChildren(new Option('All types', ''),
      ...Object.entries(d.types).map(([t, c]) => new Option(`${t} (${c})`, t, false, t === current)));
  } catch(e) { console.error('[dashboard] loadStats failed:', e); }
}

// Session identity is a registry, not a dropdown — lib/sessions.js merges the
// persisted env (worktree/branch/project) with live presence and publishes
// `sessions:tick`. Every consumer (stream rail, moment chips, traces filter)
// reads the same record, so a session's name is resolved once per page.

// ── Inline-handler exposure ───────────────────────────────────────────
// Inline `onclick="X()"` in index.html (and in dynamically-rendered
// innerHTML strings) look up `X` on the global `window`. ES modules don't
// pollute globals, so each handler the HTML names is mounted explicitly.

window.switchTab             = switchTab;
window.toggleOverflowMenu    = toggleOverflowMenu;
window.setLiveLayout         = live.setLiveLayout;
window.toggleGraph           = live.toggleGraph;
window.onS2HoursChange       = s2.onS2HoursChange;
window.onJournalsHoursChange = journals.onJournalsHoursChange;
window.onJournalsSearch      = journals.onJournalsSearch;
// Cards built by lib/activity.js and tabs/s2.js wire their own listeners
// with closure-captured refs — no window.* mount needed.
window.loadLogs              = logs.loadLogs;
window.onGraphSearch         = graph.onGraphSearch;
// Enter in the graph search box runs the REAL recall pipeline (a probe), not
// another substring pass. Typing keeps filtering locally — the two
// modes share one control because they answer the same intent at different
// depths: "show me where it is" vs "what would you actually recall".
// Wired here rather than in graph.js so the graph module doesn't import the
// recall panel that imports it back.
window.onGraphSearchKey      = (event) => {
  if (event && event.key === 'Enter') {
    graph.onGraphSearch();
    runRecallProbe(event.target && event.target.value);
    return;
  }
  graph.onGraphSearchKey(event);
};
window.onGraphRefresh        = graph.onGraphRefresh;
window.searchNodes           = explorer.searchNodes;
window.onTraceScaleChange    = traces.onTraceScaleChange;
window.loadTraces            = traces.loadTraces;
window._loadMoreTraces       = traces._loadMoreTraces;
window._revealTrace          = traces.revealTrace;
window.setTraceMode          = traces.setTraceMode;
window.loadStreams           = streams.loadStreams;
window.onStreamsSend         = streams.onStreamsSend;
window.onStreamsSendKey      = streams.onStreamsSendKey;
window.loadThalamus          = thalamus.loadThalamus;

// ── Boot ──────────────────────────────────────────────────────────────

// Page-chrome polls — 30s stats, 60s sessions. These run regardless of
// active tab (header is always visible).
poll.register({ key: 'stats',    interval: 30000, fetcher: loadStats });
poll.register({ key: 'sessions', interval: 20000, fetcher: sessionRegistry.refresh });

// Wire every tab module's polls + bus subs once (ES modules run their top
// level once per page load). Includes `graph` even though it's not a primary
// tab — Live embeds it and needs its lifecycle wired identically.
for (const name of Object.keys(TABS)) {
  try { TABS[name]?.init?.(); } catch(e) { console.error('[app] init', name, 'failed:', e); }
}

// Resolve session identity before the first paint. Every surface (stream rail,
// moment chips, Traces dropdown, Streams roster) reads names from the registry,
// so without this first pass they all render raw hex for one poll interval.
sessionRegistry.refresh();

// Activate the initial tab (Live by default — matches the active class
// in index.html). Subsequent switches go through switchTab.
try { TABS[activeTab]?.activate?.(); } catch(e) { console.error('[app] initial activate failed:', e); }
bus.publish('tab:active', { name: activeTab });
