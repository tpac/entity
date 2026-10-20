// ===========================================================================
// tabs/overview.js — system status grid + aspect taxonomy + health summary.
// ---------------------------------------------------------------------------
// The whole-brain view: is everything running, what does the taxonomy look
// like, how many nodes/edges/orphans. The `.health-*` classes below name the
// CARD TYPE (a health readout), not the tab — they stay as they are.
//
// Three render targets:
//   #status-grid    — Daemon/DB/Judge/Embedder cards (polled, 5s)
//   #aspects-grid   — 14 aspects with type + relation chips
//   #health-content — node/edge totals + insights + type breakdown
//
// All visual styling lives in components.css under the .status-card,
// .aspect-card, .health-insight, .type-count primitives. New shapes
// here go through a class — don't add fresh inline `style="..."`.
// ===========================================================================

import { api } from '/static/lib/api.js';
import { poll } from '/static/lib/poll.js';
import { escapeHtml, localTime, formatDuration } from '/static/lib/dom.js';

// ── System status (polled card grid) ────────────────────────────────

const STATUS_COMPONENTS = [
  { key: 'daemon',   label: 'Brain Daemon', icon: '🧠' },
  { key: 'brain_db', label: 'Brain DB',     icon: '💾' },
  { key: 'logs_db',  label: 'Logs DB',      icon: '📋' },
  { key: 'surface',  label: 'Surface',      icon: '⚖️' },
  { key: 'embedder', label: 'Embedder',     icon: '🔮' },
];

function _statusDetails(comp, s) {
  if (!s.alive) return s.error || 'unreachable';
  switch (comp.key) {
    case 'daemon':
      return 'PID: ' + (s.pid || '?')
           + ' · Uptime: ' + formatDuration(s.uptime);
    case 'brain_db':     return s.nodes + ' nodes · ' + (s.size_mb || '?') + 'MB';
    case 'logs_db':      return (s.size_mb || '?') + 'MB';
    case 'surface':      return localTime(s.last_run) + ' · ' + s.sample
                                + (s.model ? ' · ' + s.model : '');
    case 'embedder':     return s.model || '?';
    default:             return '';
  }
}

async function loadSystemStatus() {
  const grid = document.getElementById('status-grid');
  try {
    const d = await api.systemStatus();
    grid.innerHTML = '';
    for (const comp of STATUS_COMPONENTS) {
      const s = d.status[comp.key] || { alive: false, error: 'unknown' };
      const aliveCls = s.alive ? 'alive' : 'dead';
      const stateText = s.alive ? '● Live' : '● Down';
      const details = _statusDetails(comp, s);
      const card = document.createElement('div');
      card.className = 'status-card status-card--' + aliveCls;
      card.innerHTML =
        '<div class="status-card-row">' +
          '<span class="status-card-icon">' + comp.icon + '</span>' +
          '<div>' +
            '<div class="status-card-title">' + escapeHtml(comp.label) + '</div>' +
            '<div class="status-card-state status-card-state--' + aliveCls + '">' + stateText + '</div>' +
          '</div>' +
          '<div class="status-card-details">' + escapeHtml(details) + '</div>' +
        '</div>' +
        // The DB cards carry their file path; the daemon card the brain dir it
        // writes (system.daemon_record) — a mismatch with the DB cards is a
        // misconfiguration the Config tab spells out.
        ((s.path || s.db_dir) ? '<div class="status-card-path">' + escapeHtml(s.path || s.db_dir) + '</div>' : '');
      grid.appendChild(card);
    }
  } catch(e) {
    grid.innerHTML = '<div class="feed-empty feed-empty--error">'
                   + 'Failed to load status: ' + escapeHtml(String(e)) + '</div>';
  }
}

// ── Aspect taxonomy ─────────────────────────────────────────────────
// Source: aspects_v1.json (live at $BRAIN_DB_DIR/aspects_v1.json, seed in
// servers/scales/s2/aspects_v1.json). Counts come from brain.db.

function _aspectChip(label, count, kind) {
  return '<span class="aspect-chip aspect-chip--' + kind + '">'
       + escapeHtml(label)
       + '<span class="aspect-chip-count">' + count + '</span>'
       + '</span>';
}

function _aspectSection(label, members, kind, totalCount) {
  if (!members.length) return '';
  const more = totalCount > members.length ? totalCount - members.length : 0;
  return '<div class="aspect-card-section">'
       +   '<div class="aspect-card-section-label">' + escapeHtml(label) + '</div>'
       +   '<div>'
       +     members.map(m => _aspectChip(m.name, m.count, kind)).join('')
       +     (more ? '<span class="aspect-chip-more">+' + more + '</span>' : '')
       +   '</div>'
       + '</div>';
}

async function loadAspects() {
  try {
    const d = await api.aspects();
    const grid = document.getElementById('aspects-grid');
    if (!grid) return;
    const aspects = d.aspects || [];
    if (!aspects.length) {
      grid.innerHTML = '<div class="feed-empty">aspects_v1.json not found or empty</div>';
      return;
    }
    grid.innerHTML = aspects.map(a => {
      const lockBadge = a.locked ? ' <span class="aspect-card-locked">🔒</span>' : '';
      const dim = a.dimension
        ? '<span class="aspect-card-dim">' + escapeHtml(a.dimension) + '</span>'
        : '';
      const topTypes = a.node_types.slice(0, 12);
      const topRels  = a.edge_relations.slice(0, 12);
      return '<div class="aspect-card">'
           +   '<div class="aspect-card-head">'
           +     '<div>'
           +       '<span class="aspect-card-name">' + escapeHtml(a.name) + '</span>'
           +       lockBadge + dim
           +     '</div>'
           +     '<div class="aspect-card-totals">'
           +       a.totals.nodes + ' nodes · ' + a.totals.edges + ' edges'
           +     '</div>'
           +   '</div>'
           +   (a.meaning
                 ? '<div class="aspect-card-meaning">' + escapeHtml(a.meaning) + '</div>'
                 : '')
           +   _aspectSection('Node types',     topTypes, 'type',     a.node_types.length)
           +   _aspectSection('Edge relations', topRels,  'relation', a.edge_relations.length)
           + '</div>';
    }).join('');
  } catch(e) { console.error('loadAspects error:', e); }
}

// ── Health summary (stats + insights + types) ───────────────────────

function _healthInsight(i) {
  const sev = (i.severity || 'low').toLowerCase();
  const nodesHtml = i.nodes
    ? '<div class="health-insight-nodes">' + i.nodes.map(n =>
        '<div>&#8226; ' + escapeHtml((n.title || '').substring(0, 80))
        + ' <span class="health-insight-node-meta">('
        + escapeHtml(String(n.type || n.count || '')) + ')</span></div>'
      ).join('') + '</div>'
    : '';
  return '<div class="health-insight health-insight--' + sev + '">'
       +   '<div class="health-insight-title">' + (i.icon || '') + ' '
       +     escapeHtml(i.title || '') + '</div>'
       +   '<div class="health-insight-detail">' + escapeHtml(i.detail || '') + '</div>'
       +   nodesHtml
       + '</div>';
}

async function loadHealth() {
  try {
    const d = await api.stats();
    const ins = await api.insights();
    const hc = document.getElementById('health-content');
    const orphanClass = d.orphans > 20 ? 'bad' : d.orphans > 5 ? 'warn' : 'ok';
    const insightsHtml = (ins.insights || []).map(_healthInsight).join('');
    hc.innerHTML =
      '<div class="health-grid">'
      +   '<div class="health-card ok"><div class="hc-value">' + d.nodes + '</div><div class="hc-label">Total Nodes</div></div>'
      +   '<div class="health-card ok"><div class="hc-value">' + d.edges + '</div><div class="hc-label">Total Edges</div></div>'
      +   '<div class="health-card ok"><div class="hc-value">' + d.locked + '</div><div class="hc-label">Locked</div></div>'
      +   '<div class="health-card ' + (d.recent_24h > 0 ? 'ok' : 'warn') + '"><div class="hc-value">' + d.recent_24h + '</div><div class="hc-label">Last 24h</div></div>'
      +   '<div class="health-card ' + orphanClass + '"><div class="hc-value">' + d.orphans + '</div><div class="hc-label">Orphans</div></div>'
      + '</div>'
      + (insightsHtml
          ? '<h3 class="health-section-h3">Insights</h3>' + insightsHtml
          : '<div class="health-empty-good">No issues detected</div>')
      + '<h3 class="health-section-h3 health-section-h3--muted">Node Types</h3>'
      + '<div class="health-grid">'
      +   Object.entries(d.types).map(([t, c]) =>
            '<div class="health-card ok">'
            + '<span class="type-badge type--' + escapeHtml(t) + '">' + escapeHtml(t) + '</span>'
            + '<span class="type-count">' + c + '</span>'
            + '</div>'
          ).join('')
      + '</div>';
  } catch(e) { console.error('loadHealth failed:', e); }
}

// ── Lifecycle ─────────────────────────────────────────────────────────

export function init() {
  poll.register({
    key: 'system-status',
    interval: 5000,
    activeWhen: () => {
      const t = document.getElementById('tab-overview');
      return t && t.classList.contains('active');
    },
    fetcher: loadSystemStatus,
  });
}

export function activate() {
  loadHealth();
  loadSystemStatus();
  loadAspects();
}

export function deactivate() {}
