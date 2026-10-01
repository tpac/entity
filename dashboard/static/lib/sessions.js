// ===========================================================================
// lib/sessions.js — the session registry: one owner for "who is this stream".
// ---------------------------------------------------------------------------
// A session_id is a 36-char UUID. Nothing in the UI should ever show one raw,
// and nothing should re-derive a label for one. This module owns:
//
//   • the registry — /api/sessions (persisted env: worktree/branch/project/cwd)
//     merged with /api/self-presence (who is LIVE right now, plus their arc)
//   • the label     — sessionLabel(id) → the readable handle
//   • the hover     — sessionTooltip(id) → everything needed to FIND the thing
//   • the color     — sessionColor(id) → a stable hue per stream, so multiple
//     live streams stay distinguishable in the feed AND in the galaxy
//   • the chip      — sessionChip(id) → the rendered element
//
// The color is the part that earns its place with multiple live streams: with
// three streams interleaving in one chronological feed, the name alone makes
// you read every row to know who's talking. A consistent hue lets you see it.
//
// Hue assignment is a hash of the session id, NOT arrival order — so a stream's
// color survives a reload and matches across the feed, the stream rail, and the
// graph's activation tint. Order-based assignment would recolor everything the
// moment an older session aged out of the 7-day window.
// ===========================================================================

import { api } from '/static/lib/api.js';
import bus from '/static/lib/bus.js';
import { el, relativeTime } from '/static/lib/dom.js';

// Stream palette — distinguishable at chip size on the dark surface, and
// distinguishable from the graph's aspect hues (which are warmer/duller) so a
// stream tint never reads as a memory kind.
//
// Exported as RGB triples too: the graph tints each activation wave by the
// stream that caused it, and its sprites are built from raw channels. Both
// forms come from this one list so a stream is literally the same color in the
// feed and in the galaxy.
export const STREAM_RGB = [
  [126, 184, 255], [51, 255, 136], [255, 170, 51], [255, 102, 170],
  [69, 183, 209],  [196, 168, 240], [255, 204, 0],  [184, 255, 126],
];
const STREAM_HUES = STREAM_RGB.map(([r, g, b]) => `rgb(${r}, ${g}, ${b})`);

// session_id → record. Merged view; both sources write here.
const _registry = new Map();

function _hash(s) {
  let h = 2166136261 >>> 0;
  s = String(s || '');
  for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); }
  return (h >>> 0);
}

/** Palette slot for a stream — the shared index behind both color forms. */
export function sessionHueIndex(id) {
  return id ? (_hash(id) % STREAM_RGB.length) : -1;
}

/** Stable hue for a stream. Same id → same color, forever. */
export function sessionColor(id) {
  if (!id) return 'var(--text-muted)';
  return STREAM_HUES[sessionHueIndex(id)];
}

/** The record we know for a session — always an object, never undefined, so
 *  callers can read fields without guards on a session we've never seen. */
export function sessionInfo(id) {
  return _registry.get(id) || { id, short: (id || '').slice(0, 8) };
}

// Handles that more than one known session claims. A worktree outlives a
// session — resume it, or run two streams in it, and several session_ids share
// one handle. Rendered side by side in the rail that reads as a duplicate row,
// and two activity cards from different streams look like the same stream.
// Ambiguous handles get a four-char tail of their id appended; unique ones
// stay clean. The TAIL, not the head: Codex task ids are UUIDv7, time-ordered,
// so every session for weeks shares the leading chars and a head suffix
// collapses concurrent streams into one label. The last chars are random
// under both v4 and v7.
let _ambiguous = new Set();

function _recomputeAmbiguous() {
  const counts = new Map();
  for (const rec of _registry.values()) {
    const h = rec.handle || '';
    if (h) counts.set(h, (counts.get(h) || 0) + 1);
  }
  _ambiguous = new Set([...counts.entries()].filter(([, n]) => n > 1).map(([h]) => h));
}

/** The display handle. Falls back through worktree → branch tail → hex.
 *  Resolution lives server-side (queries/sessions._handle) so historical and
 *  live sessions agree; this is the client-side read of it, plus the
 *  disambiguating suffix when a handle is shared, and the hex fallback for ids
 *  the registry has never heard of. */
export function sessionLabel(id) {
  if (!id) return '';
  const rec = _registry.get(id);
  const handle = rec && rec.handle;
  if (!handle) return id.slice(0, 8);
  return _ambiguous.has(handle) ? handle + '·' + id.slice(-4) : handle;
}

/** Everything needed to find this stream again, as a title= string.
 *  Multi-line: browsers render \n in title attributes. */
export function sessionTooltip(id) {
  const r = sessionInfo(id);
  const lines = [];
  if (r.handle && r.handle !== r.short) lines.push(r.handle);
  if (r.handle && _ambiguous.has(r.handle)) {
    lines.push('(several sessions share this worktree — suffix disambiguates)');
  }
  lines.push('session ' + (id || ''));
  if (r.live) lines.push('● live' + (r.state ? ' · ' + r.state : ''));
  if (r.project) lines.push('project: ' + r.project);
  if (r.branch) lines.push('branch: ' + r.branch);
  if (r.worktree) lines.push('worktree: ' + r.worktree);
  if (r.cwd) lines.push('cwd: ' + r.cwd.replace(/^\/Users\/[^/]+/, '~'));
  // Latest model/host the stream rode on (per-turn values live on the traces).
  if (r.model) lines.push('model: ' + r.model + (r.host ? ' (' + r.host + ')' : ''));
  if (r.turns) lines.push(r.turns + ' turns');
  if (r.events) lines.push(r.events + ' trace events');
  if (r.last) lines.push('last seen ' + relativeTime(r.last));
  if (r.arc) lines.push('\narc: ' + r.arc);
  return lines.join('\n');
}

/** The chip: colored dot + handle, with the full hover. `compact` drops the
 *  dot for dense rows where the row itself is already tinted. */
export function sessionChip(id, { compact = false } = {}) {
  if (!id) return null;
  const color = sessionColor(id);
  return el('span', {
    class: 'session-chip',
    title: sessionTooltip(id),
    dataset: { session: id },
  },
    compact ? null : el('span', { class: 'session-dot', style: { background: color } }),
    el('span', { class: 'session-name', style: { color } }, sessionLabel(id)),
  );
}

/** Live streams, newest-active first — the stream rail's data. */
export function liveSessions() {
  return [..._registry.values()]
    .filter(r => r.live)
    .sort((a, b) => (b.last || '').localeCompare(a.last || ''));
}

/** Every known session, newest activity first — the focus picker's data. */
export function knownSessions() {
  return [..._registry.values()]
    .sort((a, b) => (b.last || '').localeCompare(a.last || ''));
}

function _upsert(id, fields) {
  if (!id) return;
  const prior = _registry.get(id) || { id, short: id.slice(0, 8) };
  _registry.set(id, { ...prior, ...fields });
}

/** Refresh both sources and publish `sessions:tick` for subscribers.
 *
 * Presence is authoritative for liveness + arc (only the daemon knows who is
 * breathing); /api/sessions is authoritative for the persisted env and trace
 * counts (it covers sessions that have since ended). A stream that is live but
 * has no trace rows yet still lands in the registry from presence alone — a
 * freshly-booted stream must be visible before it has done anything. */
export async function refresh() {
  let changed = false;
  try {
    const rows = await api.sessions();
    for (const s of (rows || [])) {
      _upsert(s.id, {
        short: s.short, handle: s.handle, first: s.first, last: s.last,
        events: s.events, branch: s.branch, worktree: s.worktree,
        project: s.project, cwd: s.cwd, turns: s.turns,
        model: s.model, host: s.host,
      });
      changed = true;
    }
  } catch (e) { console.error('[sessions] registry fetch failed:', e); }

  try {
    const p = await api.selfPresence({ limit: 20 });
    const liveIds = new Set();
    for (const s of (p.streams || [])) {
      const id = s.session_id || s.id || '';
      if (!id) continue;
      liveIds.add(id);
      const prior = sessionInfo(id);
      // Presence carries branch/worktree/cwd too, so a stream with no trace
      // rows yet still gets a real handle. Never let a blank from presence
      // erase what /api/sessions established — hence the `|| prior` chain on
      // every field, and the 'unknown' guard on branch (presence's literal
      // for "detection failed").
      const branch = (s.branch && s.branch !== 'unknown') ? s.branch : (prior.branch || '');
      const worktree = s.worktree || prior.worktree || '';
      _upsert(id, {
        live: true, state: s.state || '',
        arc: s.arc || s.focus || prior.arc || '',
        turns: s.turn_count || prior.turns || 0,
        branch, worktree,
        cwd: s.cwd || prior.cwd || '',
        model: s.model || prior.model || '',
        host: s.host || prior.host || '',
        // Same precedence the server's _handle uses: worktree, then the
        // branch tail with the shared `claude/` namespace dropped, then hex.
        handle: prior.handle
                || worktree
                || (branch ? (branch.includes('/') ? branch.slice(branch.indexOf('/') + 1) : branch) : '')
                || s.short || id.slice(0, 8),
        last: s.updated_at || prior.last || '',
        started: s.session_started_at || prior.started || '',
        inbox: s.pending_inbox_count || 0,
      });
      changed = true;
    }
    // Demote streams that dropped off the roster — otherwise a dead stream
    // keeps its live dot for the page's lifetime.
    for (const [id, rec] of _registry) {
      if (rec.live && !liveIds.has(id)) _registry.set(id, { ...rec, live: false });
    }
  } catch (e) { /* daemon down → registry still serves persisted names */ }

  if (changed) {
    _recomputeAmbiguous();
    bus.publish('sessions:tick', { sessions: knownSessions() });
  }
}

export default { sessionLabel, sessionTooltip, sessionChip, sessionColor,
                 sessionInfo, liveSessions, knownSessions, refresh };
