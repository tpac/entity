// ===========================================================================
// tabs/streams.js — the self↔self channel, read as a MESSAGING APP.
// ---------------------------------------------------------------------------
// Two columns, the shape every chat app has trained the operator on:
//
//   rail  — the CONVERSATION LIST. One row per stream of thought, plus
//           `Everyone` (the broadcasts) and `All messages` (the merged
//           feed). Each row carries the stream's liveness light, its last
//           message and when — so one column answers both "who is live" and
//           "who is talking to whom".
//
//   main  — the OPEN THREAD. A stream's thread holds every message it sent or
//           received; above it, that stream's own pane (arc, cwd, turns, and
//           its boot context on click) is the thread header, the way a chat
//           app shows contact info. Oldest at the top, newest above a pinned
//           composer addressed to whoever the thread is with.
//
// Messages read as bubbles in their SENDER's hue — the same hue the stream
// carries in the rail, the Live feed and the graph. Delivery shows as a read
// receipt: ✓ sent, ✓✓ actually consumed by a stream.
//
// The composer is the dashboard's one write path (POST /api/self-send → daemon
// self_send). The operator sends attributed ('operator-dashboard'), never as a
// stream of thought — those messages render as the reader's own, on the right.
// ===========================================================================

import { api } from '/static/lib/api.js';
import { poll } from '/static/lib/poll.js';
import { escapeHtml, relativeTime, localTime } from '/static/lib/dom.js';
import { sessionLabel, sessionColor, sessionTooltip, sessionInfo } from '/static/lib/sessions.js';
import { renderPane } from '/static/lib/stream_roster.js';

// The label server.py stamps on an operator-authored send (never a session id).
const OPERATOR = 'operator-dashboard';
// The two pseudo-threads that aren't a single counterpart.
const ALL = 'all';
const EVERYONE = 'everyone';
const _THREAD_KEY = 'dashboard.streamsThread';

// Roster + message state. _lastPresence / _lastMessages let a thread switch
// re-render synchronously without re-fetching; _headOpen tracks whether the
// thread header's pane is drilled open (it survives the 5s poll, which rebuilds
// wholesale); _bootCache holds per-stream boot renders fetched on first drill.
let _lastPresence = null;
let _lastMessages = [];
const _headOpen = new Set();
const _bootCache = {};
const _ownChatCache = {};   // sid → that stream's own operator turns (identity cue)
const _msgExpanded = new Set();   // message ids whose body is expanded

let _activeThread = ALL;
let _paintedThread = null;   // which thread the message feed currently shows
try { _activeThread = localStorage.getItem(_THREAD_KEY) || ALL; } catch (_) { /* private mode */ }

// Render signatures — the 5s poll calls _loadPresence/_loadMessages every tick;
// without these guards it rebuilt the rail + thread wholesale every time,
// resetting scroll (and the scroll inside an open boot <pre> being read). We
// re-render only on a STRUCTURAL change (excluding raw updated_at, which ticks
// every poll). Thread switches and toggles call the painters directly, so
// skipping here never strands an open pane.
let _lastPresenceFp = null;
let _lastMsgFp = null;

function _msgSignature() {
  return _lastMessages.length + '|' + (_lastMessages[0] && _lastMessages[0].id) + '|'
    + _lastMessages.reduce((n, m) => n + (m.delivered ? m.delivered.length : 0), 0);
}
function _presenceSignature() {
  const streams = (_lastPresence && _lastPresence.streams) || [];
  const lost = (_lastPresence && _lastPresence.lost) || [];
  return streams.map(s => [s.session_id, s.state, s.turn_count, s.pending_inbox_count,
    s.focus, (s.arc || '').length].join(':')).join('|')
    + '#lost:' + lost.length + '#msg:' + _msgSignature();
}

// ── Identity ───────────────────────────────────────────────────────────
// The session REGISTRY (lib/sessions.js) owns handle resolution — the same one
// the stream rail and every moment chip use, so a stream is one identity
// wherever it appears, live or long gone. The operator is not a stream: they
// get their own name.
function _handleForSession(sid) {
  if (!sid) return '?';
  if (sid === OPERATOR) return 'You';
  return sessionLabel(sid) || sid.slice(0, 8);
}

function _colorForSession(sid) {
  return sid === OPERATOR ? 'var(--accent-green)' : sessionColor(sid || '');
}

// Two letters of the handle — enough to tell two streams apart at a glance
// without turning the avatar into another label.
function _initials(sid) {
  if (sid === OPERATOR) return '🧑';
  const h = _handleForSession(sid).replace(/[^a-z0-9]/gi, '');
  return (h.slice(0, 2) || '??').toUpperCase();
}

// The presence record for a stream, or a synthetic one for a session that only
// exists in message history (the daemon's roster is live streams + recently
// lost; a week-old sender is neither).
function _streamRecord(sid) {
  const p = _lastPresence || {};
  const found = (p.streams || []).find(s => s.session_id === sid)
    || (p.lost || []).find(s => s.session_id === sid);
  return found || { session_id: sid, short: sid.slice(0, 8), state: 'lost' };
}

// ── Threads ────────────────────────────────────────────────────────────
// Which conversation(s) a message belongs to. A broadcast lives in Everyone;
// a directed message lives in BOTH parties' threads — a stream's thread is
// everything it said or heard, which is the question the operator is asking
// when they click a stream.
function _threadKeysOf(m) {
  const from = m.from_full || m.from || '';
  const push = (keys, k) => {
    if (k && k !== OPERATOR && !keys.includes(k)) keys.push(k);
    return keys;
  };
  // A broadcast belongs to Everyone AND to each party it actually touched —
  // its sender and the streams that consumed it. A stream's thread claims to
  // be everything it said or heard, so a broadcast it received belongs there
  // too; leaving it only in Everyone made that claim false.
  if (m.address === 'self:broadcast') {
    const keys = push([EVERYONE], from);
    for (const d of m.delivered || []) push(keys, d.to_full);
    return keys;
  }
  return push(push([], from), (m.address || '').replace(/^self:/, ''));
}

function _messagesFor(key) {
  if (key === ALL) return _lastMessages;
  return _lastMessages.filter(m => _threadKeysOf(m).includes(key));
}

// One row's worth of data per conversation, newest-first among the streams.
function _conversations() {
  const byKey = new Map();
  const touch = (key) => {
    if (!byKey.has(key)) byKey.set(key, { key, count: 0, last: null, pending: 0 });
    return byKey.get(key);
  };

  for (const m of _lastMessages) {
    for (const key of _threadKeysOf(m)) {
      const c = touch(key);
      c.count++;
      // _lastMessages is newest-first, so the first one seen is the latest.
      if (!c.last) c.last = m;
      if (!(m.delivered || []).length && (m.address || '') === 'self:' + key) c.pending++;
    }
  }
  // Every live (and recently lost) stream gets a row even with nothing said —
  // an empty conversation you can open and start is the point of a contact list.
  for (const s of ((_lastPresence && _lastPresence.streams) || [])
                  .concat((_lastPresence && _lastPresence.lost) || [])) {
    touch(s.session_id);
  }

  const rows = [...byKey.values()].filter(c => c.key !== EVERYONE);
  for (const c of rows) {
    const rec = _streamRecord(c.key);
    c.state = rec.state || 'lost';
    c.focus = rec.focus || '';
    // pending_inbox_count is the daemon's own count of what's waiting for a
    // live stream — authoritative where we have it.
    if (rec.pending_inbox_count) c.pending = rec.pending_inbox_count;
    c.sortAt = (c.last && c.last.created_at) || rec.updated_at || '';
  }
  rows.sort((a, b) => {
    // Live streams float above dormant/lost ones; within a band, most recent.
    const rank = s => (s === 'active' ? 0 : s === 'dormant' ? 1 : 2);
    return rank(a.state) - rank(b.state) || (b.sortAt || '').localeCompare(a.sortAt || '');
  });

  const everyone = byKey.get(EVERYONE) || { key: EVERYONE, count: 0, last: null, pending: 0 };
  return { streams: rows, everyone };
}

// ── Conversation list (the rail) ───────────────────────────────────────
// The row's second line. With messages it previews the latest one. Without
// any, it shows IDENTITY CUES — model, turns, hex — never the stream's arc or
// focus: those are built from the operator's own prompts, and a row that
// echoes the operator's words back where a message goes reads as if the stream
// had said them. Cues are what tells the operator which session this is.
function _previewOf(c) {
  if (!c.last) {
    const info = sessionInfo(c.key);
    const cues = [];
    if (info.model) cues.push(info.model);
    const turns = info.turns || 0;
    if (turns) cues.push(turns + ' turn' + (turns === 1 ? '' : 's'));
    cues.push(info.short || c.key.slice(0, 8));
    return { text: cues.join(' · '), cls: ' conv-preview--cues' };
  }
  // The mixed feeds (All / Everyone) name their speaker — without it the row
  // shows a body with no way to tell which stream said it. A single stream's
  // row doesn't need it: the row IS the speaker.
  const from = c.last.from_full || c.last.from;
  const mixed = c.key === EVERYONE || c.key === ALL;
  const who = from === OPERATOR ? 'You: '
    : (mixed ? _handleForSession(from) + ': ' : '');
  return { text: who + (c.last.body || '').replace(/\s+/g, ' ').trim(), cls: '' };
}

function _convRow(c, { title, glyph, color, live, muted, subtitle }) {
  const active = _activeThread === c.key;
  const badge = c.pending
    ? '<span class="notif show conv-badge">' + c.pending + '</span>' : '';
  const when = c.last ? relativeTime(c.last.created_at) : '';
  // A pseudo-row (All / Everyone) says what it collects when it's empty —
  // it has no session to draw identity cues from.
  const preview = (!c.last && subtitle)
    ? { text: subtitle, cls: ' conv-preview--cues' } : _previewOf(c);
  return '<div class="conv-row' + (active ? ' is-active' : '') + (muted ? ' is-muted' : '') + '"'
    + ' data-conv="' + escapeHtml(c.key) + '"'
    + (c.key === ALL || c.key === EVERYONE ? '' : ' title="' + escapeHtml(sessionTooltip(c.key)) + '"')
    + '>'
    + '<div class="conv-avatar" style="background:' + color + '">' + glyph
    +   (live ? '<span class="live-light ' + live + ' conv-light"></span>' : '')
    + '</div>'
    + '<div class="conv-main">'
    +   '<div class="conv-top">'
    +     '<span class="conv-name" style="color:' + color + '">' + escapeHtml(title) + '</span>'
    +     '<span class="conv-time">' + escapeHtml(when) + '</span>'
    +   '</div>'
    +   '<div class="conv-preview' + preview.cls + '">' + escapeHtml(preview.text) + '</div>'
    + '</div>'
    + badge
  + '</div>';
}

function _paintConvList() {
  const host = document.getElementById('conv-list');
  if (!host) return;
  const { streams, everyone } = _conversations();

  const all = { key: ALL, count: _lastMessages.length, pending: 0, last: _lastMessages[0] || null };
  let h = '<div class="conv-section">Conversations</div>';
  h += _convRow(all, { title: 'All messages', glyph: '📚', color: 'var(--text-heading)',
                       subtitle: 'every message in the window' });
  h += _convRow(everyone, { title: 'Everyone', glyph: '🌐', color: 'var(--accent-purple)',
                            subtitle: 'broadcasts to all live streams' });

  if (streams.length) h += '<div class="conv-section">Streams of thought</div>';
  for (const c of streams) {
    h += _convRow(c, {
      title: _handleForSession(c.key),
      glyph: _initials(c.key),
      color: sessionColor(c.key),
      live: c.state === 'active' ? 'active' : c.state === 'dormant' ? 'dormant' : 'lost',
      muted: c.state === 'lost',
    });
  }
  host.innerHTML = h;
}

// ── Thread header (who this conversation is with) ──────────────────────
function _paintThreadHead() {
  const host = document.getElementById('chat-thread');
  const title = document.getElementById('chat-title');
  const sub = document.getElementById('chat-sub');
  if (!host) return;

  if (_activeThread === ALL) {
    host.innerHTML = '';
    if (title) title.textContent = '💬 All messages';
    if (sub) sub.textContent = 'what the streams of thought are saying to each other';
    return;
  }
  if (_activeThread === EVERYONE) {
    host.innerHTML = '';
    if (title) title.textContent = '🌐 Everyone';
    if (sub) sub.textContent = 'broadcasts — sent to every stream that was live at the time';
    return;
  }

  const rec = _streamRecord(_activeThread);
  if (title) title.textContent = '💬 ' + _handleForSession(_activeThread);
  if (sub) sub.textContent = 'everything this stream said or heard';
  host.innerHTML = renderPane(rec, {
    open: _headOpen.has(_activeThread),
    boots: _bootCache[_activeThread],
    ownChat: _ownChatCache[_activeThread],
  });
}

// The stream's own operator turns — refetched every time its thread is opened.
// Identity, not traffic: it answers "which of my sessions is this?", which the
// handle and hex alone don't when three of them are called `main`. The cache
// is for painting instantly while the request is in flight, NOT a one-shot: a
// not-yet-spoken stream or a failed fetch must not stick for the page's life.
async function _loadOwnChat(sid) {
  if (sid === ALL || sid === EVERYONE) return;
  try {
    const body = await api.sessionMessages({ session: sid, limit: 8 });
    _ownChatCache[sid] = (body && body.messages || []).slice().reverse();  // oldest first
  } catch (_) {
    if (_ownChatCache[sid] === undefined) _ownChatCache[sid] = [];  // keep a good cache
  }
  if (_activeThread === sid) _paintThreadHead();
}

// ── Thread switching ───────────────────────────────────────────────────
// The composer follows the open thread — until the operator picks a recipient
// by hand, which is them saying "send this somewhere else without leaving the
// thread I'm reading". _toTouched is that override; a thread switch clears it.
let _toTouched = false;

function _syncComposerTarget() {
  const sel = document.getElementById('streams-send-to');
  if (!sel) return;
  const valid = v => [...sel.options].some(o => o.value === v);
  if (_toTouched) {
    if (valid(sel.value)) return;
    // The hand-picked recipient just left the roster. Retargeting a message
    // the operator wrote for THAT stream — silently, to everyone — is the one
    // outcome the composer must not produce, so say it out loud.
    _toTouched = false;
    const status = document.getElementById('streams-send-status');
    if (status) status.textContent = '⚠ that stream ended — recipient reset, check it before sending';
  }
  const target = (_activeThread === ALL || _activeThread === EVERYONE) ? 'broadcast' : _activeThread;
  sel.value = valid(target) ? target : 'broadcast';
}

function _setThread(key) {
  _activeThread = key;
  _toTouched = false;
  try { localStorage.setItem(_THREAD_KEY, key); } catch (_) { /* private mode */ }

  _syncComposerTarget();
  const input = document.getElementById('streams-send-body');
  if (input) {
    input.placeholder = (key === ALL || key === EVERYONE)
      ? 'Say something to every live stream…'
      : 'Message ' + _handleForSession(key) + '…';
  }

  _paintConvList();
  _paintThreadHead();
  _paintMessages();
  _loadOwnChat(key);
}

export async function loadStreams() {
  await _loadMessages();    // load first so the rail has previews to render
  await _loadPresence();
}

// ── Presence ───────────────────────────────────────────────────────────
async function _loadPresence() {
  try {
    _lastPresence = await api.selfPresence();
    _syncSendDropdown(_lastPresence);   // cheap; safe to refresh every tick
    const fp = _presenceSignature();
    if (fp === _lastPresenceFp) return;  // nothing structural changed
    _lastPresenceFp = fp;
    _paintConvList();
    _paintThreadHead();
  } catch (e) { console.error('[streams] presence', e); }
}

// Handle-first dropdown (branch · focus). Preserve selection across refreshes.
function _syncSendDropdown(p) {
  const sel = document.getElementById('streams-send-to');
  if (!sel) return;
  const current = sel.value;
  let html = '<option value="broadcast">everyone (broadcast)</option>';
  for (const s of (p && p.streams) || []) {
    const handle = (s.branch && s.branch !== 'unknown') ? s.branch : (s.short || s.session_id.substring(0, 8));
    const focus = s.focus ? ' — ' + s.focus.substring(0, 40) : '';
    html += '<option value="' + escapeHtml(s.session_id) + '">' + escapeHtml(handle) + escapeHtml(focus) + '</option>';
  }
  sel.innerHTML = html;
  if (current) sel.value = current;
  _syncComposerTarget();   // rebuilt options: re-apply the thread's recipient
}

// Click the rail → open that conversation.
function _onConvClick(e) {
  const row = e.target.closest('[data-conv]');
  if (!row) return;
  const key = row.getAttribute('data-conv');
  if (key && key !== _activeThread) _setThread(key);
}

// Click the thread header's title bar → drill into that stream's boot context.
async function _onThreadHeadClick(e) {
  const bar = e.target.closest('[data-stream-toggle]');
  if (!bar) return;
  const sid = bar.getAttribute('data-stream-toggle');
  if (!sid) return;
  if (_headOpen.has(sid)) {
    _headOpen.delete(sid);
    _paintThreadHead();
    return;
  }
  _headOpen.add(sid);
  _paintThreadHead();                // immediate (shows "loading boot…")
  if (_bootCache[sid] === undefined) {
    try {
      const body = await api.bootRenders({ session: sid, limit: 3 });
      _bootCache[sid] = (body && body.renders) || [];
    } catch (_) { _bootCache[sid] = []; }
    if (_headOpen.has(sid)) _paintThreadHead();
  }
}

// ── Messages (courier + delivery fan-out) ──────────────────────────────
async function _loadMessages() {
  try {
    const hours = document.getElementById('streams-hours').value;
    const body = await api.selfMessages({ hours });
    _lastMessages = (body && body.messages) || [];
    const fp = _msgSignature();
    if (fp === _lastMsgFp) return;   // unchanged — don't stomp scroll
    _lastMsgFp = fp;
    _paintConvList();
    _paintMessages();
  } catch (e) { console.error('[streams] messages', e); }
}

// A body longer than this (or taller than the clamp) gets a "show more"
// affordance. Measured in characters rather than by layout because the feed is
// rebuilt from a string — a deterministic guess beats a reflow.
const _CLAMP_CHARS = 300;
const _CLAMP_LINES = 6;

function _isClampable(body) {
  const b = body || '';
  return b.length > _CLAMP_CHARS || (b.match(/\n/g) || []).length >= _CLAMP_LINES;
}

// Day heading text for a message's local calendar day.
function _dayLabel(iso) {
  const d = new Date(iso);
  if (isNaN(d)) return '';
  const today = new Date();
  const y = new Date(today.getTime() - 86400000);
  const same = (a, b) => a.toDateString() === b.toDateString();
  if (same(d, today)) return 'Today';
  if (same(d, y)) return 'Yesterday';
  return d.toLocaleDateString(undefined, { weekday: 'long', month: 'short', day: 'numeric' });
}

// Render the open thread from cache (called by the load, the thread switch and
// the expand toggle). The API hands back newest-first; a chat reads
// oldest-first, so it's flipped here and the scroll is parked at the bottom.
function _paintMessages() {
  const msgs = _messagesFor(_activeThread);
  const countEl = document.getElementById('streams-count');
  if (countEl) countEl.textContent = msgs.length + ' message' + (msgs.length === 1 ? '' : 's');

  const host = document.getElementById('feed-streams-messages');
  if (!host) return;

  // A NEW conversation always opens at its newest message. Without this the
  // "are we at the bottom?" test below reads the scroll of the thread being
  // replaced — open a thread while scrolled up in another and you land in the
  // middle of its history, and because the test then keeps returning false,
  // that thread never auto-scrolls again.
  const threadChanged = _paintedThread !== _activeThread;
  _paintedThread = _activeThread;

  if (!msgs.length) {
    host.innerHTML = '<div class="chat-empty">'
      + '<div class="chat-empty-glyph">💬</div>'
      + '<div class="chat-empty-title">Nothing said yet</div>'
      + '<div class="chat-empty-sub">'
      + (_activeThread === ALL || _activeThread === EVERYONE
          ? 'When streams of thought message each other it shows up here. You can start the '
            + 'conversation below — older messages reap after their TTL, and survive only as S0 traces.'
          : 'Nothing to or from this stream in the window. Say something below, or widen the '
            + 'time range — messages reap after their TTL and survive only as S0 traces.')
      + '</div></div>';
    return;
  }

  // Stay pinned to the newest message unless the operator has scrolled up to
  // read history — then leave their position alone.
  const atBottom = threadChanged
    || host.scrollHeight - host.scrollTop - host.clientHeight < 80;

  const asc = msgs.slice().reverse();
  const out = [];
  let day = '';
  let prev = null;
  for (const m of asc) {
    const d = _dayLabel(m.created_at);
    if (d && d !== day) {
      day = d;
      out.push('<div class="chat-day"><span>' + escapeHtml(d) + '</span></div>');
      prev = null;                 // a new day always re-introduces the speaker
    }
    out.push(_renderMessage(m, prev));
    prev = m;
  }
  host.innerHTML = out.join('');

  if (atBottom) host.scrollTop = host.scrollHeight;
}

// One message as a chat bubble. `prev` is the message above it — a run from
// the same sender to the same recipient within 4 minutes drops the header, the
// way any chat groups a burst.
function _renderMessage(m, prev) {
  const from = m.from_full || m.from || '';
  const broadcast = m.address === 'self:broadcast';
  const toSid = broadcast ? '' : (m.address || '').replace(/^self:/, '');
  const mine = from === OPERATOR;
  const hue = _colorForSession(from);

  const grouped = !!prev && (prev.from_full || prev.from || '') === from
    && prev.address === m.address
    && Math.abs(new Date(m.created_at) - new Date(prev.created_at)) < 4 * 60000;

  const head = grouped ? '' :
    '<div class="chat-meta">'
    + '<span class="chat-from" style="color:' + hue + '"'
    +   (from === OPERATOR ? '' : ' title="' + escapeHtml(sessionTooltip(from)) + '"')
    +   '>' + escapeHtml(_handleForSession(from)) + '</span>'
    + (broadcast
        ? '<span class="chat-to chat-to--all">→ everyone</span>'
        : '<span class="chat-to">→ ' + escapeHtml(_handleForSession(toSid)) + '</span>')
    + '<span class="chat-time" title="' + escapeHtml(localTime(m.created_at) || m.created_at || '') + '">'
    +   escapeHtml(relativeTime(m.created_at)) + '</span>'
    + '</div>';

  const expanded = _msgExpanded.has(String(m.id));
  const clampable = _isClampable(m.body);
  const bodyClass = (clampable && !expanded) ? ' msg-clamp' : '';
  const moreHtml = clampable
    ? '<div class="chat-more">' + (expanded ? 'show less' : 'show more') + '</div>'
    : '';

  const refsHtml = (m.refs && m.refs.length)
    ? '<div class="chat-refs">refs: ' + m.refs.map(r => escapeHtml(String(r))).join(', ') + '</div>'
    : '';

  // Delivery as a read receipt: ✓✓ once a stream has actually consumed it.
  const delivered = m.delivered || [];
  const receipt = delivered.length
    ? '<div class="chat-receipt chat-receipt--read">✓✓ read by '
        + delivered.map(d => '<span title="' + escapeHtml(d.to_full || '') + ' @ '
            + escapeHtml(d.at || '') + '">'
            + escapeHtml(_handleForSession(d.to_full) || d.to) + '</span>').join(', ')
      + '</div>'
    : '<div class="chat-receipt chat-receipt--sent">✓ sent · not read yet</div>';

  return '<div class="chat-row' + (mine ? ' is-mine' : '') + (grouped ? ' is-grouped' : '') + '"'
    + ' data-mid="' + escapeHtml(String(m.id || '')) + '">'
    + '<div class="chat-avatar" style="background:' + (grouped ? 'transparent' : hue) + '">'
    +   (grouped ? '' : escapeHtml(_initials(from))) + '</div>'
    + '<div class="chat-stack">'
    +   head
    +   '<div class="chat-bubble" style="--from-color:' + hue + '">'
    +     '<div class="msg-body' + bodyClass + '">' + escapeHtml(m.body || '') + '</div>'
    +     moreHtml
    +     refsHtml
    +   '</div>'
    +   receipt
    + '</div>'
  + '</div>';
}

// Click a bubble → expand/collapse a long body.
function _onMessagesClick(e) {
  const row = e.target.closest('.chat-row');
  if (!row) return;
  const mid = row.getAttribute('data-mid');
  if (!mid) return;
  if (_msgExpanded.has(mid)) _msgExpanded.delete(mid);
  else _msgExpanded.add(mid);
  _paintMessages();
}

// ── Send composer (the one write path) ─────────────────────────────────
export function onStreamsSendKey(event) {
  if (event && event.key === 'Enter') { event.preventDefault(); onStreamsSend(); }
}

export async function onStreamsSend() {
  const toEl = document.getElementById('streams-send-to');
  const bodyEl = document.getElementById('streams-send-body');
  const statusEl = document.getElementById('streams-send-status');
  const to = toEl ? toEl.value : '';
  const body = bodyEl ? bodyEl.value.trim() : '';
  if (!body) { if (statusEl) statusEl.textContent = 'type a message first'; return; }
  if (statusEl) statusEl.textContent = 'sending…';
  try {
    await api.selfSend({ to, body });
    if (bodyEl) bodyEl.value = '';
    if (statusEl) {
      statusEl.textContent = '✓ sent to ' + (to === 'broadcast' ? 'everyone' : _handleForSession(to));
      setTimeout(() => { if (statusEl) statusEl.textContent = ''; }, 4000);
    }
    await loadStreams();   // refresh the thread AND the rail's previews
  } catch (e) {
    if (statusEl) statusEl.textContent = '✗ ' + (e.message || 'send failed');
    console.error('[streams] send', e);
  }
}

// ── Lifecycle ──────────────────────────────────────────────────────────
export function init() {
  // Delegated listeners on the persistent hosts — they survive the poll-driven
  // rebuilds (mirrors the traces tab pattern).
  const rail = document.getElementById('conv-list');
  if (rail && !rail._convClickBound) {
    rail.addEventListener('click', _onConvClick);
    rail._convClickBound = true;
  }
  const head = document.getElementById('chat-thread');
  if (head && !head._headClickBound) {
    head.addEventListener('click', _onThreadHeadClick);
    head._headClickBound = true;
  }
  const msgFeed = document.getElementById('feed-streams-messages');
  if (msgFeed && !msgFeed._msgClickBound) {
    msgFeed.addEventListener('click', _onMessagesClick);
    msgFeed._msgClickBound = true;
  }
  const sendTo = document.getElementById('streams-send-to');
  if (sendTo && !sendTo._toChangeBound) {
    sendTo.addEventListener('change', () => { _toTouched = true; });
    sendTo._toChangeBound = true;
  }
  poll.register({
    key: 'streams',
    interval: 5000,
    activeWhen: () => {
      const tab = document.getElementById('tab-streams');
      return tab && tab.classList.contains('active');
    },
    fetcher: loadStreams,
  });
}

export function activate() {
  _setThread(_activeThread);   // restores the composer + titles for the saved thread
  loadStreams();
}
export function deactivate() {}
