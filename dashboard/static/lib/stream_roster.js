// ===========================================================================
// lib/stream_roster.js — ONE stream of thought, as a PANE.
// ---------------------------------------------------------------------------
// A stream of thought is a live process running in a worktree, so it reads as
// a terminal/window pane — a title bar (pulsing liveness light + ⎇ branch
// handle + state) over a body — NOT another flat memory card like Traces/Live.
// Identity = the branch/worktree handle (brain principle: "one stream, one
// worktree — your handle is your branch name"); hex is the subtitle.
//
// This is the WHO of an open conversation: the Streams tab renders it as the
// thread header above the messages, the way a chat app shows contact info.
// Click the title bar → it drills open inline: full arc + the stream's OWN
// boot context (folded in — there is no separate Boot tab). Collapsed, it
// clamps the arc.
//
// Pure presentation. Structural styling lives in style.css (.stream-pane /
// .stream-titlebar / .live-light + the streamPulse keyframe); this returns an
// HTML string (matches streams.js's idiom). State (whether the pane is open,
// the boot capture to show) is owned by streams.js and passed in — so it
// re-renders correctly under the 5s presence poll.
// ===========================================================================

import { escapeHtml, relativeTime, modelChipHTML } from '/static/lib/dom.js';
import { sessionLabel, sessionColor, sessionTooltip } from '/static/lib/sessions.js';

const _LIVE = {
  active:  { dot: 'active',  label: 'active',  color: '#33d17a' },
  dormant: { dot: 'dormant', label: 'dormant', color: '#ffaa33' },
  lost:    { dot: 'lost',    label: 'lost',    color: '#777' },
};

const _ARC_CLAMP = 150;

function _shortCwd(cwd) { return (cwd || '').replace(/^\/Users\/[^/]+/, '~'); }
function _dur(iso) { const r = relativeTime(iso); return r ? r.replace(/\s*ago$/, '') : ''; }
// Handle + hue + hover all come from the session registry (lib/sessions.js) —
// the same resolution the stream rail and every moment chip use, so one stream
// reads as one identity wherever it appears. One derivation, so a handle
// can't drift between views.
function _handle(s) { return sessionLabel(s.session_id || '') || s.short || ''; }

// ── drill-down sub-blocks ───────────────────────────────────────────────────
function _bootBlock(boots) {
  if (boots === undefined) return '<div style="color:#667;font-size:11px;margin-top:8px">loading boot context…</div>';
  if (!boots || !boots.length) return '<div style="color:#566;font-size:11px;margin-top:8px">No boot capture recorded for this stream.</div>';
  const b = boots[0];
  const more = boots.length > 1 ? ' <span style="color:#566">· ' + boots.length + ' boots</span>' : '';
  return '<details class="stream-boot" style="margin-top:8px">'
    + '<summary style="cursor:pointer;list-style:none;color:#c4a8f0;font-size:11px;font-weight:600">'
    + '🌅 boot context · ' + (b.char_count || 0) + ' chars · '
    + '<span style="color:#778;font-weight:400" title="' + escapeHtml(b.created_at || '') + '">' + escapeHtml(relativeTime(b.created_at)) + '</span>' + more + '</summary>'
    + '<pre style="white-space:pre-wrap;word-break:break-word;color:#cdd;font-size:11px;line-height:1.45;margin:6px 0 2px;max-height:340px;overflow:auto;'
    + 'background:#0c0c16;border:1px solid #181826;border-radius:4px;padding:8px 10px;font-family:ui-monospace,Menlo,monospace">'
    + escapeHtml(b.text || '') + '</pre></details>';
}

// The stream's OWN conversation with the operator — an identity cue, so the
// operator can tell which of their sessions this is. Rendered as a transcript
// strip, deliberately NOT in the chat-bubble language: these lines were never
// sent to another stream, and must never read as if they were.
function _ownChatBlock(rows, open, handle) {
  if (rows === undefined) return '';
  if (!rows || !rows.length) {
    return '<div class="own-chat"><div class="own-chat-head">its own chat</div>'
      + '<div class="own-chat-empty">nothing recorded for this stream</div></div>';
  }
  const shown = rows.slice(open ? -8 : -3);
  return '<div class="own-chat' + (open ? ' is-open' : '') + '">'
    + '<div class="own-chat-head">its own chat <span>· with you, not another stream</span></div>'
    + shown.map(r => '<div class="own-chat-row">'
        + '<span class="own-chat-who own-chat-who--' + r.role + '">'
        +   (r.role === 'operator' ? 'you' : escapeHtml(handle || 'stream')) + ' ›</span>'
        + '<span class="own-chat-text" title="' + escapeHtml(r.created_at || '') + '">'
        +   escapeHtml((r.text || '').replace(/\s+/g, ' ').trim()) + '</span>'
      + '</div>').join('')
  + '</div>';
}

// ── pane ─────────────────────────────────────────────────────────────────
function _pane(s, open, boots, ownChat) {
  const live = _LIVE[s.state] || _LIVE.dormant;
  const sid = escapeHtml(s.session_id || '');
  const handle = escapeHtml(_handle(s));

  // The stream's hue — the same one its moments carry in the Live feed and
  // its activation carries in the graph, so a stream is one color everywhere.
  const hue = sessionColor(s.session_id || '');
  let h = '<div class="stream-pane ' + (s.state === 'active' ? 'is-active ' : '') + (open ? 'is-open' : '')
    + '" data-sid="' + sid + '" style="--stream-color:' + hue + '">';

  // title bar (the click target for drill-down)
  h += '<div class="stream-titlebar" data-stream-toggle="' + sid + '">'
    + '<span class="live-light ' + live.dot + '"></span>'
    + '<span class="stream-handle" style="color:' + hue + '" title="' + escapeHtml(sessionTooltip(s.session_id || '')) + '"><span class="glyph">⎇</span>' + handle + '</span>'
    + '<span class="stream-hex">' + escapeHtml(s.short || '') + '</span>'
    + '<span style="flex:1"></span>'
    + '<span class="stream-state" style="color:' + live.color + '" title="' + escapeHtml(s.updated_at || '') + '">'
    + live.label + (s.updated_at ? ' · ' + escapeHtml(relativeTime(s.updated_at)) : '') + '</span>'
    + '<span style="color:#566;font-size:10px;width:10px">' + (open ? '▾' : '▸') + '</span>'
    + '</div>';

  // body
  h += '<div class="stream-body">';

  // metrics
  const stats = [];
  if (s.cwd) stats.push('<span style="font-family:ui-monospace,monospace">' + escapeHtml(_shortCwd(s.cwd)) + '</span>');
  // What the stream rides on right now — presence mirrors the latest turn's model.
  if (s.model) stats.push(modelChipHTML(s.model, s.host));
  if (s.turn_count) stats.push(s.turn_count + ' turn' + (s.turn_count === 1 ? '' : 's'));
  const tenure = _dur(s.session_started_at);
  if (tenure) stats.push('up ' + escapeHtml(tenure));
  if (stats.length) {
    h += '<div style="color:#778;font-size:10px">' + stats.join('<span style="color:#445;margin:0 5px">·</span>');
    if (s.pending_inbox_count) h += '<span style="background:#2a1a00;border:1px solid #c83;border-radius:3px;color:#ffaa33;font-size:9px;padding:0 5px;margin-left:6px">📥 ' + s.pending_inbox_count + ' waiting</span>';
    h += '</div>';
  }

  // focus
  if (s.focus && s.focus.trim()) {
    h += '<div style="color:#cfd;font-size:12px;margin-top:6px;white-space:pre-wrap;word-break:break-word">' + escapeHtml(s.focus) + '</div>';
  }

  // arc — clamped when closed, full when drilled open
  const arc = (s.arc || '').trim();
  if (arc) {
    const shown = open ? arc : (arc.length > _ARC_CLAMP ? arc.slice(0, _ARC_CLAMP) + '…' : arc);
    h += '<div style="color:#8a8a9a;font-size:11px;margin-top:6px;line-height:1.5;white-space:pre-wrap;word-break:break-word;border-top:1px solid #15151f;padding-top:6px">'
      + escapeHtml(shown) + '</div>';
  }

  // its own chat — the recognition cue, shown collapsed AND open
  h += _ownChatBlock(ownChat, open, _handle(s));

  // drill-down: this stream's own boot context
  if (open) h += _bootBlock(boots);

  h += '</div></div>';
  return h;
}

/** Render ONE stream as a pane — the thread header of an open conversation.
 *  `s` is a presence record ({session_id, state, arc, focus, cwd, …}); `opts`
 *  = { open:bool, boots:[...], ownChat:[...] } owned by streams.js. */
export function renderPane(s, opts = {}) {
  if (!s || !s.session_id) return '';
  return _pane(s, !!opts.open, opts.boots, opts.ownChat);
}
